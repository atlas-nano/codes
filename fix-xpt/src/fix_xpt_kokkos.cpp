/* ----------------------------------------------------------------------
   fix_xpt_kokkos.cpp  —  KOKKOS-accelerated derived class of fix_xpt.

   Overrides the base-class virtual hooks (velocity gather, molecular
   Pass 1/2/3 atom loops, power-spectrum FFTs, multi-tau streams) with
   on-device Kokkos kernels; non-CUDA backends fall back to the CPU base.
---------------------------------------------------------------------- */

#include "fix_xpt_kokkos.h"

#include "atom_kokkos.h"
#include "atom_masks.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "memory_kokkos.h"

#include <mpi.h>
#include <type_traits>          // std::is_same_v for CUDA dispatch

using namespace LAMMPS_NS;
using namespace FixConst;

/* ---------------------------------------------------------------------- */

template<class DeviceType>
FixXPTKokkos<DeviceType>::FixXPTKokkos(LAMMPS *lmp, int narg, char **arg) :
  FixXPT(lmp, narg, arg)
{
  // Mark this fix as KOKKOS-aware so LAMMPS handles host/device data
  // synchronisation around our end_of_step.
  kokkosable      = 1;
  atomKK          = (AtomKokkos *) atom;
  execution_space = ExecutionSpaceFromDevice<DeviceType>::space;

  // Per-atom read mask (per-type mass k_mass is on a separate sync chain,
  // handled in init()).  X_MASK|IMAGE_MASK are needed for the on-device
  // unmap in the Pass 1/2/3 kernels.
  datamask_read   = V_MASK | MASK_MASK | TYPE_MASK | TAG_MASK
                  | X_MASK | IMAGE_MASK;
  datamask_modify = 0;       // fix_xpt does not modify atom state

  // Sentinel — force first sync on first kernel launch.
  slot_to_mol_view_generation_cached = 0xFFFFFFFFu;

  // Force first scratch View resize on first call.
  scratch_nm_cached     = -1;
  scratch_natoms_cached = -1;

  // Batched cuFFT plan lazily created on the first pwr_from_* call.
#ifdef KOKKOS_ENABLE_CUDA
  pwr_plan_kk_valid = false;
#endif
  pwr_plan_N_fft     = 0;
  pwr_plan_max_batch = 4096;  // tuneable; bounds device buffer to ~1-4 GB
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
FixXPTKokkos<DeviceType>::~FixXPTKokkos()
{
  // Destroy cuFFT plan before LAMMPS pulls down the GPU context.
#ifdef KOKKOS_ENABLE_CUDA
  if (pwr_plan_kk_valid) {
    cufftDestroy(pwr_plan_kk);
    pwr_plan_kk_valid = false;
  }
#endif
  // Consumer drops its ref to the shared vel_buf_view before member
  // destruction.  The owner may already be gone (LAMMPS destroys fixes in
  // non-strict reverse-add order), but Kokkos View ref-counting keeps the
  // allocation alive until the last consumer releases it — which is here.
  if (!owns_buffer) {
    vel_buf_view = t_vel3d();
  }
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void FixXPTKokkos<DeviceType>::init()
{
  // FP32 buffer is unsupported on the KK path: every device kernel reads
  // vel_buf_view as double.  Error out at init rather than fail at run time.
  if (buffer_precision == BUFFER_FP32)
    error->all(FLERR, "fix xpt/kk: `buffer_precision fp32` is not "
                      "supported by the KOKKOS path (V1).  Use the "
                      "plain `xpt` style on the CPU build, or stay "
                      "at fp64 with /kk.");
  // The base init builds the molecule topology from host tag/mask/type/molecule.
  atomKK->sync(Host, TAG_MASK | MASK_MASK | TYPE_MASK | MOLECULE_MASK);
  FixXPT::init();
  // Per-type mass is on a separate sync chain from the per-atom views
  // (it doesn't change during a run); push it to device once here.
  atomKK->k_mass.template modify<LMPHostType>();
  atomKK->k_mass.template sync<DeviceType>();
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void FixXPTKokkos<DeviceType>::setup(int vflag)
{
  // The base setup sizes the buffers (and builds the home layout) from host
  // tag/mask; ModifyKokkos syncs only to this fix's execution space.
  atomKK->sync(Host, TAG_MASK | MASK_MASK | TYPE_MASK | MOLECULE_MASK);
  FixXPT::setup(vflag);
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void FixXPTKokkos<DeviceType>::end_of_step()
{
  // The window-start prologue (molecule topology, home layout, slot list)
  // reads host tag/mask/type/molecule; ModifyKokkos syncs only to this fix's
  // execution space, so bring them to host here.
  if (iframe == 0)
    atomKK->sync(Host, TAG_MASK | MASK_MASK | TYPE_MASK | MOLECULE_MASK);
  // A subgroup's energy accumulation sums per-atom KE from host atom->v.
  if (pe_peratom)
    atomKK->sync(Host, V_MASK | MASK_MASK | TYPE_MASK);
  // Run the base-class step (populates host vel_buf), then lazily (re)size
  // the device-side View to match.
  FixXPT::end_of_step();
  if (!distributed()) (void) ensure_vel_buf_view_sized();
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
bool FixXPTKokkos<DeviceType>::ensure_vel_buf_view_sized()
{
  // Mirror the host vel_buf sizes (vel_buf_frames × natom_buf × 3 doubles;
  // frames = 1 for multi-tau single-frame, else nframes) on the device.
  // No-op if extents already match.  The distributed layout keeps no device
  // frame: only the replicated passes read one.
  if (distributed()) return false;
  if (vel_buf == nullptr || natom_buf <= 0) return false;
  const size_t frames = mt_vel_buf_single_frame ? size_t(1) : size_t(nframes);
  const size_t natoms = size_t(natom_buf);

  // Consumers alias the owner's ref-counted view (pointer bump) instead of
  // allocating their own, so N fixes sharing a buffer use 1× the GPU memory.
  // Falls back to own-alloc if the owner is not a FixXPTKokkos<DeviceType>.
  if (!owns_buffer && buffer_owner) {
    auto *owner_kk = dynamic_cast<FixXPTKokkos<DeviceType>*>(buffer_owner);
    if (owner_kk
        && owner_kk->vel_buf_view.extent(0) == frames
        && owner_kk->vel_buf_view.extent(1) == natoms
        && owner_kk->vel_buf_view.extent(2) == 3) {
      vel_buf_view = owner_kk->vel_buf_view;   // ref-count bump
      return false;
    }
  }

  if (vel_buf_view.extent(0) == frames &&
      vel_buf_view.extent(1) == natoms &&
      vel_buf_view.extent(2) == 3) {
    return false;
  }
  vel_buf_view = t_vel3d(Kokkos::view_alloc(Kokkos::WithoutInitializing,
                                             "fix_xpt:vel_buf_view"),
                          frames, natoms, 3);
  return true;
}

/* ----------------------------------------------------------------------
   sync_buffer_pointers_from_owner override.

   Run the base-class host pointer sync first, then alias the owner's
   device-side vel_buf_view.  Called by owner immediately after grow_buf
   (in FixXPT::grow_buf's tail loop over buffer_consumers) — so when this
   consumer's end_of_step subsequently runs, its vel_buf_view already
   points at owner's current allocation.
---------------------------------------------------------------------- */
template<class DeviceType>
void FixXPTKokkos<DeviceType>::sync_buffer_pointers_from_owner()
{
  FixXPT::sync_buffer_pointers_from_owner();
  if (!buffer_owner) return;
  auto *owner_kk = dynamic_cast<FixXPTKokkos<DeviceType>*>(buffer_owner);
  if (owner_kk) {
    vel_buf_view = owner_kk->vel_buf_view;     // ref-count bump
  }
}

/* ---------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   push_velocity_frame — KOKKOS override.

   Distributed layout: sync the per-atom arrays the host pack reads and run
   the base-class exchange (pack by home rank, MPI_Alltoallv, unpack).

   Replicated layout:
   1. Sync atomKK per-atom views to DeviceType (V, mask, tag, type).
   2. parallel_for over local atoms; each thread passing the group filter
      writes its velocity into a tag-indexed slot of a per-rank flat buffer
      + records mass at that slot.
   3. D2H mirror the flat buffer, MPI_Allreduce(SUM) into the base class's
      host vel_buf[ibuf][0] (keeps every CPU read path working unchanged).
   4. H2D mirror the Allreduced vel_buf[ibuf] back onto vel_buf_view, which
      the on-device FFT path reads directly.
---------------------------------------------------------------------- */
template<class DeviceType>
void FixXPTKokkos<DeviceType>::push_velocity_frame(int ibuf)
{
  // Distributed layout: the host pack reads x, v, image, mask, tag, type.
  if (distributed()) {
    atomKK->sync(Host, X_MASK | V_MASK | IMAGE_MASK | MASK_MASK | TAG_MASK | TYPE_MASK);
    FixXPT::push_velocity_frame(ibuf);
    return;
  }

  // Ensure the device-side velocity buffer and the persistent scratch
  // gather buffers (push_vel_local_d / push_mass_local_d) are sized.
  ensure_vel_buf_view_sized();
  ensure_scratch_views_sized();

  // Sync per-atom views from host to DeviceType (the GPU pair compute
  // already wrote velocities to k_v.view<DeviceType>() if the pair is
  // KK-accelerated; otherwise this is the canonical sync direction).
  atomKK->sync(execution_space, V_MASK | MASK_MASK | TAG_MASK | TYPE_MASK);

  // Device-side views (typed for the active execution space).
  auto v_view    = atomKK->k_v.template view<DeviceType>();
  auto mask_view = atomKK->k_mask.template view<DeviceType>();
  auto tag_view  = atomKK->k_tag.template view<DeviceType>();
  auto type_view = atomKK->k_type.template view<DeviceType>();
  auto mass_view = atomKK->k_mass.template view<DeviceType>();

  const int     nlocal_local = atom->nlocal;
  const int     natoms       = natom_buf;
  const int     groupbit_loc = groupbit;

  // Zero the persistent buffers each call so off-group slots stay 0 —
  // required for the single-rank memcpy fast path in reduce_or_copy.
  auto vel_local  = push_vel_local_d;
  auto mass_local = push_mass_local_d;
  Kokkos::deep_copy(vel_local,  0.0);
  Kokkos::deep_copy(mass_local, 0.0);

  // Per-atom kernel: write velocities + masses into tag-indexed slots.
  Kokkos::parallel_for("FixXPTKokkos::push_vel",
      Kokkos::RangePolicy<DeviceType>(0, nlocal_local),
      KOKKOS_LAMBDA(const int i) {
        if (!(mask_view(i) & groupbit_loc)) return;
        const int slot = tag_view(i) - 1;
        if (slot < 0 || slot >= natoms) return;
        vel_local(slot*3 + 0) = v_view(i, 0);
        vel_local(slot*3 + 1) = v_view(i, 1);
        vel_local(slot*3 + 2) = v_view(i, 2);
        mass_local(slot)      = mass_view(type_view(i));
      });

  // D2H mirror for the local→global reduction.
  auto vel_local_h  = Kokkos::create_mirror_view_and_copy(
                          Kokkos::HostSpace(), vel_local);
  auto mass_local_h = Kokkos::create_mirror_view_and_copy(
                          Kokkos::HostSpace(), mass_local);

  reduce_or_copy(vel_local_h.data(), vel_buf[ibuf][0], natoms * 3);

  // Multi-rank correctness rule: only overwrite mass_buf[slot] when the atom
  // is STILL local on this rank this frame (mass_local_h(slot) > 0).
  // group_slots is built once at window start; if an atom migrates away
  // mid-window its slot stays in group_slots, but mass_local was zeroed at
  // the top and the kernel won't re-fill it.  Without the guard mass_buf[slot]
  // is stomped to 0 → pwr_from_atoms skips it (m <= 0) and drops the migrated
  // atom's contribution to vac.  The CPU path writes mass_buf inline and never
  // zeroes between frames, so it has no such bug.
  for (int slot : group_slots) {
    const double m = mass_local_h(slot);
    if (m > 0.0) mass_buf[slot] = m;
  }

  // Mirror the Allreduced frame onto vel_buf_view (~natom_buf×3 doubles).
  auto vbv_subv   = Kokkos::subview(vel_buf_view, ibuf,
                                    Kokkos::ALL, Kokkos::ALL);
  auto vbv_subv_h = Kokkos::create_mirror_view(vbv_subv);
  for (int slot = 0; slot < natoms; slot++) {
    vbv_subv_h(slot, 0) = vel_buf[ibuf][slot][0];
    vbv_subv_h(slot, 1) = vel_buf[ibuf][slot][1];
    vbv_subv_h(slot, 2) = vel_buf[ibuf][slot][2];
  }
  Kokkos::deep_copy(vbv_subv, vbv_subv_h);

  // With the Pass 1/2/3 atom loops and the end-of-window FFTs on device,
  // nothing in the rest of accumulate_mol_frame reads host x/image/mask/tag/
  // type — unless the multi-tau correlator is active, whose stream push still
  // reads atom->x on host.  Keep the host sync conditional on that.
  const bool need_host_atom = do_molecule
                              && (correlator == CORR_MULTITAU);
  if (need_host_atom) {
    atomKK->sync(Host, X_MASK | IMAGE_MASK
                     | MASK_MASK | TAG_MASK | TYPE_MASK);
  }
}

/* ---------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   ensure_slot_to_mol_view_synced

   slot_to_mol[s] = molecule index for atom slot s (or -1 if not in group);
   populated by FixXPT::build_mol_topology() and stable across all the MD
   steps of a window.  We mirror it to the device once per topology
   rebuild and reuse it across every Pass 1 kernel launch in the window.

   Detection: every successful build_mol_topology() increments
   base::mol_topology_generation.  We cache the last-copied generation
   and skip the deep_copy when the counter has not changed.

   Returns true iff a copy actually fired this call.
---------------------------------------------------------------------- */
template<class DeviceType>
bool FixXPTKokkos<DeviceType>::ensure_slot_to_mol_view_synced()
{
  const size_t sz = slot_to_mol.size();
  if (sz == 0) return false;
  if (slot_to_mol_view_generation_cached == mol_topology_generation
      && slot_to_mol_view.extent(0) == sz) {
    return false;    // already up to date
  }
  if (slot_to_mol_view.extent(0) != sz) {
    slot_to_mol_view = Kokkos::View<int*, DeviceType>(
        Kokkos::view_alloc(Kokkos::WithoutInitializing,
                           "fix_xpt:slot_to_mol_view"),
        sz);
  }
  // Build a host-typed staging buffer, fill from std::vector<int>, deep_copy.
  Kokkos::View<int*, Kokkos::HostSpace> h_stage(
      Kokkos::view_alloc(Kokkos::WithoutInitializing,
                         "fix_xpt:slot_to_mol_host"),
      sz);
  for (size_t k = 0; k < sz; k++) h_stage(k) = slot_to_mol[k];
  Kokkos::deep_copy(slot_to_mol_view, h_stage);
  slot_to_mol_view_generation_cached = mol_topology_generation;
  return true;
}

/* ----------------------------------------------------------------------
   ensure_scratch_views_sized

   Lazy resize of the per-mol + per-atom device scratch Views to match
   current nmol_group / natom_buf.  Allocated WithoutInitializing; callers
   that need a zero start (the atomic_add scatters) deep_copy(view, 0.0)
   just before launching the kernel.

   Returns true iff a resize actually occurred this call.
---------------------------------------------------------------------- */
template<class DeviceType>
bool FixXPTKokkos<DeviceType>::ensure_scratch_views_sized()
{
  const int nm     = nmol_group;
  const int natoms = natom_buf;
  if (natoms <= 0) return false;
  bool resized = false;
  // Per-molecule scratch: molecular mode only (nmol_group is 0 when monatomic).
  if (nm > 0 && nm != scratch_nm_cached) {
    pass1_lcom_p_d = Kokkos::View<double*, DeviceType>(
        Kokkos::view_alloc(Kokkos::WithoutInitializing, "fix_xpt:pass1_lcom_p_d"),
        size_t(nm) * 3);
    pass1_lcom_r_d = Kokkos::View<double*, DeviceType>(
        Kokkos::view_alloc(Kokkos::WithoutInitializing, "fix_xpt:pass1_lcom_r_d"),
        size_t(nm) * 3);
    pass2_lL_d = Kokkos::View<double*, DeviceType>(
        Kokkos::view_alloc(Kokkos::WithoutInitializing, "fix_xpt:pass2_lL_d"),
        size_t(nm) * 3);
    pass2_lI_d = Kokkos::View<double*, DeviceType>(
        Kokkos::view_alloc(Kokkos::WithoutInitializing, "fix_xpt:pass2_lI_d"),
        size_t(nm) * 9);
    permol_vcom_d = Kokkos::View<double*, DeviceType>(
        Kokkos::view_alloc(Kokkos::WithoutInitializing, "fix_xpt:permol_vcom_d"),
        size_t(nm) * 3);
    permol_rcom_d = Kokkos::View<double*, DeviceType>(
        Kokkos::view_alloc(Kokkos::WithoutInitializing, "fix_xpt:permol_rcom_d"),
        size_t(nm) * 3);
    permol_omega_d = Kokkos::View<double*, DeviceType>(
        Kokkos::view_alloc(Kokkos::WithoutInitializing, "fix_xpt:permol_omega_d"),
        size_t(nm) * 3);
    scratch_nm_cached = nm;
    resized = true;
  }
  // Per-atom scratch, including the push gather buffers that every mode needs.
  if (natoms != scratch_natoms_cached) {
    pass3_vib_d = Kokkos::View<double*, DeviceType>(
        Kokkos::view_alloc(Kokkos::WithoutInitializing, "fix_xpt:pass3_vib_d"),
        size_t(natoms) * 3);
    // Per-step push gather buffers — also natom_buf-sized.
    push_vel_local_d = Kokkos::View<double*, DeviceType>(
        Kokkos::view_alloc(Kokkos::WithoutInitializing, "fix_xpt:push_vel_local_d"),
        size_t(natoms) * 3);
    push_mass_local_d = Kokkos::View<double*, DeviceType>(
        Kokkos::view_alloc(Kokkos::WithoutInitializing, "fix_xpt:push_mass_local_d"),
        size_t(natoms));
    scratch_natoms_cached = natoms;
    resized = true;
  }
  return resized;
}

/* ----------------------------------------------------------------------
   compute_mol_pass1_atoms — device kernel.

   parallel_for + atomic_add into per-rank-local device scratch, mirrored
   back to the caller's host lcom_p/lcom_r; the MPI_Allreduce(SUM) stays in
   the host caller (accumulate_mol_frame), matching the CPU path bit-for-bit.

   Why atomic_add: the scatter is many atoms → few molecule slots (small
   fan-in, ~3 atoms/molecule on water), so a per-thread atomic beats a
   ScatterView.  Double-precision atomicAdd is native on SM_60+.
---------------------------------------------------------------------- */
template<class DeviceType>
void FixXPTKokkos<DeviceType>::compute_mol_pass1_atoms(int ibuf_vel,
                                                       double *lcom_p,
                                                       double *lcom_r)
{
  // Empty-group fast path — caller's lcom_p/lcom_r already zero-init.
  const int nm = nmol_group;
  if (nm <= 0) return;

  // Velocities are read from vel_buf_view (device-resident, filled by
  // push_velocity_frame's H2D copy).  Sync per-atom views to DeviceType.
  atomKK->sync(execution_space,
               X_MASK | IMAGE_MASK | MASK_MASK | TAG_MASK | TYPE_MASK);

  ensure_slot_to_mol_view_synced();
  ensure_vel_buf_view_sized();    // safety; usually a no-op
  ensure_scratch_views_sized();

  // Device views.
  auto x_view      = atomKK->k_x.template view<DeviceType>();
  auto image_view  = atomKK->k_image.template view<DeviceType>();
  auto mask_view   = atomKK->k_mask.template view<DeviceType>();
  auto tag_view    = atomKK->k_tag.template view<DeviceType>();
  auto type_view   = atomKK->k_type.template view<DeviceType>();
  auto mass_view   = atomKK->k_mass.template view<DeviceType>();
  auto vbv         = vel_buf_view;
  auto stm         = slot_to_mol_view;
  auto lcom_p_d    = pass1_lcom_p_d;
  auto lcom_r_d    = pass1_lcom_r_d;

  const int natoms      = natom_buf;
  const int nlocal_loc  = atom->nlocal;
  const int groupbit_l  = groupbit;
  const size_t stm_size = slot_to_mol_view.extent(0);

  // Box geometry for on-device unmap.  Use triclinic formula uncondition-
  // ally — for orthogonal boxes xy=xz=yz=0 and it reduces to the simple
  // expression, matching domain->unmap() bit-for-bit.
  const double xprd = domain->xprd;
  const double yprd = domain->yprd;
  const double zprd = domain->zprd;
  const double xy   = domain->xy;
  const double xz   = domain->xz;
  const double yz   = domain->yz;

  // Zero the persistent per-mol scatter buffers before atomic_add.
  Kokkos::deep_copy(lcom_p_d, 0.0);
  Kokkos::deep_copy(lcom_r_d, 0.0);

  Kokkos::parallel_for("FixXPTKokkos::pass1",
      Kokkos::RangePolicy<DeviceType>(0, nlocal_loc),
      KOKKOS_LAMBDA(const int i) {
        if (!(mask_view(i) & groupbit_l)) return;
        const int s = tag_view(i) - 1;
        if (s < 0 || s >= natoms) return;
        if ((size_t)s >= stm_size) return;
        const int m = stm(s);
        if (m < 0) return;
        const double mi = mass_view(type_view(i));
        // Unmap (triclinic-safe; reduces to orthogonal when xy=xz=yz=0).
        const imageint im = image_view(i);
        const int ix = (im & IMGMASK) - IMGMAX;
        const int iy = ((im >> IMGBITS) & IMGMASK) - IMGMAX;
        const int iz = (im >> IMG2BITS) - IMGMAX;
        const double xu0 = x_view(i, 0) + ix * xprd + iy * xy + iz * xz;
        const double xu1 = x_view(i, 1) +              iy * yprd + iz * yz;
        const double xu2 = x_view(i, 2) +                          iz * zprd;
        const double v0 = vbv(ibuf_vel, s, 0);
        const double v1 = vbv(ibuf_vel, s, 1);
        const double v2 = vbv(ibuf_vel, s, 2);
        Kokkos::atomic_add(&lcom_p_d(m*3 + 0), mi * v0);
        Kokkos::atomic_add(&lcom_p_d(m*3 + 1), mi * v1);
        Kokkos::atomic_add(&lcom_p_d(m*3 + 2), mi * v2);
        Kokkos::atomic_add(&lcom_r_d(m*3 + 0), mi * xu0);
        Kokkos::atomic_add(&lcom_r_d(m*3 + 1), mi * xu1);
        Kokkos::atomic_add(&lcom_r_d(m*3 + 2), mi * xu2);
      });

  // D2H mirror, then copy out to caller buffers (nm*3 doubles each —
  // small even at 0p1M scale: 13550 * 3 * 8 = 325 KB per array).
  auto lcom_p_h = Kokkos::create_mirror_view_and_copy(
                      Kokkos::HostSpace(), lcom_p_d);
  auto lcom_r_h = Kokkos::create_mirror_view_and_copy(
                      Kokkos::HostSpace(), lcom_r_d);
  for (int k = 0; k < nm * 3; k++) {
    lcom_p[k] = lcom_p_h(k);
    lcom_r[k] = lcom_r_h(k);
  }
}

/* ---------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   compute_mol_pass2_atoms — device kernel.

   Uploads vcom/rcom (nm*3 each), then a parallel_for over nlocal atoms
   atomic_adds into lL_d (nm*3) and lI_d (nm*9) scratch.  Output mirrored
   back to host; caller does the MPI_Allreduce(SUM) onto gL/gI.  The small
   per-mol fan-in keeps atomic contention low.
---------------------------------------------------------------------- */
template<class DeviceType>
void FixXPTKokkos<DeviceType>::compute_mol_pass2_atoms(int           ibuf_vel,
                                                       const double *vcom,
                                                       const double *rcom,
                                                       double       *lL,
                                                       double       *lI)
{
  const int nm = nmol_group;
  if (nm <= 0) return;

  atomKK->sync(execution_space,
               X_MASK | IMAGE_MASK | MASK_MASK | TAG_MASK | TYPE_MASK);
  ensure_slot_to_mol_view_synced();
  ensure_vel_buf_view_sized();
  ensure_scratch_views_sized();

  // Zero the atomic-add targets and upload the latest host vcom/rcom.
  auto vcom_d = permol_vcom_d;
  auto rcom_d = permol_rcom_d;
  auto lL_d   = pass2_lL_d;
  auto lI_d   = pass2_lI_d;
  Kokkos::deep_copy(lL_d, 0.0);
  Kokkos::deep_copy(lI_d, 0.0);
  {
    Kokkos::View<const double*, Kokkos::HostSpace,
                 Kokkos::MemoryTraits<Kokkos::Unmanaged>> vcom_h(vcom, nm*3);
    Kokkos::View<const double*, Kokkos::HostSpace,
                 Kokkos::MemoryTraits<Kokkos::Unmanaged>> rcom_h(rcom, nm*3);
    Kokkos::deep_copy(vcom_d, vcom_h);
    Kokkos::deep_copy(rcom_d, rcom_h);
  }

  auto x_view      = atomKK->k_x.template view<DeviceType>();
  auto image_view  = atomKK->k_image.template view<DeviceType>();
  auto mask_view   = atomKK->k_mask.template view<DeviceType>();
  auto tag_view    = atomKK->k_tag.template view<DeviceType>();
  auto type_view   = atomKK->k_type.template view<DeviceType>();
  auto mass_view   = atomKK->k_mass.template view<DeviceType>();
  auto vbv         = vel_buf_view;
  auto stm         = slot_to_mol_view;

  const int natoms      = natom_buf;
  const int nlocal_loc  = atom->nlocal;
  const int groupbit_l  = groupbit;
  const size_t stm_size = slot_to_mol_view.extent(0);
  const double xprd = domain->xprd, yprd = domain->yprd, zprd = domain->zprd;
  const double xy   = domain->xy,   xz   = domain->xz,   yz   = domain->yz;

  Kokkos::parallel_for("FixXPTKokkos::pass2",
      Kokkos::RangePolicy<DeviceType>(0, nlocal_loc),
      KOKKOS_LAMBDA(const int i) {
        if (!(mask_view(i) & groupbit_l)) return;
        const int s = tag_view(i) - 1;
        if (s < 0 || s >= natoms) return;
        if ((size_t)s >= stm_size) return;
        const int m = stm(s);
        if (m < 0) return;
        const double mi = mass_view(type_view(i));
        const imageint im = image_view(i);
        const int ix = (im & IMGMASK) - IMGMAX;
        const int iy = ((im >> IMGBITS) & IMGMASK) - IMGMAX;
        const int iz = (im >> IMG2BITS) - IMGMAX;
        const double xu0 = x_view(i, 0) + ix * xprd + iy * xy + iz * xz;
        const double xu1 = x_view(i, 1) +              iy * yprd + iz * yz;
        const double xu2 = x_view(i, 2) +                          iz * zprd;
        const double rp0 = xu0 - rcom_d(m*3 + 0);
        const double rp1 = xu1 - rcom_d(m*3 + 1);
        const double rp2 = xu2 - rcom_d(m*3 + 2);
        const double vp0 = vbv(ibuf_vel, s, 0) - vcom_d(m*3 + 0);
        const double vp1 = vbv(ibuf_vel, s, 1) - vcom_d(m*3 + 1);
        const double vp2 = vbv(ibuf_vel, s, 2) - vcom_d(m*3 + 2);
        // L += m * r' × v'
        Kokkos::atomic_add(&lL_d(m*3 + 0), mi * (rp1*vp2 - rp2*vp1));
        Kokkos::atomic_add(&lL_d(m*3 + 1), mi * (rp2*vp0 - rp0*vp2));
        Kokkos::atomic_add(&lL_d(m*3 + 2), mi * (rp0*vp1 - rp1*vp0));
        // I += m * (|r'|² δ - r' r'^T)
        const double r2 = rp0*rp0 + rp1*rp1 + rp2*rp2;
        Kokkos::atomic_add(&lI_d(m*9 + 0), mi * (r2 - rp0*rp0));
        Kokkos::atomic_add(&lI_d(m*9 + 1), mi * (    - rp0*rp1));
        Kokkos::atomic_add(&lI_d(m*9 + 2), mi * (    - rp0*rp2));
        Kokkos::atomic_add(&lI_d(m*9 + 3), mi * (    - rp1*rp0));
        Kokkos::atomic_add(&lI_d(m*9 + 4), mi * (r2 - rp1*rp1));
        Kokkos::atomic_add(&lI_d(m*9 + 5), mi * (    - rp1*rp2));
        Kokkos::atomic_add(&lI_d(m*9 + 6), mi * (    - rp2*rp0));
        Kokkos::atomic_add(&lI_d(m*9 + 7), mi * (    - rp2*rp1));
        Kokkos::atomic_add(&lI_d(m*9 + 8), mi * (r2 - rp2*rp2));
      });

  // D2H mirror.
  auto lL_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), lL_d);
  auto lI_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), lI_d);
  for (int k = 0; k < nm * 3; k++) lL[k] = lL_h(k);
  for (int k = 0; k < nm * 9; k++) lI[k] = lI_h(k);
}

/* ---------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   compute_mol_pass3_atoms — device kernel.

   Per-atom write of v_vib,i = v_i - v_COM - ω × r'_i into vib_local.
   No atomics: each atom writes its own slot.  Off-group slots stay zero
   (deep_copy(0.0) on the device scratch matches the host zero-init).
---------------------------------------------------------------------- */
template<class DeviceType>
void FixXPTKokkos<DeviceType>::compute_mol_pass3_atoms(int           ibuf_vel,
                                                       const double *vcom,
                                                       const double *rcom,
                                                       const double *omega_lab,
                                                       double       *vib_local)
{
  const int nm     = nmol_group;
  const int natoms = natom_buf;
  if (nm <= 0 || natoms <= 0) return;

  atomKK->sync(execution_space,
               X_MASK | IMAGE_MASK | MASK_MASK | TAG_MASK);
  ensure_slot_to_mol_view_synced();
  ensure_vel_buf_view_sized();
  ensure_scratch_views_sized();

  auto vcom_d  = permol_vcom_d;
  auto rcom_d  = permol_rcom_d;
  auto omega_d = permol_omega_d;
  auto vib_d   = pass3_vib_d;
  // vib_d must be zeroed: kernel only writes slots of group atoms; off-group
  // slots stay 0 to match the CPU vib_local zero-init (caller's MPI_Allreduce
  // semantics depend on this).
  Kokkos::deep_copy(vib_d, 0.0);
  {
    Kokkos::View<const double*, Kokkos::HostSpace,
                 Kokkos::MemoryTraits<Kokkos::Unmanaged>> vcom_h(vcom,  nm*3);
    Kokkos::View<const double*, Kokkos::HostSpace,
                 Kokkos::MemoryTraits<Kokkos::Unmanaged>> rcom_h(rcom,  nm*3);
    Kokkos::View<const double*, Kokkos::HostSpace,
                 Kokkos::MemoryTraits<Kokkos::Unmanaged>> ome_h (omega_lab, nm*3);
    Kokkos::deep_copy(vcom_d, vcom_h);
    Kokkos::deep_copy(rcom_d, rcom_h);
    Kokkos::deep_copy(omega_d, ome_h);
  }

  auto x_view      = atomKK->k_x.template view<DeviceType>();
  auto image_view  = atomKK->k_image.template view<DeviceType>();
  auto mask_view   = atomKK->k_mask.template view<DeviceType>();
  auto tag_view    = atomKK->k_tag.template view<DeviceType>();
  auto vbv         = vel_buf_view;
  auto stm         = slot_to_mol_view;

  const int nlocal_loc  = atom->nlocal;
  const int groupbit_l  = groupbit;
  const size_t stm_size = slot_to_mol_view.extent(0);
  const double xprd = domain->xprd, yprd = domain->yprd, zprd = domain->zprd;
  const double xy   = domain->xy,   xz   = domain->xz,   yz   = domain->yz;

  Kokkos::parallel_for("FixXPTKokkos::pass3",
      Kokkos::RangePolicy<DeviceType>(0, nlocal_loc),
      KOKKOS_LAMBDA(const int i) {
        if (!(mask_view(i) & groupbit_l)) return;
        const int s = tag_view(i) - 1;
        if (s < 0 || s >= natoms) return;
        if ((size_t)s >= stm_size) return;
        const int m = stm(s);
        if (m < 0) return;
        const imageint im = image_view(i);
        const int ix = (im & IMGMASK) - IMGMAX;
        const int iy = ((im >> IMGBITS) & IMGMASK) - IMGMAX;
        const int iz = (im >> IMG2BITS) - IMGMAX;
        const double xu0 = x_view(i, 0) + ix * xprd + iy * xy + iz * xz;
        const double xu1 = x_view(i, 1) +              iy * yprd + iz * yz;
        const double xu2 = x_view(i, 2) +                          iz * zprd;
        const double rp0 = xu0 - rcom_d(m*3 + 0);
        const double rp1 = xu1 - rcom_d(m*3 + 1);
        const double rp2 = xu2 - rcom_d(m*3 + 2);
        const double omx = omega_d(m*3 + 0);
        const double omy = omega_d(m*3 + 1);
        const double omz = omega_d(m*3 + 2);
        const double vrot0 = omy*rp2 - omz*rp1;
        const double vrot1 = omz*rp0 - omx*rp2;
        const double vrot2 = omx*rp1 - omy*rp0;
        vib_d(s*3 + 0) = vbv(ibuf_vel, s, 0) - vcom_d(m*3 + 0) - vrot0;
        vib_d(s*3 + 1) = vbv(ibuf_vel, s, 1) - vcom_d(m*3 + 1) - vrot1;
        vib_d(s*3 + 2) = vbv(ibuf_vel, s, 2) - vcom_d(m*3 + 2) - vrot2;
      });

  auto vib_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), vib_d);
  for (int k = 0; k < natoms * 3; k++) vib_local[k] = vib_h(k);
}

/* ---------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   ensure_pwr_plan_batched

   Lazy-create a batched cuFFT Z2Z plan for N_fft × max_batch and the
   corresponding device buffers (interleaved complex input, per-batch
   weights, and the |X|² accumulator).  No-op once N_fft and max_batch
   are stable across windows.

   Returns true iff a (re)allocation actually fired this call.
---------------------------------------------------------------------- */
template<class DeviceType>
bool FixXPTKokkos<DeviceType>::ensure_pwr_plan_batched(int N_fft)
{
  const int max_batch = pwr_plan_max_batch;
  if (pwr_plan_N_fft == N_fft
      && pwr_fft_in_d.extent(0) == size_t(2) * size_t(N_fft) * size_t(max_batch)
      && pwr_accum_d.extent(0)  == size_t(N_fft)
      && pwr_weights_d.extent(0) == size_t(max_batch)) {
    return false;
  }
#ifdef KOKKOS_ENABLE_CUDA
  if constexpr (std::is_same<DeviceType, Kokkos::Cuda>::value) {
    if (pwr_plan_kk_valid) { cufftDestroy(pwr_plan_kk); pwr_plan_kk_valid = false; }
    int N = N_fft;
    cufftResult res = cufftPlanMany(&pwr_plan_kk,
        /*rank=*/1, &N,                     // 1D FFT, length N_fft
        /*inembed=*/nullptr, /*istride=*/1, /*idist=*/N_fft,
        /*onembed=*/nullptr, /*ostride=*/1, /*odist=*/N_fft,
        CUFFT_Z2Z, max_batch);
    if (res != CUFFT_SUCCESS) {
      error->one(FLERR, "fix xpt/kk: cufftPlanMany failed in Phase 4 V1 setup");
    }
    pwr_plan_kk_valid = true;
  }
#endif
  pwr_fft_in_d = Kokkos::View<double*, DeviceType>(
      Kokkos::view_alloc(Kokkos::WithoutInitializing, "fix_xpt:pwr_fft_in_d"),
      size_t(2) * size_t(N_fft) * size_t(max_batch));
  pwr_weights_d = Kokkos::View<double*, DeviceType>(
      Kokkos::view_alloc(Kokkos::WithoutInitializing, "fix_xpt:pwr_weights_d"),
      size_t(max_batch));
  pwr_accum_d = Kokkos::View<double*, DeviceType>(
      Kokkos::view_alloc(Kokkos::WithoutInitializing, "fix_xpt:pwr_accum_d"),
      size_t(N_fft));
  pwr_plan_N_fft = N_fft;
  return true;
}

/* ----------------------------------------------------------------------
   pwr_from_atoms — KOKKOS override.

   Batched cuFFT path.  Collects all (slot, dim) FFTs into a flat list,
   processes them in chunks of pwr_plan_max_batch.  Per chunk:
     1. Build interleaved complex input (re = buf[t][slot][d], im = 0,
        zero-padded to N_fft) on host, deep_copy to pwr_fft_in_d.
     2. Build per-batch weight vector on host, deep_copy to pwr_weights_d.
     3. cufftExecZ2Z in-place — N_fft × this_chunk batched FFTs.
     4. parallel_for accumulates Σ_b weight[b] · |X_b[k]|² into pwr_accum_d.
   After all chunks, D2H mirror pwr_accum_d → caller pwr (accumulated).

   Non-CUDA backends fall back to base-class FFTW3 (per-FFT).
---------------------------------------------------------------------- */
template<class DeviceType>
void FixXPTKokkos<DeviceType>::pwr_from_atoms(double ***buf,
                                              const std::vector<int> &slots,
                                              const double *mbuf,
                                              std::vector<double> &pwr,
                                              FFT_SCALAR *fft_buf,
                                              FFT3d *fft_vac, int ns)
{
#ifdef KOKKOS_ENABLE_CUDA
  if constexpr (std::is_same<DeviceType, Kokkos::Cuda>::value) {
    const int N_fft = 2 * ns;
    ensure_pwr_plan_batched(N_fft);
    Kokkos::deep_copy(pwr_accum_d, 0.0);

    // Build the (slot, d, weight) list of FFTs to perform.
    std::vector<int>    flat_slot, flat_dim;
    std::vector<double> flat_weight;
    flat_slot.reserve(3 * slots.size());
    flat_dim.reserve(3 * slots.size());
    flat_weight.reserve(3 * slots.size());
    for (int d = 0; d < 3; d++) {
      for (int s : slots) {
        const double m = mbuf[s];
        if (m <= 0.0) continue;
        flat_slot.push_back(s);
        flat_dim.push_back(d);
        flat_weight.push_back(m);
      }
    }
    const int n_active = (int)flat_slot.size();
    if (n_active == 0) return;

    const int max_batch = pwr_plan_max_batch;
    std::vector<double> host_chunk(size_t(2) * N_fft * max_batch, 0.0);
    std::vector<double> host_weights(max_batch, 0.0);
    Kokkos::View<double*, Kokkos::HostSpace,
                 Kokkos::MemoryTraits<Kokkos::Unmanaged>>
        h_chunk_view(host_chunk.data(),  size_t(2) * N_fft * max_batch);
    Kokkos::View<double*, Kokkos::HostSpace,
                 Kokkos::MemoryTraits<Kokkos::Unmanaged>>
        h_w_view(host_weights.data(), max_batch);

    auto fft_in_local  = pwr_fft_in_d;
    auto weights_local = pwr_weights_d;
    auto pwr_local     = pwr_accum_d;
    for (int chunk_start = 0; chunk_start < n_active; chunk_start += max_batch) {
      const int this_chunk = std::min(max_batch, n_active - chunk_start);
      // Clear the host staging buffer for the new chunk (only fill first
      // this_chunk batches; the tail stays zero so cuFFT computes 0 → no
      // contamination of pwr because the accum kernel only iterates over
      // this_chunk batches).
      std::fill(host_chunk.begin(), host_chunk.end(), 0.0);
      for (int b = 0; b < this_chunk; b++) {
        const int s = flat_slot[chunk_start + b];
        const int d = flat_dim [chunk_start + b];
        host_weights[b] = flat_weight[chunk_start + b];
        const size_t base = size_t(2) * N_fft * b;
        for (int t = 0; t < ns; t++) {
          host_chunk[base + 2*t]     = buf[t][s][d];
          host_chunk[base + 2*t + 1] = 0.0;
        }
      }
      Kokkos::deep_copy(pwr_fft_in_d,  h_chunk_view);
      Kokkos::deep_copy(pwr_weights_d, h_w_view);

      // Batched FFT (always runs max_batch FFTs; the tail computes from
      // zero inputs → zero output, ignored by the accum kernel below).
      cufftResult res = cufftExecZ2Z(pwr_plan_kk,
          reinterpret_cast<cufftDoubleComplex*>(pwr_fft_in_d.data()),
          reinterpret_cast<cufftDoubleComplex*>(pwr_fft_in_d.data()),
          CUFFT_FORWARD);
      if (res != CUFFT_SUCCESS) {
        error->one(FLERR, "fix xpt/kk: cufftExecZ2Z failed (pwr_from_atoms)");
      }

      // Accumulation kernel: this_chunk × N_fft work items.  Each thread
      // owns one (b, k) and atomic-adds w_b · (re² + im²) into pwr_d[k].
      const int N_fft_l   = N_fft;
      const int n_batch_l = this_chunk;
      Kokkos::parallel_for("FixXPTKokkos::pwr_accum_atoms_batched",
          Kokkos::RangePolicy<DeviceType>(0, size_t(N_fft_l) * n_batch_l),
          KOKKOS_LAMBDA(const int idx) {
            const int b = idx / N_fft_l;
            const int k = idx % N_fft_l;
            const size_t off = size_t(2) * N_fft_l * b + 2 * k;
            const double re = fft_in_local(off);
            const double im = fft_in_local(off + 1);
            Kokkos::atomic_add(&pwr_local(k),
                               weights_local(b) * (re*re + im*im));
          });
    }

    // D2H mirror final pwr_accum_d → caller pwr (accumulate).
    auto pwr_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), pwr_accum_d);
    for (int k = 0; k < N_fft; k++) pwr[k] += pwr_h(k);
    return;
  }
#endif
  // Non-CUDA backend: fall back to base-class FFTW3 path.
  FixXPT::pwr_from_atoms(buf, slots, mbuf, pwr, fft_buf, fft_vac, ns);
}

/* ----------------------------------------------------------------------
   pwr_from_mols — mol variant: indexed 0..nmol-1, weighted by wts[m_idx].
   Otherwise identical to pwr_from_atoms structure.
---------------------------------------------------------------------- */
template<class DeviceType>
void FixXPTKokkos<DeviceType>::pwr_from_mols(double ***buf, int nmol,
                                             const std::vector<double> &wts,
                                             std::vector<double> &pwr,
                                             FFT_SCALAR *fft_buf,
                                             FFT3d *fft_vac, int ns)
{
#ifdef KOKKOS_ENABLE_CUDA
  if constexpr (std::is_same<DeviceType, Kokkos::Cuda>::value) {
    const int N_fft = 2 * ns;
    ensure_pwr_plan_batched(N_fft);
    Kokkos::deep_copy(pwr_accum_d, 0.0);

    std::vector<int>    flat_mol, flat_dim;
    std::vector<double> flat_weight;
    flat_mol.reserve(3 * nmol);
    flat_dim.reserve(3 * nmol);
    flat_weight.reserve(3 * nmol);
    for (int d = 0; d < 3; d++) {
      for (int m_idx = 0; m_idx < nmol; m_idx++) {
        const double w = wts[m_idx];
        if (w <= 0.0) continue;
        flat_mol.push_back(m_idx);
        flat_dim.push_back(d);
        flat_weight.push_back(w);
      }
    }
    const int n_active = (int)flat_mol.size();
    if (n_active == 0) return;

    const int max_batch = pwr_plan_max_batch;
    std::vector<double> host_chunk(size_t(2) * N_fft * max_batch, 0.0);
    std::vector<double> host_weights(max_batch, 0.0);
    Kokkos::View<double*, Kokkos::HostSpace,
                 Kokkos::MemoryTraits<Kokkos::Unmanaged>>
        h_chunk_view(host_chunk.data(),  size_t(2) * N_fft * max_batch);
    Kokkos::View<double*, Kokkos::HostSpace,
                 Kokkos::MemoryTraits<Kokkos::Unmanaged>>
        h_w_view(host_weights.data(), max_batch);

    auto fft_in_local  = pwr_fft_in_d;
    auto weights_local = pwr_weights_d;
    auto pwr_local     = pwr_accum_d;
    for (int chunk_start = 0; chunk_start < n_active; chunk_start += max_batch) {
      const int this_chunk = std::min(max_batch, n_active - chunk_start);
      std::fill(host_chunk.begin(), host_chunk.end(), 0.0);
      for (int b = 0; b < this_chunk; b++) {
        const int m_idx = flat_mol[chunk_start + b];
        const int d     = flat_dim[chunk_start + b];
        host_weights[b] = flat_weight[chunk_start + b];
        const size_t base = size_t(2) * N_fft * b;
        for (int t = 0; t < ns; t++) {
          host_chunk[base + 2*t]     = buf[t][m_idx][d];
          host_chunk[base + 2*t + 1] = 0.0;
        }
      }
      Kokkos::deep_copy(pwr_fft_in_d,  h_chunk_view);
      Kokkos::deep_copy(pwr_weights_d, h_w_view);

      cufftResult res = cufftExecZ2Z(pwr_plan_kk,
          reinterpret_cast<cufftDoubleComplex*>(pwr_fft_in_d.data()),
          reinterpret_cast<cufftDoubleComplex*>(pwr_fft_in_d.data()),
          CUFFT_FORWARD);
      if (res != CUFFT_SUCCESS) {
        error->one(FLERR, "fix xpt/kk: cufftExecZ2Z failed (pwr_from_mols)");
      }

      const int N_fft_l   = N_fft;
      const int n_batch_l = this_chunk;
      Kokkos::parallel_for("FixXPTKokkos::pwr_accum_mols_batched",
          Kokkos::RangePolicy<DeviceType>(0, size_t(N_fft_l) * n_batch_l),
          KOKKOS_LAMBDA(const int idx) {
            const int b = idx / N_fft_l;
            const int k = idx % N_fft_l;
            const size_t off = size_t(2) * N_fft_l * b + 2 * k;
            const double re = fft_in_local(off);
            const double im = fft_in_local(off + 1);
            Kokkos::atomic_add(&pwr_local(k),
                               weights_local(b) * (re*re + im*im));
          });
    }

    auto pwr_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), pwr_accum_d);
    for (int k = 0; k < N_fft; k++) pwr[k] += pwr_h(k);
    return;
  }
#endif
  FixXPT::pwr_from_mols(buf, nmol, wts, pwr, fft_buf, fft_vac, ns);
}

/* ---------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   pick_mt_kk — return the device-side companion matching a base-class
   MultiTauStream by address.  Three streams (trans, rot, vib) each have
   a dedicated MultiTauStreamKokkos sized once per topology rebuild.
---------------------------------------------------------------------- */
template<class DeviceType>
typename FixXPTKokkos<DeviceType>::MultiTauStreamKokkos *
FixXPTKokkos<DeviceType>::pick_mt_kk(const MultiTauStream &s)
{
  if (&s == &mt_trans) return &mtk_trans;
  if (&s == &mt_rot)   return &mtk_rot;
  if (&s == &mt_vib)   return &mtk_vib;
  return nullptr;
}

/* ----------------------------------------------------------------------
   ensure_mt_stream_synced — lazy resize / fill of device-side ring,
   weights, and (optional) slots so the per-step inner-product kernel
   can read all its inputs from DeviceType.  Called once per
   multitau_stream_push entry; cheap no-op once layout is stable.
---------------------------------------------------------------------- */
template<class DeviceType>
bool FixXPTKokkos<DeviceType>::ensure_mt_stream_synced(
    const MultiTauStream &s, MultiTauStreamKokkos &kk,
    const double *weights, const std::vector<int> *slots)
{
  bool did = false;
  const int N  = s.n_units;
  const int MP = mt_MP;
  const int L  = (int)s.v_ring.size();
  if (!kk.initialized
      || kk.n_units != N || kk.MP != MP || kk.L != L) {
    kk.ring_d = Kokkos::View<double*, DeviceType>(
        Kokkos::view_alloc(Kokkos::WithoutInitializing, "fix_xpt:mt_ring_d"),
        size_t(L) * size_t(MP) * size_t(N) * 3);
    // Per-stream scratch: chunk holds kmax dot products, base_thens maps
    // team_rank=k → the offset of ring[Lev, (head-k) mod MP, 0, 0].
    kk.chunk_d = Kokkos::View<double*, DeviceType>(
        Kokkos::view_alloc(Kokkos::WithoutInitializing, "fix_xpt:mt_chunk_d"),
        size_t(MP));
    kk.base_thens_d = Kokkos::View<size_t*, DeviceType>(
        Kokkos::view_alloc(Kokkos::WithoutInitializing, "fix_xpt:mt_basethens_d"),
        size_t(MP));
    kk.n_units     = N;
    kk.MP          = MP;
    kk.L           = L;
    kk.initialized = true;
    Kokkos::deep_copy(kk.ring_d, 0.0);
    did = true;
  }

  // Weights cache.  For trans, weights are molmass (length n_units).
  // For rot, weights are 1.0 (length n_units).  For vib, weights are
  // mass_buf (length natom_buf, addressed by slot index).
  const int w_n = slots ? natom_buf : N;
  if (kk.weights_cached_n != w_n) {
    kk.weights_d = Kokkos::View<double*, DeviceType>(
        Kokkos::view_alloc(Kokkos::WithoutInitializing, "fix_xpt:mt_w_d"),
        size_t(w_n));
    kk.weights_cached_n = w_n;
    did = true;
  }
  // Always re-copy weights — mass_buf may be reassigned between windows.
  {
    Kokkos::View<const double*, Kokkos::HostSpace,
                 Kokkos::MemoryTraits<Kokkos::Unmanaged>>
        w_h(weights, w_n);
    Kokkos::deep_copy(kk.weights_d, w_h);
  }

  // Slots cache (distributed/vib path only).
  if (slots) {
    const int s_n = (int)slots->size();
    if (kk.slots_cached_n != s_n) {
      kk.slots_d = Kokkos::View<int*, DeviceType>(
          Kokkos::view_alloc(Kokkos::WithoutInitializing, "fix_xpt:mt_slots_d"),
          size_t(s_n));
      kk.slots_cached_n = s_n;
      did = true;
    }
    Kokkos::View<const int*, Kokkos::HostSpace,
                 Kokkos::MemoryTraits<Kokkos::Unmanaged>>
        s_h(slots->data(), s_n);
    Kokkos::deep_copy(kk.slots_d, s_h);
  }
  return did;
}

/* ----------------------------------------------------------------------
   multitau_stream_push — KOKKOS override.

   Matches FixXPT::multitau_stream_push; only the inner per-k weighted dot
   product moves onto the device:

   1. Update host ring[head] with v_curr.
   2. Push the same v_curr to the device-side ring[head].
   3. For k in [0, kmax): device reduction of the inner product, result
      accumulated into host c_sum[Lev][k].
   4. Cascade: when down_n hits mt_S, average and recurse on Lev+1 (the
      nested call re-enters this override since the method is virtual).
---------------------------------------------------------------------- */
template<class DeviceType>
void FixXPTKokkos<DeviceType>::multitau_stream_push(
    MultiTauStream& s, int Lev, const double* v_curr,
    const double* weights, const std::vector<int>* slots)
{
#ifdef KOKKOS_ENABLE_CUDA
  if constexpr (std::is_same<DeviceType, Kokkos::Cuda>::value) {
    if (Lev >= (int)s.c_sum.size()) return;
    const int MP = mt_MP;
    const int N  = s.n_units;
    if (N <= 0) return;
    MultiTauStreamKokkos *kk = pick_mt_kk(s);
    if (!kk) {
      // Shouldn't happen — unknown stream identity.  Fall back to host.
      FixXPT::multitau_stream_push(s, Lev, v_curr, weights, slots);
      return;
    }
    ensure_mt_stream_synced(s, *kk, weights, slots);

    // Step 1: update host ring (preserves host code path for any
    // future host consumers of v_ring; cheap relative to the dot product).
    double* ring   = s.v_ring[Lev].data();
    int     head   = s.head[Lev];
    std::copy(v_curr, v_curr + size_t(N) * 3, ring + size_t(head) * N * 3);
    s.count_seen[Lev]++;
    const int kmax = (int)std::min((long)MP, s.count_seen[Lev]);

    // Step 2: push v_curr into device ring at (Lev, head).
    {
      const size_t base = (size_t(Lev) * MP + head) * size_t(N) * 3;
      auto ring_slot = Kokkos::subview(kk->ring_d,
          std::make_pair(base, base + size_t(N) * 3));
      Kokkos::View<const double*, Kokkos::HostSpace,
                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>
          v_curr_h(v_curr, size_t(N) * 3);
      Kokkos::deep_copy(ring_slot, v_curr_h);
    }

    // Step 3: one batched TeamPolicy kernel handles all kmax inner
    // products.  team.league_rank() = k, team threads cooperatively reduce
    // the dot product over n_loop slots/units into chunk_d[k]; then D2H +
    // add to host c_sum[Lev].
    auto ring_d_local    = kk->ring_d;
    auto weights_d_local = kk->weights_d;
    auto slots_d_local   = kk->slots_d;
    auto chunk_d_local   = kk->chunk_d;
    auto base_thens_loc  = kk->base_thens_d;
    const int    N_local = N;
    const int    MP_local = MP;
    const bool   has_slots = (slots != nullptr);
    const int    n_loop = has_slots ? (int)slots->size() : N;
    const size_t base_curr = (size_t(Lev) * MP_local + head)
                             * size_t(N_local) * 3;

    // Pre-compute the per-k "base_then" offset on host and upload
    // (kmax × 8 B; tiny).
    {
      std::vector<size_t> base_thens_host(kmax);
      for (int k = 0; k < kmax; k++) {
        const int other = (head - k + MP) % MP;
        base_thens_host[k] = (size_t(Lev) * MP_local + other)
                             * size_t(N_local) * 3;
      }
      Kokkos::View<const size_t*, Kokkos::HostSpace,
                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>
          bh_h(base_thens_host.data(), kmax);
      auto bh_sub = Kokkos::subview(kk->base_thens_d,
                                    std::make_pair(0, kmax));
      Kokkos::deep_copy(bh_sub, bh_h);
    }

    using team_t = typename Kokkos::TeamPolicy<DeviceType>::member_type;
    Kokkos::TeamPolicy<DeviceType> policy(kmax, Kokkos::AUTO);
    if (has_slots) {
      Kokkos::parallel_for("FixXPTKokkos::mt_dot_team_slots", policy,
          KOKKOS_LAMBDA(const team_t &team) {
            const int k = team.league_rank();
            const size_t base_then = base_thens_loc(k);
            double acc = 0.0;
            Kokkos::parallel_reduce(
                Kokkos::TeamThreadRange(team, n_loop),
                [&](const int idx, double &sum) {
                  const int slot = slots_d_local(idx);
                  const double w = weights_d_local(slot);
                  const size_t bc = base_curr + size_t(slot) * 3;
                  const size_t bt = base_then + size_t(slot) * 3;
                  sum += w * (ring_d_local(bc + 0) * ring_d_local(bt + 0)
                            + ring_d_local(bc + 1) * ring_d_local(bt + 1)
                            + ring_d_local(bc + 2) * ring_d_local(bt + 2));
                }, acc);
            Kokkos::single(Kokkos::PerTeam(team), [&]() {
              chunk_d_local(k) = acc;
            });
          });
    } else {
      Kokkos::parallel_for("FixXPTKokkos::mt_dot_team", policy,
          KOKKOS_LAMBDA(const team_t &team) {
            const int k = team.league_rank();
            const size_t base_then = base_thens_loc(k);
            double acc = 0.0;
            Kokkos::parallel_reduce(
                Kokkos::TeamThreadRange(team, n_loop),
                [&](const int i, double &sum) {
                  const double w = weights_d_local(i);
                  const size_t bc = base_curr + size_t(i) * 3;
                  const size_t bt = base_then + size_t(i) * 3;
                  sum += w * (ring_d_local(bc + 0) * ring_d_local(bt + 0)
                            + ring_d_local(bc + 1) * ring_d_local(bt + 1)
                            + ring_d_local(bc + 2) * ring_d_local(bt + 2));
                }, acc);
            Kokkos::single(Kokkos::PerTeam(team), [&]() {
              chunk_d_local(k) = acc;
            });
          });
    }

    // D2H mirror chunk_d (just kmax doubles) and accumulate.
    auto chunk_sub_d = Kokkos::subview(kk->chunk_d,
                                       std::make_pair(0, kmax));
    auto chunk_sub_h = Kokkos::create_mirror_view_and_copy(
                          Kokkos::HostSpace(), chunk_sub_d);
    for (int k = 0; k < kmax; k++) s.c_sum[Lev][k] += chunk_sub_h(k);

    s.head[Lev] = (head + 1) % MP;

    // Step 4: cascade — stays host-driven, recursion lands back here
    // because this override is virtual.
    const int Lreq = (int)s.c_sum.size();
    if (Lev + 1 < Lreq) {
      double* acc = s.down_acc[Lev].data();
      for (int i = 0; i < N * 3; i++) acc[i] += v_curr[i];
      s.down_n[Lev]++;
      if (s.down_n[Lev] >= mt_S) {
        const double inv_S = 1.0 / (double)mt_S;
        std::vector<double> v_avg(size_t(N) * 3);
        for (int i = 0; i < N * 3; i++) v_avg[i] = acc[i] * inv_S;
        std::fill(s.down_acc[Lev].begin(), s.down_acc[Lev].end(), 0.0);
        s.down_n[Lev] = 0;
        multitau_stream_push(s, Lev + 1, v_avg.data(), weights, slots);
      }
    }
    return;
  }
#endif
  // Non-CUDA fallback: CPU base-class path.
  FixXPT::multitau_stream_push(s, Lev, v_curr, weights, slots);
}

/* ---------------------------------------------------------------------- */

/* ---------------------------------------------------------------------- */

// Explicit instantiation — required so the linker sees the class
// definitions for both device and host execution spaces.
namespace LAMMPS_NS {
template class FixXPTKokkos<LMPDeviceType>;
#ifdef LMP_KOKKOS_GPU
template class FixXPTKokkos<LMPHostType>;
#endif
}
