/* ----------------------------------------------------------------------
   pppm/samqeq/kk — device reciprocal-space potential for fix qeq/sam (Phase 3).
   Subclass of PPPMKokkos: reuse the device FFT/grid/greensfn/particle_map; add a
   group-masked make_rho + a u_brick potential projection + a small ad-poisson.
   DRAFT-1 (session 18) — written from the recon map; expect a build-fix round.
   Mirrors the CPU PPPMSamqeq::compute_vector exactly (same masking + ad-poisson).
-------------------------------------------------------------------------*/

#include "pppm_samqeq_kokkos.h"

#include "atom_kokkos.h"
#include "atom_masks.h"
#include "comm.h"       // comm->me (the #29 stage diagnostic prints on rank 0 only)
#include "domain.h"
#include "error.h"
#include "fft3d_kokkos.h"
#include "force.h"
#include "grid3d_kokkos.h"
#include "kokkos.h"
#include "update.h"

#include "utils.h"     // utils::logmesg for the #29 stage diagnostic

#include "platform.h"   // platform::walltime for the reciprocal sub-phase timers

#include <cmath>
#include <cstdlib>   // getenv/atoi (diagnostic gate)

using namespace LAMMPS_NS;

// grid-comm flags: use KSpace's own enumerators, qualified. In a class template, unqualified lookup
// does not see the dependent base's names, so a file-level enum silently wins -- and a file-level
// FORWARD_AD = 1 collides with KSpace::FORWARD_IK = 1 (kspace.h numbers FORWARD_RHO first), which is
// the flag PPPMKokkos forwards the ik E-field bricks with. The override below then packed u_brick in
// place of the field and the E-field ghost cells were never filled (wrong reciprocal forces within
// ~2 grid cells of every subdomain face; energies and charges unaffected). PPPMKokkos never uses
// FORWARD_AD itself, so it is free for the u_brick exchange.

/* ----------------------------------------------------------------------*/

template<class DeviceType>
PPPMSamqeqKokkos<DeviceType>::PPPMSamqeqKokkos(LAMMPS *lmp) : PPPMKokkos<DeviceType>(lmp)
{
  sam_compute_step = -1;
  sam_sensor_grpbit = sam_source_grpbit = 0;
  sam_invert = 0;
}

template<class DeviceType>
PPPMSamqeqKokkos<DeviceType>::~PPPMSamqeqKokkos()
{
  if (this->copymode) return;
}

/* ----------------------------------------------------------------------
   ★ RUN-BOUNDARY FIX (, device mirror of PPPMSamqeq::init(), pppm_samqeq.cpp): PPPMKokkos::init()
   (called at EVERY `run` command, same as the host PPPM::init()) deallocates + reallocates all device
   grid Views, and greensfn, the fk arrays, and vg are only recomputed by setup() (Verlet::setup runs it AFTER the fix's
   setup_pre_force). start_compute_device()'s self-heal sentinel (`sam_compute_step == -1`) was only set
   in the CONSTRUCTOR, so it only ever detected the very FIRST setup of the process -- on a 2nd+ run
   whose boundary solve is allowed to proceed (the `reset_timestep 0` production idiom, where
   ntimestep==0 masquerades as a fresh first setup and defeats the #9 boundary skip AND leaves
   sam_compute_step at the last step of the previous run, > ntimestep), compute_vector_device would run
   the ENTIRE boundary charge solve against an invalid/zeroed device greensfn -> the same near-indefinite
   operator / CG-stall / crash-lottery failure gdb-verified on the host path. init() is precisely the point where the grid is invalidated,
   so re-arm the sentinel here: the next compute_vector_device call self-heals via setup(). Byte-identical
   for single-run decks (constructor already set -1); multi-run decks get one redundant-but-identical
   setup() per run.
-------------------------------------------------------------------------*/

template<class DeviceType>
void PPPMSamqeqKokkos<DeviceType>::init()
{
  PPPMKokkos<DeviceType>::init();
  sam_compute_step = -1;

  // EW3DC slab correction: the HOST PPPMSamqeq::compute_vector adds the Yeh-Berkowitz dipole term to
  // the charge-solve potential (pppm_samqeq.cpp), but this device compute_vector_device does NOT (not
  // ported), so slab here would give a solve/force-inconsistent functional (energy nonconservation).
  // This class subclasses PPPMKokkos directly (not PPPMSamqeq), so it never inherited the host guard;
  // reject explicitly rather than run silently wrong.
  if (this->slabflag)
    this->error->all(FLERR,
                     "pppm/samqeq/kk does not support kspace_modify slab; use the host pppm/samqeq"
                     "(EW3DC supported there)");
}

/* ----------------------------------------------------------------------
   reciprocal energy/forces with qsqsum refreshed from the LIVE charges — the
   device analog of the #10 fix in pppm_samqeq.cpp (PPPMSamqeq::compute(), CPU
   side). This class subclasses PPPMKokkos directly (not PPPMSamqeq), so it does
   NOT inherit that override: PPPMKokkos<DeviceType>::compute() only re-measures
   qsum/qsqsum when the atom COUNT changes (pppm_kokkos.cpp), which is valid for
   fixed point charges but stale for fix qeq/sam's per-step fluctuating charges,
   biasing the reported reciprocal self-energy exactly like the pre-#10 CPU bug
   (see pppm_samqeq.cpp:212-224 for the full derivation; same reasoning applies
   here — the term is position-independent, so this is an energy-only fix, no
   force/dynamics impact).
   qsum_qsq() (KSpace base, kspace.cpp) reads the raw HOST atom->q pointer and is
   not Kokkos-aware, so pull q to the host first. This mirrors the explicit
   sync dance compute_vector_device() below does for x: under devsolve, device q
   is the canonical copy for the fix's own matvec and host q can be stale, so an
   unconditional sync is required here for correctness (harmless no-op otherwise).
-------------------------------------------------------------------------*/

template<class DeviceType>
void PPPMSamqeqKokkos<DeviceType>::compute(int eflag, int vflag)
{
  AtomKokkos *atomKK = (AtomKokkos *) this->atom;
  atomKK->sync(Host, Q_MASK);                        // qsum_qsq() reads atom->q on the host directly
  this->qsum_qsq(0);                                 // re-measure qsum/qsqsum from the current (solved) charges
  PPPMKokkos<DeviceType>::compute(eflag, vflag);      // stock device reciprocal solve; now uses fresh qsqsum
}

/* ----------------------------------------------------------------------
   allocate the base PPPM grid, then the electrolyte density Views + ensure
   d_u_brick exists (the ik scheme allocates it only for peratom; we always need
   it). Match the base brick extents (0-based Views; offset is done in index math).
-------------------------------------------------------------------------*/

template<class DeviceType>
void PPPMSamqeqKokkos<DeviceType>::allocate()
{
  PPPMKokkos<DeviceType>::allocate();

  const int e0 = this->d_density_brick.extent(0);
  const int e1 = this->d_density_brick.extent(1);
  const int e2 = this->d_density_brick.extent(2);
  d_electrolyte_density_brick = typename FFT_AT::t_FFT_SCALAR_3d("pppm/samqeq:e_density_brick", e0, e1, e2);
  d_electrolyte_density_fft = typename FFT_AT::t_FFT_SCALAR_1d("pppm/samqeq:e_density_fft",
                                                              this->d_density_fft.extent(0));
  // ★ (#29) BUG FIX: this used to be `if (d_u_brick.extent(0) == 0)`, i.e. "allocate only if it
  // has never been allocated". u_brick is OURS to size (the ik scheme allocates it only for peratom
  // output), and the PPPM grid is re-derived at EVERY run start from qsum_qsq -- with fluctuating FQ
  // charges the auto-tuned mesh really does change between runs. When it GREW, the base reallocated
  // d_density_brick at the new size while u_brick silently kept the OLD, SMALLER one, and the whole
  // pipeline (Work2ToU write, ProjectPsi read, the ghost pack/unpack) then indexed past its end:
  // out of bounds, no bounds checking in a release Kokkos build, results wrong and not reproducible
  // between identical runs. Measured: a deck whose mesh is pinned is idempotent across repeated
  // `run 0`s (-0.4754136 every time) while the same deck with the auto mesh drifted
  // -0.4754136 -> -0.48996403 -> -0.48880228. Size on the EXTENTS, never on emptiness.
  if ((int) this->d_u_brick.extent(0) != e0 || (int) this->d_u_brick.extent(1) != e1 ||
      (int) this->d_u_brick.extent(2) != e2)
    this->d_u_brick = typename FFT_AT::t_FFT_SCALAR_3d("pppm/samqeq:u_brick", e0, e1, e2);
  // ★ (row 183): the BASE allocate() above just re-created d_rho1d as a fresh ZERO View
  // (pppm_kokkos.cpp allocate(): `d_rho1d = t_FFT_SCALAR_2d_3("pppm:rho1d", nmax, ...)`), and init() calls
  // allocate() at EVERY `run`/`minimize`. Re-arm the position-cache stamp HERE, at the re-creation point, so
  // no caller (init(), a future setup_grid()/fix balance path) can inherit a "current" stamp over an empty
  // weight table. init() re-arms too ; this one is the structural guarantee.
  sam_compute_step = -1;
}

template<class DeviceType>
void PPPMSamqeqKokkos<DeviceType>::deallocate()
{
  PPPMKokkos<DeviceType>::deallocate();
  // Kokkos Views free themselves; nothing extra.
}

/* ----------------------------------------------------------------------
   lazy particle_map (once per timestep), mirroring CPU PPPMSamqeq::start_compute.
-------------------------------------------------------------------------*/

template<class DeviceType>
void PPPMSamqeqKokkos<DeviceType>::start_compute_device()
{
  if (sam_compute_step < (int) this->update->ntimestep) {
    if (sam_compute_step == -1) this->setup();
    // particle_map() resets boxlo_kk from the host `boxlo` member, so set BOTH (mirror PPPMKokkos::compute,
    // which our path bypasses). NVT keeps these fixed, but an NPT/box-changing run needs the per-step refresh.
    this->boxlo[0] = this->domain->boxlo[0];
    this->boxlo[1] = this->domain->boxlo[1];
    this->boxlo[2] = this->domain->boxlo[2];
    this->boxlo_kk[0] = this->domain->boxlo[0];
    this->boxlo_kk[1] = this->domain->boxlo[1];
    this->boxlo_kk[2] = this->domain->boxlo[2];
    // d_part2grid / d_rho1d are sized inside PPPMKokkos::compute() (which our path never calls),
    // so allocate them here on nmax growth — exactly as the CPU PPPMSamqeq::start_compute does.
    AtomKokkos *atomKK = (AtomKokkos *) this->atom;
    if (atomKK->nmax > this->nmax) {
      this->nmax = atomKK->nmax;
      this->d_part2grid = typename AT::t_int_1d_3("pppm/samqeq:part2grid", this->nmax);
      this->d_rho1d = typename FFT_AT::t_FFT_SCALAR_2d_3("pppm/samqeq:rho1d", this->nmax,
                                                        this->order/2 + this->order/2 + 1);
    }
    // electrolyte density Views + u_brick (the ik scheme leaves u_brick unallocated), keyed off the
    // now-allocated base d_density_brick so we don't depend on allocate() timing.
    // ★ (#29): these tests are on the EXTENTS, not on emptiness — see the allocate() comment. The
    // old `extent(0) == 0` form made the whole block a one-shot, so a grid that grew after the first
    // call left every one of these views undersized and the pipeline wrote past their ends.
    {
      const int e0 = this->d_density_brick.extent(0), e1 = this->d_density_brick.extent(1),
                e2 = this->d_density_brick.extent(2);
      if ((int) d_electrolyte_density_brick.extent(0) != e0 ||
          (int) d_electrolyte_density_brick.extent(1) != e1 ||
          (int) d_electrolyte_density_brick.extent(2) != e2)
        d_electrolyte_density_brick = typename FFT_AT::t_FFT_SCALAR_3d("pppm/samqeq:e_density_brick", e0, e1, e2);
      if (d_electrolyte_density_fft.extent(0) != this->d_density_fft.extent(0))
        d_electrolyte_density_fft = typename FFT_AT::t_FFT_SCALAR_1d("pppm/samqeq:e_density_fft",
                                                                    this->d_density_fft.extent(0));
      if ((int) this->d_u_brick.extent(0) != e0 || (int) this->d_u_brick.extent(1) != e1 ||
          (int) this->d_u_brick.extent(2) != e2)
        this->d_u_brick = typename FFT_AT::t_FFT_SCALAR_3d("pppm/samqeq:u_brick", e0, e1, e2);
    }
    this->particle_map();   // device functor TagPPPM_particle_map -> d_part2grid
    // ★★ (row 183) THE SECOND-RUN DEFECT (, TASKS 183).
    // The rho1d weights used to be filled in compute_vector_device under their OWN stamp
    // (`rho1d_step != ntimestep`, #30). PPPMKokkos::init() -- every `run`/`minimize` -- calls allocate(),
    // whose BASE re-creates d_rho1d as a fresh ZERO View, but rho1d_step kept the last step's value. A second
    // run whose setup solve happens at the SAME ntimestep as the previous device call (`run 0` twice,
    // `run 0` then `run N`, `run 0; reset_timestep 0; run`, a 0-iteration minimize then `run`; the setup
    // re-solve only happens at ntimestep == 0, fix_qeq_sam.cpp setup_resolve) therefore skipped the fill and
    // deposited NOTHING: rho_fft = 0, u = 0, psi = 0 on EVERY call of that setup solve (SAMQEQ_KKSP_DIAG).
    // The calibration then measured recip_self = erf(aR)/R of the probe pair EXACTLY (0.28070 =
    // erf(0.3*2.6019)/2.6019 on gself_cluster_energy vs 0.33830 correct; 0.35420 on recalib_on_grid_change, the
    // 3.95 % refusal), and the setup solve ran with no reciprocal at all (mesh pinned, no recalibration, no
    // guard: pe -5.5428 vs -6.2811). The host PPPMSamqeq computes rho1d inside make_rho and never had a stamp,
    // so host and device disagreed silently. The weights are position-derived exactly like d_part2grid (same
    // x, boxlo, delxinv, shift, order), so they belong under the SAME stamp: fill them here, right after the
    // map, and every re-arm of sam_compute_step (init(), allocate()) refreshes both.
    // The once-per-step saving is kept.
    this->copymode = 1;
    Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagSamqeqRho1d>(0, this->nlocal), *this);
    this->copymode = 0;
    sam_compute_step = (int) this->update->ntimestep;
  }
}

/* ----------------------------------------------------------------------
   device entry: accumulate the raw reciprocal potential of the source group's
   charges (device q) at the sensor group's atoms into d_vec[0,nlocal).
-------------------------------------------------------------------------*/

template<class DeviceType>
void PPPMSamqeqKokkos<DeviceType>::compute_vector_device(typename AT::t_kkfloat_1d d_vec,
    int sensor_grpbit, int source_grpbit, bool invert_source, bool q_on_device)
{
  AtomKokkos *atomKK = (AtomKokkos *) this->atom;

  // (#29) stage diagnostic — see the header. One integer test per call when unset.
  if (sam_diag_left < 0) {
    const char *e = getenv("SAMQEQ_KKSP_DIAG");
    sam_diag_left = e ? atoi(e) : 0;
  }
  const bool diag = (sam_diag_left > 0 && this->comm->me == 0);
  const int mapped_this_call = (sam_compute_step < (int) this->update->ntimestep) ? 1 : 0;
  // #29 probe: sum|q| as the HOST holds it ON ENTRY — i.e. the trial charges the caller just staged
  // — measured BEFORE the sync dance below, versus what the DEVICE ends up with after it.
  double q_host_in = 0.0, q_dev_in = 0.0;
  if (diag) { double *qh = this->atom->q; for (int i = 0; i < atomKK->nlocal; i++) q_host_in += fabs(qh[i]); }

  // ★ CUDA dynamics fix: particle_map() maps atom positions to the FFT grid and reads the DEVICE x.
  // The host charge-solve path (FixQEqSamKokkos::sync_before_solve) pulls atom data to the HOST each
  // step, so a plain flag-respecting sync<Device>(X) here can leave the device x stale -> particle_map
  // maps garbage -> "Out of range atoms - cannot compute PPPM" at the first dynamics step. (Invisible
  // under Kokkos Serial, where host==device.) Force the CANONICAL x into BOTH spaces: pull to host
  // (no-op if host is already canonical), mark host as the source, then push to device.
  const auto space = ExecutionSpaceFromDevice<DeviceType>::space;
  atomKK->sync(Host, X_MASK);
  atomKK->modified(Host, X_MASK);
  // ★★★ (#29) THE DEFECT — the reason every `-sf kk` production run gave wrong charges.
  // q used to get a plain flag-respecting sync here, and the flags cannot see what actually happens:
  // EVERY caller stages TRIAL charges into the host atom->q through a RAW POINTER — FixQEqSam::
  // add_reciprocal (the charge-solve matvec), calibrate_recip_self_probes, field_diag, and
  // FixQEqSamKokkos::device_add_reciprocal — and a raw write is invisible to Kokkos, so k_q is never
  // marked host-modified and the push is skipped after the first one of the timestep.
  // MEASURED (SAMQEQ_KKSP_DIAG, frozen-atom deck): within ONE timestep the host charges varied per
  // matvec (sum|q| = 153.0, 0.0, 250.9, 57.7, 482.2, ...) while the DEVICE stayed pinned at the first
  // call's 153.0. Every matvec therefore returned the SAME reciprocal potential, which makes the CG
  // operator AFFINE instead of linear: it stalls at resid/b ~5e-3 and charges blow up to ±3.5 e, while
  // total charge AND per-molecule nets stay perfect (the projector is fine) — the combination that
  // made this look like a solver or shielding-kernel problem for months. Step 0 looked exact only
  // because the first push of a step is the one that does happen.
  // Host q is canonical HERE BY CONSTRUCTION (every entry point stages it), so declare it and push.
  // ⚠ If an all-device path ever stages trial charges into the DEVICE q instead, it must mark
  // modified(Device, Q_MASK) and this line must become conditional; no caller does that today.
  // (#30): when the caller stages trial charges into the DEVICE q (the all-device reciprocal in
  // FixQEqSamKokkos::device_add_reciprocal) it is canonical and we must keep our hands off — pushing
  // the host's physical charges over it is #29 in mirror image, and is exactly what killed the first
  // attempt at this path ("the bespoke device-q stash produced a wrong reciprocal", 3b-DIAG 0.42).
  if (!q_on_device) atomKK->modified(Host, Q_MASK);
  atomKK->sync(space, X_MASK | (q_on_device ? 0 : Q_MASK) | MASK_MASK);
  this->x = atomKK->k_x.template view<DeviceType>();
  this->q = atomKK->k_q.template view<DeviceType>();
  d_mask_kk = atomKK->k_mask.template view<DeviceType>();
  this->nlocal = atomKK->nlocal;
  sam_sensor_grpbit = sensor_grpbit;
  sam_source_grpbit = source_grpbit;
  sam_invert = invert_source ? 1 : 0;
  d_psi = d_vec;

  if (diag) {   // what the DEVICE holds after the sync dance = what make_rho will actually use
    auto hq = Kokkos::create_mirror_view_and_copy(LMPHostType(), atomKK->k_q.template view<DeviceType>());
    for (int i = 0; i < atomKK->nlocal; i++) q_dev_in += fabs((double) hq(i));
    double *qh2 = this->atom->q;   // and whether the dance overwrote the host's trial charges
    double q_host_after = 0.0;
    for (int i = 0; i < atomKK->nlocal; i++) q_host_after += fabs(qh2[i]);
    utils::logmesg(this->lmp, "KKSP_QDIAG call={} step={} q_host_in={:.10e} q_host_after={:.10e} q_dev={:.10e}\n",
                   sam_diag_call + 1, (long) this->update->ntimestep, q_host_in, q_host_after, q_dev_in);
  }

  if (tk_on < 0) { const char *e = getenv("SAMQEQ_KK_TIME"); tk_on = (e && atoi(e)) ? 1 : 0; }
  const bool tkon = (tk_on == 1);
  double tk0 = 0.0;
  if (tkon) { Kokkos::fence(); tk0 = platform::walltime(); tk_calls++; }

  start_compute_device();   // particle_map + rho1d weights, once per step or after a re-arm (row 183)

  // (1) masked make_rho -> electrolyte density brick
  this->copymode = 1;
  Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagSamqeqMakeRhoZero>(0,
                       d_electrolyte_density_brick.size()), *this);
  Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagSamqeqMakeRhoMasked>(0, this->nlocal), *this);
  this->copymode = 0;

  if (tkon) { Kokkos::fence(); const double t = platform::walltime(); tk_rho += t - tk0; tk0 = t; }
  // (2) borrow brick2fft via the electrolyte handles (reverse_comm packs d_density_brick)
  auto save_brick = this->d_density_brick;
  auto save_fft = this->d_density_fft;
  this->d_density_brick = d_electrolyte_density_brick;
  this->d_density_fft = d_electrolyte_density_fft;
  this->gc->reverse_comm(Grid3d::KSPACE, this, KSpace::REVERSE_RHO, 1, sizeof(FFT_SCALAR),
                         this->k_gc_buf1, this->k_gc_buf2, MPI_FFT_SCALAR);
  this->brick2fft();
  this->d_density_brick = save_brick;
  this->d_density_fft = save_fft;

  if (tkon) { Kokkos::fence(); const double t = platform::walltime(); tk_b2fft += t - tk0; tk0 = t; }
  // (3) ad-poisson: work1 <- e_density_fft; FFT; xgreensfn; iFFT; -> u_brick
  this->copymode = 1;
  Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagSamqeqFillWork1>(0, this->nfft), *this);
  this->copymode = 0;
  this->fft1->compute(this->d_work1, this->d_work1, FFT3dKokkos<DeviceType>::BACKWARD);  // CPU used -1
  this->copymode = 1;
  Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagSamqeqMulGreens>(0, this->nfft), *this);
  this->copymode = 0;
  this->fft2->compute(this->d_work2, this->d_work2, FFT3dKokkos<DeviceType>::FORWARD);   // CPU used +1
  // numx/y/z_inout are set inside PPPMKokkos::poisson_ik (which we don't call) -> set them here
  // (= the inner-brick extents) so TagSamqeqWork2ToU indexes identically to TagPPPM_poisson_ik6.
  this->numz_inout = this->nzhi_in - this->nzlo_in + 1;
  this->numy_inout = this->nyhi_in - this->nylo_in + 1;
  this->numx_inout = this->nxhi_in - this->nxlo_in + 1;
  const int ninner = this->numz_inout * this->numy_inout * this->numx_inout;
  this->copymode = 1;
  Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagSamqeqWork2ToU>(0, ninner), *this);
  this->copymode = 0;

  if (tkon) { Kokkos::fence(); const double t = platform::walltime(); tk_fft += t - tk0; tk0 = t; }
  // (4) forward-comm u_brick (FORWARD_AD = our override) + project to sensor atoms
  this->gc->forward_comm(Grid3d::KSPACE, this, KSpace::FORWARD_AD, 1, sizeof(FFT_SCALAR),
                         this->k_gc_buf1, this->k_gc_buf2, MPI_FFT_SCALAR);
  if (tkon) { Kokkos::fence(); const double t = platform::walltime(); tk_ucomm += t - tk0; tk0 = t; }
  this->copymode = 1;
  Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagSamqeqProjectPsi>(0, this->nlocal), *this);
  this->copymode = 0;
  if (tkon) { Kokkos::fence(); tk_proj += platform::walltime() - tk0; }

  if (diag) {
    sam_diag_left--;
    const double s_rhofft = sam_sum_abs_1d(d_electrolyte_density_fft, this->nfft);
    const double s_uin  = sam_sum_abs_3d(this->d_u_brick,
                                         this->nzlo_in - this->nzlo_out, this->nzhi_in - this->nzlo_out + 1,
                                         this->nylo_in - this->nylo_out, this->nyhi_in - this->nylo_out + 1,
                                         this->nxlo_in - this->nxlo_out, this->nxhi_in - this->nxlo_out + 1);
    const double s_uall = sam_sum_abs_3d(this->d_u_brick, 0, (int) this->d_u_brick.extent(0),
                                         0, (int) this->d_u_brick.extent(1),
                                         0, (int) this->d_u_brick.extent(2));
    const double s_psi  = sam_sum_abs_kk(d_psi, this->nlocal);
    utils::logmesg(this->lmp,
                   "KKSP_DIAG call={} step={} mapped={} grid={}x{}x{} out=[{},{}]x[{},{}]x[{},{}]"
                   "ubrick_ext={}x{}x{} | rho_fft={:.10e} u_in={:.10e} u_ghost={:.10e} psi={:.10e}\n",
                   ++sam_diag_call, (long) this->update->ntimestep, mapped_this_call,
                   this->nx_pppm, this->ny_pppm, this->nz_pppm,
                   this->nxlo_out, this->nxhi_out, this->nylo_out, this->nyhi_out,
                   this->nzlo_out, this->nzhi_out,
                   (int) this->d_u_brick.extent(0), (int) this->d_u_brick.extent(1),
                   (int) this->d_u_brick.extent(2),
                   s_rhofft, s_uin, s_uall - s_uin, s_psi);
  }
}

/* ---- (#29) stage-diagnostic reductions (see the header). Each hosts an extended device
   lambda that captures only its own arguments, so no copymode dance is needed. ----*/

template<class DeviceType>
double PPPMSamqeqKokkos<DeviceType>::sam_sum_abs_1d(typename FFT_AT::t_FFT_SCALAR_1d v, int n) const
{
  double s = 0.0;
  auto vv = v;
  Kokkos::parallel_reduce(n, KOKKOS_LAMBDA(const int i, double &t) { t += fabs((double) vv[i]); }, s);
  return s;
}

template<class DeviceType>
double PPPMSamqeqKokkos<DeviceType>::sam_sum_abs_3d(typename FFT_AT::t_FFT_SCALAR_3d v, int z0, int z1,
                                                    int y0, int y1, int x0, int x1) const
{
  double s = 0.0;
  auto vv = v;
  const int nx = x1 - x0, ny = y1 - y0, nz = z1 - z0;
  if (nx <= 0 || ny <= 0 || nz <= 0) return 0.0;
  Kokkos::parallel_reduce(nx * ny * nz, KOKKOS_LAMBDA(const int n, double &t) {
    const int k = n / (nx * ny);
    const int j = (n - k * nx * ny) / nx;
    const int i = n - k * nx * ny - j * nx;
    t += fabs((double) vv(z0 + k, y0 + j, x0 + i));
  }, s);
  return s;
}

template<class DeviceType>
double PPPMSamqeqKokkos<DeviceType>::sam_sum_abs_kk(typename AT::t_kkfloat_1d v, int n) const
{
  double s = 0.0;
  auto vv = v;
  Kokkos::parallel_reduce(n, KOKKOS_LAMBDA(const int i, double &t) { t += fabs((double) vv(i)); }, s);
  return s;
}

/* ---- host convenience wrapper (ACCUMULATES into vec) ----*/

template<class DeviceType>
void PPPMSamqeqKokkos<DeviceType>::compute_vector(double *vec, int sensor_grpbit,
    int source_grpbit, bool invert_source)
{
  AtomKokkos *atomKK = (AtomKokkos *) this->atom;
  const int nmax = atomKK->nmax;
  if ((int) d_psi.extent(0) < nmax) d_psi = typename AT::t_kkfloat_1d("pppm/samqeq:psi", nmax);
  auto h_psi = Kokkos::create_mirror_view(d_psi);
  for (int i = 0; i < atomKK->nlocal; i++) h_psi(i) = vec[i];   // accumulate-in
  Kokkos::deep_copy(d_psi, h_psi);
  compute_vector_device(d_psi, sensor_grpbit, source_grpbit, invert_source);
  Kokkos::deep_copy(h_psi, d_psi);
  for (int i = 0; i < atomKK->nlocal; i++) vec[i] = h_psi(i);
}

/* ---- grid forward-comm override: add the u_brick (FORWARD_AD) case ----*/

template<class DeviceType>
void PPPMSamqeqKokkos<DeviceType>::pack_forward_grid_kokkos(int flag,
    FFT_DAT::tdual_FFT_SCALAR_1d &k_buf, int nlist, DAT::tdual_int_2d_lr &k_list, int index)
{
  if (flag != KSpace::FORWARD_AD) { PPPMKokkos<DeviceType>::pack_forward_grid_kokkos(flag, k_buf, nlist, k_list, index); return; }
  typename AT::t_int_2d_lr_um d_list = k_list.template view<DeviceType>();
  this->d_list_index = Kokkos::subview(d_list, index, Kokkos::ALL());
  this->d_buf = k_buf.template view<DeviceType>();
  this->nx = (this->nxhi_out - this->nxlo_out + 1);
  this->ny = (this->nyhi_out - this->nylo_out + 1);
  this->copymode = 1;
  Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagSamqeqPackU>(0, nlist), *this);
  this->copymode = 0;
}

template<class DeviceType>
void PPPMSamqeqKokkos<DeviceType>::unpack_forward_grid_kokkos(int flag,
    FFT_DAT::tdual_FFT_SCALAR_1d &k_buf, int offset, int nlist, DAT::tdual_int_2d_lr &k_list, int index)
{
  if (flag != KSpace::FORWARD_AD) { PPPMKokkos<DeviceType>::unpack_forward_grid_kokkos(flag, k_buf, offset, nlist, k_list, index); return; }
  typename AT::t_int_2d_lr_um d_list = k_list.template view<DeviceType>();
  this->d_list_index = Kokkos::subview(d_list, index, Kokkos::ALL());
  this->d_buf = k_buf.template view<DeviceType>();
  this->unpack_offset = offset;
  this->nx = (this->nxhi_out - this->nxlo_out + 1);
  this->ny = (this->nyhi_out - this->nylo_out + 1);
  this->copymode = 1;
  Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagSamqeqUnpackU>(0, nlist), *this);
  this->copymode = 0;
}

/* ---- self-contained device compute_rho1d (Horner; fills d_rho1d(i,:,)) ----*/

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void PPPMSamqeqKokkos<DeviceType>::sam_compute_rho1d(const int i, const FFT_SCALAR &dx,
    const FFT_SCALAR &dy, const FFT_SCALAR &dz) const
{
  const int ord = this->order;
  for (int k = (1-ord)/2; k <= ord/2; k++) {
    FFT_SCALAR r1 = 0, r2 = 0, r3 = 0;
    for (int l = ord-1; l >= 0; l--) {
      r1 = this->d_rho_coeff(l, k-(1-ord)/2) + r1*dx;
      r2 = this->d_rho_coeff(l, k-(1-ord)/2) + r2*dy;
      r3 = this->d_rho_coeff(l, k-(1-ord)/2) + r3*dz;
    }
    this->d_rho1d(i, k+ord/2, 0) = r1;
    this->d_rho1d(i, k+ord/2, 1) = r2;
    this->d_rho1d(i, k+ord/2, 2) = r3;
  }
}

/* ======================= device functors =======================*/

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void PPPMSamqeqKokkos<DeviceType>::operator()(TagSamqeqMakeRhoZero, const int &i) const
{
  d_electrolyte_density_brick.data()[i] = 0.0;
}

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void PPPMSamqeqKokkos<DeviceType>::operator()(TagSamqeqRho1d, const int &i) const
{
  const int nx = this->d_part2grid(i, 0), ny = this->d_part2grid(i, 1), nz = this->d_part2grid(i, 2);
  const FFT_SCALAR dx = (FFT_SCALAR)((KK_FLOAT) nx + this->shiftone_kk - (this->x(i,0) - this->boxlo_kk[0]) * this->delxinv_kk);
  const FFT_SCALAR dy = (FFT_SCALAR)((KK_FLOAT) ny + this->shiftone_kk - (this->x(i,1) - this->boxlo_kk[1]) * this->delyinv_kk);
  const FFT_SCALAR dz = (FFT_SCALAR)((KK_FLOAT) nz + this->shiftone_kk - (this->x(i,2) - this->boxlo_kk[2]) * this->delzinv_kk);
  sam_compute_rho1d(i, dx, dy, dz);
}

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void PPPMSamqeqKokkos<DeviceType>::operator()(TagSamqeqMakeRhoMasked, const int &i) const
{
  const bool in_src = ((d_mask_kk(i) & sam_source_grpbit) ? 1 : 0) != sam_invert;
  if (!in_src) return;

  Kokkos::View<FFT_SCALAR ***, Kokkos::LayoutRight, typename KKDevice<DeviceType>::value,
               Kokkos::MemoryTraits<Kokkos::Atomic | Kokkos::Unmanaged>> a_brick = d_electrolyte_density_brick;

  int nx = this->d_part2grid(i, 0);
  int ny = this->d_part2grid(i, 1);
  int nz = this->d_part2grid(i, 2);
  // (#30): d_rho1d was filled for this step by TagSamqeqRho1d — do not recompute per call
  nz -= this->nzlo_out; ny -= this->nylo_out; nx -= this->nxlo_out;

  const FFT_SCALAR z0 = (FFT_SCALAR)(this->delvolinv_kk * this->q[i]);
  for (int n = this->nlower; n <= this->nupper; n++) {
    const int mz = n + nz;
    const FFT_SCALAR y0 = z0 * this->d_rho1d(i, n + this->order/2, 2);
    for (int m = this->nlower; m <= this->nupper; m++) {
      const int my = m + ny;
      const FFT_SCALAR x0 = y0 * this->d_rho1d(i, m + this->order/2, 1);
      for (int l = this->nlower; l <= this->nupper; l++) {
        const int mx = l + nx;
        a_brick(mz, my, mx) += x0 * this->d_rho1d(i, l + this->order/2, 0);
      }
    }
  }
}

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void PPPMSamqeqKokkos<DeviceType>::operator()(TagSamqeqFillWork1, const int &i) const
{
  this->d_work1[2*i] = d_electrolyte_density_fft[i];
  this->d_work1[2*i+1] = (FFT_SCALAR) 0.0;
}

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void PPPMSamqeqKokkos<DeviceType>::operator()(TagSamqeqMulGreens, const int &i) const
{
  this->d_work2[2*i]   = this->d_work1[2*i]   * this->d_greensfn[i];
  this->d_work2[2*i+1] = this->d_work1[2*i+1] * this->d_greensfn[i];
}

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void PPPMSamqeqKokkos<DeviceType>::operator()(TagSamqeqWork2ToU, const int &ii) const
{
  const int n = ii * 2;
  int k = ii / (this->numy_inout * this->numx_inout);
  int j = (ii - k*this->numy_inout*this->numx_inout) / this->numx_inout;
  int i = ii - k*this->numy_inout*this->numx_inout - j*this->numx_inout;
  k += this->nzlo_in - this->nzlo_out;
  j += this->nylo_in - this->nylo_out;
  i += this->nxlo_in - this->nxlo_out;
  this->d_u_brick(k, j, i) = this->d_work2[n];
}

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void PPPMSamqeqKokkos<DeviceType>::operator()(TagSamqeqProjectPsi, const int &i) const
{
  if (!(d_mask_kk(i) & sam_sensor_grpbit)) return;
  const KK_FLOAT scaleinv = 1.0 / ((KK_FLOAT) this->nx_pppm * this->ny_pppm * this->nz_pppm);
  int nx = this->d_part2grid(i, 0);
  int ny = this->d_part2grid(i, 1);
  int nz = this->d_part2grid(i, 2);
  // (#30): d_rho1d is filled once per step for ALL local atoms (TagSamqeqRho1d), so the sensor
  // atoms' weights are already there — this used to recompute them on every call, a second time.
  nz -= this->nzlo_out; ny -= this->nylo_out; nx -= this->nxlo_out;
  KK_FLOAT v = 0.0;
  for (int n = this->nlower; n <= this->nupper; n++) {
    const int mz = n + nz;
    const FFT_SCALAR z0 = this->d_rho1d(i, n + this->order/2, 2);
    for (int m = this->nlower; m <= this->nupper; m++) {
      const int my = m + ny;
      const FFT_SCALAR y0 = z0 * this->d_rho1d(i, m + this->order/2, 1);
      for (int l = this->nlower; l <= this->nupper; l++) {
        const int mx = l + nx;
        const FFT_SCALAR x0 = y0 * this->d_rho1d(i, l + this->order/2, 0);
        v += x0 * this->d_u_brick(mz, my, mx);
      }
    }
  }
  d_psi(i) += v * scaleinv;
}

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void PPPMSamqeqKokkos<DeviceType>::operator()(TagSamqeqPackU, const int &i) const
{
  const double dlist = (double) this->d_list_index[i];
  const int iz = (int)(dlist / (this->nx * this->ny));
  const int iy = (int)((dlist - iz*this->nx*this->ny) / this->nx);
  const int ix = this->d_list_index[i] - iz*this->nx*this->ny - iy*this->nx;
  this->d_buf[i] = this->d_u_brick(iz, iy, ix);
}

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void PPPMSamqeqKokkos<DeviceType>::operator()(TagSamqeqUnpackU, const int &i) const
{
  const double dlist = (double) this->d_list_index[i];
  const int iz = (int)(dlist / (this->nx * this->ny));
  const int iy = (int)((dlist - iz*this->nx*this->ny) / this->nx);
  const int ix = this->d_list_index[i] - iz*this->nx*this->ny - iy*this->nx;
  this->d_u_brick(iz, iy, ix) = this->d_buf[i + this->unpack_offset];
}

/* ----------------------------------------------------------------------*/

namespace LAMMPS_NS {
template class PPPMSamqeqKokkos<LMPDeviceType>;
#ifdef LMP_KOKKOS_GPU
template class PPPMSamqeqKokkos<LMPHostType>;
#endif
}
