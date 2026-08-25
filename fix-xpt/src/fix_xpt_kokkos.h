/* -*- c++ -*- ----------------------------------------------------------
   fix_xpt_kokkos.h  —  KOKKOS-accelerated derived class of fix_xpt.

   Registers `xpt/kk`, `xpt/kk/device`, and `xpt/kk/host`.  Builds only
   when the LAMMPS KOKKOS package is installed; active under `-sf kk` or an
   explicit `fix … xpt/kk …`, otherwise the plain `xpt` CPU FixXPT is used.

   Note: bit-equivalence with the CPU path is NOT expected (mixed precision
   + reduction order).  Targets: <1 % on thermo (D, f, S, A, μ, ZPE, E);
   ns-scale convergence on Cv.
------------------------------------------------------------------------ */

#ifdef FIX_CLASS
// clang-format off
FixStyle(xpt/kk,FixXPTKokkos<LMPDeviceType>);
FixStyle(xpt/kk/device,FixXPTKokkos<LMPDeviceType>);
FixStyle(xpt/kk/host,FixXPTKokkos<LMPHostType>);
// clang-format on
#else

#ifndef LMP_FIX_XPT_KOKKOS_H
#define LMP_FIX_XPT_KOKKOS_H

#include "fix_xpt.h"
#include "kokkos_type.h"
#ifdef KOKKOS_ENABLE_CUDA
#  include <cufft.h>             // batched cuFFT (CUDA only)
#endif

namespace LAMMPS_NS {

template<class DeviceType>
class FixXPTKokkos : public FixXPT {
 public:
  typedef DeviceType device_type;
  typedef ArrayTypes<DeviceType> AT;

  // 3D row-major View for the [iframe][slot][3] velocity buffer.
  // LayoutRight matches LAMMPS' vel_buf[i][j][k] indexing.
  typedef Kokkos::View<double***, Kokkos::LayoutRight, DeviceType> t_vel3d;

  FixXPTKokkos(class LAMMPS *, int, char **);
  ~FixXPTKokkos() override;

  // Lifecycle overrides (currently pass through to the CPU base class).
  void init() override;
  void setup(int) override;
  void end_of_step() override;

  // GPU velocity gather: device gather → D2H + MPI_Allreduce(SUM) into the
  // base host vel_buf, then mirror the frame into vel_buf_view.
  void push_velocity_frame(int ibuf) override;

  // Device Pass 1 (COM momentum + mass-weighted position): atomic_add into
  // device scratch, D2H back into lcom_p/lcom_r; host MPI_Allreduce stays in
  // the base class.
  void compute_mol_pass1_atoms(int ibuf_vel,
                               double *lcom_p,
                               double *lcom_r) override;

  // Device Pass 2: atomic_add of L (nm*3) and I (nm*9) → D2H to caller.
  void compute_mol_pass2_atoms(int           ibuf_vel,
                               const double *vcom,
                               const double *rcom,
                               double       *lL,
                               double       *lI) override;

  // Device Pass 3: per-atom v_vib,i write into vib_local (no atomics),
  // D2H to caller.
  void compute_mol_pass3_atoms(int           ibuf_vel,
                               const double *vcom,
                               const double *rcom,
                               const double *omega_lab,
                               double       *vib_local) override;

  // End-of-window forward FFT.  Batched cuFFT on CUDA builds (chunked at
  // pwr_plan_max_batch), accumulating Σ_b w_b·|X_b[k]|²; non-CUDA backends
  // fall back to FixXPT::pwr_from_*.
  void pwr_from_atoms(double ***buf, const std::vector<int> &slots,
                      const double *mbuf, std::vector<double> &pwr,
                      FFT_SCALAR *fft_buf, FFT3d *fft_vac, int ns) override;
  void pwr_from_mols(double ***buf, int nmol,
                     const std::vector<double> &wts,
                     std::vector<double> &pwr,
                     FFT_SCALAR *fft_buf, FFT3d *fft_vac, int ns) override;

  // Multi-tau inner-product kernel on device (per-k weighted dot product via
  // parallel_reduce); cascade orchestration stays host-side.  Non-CUDA
  // backends fall through to FixXPT::multitau_stream_push.
  void multitau_stream_push(MultiTauStream& s, int Lev,
                            const double* v_curr,
                            const double* weights,
                            const std::vector<int>* slots) override;

 protected:
  // Device velocity buffer; lazily resized in end_of_step() to match the
  // base class's host vel_buf.
  t_vel3d vel_buf_view;

  // Lazy (re)allocate vel_buf_view to (frames, natom_buf, 3); no-op if
  // extents match, returns true on (re)alloc.  A buffer-sharing consumer
  // aliases the owner's view instead of allocating its own.
  bool ensure_vel_buf_view_sized();

  // Also re-alias a consumer's device vel_buf_view when the owner
  // (re)allocates; falls back to base behavior for mixed-style sharing.
  void sync_buffer_pointers_from_owner() override;

  // Device mirror of base::slot_to_mol for the Pass 1 kernel; re-uploaded
  // lazily when the base topology generation changes.
  Kokkos::View<int*, DeviceType> slot_to_mol_view;
  unsigned int                    slot_to_mol_view_generation_cached;
  // Refresh slot_to_mol_view if the base generation bumped; true on update.
  bool ensure_slot_to_mol_view_synced();

  // Persistent device scratch for the Pass 1/2/3 kernels; allocated once,
  // resized lazily when nmol_group or natom_buf changes.
  //
  //   pass1_lcom_p_d, pass1_lcom_r_d  — Pass 1 per-mol scatter (nm*3)
  //   pass2_lL_d,    pass2_lI_d       — Pass 2 angular momentum + inertia (nm*3, nm*9)
  //   permol_vcom_d, permol_rcom_d,
  //   permol_omega_d                  — host→device staging (nm*3 each)
  //   pass3_vib_d                     — Pass 3 vib-velocity write (natom_buf*3)
  //   push_vel_local_d                — per-step vel gather (natom_buf*3)
  //   push_mass_local_d               — per-step mass gather (natom_buf)
  Kokkos::View<double*, DeviceType> pass1_lcom_p_d, pass1_lcom_r_d;
  Kokkos::View<double*, DeviceType> pass2_lL_d,     pass2_lI_d;
  Kokkos::View<double*, DeviceType> permol_vcom_d,  permol_rcom_d, permol_omega_d;
  Kokkos::View<double*, DeviceType> pass3_vib_d;
  Kokkos::View<double*, DeviceType> push_vel_local_d, push_mass_local_d;
  int                                scratch_nm_cached;
  int                                scratch_natoms_cached;
  // Resize the scratch Views to current nmol_group/natom_buf if changed.
  // Allocated withoutInitializing (callers zero before atomic_add use).
  bool ensure_scratch_views_sized();

  // Batched cuFFT plan + persistent device input buffer.  Each pwr_from_*
  // call processes all (slot, dim) FFTs in chunks of pwr_plan_max_batch,
  // bounding the device buffer to ~1-2 GB (2·N_fft·max_batch·8B).
  // Lazy-created on first pwr_from_* call, destroyed in dtor.
#ifdef KOKKOS_ENABLE_CUDA
  cufftHandle pwr_plan_kk;
  bool        pwr_plan_kk_valid;
#endif
  int                                pwr_plan_N_fft;    // current plan N_fft
  int                                pwr_plan_max_batch;
  Kokkos::View<double*, DeviceType>  pwr_fft_in_d;      // interleaved complex (re,im)
  Kokkos::View<double*, DeviceType>  pwr_weights_d;     // per-batch weights (mass or wts)
  Kokkos::View<double*, DeviceType>  pwr_accum_d;       // |X|² accumulator, length N_fft

  // Lazy-create / resize the batched cuFFT plan + device buffers.
  // No-op once N_fft is stable across windows.
  bool ensure_pwr_plan_batched(int N_fft);

  // Device companion to MultiTauStream: mirrors the ring + cached
  // weights/slots so the inner-product kernel reads everything on-device.
  // Cascade orchestration + c_sum accumulation stay host-side.
  struct MultiTauStreamKokkos {
    int  n_units;
    int  MP;
    int  L;
    bool initialized;
    Kokkos::View<double*, DeviceType> ring_d;        // L*MP*n_units*3 flat
    Kokkos::View<double*, DeviceType> weights_d;     // n_units (or natom_buf for slots path)
    Kokkos::View<int*,    DeviceType> slots_d;       // optional (vib path)
    // Per-(stream) scratch reused across all push calls.
    Kokkos::View<double*,  DeviceType> chunk_d;      // [MP] — kmax dot-products / call
    Kokkos::View<size_t*,  DeviceType> base_thens_d; // [MP] — per-k ring offset
    int  weights_cached_n;
    int  slots_cached_n;
    MultiTauStreamKokkos() :
        n_units(0), MP(0), L(0), initialized(false),
        weights_cached_n(-1), slots_cached_n(-1) {}
  };
  MultiTauStreamKokkos mtk_trans, mtk_rot, mtk_vib;

  // Return the device-side companion matching a host MultiTauStream.
  // Identification by address — &s == &mt_{trans,rot,vib}.
  MultiTauStreamKokkos *pick_mt_kk(const MultiTauStream &s);

  // Lazy resize / fill the device mirrors when host stream layout changes.
  bool ensure_mt_stream_synced(const MultiTauStream &s,
                               MultiTauStreamKokkos &kk,
                               const double *weights,
                               const std::vector<int> *slots);
};

}    // namespace LAMMPS_NS

#endif    // LMP_FIX_XPT_KOKKOS_H
#endif    // FIX_CLASS
