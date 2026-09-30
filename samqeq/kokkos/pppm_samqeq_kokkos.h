/* -*- c++ -*- ----------------------------------------------------------
   pppm/samqeq/kk — KOKKOS device version of pppm/samqeq.

   The per-atom reciprocal-space Coulomb potential (compute_vector) on the GPU,
   so the lr_ewald>0 charge solve can run the reciprocal in the device matvec
   instead of round-tripping to the host PPPMSamqeq every iteration.

   Reuse > rewrite: subclass PPPMKokkos<DeviceType> and inherit its entire device
   FFT/grid/greensfn/particle_map/brick2fft pipeline (all protected). We add only
   (1) electrolyte (source-group) density Views and (2) two cloned device functors
   — a GROUP-MASKED make_rho and a POTENTIAL projection (u_brick -> per-atom vec) —
   plus a small device poisson (work1->FFT->xgreensfn->FFT->u_brick; NOTE PPPMKokkos
   only ships the ik/gradient poisson, so we do the ad/potential poisson ourselves,
   exactly as the CPU PPPMSamqeq does). PPPMKokkos's grid forward_comm only packs the
   ik E-field bricks (FORWARD_IK), so we override the grid pack/unpack to comm u_brick
   (a 1-value FORWARD_AD-style exchange). Kokkos Views zero-initialise, so no grid
   buffer is read uninitialised.
------------------------------------------------------------------------*/

#ifdef KSPACE_CLASS
// clang-format off
KSpaceStyle(pppm/samqeq/kk,PPPMSamqeqKokkos<LMPDeviceType>);
KSpaceStyle(pppm/samqeq/kk/device,PPPMSamqeqKokkos<LMPDeviceType>);
KSpaceStyle(pppm/samqeq/kk/host,PPPMSamqeqKokkos<LMPHostType>);
// clang-format on
#else

#ifndef LMP_PPPM_SAMQEQ_KOKKOS_H
#define LMP_PPPM_SAMQEQ_KOKKOS_H

#include "pppm_kokkos.h"
#include "samqeq_kspace.h"

namespace LAMMPS_NS {

// device functors (samqeq-specific; distinct tags from PPPMKokkos's)
struct TagSamqeqMakeRhoZero{};      // zero the electrolyte density brick
struct TagSamqeqMakeRhoMasked{};    // make_rho restricted to the source group (atomic accumulate)
struct TagSamqeqFillWork1{};        // electrolyte_density_fft -> work1 (real, imag=0)
struct TagSamqeqMulGreens{};        // work2 = work1 * greensfn
struct TagSamqeqWork2ToU{};         // work2 (real) -> u_brick (inner grid)
struct TagSamqeqProjectPsi{};       // interpolate u_brick -> per-atom potential vec (sensor group)
struct TagSamqeqRho1d{};            // fill d_rho1d for all local atoms, ONCE per step
struct TagSamqeqPackU{};            // grid forward-comm pack of u_brick
struct TagSamqeqUnpackU{};          // grid forward-comm unpack of u_brick

template<class DeviceType>
class PPPMSamqeqKokkos : public PPPMKokkos<DeviceType>, public SamqeqKspace {
 public:
  typedef DeviceType device_type;
  typedef ArrayTypes<DeviceType> AT;
  typedef FFTArrayTypes<DeviceType> FFT_AT;

  PPPMSamqeqKokkos(class LAMMPS *);
  ~PPPMSamqeqKokkos() override;

  // run-boundary re-arm, device mirror of PPPMSamqeq::init() (pppm_samqeq.cpp) -- see that file's
  // comment for the mechanism (see .cpp).
  void init() override;

  // PPPMSamqeqKokkos subclasses PPPMKokkos directly (not PPPMSamqeq), so it does NOT inherit
  // PPPMSamqeq::compute()'s qsum_qsq(0) refresh; without this override E_long is wrong under fluctuating
  // charges. See pppm_samqeq_kokkos.cpp for detail.
  void compute(int, int) override;

  // device entry: accumulate the RAW reciprocal potential (no qqrd2e) of the source
  // group's charges (atom q on device) at the sensor group's atoms into d_vec[0,nlocal).
  // q_on_device: the caller has staged the trial charges into the DEVICE q and is keeping
  // it canonical itself — do not touch the q flags or sync q here. Default false = the host-staged
  // case (every CPU-side caller), where host q is canonical by construction and must be pushed.
  // Getting this wrong in either direction silently computes the reciprocal of the wrong charges —
  // see the sync block in the .cpp.
  void compute_vector_device(typename AT::t_kkfloat_1d d_vec, int sensor_grpbit,
                             int source_grpbit, bool invert_source, bool q_on_device = false);
  // host convenience wrapper (matches the CPU PPPMSamqeq signature): vec is host, ACCUMULATES.
  // overrides the virtual PPPMSamqeq::compute_vector so the fix's dynamic_cast dispatches here.
  void compute_vector(double *vec, int sensor_grpbit, int source_grpbit, bool invert_source) override;

  void allocate() override;
  void deallocate() override;

  // grid comm override: add the u_brick (potential) FORWARD_AD case; else defer to base.
  void pack_forward_grid_kokkos(int, FFT_DAT::tdual_FFT_SCALAR_1d &, int, DAT::tdual_int_2d_lr &, int) override;
  void unpack_forward_grid_kokkos(int, FFT_DAT::tdual_FFT_SCALAR_1d &, int, int, DAT::tdual_int_2d_lr &, int) override;

  KOKKOS_INLINE_FUNCTION void operator()(TagSamqeqMakeRhoZero, const int &) const;
  KOKKOS_INLINE_FUNCTION void operator()(TagSamqeqMakeRhoMasked, const int &) const;
  KOKKOS_INLINE_FUNCTION void operator()(TagSamqeqFillWork1, const int &) const;
  KOKKOS_INLINE_FUNCTION void operator()(TagSamqeqMulGreens, const int &) const;
  KOKKOS_INLINE_FUNCTION void operator()(TagSamqeqWork2ToU, const int &) const;
  KOKKOS_INLINE_FUNCTION void operator()(TagSamqeqProjectPsi, const int &) const;
  /* The charge-assignment weights depend only on POSITIONS, which are fixed within a timestep, while
     the solve makes several reciprocal calls per step. Compute d_rho1d once per step and have
     make_rho and project_psi both read it.
     Safe against the base class overwriting d_rho1d in its own force-path make_rho: that runs AFTER
     our calls each step, and the stamp is per step, so the next step recomputes.*/
  KOKKOS_INLINE_FUNCTION void operator()(TagSamqeqRho1d, const int &) const;
  KOKKOS_INLINE_FUNCTION void operator()(TagSamqeqPackU, const int &) const;
  KOKKOS_INLINE_FUNCTION void operator()(TagSamqeqUnpackU, const int &) const;

  // self-contained copy of PPPMKokkos::compute_rho1d (which is defined in pppm_kokkos.cpp,
  // so calling it from THIS TU's device functors is unresolved under CUDA / non-RDC builds).
  KOKKOS_INLINE_FUNCTION
  void sam_compute_rho1d(const int, const FFT_SCALAR &, const FFT_SCALAR &, const FFT_SCALAR &) const;

  /* ---- STAGE DIAGNOSTIC: env SAMQEQ_KKSP_DIAG=<n> prints per-stage checksums of
     compute_vector_device for the first n calls. A fault in the reciprocal pipeline can be invisible
     at the level of the final per-atom potential -- these sums say WHICH stage a divergence enters:
       rho_brick -> rho_fft : masked make_rho + reverse_comm + brick2fft
       work2 : the ad-poisson (FFT, greensfn, FFT)
       u_inner vs u_all : the FORWARD_AD ghost exchange of u_brick (difference = ghost content)
       psi : the interpolation back to atoms
     PUBLIC because they host extended device lambdas (nvcc forbids those in non-public members).
     Free when the env var is unset (one integer test per call). ----*/
  double sam_sum_abs_1d(typename FFT_AT::t_FFT_SCALAR_1d v, int n) const;
  double sam_sum_abs_3d(typename FFT_AT::t_FFT_SCALAR_3d v, int z0, int z1, int y0, int y1,
                        int x0, int x1) const;
  double sam_sum_abs_kk(typename AT::t_kkfloat_1d v, int n) const;

 protected:
  // electrolyte (source-group) density, brick + FFT decomposition
  typename FFT_AT::t_FFT_SCALAR_3d d_electrolyte_density_brick;
  typename FFT_AT::t_FFT_SCALAR_1d d_electrolyte_density_fft;

  // per-atom output potential + the masking state captured into the functors
  typename AT::t_kkfloat_1d d_psi;       // per-atom potential (sensor group), length nmax
  typename AT::t_int_1d d_mask_kk;       // atom masks (device)
  int sam_sensor_grpbit, sam_source_grpbit;
  int sam_invert;                        // 0/1

  // grid forward-comm scratch for the u_brick pack/unpack
  typename FFT_AT::t_FFT_SCALAR_1d_um d_ubuf;
  typename AT::t_int_2d_lr_um d_ulist_index;

  int sam_compute_step;                  // lazy particle_map + rho1d-weights guard (mirrors CPU start_compute).
                                         // The weights are filled under THIS stamp, in
                                         // start_compute_device, never under their own (see the note there)
  void start_compute_device();
  // Sub-phase wall clock of compute_vector_device, accumulated here and printed by
  // FixQEqSamKokkos::post_run (which holds eksp_kk) under SAMQEQ_KK_TIME=1. The reciprocal is five
  // phases — make_rho, brick2fft+reverse comm, two FFTs with the Greens multiply, the u_brick forward
  // comm, and the projection back to atoms.
 public:
  double tk_rho = 0.0, tk_b2fft = 0.0, tk_fft = 0.0, tk_ucomm = 0.0, tk_proj = 0.0;
  long   tk_calls = 0;
  int    tk_on = -1;                     // -1 = read the env var on first use
 protected:
  int sam_diag_left = -1;                // stage diagnostic: calls still to print (-1 = read env)
  int sam_diag_call = 0;                 // call counter, for the printout
};

}    // namespace LAMMPS_NS

#endif
#endif
