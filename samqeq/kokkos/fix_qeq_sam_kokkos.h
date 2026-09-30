/* -*- c++ -*- ----------------------------------------------------------
   fix_qeq_sam_kokkos.h — KOKKOS-aware fix qeq/sam.

   Registers `qeq/sam/kk`, `qeq/sam/kk/device`, and `qeq/sam/kk/host` so a
   `-sf kk` run (or an explicit `fix ... qeq/sam/kk ...`) gets a Kokkos-aware
   charge fix instead of the bare CPU FixQEqSam.

   PHASE 2 (milestone 1, "make it RUN under -sf kk"): the charge solve + PPPM
   reciprocal still execute on the HOST (samQEq's solve is large + the
   reciprocal goes through the CPU PPPMSamqeq), but the neighbor lists are now
   NATIVE Kokkos lists (the plain CPU requests segfaulted the kk neighbor
   build — copy_to_cpu null listcopy — because samQEq's fix requests are not
   paired with a kokkos parent; see). We request
   kokkos FULL lists and, each solve, mirror their d_ilist/d_numneigh/
   d_neighbors down to host and populate the legacy ilist/numneigh/firstneigh
   the inherited host solve reads — a controlled copy_to_cpu that cannot hit
   the framework's null-listcopy bug. Atom data is synced Host before the
   solve and q is marked modified(Host) after. Subsequent milestones move
   compute_H and the matvec onto the device (Phase 2-full) and the reciprocal
   into pppm/samqeq/kk (Phase 3).
------------------------------------------------------------------------*/

#ifdef FIX_CLASS
// clang-format off
FixStyle(qeq/sam/kk,FixQEqSamKokkos<LMPDeviceType>);
FixStyle(qeq/sam/kk/device,FixQEqSamKokkos<LMPDeviceType>);
FixStyle(qeq/sam/kk/host,FixQEqSamKokkos<LMPHostType>);
// clang-format on
#else

#ifndef LMP_FIX_QEQ_SAM_KOKKOS_H
#define LMP_FIX_QEQ_SAM_KOKKOS_H

#include "fix_qeq_sam.h"
#include "kokkos_type.h"

#include <vector>

namespace LAMMPS_NS {

template<class DeviceType> class PPPMSamqeqKokkos;   // device reciprocal (Phase 3b)

// S1 (Phase 2-full): matrix-free device matvec of the short-range DSF H-block.
struct TagSamMatvecH {};
// S2: matrix-free device matvec of the ACKS2 X-block.
struct TagSamMatvecX {};
// S3: full augmented ACKS2 matvec (s+u blocks; H+X+diagonal+identity+constraint scalars).
struct TagSamMatvecFull {};
// S8: q-direct off-diagonal H-block for the Ewald path (lr_ewald=2): net J_shield = fully-shielded
// 1/∛(r³+1/g³) minus erf(αr)/r (the reciprocal adds erf(αr)/r back). Distinct from TagSamMatvecH (taper/DSF).
struct TagSamMatvecHErfc {};
// (#30): STORED-H form of the same operator. TagSamBuildHErfc evaluates each pair's H once per
// SOLVE into d_Hval; TagSamSpMV then applies it per CG iteration as a plain memory-bound product.
struct TagSamBuildHErfc {};
struct TagSamSpMV {};

/* ---------------------------------------------------------------------------------------------
   (#30): SMALL launch functors for the two hot kernels.

   ★ Why this exists at all. Every kernel in this file used to be launched as
   `parallel_for(policy, *this)`, and the fix object carries `std::vector<int> hb_neigh` /
   `hbf_neigh` — the HOST-mirrored neighbour lists, ~141 MB at 29k atoms. Kokkos copies the functor
   BY VALUE, so each launch deep-copied those vectors on the host: measured 137 ms per device matvec
   at 29k atoms, i.e. ~2 GB/s where the card does ~200, and it made the matrix-free and stored-H
   variants perform identically because the copy dwarfed both. `copymode` protects the destructor
   from freeing shared pointers; it does nothing about the std::vector members.

   These structs carry only Views and scalars, so a launch copies a few hundred bytes.
   SamShieldKernel keeps ONE implementation of the shielding kernel for both of them (and mirrors
   FixQEqSamKokkos::dev_Jshield exactly — the parity harness covers this arithmetic).
---------------------------------------------------------------------------------------------*/

template<class DeviceType>
struct SamShieldKernel {
  typedef Kokkos::View<double *, DeviceType> t_gam;
  typedef Kokkos::View<double ***, Kokkos::LayoutRight, DeviceType> t_tab;
  t_gam gamma;                 // per-type param col 4 (γ | Rc | ζ, by mode)
  t_tab J, dJ;                 // Slater tables (empty unless mode==2)
  double lambda = 0.0, rmin = 0.0, dr = 0.0;
  int mode = 0, npts = 0;      // mode: 0 cbrt, 1 PQEq Gaussian, 2 Slater

  KOKKOS_INLINE_FUNCTION
  double operator()(int ti, int tj, double r) const
  {
    if (mode == 2) {           // tabulated Slater J(r), Hermite-cubic (see SlaterJTable::eval)
      if (npts < 2 || r <= rmin) return J(ti, tj, 0);
      double x = (r - rmin) / dr;
      int n0 = (int) x;
      if (n0 >= npts - 1) n0 = npts - 2;
      const double t = x - n0, h = dr;
      const double J0 = J(ti, tj, n0), J1 = J(ti, tj, n0 + 1);
      const double D0 = dJ(ti, tj, n0), D1 = dJ(ti, tj, n0 + 1);
      const double t2 = t * t, t3 = t2 * t;
      return (2.0*t3 - 3.0*t2 + 1.0)*J0 + (t3 - 2.0*t2 + t)*h*D0
           + (-2.0*t3 + 3.0*t2)*J1 + (t3 - t2)*h*D1;
    }
    if (mode == 1) {           // PQEq Gaussian overlap; gamma is Rc here
      const double gi = gamma(ti), gj = gamma(tj);
      const double ai = lambda * 0.5 / (gi * gi), aj = lambda * 0.5 / (gj * gj);
      return erf(sqrt(ai * aj / (ai + aj)) * r) / r;
    }
    const double g = sqrt(gamma(ti) * gamma(tj));
    return 1.0 / cbrt(r * r * r + 1.0 / (g * g * g));
  }
};

// build the Ewald H-block into Hval (one thread per atom, its neighbour slots across vector lanes)
template<class DeviceType>
struct SamBuildHFunctor {
  typedef DeviceType execution_space;
  typedef ArrayTypes<DeviceType> AT;
  typedef typename Kokkos::TeamPolicy<DeviceType>::member_type member_type;
  Kokkos::View<double **, Kokkos::LayoutRight, DeviceType> Hval;
  // ★ the four per-pair GATHERS (x, type, mask, and the neighbour index) are what this kernel is
  // bound by — ~1 GB of scattered reads per build at 29k atoms. RandomAccess (= __ldg / read-only
  // cache on CUDA) is the standard remedy and is what the stock kk pair styles use for exactly these
  // views; numerically identical, it only changes how the reads are cached.
  typename AT::t_kkfloat_1d_3_lr_randomread x;
  typename AT::t_int_1d_randomread type, mask;
  typename AT::t_int_1d ilist, numneigh;          // read sequentially -> plain views
  typename AT::t_neighbors_2d_randomread neighbors;
  SamShieldKernel<DeviceType> Jsh;
  int inum, gbit;
  double swb2, qqrd2e, alpha;

  KOKKOS_INLINE_FUNCTION
  void operator()(const member_type &team) const
  {
    const int ii = team.league_rank() * team.team_size() + team.team_rank();
    if (ii >= inum) return;
    const int i = ilist(ii);
    const int jnum = numneigh(i);
    if (!(mask(i) & gbit)) {
      Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, jnum), [&](const int jj) { Hval(ii, jj) = 0.0; });
      return;
    }
    const int ti = type(i);
    const double xi = x(i, 0), yi = x(i, 1), zi = x(i, 2);
    Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, jnum), [&](const int jj) {
      double Hij = 0.0;
      const int j = neighbors(i, jj) & NEIGHMASK;
      if (mask(j) & gbit) {                    // COLUMN group test (see FixQEqSam::compute_H)
        const double dx = x(j, 0) - xi, dy = x(j, 1) - yi, dz = x(j, 2) - zi;
        const double r2 = dx * dx + dy * dy + dz * dz;
        if (r2 <= swb2) {
          const double r = sqrt(r2);
          Hij = qqrd2e * (Jsh(ti, type(j), r) - (1.0 - erfc(alpha * r)) / r);
        }
      }
      Hval(ii, jj) = Hij;                      // excluded pairs store 0 -> the apply needs no test
    });
  }
};

// apply the stored H: one streaming pass per CG iteration, no transcendentals
template<class DeviceType>
struct SamSpMVFunctor {
  typedef DeviceType execution_space;
  typedef ArrayTypes<DeviceType> AT;
  typedef typename Kokkos::TeamPolicy<DeviceType>::member_type member_type;
  Kokkos::View<double **, Kokkos::LayoutRight, DeviceType> Hval;
  Kokkos::View<const double *, DeviceType, Kokkos::MemoryTraits<Kokkos::RandomAccess>> xvec;  // gathered
  Kokkos::View<double *, DeviceType> bH;
  typename AT::t_int_1d_randomread mask;
  typename AT::t_int_1d ilist, numneigh;
  typename AT::t_neighbors_2d_randomread neighbors;
  int inum, gbit;

  KOKKOS_INLINE_FUNCTION
  void operator()(const member_type &team) const
  {
    const int ii = team.league_rank() * team.team_size() + team.team_rank();
    if (ii >= inum) return;
    const int i = ilist(ii);
    if (!(mask(i) & gbit)) { Kokkos::single(Kokkos::PerThread(team), [&]() { bH(i) = 0.0; }); return; }
    const int jnum = numneigh(i);
    double sum = 0.0;
    Kokkos::parallel_reduce(Kokkos::ThreadVectorRange(team, jnum), [&](const int jj, double &s) {
      s += Hval(ii, jj) * xvec(neighbors(i, jj) & NEIGHMASK);
    }, sum);
    Kokkos::single(Kokkos::PerThread(team), [&]() { bH(i) = sum; });
  }
};

template<class DeviceType>
class FixQEqSamKokkos : public FixQEqSam {
 public:
  typedef DeviceType device_type;
  typedef ArrayTypes<DeviceType> AT;

  FixQEqSamKokkos(class LAMMPS *, int, char **);

  void init() override;
  void init_list(int, class NeighList *) override;
  void post_run() override;   // (#30): print the device-matvec phase timings (SAMQEQ_KK_TIME=1)
  // BO charge-solve hooks
  void setup_pre_force(int) override;
  void pre_force(int) override;
  void min_pre_force(int) override;
  // XL charge-propagation hooks (no-op at runtime in BO mode)
  void initial_integrate(int) override;
  void final_integrate() override;

  // S5: host-side forward comm of the augmented (s,u) iterate for the periodic device
  // solve. pack_flag==SAMKK_FWD packs 2 doubles/atom (kk_sv[j], kk_sv[s3_NN+j]) from the
  // kk scratch buffer; every other pack_flag delegates to FixQEqSam so the host solve's
  // comms are byte-identical. forward_comm_device stays 0
  // -> CommKokkos uses this host path (mirrors fix_acks2_reaxff_kokkos, which also host-
  // round-trips the augmented vector instead of a device TagPack).
  int pack_forward_comm(int, int *, double *, int, int *) override;
  void unpack_forward_comm(int, int, double *) override;

  // S6 production wiring: when enabled + eligible, replace the host BiCGStab with the device
  // solve (result -> s; the host calculate_Q then sets q). `fix_modify <id> devsolve on|off`
  // gates it (default OFF -> M1 byte-identical). GPU reductions are not bit-reproducible, so the
  // engaged path is tolerance-accurate, not byte-identical -> opt-in.
  int BiCGStab(double *, double *) override;
  int modify_param(int, char **) override;

  // S9/Stage-3 production wiring: route the lr_ewald=2 q-direct CG (the plain BO solve AND the ASPC
  // corrector both call qeq_cg) to the device solver when `devsolve on` + eligible. Default OFF -> host.
  int qeq_cg(double *b, double *x) override;

  // shared device BiCGStab: solves the augmented ACKS2 system. x_out != null -> write the
  // converged augmented vector there (production, x_out=s); x_out == null -> compare to CPU s +
  // log (the S5 one-time self-check). Returns iteration count. Assumes upload_device_state() done.
  // MUST be public: it hosts extended __host__ __device__ lambdas, which CUDA/nvcc forbids inside a
  // private/protected member function (the enclosing function's access must be public).
  int run_device_bicgstab(double *b_host, double *x_out);

  /* ---- (#30): serve coulomb_field from the device, and then skip the host CSR H entirely ----
     coulomb_field() computes P(H·x + recip·x) — the q0 reference-charge field, which qeq_solve() needs
     on EVERY step whenever any type carries q0 (types 6-11 do on the electrolyte decks: the ions, Li+
     at +1.0). It was the last per-step consumer of the host CSR H, and the reason skipping the host
     compute_H moved the charges by 2.1e-5: the field was being built from the PREVIOUS step's matrix.
     Found by breakpointing csr_matvec_add past the 9 setup calls and reading the backtrace. With it on
     device, nothing on the host reads H under a device solve, so host_H_needed() can finally say no —
     worth 2.3x, since the host build is ~23M erf/erfc evaluations on one core (the device does the same
     matrix in 56 ms).*/
  void coulomb_field(double *x, double *out, bool project = true) override;

  bool host_H_needed() override {
    return !(device_solve_on && device_qcg_eligible() && s8_done && s9_done
             && precond_mode == 0);
  }

  // ---- S7: device per-molecule neutral projection P(v) ----
  // v[i] -= (Σ_{j∈mol(i)} v[j]) · molinv[mol(i)] — the projector onto the per-molecule-neutral subspace.
  // This is the one device primitive missing for a q-direct device qeq_cg (the lr_ewald=2 ASPC path; the H
  // matvec [TagSamMatvecH] + reciprocal [device_add_reciprocal] already exist). PUBLIC: hosts an extended device lambda.
  void project_neutral_device(Kokkos::View<double *, DeviceType> v);

  // ---- S8: device q-direct matvec out = P(η·x + H·x + reciprocal·x) ----
  // The operator of the lr_ewald=2 projected-CG (qeq_matvec): device diagonal + S1 off-diagonal H
  // (TagSamMatvecH) + reciprocal (HOST add_reciprocal bounce, correctness-first) + S7 project_neutral.
  // x is the per-atom input (local group filled); ghosts are filled internally by a host forward_comm.
  // PUBLIC: launches device parallel_fors. Scalar ridge only.
  // (#30): with_diag=false drops the η diagonal, giving exactly FixQEqSam::coulomb_field's
  // operator P(H·x + recip·x); with_proj=false skips the projector (coulomb_field's `project` arg).
  void qeq_matvec_device(Kokkos::View<double *, DeviceType> x, Kokkos::View<double *, DeviceType> out,
                         bool with_diag = true, bool with_proj = true);

  // ---- add the PPPM reciprocal to out's s-block in place, mirroring FixQEqSam::add_reciprocal ----
  // (#30): all-device when the kspace is pppm/samqeq/kk (stash device q, feed the trial charges,
  // device compute_vector, restore); host round trip otherwise. PUBLIC: hosts extended device lambdas.
  void device_add_reciprocal(Kokkos::View<double *, DeviceType> in,
                             Kokkos::View<double *, DeviceType> out);

  // ---- S9: device q-direct projected CG (the lr_ewald=2 solve, mirroring FixQEqSam::qeq_cg) ----
  // Solves Ã·x=b on the per-molecule-neutral subspace using qeq_matvec_device (S8), project_neutral_device
  // (S7), the Jacobi preconditioner Hdia_inv, and device dot/axpy. b_host = projected RHS (length>=nlocal);
  // x_out (length>=nlocal) gets the local solution. Returns iteration count. PUBLIC: hosts device lambdas.
  int run_device_qcg(double *b_host, double *x_out);

  // S1 device functor: b_H[i] = sum_{full nbrs j, r<swb} H_ij * d_xvec[j] (off-diagonal H-block only)
  KOKKOS_INLINE_FUNCTION
  void operator()(TagSamMatvecH, const int &ii) const;

  // S8 device functor: Ewald (lr_ewald=2) off-diagonal H-block, b_H[i] = sum_j J_shield_ij*d_xvec[j]
  KOKKOS_INLINE_FUNCTION
  void operator()(TagSamMatvecHErfc, const int &ii) const;

  /* ---- (#30): store H once per solve, then apply it. ------------------------------------
     The device matvec was MATRIX-FREE while the host solve is MATRIX-BASED: FixQEqSam::compute_H
     evaluates each pair's kernel ONCE per step into a CSR matrix and every CG iteration is then a
     cheap sparse product, whereas TagSamMatvecHErfc re-evaluated erf/erfc/pow for all ~800
     neighbours of every atom on EVERY iteration. At the 7-23 iterations these decks take that is an
     order of magnitude more transcendental work, and it is why one GPU matvec (29 ms at 29k atoms)
     lost to one CPU core (12 ms) doing the same physics. Same fix as fix qeq/reaxff/kk: build, then
     apply. d_Hval mirrors the neighbour list's own (i, slot) layout, so no column index is stored —
     the SpMV reads d_neighbors for j, exactly as the build did.
     Falls back to the matrix-free functor automatically if the allocation does not fit (hmat_ok).*/
  // team/vector shape (mirrors fix_qeq_reaxff_kokkos.h's constants for the same kernel)
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
  static constexpr int spmv_vectorsize = 32;
  static constexpr int spmv_teamsize = 8;
#else
  static constexpr int spmv_vectorsize = 1;
  static constexpr int spmv_teamsize = 1;
#endif

  // S2 device functor: b_X[i] = X_diag[i]*u[i] + sum_{full nbrs j} w_ij * u[j] (X-block)
  KOKKOS_INLINE_FUNCTION
  void operator()(TagSamMatvecX, const int &ii) const;
  // device weight w_ij (intra-fragment Gaussian; 0 between fragments)
  KOKKOS_INLINE_FUNCTION
  double s2_calc_w(int i, int j, int ti, int tj, double r) const;

  // S3 device functor: full augmented ACKS2 matvec, per-atom s/u blocks
  KOKKOS_INLINE_FUNCTION
  void operator()(TagSamMatvecFull, const int &ii) const;

  /* ---- : the shielding kernel ON DEVICE (cbrt / PQEq Gaussian / Slater) ----------------
     Device mirror of FixQEqSam::shielded_coulomb()'s three branches (fix_qeq_sam.cpp), i.e. the
     NET short-range kernel the lr_ewald=2 H-block is built from. Until the device H
     functors hardcoded cbrt and every non-cbrt deck fell back to the host solve entirely
     ("NOT ported"); pqeq is the standing kernel for the
     electrolyte campaigns, so that fallback covered essentially all production decks.
     Arithmetic is transcribed expression-for-expression from the host so the S8/S9 device-vs-host
     self-checks stay at round-off (the CUDA erf() may differ from libm's by ~1 ulp; that is what
     the tolerance-based check is for -- these paths were never bit-identical anyway, see the S6
     note in this header). NOT mirrored here: the two kernel MODIFIERS the host applies after the
     branch -- iondamp (Tang-Toennies) -- which device_kernel_ok() below excludes.*/
  KOKKOS_INLINE_FUNCTION
  double dev_Jshield(int ti, int tj, double r) const
  {
    if (s8_shield == SHIELD_SLATER) { double J, dJdr; slater_eval(ti, tj, r, J, dJdr); return J; }
    if (s8_shield == SHIELD_GAUSSIAN) {
      // gamma is REINTERPRETED as the PQEq Gaussian core radius Rc in this mode (host convention)
      const double gi = d_gamma(ti), gj = d_gamma(tj);
      const double ai = s8_lambda * 0.5 / (gi * gi);
      const double aj = s8_lambda * 0.5 / (gj * gj);
      const double aij = sqrt(ai * aj / (ai + aj));
      return erf(aij * r) / r;
    }
    const double g = sqrt(d_gamma(ti) * d_gamma(tj));
    return 1.0 / cbrt(r * r * r + 1.0 / (g * g * g));
  }

  // Device mirror of SlaterJTable::eval() (slater_jtable.h) -- same Hermite-cubic basis, reading the
  // dense device table instead of the host std::vector. Structurally identical to the pair style's
  // PairCoulShieldIntraKokkos::slater_eval() so fix and pair interpolate the SAME table the
  // same way (force<->solve consistency, the whole point of the shared slater_jtable.h).
  KOKKOS_INLINE_FUNCTION
  void slater_eval(int ti, int tj, const double &r, double &J, double &dJdr) const
  {
    if (slater_npts < 2 || r <= slater_rmin) {
      J = d_slater_J(ti, tj, 0);
      dJdr = d_slater_dJ(ti, tj, 0);
      return;
    }
    double x = (r - slater_rmin) / slater_dr;
    int n0 = (int) x;
    if (n0 >= slater_npts - 1) n0 = slater_npts - 2;
    const double t = x - n0;
    const double h = slater_dr;
    const double J0 = d_slater_J(ti, tj, n0),   J1 = d_slater_J(ti, tj, n0 + 1);
    const double D0 = d_slater_dJ(ti, tj, n0),  D1 = d_slater_dJ(ti, tj, n0 + 1);
    const double t2 = t * t, t3 = t2 * t;
    const double h00 = 2.0*t3 - 3.0*t2 + 1.0, h10 = t3 - 2.0*t2 + t;
    const double h01 = -2.0*t3 + 3.0*t2,      h11 = t3 - t2;
    const double h00d = 6.0*t2 - 6.0*t,       h10d = 3.0*t2 - 4.0*t + 1.0;
    const double h01d = -6.0*t2 + 6.0*t,      h11d = 3.0*t2 - 2.0*t;
    dJdr = (h00d*J0 + h10d*h*D0 + h01d*J1 + h11d*h*D1) / h;
    J = h00*J0 + h10*h*D0 + h01*J1 + h11*h*D1;
  }

 protected:
  // sync the atom arrays the host solve reads down to the host
  void sync_before_solve();
  // mark q dirty on the host so the next device kernel re-syncs it
  void modified_after_solve();

  // mirror BOTH kokkos neighbor lists into fix-owned host buffers and ALIAS the
  // lists' legacy ilist/numneigh/firstneigh onto them for the solve; restore the
  // lists' original pointers afterwards so ~NeighList frees what it owns (else
  // double-free at exit, since the buffers below are owned by the fix)
  void alias_host_neighbor_lists();
  void restore_host_neighbor_lists();
  // mirror one kokkos NeighList's device views to host into the backing buffers,
  // save the list's original legacy pointers, and alias the list onto the buffers
  void alias_one(class NeighList *klist, std::vector<int> &h_ilist,
                 std::vector<int> &h_numneigh, std::vector<int> &h_neigh,
                 std::vector<int *> &h_first, int *&sav_ilist, int *&sav_numneigh,
                 int **&sav_first, bool refresh);
  static void restore_one(class NeighList *klist, int *sav_ilist, int *sav_numneigh,
                          int **sav_first);

  // backing storage for the host-mirrored CPU lists (list = id 0 charge solve,
  // listf = id 1 full list)
  std::vector<int> hb_ilist, hb_numneigh, hb_neigh;
  std::vector<int *> hb_first;
  std::vector<int> hbf_ilist, hbf_numneigh, hbf_neigh;
  std::vector<int *> hbf_first;
  // saved original legacy pointers, restored after each solve
  int *sav_il = nullptr, *sav_nn = nullptr;   int **sav_fn = nullptr;
  int *savf_il = nullptr, *savf_nn = nullptr;  int **savf_fn = nullptr;
  bool aliased = false, aliasedf = false;

  // ---- perf: mirror CACHE for the host-mirrored neighbor lists ----
  // Rebuilding the mirrors (3 device->host copies + an O(sum numneigh) repack, per list) on every
  // hook entry was the single largest cost of the kk path: ~45 MB of D2H traffic and ~4.4M element
  // writes per call on a 3.7k-atom box, several calls per step. The mirrors only go stale when the
  // neighbor lists are REBUILT (positions moving does not change list contents), so key them on
  // neighbor->lastcall — the timestep of the last build (NeighborKokkos::build sets it) — plus
  // atom->nmax (the mirrors are sized by it) and each list's inum. Invalidated in init() because
  // Neighbor::init() resets lastcall to -1 at every run, which would otherwise let a stale mirror
  // from run N look valid at the start of run N+1. The alias/restore pointer swap still happens on
  // every call; only the copy is skipped.
  bigint mirror_stamp = -2;      // neighbor->lastcall the current mirrors were built from
  int mirror_nmax = -1;          // atom->nmax at that build
  int mirror_inum = -1;          // list->inum at that build (-1 = no list)
  int mirror_inum_f = -1;        // list_full->inum at that build
  // reentrancy guard: FixQEqBaseSam::setup_pre_force() calls pre_force() (virtual -> our pre_force),
  // so alias/restore nest. Only the OUTERMOST may save/restore, else the nested alias clobbers the saved
  // originals with the already-aliased pointers and ~NeighList double-frees our buffers.
  int alias_depth = 0;

  // ---- S1: device matrix-free DSF H-block matvec (validation harness) ----
  // one-time self-check that the device functor reproduces the host computation of
  // b_H[i] = sum_{full nbrs j, r<swb} H_ij*x[j]; engaged only on the lr_ewald=0 path.
  void validate_matvecH();
  // captured-by-value functor state (set in validate_matvecH before the parallel_for):
  typename AT::t_kkfloat_1d_3_lr d_x;           // atom positions (matches reaxff/kk x type)
  typename AT::t_int_1d d_type;                 // atom types
  typename AT::t_int_1d d_mask;                 // group masks
  typename AT::t_int_1d d_ilist_f;              // list_full d_ilist
  typename AT::t_int_1d d_numneigh_f;           // list_full d_numneigh
  typename AT::t_neighbors_2d d_neighbors_f;    // list_full d_neighbors
  Kokkos::View<double *, DeviceType> d_tap;     // 8 taper coeffs
  Kokkos::View<double *, DeviceType> d_shld;    // flattened shielding (ntp1*ntp1)
  Kokkos::View<double *, DeviceType> d_xvec;    // input vector x (per atom, incl ghosts)
  Kokkos::View<double *, DeviceType> d_bH;      // output b_H (per local atom)
  int s1_ntp1 = 0;          // ntypes+1 (shld row stride)
  int s1_gbit = 0;          // groupbit
  double s1_swb = 0.0, s1_qqrd2e = 0.0;
  bool s1_done = false;     // one-time validation flag

  // ---- S2: device X-block matvec (validation harness) ----
  void validate_matvecX();
  typename AT::t_tagint_1d d_mol;               // atom->molecule
  Kokkos::View<double *, DeviceType> d_diag;    // per-atom SOLVE diagonal = host solve_diag_of(i) (gself-folded)
  Kokkos::View<double *, DeviceType> d_chi;     // per-type chi (s1_ntp1)
  Kokkos::View<double *, DeviceType> d_uvec;    // input u-block vector
  Kokkos::View<double *, DeviceType> d_bX;      // output b_X (per local atom)
  double s2_kappa = 0.0, s2_rov = 0.0, s2_xreg = 0.0, s2_ridge = 0.0;
  bool s2_done = false;

  // ---- S4: device BiCGStab (no-ghost) ----
  void validate_solve();
  // ---- S3: full augmented ACKS2 matvec (validation harness) ----
  void validate_matvecFull();
  Kokkos::View<double *, DeviceType> d_eta;     // per-type eta (s1_ntp1)
  Kokkos::View<double *, DeviceType> d_xaug;    // augmented input (len 2*NN+2)
  Kokkos::View<double *, DeviceType> d_baug;    // augmented output
  int s3_NN = 0;            // nlocal+nghost (block stride of the augmented vector)
  bool s3_done = false;
  bool s4_done = false;     // S4/S5 device BiCGStab one-time check

  // ---- S5: ghost comm for the periodic device solve (host-roundtrip) ----
  static constexpr int SAMKK_FWD = 30;   // private pack_flag for the augmented (s,u) forward comm
  std::vector<double> kk_sv;             // host scratch = current augmented iterate (len 2*s3_NN+2)

  // ---- S6: production device solve ----
  bool device_solve_on = false;          // `fix_modify <id> devsolve on` to engage (default off)
  // upload all device state the matvec/solve needs (params once-ish + atom data/lists each call)
  // (#30): upload_device_state() ran on EVERY solve and re-created six Views (plus their host
  // mirrors) each time — the same allocate-per-call pattern that cost 60x in the matvec. The per-TYPE
  // arrays (tap/shld/eta/chi) are run constants; the per-ATOM ones only need resizing on nmax growth.
  // Persistent mirrors + a run-constant stamp; the per-atom refill stays every solve (eta_diag and the
  // compact molecule map are rebuilt by the host solve).
  void upload_device_state();
  // host fill of the per-atom solve diagonal for d_diag (local in-group rows; ghosts/out-of-group 0)
  void fill_h_diag(typename Kokkos::View<double *, DeviceType>::host_mirror_type &h, int nmax_);
  typename Kokkos::View<double *, DeviceType>::host_mirror_type h_diag_;
  typename Kokkos::View<int *, DeviceType>::host_mirror_type h_cmol_;
  int  uds_cap = -1;          // nmax the per-atom mirrors are sized for
  bool uds_types_done = false;   // per-type constants uploaded (cleared in init())
  // (run_device_bicgstab moved to the public section — CUDA forbids extended device lambdas inside a
  //  non-public member function.)
  // ---- : device shielding-kernel state (see dev_Jshield()/slater_eval() above) ----
  // Uploaded by upload_kernel_state() from the host shield_gauss/shield_lambda/gamma/slater_tabs.
  int    s8_shield = 0;      // shield_gauss (SHIELD_CBRT/PQEQ/SLATER) at last upload
  double s8_lambda = 0.0;    // shield_lambda (PQEq λ) at last upload
  // dense [ti][tj][node] Slater J/dJ tables; every type-pair table shares one (rmin,dr,npts) grid by
  // construction (slater_jtable.h), so one scalar triple suffices. Allocated only in SLATER mode.
  typedef Kokkos::View<double ***, Kokkos::LayoutRight, DeviceType> t_slater_tab_kk;
  t_slater_tab_kk d_slater_J, d_slater_dJ;
  double slater_rmin = 0.0, slater_dr = 0.0;
  int    slater_npts = 0;
  bool   slater_dev_stale = true;   // set in init() (host build_slater_tables() re-runs there), cleared
                                    // once upload_kernel_state() has re-uploaded the device copy
  bool   kernel_dev_stale = true;   // perf: gamma/lambda/mode are RUN constants (they can only change
                                    // through a fix_modify between runs, which is followed by init()), so
                                    // upload them once per run instead of once per matvec

  // ---- perf: persistent per-matvec scratch ----
  // qeq_matvec_device() used to allocate two std::vectors and four Kokkos mirror views per call —
  // ~600 cudaMalloc/cudaFree pairs and 1365 synchronous cudaMemcpy in a 2-step run (nsys). Sizes only
  // grow with atom->nmax, so hold them as members and resize on growth.
  std::vector<double> mv_hx, mv_ov;                              // host trial vector / host output
  typename Kokkos::View<double *, DeviceType>::host_mirror_type mv_hxvec, mv_hout;   // pinned-side mirrors
  int mv_cap = -1;                                               // nmax the scratch above is sized for
  void ensure_matvec_scratch(int nmax);
  bool molinv_dev_stale = true;     // per-solve, not per-matvec: molinv is rebuilt by the host solve
                                    // (build_molinv) before each solve, so upload it once per solve
  // upload gamma/lambda/mode (+ the Slater tables in SLATER mode). Called from upload_device_state()
  // and from the S1/S3 validators, which do their own (older, self-contained) uploads.
  void upload_kernel_state();

  // (was device_shield_ok): the shielding MODE no longer gates the device path -- cbrt, pqeq
  // and slater are all implemented in the device H functors now. What is still unported is the
  // kernel MODIFIER the host applies INSIDE shielded_coulomb() after the branch: the Tang-Toennies
  // interionic damping (lr_iondamp). It silently changed the HOST operator
  // while the device computed the undamped kernel -- the same silent-divergence shape the shield guard was written for,
  // and never previously guarded. Used by device_eligible()/device_qcg_eligible() (production) and by
  // the S1/S3/S8/S9 self-checks, which skip with a one-time warning rather than compare two different
  // operators.
  bool device_kernel_ok() const { return !lr_iondamp; }
  bool shield_kk_warn_done = false;      // one-time warning latch for the skip/fallback above
  void warn_shield_kk_once();

  // eligibility: the ACKS2 saddle path, lr_ewald=0 (DSF), with the default kernel (no electrode).
  // the quartic/floating-diagonal (lr_quartic), per-atom ridge (ridge_local), and the noise-free cutoff-H
  // long-range mode (lr_nrecip): the device matvec/diagonal does not implement any of these, so solving
  // with them enabled would silently use the wrong diagonal or double-count the long range (audit C4).
  // replaces the shield gate with device_kernel_ok() (iondamp), and adds the lr_alpha
  // guard below.
  // ★ lr_alpha <= 0 is REQUIRED here and is a BUG FIX, not a tightening: TagSamMatvecFull's H-block
  // computes the LEGACY tapered short-range form (Taper·qqrd2e/∛(r³+shld)), which is what the host
  // assembles only when lr_alpha <= 0 (FixQEqSam::compute_H's first line delegates to the base in that
  // case). With lr_alpha > 0 the host assembles DSF (erfc/r − e_shift − r·f_shift), which this functor
  // does not implement, so the engaged device solve silently solved a DIFFERENT operator.
  // It was invisible because the S1/S3 self-checks' host references re-implement the
  // functor's own formula instead of calling the fix's real per-pair value -- fixed in by routing
  // them through calc_Hval(). Porting the DSF/Ewald forms into the augmented functor is the follow-up
  // (§"Applied "); until then this combination falls
  // back to the host BiCGStab, which is correct.
  bool device_eligible() const {
    return lr_ewald == 0 && list_full
           && !lr_quartic && !ridge_local && !lr_nrecip && device_kernel_ok() && lr_alpha <= 0.0;
                                           //
  }
  // q-direct (lr_ewald=2) device CG eligibility: molinv built (S7/S8/S9 assumptions).
  // Same exclusions as device_eligible() above and for the same reason (audit C4); : device_kernel_ok()
  // (iondamp) replaces the shield gate -- cbrt/pqeq/slater all run on device now.
  // lr_alpha > 0 required for the mirror reason as above: TagSamMatvecHErfc computes the EWALD-SPLIT form
  // (J_shield − (1−erfc(αr))/r), which is the host's H only while lr_alpha > 0; at lr_alpha == 0 the host
  // falls back to the legacy tapered base compute_H. lr_ewald=2 decks always carry α = the PPPM g_ewald,
  // so this excludes nothing real -- it just makes the assumption explicit rather than latent.
  bool device_qcg_eligible() const {
    return lr_ewald == 2 && list_full && nmol_ > 0 && molinv
           && !lr_quartic && !ridge_local && !lr_nrecip && device_kernel_ok() && lr_alpha > 0.0;
  }

  // ---- device add_reciprocal for the lr_ewald=2 all-device matvec ----
  class PPPMSamqeqKokkos<DeviceType> *eksp_kk = nullptr;   // = the fix's eksp as the device pppm/samqeq/kk
  typename AT::t_kkfloat_1d d_prec;    // device reciprocal potential (compute_vector_device out)
  typename AT::t_kkfloat_1d d_qsave;   // stash of device atom-q while feeding trial charges
  // (device_add_reciprocal moved to the public section — : it now hosts extended device lambdas
  //  for the all-device path, and nvcc forbids those inside a private/protected member function.)

  // ---- S7: device per-molecule neutral projection (validation harness + state) ----
  // one-time self-check that project_neutral_device reproduces the host per-molecule-neutral projection on a
  // deterministic test vector (default path). Mirrors the S1-S5 validate_* methodology.
  void validate_project_neutral();
  Kokkos::View<double *, DeviceType> d_molinv;   // per-molecule 1/count (length nmol_ = nactive, B9 compact)
  Kokkos::View<double *, DeviceType> d_molsum;   // per-molecule sum scratch (length nmol_ = nactive)
  // B9: per-LOCAL-atom COMPACT molecule index (device mirror of the host cmol built in build_molinv),
  // uploaded in upload_device_state. Indexes d_molinv/d_molsum here -- d_mol (raw tagint, above) stays in
  // use for the S2 bond-softness intra-molecule EQUALITY test, which is compaction-invariant (same molecule
  // <=> same compact slot too) and so needed no change.
  Kokkos::View<int *, DeviceType> d_cmol;
  int  s7_nmol = 0;          // nmol_ at last (re)allocation of d_molinv/d_molsum
  bool s7_done = false;      // one-time validation flag

  // ---- S8: device q-direct matvec self-check (vs host FixQEqSam::qeq_matvec) ----
  void validate_qeq_matvec();
  Kokkos::View<double *, DeviceType> d_gamma;   // per-type shielding gamma (s1_ntp1) for the Ewald H-block
  double s8_alpha = 0.0;     // lr_alpha (Ewald real-space damping) at last upload
  bool s8_done = false;
  // ---- (#30) stored H (see the functors above) ----
  typedef Kokkos::View<double **, Kokkos::LayoutRight, DeviceType> t_hval_2d;
  t_hval_2d d_Hval;         // [local atom][slot] = H_ij,
                            // LayoutRight: a row must be contiguous for the vector-lane reads
  bigint hmat_step = -1;    // (#30): the stored H depends only on positions/types, which are fixed
                            // WITHIN a timestep — so rebuild once per STEP, not once per solve. Without
                            // this the q0-field call and the CG each triggered their own build (23 builds
                            // for 10 steps).
  bool hmat_stale = true;   // rebuild at the start of each SOLVE (positions/types are fixed within one,
                            // and rebuilding per solve rather than per step is automatically correct for
                            // ASPC's second call)
  bool hmat_ok = true;      // false once an allocation attempt fails -> permanent matrix-free fallback
  bool hmat_msg = false;    // one-time log of which path is in use + what it costs

  // ---- (#30) phase timing (env SAMQEQ_KK_TIME=1), printed by post_run() ----
  // nsys cannot trace GPU kernels in this environment (WSL2 blocks activity tracing; API tracing
  // still works), and phase attribution by hypothesis has now been wrong twice. These are
  // fenced wall-clock timers around each phase of the device matvec — the ground truth.
  int    tm_on = -1;        // -1 = read the env var on first use
  double tm_comm = 0.0, tm_hbuild = 0.0, tm_spmv = 0.0, tm_diag = 0.0, tm_recip = 0.0, tm_proj = 0.0;
  long   tm_calls = 0, tm_builds = 0;


  // ---- S9: device q-direct CG self-check (vs host FixQEqSam::qeq_cg on the same RHS) ----
  void validate_qcg();
  bool s9_done = false;
};

}    // namespace LAMMPS_NS

#endif    // LMP_FIX_QEQ_SAM_KOKKOS_H
#endif    // FIX_CLASS
