// clang-format off
/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation. Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software. This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
-------------------------------------------------------------------------*/

/* ----------------------------------------------------------------------
   Contributing author: Hasan Metin Aktulga, Purdue University
   (now at Lawrence Berkeley National Laboratory, hmaktulga@lbl.gov)

   Please cite the related publication:
   H. M. Aktulga, J. C. Fogarty, S. A. Pandit, A. Y. Grama,
   "Parallel Reactive Molecular Dynamics: Numerical Methods and
   Algorithmic Techniques", Parallel Computing, in press.
-------------------------------------------------------------------------*/

// samQEq self-contained ACKS2/QEq base, ported from REAXFF/fix_qeq_reaxff to drop
// the REAXFF-package dependency. This is a
// BASE CLASS ONLY — no FixStyle registration (not a user-facing fix style).
#ifndef LMP_FIX_QEQ_BASE_SAM_H
#define LMP_FIX_QEQ_BASE_SAM_H

#include "fix.h"

namespace LAMMPS_NS {

class FixQEqBaseSam : public Fix {
 public:
  FixQEqBaseSam(class LAMMPS *, int, char **);
  ~FixQEqBaseSam() override;
  int setmask() override;
  void post_constructor() override;
  void init() override;
  void init_list(int, class NeighList *) override;
  virtual void init_storage();
  void setup_pre_force(int) override;

  void setup_pre_force_respa(int, int) override;
  void pre_force_respa(int, int, int) override;

  void min_setup_pre_force(int);
  void min_pre_force(int) override;

  double compute_scalar() override;

 protected:
  int nevery, reaxflag;
  int matvecs;
  int nn, m_fill;
  int n_cap, nmax, m_cap;
  int pack_flag;
  int nlevels_respa;
  class NeighList *list;
  class FixEfield *efield;
  int *ilist, *jlist, *numneigh, **firstneigh;

  double swa, swb;     // lower/upper Taper cutoff radius
  double Tap[8];       // Taper function
  double tolerance;    // tolerance for the norm of the rel residual in CG

  // A7 (units-real policy): internal hardcoded constants across
  // samQEq are written eV-ANCHORED (as if units metal); ev_scale converts them to the deck's actual
  // energy units at point of use. Charge (e), Angstrom, and fs/ps constants are NEVER touched by this.
  // Set ONCE in the FixQEqBaseSam ctor (update->unit_style is valid there -- the `units` command
  // precedes every fix). 1.0 in metal (byte-identical: every `x*ev_scale` / `x/sqrt(ev_scale)` below is
  // then an exact x*1.0 / x/1.0, IEEE754-exact, so the 23-case metal suite is untouched).
  double ev_scale;

  double *chi, *eta, *gamma;    // qeq parameters
  double **shld;

  // fictitious charges

  double *s, *t;
  double **s_hist, **t_hist;
  int nprev;

  // ---- ASPC q-direct predictor-corrector (#16, opt-in) ----
  // UNTESTED draft (session 21): default aspc_on=0 => exact Born-Oppenheimer path (bit-suite unchanged).
  // When on: predict q from history (time-reversible Kolafa coeffs), n_corr projected-CG corrector iters,
  // omega-damped mix; the corrector acts on q DIRECTLY (constrained), NOT the s/t split (the prototype showed
  // the s/t cancellation amplifies under-convergence -> garbage). q_hist is NON-migrating: reset on neighbor
  // rebuild (local indices stay valid within a reneighbor interval). Targets 8-13x fewer matvecs at BO-quality NVE.
  int aspc_on;          // 0 = BO (default), 1 = ASPC
  int aspc_ncorr;       // corrector projected-CG iterations per solve
  int aspc_korder;      // predictor order k (history length = k+2)
  int aspc_nhist;       // = k+2
  int aspc_have;        // # valid history entries (reset on neighbor rebuild)
  bigint aspc_build;    // neighbor->ncalls at last solve (rebuild detector)
  bigint aspc_naccept;  // S1b diag: cumulative corrector ACCEPTS (commit path) -- SOLVER-DIAG acc/rej
  bigint aspc_nreject;  // S1b diag: cumulative corrector REJECTS (fell through to the full BO solve)
  double aspc_omega;    // corrector damping = (k+2)/(2k+3)
  double aspc_rtol;     // accept the capped corrector only if its rel-residual < aspc_rtol (else fall to BO).
                        // Gates on solve QUALITY, not just max|q|.
  double cg_relresid;   // last qeq_cg relative residual sqrt(|signew|)/bnorm (read by the ASPC accept gate)
  double aspc_B[8];     // predictor coefficients (length aspc_nhist)
  double **q_hist;      // [nmax][aspc_nhist] corrected-charge history (non-migrating; reset on rebuild)
  bigint ngroup_fq;     // global FQ-group atom count (for the mean-zero projection)

  // NOLINTBEGIN
  typedef struct {
    int n, m;
    int *firstnbr;
    int *numnbrs;
    int *jlist;
    double *val;
  } sparse_matrix;
  // NOLINTEND

  sparse_matrix H;
  double *Hdia_inv;
  double *b_s, *b_t;
  double *b_prc, *b_prm;
  double *chi_field;

  //CG storage
  double *p, *q, *r, *d;
  int imax, maxwarn;

  char *pertype_option;    // argument to determine how per-type info is obtained
  virtual void pertype_parameters(char *) = 0;   // FixQEqSam supplies the parameter-file reader
  void init_shielding();
  void init_taper();
  virtual void allocate_storage();
  virtual void deallocate_storage();
  void reallocate_storage();
  virtual void allocate_matrix();
  virtual void deallocate_matrix();
  void reallocate_matrix();
  void ensure_matrix_capacity();   // grow H from the CURRENT neighbour list (see .cpp)

  virtual void init_matvec() = 0;
  void init_H();
  virtual void compute_H();
  double calculate_H(double, double);
  // calc_Hval(r,ti,tj): per-type-pair hook for compute_H's H.val (added for `shield slater` -- see
  // FixQEqSam::calc_Hval, fix_qeq_sam.cpp). Default = the ORIGINAL inline shld/calculate_H lookup
  // (byte-identical for every class that doesn't override it, and for FixQEqSam itself whenever
  // shield_gauss != SHIELD_SLATER). Only compute_H's ONE call site was changed to route through this
  // (calculate_H's own signature/callers, incl. the kokkos host helpers, are UNTOUCHED).
  virtual double calc_Hval(double r, int ti, int tj);
  virtual void calculate_Q() = 0;

  // ASPC (#16): coefficient setup (the predictor-corrector itself runs in FixQEqSam::qeq_solve)
  void aspc_setup();

  int pack_forward_comm(int, int *, double *, int, int *) override;
  void unpack_forward_comm(int, int, double *) override;
  int pack_reverse_comm(int, int, double *) override;
  void unpack_reverse_comm(int, int *, double *) override;
  double memory_usage() override;
  void grow_arrays(int) override;
  void copy_arrays(int, int, int) override;
  int pack_exchange(int, double *) override;
  int unpack_exchange(int, double *) override;

  virtual double parallel_norm(double *, int);
  virtual double parallel_dot(double *, double *, int);
  virtual double parallel_vector_acc(double *, int);

  virtual void vector_sum(double *, double, double *, double, double *, int);
  virtual void vector_add(double *, double, double *, int);

  virtual void get_chi_field();

  int matvecs_s, matvecs_t;    // Iteration count for each system
};

}    // namespace LAMMPS_NS

#endif
