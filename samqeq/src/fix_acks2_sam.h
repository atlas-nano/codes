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

// samQEq self-contained ACKS2 saddle base, derived from REAXFF/fix_acks2_reaxff so
// that samQEq does not depend on the REAXFF package. BASE CLASS ONLY — no FixStyle registration (not a user-facing fix style).
#ifndef LMP_FIX_ACKS2_SAM_H
#define LMP_FIX_ACKS2_SAM_H

#include "fix_qeq_base_sam.h"

namespace LAMMPS_NS {

class FixACKS2Sam : public FixQEqBaseSam {
 public:
  FixACKS2Sam(class LAMMPS *, int, char **);
  ~FixACKS2Sam() override;
  void post_constructor() override;
  void init() override;
  void init_storage() override;
  void pre_force(int) override;

  double *get_s() { return s; }

 protected:
  int NN, last_rows_rank, last_rows_flag;

  double **s_hist_X, **s_hist_last;
  double *bcut_acks2, bond_softness, **bcut;    // acks2 parameters
  double h_ridge = 0.0;    // SAMQEQ: Tikhonov ridge added to the H-block diagonal in the
                           // matvec (eta + h_ridge), set by FixQEqSam from lr_ridge to
                           // regularize the metal-limit near-singular saddle solve. 0 = off.

  sparse_matrix X;
  double *Xdia_inv;
  double *X_diag;
  double *onsite_extra = nullptr;   // optional per-atom ADDITION to the on-site eta diagonal (owned/pointed
                                    // by a derived fix; e.g. FixQEqSam's quartic secant on the saddle path).
                                    // nullptr = off; read in sparse_matvec_acks2 + Hdia_inv.

  //BiCGStab storage
  double *g, *q_hat, *r_hat, *y, *z;

  void init_bondcut();
  void allocate_storage() override;
  void deallocate_storage() override;
  void allocate_matrix() override;
  void deallocate_matrix() override;

  void compute_X();    // NOLINT
  double calculate_X(double, double);

  virtual int BiCGStab(double *, double *);   // virtual: FixQEqSamKokkos overrides with a device solve
  int acks2_minres(double *, double *);       // MINRES on the symmetric-INDEFINITE KKT saddle (robust where
                                              // BiCGStab breaks down rho=0). Unpreconditioned (Jacobi precon is
                                              // indefinite via the negative X-block). Selected by acks2_use_minres.
  int acks2_use_minres = 0;                   // 0 = BiCGStab (default); 1 = MINRES saddle solver
  int acks2_saddle_refused = 0;               // latch -- ASPC has been refused on this saddle (warn once).
  int aspc_saddle_allow = 0;                  // opt-in escape hatch, `fix_modify <id> aspc saddle allow`.
                                              // Default 0 = enforce the aspc_setup() exclusion ("NOT engaged for
                                              // the ACKS2 saddle"): on the saddle ASPC saves little wall-clock
                                              // at a large loss of charge conservation and CT accuracy, and
                                              // aspc_rtol has no window that is both accurate and accepting.
  double acks2_relresid0 = 0.0;               // rel-residual of the ENTRY guess (the ASPC predictor), same
                                              // normalisation as acks2_relresid. The accept gate must bound the
                                              // vector actually COMMITTED, w*s + (1-w)*s_pred, not just s.
  int acks2_capped = 0;                       // 1 only while the ASPC-capped corrector solve is running, so the
                                              // "did not converge" warning stays silent for a cap that is BY DESIGN
                                              // but still fires for the uncapped fall-through solve.
  int acks2_exhausted = 0;                    // 1 if the LAST saddle solve ended by exhausting its iteration
                                              // budget or by a BiCGStab breakdown (omega/rho = 0) -- the failure
                                              // signature the no-commit escalation keys on. BiCGStab's documented
                                              // early exit (|q|^2 < tol) can leave rel-residual ~1e-4 at tol 1e-5
                                              // on a CONVERGED solve, so the residual alone is not a failure test.
  double acks2_relresid = 0.0;                // rel-residual of the LAST saddle solve (BiCGStab rnorm/bnorm,
                                              // MINRES phibar/beta1). Mirrors the CG path's cg_relresid so the
                                              // ASPC corrector can be quality-gated on the saddle too.

  // preconditioner hook (default = diagonal Jacobi); FixQEqSam overrides to add an ILU
  // saddle preconditioner for the ill-conditioned metal limit. out = M^{-1} in.
  virtual void precond_apply(double *in, double *out);
  // pre_force calls this when an UNCAPPED saddle solve ended GROSSLY unconverged (rel-residual >
  // 10x tolerance). Default: no fallback (-1) => pre_force refuses to commit. FixQEqSam overrides it with
  // one retry under the block-ILUT preconditioner: the saddle of a monatomic ion (tiny X row)
  // can stall diagonal-BiCGStab while ILU on the SAME operator converges in tens of matvecs.
  // Returns matvecs used by the retry, or -1.
  virtual int saddle_fallback(double * /*b*/, double * /*x*/) { return -1; }
  void sparse_matvec_acks2(sparse_matrix *, sparse_matrix *, double *, double *);

  int pack_forward_comm(int, int *, double *, int, int *) override;
  void unpack_forward_comm(int, int, double *) override;
  int pack_reverse_comm(int, int, double *) override;
  void unpack_reverse_comm(int, int *, double *) override;
  void more_forward_comm(double *);
  void more_reverse_comm(double *);
  double memory_usage() override;
  void grow_arrays(int) override;
  void copy_arrays(int, int, int) override;
  int pack_exchange(int, double *) override;
  int unpack_exchange(int, double *) override;

  double parallel_norm(double *, int) override;
  double parallel_dot(double *, double *, int) override;
  double parallel_vector_acc(double *, int) override;

  void vector_sum(double *, double, double *, double, double *, int) override;
  void vector_add(double *, double, double *, int) override;
  void vector_copy(double *, double *, int);
};

}    // namespace LAMMPS_NS

#endif
