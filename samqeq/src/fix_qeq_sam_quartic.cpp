// clang-format off
/* ----------------------------------------------------------------------
   samQEq (fix qeq/sam): the quartic on-site diagonal. A member of class FixQEqSam
   (declared in fix_qeq_sam.h) in its own translation unit.
-------------------------------------------------------------------------*/

#include "fix_qeq_sam.h"

#include "atom.h"
#include "comm.h"
#include "memory.h"

#include <cmath>
#if defined(_OPENMP)
#include <omp.h>
#endif

using namespace LAMMPS_NS;
using namespace FixConst;

/* ----------------------------------------------------------------------
   QUARTIC near-criticality cure: fill the per-atom solve diagonal eta_diag[i] from the LAGGED charge.
     eta_eff(q) = eta[type] + 1/6 c4 q^2 + 1/2 c q (atoms in quartic_groupbit), clamped >= quartic_etafloor;
     all other atoms keep the bare eta[type].
   Reproduces the true nonlinear on-site stationarity dE/dq = chi + eta_eff(q) q for
     E = chi q + 1/2 eta q^2 + 1/6 c q^3 + 1/24 c4 q^4
   under Picard iteration (quartic_scf). q-only => variational at the SCF minimum => NO position-force term.
   Local atoms only: the matvec/preconditioner diagonal reads eta_diag[i] for i in ilist (in-group, local).
-------------------------------------------------------------------------*/
void FixQEqSam::apply_quartic_eta()
{
  int *type = atom->type, *mask = atom->mask;
  double *qa = atom->q;
  const int nlocal = atom->nlocal;
  int nth = comm->nthreads;
  // OpenMP: each i writes only its own eta_diag[i]; anh_secant_add (fix_qeq_sam.h) is a pure inline function
  // (its `e` accumulator is this call's own local reference) -- embarrassingly parallel, no race, no reduction.
#if defined(_OPENMP)
  if (omp_go(nth, nlocal)) {
#pragma omp parallel for num_threads(nth) schedule(static)
    for (int i = 0; i < nlocal; i++) {
      const int t = type[i];
      double e = eta0 ? eta0[t] : eta[t];                               // chemical seed: see the serial branch
      const double qi = qa[i];
      const bool anh = anh_secant_add(e, qi, t, mask[i] & quartic_groupbit, quartic_gate[i]);
      const double gd = eta0 ? eta[t] - eta0[t] : 0.0;                  // floor the TOTAL diagonal (serial branch)
      if (anh && e + gd < quartic_etafloor) e = quartic_etafloor - gd;
      eta_diag[i] = e;
    }
  } else
#endif
  { for (int i = 0; i < nlocal; i++) {
    const int t = type[i];
    // Seeded from the CHEMICAL eta0, not the gself-folded eta[t]: solve_diag_of adds (eta_diag - eta0) to eta[t],
    // so a seed of eta[t] would put E_self on the diagonal twice. eta0 == eta bitwise when gself is off.
    double e = eta0 ? eta0[t] : eta[t];
    const double qi = qa[i];
    // staircase + global gated quartic SECANT (dE_anh/dq)/q: the shared on-site anharmonic model
    // (fix_qeq_sam.h) — same E(q) as compute_scalar/xl_chargeforce, factored so that
    // dE/dq − χ = q·eta_eff under the Picard iteration (q-only ⇒ variational, no position-force term).
    const bool anh = anh_secant_add(e, qi, t, mask[i] & quartic_groupbit, quartic_gate[i]);
    // The floor bounds the TOTAL solve diagonal solve_diag_of = e + (eta[t] - eta0[t]), as the saddle path does
    // (compute_saddle_onsite). gd == +0.0 exactly when gself is off.
    const double gd = eta0 ? eta[t] - eta0[t] : 0.0;
    if (anh && e + gd < quartic_etafloor) e = quartic_etafloor - gd;
    eta_diag[i] = e;
  } }
  // Picard q0 shift (fix_qeq_sam.h at anh_q0shift): eta_diag[i] is the floored secant written above, so D = eta_diag - seed
  // and the stored shift D*x_k - E'(q_k), x_k = q_k - q0, makes the Picard fixed point exact. Rows with q0 = 0: no shift.
  if (anh_q0shift) {
    anh_rhs_alloc();
    for (int i = 0; i < nlocal; i++) {
      anh_rhs[i] = 0.0;
      if (!(mask[i] & groupbit) || !anh_q0shift_row(type[i], mask[i])) continue;
      const int t = type[i];
      const double base = eta0 ? eta0[t] : eta[t];
      double f = 0.0; anh_force_add(f, qa[i], t, mask[i] & quartic_groupbit, quartic_gate[i]);   // f = -E'(q_k)
      anh_rhs[i] = (eta_diag[i] - base)*(qa[i] - q0[t]) + f;
    }
  }
}

void FixQEqSam::anh_rhs_alloc()
{
  if (atom->nmax > anh_rhs_cap) { memory->destroy(anh_rhs); memory->create(anh_rhs, atom->nmax, "samqeq:anh_rhs");
    anh_rhs_cap = atom->nmax; }
}

/* Arm the Picard q0 shift iff some row can need it: an anharmonic term on a type with q0 != 0 -- a staircase type, or
   any q0 != 0 type while a quartic wall group is set. XL never reaches the quartic solve's RHS (it propagates charges
   with the exact anh_force_add), so it is excluded. Called from init(). */
void FixQEqSam::anh_q0shift_init()
{
  anh_q0shift = 0;
  if (!lr_quartic || lr_xl || !q0) return;
  for (int t = 1; t <= atom->ntypes; t++) {
    if (q0[t] == 0.0) continue;
    if (c3_type[t] != 0.0 || c4_type[t] != 0.0 || quartic_groupbit) { anh_q0shift = 1; break; }
  }
}
