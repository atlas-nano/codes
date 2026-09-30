/* -*- c++ -*- ----------------------------------------------------------
   samqeq_ttdamp.h — Tang-Toennies damping function for the INTERIONIC
   off-diagonal FQ kernel (`iondamp`).

   f_n(x) = 1 − e^(−x) Σ_{k=0}^{n} x^k/k!, x = b_IJ·r (b in 1/Angstrom)
   df_n/dx = e^(−x) x^n/n! (the classic TT identity)

   Applied MULTIPLICATIVELY to the active shielded Coulomb kernel J(r)
   (cbrt/pqeq/slater) for DESIGNATED ion type pairs only: J_damp = f_n·J.
   f→1 at large r (the Ewald 1/r complement is untouched); f→0 at contact
   (quenches the contact charge-transfer coupling that drives the Zn–Cl
   qZn→+4/qCl→−2 FQ sloshing pathology).

   Shared, header-only, included by BOTH FixQEqSam (solve side) and
   PairCoulShieldIntra (force/energy side) so there is exactly ONE
   transcription — force↔solve consistency by construction (the
   slater_jtable.h precedent).

   UNITS: x is dimensionless (b is a length⁻¹, NO ev_scale — A7 same family
   as the Slater zeta); f is a pure factor on the raw 1/Angstrom kernel.
-------------------------------------------------------------------------*/

#ifndef LMP_SAMQEQ_TTDAMP_H
#define LMP_SAMQEQ_TTDAMP_H

#include <cmath>

namespace LAMMPS_NS {

/* f_n(x) with its exact derivative. n >= 1 (parser enforces 1..8).
   x = 0 ⇒ f = 0, df/dx = 0 (n>=1); x large ⇒ f → 1, df/dx → 0.*/
static inline double samqeq_tt_damp(double x, int n, double &dfdx)
{
  double term = 1.0, sum = 1.0;                 // term_k = x^k/k!, running
  for (int k = 1; k <= n; k++) { term *= x / (double) k; sum += term; }
  const double ex = exp(-x);
  dfdx = ex * term;                             // e^(−x)·x^n/n! (all lower terms telescope)
  return 1.0 - ex * sum;
}

}    // namespace LAMMPS_NS
#endif
