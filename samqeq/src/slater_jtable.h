/* -*- c++ -*- ----------------------------------------------------------
   slater_jtable.h — Rick's Slater-orbital Coulomb-overlap kernel J(r), tabulated.

   Rick, Stuart & Berne, JCP 101, 6141 (1994) (TIP4P-FQ): the fluctuating-charge
   sites are Slater charge densities (1s for H, 2s for O/M), so the "shielded"
   Coulomb hardness between two sites is not an ad-hoc closed form (cbrt / erf)
   but the EXACT overlap integral of two Slater densities, J(r). This header is
   the SINGLE transcription of that kernel (`slater_jraw`, a direct port of
   Rick's compiled mdtpn.f k-space quadrature, itself validated against
   examples/tip4pfq_liquid/jrab.py::jrab_raw -- see check_slater_table.py for
   the parity number) plus a lazily-built per-type-pair TABLE (`SlaterJTable`)
   so the O(nint) quadrature (nint=1000 k-points) runs ONLY at init/setup, never
   in the per-step force/solve loop. Shared, header-only, so FixQEqSam
   (fix_qeq_sam.cpp/.h) and PairCoulShieldIntra (pair_coul_shield_intra.cpp/.h)
   build/evaluate IDENTICAL tables from the SAME code -- the whole point of the
   exact kernel is force<->solve consistency, so there must be only one
   implementation to keep in sync.

   UNITS: slater_jraw returns the RAW value in 1/Angstrom (same convention as
   the cbrt/erf shielded kernels) -- the caller multiplies by
   force->qqrd2e to get energy in the deck's native units (eV in metal, kcal/mol
   in real). zeta is a length^-1 (1/Angstrom) -- NO ev_scale involved.
-------------------------------------------------------------------------*/

#ifndef LMP_SLATER_JTABLE_H
#define LMP_SLATER_JTABLE_H

#include "math_const.h"

#include <cmath>
#include <vector>

namespace LAMMPS_NS {

/* ----------------------------------------------------------------------
   slater_jraw(r, i2s, j2s, zeta_i, zeta_j, J, dJdr): EXACT transcription of
   jrab_raw (examples/tip4pfq_liquid/jrab.py), which is itself transcribed from
   Rick's compiled mdtpn.f and validated against the paper's checkpoints
   (J0_OO 371.6, J0_HH 353.0, J_MH(0.8735)=286.44, J_HH(1.5139)=203.61
   kcal/mol/e^2). kmax/nint are RICK'S OWN quadrature choices (8.0 / 1000) --
   this is the physics, not a numerics knob, and check_slater_table.py's parity
   check assumes this EXACT discretization, so do not "improve" it.

   i2s/j2s select the 2s (O/M) vs 1s (H) Slater form factor (fa/fb below); the
   `fk1^1.5` term (note the .5 power) must be transcribed exactly -- it is easy
   to typo as fk1^2 or drop the 4*zeta^2 denominator, both of which silently
   give a plausible-looking but wrong J(r).

   dJdr is the analytic r-derivative under the integral sign (Leibniz rule):
   only the sinc(k r) factor depends on r, so d(sinc)/dr is differentiated in
   closed form and integrated with the SAME fa*fb weights and k-grid as J
   itself (guarantees dJdr is the exact derivative of the SAME discretized J,
   not a separately-converged quantity -- important for force<->energy
   consistency at the table's finite quadrature resolution).
-------------------------------------------------------------------------*/
inline void slater_jraw(double r, bool i2s, bool j2s, double zeta_i, double zeta_j,
                         double &J, double &dJdr)
{
  constexpr double kmax = 8.0;     // Rick's quadrature upper limit (1/Angstrom)
  constexpr int    nint = 1000;    // Rick's quadrature point count
  const double dk = kmax / nint;

  const double zi2 = zeta_i * zeta_i, zi4 = zi2 * zi2;
  const double zj2 = zeta_j * zeta_j, zj4 = zj2 * zj2;

  double sum  = 0.5 * dk;   // Fortran: sum=0.5*dk (k=0 half-weight trapezoid endpoint); r-independent
  double dsum = 0.0;        // the k=0 term is a constant in r -> contributes 0 to dJ/dr

  for (int n = 1; n <= nint; n++) {
    const double k = n * dk;

    const double aa1 = 1.0 + (k / (2.0 * zeta_i)) * (k / (2.0 * zeta_i));
    const double aa2 = 1.0 + (k / (2.0 * zeta_j)) * (k / (2.0 * zeta_j));
    const double fk1 = 1.0 / (aa1 * aa1);
    const double fk2 = 1.0 / (aa2 * aa2);
    const double k2 = k * k, k4 = k2 * k2;

    // 2s (oxygen/M) form factor: fk1 - 3k^2 fk1^1.5/(4 zeta^2) + k^4 fk1^2/(8 zeta^4) (note the .5 power)
    const double fa = i2s ? (fk1 - 3.0 * k2 * std::pow(fk1, 1.5) / (4.0 * zi2) + k4 * fk1 * fk1 / (8.0 * zi4))
                          : fk1;
    const double fb = j2s ? (fk2 - 3.0 * k2 * std::pow(fk2, 1.5) / (4.0 * zj2) + k4 * fk2 * fk2 / (8.0 * zj4))
                          : fk2;

    const double u = k * r;
    double sinc, dsinc_dr;
    if (u < 1.0e-12) {
      // r->0 limit: sin(kr)/(kr) -> 1, its derivative -> 0 (J(r) is an even function of r). The table
      // domain never actually reaches r=0 (rmin clamp in SlaterJTable::build), this guard is just
      // defensive against a stray r=0 probe.
      sinc = 1.0; dsinc_dr = 0.0;
    } else {
      const double su = std::sin(u), cu = std::cos(u);
      sinc     = su / u;
      dsinc_dr = (u * cu - su) / (k * r * r);     // d/dr [sin(kr)/(kr)], Leibniz (fa,fb are r-independent)
    }
    sum  += sinc     * fa * fb * dk;
    dsum += dsinc_dr * fa * fb * dk;
  }
  J    = sum  * 2.0 / MathConst::MY_PI;
  dJdr = dsum * 2.0 / MathConst::MY_PI;
}

/* ----------------------------------------------------------------------
   SlaterJTable: one per-type-pair table of J(r)/dJ/dr on a uniform grid
   r in [rmin, swb], Hermite-cubic evaluated (C1-continuous -> smooth forces).
   Built ONCE at init/setup (the O(nint) quadrature above runs npts times, all
   at build() -- never per force/solve call). MEMORY: 2 doubles/point * npts
   (default 2000) = 32 KB per table; a system with nt types has
   nt*(nt+1)/2 unique (ti<=tj) tables, e.g. nt=5 -> 15 tables -> ~480 KB total
   (rank-local, MPI-replicated -- the quadrature is deterministic so every rank
   builds the identical table with no communication).
-------------------------------------------------------------------------*/
struct SlaterJTable {
  double rmin = 0.02;   // finite-at-r->0 clamp (below this, return the rmin node -- matches the physical
                        // J(0) closed form being finite, unlike bare 1/r)
  double rmax = 0.0;
  double dr = 0.0;
  int npts = 0;
  std::vector<double> Jv, dJv;   // Jv[n], dJv[n] at r = rmin + n*dr, n = 0..npts-1

  void build(bool i2s, bool j2s, double zeta_i, double zeta_j, double swb, int npoints = 2000)
  {
    npts = npoints;
    rmax = swb;
    dr = (rmax - rmin) / (npts - 1);
    Jv.assign(npts, 0.0);
    dJv.assign(npts, 0.0);
    for (int n = 0; n < npts; n++) {
      const double r = rmin + n * dr;
      slater_jraw(r, i2s, j2s, zeta_i, zeta_j, Jv[n], dJv[n]);
    }
  }

  // Hermite-cubic evaluate: C1-continuous J(r) + dJ/dr from the stored (value, derivative) pair at each
  // node -- exact force<->energy consistency at the nodes, smooth in between (unlike piecewise-linear J
  // with a naive finite-difference force, which is only C0 and has a discontinuous force at each node).
  inline double eval(double r, double &dJdr_out) const
  {
    if (r <= rmin) { dJdr_out = dJv[0]; return Jv[0]; }        // finite-at-r->0 clamp
    double x = (r - rmin) / dr;
    int n0 = (int) x;
    if (n0 >= npts - 1) n0 = npts - 2;                          // clamp at/above the top node (r ~ swb)
    const double t = x - n0;
    const double h = dr;
    const double J0 = Jv[n0], J1 = Jv[n0 + 1], D0 = dJv[n0], D1 = dJv[n0 + 1];
    const double t2 = t * t, t3 = t2 * t;
    const double h00 = 2.0*t3 - 3.0*t2 + 1.0, h10 = t3 - 2.0*t2 + t;
    const double h01 = -2.0*t3 + 3.0*t2,       h11 = t3 - t2;
    const double h00d = 6.0*t2 - 6.0*t,        h10d = 3.0*t2 - 4.0*t + 1.0;
    const double h01d = -6.0*t2 + 6.0*t,       h11d = 3.0*t2 - 2.0*t;
    dJdr_out = (h00d*J0 + h10d*h*D0 + h01d*J1 + h11d*h*D1) / h;
    return h00*J0 + h10*h*D0 + h01*J1 + h11*h*D1;
  }
};

// symmetric (ti<=tj) triangular index into a flat vector of nt*(nt+1)/2 tables, 1-based types in [1,nt].
inline int slater_tri_index(int ti, int tj, int nt)
{
  if (ti > tj) { int tmp = ti; ti = tj; tj = tmp; }
  const int row_offset = (ti - 1) * nt - (ti - 1) * (ti - 2) / 2;
  return row_offset + (tj - ti);
}

}    // namespace LAMMPS_NS
#endif
