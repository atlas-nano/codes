// clang-format off
/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing authors:
     Tod A Pascal (UCSD)
------------------------------------------------------------------------- */

#include "fix_xpt.h"

#include "atom.h"
#include "citeme.h"
#include "comm.h"
#include "compute.h"
#include "domain.h"
#include "error.h"
#include "force.h"
#include "group.h"
#include "memory.h"
#include "modify.h"
#include "input.h"
#include "pair.h"
#include "update.h"
#include "variable.h"

#include "fft3d_wrap.h"       // FFT3d + FFT_SCALAR (backend-agnostic: KISS/FFTW3/MKL)
#ifdef FFT_FFTW3
#  include <fftw3.h>           // fftw_forget_wisdom cleanup at shutdown
#endif
#include "math_eigen.h"       // MathEigen::jacobi3 (molecular inertia-tensor diagonalization)
#include "math_eigen_impl.h"  // MathEigen::Jacobi<>

#include <algorithm>
#include <cassert>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <strings.h>     // strcasecmp (POSIX)
#include <map>
#include <set>
#include <vector>

// LAMMPS mask bits
#include "fix.h"

// Shared physical constants + FIXXPT_LOG / FIXXPT_TIME diagnostic macros,
// split out so the sibling fix_xpt_*.cpp translation units share one
// definition (defines FIX_XPT_DEBUG; include after the LAMMPS headers).
#include "fix_xpt_const.h"

using namespace LAMMPS_NS;
using namespace FixConst;

// ============================================================================
// fix_xpt_entropy.cpp — hard-sphere / Desjarlais / cage / R2PT
// entropy machinery of `fix xpt` (FixXPT:: methods in a separate translation
// unit; declarations in fix_xpt.h).  Shares constants + FIXXPT_LOG/TIME via
// fix_xpt_const.h.
// ============================================================================
/* ======================================================================
   Desjarlais (2013) memory-function 2PT  — mode 3

   dawson_f(y): Dawson function F(y) = exp(-y²)·∫₀ʸ exp(t²)dt via Rybicki's
     method (Taylor series for |y|<0.2).

   sgmf_des(nu, s0, f_g, Bg, nmol): gas DoS at frequency nu using the
     Gaussian memory-function kernel with shape parameter Bg.
     A = f_g/s0·√(4Bg/π) is derived from the constraint S_gas(0)=s0.

   refine_Bg_des: bisection to find Bg such that sgmf_des at the last
     non-noise bin of pwr matches the actual DoS tail.
====================================================================== */

double FixXPT::dawson_f(double x)
{
  // Dawson integral  D(x) = e^{-x²} ∫₀ˣ e^{t²} dt  via Rybicki's method
  // (Numerical Recipes §6.10), accurate to ~1e-8 across the whole range
  // (matches scipy.special.dawsn).
  const double H  = 0.4;
  const double A1 = 2.0/3.0, A2 = 0.4, A3 = 2.0/7.0;
  const double ONE_OVER_SQRT_PI = 0.5641895835477563;

  if (std::fabs(x) < 0.2) {
    const double x2 = x*x;
    return x * (1.0 - A1*x2*(1.0 - A2*x2*(1.0 - A3*x2)));
  }

  // Tabulated weights c_i = exp(-((2i-1)H)²), i = 1..6 (cheap: 6 exp's,
  // mode-3 only).  Kept local to avoid mutable-static thread concerns.
  double c[7];
  for (int i = 1; i <= 6; i++) { const double t = (2.0*i - 1.0)*H; c[i] = std::exp(-t*t); }

  const double xx = std::fabs(x);
  const int    n0 = 2 * (int)(0.5*xx/H + 0.5);
  const double xp = xx - n0*H;
  double e1 = std::exp(2.0*xp*H);
  const double e2 = e1*e1;
  double d1 = n0 + 1.0;
  double d2 = d1 - 2.0;
  double sum = 0.0;
  for (int i = 1; i <= 6; i++, d1 += 2.0, d2 -= 2.0, e1 *= e2)
    sum += c[i] * (e1/d1 + 1.0/(d2*e1));
  double ans = ONE_OVER_SQRT_PI * std::exp(-xp*xp) * sum;
  return (x < 0.0) ? -ans : ans;
}

double FixXPT::sgmf_des(double nu, double s0, double f_g, double Bg, double nmol)
{
  if (Bg <= 0.0 || f_g <= 0.0 || s0 <= 0.0) return 0.0;
  if (nu == 0.0) return s0;           // S_gas(0) = s0 by construction
  double A   = f_g / s0 * sqrt(4.0 * Bg / PI);
  double vv  = PI * nu / 6.0 / nmol;
  double y   = vv / sqrt(4.0 * Bg);
  double D   = dawson_f(y);
  double exp_y2  = exp(-y*y);
  double exp_2y2 = exp_y2 * exp_y2;
  double denom = A*A*PI/(4.0*Bg)*exp_2y2
               + (A*D)*(A*D)/Bg - 4.0*A*y*D + vv*vv;
  if (denom <= 0.0) return 0.0;
  return A * sqrt(PI/(4.0*Bg)) * exp_y2 * f_g / denom;
}

double FixXPT::refine_Bg_des(const std::vector<double>& pwr,
                              double s0, double dnu, double nmol, double f_g)
{
  if (s0 <= 0.0 || f_g <= 0.0 || pwr.empty()) return 0.0;

  // Last bin above noise floor (avoid fitting to numerical zeros)
  double noise = 1e-6 * s0;
  int j = (int)pwr.size() - 1;
  while (j > 0 && pwr[j] < noise) j--;
  if (j == 0) return 0.0;

  double nu_tail = j * dnu;
  double target  = pwr[j];

  // Bisection: sgmf_des increases monotonically with Bg
  double Bg_lo = 1e-3, Bg_hi = 1.0;
  double sg_lo = sgmf_des(nu_tail, s0, f_g, Bg_lo, nmol);
  double sg_hi = sgmf_des(nu_tail, s0, f_g, Bg_hi, nmol);
  while (sg_lo > target && Bg_lo > 1e-12) { Bg_lo /= 2.0; sg_lo = sgmf_des(nu_tail, s0, f_g, Bg_lo, nmol); }
  while (sg_hi < target && Bg_hi < 1e8)   { Bg_hi *= 2.0; sg_hi = sgmf_des(nu_tail, s0, f_g, Bg_hi, nmol); }

  for (int it = 0; it < 80; it++) {
    double Bg_mid = 0.5*(Bg_lo + Bg_hi);
    (sgmf_des(nu_tail, s0, f_g, Bg_mid, nmol) < target ? Bg_lo : Bg_hi) = Bg_mid;
    if ((Bg_hi - Bg_lo)/(Bg_hi + Bg_lo) < 1e-8) break;
  }
  return 0.5*(Bg_lo + Bg_hi);
}

/* ======================================================================
   search2pt — Newton-Raphson for fluidicity (Lin 2003, Eq. 34)
   Solves: P(f) = 2K^{-4.5}f^{7.5} - 6K^{-3}f^5 - K^{-1.5}f^{3.5}
                + 6K^{-1.5}f^{2.5} + 2f - 2 = 0
====================================================================== */

double FixXPT::search2pt(double K)
{
  if (K <= 0.0) return 0.0;

  double fold = 0.0;
  double fnew = 0.7293 * pow(K, 0.5727);
  if (fnew > 0.5) fnew = 0.5;

  const double tol = 1e-10;
  int count = 0;
  while (fabs(fnew - fold) > tol && count < 999) {
    fold   = fnew;
    double P    = 2.0*pow(K,-4.5)*pow(fnew,7.5) - 6.0*pow(K,-3.0)*pow(fnew,5.0)
                - pow(K,-1.5)*pow(fnew,3.5) + 6.0*pow(K,-1.5)*pow(fnew,2.5)
                + 2.0*fnew - 2.0;
    double dPdf = 15.0*pow(K,-4.5)*pow(fnew,6.5) - 30.0*pow(K,-3.0)*pow(fnew,4.0)
                - 3.5*pow(K,-1.5)*pow(fnew,2.5) + 15.0*pow(K,-1.5)*pow(fnew,1.5)
                + 2.0;
    fnew = fold - P / dPdf;
    count++;
  }
  return fnew;
}




/* ======================================================================
   hs_entropy — hard-sphere Sackur-Tetrode + HS-EOS excess entropy
   per degree of freedom (in units of R), divided by HSDF/3 to give
   a per-mode weighting compatible with DOS integration.

   Returns ws such that S_gas = HSDF * ws * R   [J/(mol·K)]

                   For a one-component fluid the two reduce to identical
                   forms; the bmcsl branch is here for future mixture
                   extensions.
   hs_entropy_m    0 = rigorous (default; thermodynamically exact, no ln Z)
                   1 = lin2003  (legacy; includes the +ln Z term that arises
                                  from an NPT/NVT ensemble mismatch in the
                                  Sackur-Tetrode reference — see memory entry
                                  lin2003_compressibility_term_origin.md)
====================================================================== */

double FixXPT::hs_entropy(double y, double mass_per_atom,
                           double natom, double T, double V_m3,
                           int hs_entropy_m)
{
  // natom <= 0 is the "no gas phase" limit (hsdf → 0; callers downstream
  // multiply by hsdf so the contribution is physically zero).  Sackur-Tetrode
  // log(λ³V/N) is +inf there, and the caller's `S_gas = hsdf · ws_hs` would
  // produce 0·inf = NaN.  Match hs_entropy_lj which already has this guard,
  // and guard the same nmol<=0 case.
  if (y >= 0.74 || T <= 0.0 || V_m3 <= 0.0 || natom <= 0.0) return 0.0;

  double m_kg = mass_per_atom * 1e-3 / NA;   // kg per atom (mass in g/mol)

  // Ideal gas Sackur-Tetrode (indistinguishable particles)
  double ST = 2.5 + log(pow(2.0*PI*m_kg*KB*T/(H_SI*H_SI), 1.5) * V_m3 / natom);

  // Excess HS entropy (per HS DOF / 3 in our convention).
  // CS:    A^ex / NkT = y(4 - 3y) / (1 - y)²
  //        → S^ex / Nk = -A^ex/NkT  (rigorous)
  // BMCSL one-component:  identical to CS for monodisperse fluid.
  // Lin2003 convention adds ln Z (= ln[(1+y+y²-y³)/(1-y)³] for CS):
  //        wsehs_lin2003 = ln Z - A^ex/NkT
  double CS_exact = 0.0;
  double Z_log    = 0.0;
  if (y > 0.0 && y < 0.99) {
    // Carnahan-Starling (default)
    //   A^ex/NkT = y(4 - 3y) / (1-y)²        ⇔  S^ex/Nk = y(3y-4)/(1-y)²
    //   Z_CS log = ln[ (1 + y + y² - y³) / (1-y)³ ]
    CS_exact = y * (3.0*y - 4.0) / ((1.0 - y)*(1.0 - y));
    Z_log    = log( (1.0 + y + y*y - y*y*y) / pow(1.0 - y, 3.0) );
  }
  double wsehs = (hs_entropy_m == 1)
                 ? (Z_log + CS_exact)            // legacy lin2003 convention
                 : CS_exact;                     // rigorous (default)

  return (ST + wsehs) / 3.0;
}

/* ======================================================================
   r2pt_entropy — Sun et al. (2017) revised-2PT translational entropy
   [k_B per atom, d = 3].

   R2PT vs rigorous-HS 2PT: (i) gas fraction from f^δ = D/D₀ (Eq A8;
   δ default 1.5) instead of δ = 1; (ii) F_a-inclusive — gas entropy via
   the sum rule S_g = 3 f W_g and solid from the FULL F_s = F − f·F_g (no
   high-ν truncation); (iii) ln-z-free HS excess (= rigorous convention).
   ``f_delta1`` is the standard δ = 1 fluidicity; Δ is δ-independent, so it
   is recovered from f_delta1 (Eq A8) and re-solved at δ.  ``dos`` is the
   per-group EXTENSIVE DoS (∫ dν = 3·nmol); divided by nmol internally to
   the per-atom DoS the construction expects.  Real/metal units only (the
   Sackur-Tetrode term is SI).  Returns NaN on degenerate input.
====================================================================== */
double FixXPT::r2pt_entropy(double dnu, const std::vector<double>& dos,
                            int nused, double f_delta1, double T_K,
                            double mass_amu, double vol_A3, double nmol,
                            double delta) const
{
  const double NaN = std::numeric_limits<double>::quiet_NaN();
  if (nused < 2 || nmol <= 0.0 || (int)dos.size() < nused || dos[0] <= 0.0
      || !(f_delta1 > 0.0 && f_delta1 < 1.0)) return NaN;

  // Eq A8 with x = Δ^(-3/2)
  auto A8 = [](double f, double x, double dl) {
    return pow(x,3)*pow(f,3.0+4.5*dl) - 3.0*pow(x,2)*pow(f,3.0*dl+2.0)
         + 3.0*x*pow(f,1.0+1.5*dl) - 0.5*x*pow(f,1.0+2.5*dl) + pow(f,dl) - 1.0;
  };
  // bracketed bisection (brentq replacement); NaN if no sign change in [a,b]
  auto root = [](auto fn, double a, double b) -> double {
    double fa = fn(a), fb = fn(b);
    if (fa == 0.0) return a;
    if (fb == 0.0) return b;
    if (fa*fb > 0.0) return std::numeric_limits<double>::quiet_NaN();
    for (int it = 0; it < 300; it++) {
      double m = 0.5*(a+b), fm = fn(m);
      if (fm == 0.0 || (b-a) <= 1e-14*(std::fabs(a)+std::fabs(b))) return m;
      if (fa*fm < 0.0) { b = m; fb = fm; } else { a = m; fa = fm; }
    }
    return 0.5*(a+b);
  };

  double x = root([&](double z){ return A8(f_delta1, z, 1.0); }, 1e-6, 1e6);
  if (std::isnan(x)) return NaN;
  double fg = root([&](double ff){ return A8(ff, x, delta); }, 1e-9, 1.0-1e-12);
  if (std::isnan(fg)) return NaN;
  double gamma = x * pow(fg, 1.5*delta + 1.0);                 // Eq A4 packing
  if (!(gamma > 0.0 && gamma < 1.0)) return NaN;

  // rigorous-HS gas weight W_g per DoF: Sackur-Tetrode + Carnahan-Starling, NO ln z
  double m_kg     = mass_amu*1e-3/NA;
  double lam_inv3 = pow(2.0*PI*m_kg*KB*T_K/(H_SI*H_SI), 1.5);  // λ⁻³ [1/m³]
  double Vpa      = vol_A3*1e-30;                              // Å³ → m³
  double W_IG = (2.5 + log(lam_inv3*Vpa/fg)) / 3.0;
  double W_ex = (gamma*(3.0*gamma - 4.0)/((1.0-gamma)*(1.0-gamma))) / 3.0;
  double S_g  = 3.0*fg*(W_IG + W_ex);                          // sum-rule gas (∫F_g=3)

  // solid from the FULL F_s = F − f·F_g (Lorentzian gas); quantum W_s
  double F0    = dos[0]/nmol;
  double alpha = 12.0*fg/F0;
  double S_s   = 0.0;
  for (int j = 0; j < nused; j++) {
    double nu = j*dnu;
    double Fj = dos[j]/nmol;
    double Fg = 12.0*alpha/(alpha*alpha + 4.0*PI*PI*nu*nu);
    double Ws = 0.0;
    if (nu > 0.0) {
      double u = PLANCK*nu/T_K;                                // hc·ν/(kB·T)
      Ws = u/std::expm1(u) - std::log1p(-exp(-u));
    }
    double integ = (j == 0) ? 0.0 : (Fj - fg*Fg)*Ws;          // F_s(0)=0 cancels W_s(0) div.
    double w = (j == 0 || j == nused-1) ? 0.5 : 1.0;
    S_s += w * integ * dnu;
  }
  return S_g + S_s;
}

/* ======================================================================
   Scalar Form-B Volterra kernel K(t) of the normalized VACF cn (cn[0]=1).
   Fills K (size nvac, zeroed past the cutoff) and returns the auto cutoff
   nK (last reliable lag) from the 4 truncation guards: noise floor, finite,
   divergence clamp, swing-streak.  K(0)=2(1−Ĉ(dt))/dt² is the Form-B
   boundary (Form A gives K(0)=0, hence a wrong friction γ).  Used by
   cage_memory_entropy.
====================================================================== */
int FixXPT::volterra_kernel_scalar(const std::vector<double>& cn, int nvac,
                                   double dt, std::vector<double>& K,
                                   int nf_run)
{
  const double dt2 = dt*dt;
  K.assign(nvac, 0.0);
  K[0] = 2.0*(cn[0] - cn[1]) / dt2;                  // Form-B boundary
  const double K0abs = std::fabs(K[0]) > 0.0 ? std::fabs(K[0]) : 1.0;
  const double nf = std::sqrt(3.0) * 1e-3;           // Frobenius noise floor (diag 3×3)
  int nK = nvac - 1;
  double Kprev = std::fabs(K[0]); int swing = 0, nf_streak = 0;
  for (int n = 1; n <= nvac-2; n++) {
    // (1) noise floor → truncate.  nf_run consecutive sub-floor lags are
    // required before truncating (envelope-aware: an oscillatory librational
    // VACF passes through zero every node, so a single sub-floor sample is not
    // decay); truncate at the START of the streak, re-zeroing tentative K
    // entries inside it to keep the zero-past-cutoff contract.
    if (std::fabs(cn[n]) < nf) {
      if (++nf_streak >= nf_run) {
        nK = n - nf_streak + 1;
        for (int i = nK; i < n; i++) K[i] = 0.0;
        break;
      }
    } else nf_streak = 0;
    double cdd = (cn[n-1] - 2.0*cn[n] + cn[n+1]) / dt2;
    double s = 0.0;
    for (int j = 1; j <= n; j++) s += K[n-j]*(cn[j+1]-cn[j-1])*0.5;
    double Kn = -cdd - s;
    if (!std::isfinite(Kn)) { nK = n; break; }        // (2) conv-finite
    if (std::fabs(Kn) > 1e6*K0abs) { nK = n; break; } // (3) divergence clamp
    double Kn_norm = std::fabs(Kn);                   // (4) swing-streak
    if (n > 1 && Kprev > 1e-30) {
      double r = Kn_norm / Kprev;
      if (r > 10.0 || r < 0.1) { if (++swing >= 5) { nK = n; break; } }
      else swing = 0;
    }
    Kprev = Kn_norm;
    K[n] = Kn;
  }
  return nK;
}

/* ======================================================================
   mainlobe_cutoff — first lag n>=3 with |K[n]| < alpha·|K[0]| (the kernel
   main-lobe edge); falls back to nK_auto when no sub-alpha lag exists in
   [3, nK_auto).  Truncation-robust integration limit.
====================================================================== */
int FixXPT::mainlobe_cutoff(const std::vector<double>& K, int nK_auto, double alpha)
{
  const double K0 = std::fabs(K[0]);
  if (K0 <= 0.0) return nK_auto;
  for (int n = 3; n < nK_auto; n++)
    if (std::fabs(K[n]) < alpha*K0) return n;
  return nK_auto;
}

/* ======================================================================
   gamma_dc — DC friction γ = K̃(0) = ∫₀ K dt with trapezoid endpoint
   weights on K[:nk] (the direct DFT of the kernel at ω=0).
====================================================================== */
double FixXPT::gamma_dc(const std::vector<double>& K, int nk, double dt)
{
  if (nk < 2) return K[0]*dt;
  double s = 0.0;
  for (int n = 0; n < nk; n++) s += ((n==0||n==nk-1)?0.5:1.0)*K[n];
  return s*dt;
}

/* ======================================================================
   cage_memory_entropy — parameter-free 3PT cage-memory entropy correction
   ΔS [k_B per atom].

       ΔS = p · g(f) · ∫ cage(ν)·(1−w(ν))·(W_g − W_s(ν)) dν

   cage(ν) = const·(F_K − F_M) is the non-Markovian memory excess of the
   friction kernel: F_K = Re[1/(iω + K̃)] (full Volterra kernel) minus its
   Markovian equivalent F_M = γ/(γ²+ω²), γ = K̃(0).  K(t) is the scalar
   FORM-B Volterra kernel (K(0) = 2(1−Ĉ(dt))/dt²; required for a correct γ —
   Form A gives K(0)=0).  K̃(ω) is the direct Fourier sum (
   the truncated kernel is short so a direct DFT is identical and needs no FFT).
   w(ν)=ν²/(ν²+ν_c²) high-passes the harmonic tail; W_s=1−ln(hcν/kT) harmonic;
   W_g is the rigorous-HS per-DoF gas weight (Wg_override = wsr for the rot
   channel, which skips the HS Sackur-Tetrode/packing block).  The fluidicity
   gate g(f)=f²/(f²+f0²) enforces the f→0 harmonic-crystal limit.

   ``dos_total`` / ``dos_gas`` are PER-ATOM (∫=3); ``C_scalar`` is the clean
   un-normalised (trace/3) VACF.  Returns 0.0 on degenerate input.

   LOW-FREQUENCY SIGN STRUCTURE (verified, MB-pol water 298 K): cage(0)=0 is
   exact (F_K(0)=F_M(0)=1/γ — the diffusive pole belongs to the gas), and the
   unclipped excess is zero-sum (∫(F_K−F_M)dω = 0; both integrate to
   π/2·c(0)).  TRANS: γ is band-scale → excess positive from 0⁺, deficit
   above the band, no low-ν clipping.  ROT: γ_rot ≫ band → F_M ≈ flat 1/γ,
   so the sub-band mobility deficit (weight trapped into the librations) is
   negative excess, clipped below ν₊ ≈ 319 cm⁻¹ at 298 K.  The deficit is
   the caging seen from below.
====================================================================== */
double FixXPT::cage_memory_entropy(double dt, const std::vector<double>& C_scalar,
        int nvac, double dnu, const std::vector<double>& dos_total,
        const std::vector<double>& dos_gas, int nused, double T_K,
        double mass_amu, double vol_A3, double prefactor, int dimension,
        double Wg_override, std::vector<double>* cage_out,
        double gate_f0, double clip_eps) const
{
  if (nused < 2 || nvac < 3 || C_scalar.empty() || C_scalar[0] == 0.0
      || (int)dos_total.size() < nused || (int)dos_gas.size() < nused
      || (int)C_scalar.size() < nvac) return 0.0;

  const double tot0 = dos_total[0];
  const double beta = (clip_eps > 0.0 && tot0 > 0.0) ? 1.0/(clip_eps*tot0) : 0.0;
  auto logaddexp = [](double a, double b) {
    double m = std::max(a, b);
    return m + std::log1p(std::exp(-std::fabs(a-b)));
  };
  auto smax0 = [&](double x) {                       // smooth max(x,0)
    return (beta > 0.0) ? logaddexp(0.0, beta*x)/beta : std::max(x, 0.0);
  };
  auto smin = [&](double a, double b) {              // smooth min(a,b)
    return (beta > 0.0) ? -logaddexp(-beta*a, -beta*b)/beta : std::min(a, b);
  };

  const double c_cm_ps = VLIGHT * 1e-10;             // speed of light [cm/ps]

  // ── scalar Form-B Volterra kernel K(t) of cn = C/C[0] ───────────────────
  std::vector<double> cn(nvac);
  for (int i = 0; i < nvac; i++) cn[i] = C_scalar[i] / C_scalar[0];
  std::vector<double> K;
  int nK = volterra_kernel_scalar(cn, nvac, dt, K);
  if (nK < 1) return 0.0;

  // ── auto-cutoff friction-inflation safeguard (SPC/E fix) ────────────────
  // The auto cutoff is the right integration limit for a cleanly-decaying
  // kernel, but a *smooth* spurious tail past the main lobe — coherent, so
  // undetected by the recursion's noise/swing guards — inflates the friction
  // γ=K̃(0) and hence the Markovian reference F_M, over-counting the cage (the
  // SPC/E water failure mode: γ ~4×, trans cage 2.66→1.36 J/mol/K).  Detect it
  // by comparing γ at the auto cutoff with γ at the main-lobe cutoff (first
  // |K|<α·|K0|); when the post-main-lobe tail shifts γ by more than the
  // tolerance, fall back to the truncation-robust main-lobe cutoff (and warn).
  {
    int nK_main = mainlobe_cutoff(K, nK, CAGE_MAINLOBE_ALPHA);
    if (nK_main < nK) {
      double g_auto = gamma_dc(K, nK, dt);
      double g_main = gamma_dc(K, nK_main, dt);
      if (std::fabs(g_main) > 0.0
          && std::fabs(g_auto - g_main) > CAGE_TAIL_TOL*std::fabs(g_main)) {
        if (me == 0)
          utils::logmesg(lmp, "FixXPT::{}-{}: cage_memory_entropy: auto-cutoff "
              "friction inflated {:.2g}x by a smooth post-main-lobe kernel tail "
              "(gamma_auto={:.3g} vs gamma_main={:.3g}; t_auto={:.3g} vs "
              "t_main={:.3g} ps); falling back to the main-lobe cutoff.\n",
              id, group->names[igroup], g_auto/g_main, g_auto, g_main,
              nK*dt, nK_main*dt);
        nK = nK_main;
      }
    }
  }

  // trapezoid endpoint weights on the truncated kernel, then K̃(ω) by direct DFT
  std::vector<double> Kv(K.begin(), K.begin()+nK);
  Kv[0]    *= 0.5;
  Kv[nK-1] *= 0.5;
  const double phistep = 2.0*PI*dnu*c_cm_ps*dt;      // phase per (n·k)
  double cn_int = 0.0;                               // ∫cn dt (trapezoid)
  for (int i = 0; i < nvac; i++)
    cn_int += ((i==0 || i==nvac-1) ? 0.5 : 1.0) * cn[i] * dt;
  if (cn_int == 0.0) return 0.0;
  const double cage_const = tot0 / cn_int;

  // ── per-frequency cage(ν) ───────────────────────────────────────────────
  std::vector<double> cage(nused, 0.0);
  // γ = Re K̃(0) = dt·Σ Kv
  double gamma = 0.0;
  for (int n = 0; n < nK; n++) gamma += Kv[n];
  gamma *= dt;
  // Om0² from the DoS 2nd moment, hoisted ahead of the cage loop for the
  // high-pass ν_c downstream.  (Trapezoid dx=1 in both accumulators → cancels.)
  double num_Om = 0.0, den_Om = 0.0;
  for (int k = 0; k < nused; k++) {
    double wa = 2.0*PI*(k*dnu)*c_cm_ps;
    double wtrap = (k==0 || k==nused-1) ? 0.5 : 1.0;
    num_Om += wtrap * wa*wa * dos_total[k];
    den_Om += wtrap * dos_total[k];
  }
  const double Om0sq = (den_Om > 0.0) ? num_Om/den_Om : 0.0;  // [1/ps²]
  for (int k = 0; k < nused; k++) {
    double nu = k*dnu;
    double wa = 2.0*PI*nu*c_cm_ps;                   // angular freq [1/ps]
    double re = 0.0, im = 0.0;
    for (int n = 0; n < nK; n++) {
      double ph = phistep * (double)n * (double)k;
      re += Kv[n]*cos(ph);
      im -= Kv[n]*sin(ph);
    }
    re *= dt; im *= dt;                              // K̃(ω_k)
    double F_K = re / (re*re + (wa+im)*(wa+im));     // Re[1/(iω+K̃)]
    double F_ref = gamma / (gamma*gamma + wa*wa);     // Markovian reference F_M
    double solid2 = smax0(dos_total[k] - dos_gas[k]);
    cage[k] = smax0(smin(cage_const*(F_K - F_ref), solid2));
  }
  // ── high-pass w(ν), fluidicity f, gas weight W_g, gate, integral ────────
  double Om0 = std::sqrt(Om0sq);                     // pre-pass moment above
  double nuc = Om0 / (2.0*PI*c_cm_ps);

  double f = 0.0;                                    // ∫gas/dim
  for (int k = 0; k < nused; k++)
    f += ((k==0||k==nused-1)?0.5:1.0) * dos_gas[k] * dnu;
  f /= (double)dimension;
  if (!(f > 0.0 && f < 1.0)) return 0.0;
  if (cage_out) *cage_out = cage;                    // per-atom cage DoS (success path)

  double Wg;
  if (!std::isnan(Wg_override)) {
    Wg = Wg_override;                                // rot channel: rigid-rotor weight
  } else {
    // packing γ from f via the d-dim Enskog contact relation (bisection)
    double gmax = (dimension==3) ? 0.74 : (dimension==2 ? 0.9 : 0.999);
    auto contact = [&](double g)->double {
      if (dimension==3) return (1.0 - g/2.0)/pow(1.0-g,3);
      if (dimension==2) return (1.0 - 7.0*g/16.0)/pow(1.0-g,2);
      return 1.0/(1.0-g);
    };
    double a = 1e-12, b = gmax, fa = contact(a)-1.0/f, fb = contact(b)-1.0/f, yhs = gmax;
    if (fa*fb <= 0.0) {
      for (int it=0; it<200; it++){ double mm=0.5*(a+b), fm=contact(mm)-1.0/f;
        if (fm==0.0 || (b-a)<1e-13) { yhs=mm; break; }
        if (fa*fm<0.0){b=mm;fb=fm;} else {a=mm;fa=fm;} yhs=0.5*(a+b); }
    }
    double m_kg     = mass_amu*1e-3/NA;
    double lam_invd = pow(2.0*PI*m_kg*KB*T_K/(H_SI*H_SI), dimension/2.0);
    double Vpa      = vol_A3 * pow(1e-10, dimension);
    double hs_ex    = (yhs<=0.0) ? 0.0
                    : (dimension==3 ? yhs*(3.0*yhs-4.0)/((1.0-yhs)*(1.0-yhs))
                    : (dimension==2 ? -(9.0/8.0)*yhs/(1.0-yhs)+(7.0/8.0)*std::log(1.0-yhs)
                    :  std::log(1.0-yhs)));
    Wg = (dimension/2.0 + 1.0 + std::log(lam_invd*Vpa/f))/dimension + hs_ex/dimension;
  }

  const double gate = (f*f) / (f*f + gate_f0*gate_f0);
  double dS = 0.0;
  for (int k = 0; k < nused; k++) {
    double nu = k*dnu;
    double w  = (nu*nu) / (nu*nu + nuc*nuc);
    double Ws = (nu > 0.0) ? (1.0 - std::log(PLANCK*nu/T_K)) : 0.0;
    double integ = cage[k]*(1.0 - w)*(Wg - Ws);
    dS += ((k==0||k==nused-1)?0.5:1.0) * integ * dnu;
  }
  return prefactor * gate * dS;
}

/* ======================================================================
   hs_compressibility — hard-sphere Z(y) = pV/(NkT) at packing fraction y.

   Z_CS = (1 + y + y² − y³) / (1 − y)³   (Carnahan-Starling)
                      Boublik fit — NOT the CS one-fluid limit)
====================================================================== */
double FixXPT::hs_compressibility(double y)
{
  if (y <= 0.0)  return 1.0;        // ideal gas limit
  if (y >= 0.99) return 0.0;        // unphysical
  double one_my   = 1.0 - y;
  double one_my3  = one_my * one_my * one_my;
  return (1.0 + y + y*y - y*y*y) / one_my3;
}

/* ======================================================================
   z_sim_from_pressure -- Z = P V / (N kB T) from the LAMMPS virial.
   Replaces hs_compressibility(y) in the mu_q gas-PV term when
   use_sim_z_mode = 1; affects mu_q only, not entropies.

   Native pressure unit per LAMMPS units style:
       real     atm    -> Pa  factor 101325
       metal    bar    -> Pa  factor 1e5
       si       Pa     -> Pa  factor 1
       cgs      dyn/cm2 -> Pa  factor 0.1
       lj       reduced (formula reduces to P V / (N T) with kB = 1)
   In the LJ branch (units_lj == 1) kB is implicit (= 1), so the formula
   becomes Z = P V / (N T).  All other branches convert P to Pa and the
   group volume to m3 via the same vol_to_angst3 factor (native -> A^3
   then -> m3 via x 1e-30) used elsewhere in the engine.
====================================================================== */
double FixXPT::z_sim_from_pressure(double P_native, double V_native,
                                    int N, double T, int units_lj,
                                    double vol_to_angst3)
{
  if (N <= 0 || T <= 0.0 || V_native <= 0.0) return 1.0;

  if (units_lj) {
    // P* V* / (N T*)   (kB == 1 in LJ reduced units)
    return P_native * V_native / (static_cast<double>(N) * T);
  }

  // Non-LJ: convert native pressure to Pa.  Detect units style from LAMMPS.
  const char *us = update->unit_style;
  double press_to_pa;
  if      (us && !strcmp(us, "real"))     press_to_pa = 101325.0;     // atm -> Pa
  else if (us && !strcmp(us, "metal"))    press_to_pa = 1.0e5;        // bar -> Pa
  else if (us && !strcmp(us, "si"))       press_to_pa = 1.0;
  else if (us && !strcmp(us, "cgs"))      press_to_pa = 0.1;          // dyn/cm2 -> Pa
  else                                     press_to_pa = 101325.0;     // default

  // V_native -> m^3.  vol_to_angst3 maps native vol -> A^3; A^3 -> m^3 is x 1e-30.
  double V_m3 = V_native * vol_to_angst3 * 1.0e-30;

  const double kB = 1.380649e-23;   // J/K
  double P_pa = P_native * press_to_pa;
  return (P_pa * V_m3) / (static_cast<double>(N) * kB * T);
}

/* ======================================================================
   hs_entropy_lj -- Sackur-Tetrode + Carnahan-Starling in LJ reduced units.
   Convention: hbar=1, h=2pi, kB=1 (all in reduced units).
   de Broglie wavelength: lambda = h/sqrt(2pi m T) = sqrt(2pi/(m T))
   Returns ws such that S_gas = HSDF * ws   [kB per group]
====================================================================== */

double FixXPT::hs_entropy_lj(double y, double mass_star,
                              double natom, double T_star, double V_star,
                              double hbar_star,
                              int hs_entropy_m)
{
  if (y >= 0.74 || T_star <= 0.0 || V_star <= 0.0 || natom <= 0.0) return 0.0;

  double ST = 2.5 + log(pow(mass_star * T_star / (2.0 * PI * hbar_star * hbar_star), 1.5)
                        * V_star / natom);

  // Excess HS entropy — same hs_entropy_m logic as the real-units path;
  // the dimensionless form is identical (no SI prefactors).
  double CS_exact = 0.0;
  double Z_log    = 0.0;
  if (y > 0.0 && y < 0.99) {
    CS_exact = y * (3.0*y - 4.0) / ((1.0 - y)*(1.0 - y));
    Z_log    = log( (1.0 + y + y*y - y*y*y) / pow(1.0 - y, 3.0) );
  }
  double wsehs = (hs_entropy_m == 1) ? (Z_log + CS_exact) : CS_exact;

  return (ST + wsehs) / 3.0;
}

/* ======================================================================
   hs_entropy_rot — rigid-rotor gas entropy weighting for real/metal units.
   I1, I2, I3 in (g/mol)*Å² (principal moments; I3=0 for linear).
   natom_hs_rot = gas fraction of rotational DOF (hsdf_rot / ndof_rot).
   Returns ws such that S_rot_gas = HSDF_rot * ws * R  [J/(mol·K)].

   For nonlinear: S_rot = R * [3/2 + ln(sqrt(pi) * sqrt(I1*I2*I3) * (8pi^2 kT)^(3/2) / (sigma*h^3))]
   For linear:    S_rot = R * [1   + ln(8pi^2 * I * kT / (sigma * h^2))]
====================================================================== */

double FixXPT::hs_entropy_rot(double y, double I1, double I2, double I3,
                               double natom_hs_rot, double T, int sigma_rot, bool linear)
{
  (void)y; (void)natom_hs_rot;  // packing fraction not used in rigid-rotor formula
  if (T <= 0.0 || I1 <= 0.0) return 0.0;
  // Convert I from (g/mol)*Å² → kg*m²/molecule
  double fac = 1e-3 / NA * 1e-20;  // (g/mol)*Å² → kg·m²/molecule
  double I1k = I1 * fac;
  double I2k = (I2 > 0.0) ? I2 * fac : I1k;

  double SR;
  if (linear) {
    // I_eff = sqrt(I1k * I2k)  (geometric mean of 2 non-zero moments)
    double I_eff = sqrt(I1k * I2k);
    SR = 1.0 + log(8.0*PI*PI * I_eff * KB * T / ((double)sigma_rot * H_SI*H_SI));
  } else {
    double I3k = (I3 > 0.0) ? I3 * fac : I2k;
    double lnI = 0.5 * log(I1k * I2k * I3k);
    SR = 1.5 + log(sqrt(PI) / (double)sigma_rot)
       + 1.5 * log(8.0*PI*PI * KB * T / (H_SI*H_SI))
       + lnI;
  }

  if (SR <= 0.0) return 0.0;
  int ndof_rot = linear ? 2 : 3;
  return SR / (double)ndof_rot;
}

/* ======================================================================
   hs_entropy_rot_lj — rigid-rotor gas entropy in LJ reduced units.
   I_star in m_lj*sigma² (LJ reduced moment of inertia).
   Convention: hbar_star = h_star/(2pi), same as translational.
   For linear: 1 + ln(I_star * T_star / (sigma_rot * hbar_star^2))
   For nonlinear: 1.5 + ... (using I_star^(3/2))
====================================================================== */

double FixXPT::hs_entropy_rot_lj(double y, double I_star,
                                  double natom_hs_rot, double T_star,
                                  int sigma_rot, bool linear, double hbar_star)
{
  (void)y; (void)natom_hs_rot;  // packing fraction not used in rigid-rotor formula
  if (T_star <= 0.0 || I_star <= 0.0) return 0.0;
  double h_star = 2.0 * PI * hbar_star;
  double SR;
  if (linear) {
    // S_rot / R = 1 + ln(8pi^2 * I* * T* / (sigma * h*^2))
    // In reduced units: h*=2pi*hbar*, so 8pi^2/h*^2 = 8pi^2/(4pi^2*hbar*^2) = 2/hbar*^2
    SR = 1.0 + log(2.0 * I_star * T_star / ((double)sigma_rot * hbar_star*hbar_star));
  } else {
    // S_rot / R = 1.5 + ln(sqrt(pi) * I*^(3/2) * T*^(3/2) / (sigma * (h*/2pi)^3))
    // = 1.5 + ln(sqrt(pi/sigma) * (I*T*)^(3/2) / hbar*^3)
    SR = 1.5 + log(sqrt(PI / (double)sigma_rot)
                   * pow(I_star * T_star, 1.5) / (hbar_star*hbar_star*hbar_star));
  }
  if (SR <= 0.0) return 0.0;
  int ndof_rot = linear ? 2 : 3;
  return SR / (double)ndof_rot;
}
