"""
Mathematical utility functions for the 2PT thermodynamics engine.

All functions are pure NumPy/SciPy; no MDAnalysis dependency.
"""

from __future__ import annotations
from scipy.special import dawsn

import logging
import math
import numpy as np

log = logging.getLogger(__name__)

# NumPy 2.0 renamed trapz → trapezoid; support both
try:
    _trapz = np.trapezoid
except AttributeError:
    _trapz = np.trapz

from scipy.integrate import cumulative_trapezoid as _cumtrapz
from scipy.linalg import eigh as _eigh

from pyxpt.constants import PI, KB, H, R, VLIGHT, NA
from pyxpt.core import fft as _onfft


# ── Numerical-linear-algebra helpers ─────────────────────────────────────────

def safe_matrix_inverse(A: np.ndarray, *,
                        cond_warn: float = 1e10,
                        rcond_pinv: float = 1e-10,
                        label: str | None = None,
                        ) -> tuple[np.ndarray, float, bool]:
    """Condition-number-guarded matrix inverse with pseudo-inverse fallback.

    Calls ``np.linalg.cond(A)`` first; if the result exceeds ``cond_warn``,
    returns ``np.linalg.pinv(A, rcond=rcond_pinv)`` with a WARNING log,
    otherwise returns ``np.linalg.inv(A)``.  A bare ``LinAlgError`` from the
    direct inverse also falls back to the pseudo-inverse path.

    The site this guards in pyxpt is:

      • ``invert_kernel_matrix``: ``inv(C_matrix[0])``  — VACF at t=0
        (the 3PT cage's Form-B Volterra inversion)

    Near-singular cases (condition number 10⁸–10¹² without exact zero
    eigenvalues) were the silent-failure regime that produced the SPC/Ew
    NaCl(aq) block-Volterra catastrophe documented in
    ``ms_followup_block_volterra_scalar``.  Direct ``np.linalg.inv`` only
    raises on exact singularity; this helper catches the ill-conditioned
    range and switches to pinv with surface-able diagnostics.

    Parameters
    ----------
    A : (m, m) np.ndarray
        Square matrix to invert.
    cond_warn : float, default 1e10
        Condition-number threshold for pinv fallback + WARNING log.
    rcond_pinv : float, default 1e-10
        Relative cutoff for small singular values in the pinv fallback.
    label : str, optional
        Diagnostic tag for the WARNING log (e.g., ``"C(0) block (3 species)"``).
        Pass ``None`` to suppress logging.

    Returns
    -------
    A_inv : (m, m) np.ndarray — the inverse or pseudo-inverse.
    cond  : float             — the condition number (np.inf if exactly singular).
    fallback : bool           — True if the pinv path was taken.
    """
    try:
        cond = float(np.linalg.cond(A))
    except (np.linalg.LinAlgError, ValueError):
        cond = float('inf')
    if not np.isfinite(cond) or cond > cond_warn:
        if label is not None:
            log.warning(
                "safe_matrix_inverse(%s): condition number %.3g exceeds "
                "%.0e — using pseudo-inverse with rcond=%.0e.  Result may "
                "lose information from the near-null subspace; downstream "
                "diffusion / kernel quantities should be treated as "
                "diagnostic-only.", label, cond, cond_warn, rcond_pinv)
        return np.linalg.pinv(A, rcond=rcond_pinv), cond, True
    try:
        return np.linalg.inv(A), cond, False
    except np.linalg.LinAlgError:
        if label is not None:
            log.warning(
                "safe_matrix_inverse(%s): direct inverse failed (LinAlgError); "
                "using pseudo-inverse with rcond=%.0e.", label, rcond_pinv)
        return np.linalg.pinv(A, rcond=rcond_pinv), cond, True


# ── Quantum weighting functions ───────────────────────────────────────────────

def polylog(n: float, z: float) -> float:
    """Polylogarithm Li_n(z) via series summation (converges for |z| < 1)."""
    total = prev = 0.0
    k = 1
    while True:
        prev = total
        total += z**k / k**n
        if total == prev:
            break
        k += 1
    return total


def sqweighting(u: float) -> float:
    """
    Quantum entropy weighting integral S_q(u)/u for the solid component.

    Numerically stable for all u ≥ 0; asymptotic form used above u=500.
    """
    if u == 0.0 or math.isnan(u):
        return 0.0
    pi = PI
    if u > 500.0:
        return pi**2 / (3.0 * u)
    # log(-1 + exp(u)) = u + log(1 - exp(-u))  — no overflow
    log_term = u + math.log(1.0 - math.exp(-u))
    wsq = pi**2 / 3.0 - u**2 + u * log_term - 2.0 * polylog(2.0, math.exp(-u))
    return wsq / u


def quantum_weights(u: float) -> tuple[float, float, float, float]:
    """
    Return (weq, wsq, waq, wcvq) quantum harmonic-oscillator weightings
    for a single dimensionless frequency u = hν/kT.

    All quantities are zero when u ≤ 0 or u is NaN.
    Numerically stable for all u ≥ 0; asymptotic form used above u=500.

    Debye solid chemical potential
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    For the solid component, the Gibbs chemical potential weighting is:

        wμq_solid = waq  (= u/2 + ln(1 − e^{−u}))

    Derivation: for a Debye/harmonic solid of N atoms with DoS g(ν),
    the partition function is Z = ∏_k e^{-u_k/2}/(1-e^{-u_k}), so:

        F/N = kT ∫ g(ν) · waq(ν) dν   (extensive in N)
        μ_A = (∂F/∂N)_{V,T} = F/N     (independent-mode solid)

    In the harmonic (Debye) approximation, mode frequencies are independent
    of volume (∂ω_k/∂V ≡ 0 by construction), so P = −(∂F/∂V) = 0, giving:

        G_solid = A_solid  →  μ_solid = A_solid/N  →  wμq_solid = waq

    Classical limit: Z_k^{cl} = kT/hν = 1/u  →  F^{cl}/N = kT ln(u)
        wμc_solid = ln(u)  (= wac_sol in the code)
    """
    if u <= 0.0 or math.isnan(u):
        return 0.0, 0.0, 0.0, 0.0
    if u > 500.0:
        # Deep quantum limit: exp(-u) → 0, exp(u) → ∞
        # weq  → u/2  (zero-point dominates)
        # wsq  → 0    (Planck factor → 0)
        # waq  → u/2  (log(1/exp(-u/2)) = u/2)
        # wcvq → 0    (exp(-u) → 0)
        return u / 2.0, 0.0, u / 2.0, 0.0
    exp_mu = math.exp(-u)
    exp_u  = math.exp(u)   # safe: u ≤ 500, exp(500) ≈ 5e217 < float max
    weq  = u / 2.0 + u / (exp_u - 1.0)
    wsq  = u / (exp_u - 1.0) - math.log(1.0 - exp_mu)
    waq  = math.log((1.0 - exp_mu) / math.exp(-u / 2.0))
    # wcvq = u² eᵘ / (eᵘ - 1)² = u² / (eᵘ/² - e⁻ᵘ/²)²
    sh = math.sinh(u / 2.0)
    wcvq = (u / (2.0 * sh))**2 if sh > 0 else 0.0
    return weq, wsq, waq, wcvq


# ── Beyond-Debye solid entropy correction ────────────────────────────────────

def d2wsq_du2(u: float) -> float:
    """
    Second derivative of wsq(u) = u/(eᵘ−1) − ln(1−e^{−u}).

    d²wsq/du² = eᵘ · n² · [u(1+2n) − 1]   where n = 1/(eᵘ−1)

    Limiting behavior:
      u → 0+:  d²wsq/du² → 1/u²  (diverges; soft-mode regime)
      u → ∞:   d²wsq/du² → 0     (exponential suppression)

    Always ≥ 0 for u > 0 (wsq is convex; Jensen correction is non-negative).
    """
    if u <= 0.0 or math.isnan(u):
        return 0.0
    if u > 500.0:
        return 0.0
    if u < 1e-8:
        return 1.0 / (u * u)
    exp_u = math.exp(u)
    n = 1.0 / (exp_u - 1.0)
    return exp_u * n * n * (u * (1.0 + 2.0 * n) - 1.0)


def wsq_lorentzian_avg(u0: float, gamma_u: float, n_pts: int = 64) -> float:
    """
    Lorentzian-weighted average of wsq for a damped quantum oscillator.

    Evaluates ∫₀^∞ wsq(u) · L(u; u0, γ/2) du / ∫₀^∞ L(u; u0, γ/2) du
    where L(u; u0, g) = (g/π) / ((u−u0)² + g²),  g = gamma_u/2.

    This is the quantum entropy of a mode broadened to a Lorentzian spectral
    function, as opposed to wsq(u0) for the sharp harmonic reference.

    Parameters
    ----------
    u0      : float — peak position ħω₀/kBT
    gamma_u : float — full linewidth ħΓ/kBT (FWHM in u units)
    n_pts   : int   — Gauss-Legendre quadrature points

    Returns
    -------
    wsq_avg : float ≥ wsq(u0)   (Jensen inequality: wsq is convex)
    """
    if u0 <= 0.0:
        return 0.0
    if gamma_u < 1e-12 * max(u0, 1.0):
        return quantum_weights(u0)[1]

    g = 0.5 * gamma_u
    u_lo = max(1e-10, u0 - 15.0 * g)
    u_hi = u0 + 15.0 * g

    pts, wts = np.polynomial.legendre.leggauss(n_pts)
    u_gl = 0.5 * (u_hi + u_lo) + 0.5 * (u_hi - u_lo) * pts
    du   = 0.5 * (u_hi - u_lo)

    wsq_v = np.array([quantum_weights(float(_u))[1] for _u in u_gl])
    L_v   = g / PI / ((u_gl - u0) ** 2 + g * g)

    norm = float(np.dot(L_v, wts)) * du
    if norm < 1e-30:
        return quantum_weights(u0)[1]
    return float(np.dot(wsq_v * L_v, wts)) * du / norm


# ── 2PT fluidicity solver ─────────────────────────────────────────────────────

def search_xpt(K: float) -> float:
    """
    Find fluidicity f satisfying the 2PT cubic (eqn 34 of Lin et al. 2010).

    Newton–Raphson iteration starting from f₀ = 0.7293·K^0.5727.
    The polynomial uses fractional powers of f (f**7.5, f**3.5, ...), so
    f must remain strictly positive at every iterate — otherwise Python's
    ``f ** 3.5`` returns a complex number for f < 0 and downstream
    comparisons fail (TypeError: complex vs float).  Newton can overshoot
    into f<0 or f>1 when |dP| is small (sparse / ill-conditioned K), so
    clamp to (eps, 1-eps) and bisect-back-off after each step.
    """
    K   = max(float(K), 1e-300)
    eps = 1e-12
    f   = min(max(0.7293 * K**0.5727, eps), 1.0 - eps)
    tol = 1e-10
    for _ in range(999):
        f_old = f
        P = (2.0*K**-4.5 * f**7.5 - 6.0*K**-3.0 * f**5.0
             - K**-1.5 * f**3.5 + 6.0*K**-1.5 * f**2.5 + 2.0*f - 2.0)
        dP = (15.0*K**-4.5 * f**6.5 - 30.0*K**-3.0 * f**4.0
              - 3.5*K**-1.5 * f**2.5 + 15.0*K**-1.5 * f**1.5 + 2.0)
        if dP == 0.0 or not math.isfinite(dP):
            break
        f_new = f_old - P / dP
        # If Newton overshoots out of (0, 1) or returns NaN/Inf, halve the
        # step toward the nearer bracket boundary and try again.  This
        # preserves quadratic convergence in the well-behaved regime while
        # turning a divergent step into a damped bisection.
        if not math.isfinite(f_new) or f_new <= 0.0 or f_new >= 1.0:
            f_new = (f_old + (eps if f_new <= 0.0 else (1.0 - eps))) * 0.5
        f = f_new
        if abs(f - f_old) <= tol:
            break
    return float(f)


# The 2PT fluidicity closure in d dimensions (derived + verified to reduce
# EXACTLY to search_xpt / the Carnahan-Starling chain at d=3):
#   (i)  γ = Δ^(-d/(d-1)) f^((2d-1)/(d-1))   [d-dim Chapman-Enskog diffusivity]
#   (ii) 1/f = g_d(γ)                         [Enskog contact value]
# with g_3 = Carnahan-Starling sphere, g_2 = Henderson disk, g_1 = Tonks rod.
# Santos, Yuste & Lopez de Haro, Entropy 22, 469 (2020), Table 1 (monocomponent
# d=4), cross-checked against Clisby-McCoy reduced virials (B3/B2^2=0.5063 ->
# b3=32.41 ; B4/B2^3=0.1518 -> b4=77.74).  The LM form reproduces b2,b3,b4 EXACTLY
# (verified by Taylor expansion, independent of zeta):
#   Z(eta) = 1 + b2 eta (1 + (b3/b2 - z b4/b3) eta)
#                  / (1 - z (b4/b3) eta + (z-1)(b4/b2) eta^2),  z = z0 + z1 eta/eta_cp
# Contact value from Z = 1 + 2^(d-1) eta g  ->  g4 = (Z-1)/(8 eta).
# Excess entropy s_ex = -int_0^eta (Z-1)/eta' deta' (LM has no closed integral;
# evaluated by quadrature, accurate to machine precision).  Validated:
# hs4d_check.py in the lj_4D campaign (Z>0, monotonic, contact>=1, no pole to 0.60).
_HS4D_B2, _HS4D_B3, _HS4D_B4 = 8.0, 32.406, 77.7452
_HS4D_Z0, _HS4D_Z1 = 1.2973, -0.062
_HS4D_ETACP = math.pi**2 / 16.0          # D4 close packing = 0.61685

def _hs4d_Z(eta: float) -> float:
    """d=4 hard-hypersphere compressibility factor (Luban-Michels)."""
    z = _HS4D_Z0 + _HS4D_Z1 * eta / _HS4D_ETACP
    c = _HS4D_B4 / _HS4D_B3                       # b4/b3
    e = _HS4D_B4 / _HS4D_B2                       # b4/b2
    num = 1.0 + (_HS4D_B3 / _HS4D_B2 - z * c) * eta
    den = 1.0 - z * c * eta + (z - 1.0) * e * eta * eta
    return 1.0 + _HS4D_B2 * eta * num / den


def hs_contact_dgen(gamma: float, d: int) -> float:
    """Hard-sphere contact value g_d(η⁺) in d dimensions."""
    if d == 3: return (1.0 - gamma/2.0) / (1.0 - gamma)**3        # Carnahan-Starling
    if d == 2: return (1.0 - 7.0*gamma/16.0) / (1.0 - gamma)**2   # Henderson disk
    if d == 4:                                                     # Luban-Michels (4D)
        if gamma <= 0.0: return 1.0
        return (_hs4d_Z(gamma) - 1.0) / (8.0 * gamma)             # Z=1+2^(d-1) eta g
    if d == 1: return 1.0 / (1.0 - gamma)                          # Tonks rod
    raise ValueError(f"dimension {d} not supported")

def hs_Z_dgen(eta: float, d: int) -> float:
    """Hard-sphere compressibility Z = PV/Nk_BT in d dimensions."""
    if d == 3: return (1.0 + eta + eta**2 - eta**3) / (1.0 - eta)**3   # CS
    if d == 2: return (1.0 + eta**2/8.0) / (1.0 - eta)**2              # Henderson
    if d == 4: return _hs4d_Z(eta)                                     # Luban-Michels (4D)
    if d == 1: return 1.0 / (1.0 - eta)                                # Tonks
    raise ValueError(f"dimension {d} not supported")

def hs_excess_entropy_dgen(eta: float, d: int) -> float:
    """Rigorous hard-sphere excess entropy S^ex/Nk_B = −A^ex/Nk_BT in d dims
    (closed forms d=1,2,3; d=4 by quadrature of the LM EOS)."""
    if eta <= 0.0: return 0.0
    if d == 3: return eta*(3.0*eta - 4.0) / (1.0 - eta)**2                          # CS
    if d == 2: return -(9.0/8.0)*eta/(1.0 - eta) + (7.0/8.0)*math.log(1.0 - eta)    # Henderson disk
    if d == 4:                                                                      # Luban-Michels (4D)
        from scipy.integrate import quad
        val, _ = quad(lambda e: (_hs4d_Z(e) - 1.0) / e, 0.0, eta, limit=200)
        return -val
    if d == 1: return math.log(1.0 - eta)                                           # Tonks rod
    raise ValueError(f"dimension {d} not supported")

def solve_xpt_dgen(K: float, d: int) -> float:
    """Fluidicity f for dimensionless diffusivity K=Δ in d dimensions.
    For d=3 this matches ``search_xpt(K)`` to machine precision.

    The closure 1/f=g_d(γ), γ=Δ^(-d/(d-1))f^((2d-1)/(d-1)) DEGENERATES (not
    diverges) at d=1: f^((2d-1)/d)=Δγ^((d-1)/d) → f=Δ (the γ exponent (d-1)/d→0),
    and 1/f=g_1(γ)=1/(1-γ) → γ=1-f.  So the 1D fluidicity is the normalized
    diffusivity directly.  The 2PT gas REFERENCE is the idealized passing 1D
    gas; a single-file channel (small Fickian D) → small Δ → small f → mostly
    solid, as physically expected (Tonks-rod HS excess via packing γ=1-f)."""
    if d == 3:
        return search_xpt(K)        # bit-identical legacy path
    if d == 1:
        return float(min(max(K, 1e-9), 1.0 - 1e-9))   # f = Δ (degenerate limit)
    from scipy.optimize import brentq
    a = d/(d-1.0); b = (2.0*d-1.0)/(d-1.0)
    fmax = K**(d/(2.0*d-1.0))        # f at which γ=1 (physical cap)
    hi = min(1.0 - 1e-9, fmax*(1.0 - 1e-9))
    if hi <= 1e-9:
        return 1e-9
    def resid(f):
        return 1.0/f - hs_contact_dgen(K**(-a)*f**b, d)
    try:
        return float(brentq(resid, 1e-9, hi, xtol=1e-13, rtol=1e-12, maxiter=200))
    except ValueError:
        return float(hi)

def packing_from_fluidicity_dgen(f: float, K: float, d: int) -> float:
    """Packing fraction γ from fluidicity f and dimensionless diffusivity K
    via the closure γ = K^(-d/(d-1)) f^((2d-1)/(d-1))  (valid for all d≥2;
    reproduces the 3D 2PT packing at d=3)."""
    return K**(-d/(d-1.0)) * f**((2.0*d-1.0)/(d-1.0))


def packing_from_f_dgen(f: float, d: int) -> float:
    """Packing fraction γ directly from fluidicity f by inverting the Enskog
    contact relation 1/f = g_d(γ)  (equivalent to the K-based closure; needs no
    diffusivity).  At d=3 reproduces the standard 2PT packing y."""
    from scipy.optimize import brentq
    f = float(max(1e-9, min(f, 1.0 - 1e-9)))
    gmax = {1: 0.999, 2: 0.9, 3: 0.74, 4: 0.60}[d]   # d=4: below D4 close-pack pi^2/16=0.617
    try:
        return float(brentq(lambda g: hs_contact_dgen(g, d) - 1.0/f,
                            1e-12, gmax, xtol=1e-13, rtol=1e-12, maxiter=200))
    except ValueError:
        return gmax


def enskog_K_parameter(s0: float, T_K: float,
                       mass_per_mol_g_mol: float,
                       nmol: int, vol_A3: float,
                       dimension: int = 3) -> float:
    """Compute the dimensionless 2PT K-parameter from kinetic inputs.

    Implements Lin-Maiti-Goddard 2003 eq.~14:

    .. math::

       K = \\frac{s_0}{c \\cdot 10^{-2} N}\\sqrt{\\frac{\\pi N_A k_B T}{m}}
           \\cdot \\frac{2}{9}\\left(\\frac{N}{V}\\right)^{1/3}\\cdot 10^{10}
           \\cdot (6/\\pi)^{2/3}

    Inputs:
    - ``s0``: DoS at zero frequency [cm] (the per-mol-normalised value).
    - ``T_K``: thermodynamic temperature [K].
    - ``mass_per_mol_g_mol``: molecular weight per molecule [g/mol]
      (e.g. ``grp.mass / grp.nmol``).
    - ``nmol``: number of molecules in the group.
    - ``vol_A3``: group volume [Å³].

    Returns the dimensionless K ready to feed into ``search_xpt(K)``
    for the fluidicity self-consistent solve.

    The internal `* 1e-3` converts mass from g/mol to kg/mol for the
    sqrt term; the `* 1e10` converts the volume factor from Å to m.

    ``dimension`` generalizes the 3D constants to d dimensions:
    ``2/9 → 2/(3d)``, ``(N/V)^(1/3) → (N/V)^(1/d)`` (V is the Å^d volume:
    Å³ bulk, Å² slit, Å channel), and ``(6/π)^(2/3) → v_d^(-(d-1)/d)`` with
    ``v_d`` the d-ball packing coefficient (π/6, π/4, 1 for d=3,2,1).  d=3 is
    bit-identical to the original.  Feed the result to ``solve_xpt_dgen(K, d)``.

    Source of truth for the K formula — used by the 2PT/3PT path
    (in `_xpt_compute.py`) for both translational and rotational channels.
    Single point of update if the dimensional convention ever changes.
    """
    v_d = {1: 1.0, 2: PI/4.0, 3: PI/6.0, 4: PI*PI/32.0}[dimension]   # diameter-1 d-ball volume
    return (s0 / VLIGHT * 1e-2) / nmol * math.sqrt(
        PI * NA * KB * T_K / (mass_per_mol_g_mol * 1e-3)
    ) * 2.0 / (3.0*dimension) * (nmol / vol_A3)**(1.0/dimension) * 1e10 \
        * v_d**(-(dimension - 1.0)/dimension)


def xpt_partition(s0: float, sv: float, nu: float, nmol: int,
                     f: float, fmf: float, dimension: int = 3) -> tuple[float, float]:
    """
    Partition DoS value *sv* at frequency *nu* into gas and solid fractions.

    The diffusive (gas) Lorentzian half-width carries the dimensionality through
    ``2·d·N·f`` (= ``6·N·f`` at d=3): S_gas(ν) = s0/(1+(π s0 ν/(2dNf))²), which
    integrates to d·N·f as required by the d-DoF sum rule.

    Returns (gas, solid).
    """
    if f == 0.0:
        return 0.0, sv
    if nu == 0.0:
        return s0, 0.0
    gas = (s0 / (1.0 + (PI * s0 * nu / (2.0 * dimension * nmol * f))**2)
           if f == fmf else _Sgmf(nu, s0, f, nmol, fmf))
    gas = min(gas, sv)
    return gas, sv - gas


def trapz_dos(pwr: np.ndarray, fmin: float) -> float:
    """Trapezoid integration of a DoS array with uniform spacing *fmin*."""
    if len(pwr) == 0:
        return 0.0
    return float(_trapz(pwr, dx=fmin))


def hsdf(pwr: np.ndarray, fmin: float, nmol: int,
         f: float, fmf: float, dimension: int = 3) -> float:
    """Hard-sphere degrees of freedom (integral of gas DoS).

    ``dimension`` must match the gas-Lorentzian half-width 2·d·N·f used in the
    DoS partition (it defaulted to 3 and silently narrowed the gas at d≠3,
    desyncing nmol_hs / packing y from the actual dos_gas)."""
    n = len(pwr)
    gas = np.array([xpt_partition(pwr[0], pwr[j], fmin*j, nmol, f, fmf,
                                  dimension=dimension)[0]
                    for j in range(n)])
    return float(_trapz(gas, dx=fmin))


# ── Mixture hard-sphere chemical potential (one-fluid CS) ─────────────────────

def mixture_packing_fraction(y_arr: np.ndarray) -> float:
    """
    Total packing fraction η_mix = Σ_i y_i for a hard-sphere mixture.

    y_arr : packing fractions of each species (one per group), already computed
            from the 2PT fluidicity solve for each group independently.

    η_mix is used in the one-fluid CS approximation for the mixture chemical
    potential:  Z_mix = (1+η+η²-η³)/(1-η)³  with η = η_mix.

    This approximation:
    - Reduces exactly to single-component CS when ngrp=1 (η_mix = y_i)
    - Captures the key mixture effect: a denser total packing → higher μ for all
      species, reflecting cross-species excluded volume
    - Is consistent with the CS framework already used for the entropy (wsehs)

    Note: The BMCSL (Boublik-Mansoori-Carnahan-Starling-Leland) equation gives a
    rigorous species-resolved mixture μ but reduces to the Boublik (1970) EOS for
    a pure fluid — a different (though close) approximation from CS.  Using the
    one-fluid CS with η_mix is therefore more consistent with the rest of 2PT.
    """
    return float(np.sum(y_arr))


# ── Boublik / BMCSL hard-sphere EOS ──────────────────────────────────────────

def _boublik_Z(eta: float) -> float:
    """
    Compressibility factor Z for a pure hard-sphere fluid (Boublik 1970).

    Z_B = (1 + 4η − 3η²) / (1 − η)³

    Reduces to Z_CS = (1+η+η²−η³)/(1−η)³ (Carnahan-Starling) only at low η;
    both are accurate fits to simulation data with slightly different weights
    for the 4th-order virial correction.
    """
    return (1.0 + 4.0*eta - 3.0*eta**2) / (1.0 - eta)**3


def _boublik_wsehs(eta: float) -> float:
    """
    Entropy weighting for the Boublik hard-sphere EOS.

    wsehs_B = ln(Z_B) − A^ex_B/NkT
            = ln(Z_B) − (6η − 5η²)/(1 − η)² + ln(1 − η)

    Derivation:
      For hard spheres, A^ex is T-independent (no energy scale), so S^ex = −A^ex/T:
        S^ex_B/NkB = −A^ex_B/NkT
      The total HS entropy correction (vs ideal gas) is:
        wsehs_B = ln(Z_B) + S^ex_B/NkB = ln(Z_B) − A^ex_B/NkT
      where A^ex_B/NkT = (6η − 5η²)/(1 − η)² − ln(1 − η)   [Boublik 1970]

    Analogue of the CS formula wsehs_CS = ln(Z_CS) − A^ex_CS/NkT
    where A^ex_CS/NkT = (4η − 3η²)/(1 − η)² [Carnahan-Starling].
    """
    Z_B = _boublik_Z(eta)
    A_ex_NkT = (6.0*eta - 5.0*eta**2) / (1.0 - eta)**2 - math.log(1.0 - eta)
    return math.log(Z_B) - A_ex_NkT



def _bmcsl_xi(sigmas: np.ndarray, rho_gas: np.ndarray) -> tuple[float, float, float, float]:
    """
    Compute BMCSL moments ξₙ = (π/6) Σᵢ ρᵢ σᵢⁿ for n = 0,1,2,3.

    Parameters
    ----------
    sigmas  : (ngrp,) hard-sphere diameters σᵢ (Å)
    rho_gas : (ngrp,) gas-phase number densities ρᵢ = N_i / V_total (Å⁻³)

    Returns
    -------
    ξ₀, ξ₁, ξ₂, ξ₃  (ξ₃ = η_mix = total packing fraction)

    Notes
    -----
    Retained for diagnostic output (σᵢ, ξ₃) and future species-resolved
    mixture μ implementations.
    """
    fac = PI / 6.0
    xi0 = fac * float(np.sum(rho_gas * sigmas**0))
    xi1 = fac * float(np.sum(rho_gas * sigmas**1))
    xi2 = fac * float(np.sum(rho_gas * sigmas**2))
    xi3 = fac * float(np.sum(rho_gas * sigmas**3))  # = η_mix
    return xi0, xi1, xi2, xi3


def bmcsl_sigma_from_y(y_i: float, rho_gas_i: float) -> float:
    """
    Hard-sphere diameter σᵢ (Å) from packing fraction yᵢ and gas density ρᵢ.

    yᵢ = (π/6) ρᵢ σᵢ³  →  σᵢ = (6yᵢ / (π ρᵢ))^{1/3}

    rho_gas_i : number density in Å⁻³ (= N_i^gas / V_total)
    """
    if rho_gas_i <= 0.0 or y_i <= 0.0:
        return 0.0
    return (6.0 * y_i / (PI * rho_gas_i)) ** (1.0 / 3.0)


# ── Hard-sphere entropy weightings ────────────────────────────────────────────

def hs_weighting(y: float, mass_per_mol: float, nmol: float,
                 T_trans: float, T_rot: float,
                 volume: float, rT: np.ndarray,
                 rotsym: float,
                 eta_mix: float | None = None,
                 hs_eos: str = "cs",
                 sigma_i: float = 0.0,
                 xi: tuple[float, float, float, float] | None = None,
                 hs_entropy: str = "rigorous",
                 z_override: float | None = None,
                 dimension: int = 3,
                 ) -> dict[str, float]:
    """
    Compute hard-sphere thermodynamic weighting factors.

    ``dimension``: 3 = bulk (default, unchanged).  For d=2/1 the
    translational gas weights use the d-dim Sackur-Tetrode ((d/2+1)+ln(λ⁻ᵈV_d/N),
    per-DoF /d) and the d-dim hard-sphere excess entropy / compressibility
    (Henderson disk for d=2, Tonks rod for d=1).  ``volume`` is then the Å^d
    measure (area for slits, length for channels).  bmcsl is 3D-only.

    Parameters
    ----------
    y            : packing fraction of this species (from 2PT fluidicity solve)
    mass_per_mol : molecular mass (g/mol)
    nmol         : number of molecules in the group
    T_trans      : translational temperature (K)
    T_rot        : rotational temperature (K)
    volume       : group volume (Å³) — used for de Broglie wavelength
    rT           : (3,) principal rotational temperatures (K); rT[2]<0 → linear
    rotsym       : rotational symmetry number
    eta_mix      : total mixture packing fraction η_mix = Σ_i y_i (from
                   mixture_packing_fraction()).  Used in both CS and BMCSL modes for
                   the one-fluid mixture chemical potential.  None for pure fluids.
    hs_eos       : "cs"    — Carnahan-Starling (default; Lin 2003)
                   "bmcsl" — Boublik (1970) EOS for both entropy and chemical potential
    sigma_i      : hard-sphere diameter σᵢ (Å) — passed through for diagnostic output;
                   not currently used in the wmp calculation.
    xi           : BMCSL moments (ξ₀,ξ₁,ξ₂,ξ₃) from _bmcsl_xi() — for diagnostics;
                   not currently used in wmp.
    hs_entropy   : "rigorous" — Thermodynamically exact: wsehs = S^ex/NkB = −A^ex/NkT
                                (no ln(Z) term; gives 3·wmp = βμ_standard exactly).
                                **Default**, and forced for all GLE modes (5, 6, 7).
                   "lin2003"  — Lin 2003 convention: wsehs includes extra ln(Z) term.
                                Use only to reproduce legacy Lin 2003 / Pascal 2011
                                benchmark numbers; not selectable with GLE modes.
    z_override   : float | None — when provided, replaces Z(η) in the chemical
                                potential weight wmp = wap + Z/3 with Z_sim =
                                PV/(NkT) from the user-supplied P, V, T, N.
                                Bypasses the HS EOS overestimate at η > 0.50;
                                does not affect entropy weights.

    Returns
    -------
    dict with keys: wep, wap, wmp, wcvp, wsehs, wsp, wspd, wsr, war, wmr, wer, wcvr

    Notes
    -----
    Lin 2003 convention (hs_entropy="lin2003"):
      CS:    wsehs = ln(Z_CS) + η(3η−4)/(1−η)²
      BMCSL: wsehs = ln(Z_B) − (6η−5η²)/(1−η)² + ln(1−η)

    Rigorous convention (hs_entropy="rigorous"):
      CS:    wsehs = η(3η−4)/(1−η)²                      [= S^ex_CS/NkB]
      BMCSL: wsehs = −(6η−5η²)/(1−η)² + ln(1−η)         [= −A^ex_B/NkT]
      In this convention 3·wmp = βμ_standard = ln(ρλ³) + βμ^ex exactly.

    Chemical potential (both conventions):
      wmp = wap + Z(η_eff)/3     where η_eff = η_mix or y; Z = Z_CS or Z_B.

    The rotational analogue wmr = war + 1/ndof_rot (ideal rotor PV = kT).
    """
    zero = dict(wep=0.0, wap=0.0, wmp=0.0, wcvp=0.0, wsehs=0.0,
                wsp=0.0, wspd=0.0, wsr=0.0, war=0.0, wmr=0.0, wer=0.0, wcvr=0.0)
    if hs_eos == "bmcsl" and dimension != 3:
        raise NotImplementedError("bmcsl EOS is 3D-only; use hs_eos=cs for d<3.")
    y_cap = {1: 0.99, 2: 0.90, 3: 0.74, 4: 0.60}[dimension]   # near-close-packing guard (d=4: <pi^2/16)
    if y > y_cap:
        return zero
    # nmol → 0 is the "no gas phase" limit (e.g. f_gle = 0 after GLE Volterra
    # collapse; the GLE re-partition then yields a zero gas DoS and
    # ``hsdf_gle = 0`` → ``nmol_hs = 0``).  The Sackur-Tetrode log(λ³V/N)
    # is undefined here, but the gas-side integrals in integrate_thermo
    # are pwr_g · wsp with pwr_g ≡ 0, so the contribution is physically
    # zero.  Returning the zero dict makes that explicit and avoids the
    # 0·inf → NaN cascade that nan_to_num then silently zeroes (which
    # also clobbers the solid-side baseline in result.S_quantum).
    if nmol <= 0.0:
        return zero

    m_kg   = mass_per_mol * 1e-3 / NA         # kg per molecule
    vol_m3 = volume * (1e-10)**dimension       # Å^d → m^d  (d=3: ×1e-30)
    lam3   = (2.0 * PI * m_kg * KB * T_trans / H**2)**(dimension/2.0)   # λ^{-d}

    wsp  = (dimension/2.0 + 1.0) + math.log(lam3 * vol_m3 / nmol)   # indistinguishable
    wspd = (dimension/2.0)       + math.log(lam3 * vol_m3)           # distinguishable

    # ── Entropy (excess hard-sphere correction wsehs) ─────────────────────────
    if hs_eos == "bmcsl":
        if hs_entropy == "rigorous":
            # Exact Boublik: wsehs = S^ex_B/NkB = −A^ex_B/NkT  (no ln(Z_B))
            # A^ex_B/NkT = (6η−5η²)/(1−η)² − ln(1−η)
            A_ex_NkT = (6.0*y - 5.0*y**2) / (1.0 - y)**2 - math.log(1.0 - y)
            wsehs = -A_ex_NkT
        else:  # lin2003
            wsehs = _boublik_wsehs(y)   # = ln(Z_B) − A^ex_B/NkT
    else:  # "cs" (default; d-dim HS excess: CS sphere / Henderson disk / Tonks rod)
        if hs_entropy == "rigorous":
            # Exact: wsehs = S^ex/NkB  (no ln(Z))
            wsehs = hs_excess_entropy_dgen(y, dimension)
        else:  # lin2003: wsehs = ln(Z_d) + S^ex/NkB
            wsehs = math.log(hs_Z_dgen(y, dimension)) + hs_excess_entropy_dgen(y, dimension)

    wsp  = (wsp  + wsehs) / dimension
    wspd = (wspd + wsehs) / dimension
    wep  = wcvp = 0.5
    wap  = wep - wsp

    # ── Chemical potential weighting wmp ──────────────────────────────────────
    if z_override is not None:
        # Z_sim from user-supplied P,V,T,N bypasses HS EOS overestimate (Sec V.D)
        wmp = wap + float(z_override) / dimension
    elif hs_eos == "bmcsl":
        # One-fluid Boublik: wmp = wap + Z_B(η_eff)/3
        #   pure fluid:  η_eff = y   → exact Boublik single-component
        #   mixture:     η_eff = η_mix = Σ_i y_i  (one-fluid approximation)
        # This is analogous to one-fluid CS used in the CS branch, but uses Z_B
        # (Boublik 1970) instead of Z_CS, keeping entropy and μ on the same EOS.
        eta = eta_mix if (eta_mix is not None and eta_mix < 0.74) else y
        Z_b = _boublik_Z(eta)
        wmp = wap + Z_b / 3.0
    else:
        # CS/Henderson/Tonks: Z_d(η_eff)/d
        eta = eta_mix if (eta_mix is not None and eta_mix < y_cap) else y
        wmp = wap + hs_Z_dgen(eta, dimension) / dimension

    # ── Rigid rotor ───────────────────────────────────────────────────────────
    wsr = war = wmr = wer = wcvr = 0.0
    if rT[0] > 1e-4:
        rs = rotsym
        if rT[2] < 0:   # linear
            wsr = (1.0 + math.log(T_rot / math.sqrt(rT[0]*rT[1]) / rs)) / 2.0
        else:
            wsr = (1.5 + math.log(math.sqrt(PI * T_rot**3 / (rT[0]*rT[1]*rT[2])) / rs)) / 3.0
        war = wep - wsr
        # Rotational chemical potential: μ_rot = A_rot + PV_rot = A_rot + kT (ideal rotor)
        # per rotational DOF: wmr = war + 1/(number of rot DOF)
        ndof_rot = 2.0 if rT[2] < 0 else 3.0
        wmr = war + 1.0 / ndof_rot
        wer = wep
        wcvr = 0.5

    return dict(wep=wep, wap=wap, wmp=wmp, wcvp=wcvp, wsehs=wsehs,
                wsp=wsp, wspd=wspd, wsr=wsr, war=war, wmr=wmr, wer=wer, wcvr=wcvr)


# ── Memory-function variant ───────────────────────────────────────────────────

def _cal_AB(s0: float, fxpt: float, fmf: float) -> tuple[float, float]:
    alpha = fxpt / s0
    a = 16.0 * (PI * (s0**2/fmf**2 - 1.0/alpha**2))**2
    b = -8.0 * PI * ((4.0+PI)*s0**2/fmf**2 + (4.0-PI)/alpha**2)
    c = (PI - 4.0)**2
    disc = b**2 - 4.0*a*c
    B = (-b + math.sqrt(max(0.0, disc))) / (2.0*a)
    A = math.sqrt(abs(4.0*B*fmf**2 / PI / s0**2))
    return A, B


def _Sgmf(nu: float, s0: float, fxpt: float, nmol: int, fmf: float) -> float:
    """Memory-function gas DoS (Desjarlais 2013)."""
    if fmf == fxpt:
        return s0 / (1.0 + (PI*s0*nu / (6.0*nmol*fxpt))**2)
    A, B = _cal_AB(s0, fxpt, fmf)
    vv = PI * nu / 6.0 / nmol
    y  = vv / math.sqrt(4.0 * B)
    c1, c2 = 214.0/155.0, 503.0/754.0
    D = 0.0 if y == 0.0 else (
        (1.0 - math.exp(-c1*y**2)) / (2.0*y)
        + (1.0 - (1.0 + y**2)*math.exp(-y**2)) / (4.0*y**3)
        + (0.875 - 0.5*c1)*y*math.exp(-c2*y**2)
    )
    denom = (A**2*PI/4.0/B*math.exp(-2*y**2)
             + (A*D)**2/B - 4.0*A*y*D + vv**2)
    return (A*math.sqrt(PI/4.0/B)*math.exp(-y**2))*fmf/denom if denom else 0.0

def _Sgmf_des(nu: float, s0: float, f_g: float, B: float, nmol: int) -> float:
    """Gas DOS for the Desjarlais (2013) memory-function variant."""
    if B <= 0 or f_g <= 0 or s0 <= 0:
        return 0.0
    
    A = f_g / s0 * math.sqrt(4.0 * B / PI)
    vv = PI * nu / 6.0 / nmol
    y  = vv / math.sqrt(4.0 * B)
    
    # Replace the legacy c1/c2 algebraic approximation with exact SciPy math
    D = dawsn(y)
    
    denom = (A**2 * PI / 4.0 / B * math.exp(-2.0 * y**2)
             + (A * D)**2 / B - 4.0 * A * y * D + vv**2)
             
    if denom <= 0:
        return 0.0
        
    return (A * math.sqrt(PI / 4.0 / B) * math.exp(-y**2)) * f_g / denom

def refine_Bg_desjarlais(
    pwr: np.ndarray,
    s0: float,
    nu: float,
    nmol: int,
    f_g: float,
    *,
    tol: float = 1e-8,
    maxiter: int = 80,
):
    """
    Determine the Desjarlais MF shape parameter Bg (dimensionless, Lin-2PT scaled)
    by matching the MF gas DoS tail to the total DoS tail.

    MF tail increases monotonically with Bg in the scaled formulation.
    """

    if s0 <= 0.0 or f_g <= 0.0 or pwr.size == 0:
        return 0.0

    # --- select a non-noisy tail frequency ---
    noise_floor = 1e-6 * s0
    j = pwr.size - 1
    while j > 0 and pwr[j] < noise_floor:
        j -= 1
    if j == 0:
        return 0.0

    nu_tail = j * nu
    target = pwr[j]

    # --- initial bracket ---
    Bg_lo = 1e-3
    Bg_hi = 1.0

    sg_lo = _Sgmf_des(nu_tail, s0, f_g, Bg_lo, nmol)
    sg_hi = _Sgmf_des(nu_tail, s0, f_g, Bg_hi, nmol)

    # Ensure the lower bound is actually below the target
    while sg_lo > target:
        Bg_lo /= 2.0
        if Bg_lo < 1e-12:  # Prevent infinite underflow
            break
        sg_lo = _Sgmf_des(nu_tail, s0, f_g, Bg_lo, nmol)

    # Ensure the upper bound is actually above the target
    while sg_hi < target:
        Bg_hi *= 2.0
        if Bg_hi > 1e8:
            break
        sg_hi = _Sgmf_des(nu_tail, s0, f_g, Bg_hi, nmol)

    # --- bisection ---
    for _ in range(maxiter):
        Bg_mid = 0.5 * (Bg_lo + Bg_hi)
        sg_mid = _Sgmf_des(nu_tail, s0, f_g, Bg_mid, nmol)

        if sg_mid < target:
            Bg_lo = Bg_mid
        else:
            Bg_hi = Bg_mid

        if abs(Bg_hi - Bg_lo) / (Bg_hi + Bg_lo) < tol:
            break

    return 0.5 * (Bg_lo + Bg_hi)

def xpt_partition_des(s0: float, sv: float, nu: float, nmol: int,
                          f_g: float, B_g: float) -> tuple[float, float]:
    """
    Partition DoS value *sv* at frequency *nu* using the Desjarlais memory-function gas DOS.

    Returns (gas, solid).
    """
    if f_g == 0.0:
        return 0.0, sv
    if nu == 0.0:
        return s0, 0.0
    gas = _Sgmf_des(nu, s0, f_g, B_g, nmol)
    gas = min(gas, sv)
    return gas, sv - gas

def hsdf_des(pwr: np.ndarray, fmin: float, nmol: int,
             f_g: float, B_g: float) -> float:
    """Hard-sphere degrees of freedom using Desjarlais gas DOS."""
    n = len(pwr)
    gas = np.array([xpt_partition_des(pwr[0], pwr[j], fmin*j, nmol, f_g, B_g)[0]
                    for j in range(n)])
    return float(_trapz(gas, dx=fmin))


# ── Inertia tensor / principal moments ───────────────────────────────────────

def _inertia_tensor(positions: np.ndarray, masses: np.ndarray) -> np.ndarray:
    """Build 3×3 inertia tensor from positions (n,3) and masses (n,)."""
    rx, ry, rz = positions[:, 0], positions[:, 1], positions[:, 2]
    I = np.empty((3, 3))
    I[0, 0] = np.dot(masses, ry**2 + rz**2)
    I[1, 1] = np.dot(masses, rx**2 + rz**2)
    I[2, 2] = np.dot(masses, rx**2 + ry**2)
    I[0, 1] = I[1, 0] = -np.dot(masses, rx * ry)
    I[0, 2] = I[2, 0] = -np.dot(masses, rx * rz)
    I[1, 2] = I[2, 1] = -np.dot(masses, ry * rz)
    return I


def principal_moments(positions: np.ndarray, masses: np.ndarray
                      ) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute principal moments of inertia and eigenvectors for a molecule.

    Parameters
    ----------
    positions : (n, 3) array of atomic positions relative to COM (Å)
    masses    : (n,) array of atomic masses (g/mol)

    Returns
    -------
    evals : (3,) principal moments in kg·m²   (ascending order)
    evecs : (3, 3) eigenvectors (columns)
    """
    I = _inertia_tensor(positions, masses)
    evals, evecs = _eigh(I)   # ascending, real symmetric
    # Convert (g/mol)·Å² → kg·m²
    conv = 1e-3 / NA * 1e-20
    return evals * conv, evecs


def decompose_velocities(positions: np.ndarray, velocities: np.ndarray,
                         masses: np.ndarray
                         ) -> tuple[np.ndarray, np.ndarray, np.ndarray,
                                    np.ndarray, np.ndarray, np.ndarray]:
    """
    Decompose atomic velocities into translational, rotational, and
    vibrational components for a single molecule.

    Parameters
    ----------
    positions  : (n, 3) Å  — atomic positions
    velocities : (n, 3) Å/ps
    masses     : (n,)  g/mol

    Returns
    -------
    v_trans     : (n, 3)  translational velocity (= COM velocity for each atom)
    v_rot       : (n, 3)  rotational velocity
    v_vib       : (n, 3)  vibrational = total − trans − rot
    anguv       : (3,)    mass-weighted angular velocity (lab frame), units
                          sqrt(g/mol)·Å/ps — the form used by 2PT for
                          unit-consistent rotational VAC.
    pI          : (3,)    principal moments of inertia (kg·m²)
    omega_lab   : (3,)    true angular velocity ω = I⁻¹·L in the lab frame,
                          units rad/ps — used by transport / NMR paths.
    """
    n = len(masses)
    total_mass = masses.sum()

    if n == 1:
        v_trans = velocities.copy()
        v_rot   = np.zeros((1, 3))
        v_vib   = np.zeros((1, 3))
        anguv   = np.zeros(3)
        pI      = np.full(3, -1.0)
        omega_lab = np.zeros(3)
        return v_trans, v_rot, v_vib, anguv, pI, omega_lab

    # COM position and velocity
    com_pos = (masses[:, None] * positions).sum(0) / total_mass
    com_vel = (masses[:, None] * velocities).sum(0) / total_mass

    rel_pos = positions - com_pos    # (n, 3)
    rel_vel = velocities - com_vel   # (n, 3)

    # Angular momentum L = sum m_i (r_i × v_i)
    L = np.sum(masses[:, None] * np.cross(rel_pos, rel_vel), axis=0)

    # Inertia tensor and principal moments
    pI, evecs = principal_moments(rel_pos, masses)

    # Inertia tensor (for omega solve; rel_pos already COM-centred)
    I = _inertia_tensor(rel_pos, masses)

    try:
        omega = np.linalg.solve(I, L)
    except np.linalg.LinAlgError:
        omega = np.zeros(3)

    v_trans = np.tile(com_vel, (n, 1))
    v_rot   = np.cross(omega, rel_pos)   # (n, 3)
    v_vib   = rel_vel - v_rot

    # Angular velocity weighted by sqrt(I), expressed in the LAB frame.
    # Matches legacy C++ which stores anguv_lab = evecs @ (omega_body * sqrt(I)):
    #   anguv[i] = sum_k pomega[k] * evecs[i][k] * sqrt(evl[k])
    # where pomega = principal-axis angular velocity (omega in body frame).
    # Using lab-frame anguv ensures its VACF captures the slow molecular
    # reorientation (matching legacy S(0) and diffusivity).
    #
    # pI from principal_moments is in kg·m²; omega is in rad/ps (MD units).
    # Convert pI to MD units (g/mol·Å²) so anguv has units sqrt(g/mol)·Å/ps,
    # consistent with the translational VAC units (g/mol)·(Å/ps)².
    # Conversion: (g/mol)·Å² → kg·m² uses conv = 1e-3/NA * 1e-20
    conv = 1e-3 / NA * 1e-20   # (g/mol)·Å² → kg·m²
    pI_md = pI / conv           # kg·m² → (g/mol)·Å²
    sqrt_pI = np.sqrt(np.maximum(pI_md, 0.0))
    omega_body = evecs.T @ omega         # omega in principal-axis (body) frame
    anguv      = evecs @ (omega_body * sqrt_pI)  # rotate back to lab frame

    return v_trans, v_rot, v_vib, anguv, pI, omega


def decompose_velocities_batch(
    pos_batch: np.ndarray,
    vel_batch: np.ndarray,
    masses_batch: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Vectorised molecular velocity decomposition for a batch of molecules.

    All molecules must have the same number of atoms (``na``).  Call
    separately for each molecule type when sizes differ.

    Parameters
    ----------
    pos_batch    : (nmol, na, 3) positions, Å
    vel_batch    : (nmol, na, 3) velocities, Å/ps
    masses_batch : (nmol, na)   masses, g/mol

    Returns
    -------
    v_trans   : (nmol, na, 3)
    v_rot     : (nmol, na, 3)
    v_vib     : (nmol, na, 3)
    anguv     : (nmol, 3)  mass-weighted angular velocity in lab frame
                            [sqrt(g/mol)·Å/ps] — used by 2PT VAC for
                            unit-consistent rotational kinetic energy.
    pI        : (nmol, 3)  principal moments of inertia, kg·m²
    omega_lab : (nmol, 3)  true angular velocity ω = I⁻¹·L in lab frame
                            [rad/ps] — used by transport / NMR paths.
    """
    nmol, na = masses_batch.shape[:2]

    # Single-atom molecules: all DOF is translational; no rotation or vibration.
    # The inertia tensor is identically zero, making the solve singular — skip it.
    if na == 1:
        v_trans = vel_batch.copy()                        # (nmol, 1, 3)
        v_rot   = np.zeros_like(vel_batch)
        v_vib   = np.zeros_like(vel_batch)
        anguv   = np.zeros((nmol, 3))
        pI      = np.full((nmol, 3), -1.0)
        omega_lab = np.zeros((nmol, 3))
        return v_trans, v_rot, v_vib, anguv, pI, omega_lab

    m = masses_batch                           # (nmol, na)
    total_mass = m.sum(axis=1, keepdims=True)  # (nmol, 1)

    # Centre-of-mass position and velocity
    com_pos = (m[:, :, None] * pos_batch).sum(1) / total_mass  # (nmol, 3)
    com_vel = (m[:, :, None] * vel_batch).sum(1) / total_mass  # (nmol, 3)

    rel_pos = pos_batch - com_pos[:, None, :]   # (nmol, na, 3)
    rel_vel = vel_batch - com_vel[:, None, :]   # (nmol, na, 3)

    # Angular momentum L = Σ mᵢ (rᵢ × vᵢ)
    L = (m[:, :, None] * np.cross(rel_pos, rel_vel)).sum(1)   # (nmol, 3)

    # Batched inertia tensor (nmol, 3, 3)
    rx = rel_pos[:, :, 0]; ry = rel_pos[:, :, 1]; rz = rel_pos[:, :, 2]
    I = np.empty((len(m), 3, 3))
    I[:, 0, 0] = (m * (ry**2 + rz**2)).sum(1)
    I[:, 1, 1] = (m * (rx**2 + rz**2)).sum(1)
    I[:, 2, 2] = (m * (rx**2 + ry**2)).sum(1)
    I[:, 0, 1] = I[:, 1, 0] = -(m * rx * ry).sum(1)
    I[:, 0, 2] = I[:, 2, 0] = -(m * rx * rz).sum(1)
    I[:, 1, 2] = I[:, 2, 1] = -(m * ry * rz).sum(1)

    # omega = I⁻¹ L  (batched, via pseudo-inverse to handle rank-deficient I).
    # Linear/diatomic molecules have a zero eigenvalue along the bond axis;
    # pinv returns ω=0 for that axis (minimum-norm solution), which is correct.
    omega = (np.linalg.pinv(I) @ L[:, :, None]).squeeze(-1)   # (nmol, 3)

    # Velocity components
    v_rot   = np.cross(omega[:, None, :], rel_pos)   # (nmol, na, 3)
    v_vib   = rel_vel - v_rot
    v_trans = np.broadcast_to(com_vel[:, None, :], pos_batch.shape).copy()

    # Principal moments and anguv
    pI_eval, pI_evec = np.linalg.eigh(I)      # (nmol, 3), (nmol, 3, 3)
    # pI_eval is in (g/mol)·Å²; convert to kg·m² for return value
    conv  = 1e-3 / NA * 1e-20
    pI_kg = pI_eval * conv

    sqrt_pI    = np.sqrt(np.maximum(pI_eval, 0.0))         # (nmol, 3)
    # omega in body (principal-axis) frame: evec.T @ omega per molecule
    omega_body = np.einsum('mji,mj->mi', pI_evec, omega)   # (nmol, 3)
    anguv      = np.einsum('mij,mj->mi', pI_evec, omega_body * sqrt_pI)  # (nmol, 3)

    return v_trans, v_rot, v_vib, anguv, pI_kg, omega


# ── Memory-kernel utilities ──────────────────────────────────────────────────
#
# The normalized velocity autocorrelation obeys the Volterra equation
#   dĈ/dt = -∫₀ᵗ K(t-s) Ĉ(s) ds
# from which the memory friction kernel K(t) is inverted numerically (used by
# the cage construction).  Units: time in ps, K(t) in 1/ps², and ∫K dt in 1/ps
# (the reduced friction γ = ∫₀^∞K dt).


def detect_plateau(S: np.ndarray, time: np.ndarray,
                   frac: float = 0.1, tol: float = 0.02
                   ) -> tuple[float, float, bool]:
    """
    Detect the first plateau in a running-integral array S(t).

    A plateau is declared at the first interior point i where the local
    slope over a window of width ``frac · n`` falls below ``tol · |S[i]|``.
    Returns ``(S_plateau, t_plateau, detected)``:

    - **Detected case** (``detected=True``): ``S_plateau`` is the value at
      the first plateau bin; ``t_plateau`` is that bin's time.
    - **Fallback case** (``detected=False``): the array is too short or
      too noisy for a plateau to be found; ``S_plateau = S[-1]`` and
      ``t_plateau = time[-1]`` (the trailing value).  Callers should log
      a warning and decide whether to use the fallback value or skip the
      entropy contribution entirely.

    The ``detected`` flag lets callers distinguish a real plateau from a
    trailing-noise fallback, rather than silently substituting ``S[-1]``.

    Notes
    -----
    For backward compatibility, callers that ignored the second tuple
    element (`S, _ = detect_plateau(...)`) will now also discard the
    bool — that pattern is harmless because the fallback value remains
    `S[-1]`.  Callers that unpack three values can act on
    ``detected``.
    """
    n = len(S)
    w = max(int(frac * n), 1)
    for i in range(w, n - w):
        slope = abs((S[i + w] - S[i - w]) / (time[i + w] - time[i - w]))
        ref   = abs(S[i]) if abs(S[i]) > 1e-30 else 1.0
        if slope < tol * ref:
            return float(S[i]), float(time[i]), True
    return float(S[-1]), float(time[-1]), False


def gk_viscosity(stress_Pa: np.ndarray, dt_ps: float,
                 V_m3: float, T: float) -> float:
    """
    Standard Green-Kubo viscosity from off-diagonal stress time series.

    η = (V / kB T) × ∫ <P_αβ(0) P_αβ(t)> dt

    Parameters
    ----------
    stress_Pa : (nframes, 3) Pa — the 3 off-diagonal stress components
                (e.g. xy, xz, yz averaged for better statistics)
    dt_ps     : float, ps — frame spacing
    V_m3      : float, m³ — system volume
    T         : float, K  — temperature

    Returns
    -------
    η : float, Pa·s
    """
    nf = stress_Pa.shape[0]
    if nf < 2:
        return 0.0

    dt_s  = dt_ps * 1e-12
    acf   = np.zeros(nf)
    for d in range(stress_Pa.shape[1]):
        x = stress_Pa[:, d] - stress_Pa[:, d].mean()
        N2 = 2 * nf
        F  = _onfft.rfft(x, n=N2)
        acf += _onfft.irfft(np.abs(F)**2)[:nf].real / nf
    acf /= stress_Pa.shape[1]

    return V_m3 / (KB * T) * float(_trapz(acf, dx=dt_s))


def invert_kernel_matrix(dt: float, C_matrix: np.ndarray,
                          *,
                          noise_floor: float = 1e-3,
                          divergence_factor: float = 1e6,
                          swing_threshold: float = 10.0,
                          swing_window: int = 5,
                          nf_run: int = 1,
                          label: str | None = None,
                          ) -> np.ndarray:
    """Discrete Volterra inversion of the matrix memory kernel
    via the second-derivative (Form B) GLE.

    Continuous form (differentiated GLE):

        C̈(t) = -K(t)·C(0) - ∫₀ᵗ K(t-s)·Ċ(s) ds

    Normalising by C(0) so that Ĉ(t) = C(t)·C(0)⁻¹ has Ĉ(0) = I:

        C̈̂(t) = -K(t) - ∫₀ᵗ K(t-s)·Ċ̂(s) ds

    For an equilibrium VACF (Ĉ(t) even, Ċ̂(0) = 0), even-extension at the
    boundary gives the correct K(0):

        K(0) = -C̈̂(0) = 2·(Ĉ(0) - Ĉ(dt)) / dt²

    Note the factor of 2 — it comes from C̈̂(0) = (Ĉ(-dt) - 2Ĉ(0) + Ĉ(dt))/dt²
    = 2(Ĉ(dt) - Ĉ(0))/dt² under Ĉ(-dt) = Ĉ(dt).

    For n ≥ 1, central differences:
        C̈̂[n] = (Ĉ[n-1] - 2Ĉ[n] + Ĉ[n+1]) / dt²
        Ċ̂[j] = (Ĉ[j+1] - Ĉ[j-1]) / (2·dt)

    and the recursion (with dt·Ċ̂[j] = (Ĉ[j+1] - Ĉ[j-1])/2 absorbing the dt):

        K[n] = -C̈̂[n] - Σ_{j=1..n} K[n-j] · (Ĉ[j+1] - Ĉ[j-1]) / 2

    Compared to the Form A recursion (first-derivative GLE) that uses
    ``M[n] = (Ĉ[n-1]-Ĉ[n+1])/(2dt²) - Σ M[n-j]·Ĉ[j]`` this scheme is
    mathematically consistent at the boundary (K[0] from the second-derivative
    formula matches the recursion's view of K[0]).  Empirical accuracy on a
    smooth synthetic kernel K(t) = exp(-t/τ_K): RMS error 7×10⁻⁴ over the
    first 200 lags vs 5×10⁻² for Form A (70× improvement); the P1 entropy
    plateau is at machine precision (2×10⁻¹⁶) on the analytically-Markovian
    pure-exponential reference, vs 5×10⁻⁵ for Form A.

    Caveat: the central second-difference at t=0 assumes Ĉ(t) has a smooth
    even Taylor expansion at t=0.  This holds for all classical-mechanical
    velocity ACFs (Ċ(0) = 0 by symmetry; finite C̈(0)).  It does *not* hold
    for purely stochastic abstractions like Ornstein-Uhlenbeck C(t)=exp(-γ|t|),
    where Ĉ has a kink — but those aren't real MD data.

    Parameters
    ----------
    C_matrix : (N, m, m) np.ndarray
        Velocity autocorrelation matrix; m = 3 for a single group (spatial
        tensor) or m = ngrp·3 for a mixture block matrix.
    dt : float
        Time step [ps].
    noise_floor : float, default 1e-3
        Stop the recursion when ``‖Ĉ[i]‖_F / m < noise_floor`` (per-dimension
        Frobenius noise floor).  Increase for supercooled / glassy systems
        where C(t) plateaus and never decays into noise.
    divergence_factor : float, default 1e6
        Stop when ``max|K[i]| > divergence_factor · max|K[0]|`` — catches
        catastrophic blow-up.  Per-element max rather than trace so a wildly
        oscillating off-diagonal triggers the clamp even when the trace is
        stable.
    swing_threshold : float, default 10.0
        Adaptive instability check: when ``swing_window`` consecutive lags
        all satisfy ``‖K[i]‖_F / ‖K[i-1]‖_F > swing_threshold`` or its
        reciprocal, the recursion is declared noise-dominated and truncated.
        Catches gradual noise accumulation that doesn't trip
        ``divergence_factor``.  Set to ``inf`` to disable.
    swing_window : int, default 5
        Number of consecutive swings required before truncating.
    nf_run : int, default 1
        cage_nf_run: number of CONSECUTIVE
        sub-noise-floor lags required before the noise-floor guard truncates.
        1 = legacy first-lag behavior.  Oscillatory (librational) VACFs pass
        through zero every node, so a single sub-floor sample is not decay —
        nf_run >= 3 makes the guard envelope-aware.  Truncation lands at the
        START of the streak (tentative kernel entries inside the streak are
        re-zeroed).
    label : str, optional
        Diagnostic tag for the truncation log line.  When ``None`` no log is
        emitted; truncation is silent for backward compatibility.

    Returns
    -------
    M : (N, m, m) np.ndarray
        Memory kernel in units [1/ps²].  Zero past the truncation lag.

    Notes
    -----
    The per-lag noise/divergence guards (validated on the LJ-Ar 28-state grid)
    generalise from scalar |K[i]| to per-element max|K[i,j,k]| and Frobenius
    norm for the swing-streak check.  This closes
    the silent-failure regime documented in
    ``ms_followup_block_volterra_scalar``, where the SPC/Ew NaCl(aq) block
    kernel was producing D_gle/D_xpt ~ 10⁵× without surfacing any warning.
    The ``C(0)`` inverse is now condition-number-guarded via
    ``safe_matrix_inverse`` (pinv fallback at cond > 1e10 with WARNING log).
    """
    n_lags = C_matrix.shape[0]
    m      = C_matrix.shape[1]
    M = np.zeros((n_lags, m, m))

    # 1. Normalize the correlation matrix by C(0) inverse.  Condition-number-
    # guarded: near-singular C(0) (cond > 1e10) falls back to pinv with a
    # WARNING; bare LinAlgError on exactly-singular C(0) also falls back.
    C0_inv, _C0_cond, _C0_pinv = safe_matrix_inverse(
        C_matrix[0],
        label=(f"C(0) for invert_kernel_matrix({label})" if label else None))
    if _C0_pinv and label is None:
        # Even without an explicit label, surface this as INFO so the
        # downstream kernel quality is interpretable.
        log.info("invert_kernel_matrix: C(0) ill-conditioned (cond=%.3g); "
                 "kernel built from pseudo-inverse normalisation.", _C0_cond)

    C_hat = np.einsum('nij,jk->nik', C_matrix, C0_inv)

    # 2. Boundary K(0) from even-extension second difference (Form B).
    # Ĉ(-dt) = Ĉ(dt) ⇒ C̈̂(0) = 2(Ĉ(dt) - Ĉ(0))/dt² ⇒ K(0) = 2(Ĉ(0)-Ĉ(dt))/dt².
    M[0] = 2.0 * (C_hat[0] - C_hat[1]) / (dt**2)

    # Stability reference: per-element max-magnitude of M[0] used to detect
    # divergence at any matrix element (not just the trace, which can mask
    # a wildly oscillating off-diagonal — extension of the scalar-path
    # divergence clamp to the matrix case).
    M0_max = float(np.max(np.abs(M[0])))
    if M0_max <= 0.0:
        M0_max = 1.0

    # 3. Form B recursion (consistent with the boundary K[0]):
    #   K[n] = -C̈̂[n] - Σ_{j=1..n} K[n-j] · Ċ̂[j] · dt
    #        = -C̈̂[n] - Σ_{j=1..n} K[n-j] · (Ĉ[j+1] - Ĉ[j-1]) / 2
    #
    # Four termination guards:
    #   (1) Frobenius noise floor on Ĉ[n]
    #   (2) Conv-finite check on the convolution accumulator
    #   (3) Per-element divergence clamp (max|K[n,i,j]| > divergence_factor · M0_max)
    #   (4) Adaptive swing-streak on ‖K[n]‖_F (consecutive lag-to-lag ratio out of band)
    _swing_streak = 0
    _nf_streak    = 0
    _M_prev_norm  = float(np.linalg.norm(M[0]))
    _truncation = None    # (idx, reason, c_norm, k_max) for INFO log

    for n in range(1, n_lags - 1):
        # (1) Frobenius noise floor on Ĉ[n] (per-element normalised).
        # — require nf_run consecutive sub-floor lags and truncate at the
        # START of the streak (tentative M entries inside it are re-zeroed),
        # so an oscillatory VACF node cannot truncate the kernel.  nf_run = 1
        # reproduces the legacy first-lag break exactly.
        c_norm = float(np.linalg.norm(C_hat[n]))
        if c_norm / m < noise_floor:
            _nf_streak += 1
            if _nf_streak >= nf_run:
                n0 = n - _nf_streak + 1
                M[n0:n] = 0.0
                _truncation = (n0, "noise_floor", c_norm, 0.0)
                break
        else:
            _nf_streak = 0

        # C̈̂[n] via central second-difference (n-1, n+1 both in range for n ≥ 1)
        Cdd_n = (C_hat[n-1] - 2.0 * C_hat[n] + C_hat[n+1]) / (dt ** 2)

        # Convolution sum with central-difference Ċ̂[j], j = 1..n.
        conv = np.zeros((m, m))
        for j in range(1, n + 1):
            Cdot_j = (C_hat[j+1] - C_hat[j-1]) / 2.0
            conv += M[n-j] @ Cdot_j

        # (2) Conv-finite check: catch overflow / NaN before it propagates.
        if not np.isfinite(conv).all():
            _truncation = (n, "conv_nonfinite", c_norm, 0.0)
            break

        with np.errstate(over='ignore', invalid='ignore'):
            M_n = -Cdd_n - conv

        if not np.isfinite(M_n).all():
            _truncation = (n, "k_nonfinite", c_norm, 0.0)
            break

        # (3) Per-element divergence clamp.  Extension of the scalar-path
        # absolute clamp to the matrix case — uses max(|K[n,i,j]|) rather
        # than Tr[K[n]] so a wildly oscillating off-diagonal triggers the
        # clamp even when the trace is stable.
        k_max = float(np.max(np.abs(M_n)))
        if k_max > divergence_factor * M0_max:
            _truncation = (n, "divergence_clamp", c_norm, k_max)
            break

        # (4) Adaptive swing-streak on Frobenius norm.
        M_n_norm = float(np.linalg.norm(M_n))
        if n > 1 and _M_prev_norm > 1e-30:
            ratio = M_n_norm / _M_prev_norm
            if ratio > swing_threshold or ratio < 1.0 / swing_threshold:
                _swing_streak += 1
                if _swing_streak >= swing_window:
                    _truncation = (n, "swing_instability", c_norm, k_max)
                    break
            else:
                _swing_streak = 0

        M[n] = M_n
        _M_prev_norm = M_n_norm

    if _truncation is not None and label is not None:
        idx, reason, c_val, k_val = _truncation
        log.info("invert_kernel_matrix(%s) truncated at lag %d (t=%.4g ps): "
                 "%s  ‖Ĉ‖=%.3e  max|K|=%.3e  (kernel zero past this lag)",
                 label, idx, idx * dt, reason, c_val, k_val)
    return M


def _pade_kernel_fit(dt: float, C_scalar: np.ndarray,
                     M: int, N: int,
                     n_freq: int | None, omega_max: float | None,
                     stabilize: bool, label: str | None):
    """Padé rational fit of K̃(s) ≈ P_M(s)/Q_N(s) from a scalar VACF C(t).

    Returns the partial-fraction representation
    ``(poles, residues, gamma_inst)`` such that
    ``K̃(s) = gamma_inst + Σ_k residues_k / (s − poles_k)``.

    Shared by ``invert_kernel_pade`` (which inverse-Laplace transforms the
    poles back to K(t)) and ``kvol_gas_dos_split`` (which classifies the
    poles into diffusive/resonant and builds the gas DoS from the diffusive
    subset).  Returns ``(None, None, None)`` when the fit is ill-posed
    (degenerate VACF, LSQ failure, or residue decomposition failure).
    """
    import scipy.signal as _ssig
    N_t = len(C_scalar)
    if N_t < 4 or M < 0 or N < 1:
        return None, None, None
    if n_freq is None:
        n_freq = max(4 * (M + N + 1), 32)
    if omega_max is None:
        # Adaptive default: fit only the band where K̃ is well-conditioned.
        # Form-B boundary K(0) ≈ γ_inst/dt + K_mem(0+); discrete K(0)·dt
        # gives the dominant timescale γ_inst, and the kernel's natural
        # bandwidth is ~γ_inst.  Fitting out to ~3·γ_inst captures the
        # full smooth shape without picking up the high-ω C̃ aliasing.
        # Empirically (exp-cos synthetic γ=2, ω₀=0.5): the optimum is
        # omega_max ≈ 2-3·K(0)·dt, with the fit degrading rapidly above
        # that as C̃ aliasing contaminates K̃.  Floor at 5 ps⁻¹ to
        # ensure adequate fit-frequency coverage for slow kernels.
        # Form-B K(0) = 2(Ĉ(0)-Ĉ(dt))/dt² ≈ 2γ_inst/dt + (ω₀² - γ_inst²)
        # for a damped-oscillator C(t).  Discrete K(0)·dt ≈ 2γ_inst (the
        # factor 2 is the even-extension convention), so γ_inst ≈
        # K(0)·dt/2.  Empirically the fit sweet spot is omega_max ≈
        # (2-3)·γ_inst — too low loses Lorentzian shape, too high lets
        # C̃ aliasing leak into K̃.  Use 2.5·γ_inst with a 5 ps⁻¹ floor.
        K0_estimate = abs(2.0 * (C_scalar[0] - C_scalar[1]) / (C_scalar[0] * dt**2))
        gamma_est   = K0_estimate * dt / 2.0
        omega_max   = max(5.0, 2.5 * gamma_est)
        # Hard cap at half the discrete Nyquist to never cross into the
        # alias-dominated region regardless of K(0) estimate.
        omega_max   = min(omega_max, 0.5 * np.pi / dt)

    # 1. K̃(iω_k) at the fit frequencies via direct quadrature on C(t).
    omega_grid = np.linspace(0.0, omega_max, n_freq)
    K_tilde = kvol_K_from_C(dt, np.asarray(C_scalar, dtype=float), omega_grid,
                              label=(f"pade-fit({label})" if label else None))
    if not np.isfinite(K_tilde).all():
        K_tilde = np.nan_to_num(K_tilde, nan=0.0, posinf=0.0, neginf=0.0)

    # 2. Build the least-squares system at the fit frequencies.
    #    Row k for each ω_k (complex):
    #      Σ_{l=0..M} a_l · (iω_k)^l  −  K̃_k · Σ_{j=1..N} b_j · (iω_k)^j  =  K̃_k
    s = 1j * omega_grid                      # (n_freq,)
    # Vandermonde-like blocks for s^l and s^j
    A_num = np.column_stack([s**l for l in range(M + 1)])           # (n_freq, M+1)
    A_den = np.column_stack([-K_tilde * s**j for j in range(1, N + 1)])
    A_complex = np.hstack([A_num, A_den])    # (n_freq, M+N+1)
    b_complex = K_tilde                       # (n_freq,)
    # Split into real and imag → 2 n_freq real equations.
    A_real = np.vstack([A_complex.real, A_complex.imag])
    b_real = np.concatenate([b_complex.real, b_complex.imag])
    try:
        sol, *_ = np.linalg.lstsq(A_real, b_real, rcond=None)
    except np.linalg.LinAlgError:
        if label is not None:
            log.warning("_pade_kernel_fit(%s): LSQ failed.", label)
        return None, None, None
    a_coef = sol[:M + 1]                      # P_M coefficients: a_0 .. a_M
    b_coef = np.concatenate(([1.0], sol[M + 1:]))  # Q_N: b_0=1, b_1..b_N

    # 3. Find poles, optionally reflect RHP → LHP for stability.
    # scipy.signal.residue uses descending-power coefficient convention.
    P_desc = a_coef[::-1]
    Q_desc = b_coef[::-1]
    try:
        residues, poles, direct = _ssig.residue(P_desc, Q_desc)
    except Exception as e:
        if label is not None:
            log.warning("_pade_kernel_fit(%s): residue decomposition "
                        "failed (%s).", label, e)
        return None, None, None
    if stabilize:
        # Reflect any RHP poles (Re(p) > 0) into the LHP.  Update
        # residues to keep the rational function consistent.
        rhp_mask = poles.real > 0.0
        if rhp_mask.any():
            if label is not None:
                log.info("_pade_kernel_fit(%s): reflecting %d RHP poles "
                         "into LHP for stability.", label, int(rhp_mask.sum()))
            poles = np.where(rhp_mask, -poles.real + 1j*poles.imag, poles)
            # Note: this changes the residues; we leave them as fit because
            # the partial-fraction representation with reflected poles is
            # an approximation.  For a robust implementation, refit P given
            # the reflected Q — deferred.

    # γ_inst from the direct polynomial.  When M == N, direct is a single
    # scalar (the constant a_M/b_N).  When M < N, direct is empty (γ_inst = 0).
    gamma_inst = float(np.real(direct[0])) if len(direct) > 0 else 0.0
    return poles, residues, gamma_inst


def kvol_gas_dos_split(dt: float, C_scalar: np.ndarray,
                       omega: np.ndarray, dos_total: np.ndarray,
                       ref_gas_area: float, dnu: float,
                       *,
                       M: int = 3, N: int = 4,
                       omega_max: float | None = None,
                       label: str | None = None) -> np.ndarray | None:
    """Kernel-derived GAS DoS via Padé pole classification + 2-anchor norm.

    Builds the diffusive (gas) density of states from the *overdamped* poles
    of the memory kernel K̃(s) only, then normalizes with two anchors so the
    result is a valid 2PT gas component:

      * pin:    ``gas[0] == dos_total[0]``       → ``solid(0) = total(0) − gas(0) = 0``
      * width:  ``∫ gas dν == ref_gas_area`` (3fN) → 2PT fluidicity / DoF preserved

    A single scalar normalization cannot satisfy both anchors because the
    kernel friction K̃(0) is generally far broader than the 2PT fluidicity
    Lorentzian (on dense LJ-Ar, K̃(0) ≈ 9 ps⁻¹ → kernel f ≈ 0.8 vs 2PT
    f ≈ 0.33).  The second anchor is met by frequency-rescaling the gas
    kernel, K̃_gas(iω/β), and solving for the scale β that matches the area.

    Pole classification: a pole ``p = −a + ib`` is *resonant* (cage-rattle /
    backscatter — belongs to the solid) when ``|b| > |a|`` (underdamped) and
    *diffusive* (broadband friction — belongs to the gas) otherwise.  Only
    the diffusive poles enter K̃_gas, so the resonance structure stays in the
    solid residual where the harmonic-oscillator weight applies.

    This replaces the earlier ``F_g(ν) = Re[1/(K̃(iω)+iω)]`` form built from
    the *full* Volterra kernel, which — by the GLE identity
    ``C̃(iω) = 1/(iω + K̃(iω))`` — is identically the *total* DoS shape, not a
    gas component, and therefore produced a finite (unphysical) solid DoS at
    ν=0 (fixed 2026-06-02).

    Parameters
    ----------
    dt           : float — VACF time step [ps].
    C_scalar     : (N_t,) — scalar (trace/3) VACF, un-normalised.
    omega        : (n,)   — angular-frequency grid [1/ps] of the DoS bins.
    dos_total    : (n,)   — total DoS on the same grid (``dos_total[0]`` is the
                            ν=0 value the gas is pinned to).
    ref_gas_area : float  — target gas DoF, ``∫ gas dν`` (= 3fN; pass the
                            Lin-Goddard Lorentzian gas-DoS integral).
    dnu          : float  — DoS frequency-bin spacing [cm⁻¹] for the area
                            quadrature.
    M, N         : int    — Padé numerator/denominator degree (default 3/4).
                            ``M < N`` forces the direct (γ_inst δ-) term to
                            zero, which removes a broad instantaneous-friction
                            floor from the gas DoS and makes the β area-match
                            reachable across fit bands (M==N can leave the gas
                            floor above 3fN so β cannot shrink it — falls back
                            to the Lorentzian then).
    omega_max    : float  — optional Padé fit-band cap [1/ps].
    label        : str    — optional diagnostic tag.

    Returns
    -------
    gas : (n,) float — the gas DoS, clipped to ``[0, dos_total]``; or ``None``
          when the construction fails (degenerate kernel, no diffusive pole,
          or β cannot bracket the target area), in which case the caller
          keeps the Lin-Goddard Lorentzian.
    """
    from scipy.optimize import brentq
    C_scalar = np.asarray(C_scalar, dtype=float)
    if C_scalar.size < 4 or not np.isfinite(C_scalar).all() or C_scalar[0] <= 0.0:
        return None
    dos_total = np.asarray(dos_total, dtype=float)
    if dos_total.size == 0 or dos_total[0] <= 0.0 or ref_gas_area <= 0.0:
        return None

    poles, residues, gamma_inst = _pade_kernel_fit(
        dt, C_scalar, M, N, None, omega_max, True, label)
    if poles is None:
        return None
    # Diffusive (gas) poles: overdamped / near-real (|Im| ≤ |Re|).  The
    # underdamped poles are cage resonances and stay in the solid residual.
    diff = np.abs(poles.imag) <= np.abs(poles.real)
    if not diff.any():
        return None
    p_d = poles[diff]
    r_d = residues[diff]
    iw  = 1j * np.asarray(omega, dtype=float)

    def _gas(beta: float):
        # K̃_gas evaluated on a frequency axis rescaled by β (broadens/narrows
        # the diffusive friction to tune the gas DoF without moving the ν=0
        # pin or reintroducing the cage resonances).
        Kgas = np.full(iw.shape, gamma_inst, dtype=complex)
        for p, r in zip(p_d, r_d):
            Kgas = Kgas + r / (iw / beta - p)
        Fg = np.real(1.0 / (iw + Kgas))
        Fg = np.nan_to_num(Fg, nan=0.0, posinf=0.0, neginf=0.0)
        if Fg[0] <= 0.0:
            return None
        Fg = Fg * (dos_total[0] / Fg[0])          # anchor 1: pin ν=0
        Fg = np.minimum(Fg, dos_total)            # solid ≥ 0 (2PT convention)
        Fg = np.maximum(Fg, 0.0)
        return Fg

    def _area_err(beta: float) -> float:
        g = _gas(beta)
        if g is None:
            return np.nan
        return float(np.trapezoid(g, dx=dnu)) - ref_gas_area

    # Anchor 2: solve β so ∫gas = 3fN.  The pinned-and-clipped gas area is a
    # non-monotonic function of β (it dips as the gas narrows, then peaks as
    # the friction broadens), so a two-endpoint bracket is unreliable.  Scan a
    # log-β grid, take the *first* sign change (smallest β → narrowest gas with
    # the sharpest kernel tail cutoff), and refine with brentq.  Fall back to
    # the Lorentzian when no grid point crosses the target (3fN unreachable —
    # e.g. an M==N fit whose γ_inst floor sits above the target area).
    betas = np.logspace(-2.0, 2.0, 400)
    errs  = np.array([_area_err(b) for b in betas])
    finite = np.isfinite(errs)
    if not finite.any():
        if label is not None:
            log.warning("kvol_gas_dos_split(%s): no finite gas DoS over the β "
                        "scan; falling back to Lorentzian.", label)
        return None
    sgn = np.sign(errs)
    cross = np.where(finite[:-1] & finite[1:] & (sgn[:-1] != sgn[1:]))[0]
    if cross.size == 0:
        if label is not None:
            log.warning("kvol_gas_dos_split(%s): target gas area %.4g not "
                        "reached over β∈[%.2g,%.2g] (area∈[%.3g,%.3g]); "
                        "falling back to Lorentzian.", label, ref_gas_area,
                        betas[0], betas[-1],
                        float(ref_gas_area + np.nanmin(errs)),
                        float(ref_gas_area + np.nanmax(errs)))
        return None
    i = int(cross[0])
    try:
        beta = brentq(_area_err, betas[i], betas[i + 1],
                      xtol=1e-4, rtol=1e-4, maxiter=100)
    except Exception as e:
        if label is not None:
            log.warning("kvol_gas_dos_split(%s): β refine failed (%s); "
                        "falling back to Lorentzian.", label, e)
        return None
    gas = _gas(beta)
    if gas is None:
        return None
    if label is not None:
        log.info("kvol_gas_dos_split(%s, [%d/%d]): %d diffusive / %d resonant "
                 "poles, β=%.4g, gas(0)=%.4g (pin total(0)=%.4g), ∫gas=%.4g "
                 "(target 3fN=%.4g)", label, M, N, int(diff.sum()),
                 int((~diff).sum()), beta, float(gas[0]), float(dos_total[0]),
                 float(np.trapezoid(gas, dx=dnu)), ref_gas_area)
    return gas


def cage_memory_entropy(dt: float, C_scalar: np.ndarray,
                        nu_cm: np.ndarray, dos_total: np.ndarray,
                        dos_gas: np.ndarray, T_K: float,
                        mass_amu: float, vol_per_atom_A3: float,
                        *,
                        prefactor: float | None = 1.0 / 3.0,
                        dimension: int = 3,
                        gate_f0: float = 0.01,
                        clip_eps: float = 1e-3,
                        nuc_scale: float = 1.0,
                        mainlobe_alpha: float = 0.02,
                        tail_tol: float = 1.0,
                        ref: str = "markov",
                        nf_run: int = 1,
                        taper: str = "none",
                        Wg_override: float | None = None,
                        label: str | None = None,
                        cage_out: list | None = None) -> float | None:
    """Cross-family **cage-memory entropy correction** ΔS [k_B per atom].

    A parameter-free post-correction to the rigorous-HS 2PT entropy that
    recovers the systematic solid-side deficit of rigorous-HS in structured
    liquids (liquid metals, dense LJ).  ΔS is the entropy difference between
    treating the *non-Markovian cage* motion as gas-like vs solid-like, taken
    over the low-frequency (diffusive-coupled) window:

        ΔS = p · ∫ cage(ν) · (1 − w(ν)) · (W_g − W_s(ν)) dν ,

      * ``cage(ν) = const·(F_K − F_M)`` — the memory excess of the friction
        kernel: ``F_K = Re[1/(iω + K̃)]`` (full Volterra kernel) minus its
        Markovian equivalent ``F_M = Re[1/(iω + γ)]``, ``γ = K̃(0)``.  Pinned
        by ``const = dos_total[0]/∫(C/C₀)dt`` and clipped to ``[0, solid2]``,
        ``solid2 = max(dos_total − dos_gas, 0)`` so the Einstein residual
        ``solid2 − cage`` stays non-negative.  ``cage(0)=0`` automatically
        (``F_K(0)=F_M(0)=1/γ``).

        Low-frequency sign structure (verified, MB-pol water 298 K): the
        unclipped excess is a zero-sum redistribution
        (``∫(F_K−F_M)dω = 0`` — both integrate to ``π/2·c(0)``).  The TRANS
        channel has no low-ν clipping: γ is band-scale, the excess is
        positive from 0⁺ and the deficit sits above the band.  The ROT
        channel clips everything below the libration band (ν₊≈319 cm⁻¹ at
        298 K): γ_rot ≫ band makes F_M essentially flat, so the sub-band
        mobility deficit (weight trapped and pushed up into librations) is
        negative excess.  The deficit is the caging seen from below; its
        magnitude sits in the sub-band mobility deficit below the libration band.
        


        ``nf_run`` / ``taper`` (cage_nf_run /
        cage_taper keywords): ``nf_run`` makes the Volterra noise-floor guard
        envelope-aware (consecutive sub-floor lags required; 1 = legacy) so a
        librational VACF node cannot truncate the kernel at random;
        ``taper="hann"`` rolls the truncated kernel to zero before the CZT,
        suppressing the sinc ringing the clip would convert into fake band
        gaps.  Defaults preserve legacy behavior bit-for-bit.
      * ``w(ν) = ν²/(ν² + ν_c²)`` high-passes out the harmonic tail; ``ν_c =
        Ω₀/2πc`` with ``Ω₀`` the Einstein frequency from the 2nd moment of the
        total DoS.
      * ``W_g`` = rigorous-HS gas weight per DoF (Sackur-Tetrode + Carnahan-
        Starling excess, NO ln z), ``W_s = 1 − ln(hcν/kT)`` harmonic.

    **Prefactor.** Originally the 2PT fluidicity ``f`` was used, but a p* sweep
    over LJ-Ar + EAM + Sutton-Chen liquid metals (2026-06) showed ×f is an
    LJ-family coincidence (p* tracks f only within the LJ family).  The
    cross-family universal value is ``p ≈ 1/3`` (≈ per-translational-DoF),
    which removes the low-fluidicity under-correction (Na/Ag) and makes the
    correction beat the TI-tuned R2PT (Sun et al. 2017) parameter-free.
    Default ``1/3``; pass ``prefactor=None`` to recover the legacy ×f.

    Parameters
    ----------
    dt              : VACF time step [ps].
    C_scalar        : (N_t,) scalar (trace/3) VACF, un-normalised.
    nu_cm           : (n,) DoS frequency grid [cm⁻¹], uniform spacing.
    dos_total       : (n,) total DoS on ``nu_cm`` (per atom, ∫ dν = 3).
    dos_gas         : (n,) standard 2PT Lin-Goddard Lorentzian gas DoS.
    T_K             : temperature [K].
    mass_amu        : atomic mass [amu].
    vol_per_atom_A3 : volume per atom [Å³].
    prefactor       : cage prefactor ``p`` (default 1/3; ``None`` → legacy ×f).
    label           : optional diagnostic tag.
    cage_out        : optional list; if given, the per-atom cage DoS array
                      ``cage(ν)`` (on ``nu_cm``, clipped to ``[0, solid2]``) is
                      appended to it on the success path — for the ``.pwr``
                      cage-column writer.  Left empty on any ``None`` return.

    Returns
    -------
    dS : float — cage entropy correction [k_B per atom]; or ``None`` when the
         kernel inversion is degenerate or the fluidicity is unphysical.
    """
    from scipy.signal import czt

    nu = np.asarray(nu_cm, dtype=float)
    tot = np.asarray(dos_total, dtype=float)
    gas = np.asarray(dos_gas, dtype=float)
    C = np.asarray(C_scalar, dtype=float)
    if nu.size < 2 or C.size < 2 or C[0] == 0.0:
        return None
    dnu = nu[1] - nu[0]

    # ── smooth (C^∞) clip ───────────────────────────────────────────────────
    # The cage clip is implemented as a softplus/LogSumExp regularization with
    # width 1/β = clip_eps·S(0) (β→∞, i.e. clip_eps→0, recovers the hard clip
    # max[min(·,solid2),0]).  The width sits at the DoS noise floor, so reported
    # entropies are unchanged (<1e-4 k_B), while all thermodynamic derivatives
    # (Cv=∂S/∂T, ∂S/∂ρ, …) stay continuous across the phase diagram.
    _beta = (1.0 / (clip_eps * tot[0])) if (clip_eps > 0.0 and tot[0] > 0.0) else None
    def _smax0(x):            # smooth max(x, 0)
        return x if _beta is None else np.logaddexp(0.0, _beta * x) / _beta
    def _smin(a, b):          # smooth min(a, b)
        return np.minimum(a, b) if _beta is None else -np.logaddexp(-_beta * a, -_beta * b) / _beta

    solid2 = _smax0(tot - gas)

    hc_k = 100.0 * H * VLIGHT / KB                # hc/k_B [cm·K] ≈ 1.43877
    u = np.where(nu > 0, hc_k * nu / T_K, 1e-9)
    Ws = np.where(nu > 0, 1.0 - np.log(u), 0.0)

    c_cm_ps = VLIGHT * 1e-10                       # speed of light [cm/ps]
    wa = 2.0 * PI * nu * c_cm_ps                   # angular frequency [1/ps]

    # scalar (trace/3) Volterra kernel K̃(ω) via the time-domain round-trip
    cn = C / C[0]
    Cm = np.zeros((cn.size, 3, 3))
    for i in range(3):
        Cm[:, i, i] = cn
    K = np.einsum('tii->t', invert_kernel_matrix(dt, Cm, nf_run=nf_run)) / 3.0
    nz = np.nonzero(K)[0]
    if not nz.size:
        if label:
            log.warning("cage_memory_entropy(%s): degenerate kernel", label)
        return None
    nK_auto = int(nz[-1]) + 1

    # ── auto-cutoff failure diagnostic ──────────────────────────────────────
    # The auto support (last non-zero lag of the guarded inversion) is the right
    # integration limit for a cleanly-decaying kernel, but a *smooth* spurious
    # tail beyond the main lobe — coherent, so undetected by the inversion's
    # noise/swing guards — inflates the friction γ=K̃(0) and hence the Markovian
    # reference F_M, over-counting the cage (the SPC/E water failure mode).  We
    # detect it by comparing the friction at the auto cutoff with that at the
    # main-lobe cutoff (first |K|<mainlobe_alpha·|K(0)|): when the post-main-lobe
    # tail shifts γ by more than tail_tol, it dominates the friction and we fall
    # back to the truncation-robust main-lobe cutoff (and warn).
    def _gamma_dc(nk):                       # K̃(0) = ∫K dt (matches czt[0].real)
        if nk < 2:
            return float(K[0]) * dt
        kv = K[:nk].copy(); kv[0] *= 0.5; kv[-1] *= 0.5
        return float(kv.sum()) * dt
    K0 = abs(K[0])
    _below = np.where(np.abs(K[3:]) < mainlobe_alpha * K0)[0] if K0 > 0.0 else np.array([], dtype=int)
    nK_main = (int(_below[0]) + 3) if _below.size else nK_auto
    nK = nK_auto
    if nK_main < nK_auto and tail_tol is not None:
        g_auto, g_main = _gamma_dc(nK_auto), _gamma_dc(nK_main)
        if abs(g_main) > 0.0 and abs(g_auto - g_main) > tail_tol * abs(g_main):
            log.warning("cage_memory_entropy%s: auto-cutoff friction inflated %.2gx by a "
                        "smooth post-main-lobe kernel tail (gamma_auto=%.3g vs "
                        "gamma_mainlobe=%.3g; t_auto=%.3g vs t_main=%.3g ps); "
                        "falling back to the main-lobe cutoff.",
                        f"({label})" if label else "",
                        g_auto / g_main, g_auto, g_main, nK_auto * dt, nK_main * dt)
            nK = nK_main

    Kv = K[:nK].copy()
    # truncation at t_cut rings K̃(ω) with period ~1/(c·t_cut), which the
    # [0, solid2] clip converts into spurious cage band gaps.  γ below is the
    # tapered kernel's DC value (self-consistent reference); the tail
    # safeguard above intentionally compared UNtapered γ values.  Mirrors
    if taper == "hann" and nK > 2:
        Kv *= 0.5 * (1.0 + np.cos(PI * np.arange(nK) / (nK - 1)))
    Kv[0] *= 0.5
    Kv[-1] *= 0.5
    Ktil = czt(Kv, m=wa.size,
               w=np.exp(-1j * 2.0 * PI * dnu * c_cm_ps * dt), a=1.0) * dt

    gamma = float(Ktil[0].real)
    if label and (nf_run > 1 or taper != "none"):
        log.info("cage_memory_entropy(%s): kernel t_cut=%.4g ps (nK=%d, "
                 "nf_run=%d), taper=%s, gamma=%.4g 1/ps",
                 label, nK * dt, nK, nf_run, taper, gamma)
    const = tot[0] / np.trapezoid(cn, dx=dt)
    F_K = np.real(1.0 / (1j * wa + Ktil))
    # Om0² (DoS 2nd moment) hoisted ahead of the cage construction: needed by
    Om0sq = float(np.trapezoid(wa**2 * tot, dx=1.0) / np.trapezoid(tot, dx=1.0))
    F_ref = gamma / (gamma**2 + wa**2)
    cage = _smax0(_smin(const * (F_K - F_ref), solid2))

    Om0 = np.sqrt(Om0sq)
    nuc = nuc_scale * Om0 / (2.0 * PI * c_cm_ps)   # nuc_scale=1: parameter-free Einstein cutoff
    w = nu**2 / (nu**2 + nuc**2)

    f = float(np.trapezoid(gas, dx=dnu) / dimension)   # gas DoF fraction (∫gas=d·f)
    if not (0.0 < f < 1.0):
        if label:
            log.warning("cage_memory_entropy(%s): unphysical fluidicity f=%.3f",
                        label, f)
        return None
    if Wg_override is not None:
        # Rotational (or other non-translational) channel: the per-DoF gas weight
        # is supplied directly (e.g. the rigid/free-rotor weight wsr).  The
        # hard-sphere Sackur-Tetrode + Carnahan-Starling excess machinery is
        # meaningless here, so skip packing_from_f_dgen / hs_excess_entropy_dgen
        # entirely and use the override scalar as W_g.
        Wg = float(Wg_override)
    else:
        y = packing_from_f_dgen(f, dimension)      # HS packing γ (d-dim contact relation)
        m_kg = mass_amu * 1e-3 / NA
        lam_invd = (2.0 * PI * m_kg * KB * T_K / H**2)**(dimension/2.0)   # λ⁻ᵈ [1/m^d]
        Vpa = vol_per_atom_A3 * (1e-10)**dimension     # [m^d]  (d=3: ×1e-30)
        Wg = ((dimension/2.0 + 1.0 + np.log(lam_invd * Vpa / f)) / dimension
              + hs_excess_entropy_dgen(y, dimension) / dimension)

    p = f if prefactor is None else prefactor
    if cage_out is not None:
        cage_out.append(cage)          # per-atom cage DoS for the .pwr writer

    # ── fluidicity gate (enforces the f→0 Debye–Einstein crystalline limit) ──
    # The bare memory excess F_K−F_M does not vanish in a crystal (γ=K̃(0) is
    # finite, so the Markovian Lorentzian F_M is a fictitious diffusive baseline),
    # which would make the cage spuriously over-correct the harmonic solid as
    # f→0.  Gate the cage amplitude by g(f)=f²/(f²+f0²): g→1 in the fluid regime
    # (every homogeneous fluid has f≳0.15) and g→0 as f→0 (3PT→rigorous-HS = the
    # harmonic crystal).  f0 sits in the empty fluidicity gap (max solid f≈4e-4,
    # min fluid f≈0.15), so the result is independent of its value there — the
    # gate is a fluid/crystal switch, not a fitted parameter.
    gate = (f * f) / (f * f + gate_f0 * gate_f0)
    dS = float(p * gate * np.trapezoid(cage * (1.0 - w) * (Wg - Ws), dx=dnu))
    if label and gate < 0.99:
        log.info("cage_memory_entropy(%s): fluidicity gate g(f=%.4f)=%.3f "
                 "(f0=%.3g) → cage suppressed toward the harmonic-crystal limit",
                 label, f, gate, gate_f0)
    return dS


def r2pt_entropy(nu_cm: np.ndarray, F_total: np.ndarray, f_delta1: float,
                 T_K: float, mass_amu: float, vol_per_atom_A3: float,
                 *, delta: float = 1.5, label: str | None = None) -> float | None:
    """Sun et al. (2017) revised-2PT (R2PT) translational entropy [k_B per atom, d=3].

    R2PT vs rigorous-HS 2PT: (i) gas fraction from f^δ = D/D₀ (Eq A8; δ=1.5 default)
    instead of δ=1; (ii) F_a-inclusive — gas entropy via the sum rule S_g = 3 f W_g
    and solid from the FULL F_s = F − f·F_g (no high-ν truncation); (iii) ln-z-free
    HS excess (= rigorous convention).  Validated against Sun Table I (≤0.02 k_B;
    cf. scripts/r2pt.py).  ``f_delta1`` is pyxpt's standard δ=1 fluidicity — Δ is
    δ-independent, so it is recovered from f_delta1 and Eq A8 re-solved at δ.
    Sun's R2PT is 3-dimensional; returns ``None`` for degenerate input.
    """
    from scipy.optimize import brentq
    nu = np.asarray(nu_cm, float); F = np.asarray(F_total, float)
    if nu.size < 2 or F[0] <= 0.0 or not (0.0 < f_delta1 < 1.0):
        return None
    dnu = nu[1] - nu[0]
    def _A8(f, x, dl):                       # Eq A8 with x = Δ^(-3/2)
        return (x**3*f**(3+4.5*dl) - 3*x**2*f**(3*dl+2) + 3*x*f**(1+1.5*dl)
                - 0.5*x*f**(1+2.5*dl) + f**dl - 1.0)
    try:
        x  = brentq(lambda z: _A8(f_delta1, z, 1.0), 1e-6, 1e6)   # Δ from the δ=1 f
        fg = brentq(lambda f: _A8(f, x, delta), 1e-9, 1.0 - 1e-12)  # re-solve at δ
    except ValueError:
        if label:
            log.warning("r2pt_entropy(%s): Eq A8 has no root (f_δ1=%.3f) — skipping",
                        label, f_delta1)
        return None
    gamma = x * fg**(1.5*delta + 1.0)        # Eq A4 packing fraction
    if not (0.0 < gamma < 1.0):
        return None
    # rigorous-HS gas weight W_g per DoF: Sackur-Tetrode + Carnahan-Starling, NO ln z
    m_kg = mass_amu*1e-3/NA
    lam_inv3 = (2.0*PI*m_kg*KB*T_K/H**2)**1.5
    Vpa = vol_per_atom_A3*(1e-10)**3
    W_IG = (2.5 + np.log(lam_inv3*Vpa/fg)) / 3.0
    W_ex = (gamma*(3.0*gamma - 4.0)/(1.0 - gamma)**2) / 3.0
    S_g  = 3.0*fg*(W_IG + W_ex)              # sum-rule gas (∫F_g = 3); includes F_a tail
    # solid from the FULL F_s = F − f·F_g (Lorentzian gas); quantum W_s
    F0 = F[0]; alpha = 12.0*fg/F0
    Fg = 12.0*alpha/(alpha**2 + 4.0*PI**2*nu**2)
    hc_k = 100.0*H*VLIGHT/KB
    u = np.where(nu > 0, hc_k*nu/T_K, 1e-12)
    Ws = np.where(nu > 0, u/np.expm1(u) - np.log1p(-np.exp(-u)), 0.0)
    integ = (F - fg*Fg)*Ws; integ[0] = 0.0   # F_s(0)=0 cancels the W_s(0) divergence
    S_s = float(np.trapezoid(integ, dx=dnu))
    return float(S_g + S_s)


def desjarlais_mf_gas_dos(dt: float, C_scalar: np.ndarray,
                          omega: np.ndarray, dos_total: np.ndarray,
                          ref_gas_area: float, dnu: float,
                          *,
                          label: str | None = None) -> np.ndarray | None:
    """Parametric Desjarlais memory-function gas DoS (moment-matched Gaussian).

    Desjarlais's *original* construction: model the gas-side memory kernel as a
    smooth parametric Gaussian whose low-order spectral moments match the
    observed velocity power spectrum, rather than inverting the trajectory's
    Volterra kernel (cf. :func:`kvol_gas_dos_split`).  Because the kernel is a
    two-parameter Gaussian it cannot reconstruct the full structured spectrum
    (it gives a smooth envelope), unlike the full Volterra kernel whose
    Desjarlais image is identically the total DoS.

    Construction
    ------------
    From the velocity DoS ``dos_total`` (∝ the power spectrum) take the moment
    ratios on the angular-frequency grid ``omega``:

        Ω₀² = M₂/M₀          (Einstein frequency²; = K(0) of the kernel)
        Δ²  = M₄/M₂ − M₂/M₀   (kernel curvature; the kernel's 2nd moment)

    with ``Mₙ = ∫ ωⁿ S(ω) dω``.  The Gaussian memory kernel is
    ``K(t) = Ω₀² exp(−½ Δ² t²)`` (so ``K(0)=Ω₀²``, ``K'(0)=0`` — physical, unlike
    a single exponential), whose Laplace image ``K̃(iω)`` gives the gas DoS via
    the Desjarlais relation ``F_g(ν) = Re[1/(K̃(iω)+iω)]``.

    Normalisation is the same two-anchor scheme as :func:`kvol_gas_dos_split`:
    pin ``gas(0)=total(0)`` (⇒ ``solid(0)=0``) and frequency-rescale to
    ``∫gas = ref_gas_area = 3fN``.  Returns ``None`` (caller keeps the
    Lin–Goddard Lorentzian) on degenerate moments (Δ² ≤ 0) or an unreachable
    area target.
    """
    from scipy.optimize import brentq
    S = np.asarray(dos_total, dtype=float)
    w = np.asarray(omega, dtype=float)
    if S.size < 4 or S[0] <= 0.0 or ref_gas_area <= 0.0:
        return None
    M0 = float(np.trapezoid(S, dx=1.0))
    M2 = float(np.trapezoid(w**2 * S, dx=1.0))
    M4 = float(np.trapezoid(w**4 * S, dx=1.0))
    if M0 <= 0.0 or M2 <= 0.0:
        return None
    Omega0_sq = M2 / M0
    Delta_sq  = M4 / M2 - M2 / M0
    if not np.isfinite(Delta_sq) or Delta_sq <= 0.0:
        if label is not None:
            log.warning("desjarlais_mf_gas_dos(%s): non-positive kernel "
                        "curvature Δ²=%.3g (4th moment); falling back to "
                        "Lorentzian.", label, Delta_sq)
        return None

    # Gaussian kernel sampled on a FINE internal grid (the kernel is analytic,
    # so we are not tied to the VACF dump spacing).  dt_k is chosen to both
    # resolve the kernel (dt_k ≪ 1/Δ) and keep the Laplace integral below the
    # ω-grid Nyquist (ω_max·dt_k ≪ π) — the VACF spacing `dt` (often ~0.06 ps
    # for 8-step dumps) would otherwise alias K̃ at high ω.  The grid spans ≈6/Δ
    # (where the Gaussian has decayed to ~10⁻⁸).
    _wmax = float(max(np.max(np.abs(w)), 1.0))
    dt_k  = min(0.5 / np.sqrt(Delta_sq), 0.5 / _wmax)
    n_k   = int(np.ceil(6.0 / (np.sqrt(Delta_sq) * dt_k))) + 1
    tk    = np.arange(n_k) * dt_k
    Kt    = Omega0_sq * np.exp(-0.5 * Delta_sq * tk**2)
    tw    = np.ones(n_k); tw[0] = tw[-1] = 0.5          # trapezoid endpoints
    Kt_tw = Kt * tw
    tot0  = float(S[0])
    iw = 1j * w

    def _gas(beta: float):
        # K̃(iω/β) = ∫₀^∞ K(t) e^{-i(ω/β)t} dt  (direct trapezoidal Laplace)
        Ktil = (np.exp(-1j * np.outer(w / beta, tk)) @ Kt_tw) * dt_k
        Fg = np.real(1.0 / (iw + Ktil))
        Fg = np.nan_to_num(Fg, nan=0.0, posinf=0.0, neginf=0.0)
        if Fg[0] <= 0.0:
            return None
        Fg = Fg * (tot0 / Fg[0])             # pin ν=0
        return np.clip(np.minimum(Fg, S), 0.0, None)

    def _err(beta: float) -> float:
        g = _gas(beta)
        return np.nan if g is None else float(np.trapezoid(g, dx=dnu)) - ref_gas_area

    betas = np.logspace(-2.0, 2.0, 400)
    errs  = np.array([_err(b) for b in betas])
    fin   = np.isfinite(errs); sgn = np.sign(errs)
    cross = np.where(fin[:-1] & fin[1:] & (sgn[:-1] != sgn[1:]))[0]
    if cross.size == 0:
        if label is not None:
            log.warning("desjarlais_mf_gas_dos(%s): area %.4g not reached over "
                        "β scan; falling back to Lorentzian.", label, ref_gas_area)
        return None
    i = int(cross[0])
    try:
        beta = brentq(_err, betas[i], betas[i + 1], xtol=1e-4, rtol=1e-4, maxiter=100)
    except Exception:
        return None
    gas = _gas(beta)
    if gas is not None and label is not None:
        log.info("desjarlais_mf_gas_dos(%s): Ω₀=%.3g 1/ps, Δ=%.3g 1/ps, β=%.4g, "
                 "gas(0)=%.4g (pin total(0)=%.4g), ∫gas=%.4g (target 3fN=%.4g)",
                 label, np.sqrt(Omega0_sq), np.sqrt(Delta_sq), beta,
                 float(gas[0]), tot0, float(np.trapezoid(gas, dx=dnu)), ref_gas_area)
    return gas


def invert_kernel_pade(dt: float, C_scalar: np.ndarray,
                         *,
                         M: int = 4,
                         N: int = 4,
                         n_freq: int | None = None,
                         omega_max: float | None = None,
                         stabilize: bool = True,
                         label: str | None = None,
                         ) -> np.ndarray:
    """Padé spectral-approximation inversion of scalar K(t) from C(t) (MV5.5c).

    Fits a rational function $\\tilde K(s) \\approx P_M(s) / Q_N(s)$ to the
    per-frequency K̃(iω) values computed from C̃ on a band-limited grid,
    then inverse-Laplace transforms analytically via partial fractions
    to produce a smooth K(t) without IFFT-aliasing artefacts.

    Strategy
    --------
    1.  Compute K̃(iω_k) at n_freq uniformly-spaced frequencies in
        [0, omega_max] via the existing C(0)·C̃(iω)⁻¹ − iω formula
        (using direct quadrature on C(t) for ω_k below the discrete
        Nyquist).  This gives a clean K̃ shape at low/mid frequency
        where the discrete sampling faithfully represents the
        continuous Laplace.
    2.  Fit complex-valued least squares: minimise
        Σ_k |P_M(iω_k) − K̃(iω_k)·Q_N(iω_k)|² with Q_N(0) = 1
        normalisation.  Splits into a real (2·n_freq, M+N+1) system.
    3.  Optionally reflect RHP poles into the LHP (``stabilize=True``)
        for causal K(t).
    4.  Partial-fraction decompose P/Q via ``scipy.signal.residue``;
        each pole p_k contributes A_k·exp(p_k·t)·θ(t) to K(t); the
        constant direct term contributes γ_inst·δ(t).
    5.  Reconstruct discrete K[n]:
            K[0] = γ_inst/dt + Σ A_k                  (δ + smooth at t=0⁺)
            K[n>0] = Re[Σ A_k · exp(p_k · n·dt)]      (real by Hermitian
                                                       symmetry of poles)

    Compared to MV5.5d (δ separation + apodization):
    Padé regularises the entire spectrum globally via the low-order
    rational form; smoothness comes from the parametric constraint
    rather than post-hoc apodization.  Analytic inverse-Laplace
    eliminates IFFT-aliasing entirely.

    Degree selection (M, N): low order (e.g. M=N=2-4) captures the
    dominant memory time-scales without overfitting noise.  For a
    pure exponential C(t)=exp(-γt) the exact form is Padé[1,1]; for
    underdamped cosine C(t)=exp(-γt)cos(ω₀t) it's Padé[1,1] in s.
    Anharmonic / multi-time-scale memory typically wants M=N=4-8.

    Parameters
    ----------
    dt        : float — time step [ps] for the discrete output K[n]
    C_scalar  : (N_t,) — scalar VACF samples (un-normalised; trace/3
                if the caller has a matrix VAC)
    M         : int, default 4 — numerator degree
    N         : int, default 4 — denominator degree (Q(0)=1 normalisation)
    n_freq    : int, optional — number of K̃ fit frequencies.  Default
                4·(M+N+1) for a well-conditioned least squares.
    omega_max : float, optional — upper fit frequency [1/ps].  Default
                0.5 · π/dt (half the discrete Nyquist) to stay below
                the aliasing band.
    stabilize : bool, default True — reflect RHP poles into LHP so the
                reconstructed K(t) decays.
    label     : str, optional — diagnostic tag for the INFO log line.

    Returns
    -------
    K : (N_t,) [ps⁻²] — scalar memory kernel; K[0] includes the
        γ_inst/dt + smooth-tail Form-B convention, K[n>0] is the
        rational reconstruction.
    """
    N_t = len(C_scalar)
    if N_t < 4 or M < 0 or N < 1:
        return np.zeros(N_t)

    # 1-3. Padé rational fit → partial-fraction (poles, residues, γ_inst).
    poles, residues, gamma_inst = _pade_kernel_fit(
        dt, C_scalar, M, N, n_freq, omega_max, stabilize, label)
    if poles is None:
        if label is not None:
            log.warning("invert_kernel_pade(%s): fit failed; returning zeros.",
                        label)
        return np.zeros(N_t)

    # 4. Reconstruct K(t) at the discrete grid.
    K = np.zeros(N_t)
    t = np.arange(N_t) * dt
    smooth_tail_at_0 = float(np.real(np.sum(residues)))
    K[0] = gamma_inst / dt + smooth_tail_at_0
    if N_t > 1:
        # K[n>0] = Re[Σ residues_k · exp(poles_k · t_n)]
        # Vectorised: (N_t-1, n_poles) outer product
        exp_pt = np.exp(np.outer(t[1:], poles))             # (N_t-1, n_poles)
        K[1:] = np.real(exp_pt @ residues)

    if label is not None:
        log.info("invert_kernel_pade(%s, [%d/%d]): γ_inst=%.3g, "
                 "n_poles=%d (max Re(p)=%.3g), K[0]=%.3g, K[5]=%.3g",
                 label, M, N, gamma_inst, len(poles),
                 float(np.max(poles.real)) if len(poles) else 0.0,
                 K[0], K[5] if N_t > 5 else 0.0)
    return K


def kvol_K_from_C(dt: float, C_matrix: np.ndarray,
                    omega_grid: np.ndarray,
                    *,
                    cond_warn: float = 1e10,
                    label: str | None = None,
                    ) -> np.ndarray:
    """Direct K̃(iω) evaluation at an arbitrary frequency grid, bypassing
    the K(t) round-trip (MV5.5a Phase 2).

    Computes K̃(iω) at the user-supplied ``omega_grid`` (in units of
    1/time, matching ``dt``) via:

      (1) C̃(iω_k) = CZT(C_matrix; geometric W on ω_grid) · dt
          — trapezoidal-weighted to match np.trapezoid output to FP
          precision
      (2) K̃(iω_k) = C(0)·C̃(iω_k)⁻¹ − iω_k I    per-frequency
          — using safe_matrix_inverse for cond-guarded per-frequency
          inversion (pinv fallback at cond > cond_warn)

    For the kvol-DoS use case the input ``omega_grid`` is the engine's
    ``2π·_pwrfreq·c_cm_ps`` array (kvol cm⁻¹ grid mapped to 1/ps).
    Returns K̃(iω) directly at those frequencies — the Desjarlais Eq. 16
    DoS construction can read this without going through the K(t)→Laplace
    round-trip that the current pipeline does.

    For the scalar case (m=1) the result is shape ``(M,)`` complex.  For
    matrix (m ≥ 3) the result is shape ``(M, m, m)`` complex.

    The kvol DoS only needs K̃ at the frequency band where Desjarlais
    Eq. 16 is integrated (typically ≤ ~3000 cm⁻¹, well below the discrete
    Nyquist for production timesteps), so no Nyquist-edge cancellation is
    required.

    Restrictions:
      • Caller is responsible for keeping ``omega_grid`` below the
        discrete-sampling Nyquist edge (typically ω_max < 0.5·π/dt for
        the per-frequency inversion to be physically meaningful).
      • Above that band, C̃(iω) suffers the same bandwidth-limit issue
        that MV5 documents; K̃ values become unreliable.  An upper-
        cutoff or apodization layer (MV5.5b) would be a future
        enhancement for callers that need K̃ above the band.

    Parameters
    ----------
    dt        : float — time-domain sample spacing of C_matrix [ps]
    C_matrix  : (N,) or (N, m, m) — un-normalised VACF samples
    omega_grid: (M,) — output angular-frequency grid [1/ps]
    cond_warn : float, default 1e10 — pinv fallback threshold per
                frequency.  Same as the time-domain ``safe_matrix_inverse``
                for cross-path consistency.
    label     : str, optional — diagnostic tag aggregated into a single
                INFO log line when any frequency triggers the pinv
                fallback.

    Returns
    -------
    K_tilde : (M,) complex if scalar input, (M, m, m) complex if matrix.
    """
    import scipy.signal as _ssig
    if C_matrix.ndim == 1:
        # Scalar fast path (m=1).
        N = C_matrix.shape[0]
        C_tz = C_matrix.copy()
        C_tz[0]  *= 0.5
        C_tz[-1] *= 0.5
        # Uniform omega grid; assert and compute step.
        if len(omega_grid) < 2:
            domega = float(omega_grid[0]) if len(omega_grid) > 0 else 0.0
        else:
            domega = float(omega_grid[1] - omega_grid[0])
        W = np.exp(-1j * domega * dt)
        C_tilde = _ssig.czt(C_tz, m=int(omega_grid.size), w=W, a=1.0) * dt
        C0 = float(C_matrix[0])
        # Scalar K̃(iω) = C(0) / C̃(iω) - iω.  C̃ can be ≈ 0 at high ω
        # (the Nyquist-band failure regime); guard with absolute-value
        # threshold and emit aggregated INFO if it fires.
        n_pinv = int(np.sum(np.abs(C_tilde) < (np.max(np.abs(C_tilde)) / cond_warn)))
        with np.errstate(divide='ignore', invalid='ignore'):
            K_tilde = C0 / C_tilde - 1j * omega_grid
        # Replace any non-finite K̃ from C̃ ≈ 0 with 0 (the kvol DoS
        # construction then floors the resulting F_g contribution at zero).
        K_tilde = np.where(np.isfinite(K_tilde), K_tilde, 0.0)
        if n_pinv > 0 and label is not None:
            log.info("kvol_K_from_C(%s, scalar): %d/%d frequencies hit "
                     "|C̃| < max(|C̃|)/%.0e — K̃ set to 0 there.",
                     label, n_pinv, len(omega_grid), cond_warn)
        return K_tilde
    # Matrix path (m ≥ 1).
    N, m, _ = C_matrix.shape
    # CZT operates along axis 0; apply trapezoidal weights at endpoints.
    C_tz = C_matrix.copy()
    C_tz[0]  *= 0.5
    C_tz[-1] *= 0.5
    if len(omega_grid) < 2:
        domega = float(omega_grid[0]) if len(omega_grid) > 0 else 0.0
    else:
        domega = float(omega_grid[1] - omega_grid[0])
    W = np.exp(-1j * domega * dt)
    # Flatten m×m → m² and CZT each component separately (CZT is 1-D in scipy).
    C_flat = C_tz.reshape(N, m * m)
    C_tilde_flat = np.empty((omega_grid.size, m * m), dtype=complex)
    for c in range(m * m):
        C_tilde_flat[:, c] = _ssig.czt(C_flat[:, c],
                                         m=int(omega_grid.size), w=W, a=1.0)
    C_tilde = C_tilde_flat.reshape(omega_grid.size, m, m) * dt
    # Per-frequency K̃ = C(0)·C̃⁻¹ − iω·I via safe_matrix_inverse.
    K_tilde = np.empty((omega_grid.size, m, m), dtype=complex)
    C0      = C_matrix[0].astype(float)
    eye_m   = np.eye(m)
    n_pinv  = 0
    for k in range(omega_grid.size):
        Ck_inv, _, fb = safe_matrix_inverse(C_tilde[k], cond_warn=cond_warn,
                                              label=None)
        if fb:
            n_pinv += 1
        K_tilde[k] = C0 @ Ck_inv - 1j * float(omega_grid[k]) * eye_m
    if n_pinv > 0 and label is not None:
        log.info("kvol_K_from_C(%s, matrix m=%d): %d/%d frequencies needed "
                 "pinv fallback (cond > %.0e) — K̃ at those ω is "
                 "diagnostic-only.", label, m, n_pinv, omega_grid.size,
                 cond_warn)
    return K_tilde


# ── Relocated pressure-unit + shear-viscosity helpers (formerly thermo/transport.py) ──
_PRESS_TO_PA = {
    "real":  101325.0,    # atm  → Pa
    "metal": 1.0e5,       # bar  → Pa
    "si":    1.0,         # Pa already
    "":      101325.0,    # auto / blank → assume real (LAMMPS default)
}


def lammps_press_to_pa(units: str) -> float:
    """Conversion factor: LAMMPS pressure unit → Pa."""
    return _PRESS_TO_PA.get((units or "").strip().lower(), 101325.0)

def shear_viscosity_from_stress_block(stress_block_Pa: np.ndarray,
                                        dt_ps: float, V_m3: float,
                                        T_K: float,
                                        corlen: float = 0.5
                                        ) -> tuple[float, np.ndarray]:
    """
    Shear viscosity from off-diagonal stress autocorrelation:

        η = (V/kT) × ∫₀^∞ ⟨σ_αβ(0) σ_αβ(t)⟩ dt    (averaged over αβ ∈ {xy, xz, yz})

    Parameters
    ----------
    stress_block_Pa : (nf, 3) — system off-diagonal stress σ_xy, σ_xz, σ_yz [Pa]
    dt_ps           : float   — frame spacing [ps]
    V_m3            : float   — system volume [m³]
    T_K             : float   — temperature [K]

    Returns
    -------
    eta_PaS : float            — shear viscosity [Pa·s]
    G_t_Pa  : (nf,) np.ndarray — stress relaxation function G(t) = ⟨σ(0)σ(t)⟩/V·kT * V
                                  in Pa (the time-domain shear modulus)
    """
    nf = stress_block_Pa.shape[0]
    if nf < 4:
        return 0.0, np.zeros(0)
    pad = 2 * nf
    acf = np.zeros(nf, dtype=np.float64)
    for d in range(stress_block_Pa.shape[1]):
        x   = stress_block_Pa[:, d] - stress_block_Pa[:, d].mean()
        F   = _onfft.rfft(x, n=pad)
        psd = np.abs(F) ** 2
        c   = _onfft.irfft(psd, n=pad)[:nf].real / nf
        acf += c
    acf /= stress_block_Pa.shape[1]   # average over the 3 off-diagonals
    # G(t) = (V/kT) · ⟨σ(0)σ(t)⟩  has units Pa
    G_t_Pa = (V_m3 / (KB * T_K)) * acf
    # η = ∫G dt with dt in seconds, truncated at corlen·nf to discard the
    # long-time-tail noise that otherwise dominates the trapezoid endpoint
    # correction.
    dt_s  = dt_ps * 1e-12
    n_int = max(int(corlen * nf), 4)
    eta   = float(_trapz(G_t_Pa[:n_int], dx=dt_s))
    return eta, G_t_Pa


