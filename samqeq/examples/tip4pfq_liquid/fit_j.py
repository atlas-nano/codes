#!/usr/bin/env python3
"""
Fit the cube-root and Gaussian kernels of samQEq to the exact Slater-orbital Coulomb integrals of TIP4P-FQ
(jrab.py), at the two intramolecular distances a rigid molecule samples: H-H at 1.5139 A and M-H at 0.8735 A
(from r_OH = 0.9572 A, r_OM = 0.15 A and theta_HOH = 104.52 deg). The O site carries no charge, so these are
the only two charge-response pairs. The M site uses the O 2s form factor, as in the original model. Writes
fit_results.txt; the Gaussian radii are those of water4_liquid_gauss.param.
"""
import math
import numpy as np
from jrab import J_OH, J_HH, J_OO, jrab_raw, QQRD2E_KCAL, ZETA_O, ZETA_H

# ---- rigid TIP4P-FQ geometry (Table I) ----
R_OH = 0.9572
THETA = math.radians(104.52)
R_OM = 0.15
R_HH = 2.0 * R_OH * math.sin(THETA / 2.0)          # = 1.5152 (paper quotes 1.5139, use paper's value as the fit target)
R_HH_PAPER = 1.5139
# M-H distance via law of cosines in triangle O-H-M (half-angle between OH bond and bisector = THETA/2)
R_MH = math.sqrt(R_OH**2 + R_OM**2 - 2.0*R_OH*R_OM*math.cos(THETA/2.0))

print(f"Geometry check: R_HH (from Table I roh,theta) = {R_HH:.4f} A  (paper checkpoint quotes 1.5139)")
print(f"Geometry check: R_MH (law of cosines, r_OM=0.15 off bisector) = {R_MH:.4f} A  (NOT r_OH=0.9572)")
print()

# Targets in raw units (1/A): the Coulomb constant multiplies both sides of the matching condition.
target_HH_raw = jrab_raw(R_HH_PAPER, False, False, ZETA_H, ZETA_H)
target_MH_raw = jrab_raw(R_MH, True, False, ZETA_O, ZETA_H)
target_MH_raw_at_0p9572 = jrab_raw(R_OH, True, False, ZETA_O, ZETA_H)

print(f"target J_HH_raw(r={R_HH_PAPER}) = {target_HH_raw:.6f} 1/A  ({target_HH_raw*QQRD2E_KCAL:.2f} kcal/mol/e^2)")
print(f"target J_MH_raw(r={R_MH:.4f})  = {target_MH_raw:.6f} 1/A  ({target_MH_raw*QQRD2E_KCAL:.2f} kcal/mol/e^2)  <- fit anchor")
print(f"  (for reference, J_MH_raw at literal r=0.9572 = {target_MH_raw_at_0p9572*QQRD2E_KCAL:.2f} kcal/mol/e^2)")
print()

# ---------------------------------------------------------------------
# Kernel A: cbrt  sh(r) = 1/(r^3 + 1/gamma^3)^(1/3),  gamma_ij = sqrt(gamma_i*gamma_j)
# ---------------------------------------------------------------------
def sh_cbrt(r, gamma):
    return 1.0 / (r**3 + 1.0/gamma**3) ** (1.0/3.0)

def solve_gamma_cbrt(r, target_raw):
    # sh(r,gamma)=target -> 1/gamma^3 = 1/target^3 - r^3
    inv_g3 = 1.0/target_raw**3 - r**3
    if inv_g3 <= 0:
        raise ValueError("target too large for cbrt kernel at this r (unshielded 1/r already exceeds target)")
    return inv_g3 ** (-1.0/3.0)

gamma_H_cbrt = solve_gamma_cbrt(R_HH_PAPER, target_HH_raw)          # H-H same-type -> gamma_ij=gamma_H directly
gamma_HM_eff_cbrt = solve_gamma_cbrt(R_MH, target_MH_raw)           # = sqrt(gamma_M*gamma_H)
gamma_M_cbrt = gamma_HM_eff_cbrt**2 / gamma_H_cbrt

print("=== cbrt kernel fit ===")
print(f"gamma_H = {gamma_H_cbrt:.6f} A")
print(f"gamma_M = {gamma_M_cbrt:.6f} A   (gamma_ij(M,H) = sqrt(gamma_M*gamma_H) = {math.sqrt(gamma_M_cbrt*gamma_H_cbrt):.6f})")
print(f"check H-H: sh={sh_cbrt(R_HH_PAPER, gamma_H_cbrt):.6f} vs target {target_HH_raw:.6f}")
print(f"check M-H: sh={sh_cbrt(R_MH, math.sqrt(gamma_M_cbrt*gamma_H_cbrt)):.6f} vs target {target_MH_raw:.6f}")
print()

# ---------------------------------------------------------------------
# Kernel B: gaussian (pqeq)  sh(r) = erf(alpha_ij r)/r,
#   alpha_i = lambda*0.5/gamma_i^2  (lambda = 0.462770, hardcoded in FixQEqSam)
#   alpha_ij = sqrt(alpha_i*alpha_j/(alpha_i+alpha_j))
# ---------------------------------------------------------------------
LAM = 0.462770

def sh_gauss_same(r, gamma):
    a = LAM*0.5/gamma**2
    aij = math.sqrt(a*a/(2.0*a))   # = sqrt(a/2)
    return math.erf(aij*r)/r

def sh_gauss_cross(r, gamma_i, gamma_j):
    ai = LAM*0.5/gamma_i**2
    aj = LAM*0.5/gamma_j**2
    aij = math.sqrt(ai*aj/(ai+aj))
    return math.erf(aij*r)/r

def bisect(f, lo, hi, tol=1e-12, maxit=200):
    flo = f(lo)
    for _ in range(maxit):
        mid = 0.5*(lo+hi)
        fm = f(mid)
        if flo*fm <= 0:
            hi = mid
        else:
            lo = mid; flo = fm
        if hi-lo < tol:
            break
    return 0.5*(lo+hi)

# H-H: bisect over a broad bracket.
gamma_H_gauss = bisect(lambda g: sh_gauss_same(R_HH_PAPER, g) - target_HH_raw, 0.01, 10.0)

# M-H: solve gamma_M given gamma_H_gauss fixed
gamma_M_gauss = bisect(lambda g: sh_gauss_cross(R_MH, g, gamma_H_gauss) - target_MH_raw, 0.01, 10.0)

print("=== gaussian (pqeq) kernel fit ===")
print(f"gamma_H (Rc_H) = {gamma_H_gauss:.6f} A")
print(f"gamma_M (Rc_M) = {gamma_M_gauss:.6f} A")
print(f"check H-H: sh={sh_gauss_same(R_HH_PAPER, gamma_H_gauss):.6f} vs target {target_HH_raw:.6f}")
print(f"check M-H: sh={sh_gauss_cross(R_MH, gamma_M_gauss, gamma_H_gauss):.6f} vs target {target_MH_raw:.6f}")
print()

# ---------------------------------------------------------------------
# Diagnostic only: residual of the fitted M-M kernel against the exact integral over r in [2.4, 6] A, the
# first intermolecular shell, where the fit is not constrained.
# ---------------------------------------------------------------------
r_diag = np.linspace(2.4, 6.0, 37)
true_MM_raw = jrab_raw(r_diag, True, True, ZETA_O, ZETA_H*0+ZETA_O)  # M-M (2s-2s)
true_HH_raw = jrab_raw(r_diag, False, False, ZETA_H, ZETA_H)

cbrt_MM = np.array([sh_cbrt(r, gamma_M_cbrt) for r in r_diag])
cbrt_HH = np.array([sh_cbrt(r, gamma_H_cbrt) for r in r_diag])
gauss_MM = np.array([sh_gauss_same(r, gamma_M_gauss) for r in r_diag])
gauss_HH = np.array([sh_gauss_same(r, gamma_H_gauss) for r in r_diag])

def rmse(a, b):
    return float(np.sqrt(np.mean((a-b)**2)))

print("=== diagnostic RMSE over r in [2.4,6] A (raw 1/A units; x332.06 for kcal/mol/e^2) ===")
print(f"cbrt   M-M: {rmse(cbrt_MM, true_MM_raw):.6e}   H-H: {rmse(cbrt_HH, true_HH_raw):.6e}")
print(f"gauss  M-M: {rmse(gauss_MM, true_MM_raw):.6e}   H-H: {rmse(gauss_HH, true_HH_raw):.6e}")
print()

# ---------------------------------------------------------------------
# Tabulation for r in [0.5, 8] A
# ---------------------------------------------------------------------
r_tab = np.array([0.5,0.6,0.7,0.8,0.8736,0.9572,1.0,1.2,1.5139,1.6,2.0,2.4,3.0,4.0,5.0,6.0,8.0])
print("=== J(r) reference table (kcal/mol/e^2) ===")
print(f"{'r(A)':>8} {'J_OO(2s-2s)':>12} {'J_OH(2s-1s)':>12} {'J_HH(1s-1s)':>12} {'332.06/r':>10}")
for r in r_tab:
    print(f"{r:8.4f} {J_OO(r):12.3f} {J_OH(r):12.3f} {J_HH(r):12.3f} {QQRD2E_KCAL/r:10.3f}")
print()

# Save fitted gammas for downstream param-file building
with open("fit_results.txt", "w") as f:
    f.write(f"R_HH_fit={R_HH_PAPER}\nR_MH_fit={R_MH}\n")
    f.write(f"cbrt gamma_H={gamma_H_cbrt}\ncbrt gamma_M={gamma_M_cbrt}\ncbrt gamma_O={gamma_M_cbrt}\n")
    f.write(f"gauss gamma_H={gamma_H_gauss}\ngauss gamma_M={gamma_M_gauss}\ngauss gamma_O={gamma_M_gauss}\n")
print("Wrote fit_results.txt")
