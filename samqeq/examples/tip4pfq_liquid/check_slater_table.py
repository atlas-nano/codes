#!/usr/bin/env python3
"""
Parity checker for the samQEq C++ Slater-overlap kernel (slater_jtable.h::slater_jraw).

This is an INDEPENDENT re-implementation (in numpy, from the C++ source's own constants
and algebra -- NOT an import of jrab.py) of slater_jtable.h::slater_jraw, checked against
jrab.py's jrab_raw (the validated Python reference the C++ was transcribed from) over
r in [0.05, 10] Angstrom for the three pairs Rick's TIP4P-FQ actually samples:
  O-O / M-M  (2s-2s, zeta_O)
  M-H        (2s(O)-1s(H), zeta_O/zeta_H)   -- Rick's "johr" pairing
  H-H        (1s-1s, zeta_H)                -- Rick's "jhhr" pairing

Only transcription errors should show up here (same quadrature, same kmax/nint) -- target
max relative deviation < 1e-10 (in practice this is a same-algorithm-different-language
comparison, so it should land near machine epsilon, ~1e-14).

Run: python3 check_slater_table.py
"""
import math
import numpy as np

from jrab import jrab_raw, ZETA_O, ZETA_H   # the validated Python reference (jrab.py, this dir)


def slater_jraw_cpp_reimpl(r, i2s, j2s, zeta_i, zeta_j, kmax=8.0, nint=1000):
    """Fresh re-implementation of slater_jtable.h::slater_jraw's algebra (C++ source, NOT jrab.py),
    written independently from the C++ constants/expression to catch transcription drift between
    the two.  Returns (J, dJdr) exactly like the C++ function (raw 1/Angstrom units)."""
    dk = kmax / nint
    zi2, zi4 = zeta_i * zeta_i, zeta_i ** 4
    zj2, zj4 = zeta_j * zeta_j, zeta_j ** 4

    sum_ = 0.5 * dk
    dsum = 0.0
    for n in range(1, nint + 1):
        k = n * dk
        aa1 = 1.0 + (k / (2.0 * zeta_i)) ** 2
        aa2 = 1.0 + (k / (2.0 * zeta_j)) ** 2
        fk1 = 1.0 / aa1 ** 2
        fk2 = 1.0 / aa2 ** 2
        k2, k4 = k * k, k ** 4

        fa = (fk1 - 3.0 * k2 * fk1 ** 1.5 / (4.0 * zi2) + k4 * fk1 ** 2 / (8.0 * zi4)) if i2s else fk1
        fb = (fk2 - 3.0 * k2 * fk2 ** 1.5 / (4.0 * zj2) + k4 * fk2 ** 2 / (8.0 * zj4)) if j2s else fk2

        u = k * r
        if u < 1.0e-12:
            sinc, dsinc_dr = 1.0, 0.0
        else:
            su, cu = math.sin(u), math.cos(u)
            sinc = su / u
            dsinc_dr = (u * cu - su) / (k * r * r)
        sum_ += sinc * fa * fb * dk
        dsum += dsinc_dr * fa * fb * dk

    J = sum_ * 2.0 / math.pi
    dJdr = dsum * 2.0 / math.pi
    return J, dJdr


def main():
    r_grid = np.linspace(0.05, 10.0, 80)

    pairs = [
        ("O-O/M-M (2s-2s)", True,  True,  ZETA_O, ZETA_O),
        ("M-H (2s-1s)",     True,  False, ZETA_O, ZETA_H),
        ("H-H (1s-1s)",     False, False, ZETA_H, ZETA_H),
    ]

    overall_max_rel = 0.0
    for name, i2s, j2s, zi, zj in pairs:
        ref = jrab_raw(r_grid, i2s, j2s, zi, zj)                       # jrab.py (the validated reference)
        mine = np.array([slater_jraw_cpp_reimpl(r, i2s, j2s, zi, zj)[0] for r in r_grid])   # C++-formula reimpl
        rel = np.abs(mine - ref) / np.maximum(np.abs(ref), 1e-30)
        max_rel = rel.max()
        overall_max_rel = max(overall_max_rel, max_rel)
        print(f"{name:20s}: max|J_cpp-J_ref|/|J_ref| = {max_rel:.3e}  "
              f"(J_ref range [{ref.min():.4f}, {ref.max():.4f}] raw 1/A)")

        # also spot-check the derivative against a central finite difference of the REFERENCE
        # (independent of both implementations' analytic-derivative algebra)
        h = 1.0e-5
        fd = (jrab_raw(r_grid + h, i2s, j2s, zi, zj) - jrab_raw(r_grid - h, i2s, j2s, zi, zj)) / (2 * h)
        mine_d = np.array([slater_jraw_cpp_reimpl(r, i2s, j2s, zi, zj)[1] for r in r_grid])
        rel_d = np.abs(mine_d - fd) / np.maximum(np.abs(fd), 1e-30)
        print(f"{'':20s}  dJ/dr vs central-FD(jrab_raw): max rel = {rel_d.max():.3e}")

    print()
    print(f"OVERALL max relative deviation (J, all 3 pairs, r in [0.05,10]): {overall_max_rel:.3e}")
    print("target: < 1e-10 (same quadrature -> only transcription errors would show)")
    print("PASS" if overall_max_rel < 1e-10 else "FAIL")

    # paper checkpoints, sanity (raw*QQRD2E_KCAL should match jrab.py's own printed checkpoints)
    QQRD2E_KCAL = 332.0637
    print()
    print("checkpoint sanity (kcal/mol/e^2, cpp-formula reimpl):")
    j_oo, _ = slater_jraw_cpp_reimpl(0.05, True, True, ZETA_O, ZETA_O)
    j_mh, _ = slater_jraw_cpp_reimpl(0.8735, True, False, ZETA_O, ZETA_H)
    j_hh, _ = slater_jraw_cpp_reimpl(1.5139, False, False, ZETA_H, ZETA_H)
    print(f"  J_OO(r=0.05)     = {j_oo*QQRD2E_KCAL:.2f}  (~371.6 near r->0 closed form)")
    print(f"  J_MH(r=0.8735)   = {j_mh*QQRD2E_KCAL:.2f}  (target 286.44)")
    print(f"  J_HH(r=1.5139)   = {j_hh*QQRD2E_KCAL:.2f}  (target 203.61)")


if __name__ == "__main__":
    main()
