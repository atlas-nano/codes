#!/usr/bin/env python3
"""
Python transcription of Rick's `jrab` Slater-orbital Coulomb-overlap integral
(mdtpn.f, subroutine jrab + the jab1/jab2/jab3/johr/jhhr setup in ewsetup()).

Rick, Stuart & Berne, JCP 101, 6141 (1994) TIP4P-FQ: the fluctuating charges
sit on the two H sites (1s Slater, zeta_H) and the M site (uses the OXYGEN
2s Slater form factor, zeta_O, even though M is geometrically offset from O
by r_OM=0.15 A along the bisector -- Rick's code flags this via itype/jtype==4).

jrab returns a RAW value in 1/Angstrom (Coulomb-kernel-like, atomic-units
convention with zeta already converted to 1/Angstrom = zeta[1/bohr]/a0).
Multiply by the Coulomb constant (332.0637 kcal*Angstrom/mol/e^2) to get
J(r) in kcal/mol/e^2, matching the paper's Table I values.

Validated checkpoints (paper / mdtpn.f closed forms):
  J0_OO  = 93/256 * zeta_O / a0 * 332.0637 -> 371.6 kcal/mol/e^2  (2s-2s, r->0)
  J0_HH  = 5/8    * zeta_H / a0 * 332.0637 -> 353.0 kcal/mol/e^2  (1s-1s, r->0)
  J_MH(0.8735) = 286.4 kcal/mol/e^2   (2s(O)-1s(H) at the rigid M-H distance)
  J_HH(1.5139) = 203.6 kcal/mol/e^2   (1s(H)-1s(H) at the TIP4P H-H distance)
  large r: J(r) -> 332.0637/r kcal/mol/e^2  (bare Coulomb, point-charge limit)
"""
import numpy as np

A0 = 0.529177          # Bohr radius, Angstrom/bohr
QQRD2E_KCAL = 332.0637  # kcal*Angstrom/(mol*e^2)  (Rick's/standard CHARMM-style constant)

ZETA_O_BOHR = 1.63      # 2s Slater exponent, oxygen  (bohr^-1)
ZETA_H_BOHR = 0.90      # 1s Slater exponent, hydrogen (bohr^-1)
ZETA_O = ZETA_O_BOHR / A0   # -> 1/Angstrom
ZETA_H = ZETA_H_BOHR / A0


def jrab_raw(r, is_i_2s, is_j_2s, zeta_i, zeta_j, kmax=8.0, nint=1000):
    """Direct transcription of the Fortran jrab() k-space quadrature.
    r, zeta_i, zeta_j all in Angstrom / 1/Angstrom (i.e. zeta already /a0).
    is_i_2s/is_j_2s: True selects the oxygen 2s Slater form factor (Fortran itype==4),
    False selects the plain 1s Slater form factor (Fortran itype!=4).
    Returns the RAW (1/Angstrom) value; caller multiplies by 332.0637 for kcal/mol/e^2.
    """
    r = np.asarray(r, dtype=float)
    dk = kmax / nint
    k = np.arange(1, nint + 1) * dk   # rk = k*dk, k=1..nint (matches Fortran do 100 loop)

    aa1 = 1.0 + (k / (2.0 * zeta_i)) ** 2
    aa2 = 1.0 + (k / (2.0 * zeta_j)) ** 2
    fk1 = 1.0 / aa1 ** 2
    fk2 = 1.0 / aa2 ** 2

    if is_i_2s:
        fa = fk1 - 3.0 * k ** 2 * fk1 ** 1.5 / (4.0 * zeta_i ** 2) + k ** 4 * fk1 ** 2 / (8.0 * zeta_i ** 4)
    else:
        fa = fk1
    if is_j_2s:
        fb = fk2 - 3.0 * k ** 2 * fk2 ** 1.5 / (4.0 * zeta_j ** 2) + k ** 4 * fk2 ** 2 / (8.0 * zeta_j ** 4)
    else:
        fb = fk2

    # scalar-r and array-r both supported
    scalar = (r.ndim == 0)
    rr = np.atleast_1d(r)
    out = np.empty_like(rr)
    for idx, rv in enumerate(rr):
        if rv == 0.0:
            sinc = np.ones_like(k)   # lim sin(kr)/(kr) -> 1
        else:
            sinc = np.sin(k * rv) / (k * rv)
        intgrnd = sinc * fa * fb
        s = 0.5 * dk + np.sum(intgrnd) * dk    # Fortran: sum=0.5*dk (k=0 half-weight); loop adds full-weight dk terms
        out[idx] = s * 2.0 / np.pi
    return out[0] if scalar else out


def J_OO(r):
    """M-M (2s-2s, zeta_O) in kcal/mol/e^2."""
    return QQRD2E_KCAL * jrab_raw(r, True, True, ZETA_O, ZETA_O)


def J_OH(r):
    """M-H (2s(O)-1s(H)) in kcal/mol/e^2 -- Rick's 'johr' pairing."""
    return QQRD2E_KCAL * jrab_raw(r, True, False, ZETA_O, ZETA_H)


def J_HH(r):
    """H-H (1s(H)-1s(H)) in kcal/mol/e^2 -- Rick's 'jhhr' pairing."""
    return QQRD2E_KCAL * jrab_raw(r, False, False, ZETA_H, ZETA_H)


def J0_closed_form():
    """r->0 closed forms used directly in ewsetup() (jab1(0), jab2(0))."""
    j0_oo = QQRD2E_KCAL * (93.0 / 256.0) * ZETA_O
    j0_hh = QQRD2E_KCAL * (5.0 / 8.0) * ZETA_H
    return j0_oo, j0_hh


if __name__ == "__main__":
    j0_oo, j0_hh = J0_closed_form()
    print(f"J0_OO (r->0, closed form)     = {j0_oo:.2f}  (target 371.6)")
    print(f"J0_HH (r->0, closed form)     = {j0_hh:.2f}  (target 353.0)")
    print(f"J_OO(r=0.05) numeric quad     = {J_OO(0.05):.2f}  (sanity vs closed form)")
    print(f"J_HH(r=0.05) numeric quad     = {J_HH(0.05):.2f}  (sanity vs closed form)")
    print(f"J_OH(0.9572)                  = {J_OH(0.9572):.2f}  (target 286.4)")
    print(f"J_HH(1.5139)                  = {J_HH(1.5139):.2f}  (target 203.6)")
    for r in [1.0, 2.0, 4.0, 6.0, 8.0]:
        print(f"large-r check r={r}: J_OO={J_OO(r):.4f}  332.0637/r={QQRD2E_KCAL/r:.4f}")
