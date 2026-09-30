# slab_ewald_sawtooth — validation record (2026-08-23)

Pins the path opened by narrowing the samQEq periodic-direction efield guard: an ATOM-STYLE
sawtooth potential through full Ewald on a z-PERIODIC cell. Same water slab as slab_ewald3dc.

Controls run before freezing the golden:
- A. **Field couples.** max|q(E) − q(0)| = 8.97e-3 e at E = 0.05 V/Å, 8.97e-2 at 0.5, 3.59e-1 at 2.0
  — exactly linear, as a linear-response solve must be. (A first version of this control reported
  0 because it compared charges.dump with itself after the E=0 run overwrote it. Re-run with
  separate files. The code was never wrong.)
- B. **Guard still refuses the ill-defined form.** The same deck with `fix efield 0 0 0.05`
  (constant E_z, z periodic) errors with "Must not have electric field component in direction of
  periodic boundary" — the guard narrowing did not open that.
- C. Tripwire 5 (fix_qeq_base_sam.cpp, temporary) confirmed chi_field = qe2f·efield[i][3] is filled
  on 144 atoms with max 0.196 V, i.e. φ = −E(z − zc) over the slab, before the solve.

Golden: 153-line charges.dump, pe = -232.79514124516 (metal units), np=1, rtol 1e-9.
