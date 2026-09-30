# slab_ewald_sawtooth — validation notes

Pins an ATOM-STYLE sawtooth potential through full Ewald on a z-PERIODIC cell, the one applied-field
form the periodic-direction efield guard admits. Same water slab as slab_ewald3dc.

Controls:
- A. **Field couples.** max|q(E) − q(0)| = 8.97e-3 e at E = 0.05 V/Å, 8.97e-2 at 0.5, 3.59e-1 at 2.0
  — exactly linear, as a linear-response solve must be.
- B. **Guard refuses the ill-defined form.** The same deck with `fix efield 0 0 0.05`
  (constant E_z, z periodic) errors with "Must not have electric field component in direction of
  periodic boundary".
- C. chi_field = qe2f·efield[i][3] is filled on 144 atoms with max 0.196 V, i.e. φ = −E(z − zc) over
  the slab, before the solve.

Golden: 153-line charges.dump, pe = -232.79514124516 (metal units), np=1, rtol 1e-9.
