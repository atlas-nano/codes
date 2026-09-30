# TIP4P-FQ, ported without refitting

TIP4P-FQ (Rick, Stuart and Berne, J. Chem. Phys. 101, 6141 (1994)) in samQEq from the model's own parameters.

| deck | what it runs | expected |
|---|---|---|
| `in.gas_slater` | one molecule, exact Slater kernel, free boundaries, 40 Å box, one solve | μ = 1.8611 D (published 1.85 D) |
| `in.tip4pfq_xl_slater_lr1` | 256-molecule liquid, exact Slater kernel, intramolecular pairs shielded and intermolecular pairs bare (`lr_ewald 1`), the structure of the original model | μ = 2.635 ± 0.002 D (published 2.637 D) |
| `in.tip4pfq_xl_slater` | the same liquid with every pair shielded (`lr_ewald 2`) | μ ≈ 2.34 D, about 11% under-polarized |
| `in.tip4pfq_liquid` | the liquid on the Gaussian kernel, radii fitted by `fit_j.py`; the water of the paper's molecules-and-conductor demonstration | stable at 0.2 fs |

The liquid value is the mean molecular dipole over eight runs of `in.tip4pfq_xl_slater_lr1` with different velocity
seeds, each 20 ps of equilibration and 100 ps of production. It is the same at the shipped thermostat coupling
(0.02 ps, 2.6337 ± 0.0015 D) and at 0.1 ps (2.6364 ± 0.0013 D). The shipped deck runs 3.75 ps and lands within
about 0.02 D of it.

The periodic, infinite-dilution gas dipole quoted in the paper (1.8634 D) comes from `examples/kernel_comparison`.

Scripts (numpy):
- `jrab.py`: the Slater-orbital Coulomb integral of the original code.
- `check_slater_table.py`: checks the package's tabulated kernel against `jrab.py` (maximum relative deviation about 5e-15).
- `fit_j.py`: fits the cube-root and Gaussian kernels to the two intramolecular integrals (J_HH at 1.5139 Å, J_MH at
  0.8735 Å) and writes `fit_results.txt`; its Gaussian radii are those of `water4_liquid_gauss.param`.
