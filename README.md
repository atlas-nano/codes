# ATLAS codes

Public research codes from the ATLAS Materials Physics Laboratory
(UC San Diego), one subdirectory per code.

| code | what |
|---|---|
| [`py-xPT/`](py-xPT/) | Python implementation of the 2PT and 3PT (Three-Phase Explicit Anharmonic Thermodynamics) entropy methods: velocity density-of-states partitioning, Mori–Zwanzig memory-kernel cage extraction, hard-sphere/harmonic weighting, Yeh–Hummer finite-size corrections. |
| [`lj-4d-md/`](lj-4d-md/) | Standalone four-dimensional Lennard-Jones MD engine (C/OpenMP + CUDA) used for the cross-dimensional `p = 1/d` cage-prefactor test. |

Each subdirectory is self-contained with its own README, license, and examples.

## Installing py-xPT from this repository

```bash
pip install "git+https://github.com/atlas-nano/codes.git#subdirectory=py-xPT"
```
