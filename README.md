# ATLAS codes

Public research codes from the ATLAS Materials Physics Laboratory
(UC San Diego), one subdirectory per code.

| code | what |
|---|---|
| [`py-xPT/`](py-xPT/) | Python implementation of the 2PT and 3PT (Three-Phase Explicit Anharmonic Thermodynamics) entropy methods: velocity density-of-states partitioning, Mori–Zwanzig memory-kernel cage extraction, hard-sphere/harmonic weighting, Yeh–Hummer finite-size corrections. MIT; Zenodo [10.5281/zenodo.21447746](https://doi.org/10.5281/zenodo.21447746). |
| [`lj-4d-md/`](lj-4d-md/) | Standalone four-dimensional Lennard-Jones MD engine (C/OpenMP + CUDA) used for the cross-dimensional `p = 1/d` cage-prefactor test. MIT; Zenodo [10.5281/zenodo.21447750](https://doi.org/10.5281/zenodo.21447750). |
| [`DMAx/`](DMAx/) | High-throughput workflow for Dynamic Mechanical Analysis (DMA) simulations with LAMMPS. |
| [`2pt-legacy/`](2pt-legacy/) | Original Two-Phase Thermodynamics (2PT) reference implementation (v1.4): the solid+gas hard-sphere density-of-states partition for liquid thermodynamics, with user guide and a LAMMPS example. Superseded by `py-xPT`; retained for reference. Zenodo [10.5281/zenodo.7731073](https://doi.org/10.5281/zenodo.7731073). |

Each subdirectory is self-contained with its own README, license, and examples.

## Installing py-xPT from this repository

```bash
pip install "git+https://github.com/atlas-nano/codes.git#subdirectory=py-xPT"
```

This builds and installs **only** py-xPT; the other codes in this repository are
not pulled into your environment. Append `@3pt-v1` before `#subdirectory` to pin
the release used in the 3PT paper.

## Checking out a single code

To get just one code's source tree (e.g. `py-xPT/`) without the rest:

```bash
git clone --no-checkout --filter=blob:none https://github.com/atlas-nano/codes.git
cd codes && git sparse-checkout set py-xPT && git checkout
```

The repository ships no trajectories and is small, so a plain
`git clone` followed by `cd py-xPT` also works.
