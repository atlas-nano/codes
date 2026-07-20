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
