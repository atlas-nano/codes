# lj-4d-md — a four-dimensional Lennard-Jones MD engine

A standalone molecular-dynamics engine for the **four-dimensional**
Lennard-Jones (12-6) liquid, written for the cross-dimensional test of the
cage-entropy prefactor `p = 1/d` in the Three-Phase Explicit Anharmonic
Thermodynamics (3PT) paper. Standard MD codes hard-wire three Cartesian
components; this engine integrates genuinely 4D kinematics on a four-torus.

**License:** MIT · **Archived on Zenodo:** https://doi.org/10.5281/zenodo.21447750

## Contents

| path | what |
|---|---|
| `engine/md4d.c` | CPU engine (C, OpenMP). LJ(12-6), force-shifted cutoff, NVT Nosé–Hoover chain (MTK, length 3) or NVE, optional Berendsen barostat, Frenkel–Ladd springs, g(r). Reduced LJ units, `DIM = 4` at compile time. |
| `engine/md4d_gpu.cu` | Single-GPU CUDA port (FP64), same CLI and physics; validated against the CPU engine via energy conservation. |
| `examples/00_build.sh` | Build both binaries (`gcc -O3 -fopenmp`; `nvcc` optional — set `CUDA=`). |
| `examples/01_decisive_point.sh` | The decisive strongly caged state ρ\*=1.10, T\*=1.0 (N = 6⁴ = 1296, γ/Ω₀ ≈ 3.2). |
| `examples/02_isotherm.sh` | A T\* isotherm: fluid branch (hypercubic lattice) + D₄-crystal branch. |
| `examples/pyxpt_d4.ini` | py-xPT control template for analyzing a run (`trajectory_format = MD4D`, `dimension = 4`). |

## Physics notes

- Force-shifted LJ cutoff (`u` and `f` continuous, `f(rc) = 0`), so the inverted
  velocity memory kernel is free of cutoff force discontinuities.
- Deterministic (Nosé–Hoover) thermostat on purpose: a stochastic thermostat
  would corrupt the velocity memory kernel that the 3PT analysis inverts.
- Brute-force O(N²) pair loop; robust for N up to a few thousand
  (OpenMP / one CUDA thread per atom). Requires box `L > 2 rc`.
- Initial lattices: hypercubic (`lattice=hc`) or D₄ (`lattice=d4`, the 4D
  close packing). Velocities can be written every step (`velstride=1`) into a
  compact `.vel` binary + `.meta` sidecar.

## Analysis

Entropy analysis (2PT/3PT) of `.vel`/`.meta` output is performed with the
[py-xPT](../py-xPT) package in this repository, which ships the `MD4D`
trajectory reader and dimension-generalized (`dimension = 4`) gas/cage/solid
weights. See `examples/pyxpt_d4.ini`.

## Reference

If you use this engine, please cite the 3PT manuscript (see the repository
root README for the citation).
