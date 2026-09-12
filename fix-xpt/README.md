# FixXPT

**On-the-fly two- and three-phase thermodynamics (1PT / 2PT / 3PT) for
LAMMPS.** A native LAMMPS `fix` that computes absolute entropy *S*,
Helmholtz free energy *A*, internal energy *E*, chemical potential *μ*,
heat capacity *C*ᵥ, zero-point energy, and self-diffusivity *D* directly
from the equilibrium MD velocity trajectory — in memory, with **no
trajectory files and no post-processing step**.

**License:** MIT · **Archived on Zenodo:** v1.0.1
https://doi.org/10.5281/zenodo.22729714 (all versions:
https://doi.org/10.5281/zenodo.21456029) · **Accompanying paper:**
*"On-the-fly Estimation of Quantum-Corrected Thermodynamics in LAMMPS"*
(in preparation). The
stand-alone Python post-processor
[`py-xPT`](https://github.com/atlas-nano/codes/tree/main/py-xPT)
implements the same methods for offline trajectory analysis and is the
reference against which FixXPT is cross-validated.

## What it computes

The velocity autocorrelation function (VACF) is accumulated on the fly
via the Wiener–Khinchin theorem (or an on-line multi-tau correlator),
cosine-transformed to a density of states (DoS), and partitioned:

- **1PT** — all-solid harmonic (Debye–Einstein) baseline.
- **2PT** (Lin–Blanco–Goddard) — the DoS is split into a hard-sphere
  gas and a harmonic solid. Entropy refinements: `rigorous`
  (Sackur–Tetrode + Carnahan–Starling, no `+ln Z`), `lin2003`
  (the standard published `+ln Z` convention), `desjarlais`
  (Gaussian memory-function gas DoS for structured liquids / liquid
  metals), and `r2pt` (reparameterised gas fraction, Sun 2017).
- **3PT** — the rigorous-HS 2PT baseline plus an explicit, parameter-free
  **memory cage**: the non-Markovian excess of the Mori–Zwanzig friction
  kernel over its Markovian (Lorentzian) counterpart, applied to the
  translational and (for molecules) rotational channels. The cage
  construction and its validation are developed in Buarque, Gascon &
  Pascal, *"An anharmonic liquid-entropy functional from the
  Mori–Zwanzig memory kernel"* (J. Chem. Phys., in review); FixXPT
  implements the `refinement none` variant of that model.

Molecular systems are decomposed into translational, rotational, and
vibrational channels (per-frame inertia-tensor diagonalisation). All
eight LAMMPS unit systems are supported.

## Build

FixXPT is a self-contained LAMMPS package (`XPT`); it needs only LAMMPS's
built-in FFT support (FFTW3, MKL, or the bundled KISS FFT). It does **not
build LAMMPS for you** — copy the sources into your LAMMPS tree and
rebuild with your usual toolchain.

The six core files form the `XPT` package. The two KOKKOS files
(`fix_xpt_kokkos.*`, the `xpt/kk` styles) follow the LAMMPS convention for
accelerator variants and go in the `KOKKOS` package directory, so a build
without KOKKOS never compiles them:

```bash
mkdir -p /path/to/lammps/src/XPT
cp src/fix_xpt.h src/fix_xpt.cpp src/fix_xpt_accumulate.cpp src/fix_xpt_analysis.cpp \
   src/fix_xpt_entropy.cpp src/fix_xpt_const.h /path/to/lammps/src/XPT/
cp src/fix_xpt_kokkos.h src/fix_xpt_kokkos.cpp /path/to/lammps/src/KOKKOS/   # xpt/kk (optional)
cp doc/fix_xpt.rst /path/to/lammps/doc/src/                                     # docs (optional)
```

**With CMake:** LAMMPS's CMake builds only the packages it lists, so add `XPT`
to the `set(STANDARD_PACKAGES ...)` list in `cmake/CMakeLists.txt` once. Then

```bash
cd /path/to/lammps/build
cmake ../cmake -D PKG_XPT=yes [ -D PKG_KOKKOS=yes ... ]
cmake --build . -j
```

With `PKG_KOKKOS=yes`, CMake compiles `src/KOKKOS/fix_xpt_kokkos.cpp` and
registers `xpt/kk`, `xpt/kk/device` and `xpt/kk/host`.

**With `make`:**

```bash
cd /path/to/lammps/src
make yes-xpt           # copies the XPT package into src/
make mpi               # or your usual target
```

For the `xpt/kk` styles in a `make` build, install KOKKOS as usual and copy
`fix_xpt_kokkos.h` and `fix_xpt_kokkos.cpp` into `src/` as well.

## Usage

```
fix ID group-ID xpt Nevery Nframes prefix [keyword value ...]
```

Sample velocities every `Nevery` steps; complete one analysis window
every `Nframes` samples; write results to `prefix.thermo`, `prefix.pwr`
(DoS) and `prefix.vac` (VACF), and to the LAMMPS thermo vector
(`f_ID[n]`). Common keywords:

| keyword | meaning |
|---|---|
| `mode 1PT\|2PT\|3PT` | phase partition (default `2PT`) |
| `refinement rigorous\|lin2003\|desjarlais\|r2pt` | 2PT sub-method (default `rigorous`) |
| `molecule` | translational/rotational/vibrational decomposition |
| `symmetry <Schoenflies>` | point group → rotational symmetry number (e.g. `C2v`, `D6h`) |
| `classical` | also output classical (non-quantum) thermodynamics |
| `normalize` | per-atom / per-molecule output |
| `correlator fft\|multitau` | VACF algorithm (default `fft`) |
| `buffer_layout distributed\|replicated` | where the frame history lives (default `distributed`: one home rank per atom, ~1/P of the history per rank). On a single rank an atomic group selects `replicated`, which is faster there |
| `epsilon`/`sigma`/`mass` | physical LJ parameters for ħ\* scaling in `lj` units |
| `volume v_name` | override the cell volume (e.g. slab geometries) |

Minimal example (liquid argon, real units, monoatomic):

```lammps
fix mom all momentum 100 linear 1 1 1          # see "Protocol" below
fix xpt all xpt 5 512 ar_real classical normalize
```

Molecular example (SPC/E water):

```lammps
fix xpt all xpt 2 5000 water classical normalize molecule symmetry C2v mode 2PT
```

See `examples/` for complete, runnable decks: liquid argon (real and LJ
units, and with the Desjarlais refinement), SPC/E water (single-line and
INI-file syntax), liquid benzene, a dynamic-group solvation example, a
liquid-sodium melt, a six-analysis shared-buffer sweep, and an
`electron`-units check whose ideal-gas
construction makes the reported `T_vac` reproduce the imposed temperature
exactly, so any error in the unit conversions is visible in one column.

## Sharing one velocity buffer across fixes

The velocity buffer is the dominant memory cost, and comparing analyses of the
same trajectory is the common case. Declaring one fix per setting would
ordinarily allocate one complete velocity history per setting.

It does not here. A fix checks, at the start of each `run`, whether an earlier
`fix xpt` is accumulating the same trajectory, and reuses its buffer if so. The
configurations must agree on

    group · Nframes · Nevery · correlator · buffer_precision · buffer_layout · molecule on/off

`mode` and `refinement` are deliberately *not* part of that test, because both
act on the density of states rather than on the trajectory. So this

```lammps
fix s1 all xpt 5 4096 ar.1pt   mode 1PT
fix s2 all xpt 5 4096 ar.rig   mode 2PT refinement rigorous
fix s3 all xpt 5 4096 ar.lin   mode 2PT refinement lin2003
fix s4 all xpt 5 4096 ar.des   mode 2PT refinement desjarlais
fix s5 all xpt 5 4096 ar.r2pt  mode 2PT refinement r2pt
fix s6 all xpt 5 4096 ar.3pt   mode 3PT
```

costs one velocity history, not six. `s1` owns the buffer (and, with the default
distributed layout, its home layout); `s2`–`s6` re-point at it and skip frame
accumulation, so they add no memory, no per-frame staging and no per-frame
communication. Each still runs its own window-end analysis, which
is the part that actually differs. Sharing is reported in the log:

    FixXPT::s2-all: sharing velocity buffer with FixXPT::s1-all (nevery=5, nframes=4096, FFT, do_molecule=0)

One restriction: sharing applies to the FFT correlator only. The multi-tau ring
buffers are per-fix, so a multi-tau consumer would reach the end of its window
with no autocorrelation of its own; multi-tau fixes always own their buffers.

See `examples/in.2pt_shared_buffer`.

## Protocol — two required deck settings

FixXPT infers thermodynamics from the raw group velocities, so two
conventions must be set by the user:

1. **Remove center-of-mass drift** (`fix momentum 100 linear 1 1 1`).
   FixXPT has no internal mechanism to subtract group-COM motion from
   VACF(0); residual drift over long trajectories inflates *D* by orders
   of magnitude.
2. **For small flexible groups, match the thermostat's degree-of-freedom
   convention.** FixXPT uses `DOF = dim·N − N_constraint` (the COM
   dimension is *not* subtracted); LAMMPS's default `compute temp`
   subtracts `dim`. The difference is negligible for *N* ≳ 100 but
   reaches ~9% for an ~10-atom solute. Pair the thermostat with a
   `compute temp` modified by `compute_modify … extra/dof 0`.

## Test run

CPC's reviewers should start here. The designated test case is liquid argon in
LJ reduced units, which exercises the full pipeline (velocity accumulation, the
Wiener--Khinchin VACF, the DoS, the gas/solid partition, the fluidicity solve
and the thermodynamic integration) in about a minute on one core.

```
cd examples
lmp -in in.2pt_argon_lj
```

It writes three files, all of which are included here as reference output:

| file | contents |
|---|---|
| `ar_lj_ar.thermo` | one row per completed analysis window: S*, A*, E*, Cv*, D*, f, T*, … |
| `ar_lj_ar.pwr`  | the density of states, and its gas and solid components |
| `ar_lj_ar.vac`  | the normalised velocity autocorrelation function |

Twenty consecutive 512-frame windows are analysed. The run reproduces the
reference to within the per-window statistical scatter; the window-to-window
spread on `S*` is itself a few times larger than any platform difference, so
compare window averages rather than individual rows. Representative values for
this state point (rho* = 0.85, T* ~ 0.84) are S* ~ 7.3 kB/atom, f ~ 0.43 and
D* ~ 7.9e-2 sigma^2/tau.

The SPC/Ew water case (`in.spcew_water_ini`, reference `spcew_water_ini.thermo`)
exercises the molecular pipeline: translational, rotational and vibrational
channels, rigid-water constraints, and the INI-file input syntax.

> The reference outputs were generated with the code as shipped here. Absolute
> values depend on the FFT backend only at the round-off level, but the
> `refinement` default is `rigorous`, so entropies from older releases that
> defaulted to the Lin-2003 `+ln Z` convention sit about 0.5 kB/atom higher.

## Output files

- `prefix.thermo` — one row per analysis window: `S_q A_q E_q Cv_q D f
  T_vac Natom V P_avg E_md [classical cols] [molecular cols] ZPE_q μ_q μ_c`.
  A header `# Units:` line documents the per-style output units.
- `prefix.pwr` — the density of states (gas / solid / cage columns).
- `prefix.vac` — the velocity autocorrelation function.

## Changes

- **1.0.1**
  - **Distributed frame buffers** (`buffer_layout distributed`, the new default): every group atom
    has one home rank per window and each frame travels home in one `MPI_Alltoallv`, so a rank stores
    about 1/P of the history instead of all of it. Node memory for a 5184-atom water window on 128
    ranks falls from 309 GB to 15 GB, and the fix's cost now falls with the rank and GPU count.
    `buffer_layout replicated` keeps the 1.0.0 behaviour and is selected automatically on a single
    rank for atomic groups, where it is faster.
  - **Restart format v2**: multi-rank restarts store global sums, so a `run 0` reanalysis reproduces
    the window it restarted from on any rank count. A v1 file is refused on more than one rank.
  - `xpt/kk` no longer crashes on monatomic systems (the push buffers were never allocated when
    `nmol_group == 0`).
  - `xpt/kk` subgroup `E_md` and `Cv` were computed from stale host velocities; the host copy is now
    synced before the energy tally. VACF-derived quantities were unaffected.
  - `run 0` restart reanalysis filled no masses, so a restarted monatomic analysis returned
    fluidicity 1.0 and D = 0.
  - Multi-tau: the first window dropped frame 0 of the molecular streams, a dynamic group dropped
    frame 0 on every window, and a partial final window was transformed at the wrong FFT length.
  - Atom styles with per-atom masses (`rmass`) are refused at init rather than segfaulting.
  - The build recipe in this README failed on a stock LAMMPS tree: the Kokkos sources belong in
    `src/KOKKOS/`, and `XPT` must be listed in `STANDARD_PACKAGES`.

## Citing

If you use FixXPT, please cite this release
([10.5281/zenodo.22729714](https://doi.org/10.5281/zenodo.22729714), all versions:
[10.5281/zenodo.21456029](https://doi.org/10.5281/zenodo.21456029)), the
accompanying LAMMPS-implementation paper (in preparation), and — for the
3PT cage — Buarque, Gascon & Pascal, *J. Chem. Phys.* (in review, 2026).
The `py-xPT` post-processor is archived at
[10.5281/zenodo.21447746](https://doi.org/10.5281/zenodo.21447746).

Part of the [ATLAS codes](https://github.com/atlas-nano/codes)
collection, ATLAS Materials Physics Laboratory, UC San Diego.
