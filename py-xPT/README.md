# py-xPT

**Two- and three-phase thermodynamics (2PT / 3PT)** — absolute entropy, free
energy, and heat capacity of a liquid or solid from a *single* equilibrium
molecular-dynamics trajectory, via the velocity density of states (DoS).

**License:** MIT · **Archived on Zenodo:** v2.1.0 https://doi.org/10.5281/zenodo.23374428 (all versions: https://doi.org/10.5281/zenodo.21447745)

This is the reference implementation accompanying the manuscript *"An anharmonic
liquid-entropy functional from the Mori–Zwanzig memory kernel."* It computes:

- **2PT** (Lin–Blanco–Goddard): partition the DoS into a hard-sphere gas and a
  harmonic solid. Refinements: `rigorous` (Sackur–Tetrode + Carnahan–Starling,
  no `+ln Z`), `lin2003` (empirical `+ln Z`), `desjarlais` (memory-function gas
  DoS), `r2pt` (reparameterised gas fraction).
- **3PT**: the rigorous-HS 2PT baseline plus an explicit **memory cage** —
  the non-Markovian excess of the Mori–Zwanzig friction kernel over its
  Markovian (Lorentzian) counterpart, weighted as a bounded hard-sphere fluid
  with a dimension-only prefactor `1/d`. Applied to both the translational and
  (for molecular liquids) rotational channels.

## Install

```bash
pip install -e .            # editable, from the repo root
# or run in place:
export PYTHONPATH=$PWD/src
```
Requires Python ≥ 3.10, numpy, scipy, MDAnalysis.

## Quick start

```bash
pyxpt control.ini           # writes <prefix>.thermo, .pwr, .vac, .3n, .out.log
```

or from Python:

```python
from pyxpt import run
result = run("control.ini")
```

## Minimal control file

Monatomic Lennard-Jones (3PT), reduced-unit output:

```ini
[files]
trajectory        = lj.lammpstrj
trajectory_format = LAMMPSDUMP
[system_properties]
timestep    = 0.008          # ps between dumped frames
lammps_units = real
[thermodynamics]
mode      = 3PT              # 2PT | 3PT
molecular = false            # true -> trans/rot/vib decomposition (needs topology)
hs_eos    = cs               # Carnahan-Starling
[output]
prefix    = lj
out_units = lj               # report in LJ reduced units
lj_sigma  = 3.405
lj_epsilon = 0.2381
lj_mass   = 39.948
```

Molecular water (3PT, translational + rotational cage) needs a topology and
molecule metadata:

```ini
[files]
trajectory        = water.lammpstrj
trajectory_format = LAMMPSDUMP
topology          = data.water
topology_format   = LAMMPS
[system_properties]
timestep    = 0.002
lammps_units = real
temperature = 298.0
volume      = 53646.6
[thermodynamics]
mode       = 3PT
molecular  = true
hs_eos     = cs
mol_rotsym = 2
constraints = 5184
[output]
prefix    = water
normalize = 1
```

### Modes and refinements
- `mode = 2PT` + `refinement = rigorous|lin2003|desjarlais|r2pt`
- `mode = 3PT` — the bare memory cage on top of rigorous-HS 2PT (the published
  3PT). `refinement` must be `none`.
- `cage_ws = consistent|legacy` (`[thermodynamics]`, default `consistent`) — the harmonic
  weight subtracted in the cage reweighting. `consistent` gives the quantum entropy the
  correction computed with the quantum harmonic weight and the classical entropy the one
  computed with the classical weight; `legacy` applies the classical weight to both, as
  releases up to 1.0.1 did.
- Cage numerics (`[thermodynamics]`): `cage_quadrature = trapezoid|rectangle` (default
  `trapezoid`, second order in the frame spacing); `cage_nf_run` (default 3) and
  `cage_nf_run_rot` (default 1), the number of consecutive lags below the noise floor that end
  the translational and the rotational kernel. The translational cage's gas weight is the 2PT
  baseline's own gas weight, at the baseline's fluidicity and packing fraction. A cage entropy
  that is negative, non-finite, or larger than 3 k_B per degree of freedom is reported and set
  to zero. The kernel must resolve the Einstein period: keep the frame spacing at
  dt * Omega_0 <= 0.066 or below.

### Dimension (experimental)
`dimension = 1|2|4` (`[thermodynamics]`, default 3) evaluates the fluidicity closure, the
hard-sphere reference (hard rods, hard disks) and the gas weight in d dimensions. The velocity
components come from the trajectory: a slit or channel keeps all three (the confined axis enters
the DoS as a vibrational channel of the d-dimensional gas), while a strictly two-dimensional run
(two-column data, or v_z ≡ 0) keeps two. Unless the trajectory itself is d-dimensional, set
`volume` in `[system_properties]` to the d-dimensional measure (Å^d, e.g. the in-plane area of a
slit). **This branch is experimental and has not been validated.** A dilute two-dimensional
Lennard-Jones fluid misses same-Hamiltonian thermodynamic integration by 0.2–0.25 k_B per atom,
whereas the three-dimensional branch agrees to within 0.04 k_B in the same test. Treat
entropies at d ≠ 3 as exploratory; py-xPT logs a warning whenever `dimension ≠ 3`.

## Outputs
- `<prefix>.thermo` — entropy (`S_q`), cage entropy (`S_cage`), free energy
  (`A_q`), chemical potential (`μ_q`), heat capacity (`Cv_q`), fluidicity, per
  channel (and per group for mixtures / molecular channels).
- `<prefix>.pwr` — the gas | cage | solid DoS decomposition.
- `<prefix>.vac`, `.3n` — velocity autocorrelation and cumulative DoS integral.

## Tests
```bash
pytest                       # end-to-end regression on a bundled mini LJ trajectory
```

## Changes
- **2.1.0** — the Volterra kernel inversion now evaluates its convolution with trapezoidal
  product-integration weights (`cage_quadrature = trapezoid`, the new default) instead of unit
  weights. The previous rule was first order in the frame spacing and returned a friction that
  does not satisfy the Green-Kubo identity gamma * int c dt = 1: on a set of Lennard-Jones
  liquids sampled every 8 fs the product is 1.37 in the median, and on liquid metals sampled
  every 10 fs it is 2.0. The trapezoidal rule restores the identity where the correlation
  function is resolved (1.01 in the median for those Lennard-Jones states) and reduces the
  error in gamma from 14% to 0.5% on a single-exponential model kernel. **Reported entropies
  change** (see below for the bundled example). Where the
  frame spacing is too coarse (dt * Omega_0 above about 0.1) neither rule is converged, and the
  fix does not repair that; check convergence against the frame spacing.
  `cage_quadrature = rectangle` reproduces the results of 2.0.1 and earlier.
  The translational cage now takes the 2PT baseline's own gas weight W_g (Sackur-Tetrode on the
  integrated gas, hard-sphere excess at the baseline's packing fraction) instead of a weight
  built from a separate packing of the cage's integrated fluidicity; the two disagree in dense
  liquids, where the separate packing approached close packing and could make the cage
  negative. This raises the translational cage by about 30% in liquid metals and dense
  Lennard-Jones liquids and leaves the rotational cage unchanged. A cage entropy that is
  negative, non-finite, or larger than 3 k_B per degree of freedom is now reported and set to
  zero. The kernel-truncation run length is set per channel (`cage_nf_run`, default 3, for
  translation; `cage_nf_run_rot`, default 1, for rotation): requiring three consecutive lags
  below the noise floor keeps an isolated node of an oscillatory correlation function from
  ending the translational kernel. The cage term of the bundled example is 0.216 k_B/atom.
- **2.0.1** — fixes to the `dimension ≠ 3` branch, which is now marked experimental (see
  *Dimension*). Results at `dimension = 3` are bit-identical to 2.0.0.
  - `dimension = 2` or `1` could not read a three-component trajectory such as a LAMMPS dump. The
    number of velocity components now comes from the trajectory instead of from `dimension`.
  - Slit and channel mode: the hard-disk packing fraction came out at 2/3 of its value, because the
    gas fraction was taken against all three velocity axes. It now uses the gas number fraction.
  - The low-memory batched path crashed for `dimension ≠ 3`.
  - The Enskog parameter K used 2/(3d), which is right only if the three-dimensional
    Chapman–Enskog prefactor 3/8 held in every dimension. It now uses 1/(4 d c_d) with
    c_d = d Γ(d/2)/(4 π^((d−2)/2)) (c_2 = 1/2 for hard disks), so K is 3/4 of its former value at
    d = 2 and 1.18 times it at d = 4.
  - The reported self-diffusivity used D = S(0) k_BT/(12 m N) in every dimension. It now uses
    S(0) k_BT/(4 d m N), which is 3/2 of the former value at d = 2.
  - `constraints` placed under `[system_properties]` was silently ignored. py-xPT now warns
    (the key is read only from `[thermodynamics]`).
- **2.0.0** — the 3PT cage correction subtracts the harmonic weight that matches the
  entropy it corrects: the quantum weight for `S_q`/`A_q`/`μ_q` and the classical weight for
  the classical quantities (`cage_ws = consistent`, the new default). Releases up to 1.0.1
  subtracted the classical weight for both, which overstates the quantum cage entropy where the
  cage band lies at high frequency relative to k_BT. For Lennard-Jones liquids and liquid metals
  the change is below 0.01 k_B per atom; for the rotational cage of liquid water at 298 K it is
  about −0.8 J mol⁻¹ K⁻¹. `cage_ws = legacy` reproduces the 1.0.1 values. Also fixed: the
  `S_cage` row of the `.thermo` file was not converted for `out_units = lj`, `kcal/mol` or
  `eV`, and was reported in J mol⁻¹ K⁻¹ under those unit labels.
- **1.0.1** — linear molecules: the rotational temperature of the axial slot now uses the
  perpendicular moment. The axial moment of a linear molecule is zero to roundoff, and in
  1.0.0 the sign of that roundoff decided whether the rotational gas term diverged or was
  skipped, so the rotational entropy of linear species (e.g. OH⁻) varied from seed to seed.
  Nonlinear and monatomic results are unchanged.

## Cite
1. Lin, Blanco, Goddard. *J. Chem. Phys.* **2003**, 119, 11792.
2. Lin, Maiti, Goddard. *J. Phys. Chem. B* **2010**, 114, 8191.
3. Pascal, Lin, Goddard. *PCCP* **2011**, 13, 169.
