# py-xPT

**Two- and three-phase thermodynamics (2PT / 3PT)** — absolute entropy, free
energy, and heat capacity of a liquid or solid from a *single* equilibrium
molecular-dynamics trajectory, via the velocity density of states (DoS).

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

## Cite
1. Lin, Blanco, Goddard. *J. Chem. Phys.* **2003**, 119, 11792.
2. Lin, Maiti, Goddard. *J. Phys. Chem. B* **2010**, 114, 8191.
3. Pascal, Lin, Goddard. *PCCP* **2011**, 13, 169.
