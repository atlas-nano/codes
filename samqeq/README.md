# samQEq

**samQEq** is a configurable fluctuating-charge (charge-equilibration) package for LAMMPS. The choices
that usually distinguish one charge-equilibration code from another, namely the shielding kernel, the
long-range treatment and the charge constraint, are run-time settings of one energy functional and one
matrix-free solver. The package provides `fix qeq/sam`, the companion pair style `coul/shield/intra`, the
per-atom reciprocal-space solver `kspace_style pppm/samqeq`, `compute dipole/samqeq`, and
Kokkos device versions of the fix, pair and kspace styles.

## Contents

- [What the package solves](#what-the-package-solves)
- [Build](#build)
- [Usage](#usage)
  - [Parameter file](#parameter-file)
  - [Units convention](#units-convention)
  - [Molecule-optional solve](#molecule-optional-solve)
- [Long-range routes](#long-range-routes)
- [Shielding kernels](#shielding-kernels)
- [Conductors](#conductors)
- [Solver and run-time controls](#solver-and-run-time-controls)
- [Coupling to other potentials](#coupling-to-other-potentials)
- [Observables and free energies](#observables-and-free-energies)
- [GPU (Kokkos)](#gpu-kokkos)
- [OpenMP threading](#openmp-threading)
- [Verification](#verification)
- [Limitations](#limitations)
- [License](#license)

Step-by-step guides are in [`doc/`](doc/README.md).

## What the package solves

At every solve the fix minimizes the second-order charge-equilibration energy

    E(q) = Σ_i (χ_i q_i + ½ η_i q_i²) + ½ Σ_{i≠j} J_ij(r_ij) q_i q_j

subject to a neutrality constraint, where χ and η are the per-type electronegativity and hardness and
`J_ij` is a shielded Coulomb kernel that tends to `1/r` at large separation. Charges are written as
`q = q⁰ + Δ`, so an ion keeps its formal charge `q⁰` and only the response `Δ` is solved for.

- **Kernel.** Three shielding kernels (cube-root, Gaussian overlap and the exact Slater-orbital Coulomb
  integral) are selectable at run time, and the pair style mirrors the choice so that forces and the charge
  solve use one kernel.
- **Long range.** The reciprocal-space contribution is evaluated inside the matrix-vector product, so a
  single projected Krylov solve returns fully long-range charges with no self-consistent outer loop.
- **Constraint.** An orthogonal projector imposes one neutrality constraint per fragment (molecule ID), or a
  single global constraint when the system has no molecule IDs.
- **Conductors.** The finite-width self-energy of a Gaussian charge can be added to the on-site hardness of
  designated types, which keeps the operator positive definite in the metallic limit.
- **Dynamics.** Charges are minimized each step (Born–Oppenheimer), predicted and corrected (ASPC), or
  propagated by an extended Lagrangian.

## Build

The package builds inside a LAMMPS source tree with CMake. It is tested against LAMMPS version 2 Sep 2026
(the `develop` branch at commit `d71abe6102`). It requires the **KSPACE, MOLECULE, RIGID and
EXTRA-FIX** packages; the device styles additionally require **KOKKOS**. The ACKS2/QEq solver base
and the per-atom reciprocal routine ship in-package, so neither REAXFF nor ELECTRODE is needed.

Three edits to the host tree are required, and all three ship as patches in `patches/`; apply each with
`patch -p1` from the LAMMPS root:

| patch | purpose |
|---|---|
| `cmake_register_samqeq.patch` | registers `SAMQEQ` in `cmake/CMakeLists.txt` (the `STANDARD_PACKAGES` list and `pkg_depends(SAMQEQ KSPACE)`) |
| `kspace_pppm_peratom_brick_zeroing.patch` | zeroes the per-atom particle–mesh accumulators in `PPPM::allocate_peratom()`; without it a per-atom energy can be read from uninitialized memory |
| `fix_efield-friend.patch` | a friend declaration in `src/fix_efield.h`, so the fix can read an applied field |

Without the registration CMake completes without error and produces a LAMMPS binary that carries none of
the package's styles, so check every new build (step 5 below).

```bash
cp -r src <lammps>/src/SAMQEQ # 1. the package sources
cp kokkos/* <lammps>/src/SAMQEQ # only for a Kokkos build
cd <lammps> && for p in <this-release>/patches/*.patch; do patch -p1 < $p; done # 2. host-tree edits
mkdir build && cd build # 3. configure
cmake ../cmake -D PKG_SAMQEQ=on -D PKG_KSPACE=on -D PKG_MOLECULE=on -D PKG_RIGID=on \
      -D PKG_EXTRA-FIX=on # + -D PKG_KOKKOS=on ... for GPU, -D PKG_REAXFF=on for examples/reaxff_rdx
make -j # 4. build
./lmp -h | grep -c samqeq # 5. smoke test: a positive count means the styles are registered
LMP=$PWD/lmp bash <this-release>/tests/run_tests.sh # 6. test suite: expect failed=0
```

The `kokkos/` sources are kept apart because CMake compiles every `.cpp` in the package directory: copied
into a build without KOKKOS, they fail to compile. A LAMMPS version bump can silently drop a host-tree patch,
so re-apply all three after updating LAMMPS; the patches carry context and apply to nearby versions, but
only the version above is tested.

## Usage

```lammps
units metal
atom_style full # molecule IDs define the fragments
fix chg all qeq/sam <Nevery> <cutlo> <cuthi> <tol> <paramfile>
```

The command layout follows `fix qeq/reaxff`. Optional trailing keywords on the fix command:

| keyword | effect |
|---|---|
| `maxiter <N>` | Krylov iteration cap (default 1000). |
| `nowarn` | Suppress the non-convergence warning. It hides the one signal that the charges are not converged, so use it only when the cause is understood. |

### Parameter file

```
gamma_align kappa_bond r_ov [r_loc] [lr_alpha] [lr_ewald] [lr_ridge]
<type> <chi> <eta> <gamma> <q0> <eHOMO> <eLUMO> [<c3> <c4>] # one line per atom type
```

- `chi`, `eta`: electronegativity and hardness. `eta` must exceed the Coulomb coupling at bonding distance.
- `gamma`: the fourth column is the kernel parameter (γ for `cbrt`, the Gaussian radius `Rc` for
  `gaussian`, the Slater exponent ζ for `slater`).
- `q0`: reference charge.
- `eHOMO`, `eLUMO`: read and ignored; supply placeholders such as `-7 3`.
- `c3`, `c4` (optional, default 0): cubic and quartic on-site coefficients, adding `c3 q³ + c4 q⁴` to the
  site energy. A nonzero value turns on the nonlinear solve described under `quartic` below.
- `gamma_align`: read and required to be positive; no solve path uses it.
- `kappa_bond`, `r_ov`: amplitude and length of the intra-fragment bond softness `κ·exp(−r²/r_ov²)`.
- `r_loc`: read and unused; any value is accepted.
- `lr_alpha`: damping α of the damped-shifted-force route (0 selects the short-range route).
- `lr_ewald`: long-range route, 0, 1 or 2 (see [below](#long-range-routes)). Any other value is refused.
- `lr_ridge`: Tikhonov diagonal hardness (eV/e²), 0 = off.

### Units convention

`fix qeq/sam` supports `units metal` (eV) and `units real` (kcal/mol); any other unit style is refused.

- **Everything typed in the deck is in the deck's own energy units.** Every parameter-file column and every
  numeric `fix_modify` argument is read as eV under `units metal` and as kcal/mol under `units real`. The same
  physical model under `units real` needs its energy-dimensioned numbers scaled by 23.060549; charges are
  unchanged.
- **Internal constants are anchored in eV** and converted once, so they have the same physical meaning in
  both unit systems.
- **Solver tolerances mean the same physical convergence in both unit systems** (the residual criteria are
  rescaled internally).

`fix efield` coupling works in both `metal` and `real`. Log lines that report an energy print eV regardless of
the unit style.

### Molecule-optional solve

Molecule IDs are not required. Under `atom_style charge` every atom in the group belongs to one fragment, and
the projector enforces global charge neutrality, the standard QEq constraint for a bare solid. The Kokkos fix
refuses this case; use `fix qeq/sam`. Test case: `tests/cases/charge_solid_global`.

## Long-range routes

`lr_ewald` selects the route. The real-space part of the Ewald split is assembled with the short-range
operator, and the reciprocal per-atom potential comes from `kspace_style pppm/samqeq`, evaluated inside the
matrix-vector product. The solve is a projected conjugate gradient in the per-fragment-neutral subspace.

| `lr_ewald` | route |
|---|---|
| `0` | short-range (tapered) or, with `lr_alpha > 0`, damped-shifted force |
| `1` | intramolecular shielding only; intermolecular pairs are bare `1/r` |
| `2` | all pairs shielded, `J(r) → 1/r` at long range; stable, energy-conserving NVE |

**Which one to pick.** All-pair shielding (`lr_ewald 2`) is the native samQEq choice for a model parameterized
under it. Published fluctuating-charge models with bare intermolecular Coulomb shield only the fixed
intramolecular geometry, and porting one faithfully (its own raw parameters, no refit) needs `lr_ewald 1`. With
the exact Slater kernel and Rick's raw exponents, `lr_ewald 1` reproduces TIP4P-FQ's gas dipole (1.8611 D
against 1.85 D) and liquid dipole (2.635 ± 0.002 D over eight 100 ps runs, against 2.637 D), while `lr_ewald 2` under the same kernel
under-polarizes the liquid by about 11% (2.340 D). The difference is structural (which pairs are shielded), not a
kernel-fit residual. `lr_ewald 1` runs closer to criticality and needs NVT.

Force consistency uses `coul/shield/intra` overlaid on `coul/long`:

```lammps
pair_style hybrid/overlay lj/cut 10.0 coul/long 10.0 coul/shield/intra 5.0 all
pair_coeff * * coul/long
pair_coeff * * coul/shield/intra <gamma>
kspace_style pppm/samqeq 1.0e-6
```

The pair scope (`intra` or `all`) must match `lr_ewald` (`intra` for 1, `all` for 2); the fix checks this at
initialization and refuses a mismatch, because the charges are identical either way and only the forces move.
Under `hybrid/overlay`, give the `coul/shield/intra` coefficients as `pair_coeff * *` or one line per type
pair: LAMMPS does not mix a sub-style across type pairs that another sub-style (here `coul/long`) already
covers, so per-type lines alone leave the cross pairs out of the forces. The fix refuses such a deck and
names the missing pairs.
The fix takes the Ewald split from the kspace style, so `kspace_modify gewald` is optional.

## Shielding kernels

`fix_modify <id> shield cbrt|gaussian [<lambda>]|slater [2s <type>...]`, mirrored by
`pair_style coul/shield/intra <cut> [intra|all] [cbrt|gaussian|slater] [2s <type>...]`:

| mode | `J(r)` | fourth column | notes |
|---|---|---|---|
| `cbrt` (default) | `1/∛(r³ + 1/γ³)` | γ | the native samQEq kernel |
| `gaussian` | `erf(α_ij r)/r` | Gaussian radius `Rc` | PQEq-form Gaussian overlap, optional `λ` |
| `slater` | exact Slater-orbital Coulomb integral | exponent ζ (1/Å) | published Slater-overlap models port with no refit; `2s <type>...` marks types with the 2s form factor (default 1s) |

All three tend to `1/r` at large `r` and are unit-correct in both `metal` and `real`. The tabulated Slater
kernel agrees with an independent reference quadrature to a maximum relative deviation of 5.2×10⁻¹⁵
(`doc/02_long_range_and_solvers.md`). `shield slater` with `lr_alpha > 0` is refused.

- **`fix_modify <id> iondamp <typeI> <typeJ> <b> [<n>]`** applies Tang–Toennies damping `f_n(b·r)` on top of the
  active kernel for designated ion pairs (mirrored by the pair keyword `iondamp`), quenching contact
  over-transfer while leaving the long-range complement untouched. Supported on `lr_ewald 2` and the
  short-range route; refused on `lr_ewald 1`, damped-shifted force and the Kokkos fix.
- **`fix_modify <id> shieldpair <ti> <tj> <R_ij>`** sets a per-type-pair Gaussian radius in place of the
  combination rule. The fix pushes the table into `coul/shield/intra` so that forces and solve share one
  kernel. Not available on the Kokkos kernels.

## Conductors

`fix_modify <id> gself on|off [width <Å>] [types <t1> ...]` adds the finite-width self-energy
`K_e/(√π w)` of a Gaussian charge of width `w` to the on-site hardness of the listed types (default: all
types, `w` = 0.5 Å). Widths above about 0.75 Å let the layer charges of a slab alternate in sign, so keep
`w` at or below 0.5 Å. Without it, the operator of a metallic slab becomes
indefinite once η falls below the contact value of its own kernel (about 3.5 eV for the Au(111) slab of the paper); with it, the smallest eigenvalue stays positive.

## Solver and run-time controls

| `fix_modify <id> ...` | effect |
|---|---|
| `energy yes` | Count the on-site self-energy `Σ(χ_i q_i + ½ η_i q_i²)` in the potential energy (force-free). |
| `eselfref qs` \| `q` | Basis of the reported self-energy (reporting only). |
| `xl <q_mass> <q_tdamp>` [`<T_q>`] [`<seed>`] [`masswt`] | Extended-Lagrangian charge dynamics (velocity-Verlet charges, `lr_ewald > 0`). On the same benchmark in NVE the nuclear temperature falls from 320 to 69 K in 10 ps; use it under a thermostat. `T_q` adds a fluctuation–dissipation-balanced Langevin charge thermostat; `masswt` sets `m_i = m0·eta[type]` for a uniform fictitious frequency. |
| `aspc <n_corr>` [`order`] \| `rtol <v>` \| `off` | Always-stable predictor–corrector (`lr_ewald 2`). A corrected step is accepted only if it is bounded and its relative residual is below `rtol` (default 0.01); otherwise the step falls back to the exact solve. The acceptance test does not bound energy conservation: on the 64-molecule water benchmark in NVE, `aspc 2` needs 2 matvecs per step against 18 for the exact solve, accepts every step at residuals near 5×10⁻⁴, and cools the system by about 1200 K/ns. Use it under a thermostat. |
| `spikeguard <q_spike> <k_spike>` \| `off` | Soft restoring force on any charge above `q_spike` under XL (default on; inactive in normal dynamics). |
| `ridge eig <floor>` [`<m>`] [`<every>`] [`<delta>`] | Near-critical regularizer: an `m`-step Lanczos estimate of the smallest eigenvalue sets a Tikhonov ridge so that `λ_min + ridge ≥ floor`. The Lanczos value is an upper bound, so use `m ≥ 60` on interface cells; `<delta>` budgets the estimate error. Other modes: `ridge <q_onset> <gain>`, `ridge local ...`, `ridge off`. |
| `rnd off` \| `<qfreeze>` [`<maxconsec>`] | Freeze charges above `qfreeze` for at most `maxconsec` steps instead of aborting a near-critical solve. |
| `precond ilu` [`drop_tol`] \| `diag` | Preconditioner for the ACKS2 saddle: diagonal (default) or a centralized block-ILUT for the near-singular metallic limit. |
| `bondsoft off` \| `<kappa> <bcut>` | Standard ACKS2 bond softness. Arming it selects MINRES and the Gaussian taper-cutoff operator. |
| `recip_self peratom` \| `auto` \| `<raw>` | Grid self-term subtracted from the diagonal. `peratom`: the exact per-atom value from `pppm/samqeq` (recommended where the mesh allows, at least 9 points per direction at order 5; not on Kokkos). `auto` (default): a scalar calibrated by deterministic probes. `<raw>`: pin the scalar, in the kspace style's internal units (1/Å; multiply by the Coulomb constant for eV/e²). A pin belongs to one Ewald split; the fix warns when it departs by more than 2% from the converged-mesh limit `2 g_ewald/√π` of the current split. |
| `recip_probes <K>` | Number of deterministic probe pairs for the grid self-term calibration (default 16). The calibration is repeated only when the grid changes, and a jump at an unchanged grid is refused. |
| `devsolve on` \| `off` | Kokkos only: run the solve on the device (default off). |
| `solver cg` \| `minres [<qcap>]` \| `cg nofallback` | Krylov solver (default `cg`, with automatic MINRES fallback on divergence; `nofallback` disables it). `<qcap>` enables the stall-gated charge truncation of MINRES. |
| `solver warmstart on` \| `off` | Seed the solve from the previously converged charges (default off). |
| `shieldcheck <tol>` \| `on` \| `off` | Initialization checks that the pair style and the fix use the same kernel, parameters, cutoff range, scope and type-pair coverage. `<tol>` is the range tolerance in eV per unit charge pair (default 0.02). `off` disables every check and warns. |
| `ionfield on` \| `off` | Fixed charges outside the fix group enter the solve through their field (`on`). A deck with such charges must say which it intends; `off` acknowledges an ion-blind solve. |
| `quartic <group> <c4>` [`<c>`] [`<niter>`] [`<mix>`] [`<fld0>`] [`<qref>`] | Quartic on-site wall `c4 q⁴` on the group (with `qref > 0`, on `|q| − qref` only), solved by damped Picard iteration (`niter` passes, mixing `mix`). It acts on the charges only and adds no force term, so dynamics under it do not conserve energy. |
| `shield ...`, `iondamp ...`, `shieldpair ...`, `gself ...` | See the sections above. |

## Coupling to other potentials

`fix qeq/sam` builds its own neighbor list and reads its own parameter file, so it can be overlaid on any
LAMMPS potential.

- **Fixed-charge force fields.** Overlay `coul/shield/intra` and `coul/long` on the existing pair style as in
  [Long-range routes](#long-range-routes).
- **ReaxFF.** Replace `fix qeq/reaxff` with `fix qeq/sam` (enable `PKG_REAXFF`). The fix needs its own
  parameter file; ReaxFF was parameterized against its own charges, so re-validate after swapping the charge
  method.

## Observables and free energies

- **`compute dipole/samqeq [debye]`** returns per-molecule dipoles from the live charges: the mean `|μ|` and a
  5-vector. A molecule with nonzero net charge has an origin-dependent dipole; the compute handles it
  explicitly (see the header of `compute_dipole_samqeq.cpp`). Molecule ID 0 is excluded.
- **`energy yes`** makes `pe` the complete samQEq energy (pair + kspace + self-energy), and `compute fep` then
  works directly with `fix qeq/sam` for pair-coefficient and fixed-charge perturbations. See
  [`doc/10_free_energy.md`](doc/10_free_energy.md).

## GPU (Kokkos)

`-sf kk` selects `fix qeq/sam/kk`, `coul/shield/intra/kk` and `pppm/samqeq/kk`. On a single node of four V100
GPUs the largest tested systems run at about twice the throughput of a 128-core CPU node. Multi-node GPU
scaling is limited by the collective communication inside the iterative solve.

The device path refuses, at initialization, the configurations it does not implement: molecule-free
long-range systems, `recip_self peratom`, `kspace_modify slab`, `iondamp` and `shieldpair`.
Routes the device solve does not cover run on the host. With GPU-aware MPI, pass
`-pk kokkos gpu/aware on`.

## OpenMP threading

`package omp N` threads the hot loops inside the fix, with no new keyword or style variant. Threading engages
only above about 4096 atoms per rank, because fork/join overhead makes small cells slower. Results are
bit-identical run to run at a fixed thread count but not across thread counts. On a 5184-atom deck with one
MPI rank, four threads give 1.23×; two MPI ranks give 1.80×, so MPI ranks remain the first choice.

## Verification

- **A published model, without refitting.** TIP4P-FQ with the exact Slater kernel and `lr_ewald 1`
  reproduces the gas and liquid dipoles quoted above (`examples/tip4pfq_liquid/`).
- **ReaxFF charge equilibration.** The native ReaxFF charge-equilibration step is reproduced to about
  10⁻⁹ e.
- **Kernel comparison.** `examples/kernel_comparison/` runs the same system under two kernels, with both decks
  and the reference output.
- **Test suite.** `tests/` holds self-contained cases, each compared against committed per-atom
  charges at 10⁻⁹ relative tolerance. `tests/CASES.txt` lists them. Run with

  ```bash
  LMP=/path/to/lmp tests/run_tests.sh # all cases, np=1
  LMP=/path/to/lmp tests/run_tests.sh <case> ...
  ```

- **Initialization checks.** At startup the fix asserts that the pair style and the solve use the same kernel,
  kernel parameters, cutoff and pair scope, and refuses a mismatch rather than integrating forces from one
  operator while solving charges against another.

## Limitations

- Physical validation is limited to water models, one gold-slab configuration and two ReaxFF systems.
- The default atomic-charge solve should not be used for the longitudinal response of long molecules; the
  bond-softness kernel that restores the correct scaling needs two parameters the shipped sets do not supply.
- MINRES is unpreconditioned.
- The device solve covers the atomic-charge path on a single node; other routes solve on the host.

## Version and citation

This is samQEq 1.0.1, archived at Zenodo, https://doi.org/10.5281/zenodo.23051745. If you use it, please cite
that record and the accompanying paper: N. Solan, D. Sun and T. A. Pascal, "samQEq: unified charge equilibration
from molecules to conductors", Computer Physics Communications (submitted).

## License

Derived from LAMMPS (`fix_acks2_reaxff`) and distributed under the GNU General Public License, version 2.
