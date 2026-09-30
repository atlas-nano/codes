# samQEq HOWTOs

Practical, copy-pasteable guides for `fix qeq/sam` (package SAMQEQ): a polarizable-water equilibrium run,
the long-range solvers, and free-energy methods.

Each guide is self-contained: the minimal deck, the relevant `fix_modify` keywords, and the gotchas. The decks
assume `units metal`, `atom_style full`, and a build with packages **KSPACE + SAMQEQ** (and
**FEP** for the free-energy guide). `units real` is equally supported (param-file and
`fix_modify` numbers stay in the deck's own energy units — see the top-level README's *Units convention*); a
`atom_style full` (with
molecule IDs) is the common case in these guides because the per-fragment charge constraint keys off
fragment structure, but `fix qeq/sam` also runs on bare `atom_style charge` (no
molecule IDs at all), falling back to a single global-neutrality fragment; see the top-level README's
*Molecule-optional solve* section.

## Reading order

| # | Guide | What it covers |
|---|-------|----------------|
| 01 | [Equilibrium MD with fluctuating charges](01_equilibrium_md.md) | The simplest run: polarizable fluctuating-charge water. |
| 02 | [Long-range Coulomb & the solvers](02_long_range_and_solvers.md) | `lr_ewald` (incl. native-vs-faithful-port design guidance), the shielding kernel (`shield cbrt\|gaussian\|slater`), projected CG, MINRES, the eig-ridge, warm-start, ASPC predictor–corrector, XL charges, OpenMP threading. |
| 10 | [Free energy: total energy and compute fep](10_free_energy.md) | `energy yes` and FEP. |

## The one-paragraph mental model

`fix qeq/sam` solves the fluctuating charges each step by minimizing
`E = Σ_i(χ_i q_i + ½ η_i q_i²) + ½ Σ_{i≠j} J_ij(r) q_i q_j` subject to a per-fragment (molecule-ID) charge
constraint. `χ` and `η` are the Mulliken electronegativity and hardness. Everything else is a mode on
top: the long-range route and its solvers (02) and the total energy used by free-energy methods (10).
All capabilities are opt-in via `fix_modify` and default to the plain polarizable solve.

## Conventions used throughout

- The fix is created as `fix <id> <group> qeq/sam <Nevery> <cutlo> <cuthi> <tol> <paramfile>`.
- Parameter file rows: `type chi eta gamma q0 eHOMO eLUMO` (a header line carries
  `gamma_align kappa_bond r_ov r_loc lr_alpha lr_ewald [lr_ridge r_bond ...]`). ⚠ The **sixth**
  field is `lr_ewald`, the long-range mode, and the deck's `coul/shield/intra` scope has to match
  it. See guide 02 for the header.
- `f_<id>` is the fix's global scalar (the QEq self-energy).
- Long-range solves need `kspace_style pppm/samqeq` and a `coul/long` + `coul/shield/intra` pair.
