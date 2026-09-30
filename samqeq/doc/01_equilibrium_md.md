# 01 — Equilibrium MD with fluctuating charges

The simplest samQEq run: a polarizable fluctuating-charge liquid where charge equilibrates **within** each
molecule but **not** between molecules. This is plain QEq/FQ, the baseline
on which every other capability builds.

## When to use

Polarizable electrostatics for a molecular liquid or solid: the per-atom charges respond to the local field
(gas→liquid dipole enhancement, field-dependent polarization), and molecules stay neutral.

## Minimal deck (SPC-FQ water, short-range)

```
units metal
atom_style full
boundary p p p
read_data water.data # O (type 1) + H (type 2), molecule IDs per water

pair_style hybrid/overlay lj/cut 10.0 coul/shield/intra 5.0
pair_coeff 1 1 lj/cut 0.01275 3.176
pair_coeff 1 2 lj/cut 0.0 1.0
pair_coeff 2 2 lj/cut 0.0 1.0
pair_coeff * * coul/shield/intra 1.11
neighbor 2.0 bin

group water type 1 2
fix chg water qeq/sam 1 0.0 10.0 1.0e-6 spcfq.param

velocity water create 300.0 12345 mom yes rot yes
fix rigidw water rigid/nvt/small molecule temp 300.0 300.0 0.1
timestep 0.0005
thermo 200
run 20000
```

`spcfq.param`:

```
# gamma_align kappa_bond r_ov r_loc lr_alpha lr_ewald
0.30 10.0 2.0 3.5 0.0 0
1 7.68 15.91 1.11 0.0 -7.0 3.0 # Ow
2 4.50 17.01 1.11 0.0 -7.0 3.0 # Hw
```

## Key points

- **Every number you type is in the deck's own energy units.** This deck is `units metal` (eV); switch to
  `units real` and every param-file column and `fix_modify` numeric argument is interpreted in kcal/mol instead
  — no separate real-units param file convention, just scale the energy-dimensioned numbers by ×23.060549 if you
  want the same physical water (charge-dimensioned values like `q0` are unchanged). See the top-level README's
  *Units convention* section for the full R1/R2/R3 policy.
- **Charge equilibrates within each molecule.** Charge equilibrates only
  inside each molecule ID (per-fragment neutrality). Different molecules cannot exchange net charge.
- **`coul/shield/intra`** is the shielded Coulomb whose kernel matches the QEq solve's intramolecular term — use
  it (not bare `lj/cut/coul/cut`) so the pair forces are consistent with the solved charges and NVE conserves
  energy. Set its single coefficient to `√(γ_iγ_j)` (= the param `gamma`, 1.11 here).
- **Rigid water** (`fix rigid/.../small molecule`) is the usual choice; the FQ charges still fluctuate on the
  rigid geometry. For flexible water add SHAKE on the O–H bonds + angle.
- **Charge fluctuations are physical:** the per-molecule dipole rises from the gas value to the liquid value as
  the field builds up. Dump `q` to see it: `dump d all custom 200 t.lammpstrj id type xu yu zu q`.

## Validation you should see

A 3-site SPC-FQ model gives a gas dipole ~1.9 D rising to a liquid dipole ~2.5–2.7 D. If the liquid dipole is
~1 D you are probably using a detuned `χ_O`.

## Next

Add long-range Coulomb and learn the solver controls → [02](02_long_range_and_solvers.md).
