# 10 — Free energy: total energy and `compute fep`

samQEq exposes the **complete** potential energy, so standard free-energy methods (FEP, thermodynamic
integration) work.

## Total energy (`energy yes`)

```
fix_modify chg energy yes
```

This folds the QEq on-site self-energy `Σ(χ_i q_i + ½ η_i q_i²)` into the global potential energy, so thermo `pe`
becomes `pair + kspace + self` — the complete samQEq energy. Two facts to rely on:

- The contribution is **force-free**: the self-energy is position-independent at the solved-charge minimum, so by
  Hellmann–Feynman it adds exactly zero force/virial (verified atom-by-atom: forces are byte-identical with
  `energy yes` vs `no`). It is safe to leave on during dynamics.
- The self-energy alone is also available as the fix **global scalar** `f_<id>` (e.g. `f_chg`), independent of the
  `energy yes` toggle.

## `compute fep` (FEP package)

With `energy yes`, `compute fep` works directly with `fix qeq/sam`:

```
fix_modify chg energy yes
variable dlj equal 0.0015
compute fp all fep 300.0 pair lj/cut epsilon 1 1 v_dlj
thermo_style custom step pe c_fp[1] c_fp[2] # c_fp[1]=dU, c_fp[2]=exp(-dU/kT)
```

- The perturbation delta must be a `v_` **variable** (not a literal).
- The FQ charges are solved each step, so the perturbed-state energy reflects the polarizable response of the
  charges that are *not* perturbed.
- For a **same-box** perturbation the large constant on-site terms (the fixed core-charge self-energy, the Ewald
  self) cancel in the difference `ΔU`, so `energy yes` gives the right total. (Across *different* boxes — a
  gas↔liquid ΔHvap — the on-site self-energy is not additive; see the ΔHvap note below.)
- This is the right tool for **pair-coefficient / fixed-charge** (alchemical λ) perturbations with polarizable
  charges.
- A perturbation of the charge parameters themselves (χ, η or q⁰) is not captured: `compute fep` re-evaluates
  the energy without re-solving the charges.

## ΔHvap: use the cohesion (pe) value, not the total energy

A heat of vaporization is an **intermolecular** energy, so use the **cohesion** value (pe = pair + kspace, gas
vs liquid at the same recipe). This is the physically correct number, not
an approximation. Do **not** add the on-site QEq self-energy (`energy yes`/`f_chg`) here: the self-energy is an
**intramolecular** quantity, and the gas→liquid polarization raises it by a large amount
that is **offset by the intermolecular Coulomb stabilization
already counted in pe**. The two are therefore *not additive across the phase change* — adding the on-site term
double-counts and yields an unphysical (even negative) ΔHvap. (This is specific to a gas↔liquid, **different-box**
total-energy difference; for a **same-box** FEP/TI the on-site terms cancel and `energy yes` is the right total —
see above.) If a polarization correction is wanted, it is only the small gas-relaxation term (one isolated
molecule's total energy at its liquid charges vs its gas-equilibrium charges).
