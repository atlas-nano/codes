# Native SPC-FQ water

The experiment-matched samQEq water, with the two runs that reproduce its liquid properties: an NPT
density run and an entropy run.

## The model

Fit to the exact experimental monomer dipole through the analytic monomer relation
`q_H = Δχ / (2η_O + η_H − 4J_OH + J_HH)`, with the on-site hardness set for a stable liquid and the
oxygen–oxygen dispersion refit to the liquid density:

| | value |
|---|---|
| parameters | `spcfq_expt.param` (`lr_ewald = 2`, all-pairs shielding) |
| O–O Lennard-Jones | ε = 0.01275 eV, **σ = 3.144 Å** |
| hydrogen dispersion | none |
| shielding | cube-root kernel, γ = 1.11 |

⚠ **σ = 3.144 is the value that reproduces the table below.** σ = 3.176, found in other SPC-FQ decks,
does not.

## Running them

```bash
mpirun -np 32 lmp -in in.npt_canon # 30 ps NPT, writes spcfq_s3144.restart
mpirun -np 32 lmp -in in.xpt_canon # NVE from that restart, entropy via fix xpt
```

The two are a chain: the NPT run writes the restart the entropy run reads, and the paths are
relative so the pair works inside a container. `in.xpt_canon` additionally requires the **XPT**
package; `in.npt_canon` needs only SAMQEQ, KSPACE, MOLECULE and RIGID.

## ★ The grid self-term is pinned, and it has to be

`in.npt_canon` carries

```
fix_modify chg recip_self 0.37933
```

Under constant pressure the cell volume changes, the particle-mesh grid is re-derived, and the
probe-pair estimator re-selects its pair. Unpinned, this deck recalibrates **384 times over one
30 ps run**, across 0.37573–0.38261 — a 1.83 % spread. The self-term is subtracted from the solve
diagonal, so that drift silently re-tunes the model mid-run; the run-time guard aborts such a run
rather than let it finish with a different Hamiltonian.

0.37933 is the value the calibration returns for this system at its production geometry, identical
to five decimals at 1, 2, 4 and 8 MPI ranks.

⚠ **Do not copy that number to another system.** It is geometry-specific. For a different box, run
once and take the value the calibration reports, or use `fix_modify chg recip_probes <K>`.

⚠ **`recip_probes` is not a drop-in substitute here.** On this system it shifts the density by
+4.2 % and the liquid dipole by +13.2 %. It is a different operator, not a better estimate of the
same one.

## What they reproduce

Three independent 30 ps trajectories, differing only in the initial velocity seed, averaged over the
back 15 ps:

| property | measured | experiment |
|---|---|---|
| density (g cm⁻³) | **0.9970 ± 0.0008** | 0.997 |
| liquid dipole (D) | **2.7972 ± 0.0017** | ~2.9 |
| ⟨q_O⟩ (e) | −1.0020 ± 0.0006 | — |
| gas dipole (D) | 1.855 | 1.855 (exact by construction) |

Uncertainties are the standard error of three seeds. ΔH_vap and the 2PT/3PT entropies come from
`in.xpt_canon`.

## The one deck contract that will bite you

`pair_style … coul/shield/intra 5.0 **all**` — the `all` keyword is **required**. The parameter
file's sixth header field selects all-pairs shielding for the *solve*; without `all` the pair style
corrects intramolecular pairs only, so every intermolecular pair is shielded in the charge matrix
and bare in the forces. The solved charges are unchanged, which is what makes it silent: only the
forces move. Under NPT it is fatal — the box collapses and the charges run to |q| ≈ 2 e within a
couple of picoseconds. The fix refuses such a deck at initialization with a `SCOPE` error.
