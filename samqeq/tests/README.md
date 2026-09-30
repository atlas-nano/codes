# samQEq test suite

Bit-correctness tests for `fix qeq/sam`, `pair coul/shield/intra`,
`compute dipole/samqeq`, and `kspace_style pppm/samqeq`. Each test runs a small,
self-contained deck at **np=1** (so the result is deterministic / bit-reproducible)
and diffs the solved per-atom charges against a committed golden output. A passing
diff means a code change did **not** perturb the numerical result.

## Run

```
./run_tests.sh # run all cases, compare to golden
./run_tests.sh spcfq_gas_dsf # run only the named case(s)
./run_tests.sh --update # (re)generate golden outputs (after an intended change)
LMP=/path/to/lmp ./run_tests.sh # override the LAMMPS binary (default: ~/codes/lammps/lammps-dev2/build/lmp_parallel)
RTOL=1e-7 ./run_tests.sh # loosen tolerance (default rtol 1e-9, atol 1e-12)
```

Exit status is non-zero if any case fails, so it works in CI / pre-commit.

## Layout

```
run_tests.sh driver
lib/numdiff.py numeric dump comparator (abs/rel tolerance, id-matched)
cases/<name>/
    in.lammps deck; must write charges.dump (id type q), may write pe.txt
    *.param *.mol self-contained inputs
    golden/charges.dump committed reference output
    golden/pe.txt committed reference system PE
```

## Cases

`CASES.txt` lists every case; the table below describes a selection, and each case's `in.lammps`
header states what it checks. Each case diffs `charges.dump` (always) and, if present, a scalar
`pe.txt` (system PE, or a fix scalar); both must match the golden.

| case | exercises | reference |
|------|-----------|-----------|
| `spcfq_gas_dsf` | core solver / gate / per-mol-neutral projection, no kspace | symmetric SPC-FQ gas, O=−0.680 e |
| `spcfq_gas_ewald`| `pppm/samqeq` `compute_vector` on a single molecule | gas dipole **1.853 D** (≈ recorded 1.855 D) |
| `spcfq_liq_ewald`| `pppm/samqeq` long-range path in a periodic liquid (lr_ewald=2) | per-molecule-neutral charges, PE −312.2 eV |
| `spcfq_xl` | extended-Lagrangian charge dynamics: 10-step NVE with charges as dynamic DOF | fluctuating charges (O≈−0.81) |
| `spcfq_xl_staircase` | XL **+** the per-type IP-staircase on-site anharmonicity (`c3`/`c4`, param cols 8/9) for the O type — exercises the XL staircase force branch + `apply_quartic_eta()` | XL stays bounded/deterministic with the anharmonic term active |
| `aspc_corrector` | ASPC: short rigid-water NVE with the q-direct predictor–corrector engaged (`fix_modify chg aspc 2`) — pins the predictor history fill + `n_corr=2` corrector + ω-mix | deterministic charges at np=1 |
| `tip4pfq_gas` | TIP4P-FQ 4-site (massless M) augmented solve | O=0, H=+0.447, **M=−0.894 e** |
| `second_run_molinv` | a SECOND `run`'s `setup_pre_force` (mid-deck `reset_timestep`+`dump`) re-enters `build_molinv`, a path no single-run case reaches | 27 SPC-FQ mols, both runs complete, charges reproduce |
| `recalib_on_grid_change` | signature-gated `recip_self` recalibration: a mid-deck `reset_timestep`+integrator-swap+fresh-dump second `run`, which must not leave PPPM with stale grid state ("Out of range") | frozen (`run 0` both times), charges reproduce byte-for-byte across the two runs |
| `charge_solid_global` | bare `atom_style charge` (no molecule IDs) periodic binary solid on the full long-range path (`lr_ewald=2`) — the molecule-free fallback makes the projector enforce GLOBAL charge neutrality instead of erroring | two-`chi` slab, charge transfers across the interface, `sum(q) ~ 3e-12` |

## Notes

- **np=1 is required** for bit reproducibility — MPI reductions and FFT
  decomposition are not bit-identical across processor counts. The physics is
  validated separately at production np; these tests only guard against
  *unintended* numerical drift from refactors.
- Golden outputs are machine/compiler-specific at the 1e-9 level (FFT/BLAS
  rounding). On a new toolchain, re-baseline with `--update` after confirming the
  physics, or run with a looser `RTOL`.
- After an *intentional* numerical change, regenerate goldens with `--update` and
  review the diff before committing.

## Continuous integration (git hook)

A tracked pre-commit hook (`tests/hooks/pre-commit`) runs the suite and blocks a
commit that changes the result of any case. Install it once per clone:

```
git config core.hooksPath tests/hooks
```

It runs only when the commit touches `src/` or `tests/`, and skips cleanly if
the LAMMPS binary isn't found (so it never blocks a commit you can't test). It checks
the **currently-built** binary, so rebuild after editing `src/` before relying
on it; bypass any time with `git commit --no-verify`.

For hosted CI, a runner would need to build LAMMPS (KSPACE + SAMQEQ, plus the
`patches/fix_efield-friend.patch`) and then run `tests/run_tests.sh`.

## Adding a case

1. `mkdir cases/<name>`, add `in.lammps` (must `write_dump ... charges.dump`),
   and copy the `*.param` / `*.mol` it needs into the case dir.
2. `./run_tests.sh --update <name>` to record the golden.
3. Eyeball the golden for physical sanity, then commit it.

## Rank reproducibility of these decks
The suite runs at np=1 by design. Do NOT read a 1-rank vs 4-rank difference on a case built with
`create_atoms ... mol <template> <seed>` as a solver defect: LAMMPS draws the per-molecule random
orientation from a per-processor stream, so the GEOMETRY differs by decomposition (on spcfq_liq_ewald
most atoms move and the charges differ accordingly, for every calibration route and at any tolerance).
Multi-rank consistency of the solve is tested on data-file decks (`parallel/`).

## Multi-rank check
`parallel/parallel.sh` runs the multi-rank `build_molinv` check (`parallel/molinv/`, 8100 SPC-FQ waters on
`processors 2 1 1`, rank-asymmetric storage regrow at step ~126) at np 2 under `timeout` and FAILS on HANG, ABORT,
VACUOUS (the regrow did not happen) or WRONG (step-0 charges / pe 0-50 vs the 1-rank golden). ~1-5 min. Run it by hand,
with `perturb.sh`; it is not in the pre-commit hook. `--update` regenerates the golden.
