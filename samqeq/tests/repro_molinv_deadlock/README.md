# repro_molinv_deadlock — build_molinv S4-cache MPI desync repro

NOT part of the golden bit-suite (deliberately outside `cases/`): this is a
multi-rank hang/recovery test, not an np1 bit-identity test.

Defect: the build_molinv cache gate branched a collective on rank-LOCAL state
(`molinv_valid`/`molinv`/`mol2c`, all cleared per-rank by
`reallocate_storage() -> deallocate_storage()` when `atom->nmax` regrows on one
rank at an exchange event). Ranks split between the collective rebuild and the
no-comm cache path -> mismatched collectives. Fix : the gate verdict is
Allreduce(MIN)'d, plus a one-shot desync warning.

Run: `./run_repro.sh` (see its header for verdicts). Key point: on a post-fix
binary a PASS requires BOTH completion AND the logged warning
`samqeq: build_molinv cache validity diverged across ranks at step N` —
completion without the warning means the deck no longer crosses an nmax
multiple on rank 0 and must be retuned (`vdense`, see in.lammps header).

Recorded pre-fix baseline (2026-08-08, binary built Aug 8 07:23):
- np2: aborts between thermo steps 100 and 125 (three runs, same window) with
  `An error occurred in MPI_Allreduce ... MPI_ERR_TRUNCATE: message truncated`
  — build_molinv's 1-tagint rebuild reduce on the regrown rank colliding with
  project_neutral's nmol_+tail-double reduce on the other. (The np4 campaign
  saw the hang form of the same mismatch.)
- np1 control: same deck, clean through step 300 (`REPRO_COMPLETED`) — the
  failure is decomposition-linked, not physics.
- Sizing (run 0): rank 0 nlocal+nghost = 4957+10649 = 15606, i.e. 778 under the
  16384 AtomVec DELTA quantum; rank 1 streams 1000 molecules (3000 atoms) into
  rank 0 at 20 A/ps.

Raw logs kept here: `size.log`, `serial_ctrl.log`, `prefix_fail.log/.out`
(first hit), `prefix_fail2.log/.out` (ORTE aggregation off — shows the
truncate), `baseline_20260808.txt` + `repro.log/.out` (shipped-deck driver
baseline).
