#!/usr/bin/env bash
# repro driver: build_molinv S4-cache MPI deadlock (per-rank nmax growth).
# NOT part of the golden bit-suite (lives outside cases/ on purpose: it is a
# hang/recovery test, not a bit-identity test, and it needs np>1).
#
# Usage:
#   ./run_repro.sh            # size, then run; report PASS/FAIL
#   LMP=/path/to/lmp ./run_repro.sh
#
# Verdicts:
#   PRE-FIX binary : "FAIL (defect reproduced)" -- EITHER the timeout expires
#                    with the run stalled (the campaign's np4 signature) OR the
#                    run aborts with MPI_ERR_TRUNCATE inside MPI_Allreduce (this
#                    deck's np2/OpenMPI signature; recorded baseline 2026-08-08:
#                    died between thermo steps 100-125, twice, same window).
#                    Both are the same defect: ranks split at the cache gate and
#                    their next collectives mismatch.
#   POST-FIX binary: "PASS" -- run completes AND the log carries the 
#                    desync warning (proof the divergence was exercised).
#   "INCONCLUSIVE" -- run completes WITHOUT the warning: the deck no longer
#                    drives rank 0 across a 16384 multiple; retune vdense so the
#                    sizing pass lands nlocal+nghost(max) 300-1500 under one.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LMP="${LMP:-$HOME/codes/lammps/lammps-dev2/build/lmp_parallel}"
TMO="${TMO:-2400}"   # seconds; the healthy post-fix run finishes well inside this

cd "$HERE"
echo "== sizing pass (run 0, np2) =="
mpirun -np 2 "$LMP" -in in.lammps -var nsteps 0 -log size.log > size.out 2>&1 || { echo "sizing run failed"; tail -5 size.out; exit 2; }
nl=$(awk '/^Nlocal:/{print $4}' size.log); ng=$(awk '/^Nghost:/{print $4}' size.log)
peak=$(printf '%.0f' "$(echo "$nl + $ng" | bc -l)")
mult=$(( (peak/16384 + 1) * 16384 ))
echo "rank-0 peak nlocal+nghost ~= $peak; next nmax multiple = $mult; headroom = $((mult-peak))"
if [ $((mult-peak)) -gt 2000 ]; then
  echo "WARNING: headroom > 2000 -- the stream may not cross the multiple; increase vdense"
fi

echo "== repro run (np2, timeout ${TMO}s) =="
timeout "$TMO" mpirun --mca orte_base_help_aggregate 0 -np 2 "$LMP" -in in.lammps -log repro.log > repro.out 2>&1
rc=$?
if [ $rc -eq 124 ]; then
  echo "FAIL (defect reproduced, hang form): run stalled and hit the ${TMO}s timeout."
  echo "  last thermo line:"; grep -hE "^ *[0-9]+ +-?[0-9]" repro.out repro.log 2>/dev/null | tail -1 | sed 's/^/    /'
  exit 1
fi
if grep -q "MPI_ERR_TRUNCATE\|An error occurred in MPI_Allreduce" repro.out 2>/dev/null; then
  echo "FAIL (defect reproduced, abort form): mismatched MPI_Allreduce (MPI_ERR_TRUNCATE)."
  echo "  last thermo line:"; grep -hE "^ *[0-9]+ +-?[0-9]" repro.out repro.log 2>/dev/null | tail -1 | sed 's/^/    /'
  exit 1
fi
if grep -q "REPRO_COMPLETED" repro.log 2>/dev/null || grep -q "REPRO_COMPLETED" repro.out 2>/dev/null; then
  if grep -q "build_molinv cache validity diverged" repro.log repro.out 2>/dev/null; then
    echo "PASS: run completed AND the desync warning fired (divergence exercised and recovered)."
    exit 0
  fi
  echo "INCONCLUSIVE: run completed but the desync warning never fired -- the deck did not"
  echo "cross an nmax multiple. Retune vdense (see header) and re-run."
  exit 3
fi
echo "run ended abnormally (rc=$rc, no completion marker); tail of repro.out:"
tail -10 repro.out | sed 's/^/    /'
exit 2
