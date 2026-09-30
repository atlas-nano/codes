#!/usr/bin/env bash
# samQEq bit-correctness regression tests.
#
# Each cases/<name>/ holds a self-contained LAMMPS deck (in.lammps) plus its
# .param/.mol inputs. The deck must write `charges.dump` (id type q) and may
# write `pe.txt` (system PE). Runs are done at np=1 so they are deterministic /
# bit-reproducible; the resulting charges are diffed against the committed
# golden output. A passing diff means the code change did NOT perturb the
# numerical result.
#
# Usage:
#   ./run_tests.sh                  # run all cases, compare to golden
#   ./run_tests.sh spcfq_gas_dsf    # run only named case(s)
#   ./run_tests.sh --update         # (re)generate golden outputs for all cases
#   LMP=/path/to/lmp ./run_tests.sh # override the LAMMPS binary
#   RTOL=1e-7 ./run_tests.sh        # loosen tolerance (default 1e-9)
set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LMP="${LMP:-$HOME/codes/lammps/lammps-dev2/build/lmp_parallel}"
NP="${NP:-1}"            # keep at 1: MPI reductions are not bit-identical across proc counts
RTOL="${RTOL:-1e-9}"
ATOL="${ATOL:-1e-12}"

UPDATE=0
CASES=()
for a in "$@"; do
  if [ "$a" = "--update" ]; then UPDATE=1; else CASES+=("$a"); fi
done
if [ ${#CASES[@]} -eq 0 ]; then
  while IFS= read -r d; do CASES+=("$d"); done < <(cd "$HERE/cases" && for x in */; do echo "${x%/}"; done)
fi

if [ ! -x "$LMP" ]; then echo "ERROR: LAMMPS binary not found/executable: $LMP" >&2; exit 2; fi

pass=0; fail=0
for c in "${CASES[@]}"; do
  dir="$HERE/cases/$c"
  if [ ! -f "$dir/in.lammps" ]; then echo "SKIP $c (no in.lammps)"; continue; fi
  work="$(mktemp -d)"
  # copy ALL case input files (in.lammps, *.param, *.mol, data.*, etc.); cp (non-recursive) skips the golden/ dir
  cp "$dir"/* "$work"/ 2>/dev/null
  # A case directory may carry a stale run artifact that was committed by accident. Copying it in
  # makes the case UNFALSIFIABLE: LAMMPS can die before writing anything and the check below still
  # finds a charges.dump -- the previous good run's -- and diffs it clean; two cases once passed
  # that way. Delete the outputs in the work dir before running, so a
  # charges.dump present afterwards can only be one this run produced.
  rm -f "$work/charges.dump" "$work/pe.txt" "$work/log.lammps"
  ( cd "$work" && mpirun -np "$NP" "$LMP" -in in.lammps > run.log 2>&1 )
  rc=$?
  # A non-zero exit is a failure even if the outputs somehow exist: the run did not finish.
  if [ "$rc" -ne 0 ]; then
    echo "FAIL $c (LAMMPS exit $rc; tail of run.log:)"
    tail -5 "$work/run.log" | sed 's/^/    /'
    fail=$((fail+1)); rm -rf "$work"; continue
  fi
  if [ ! -f "$work/charges.dump" ]; then
    echo "FAIL $c (no charges.dump produced; tail of run.log:)"
    tail -5 "$work/run.log" | sed 's/^/    /'
    fail=$((fail+1)); rm -rf "$work"; continue
  fi
  if [ "$UPDATE" = "1" ]; then
    mkdir -p "$dir/golden"
    cp "$work/charges.dump" "$dir/golden/charges.dump"
    [ -f "$work/pe.txt" ] && cp "$work/pe.txt" "$dir/golden/pe.txt"
    echo "UPDATED $c"
  else
    if [ ! -f "$dir/golden/charges.dump" ]; then
      echo "FAIL $c (no golden; run with --update first)"; fail=$((fail+1)); rm -rf "$work"; continue
    fi
    out="$(python3 "$HERE/lib/numdiff.py" "$dir/golden/charges.dump" "$work/charges.dump" "$RTOL" "$ATOL")"
    ok=$?
    # also check the recorded scalar (system PE, or a fix scalar) if a golden exists
    if [ $ok -eq 0 ] && [ -f "$dir/golden/pe.txt" ]; then
      peout="$(python3 "$HERE/lib/numdiff.py" "$dir/golden/pe.txt" "$work/pe.txt" "$RTOL" "$ATOL")"
      if [ $? -ne 0 ]; then ok=1; out="pe.txt: $peout"; fi
    fi
    if [ $ok -eq 0 ]; then echo "PASS $c  (q: $out)"; pass=$((pass+1)); else echo "FAIL $c  ($out)"; fail=$((fail+1)); fi
  fi
  rm -rf "$work"
done

echo "-----"
if [ "$UPDATE" = "1" ]; then echo "golden update complete"; exit 0; fi
echo "passed=$pass failed=$fail"
[ "$fail" -eq 0 ]
