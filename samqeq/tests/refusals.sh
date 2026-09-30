#!/usr/bin/env bash
# samQEq refusal tests. run_tests.sh diffs numbers and has no way to expect an error; this script does.
#
# For every cases/<name>/in.refuse_* deck:
#   - a deck with a line `# EXPECT: <string>` PASSes only if LAMMPS exits NON-zero AND <string> is in its output
#     (the refusal fired, for the stated reason);
#   - a deck with a line `# EXPECT_OK` is the positive control (the way out the refusal names): it PASSes only if
#     LAMMPS exits 0 AND writes charges.dump.
# For every cases/<name>/in.warn_* deck (warnings, not refusals):
#   - `# EXPECT_WARN: <string>` PASSes only if LAMMPS exits 0, writes charges.dump AND prints `WARNING: <string>`;
#   - `# EXPECT_NOWARN: <string>` is the negative control: same, but that warning must be ABSENT.
#   (`WARNING: ` is prepended to the grep so the deck's own comment line can never satisfy it.)
#
# Usage:
#   ./refusals.sh                   # all probes
#   LMP=/path/to/lmp ./refusals.sh  # override the LAMMPS binary
set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LMP="${LMP:-$HOME/codes/lammps/lammps-dev2/build/lmp_parallel}"
if [ ! -x "$LMP" ]; then echo "ERROR: LAMMPS binary not found/executable: $LMP" >&2; exit 2; fi

pass=0; fail=0
for deck in "$HERE"/cases/*/in.refuse_* "$HERE"/cases/*/in.warn_*; do
  [ -f "$deck" ] || continue
  dir="$(dirname "$deck")"; name="$(basename "$dir")/$(basename "$deck")"
  expect="$(sed -n 's/^# EXPECT: //p' "$deck" | head -1)"
  ok=0; grep -q '^# EXPECT_OK' "$deck" && ok=1
  warn="$(sed -n 's/^# EXPECT_WARN: //p' "$deck" | head -1)"
  nowarn="$(sed -n 's/^# EXPECT_NOWARN: //p' "$deck" | head -1)"
  if [ -z "$expect" ] && [ $ok -eq 0 ] && [ -z "$warn" ] && [ -z "$nowarn" ]; then
    echo "FAIL $name (no '# EXPECT:', '# EXPECT_OK', '# EXPECT_WARN:' or '# EXPECT_NOWARN:' line)"; fail=$((fail+1)); continue; fi
  work="$(mktemp -d)"
  cp "$dir"/* "$work"/ 2>/dev/null
  rm -f "$work/charges.dump" "$work/pe.txt" "$work/log.lammps"
  ( cd "$work" && mpirun -np 1 "$LMP" -in "$(basename "$deck")" -nb > run.log 2>&1 )
  rc=$?
  if [ -n "$warn" ] || [ -n "$nowarn" ]; then
    if [ $rc -ne 0 ] || [ ! -s "$work/charges.dump" ]; then
      echo "FAIL $name (warning probe must run; rc $rc; tail of run.log:)"; tail -3 "$work/run.log" | sed 's/^/    /'; fail=$((fail+1))
    elif [ -n "$warn" ] && ! grep -qF -- "WARNING: $warn" "$work/run.log"; then echo "FAIL $name (expected warning missing)"; fail=$((fail+1))
    elif [ -n "$nowarn" ] && grep -qF -- "WARNING: $nowarn" "$work/run.log"; then echo "FAIL $name (warning fired but must be silent)"; fail=$((fail+1))
    else echo "PASS $name (runs; warning ${warn:+present}${nowarn:+absent} as expected)"; pass=$((pass+1)); fi
  elif [ $ok -eq 1 ]; then
    if [ $rc -eq 0 ] && [ -s "$work/charges.dump" ]; then echo "PASS $name (runs: rc 0, charges.dump written)"; pass=$((pass+1))
    else echo "FAIL $name (control must run; rc $rc; tail of run.log:)"; tail -3 "$work/run.log" | sed 's/^/    /'; fail=$((fail+1)); fi
  else
    if [ $rc -ne 0 ] && grep -qF -- "$expect" "$work/run.log"; then echo "PASS $name (refused: rc $rc, message found)"; pass=$((pass+1))
    elif [ $rc -eq 0 ]; then echo "FAIL $name (NOT refused: LAMMPS exited 0)"; fail=$((fail+1))
    else echo "FAIL $name (rc $rc but expected message missing; tail of run.log:)"; tail -3 "$work/run.log" | sed 's/^/    /'; fail=$((fail+1)); fi
  fi
  rm -rf "$work"
done
echo "-----"
echo "refusals passed=$pass failed=$fail"
[ $fail -eq 0 ]
