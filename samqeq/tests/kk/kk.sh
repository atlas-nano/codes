#!/usr/bin/env bash
# tests/kk/kk.sh — the Kokkos/GPU arm of the samQEq test suite.
#
# Runs cases/<name>/in.lammps with `-k on g 1 -sf kk` at np 1 and compares to the HOST goldens. The host
# runner (run_tests.sh) never exercises a device path, so device-only defects are caught only here.
#
# Expectations live in kk/expect.txt, one line per case:   <case>  PASS <rtol> | REFUSED <error-substring> | XFAIL <why>
#   PASS    ran, charges (and pe.txt) match the host golden within <rtol> (ATOL 1e-12)
#   REFUSED LAMMPS must stop with an ERROR line containing <substring> (refused-by-design on kk: slab, peratom,
#           atom_style charge) — a case that unexpectedly RUNS is a FAIL (a refusal silently lifted)
#   XFAIL   known device discrepancy or crash — reported, not counted as pass;
#           a case that unexpectedly PASSES is reported loudly so the expectation gets retired
# Cases absent from expect.txt are FAIL "unclassified" — every suite case must be classified for the device arm.
#
# Exit: 0 all as expected; 1 a case failed; 3 SKIPPED (no Kokkos styles in the binary, or no usable GPU) —
# distinct from 0 on purpose, so that a CI cannot report "no GPU" as green.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; T="$HERE/.."
LMP="${LMP:-$HOME/codes/lammps/lammps-dev2/build/lmp_parallel}"
KK="${KK:--k on g 1 -sf kk}"; ATOL="${ATOL:-1e-12}"; EXPECT="${EXPECT:-$HERE/expect.txt}"

# ---- preflight: say loudly why nothing ran, and exit 3, never 0 ------------------------------------------
[ -x "$LMP" ] || { echo "SKIP kk arm: no binary at $LMP"; exit 3; }
"$LMP" -h 2>/dev/null | grep -q "qeq/sam/kk" || { echo "SKIP kk arm: $LMP has no qeq/sam/kk style (built without KOKKOS+SAMQEQ)"; exit 3; }
pf="$(mktemp -d)"; printf 'print "kk-preflight-ok"\n' > "$pf/in.pf"
if ! ( cd "$pf" && timeout 120 "$LMP" $KK -in in.pf -log none -screen screen.txt >/dev/null 2>&1 && grep -q kk-preflight-ok screen.txt ); then
  echo "SKIP kk arm: '$LMP $KK' cannot start (no GPU / Kokkos device init failed):"; tail -3 "$pf/screen.txt" | sed 's/^/    /'; rm -rf "$pf"; exit 3; fi
rm -rf "$pf"

CASES=("$@"); [ ${#CASES[@]} -eq 0 ] && while IFS= read -r d; do CASES+=("$d"); done < <(cd "$T/cases" && for x in */; do echo "${x%/}"; done)
pass=0; fail=0; refused=0; xfail=0; xpass=0
for c in "${CASES[@]}"; do
  dir="$T/cases/$c"; [ -f "$dir/in.lammps" ] || { echo "SKIP $c (no in.lammps)"; continue; }
  exp="$(awk -v c="$c" '$1==c {print; exit}' "$EXPECT")"; kind="$(echo "$exp" | awk '{print $2}')"; arg="$(echo "$exp" | cut -d' ' -f3- )"
  [ -z "$kind" ] && { echo "FAIL $c (unclassified in $EXPECT)"; fail=$((fail+1)); continue; }
  w="$(mktemp -d)"; cp "$dir"/* "$w"/ 2>/dev/null; rm -f "$w/charges.dump" "$w/pe.txt" "$w/log.lammps"
  ( cd "$w" && timeout 600 "$LMP" $KK -in in.lammps -log log.lammps -screen none > run.log 2>&1 ); rc=$?
  err="$(grep -m1 '^ERROR' "$w/log.lammps" 2>/dev/null | cut -c1-160)"
  case "$kind" in
    REFUSED) if [ $rc -ne 0 ] && [[ "$err" == *"$arg"* ]]; then echo "REFUSED $c (as expected)"; refused=$((refused+1));
             else echo "FAIL $c (expected refusal '$arg'; rc=$rc; ${err:-no ERROR line})"; fail=$((fail+1)); fi ;;
    PASS|XFAIL)
      if [ $rc -ne 0 ] || [ ! -f "$w/charges.dump" ]; then res="rc=$rc ${err:-$(tail -1 "$w/run.log" | cut -c1-120)}"; ok=1
      else rtol="$([ "$kind" = PASS ] && echo "$arg" || echo 1e-9)"
           res="$(python3 "$T/lib/numdiff.py" "$dir/golden/charges.dump" "$w/charges.dump" "$rtol" "$ATOL")"; ok=$?
           if [ $ok -eq 0 ] && [ -f "$dir/golden/pe.txt" ]; then pe="$(python3 "$T/lib/numdiff.py" "$dir/golden/pe.txt" "$w/pe.txt" "$rtol" "$ATOL")" || { ok=1; res="pe.txt: $pe"; }; fi; fi
      if [ "$kind" = PASS ]; then
        if [ $ok -eq 0 ]; then echo "PASS $c  ($res)"; pass=$((pass+1)); else echo "FAIL $c  ($res)"; fail=$((fail+1)); fi
      else
        if [ $ok -eq 0 ]; then echo "XPASS $c  (expected to fail: $arg -- retire the XFAIL)"; xpass=$((xpass+1)); else echo "XFAIL $c  ($arg; $res)"; xfail=$((xfail+1)); fi
      fi ;;
    *) echo "FAIL $c (bad expectation kind '$kind')"; fail=$((fail+1)) ;;
  esac
  rm -rf "$w"
done
echo "-----"; echo "kk arm: passed=$pass refused-as-expected=$refused xfail=$xfail xpass=$xpass failed=$fail"
[ "$fail" -eq 0 ]
