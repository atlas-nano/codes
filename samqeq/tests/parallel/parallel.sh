#!/usr/bin/env bash
# samQEq multi-rank check. Run by hand, next to perturb.sh; NOT in the pre-commit hook (it takes ~4-5 min).
#
# molinv/: the build_molinv cache under rank-asymmetric storage regrowth. A 8100-atom SPC-FQ water box on
# `processors 2 1 1` whose second rank's atom storage regrows at step ~126, so the ranks disagree on the molinv
# cache verdict. build_molinv must reconcile that verdict across ranks before its collectives; if it does not, the
# ranks enter different Allreduces and the run hangs. The check FAILS on any of:
#   1. HANG      - the run does not finish within TMO seconds (default 900; rc 124 from `timeout`)
#   2. ABORT     - non-zero exit, or no REPRO_COMPLETED line
#   3. VACUOUS   - the recovery warning "build_molinv cache validity diverged across ranks" never printed, i.e. the
#                  asymmetric regrow did not happen and the check tested nothing (layout/deck/binary changed)
#   4. WRONG     - step-0 charges or pe at steps 0-50 differ from the 1-rank reference (golden/): max|dq| > 1e-9 e or
#                  pe rel. diff > 1e-9
#
# Usage:  ./parallel.sh                 # the check at np 2
#         ./parallel.sh --update        # regenerate golden/ from a 1-rank run of the same deck
#         LMP=/path/to/lmp TMO=1200 ./parallel.sh
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LMP="${LMP:-$HOME/codes/lammps/lammps-dev2/build/lmp_parallel}"
TMO="${TMO:-900}"
[ -x "$LMP" ] || { echo "ERROR: LAMMPS binary not found: $LMP" >&2; exit 2; }
D="$HERE/molinv"; W="$(mktemp -d)"; cp "$D"/in.lammps "$D"/spcfq.mol "$D"/spcfq_expt.param "$D"/cfg_yz10.data "$W"/
ARGS=(-in in.lammps -var usecfg 1 -var cfg cfg_yz10.data -var vyz 10 -var nsteps 600)

if [ "${1:-}" = "--update" ]; then
  ( cd "$W" && timeout -k 30 "$TMO" mpirun -np 1 --bind-to none "$LMP" "${ARGS[@]}" -var px 1 -var py 1 -var pz 1 \
      -log run.log > run.out 2>&1 ); rc=$?
  if [ $rc -ne 0 ] || ! grep -q REPRO_COMPLETED "$W/run.out"; then echo "UPDATE FAILED (rc $rc)"; tail -5 "$W/run.out"; exit 1; fi
  mkdir -p "$D/golden"; cp "$W/charges0.dump" "$D/golden/charges0_np1.dump"
  awk '/^ *Step/{f=1;next} f&&$1~/^[0-9]+$/&&$1<=50{print $1,$3}' "$W/run.log" > "$D/golden/pe_np1_0-50.txt"
  echo "golden updated from np1 ($(wc -l < "$D/golden/pe_np1_0-50.txt") thermo rows)"; rm -rf "$W"; exit 0
fi

t0=$(date +%s)
( cd "$W" && timeout -k 30 "$TMO" mpirun -np 2 --bind-to none "$LMP" "${ARGS[@]}" -var px 2 -var py 1 -var pz 1 \
    -log run.log > run.out 2>&1 ); rc=$?
dt=$(( $(date +%s) - t0 )); fail=0
if [ $rc -eq 124 ] || [ $rc -eq 137 ]; then echo "FAIL molinv: HANG (no finish within ${TMO}s, rc $rc) -- the build_molinv deadlock class is back"; fail=1
elif [ $rc -ne 0 ] || ! grep -q REPRO_COMPLETED "$W/run.out"; then echo "FAIL molinv: ABORT (rc $rc): $(grep -m1 -E 'ERROR|MPI_ERR' "$W/run.out" | cut -c1-120)"; fail=1
fi
if [ $fail -eq 0 ]; then
  w=$(grep -m1 -oE 'build_molinv cache validity diverged across ranks at step [0-9]+' "$W/run.out")
  if [ -z "$w" ]; then echo "FAIL molinv: VACUOUS (no rank-asymmetric regrow happened; the check exercised nothing)"; fail=1; fi
fi
if [ $fail -eq 0 ]; then
  awk '/^ *Step/{f=1;next} f&&$1~/^[0-9]+$/&&$1<=50{print $1,$3}' "$W/run.log" > "$W/pe_np2.txt"
  res=$(python3 - "$D/golden/charges0_np1.dump" "$W/charges0.dump" "$D/golden/pe_np1_0-50.txt" "$W/pe_np2.txt" <<'PY'
import sys
def q(p):
    L=open(p).read().split('\n'); i=[k for k,l in enumerate(L) if l.startswith('ITEM: ATOMS')][0]
    return {int(l.split()[0]): float(l.split()[-1]) for l in L[i+1:] if l.strip()}
a,b=q(sys.argv[1]),q(sys.argv[2])
dq=max(abs(a[k]-b[k]) for k in a) if a.keys()==b.keys() else float('inf')
pa=[l.split() for l in open(sys.argv[3]) if l.strip()]; pb=[l.split() for l in open(sys.argv[4]) if l.strip()]
dp=max(abs(float(x[1])-float(y[1]))/max(1,abs(float(x[1]))) for x,y in zip(pa,pb)) if len(pa)==len(pb) and pa else float('inf')
print(('OK' if dq<=1e-9 and dp<=1e-9 else 'BAD'), f'max|dq| {dq:.1e}, pe rel {dp:.1e} over {len(pa)} rows')
PY
)
  case "$res" in OK*) ;; *) echo "FAIL molinv: WRONG (${res#BAD })"; fail=1;; esac
fi
if [ $fail -eq 0 ]; then echo "PASS molinv np2: completed in ${dt}s, recovered ($w), ${res#OK }"; rm -rf "$W"
else echo "  work dir kept: $W"; fi
# np-invariance of the recip_self calibration under five processor grids (+ a sort-order arm).
LMP="$LMP" "$HERE/npinv.sh" || fail=1
echo "-----"; echo "parallel: $( [ $fail -eq 0 ] && echo PASS || echo FAIL )"
exit $fail
