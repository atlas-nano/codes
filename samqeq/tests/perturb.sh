#!/usr/bin/env bash
# samQEq uninitialised-read check. For every case, run the deck twice at
# NP ranks on the SAME binary -- plain, and with MALLOC_PERTURB_=165 (glibc fills every fresh allocation with garbage) --
# and require the two to be BYTE-IDENTICAL (charges.dump and pe.txt). A code path that reads a never-written slot
# changes the result or aborts; nothing else can. No goldens, no tolerance: np-N vs np-1 round-off plays no part.
#
# Usage:  ./perturb.sh                  # NP=2, all cases
#         NP=4 ./perturb.sh <case>...   # np 3-4 as well
#         LMP=/path/to/lmp ./perturb.sh
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LMP="${LMP:-$HOME/codes/lammps/lammps-dev2/build/lmp_parallel}"
NP="${NP:-2}"
[ -x "$LMP" ] || { echo "ERROR: LAMMPS binary not found: $LMP" >&2; exit 2; }
CASES=("$@")
[ ${#CASES[@]} -eq 0 ] && while IFS= read -r d; do CASES+=("$d"); done < <(cd "$HERE/cases" && for x in */; do echo "${x%/}"; done)
ident=0; diff=0; abort=0
for c in "${CASES[@]}"; do
  dir="$HERE/cases/$c"; [ -f "$dir/in.lammps" ] || continue
  res=()
  for mode in plain perturb; do
    w="$(mktemp -d)"; cp "$dir"/* "$w"/ 2>/dev/null; rm -f "$w/charges.dump" "$w/pe.txt" "$w/log.lammps"
    if [ $mode = perturb ]; then ( cd "$w" && MALLOC_PERTURB_=165 mpirun -np "$NP" -x MALLOC_PERTURB_ "$LMP" -in in.lammps > run.log 2>&1 )
    else ( cd "$w" && mpirun -np "$NP" "$LMP" -in in.lammps > run.log 2>&1 ); fi
    rc=$?; res+=("$w:$rc")
  done
  wp=${res[0]%:*}; rp=${res[0]##*:}; wq=${res[1]%:*}; rq=${res[1]##*:}
  if [ "$rp" -ne 0 ]; then echo "SKIP  $c (plain np$NP itself failed, rc $rp)"
  elif [ "$rq" -ne 0 ] || [ ! -s "$wq/charges.dump" ]; then echo "ABORT $c (perturbed rc $rq: $(grep -m1 ERROR "$wq/run.log" | cut -c1-90))"; abort=$((abort+1))
  elif cmp -s "$wp/charges.dump" "$wq/charges.dump" && { [ ! -f "$wp/pe.txt" ] || cmp -s "$wp/pe.txt" "$wq/pe.txt"; }; then echo "IDENT $c"; ident=$((ident+1))
  else echo "DIFF  $c ($(python3 "$HERE/lib/numdiff.py" "$wp/charges.dump" "$wq/charges.dump" 0 0 2>&1 | tail -1 | cut -c1-60))"; diff=$((diff+1)); fi
  rm -rf "$wp" "$wq"
done
echo "-----"; echo "perturb np$NP: IDENT=$ident DIFF=$diff ABORT=$abort"
[ $diff -eq 0 ] && [ $abort -eq 0 ]
