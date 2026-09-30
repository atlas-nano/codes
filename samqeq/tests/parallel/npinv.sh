#!/usr/bin/env bash
# samQEq np-INVARIANCE of the recip_self calibration (, TASKS row 182, ledger L52). Release check, by hand;
# parallel.sh calls it. Each fixture runs `run 0` under five processor grids -- 1 1 1 / 2 1 1 / 1 1 2 / 2 2 1 / 1 4 1
# (the default grid of a cubic box splits z, which hides an x/y-split pair change) -- and FAILS unless:
#   - every run exits 0 (no "no rank owning two fix-group atoms" refusal, the earlier np4 failure),
#   - the logged recip_self value is present and identical to the 1x1x1 run (a missing value FAILS: never vacuous),
#   - every charge agrees with the 1x1x1 run to 1e-9 e (MPI reduction order forbids %.17g equality).
# Fixtures are the goldens with the orientation PINNED: `create_atoms ... mol` without `rotate` draws each molecule's
# orientation from RanMars(seed + comm->me) (src/create_atoms.cpp:365), so an unpinned golden is a DIFFERENT geometry at
# np 2 (ledger L51). The data-file fixtures need no pin.
#   spcfq_gas_ewald, ionfield_cl_water  3-atom groups: the earlier tiny-group legacy fallback (probe route after )
#   metal_slab_gself                    240-atom Au lattice, G2-window miss -> legacy pair under the default K=16
#   shield_special00                    375-atom liquid water, G2-window miss -> legacy pair; the earlier pick moved q by
#                                       0.056 e with `atom_modify sort 0` alone. Its extra arm `nosort` (1x1x1, sort 0)
#                                       must match too: the pair may depend on neither decomposition nor sort order.
# Pre-binaries FAIL this check (spcfq_gas_ewald/ionfield_cl_water at 4xy, a refusal at 4y; shield_special00 nosort).
# Usage: ./npinv.sh            LMP=/path/to/lmp ./npinv.sh
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LMP="${LMP:-$HOME/codes/lammps/lammps-dev2/build/lmp_parallel}"
[ -x "$LMP" ] || { echo "ERROR: LAMMPS binary not found: $LMP" >&2; exit 2; }
CASES="$HERE/../cases"; W="$(mktemp -d)"; fail=0
declare -A GRID=( [1]="1 1 1" [2x]="2 1 1" [2z]="1 1 2" [4xy]="2 2 1" [4y]="1 4 1" [nosort]="1 1 1" )

maxdq() {   # max |q_a - q_b| over matching ids; inf if the id sets differ
  python3 - "$1" "$2" <<'PY'
import sys
def q(p):
    L = open(p).read().split('\n'); i = [k for k, l in enumerate(L) if l.startswith('ITEM: ATOMS')][-1]
    return {int(l.split()[0]): float(l.split()[2]) for l in L[i+1:] if l.strip()}
a, b = q(sys.argv[1]), q(sys.argv[2])
print(max(abs(a[k] - b[k]) for k in a) if a.keys() == b.keys() else float('inf'))
PY
}

for fx in spcfq_gas_ewald ionfield_cl_water metal_slab_gself shield_special00; do
  grids="1 2x 2z 4xy 4y"; [ "$fx" = shield_special00 ] && grids="$grids nosort"
  ref=""; refrs=""; ffail=0
  for g in $grids; do
    np=1; case "$g" in 2*) np=2;; 4*) np=4;; esac
    d="$W/${fx}_$g"; mkdir -p "$d"; cp "$CASES/$fx"/* "$d"/ 2>/dev/null; rm -rf "$d/golden"
    sed -i -e 's|^\(create_atoms .* mol [A-Za-z0-9_]* [0-9]*\) *$|\1 rotate 0.0 0.0 0.0 1.0|' \
           -e "0,/^units/s|^units\(.*\)$|units\1\nprocessors ${GRID[$g]}|" "$d/in.lammps"
    [ "$g" = nosort ] && sed -i '0,/^read_data/s|^\(read_data.*\)$|\1\natom_modify     sort 0 0.0|' "$d/in.lammps"
    ( cd "$d" && timeout -k 30 600 mpirun -np "$np" --bind-to none "$LMP" -in in.lammps > run.out 2>&1 ); rc=$?
    rs=$(grep -m1 -oE 'recip_self=[0-9.]+' "$d/run.out")
    if [ $rc -ne 0 ] || [ ! -s "$d/charges.dump" ]; then
      echo "FAIL npinv $fx@$g: rc $rc $(grep -m1 ERROR "$d/run.out" | cut -c1-110)"; ffail=1; continue; fi
    if [ -z "$rs" ]; then echo "FAIL npinv $fx@$g: no recip_self line (vacuous)"; ffail=1; continue; fi
    if [ -z "$ref" ]; then ref="$d/charges.dump"; refrs="$rs"; continue; fi
    [ "$rs" = "$refrs" ] || { echo "FAIL npinv $fx@$g: $rs != 1x1x1 $refrs"; ffail=1; }
    dq=$(maxdq "$ref" "$d/charges.dump")
    awk -v x="$dq" 'BEGIN{exit !(x<=1e-9)}' || { echo "FAIL npinv $fx@$g: max|dq| $dq > 1e-9 vs 1x1x1"; ffail=1; }
  done
  [ $ffail -eq 0 ] && echo "PASS npinv $fx: grids ${grids// //}, $refrs" || fail=1
done
rm -rf "$W"; exit $fail
