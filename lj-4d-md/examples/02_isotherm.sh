#!/bin/bash
# A T*=3.0 isotherm: dilute gas -> dense fluid (hc lattice, N=1296) and the
# high-pressure D4 crystal branch (d4 lattice, N=4096).  Maps where the 3PT/TI
# divergence builds; the fluid branch also gives P,U for the EOS-integration TI.
#
# Pass T as $1 (default 3.0).  Production used nprod~40000; shorter here for a demo.
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MD="$ROOT/engine/md4d"
T="${1:-3.0}"
OUT="$ROOT/runs/example_isotherm_T${T}"
mkdir -p "$OUT"
COMMON="rc=2.5 dt=0.00371 thermostat=1 tdamp=0.1 nequil=6000 nprod=12000 thermoevery=500 velstride=1"

echo "=== FLUID branch (hc lattice, N=1296), T*=$T ==="
for RHO in 0.10 0.30 0.50 0.70 0.90 1.10 1.30; do
  echo "  fluid rho*=$RHO"
  OMP_NUM_THREADS="${OMP_NUM_THREADS:-15}" $MD $COMMON \
      n=6 lattice=hc rho=$RHO T=$T prefix="$OUT/r${RHO}" > "$OUT/r${RHO}.stdout" 2>&1
done

echo "=== SOLID branch (D4 lattice, N=4096), T*=$T ==="
for RHO in 1.50 1.70 1.90 2.10; do
  echo "  solid rho*=$RHO"
  OMP_NUM_THREADS="${OMP_NUM_THREADS:-15}" $MD $COMMON \
      n=8 lattice=d4 rho=$RHO T=$T prefix="$OUT/s${RHO}" > "$OUT/s${RHO}.stdout" 2>&1
done

echo "=== isotherm T*=$T done -> $OUT ==="
echo "Analyse each state with py-xPT (see examples/pyxpt_d4.ini)"
