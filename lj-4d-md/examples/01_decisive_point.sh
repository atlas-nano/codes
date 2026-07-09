#!/bin/bash
# Run the decisive 4D-LJ state: rho*=1.10, T*=1.0 (strongly caged, gamma/Omega0~3.2).
# N = 6^4 = 1296, force-shifted LJ at rc=2.5, NVT.  Velocities every step for 3PT.
#
# Production used nequil=20000 nprod=40000 (~prod_r110_T10 in runs/); this example
# uses a shorter run for a quick demo — lengthen for publication-quality statistics.
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MD="$ROOT/engine/md4d"
OUT="$ROOT/runs/example_r110_T10"
mkdir -p "$(dirname "$OUT")"

OMP_NUM_THREADS="${OMP_NUM_THREADS:-15}" "$MD" \
    n=6 lattice=hc rho=1.10 T=1.0 rc=2.5 dt=0.00371 \
    thermostat=1 prodthermostat=1 tdamp=0.1 \
    nequil=20000 nprod=40000 thermoevery=500 velstride=1 \
    prefix="$OUT" | tee "$OUT.stdout"

echo "=== wrote $OUT.vel / .meta / .thermo ==="
echo "Analyse with:  python -m pyxpt examples/pyxpt_d4.ini   (edit trajectory= to $OUT)"
