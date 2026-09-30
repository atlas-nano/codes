#!/usr/bin/env python3
"""Compare the samQEq run of a system with its native ReaxFF reference: per-atom charges, forces and the
ReaxFF energy. Usage: python3 compare.py rdx|water  (run in.<sys>_reference and in.<sys>_samqeq first)."""
import sys


def dump(p):
    rows, on = {}, False
    for l in open(p):
        if l.startswith("ITEM: ATOMS"):
            on = True; continue
        if on:
            a = l.split(); rows[int(a[0])] = [float(x) for x in a[2:6]]
    return rows


def energy(p):
    return float(open(p).read().split()[1])


s = sys.argv[1]
ref, sam = dump(f"charges_{s}_reference.dump"), dump(f"charges_{s}_samqeq.dump")
assert set(ref) == set(sam)
dq = max(abs(sam[k][0] - ref[k][0]) for k in ref)
df = max(abs(sam[k][j] - ref[k][j]) for k in ref for j in (1, 2, 3))
de = abs(energy(f"energies_{s}_samqeq.txt") - energy(f"energies_{s}_reference.txt"))
print(f"{s}: N = {len(ref)}  max|dq| = {dq:.2e} e  max|dF| = {df:.2e} kcal/mol/A  |dE_reaxff| = {de:.2e} kcal/mol")
