#!/usr/bin/env python3
"""Numeric diff of two LAMMPS dump files (or any whitespace tables under an
'ITEM: ATOMS' header). Rows are matched by the first column (atom id), then
every field is compared: numeric fields by abs/rel tolerance, string fields
exactly. Exit 0 (PASS) iff every field is within tolerance.

Usage: numdiff.py golden.dump new.dump [rtol] [atol]
Defaults rtol=1e-9 atol=1e-12 (effectively bit-for-bit on a fixed machine/np;
np=1 LAMMPS runs are deterministic, so a passing diff means the code change did
not perturb the result).
"""
import sys


def load(path):
    """Return (rows, is_dump). Dump mode: the table under 'ITEM: ATOMS' (rows
    matched by atom id). Flat mode (no such header, e.g. a pe.txt scalar): every
    non-empty, non-comment line as a row, compared positionally."""
    with open(path) as f:
        lines = f.readlines()
    start = None
    for i, l in enumerate(lines):
        if l.startswith("ITEM: ATOMS"):
            start = i + 1
            break
    if start is None:
        rows = [l.split() for l in lines if l.strip() and not l.lstrip().startswith("#")]
        return rows, False
    rows = []
    for l in lines[start:]:
        if l.startswith("ITEM:"):
            break
        parts = l.split()
        if parts:
            rows.append(parts)
    return rows, True


def main():
    if len(sys.argv) < 3:
        sys.exit("usage: numdiff.py golden.dump new.dump [rtol] [atol]")
    golden, new = sys.argv[1], sys.argv[2]
    rtol = float(sys.argv[3]) if len(sys.argv) > 3 else 1e-9
    atol = float(sys.argv[4]) if len(sys.argv) > 4 else 1e-12
    (a, a_dump), (b, b_dump) = load(golden), load(new)
    if len(a) != len(b):
        print(f"row count differs: {len(a)} (golden) vs {len(b)} (new)")
        sys.exit(1)
    if a_dump and b_dump:        # dump files: match rows by atom id (col 0)
        a.sort(key=lambda r: int(r[0]))
        b.sort(key=lambda r: int(r[0]))
    maxabs = maxrel = 0.0
    worst = None
    for ra, rb in zip(a, b):
        if len(ra) != len(rb):
            print(f"column count differs at id {ra[0]}")
            sys.exit(1)
        for ca, cb in zip(ra, rb):
            try:
                fa, fb = float(ca), float(cb)
            except ValueError:
                if ca != cb:
                    print(f"string field differs at id {ra[0]}: {ca} != {cb}")
                    sys.exit(1)
                continue
            d = abs(fa - fb)
            denom = max(abs(fa), abs(fb))
            rel = d / denom if denom > 0 else 0.0
            if d > maxabs:
                maxabs = d
            if rel > maxrel:
                maxrel = rel
            if d > atol + rtol * denom:
                worst = (ra[0], ca, cb, d, rel)
    if worst:
        print(f"max abs {maxabs:.3e} rel {maxrel:.3e}; "
              f"worst id {worst[0]}: {worst[1]} vs {worst[2]} "
              f"(abs {worst[3]:.3e} rel {worst[4]:.3e}) > rtol {rtol:g} atol {atol:g}")
        sys.exit(1)
    print(f"max abs {maxabs:.3e} rel {maxrel:.3e} (rtol {rtol:g} atol {atol:g})")
    sys.exit(0)


if __name__ == "__main__":
    main()
