# Provenance

This public release of **samQEq** (version 1.0.0) was cut from the private development
repository `atlas-nano/samQEq`, branch `dev`, at commit

    804b1f318679ef0d2d4fcb2f46207f6986e3a40e
    (804b1f3, 2026-09-26)

with the release patch `capsule_all_vs_804b1f3.patch` (md5 4322177bc23f4e5b72af12ddb491e8f1) applied,
by `internal/make_release.py`, which is the only supported way to produce it.

## What differs from that commit

Nothing functional. Internal working references were removed from comments and
from runtime message strings: session tags, ledger numbers, and pointers to
design documents that do not ship. 1506 such references were removed across
70 files, and the advice around each was left in place.

The script refuses to write a release unless both of the following hold, and
they held for this one:

- **No forbidden token survives.** No `internal/*.md` path, no bare
  `SPEC_*.md` / `SCOPE_*.md` / `HANDOFF_*.md`, no `RESULTS_M0`, no ledger
  number, no session tag, anywhere in any shipped file.
- **No code changed.** For every source, the token stream outside comments and
  string contents is byte-identical to that commit with the patch applied, checked with a
  comment- and string-aware scanner. Comments and message text may differ;
  code may not.

Changed message strings are a genuine change to what a user sees, so the
release must be build-validated before publication.

## Contents

| | |
|---|---|
| `src/` | the samQEq LAMMPS package |
| `tests/` | the bit-correctness suite and its runner |
| `kokkos/` | the Kokkos device styles (copy only into a Kokkos build) |
| `examples/` | runnable decks with reference output |
| `doc/` | usage notes |
| `patches/` | the three required host-tree patches |

## The patches are not optional

Apply each file in `patches/` with `patch -p1` from the LAMMPS root (README, "Build").
`cmake_register_samqeq.patch` registers the package with CMake; without it the build
completes and the binary carries none of the package's styles.
`kspace_pppm_peratom_brick_zeroing.patch` zeroes the per-atom bricks in
`PPPM::allocate_peratom()`. Stock LAMMPS never needs it, because
`poisson_peratom()` fills those cells before anything reads them. samQEq breaks
that ordering: its first `compute_vector()` runs in the fix's
`setup_pre_force`, before any `PPPM::compute()`. `fix_efield-friend.patch` lets the
fix read an applied field. A LAMMPS version bump can silently drop any of them.
