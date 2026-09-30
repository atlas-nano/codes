spcfq_xl_staircase — B3 (XL completion) regression case.

Same 64-water SPC-FQ / rigid-molecule / XL setup as ../spcfq_xl/, but the O-type param
line carries nonzero columns 8/9 (c3_type=2.0, c4_type=8.0 eV-units): the #13 Phase B
per-type IP-staircase on-site anharmonicity. Nonzero c3_type/c4_type auto-enable
lr_quartic at param load (fix_qeq_sam.cpp ~line 433), which under `fix_modify xl`
exercises three of the audit's Tier-B XL fixes together:

  - B3.1 the staircase restoring force in xl_chargeforce() (fix_qeq_sam_xl.cpp) —
    previously MISSING entirely for XL (audit finding #5): the diagonal stiffened via
    lr_quartic but no -dE/dq force was ever applied, so XL propagated staircase-carrying
    atoms on the wrong PES.
  - B3.2 the quartic diagonal fill (apply_quartic_eta(), called from final_integrate
    before xl_chargeforce/qeq_matvec), so the diagonal is current in XL mode whenever
    lr_quartic is active (audit finding #6).
  - (B3.3 the field gate is NOT exercised here: no `fix_modify quartic` group/gate is
    set, so quartic_fld0=0 and compute_quartic_gate() is skipped — gate stays the
    byte-identical no-op value 1.0. Covering the gate-recompute path needs a second,
    ion-shell-bearing case; out of scope for this staircase-only regression.)

No golden/ directory is checked in yet — the orchestrator generates it after the next
build (see tests/run_tests.sh --update). The values (c3=2.0, c4=8.0) were chosen to be
large enough to visibly shift the O charge trajectory over the 10-step NVE run while
staying well inside the stable/bounded XL regime (no near-crit blowup expected).
