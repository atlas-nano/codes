spcfq_xl_staircase — XL with the on-site staircase anharmonicity.

Same 64-water SPC-FQ / rigid-molecule / XL setup as ../spcfq_xl/, but the O-type param
line carries nonzero columns 8/9 (c3_type=2.0, c4_type=8.0 eV-units): the per-type
IP-staircase on-site anharmonicity. Nonzero c3_type/c4_type enable lr_quartic at param
load (fix_qeq_sam.cpp), which under `fix_modify xl` exercises:

  - the staircase restoring force -dE/dq in xl_chargeforce() (fix_qeq_sam_xl.cpp), so XL
    propagates staircase-carrying atoms on the correct PES;
  - the quartic diagonal fill (apply_quartic_eta(), called from final_integrate
    before xl_chargeforce/qeq_matvec), so the diagonal is current in XL mode whenever
    lr_quartic is active.

The field gate is NOT exercised here: no `fix_modify quartic` group/gate is set, so
quartic_fld0=0 and compute_quartic_gate() is skipped (gate = 1.0).

The values (c3=2.0, c4=8.0) are large enough to visibly shift the O charge trajectory
over the 10-step NVE run while staying well inside the stable/bounded XL regime.
