# slab_ewald3dc — validation notes

Pins the EW3DC (Yeh–Berkowitz) slab correction in `src/pppm_samqeq.cpp` and the
slab-aware recip_self calibration in `src/fix_qeq_sam_lr.cpp`.

## Physics validation beyond bit-pinning (by hand)
- **Solve↔force consistency:** run a longer NVE (e.g. 5–20 ps) of
  this slab; total energy (pe + fix chg self-energy, `f_chg`) must not drift
  systematically. A drift that appears ONLY with `kspace_modify slab` on would
  indicate the compute_vector dipole term and PPPM::slabcorr() disagree.
- **Convergence to the open-boundary limit:** compare per-atom forces/solved
  charges of this deck against the SAME slab in a 3d-periodic box with a very
  large vacuum gap along z (e.g. zhi stretched 5–10x, boundary p p p, no slab
  keyword). EW3DC should reproduce that limit to ~the PPPM accuracy as the
  volfactor/gap grow.
- **Symmetry:** for a config mirror-symmetric about the slab midplane, M_z of
  the solved charges should be ~0 and the slabcorr energy ~0; f_z on mirror
  atom pairs should be equal/opposite.
- **Calibration:** with `kspace_modify slab` the log line
  `samqeq: calibrated grid reciprocal self-term recip_self=...` should be close
  (~few 0.01 eV/e) to the value of the same system without slab in the
  vacuum-extended box; the (2π/V)(z_A−z_B)² probe contamination is subtracted
  analytically in calibrate_recip_self. Without that subtraction recip_self
  shifts by exactly that amount and the QEq diagonal goes off-eta.
