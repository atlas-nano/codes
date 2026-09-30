# metal_slab_gself — validation record (2026-08-23)

Pins `fix_modify gself`: the finite-width charge self-energy K_e·sqrt(2α/π) folded into η from eta0
at init(). 240-atom Au(111) slab, PQEq kernel, full Ewald (pinned mesh), E = 0.5 V/Å, η_Au = 5.172
(the physical Thomas–Fermi hardness, l_TF = 0.72 Å — NOT touched), Gaussian width w = 0.5 Å (η⁻¹ = w√2 = 0.71 Å in Scalfi's
convention; Scalfi et al. used η⁻¹ = 0.56 Å).

Controls run before freezing:
- **Negative control (the golden depends on the term):** same deck, `gself off`:
    layer OFF ON
      1 -2.1707 -1.0562
      2 +1.4896 -0.1690 <- staggered (OFF) vs monotone (ON)
      3 -0.0224 +0.0001
      4 -1.4479 +0.1703
      5 +2.1514 +1.0548
  max|q_on − q_off| = 3.5e-2 e per atom. Gauss limit for this cell/field = 0.994 e per face.
- **Conductor limit (Scalfi JCP 153,174704 Fig 2a):** with gself on, η = 5.172 / 2.5 / 1.0 / 0.3 gives
  face −1.056 / −1.105 / −1.137 / −1.154 and interior −0.169 / −0.127 / −0.095 / −0.078: monotone at
  every η, interior → 0 as the screening length shortens. Without gself the same sweep reads
  1.8× / 2.5× / 3.9× / 5.6× Gauss, staggered, diverging.
- **Exactness of the fold:** at the default width (per-type PQEq Rc/√λ, E_self = 3.416 eV) the real knob
  reproduces a hand-emulation through η to every printed digit.
- The rest of the suite (32 cases) is byte-identical with gself off (default).

(First freeze attempt captured the negative-control run's pe.txt (2.680) with the ON charges — my copy
ordering; regenerated from a clean ON run: pe = 2.22754439581523, verified twice.)
- **2026-09-16 re-baseline (pe.txt only):** the reported energy now includes gself's E_self
  (gself_energy_20260916). pe moved 2.35068898731 -> 2.73755775570, i.e.
  +0.38686876839 eV = ½·E_self(0.50)·Σq² from the golden charges (predicted 0.38686876840). charges.dump unchanged.
