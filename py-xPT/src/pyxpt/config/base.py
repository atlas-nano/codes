# -*- coding: utf-8 -*-
"""Config dataclass and validation for py-xPT."""
from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field

from pyxpt.constants import ModexPT

log = logging.getLogger(__name__)

# ── Velocity scale presets ────────────────────────────────────────────────────

_VEL_SCALE_PRESETS: dict[str, float] = {
    "LAMMPS": 1000.0,
    "NAMD":   20.45482706,
    "CP2K":   2188491.52e-2,
}

TOPOLOGY_FORMAT_MAP: dict[str, str] = {
    "LAMMPS":  "DATA", "LMPDATA": "DATA", "BGF":     "BGF",
    "AMBER":   "PRMTOP", "PRMTOP": "PRMTOP", "PDB":  "PDB",
    "GRO":     "GRO", "PSF":      "PSF",
}

TRAJECTORY_FORMAT_MAP: dict[str, str] = {
    "LAMMPS":     "LAMMPSDUMP", "LMPTRJ":    "LAMMPSDUMP", "LAMMPSDUMP": "LAMMPSDUMP",
    "AMBER":      "NCDF", "AMBERTRJ":  "TRJ", "CHARMM":    "DCD",
    "CHARMMTRJ":  "DCD", "DCD":       "DCD", "XTC":       "XTC",
    "TRR":        "TRR", "XYZ":       "XYZ",
    "DLPOLY":     "DLPOLY", "DL_POLY": "DLPOLY", "DLPOLYSPLIT": "DLPOLY",
}


# recipes (minimal / p1 / p1_spec / mr_safe) were bandwidth for users
# navigating the 15-knob INI surface; after consolidation the surface
# is ~5 user-facing knobs and presets add no value.


# ── Config dataclass ──────────────────────────────────────────────────────────

@dataclass
class Config:
    # [files]
    topology: str = ""
    topology_format: str = ""
    trajectory: list[str] = field(default_factory=list)
    trajectory_format: str = ""
    # Separate velocity file (trajectory_format = DLPOLY): positions/cell come
    # from `trajectory`, velocities from `velocity_file`, read frame-locked.
    velocity_file: str = ""
    group_file: str = "none"

    # [frames]
    start: int = 1
    stop: int = 0
    step: int = 1

    # [thermodynamics]  (aliases: [xpt], [2pt], [analysis])
    # Publishable taxonomy: mode = 1PT | 2PT | 3PT.  The 2PT gas/solid treatment
    # is selected by `refinement`; 3PT implies rigorous + cage.
    mode: ModexPT = ModexPT.TWO_PT
    # 2PT refinement (mode = 2PT or 3PT):
    #   rigorous   — rigorous-HS Sackur-Tetrode + CS excess (no ln z) [default]
    #   lin2003    — empirical Lin-Goddard +ln z convention (literature comparison)
    #   desjarlais — Desjarlais 2013 moment-matched memory-function gas DoS
    #   r2pt       — Sun 2017 revised 2PT (δ-tuned fluidicity, F_a-inclusive)
    refinement: str = "rigorous"
    # R2PT gas-fraction exponent δ (Sun 2017, f^δ = D/D₀); 1.5 = LJ-Ar/metals default.
    r2pt_delta: float = 1.5
    molecular: bool = False
    corlen: float = 0.5
    mol_linear: int = 0
    mol_rotsym: int = 2
    constraints: int = 0
    per_molecule: bool = False
    check_grp_eng: bool = True
    vel_scale: float = 1.0
    lammps_units: str = ""
    use_vrotat: bool = False
    use_ind: bool = True
    use_gpu: bool = False
    apodization: str = "none"
    apodization_alpha: float = 0.5
    hs_eos: str = "cs"
    hs_entropy: str = "rigorous"
    # 3 = bulk (default; identical to all prior behavior).  2 = 2D slit/interface
    # diffusion, 1 = 1D channel: the fluidicity closure, hard-sphere excess
    # entropy, Sackur-Tetrode gas weight, and gas-Lorentzian sum rule are computed
    # in d dimensions (Henderson hard-disk for d=2, Tonks rod for d=1).
    # DECOUPLED from the total velocity-DoF count, which is auto-detected from the
    # per-axis velocity variance: a slit/channel has 3 velocity DoF (the confined
    # axis is a thermal vibrational/solid DoF) but 2D/1D diffusion → dimension=2/1
    # with 3 DoF; a genuinely-2D run (v_z≡0) has 2 DoF and dimension=2.  Native
    # confined-fluid entropy (cf. PNAS 108, 11794; JPCL 12, 9162).
    dimension: int = 3
    # ΔS = (1/3)∫cage·(1−w)·(W_g−W_s)dν to the rigorous-HS entropy, recovering the
    # solid-side deficit of rigorous-HS in structured liquids (liquid metals, dense
    # LJ).  Designed for hs_entropy=rigorous on monatomic groups; off by default.
    cage_entropy: bool = False
    # Rotational analogue of cage_entropy: applies the cage-memory machinery to
    # the ROTATIONAL channel of molecular liquids, with the rotational "gas"
    # weight being the free/rigid-rotor weight (NOT hard-sphere) and prefactor
    # p_rot = 1/d_rot.  Off by default; independent of cage_entropy.
    cage_entropy_rot: bool = False
    # Volterra-kernel noise-floor / truncation controls for the cage memory
    # excess (defaults reproduce the published bare-cage path bit-for-bit).
    cage_nf_run: int = 1          # consecutive sub-noise-floor VACF lags before truncation
    cage_taper: str = "none"      # none|hann half-Hann taper on the truncated kernel
    cage_tail_tol: float = 1.0    # auto-cutoff -> main-lobe fallback tolerance
    # Debye gas-gate: zero the fluidicity in (near-)crystalline channels, where a
    # ~ν² low-frequency DoS would otherwise be credited spurious diffusive entropy.
    # "none" (default) = off; "debye" applies the gate; "debye_warn" only reports.
    gas_gate: str = "none"
    gas_gate_tol: float = 0.01
    # ── HS reference diffusivity scale (P4, doc option C) ────────────────────
    # The 2PT fluidicity f = D / D₀^HS uses Lin 2003's analytic Chapman-Enskog
    # formula for D₀^HS implicitly inside the K parameter of search_xpt(K).
    # An alternative (option C) replaces this analytic D₀ with
    # a value measured from a separate dilute-MD or HS-MD simulation:
    #     D₀^GK_user = S_HS(0) / (12·m·N·k_B·T)
    # Since the engine doesn't carry the analytic D₀^HS explicitly (it's
    # encoded in the K formula), we expose a DIMENSIONLESS scale factor:
    #     d0_hs_scale = D₀^user / D₀^Enskog
    # which the engine applies as  K → K / d0_hs_scale  in the f-solve.
    # The user computes the ratio externally:
    #   1. Run a separate HS-MD at the matched (T*, ρ*) state point.
    #   2. Extract D_MD from the VACF integral or MSD slope.
    #   3. Compute D₀^Enskog analytically: 3/(8·ρ·σ²)·√(k_B·T/(π·m)).
    #   4. Set d0_hs_scale = D_MD / D₀^Enskog.
    # Default 1.0 preserves Enskog-CE behaviour.  Mostly a sanity-check knob
    # — for monoatomic LJ-Ar the empirical Enskog D₀ is well-validated.
    d0_hs_scale: float = 1.0

    # ── F1 strict-stationarity opt-in ────────────────────────────────────────
    # When True, the post-hoc stationarity diagnostics (block-KE drift > 5 %
    # and VACF tail > 10 % at t_max/2) become RuntimeErrors instead of
    # warnings.  Use in batch / CI runs where you want to fail loudly on
    # bad-trajectory inputs.  Default False preserves the legacy warn-and-
    # continue behaviour for interactive analysis.
    strict_stationarity: bool = False

    use_sim_z: bool = False
    fixed_fluidicity: list[float] = field(default_factory=list)

    # ── System-COM velocity subtraction (Phase MR-DRIFT) ─────────────────────
    # In an equilibrium MD trajectory the centre-of-mass velocity of the
    # whole system should be zero — but NPT runs without ``fix momentum`` /
    # ``fix recenter`` accumulate a non-zero net drift over time.  That drift
    # contaminates the velocity ACF as a (nearly) constant offset
    # ``M_total · ⟨|v_sys|²⟩`` that integrates linearly with τ, dominating
    # the apparent VACF for any window beyond a few ps and inflating the
    # extracted D by 1–3 orders of magnitude on long trajectories.
    #
    # When ``subtract_com_velocity = True`` the engine subtracts
    # ``v_sys_COM(t) = (Σ_i m_i v_i(t)) / M_total`` from every atom's
    # velocity at every frame before any downstream processing.  This is
    # the velocity-space analog of LAMMPS's ``compute msd com=yes``.  Default
    # ON because it is correct for any equilibrium analysis; turn off
    # explicitly for non-equilibrium MD where the bulk drift carries
    # physical information (e.g. shear / electric-field-driven flow).
    subtract_com_velocity: bool = True

    # ── Finite-size correction on the 2PT fluidicity (Yeh-Hummer) ────────────
    # The Lin-Goddard cubic was derived for an infinite hard-sphere reference,
    # but the diffusion enters via S(0) computed under PBC.  Yeh & Hummer
    # (J. Phys. Chem. B 108, 15873, 2004) showed D_PBC = D_inf − ξ k_B T /
    # (6π η L) with ξ = 2.837297 for cubic boxes.  Correcting D_PBC upward
    # (equivalently, scaling K = K(D)) before solve gives a consistent f.
    #   "off"          legacy behavior (default for backwards compatibility)
    #   "yeh-hummer"   require viscosity (auto from stress or fsc_viscosity)
    #   "auto"         apply if viscosity is available; warn-and-skip otherwise
    finite_size_correction: str = "off"
    # User-supplied shear viscosity for FSC, in Pa·s.  When > 0 this value is
    # used directly; when ≤ 0 the engine computes η inline from the stress
    # accumulator.  Set this if disable_xpt is on or no stress data is dumped.
    fsc_viscosity: float = -1.0

    # ── [multires] — multi-resolution VACF stitching (Phase MR1) ──────────────
    # Populated by config.multires.parse_multires; default is a disabled stub.
    # The engine wire-up (Phase MR2) reads cfg.multires.enabled to switch into
    # multi-trajectory accumulation.  When None or disabled, the legacy
    # single-trajectory pipeline runs unchanged.
    multires: object = None
    # Phase MEM-B: skip the 2PT thermodynamic pipeline entirely.  When True
    # the engine does not allocate the per-atom velocity time series _vacvv
    # nor the per-molecule angular-velocity / dipole arrays, and skips
    # _compute_vac, _compute_thermo, the .thermo/.vac/.pwr/.3n
    # writers, and any 2PT-only stationarity checks.  Use this for
    # mechanics-only or transport-only runs on large trajectories where the
    # ~5 GB of 2PT working memory is not affordable.  Auto-disables
    # transport_diffusion, transport_rotation
    # (which need per-atom or per-molecule velocity time series), and
    # induced-dipole contributions to ε(0).
    disable_xpt: bool = False

    # ── Phase 1 restructure: explicit per-module ``enabled`` flags ───────────
    # Each major analysis module gets an explicit on/off switch.  The
    # ``transport_enabled`` / ``mechanics_enabled`` flags default to
    # ``None`` and are auto-derived in :func:`validate`
    # from the per-property flags below them (so existing INI files keep
    # working with no change), but a user can force-disable a module
    # by setting ``enabled = 0`` in the corresponding section.
    #
    # ``thermodynamics_enabled`` is True by default (most users want 2PT).
    # Setting either ``[thermodynamics] enabled = 0`` or ``disable_xpt = 1``
    # turns it off.  The two flags are kept synonymous.
    #
    # At startup :func:`validate` checks that AT LEAST ONE module is
    # enabled and raises ValueError otherwise.
    # ``thermodynamics_enabled`` follows the same auto-derivation pattern as
    # the other module flags: ``None`` means "user did not set it" and
    # :func:`validate` resolves it from context (presence of [thermodynamics]
    # properties, presence of other-module activations, ``disable_xpt``).
    thermodynamics_enabled: bool | None = None
    transport_enabled: bool | None = None
    mechanics_enabled: bool | None = None

    # ── [memory] — RAM-budget driven atom batching (Phase MEM-R1) ────────────
    # When the engine's estimated peak memory exceeds the budget, atoms (or
    # molecules, when molecular=1) are split into batches and the trajectory
    # is re-read once per batch.  The same ``vac_sum`` accumulator is summed
    # across batches before _compute_postvac runs, so results are bit-
    # identical (modulo float32 sum order) to the single-pass path.
    #
    #   ram_budget_gb        explicit budget in GB (overrides fraction)
    #   ram_budget_fraction  fraction of available system RAM (default 0.5)
    #   single_pass_only     raise MemoryError instead of multi-pass
    ram_budget_gb: float = 0.0
    ram_budget_fraction: float = 0.5
    single_pass_only: bool = False

    # ── [transport] — Green-Kubo transport coefficients ───────────────────────
    # All per-property flags default to False; the section is opt-in.
    transport_diffusion: bool = False                 # translational D (cm²/s)
    transport_rotation: bool = False                  # rotational D (1/ps; molecular)
    transport_shear_viscosity: bool = False           # η_shear (Pa·s)
    transport_bulk_viscosity: bool = False            # η_bulk (Phase T4)
    transport_electrical_conductivity: bool = False   # σ_el (Phase T2)
    # σ_el current convention.  True → molecular current J = Σ_m Q_m · v_m_COM
    # (proper σ_DC for electrolyte solutions; neutral solvents drop out by
    # construction).  False → atomic current J = Σ_i q_i · v_i (full-spectrum
    # σ(ω); includes solvent dipole-flux contributions).  Default True since
    # σ_DC is the impedance-comparison value most users expect.  See
    transport_electrical_molecular_current: bool = True
    transport_thermal_conductivity: bool = False      # κ (Phase T3)
    transport_dielectric: bool = False                # ε(0), ε(ω) (Phase T4)
    transport_averaging: str = "block"     # none | block | bootstrap | block-bootstrap
    transport_n_blocks: int = 5
    transport_n_bootstrap: int = 200
    transport_report_uncertainty: bool = True
    transport_burn_in_frames: int = 0
    # Fraction of each block over which to integrate the flux ACF.  GK
    # integrals over the full block accumulate long-time noise that tends
    # to cancel against the peak in the trapezoid endpoint correction.
    # Truncating to ~50 % of the block (the default) discards the noisy
    # tail and gives a stable estimate.  Set higher (≤1) for very slow
    # transport (e.g. dense liquids near Tg); lower for very fast.
    transport_acf_corlen: float = 0.5
    # κ-specific truncation: the heat-flux ACF for atomic / simple liquids
    # decays much faster than the velocity or stress ACF, so a global
    # ``corlen`` of 0.5 over-integrates noise for κ.  When
    # ``transport_kappa_truncation == "auto"`` we instead integrate up to
    # the first negative excursion of the ACF beyond ``min_lag_ps``
    # (Helfand-plateau).  Set to "manual" to fall back to corlen × nf.
    transport_kappa_truncation: str = "auto"
    # ``min_lag_ps`` should cover the slowest vibrational period contributing
    # to the heat-flux ACF.  0.5 ps skips C–H stretches (~10 fs), bond
    # bends and ring-mode oscillations (50–200 fs), which otherwise produce
    # spurious early zeros in the ACF.  Atomic / monatomic systems are
    # unaffected (their ACF first zero is typically beyond ~1 ps).
    transport_kappa_min_lag_ps: float = 0.5

    # ── [mechanics] — Frequency-dependent linear response and elastic constants ─
    # All flags opt-in (default off).  Phase M1 covers DMA + bulk DMA + ε(ω);
    # Phase M2 will add the C_ij stress-fluctuation method and derived properties.
    mechanics_dma_shear: bool = False           # G'(ω), G''(ω), tan δ_s, G_∞, τ_M
    mechanics_dma_bulk: bool = False            # K'(ω), K''(ω), tan δ_b, K_∞, τ_K
    mechanics_dielectric_spectrum: bool = False  # ε'(ω), ε''(ω), tan δ_ε
    # Phase M2 — elastic constants and derived properties.
    mechanics_elastic_constants: bool = False   # full 6×6 C_ij tensor [Pa]
    # Method choice.  "auto" picks strain-fluctuation when the trajectory is
    # NPT (relative box-volume std > 1e-4) and stress-fluctuation otherwise.
    # "strain_fluct" forces NPT-style box-fluctuation analysis; "stress_fluct"
    # forces the NVT-style stress-covariance analysis (which omits the Born
    # hessian and is therefore a *partial* C_ij — see HOWTO).
    mechanics_elastic_method: str = "auto"      # auto | strain_fluct | stress_fluct
    mechanics_vrh_averages: bool = True         # Voigt-Reuss-Hill K, G, E, ν
    mechanics_sound_velocity: bool = True       # v_L, v_T, v_avg from C_ij
    mechanics_anisotropy: bool = True           # Zener A_Z, universal A^U
    mechanics_born_stability: bool = True       # eigenvalues of C > 0?
    mechanics_debye_temperature: bool = True    # Θ_D from sound-velocity avg
    mechanics_pugh_ratio: bool = True           # G/K (ductility indicator)
    # Phase M3b — multi-mode Maxwell (Prony-series) fit to G(t)/K(t)
    mechanics_maxwell_fit: bool = False         # fit G(t) and K(t) to Σ Gᵢ·exp(-t/τᵢ)
    mechanics_maxwell_n_modes: int = 8          # number of log-spaced τ grid points
    mechanics_maxwell_include_G_inf: bool = False  # fit non-zero G_∞ (solids only)
    mechanics_maxwell_regularization: float = 1e-3  # Tikhonov L2 penalty (rel. to ||G||)
    # Statistical / discretization knobs.  When unset, mechanics inherits the
    # corresponding [transport] knob (acf_corlen, n_blocks, etc.) so a single
    # set of integration choices applies across both pipelines.
    mechanics_n_freqs: int = 200                 # log-spaced frequency points
    mechanics_acf_corlen: float = 0.5            # fraction of block to integrate

    # [system_properties]  (alias: [md])
    timestep: float = 0.001
    dump_freq: int = 0
    temperature: float = 0.0
    pressure: float = 0.0
    volume: float = 0.0
    void_volume: float = 0.0
    energy_avg: float = 0.0
    energy_std: float = 0.0
    energy_units: str = "kcal/mol"

    # ── Reference-pressure correction (Tier 1 NPT support) ──────────────────
    # When set, an additional column μ_q_at_ref_P = μ_q + V·(P_ref − P_sim)
    # is emitted in .thermo (extensive, kJ/mol/SimBox) where V is the group
    # volume and P_sim is the simulation's average pressure (from cfg.pressure
    # or the trajectory).  Units are *raw LAMMPS pressure units* (same as
    # cfg.pressure): atm for units=real, bar for units=metal, reduced units
    # for units=lj.  The engine converts both to GPa internally via
    # ``lammps_press_to_pa`` before applying the correction.
    #
    # Use case: cross-system comparisons in osmotic-pressure / activity work
    # where pure solvent and solution NVT productions may sit at very
    # different effective pressures (e.g. -300 atm vs -50 atm due to
    # incomplete NPT equilibration).  Set to None to disable.
    reference_pressure: float | None = None

    # ── Ensemble hint (Tier 2 NPT support) ──────────────────────────────────
    # Tells the engine how to derive the analysis state point from the
    # trajectory.  Affects how V (and to a lesser extent P) are extracted:
    #   "nvt"  (default): use cfg.volume / cfg.pressure if overridden in the
    #                     .ini, else use trajectory averages.  Backward-
    #                     compatible behaviour.
    #   "npt"            : FORCE per-frame V/P averages from the trajectory;
    #                     ignore any cfg.volume override.  Also enables a
    #                     diagnostic warning when std(V)/<V> > 5%, indicating
    #                     unexpectedly large volume fluctuations that may
    #                     invalidate scalar-V analysis assumptions.
    #   "auto"           : if std(V)/<V> > 1% across frames, behave as "npt";
    #                     otherwise behave as "nvt".
    ensemble: str = "nvt"

    # [output]
    prefix: str = "pyxpt.out"
    show_xpt_split: bool = False
    show_classical_thermo: bool = False
    normalize: bool = False
    out_units: str = "kj/mol"
    lj_sigma: float = 1.0
    lj_epsilon: float = 1.0
    lj_mass: float = 1.0
    # MC1 diagnostic: cross-channel velocity autocorrelators between
    # trans/rot/vib (writes <prefix>.cross_vac + .cross_pwr + summary log
    # of the dimensionless coupling ρ_αβ = max_t |trace C_αβ(t)| /
    # √(C_αα(0)·C_ββ(0)) per group).  Molecular runs only.
    cross_vac_diagnostic: bool = False

    # spdos sub-settings (formerly [spdos_raman])
    spdos_raman: bool = False
    spdos_nmd_file: str = ""
    spdos_raman_species: str = ""
    spdos_write_per_mode: bool = False
    spdos_auto_nmd: bool = False
    spdos_point_group: str = "C1"
    spdos_linear: bool = False
    spdos_auto_nmd_natoms: int = 0

    # hybrid sub-settings (formerly [hybrid_raman])
    hybrid_raman: bool = False
    hybrid_raman_epsilon: float = 0.01
    hybrid_raman_crossover: float = 2200.0
    hybrid_raman_taper: float = 150.0
    hybrid_raman_smooth_sigma: float = 10.0
    hybrid_raman_floor_window: float = 150.0
    hybrid_raman_bpm_column: str = "total"

    # bond polarizability table (read from [bond_polarizability])
    bond_polarizability: dict = field(default_factory=dict)

    # ── [nmr] — NMR relaxation ────────────────────────────────────────────────
    nmr_enabled: bool = False
    nmr_spin_dump: str = ""          # LAMMPS spin dump file (sx,sy,sz per atom)
    nmr_nucleus: str = "1H"          # NMR-active nucleus label
    nmr_larmor_freq: float = 500e6   # Hz — nuclear Larmor frequency
    nmr_larmor_freq_e: float = 0.0   # Hz — electronic Larmor freq (0 = compute from g_e·B0)
    nmr_hyperfine_a: float = 0.0     # MHz — isotropic Fermi contact coupling
    nmr_electron_g: float = 2.0023   # electronic g-factor
    nmr_corlen: float = 0.5          # spin ACF correlation length (fraction of trajectory)
    nmr_electron_distance: float = 0.0  # Å — electron-nucleus distance for dipolar T1/T2 (0 = skip)

    # Quadrupolar relaxation via EFG ACF
    nmr_quadrupolar: bool = False          # enable EFG-based quadrupolar T1/T2
    nmr_efg_nucleus: str = "2H"            # quadrupolar nucleus label (must be in QUAD_NUCLEI)
    nmr_efg_target_ids: list[int] = field(default_factory=list)  # 1-based atom IDs to average over
    nmr_efg_trajectory: str = ""           # LAMMPS position dump with columns id x y z [q]
    nmr_efg_cutoff: float = 10.0           # Å — point-charge sum cutoff
    nmr_efg_sternheimer: float = 0.0       # γ_∞ Sternheimer antishielding factor

    # Multi-site spin ACFs
    nmr_spin_sites_file: str = ""   # path to INI file defining per-site atom ID lists
    nmr_auto_sites: bool = False     # True = auto-group atoms by LAMMPS type column

    # ── [nmr] — outer-sphere (translational) relaxation ──────────────────────
    nmr_os_enabled: bool = False                           # enable outer-sphere calculation
    nmr_os_trajectory: str = ""                            # position dump (defaults to efg_trajectory)
    nmr_os_metal_ids: list[int] = field(default_factory=list)    # 1-based LAMMPS IDs of paramagnetic centers
    nmr_os_nucleus_ids: list[int] = field(default_factory=list)  # 1-based LAMMPS IDs of observed nuclei
    nmr_os_d_inner: float = 0.0          # Å — inner exclusion radius (0 = no inner cutoff)
    nmr_os_d_outer: float = 0.0          # Å — outer shell cutoff (required when os_enabled)
    nmr_os_corlen: float = 0.5           # fraction of trajectory for OS ACF
    nmr_os_cross_metal: bool = False     # coherent sum over metals (default: incoherent per-metal)
    nmr_os_s_spin: float = 0.5           # electron spin quantum number of paramagnetic center
    nmr_tau_r: float = 0.0               # ns — global rotational correlation time for LS correction (0 = disabled)

    # ── [nmrd] — field-cycling NMRD dispersion profile ────────────────────────
    nmrd_enabled:    bool  = False
    nmrd_freq_min:   float = 0.01    # MHz — log grid start
    nmrd_freq_max:   float = 500.0   # MHz — log grid end
    nmrd_n_points:   int   = 150     # points on log grid
    nmrd_include_is: bool  = True    # inner-sphere dipolar channel
    nmrd_include_os: bool  = True    # outer-sphere dipolar channel
    nmrd_include_fc: bool  = True    # Fermi contact channel
    nmrd_tau_m:      float = 0.0     # ns — IS water exchange time (0 = fast exchange)
    nmrd_acf_fit:    str   = "biexp" # "biexp" | "raw"

    # Internal tracking
    _temperature_provided: bool = False
    _pressure_provided: bool = False
    _volume_provided: bool = False
    _energy_avg_provided: bool = False
    _vel_scale_provided: bool = False
    _energy_units_provided: bool = False
    _thermo_section_seen: bool = False
    _energy_to_kcal_conv: float = 1.0
    _lj_epsilon_j_mol: float = 1.0

    @property
    def do_xpt(self) -> bool:
        """True for any mode with a gas phase (2PT/3PT)."""
        return self.mode != ModexPT.ONE_PT

    @property
    def do_desjarlais(self) -> bool:
        """2PT 'desjarlais' refinement: moment-matched memory-function gas DoS."""
        return self.do_xpt and self.refinement == "desjarlais"

    @property
    def do_r2pt(self) -> bool:
        """2PT 'r2pt' refinement: Sun 2017 revised 2PT (δ-tuned f, F_a-inclusive)."""
        return self.do_xpt and self.refinement == "r2pt"

    @property
    def temperature_override(self) -> bool: return self._temperature_provided
    @property
    def pressure_override(self) -> bool: return self._pressure_provided
    @property
    def volume_override(self) -> bool: return self._volume_provided
    @property
    def energy_override(self) -> bool: return self._energy_avg_provided


# ── Validation ────────────────────────────────────────────────────────────────

def validate(cfg: Config) -> None:
    """Validate a fully-loaded Config; raise ValueError on bad inputs."""
    # ── Phase 1 restructure: per-module enabled flags ───────────────────────
    # Auto-derive *_enabled from the per-property flags below them when the
    # user hasn't explicitly set the section-level switch (preserves
    # existing INI-file behaviour: setting transport_diffusion=1 implies
    # transport is enabled).  ``[thermodynamics] enabled`` and the legacy
    # ``disable_xpt`` flag are kept synonymous.
    if cfg.transport_enabled is None:
        cfg.transport_enabled = bool(
            cfg.transport_diffusion or cfg.transport_rotation
            or cfg.transport_shear_viscosity or cfg.transport_bulk_viscosity
            or cfg.transport_electrical_conductivity
            or cfg.transport_thermal_conductivity
            or cfg.transport_dielectric
        )
    if cfg.mechanics_enabled is None:
        cfg.mechanics_enabled = bool(
            cfg.mechanics_dma_shear or cfg.mechanics_dma_bulk
            or cfg.mechanics_dielectric_spectrum
            or cfg.mechanics_elastic_constants
            or cfg.mechanics_maxwell_fit
        )
    # ``thermodynamics_enabled`` resolution:
    #   - ``disable_xpt = 1``                                       → False
    #   - user explicitly set ``[thermodynamics] enabled``           → that value
    #   - user wrote a [thermodynamics] section (any thermo property) → True
    #     (clear opt-in; stays True even when transport/mechanics are also on)
    #   - otherwise: True only if no other module is on (preserves
    #     "empty INI defaults to 2PT" legacy behaviour); False if at
    #     least one of transport / mechanics / nmr / nmrd
    #     is active (so a transport-only INI doesn't need an explicit
    #     ``[thermodynamics] enabled = 0`` line).
    if cfg.disable_xpt:
        cfg.thermodynamics_enabled = False
    elif cfg.thermodynamics_enabled is None:
        if cfg._thermo_section_seen:
            cfg.thermodynamics_enabled = True
        else:
            _other_module_on = (
                bool(cfg.transport_enabled) or bool(cfg.mechanics_enabled)
                or bool(cfg.nmr_enabled) or bool(cfg.nmrd_enabled)
                or bool(cfg.bond_polarizability) or bool(cfg.spdos_raman)
                or bool(cfg.hybrid_raman)
            )
            cfg.thermodynamics_enabled = not _other_module_on

    enabled_modules = []
    if cfg.thermodynamics_enabled: enabled_modules.append("thermodynamics")
    if not enabled_modules:
        raise ValueError(
            "No analysis module enabled.  Set [thermodynamics] enabled = 1 "
            "(or mode = 2PT|3PT) to run an entropy calculation."
        )
    log.info("Active modules: %s", ", ".join(enabled_modules))

    nmr_only = cfg.nmr_enabled and (
        cfg.nmr_spin_dump or (cfg.nmr_quadrupolar and cfg.nmr_efg_trajectory)
    ) and not cfg.trajectory
    multires_active = (
        cfg.multires is not None
        and getattr(cfg.multires, "enabled", False)
        and getattr(cfg.multires, "num_trajectories", 0) > 0
    )
    if not cfg.trajectory and not nmr_only and not multires_active:
        raise ValueError("No trajectory file(s) specified")
    if cfg.molecular and not cfg.topology:
        raise ValueError("Molecular 2PT requires a topology file")
    if cfg.hs_eos not in ("cs", "bmcsl"):
        raise ValueError(f"Invalid hs_eos '{cfg.hs_eos}'. Valid: cs, bmcsl")
    if cfg.hs_entropy not in ("lin2003", "rigorous"):
        raise ValueError(f"Invalid hs_entropy '{cfg.hs_entropy}'. Valid: lin2003, rigorous")
    if cfg.d0_hs_scale <= 0.0:
        raise ValueError(f"d0_hs_scale must be > 0 (got {cfg.d0_hs_scale}). "
                         "Default 1.0 preserves Enskog D₀^HS.")
    if cfg.dimension not in (1, 2, 3, 4):
        raise ValueError(
            f"dimension must be 1, 2, 3, or 4 (got {cfg.dimension}). "
            "3 = bulk (default); 2 = quasi-2D slit/interface; 1 = 1D channel; "
            "4 = d=4 hypersphere LJ (p=1/d test, 4D trajectories only).")
    if cfg.use_sim_z and not (cfg._pressure_provided and cfg._volume_provided
                              and cfg._temperature_provided):
        raise ValueError(
            "use_sim_z=true requires pressure, volume, and temperature to be "
            "specified in [system_properties]"
        )
    _avg_methods = ("none", "block", "bootstrap", "block-bootstrap")
    if cfg.transport_averaging not in _avg_methods:
        raise ValueError(
            f"Invalid transport_averaging '{cfg.transport_averaging}'. "
            f"Valid: {', '.join(_avg_methods)}"
        )
    if cfg.transport_n_blocks < 1:
        raise ValueError(
            f"transport_n_blocks must be >= 1; got {cfg.transport_n_blocks}"
        )
    if cfg.transport_n_bootstrap < 1:
        raise ValueError(
            f"transport_n_bootstrap must be >= 1; got {cfg.transport_n_bootstrap}"
        )
    if cfg.transport_burn_in_frames < 0:
        raise ValueError(
            f"transport_burn_in_frames must be >= 0; got {cfg.transport_burn_in_frames}"
        )
    if not (0.0 < cfg.transport_acf_corlen <= 1.0):
        raise ValueError(
            f"transport_acf_corlen must be in (0, 1]; got {cfg.transport_acf_corlen}"
        )
    if cfg.transport_kappa_truncation not in ("auto", "manual"):
        raise ValueError(
            "transport_kappa_truncation must be 'auto' or 'manual'; got "
            f"{cfg.transport_kappa_truncation!r}"
        )
    if cfg.transport_kappa_min_lag_ps < 0.0:
        raise ValueError(
            "transport_kappa_min_lag_ps must be >= 0; got "
            f"{cfg.transport_kappa_min_lag_ps}"
        )
    if cfg.fixed_fluidicity:
        bad = [f for f in cfg.fixed_fluidicity if not (0.0 <= f <= 1.0)]
        if bad:
            raise ValueError(f"fixed_fluidicity values must be in [0,1]; got {bad}")
    if cfg.finite_size_correction not in ("off", "auto", "yeh-hummer"):
        raise ValueError(
            f"finite_size_correction must be off | auto | yeh-hummer; "
            f"got {cfg.finite_size_correction!r}"
        )
    if cfg.nmr_enabled and not cfg.nmr_spin_dump:
        log.warning("[nmr] enabled but no spin_dump file specified")

    if cfg.lammps_units:
        unit_map = {
            "real":     (1000.0,    "kcal/mol"),
            "metal":    (1.0,       "ev"),
            "si":       (0.01,      "kj/mol"),
            "cgs":      (1.0e-4,    "kcal/mol"),
            "electron": (21.8769,   "kcal/mol"),
        }
        if cfg.lammps_units in unit_map:
            v_scale, e_unit = unit_map[cfg.lammps_units]
            if not cfg._vel_scale_provided: cfg.vel_scale    = v_scale
            if not cfg._energy_units_provided: cfg.energy_units = e_unit
    elif cfg.vel_scale == 1.0 and not cfg._vel_scale_provided \
            and cfg.trajectory_format in ("LAMMPSDUMP", ""):
        cfg.vel_scale = _VEL_SCALE_PRESETS["LAMMPS"]

    energy_unit_conv = {"kcal/mol": 1.0, "kj/mol": 0.239006, "ev": 23.0605}
    unit_key = cfg.energy_units.replace(" ", "")
    if unit_key not in energy_unit_conv:
        raise ValueError(f"Invalid energy_units: {cfg.energy_units}")
    conv = energy_unit_conv[unit_key]
    cfg._energy_to_kcal_conv = conv
    if cfg.energy_avg != 0.0: cfg.energy_avg *= conv
    if cfg.energy_std != 0.0: cfg.energy_std *= conv

    valid_out = {"kj/mol", "kcal/mol", "ev", "lj"}
    if cfg.out_units not in valid_out:
        raise ValueError(f"Invalid out_units '{cfg.out_units}'. Valid: {valid_out}")
    if cfg.out_units == "lj":
        cfg._lj_epsilon_j_mol = cfg.lj_epsilon * conv * 4184.0
        if cfg._lj_epsilon_j_mol <= 0:
            raise ValueError("lj_epsilon must be > 0 for LJ reduced units")

    # ── Startup advisories ────────────────────────────────────────────────────
    # These are quiet correctness traps: things that don't fail loudly but
    # silently produce wrong numbers.  Surface them at startup so the user
    # decides rather than silently shipping wrong thermo/D/μ.
    import logging
    _log = logging.getLogger("pyxpt")

    # No topology → masses default to 1 g/mol → everything mass-dependent
    # (T from VAC, D, μ, energies, ZPE) is off by ~m_atom.  lj_mass in the
    # [output] section is honoured only when out_units = lj; in any other
    # case the user MUST supply a data file.
    if not cfg.topology:
        if cfg.out_units == "lj" and cfg.lj_mass != 1.0:
            _log.info("No topology supplied — falling back to lj_mass = %.4g g/mol "
                      "for all atoms.", cfg.lj_mass)
        else:
            _log.warning("No topology / data file supplied (`[files] topology`). "
                         "Atom masses will default to 1 g/mol — ALL mass-dependent "
                         "quantities (T from VAC, D, μ, energies, ZPE) will be wrong. "
                         "Supply a LAMMPS data file (or set lj_mass when out_units = lj).")

    # Thermodynamics on but no MD energies → Eo correction (E_md - E_classical)
    # and Cv_fluct = σ²(E)/(RT²) are silently skipped.  Final S_q is correct
    # to the extent that the DoS is normalised, but E_q / A_q / μ_q / Cv_q
    # lose physically meaningful contributions.
    if cfg.thermodynamics_enabled:
        if not cfg._energy_avg_provided and cfg.energy_avg == 0.0:
            _log.warning("[thermodynamics] enabled but `energy_avg` not supplied. "
                         "E_q / A_q / μ_q will skip the Eo = E_md − E_classical shift. "
                         "Add `energy_avg = <total MD energy>` under [system_properties] "
                         "to enable the correction.")
        if cfg.energy_std == 0.0:
            _log.warning("[thermodynamics] enabled but `energy_std` not supplied. "
                         "Cv_q will fall back to the DoS-only quantum-classical "
                         "difference and miss the σ²(E)/(RT²) fluctuation term. "
                         "Add `energy_std = <std dev of MD energy>` under [system_properties] "
                         "for the fluctuation-aware Cv_q.")


# ── Summary printer ───────────────────────────────────────────────────────────

def print_summary(cfg: Config, file=sys.stdout) -> None:
    lines = [
        "",
        "  ── Configuration ─────────────────────────────────────",
        f"  topology         : {cfg.topology or '(none)'}",
        f"  trajectory       : {', '.join(cfg.trajectory)}",
        f"  mode             : {cfg.mode} ({cfg.mode.name})",
    ]
    if cfg.trajectory_format == "LAMMPSDUMP":
        lines.append(f"  lammps_units     : {cfg.lammps_units or '(auto/none)'}")
    lines.append(f"  output_units     : {cfg.out_units.upper()}")
    if cfg.out_units == "lj":
        lines += [
            f"  LJ sigma         : {cfg.lj_sigma} Å",
            f"  LJ epsilon       : {cfg.lj_epsilon} {cfg.energy_units}",
            f"  LJ mass          : {cfg.lj_mass} g/mol",
        ]
    if cfg.nmr_enabled:
        lines.append(f"  nmr nucleus      : {cfg.nmr_nucleus}  ω_I = {cfg.nmr_larmor_freq:.3g} Hz")
    if cfg.nmr_quadrupolar:
        lines.append(f"  nmr quadrupolar  : {cfg.nmr_efg_nucleus}  cutoff = {cfg.nmr_efg_cutoff} Å")
    lines += ["  ─────────────────────────────────────────────────────", ""]
    print("\n".join(lines), file=file)
