# -*- coding: utf-8 -*-
"""Parse [thermodynamics] / [system_properties] config sections.

The publishable taxonomy is the only surface: mode = 1PT | 2PT | 3PT
(case-insensitive strings via the ModexPT enum).  The retired research
modes (HS/MF/GLE/MZV) and their knobs are rejected with a migration error.
No legacy integer aliases, no deprecated knob bridges, no preset system.
"""
from __future__ import annotations

import configparser
import logging

from .base import Config, _VEL_SCALE_PRESETS

log = logging.getLogger(__name__)


def parse_xpt(parser: configparser.ConfigParser, cfg: Config) -> None:
    """Read [thermodynamics] (aliases: [xpt], [2pt], [analysis]) into cfg."""
    sections_present = [n for n in ("thermodynamics", "xpt", "2pt", "analysis")
                        if parser.has_section(n)]
    if len(sections_present) > 1:
        log.info("Merging multiple analysis sections: %s "
                 "(later overrides earlier on key conflict)",
                 ", ".join(sections_present))
    if sections_present:
        cfg._thermo_section_seen = True
    for sec_name in sections_present:
        s = parser[sec_name]
        if "strict_stationarity" in s:
            cfg.strict_stationarity = s.getboolean("strict_stationarity")
        # ── mode (1PT | 2PT | 3PT, case-insensitive) ────────────────────────
        from pyxpt.constants import ModexPT
        if "mode" in s:
            raw_mode = s["mode"].strip().upper()
            if raw_mode in ("HS", "MF"):
                raise ValueError(
                    f"mode = {raw_mode!r} is defunct. Use the 1PT/2PT/3PT taxonomy: "
                    "mode=2PT + refinement=rigorous|lin2003|desjarlais|r2pt, or "
                    "mode=3PT (cage).  (HS -> 2PT+rigorous; MF -> 2PT+desjarlais.)"
                )
            try:
                cfg.mode = ModexPT(raw_mode)
            except ValueError:
                raise ValueError(
                    f"mode = {s['mode'].strip()!r} not recognised; valid: 1PT | 2PT | 3PT"
                ) from None
        # ── refinement ──────────────────────────────────────────────────────
        # 2PT: gas--solid treatment (rigorous|lin2003|desjarlais|r2pt).
        if "refinement" in s:
            raw = s["refinement"].strip().lower()
            if cfg.mode == ModexPT.THREE_PT:
                if raw != "none":
                    raise ValueError(
                        f"refinement = {raw!r} for mode=3PT: must be 'none' "
                        "(the published bare memory cage)"
                    )
            elif raw not in ("rigorous", "lin2003", "desjarlais", "r2pt"):
                raise ValueError(
                    f"refinement = {raw!r} for mode=2PT: must be rigorous | lin2003 | desjarlais | r2pt"
                )
            cfg.refinement = raw
        if "r2pt_delta" in s: cfg.r2pt_delta = s.getfloat("r2pt_delta")
        # reject keys removed by the taxonomy (hard migration)
        for _old, _hint in (("hs_entropy", "refinement=rigorous|lin2003"),
                            ("mf_variant", "refinement=desjarlais (fmf/lin2021 retired)")):
            if _old in s:
                raise ValueError(f"[thermodynamics] {_old!r} removed; use {_hint}.")
        cfg.molecular     = s.getboolean("molecular",      cfg.molecular)
        cfg.corlen        = s.getfloat("corlen",           cfg.corlen)
        cfg.constraints   = s.getint("constraints",        cfg.constraints)
        cfg.per_molecule  = s.getboolean("per_molecule",   cfg.per_molecule)
        cfg.check_grp_eng = s.getboolean("check_grp_eng",  cfg.check_grp_eng)
        cfg.use_vrotat    = s.getboolean("use_vrotat",      cfg.use_vrotat)
        cfg.use_gpu       = s.getboolean("use_gpu",         cfg.use_gpu)
        if "hs_eos"     in s: cfg.hs_eos     = s["hs_eos"].strip().lower()
        if "dimension"  in s: cfg.dimension  = int(s["dimension"])
        if "cage_entropy" in s:     cfg.cage_entropy     = s.getboolean("cage_entropy")
        if "cage_entropy_rot" in s: cfg.cage_entropy_rot = s.getboolean("cage_entropy_rot")
        if "cage_nf_run" in s:
            cfg.cage_nf_run = int(s["cage_nf_run"])
            if cfg.cage_nf_run < 1:
                raise ValueError("[thermodynamics] cage_nf_run must be >= 1")
        if "cage_taper" in s:
            cfg.cage_taper = s["cage_taper"].strip().lower()
            if cfg.cage_taper not in ("none", "hann"):
                raise ValueError(f"[thermodynamics] cage_taper must be none or "
                                 f"hann (got '{cfg.cage_taper}')")
        if "cage_tail_tol" in s:
            cfg.cage_tail_tol = s.getfloat("cage_tail_tol")
            if cfg.cage_tail_tol <= 0.0:
                raise ValueError("[thermodynamics] cage_tail_tol must be > 0")
        if "gas_gate" in s:
            cfg.gas_gate = s["gas_gate"].strip().lower()
            if cfg.gas_gate not in ("none", "debye", "debye_warn"):
                raise ValueError(f"[thermodynamics] gas_gate must be none, debye or "
                                 f"debye_warn (got '{cfg.gas_gate}')")
        if "gas_gate_tol" in s:
            cfg.gas_gate_tol = s.getfloat("gas_gate_tol")
            if cfg.gas_gate_tol <= 0.0:
                raise ValueError("[thermodynamics] gas_gate_tol must be > 0")
        # ── derive internal HS-entropy convention + cage from (mode, refinement) ──
        # lin2003 and desjarlais both use the empirical Lin-Goddard +ln Z entropy
        # convention (Desjarlais 2013 builds on the +ln Z 2PT framework); rigorous
        # and r2pt use the rigorous Sackur-Tetrode + CS excess (r2pt applies its own
        # ln z-free weight internally).
        cfg.hs_entropy = ("lin2003" if cfg.refinement in ("lin2003", "desjarlais")
                          else "rigorous")
        if cfg.mode == ModexPT.THREE_PT:
            cfg.cage_entropy = True
            cfg.cage_entropy_rot = True   # engine auto-skips rotational cage for monatomic
            cfg.refinement = "none"       # bare memory cage (the published 3PT)
        if "d0_hs_scale" in s:
            cfg.d0_hs_scale = s.getfloat("d0_hs_scale")
        # Phase MEM-B: 'disable_xpt' / 'skip_xpt' / 'no_xpt' all map to the
        # same flag.  See Config.disable_xpt docstring.  The legacy '*_2pt'
        # spellings are kept as back-compat aliases.
        for _key in ("disable_xpt", "skip_xpt", "no_xpt", "mechanics_only",
                     "disable_2pt", "skip_2pt", "no_2pt"):
            if _key in s:
                cfg.disable_xpt = s.getboolean(_key)
        if "enabled" in s:
            cfg.thermodynamics_enabled = s.getboolean("enabled")
        # Phase MR-DRIFT: subtract system COM velocity per frame.  Default
        # True (correct for equilibrium MD); set False for non-equilibrium
        # runs where COM drift carries physical info.
        for _key in ("subtract_com_velocity", "remove_com_drift",
                     "drift_correction"):
            if _key in s:
                cfg.subtract_com_velocity = s.getboolean(_key)
        cfg.use_sim_z = s.getboolean("use_sim_z", cfg.use_sim_z)
        if "ensemble" in s:
            v = s["ensemble"].strip().lower()
            if v not in ("nvt", "npt", "auto"):
                raise ValueError(f"[thermodynamics] ensemble={v!r}: must be nvt|npt|auto")
            cfg.ensemble = v
        if "apodization" in s:
            cfg.apodization = s["apodization"].strip().lower()
        if "apodization_alpha" in s:
            cfg.apodization_alpha = s.getfloat("apodization_alpha")
        if "lammps_units" in s:
            cfg.lammps_units = s["lammps_units"].strip().lower()
        if "vel_scale" in s:
            raw_vs = s["vel_scale"].strip().upper()
            cfg.vel_scale = _VEL_SCALE_PRESETS.get(
                raw_vs, float(raw_vs) if raw_vs.replace(".", "", 1).isdigit() else 1.0
            )
            cfg._vel_scale_provided = True
        if "fixed_fluidicity" in s:
            cfg.fixed_fluidicity = [float(v) for v in s["fixed_fluidicity"].split()]
        # Finite-size correction (Yeh-Hummer) on fluidicity solve.  Accepts
        # several spellings of the abbreviation; True → "auto", False → "off".
        for _key in ("finite_size_correction", "fsc", "yeh_hummer", "yeh-hummer"):
            if _key in s:
                raw_fsc = s[_key].strip().lower()
                if raw_fsc in ("true", "yes", "on", "1"):
                    cfg.finite_size_correction = "auto"
                elif raw_fsc in ("false", "no", "off", "0"):
                    cfg.finite_size_correction = "off"
                else:
                    cfg.finite_size_correction = raw_fsc
        for _key in ("fsc_viscosity", "viscosity_for_fsc", "fsc_eta"):
            if _key in s:
                cfg.fsc_viscosity = s.getfloat(_key)
        if "mol_linear" in s and s["mol_linear"].strip().lower() != "g":
            cfg.mol_linear = int(s["mol_linear"])
        if "mol_rotsym" in s and s["mol_rotsym"].strip().lower() != "g":
            cfg.mol_rotsym = int(s["mol_rotsym"])


def parse_md(parser: configparser.ConfigParser, cfg: Config) -> None:
    """Read [system_properties] (alias: [md]) into cfg."""
    sections_present = [n for n in ("system_properties", "md")
                        if parser.has_section(n)]
    if len(sections_present) > 1:
        log.info("Merging multiple system-properties sections: %s",
                 ", ".join(sections_present))
    for sec_name in sections_present:
        s = parser[sec_name]
        cfg.timestep  = s.getfloat("timestep", cfg.timestep)
        cfg.dump_freq = s.getint("dump_freq",  cfg.dump_freq)
        if "temperature" in s:
            cfg.temperature = s.getfloat("temperature")
            cfg._temperature_provided = True
        if "pressure" in s:
            cfg.pressure = s.getfloat("pressure")
            cfg._pressure_provided = True
        if "reference_pressure" in s:
            cfg.reference_pressure = s.getfloat("reference_pressure")
        if "volume" in s:
            cfg.volume = s.getfloat("volume")
            cfg._volume_provided = True
        if "energy_avg" in s:
            cfg.energy_avg = s.getfloat("energy_avg")
            cfg._energy_avg_provided = True
        cfg.void_volume = s.getfloat("void_volume", cfg.void_volume)
        if "energy_units" in s:
            cfg.energy_units = s["energy_units"].strip().lower()
            cfg._energy_units_provided = True
        cfg.energy_std = s.getfloat("energy_std", cfg.energy_std)
        # `lammps_units` is conventionally a [thermodynamics] key, but the
        # transport / mechanics / ir_raman pipelines need it for unit
        # conversions too — accept it under [system_properties] as well so
        # transport-only INI files don't need a [thermodynamics] section.
        if "lammps_units" in s:
            cfg.lammps_units = s["lammps_units"].strip().lower()


def parse_transport(parser: configparser.ConfigParser, cfg: Config) -> None:
    """Read [transport] into cfg.

    All keys are optional; the section is opt-in (no default activations).
    Per-property bool flags accept short aliases (e.g. ``shear`` for
    ``shear_viscosity``); see the table below.
    """
    if not parser.has_section("transport"):
        return
    s = parser["transport"]
    # Phase 1 restructure: explicit module enable flag.  When unset
    # the per-property flags below auto-enable the module via
    # :func:`pyxpt.config.base.validate`.
    if "enabled" in s:
        cfg.transport_enabled = s.getboolean("enabled")

    # Per-property flags.  Accept long form (matches Config field name minus
    # the "transport_" prefix) and a short alias for the most common ones.
    _bool_aliases = {
        "diffusion":               "transport_diffusion",
        "rotation":                "transport_rotation",
        "shear_viscosity":         "transport_shear_viscosity",
        "shear":                   "transport_shear_viscosity",
        "bulk_viscosity":          "transport_bulk_viscosity",
        "bulk":                    "transport_bulk_viscosity",
        "viscosity":               "transport_shear_viscosity",   # default to shear
        "electrical_conductivity": "transport_electrical_conductivity",
        "electrical":              "transport_electrical_conductivity",
        "electrical_molecular_current": "transport_electrical_molecular_current",
        "molecular_current":       "transport_electrical_molecular_current",
        "mol_current":             "transport_electrical_molecular_current",
        "ion_current":             "transport_electrical_molecular_current",
        "thermal_conductivity":    "transport_thermal_conductivity",
        "thermal":                 "transport_thermal_conductivity",
        "dielectric":              "transport_dielectric",
    }
    for key, attr in _bool_aliases.items():
        if key in s:
            setattr(cfg, attr, s.getboolean(key))

    if "averaging" in s:
        cfg.transport_averaging = s["averaging"].strip().lower()
    if "n_blocks" in s:
        cfg.transport_n_blocks = s.getint("n_blocks")
    if "n_bootstrap" in s:
        cfg.transport_n_bootstrap = s.getint("n_bootstrap")
    if "report_uncertainty" in s:
        cfg.transport_report_uncertainty = s.getboolean("report_uncertainty")
    if "burn_in_frames" in s:
        cfg.transport_burn_in_frames = s.getint("burn_in_frames")
    if "acf_corlen" in s:
        cfg.transport_acf_corlen = s.getfloat("acf_corlen")
    if "kappa_truncation" in s:
        cfg.transport_kappa_truncation = s["kappa_truncation"].strip().lower()
    if "kappa_min_lag_ps" in s:
        cfg.transport_kappa_min_lag_ps = s.getfloat("kappa_min_lag_ps")
