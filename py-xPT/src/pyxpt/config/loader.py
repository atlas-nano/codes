# -*- coding: utf-8 -*-
"""Top-level config loader."""
from __future__ import annotations

import configparser
import logging
from pathlib import Path

from .base import Config, TOPOLOGY_FORMAT_MAP, TRAJECTORY_FORMAT_MAP, validate
from .thermo import parse_xpt, parse_md

log = logging.getLogger(__name__)


def load(path: str | Path) -> Config:
    """Parse a py-xPT INI control file and return a validated Config."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Control file not found: {path}")

    log.info("Reading control file %s", path)
    raw = path.read_text()

    parser = configparser.ConfigParser(
        inline_comment_prefixes=("#", "!"),
        comment_prefixes=("#", "!", "*", "//"),
        strict=False,
        interpolation=None,
    )
    parser.read_string(raw)
    cfg = Config()

    # ── [files] ───────────────────────────────────────────────────────────────
    if parser.has_section("files"):
        s = parser["files"]
        if "topology" in s:
            cfg.topology = s["topology"].strip()
        if "topology_format" in s:
            cfg.topology_format = TOPOLOGY_FORMAT_MAP.get(
                s["topology_format"].strip().upper(), s["topology_format"].strip().upper()
            )
        if "trajectory" in s:
            cfg.trajectory = s["trajectory"].split()
        if "velocity_file" in s:
            cfg.velocity_file = s["velocity_file"].strip()
        if "trajectory_format" in s:
            cfg.trajectory_format = TRAJECTORY_FORMAT_MAP.get(
                s["trajectory_format"].strip().upper(), s["trajectory_format"].strip().upper()
            )
        if "group_file" in s:
            cfg.group_file = s["group_file"].strip()

    # ── [frames] ──────────────────────────────────────────────────────────────
    if parser.has_section("frames"):
        s = parser["frames"]
        cfg.start = s.getint("start", cfg.start)
        cfg.stop  = s.getint("stop",  cfg.stop)
        cfg.step  = s.getint("step",  cfg.step)

    # ── [output] ──────────────────────────────────────────────────────────────
    if parser.has_section("output"):
        s = parser["output"]
        cfg.prefix               = s.get("prefix",               cfg.prefix).strip()
        # 'show_2pt_split' kept as a back-compat alias for 'show_xpt_split'.
        cfg.show_xpt_split       = s.getboolean("show_xpt_split",
                                   s.getboolean("show_2pt_split", cfg.show_xpt_split))
        cfg.show_classical_thermo = s.getboolean("show_classical_thermo", cfg.show_classical_thermo)
        cfg.normalize            = s.getboolean("normalize",             cfg.normalize)
        cfg.out_units            = s.get("out_units", cfg.out_units).strip().lower()
        cfg.lj_sigma             = s.getfloat("lj_sigma",   cfg.lj_sigma)
        cfg.lj_epsilon           = s.getfloat("lj_epsilon", cfg.lj_epsilon)
        cfg.lj_mass              = s.getfloat("lj_mass",    cfg.lj_mass)
        cfg.cross_vac_diagnostic = s.getboolean("cross_vac_diagnostic",
                                                cfg.cross_vac_diagnostic)

    # ── [memory] — RAM-budget driven atom batching (Phase MEM-R1) ────────────
    if parser.has_section("memory"):
        s = parser["memory"]
        if "ram_budget_gb" in s:
            cfg.ram_budget_gb = s.getfloat("ram_budget_gb")
        # alias: just ``budget_gb``
        if "budget_gb" in s:
            cfg.ram_budget_gb = s.getfloat("budget_gb")
        if "ram_budget_fraction" in s:
            cfg.ram_budget_fraction = s.getfloat("ram_budget_fraction")
        if "fraction" in s:
            cfg.ram_budget_fraction = s.getfloat("fraction")
        if "single_pass_only" in s:
            cfg.single_pass_only = s.getboolean("single_pass_only")

    # ── module parsers ────────────────────────────────────────────────────────
    parse_xpt(parser, cfg)
    parse_md(parser, cfg)

    validate(cfg)
    return cfg

