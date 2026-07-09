# -*- coding: utf-8 -*-
"""
py-xPT — two-/three-phase thermodynamics (2PT/3PT) entropy from MD trajectories.

Quick start
-----------
    from pyxpt import run
    result = run("control.ini")

    from pyxpt import load_config
    from pyxpt.io.trajectory import System
    from pyxpt.thermo import xPTEngine
    cfg    = load_config("control.ini")
    system = System.from_config(cfg)
    engine = xPTEngine(cfg, system)
    engine.accumulate(system.iter_frames(cfg))
    result = engine.compute()
    engine.write(result)
"""
from .config import Config, load as load_config
from .io.trajectory import System, FrameData
from .thermo.engine import xPTEngine, xPTResult
from .analysis import run
from .constants import COPYRIGHT, ModexPT
from . import core

__all__ = [
    "run",
    "Config", "load_config",
    "System", "FrameData",
    "xPTEngine", "xPTResult",
    "COPYRIGHT", "ModexPT",
    "core",
]
__version__ = "1.0.0"
