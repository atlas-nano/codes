# -*- coding: utf-8 -*-
"""
py-xPT — Core computational infrastructure
=============================================

Analysis-agnostic kernels (VACF, FFT) used by the pyxpt thermodynamics module.
This subpackage has no internal dependencies on other pyxpt modules
and is safe to import standalone.

Modules
-------
:mod:`pyxpt.core.fft`
    Apodization windows for spectral analysis.

:mod:`pyxpt.core.gpu`
    GPU backend abstraction (numpy / CuPy).

:mod:`pyxpt.core.vac`
    Wiener-Khinchin FFT kernel for the velocity autocorrelation
    tensor.  Pure numpy/CuPy operating on plain arrays — no engine
    state.  Used by every pyxpt pipeline that needs the mass-
    weighted 3×3 VAC tensor.
"""
from . import fft
from . import gpu
from . import vac

# Re-export the most commonly used kernel functions
from .vac import compute_vac_tensor, subtract_system_com_velocity

__all__ = [
    "fft",
    "gpu",
    "vac",
    "compute_vac_tensor",
    "subtract_system_com_velocity",
]
