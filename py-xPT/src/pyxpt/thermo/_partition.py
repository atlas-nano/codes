# -*- coding: utf-8 -*-
"""
Gas/solid DoS partition kernel — extracted from
``thermo/engine.py:_partition_dos``.

Public API
----------
:func:`partition_dos(engine, dos, fxpt, fmf, Bg_arr, y_arr=None)`
    Split the per-channel DoS into gas + solid contributions using one
    of three partition shapes:

    - **Lin-Goddard Lorentzian** (modes 2/3/5/6) — vectorised fast path.
    - **Desjarlais memory-function** (mode 3 path) — scalar fallback.

Like the other ``_*_compute`` modules, this reaches into engine state
via the ``engine`` argument but introduces no new state.
Bit-exact with the pre-extraction implementation on the LJ-Ar
regression suite (D1).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

try:
    _trapz = np.trapezoid
except AttributeError:
    _trapz = np.trapz

from pyxpt.constants import PI, VelType, ModexPT
from pyxpt.thermo.utility import (
    xpt_partition, xpt_partition_des,
)

if TYPE_CHECKING:  # pragma: no cover
    from pyxpt.thermo.engine import xPTEngine


TRANS = int(VelType.TRANS)
ANGUL = int(VelType.ANGUL)
ROTAT = int(VelType.ROTAT)


def partition_dos(engine: "xPTEngine", dos: np.ndarray,
                   fxpt: np.ndarray, fmf: np.ndarray,
                   Bg_arr: np.ndarray,
                   y_arr: np.ndarray | None = None,
                   ) -> tuple[np.ndarray, np.ndarray]:
    """Split DOS into gas and solid contributions.

    See engine.xPTEngine._partition_dos for the original docstring;
    behaviour and signature are preserved.
    """
    ngrp  = engine.sys.ngrp
    vt    = engine._vactype
    nused = engine._nused
    nu    = engine._pwrfreq
    if y_arr is None:
        y_arr = np.zeros(ngrp)

    dos_gas  = np.zeros_like(dos)
    dos_sol  = np.zeros_like(dos)
    d_dim    = int(engine.cfg.dimension)        # translational gas-Lorentzian dimensionality

    for gi, grp in enumerate(engine.sys.groups):
        # TAP 04/05/2026 fix: don't fall back to 1 when grp.nmol = 0.
        nmol = grp.nmol

        # ── 2PT Lin-Goddard partition (vectorized standard path) ──────────
        # Lin-Goddard Lorentzian (f == fmf case) broadcasts trivially over
        # the frequency axis at C-loop speed.  Desjarlais and memory-function
        # variants have non-vectorizable inner functions; they fall back to
        # the original per-frequency scalar loop.
        freq_arr = np.arange(nused, dtype=float) * nu
        use_vec_trans = (
            not engine._do_desjarlais
            and abs(fxpt[0, gi] - fmf[0, gi]) < 1e-12
            and fxpt[0, gi] > 0.0
        )
        if use_vec_trans:
            s_total = dos[gi, TRANS, :nused]
            s0 = float(s_total[0])
            f_t = fxpt[0, gi]
            # Lin-Goddard diffusive Lorentzian half-width carries the spatial
            # dimensionality as 2·d·N·f (= 6·N·f at d=3); must match the scalar
            # xpt_partition (utility.py) for d≠3 — was hardwired to 6.0.
            denom = 1.0 + (PI * s0 * freq_arr / (2.0 * d_dim * nmol * f_t)) ** 2
            gas_t = s0 / denom
            gas_t[0] = s0    # ν=0 special case (matches scalar xpt_partition)
            gas_t = np.minimum(gas_t, s_total)
            dos_gas[gi, TRANS, :nused] = gas_t
            dos_sol[gi, TRANS, :nused] = s_total - gas_t

        use_vec_rot = (
            engine._molecular and engine._rot_type < vt
            and not engine._do_desjarlais
            and abs(fxpt[1, gi] - fmf[1, gi]) < 1e-12
            and fxpt[1, gi] > 0.0
        )
        if use_vec_rot:
            s_total_r = dos[gi, engine._rot_type, :nused]
            s0_r = float(s_total_r[0])
            f_r = fxpt[1, gi]
            denom_r = 1.0 + (PI * s0_r * freq_arr / (6.0 * nmol * f_r)) ** 2
            gas_r = s0_r / denom_r
            gas_r[0] = s0_r
            gas_r = np.minimum(gas_r, s_total_r)
            dos_gas[gi, engine._rot_type, :nused] = gas_r
            dos_sol[gi, engine._rot_type, :nused] = s_total_r - gas_r

        # Other velocity types: all-solid (vectorized)
        skip = {TRANS}
        if engine._molecular and engine._rot_type < vt:
            skip.add(engine._rot_type)
            other_rot = ANGUL if engine._rot_type == ROTAT else ROTAT
            skip.add(other_rot)
        for k in range(vt):
            if k in skip:
                continue
            dos_sol[gi, k, :nused] = dos[gi, k, :nused]

        # Scalar fallback for Desjarlais / memory-function paths
        if not (use_vec_trans and (use_vec_rot or not engine._molecular
                                     or engine._rot_type >= vt)):
            for i in range(nused):
                freq = nu * i
                # Translation
                if not use_vec_trans:
                    if engine._do_desjarlais:
                        g, sl = xpt_partition_des(
                            dos[gi, TRANS, 0], dos[gi, TRANS, i],
                            freq, nmol, fxpt[0, gi], Bg_arr[0, gi])
                    else:
                        g, sl = xpt_partition(
                            dos[gi, TRANS, 0], dos[gi, TRANS, i],
                            freq, nmol, fxpt[0, gi], fmf[0, gi], dimension=d_dim)
                    dos_gas[gi, TRANS, i] = g
                    dos_sol[gi, TRANS, i] = sl
                # Rotation
                if (engine._molecular and engine._rot_type < vt
                        and not use_vec_rot):
                    if engine._do_desjarlais:
                        g_a, sl_a = xpt_partition_des(
                            dos[gi, engine._rot_type, 0],
                            dos[gi, engine._rot_type, i],
                            freq, nmol, fxpt[1, gi], Bg_arr[1, gi],
                        )
                    else:
                        g_a, sl_a = xpt_partition(
                            dos[gi, engine._rot_type, 0],
                            dos[gi, engine._rot_type, i],
                            freq, nmol, fxpt[1, gi], fmf[1, gi],
                        )
                    dos_gas[gi, engine._rot_type, i] = g_a
                    dos_sol[gi, engine._rot_type, i] = sl_a

    return dos_gas, dos_sol
