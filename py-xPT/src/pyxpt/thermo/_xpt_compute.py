# -*- coding: utf-8 -*-
"""
2PT-HS / 2PT-MF / Desjarlais compute kernel — extracted from
``thermo/engine.py:_calc_xpt``.

Public API
----------
:func:`compute_xpt(engine, dos, vacT, vacDF, V_use, T_use)`
    Compute fluidicity, packing fraction and hard-sphere DoF for the
    standard 2PT modes (HS, MF / des, MF / lin2021).

Like the other ``_*_compute`` modules, this reaches into engine state via
the ``engine`` argument but introduces no new state.
"""
from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

import numpy as np

from pyxpt.constants import PI, KB, NA, VLIGHT, R, VelType, ModexPT
from pyxpt.thermo.utility import (
    search_xpt, hsdf as _hsdf, trapz_dos,
    refine_Bg_desjarlais, hsdf_des,
    enskog_K_parameter, solve_xpt_dgen,
)

if TYPE_CHECKING:  # pragma: no cover
    from pyxpt.thermo.engine import xPTEngine

log = logging.getLogger(__name__)

TRANS = int(VelType.TRANS)


def compute_xpt(engine: "xPTEngine",
                 dos: np.ndarray, vacT: np.ndarray, vacDF: np.ndarray,
                 V_use: float, T_use: float):
    """
    Compute fluidicity, packing fraction and hard-sphere DOF
    for each group.

    Returns
    -------
    fxpt     : (2, ngrp)  translational and rotational fluidicity
    fmf      : (2, ngrp)  memory-function fluidicity (= fxpt for Desjarlais)
    y_arr    : (ngrp,)    packing fraction
    hsdf_arr : (2, ngrp)  hard-sphere DOF
    Bg_arr   : (2, ngrp)  Gaussian kernel width B_g (Desjarlais mode only; 0 otherwise)
    """
    s    = engine.sys
    ngrp = s.ngrp
    nu   = engine._pwrfreq
    nused= engine._nused

    fxpt     = np.zeros((2, ngrp))
    fmf_arr  = np.zeros((2, ngrp))
    y_arr    = np.zeros(ngrp)
    hsdf_arr = np.zeros((2, ngrp))
    Bg_arr   = np.zeros((2, ngrp))

    # d=3 (default) is bit-identical to all prior behavior.  d=1 (single-file
    # channels) has D→0 hard-rod physics not captured by the Chapman-Enskog
    # closure and is rejected here (needs separate single-file treatment).
    d_dim = int(engine.cfg.dimension)
    if d_dim == 1:
        log.warning(
            "dimension=1: the 2PT fluidicity NORMALIZATION is ill-defined in 1D "
            "(hard rods have no Chapman-Enskog self-diffusion — single-file/no-"
            "passing), so f tends to saturate (→0 jammed, →1 dilute) rather than "
            "give physical intermediate values. The closure STRUCTURE (f=Δ, γ=1-f) "
            "and the Tonks-rod HS excess / 1D Sackur-Tetrode are correct; the "
            "f→0 single-file (all-solid) limit is reliable. Intermediate-f 1D "
            "channels need a single-file mobility or DoS-lineshape fluidicity "
            "(open problem).")
    def _solve_f(K):
        return solve_xpt_dgen(K, d_dim)

    # Mode 1 (1PT / ONE_PT): all-solid — no gas component anywhere.
    # Return zero fluidicity immediately; _partition_dos handles f=0 correctly.
    if not engine._apply_xpt:
        return fxpt, fmf_arr, y_arr, hsdf_arr, Bg_arr

    # Resolve fixed_fluidicity list → per-group lookup (validate length here
    # where ngrp is known).
    _ff = engine.cfg.fixed_fluidicity
    if _ff and len(_ff) not in (1, ngrp):
        raise ValueError(
            f"fixed_fluidicity has {len(_ff)} value(s) but the system has "
            f"{ngrp} group(s).  Supply 1 value (applied to all groups) or "
            f"exactly {ngrp} value(s)."
        )

    # Yeh-Hummer FSC: resolve a single shear viscosity once for all groups.
    _fsc_mode = engine.cfg.finite_size_correction
    _fsc_eta  = engine._yeh_hummer_eta(T_use, V_use) if _fsc_mode != "off" else 0.0
    if _fsc_mode == "yeh-hummer" and _fsc_eta <= 0.0:
        raise ValueError(
            "finite_size_correction = yeh-hummer requested but no shear "
            "viscosity is available. Set [2pt] fsc_viscosity = <Pa·s> "
            "or enable [transport] shear_viscosity."
        )
    if _fsc_mode == "auto" and _fsc_eta <= 0.0:
        log.warning("finite_size_correction = auto: no viscosity available, "
                    "skipping Yeh-Hummer correction on fluidicity.")
    # Per-group FSC scales (1.0 = no correction).  Cached on self so that
    # _compute_transport can rescale D-derived outputs.
    engine._fsc_scale_trans = np.ones(ngrp)
    engine._fsc_scale_rot   = np.ones(ngrp)

    for gi, grp in enumerate(s.groups):
        #nmol  = max(grp.nmol if engine._molecular else 1, 1)
        nmol = grp.nmol #TAP 4/5/2026: changed here. there was a bug in legacy code
        gmass = grp.mass
        gvol  = grp.volume if grp.volume > 0 else V_use  # V_total default → mixing entropy included
        T_t   = vacT[gi, TRANS]

        pwr_t = dos[gi, TRANS, :nused]
        s0t   = pwr_t[0]

        if s0t > 0 and T_t > 0 and gmass > 0 and gvol > 0:
            K = enskog_K_parameter(s0t, T_t, gmass / nmol, nmol, gvol, dimension=d_dim)

            # Yeh-Hummer: scale K = K(D) by D_inf/D_PBC.  The DoS array
            # itself is left untouched; only the fluidicity solve and the
            # packing fraction (which depends on K) see the corrected K.
            fsc_scale = engine._yeh_hummer_scale(
                gvol, gmass, nmol, T_t, s0t, _fsc_eta,
            ) if _fsc_eta > 0.0 else 1.0
            engine._fsc_scale_trans[gi] = fsc_scale
            # P4: D₀^HS scale override (option C — MD-derived reference).
            # K ∝ D / D₀^HS, so K → K / d0_hs_scale replaces D₀^HS.
            d0_scale = float(engine.cfg.d0_hs_scale)
            K_eff = K * fsc_scale / d0_scale
            if d0_scale != 1.0:
                log.info(
                    "D₀^HS override: group %d  d0_hs_scale=%.4f  "
                    "f shifts %.4f → %.4f",
                    gi, d0_scale,
                    _solve_f(K * fsc_scale),
                    _solve_f(K_eff),
                )
            if fsc_scale != 1.0:
                log.info(
                    "FSC (Yeh-Hummer): group %d  η=%.3e Pa·s  L=%.3f Å  "
                    "D_inf/D_PBC=%.4f  f shifts %s",
                    gi, _fsc_eta, (gvol)**(1/3),
                    fsc_scale,
                    f"{_solve_f(K):.4f} → {_solve_f(K_eff):.4f}"
                    if not _ff else "(fixed_fluidicity overrides)",
                )

            if _ff:
                # Fixed fluidicity: bypass self-consistent solve.
                # The gas/solid DoS split and packing fraction are computed
                # using the prescribed f rather than the Enskog-matched value.
                f = float(_ff[gi] if len(_ff) > 1 else _ff[0])
            else:
                f = _solve_f(K_eff)
            if engine._do_desjarlais:
                fm = f   # f_g is fixed; B_g is varied instead
                Bg = refine_Bg_desjarlais(pwr_t, s0t, nu, nmol, f)
                Bg_arr[0, gi] = Bg
                hsd = hsdf_des(pwr_t, nu, nmol, f, Bg)
            else:
                fm = f
                hsd = _hsdf(pwr_t, nu, nmol, f, fm, dimension=d_dim)

            fxpt[0, gi]    = f
            fmf_arr[0, gi] = fm

            # packing factor exponent d/(d-1) (= 1.5 at d=3, exact).  d=1 is the
            # degenerate limit γ=1-f (the exponent diverges; handle separately).
            tdf = trapz_dos(pwr_t, nu)
            if d_dim == 1:
                y_arr[gi] = max(0.0, 1.0 - fm)        # Tonks packing γ = 1 − f
            else:
                ry = (fm / K_eff)**(d_dim/(d_dim-1.0)) if K_eff > 0 else 0.0
                y_arr[gi] = ry * (hsd / tdf if tdf > 0 else 0.0)
            hsdf_arr[0, gi] = hsd

        if engine._molecular and engine._rot_type < engine._vactype:
            # Rotational fluidicity from the active rotational velocity type
            # (ANGUL = ω·√I per mol, or ROTAT = ω×r per atom, per use_vrotat flag)
            pwr_a = dos[gi, engine._rot_type, :nused]
            s0a   = pwr_a[0]

            T_a   = vacT[gi, engine._rot_type]
            if s0a > 0 and T_a > 0 and gmass > 0 and gvol > 0:
                K_r = enskog_K_parameter(s0a, T_a, gmass / nmol, nmol, gvol)
                fsc_scale_rot = engine._yeh_hummer_scale_rot(
                    gvol, nmol, _fsc_eta,
                ) if _fsc_eta > 0.0 else 1.0
                engine._fsc_scale_rot[gi] = fsc_scale_rot
                # P4: same D₀^HS override applies to rotational K.
                K_r_eff = K_r * fsc_scale_rot / float(engine.cfg.d0_hs_scale)
                if fsc_scale_rot != 1.0:
                    log.info(
                        "FSC (rot, Yeh-Hummer): group %d  scale=%.6f  "
                        "f_rot shifts %.4f → %.4f",
                        gi, fsc_scale_rot,
                        search_xpt(K_r), search_xpt(K_r_eff),
                    )
                f_r = search_xpt(K_r_eff)
                if engine._do_desjarlais:
                    fm_r = f_r
                    Bg_r = refine_Bg_desjarlais(pwr_a, s0a, nu, nmol, f_r)
                    Bg_arr[1, gi] = Bg_r
                    hsdf_arr[1, gi] = hsdf_des(pwr_a, nu, nmol, f_r, Bg_r)
                else:
                    fm_r = f_r
                    hsdf_arr[1, gi] = _hsdf(pwr_a, nu, nmol, f_r, fm_r)
                fxpt[1, gi]    = f_r
                fmf_arr[1, gi] = fm_r

        if y_arr[gi] > 0.74:
            fxpt[:, gi] = 0.0

    return fxpt, fmf_arr, y_arr, hsdf_arr, Bg_arr

