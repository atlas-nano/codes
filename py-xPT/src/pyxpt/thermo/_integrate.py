# -*- coding: utf-8 -*-
"""
Thermodynamic-integration kernels — extracted from
``thermo/engine.py:_integrate_thermo`` + ``_integrate_gas_solid_split``.

Public API
----------
:func:`integrate_thermo(engine, …)`
    Integrate quantum/classical weighting functions over the gas/solid DoS
    to obtain (S, A, μ, E, Cv, dof) for every group × velocity type.
    Dispatches the gas/solid split via :func:`integrate_gas_solid_split`
    when ``cfg.show_xpt_split`` is set.

:func:`integrate_gas_solid_split(engine, …)`
    Compute separate thermodynamic properties for gas and solid phases
    and store in ``result.{property}_gas`` / ``result.{property}_solid``.

Like the other ``_*_compute`` modules, this reaches into engine state via
the ``engine`` argument but introduces no new state.  Bit-exact with
the pre-extraction implementation on the LJ-Ar regression suite (D1).
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

try:
    _trapz = np.trapezoid
except AttributeError:
    _trapz = np.trapz

from pyxpt.constants import (
    PI, KB, H, R, VLIGHT, NA, CALTOJ, VelType, ModexPT,
)
from pyxpt.thermo.utility import (
    hs_weighting, quantum_weights,
)

if TYPE_CHECKING:  # pragma: no cover
    from pyxpt.thermo.engine import xPTEngine, xPTResult

log = logging.getLogger(__name__)

TRANS  = int(VelType.TRANS)
IMVIB  = int(VelType.IMVIB)
ROTAT  = int(VelType.ROTAT)
TOTAL  = int(VelType.TOTAL)


def integrate_thermo(engine: "xPTEngine",
                      result, dos_gas, dos_sol, dos_all,
                      vacT, vacDF, V_use, P_use, T_use, trjE,
                      fxpt, hsdf_arr, y_arr,
                      ngrp, vt,
                      Bg_arr=None, dos_gas_pre=None) -> None:
    """
    Integrate quantum/classical weighting functions over the DoS to
    obtain thermodynamic properties for every group and velocity type.

    Bg_arr : (2, ngrp) optional — memory-time array from GLE-FMF modes.
        If provided and mode is GLE_FINITEMEM_*, the spectral excess entropy
        s_ex = -(R/π) ∫ φ(ν)·S_gas(ν) dν  with  φ = (ντ)²/(1+(ντ)²)
        is added to S_quantum/S_classical.
    dos_gas_pre : (ngrp, vt, nused) optional — the PRE-gas_gate gas DoS.  Used
        only to fill the per-channel gas-entropy diagnostic
        ``result.gas_entropy_component`` (the gas-phase quantum entropy each
        channel carries at its natural fluidicity, = what a Debye gate removes).
        Defaults to ``dos_gas`` (identical unless a gas_gate=debye zeroed a
        fluidicity); never affects the headline thermodynamics.
    """
    if dos_gas_pre is None:
        dos_gas_pre = dos_gas
    nu    = engine._pwrfreq
    nused = engine._nused
    freqs = result.frequencies
    s     = engine.sys
    hs_eos = engine.cfg.hs_eos
    d_dim  = int(engine.cfg.dimension)        # translational dimensionality (3=bulk)

    # Pre-compute BMCSL parameters (σᵢ, ξ moments) once for all groups
    if hs_eos == "bmcsl":
        bmcsl_sigmas, bmcsl_xi = engine._bmcsl_params(y_arr, V_use)
        eta_mix_cs = None   # not used in BMCSL mode
    else:
        bmcsl_sigmas = None
        bmcsl_xi = None
        eta_mix_cs = engine._eta_mix(y_arr)

    # When normalize is active, eng_avg/eng_std are stored per atom/molecule;
    # scale back to total-group values for the thermodynamic integrals.
    _do_norm_eng = engine.cfg.normalize or (engine._molecular and engine.cfg.per_molecule)

    # Z_sim from user-supplied P, V, T (Sec V.D bypass of HS EOS overestimate).
    # P [GPa] × V [Å³] × 1e-21 / (N × kB × T) [J] = dimensionless Z.
    _use_sim_z = bool(engine.cfg.use_sim_z)

    for gi in range(ngrp):
        grp   = s.groups[gi]
        #nmol  = (grp.nmol if engine._molecular else 1, 1)
        nmol  = grp.nmol #TAP 04/06/2026 Bugfix for previous line
        gmass = grp.mass
        gvol  = grp.volume if grp.volume > 0 else V_use  # V_total default → mixing entropy included
        _N_eng = max(grp.nmol if engine._molecular else grp.natom, 1) if _do_norm_eng else 1

        rT = np.zeros(3)
        if engine._molecular and engine._pI_acc.size > 0:
            engine._get_rot_temp(gi, rT)

        # Per-group Z_sim (computed once before the velocity loop so it is
        # consistent across translational/rotational components).
        _z_sim = None
        if _use_sim_z and nmol > 0 and vacT[gi, TRANS] > 0.0:
            _z_sim = (P_use * gvol * 1e-21) / (nmol * KB * vacT[gi, TRANS])

        for k in range(vt):
            T = vacT[gi, k]

            # Skip velocity components with negligible temperature
            if T <= 1e-20:  # Essentially zero in double precision
                continue

            # Effective gas-molecule count for Sackur-Tetrode entropy (C++ convention)
            nmol_hs = hsdf_arr[0, gi] / d_dim
            hs = hs_weighting(
                y_arr[gi], gmass / nmol, nmol_hs, vacT[gi, TRANS],
                vacT[gi, ROTAT] if vt > ROTAT else 0.0,
                gvol, rT, grp.rotsym,
                eta_mix=eta_mix_cs,
                hs_eos=hs_eos,
                sigma_i=float(bmcsl_sigmas[gi]) if bmcsl_sigmas is not None else 0.0,
                xi=bmcsl_xi,
                hs_entropy=engine.cfg.hs_entropy,
                z_override=_z_sim,
                dimension=d_dim,
            )

            # Arrays of gas and solid DoS
            pwr_g = dos_gas[gi, k, :nused]
            pwr_s = dos_sol[gi, k, :nused] if k < dos_sol.shape[1] else np.zeros(nused)

            scaled_temp = KB * T / (H * VLIGHT * 100.0)   # 1/cm
            u_arr = freqs / scaled_temp if scaled_temp > 0 else np.zeros(nused)

            # Vectorised quantum weights
            weq_v  = np.zeros(nused)
            wsq_v  = np.zeros(nused)
            waq_v  = np.zeros(nused)
            wcvq_v = np.zeros(nused)
            for i in range(nused):
                weq_v[i], wsq_v[i], waq_v[i], wcvq_v[i] = quantum_weights(u_arr[i])

            # Classical harmonic-solid weights: S_c = R·∫ g_s·(1-ln u) dω,
            # A_c = RT·∫ g_s·ln(u) dω.  At u=0, pwr_s=0 so the product is 0.
            _safe_ln = np.log(np.where(u_arr > 0, u_arr, 1.0))
            wsc_sol = np.where(u_arr > 0, 1.0 - _safe_ln, 0.0)
            wac_sol = np.where(u_arr > 0, _safe_ln, 0.0)

            wep = hs["wep"]; wsp = hs["wsp"]; wap = hs["wap"]; wmp = hs["wmp"]
            wcvp = hs["wcvp"]; wspd = hs["wspd"]
            wsr  = hs["wsr"];  war  = hs["war"];  wmr  = hs["wmr"]
            wer  = hs["wer"];  wcvr = hs["wcvr"]

            # Apply rotational hard-sphere correction for the active rotational type.
            # Adds wsr/war terms for gas-phase hard-sphere rotational entropy/free energy.
            w_rot_extra = (k == engine._rot_type and engine._molecular)

            # Trapezoid integration
            dof_int = _trapz(pwr_g + pwr_s, dx=nu)
            zpe_int = _trapz(pwr_s * u_arr * 0.5, dx=nu) if k == IMVIB else 0.0
            Eq_int  = _trapz(pwr_s * weq_v + pwr_g * wep, dx=nu)
            Ec_int  = _trapz(pwr_s + pwr_g * wep, dx=nu)
            Sq_int  = _trapz(pwr_s * wsq_v + pwr_g * wsp, dx=nu)
            Sc_int  = _trapz(pwr_s * wsc_sol + pwr_g * wsp, dx=nu)
            Aq_int  = _trapz(pwr_s * waq_v + pwr_g * wap, dx=nu)
            Ac_int  = _trapz(pwr_s * wac_sol + pwr_g * wap, dx=nu)
            # G = A + PV:  solid uses waq (PV_solid ≈ 0 for harmonic solid);
            # gas uses wmp = wap + Z_CS/3 (Carnahan-Starling PV/N per trans DOF)
            Gq_int  = _trapz(pwr_s * waq_v + pwr_g * wmp, dx=nu)
            Gc_int  = _trapz(pwr_s * wac_sol + pwr_g * wmp, dx=nu)
            Cvq_int = _trapz(pwr_s * wcvq_v + pwr_g * wcvp, dx=nu)
            Cvc_int = _trapz(pwr_s + pwr_g * wcvp, dx=nu)

            if w_rot_extra:
                Eq_int  += _trapz(pwr_g * (wer - wep), dx=nu)
                Ec_int  += _trapz(pwr_g * (wer - wep), dx=nu)
                Sq_int  += _trapz(pwr_g * (wsr - wsp), dx=nu)
                Sc_int  += _trapz(pwr_g * (wsr - wsp), dx=nu)
                Aq_int  += _trapz(pwr_g * (war - wap), dx=nu)
                Ac_int  += _trapz(pwr_g * (war - wap), dx=nu)
                Gq_int  += _trapz(pwr_g * (wmr - wmp), dx=nu)
                Gc_int  += _trapz(pwr_g * (wmr - wmp), dx=nu)
                Cvq_int += _trapz(pwr_g * (wcvr - wcvp), dx=nu)
                Cvc_int += _trapz(pwr_g * (wcvr - wcvp), dx=nu)

            # gas_gate diagnostic: the gas-phase quantum entropy this channel's
            # gas DoS contributes at its NATURAL (pre-gate) fluidicity.  Mirrors
            # the gas terms of Sq_int above (pwr_g·wsp, plus the rotational
            # hard-sphere wsr−wsp) but evaluated on dos_gas_pre, so it equals the
            # amount a Debye gate removes from S_quantum on a gated channel.
            pwr_g_pre = dos_gas_pre[gi, k, :nused]
            Sgas_pre  = _trapz(pwr_g_pre * wsp, dx=nu)
            if w_rot_extra:
                Sgas_pre += _trapz(pwr_g_pre * (wsr - wsp), dx=nu)
            result.gas_entropy_component[gi, k] = Sgas_pre * R

            # ── Apply Native Scaling (E in kJ/mol, S in J/mol) ──
            result.dof[gi, k]        = dof_int
            result.zpe[gi, k]        = zpe_int * T * R * 1e-3
            result.E_classical[gi,k] = Ec_int  * T * R * 1e-3
            result.E_quantum[gi, k]  = Eq_int  * T * R * 1e-3
            result.S_quantum[gi, k]  = Sq_int  * R
            result.S_classical[gi, k]= Sc_int  * R
            result.A_quantum[gi, k]  = Aq_int  * T * R * 1e-3
            result.A_classical[gi, k]= Ac_int  * T * R * 1e-3
            result.mu_quantum[gi, k]  = Gq_int  * T * R * 1e-3
            result.mu_classical[gi, k]= Gc_int  * T * R * 1e-3

            # Cv quantum/classical: if energy_std is available, use fluctuation-based correction
            Cvc_dos = Cvc_int * R        # Cv from DoS integration
            Cvq_dos = Cvq_int * R        # Cv from DoS integration

            sys_dof_total = float(np.sum(vacDF[:, vt - 1]))
            sys_dof_total = sys_dof_total if sys_dof_total > 0 else 1.0

            # Use the thermodynamic temperature (from config or trajectory average)
            # for the fluctuation Cv formula sigma²/(R·T²), not the VAC-derived
            # temperature (which is a kinetic average and differs in molecular mode).
            T_total_grp = T_use

            if grp.eng_std > 0 and T_total_grp > 0:
                # Group has its OWN energy std (from per-atom trajectory or GroupEnergyStd).
                # Cv_total = sigma²_group / (R·T_total²); distribute sub-columns by system
                # velocity-type fraction — matching legacy C++ behaviour:
                #   Cvc[j] = Cvc[TOTAL] * vacDF[sys][j] / vacDF[sys][TOTAL]
                # When normalize is active, eng_std is per atom/molecule; multiply by
                # _N_eng so that sigma2_E is the total-group variance, then the output
                # normalization divides by _N_eng to recover the per-particle Cv.
                sigma2_E = (_N_eng * grp.eng_std ** 2) * 1e6 * (CALTOJ ** 2)
                Cvc_fluct_total = sigma2_E / (R * T_total_grp * T_total_grp)
                if k == vt - 1:
                    Cvc_fluct = Cvc_fluct_total
                else:
                    Cvc_fluct = Cvc_fluct_total * (float(np.sum(vacDF[:, k])) / sys_dof_total)
                result.Cv_quantum[gi, k]  = Cvc_fluct + (Cvq_dos - Cvc_dos)
                result.Cv_classical[gi, k] = Cvc_fluct

            elif engine.cfg.energy_std > 0 and T_total_grp > 0:
                # System-level std: compute system Cv and distribute by group DOF fraction
                # for TOTAL column, then by system velocity-type fraction for sub-columns —
                # matching legacy C++ sigma2x[i] = sigma2x[sys] * DOF[i] / DOF[sys].
                sigma2_E = (engine.cfg.energy_std ** 2) * 1e6 * (CALTOJ ** 2)
                Cvc_sys = sigma2_E / (R * T_total_grp * T_total_grp)
                grp_dof_frac = (vacDF[gi, vt - 1] / sys_dof_total
                                if sys_dof_total > 0 else 0.0)
                Cvc_fluct_total = Cvc_sys * grp_dof_frac   # group's Cv = system Cv × DOF fraction
                if k == vt - 1:
                    Cvc_fluct = Cvc_fluct_total
                else:
                    Cvc_fluct = Cvc_fluct_total * (float(np.sum(vacDF[:, k])) / sys_dof_total)
                result.Cv_quantum[gi, k]  = Cvc_fluct + (Cvq_dos - Cvc_dos)
                result.Cv_classical[gi, k] = Cvc_fluct

            else:
                # No fluctuation data: use DoS integrals directly
                result.Cv_quantum[gi, k]  = Cvq_dos
                result.Cv_classical[gi, k] = Cvc_dos
            result.S_dos0[gi, k]     = dos_all[gi, k, 0]

            # MD energy and reference energy correction.
            # If the group has its OWN energy (from per-atom trajectory or
            # GroupEnergyAvg in the group file), use it directly for the TOTAL
            # column and distribute sub-columns by the group's own DOF fraction —
            # matching legacy C++ behaviour.
            # Otherwise fall back to the system energy (config override or trajectory
            # average) distributed by system DOF fraction.
            # ── MD Energy Correction (J/mol) ──
            if grp.eng_avg != 0:
                E_total = grp.eng_avg * _N_eng * 4.184  # kcal/mol -> kJ/mol (total group)
                grp_dof_total = vacDF[gi, vt - 1] if vacDF[gi, vt - 1] > 0 else 1.0
                E_md = E_total * vacDF[gi, k] / grp_dof_total
            else:
                sys_energy = engine.cfg.energy_avg if engine.cfg.energy_avg != 0 else trjE
                E_md = sys_energy * 4.184 * (vacDF[gi, k] / sys_dof_total
                                              if sys_dof_total > 0 else 0.0)

            result.E_md[gi, k] = E_md
            Eo = E_md - result.E_classical[gi, k]
            result.E_quantum[gi, k] += Eo
            result.A_quantum[gi, k] += Eo
            result.mu_quantum[gi, k] += Eo

            # Diffusivity (cm²/s) — same formula for all velocity types.
            # FSC: report D_inf rather than D_PBC.  Translational and
            # rotational get their own per-group scales; internal
            # vibrational modes are unaffected by Yeh-Hummer.
            if gmass > 0 and T > 0:
                _fsc_t = (engine._fsc_scale_trans[gi]
                          if engine._fsc_scale_trans is not None else 1.0)
                _fsc_r = (engine._fsc_scale_rot[gi]
                          if engine._fsc_scale_rot is not None else 1.0)
                if k == TRANS:
                    scale = _fsc_t
                elif k == engine._rot_type and engine._molecular:
                    scale = _fsc_r
                elif TOTAL < engine._vactype and k == TOTAL:
                    # The TOTAL row aggregates all components.  Use the
                    # translational scale as a representative — it dominates
                    # the magnitude; rotational FSC is two orders of
                    # magnitude smaller and the geometric average is a
                    # vanishing correction on top of that.
                    scale = _fsc_t
                else:
                    scale = 1.0
                result.diffusivity[gi, k] = (
                    result.S_dos0[gi, k] * R * T / (12.0 * VLIGHT * gmass)
                ) * 1e5 * scale

    # ── For molecular mode: overwrite TOTAL column = sum(Trans + rot_type + IMVIB) ──
    if engine._molecular and vt > TOTAL:
        for attr in ("dof", "zpe", "E_quantum", "E_classical", "E_md",
                     "S_classical", "S_quantum", "A_quantum", "A_classical",
                     "mu_quantum", "mu_classical",
                     "Cv_quantum", "Cv_classical","S_dos0", "diffusivity"):
            arr = getattr(result, attr)
            for gi in range(ngrp):
                arr[gi, TOTAL] = sum(arr[gi, k] for k in (TRANS, engine._rot_type, IMVIB))

    # ── Compute gas/solid split if requested ──────────────────────────────
    if engine.cfg.show_xpt_split:
        integrate_gas_solid_split(engine, 
            result, dos_gas, dos_sol, dos_all, vacT, vacDF,
            V_use, P_use, T_use, trjE, fxpt, hsdf_arr, y_arr, ngrp, vt
        )

    # Clean up NaN / Inf
    for attr in ("dof", "zpe", "E_quantum", "E_classical", "E_md",
                  "S_quantum", "A_quantum", "mu_quantum", "S_classical", "A_classical",
                  "mu_classical", "Cv_quantum", "Cv_classical", "S_dos0", "diffusivity",
                  # Gas/solid split
                  "dof_gas", "dof_solid", "T_gas", "T_solid",
                  "zpe_gas", "zpe_solid", "E_quantum_gas", "E_quantum_solid",
                  "E_classical_gas", "E_classical_solid", "E_md_gas", "E_md_solid",
                  "S_quantum_gas", "S_quantum_solid", "S_classical_gas", "S_classical_solid",
                  "A_quantum_gas", "A_quantum_solid", "A_classical_gas", "A_classical_solid",
                  "mu_quantum_gas", "mu_quantum_solid", "mu_classical_gas", "mu_classical_solid",
                  "Cv_quantum_gas", "Cv_quantum_solid", "Cv_classical_gas",
                  "Cv_classical_solid", "S_dos0_gas", "diffusivity_gas"):
        if hasattr(result, attr):
            arr = getattr(result, attr)
            np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0, copy=False)



def integrate_gas_solid_split(engine: "xPTEngine",
                               result, dos_gas, dos_sol, dos_all,
                               vacT, vacDF, V_use, P_use, T_use, trjE,
                               fxpt, hsdf_arr, y_arr, ngrp, vt) -> None:
    """
    Compute separate thermodynamic properties for gas and solid phases.
    Store results in result.{property}_gas and result.{property}_solid.
    """
    nu    = engine._pwrfreq
    nused = engine._nused
    freqs = result.frequencies
    s     = engine.sys
    hs_eos = engine.cfg.hs_eos
    d_dim  = int(engine.cfg.dimension)        # translational dimensionality (3=bulk)

    # Pre-compute BMCSL parameters once for all groups
    if hs_eos == "bmcsl":
        bmcsl_sigmas, bmcsl_xi = engine._bmcsl_params(y_arr, V_use)
        eta_mix_cs = None
    else:
        bmcsl_sigmas = None
        bmcsl_xi = None
        eta_mix_cs = engine._eta_mix(y_arr)

    _do_norm_eng = engine.cfg.normalize or (engine._molecular and engine.cfg.per_molecule)
    _use_sim_z = bool(engine.cfg.use_sim_z)

    for gi in range(ngrp):
        grp   = s.groups[gi]
        nmol  = grp.nmol   # <-- BUGFIX 1: Restored synchronization with main loop
        gmass = grp.mass
        gvol  = grp.volume if grp.volume > 0 else V_use  # V_total default → mixing entropy included
        _N_eng = max(grp.nmol if engine._molecular else grp.natom, 1) if _do_norm_eng else 1

        rT = np.zeros(3)
        if engine._molecular and engine._pI_acc.size > 0:
            engine._get_rot_temp(gi, rT)

        _z_sim = None
        if _use_sim_z and nmol > 0 and vacT[gi, TRANS] > 0.0:
            _z_sim = (P_use * gvol * 1e-21) / (nmol * KB * vacT[gi, TRANS])

        for k in range(vt):
            T = vacT[gi, k]

            # Skip velocity components with negligible temperature (same as main integration)
            if T <= 1e-20:  # Essentially zero in double precision
                continue

            nmol_hs = hsdf_arr[0, gi] / d_dim
            hs = hs_weighting(
                y_arr[gi], gmass / nmol, nmol_hs, vacT[gi, TRANS],
                vacT[gi, ROTAT] if vt > ROTAT else 0.0,
                gvol, rT, grp.rotsym,
                eta_mix=eta_mix_cs,
                hs_eos=hs_eos,
                sigma_i=float(bmcsl_sigmas[gi]) if bmcsl_sigmas is not None else 0.0,
                xi=bmcsl_xi,
                hs_entropy=engine.cfg.hs_entropy,
                z_override=_z_sim,
                dimension=d_dim,
            )

            # Get gas and solid DoS separately
            pwr_g = dos_gas[gi, k, :nused]
            pwr_s = dos_sol[gi, k, :nused] if k < dos_sol.shape[1] else np.zeros(nused)

            scaled_temp = KB * T / (H * VLIGHT * 100.0)
            u_arr = freqs / scaled_temp if scaled_temp > 0 else np.zeros(nused)

            # Quantum weights
            weq_v = np.zeros(nused)
            wsq_v = np.zeros(nused)
            waq_v = np.zeros(nused)
            wcvq_v = np.zeros(nused)
            for i in range(nused):
                weq_v[i], wsq_v[i], waq_v[i], wcvq_v[i] = quantum_weights(u_arr[i])

            # Classical harmonic-solid weights
            _safe_ln = np.log(np.where(u_arr > 0, u_arr, 1.0))
            wsc_sol = np.where(u_arr > 0, 1.0 - _safe_ln, 0.0)
            wac_sol = np.where(u_arr > 0, _safe_ln, 0.0)

            wep = hs["wep"]; wsp = hs["wsp"]; wap = hs["wap"]; wmp = hs["wmp"]
            wcvp = hs["wcvp"]
            wsr = hs["wsr"]; war = hs["war"]; wmr = hs["wmr"]
            wer = hs["wer"]; wcvr = hs["wcvr"]

            w_rot_extra = (k == engine._rot_type and engine._molecular)

            # ── GAS PHASE ─────────────────────────────────────────────────
            dof_g = _trapz(pwr_g, dx=nu)
            zpe_g = _trapz(pwr_g * u_arr * 0.5, dx=nu) if k == IMVIB else 0.0
            Eq_g  = _trapz(pwr_g * wep, dx=nu)
            Ec_g  = _trapz(pwr_g * wep, dx=nu)
            Sq_g  = _trapz(pwr_g * wsp, dx=nu)
            Sc_g  = _trapz(pwr_g * wsp, dx=nu)
            Aq_g  = _trapz(pwr_g * wap, dx=nu)
            Ac_g  = _trapz(pwr_g * wap, dx=nu)
            # Gas G: A_gas + PV_gas; PV per trans DOF = Z_CS·kT → weight = wmp = wap + Z_CS/3
            Gq_g  = _trapz(pwr_g * wmp, dx=nu)
            Gc_g  = _trapz(pwr_g * wmp, dx=nu)
            Cvq_g = _trapz(pwr_g * wcvp, dx=nu)
            Cvc_g = _trapz(pwr_g * wcvp, dx=nu)

            if w_rot_extra:
                Eq_g  += _trapz(pwr_g * (wer - wep), dx=nu)
                Ec_g  += _trapz(pwr_g * (wer - wep), dx=nu)
                Sq_g  += _trapz(pwr_g * (wsr - wsp), dx=nu)
                Sc_g  += _trapz(pwr_g * (wsr - wsp), dx=nu)
                Aq_g  += _trapz(pwr_g * (war - wap), dx=nu)
                Ac_g  += _trapz(pwr_g * (war - wap), dx=nu)
                Gq_g  += _trapz(pwr_g * (wmr - wmp), dx=nu)
                Gc_g  += _trapz(pwr_g * (wmr - wmp), dx=nu)
                Cvq_g += _trapz(pwr_g * (wcvr - wcvp), dx=nu)
                Cvc_g += _trapz(pwr_g * (wcvr - wcvp), dx=nu)

            # ── SOLID PHASE ───────────────────────────────────────────────
            dof_s = _trapz(pwr_s, dx=nu)
            zpe_s = _trapz(pwr_s * u_arr * 0.5, dx=nu) if k == IMVIB else 0.0
            Eq_s  = _trapz(pwr_s * weq_v, dx=nu)
            Ec_s  = _trapz(pwr_s, dx=nu)
            Sq_s  = _trapz(pwr_s * wsq_v, dx=nu)
            Sc_s  = _trapz(pwr_s * wsc_sol, dx=nu)
            Aq_s  = _trapz(pwr_s * waq_v, dx=nu)
            Ac_s  = _trapz(pwr_s * wac_sol, dx=nu)
            # Solid G: harmonic Debye model → modes V-independent → P=0 → G=A
            # Derivation: Z_solid = ∏_k e^{-u_k/2}/(1-e^{-u_k})
            #   F/N = kT ∫g(ν)·waq(ν)dν; μ = F/N (extensive solid)
            #   PV = 0 in harmonic limit (∂ω/∂V = 0) → G = A → wμq_solid = waq
            Gq_s  = _trapz(pwr_s * waq_v, dx=nu)    # same as Aq_s: G_solid = A_solid
            Gc_s  = _trapz(pwr_s * wac_sol, dx=nu)  # classical: G_solid = A_solid = kT·ln(u)
            Cvq_s = _trapz(pwr_s * wcvq_v, dx=nu)
            Cvc_s = _trapz(pwr_s, dx=nu)

            # ── MD energy correction (Eo = E_md - E_classical) ────────────
            if grp.eng_avg != 0:
                E_total = grp.eng_avg * _N_eng * 4.184  # kcal/mol -> kJ/mol (total group)
                grp_dof_total = vacDF[gi, vt - 1] if vacDF[gi, vt - 1] > 0 else 1.0
                E_md_total = E_total * vacDF[gi, k] / grp_dof_total
            else:
                sys_energy = engine.cfg.energy_avg if engine.cfg.energy_avg != 0 else trjE
                _sys_dof_vt = float(np.sum(vacDF[:, vt - 1]))
                E_md_total = sys_energy * 4.184 * (vacDF[gi, k] / _sys_dof_vt
                                                    if _sys_dof_vt > 0 else 0.0)

            Ec_total = (Ec_g + Ec_s) * T * R * 1e-3
            Eo = E_md_total - Ec_total

            dos_tot = dof_g + dof_s
            if dos_tot > 1e-10:
                frac_g = dof_g / dos_tot
                frac_s = dof_s / dos_tot
            else:
                frac_g = frac_s = 0.0

            Eo_g = Eo * frac_g
            Eo_s = Eo * frac_s

            # ── Cv fluctuation correction ──────────────────────────────────
            sys_dof_total = float(np.sum(vacDF[:, vt - 1]))
            sys_dof_total = sys_dof_total if sys_dof_total > 0 else 1.0
            T_total_grp = T_use

            if grp.eng_std > 0 and T_total_grp > 0:
                sigma2_E = (_N_eng * grp.eng_std ** 2) * 1e6 * (CALTOJ ** 2)
                Cvc_fluct_total = sigma2_E / (R * T_total_grp * T_total_grp)
                if k == vt - 1:
                    Cvc_fluct = Cvc_fluct_total
                else:
                    Cvc_fluct = Cvc_fluct_total * (float(np.sum(vacDF[:, k])) / sys_dof_total)
                Cvc_fluct_g = Cvc_fluct * frac_g
                Cvc_fluct_s = Cvc_fluct * frac_s
                Cvq_g_final = Cvc_fluct_g + (Cvq_g - Cvc_g) * R
                Cvq_s_final = Cvc_fluct_s + (Cvq_s - Cvc_s) * R
                Cvc_g_final = Cvc_fluct_g
                Cvc_s_final = Cvc_fluct_s
            elif engine.cfg.energy_std > 0 and T_total_grp > 0:
                sigma2_E = (engine.cfg.energy_std ** 2) * 1e6 * (CALTOJ ** 2)
                Cvc_sys = sigma2_E / (R * T_total_grp * T_total_grp)
                grp_dof_frac = (vacDF[gi, vt - 1] / sys_dof_total if sys_dof_total > 0 else 0.0)
                Cvc_fluct_total = Cvc_sys * grp_dof_frac
                if k == vt - 1:
                    Cvc_fluct = Cvc_fluct_total
                else:
                    Cvc_fluct = Cvc_fluct_total * (float(np.sum(vacDF[:, k])) / sys_dof_total)
                Cvc_fluct_g = Cvc_fluct * frac_g
                Cvc_fluct_s = Cvc_fluct * frac_s
                Cvq_g_final = Cvc_fluct_g + (Cvq_g - Cvc_g) * R
                Cvq_s_final = Cvc_fluct_s + (Cvq_s - Cvc_s) * R
                Cvc_g_final = Cvc_fluct_g
                Cvc_s_final = Cvc_fluct_s
            else:
                Cvq_g_final = Cvq_g * R
                Cvq_s_final = Cvq_s * R
                Cvc_g_final = Cvc_g * R
                Cvc_s_final = Cvc_s * R

            # ── Store gas phase results ────────────────────────────────────
            # <-- BUGFIX 2: Placed 1e-3 onto Eq_g * T * R where it belongs
            result.dof_gas[gi, k]          = dof_g
            result.T_gas[gi, k]            = T
            result.zpe_gas[gi, k]          = zpe_g * T * R * 1e-3
            result.E_classical_gas[gi, k]  = Ec_g * T * R * 1e-3
            result.E_quantum_gas[gi, k]    = Eq_g * T * R * 1e-3 + Eo_g
            result.S_quantum_gas[gi, k]    = Sq_g * R
            result.S_classical_gas[gi, k]  = Sc_g * R
            result.A_quantum_gas[gi, k]    = Aq_g * T * R * 1e-3 + Eo_g
            result.A_classical_gas[gi, k]  = Ac_g * T * R * 1e-3
            result.mu_quantum_gas[gi, k]    = Gq_g * T * R * 1e-3 + Eo_g
            result.mu_classical_gas[gi, k]  = Gc_g * T * R * 1e-3
            result.Cv_quantum_gas[gi, k]   = Cvq_g_final
            result.Cv_classical_gas[gi, k] = Cvc_g_final
            result.S_dos0_gas[gi, k]       = dos_gas[gi, k, 0]
            if gmass > 0 and T > 0:
                result.diffusivity_gas[gi, k] = (
                    result.S_dos0_gas[gi, k] * R * T / (12.0 * VLIGHT * gmass)
                ) * 1e5

            # ── Store solid phase results ──────────────────────────────────
            # <-- BUGFIX 3: Same scaling fix for the solid components
            result.dof_solid[gi, k]          = dof_s
            result.T_solid[gi, k]            = 0.0 if dof_s < 1e-6 else T
            result.zpe_solid[gi, k]          = zpe_s * T * R * 1e-3
            result.E_classical_solid[gi, k]  = Ec_s * T * R * 1e-3
            result.E_quantum_solid[gi, k]    = Eq_s * T * R * 1e-3 + Eo_s
            result.S_quantum_solid[gi, k]    = Sq_s * R
            result.S_classical_solid[gi, k]  = Sc_s * R
            result.A_quantum_solid[gi, k]    = Aq_s * T * R * 1e-3 + Eo_s
            result.A_classical_solid[gi, k]  = Ac_s * T * R * 1e-3
            result.mu_quantum_solid[gi, k]    = Gq_s * T * R * 1e-3 + Eo_s
            result.mu_classical_solid[gi, k]  = Gc_s * T * R * 1e-3
            result.Cv_quantum_solid[gi, k]   = Cvq_s_final
            result.Cv_classical_solid[gi, k] = Cvc_s_final

            # ── MD energy split (kJ/mol) ──
            if grp.eng_avg != 0:
                E_total = grp.eng_avg * _N_eng * 4.184  # kcal/mol -> kJ/mol (total group)
                grp_dof_total = vacDF[gi, vt - 1] if vacDF[gi, vt - 1] > 0 else 1.0
                E_md_total = E_total * vacDF[gi, k] / grp_dof_total
            else:
                sys_energy = engine.cfg.energy_avg if engine.cfg.energy_avg != 0 else trjE
                _sys_dof_vt = float(np.sum(vacDF[:, vt - 1]))
                E_md_total = sys_energy * 4.184 * (vacDF[gi, k] / _sys_dof_vt
                                                    if _sys_dof_vt > 0 else 0.0)

            result.E_md_gas[gi, k]   = E_md_total * frac_g
            result.E_md_solid[gi, k] = E_md_total * frac_s

    # ── Overwrite TOTAL column as sum of TRANS+rot_type+IMVIB (molecular only) ──
    ngrp = engine.sys.ngrp
    if engine._molecular and vt > TOTAL:
        for attr in ("dof_gas", "dof_solid", "zpe_gas", "zpe_solid",
                     "E_quantum_gas", "E_quantum_solid",
                     "E_classical_gas", "E_classical_solid",
                     "E_md_gas", "E_md_solid",
                     "S_quantum_gas", "S_quantum_solid",
                     "S_classical_gas", "S_classical_solid",
                     "A_quantum_gas", "A_quantum_solid",
                     "A_classical_gas", "A_classical_solid",
                     "mu_quantum_gas", "mu_quantum_solid",
                     "mu_classical_gas", "mu_classical_solid",
                     "Cv_quantum_gas", "Cv_quantum_solid",
                     "Cv_classical_gas", "Cv_classical_solid",
                     "S_dos0_gas", "diffusivity_gas"):
            if not hasattr(result, attr):
                continue
            arr = getattr(result, attr)
            for gi in range(ngrp):
                arr[gi, TOTAL] = sum(arr[gi, k] for k in (TRANS, engine._rot_type, IMVIB))


