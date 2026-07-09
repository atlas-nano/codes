"""
VAC accumulation and xPT (1PT/2PT/3PT) thermodynamics computation engine (v2).

Key design choices vs. the v1 port
-----------------------------------
* Velocities are accumulated from the trajectory *in one vectorised pass*
  into a (vactype, nsteps, natom, 3) float32 array rather than reading the
  trajectory twice (once per atom pass as in the original C++).
* The ACF is computed for all atoms simultaneously via a single batched FFT
  call, avoiding the per-atom Python loop.
* All thermodynamic integration uses ``numpy.trapz`` / ``scipy.integrate``
  rather than explicit loops.
* ``numpy.einsum`` is used for mass-weighted accumulation into group VACs.
* Logging replaces direct ``print`` calls.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Sequence

import numpy as np

# NumPy 2.0 renamed trapz → trapezoid; support both
try:
    _trapz = np.trapezoid
except AttributeError:
    _trapz = np.trapz

from scipy.integrate import cumulative_trapezoid
from scipy.ndimage import gaussian_filter1d, minimum_filter1d

from pyxpt.constants import (
    PI, KB, H, R, VLIGHT, NA, CALTOJ, ELEMENTS,
    VelType, VELTYPE, ModexPT, COPYRIGHT,
    YEH_HUMMER_XI, YEH_HUMMER_XI_ROT,
)

from pyxpt.config import Config
from pyxpt.io.trajectory import System, FrameData, GroupInfo
from pyxpt.thermo.utility import (
    search_xpt, xpt_partition, hsdf as _hsdf, trapz_dos,
    hs_weighting, sqweighting, quantum_weights,
    mixture_packing_fraction, bmcsl_sigma_from_y, _bmcsl_xi,
    refine_Bg_desjarlais, xpt_partition_des, hsdf_des,
    decompose_velocities, decompose_velocities_batch,
    detect_plateau,
)
from pyxpt.core.gpu import get_backend
from pyxpt.core import fft as _onfft

log = logging.getLogger(__name__)

# Convenience aliases
TRANS = int(VelType.TRANS)
ANGUL = int(VelType.ANGUL)
IMVIB = int(VelType.IMVIB)
ROTAT = int(VelType.ROTAT)
TOTAL = int(VelType.TOTAL)


# ── Kabsch rotation helper ────────────────────────────────────────────────────

def _kabsch_rotation(ref_pos: np.ndarray, cur_pos: np.ndarray,
                     masses: np.ndarray) -> np.ndarray:
    """
    Find rotation matrix R (3×3) minimising ||cur_pos − ref_pos @ R.T||²_F
    (mass-weighted Kabsch algorithm via SVD).

    Both *ref_pos* and *cur_pos* must already be COM-centred.

    Convention:  cur_pos ≈ ref_pos @ R.T
    To rotate a lab-frame velocity v to the reference frame:  v_ref = v @ R
    """
    w = masses / masses.sum()
    H = ref_pos.T @ (cur_pos * w[:, None])   # (3, 3) weighted covariance
    U, _, Vt = np.linalg.svd(H)
    # Ensure proper rotation (det = +1, not a reflection)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return R   # (3, 3)


# ── Warning collector (C3 silent-fallback surface) ──────────────────────────


class _WarningCollector(logging.Handler):
    """Capture WARNING-level records from pyxpt loggers during compute().

    Attached at the start of :meth:`xPTEngine.compute` and detached in
    a ``finally`` block, then read by :meth:`xPTEngine._write_log` to
    emit a single "WARNINGS / FALLBACKS APPLIED" summary at the top of
    ``.out.log``.  Lets every existing silent-fallback warning
    (kernel-inversion truncation, plateau-not-detected, scalar→matrix
    promotion, etc.) surface in one place without modifying the warning
    emitters themselves.

    The handler is **module-local** — it captures records propagated
    through any pyxpt.* logger but doesn't interfere with the user's
    own logging configuration (console output, custom handlers, etc.).
    """

    def __init__(self):
        super().__init__(level=logging.WARNING)
        self.records: list[str] = []
        # Strip the timestamp / level prefix so the summary block stays terse.
        # Format is ``logger.name: message`` (e.g.,
        # ``pyxpt.thermo.utility: cage-memory entropy ...``).
        self.setFormatter(logging.Formatter("%(name)s: %(message)s"))

    def emit(self, record):
        try:
            self.records.append(self.format(record))
        except Exception:
            self.handleError(record)


# ── Result container ──────────────────────────────────────────────────────────

class xPTResult:
    """
    Container for all per-group thermodynamic results.

    Attributes mirror the columns in the original ``.thermo`` output file.
    All energies in kJ/mol/SimBox, entropies in J/(K·mol)/SimBox.
    
    Gas/solid split properties (when show_xpt_split enabled):
    - dof_gas, dof_solid: degrees of freedom
    - T_gas, T_solid: temperatures from gas/solid DOS weighting
    - zpe_gas, zpe_solid, etc.: split thermodynamic properties

    Thermodynamics with classical (instead of quantum) weights (when show_classical_thermo enabled):
    - A_c, S_c, E_c
    - A_c_gas, A_c_solid, etc.: split thermodynamic properties
    """
    __slots__ = (
        "ngrp", "vactype",
        "unit_E", "unit_S", "unit_T", "unit_P", "unit_V", "unit_D",
        "temperature",    # (ngrp, vactype)
        "pressure",       # scalar
        "volume",         # (ngrp,)
        "dof",            # (ngrp, vactype)
        "zpe",            # (ngrp, vactype)  zero-point energy
        "E_md",           # (ngrp, vactype)  MD total energy
        "E_quantum",      # (ngrp, vactype)  quantum energy
        "E_classical",    # (ngrp, vactype)  classical energy
        "S_quantum",      # (ngrp, vactype)  quantum entropy
        "S_classical",    # (ngrp, vactype)  classical entropy
        "A_quantum",      # (ngrp, vactype)  quantum Helmholtz free energy
        "A_classical",    # (ngrp, vactype)  classical Helmholtz free energy
        "mu_quantum",      # (ngrp, vactype)  quantum chemical potential μ = G/N = A_solid/N + (A+PV)_gas/N
        "mu_classical",    # (ngrp, vactype)  classical chemical potential
        "mu_quantum_gas",  # (ngrp, vactype)  μ gas component: (A_gas + Z_CS·kT)/N_gas
        "mu_quantum_solid",# (ngrp, vactype)  μ solid component: A_solid/N_solid (harmonic → PV=0)
        "mu_classical_gas",# (ngrp, vactype)  classical μ gas component
        "mu_classical_solid",# (ngrp, vactype) classical μ solid component
        "mu_quantum_at_ref_p",   # (ngrp, vactype) μ_q + v̄·(P_ref−P_sim), reference-pressure corrected
        "mu_classical_at_ref_p", # (ngrp, vactype) classical analogue
        "Cv_quantum",     # (ngrp, vactype)  quantum Cv
        "Cv_classical",   # (ngrp, vactype)  classical Cv
        "S_dos0",         # (ngrp, vactype)  DoS at zero frequency (cm)
        "diffusivity",    # (ngrp, vactype)  cm²/s
        "fluidicity",     # (2, ngrp)        trans/rot
        "packing_fraction",  # (ngrp,)       hard-sphere packing fraction y_i
        # ── gas_gate (Debye-consistency) diagnostic (gate c, 2026-06-13) ─────────
        # Per gas-gate channel, indexed [frow, gi] with frow 0=TRANS, 1=ROT (same
        # rows as `fluidicity`).  Populated only when cfg.gas_gate != "none".
        "gas_entropy_component",   # (ngrp, vactype)  per-channel gas-phase quantum
                                   # entropy [J/mol·K] at the NATURAL (pre-gate)
                                   # fluidicity — the gas S each channel carries;
                                   # the magnitude a debye gate would remove.
        "gas_gate_fired",          # (2, ngrp) bool — channel gated (P < tol)
        "gas_gate_plateau_ratio",  # (2, ngrp) float — P=<S>[0,10]/<S>[20,40]; NaN if not evaluated
        "gas_gate_fluidicity_pre", # (2, ngrp) float — channel fluidicity before the gate
        "gas_gate_dS_removed",     # (2, ngrp) float — spurious gas entropy
                                   # [J/mol·K PER molecule/atom] removed (gas_gate=
                                   # debye) or that WOULD be removed (debye_warn);
                                   # 0 where the gate did not fire
        # ── [transport] (Phase T1+) ──────────────────────────────────────────
        # Each entry is (mean, stderr, ci_lo, ci_hi); ci_* = NaN when not requested.
        "transport_D_trans",         # (ngrp, 4)   translational D [cm²/s]
        "transport_D_rot",           # (ngrp, 4)   I_ω = ∫⟨ω·ω⟩ dt per-axis-per-molecule [1/ps]
        "transport_omega_sq",        # (ngrp,)     ⟨ω_α²⟩(0) per-axis [1/ps²]
        "transport_eta_shear",       # (4,)        shear viscosity [Pa·s]
        "transport_eta_bulk",        # (4,)        bulk viscosity η_b [Pa·s]
        # ── [mechanics] — frequency-dependent linear response (Phase M1) ────
        "mechanics_G_t",             # (n_lag,)    shear relaxation G(t) [Pa] (first-block trace)
        "mechanics_G_t_dt_ps",       # scalar      time step of G_t [ps]
        "mechanics_G_freq_THz",      # (nfreq,)    frequency axis for G'(ω), G''(ω)
        "mechanics_G_prime",         # (nfreq,)    storage modulus G'(ω) [Pa]
        "mechanics_G_double",        # (nfreq,)    loss modulus    G''(ω) [Pa]
        "mechanics_tan_delta_shear", # (nfreq,)    G''/G' loss tangent
        "mechanics_K_t",             # (n_lag,)    bulk relaxation K(t) [Pa] (first-block trace)
        "mechanics_K_t_dt_ps",       # scalar      time step of K_t [ps]
        "mechanics_K_freq_THz",      # (nfreq,)    frequency axis for K'(ω), K''(ω)
        "mechanics_K_prime",         # (nfreq,)    storage bulk modulus K'(ω) [Pa]
        "mechanics_K_double",        # (nfreq,)    loss bulk modulus    K''(ω) [Pa]
        "mechanics_tan_delta_bulk",  # (nfreq,)    K''/K' loss tangent
        "mechanics_eps_freq_THz",    # (nfreq,)    frequency axis for ε(ω)
        "mechanics_eps_real",        # (nfreq,)    real part ε'(ω)
        "mechanics_eps_imag",        # (nfreq,)    imaginary part ε''(ω)
        "mechanics_tan_delta_eps",   # (nfreq,)    ε''/ε' dielectric loss tangent
        "mechanics_M_t_eA",          # (n_lag,3)   total dipole trace [e·Å]  (stashed)
        "mechanics_M_dt_ps",         # scalar      time step of M_t [ps]
        "mechanics_M_V_m3",          # scalar      cell volume [m³] for ε(ω) prefactor
        "mechanics_M_T_K",           # scalar      temperature [K] for ε(ω) prefactor
        # Phase M2 — elastic-constant tensor and derived quantities
        "mechanics_C_ij",            # (6, 6)      elastic constants [Pa]
        "mechanics_elastic_method_used",  # str    "strain_fluct" | "stress_fluct" | ""
        "mechanics_K_VRH",           # scalar      Voigt-Reuss-Hill bulk modulus [Pa]
        "mechanics_G_VRH",           # scalar      Voigt-Reuss-Hill shear modulus [Pa]
        "mechanics_E_VRH",           # scalar      Young's modulus [Pa]
        "mechanics_nu_VRH",          # scalar      Poisson's ratio [-]
        "mechanics_v_L",             # scalar      longitudinal sound velocity [m/s]
        "mechanics_v_T",             # scalar      transverse sound velocity [m/s]
        "mechanics_v_D",             # scalar      Debye-averaged velocity [m/s]
        "mechanics_A_zener",         # scalar      Zener anisotropy A_Z [-]
        "mechanics_A_universal",     # scalar      universal anisotropy A^U [-]
        "mechanics_pugh_ratio",      # scalar      G/K ductility indicator [-]
        "mechanics_born_stable",     # bool        all C-eigenvalues > 0
        "mechanics_C_eigenvalues",   # (6,)        eigenvalues of C [Pa]
        "mechanics_debye_T",         # scalar      Debye temperature [K]
        # Phase M3b — multi-mode Maxwell (Prony) fit results
        "mechanics_G_maxwell_tau",   # (n_modes,)  shear relaxation times [ps]
        "mechanics_G_maxwell_amp",   # (n_modes,)  shear mode amplitudes [Pa]
        "mechanics_G_maxwell_inf",   # scalar      shear long-time G_∞ [Pa]
        "mechanics_G_maxwell_resid", # scalar      RMS residual / ||G(t)||
        "mechanics_K_maxwell_tau",   # (n_modes,)  bulk relaxation times [ps]
        "mechanics_K_maxwell_amp",   # (n_modes,)  bulk mode amplitudes [Pa]
        "mechanics_K_maxwell_inf",   # scalar      bulk long-time K_∞ [Pa]
        "mechanics_K_maxwell_resid", # scalar      RMS residual / ||K(t)||
        "mechanics_method_log",      # list[str]   per-property method strings (.mechanics output)
        # Gas/solid split properties (computed when show_xpt_split = true)
        "dof_gas",        # (ngrp, vactype)
        "dof_solid",      # (ngrp, vactype)
        "T_gas",          # (ngrp, vactype)
        "T_solid",        # (ngrp, vactype)
        "zpe_gas",        # (ngrp, vactype)
        "zpe_solid",      # (ngrp, vactype)
        "E_md_gas",       # (ngrp, vactype)
        "E_md_solid",     # (ngrp, vactype)
        "E_quantum_gas",  # (ngrp, vactype)
        "E_quantum_solid",# (ngrp, vactype)
        "E_classical_gas",# (ngrp, vactype)
        "E_classical_solid",# (ngrp, vactype)
        "S_quantum_gas",  # (ngrp, vactype)
        "S_quantum_solid",# (ngrp, vactype)
        "S_classical_gas",# (ngrp, vactype)
        "S_classical_solid",# (ngrp, vactype)
        "A_quantum_gas",  # (ngrp, vactype)
        "A_quantum_solid",# (ngrp, vactype)
        "A_classical_gas",# (ngrp, vactype)
        "A_classical_solid",# (ngrp, vactype)
        "Cv_quantum_gas", # (ngrp, vactype)
        "Cv_quantum_solid",# (ngrp, vactype)
        "Cv_classical_gas",# (ngrp, vactype)
        "Cv_classical_solid",# (ngrp, vactype)
        "S_dos0_gas",     # (ngrp, vactype)
        "diffusivity_gas",# (ngrp, vactype)
        # Full spectra
        "dos",            # (ngrp, vactype, nused)  full power spectrum
        "dos_gas",        # (ngrp, vactype, nused)
        "dos_solid",      # (ngrp, vactype, nused)
        "dos_cage",       # (ngrp, vactype, nused) — cage excess, TRANS + rot channel
        "S_cage",         # (ngrp, vactype) — cage-memory + anharmonic ΔS already folded
                          #   into S_quantum (3PT solid-side correction), kept for display
        "vac",            # (ngrp, vactype, vacmaxf, 3, 3)  autocorrelation tensor
        "frequencies",    # (nused,) cm⁻¹
        # IR/Raman split (only populated when partial charges are present)
        "has_ir",            # bool
        "dos_ir",            # (ngrp, nused)  IR-active DoS (classical, DACF)
        "dos_raman",         # (ngrp, nused)  Raman residual: basis − dos_ir  (classical)
        "dos_ir_qcf",        # (ngrp, nused)  IR intensity QCF-corrected: dos_ir × u/(1−e⁻ᵘ)
        "dos_raman_qcf",     # (ngrp, nused)  Raman residual QCF-corrected: dos_raman × 1/(1−e⁻ᵘ)
        # BPM Raman (only populated when [bond_polarizability] params provided)
        "has_bpm",               # bool
        "dos_raman_bpm",         # (ngrp, nused)  BPM Raman, Frobenius norm (classical)
        "dos_raman_bpm_qcf",     # (ngrp, nused)  Frobenius × QCF
        "dos_raman_bpm_iso",     # (ngrp, nused)  isotropic Placzek invariant ā'² (classical)
        "dos_raman_bpm_iso_qcf", # (ngrp, nused)  isotropic × QCF
        "dos_raman_bpm_aniso",   # (ngrp, nused)  anisotropic Placzek invariant γ'² (classical)
        "dos_raman_bpm_aniso_qcf",# (ngrp, nused) anisotropic × QCF
        # SP-DoS Raman (only populated when [spdos_raman] is active)
        "has_spdos",                 # bool
        "dos_raman_spdos",           # (ngrp, nused)  SP-DoS Raman (classical, rescaled)
        "dos_raman_spdos_qcf",       # (ngrp, nused)  SP-DoS × Raman QCF
        # MC1: cross-channel velocity autocorrelators (off-diagonals of the
        # trans/rot/vib velocity tensor).  Populated only when the user sets
        # [output] cross_vac_diagnostic = 1 on a molecular run.
        "has_cross_vac",     # bool
        "cross_vac_pairs",   # list[(int, int)] velocity-type pairs computed
        "cross_vac",         # (ngrp, n_pairs, vacmaxf, 3, 3)  C_αβ(t)
        "cross_pwr",         # (ngrp, n_pairs, nused, 3, 3)    |S_αβ(ν)|
        "cross_coupling",    # (ngrp, n_pairs) max_t |tr C_αβ|/√(tr C_αα(0) tr C_ββ(0))
    )

    def __init__(self, ngrp: int, vactype: int, nused: int, vacmaxf: int,
                 nd: int = 3) -> None:
        shape = (ngrp, vactype)
        # Initialize default native py-xPT thermodynamics units
        self.unit_E = "kJ/mol"
        self.unit_S = "J/mol·K"
        self.unit_T = "K"
        self.unit_P = "GPa"
        self.unit_V = "Å³"
        self.unit_D = "cm²/s"
        for attr in ("temperature", "dof", "zpe", "E_quantum", "E_classical",
                     "E_md", "S_quantum", "S_classical", "A_quantum", "A_classical",
                     "mu_quantum", "mu_classical",
                     "mu_quantum_gas", "mu_quantum_solid",
                     "mu_classical_gas", "mu_classical_solid",
                     "mu_quantum_at_ref_p", "mu_classical_at_ref_p",
                     "Cv_quantum", "Cv_classical", "S_dos0", "diffusivity",
                     # Gas/solid split properties
                     "dof_gas", "dof_solid", "T_gas", "T_solid",
                     "zpe_gas", "zpe_solid",
                     "E_md_gas", "E_md_solid",
                     "E_quantum_gas", "E_quantum_solid",
                     "E_classical_gas", "E_classical_solid",
                     "S_quantum_gas", "S_quantum_solid",
                     "S_classical_gas", "S_classical_solid",
                     "A_quantum_gas", "A_quantum_solid",
                     "A_classical_gas", "A_classical_solid",
                     "Cv_quantum_gas", "Cv_quantum_solid",
                     "Cv_classical_gas", "Cv_classical_solid",
                     "S_dos0_gas", "diffusivity_gas"):
            object.__setattr__(self, attr, np.zeros(shape))
        object.__setattr__(self, "pressure", 0.0)
        object.__setattr__(self, "volume", np.zeros(ngrp))
        object.__setattr__(self, "fluidicity", np.zeros((2, ngrp)))
        object.__setattr__(self, "packing_fraction", np.zeros(ngrp))
        # gas_gate diagnostic fields (filled by compute() when gas_gate != none)
        object.__setattr__(self, "gas_entropy_component", np.zeros(shape))
        object.__setattr__(self, "gas_gate_fired", np.zeros((2, ngrp), dtype=bool))
        object.__setattr__(self, "gas_gate_plateau_ratio", np.full((2, ngrp), np.nan))
        object.__setattr__(self, "gas_gate_fluidicity_pre", np.zeros((2, ngrp)))
        object.__setattr__(self, "gas_gate_dS_removed", np.zeros((2, ngrp)))
        # Transport coefficient placeholders (filled by _compute_transport)
        object.__setattr__(self, "transport_D_trans", np.full((ngrp, 4), float("nan")))
        object.__setattr__(self, "transport_D_rot",   np.full((ngrp, 4), float("nan")))
        object.__setattr__(self, "transport_omega_sq", np.full(ngrp, float("nan")))
        object.__setattr__(self, "transport_eta_shear",  np.full(4, float("nan")))
        object.__setattr__(self, "transport_eta_bulk",       np.full(4, float("nan")))
        # Mechanics placeholders (filled by _compute_mechanics post-pass)
        object.__setattr__(self, "mechanics_G_t",            np.zeros(0))
        object.__setattr__(self, "mechanics_G_t_dt_ps",      0.0)
        object.__setattr__(self, "mechanics_G_freq_THz",     np.zeros(0))
        object.__setattr__(self, "mechanics_G_prime",        np.zeros(0))
        object.__setattr__(self, "mechanics_G_double",       np.zeros(0))
        object.__setattr__(self, "mechanics_tan_delta_shear", np.zeros(0))
        object.__setattr__(self, "mechanics_K_t",            np.zeros(0))
        object.__setattr__(self, "mechanics_K_t_dt_ps",      0.0)
        object.__setattr__(self, "mechanics_K_freq_THz",     np.zeros(0))
        object.__setattr__(self, "mechanics_K_prime",        np.zeros(0))
        object.__setattr__(self, "mechanics_K_double",       np.zeros(0))
        object.__setattr__(self, "mechanics_tan_delta_bulk", np.zeros(0))
        object.__setattr__(self, "mechanics_eps_freq_THz",   np.zeros(0))
        object.__setattr__(self, "mechanics_eps_real",       np.zeros(0))
        object.__setattr__(self, "mechanics_eps_imag",       np.zeros(0))
        object.__setattr__(self, "mechanics_tan_delta_eps",  np.zeros(0))
        object.__setattr__(self, "mechanics_M_t_eA",         np.zeros((0, 3)))
        object.__setattr__(self, "mechanics_M_dt_ps",        0.0)
        object.__setattr__(self, "mechanics_M_V_m3",         0.0)
        object.__setattr__(self, "mechanics_M_T_K",          0.0)
        # Phase M2 elastic-constant placeholders
        object.__setattr__(self, "mechanics_C_ij",           np.full((6, 6), float("nan")))
        object.__setattr__(self, "mechanics_elastic_method_used", "")
        object.__setattr__(self, "mechanics_K_VRH",          float("nan"))
        object.__setattr__(self, "mechanics_G_VRH",          float("nan"))
        object.__setattr__(self, "mechanics_E_VRH",          float("nan"))
        object.__setattr__(self, "mechanics_nu_VRH",         float("nan"))
        object.__setattr__(self, "mechanics_v_L",            float("nan"))
        object.__setattr__(self, "mechanics_v_T",            float("nan"))
        object.__setattr__(self, "mechanics_v_D",            float("nan"))
        object.__setattr__(self, "mechanics_A_zener",        float("nan"))
        object.__setattr__(self, "mechanics_A_universal",    float("nan"))
        object.__setattr__(self, "mechanics_pugh_ratio",     float("nan"))
        object.__setattr__(self, "mechanics_born_stable",    False)
        object.__setattr__(self, "mechanics_C_eigenvalues",  np.full(6, float("nan")))
        object.__setattr__(self, "mechanics_debye_T",        float("nan"))
        object.__setattr__(self, "mechanics_G_maxwell_tau",  np.zeros(0))
        object.__setattr__(self, "mechanics_G_maxwell_amp",  np.zeros(0))
        object.__setattr__(self, "mechanics_G_maxwell_inf",  float("nan"))
        object.__setattr__(self, "mechanics_G_maxwell_resid", float("nan"))
        object.__setattr__(self, "mechanics_K_maxwell_tau",  np.zeros(0))
        object.__setattr__(self, "mechanics_K_maxwell_amp",  np.zeros(0))
        object.__setattr__(self, "mechanics_K_maxwell_inf",  float("nan"))
        object.__setattr__(self, "mechanics_K_maxwell_resid", float("nan"))
        object.__setattr__(self, "mechanics_method_log",     [])
        object.__setattr__(self, "dos",      np.zeros((ngrp, vactype, nused)))
        object.__setattr__(self, "dos_gas",  np.zeros((ngrp, vactype, nused)))
        object.__setattr__(self, "dos_solid",np.zeros((ngrp, vactype, nused)))
        # solid DoS; TRANS channel (cage_entropy) and rotational channel
        # ROTAT/ANGUL (cage_entropy_rot, block 9d).
        object.__setattr__(self, "dos_cage", np.zeros((ngrp, vactype, nused)))
        object.__setattr__(self, "S_cage",   np.zeros((ngrp, vactype)))
        object.__setattr__(self, "vac",      np.zeros((ngrp, vactype, vacmaxf, nd, nd)))
        object.__setattr__(self, "frequencies", np.zeros(nused))
        object.__setattr__(self, "ngrp", ngrp)
        object.__setattr__(self, "vactype", vactype)
        object.__setattr__(self, "has_ir", False)
        for attr in ("dos_ir", "dos_raman", "dos_ir_qcf", "dos_raman_qcf"):
            object.__setattr__(self, attr, np.zeros((ngrp, nused)))
        object.__setattr__(self, "has_bpm", False)
        for attr in ("dos_raman_bpm",     "dos_raman_bpm_qcf",
                     "dos_raman_bpm_iso", "dos_raman_bpm_iso_qcf",
                     "dos_raman_bpm_aniso", "dos_raman_bpm_aniso_qcf"):
            object.__setattr__(self, attr, np.zeros((ngrp, nused)))
        object.__setattr__(self, "has_spdos", False)
        for attr in ("dos_raman_spdos", "dos_raman_spdos_qcf"):
            object.__setattr__(self, attr, np.zeros((ngrp, nused)))
        # MC1 cross-channel diagnostic placeholders (filled by _compute_cross_vac)
        object.__setattr__(self, "has_cross_vac",   False)
        object.__setattr__(self, "cross_vac_pairs", [])
        object.__setattr__(self, "cross_vac",       np.zeros((ngrp, 0, vacmaxf, nd, nd)))
        object.__setattr__(self, "cross_pwr",       np.zeros((ngrp, 0, nused, nd, nd)))
        object.__setattr__(self, "cross_coupling",  np.zeros((ngrp, 0)))


# ── Main engine ───────────────────────────────────────────────────────────────

class xPTEngine:
    """
    Full xPT (1PT/2PT/3PT) computation engine.

    Usage
    -----
    >>> engine = xPTEngine(cfg, system)
    >>> engine.accumulate(frame_iterator)
    >>> result = engine.compute()
    >>> engine.write(result)
    """

    def __init__(self, cfg: Config, system: System) -> None:
        self.cfg = cfg
        self.sys = system
        self._nsteps = 0
        self._vacmaxf = 0
        self._tot_N = 0
        self._nused = 0
        self._vacdtime = 0.0   # ps between frames used
        self._pwrfreq = 0.0    # cm⁻¹ per bin

        # Initialize GPU backend
        self._backend = get_backend(use_gpu=cfg.use_gpu)

        # Derived flags
        self._molecular      = cfg.molecular
        # Drives the VAC tensor / DoS / DoF axis count.  For d≠3 the trajectory
        # reader supplies (natom, d) velocities and cfg.dimension == d; the
        # molecular decomposition (rot/vib) is only supported at d=3.
        self._nd        = int(getattr(cfg, "dimension", 3) or 3)
        self._apply_xpt      = cfg.do_xpt
        self._do_desjarlais  = cfg.do_desjarlais
        self._vactype   = VELTYPE + 1 if self._molecular else 1
        # Phase MEM-B: skip 2PT machinery (per-atom VAC, DoS, partition,
        # free energy) for mechanics-only / transport-only workflows.
        # Auto-disable 2PT-dependent transport flags with a warning.
        self._disable_xpt = bool(cfg.disable_xpt)
        if self._disable_xpt:
            # transport_diffusion needs the full _vacvv per-atom velocity
            # time series for the VACF integral; transport_rotation needs
            # the per-molecule angular-velocity stream from
            # decompose_velocities_batch.  Both are skipped under
            # disable_xpt.
            for _flag, _attr in (
                ("transport_diffusion",      "transport_diffusion"),
                ("transport_rotation",       "transport_rotation"),
            ):
                if getattr(cfg, _attr, False):
                    log.warning("disable_xpt=True: auto-disabling %s "
                                "(needs per-atom velocity time series)", _flag)
                    setattr(cfg, _attr, False)
        # Active rotational velocity type: ROTAT (ω×r per atom) or ANGUL (ω·√I per mol)
        self._rot_type  = ROTAT if (self._molecular and cfg.use_vrotat) else ANGUL

        # Accumulated arrays (allocated in _setup)
        self._vacvv: np.ndarray     = np.array([])  # (vactype, nsteps, natom, 3)
        self._angvv: np.ndarray     = np.array([])  # (nsteps, nmol, 3) mass-weighted angular velocity per molecule [sqrt(g/mol)·Å/ps]
        self._omega_lab: np.ndarray = np.array([])  # (nsteps, nmol, 3) true angular velocity per molecule [rad/ps]
        self._dipvv: np.ndarray     = np.array([])  # (nsteps, nmol, 3) charge-weighted velocity for DACF
        self._inddip: np.ndarray    = np.array([])  # (nsteps, nmol, 3) per-mol induced dipole sum (AMOEBA, e·Å)
        self._polvv: np.ndarray     = np.array([])  # (nsteps, nmol, 6) anisotropic bond polarizability
        self._charge_sum: np.ndarray = np.array([]) # (natom,) running sum of per-frame charges

        # BPM bond arrays (precomputed in _setup; empty when no [bond_polarizability] params)
        self._bpm_mol_ids:    np.ndarray = np.array([], dtype=np.intp)  # (nbonds,) mol index
        self._bpm_ai:         np.ndarray = np.array([], dtype=np.intp)  # (nbonds,) global atom i
        self._bpm_aj:         np.ndarray = np.array([], dtype=np.intp)  # (nbonds,) global atom j
        self._bpm_dalpha:     np.ndarray = np.array([])  # (nbonds,) Δα⁰ = α∥ − α⊥       (Å³)
        self._bpm_alpha_perp: np.ndarray = np.array([])  # (nbonds,) α_⊥⁰                 (Å³)
        self._bpm_d_par:      np.ndarray = np.array([])  # (nbonds,) ∂α∥/∂r               (Å²)
        self._bpm_d_perp:     np.ndarray = np.array([])  # (nbonds,) ∂α⊥/∂r               (Å²)

        # SP-DoS normal-mode projection arrays (set in _setup when spdos_raman=1)
        self._spdos_vv: np.ndarray           = np.array([])  # (nframes, nmol, n_raman)
        self._spdos_eigvecs: np.ndarray      = np.array([])  # (n_raman, 3*natoms_mol)
        self._spdos_sqrt_masses: np.ndarray  = np.array([])  # (3*natoms_mol,) sqrt(m) per coord
        self._spdos_ref_pos: np.ndarray      = np.array([])  # (natoms_mol, 3) COM-centred ref
        self._spdos_ref_masses: np.ndarray   = np.array([])  # (natoms_mol,) g/mol
        self._spdos_n: int = 0                               # number of Raman-active modes
        self._spdos_natoms_mol: int = 0                      # atoms per molecule in .nmd
        # Auto-NMD accumulators (used when spdos_auto_nmd=True)
        self._spdos_auto_cov:     np.ndarray    = np.array([])  # (3N, 3N)
        self._spdos_auto_mean:    np.ndarray    = np.array([])  # (3N,)
        self._spdos_auto_n:       int           = 0
        self._spdos_vel_all:      np.ndarray    = np.array([])  # (nframes, n_tracked, 3N)
        self._spdos_auto_mol_ids: np.ndarray    = np.array([], dtype=np.intp)  # global mol ids
        self._spdos_auto_mol_map: dict[int, int] = {}  # global mol id → compact index
        self._frame_T: list[float]  = []
        self._frame_P: list[float]  = []
        self._frame_V: list[float]  = []
        self._frame_E: list[float]  = []
        # Tier 2 NPT diagnostics (populated in _resolve_state_use)
        self._npt_v_disp: float     = 0.0    # std(V)/⟨V⟩
        self._npt_v_mean: float     = 0.0    # ⟨V⟩ in Å³
        self._ensemble_resolved: str = "nvt"
        # Phase M2: per-frame 3×3 box matrix (Å) for the strain-fluctuation
        # elastic-constant method.  Populated unconditionally — the storage
        # cost is 72 bytes/frame, negligible.
        self._frame_box: list[np.ndarray] = []
        self._pI_acc: np.ndarray    = np.array([])  # (nmol, 3) principal moments
        self._atom_eng: np.ndarray  = np.array([])  # (natom, nsteps)

        # Precomputed molecule batching (set in _setup)
        # Each entry: (mol_ids_matrix, masses_matrix, mol_order)
        #   mol_ids_matrix : (nmol_in_group, na) int   — atom indices per molecule
        #   masses_matrix  : (nmol_in_group, na) float — masses per atom per molecule
        #   mol_order      : (nmol_in_group,) int      — molecule .id for each row
        self._mol_batches: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []

        # Group energy stats accumulated as lists
        self._grp_eng: list[list[float]] = [[] for _ in system.groups]

        # Per-frame stress tensor (6 components) — populated when LAMMPS dump
        # includes c_stress[1..6] columns; used for Green-Kubo viscosity.
        # Each entry is a (6,) float64 array [bar·Å³], summed over atoms.
        self._frame_stress: list[np.ndarray] = []
        # Per-group Yeh-Hummer FSC scale on D_PBC.  1.0 means "no correction"
        # (default).  Populated by _calc_xpt when FSC is
        # active so that downstream (_compute_transport,
        # _integrate_thermo) can rescale D-derived quantities consistently.
        self._fsc_scale_trans: np.ndarray | None = None
        # Rotational FSC scale.  Decays as (a/L)³ — typically O(1e-4) for
        # small molecules in 5-nm boxes, but populated for completeness so
        # f_rot, D_rot, and rotational thermo are internally consistent.
        self._fsc_scale_rot: np.ndarray | None = None
        self._frame_current: list[np.ndarray] = []   # J(t) = Σ qᵢ vᵢ [e·Å/ps]
        # σ_el molecular-current accumulator: J_mol(t) = Σ_m Q_m · v_m_COM,
        # which excludes neutral solvent dipole flux by construction.
        # Populated only when transport_electrical_molecular_current=True.
        self._frame_current_mol: list[np.ndarray] = []
        # Per-frame M_neutral(t) = Σ_i q_neutral_i · r_i(t) where
        # q_neutral_i = q_i − Q_m·m_i/M_m subtracts the bulk-translation
        # contribution of every charged molecule.  For pure solvent (every
        # mol neutral) M_neutral = Σ q·r (bounded); for an electrolyte it
        # excludes the ion-translation Brownian walk that would otherwise
        # blow up the velocity-cumsum estimate of ε(0).  Computed from
        # positions directly — no cumulative integration of J needed.
        self._frame_M_neutral_eA: list[np.ndarray] = []
        # Per-atom-type velocity sums for the κ convective-baseline correction
        # under disable_xpt (where the full _vacvv per-atom velocity history
        # is not stored).  Σᵢ⟨eᵢ⟩·vᵢ is approximated as Σ_type ⟨e_type⟩·V_type
        # which is exact when atoms of the same type have similar ⟨e_i⟩ (true
        # in equilibrium for chemically homogeneous systems).  Storage:
        # n_types × n_frames × 3 × 8 ≈ 90 KB for a typical 4-type / 1k-frame
        # run.  Indices grouping atoms by type_label set in _setup.
        self._frame_v_per_type: list[np.ndarray] = []
        self._kappa_type_indices: list[np.ndarray] = []
        # Pre-computed effective per-atom charges
        #   q_eff_i      = Q_m · m_i / M_m  (gives J_mol  when dotted with v)
        #   q_neutral_i  = q_i − q_eff_i    (gives M_water when dotted with r)
        # Both set in _setup when the system has static charges.
        self._q_eff_static: np.ndarray | None = None
        self._q_neutral_static: np.ndarray | None = None
        # Heat-current components, accumulated per frame for two-pass per-atom
        # time-mean subtraction in `_compute_transport`.  See Hardy-Allen
        # fluctuation form in transport.heat_current_components_per_frame_SI.
        self._frame_eV: list[np.ndarray] = []        # Σ eᵢ vᵢ      [J·m/s]
        self._frame_Sv: list[np.ndarray] = []        # Σ Sᵢ vᵢ      [J·m/s]
        self._atom_e_sum_J: np.ndarray | None = None # running Σ_t eᵢ(t)  [J]
        self._heat_flux_units: str = ""              # cached for post-pass

        # ── Phase MEM-R1 — atom-batch state ───────────────────────────────────
        # ``_atom_mask`` selects a subset of global atom indices for the current
        # pass.  When ``None`` (default), the engine processes every atom
        # exactly as before.  When set, ``_vacvv`` is allocated to length
        # ``len(atom_mask)`` and per-frame velocity gathering writes only the
        # masked atoms.  Per-molecule arrays (``_angvv`` etc.) and per-frame
        # scalar accumulators (``_frame_T``, ``_frame_box``, ...) remain full-
        # size — they are needed by ``_compute_postvac`` and are filled on the
        # *first* batch only (subsequent batches set
        # ``_skip_full_arrays_this_pass=True`` to skip the redundant work).
        # ``_skip_per_mol_in_compute_vac`` controls whether the ANGUL branch
        # of ``_compute_vac`` (which uses the full _angvv) runs — it should
        # contribute exactly once across all batches, not n_batches times.
        self._atom_mask:                    np.ndarray | None = None
        self._atom_local_idx:               np.ndarray | None = None  # (natom,) global→local map, -1 outside mask
        self._skip_full_arrays_this_pass:   bool = False
        self._skip_per_mol_in_compute_vac:  bool = False

        # Phase 2 restructure: when ``compute()`` invokes
        # ``_compute_postvac``, it owns the transport / mechanics
        # dispatch, so the postvac call sets this flag to avoid double-
        # running those passes.  Multi-res callers (which jump straight
        # into ``_compute_postvac`` without going through ``compute()``)
        # leave this False so the post-passes still fire as before.
        self._postvac_skip_transport_mechanics: bool = False

    # ── Phase MEM-R1 — batch mask helpers ─────────────────────────────────────

    def set_atom_mask(self, atom_ids: np.ndarray | None,
                      *, skip_full_arrays: bool = False,
                      skip_per_mol_in_compute_vac: bool = False) -> None:
        """Restrict per-atom storage and processing to the given subset.

        Parameters
        ----------
        atom_ids
            Global atom indices belonging to this batch, or ``None`` to clear
            the mask.  When ``None`` the engine reverts to full-system mode.
        skip_full_arrays
            When ``True``, :meth:`accumulate` skips the per-frame scalar
            accumulators (``_frame_T``/``_frame_V``/``_frame_stress`` ...) and
            the per-molecule arrays (``_angvv``, ``_omega_lab``, ``_dipvv``,
            ``_inddip``).  Set this on every pass *except the first* during
            multi-batch execution so those full-size arrays are filled exactly
            once.
        skip_per_mol_in_compute_vac
            When ``True``, :meth:`_compute_vac` skips the ANGUL branch (which
            uses ``_angvv`` — already filled on the first pass).  Set this on
            every pass except the first; otherwise the ANGUL contribution
            would be summed n_batches times.
        """
        if atom_ids is None:
            self._atom_mask = None
            self._atom_local_idx = None
            self._skip_full_arrays_this_pass = False
            self._skip_per_mol_in_compute_vac = False
            return

        atom_ids = np.asarray(atom_ids, dtype=np.intp)
        natom = self.sys.natom
        local = np.full(natom, -1, dtype=np.intp)
        local[atom_ids] = np.arange(len(atom_ids), dtype=np.intp)
        self._atom_mask = atom_ids
        self._atom_local_idx = local
        self._skip_full_arrays_this_pass = bool(skip_full_arrays)
        self._skip_per_mol_in_compute_vac = bool(skip_per_mol_in_compute_vac)

    def free_batch_arrays(self) -> None:
        """Release the masked ``_vacvv`` allocation between passes.

        Called between successive atom batches to drop the multi-GB
        velocity time series before the next pass re-allocates it (with a
        possibly different atom count).  Per-molecule and per-frame
        accumulators are preserved.
        """
        self._vacvv = np.zeros((0,), dtype=np.float32)

    # ── Setup ─────────────────────────────────────────────────────────────────

    def _setup(self, n_total_frames: int) -> None:
        """Allocate velocity storage arrays."""
        cfg = self.cfg
        self._nsteps = n_total_frames
        corlen = cfg.corlen
        self._vacmaxf = int(corlen * (n_total_frames - 1)) + 1
        # FFT-friendly tot_N: round up to a 5-smooth number (only 2,3,5
        # factors).  pocketfft handles such lengths in O(N log N); arbitrary
        # composite lengths fall back to Bluestein's O(N log N) but with a
        # 2–3× constant factor.  E.g. for 10001-frame runs the natural
        # tot_N = 15002 = 2·13·577 hits Bluestein and FFT is ~3× slower
        # than the next-fast 16384 = 2^14.  Up-rounding to scipy's
        # next_fast_len keeps the input zero-padded in the extra bins, so
        # all physics is preserved (S(0) = sum, frequencies refined slightly).
        n_fft_raw = n_total_frames + self._vacmaxf
        try:
            from scipy.fft import next_fast_len
            self._tot_N = int(next_fast_len(n_fft_raw, real=False))
        except ImportError:                          # fallback: next power of 2
            self._tot_N = 1 << (n_fft_raw - 1).bit_length()
        self._nused   = self._tot_N // 2
        if self._tot_N != n_fft_raw:
            log.info(
                "FFT length: tot_N %d → %d (5-smooth round-up; saves "
                "Bluestein factor on prime-rich factorizations)",
                n_fft_raw, self._tot_N,
            )

        s = self.sys
        # Phase MEM-R1: ``_atom_mask`` (set via :meth:`set_atom_mask`) shrinks
        # ``_vacvv``'s atom dimension to the current batch.  When unset, full
        # natom is allocated as before.
        n_active_atoms = (len(self._atom_mask) if self._atom_mask is not None
                          else s.natom)
        # Phase 2 fix: even under disable_xpt = True, the per-atom velocity
        # array is required by the transport pipeline whenever D_α, D_rot,
        # or Λ_{αβ} is requested (those are computed from the full velocity
        # time series via Wiener-Khinchin).  Skip the allocation only when
        # *no* downstream consumer needs it.
        _transport_needs_vac = bool(
            getattr(self.cfg, "transport_diffusion", False)
            or getattr(self.cfg, "transport_rotation", False)
        )
        if self._disable_xpt and not _transport_needs_vac:
            log.info("disable_xpt=True: skipping per-atom velocity "
                     "time-series allocation (saves ~%.1f GB)",
                     self._vactype * n_total_frames * s.natom * 3 * 4 / 1e9)
            # Zero-sized placeholders so size() checks downstream are False.
            self._vacvv = np.zeros((0,), dtype=np.float32)
        else:
            if self._disable_xpt:
                log.info(
                    "disable_xpt=True but transport requires per-atom "
                    "velocities (D / D_rot enabled) — keeping the "
                    "velocity array allocated."
                )
            log.info(
                "Allocating velocity array: shape=(%d, %d, %d, %d) dtype=float32",
                self._vactype, n_total_frames, n_active_atoms, self._nd,
            )
            self._vacvv = np.zeros(
                (self._vactype, n_total_frames, n_active_atoms, self._nd), dtype=np.float32
            )
        if s.has_frame_charges:
            # Running sum for trajectory-averaged per-atom charges.
            self._charge_sum = np.zeros(s.natom, dtype=np.float64)

        # Pre-compute q_eff per atom for the σ_el molecular-current path
        # AND the dielectric J_neutral path.  q_eff_i = Q_m · m_i / M_m makes
        #   J_mol     = Σ q_eff · v     (= Σ_m Q_m v_m_COM, ions only)
        #   J_neutral = Σ (q − q_eff)·v (= solvent dipole flux, no ion drift)
        # so subtracting one from the other isolates the pieces needed for
        # the two different transport quantities.  Built whenever the
        # system has static charges and a non-trivial molecular topology
        # (so ε(0) gets the J_neutral fix even when σ_el is left in
        # atomic-current mode).
        if (s.has_charges and not s.has_frame_charges and s.nmol > 0
                and (cfg.transport_electrical_conductivity
                     or cfg.transport_dielectric)):
            qa = np.array([a.charge for a in s.atoms], dtype=np.float64)
            ma = np.array([a.mass   for a in s.atoms], dtype=np.float64)
            mol_id_per_atom = np.full(s.natom, -1, dtype=int)
            for m in s.mols:
                for ai in m.atom_ids:
                    mol_id_per_atom[ai] = m.id
            mol_M = np.zeros(s.nmol, dtype=np.float64)
            mol_Q = np.zeros(s.nmol, dtype=np.float64)
            for m in s.mols:
                if not m.atom_ids:
                    continue
                idx = np.asarray(m.atom_ids, dtype=int)
                mol_M[m.id] = float(ma[idx].sum())
                mol_Q[m.id] = float(qa[idx].sum())
            safe_M = np.where(mol_M > 0, mol_M, 1.0)
            valid = mol_id_per_atom >= 0
            q_eff = np.zeros(s.natom, dtype=np.float64)
            q_eff[valid] = (mol_Q[mol_id_per_atom[valid]]
                            * ma[valid]
                            / safe_M[mol_id_per_atom[valid]])
            self._q_eff_static = q_eff
            self._q_neutral_static = qa - q_eff
            n_charged_mols = int((np.abs(mol_Q) > 1e-6).sum())
            log.info("σ_el / ε(0): molecular topology resolved; %d/%d "
                     "molecules carry net charge.  J_mol = Σ q_eff·v "
                     "(ions only); M_neutral = Σ q_neutral·r "
                     "(solvent dipole, bounded).",
                     n_charged_mols, s.nmol)
        elif (cfg.transport_electrical_conductivity
                and cfg.transport_electrical_molecular_current
                and s.has_frame_charges):
            log.warning("σ_el: per-frame charges in dump; "
                        "molecular-current convention not yet supported with "
                        "fluctuating charges.  Falling back to atomic current "
                        "Σ qᵢ vᵢ.  Set [transport] molecular_current = 0 to "
                        "silence this warning.")

        # Per-atom-type indices for the κ convective-baseline correction.
        # Grouped under disable_xpt where _vacvv is not allocated; building
        # this is cheap so we do it whenever transport_thermal_conductivity
        # is enabled.
        if cfg.transport_thermal_conductivity and self._disable_xpt:
            type_groups: dict[str, list[int]] = {}
            for i, a in enumerate(s.atoms):
                type_groups.setdefault(a.type_label, []).append(i)
            self._kappa_type_indices = [np.asarray(idx, dtype=int)
                                         for idx in type_groups.values()]
            log.info("κ: %d atom types resolved for type-averaged ⟨eᵢ⟩·vᵢ "
                     "convective-baseline correction (disable_xpt path)",
                     len(self._kappa_type_indices))

        # Phase MEM-R1: per-molecule arrays are filled on the *first* batch
        # only.  Skip allocation on subsequent batches so the data populated
        # during pass 0 is preserved for the final ``_compute_postvac`` /
        # ``_compute_dacf`` call.
        if (self._molecular and not self._disable_xpt
                and not self._skip_full_arrays_this_pass):
            self._pI_acc = np.zeros((s.nmol, 3))
            if self._rot_type == ANGUL:
                self._angvv = np.zeros((n_total_frames, s.nmol, 3), dtype=np.float32)
                self._omega_lab = np.zeros((n_total_frames, s.nmol, 3), dtype=np.float32)
            if s.has_charges or s.has_frame_charges:
                self._dipvv = np.zeros((n_total_frames, s.nmol, 3), dtype=np.float32)
            if s.has_induced_dipoles and cfg.use_ind:
                self._inddip = np.zeros((n_total_frames, s.nmol, 3), dtype=np.float32)
                log.info("Induced dipoles will be included in IR DACF (use_ind=1).")
            elif s.has_induced_dipoles and not cfg.use_ind:
                log.info("Induced dipoles present in trajectory but use_ind=0; "
                         "computing charge-flux-only IR.")

            # Precompute batched molecule index/mass arrays grouped by molecule size.
            # All molecules of the same atom count are stacked into one matrix so
            # decompose_velocities_batch() can process them with a single numpy call.
            masses_all = np.array([a.mass for a in s.atoms])
            size_groups: dict[int, list] = {}
            for mol in s.mols:
                ids = mol.atom_ids
                size_groups.setdefault(len(ids), []).append((mol.id, ids))
            self._mol_batches = []
            for _na, entries in sorted(size_groups.items()):
                mol_order = np.array([e[0] for e in entries], dtype=np.intp)
                mol_ids   = np.array([e[1] for e in entries], dtype=np.intp)  # (ng, na)
                mol_masses = masses_all[mol_ids]                               # (ng, na)
                self._mol_batches.append((mol_ids, mol_masses, mol_order))

            # ── BPM bond arrays ───────────────────────────────────────────────
            # Precompute flat arrays over all parameterised bonds so that the
            # per-frame polarizability computation is fully vectorised.
            bp = cfg.bond_polarizability
            if bp:
                mol_ids_list = []
                ai_list, aj_list = [], []
                da_list, aperp_list, dpar_list, dperp_list = [], [], [], []
                for mol in s.mols:
                    for (ai, aj) in mol.bonds:
                        ti = s.atoms[ai].type_label.upper()
                        tj = s.atoms[aj].type_label.upper()
                        key = tuple(sorted([ti, tj]))
                        if key in bp:
                            a_par, a_perp, d_par, d_perp = bp[key]
                            mol_ids_list.append(mol.id)
                            ai_list.append(ai)
                            aj_list.append(aj)
                            da_list.append(a_par - a_perp)
                            aperp_list.append(a_perp)
                            dpar_list.append(d_par)
                            dperp_list.append(d_perp)
                if mol_ids_list:
                    self._bpm_mol_ids    = np.array(mol_ids_list, dtype=np.intp)
                    self._bpm_ai         = np.array(ai_list,       dtype=np.intp)
                    self._bpm_aj         = np.array(aj_list,       dtype=np.intp)
                    self._bpm_dalpha     = np.array(da_list,       dtype=np.float64)
                    self._bpm_alpha_perp = np.array(aperp_list,    dtype=np.float64)
                    self._bpm_d_par      = np.array(dpar_list,     dtype=np.float64)
                    self._bpm_d_perp     = np.array(dperp_list,    dtype=np.float64)
                    self._polvv = np.zeros((n_total_frames, s.nmol, 6), dtype=np.float32)
                    has_deriv = any(d != 0.0 for d in dpar_list + dperp_list)
                    log.info("BPM: %d bonds parameterised for Raman PACF "
                             "(%d bond type(s)%s).", len(mol_ids_list), len(bp),
                             "; isotropic stretch term active" if has_deriv else "")
                else:
                    log.warning("bond_polarizability specified but no bonds matched "
                                "atom type labels. Check topology type labels vs. "
                                "[bond_polarizability] keys.")

            # ── SP-DoS normal-mode projection ─────────────────────────────────
            if cfg.spdos_raman:
                if cfg.spdos_auto_nmd:
                    # Auto-NMD: determine which molecule size to track
                    target_na = cfg.spdos_auto_nmd_natoms
                    if target_na == 0:
                        target_na = max(size_groups.keys())
                        log.info("auto_nmd: auto-detected molecule size = %d atoms.",
                                 target_na)
                    if target_na not in size_groups:
                        log.warning(
                            "auto_nmd: no molecules with %d atoms found; "
                            "SP-DoS disabled.", target_na)
                    else:
                        entries      = size_groups[target_na]
                        mol_order_arr = np.array([e[0] for e in entries], dtype=np.intp)
                        mol_ids_arr   = np.array([e[1] for e in entries], dtype=np.intp)
                        mol_masses_0  = masses_all[mol_ids_arr[0]]  # (target_na,)
                        n3            = 3 * target_na
                        n_tracked     = len(mol_order_arr)
                        self._spdos_auto_mol_ids = mol_order_arr
                        self._spdos_auto_mol_map = {
                            int(gid): i for i, gid in enumerate(mol_order_arr)}
                        self._spdos_auto_cov  = np.zeros((n3, n3), dtype=np.float64)
                        self._spdos_auto_mean = np.zeros(n3,       dtype=np.float64)
                        self._spdos_natoms_mol   = target_na
                        self._spdos_ref_masses   = mol_masses_0.copy()
                        self._spdos_sqrt_masses  = np.repeat(
                            np.sqrt(mol_masses_0), 3)
                        # _spdos_ref_pos will be set from first frame
                        mem_gb = n_total_frames * n_tracked * n3 * 4 / 1e9
                        if mem_gb > 2.0:
                            log.warning(
                                "auto_nmd: velocity buffer requires %.1f GB; "
                                "consider using a smaller system or an explicit "
                                ".nmd file.", mem_gb)
                        self._spdos_vel_all = np.zeros(
                            (n_total_frames, n_tracked, n3), dtype=np.float32)
                        log.info(
                            "auto_nmd: tracking %d molecules (%d atoms each, "
                            "point_group=%s).",
                            n_tracked, target_na, cfg.spdos_point_group)
                elif not cfg.spdos_nmd_file:
                    log.warning("spdos_raman=1 but neither spdos_auto_nmd nor "
                                "spdos_nmd_file specified; SP-DoS disabled.")
                else:
                    raise RuntimeError("SP-DoS normal-mode Raman is not available in this 2PT/3PT-only build of py-xPT.")
                    nmd = NormalModeData.from_file(cfg.spdos_nmd_file)
                    if cfg.spdos_raman_species:
                        sp_set = set(cfg.spdos_raman_species.split())
                        mask = np.array([lbl in sp_set for lbl in nmd.labels])
                    else:
                        mask = nmd.raman_active.astype(bool)
                    n_r = int(mask.sum())
                    if n_r == 0:
                        log.warning("SP-DoS: no Raman-active modes in %s; "
                                    "SP-DoS disabled.", cfg.spdos_nmd_file)
                    else:
                        self._spdos_eigvecs     = nmd.eigvecs[mask].astype(np.float64)
                        self._spdos_n           = n_r
                        self._spdos_natoms_mol  = nmd.natoms
                        self._spdos_sqrt_masses = np.repeat(
                            np.sqrt(nmd.masses), 3)   # (3*natoms,)
                        m_frac = nmd.masses / nmd.masses.sum()
                        self._spdos_ref_pos    = (nmd.ref_pos
                                                   - (nmd.ref_pos * m_frac[:, None]).sum(0))
                        self._spdos_ref_masses = nmd.masses.copy()
                        self._spdos_vv = np.zeros(
                            (n_total_frames, s.nmol, n_r), dtype=np.float32)
                        log.info(
                            "SP-DoS: %d Raman-active modes loaded from %s "
                            "(molecule=%s, point_group=%s).",
                            n_r, cfg.spdos_nmd_file, nmd.molecule, nmd.point_group)

    # ── Accumulation ──────────────────────────────────────────────────────────

    def accumulate(self, frames: Sequence[FrameData]) -> None:
        """
        Accumulate all trajectory frames into velocity storage.

        *frames* may be a pre-collected list **or** a lazy generator.
        When a generator is passed, :meth:`_setup` must have already been
        called (e.g. via ``engine._setup(system.total_frames(cfg))``);
        otherwise the iterator is materialised into a list first so that
        the total frame count is known for array allocation.
        """
        if self._nsteps == 0:
            # Frame count unknown — materialise once to determine it
            frames = list(frames)
            if not frames:
                raise RuntimeError("No trajectory frames to process.")
            self._setup(len(frames))

        s   = self.sys
        cfg = self.cfg

        # Static topology charges — used as fallback when dump has no "q" column.
        # For fluctuating charge models, fd.charges overrides this per frame.
        _static_charges = np.array([a.charge for a in s.atoms]) if s.has_charges else None

        _ts0: float = 0.0   # timestep of first frame  (for Δt detection)
        _ts1: float = 0.0   # timestep of second frame

        # Phase MEM-R1: helpers for atom-batch mode.  When _atom_mask is set,
        # _vacvv has shape (vactype, ns, len(mask), 3), so per-frame velocity
        # writes must subset to the mask before storing.  When _skip_full_arrays
        # _this_pass is True (subsequent batches), per-frame scalar accumulators
        # and per-molecule arrays are NOT touched — they were populated during
        # the first batch and the data is identical.
        _mask = self._atom_mask
        _skip_full = self._skip_full_arrays_this_pass

        def _store_vacvv(ti: int, fi_: int, src: np.ndarray) -> None:
            """Write ``src`` (shape (natom, 3)) into ``_vacvv[ti, fi_, :, :]``,
            slicing by the atom mask when one is set."""
            if _mask is None:
                self._vacvv[ti, fi_] = src.astype(np.float32)
            else:
                self._vacvv[ti, fi_] = src[_mask].astype(np.float32)

        # Phase MR-DRIFT: subtract per-frame system-COM velocity from every
        # atom's velocity before any downstream use.  In an equilibrium MD
        # trajectory ⟨v_sys_COM⟩ should be zero, but NPT runs without
        # ``fix momentum`` accumulate net drift that contributes a constant
        # offset M_total · ⟨|v_sys|²⟩ to the mass-weighted VACF, integrating
        # linearly with τ and dominating D for any window > a few ps.
        # The subtraction is the velocity-space analog of LAMMPS's
        # ``compute msd com=yes``.  Disabled by user setting
        # subtract_com_velocity = False (e.g. for shear / driven MD where
        # the bulk drift IS the physics).
        _do_drift_subtract = bool(getattr(cfg, "subtract_com_velocity", True))
        if _do_drift_subtract:
            _atom_masses_full = np.array([a.mass for a in s.atoms],
                                          dtype=np.float64)
            _M_total = float(_atom_masses_full.sum())
            _drift_speed_sq_sum = 0.0   # for ⟨|v_sys|²⟩ diagnostic
            _drift_n_frames = 0


        self._rdf_acc = None
        self._orient_acc = None
        self._coll_acc = None

        for fi, fd in enumerate(frames):
            vel = fd.velocities   # (natom, 3) Å/ps



            # Per-frame system-COM velocity removal (see comment above).
            if _do_drift_subtract and _M_total > 0.0:
                _v_sys_COM = ((vel * _atom_masses_full[:, None]).sum(axis=0)
                               / _M_total)
                vel = vel - _v_sys_COM[None, :]
                if not _skip_full:
                    _drift_speed_sq_sum += float((_v_sys_COM * _v_sys_COM).sum())
                    _drift_n_frames += 1

            # Per-frame charges override static topology charges when present
            charges_this = fd.charges if fd.charges is not None else _static_charges

            # Accumulate per-frame charges for trajectory-averaged ion detection
            # (skipped on Phase MEM-R1 subsequent batches — already done on pass 0)
            if (fd.charges is not None and self._charge_sum.size > 0
                    and not _skip_full):
                self._charge_sum += fd.charges.astype(np.float64)

            if self._disable_xpt:
                # Phase MEM-B: skip per-atom velocity time series + molecular
                # decomposition + per-mol dipole/angular-velocity arrays.
                # Per-frame transport-side accumulators (_frame_T, _frame_V,
                # _frame_stress, _frame_current, _frame_eV, _frame_box) still
                # populate below.
                pass
            elif not self._molecular:
                # Atom-total velocity only
                _store_vacvv(TRANS, fi, vel)
            else:
                # Decompose all molecules at once, grouped by molecule size
                v_t = np.empty_like(vel)
                v_r = np.empty_like(vel)
                v_v = np.empty_like(vel)

                for mol_ids, mol_masses, mol_order in self._mol_batches:
                    # mol_ids: (ng, na) int indices; mol_masses: (ng, na)
                    pos_b = fd.positions[mol_ids]   # (ng, na, 3)
                    vel_b = vel[mol_ids]             # (ng, na, 3)

                    vt_b, vr_b, vv_b, anguv_b, pI_b, omega_b = (
                        decompose_velocities_batch(pos_b, vel_b, mol_masses)
                    )

                    # Scatter back to full atom arrays
                    v_t[mol_ids] = vt_b
                    v_r[mol_ids] = vr_b
                    v_v[mol_ids] = vv_b

                    # Cumulative / per-molecule stores: skipped on Phase MEM-R1
                    # subsequent batches because the underlying _angvv/_dipvv/
                    # _pI_acc arrays already contain the right values from
                    # batch 0.
                    if not _skip_full:
                        self._pI_acc[mol_order] += pI_b

                        if self._rot_type == ANGUL:
                            self._angvv[fi, mol_order, :] = anguv_b.astype(np.float32)
                            self._omega_lab[fi, mol_order, :] = omega_b.astype(np.float32)

                    if (not _skip_full
                            and charges_this is not None and self._dipvv.size > 0):
                        q_b = charges_this[mol_ids]          # (ng, na)
                        # Full molecular dipole velocity: μ̇_m = Σᵢ qᵢ vᵢ using
                        # the complete atomic velocity (trans + rot + vib).
                        # For neutral molecules the translational term
                        # Σᵢ qᵢ v_trans,i = Q_mol v_cm = 0 cancels exactly, so
                        # the result equals (rot + vib) without any extra cost.
                        # For ions (Q_mol ≠ 0) the translational contribution is
                        # correctly included.  For fluctuating charges the
                        # charge-reorganisation-during-translation term is also
                        # captured, giving the intermolecular phonon IR band.
                        self._dipvv[fi, mol_order, :] = (
                            (q_b[:, :, None] * vel_b).sum(axis=1)
                        ).astype(np.float32)

                    if (not _skip_full
                            and fd.induced_dipoles is not None and self._inddip.size > 0):
                        # Per-molecule induced dipole: P_ind,m = Σᵢ∈m p_ind,i  (e·Å)
                        # The time derivative dP_ind,m/dt is computed in _compute_dacf
                        # via ω-multiplication in the frequency domain, avoiding
                        # numerical differentiation of noisy per-frame data.
                        ind_b = fd.induced_dipoles[mol_ids]  # (ng, na, 3)
                        self._inddip[fi, mol_order, :] = (
                            ind_b.sum(axis=1)
                        ).astype(np.float32)

                    # ── SP-DoS: project v_vib onto Raman-active normal modes ──
                    # Rotate each molecule's vibrational velocity (lab frame) to
                    # the reference molecular frame via Kabsch rotation, then
                    # project onto the mass-weighted normal mode eigenvectors.
                    # Only processes batches whose atom count matches the .nmd file.
                    if (not _skip_full
                            and self._spdos_vv.size > 0
                            and mol_ids.shape[1] == self._spdos_natoms_mol):
                        _na   = self._spdos_natoms_mol
                        _tot  = float(self._spdos_ref_masses.sum())
                        _sm3  = self._spdos_sqrt_masses          # (3*na,)
                        _ref  = self._spdos_ref_pos              # (na, 3) COM-centred
                        _mref = self._spdos_ref_masses           # (na,)
                        for _mi in range(len(mol_order)):
                            cur  = pos_b[_mi]                    # (na, 3)
                            com  = (cur * _mref[:, None]).sum(0) / _tot
                            R    = _kabsch_rotation(_ref, cur - com, _mref)
                            vref = vv_b[_mi] @ R                 # (na, 3) in ref frame
                            vmw  = (vref * _sm3.reshape(_na, 3)).ravel()  # (3*na,)
                            self._spdos_vv[fi, mol_order[_mi]] = (
                                self._spdos_eigvecs @ vmw
                            ).astype(np.float32)

                    # ── Auto-NMD: accumulate displacement covariance ───────────
                    # Runs when spdos_auto_nmd=True (eigvecs not yet known).
                    # Stores mass-weighted vibrational velocities for projection
                    # after _finalize_auto_nmd() builds the eigenvectors.
                    if (not _skip_full
                            and self._spdos_vel_all.size > 0
                            and mol_ids.shape[1] == self._spdos_natoms_mol):
                        _na   = self._spdos_natoms_mol
                        _tot  = float(self._spdos_ref_masses.sum())
                        _sm3  = self._spdos_sqrt_masses
                        _mref = self._spdos_ref_masses
                        if not self._spdos_ref_pos.size:
                            # Bootstrap reference from first molecule of first frame
                            cur0 = pos_b[0]
                            com0 = (cur0 * _mref[:, None]).sum(0) / _tot
                            self._spdos_ref_pos = (cur0 - com0).copy()
                        _ref = self._spdos_ref_pos
                        for _mi in range(len(mol_order)):
                            gid  = int(mol_order[_mi])
                            cidx = self._spdos_auto_mol_map.get(gid, -1)
                            if cidx < 0:
                                continue
                            cur  = pos_b[_mi]
                            com  = (cur * _mref[:, None]).sum(0) / _tot
                            R    = _kabsch_rotation(_ref, cur - com, _mref)
                            # Displacement from reference in aligned frame
                            u_ref = (cur - com) @ R - _ref    # (na, 3)
                            q_mw  = (u_ref * _sm3.reshape(_na, 3)).ravel()
                            # Vibrational velocity in aligned frame
                            vref = vv_b[_mi] @ R
                            vmw  = (vref * _sm3.reshape(_na, 3)).ravel()
                            # Accumulate covariance statistics
                            self._spdos_auto_mean += q_mw
                            self._spdos_auto_cov  += np.outer(q_mw, q_mw)
                            self._spdos_auto_n    += 1
                            # Store velocity for later projection
                            self._spdos_vel_all[fi, cidx] = vmw.astype(np.float32)

                _store_vacvv(TRANS, fi, v_t)
                _store_vacvv(ROTAT, fi, v_r)
                _store_vacvv(IMVIB, fi, v_v)
                _store_vacvv(TOTAL, fi, vel)

                # ── BPM: compute anisotropic bond polarizability α̃_m(t) ──────
                # Full molecular polarizability tensor (Silberstein / Bader-Berne BPM):
                #   α_m = Σ_b [ α_⊥(r) I + Δα(r) r̂ r̂^T ]
                # where α_⊥(r) = α_⊥⁰ + ∂α_⊥/∂r · r  and  Δα(r) = Δα⁰ + (∂α∥/∂r − ∂α_⊥/∂r)·r
                # Stored as 6-component symmetric tensor [xx,yy,zz,xy,xz,yz].
                # When derivatives are 0 (2-param format), α_⊥ is constant and the
                # isotropic Σ_b α_⊥ I term contributes only at ω=0 (removed by ω²).
                if (not _skip_full
                        and self._polvv.size > 0 and self._bpm_mol_ids.size > 0):
                    r_vecs = (fd.positions[self._bpm_aj]
                              - fd.positions[self._bpm_ai])    # (nbonds, 3)
                    r_lens = np.linalg.norm(r_vecs, axis=1, keepdims=True)
                    np.maximum(r_lens, 1e-10, out=r_lens)
                    rh = r_vecs / r_lens                        # (nbonds, 3) unit vectors
                    r_lens_1d    = r_lens[:, 0]               # (nbonds,)
                    # r-dependent α_⊥(r) and Δα(r) — derivatives are 0 when
                    # only 2-parameter format is given (backward-compatible)
                    alpha_perp_r = (self._bpm_alpha_perp
                                    + self._bpm_d_perp * r_lens_1d)
                    delta_alpha  = (self._bpm_dalpha
                                    + (self._bpm_d_par - self._bpm_d_perp)
                                    * r_lens_1d)
                    pol_frame = np.zeros((s.nmol, 6), dtype=np.float64)
                    for c, (xi, xj) in enumerate(
                            [(0,0),(1,1),(2,2),(0,1),(0,2),(1,2)]):
                        if xi == xj:
                            # diagonal: isotropic α_⊥(r) + anisotropic Δα(r) r̂ᵢ²
                            wts = alpha_perp_r + delta_alpha * rh[:, xi] ** 2
                        else:
                            # off-diagonal: only anisotropic Δα(r) r̂ᵢ r̂ⱼ
                            wts = delta_alpha * rh[:, xi] * rh[:, xj]
                        pol_frame[:, c] = np.bincount(
                            self._bpm_mol_ids, weights=wts, minlength=s.nmol)
                    self._polvv[fi] = pol_frame.astype(np.float32)

            # ── Phase MEM-R1: skip per-frame scalars on subsequent batches ──
            # All the lists / accumulators below are append-only or running
            # sums; they would double-count if filled on more than one pass.
            # The data populated during batch 0 is identical regardless of
            # which atom subset is in _vacvv, so we simply skip here.
            if _skip_full:
                continue

            # Track first two timesteps for Δt detection
            if fi == 0:
                _ts0 = fd.timestep
            elif fi == 1:
                _ts1 = fd.timestep

            # Scalar thermodynamics
            self._frame_T.append(fd.temperature)
            self._frame_P.append(fd.pressure)
            self._frame_V.append(fd.volume)
            self._frame_E.append(fd.total_energy)
            # Per-frame box matrix (Å, row-vector convention) — used by the
            # strain-fluctuation elastic-constant method (Phase M2).
            if hasattr(fd, "box") and fd.box is not None:
                self._frame_box.append(np.asarray(fd.box, dtype=np.float64).copy())

            # Per-atom energies (if available in trajectory)
            if fd.atom_energies is not None:
                for gi, grp in enumerate(s.groups):
                    atom_ids = np.array(grp.atom_ids)
                    grp_atom_eng = fd.atom_energies[atom_ids]
                    grp_eng_sum = np.sum(grp_atom_eng)   # total group energy (kcal/mol)
                    self._grp_eng[gi].append(float(grp_eng_sum))

            # Per-atom stresses (if available in trajectory — LAMMPS c_stress[1..6])
            # fd.stresses is (natom, 6) [bar·Å³]; sum over atoms → system stress
            if (hasattr(fd, 'stresses') and fd.stresses is not None
                    and fd.stresses.size > 0):
                self._frame_stress.append(fd.stresses.sum(axis=0).copy())

            # Per-frame ionic-current vectors for σ_el (Phase T2) and the
            # atomic-current dielectric flux (used by ε(0) cumulative sum).
            #   _frame_current     = Σᵢ qᵢ vᵢ        (atomic; for ε(0) cumsum
            #                                          and as a σ_el fallback)
            #   _frame_current_mol = Σ_m Q_m v_m_COM  (molecular; correct σ_DC
            #                                          when the system has a
            #                                          neutral solvent that
            #                                          would otherwise pollute
            #                                          the GK integral)
            # Both are 3 floats per frame so storage is negligible.
            if charges_this is not None and np.any(charges_this != 0.0):
                self._frame_current.append(
                    (charges_this[:, None] * vel).sum(axis=0).astype(np.float64)
                )
                if self._q_eff_static is not None:
                    self._frame_current_mol.append(
                        (self._q_eff_static[:, None] * vel).sum(axis=0).astype(np.float64)
                    )
                if self._q_neutral_static is not None:
                    self._frame_M_neutral_eA.append(
                        (self._q_neutral_static[:, None] * fd.positions).sum(axis=0).astype(np.float64)
                    )

            # Per-frame Hardy-Allen heat-flux components for GK thermal
            # conductivity (Phase T3).  Store Σ eᵢvᵢ and Σ Sᵢvᵢ separately so
            # `_compute_transport` can subtract the per-atom time-mean energy
            # ⟨eᵢ⟩_t before forming J_q (the fluctuation form, required when
            # per-atom energies have large atom-to-atom variation).
            if (fd.atom_energies is not None and fd.atom_energies.size > 0
                    and hasattr(fd, 'stresses') and fd.stresses is not None
                    and fd.stresses.size > 0
                    and fd.stresses.shape[1] == 6):
                raise RuntimeError("thermal-conductivity transport is not available in this 2PT/3PT-only build of py-xPT.")
                eV_t, Sv_t, e_atom_J = heat_current_components_per_frame_SI(
                    fd.atom_energies, vel, fd.stresses,
                    lammps_units=self.cfg.lammps_units,
                )
                self._frame_eV.append(eV_t)
                self._frame_Sv.append(Sv_t)
                if self._atom_e_sum_J is None:
                    self._atom_e_sum_J = e_atom_J.copy()
                else:
                    self._atom_e_sum_J += e_atom_J
                self._heat_flux_units = self.cfg.lammps_units or ""
                # Per-type velocity sums for κ convective-baseline correction
                # under disable_xpt (cheap; only when type-grouping has been
                # set up at _setup time).
                if self._kappa_type_indices:
                    n_types = len(self._kappa_type_indices)
                    v_per_type = np.empty((n_types, 3), dtype=np.float64)
                    for ti, idx in enumerate(self._kappa_type_indices):
                        v_per_type[ti] = vel[idx].sum(axis=0)
                    self._frame_v_per_type.append(v_per_type)

        # Phase MR-DRIFT diagnostic: report the trajectory-averaged drift
        # speed when the subtraction is active.  In a properly-equilibrated
        # NPT trajectory this should be ≪ 1 Å/ps (much smaller than the
        # ~few Å/ps thermal velocity); values comparable to thermal speeds
        # indicate a missing ``fix momentum`` in the LAMMPS production.
        # Without the subtraction the integrated drift offset on long
        # trajectories can inflate D by 1–3 orders of magnitude.
        if _do_drift_subtract and not _skip_full and _drift_n_frames > 0:
            rms_drift = float(np.sqrt(_drift_speed_sq_sum / _drift_n_frames))
            log.info("System COM drift subtracted; ⟨|v_sys|²⟩^½ = %.3e Å/ps "
                     "over %d frames (set [thermodynamics] "
                     "subtract_com_velocity = 0 to disable).",
                     rms_drift, _drift_n_frames)

        # Determine dump time step from frame timestamps (or config fallback).
        # Phase MEM-R1: when _skip_full_arrays_this_pass is set the per-frame
        # loop short-circuited before writing _ts0/_ts1 — preserve the value
        # established on pass 0 instead of falling through to cfg.timestep
        # (which would silently divide _vacdtime by the dump_freq → S(0)/D
        # off by an integer factor on multi-pass runs).
        if not self._skip_full_arrays_this_pass:
            nf = len(self._frame_T)
            if nf >= 2 and _ts1 > _ts0:
                self._vacdtime = _ts1 - _ts0
            elif cfg.dump_freq > 0:
                self._vacdtime = cfg.dump_freq * cfg.timestep
            else:
                self._vacdtime = cfg.timestep

        # Compute per-group energy statistics from accumulated per-atom energies (if present)
        self._compute_group_energies()

        # ── Auto-NMD finalization ─────────────────────────────────────────────
        if self._spdos_vel_all.size > 0:
            T_frames = [t for t in self._frame_T if t > 0]
            T_auto   = float(np.mean(T_frames)) if T_frames else 300.0
            if cfg._temperature_provided:
                T_auto = cfg.temperature
            self._finalize_auto_nmd(T_auto)

        log.info("Accumulated %d frames  Δt = %.4g ps", self._nsteps, self._vacdtime)


    # ── Computation ───────────────────────────────────────────────────────────

    def compute(self) -> xPTResult:
        """Run analysis pipelines based on the per-module ``*_enabled``
        flags resolved during config validation.

        Phase 2 restructure: each module gates independently.  The
        engine computes the VAC tensor only when at least one consuming
        module is on (thermo / IR-Raman / transport_diffusion /
        transport_rotation).  Bit-exact
        compatible with the legacy ``disable_xpt = 1`` flag (now a
        synonym of ``thermodynamics_enabled = 0``).
        """
        s     = self.sys
        ngrp  = s.ngrp
        vt    = self._vactype
        nused = self._nused
        cfg   = self.cfg

        result = xPTResult(ngrp, vt, nused, self._vacmaxf, self._nd)

        # ── C3: silent-fallback collection ──────────────────────────────────
        # Attach a WARNING-level handler to the pyxpt logger root for the
        # duration of compute().  The captured records get summarised in
        # the .out.log header by _write_log so users have one place to spot
        # fallbacks instead of grepping the run log.
        _warning_collector = _WarningCollector()
        _warn_logger = logging.getLogger("pyxpt")
        _warn_logger.addHandler(_warning_collector)

        thermo_on   = (getattr(cfg, "thermodynamics_enabled", True)
                       and not self._disable_xpt)
        transport_on = (bool(getattr(cfg, "transport_enabled", False))
                        and self._transport_any_enabled())
        mechanics_on = (bool(getattr(cfg, "mechanics_enabled", False))
                        and self._mechanics_any_enabled())

        # VAC tensor is needed for thermo and the per-atom transport
        # diagnostics (D, D_rot).  Other transport coeffs
        # (η_shear from stress, σ_el from currents, κ from heat flux) do
        # not need it; mechanics doesn't need it at all.
        need_vac = thermo_on or (
            transport_on and (
                cfg.transport_diffusion or cfg.transport_rotation
            )
        )

        # Resolve T / P / V / E once (cheap; used by every downstream pass).
        T_use, P_use, V_use, _ = self._resolve_state_use()

        if need_vac:
            log.info("Computing VAC via FFT ...")
            vac_sum = self._compute_vac()
            # compute() owns the transport / mechanics dispatch below; ask
            # _compute_postvac to skip them so they don't run twice.
            self._postvac_skip_transport_mechanics = True
            try:
                self._compute_postvac(result, vac_sum,
                                        run_thermo=thermo_on)
            finally:
                self._postvac_skip_transport_mechanics = False
        else:
            log.info("Skipping VAC / 2PT / IR-Raman: no consuming module"
                     " enabled.  Running only [transport] and [mechanics]"
                     " post-passes.")
            log.info("Avg T=%.2f K  V=%.2f Å³", T_use, V_use)

        if transport_on:
            log.info("Computing transport coefficients ...")
            self._compute_transport(result, T_use, V_use)

        if mechanics_on:
            log.info("Computing mechanical / dynamic-mechanical properties ...")
            self._compute_mechanics(result)

        log.info("Computation complete.")
        # C3: detach warning collector + stash records for _write_log.
        _warn_logger.removeHandler(_warning_collector)
        self._captured_warnings = list(_warning_collector.records)
        return result

    def _resolve_state_use(self) -> tuple[float, float, float, float]:
        """Resolve (T_use, P_use, V_use, E_use) from cfg overrides or the
        trajectory averages.  Single source of truth used by both the
        VAC-needing path and the transport/mechanics-only path.

        P_use is normalized to **GPa**, the internal unit expected by
        downstream consumers (`_z_sim` for use_sim_z, reduced-P* reporter).
        LAMMPS `units real` emits atm; `units metal` emits bar; `units si`
        emits Pa.  For `units lj` no conversion is performed — pressure
        flows through as-is in reduced units, which is the convention used
        by callers when ``cfg.lammps_units == 'lj'``.

        Tier 2 NPT support (cfg.ensemble):
          - "nvt"  (default): respect cfg overrides; backward-compatible.
          - "npt": FORCE per-frame V/P averages from trajectory regardless
                   of cfg.volume / cfg.pressure overrides.  Also logs the
                   volume fluctuation statistic std(V)/<V>, and warns when
                   it exceeds 5% (large fluctuations may indicate non-
                   equilibrium or invalidate scalar-V analysis assumptions).
          - "auto": treat as "npt" when std(V)/<V> > 1%, else "nvt".
        """
        cfg = self.cfg
        trjT, trjP, trjV, trjE = self._avg_thermo()

        # Tier 2: compute V fluctuation statistic for ensemble dispatch / diagnostic
        v_frames = [v for v in self._frame_V if v > 0]
        if v_frames:
            v_arr  = np.array(v_frames, dtype=np.float64)
            v_mean = float(v_arr.mean())
            v_std  = float(v_arr.std(ddof=1)) if len(v_arr) > 1 else 0.0
            v_disp = v_std / v_mean if v_mean > 0 else 0.0
        else:
            v_mean = v_std = v_disp = 0.0

        ensemble = (cfg.ensemble or "nvt").strip().lower()
        if ensemble == "auto":
            # Auto-detect: > 1% std(V)/<V> ⇒ treat as NPT
            ensemble_resolved = "npt" if v_disp > 0.01 else "nvt"
            log.info("Ensemble auto-detect: std(V)/⟨V⟩ = %.4g → %s",
                     v_disp, ensemble_resolved.upper())
        else:
            ensemble_resolved = ensemble

        # Stash the V dispersion for downstream consumers (writers, diagnostics)
        self._npt_v_disp = v_disp
        self._npt_v_mean = v_mean
        self._ensemble_resolved = ensemble_resolved

        # State point selection
        if ensemble_resolved == "npt":
            # Use trajectory-derived V (the meaningful NPT quantity).
            # For P: only use trjP when frames actually carried pressure data
            # (i.e. non-zero with non-trivial dispersion).  NCDF dumps from
            # LAMMPS typically don't record per-frame P, so trjP = 0 and we
            # must fall back to cfg.pressure (set from .eng averages).
            T_use = cfg.temperature if cfg.temperature_override else trjT
            p_frames = [p for p in self._frame_P if p != 0.0]
            traj_has_P = len(p_frames) > 0
            if traj_has_P:
                P_raw = trjP
            else:
                P_raw = cfg.pressure if cfg.pressure_override else trjP
            V_use = trjV - cfg.void_volume
            if cfg.volume_override:
                log.info("Ensemble=NPT: ignoring cfg.volume override; using "
                         "trajectory ⟨V⟩ = %.3f Å³ (override was %.3f)",
                         trjV, cfg.volume)
            if not traj_has_P and cfg.pressure_override:
                log.info("Ensemble=NPT: trajectory has no per-frame P data; "
                         "falling back to cfg.pressure = %g (raw LAMMPS units)",
                         cfg.pressure)
            if v_disp > 0.05:
                log.warning("Ensemble=NPT: std(V)/⟨V⟩ = %.4g exceeds 5%% — "
                            "large volume fluctuations may invalidate "
                            "scalar-V (Sackur-Tetrode, HS packing) analysis. "
                            "Consider longer equilibration or smaller "
                            "barostat damping.", v_disp)
        else:  # NVT (default)
            T_use = cfg.temperature if cfg.temperature_override else trjT
            P_raw = cfg.pressure    if cfg.pressure_override    else trjP
            V_use = (cfg.volume - cfg.void_volume
                     if cfg.volume_override
                     else trjV - cfg.void_volume)

        units = (cfg.lammps_units or "").strip().lower()
        if units == "lj":
            P_use = P_raw
        else:
            from pyxpt.thermo.utility import lammps_press_to_pa
            P_use = P_raw * lammps_press_to_pa(units) * 1e-9   # Pa → GPa

        E_use = cfg.energy_avg  if cfg.energy_override else trjE
        return T_use, P_use, V_use, E_use

    # ── Post-VAC pipeline ─────────────────────────────────────────────────────
    def _compute_postvac(self, result: xPTResult,
                          vac_sum: np.ndarray,
                          *,
                          run_thermo: bool = True,
                          ) -> xPTResult:
        """Run all pipeline steps after the VAC tensor has been built.

        The thermo (xPT) block is gated by ``run_thermo`` (default True).

        The pre-amble (averaged T/P/V/E, DOF accounting, vacT, dos,
        result.vac, result.frequencies, result.dos) always runs because
        every downstream consumer needs at least the dos and frequency
        axis.
        """
        s     = self.sys
        ngrp  = s.ngrp
        vt    = self._vactype
        nused = self._nused

        # ── 2. Averaged thermodynamic quantities ──────────────────────────────
        trjT, trjP, trjV, trjE = self._avg_thermo()

        # ── 3. Apply T, P, V, E overrides from config ────────────────────────
        T_use, P_use, V_use, E_use = self._resolve_state_use()

        # Log the final values to use (trajectory or config override)
        log.info("Avg T=%.2f K  P=%.4f GPa  V=%.2f Å³  E=%.3f kcal/mol",
                 T_use, P_use, V_use, E_use)

        # ── 4. DOF accounting ─────────────────────────────────────────────────
        vacDF = self._compute_dof(vac_sum)       # (ngrp, vactype) — auto-detect active DoF
        trdf  = self._translation_rot_dof()
        total_df = float(np.sum(vacDF[:, vt - 1]))
        for i in range(ngrp):
            for j in range(vt):
                vacDF[i, j] -= trdf * vacDF[i, j] / total_df if total_df > 0 else 0.0

        # ── 5. Temperature from VAC at lag 0 ─────────────────────────────────
        vacT = self._vac_temperature(vac_sum, vacDF)  # (ngrp, vactype)

        # Temperature consistency check (non-group runs only, matches legacy behaviour).
        # Compare VAC-derived system temperature against the input T; warn if they
        # differ by more than 0.1% — likely a wrong vel_scale or constraint count.
        if ngrp == 1 and T_use > 0:
            vac_T_sys = vacT[ngrp - 1, vt - 1]
            if vac_T_sys > 0 and abs(vac_T_sys - T_use) / T_use > 0.001:
                log.warning(
                    "Input temperature %.2f K and VAC-derived temperature %.2f K "
                    "differ by %.2f%%. Check constraints or velocity scaling factor.",
                    T_use, vac_T_sys, (vac_T_sys - T_use) / T_use * 100.0,
                )

        # Stationarity diagnostic: block-KE consistency + VACF tail decay
        self._check_stationarity(vac_sum)

        # ── 6. Power spectrum ─────────────────────────────────────────────────
        self._pwrfreq = 1.0e10 / (self._vacdtime * self._tot_N * VLIGHT)
        freqs = np.arange(nused) * self._pwrfreq
        result.frequencies[:] = freqs

        dos = self._power_spectrum(vac_sum, vacT)     # (ngrp, vactype, tot_N)
        result.dos[:] = dos[:, :, :nused]

        # ── 7. Store VAC ──────────────────────────────────────────────────────
        result.vac[:] = vac_sum[:, :, :self._vacmaxf, :, :]

        # ── 7b. MC1 cross-channel diagnostic (off-diagonal trans/rot/vib) ─────
        if getattr(self.cfg, "cross_vac_diagnostic", False):
            if self._molecular and self._vacvv.size > 0:
                self._compute_cross_vac(result, vacT, nused)
            else:
                log.warning("cross_vac_diagnostic = 1 ignored: requires a "
                            "molecular run with per-atom velocities (got "
                            "molecular=%s, vacvv.size=%d)",
                            self._molecular, self._vacvv.size)

        # ── 8. 2PT partitioning ───────────────────────────────────────────────
        # The fluidicity solve + dos_gas/dos_sol partition is needed by both
        # the thermodynamic integration AND the IR/Raman split (which uses
        # dos_sol to compute the solid-mode contribution to the dipole DACF).
        # Run it whenever either pipeline is on; skip outright otherwise.
        fxpt = fmf_arr = y_arr = hsdf_arr = Bg_arr = None
        dos_gas = dos_sol = None
        if run_thermo:
            fxpt, fmf_arr, y_arr, hsdf_arr, Bg_arr = self._calc_xpt(
                dos, vacT, vacDF, V_use, T_use)

            # In a (near-)crystal the low-ν DoS rises as ν² with no diffusive
            # plateau; the 2PT split nevertheless books the small residual
            # mobility as an HS gas (ice Ih T230: f=0.017 → ~0.9 J/mol/K
            # spurious).  When the plateau ratio P = ⟨S⟩_[0,10] / ⟨S⟩_[20,40] cm⁻¹
            # falls below gas_gate_tol (ice 0.001 vs deepest liquid 0.020; default
            # tol 0.01) the channel is crystal-like.  gas_gate=debye zeroes its
            # fluidicity (removes the gas component); gas_gate=debye_warn records
            # the would-be removal WITHOUT touching the headline (bit-exact with
            # "none").  Either way the per-channel diagnostics (plateau ratio,
            # pre-gate fluidicity, removed gas entropy) land in result.gas_gate_*.
            # NOTE the two indexings: dos is keyed by VelType (k), the fluidicity
            # array by row (frow: 0=trans, 1=rot) — they coincide only for
            # ANGUL rotation, so map them explicitly.
            fxpt_pre = fxpt.copy()
            _gas_gate_zeroed = False
            if self.cfg.gas_gate in ("debye", "debye_warn"):
                _warn_only = self.cfg.gas_gate == "debye_warn"
                _nu_g = freqs[:nused]
                _lo_m = (_nu_g >= 0.0) & (_nu_g <= 10.0)
                _mi_m = (_nu_g >= 20.0) & (_nu_g <= 40.0)
                _gate_pairs = [(TRANS, 0)]                  # (VelType, fluidicity row)
                if self._molecular:
                    _gate_pairs.append((self._rot_type, 1))
                if _lo_m.sum() < 3 or _mi_m.sum() < 3:
                    log.warning("gas_gate=%s: frequency grid too coarse to "
                                "resolve the 0-10/20-40 cm^-1 bands; gate skipped",
                                self.cfg.gas_gate)
                else:
                    for gi in range(ngrp):
                        for k, frow in _gate_pairs:
                            _S = dos[gi, k, :nused]
                            _mid = float(np.mean(_S[_mi_m]))
                            if _mid <= 0.0:
                                continue
                            _P = max(float(np.mean(_S[_lo_m])), 0.0) / _mid
                            result.gas_gate_plateau_ratio[frow, gi] = _P
                            result.gas_gate_fluidicity_pre[frow, gi] = fxpt[frow, gi]
                            if _P < self.cfg.gas_gate_tol and fxpt[frow, gi] > 0.0:
                                result.gas_gate_fired[frow, gi] = True
                                log.warning(
                                    "gas_gate=%s grp%d %s: plateau ratio P=%.4g < "
                                    "tol %.3g — no diffusive plateau (Debye nu^2 / "
                                    "crystal-like); %s gas component (f=%.4g)",
                                    self.cfg.gas_gate, gi + 1,
                                    "trans" if frow == 0 else "rot", _P,
                                    self.cfg.gas_gate_tol,
                                    "removing" if not _warn_only
                                    else "WARN-ONLY: would remove",
                                    fxpt[frow, gi])
                                if not _warn_only:
                                    fxpt[frow, gi] = 0.0
                                    _gas_gate_zeroed = True

            result.fluidicity[:] = fxpt
            result.packing_fraction[:] = y_arr

            dos_gas, dos_sol = self._partition_dos(dos, fxpt, fmf_arr,
                                                     Bg_arr, y_arr)
            result.dos_gas[:]   = dos_gas[:, :, :nused]
            result.dos_solid[:] = dos_sol[:, :, :nused]
            # Pre-gate gas DoS for the per-channel gas-entropy diagnostic
            # (result.gas_entropy_component).  Identical to dos_gas unless a
            # `debye` gate actually zeroed a fluidicity, so the common path pays
            # nothing.  y_arr is unchanged by the gate, so both partitions share it.
            if _gas_gate_zeroed:
                dos_gas_pre, _ = self._partition_dos(dos, fxpt_pre, fmf_arr,
                                                     Bg_arr, y_arr)
            else:
                dos_gas_pre = dos_gas

        # ── 9. Thermodynamic integration ──────────────────────────────────────
        if run_thermo:
            log.info("Integrating thermodynamic weighting functions ...")
            self._integrate_thermo(
                result, dos_gas, dos_sol, dos, vacT, vacDF,
                V_use, P_use, T_use, trjE, fxpt, hsdf_arr, y_arr, ngrp, vt,
                Bg_arr=Bg_arr, dos_gas_pre=dos_gas_pre,
            )

            # gas_gate: book the spurious gas entropy per fired channel from the
            # per-channel gas-entropy diagnostic (computed in the integrator from
            # the pre-gate gas DoS).  This is the magnitude removed (debye) or that
            # WOULD be removed (debye_warn); zero on channels the gate left alone.
            # Reported PER molecule/atom (÷ group count) so it matches the cage
            # diagnostics and the campaign's per-mol numbers (ice ~0.83); the
            # gas_entropy_component field itself stays extensive (mirrors the split
            # S_quantum_gas).
            if self.cfg.gas_gate in ("debye", "debye_warn"):
                for gi in range(ngrp):
                    grp = self.sys.groups[gi]
                    _cnt = max(grp.nmol if self._molecular else grp.natom, 1)
                    for k, frow in ((TRANS, 0), (self._rot_type, 1)):
                        if frow == 1 and not self._molecular:
                            continue
                        if result.gas_gate_fired[frow, gi]:
                            result.gas_gate_dS_removed[frow, gi] = \
                                float(result.gas_entropy_component[gi, k]) / _cnt

            # ── 9b2. R2PT refinement (Sun 2017) ─────────────────────────────────
            # Override the translational gas+solid entropy with the revised-2PT
            # value: δ-resolved fluidicity (Eq A8) + F_a-inclusive sum-rule gas /
            # full-F_s solid.  Applied as a delta off the rigorous-HS entropy
            # (reusing the cage post-correction machinery).  2PT-only (3PT uses cage).
            if self.cfg.do_r2pt:
                from pyxpt.thermo.utility import r2pt_entropy
                _freq_cm = result.frequencies[:nused]
                _TOT = result.S_quantum.shape[1] - 1
                for gi, grp in enumerate(self.sys.groups):
                    cnt = max(grp.nmol, 1)
                    T_loc = vacT[gi, TRANS] if vacT[gi, TRANS] > 0.0 else T_use
                    S_r2 = r2pt_entropy(
                        _freq_cm, dos[gi, TRANS, :nused] / cnt,
                        float(fxpt[TRANS, gi]), T_loc,
                        grp.mass / cnt, V_use / cnt,
                        delta=self.cfg.r2pt_delta, label=f"grp{gi+1}")
                    if S_r2 is None:
                        log.warning("R2PT group %d skipped (Eq A8 no root); "
                                    "translational entropy left at rigorous-HS.", gi + 1)
                        continue
                    S_rig = float(result.S_quantum[gi, TRANS]) / (cnt * R)   # k_B/atom
                    dS_ext = (S_r2 - S_rig) * cnt
                    A_ex_kJ = -dS_ext * R * T_loc * 1e-3
                    _chans = (TRANS,) if _TOT == TRANS else (TRANS, _TOT)
                    for _c in _chans:
                        result.S_quantum[gi, _c]   += dS_ext * R
                        result.S_classical[gi, _c] += dS_ext * R
                        result.A_quantum[gi, _c]    += A_ex_kJ
                        result.A_classical[gi, _c]  += A_ex_kJ
                        result.mu_quantum[gi, _c]   += A_ex_kJ
                        result.mu_classical[gi, _c] += A_ex_kJ
                    log.info("R2PT(δ=%.2f) group %d: S*=%.4f k_B/atom "
                             "(rigorous-HS %.4f, ΔS=%+.4f)",
                             self.cfg.r2pt_delta, gi + 1, S_r2, S_rig, S_r2 - S_rig)

            # Parameter-free ΔS = (1/3)∫cage·(1−w)·(W_g−W_s)dν added to the
            # rigorous-HS entropy; recovers the solid-side deficit of rigorous-HS
            # in structured liquids (liquid metals, dense LJ).  Uses the standard
            # 2PT Lorentzian gas (``dos_gas``), the total DoS, and the VACF kernel
            # — mode-independent, so it complements hs_entropy=rigorous directly.
            # In-memory DoS is un-normalised (∫=3N); divide by the per-group
            # particle count to get the per-atom DoS the construction expects.
            if self.cfg.cage_entropy:
                from pyxpt.thermo.utility import cage_memory_entropy
                if self.cfg.hs_entropy != "rigorous":
                    log.warning("cage_entropy is designed for hs_entropy=rigorous; "
                                "adding it on top of '%s' may double-correct.",
                                self.cfg.hs_entropy)
                _freq_cm = result.frequencies[:nused]
                for gi, grp in enumerate(self.sys.groups):
                    cnt = max(grp.nmol, 1)              # monatomic: atoms == molecules
                    T_loc = vacT[gi, TRANS] if vacT[gi, TRANS] > 0.0 else T_use
                    # kernel inversion uses the clean VACF (first vacmaxf lags),
                    # not the zero-padded/wrapped FFT tail (:nused)
                    C_sc = np.einsum('tii->t',
                                     vac_sum[gi, TRANS, :self._vacmaxf]) / self._nd
                    # V_use is the total box volume [Å³]; result.volume is not yet
                    # assigned at this point in the flow.
                    _d = int(self.cfg.dimension)
                    _cage_arr: list = []
                    # env-gated save of the cage inputs for the SI gate/filter
                    # sensitivity sweep (CAGE_SAVE_INPUTS=<prefix>); diagnostic only.
                    import os as _os_cage
                    _csi = _os_cage.environ.get("CAGE_SAVE_INPUTS")
                    if _csi:
                        np.savez(f"{_csi}_grp{gi+1}.npz",
                                 dt=self._vacdtime, C_scalar=C_sc, nu_cm=_freq_cm,
                                 dos_total=dos[gi, TRANS, :nused] / cnt,
                                 dos_gas=dos_gas[gi, TRANS, :nused] / cnt,
                                 T_K=T_loc, mass_amu=grp.mass / cnt,
                                 vol_per_atom=V_use / cnt, dimension=_d)
                    dS = cage_memory_entropy(
                        self._vacdtime, C_sc, _freq_cm,
                        dos[gi, TRANS, :nused] / cnt,        # → per-atom total DoS
                        dos_gas[gi, TRANS, :nused] / cnt,    # → per-atom Lorentzian gas
                        T_loc, grp.mass / cnt, V_use / cnt,  # V_use is the d-dim measure (Å^d)
                        prefactor=1.0 / _d, dimension=_d,    # per-DoF prefactor 1/d
                        ref="markov",
                        nf_run=self.cfg.cage_nf_run, taper=self.cfg.cage_taper,
                        tail_tol=self.cfg.cage_tail_tol,
                        label=f"grp{gi+1}", cage_out=_cage_arr)
                    if not dS:
                        continue
                    # stash the cage DoS for the .pwr writer (un-normalised, ∫=3N
                    # convention like dos/dos_gas/dos_solid → ×cnt back from per-atom)
                    if _cage_arr:
                        result.dos_cage[gi, TRANS, :nused] = _cage_arr[0] * cnt
                    # dS is per-atom [k_B]; result.S_quantum is extensive (the
                    # .thermo writer divides by the particle count), so scale by cnt.
                    dS_ext = dS * cnt
                    A_ex_kJ = -dS_ext * R * T_loc * 1e-3
                    # translational cage correction; for molecular groups also
                    # propagate to the aggregate TOTAL channel (last vt slot) so
                    # the reported total reflects it.  For monatomic, TRANS is the
                    # only/total channel (TOT==TRANS) → add once.
                    _TOT = result.S_quantum.shape[1] - 1
                    _chans = (TRANS,) if _TOT == TRANS else (TRANS, _TOT)
                    for _c in _chans:
                        result.S_quantum[gi, _c]   += dS_ext * R
                        result.S_classical[gi, _c] += dS_ext * R
                        result.S_cage[gi, _c]      += dS_ext * R
                        result.A_quantum[gi, _c]    += A_ex_kJ
                        result.A_classical[gi, _c]  += A_ex_kJ
                        result.mu_quantum[gi, _c]   += A_ex_kJ
                        result.mu_classical[gi, _c] += A_ex_kJ
                    log.info("Cage-memory entropy group %d: ΔS=%.4g J/mol/K/mol "
                             "(prefactor=1/%d, d=%d)", gi + 1, dS * R, _d, _d)

            # ── 9d. Rotational cage-memory entropy post-correction ──────────────
            # Same cage machinery applied to the ROTATIONAL channel of molecular
            # liquids.  The rotational "gas" weight is the free/rigid-rotor weight
            # wsr (NOT the hard-sphere translational gas), supplied via
            # Wg_override so the HS Sackur-Tetrode / packing block is skipped.
            # Prefactor p_rot = 1/d_rot with d_rot = 3 (nonlinear) or 2 (linear).
            _rot_ch = self._rot_type   # ROTAT or ANGUL, whichever is active
            _is_mol_rot = self._molecular and _rot_ch < self._vactype
            if self.cfg.cage_entropy_rot and _is_mol_rot:
                from pyxpt.thermo.utility import cage_memory_entropy
                if self.cfg.hs_entropy != "rigorous":
                    log.warning("cage_entropy_rot is designed for "
                                "hs_entropy=rigorous; adding it on top of '%s' "
                                "may double-correct.", self.cfg.hs_entropy)
                _freq_cm = result.frequencies[:nused]
                _TOT = result.S_quantum.shape[1] - 1
                for gi, grp in enumerate(self.sys.groups):
                    cnt = max(grp.nmol, 1)
                    T_loc = (vacT[gi, _rot_ch]
                             if vacT[gi, _rot_ch] > 0.0 else T_use)
                    # d_rot = 2 (linear) else 3.  Prefer the per-group linear flag;
                    # fall back to cfg.mol_linear when group topology is absent.
                    _lin = bool(grp.linear) or bool(self.cfg.mol_linear)
                    d_rot = 2 if _lin else 3
                    # ── rigid/free-rotor per-DoF gas weight wsr (utility formula) ──
                    rT = np.zeros(3)
                    if self._molecular and self._pI_acc.size > 0:
                        self._get_rot_temp(gi, rT)
                    rs = float(grp.rotsym if grp.rotsym > 0 else 1)
                    wsr = 0.0
                    if rT[0] > 1e-4 and T_loc > 0.0:
                        if _lin or rT[2] < 0:   # linear
                            wsr = (1.0 + math.log(
                                T_loc / math.sqrt(rT[0]*rT[1]) / rs)) / 2.0
                        else:
                            wsr = (1.5 + math.log(math.sqrt(
                                PI * T_loc**3 / (rT[0]*rT[1]*rT[2])) / rs)) / 3.0
                    if wsr == 0.0:
                        continue
                    C_sc = np.einsum('tii->t',
                                     vac_sum[gi, _rot_ch, :self._vacmaxf]) / 3.0
                    _cage_arr: list = []
                    # env-gated save of the ROTATIONAL cage inputs for the SI
                    # gate/filter sensitivity sweep (CAGE_SAVE_INPUTS=<prefix>);
                    # mirrors the translational hook in block 9c.  Stores the
                    # rigid/free-rotor gas weight wsr (Wg_override) so the
                    # post-eval can reconstruct ΔS without the HS Sackur-Tetrode
                    # machinery.  Diagnostic only; off by default.
                    import os as _os_cage_rot
                    _csi_rot = _os_cage_rot.environ.get("CAGE_SAVE_INPUTS")
                    if _csi_rot:
                        np.savez(f"{_csi_rot}_grp{gi+1}_rot.npz",
                                 dt=self._vacdtime, C_scalar=C_sc, nu_cm=_freq_cm,
                                 dos_total=dos[gi, _rot_ch, :nused] / cnt,
                                 dos_gas=dos_gas[gi, _rot_ch, :nused] / cnt,
                                 T_K=T_loc, wsr=wsr, dimension=d_rot)
                    dS = cage_memory_entropy(
                        self._vacdtime, C_sc, _freq_cm,
                        dos[gi, _rot_ch, :nused] / cnt,      # → per-mol rot total DoS
                        dos_gas[gi, _rot_ch, :nused] / cnt,  # → per-mol rot Lorentzian gas
                        T_loc, grp.mass / cnt, V_use / cnt,  # mass/vol unused (Wg_override)
                        prefactor=1.0 / d_rot, dimension=d_rot,
                        ref="markov",
                        nf_run=self.cfg.cage_nf_run, taper=self.cfg.cage_taper,
                        tail_tol=self.cfg.cage_tail_tol,
                        Wg_override=wsr,
                        label=f"grp{gi+1}-rot", cage_out=_cage_arr)
                    if not dS:
                        continue
                    if _cage_arr:
                        result.dos_cage[gi, _rot_ch, :nused] = _cage_arr[0] * cnt
                    dS_ext = dS * cnt
                    A_ex_kJ = -dS_ext * R * T_loc * 1e-3
                    _chans = (_rot_ch,) if _TOT == _rot_ch else (_rot_ch, _TOT)
                    for _c in _chans:
                        result.S_quantum[gi, _c]   += dS_ext * R
                        result.S_classical[gi, _c] += dS_ext * R
                        result.S_cage[gi, _c]      += dS_ext * R
                        result.A_quantum[gi, _c]    += A_ex_kJ
                        result.A_classical[gi, _c]  += A_ex_kJ
                        result.mu_quantum[gi, _c]   += A_ex_kJ
                        result.mu_classical[gi, _c] += A_ex_kJ
                    log.info("Cage-memory entropy (rot) group %d: "
                             "ΔS_cage^rot=%.4g J/mol/K/mol "
                             "(prefactor=1/%d, d_rot=%d, wsr=%.4f)",
                             gi + 1, dS * R, d_rot, d_rot, wsr)


        result.temperature[:] = vacT
        result.pressure = P_use
        # Per-group volume: use GroupVolume from group file if set (user override),
        # otherwise use V_total for all groups.  Using V_total gives the correct
        # Sackur-Tetrode translational entropy for each component in a mixture,
        # which automatically includes the ideal entropy of mixing (−kB·ln x_i per molecule).
        # When a group-specific volume is set, the mixing entropy is absent and
        # the correction Δμ_i = kT·ln(x_i) is reported as a comment in the .thermo file.
        for gi, grp in enumerate(self.sys.groups):
            if grp.volume > 0:
                result.volume[gi] = grp.volume   # user override: mixing entropy not included
            else:
                result.volume[gi] = V_use         # default: V_total → mixing entropy included

        # ── 11. [transport] / [mechanics] — gated by their own enable flags ──
        # When ``_compute_postvac`` is reached from ``compute()``, the
        # caller has already enforced the per-module gating; running these
        # again here would double-up the post-passes.  Multi-resolution
        # callers (which jump directly to ``_compute_postvac`` to reuse the
        # 2PT / IR-Raman pipeline on a stitched VAC) still need them, so
        # only invoke when the respective per-property flag is set AND the
        # section's enabled flag is on.
        cfg = self.cfg
        if (bool(getattr(cfg, "transport_enabled", False))
                and self._transport_any_enabled()
                and not getattr(self, "_postvac_skip_transport_mechanics", False)):
            log.info("Computing transport coefficients ...")
            self._compute_transport(result, T_use, V_use)

        if (bool(getattr(cfg, "mechanics_enabled", False))
                and self._mechanics_any_enabled()
                and not getattr(self, "_postvac_skip_transport_mechanics", False)):
            log.info("Computing mechanical / dynamic-mechanical properties ...")
            self._compute_mechanics(result)

        # ── Reference-pressure correction (Tier 1 NPT support) ────────────────
        # When cfg.reference_pressure is set, compute μ_q_at_ref_P = μ_q +
        # v̄·(P_ref − P_sim) per group.  v̄ is the per-molecule volume of
        # the group (result.volume[gi] / nmol).
        #
        # Units convention: cfg.reference_pressure is in the SAME raw LAMMPS
        # pressure units as cfg.pressure (atm for units=real, bar for metal,
        # reduced for lj).  Internally we convert both to GPa via
        # lammps_press_to_pa, matching the result.pressure convention.
        #
        # Use case: cross-system comparisons in osmotic-pressure work where
        # pure solvent and solution NVT productions may sit at very
        # different effective pressures (e.g. -300 atm vs -50 atm).
        result.mu_quantum_at_ref_p[:]   = result.mu_quantum[:]
        result.mu_classical_at_ref_p[:] = result.mu_classical[:]
        if cfg.reference_pressure is not None:
            units = (cfg.lammps_units or "").strip().lower()
            if units == "lj":
                P_ref_GPa = cfg.reference_pressure
            else:
                from pyxpt.thermo.utility import lammps_press_to_pa
                P_ref_GPa = cfg.reference_pressure * lammps_press_to_pa(units) * 1e-9
            P_sim_GPa = float(result.pressure)        # already GPa via _resolve_state_use
            dP_GPa = P_ref_GPa - P_sim_GPa
            # TOTAL column index (last vactype slot — same as TOTAL constant)
            TOTAL_IDX = result.mu_quantum.shape[1] - 1
            for gi in range(ngrp):
                n_mol = (self.sys.groups[gi].nmol if self._molecular
                         else self.sys.groups[gi].natom)
                if n_mol <= 0:
                    continue
                V_grp_A3 = float(result.volume[gi])     # Å³ total of group volume
                # Δμ per molecule = v̄ · ΔP = (V/N) · ΔP
                # In kJ/mol/molecule: (V/N)[Å³] · ΔP[GPa] · 0.602214076
                # Result is in kJ/mol/SimBox (extensive), so multiply by N_mol:
                #   ΔΜ_box = N · (V/N) · ΔP · 0.6022 = V · ΔP · 0.6022
                # (equivalent to PV correction with extensive V.)
                #
                # The correction is a system-level (total μ_q) shift; it is NOT
                # decomposed across Trans/Rot/Imvib gas/solid channels.  Apply
                # only to the TOTAL column.  Other vactype columns of
                # mu_quantum_at_ref_p remain equal to the uncorrected
                # mu_quantum (via the [:] copy above).
                dmu_kJ_box = V_grp_A3 * dP_GPa * 0.602214076
                result.mu_quantum_at_ref_p[gi, TOTAL_IDX]   += dmu_kJ_box
                result.mu_classical_at_ref_p[gi, TOTAL_IDX] += dmu_kJ_box
            log.info("Reference-pressure correction applied: P_ref=%g (raw) → "
                     "%g GPa, P_sim=%g GPa, ΔP=%g GPa → μ_q_at_ref_P column emitted",
                     cfg.reference_pressure, P_ref_GPa, P_sim_GPa, dP_GPa)

        # ── Apply Output Unit Scaling ──────────────────────────────────────────
        # Called once at the end for ALL modes so GLE modes don't double-convert.
        self._apply_unit_conversions(result)

        log.info("Computation complete.")
        return result

    # ── Transport gating ──────────────────────────────────────────────────────
    def _transport_any_enabled(self) -> bool:
        c = self.cfg
        return any((
            c.transport_diffusion,
            c.transport_rotation,
            c.transport_shear_viscosity,
            c.transport_bulk_viscosity,
            c.transport_electrical_conductivity,
            c.transport_thermal_conductivity,
            c.transport_dielectric,
            # Mechanics consumes traces produced by [transport]; auto-enable
            # the corresponding [transport] flags via _mechanics_dependencies()
            # if the user only asked for [mechanics].
            c.mechanics_dma_shear,
            c.mechanics_dma_bulk,
            c.mechanics_dielectric_spectrum,
        ))

    def _mechanics_any_enabled(self) -> bool:
        c = self.cfg
        return any((
            c.mechanics_dma_shear,
            c.mechanics_dma_bulk,
            c.mechanics_dielectric_spectrum,
            c.mechanics_elastic_constants,
            c.mechanics_maxwell_fit,
        ))

    # ── VAC via FFT ───────────────────────────────────────────────────────────

    def _check_stationarity(self, vac_sum: np.ndarray) -> None:
        """
        Post-hoc stationarity diagnostics for the trajectory.

        Two checks emit WARNING by default; the ``[thermodynamics]
        strict_stationarity = 1`` flag (F1) promotes them to a
        ``RuntimeError`` so production runs fail loudly on bad data:

        1. **Block kinetic-energy consistency**: split the translational
           velocity time series into N_block=4 equal chunks, compute mean
           KE per block, warn/error if (max−min)/mean > 5% — suggests the
           trajectory is not thermalised or has a temperature drift.

        2. **VACF tail decay**: check that the scalar VACF trace has decayed
           to below 10% of C(0) by t_max/2.  A persistent tail indicates
           either insufficient trajectory length for the slowest modes or
           genuine non-stationary dynamics; either way, modes 5–7 results
           may be affected (Sec V.G of ms2_gle_vs_hs_vs_des_monoatomic).
        """
        N_BLOCK     = 4
        KE_DRIFT    = 0.05
        VACF_TAIL   = 0.10
        _strict     = bool(getattr(self.cfg, "strict_stationarity", False))

        def _flag(msg: str, *args) -> None:
            if _strict:
                raise RuntimeError(
                    "strict_stationarity = 1: " + (msg % args) +
                    "  Disable with strict_stationarity = 0 if this is "
                    "expected (e.g., warm-up ramp; non-equilibrium run)."
                )
            log.warning(msg, *args)

        # Block-KE consistency on translational velocities (TRANS = 0)
        if self._vacvv.size > 0 and self._vacvv.shape[0] > TRANS:
            v_trans = self._vacvv[TRANS]   # (ns, natom, 3)
            ns      = v_trans.shape[0]
            blk_sz  = ns // N_BLOCK
            if blk_sz >= 2:
                # mean |v|² per frame, summed over atoms (proxy for 2·KE/m̄)
                ke_frame  = np.einsum('faj,faj->f', v_trans, v_trans)
                blk_means = np.array([
                    ke_frame[i*blk_sz:(i+1)*blk_sz].mean() for i in range(N_BLOCK)
                ])
                m_mean = float(blk_means.mean())
                if m_mean > 0.0:
                    rel_drift = float(np.ptp(blk_means) / m_mean)
                    if rel_drift > KE_DRIFT:
                        _flag(
                            "Stationarity: block-KE drift %.1f%% across %d blocks "
                            "exceeds %.0f%% threshold — trajectory may not be "
                            "thermalised; check the temperature time series.",
                            rel_drift * 100.0, N_BLOCK, KE_DRIFT * 100.0,
                        )

        # VACF tail check: |C_sc(t_max/2)| / |C_sc(0)| < VACF_TAIL
        for gi in range(vac_sum.shape[0]):
            C_sc = np.einsum('tii->t', vac_sum[gi, TRANS]) / 3.0
            if C_sc.shape[0] < 4 or abs(C_sc[0]) < 1e-30:
                continue
            t_half = self._vacmaxf // 2 if self._vacmaxf > 0 else len(C_sc) // 2
            if t_half < 2:
                continue
            ratio = abs(C_sc[t_half]) / abs(C_sc[0])
            if ratio > VACF_TAIL:
                _flag(
                    "Stationarity: VACF group %d has not decorrelated by t=%d "
                    "lags (|C(t)/C(0)|=%.3f > %.2f); trajectory may be too "
                    "short or system non-stationary — modes 5–7 K_int "
                    "and beyond-Debye correction may be biased.",
                    gi, t_half, ratio, VACF_TAIL,
                )

    def _compute_vac(self) -> np.ndarray:
        """
        Compute the 3×3 mass-weighted velocity autocorrelation tensor for every
        group and velocity type via the Wiener–Khinchin theorem.

        Returns
        -------
        vac_sum : (ngrp, vactype, tot_N, 3, 3)
            C_ij(t) = Σ_atoms m_a <v_i(0) v_j(t)>  summed over atoms in each group.
            Diagonal elements give the standard (scalar) VAC; off-diagonal elements
            capture anisotropy and are used by the matrix admittance kernel inversion.

        Thin wrapper over :func:`pyxpt.core.vac.compute_vac_tensor` —
        handles the per-velocity-type branching (TRANS / ROTAT / IMVIB /
        TOTAL use ``_vacvv``; ANGUL uses ``_angvv`` with unit weights),
        the optional atom mask (Phase MEM-R1), and assembly of the
        per-group tensors into a single ``(ngrp, vt, tot_N, 3, 3)``
        array.
        """
        from pyxpt.core.vac import compute_vac_tensor

        s     = self.sys
        N     = self._tot_N
        ns    = self._nsteps
        vt    = self._vactype

        # Phase MEM-R1: when an atom mask is active, _vacvv has shape
        # (vt, ns, len(mask), 3) and group-atom indices must be remapped to
        # local positions within the mask.  Build the per-batch local index
        # tables lazily so single-pass runs incur zero overhead.
        _atom_mask = self._atom_mask
        _local_idx = self._atom_local_idx     # (natom,) global→local, -1 outside mask
        _has_mask  = _atom_mask is not None

        masses_full = np.array([a.mass for a in s.atoms])   # (natom,)
        if _has_mask:
            masses_masked = masses_full[_atom_mask]
        else:
            masses_masked = masses_full
        vac_sum = np.zeros((s.ngrp, vt, N, self._nd, self._nd))

        for tp in range(vt):
            if tp == ANGUL and self._molecular and self._rot_type == ANGUL and self._angvv.size > 0:
                # Phase MEM-R1: ``_angvv`` is full-size (filled on the first
                # batch).  Its contribution to ``vac_sum`` should be summed
                # exactly once across all batches.  Skip on subsequent
                # batches.
                if self._skip_per_mol_in_compute_vac:
                    continue
                src     = self._angvv                         # float32 (ns, n_mols, 3)
                wt      = np.ones(self._angvv.shape[1])
                grp_idx = {gi: np.array(grp.mol_ids)
                           for gi, grp in enumerate(s.groups) if grp.mol_ids}
            else:
                src     = self._vacvv[tp]                     # float32 (ns, n_part, 3)
                wt      = masses_masked
                if _has_mask:
                    # Remap each group's global atom IDs to local positions
                    # in the masked _vacvv.  Atoms outside the current batch
                    # are dropped — they contribute via other batches.
                    grp_idx = {}
                    for gi, grp in enumerate(s.groups):
                        if not grp.atom_ids:
                            continue
                        gids = np.asarray(grp.atom_ids, dtype=np.intp)
                        lids = _local_idx[gids]
                        keep = lids >= 0
                        if keep.any():
                            grp_idx[gi] = lids[keep]
                else:
                    grp_idx = {gi: np.array(grp.atom_ids)
                               for gi, grp in enumerate(s.groups) if grp.atom_ids}

            # Delegate the FFT kernel to core.vac.compute_vac_tensor; route
            # the per-group result back into the (ngrp, vt, N, 3, 3) tensor.
            per_group_vac = compute_vac_tensor(
                src, wt, grp_idx, tot_N=N, ns=ns, backend=self._backend,
            )
            for gi, vac_g in per_group_vac.items():
                vac_sum[gi, tp] = vac_g

        return vac_sum

    # ── MC1: cross-channel correlator diagnostic ─────────────────────────────

    def _compute_cross_vac(self, result: "xPTResult",
                           vacT: np.ndarray, nused: int) -> None:
        """Compute the off-diagonal trans/rot/vib correlators and attach
        them to *result*.

        Three pairs are computed for a molecular run that exposes per-atom
        translational, rotational, and internal-vibration velocity time
        series in ``self._vacvv``:

        ============================================  ==============
        :math:`C_{TR}(t) = \\langle v_T(0)\\,v_R(t)\\rangle`   trans × rot
        :math:`C_{TV}(t) = \\langle v_T(0)\\,v_V(t)\\rangle`   trans × vib
        :math:`C_{RV}(t) = \\langle v_R(0)\\,v_V(t)\\rangle`   rot   × vib
        ============================================  ==============

        Per group we record the full 3×3 time-domain tensor, its rfft
        spectrum, and a scalar diagnostic

        .. math::

           \\rho_{\\alpha\\beta} \\;=\\;
             \\frac{\\max_t |\\operatorname{tr}\\,C_{\\alpha\\beta}(t)|}
                   {\\sqrt{\\operatorname{tr}\\,C_{\\alpha\\alpha}(0)\\;
                           \\operatorname{tr}\\,C_{\\beta\\beta}(0)}}

        Result fields populated: ``has_cross_vac``, ``cross_vac_pairs``,
        ``cross_vac``, ``cross_pwr``, ``cross_coupling``.
        """
        from pyxpt.core.vac import compute_cross_vac_tensor

        s     = self.sys
        N     = self._tot_N
        ns    = self._nsteps
        vacmaxf = self._vacmaxf

        # Per-atom rotational velocity time series always lives in
        # ``_vacvv[ROTAT]`` (line ~1343 of accumulate(); it's stored
        # unconditionally on the molecular path).  ``_rot_type == ANGUL``
        # only changes where the per-MOLECULE angular velocity is stored
        # (``_angvv``), which is the wrong basis for cross-channel
        # correlations against per-atom trans/vib.  Always pin to ROTAT.
        rot_ch = ROTAT
        candidate_pairs = ((TRANS, rot_ch), (TRANS, IMVIB), (rot_ch, IMVIB))

        def _slot_has_data(k: int) -> bool:
            if not (0 <= k < self._vacvv.shape[0] and self._vacvv[k].size > 0):
                return False
            # Reject all-zero slots (e.g., uninstantiated channels)
            return bool(np.any(self._vacvv[k]))

        pairs: list[tuple[int, int]] = []
        for a, b in candidate_pairs:
            if _slot_has_data(a) and _slot_has_data(b):
                pairs.append((a, b))
        if not pairs:
            log.warning("cross_vac_diagnostic: no valid trans/rot/vib pairs "
                        "with populated _vacvv; skipping.")
            return

        # Atom mask / weight bookkeeping mirrors _compute_vac
        _atom_mask = self._atom_mask
        _local_idx = self._atom_local_idx
        _has_mask  = _atom_mask is not None
        masses_full = np.array([a.mass for a in s.atoms])
        masses_masked = masses_full[_atom_mask] if _has_mask else masses_full

        if _has_mask:
            grp_idx = {}
            for gi, grp in enumerate(s.groups):
                if not grp.atom_ids:
                    continue
                gids = np.asarray(grp.atom_ids, dtype=np.intp)
                lids = _local_idx[gids]
                keep = lids >= 0
                if keep.any():
                    grp_idx[gi] = lids[keep]
        else:
            grp_idx = {gi: np.array(grp.atom_ids)
                       for gi, grp in enumerate(s.groups) if grp.atom_ids}

        ngrp = s.ngrp
        npair = len(pairs)
        cross_vac_full = np.zeros((ngrp, npair, vacmaxf, 3, 3))
        cross_pwr_full = np.zeros((ngrp, npair, nused, 3, 3))
        coupling       = np.zeros((ngrp, npair))

        # Need the diagonal C_αα(0) for the dimensionless ρ_αβ.  These are
        # already in self._vac_sum if it was stashed; otherwise pull the
        # diagonal traces directly from result.vac (already filled).
        # result.vac has shape (ngrp, vt, vacmaxf, 3, 3); index [:, :, 0]
        # gives C(t=0).
        trC0 = np.zeros((ngrp, self._vactype))
        for gi in range(ngrp):
            for k in range(self._vactype):
                m = result.vac[gi, k, 0]
                trC0[gi, k] = m[0, 0] + m[1, 1] + m[2, 2]

        log.info("MC1: computing cross-channel correlators for %d pair(s) ...",
                 npair)
        vel_name = {TRANS: "trans", ROTAT: "rotat", ANGUL: "angul",
                    IMVIB: "imvib", TOTAL: "total"}
        for pi, (a, b) in enumerate(pairs):
            src_a = self._vacvv[a]
            src_b = self._vacvv[b]
            per_group = compute_cross_vac_tensor(
                src_a, src_b, masses_masked, grp_idx,
                tot_N=N, ns=ns, backend=self._backend,
            )
            for gi, mat in per_group.items():
                # Time-domain forward lags only
                cross_vac_full[gi, pi] = mat[:vacmaxf]
                # Spectrum of trace(C_αβ(t)) via real FFT of zero-padded ACF
                tr_t = mat[:, 0, 0] + mat[:, 1, 1] + mat[:, 2, 2]   # (N,)
                # Use rfft directly on the trace; first nused bins match the
                # diagonal-channel frequency axis.
                S = np.fft.rfft(tr_t)
                # Per-component spectrum (modulus) — handy for cross-band
                # diagnostics.  Stored as |S_ij(ν)|.
                S_ij = np.fft.rfft(mat, axis=0)
                cross_pwr_full[gi, pi, :, :, :] = np.abs(S_ij[:nused, :, :])
                # ρ_αβ diagnostic
                denom = np.sqrt(max(trC0[gi, a], 0.0) * max(trC0[gi, b], 0.0))
                if denom > 0.0:
                    coupling[gi, pi] = float(np.max(np.abs(tr_t)) / denom)
                else:
                    coupling[gi, pi] = float("nan")
                log.info("  group %d  %s × %s : ρ = %.4f  (max |tr C|=%.3e, "
                         "√(tr Cαα·tr Cββ)=%.3e)",
                         gi + 1, vel_name.get(a, str(a)), vel_name.get(b, str(b)),
                         coupling[gi, pi],
                         float(np.max(np.abs(tr_t))), denom)

        object.__setattr__(result, "has_cross_vac",   True)
        object.__setattr__(result, "cross_vac_pairs", pairs)
        object.__setattr__(result, "cross_vac",       cross_vac_full)
        object.__setattr__(result, "cross_pwr",       cross_pwr_full)
        object.__setattr__(result, "cross_coupling",  coupling)

    def _compute_group_energies(self) -> None:
        """
        Compute per-group energy averages and std dev from accumulated per-atom energies.
        
        This is called after accumulation if per-atom energies were present in the trajectory.
        Updates each group's eng_avg and eng_std fields, which will override config file values.
        """
        has_per_atom_energies = any(grp_eng for grp_eng in self._grp_eng)
        if not has_per_atom_energies:
            return  # No per-atom energies were accumulated
        
        log.info("Computing per-group energy statistics from per-atom energies in trajectory")

        do_norm = self.cfg.normalize or (self._molecular and self.cfg.per_molecule)

        for gi, grp in enumerate(self.sys.groups):
            if not self._grp_eng[gi]:
                continue

            eng_array = np.array(self._grp_eng[gi])
            old_avg = grp.eng_avg
            old_std = grp.eng_std

            # Frame-average the total group energy, then normalise per atom/molecule
            # when normalize=1 so that eng_avg/eng_std carry the same per-particle
            # units that are written to the output file.
            grp.eng_avg = float(np.mean(eng_array))
            grp.eng_std = float(np.std(eng_array))

            if do_norm:
                N = max(grp.nmol if self._molecular else grp.natom, 1)
                grp.eng_avg /= N
                grp.eng_std /= N

            if old_avg != 0 or old_std != 0:
                log.info("Group %d: Trajectory per-atom E overrides config "
                        "(E_avg: %.4f → %.4f kcal/mol, E_std: %.4f → %.4f kcal/mol)",
                        gi, old_avg, grp.eng_avg, old_std, grp.eng_std)
            else:
                log.info("Group %d: Using trajectory per-atom E "
                        "(E_avg=%.4f kcal/mol, E_std=%.4f kcal/mol)",
                        gi, grp.eng_avg, grp.eng_std)

    # ── Thermodynamics helpers ────────────────────────────────────────────────

    def _avg_thermo(self) -> tuple[float, float, float, float]:
        T = float(np.mean(self._frame_T)) if any(self._frame_T) else 0.0
        P = float(np.mean(self._frame_P)) if any(self._frame_P) else 0.0
        V = float(np.mean(self._frame_V)) if any(self._frame_V) else 0.0
        E = float(np.mean(self._frame_E)) if any(self._frame_E) else 0.0
        return T, P, V, E

    def _compute_dof(self, vac_sum: np.ndarray | None = None) -> np.ndarray:
        """Degrees of freedom per group and velocity type.

        The translational DoF count is AUTO-DETECTED from the per-axis
        velocity variance (diag of VAC(0)) — an axis with ~zero variance (e.g.
        v_z≡0 in a genuinely-2D run) drops out, giving the true active DoF.
        This is DECOUPLED from cfg.dimension (the gas/diffusion dimension):
          • bulk:           3 active axes → 3N DoF, gas dim 3
          • genuinely-2D:   v_z≡0 → 2N DoF, gas dim 2
          • slit (3D conf): v_z≠0 (thermal) → 3N DoF, but gas/diffusion is 2D
            (cfg.dimension=2) — the confined axis is a real vibrational DoF.
        """
        s     = self.sys
        ngrp  = s.ngrp
        vt    = self._vactype
        vacDF = np.zeros((ngrp, vt))

        for gi, grp in enumerate(s.groups):
            if not self._molecular:
                n_active = self._nd
                if vac_sum is not None:
                    var = np.array([float(vac_sum[gi, TRANS, 0, a, a])
                                    for a in range(self._nd)])
                    if var.max() > 0:
                        n_active = int(np.sum(var > 1e-6 * var.max()))
                vacDF[gi, 0] = float(n_active) * grp.natom
            else:
                for mid in grp.mol_ids:
                    mol = s.mols[mid]
                    na  = len(mol.atom_ids)
                    lin = int(grp.linear or mol.linear)
                    if na == 1:
                        vacDF[gi, TRANS] += 3.0
                        vacDF[gi, TOTAL] += 3.0
                    elif na == 2:
                        vacDF[gi, TRANS] += 3.0
                        vacDF[gi, ANGUL] += 2.0
                        vacDF[gi, ROTAT] += 2.0
                        vacDF[gi, IMVIB] += 1.0
                        vacDF[gi, TOTAL] += 6.0
                    else:
                        vacDF[gi, TRANS] += 3.0
                        vacDF[gi, ANGUL] += 3.0 - lin
                        vacDF[gi, ROTAT] += 3.0 - lin
                        vacDF[gi, IMVIB] += 3.0*na - 6.0 + lin
                        vacDF[gi, TOTAL] += 3.0 * na
            # Apply constraints
            if not self._molecular:
                vacDF[gi, 0] -= grp.constraint
            else:
                vacDF[gi, IMVIB] -= grp.constraint
                vacDF[gi, TOTAL] -= grp.constraint
        return vacDF

    def _translation_rot_dof(self) -> float:
        s = self.sys
        if s.periodic:
            return 3.0
        if s.natom == 1:
            return 3.0
        if s.natom == 2:
            return 5.0
        return 6.0

    def _vac_temperature(self, vac_sum: np.ndarray,
                          vacDF: np.ndarray) -> np.ndarray:
        """Kinetic temperature from VAC at lag 0: T = VAC(0) / (0.1·R·DOF)."""
        ngrp = self.sys.ngrp
        vt   = self._vactype
        vacT = np.zeros((ngrp, vt))
        for gi in range(ngrp):
            for k in range(vt):
                df = vacDF[gi, k]
                if df > 0:
                    vac0_trace = np.trace(vac_sum[gi, k, 0])
                    vacT[gi, k] = vac0_trace / (0.1 * R * df)
        return vacT

    def _power_spectrum(self, vac_sum: np.ndarray, vacT: np.ndarray) -> np.ndarray:
        """
        Convert VAC -> density of states via FFT (Wiener–Khinchin).
        Result shape: (ngrp, vactype, tot_N).
        """
        ngrp = self.sys.ngrp
        vt   = self._vactype
        N    = self._tot_N
        dos  = np.zeros((ngrp, vt, N))

        for gi in range(ngrp):
            for k in range(vt):
                T = vacT[gi, k]
                if T <= 0:
                    continue

                # vac_sum[gi, k] is (N, d, d); take the trace (sum of diagonals)
                # over all d velocity axes to collapse it to (N,).
                sig_matrix = vac_sum[gi, k]
                sig_trace = np.einsum('tii->t', sig_matrix)

                # Now sig is a 1D array of length N, matching the original logic
                sig = sig_trace * 20.0 / (R * T)

                # Transfer to GPU, perform FFT, transfer back
                sig_gpu = self._backend.to_gpu(sig)
                F_gpu = self._backend.ifft(sig_gpu)
                F = self._backend.to_cpu(F_gpu)

                # FFTW (legacy) doesn't normalize by 1/N, but NumPy's ifft does
                # F.real is now (N,), perfectly matching dos[gi, k]
                dos[gi, k] = F.real * N * self._vacdtime * VLIGHT * 1e-10

        return dos

    # ── Per-axis trans DoS partition (Sec V.B mitigation, Phase 4b-2) ─────────
    def _per_axis_partition_trans(self, vac_sum: np.ndarray, gi: int,
                                    f_axes: np.ndarray, nmol: int,
                                    T: float) -> tuple[np.ndarray, np.ndarray]:
        """
        Build trans gas/solid DoS by summing per-axis 2PT Lorentzian partitions.

        For each Cartesian axis α the diagonal mass-weighted VACF
        ``vac_sum[gi, TRANS, :, α, α]`` is FFT'd to obtain a per-axis DoS
        ``pwr_α(ν)`` (same normalisation as ``_power_spectrum`` but applied to
        a single diagonal element — integrates to N, i.e. one trans DoF per
        atom, instead of 3N for the full trace).  Each axis is then partitioned
        into gas and solid via the standard 2PT Lorentzian using its own
        fluidicity ``f_α``, and the three contributions are summed.

        Compared to the scalar approach (single ``f_GLE`` applied to the full
        trans DoS):
          - Total gas DoF is preserved: ∫ pwr_g_total dν = N · Σ f_α
            (= 3N · ⟨f⟩ when the per-axis mean equals the scalar f_GLE).
          - The shape of pwr_g and pwr_s differs when f_α are unequal — for
            anisotropic systems (confined fluids, layered interfaces) the
            scalar Lorentzian smears across what is genuinely two distinct
            sub-spectra.  Sackur–Tetrode entropy is unaffected (depends only
            on the integral); quantum-solid entropy, beyond-Debye correction,
            and the kernel gas filter all see the new shape and respond.

        Parameters
        ----------
        vac_sum  : (ngrp, vt, N, 3, 3) — full VACF tensor block
        gi       : int — group index
        f_axes   : (3,) — per-axis fluidicities (f_xx, f_yy, f_zz)
        nmol     : int — number of independent diffusing units (atoms in group)
        T        : float — translational temperature for normalisation [K]

        Returns
        -------
        pwr_g_sum, pwr_s_sum : (nused,) per-axis-summed gas and solid DoS
        """
        nused   = self._nused
        nu_step = self._pwrfreq
        N       = self._tot_N
        dt      = self._vacdtime

        pwr_g_sum = np.zeros(nused)
        pwr_s_sum = np.zeros(nused)
        if T <= 0.0:
            return pwr_g_sum, pwr_s_sum

        # The 2PT Lorentzian formula uses width parameter 6·nmol·f, which encodes
        # the 3D dimensionality (3 spatial directions × 2-pt factor).  For
        # per-axis (1D) partitioning, the corresponding factor is 2·nmol·f, so
        # we pass nmol_eff = nmol/3 to xpt_partition.  This guarantees that
        # the isotropic limit (f_x = f_y = f_z) reduces exactly to the scalar
        # partition: 3·L(s0_α, f_α, nmol/3) = L(3·s0_α, f_α, nmol).
        nmol_eff = max(int(round(nmol / 3.0)), 1)
        for axis in range(3):
            f_alpha = float(f_axes[axis])
            if f_alpha <= 0.0:
                continue
            sig_alpha = vac_sum[gi, TRANS, :, axis, axis] * 20.0 / (R * T)
            F = _onfft.ifft(sig_alpha)
            pwr_alpha_full = F.real * N * dt * VLIGHT * 1e-10
            pwr_alpha = pwr_alpha_full[:nused]
            s0_alpha  = float(pwr_alpha[0])
            for j in range(nused):
                gas_j, sol_j = xpt_partition(
                    s0_alpha, float(pwr_alpha[j]),
                    nu_step * j, nmol_eff, f_alpha, f_alpha,
                )
                pwr_g_sum[j] += gas_j
                pwr_s_sum[j] += sol_j

        return pwr_g_sum, pwr_s_sum

    # ── Yeh-Hummer finite-size correction on D_PBC ───────────────────────────
    def _yeh_hummer_eta(self, T_use: float, V_use: float) -> float:
        """Return η [Pa·s] used by the FSC; 0.0 means "skip correction".

        Source priority:
          1. ``cfg.fsc_viscosity`` if positive (user override)
          2. one-shot GK integral on ``self._frame_stress`` if non-empty
          3. 0.0 — caller must skip / warn

        For ``lammps_units = lj`` the SI conversion of stress and volume is
        not well-defined (LJ ε, σ, m are arbitrary), so an inline GK estimate
        cannot be trusted as Pa·s.  In that case only an explicit
        ``fsc_viscosity`` (in real Pa·s) is honored.
        """
        c = self.cfg
        if c.fsc_viscosity > 0.0:
            return float(c.fsc_viscosity)
        if (c.lammps_units or "").strip().lower() == "lj":
            log.info(
                "FSC: lammps_units = lj — cannot infer η in Pa·s without an "
                "explicit ε, σ, m mapping. Set [2pt] fsc_viscosity = <Pa·s> "
                "to enable Yeh-Hummer correction. Skipping.")
            return 0.0
        if not getattr(self, "_frame_stress", None):
            return 0.0
        from pyxpt.thermo.utility import (
            shear_viscosity_from_stress_block, lammps_press_to_pa,
        )
        press_to_Pa = lammps_press_to_pa(c.lammps_units)
        V_m3 = float(V_use) * 1e-30
        stress_arr = np.asarray(self._frame_stress)
        sigma_off = stress_arr[:, 3:6] / float(V_use) * press_to_Pa
        try:
            eta, _ = shear_viscosity_from_stress_block(
                sigma_off, self._vacdtime, V_m3, T_use,
                corlen=c.transport_acf_corlen,
            )
        except Exception:
            return 0.0
        return float(eta) if eta > 0 else 0.0

    def _yeh_hummer_scale(self, gvol: float, gmass: float, nmol: int,
                          T_t: float, s0t: float, eta: float) -> float:
        """Multiplicative scale on K (= scale on D_PBC) so that the corrected
        K gives an infinite-system fluidicity.  Returns 1.0 to disable.

        Identity used:
            scale = D_inf / D_PBC = 1 + (ξ k_B T) / (6π η L D_PBC)
        with L = V^(1/3) (group volume).  D_PBC is recovered from s0t via
            D_cm2s = s0t · R · T / (12 c_light · m_total) × 1e5   (existing eq.)
        """
        if eta <= 0.0 or s0t <= 0.0 or gvol <= 0.0 or gmass <= 0.0 or nmol <= 0:
            return 1.0
        # D_PBC in m²/s.  Same formula as engine.py:1872 but kept in SI.
        D_cm2s = s0t * R * T_t / (12.0 * VLIGHT * gmass) * 1e5
        D_m2s = D_cm2s * 1e-4
        if D_m2s <= 0.0:
            return 1.0
        L_m = (gvol * 1e-30) ** (1.0 / 3.0)   # gvol is Å³
        delta_D = YEH_HUMMER_XI * KB * T_t / (6.0 * PI * eta * L_m)
        scale = 1.0 + delta_D / D_m2s
        # Sanity guard: a Yeh-Hummer scale > 2 means either the box is far too
        # small for 2PT (Lin-Goddard breaks down before FSC even applies), or
        # the inputs are in incompatible units.  Skip the correction in that
        # case; the user gets uncorrected (legacy) results plus a warning.
        if scale > 2.0:
            log.warning(
                "FSC: Yeh-Hummer scale = %.2f for group with L=%.2f Å, "
                "η=%.2e Pa·s, D_PBC=%.2e m²/s — implausibly large. "
                "Box may be too small for 2PT, or units are inconsistent. "
                "Skipping correction.", scale, L_m * 1e10, eta, D_m2s)
            return 1.0
        return scale

    def _yeh_hummer_scale_rot(self, gvol: float, nmol: int,
                              eta: float) -> float:
        """Rotational Yeh-Hummer scale on K_rot.

        Uses the Stokes-Einstein-Debye rotational mobility together with the
        rotational lattice-image sum:
            scale_R = 1 + ξ_R · (a/L)³
        with ``a = (3·V_mol / 4π)^(1/3)`` (volume-equivalent sphere; per-mol
        volume V_mol = gvol/nmol) and ``L = gvol^(1/3)``.  Skips silently
        (returns 1.0) when ``η ≤ 0`` or the geometry is degenerate.

        The result is independent of η — it cancels between Stokes-Einstein-
        Debye D_R and the rotational image-sum — so this function only needs
        η > 0 as a "rotational correction is enabled" gate, not a quantitative
        input.  Larger molecules (a/L → 0.5) get larger corrections; for
        small molecules in pyxpt-typical boxes the magnitude is < 0.1 %.
        """
        if eta <= 0.0 or gvol <= 0.0 or nmol <= 0:
            return 1.0
        V_mol_A3 = gvol / max(nmol, 1)
        if V_mol_A3 <= 0.0:
            return 1.0
        a_m = (3.0 * V_mol_A3 / (4.0 * PI)) ** (1.0/3.0) * 1e-10  # Å → m
        L_m = (gvol * 1e-30) ** (1.0/3.0)
        if L_m <= 0.0 or a_m <= 0.0:
            return 1.0
        ratio = a_m / L_m
        scale = 1.0 + YEH_HUMMER_XI_ROT * ratio**3
        # If a/L > 0.5 the volume-equivalent sphere overlaps multiple periodic
        # images — outside the regime where the rotational image-sum applies.
        if ratio > 0.5:
            log.warning(
                "FSC rotational: a/L = %.3f > 0.5 — molecule too large for "
                "the rigid-rotor image-sum approximation. Skipping.", ratio)
            return 1.0
        return scale

    # ── 2PT partitioning ─────────────────────────────────────────────────────
    def _calc_xpt(self, dos: np.ndarray, vacT: np.ndarray, vacDF: np.ndarray,
                V_use: float, T_use: float
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Fluidicity / packing fraction / HS DoF for modes 2/3/4.

        B1 split (2026-05-16): implementation moved to
        :mod:`pyxpt.thermo._xpt_compute`.
        """
        from pyxpt.thermo._xpt_compute import compute_xpt
        return compute_xpt(self, dos, vacT, vacDF, V_use, T_use)

    def _partition_dos(self, dos: np.ndarray,
                        fxpt: np.ndarray, fmf: np.ndarray,
                        Bg_arr: np.ndarray,
                        y_arr: np.ndarray | None = None,
                        ) -> tuple[np.ndarray, np.ndarray]:
        """Split DoS into gas/solid contributions.

        The partition logic lives in :mod:`pyxpt.thermo._partition` and
        handles the Lin-Goddard Lorentzian and Desjarlais memory-function
        gas-DoS shapes (modes 2PT/3PT).
        """
        from pyxpt.thermo._partition import partition_dos
        return partition_dos(self, dos, fxpt, fmf, Bg_arr, y_arr)

    def _eta_mix(self, y_arr: np.ndarray) -> float | None:
        """
        Return η_mix = Σ_i y_i for use in the one-fluid CS chemical potential,
        or None if there is only one group (pure fluid — single-component CS is exact).
        """
        if self.sys.ngrp < 2:
            return None
        return mixture_packing_fraction(y_arr)

    def _bmcsl_params(self, y_arr: np.ndarray, V_use: float,
                      ) -> tuple[np.ndarray, tuple[float, float, float, float] | None]:
        """
        Compute BMCSL hard-sphere diameters σᵢ and moments (ξ₀,ξ₁,ξ₂,ξ₃).

        Parameters
        ----------
        y_arr : (ngrp,) packing fractions from 2PT fluidicity solve
        V_use : total system volume (Å³)

        Returns
        -------
        sigmas : (ngrp,) σᵢ in Å (0 for zero-fluidicity groups)
        xi     : (ξ₀,ξ₁,ξ₂,ξ₃) BMCSL moments, or None for single-group systems
        """
        s = self.sys
        ngrp = s.ngrp
        sigmas = np.zeros(ngrp)
        rho_gas = np.zeros(ngrp)

        for gi, grp in enumerate(s.groups):
            nmol = grp.nmol if self._molecular else grp.natom
            f_i = y_arr[gi]   # packing fraction of gas component of species i
            if f_i <= 0.0 or V_use <= 0.0:
                continue
            # Gas-phase number density: ρᵢ = N_i_gas / V_total
            # N_i_gas ≈ f_i × N_i (fluidicity × total count)
            # But yᵢ is already the packing fraction of the gas portion:
            #   yᵢ = (π/6) ρᵢ^gas σᵢ³, so we need ρᵢ^gas separately.
            # From 2PT: ρᵢ^gas = (gas DOS integral / 3) / (V_total) in molecule/Å³
            # Simpler: use total ρᵢ = nmol/V_use to approximate gas ρ.
            # This is fine because yᵢ = (π/6) ρᵢ σᵢ³ uses total ρᵢ per Lin 2003.
            rho_i = nmol / V_use   # Å⁻³
            rho_gas[gi] = rho_i
            sigmas[gi] = bmcsl_sigma_from_y(f_i, rho_i)

        if ngrp < 2:
            return sigmas, None   # pure fluid: xi not needed

        # Only include groups with nonzero sigma in the mixture moments
        mask = sigmas > 0.0
        xi = _bmcsl_xi(sigmas[mask], rho_gas[mask])
        return sigmas, xi

    # ── Thermodynamic integration ─────────────────────────────────────────────

    def _integrate_thermo(self, result: xPTResult,
                           dos_gas, dos_sol, dos_all,
                           vacT, vacDF, V_use, P_use, T_use, trjE,
                           fxpt, hsdf_arr, y_arr,
                           ngrp, vt,
                           Bg_arr=None, dos_gas_pre=None) -> None:
        """Integrate thermodynamic weighting functions over DoS.

        B1 split (2026-05-16): implementation moved to
        :mod:`pyxpt.thermo._integrate`.  See ``integrate_thermo`` there.
        """
        from pyxpt.thermo._integrate import integrate_thermo
        integrate_thermo(self, result, dos_gas, dos_sol, dos_all,
                         vacT, vacDF, V_use, P_use, T_use, trjE,
                         fxpt, hsdf_arr, y_arr, ngrp, vt, Bg_arr,
                         dos_gas_pre=dos_gas_pre)

    def _integrate_gas_solid_split(self,
                                   result: xPTResult,
                                   dos_gas, dos_sol, dos_all,
                                   vacT, vacDF, V_use, P_use, T_use, trjE,
                                   fxpt, hsdf_arr, y_arr, ngrp, vt) -> None:
        """Gas/solid split thermodynamic integration.

        B1 split (2026-05-16): implementation moved to
        :mod:`pyxpt.thermo._integrate`.
        """
        from pyxpt.thermo._integrate import integrate_gas_solid_split
        integrate_gas_solid_split(self, result, dos_gas, dos_sol, dos_all,
                                  vacT, vacDF, V_use, P_use, T_use, trjE,
                                  fxpt, hsdf_arr, y_arr, ngrp, vt)

    def _get_rot_temp(self, gi: int, rT: np.ndarray) -> None:
        """Populate rT (3,) with average rotational temperatures for group gi."""
        s   = self.sys
        grp = s.groups[gi]
        n   = 0
        for mid in grp.mol_ids:
            pI = self._pI_acc[mid] / self._nsteps
            for k in range(3):
                if pI[k] > 0:
                    rT[k] += H**2 / (8.0 * PI**2 * pI[k] * KB)
            n += 1
        if n > 0:
            rT /= n
        if grp.linear:
            rT[2] = -999.0

    # ── IR / Raman methods live in spectra/engine.py (SpectralMixin) ─────────

    # ── Output writers ────────────────────────────────────────────────────────

    def _norm_factors(self) -> tuple[bool, np.ndarray]:
        """Return (do_normalize, per-group factor array).

        ``do_normalize`` is True when either ``normalize`` or (for molecular
        mode) ``per_molecule`` is set in the config.  The factor array has
        shape ``(ngrp,)``; multiply a per-group quantity by ``factors[gi]``
        to convert it from per-SimBox to per-molecule (molecular) or per-atom
        (monoatomic).
        """
        do_norm = self.cfg.normalize or (self._molecular and self.cfg.per_molecule)
        if do_norm:
            factors = np.array([
                1.0 / max(grp.nmol if self._molecular else grp.natom, 1)
                for grp in self.sys.groups
            ])
        else:
            factors = np.ones(self.sys.ngrp)
        return do_norm, factors

    # ── GLE (Generalized Langevin Equation) analysis ─────────────────────────

    def _compute_group_cross_vac(self, tp: int) -> np.ndarray:
        """
        Cross-group collective velocity autocorrelations for mixture block-matrix GLE.

        Each group's "collective velocity" is W_I(t) = Σ_{a∈I} m_a v_a(t).
        The cross-VACF is:

            C_IJ(t, α, β) = <W_I^α(0) W_J^β(t)>

        Diagonal blocks (I=J) include both self (a=b) and distinct (a≠b)
        contributions.  Off-diagonal blocks (I≠J) are the cross-species
        collective correlations that encode inter-species friction coupling.

        For an isotropic equilibrium system C_JI(t,β,α) = C_IJ(t,α,β), so
        only the upper triangle is computed and mirrored.

        Parameters
        ----------
        tp : int
            Velocity type index (TRANS or ANGUL/ROTAT).

        Returns
        -------
        vac_cross : (ngrp, ngrp, vacmaxf, 3, 3) float64
        """
        s       = self.sys
        ngrp    = s.ngrp
        N       = self._tot_N
        ns      = self._nsteps
        vacmaxf = self._vacmaxf

        if tp == ANGUL and self._molecular and self._rot_type == ANGUL and self._angvv.size > 0:
            block   = self._angvv.astype(np.float64)     # (ns, nmol, 3)
            masses  = np.ones(block.shape[1])
            grp_idx = {gi: np.array(grp.mol_ids)
                       for gi, grp in enumerate(s.groups) if grp.mol_ids}
            npart   = block.shape[1]
        else:
            block   = self._vacvv[tp].astype(np.float64) # (ns, natom, 3)
            masses  = np.array([a.mass for a in s.atoms])
            grp_idx = {gi: np.array(grp.atom_ids)
                       for gi, grp in enumerate(s.groups) if grp.atom_ids}
            npart   = s.natom

        padded = np.zeros((N, npart, 3))
        padded[:ns] = block

        # FFT of all particles along the time axis
        padded_gpu = self._backend.to_gpu(padded)
        F_gpu = self._backend.fft(padded_gpu, axis=0)    # (N, npart, 3) complex
        F_cpu = self._backend.to_cpu(F_gpu)

        # Group-total mass-weighted FFT: F_I[ω, α] = Σ_{a∈I} m_a F[ω, a, α]
        F_grp = np.zeros((ngrp, N, 3), dtype=complex)
        for gi, idx in grp_idx.items():
            F_grp[gi] = (F_cpu[:, idx, :] * masses[idx][None, :, None]).sum(axis=1)

        # Cross-power-spectra → cross-VACFs via IFFT
        # C_IJ(t, α, β) = IFFT[ F_I[ω,α]* × F_J[ω,β] / N ] × N/ns
        vac_cross = np.zeros((ngrp, ngrp, vacmaxf, 3, 3))
        for I in range(ngrp):
            for J in range(I, ngrp):   # upper triangle + diagonal
                # Shape (N, 3, 3): outer product over spatial directions
                pwr = (F_grp[I, :, :, None].conj() * F_grp[J, :, None, :]) / N
                pwr_2d  = pwr.reshape(N, 9)   # flatten spatial for vectorised IFFT
                pwr_gpu = self._backend.to_gpu(pwr_2d)
                c_gpu   = self._backend.ifft(pwr_gpu, axis=0)
                c_real  = self._backend.to_cpu(c_gpu.real) * N / ns   # (N, 9)
                c_block = c_real[:vacmaxf].reshape(vacmaxf, 3, 3)

                vac_cross[I, J] = c_block
                if I != J:
                    # C_JI(t,β,α) = C_IJ(t,α,β)^T  (time-reversal + isotropy)
                    vac_cross[J, I] = c_block.transpose(0, 2, 1)

        return vac_cross

    # ── Phase T1: Green-Kubo transport coefficients ──────────────────────────
    def _compute_transport(self, result: "xPTResult",
                            T_use: float, V_use: float) -> None:
        """Phase T1 transport-property pipeline.

        Phase 3 split: implementation lives in
        :mod:`pyxpt.transport._compute`.  This method is a one-line
        delegation so the per-module subpackage owns its own compute
        kernel and can be packaged separately.
        """
        raise RuntimeError("transport module is not available in this 2PT/3PT-only build of py-xPT.")

    # ── [mechanics] post-pass (Phase M1) ──────────────────────────────────────
    def _compute_mechanics(self, result: "xPTResult") -> None:
        """Frequency-domain linear response from the time-domain traces stashed
        by ``_compute_transport``.

        Phase 3 split: implementation lives in
        :mod:`pyxpt.mechanics._compute`.  This method is a one-line
        delegation so the per-module subpackage owns its own compute
        kernel and can be packaged separately.
        """
        raise RuntimeError("mechanics module is not available in this 2PT/3PT-only build of py-xPT.")

    def write(self, result: xPTResult) -> None:
        """Write all output files for *result*."""
        prefix = Path(self.cfg.prefix)
        prefix.parent.mkdir(parents=True, exist_ok=True)
        p = str(prefix)

        log_path = p + ".out.log"
        with open(log_path, "w") as fh:
            print(COPYRIGHT, file=fh)
            self._write_log(result, fh)

        # F3: terse .summary.log for batch / downstream scripts that just
        # need the headline numbers without grepping the large .out.log.
        self._write_summary_file(result, p + ".summary.log")

        # Phase MEM-B: when 2PT is disabled, .vac/.pwr/.3n/.thermo would all
        # be empty (no DoS/free-energy data); skip those writers entirely.
        if not self._disable_xpt:
            self._write_vac_file(result,    p + ".vac")
            self._write_pwr_file(result,    p + ".pwr")
            self._write_int_file(result,    p + ".3n")
            self._write_thermo_file(result, p + ".thermo")
        if getattr(result, "has_cross_vac", False):
            self._write_cross_vac_file(result, p + ".cross_vac")
            self._write_cross_pwr_file(result, p + ".cross_pwr")
        if self._transport_any_enabled():
            self._write_transport_file(result, p + ".transport")
        if self._mechanics_any_enabled():
            self._write_mechanics_file(result, p + ".mechanics")
        if result.has_ir:
            self._write_ir_file(result,    p + ".ir")
            self._write_raman_file(result, p + ".raman")
            if self.cfg.hybrid_raman:
                if result.has_bpm:
                    self._write_raman_hybrid_file(result, p + ".raman-hybrid")
                else:
                    log.warning("hybrid_raman = 1 but no BPM parameters were loaded; "
                                "skipping .raman-hybrid output.")
        if result.has_spdos and self.cfg.spdos_write_per_mode:
            log.info("spdos_write_per_mode: per-mode diagnostic not yet implemented; "
                     "SP-DoS columns are in %s.raman", p)

        log.info("Output written to %s.*", prefix)

    # ── Log file (summary + inline VAC + inline DoS tables) ──────────────────

    def _write_summary_file(self, result: xPTResult, path: str) -> None:
        """Write a terse machine-friendly DIAGNOSTICS file (F3).

        This file carries ONLY run diagnostics and code-path outputs — the
        warnings/fallback count, the NPT ensemble check, the per-channel
        plateau-detected flags, and the gas_gate (Debye-consistency) report.
        It deliberately does NOT duplicate any physical results: the headline
        thermodynamics (T, V, S, A, μ, Cv) live in ``.thermo``; the DoS / VACF
        / cumulative integrals live in ``.pwr`` / ``.vac`` / ``.3n``.  Designed
        for batch aggregators that scan many runs to flag the ones that need a
        closer look (``grep 'n_warnings = [1-9]'``, ``grep 'gas_gate fired'``,
        plateau flag = 0).

        Layout:

        ::

            # pyxpt diagnostics — <prefix>
            # (results in .thermo / .pwr / .vac / .3n; this file is diagnostics only)
            # n_warnings = <int>
            # ensemble = nvt|npt   V_disp = std(V)/<V>
            # gas_gate = none|debye|debye_warn   fired = 0|1
            #   gas_gate fired: grp1 trans  P=...  f_pre=...  dS=... <unit_S>
            # gi  plat_t plat_r  gg
               1     0      0     0

        Empty (header only) when no thermodynamics module ran (mechanics-only).
        """
        ngrp = result.ngrp
        warnings_n = len(getattr(self, "_captured_warnings", []) or [])

        with open(path, "w") as fh:
            print(f"# pyxpt diagnostics — {Path(path).stem}", file=fh)
            print("# (results in .thermo / .pwr / .vac / .3n; "
                  "this file is diagnostics only)", file=fh)
            print(f"# n_warnings = {warnings_n}", file=fh)
            # Tier 2 NPT diagnostic
            print(f"# ensemble = {getattr(self, '_ensemble_resolved', 'nvt')}   "
                  f"V_disp = std(V)/<V> = {getattr(self, '_npt_v_disp', 0.0):.4g}",
                  file=fh)
            # gas_gate (Debye-consistency) diagnostic: list every channel the gate
            # flagged crystal-like, with the plateau ratio, the pre-gate fluidicity,
            # and the spurious gas entropy removed / would-be removed.
            _gg = getattr(self.cfg, "gas_gate", "none")
            if _gg in ("debye", "debye_warn") and hasattr(result, "gas_gate_fired"):
                _fired = bool(result.gas_gate_fired.any())
                print(f"# gas_gate = {_gg}   fired = {int(_fired)}   "
                      f"({'removed' if _gg == 'debye' else 'WARN-ONLY would remove'} "
                      f"the gas entropy below)", file=fh)
                if _fired:
                    for gi in range(ngrp):
                        for frow, nm in ((0, "trans"), (1, "rot")):
                            if result.gas_gate_fired[frow, gi]:
                                print(f"#   gas_gate fired: grp{gi+1} {nm}  "
                                      f"P={result.gas_gate_plateau_ratio[frow, gi]:.4g}"
                                      f"  f_pre={result.gas_gate_fluidicity_pre[frow, gi]:.4g}"
                                      f"  dS={result.gas_gate_dS_removed[frow, gi]:.3f} "
                                      f"{result.unit_S}/mol", file=fh)
            # Per-group diagnostic flags only (no thermodynamic values).
            print(f"# {'gi':>3}  {'plat_t':>6} {'plat_r':>6} {'gg':>4}", file=fh)
            for gi in range(ngrp):
                # plateau-detected flags (not tracked in this build)
                pl_t = pl_r = 0
                # gas_gate fired on any channel of this group
                gg = 0
                if (hasattr(result, "gas_gate_fired")
                        and result.gas_gate_fired.shape == (2, ngrp)):
                    gg = int(bool(result.gas_gate_fired[:, gi].any()))
                print(f"  {gi+1:>3}  {pl_t:>6d} {pl_r:>6d} {gg:>4d}", file=fh)

    def _write_log(self, result: xPTResult, fh) -> None:
        """Write the main .out.log: averaged thermo, VAC table, DoS table."""
        ngrp = result.ngrp
        vt   = result.vactype

        # ── C3 silent-fallback summary block ─────────────────────────────────
        # Surface every WARNING-level record from compute() in one prominent
        # place at the top of the log.  Common entries include kernel-inversion
        # truncation, plateau-not-detected, scalar→matrix promotion, and
        # mixture-block ill-conditioned fallbacks.  Empty section is omitted.
        _warnings = getattr(self, "_captured_warnings", None)
        if _warnings:
            n = len(_warnings)
            print(f"\n  ── Diagnostics: {n} WARNING(s) / FALLBACK(s) ─"
                  f"────────────────────────────", file=fh)
            print(f"   Inspect each before trusting the headline numbers.  "
                  f"Common categories:", file=fh)
            print(f"     kernel-inversion truncation, plateau-not-detected,",
                  file=fh)
            print(f"     scalar→matrix promotion, mixture-block fallback, "
                  f"FSC absent.", file=fh)
            print(f"  ──────────────────────────────────────────────────"
                  f"────────────────────────", file=fh)
            for i, w in enumerate(_warnings, 1):
                # Keep each entry on a single line; longer warnings get
                # truncated to ~140 chars for the summary view (the full
                # text is still in the run-time log if the user piped it).
                line = w.replace("\n", " ")
                if len(line) > 140:
                    line = line[:137] + "..."
                print(f"   {i:>3}. {line}", file=fh)
            print(file=fh)

        # ── Scalar thermodynamics ────────────────────────────────────────────
        _dof_tot = float(np.sum(result.dof[:, vt - 1]))
        T_sys = (sum(float(result.temperature[gi, vt-1]) * float(result.dof[gi, vt-1])
                     for gi in range(ngrp)) / _dof_tot
                 if _dof_tot > 0 else float(result.temperature[0, vt - 1]))
        V_sys = float(result.volume[0])
        E_sys = sum(float(result.E_md[gi, vt - 1]) for gi in range(ngrp))
        if self.cfg.out_units == "lj":
            print(f"\n{'LJ Reference Parameters:':>26}", file=fh)
            print(f"\t{'σ (sigma)':14}: {self.cfg.lj_sigma:.4f} Å", file=fh)
            print(f"\t{'ε (epsilon)':14}: {self.cfg.lj_epsilon:.4f} {self.cfg.energy_units}", file=fh)
            print(f"\t{'Mass':14}: {self.cfg.lj_mass:.4f} g/mol", file=fh)
        print(f"\n{'Avg Thermo Vals:':>22}", file=fh)
        print(f"\t{'Temperature':14}: {T_sys:.3f} {result.unit_T}", file=fh)
        print(f"\t{'Energy':14}: {E_sys:.3f} {result.unit_E}", file=fh)
        print(f"\t{'Pressure':14}: {result.pressure:.4f} {result.unit_P}", file=fh)
        print(f"\t{'Volume':14}: {V_sys:.3f} {result.unit_V}", file=fh)

        if ngrp > 1:
            print(f"\n  {'Group energies (kJ/mol):':}", file=fh)
            for gi in range(ngrp):
                print(f"    Group {gi+1}: {result.E_md[gi, vt-1]:.3f}", file=fh)
            print(f"\n  {'Group volumes (Å³):':}", file=fh)
            for gi in range(ngrp):
                print(f"    Group {gi+1}: {result.volume[gi]:.3f}", file=fh)

        # ── MC1 cross-channel coupling diagnostic ────────────────────────────
        if getattr(result, "has_cross_vac", False):
            vel_name = {0: "trans", 1: "angul", 2: "imvib",
                        3: "rotat", 4: "total"}
            pairs = result.cross_vac_pairs
            print(f"\n Cross-channel coupling ρ_αβ "
                  f"= max_t |tr C_αβ(t)| / √(tr C_αα(0)·tr C_ββ(0))", file=fh)
            print(f"   (decision gate: < 0.05 → diagonal 2PT valid; "
                  f"≥ 0.05–0.10 → off-diagonal channel coupling matters)",
                  file=fh)
            for gi in range(ngrp):
                for pi, (a, b) in enumerate(pairs):
                    print(f"   G{gi+1:03d}  {vel_name.get(a, str(a)):>6} × "
                          f"{vel_name.get(b, str(b)):<6}  ρ = "
                          f"{result.cross_coupling[gi, pi]:.4f}", file=fh)

        # ── VAC table ────────────────────────────────────────────────────────
        mf = result.vac.shape[2]
        print(f"\n Velocity autocorrelation function (g/mol·Å²/ps²)", file=fh)
        hdr_cols = "  ".join(
            f"{'VACcmt' if k==0 and vt>1 else 'VACtot' if k==vt-1 and vt>1 else 'VAC':>6}"
            f"[G{gi+1:03d}]"
            for gi in range(ngrp) for k in range(vt)
        )
        print(f" {'time(ps)':>13}  {hdr_cols}", file=fh)
        for i in range(mf):
            t_val = f" {self._vacdtime * i:13.3f}"
            row_vals = "  ".join(
                # Use np.trace() to collapse the 3x3 matrix into a scalar
                f"{np.trace(result.vac[gi, k, i]):13.3f}"
                for gi in range(ngrp) for k in range(vt)
            )
            print(f"{t_val}  {row_vals}", file=fh)

        # ── Power spectrum table ──────────────────────────────────────────────
        nused = result.frequencies.size
        # cage column on TRANS (cage_entropy) and the rotational channel
        # (cage_entropy_rot) when show_xpt_split is on — mirrors _write_pwr_file.
        split_on   = bool(getattr(self.cfg, "show_xpt_split", False))
        cage_trans = split_on and bool(getattr(self.cfg, "cage_entropy", False))
        cage_rot   = split_on and bool(getattr(self.cfg, "cage_entropy_rot", False))
        rot_ch     = getattr(self, "_rot_type", -1)
        print(f"\n Power Spectrum / Density of States (cm)", file=fh)
        pwr_hdr_parts = []
        for gi in range(ngrp):
            for k in range(vt):
                tag = f"[G{gi+1:03d},T{k}]"
                pwr_hdr_parts.append(f"{'PWR':>6}{tag}")
                if k != IMVIB:
                    pwr_hdr_parts.append(f"{'GAS':>6}{tag}")
                    if (cage_trans and k == TRANS) or (cage_rot and k == rot_ch):
                        pwr_hdr_parts.append(f"{'CAGE':>6}{tag}")
                    pwr_hdr_parts.append(f"{'SOL':>6}{tag}")
        print(f" {'freq(cm-1)':>13}  {'  '.join(pwr_hdr_parts)}", file=fh)
        for i in range(nused):
            f_val = f" {result.frequencies[i]:13.4f}"
            pwr_parts = []
            for gi in range(ngrp):
                for k in range(vt):
                    pwr_parts.append(f"{result.dos[gi, k, i]:13.4f}")
                    if k != IMVIB:
                        pwr_parts.append(f"{result.dos_gas[gi, k, i]:13.4f}")
                        if (cage_trans and k == TRANS) or (cage_rot and k == rot_ch):
                            pwr_parts.append(f"{result.dos_cage[gi, k, i]:13.4f}")
                            pwr_parts.append(
                                f"{result.dos_solid[gi, k, i] - result.dos_cage[gi, k, i]:13.4f}")
                        else:
                            pwr_parts.append(f"{result.dos_solid[gi, k, i]:13.4f}")
            print(f"{f_val}  {'  '.join(pwr_parts)}", file=fh)

        # ── Thermodynamic properties table ───────────────────────────────────
        self._write_thermo_table(result, fh)

    # ── Output unit converter ────────────────────────────────────────────────
    def _apply_unit_conversions(self, result: xPTResult) -> None:
        """Scale all thermodynamic outputs based on the requested unit system."""
        cfg = self.cfg
        ngrp = result.ngrp

        mult_E, mult_S, mult_T, mult_P, mult_V, mult_D = 1.0, 1.0, 1.0, 1.0, 1.0, 1.0

        if cfg.out_units == "j/mol":
            mult_E = 1000.0  # kJ/mol -> J/mol
            result.unit_E = "J/mol"
        elif cfg.out_units == "kcal/mol":
            mult_E = 1.0 / 4.184  # Native kJ/mol -> kcal/mol
            mult_S = 1.0 / 4.184  # Native J/K -> cal/K
            result.unit_E = "kcal/mol"
            result.unit_S = "cal/mol·K"
        elif cfg.out_units == "ev":
            J_TO_EV = 1.03642688e-5  
            mult_E = 1000.0 * J_TO_EV  # Native kJ/mol -> eV
            mult_S = J_TO_EV           # Native J/K -> eV/K
            result.unit_E = "eV"
            result.unit_S = "eV/K"
        elif cfg.out_units == "lj":
            eps_j = cfg._lj_epsilon_j_mol
            eps_kj = eps_j / 1000.0
            sigma = cfg.lj_sigma
            mass = cfg.lj_mass
            
            # E* = E_native(kJ/mol) / eps_kj(kJ/mol)
            mult_E = 1.0 / eps_kj
            # S* = S_native(J/mol·K) / R(J/mol·K)
            mult_S = 1.0 / R
            # T* = k_B T / eps_part = R T / eps_mol(J)
            mult_T = R / eps_j
            # P* = P * sigma^3 / eps_part (P in GPa = 1e9 J/m^3, sigma in m = 1e-10)
            mult_P = 1e9 * (sigma * 1e-10)**3 * NA / eps_j
            # V* = V / sigma^3 (V is in Ang^3, sigma is in Ang)
            mult_V = 1.0 / (sigma**3)
            # D* = D * (m/eps_part)^0.5 / sigma (D_raw is cm^2/s = 1e-4 m^2/s)
            eps_part = eps_j / NA
            m_part = mass * 1e-3 / NA
            mult_D = 1e-4 * math.sqrt(m_part / eps_part) / (sigma * 1e-10)

            result.unit_E = "E*"
            result.unit_S = "S*"
            result.unit_T = "T*"
            result.unit_P = "P*"
            result.unit_V = "V*"
            result.unit_D = "D*"

        # ── Apply Multipliers to Result Arrays ──
        energy_attrs = [
            "zpe", "E_classical", "E_quantum", "E_md",
            "A_quantum", "A_classical", "mu_quantum", "mu_classical",
            "zpe_gas", "zpe_solid",
            "E_classical_gas", "E_classical_solid",
            "E_quantum_gas", "E_quantum_solid",
            "E_md_gas", "E_md_solid",
            "A_quantum_gas", "A_quantum_solid",
            "A_classical_gas", "A_classical_solid",
            "mu_quantum_gas", "mu_quantum_solid",
            "mu_classical_gas", "mu_classical_solid",
            "mu_quantum_at_ref_p", "mu_classical_at_ref_p",
        ]
        entropy_attrs = [
            "S_quantum", "S_classical",
            "Cv_quantum", "Cv_classical",
            "S_quantum_gas", "S_quantum_solid", 
            "S_classical_gas", "S_classical_solid", 
            "Cv_quantum_gas", "Cv_quantum_solid",
            "Cv_classical_gas", "Cv_classical_solid"
        ]
        temp_attrs = ["temperature", "T_gas", "T_solid"]
        
        result.pressure *= mult_P
        for i in range(ngrp):
            result.volume[i] *= mult_V

        for attr in energy_attrs:
            if hasattr(result, attr):
                arr = getattr(result, attr)
                if arr is not None: arr *= mult_E
                
        for attr in entropy_attrs:
            if hasattr(result, attr):
                arr = getattr(result, attr)
                if arr is not None: arr *= mult_S
                
        for attr in temp_attrs:
            if hasattr(result, attr):
                arr = getattr(result, attr)
                if arr is not None: arr *= mult_T
                
        result.diffusivity *= mult_D
        if hasattr(result, "diffusivity_gas") and getattr(result, "diffusivity_gas") is not None:
            result.diffusivity_gas *= mult_D

    # ── Standalone file writers ───────────────────────────────────────────────

    def _write_cross_vac_file(self, result: xPTResult, path: str) -> None:
        """Write MC1 off-diagonal C_αβ(t) to <prefix>.cross_vac.

        Columns: time_ps, then for each group × pair × (i,j) the
        ``cross_vac[gi, pair, t, i, j]`` value.  Cross-correlators are
        *not* (i,j)-symmetric, so all 9 components per pair are written.
        A summary header lists the scalar diagnostic ρ_αβ per group.
        """
        ngrp = result.ngrp
        pairs = result.cross_vac_pairs
        npair = len(pairs)
        mf = result.cross_vac.shape[2]
        if npair == 0 or mf == 0:
            return

        vel_name = {0: "trans", 1: "angul", 2: "imvib", 3: "rotat", 4: "total"}

        ncols = 1 + ngrp * npair * 9
        rows = np.zeros((mf, ncols))
        rows[:, 0] = np.arange(mf) * self._vacdtime

        col = 1
        comp_labels = ["xx", "xy", "xz", "yx", "yy", "yz", "zx", "zy", "zz"]
        comp_idx = [(0,0),(0,1),(0,2),(1,0),(1,1),(1,2),(2,0),(2,1),(2,2)]
        hdr_lines = [
            "# MC1 cross-channel velocity autocorrelators C_αβ(t)",
            "# pair α×β:  trace ρ_αβ = max_t |tr C_αβ| / √(tr C_αα(0) tr C_ββ(0))",
        ]
        for gi in range(ngrp):
            for pi, (a, b) in enumerate(pairs):
                hdr_lines.append(
                    "#   G%d  %s × %s :  ρ = %.4f"
                    % (gi + 1, vel_name.get(a, str(a)), vel_name.get(b, str(b)),
                       float(result.cross_coupling[gi, pi]))
                )

        hdr_cols = [f"{'time_ps':>12}"]
        for gi in range(ngrp):
            for pi, (a, b) in enumerate(pairs):
                tag = f"G{gi+1}_{vel_name.get(a, str(a))}x{vel_name.get(b, str(b))}"
                for lab in comp_labels:
                    hdr_cols.append(f"{tag + '_' + lab:>18}")
                for ci, (i, j) in enumerate(comp_idx):
                    rows[:, col] = result.cross_vac[gi, pi, :, i, j]
                    col += 1

        with open(path, "w") as fh:
            for line in hdr_lines:
                print(line, file=fh)
            print("# " + " ".join(hdr_cols), file=fh)
            np.savetxt(fh, rows, fmt="%14.6e")

    def _write_cross_pwr_file(self, result: xPTResult, path: str) -> None:
        """Write MC1 cross-channel power spectra |S_αβ(ν)| to
        <prefix>.cross_pwr.

        Columns: frequency [cm⁻¹], then for each group × pair × (i,j)
        the modulus ``|S_αβ_ij(ν)|`` of the cross-spectrum.  No thermal
        normalization is applied — this is the raw rfft of C_αβ(t)
        (interpret bandwise alongside the diagonal-channel DoS in
        ``<prefix>.pwr``).
        """
        ngrp = result.ngrp
        pairs = result.cross_vac_pairs
        npair = len(pairs)
        nused = result.frequencies.size
        if npair == 0 or nused == 0:
            return

        vel_name = {0: "trans", 1: "angul", 2: "imvib", 3: "rotat", 4: "total"}
        comp_labels = ["xx", "xy", "xz", "yx", "yy", "yz", "zx", "zy", "zz"]
        comp_idx = [(0,0),(0,1),(0,2),(1,0),(1,1),(1,2),(2,0),(2,1),(2,2)]

        ncols = 1 + ngrp * npair * 9
        rows = np.zeros((nused, ncols))
        rows[:, 0] = result.frequencies
        col = 1
        hdr_cols = [f"{'freq_cm-1':>12}"]
        for gi in range(ngrp):
            for pi, (a, b) in enumerate(pairs):
                tag = f"G{gi+1}_{vel_name.get(a, str(a))}x{vel_name.get(b, str(b))}"
                for lab in comp_labels:
                    hdr_cols.append(f"{'|'+tag+'_'+lab+'|':>18}")
                for ci, (i, j) in enumerate(comp_idx):
                    rows[:, col] = result.cross_pwr[gi, pi, :, i, j]
                    col += 1
        with open(path, "w") as fh:
            print("# MC1 cross-channel power spectra |S_αβ_ij(ν)|  (raw rfft "
                  "of C_αβ(t); no thermal normalization)", file=fh)
            print("# " + " ".join(hdr_cols), file=fh)
            np.savetxt(fh, rows, fmt="%14.6e")

    def _write_vac_file(self, result: xPTResult, path: str) -> None:
        """Write VAC vs. lag time to <prefix>.vac (tab-separated)."""
        ngrp = result.ngrp
        vt   = result.vactype
        mf   = result.vac.shape[2]

        rows = np.zeros((mf, 1 + ngrp * vt))
        rows[:, 0] = np.arange(mf) * self._vacdtime
        for gi in range(ngrp):
            for k in range(vt):
                # Extract the 3x3 matrix across all lags, then sum the diagonals
                vac_mat = result.vac[gi, k] # Shape: (mf, 3, 3)
                vac_trace = vac_mat[:, 0, 0] + vac_mat[:, 1, 1] + vac_mat[:, 2, 2]

                rows[:, 1 + gi * vt + k] = vac_trace

        vel_type_name = {0: "trans", 1: "angul", 2: "imvib", 3: "rotat", 4: "total"}
        hdr_parts = [f"{'time_ps':>12}"] + [
            f"{'VAC_G'+str(gi+1)+'_'+vel_type_name.get(k,str(k)):>14}"
            for gi in range(ngrp) for k in range(vt)
        ]
        np.savetxt(path, rows, header=" ".join(hdr_parts), fmt="%14.4f")



    def _write_pwr_file(self, result: xPTResult, path: str) -> None:
        """Write 2PT power spectrum (DoS) to <prefix>.pwr (tab-separated).

        Imvib has no gas/solid split in 2PT (purely solid-like), so only the
        total DoS column is written for that velocity type.

        When normalize is active the DoS columns are divided by the per-group
        molecule count (molecular mode) or atom count (monoatomic mode) so
        that integrating the DoS gives degrees of freedom per molecule/atom
        rather than for the whole simulation box.

        When ``show_xpt_split`` is active together with a cage correction, the
        affected channel's split is written as ``gas | cage | sol`` (the
        cage-memory excess carved out of the solid), with
        ``sol = DoS_total − gas − cage``.  ``cage_entropy`` produces this on the
        TRANS channel; ``cage_entropy_rot`` produces it on the rotational channel
        (ROTAT/ANGUL).  All other channels keep the standard ``gas | sol`` pair.

        IR/Raman spectra are written to separate .ir and .raman files.
        """
        ngrp  = result.ngrp
        vt    = result.vactype
        nused = result.frequencies.size

        do_normalize, nf_arr = self._norm_factors()

        # correction, the affected channel's split becomes gas | cage | sol, with
        # the cage excess carved out of the solid (sol = total − gas − cage).
        # cage_entropy → TRANS channel; cage_entropy_rot → rotational channel
        # (ROTAT/ANGUL, populated in block 9d).  Other channels and the non-cage
        # case keep the standard gas | sol pair.
        split_on   = bool(getattr(self.cfg, "show_xpt_split", False))
        cage_trans = split_on and bool(getattr(self.cfg, "cage_entropy", False))
        cage_rot   = split_on and bool(getattr(self.cfg, "cage_entropy_rot", False))
        rot_ch     = getattr(self, "_rot_type", -1)

        vel_type_name = {0: "trans", 1: "angul", 2: "imvib", 3: "rotat", 4: "total"}
        cols      = [result.frequencies]
        hdr_parts = [f"{'freq_cm-1':>12}"]
        for gi in range(ngrp):
            nf = nf_arr[gi]
            for k in range(vt):
                n = f"G{gi+1}_{vel_type_name.get(k, str(k))}"
                cols.append(result.dos[gi, k] * nf)
                hdr_parts.append(f"{'DoS_'+n:>14}")
                if k in (IMVIB, TOTAL):
                    continue
                cols.append(result.dos_gas[gi, k] * nf)
                hdr_parts.append(f"{'gas_'+n:>14}")
                if (cage_trans and k == TRANS) or (cage_rot and k == rot_ch):
                    cols.append(result.dos_cage[gi, k] * nf)
                    cols.append((result.dos_solid[gi, k]
                                 - result.dos_cage[gi, k]) * nf)
                    hdr_parts += [f"{'cage_'+n:>14}", f"{'sol_'+n:>14}"]
                else:
                    cols.append(result.dos_solid[gi, k] * nf)
                    hdr_parts.append(f"{'sol_'+n:>14}")
        rows = np.column_stack(cols)

        if do_normalize:
            kind = "molecule" if self._molecular else "atom"
            norm_lines = []
            for gi, grp in enumerate(self.sys.groups):
                count = grp.nmol if self._molecular else grp.natom
                norm_lines.append(
                    f"Group {gi+1}: DoS normalized by {count} {kind}{'s' if count != 1 else ''}"
                )
            norm_comment = "Normalizing DoS columns — " + "; ".join(norm_lines) + "\n"
        else:
            norm_comment = ""

        np.savetxt(path, rows,
                   header=norm_comment + " ".join(hdr_parts),
                   fmt="%14.4f")

    def _write_int_file(self, result: xPTResult, path: str) -> None:
        """Write cumulative DoS integral to <prefix>.3n (tab-separated)."""
        ngrp  = result.ngrp
        vt    = result.vactype
        nused = result.frequencies.size
        nu    = self._pwrfreq

        rows = np.zeros((nused, 1 + ngrp * vt))
        rows[:, 0] = result.frequencies
        col = 1
        for gi in range(ngrp):
            for k in range(vt):
                rows[:, col] = cumulative_trapezoid(
                    result.dos[gi, k], dx=nu, initial=0.0
                )
                col += 1

        vel_type_name = {0: "trans", 1: "angul", 2: "imvib", 3: "rotat", 4: "total"}
        hdr_parts = [f"{'freq_cm-1':>12}"] + [
            f"{'INT_G'+str(gi+1)+'_'+vel_type_name.get(k,str(k)):>14}"
            for gi in range(ngrp) for k in range(vt)
        ]
        np.savetxt(path, rows, header=" ".join(hdr_parts), fmt="%14.4f")

    def _write_thermo_file(self, result: xPTResult, path: str) -> None:
        """Write thermodynamic property table to <prefix>.thermo."""
        lines = [COPYRIGHT, ""]
        self._write_thermo_table(result, lines_out=lines)
        Path(path).write_text("\n".join(lines) + "\n")

    def _write_transport_file(self, result: xPTResult, path: str) -> None:
        """Write Green-Kubo transport coefficients to <prefix>.transport.

        The format is a plain-text key=value table with explicit units and
        per-property method strings (block count, averaging method, kref
        choice).  When ``cfg.transport_report_uncertainty`` is true and the
        averaging method produces an uncertainty, ± stderr (or 95 % CI for
        bootstrap variants) is reported alongside the mean.

        For η_shear, the stress relaxation function G(t) from the first
        block is appended as a two-column (t [ps], G(t) [Pa]) block at the
        end of the file when computed.
        """
        c     = self.cfg
        ngrp  = self.sys.ngrp
        lines: list[str] = [COPYRIGHT, ""]
        lines.append(" Green-Kubo transport coefficients")
        lines.append(f"   averaging          : {c.transport_averaging}")
        lines.append(f"   n_blocks           : {c.transport_n_blocks}")
        if c.transport_averaging in ("bootstrap", "block-bootstrap"):
            lines.append(f"   n_bootstrap        : {c.transport_n_bootstrap}")
        lines.append(f"   burn_in_frames     : {c.transport_burn_in_frames}")
        lines.append("")

        def _fmt_quad(q: np.ndarray, unit: str) -> str:
            mean, se, lo, hi = float(q[0]), float(q[1]), float(q[2]), float(q[3])
            s = f"{mean:14.6e}  {unit}"
            if c.transport_report_uncertainty:
                if se > 0.0:
                    s += f"  ± {se:.3e}"
                if np.isfinite(lo) and np.isfinite(hi):
                    s += f"  CI95: [{lo:.3e}, {hi:.3e}]"
            return s

        if c.transport_diffusion:
            lines.append(" Translational self-diffusion D")
            for gi in range(ngrp):
                d = result.transport_D_trans[gi]
                if not np.isnan(d[0]):
                    lines.append(f"   group {gi+1:2d}: D_trans = "
                                  + _fmt_quad(d, "cm²/s"))
            lines.append("")

        if c.transport_rotation and self._molecular:
            lines.append(" Rotational dynamics")
            lines.append("   I_ω  = ∫⟨ω(0)·ω(t)⟩ dt    (per-axis, per-molecule mean) [1/ps]")
            lines.append("   τ_ω  = I_ω / ⟨ω_α²⟩(0)    (angular-velocity correlation time) [ps]")
            for gi in range(ngrp):
                d = result.transport_D_rot[gi]
                if not np.isnan(d[0]):
                    wsq = float(result.transport_omega_sq[gi])
                    tau_w = (d[0] / wsq) if wsq > 0.0 else float("nan")
                    lines.append(f"   group {gi+1:2d}: I_ω   = "
                                  + _fmt_quad(d, "1/ps"))
                    lines.append(f"            ⟨ω_α²⟩ = "
                                  f"{wsq:14.6e}  (rad/ps)²")
                    if not math.isnan(tau_w):
                        lines.append(f"            τ_ω   = "
                                      f"{tau_w:14.6e}  ps")
            lines.append("")

        if c.transport_shear_viscosity:
            lines.append(" Shear viscosity η_shear")
            eta = result.transport_eta_shear
            if not np.isnan(eta[0]):
                lines.append(f"             η_shear = " + _fmt_quad(eta, "Pa·s"))
            lines.append("")

        if c.transport_bulk_viscosity:
            lines.append(" Bulk viscosity η_bulk  (virial pressure trace)")
            etab = result.transport_eta_bulk
            if not np.isnan(etab[0]):
                lines.append(f"             η_bulk  = " + _fmt_quad(etab, "Pa·s"))
            lines.append("")

        if c.transport_dielectric:
            log.info("transport_dielectric requested but the transport "
                     "subsystem is not available in this 2PT/3PT build; "
                     "skipping ε(0).")

        Path(path).write_text("\n".join(lines) + "\n")

    def _write_mechanics_file(self, result: "xPTResult", path: str) -> None:
        """Write [mechanics] outputs to <prefix>.mechanics.

        Contains the time-domain relaxation traces and the frequency-domain
        spectra (storage / loss moduli, loss tangents, dielectric spectrum).
        Created only when at least one mechanics_* flag is enabled.
        """
        c = self.cfg
        lines: list[str] = [COPYRIGHT, ""]
        lines.append(" Mechanical / dynamic-mechanical analysis")
        lines.append(f"   averaging          : {c.transport_averaging}  (inherits [transport])")
        lines.append(f"   n_blocks           : {c.transport_n_blocks}")
        lines.append(f"   acf_corlen         : {c.transport_acf_corlen}")
        lines.append(f"   n_freqs            : {c.mechanics_n_freqs} (log-spaced)")
        lines.append("")

        # ── (a) Shear DMA ─────────────────────────────────────────────────
        if (c.mechanics_dma_shear and result.mechanics_G_t.size > 0
                and result.mechanics_G_t_dt_ps > 0.0):
            dt = result.mechanics_G_t_dt_ps
            lines.append(" Shear stress relaxation function G(t)  [first block]")
            lines.append("       t [ps]            G(t) [Pa]")
            for k, g in enumerate(result.mechanics_G_t):
                lines.append(f"   {k * dt:14.6e}   {float(g):14.6e}")
            lines.append("")
            if (result.mechanics_G_freq_THz.size > 1
                    and result.mechanics_G_prime.size == result.mechanics_G_freq_THz.size):
                lines.append(" Shear moduli G'(ω), G''(ω) and tan δ_s")
                lines.append("      f [THz]         G'(ω) [Pa]      G''(ω) [Pa]    tan δ_s")
                for k in range(result.mechanics_G_freq_THz.size):
                    td = float(result.mechanics_tan_delta_shear[k]) if result.mechanics_tan_delta_shear.size else float("nan")
                    lines.append(
                        f"   {float(result.mechanics_G_freq_THz[k]):14.6e}"
                        f"   {float(result.mechanics_G_prime[k]):14.6e}"
                        f"   {float(result.mechanics_G_double[k]):14.6e}"
                        f"   {td:12.4e}"
                    )
                lines.append("")

        # ── (b) Bulk DMA ──────────────────────────────────────────────────
        if (c.mechanics_dma_bulk and result.mechanics_K_t.size > 0
                and result.mechanics_K_t_dt_ps > 0.0):
            dt = result.mechanics_K_t_dt_ps
            lines.append(" Bulk relaxation function K(t)  [first block]")
            lines.append("       t [ps]            K(t) [Pa]")
            for k, k_val in enumerate(result.mechanics_K_t):
                lines.append(f"   {k * dt:14.6e}   {float(k_val):14.6e}")
            lines.append("")
            if (result.mechanics_K_freq_THz.size > 1
                    and result.mechanics_K_prime.size == result.mechanics_K_freq_THz.size):
                lines.append(" Bulk moduli K'(ω), K''(ω) and tan δ_b")
                lines.append("      f [THz]         K'(ω) [Pa]      K''(ω) [Pa]    tan δ_b")
                for k in range(result.mechanics_K_freq_THz.size):
                    td = float(result.mechanics_tan_delta_bulk[k]) if result.mechanics_tan_delta_bulk.size else float("nan")
                    lines.append(
                        f"   {float(result.mechanics_K_freq_THz[k]):14.6e}"
                        f"   {float(result.mechanics_K_prime[k]):14.6e}"
                        f"   {float(result.mechanics_K_double[k]):14.6e}"
                        f"   {td:12.4e}"
                    )
                lines.append("")

        # ── (c) Dielectric spectrum ε(ω) ──────────────────────────────────
        if (c.mechanics_dielectric_spectrum
                and result.mechanics_eps_freq_THz.size > 1):
            lines.append(" Dielectric ε'(ω), ε''(ω) and tan δ_ε  [single-trajectory]")
            lines.append("      f [THz]         ε'(ω)            ε''(ω)         tan δ_ε")
            for k in range(result.mechanics_eps_freq_THz.size):
                td = float(result.mechanics_tan_delta_eps[k]) if result.mechanics_tan_delta_eps.size else float("nan")
                lines.append(
                    f"   {float(result.mechanics_eps_freq_THz[k]):14.6e}"
                    f"   {float(result.mechanics_eps_real[k]):14.6e}"
                    f"   {float(result.mechanics_eps_imag[k]):14.6e}"
                    f"   {td:12.4e}"
                )
            lines.append("")

        # ── (d) Elastic constants C_ij and derived quantities (Phase M2) ──
        # Print whenever any element is finite — partial tensors (e.g.
        # K-only from isotropic NPT) are still useful.
        if (c.mechanics_elastic_constants
                and np.any(np.isfinite(result.mechanics_C_ij))):
            method_used = result.mechanics_elastic_method_used
            method_label = {
                "strain_fluct": "strain-fluctuation (Parrinello-Rahman, NPT)",
                "stress_fluct": "stress-fluctuation (NVT, partial — Born hessian omitted)",
            }.get(method_used, method_used)
            lines.append(f" Elastic constants C_ij  [Pa]   ({method_label})")
            lines.append("   Voigt order: 1=xx, 2=yy, 3=zz, 4=yz, 5=xz, 6=xy")
            lines.append("   ┌" + "─" * 84 + "┐")
            for i in range(6):
                row = "   │ " + "  ".join(
                    (f"{result.mechanics_C_ij[i, j]:12.4e}"
                     if np.isfinite(result.mechanics_C_ij[i, j])
                     else f"{'         nan':>12s}")
                    for j in range(6)
                ) + "  │"
                lines.append(row)
            lines.append("   └" + "─" * 84 + "┘")
            lines.append("")

            if c.mechanics_vrh_averages and np.isfinite(result.mechanics_K_VRH):
                lines.append(" Voigt-Reuss-Hill averages")
                lines.append(f"   K_VRH      = {result.mechanics_K_VRH:14.6e}  Pa  (bulk modulus)")
                lines.append(f"   G_VRH      = {result.mechanics_G_VRH:14.6e}  Pa  (shear modulus)")
                lines.append(f"   E_VRH      = {result.mechanics_E_VRH:14.6e}  Pa  (Young's modulus)")
                lines.append(f"   ν_VRH      = {result.mechanics_nu_VRH:14.6f}     (Poisson's ratio)")
                lines.append("")

            if c.mechanics_pugh_ratio and np.isfinite(result.mechanics_pugh_ratio):
                kp = result.mechanics_pugh_ratio
                tag = "ductile" if kp < 0.57 else "brittle"
                lines.append(f" Pugh's ratio  G/K = {kp:8.4f}  ({tag} regime; threshold 0.57)")
                lines.append("")

            if c.mechanics_anisotropy:
                lines.append(" Anisotropy indices")
                if np.isfinite(result.mechanics_A_zener):
                    az = result.mechanics_A_zener
                    lines.append(f"   Zener A_Z      = {az:10.4f}  (= 1 for isotropic / cubic-isotropic)")
                if np.isfinite(result.mechanics_A_universal):
                    au = result.mechanics_A_universal
                    lines.append(f"   Universal A^U  = {au:10.4f}  (≥ 0; 0 for isotropic)")
                lines.append("")

            if c.mechanics_sound_velocity and np.isfinite(result.mechanics_v_L):
                lines.append(" Sound velocities  (from VRH moduli + density)")
                lines.append(f"   v_L (longitudinal) = {result.mechanics_v_L:10.2f}  m/s")
                lines.append(f"   v_T (transverse)   = {result.mechanics_v_T:10.2f}  m/s")
                lines.append(f"   v_D (Debye-avg)    = {result.mechanics_v_D:10.2f}  m/s")
                lines.append("")

            if c.mechanics_debye_temperature and np.isfinite(result.mechanics_debye_T):
                lines.append(f" Debye temperature  Θ_D = {result.mechanics_debye_T:10.2f}  K")
                lines.append("")

            if (c.mechanics_born_stability
                    and np.all(np.isfinite(result.mechanics_C_ij))):
                stable = result.mechanics_born_stable
                lines.append(f" Born stability       = {'STABLE' if stable else 'UNSTABLE'}")
                lines.append(f"   eigenvalues of C   = "
                             + "  ".join(f"{e:.3e}" for e in result.mechanics_C_eigenvalues))
                if not stable:
                    lines.append("   ⚠ at least one eigenvalue ≤ 0 — system is mechanically unstable")
                lines.append("")
            elif c.mechanics_born_stability:
                lines.append(" Born stability       = (not assessable — partial C_ij tensor)")
                lines.append("")

        # ── (e) Multi-mode Maxwell (Prony) fits to G(t) and K(t) (M3b) ────
        def _emit_prony(label, tau, amp, G_inf, resid, unit="Pa"):
            if tau.size == 0:
                return
            lines.append(f" {label} Prony series fit  ({label} = G_∞ + Σ Gᵢ·exp(−t/τᵢ))")
            lines.append(f"   residual / ||{label}|| = {resid:.4f}; G_∞ = {G_inf:.4e} {unit}")
            lines.append(f"   τᵢ [ps]              Gᵢ [{unit}]          fraction (Gᵢ/Σ Gⱼ)")
            G_total = float(amp.sum()) if amp.size else 0.0
            for ti, gi in zip(tau, amp):
                if gi > 1e-3 * (amp.max() if amp.size else 0.0):
                    frac = (gi / G_total) if G_total > 0 else 0.0
                    lines.append(f"   {ti:14.4e}     {gi:14.4e}     {frac:8.4f}")
            lines.append("")
        if c.mechanics_maxwell_fit:
            _emit_prony("G(t)",
                        np.asarray(result.mechanics_G_maxwell_tau),
                        np.asarray(result.mechanics_G_maxwell_amp),
                        result.mechanics_G_maxwell_inf,
                        result.mechanics_G_maxwell_resid)
            _emit_prony("K(t)",
                        np.asarray(result.mechanics_K_maxwell_tau),
                        np.asarray(result.mechanics_K_maxwell_amp),
                        result.mechanics_K_maxwell_inf,
                        result.mechanics_K_maxwell_resid)

        # Method log
        if result.mechanics_method_log:
            lines.append(" Method log:")
            for s in result.mechanics_method_log:
                lines.append(f"   • {s}")
            lines.append("")

        Path(path).write_text("\n".join(lines) + "\n")

    def _write_thermo_table(self, result: xPTResult,
                             fh=None, lines_out: list | None = None) -> None:
        """
        Write the formatted thermodynamic property table.

        Output goes to *fh* (file handle) when given, or appended to
        *lines_out* (list of strings) otherwise.
        
        If show_xpt_split is enabled, outputs 3 columns per property:
        gas phase, solid phase, and total.
        
        Note: ANGUL column is skipped for molecular mode (always zero).
        """
        ngrp = result.ngrp
        vt   = result.vactype
        show_xpt_split = self.cfg.show_xpt_split
        show_classical = self.cfg.show_classical_thermo

        # For molecular mode: show Trans, active rotational type as "Rot", Imvib, Total.
        # Skip the inactive rotational type (ROTAT when using vangul, ANGUL when using vrotat).
        vel_type_name = {0: "Trans", 1: "Rot", 2: "Imvib", 3: "Rot", 4: "Total"}
        if self._molecular:
            other_rot = ANGUL if self._rot_type == ROTAT else ROTAT
            vel_types_to_show = [k for k in range(vt) if k != other_rot]
        else:
            vel_types_to_show = list(range(vt))

        def emit(line: str) -> None:
            if fh is not None:
                print(line, file=fh)
            if lines_out is not None:
                lines_out.append(line)

        # Column header — each sub-column is 10 chars wide to match data fmt "10.xf"
        def _col_label(gi: int, k: int) -> str:
            return f"G{gi+1}_{vel_type_name.get(k, str(k))}"

        if show_xpt_split:
            col_hdr = "  ".join(
                f"{_col_label(gi,k)+'.'+ph:>10}"
                for gi in range(ngrp) for k in vel_types_to_show
                for ph in (["t"] if k in (IMVIB, TOTAL) else ["g", "s", "t"])
            )
        else:
            col_hdr = "  ".join(
                f"{_col_label(gi, k):>10}"
                for gi in range(ngrp) for k in vel_types_to_show
            )

        n_cols = (ngrp * sum(1 if k in (IMVIB, TOTAL) else 3 for k in vel_types_to_show)
                  if show_xpt_split else ngrp * len(vel_types_to_show))

        # Normalization: divide extensive quantities by nmol (molecular) or natom (mono).
        do_normalize, norm_factor = self._norm_factors()
        if do_normalize:
            # For molecular mode, dividing by nmol converts "per sim-box" → "per molecule"
            # (the same unit as result.unit_E, e.g. kJ/mol), so no suffix is needed.
            # For mono mode, dividing by natom gives "per atom", shown as /atom.
            norm_denom = "" if self._molecular else "atom"

            def N(arr: np.ndarray | None) -> np.ndarray | None:
                """Apply per-group normalization to a (ngrp, vt) array."""
                if arr is None:
                    return None
                return arr * norm_factor[:, None]
        else:
            norm_denom = "SimBox"
            def N(arr: np.ndarray | None) -> np.ndarray | None:  # type: ignore[misc]
                return arr

        # Helper: build "unit/denom" or just "unit" when denom is empty
        def _ulabel(unit: str) -> str:
            return f"{unit}/{norm_denom}" if norm_denom else unit

        emit(f"\n Calculation of Thermodynamic Properties")

        if do_normalize:
            kind = "molecule" if self._molecular else "atom"
            if ngrp == 1:
                count = (self.sys.groups[0].nmol if self._molecular
                         else self.sys.groups[0].natom)
                emit(f"# Normalizing extensive thermodynamic properties by "
                     f"{count} {kind}{'s' if count != 1 else ''}")
            else:
                emit(f"# Normalizing extensive thermodynamic properties per group:")
                for gi, grp in enumerate(self.sys.groups):
                    count = grp.nmol if self._molecular else grp.natom
                    emit(f"#   Group {gi+1}: {count} {kind}{'s' if count != 1 else ''}")

        if self.cfg.out_units == "lj":
            emit(f"  * Reduced Units derived via: σ = {self.cfg.lj_sigma} Å, "
                 f"ε = {self.cfg.lj_epsilon} {self.cfg.energy_units}, "
                 f"mass = {self.cfg.lj_mass} g/mol")

        emit(f"  {'Property':<30}  {col_hdr}")
        emit(" " + "─" * (32 + 12 * n_cols))

        def row(label: str, arr: np.ndarray, arr_gas: np.ndarray | None = None,
                arr_solid: np.ndarray | None = None, fmt: str = "10.3f") -> None:
            if show_xpt_split and arr_gas is not None and arr_solid is not None:
                # Gas / Solid / Total columns; Total column emits total only (no g/s split)
                parts = []
                for gi in range(ngrp):
                    for k in vel_types_to_show:
                        if k in (IMVIB, TOTAL):
                            parts.append(f"{arr[gi, k]:{fmt}}")
                        else:
                            parts.append(
                                f"{arr_gas[gi, k]:{fmt}}  {arr_solid[gi, k]:{fmt}}  {arr[gi, k]:{fmt}}"
                            )
                vals = "  ".join(parts)
            else:
                # Total only
                vals = "  ".join(f"{arr[gi, k]:{fmt}}"
                                 for gi in range(ngrp) for k in vel_types_to_show)
            emit(f"  {label:<30}  {vals}")

        def row_scalar(label: str, vals_list: list, fmt: str = "10.3f") -> None:
            vals = "  ".join(f"{v:{fmt}}" for v in vals_list)
            emit(f"  {label:<30}  {vals}")

        # Per-group/type rows
        row("DOF", result.dof,
            result.dof_gas if show_xpt_split else None,
            result.dof_solid if show_xpt_split else None)
        
        row(f"Temperature ({result.unit_T})", result.temperature,
            result.T_gas if show_xpt_split else None,
            result.T_solid if show_xpt_split else None, "10.2f")

        row(f"Pressure ({result.unit_P})",
            np.full((ngrp, vt), result.pressure),
            np.full((ngrp, vt), result.pressure) if show_xpt_split else None,
            np.full((ngrp, vt), result.pressure) if show_xpt_split else None, "10.4f")

        vol_arr = np.array([[result.volume[gi]] * vt for gi in range(ngrp)])
        row(f"Volume ({result.unit_V})", vol_arr,
            vol_arr if show_xpt_split else None,
            vol_arr if show_xpt_split else None, "10.2f")

        row(f"ZPE ({_ulabel(result.unit_E)})", N(result.zpe),
            N(result.zpe_gas) if show_xpt_split else None,
            N(result.zpe_solid) if show_xpt_split else None)

        row(f"E_md ({_ulabel(result.unit_E)})", N(result.E_md),
            N(result.E_md_gas) if show_xpt_split else None,
            N(result.E_md_solid) if show_xpt_split else None)

        row(f"E_q ({_ulabel(result.unit_E)})", N(result.E_quantum),
            N(result.E_quantum_gas) if show_xpt_split else None,
            N(result.E_quantum_solid) if show_xpt_split else None)

        row(f"S_q ({_ulabel(result.unit_S)})", N(result.S_quantum),
            N(result.S_quantum_gas) if show_xpt_split else None,
            N(result.S_quantum_solid) if show_xpt_split else None)

        # 3PT cage-memory + anharmonic correction (already folded into S_q above).
        # It is a solid-side correction, so in split mode it appears under the
        # ".s" (solid) sub-column of the Trans and Rot channels (gas column = 0).
        if self.cfg.mode == ModexPT.THREE_PT:
            row(f"S_cage ({_ulabel(result.unit_S)})", N(result.S_cage),
                N(np.zeros_like(result.S_cage)) if show_xpt_split else None,
                N(result.S_cage) if show_xpt_split else None)

        row(f"A_q ({_ulabel(result.unit_E)})", N(result.A_quantum),
            N(result.A_quantum_gas) if show_xpt_split else None,
            N(result.A_quantum_solid) if show_xpt_split else None)

        row(f"μ_q ({_ulabel(result.unit_E)})", N(result.mu_quantum),
            N(result.mu_quantum_gas) if show_xpt_split else None,
            N(result.mu_quantum_solid) if show_xpt_split else None)

        # Tier 1 NPT support: emit pressure-corrected μ_q if cfg.reference_pressure
        # was set.  Only the Total column carries the V·(P_ref − P_sim)
        # correction; other vactype columns (Trans.g/s, Rot.g/s, Imvib) keep
        # the uncorrected μ_q values (the PV shift is a system-level scalar
        # and doesn't decompose across DoF channels).  Pass the same array
        # for arr_gas/arr_solid so the row width matches the show_xpt_split
        # header and the .thermo parser handles it.
        if getattr(self.cfg, "reference_pressure", None) is not None:
            row(f"μ_q_at_ref_P ({_ulabel(result.unit_E)})",
                N(result.mu_quantum_at_ref_p),
                N(result.mu_quantum_at_ref_p) if show_xpt_split else None,
                N(result.mu_quantum_at_ref_p) if show_xpt_split else None)

        # ── Mixing entropy comment (only when user has overridden group volumes) ──
        # When grp.volume > 0 the Sackur-Tetrode entropy uses that volume instead
        # of V_total, so the ideal translational mixing entropy is absent.
        # Δμ_i = kT·ln(x_i) is the per-molecule correction needed to recover it.
        override_groups = [gi for gi, grp in enumerate(self.sys.groups)
                           if grp.volume > 0]
        if override_groups and ngrp > 1:
            N_total = sum(
                (self.sys.groups[gi].nmol if self._molecular
                 else self.sys.groups[gi].natom)
                for gi in range(ngrp)
            )
            T_avg = float(np.mean(result.temperature[:, 0]))   # representative T
            # kT in the output energy unit
            kT_out = KB * NA * T_avg * 1e-3 * {   # native kJ/mol per molecule
                "kj/mol":   1.0,
                "j/mol":    1000.0,
                "kcal/mol": 1.0 / 4.184,
                "ev":       1000.0 * 1.03642688e-5,
            }.get(self.cfg.out_units, 1.0)
            emit(f"#")
            emit(f"# NOTE: group volumes were overridden for group(s) "
                 f"{[gi+1 for gi in override_groups]}.")
            emit(f"# The Sackur-Tetrode entropy uses the specified volume, so the")
            emit(f"# ideal translational mixing entropy is not included in μ_q above.")
            emit(f"# Correction to add: Δμ_i = kT·ln(x_i)  (negative for x_i < 1)")
            emit(f"#   T_avg = {T_avg:.2f} K,  kT = {kT_out:.4f} {result.unit_E}/molecule")
            for gi in range(ngrp):
                grp = self.sys.groups[gi]
                N_i = grp.nmol if self._molecular else grp.natom
                x_i = N_i / N_total if N_total > 0 else 1.0
                dmu = kT_out * math.log(x_i) if x_i > 0 else 0.0
                vol_flag = " (overridden)" if grp.volume > 0 else " (V_total)"
                emit(f"#   Group {gi+1}: x_{gi+1} = {N_i}/{N_total} = {x_i:.6f}"
                     f"  Δμ_{gi+1} = {dmu:+.4f} {result.unit_E}/molecule"
                     f"  [vol{vol_flag}]")

        row(f"Cv_q ({_ulabel(result.unit_S)})", N(result.Cv_quantum),
            N(result.Cv_quantum_gas) if show_xpt_split else None,
            N(result.Cv_quantum_solid) if show_xpt_split else None)

        if show_classical:
            row(f"E_c ({_ulabel(result.unit_E)})", N(result.E_classical),
                N(result.E_classical_gas) if show_xpt_split else None,
                N(result.E_classical_solid) if show_xpt_split else None)

            row(f"S_c ({_ulabel(result.unit_S)})", N(result.S_classical),
                N(result.S_classical_gas) if show_xpt_split else None,
                N(result.S_classical_solid) if show_xpt_split else None)

            row(f"A_c ({_ulabel(result.unit_E)})", N(result.A_classical),
                N(result.A_classical_gas) if show_xpt_split else None,
                N(result.A_classical_solid) if show_xpt_split else None)

            row(f"μ_c ({_ulabel(result.unit_E)})", N(result.mu_classical),
                N(result.mu_classical_gas) if show_xpt_split else None,
                N(result.mu_classical_solid) if show_xpt_split else None)

            row(f"Cv_c ({_ulabel(result.unit_S)})", N(result.Cv_classical),
                N(result.Cv_classical_gas) if show_xpt_split else None,
                N(result.Cv_classical_solid) if show_xpt_split else None)

        # NOTE: Keeping S(0) natively in cm/mol to avoid conflating with Diffusivity
        row("S(0) (cm/mol/SimBox)", result.S_dos0,
            result.S_dos0_gas if show_xpt_split else None,
            np.zeros_like(result.S_dos0) if show_xpt_split else None, "10.4f")

        row(f"Diffusivity ({result.unit_D})", result.diffusivity,
            result.diffusivity_gas if show_xpt_split else None,
            np.zeros_like(result.diffusivity) if show_xpt_split else None, "10.4e")

        # Fluidicity — non-zero only in the relevant velocity column;
        # in split mode, only in the gas sub-column.
        rot_col = self._rot_type  # ANGUL or ROTAT
        f_trans = np.zeros((ngrp, vt))
        f_rot   = np.zeros((ngrp, vt))
        for gi in range(ngrp):
            f_trans[gi, TRANS] = result.fluidicity[0, gi]
            if TOTAL < vt:
                f_trans[gi, TOTAL] = result.fluidicity[0, gi]
            if self._molecular:    
                if rot_col < vt:
                    f_rot[gi, rot_col] = result.fluidicity[1, gi]
                if rot_col < vt and TOTAL < vt:
                    f_rot[gi, TOTAL] = result.fluidicity[1, gi]

        zeros = np.zeros((ngrp, vt))
        if show_xpt_split:
            row("Fluidicity (trans)", f_trans, f_trans, zeros, "10.4e")
            if self._molecular:
                row("Fluidicity (rot)",   f_rot,   f_rot,   zeros, "10.4e")
        else:
            row("Fluidicity (trans)", f_trans, fmt="10.4e")
            if self._molecular:
                row("Fluidicity (rot)",   f_rot,   fmt="10.4e")

        # ── Hard-sphere EOS diagnostics (2PT only) ────────────────────────────
        # The HS gas/solid partition is the 2PT construction; 3PT reports its
        # cage correction instead (S_cage above) and 1PT has no HS reference.
        if self.cfg.mode == ModexPT.TWO_PT:
            hs_eos     = self.cfg.hs_eos
            hs_entropy = self.cfg.hs_entropy
            y_arr      = result.packing_fraction
            emit(f"# Hard-sphere EOS: {hs_eos.upper()}  entropy: {hs_entropy}")
            if self.cfg.fixed_fluidicity:
                _ff = self.cfg.fixed_fluidicity
                vals = [float(_ff[gi] if len(_ff) > 1 else _ff[0]) for gi in range(ngrp)]
                emit(f"# Fixed fluidicity: {' '.join(f'{v:.6f}' for v in vals)}"
                     f"  (self-consistent solve bypassed; gas/solid split is fixed)")
            if hs_eos == "bmcsl" and ngrp > 0:
                # Reconstruct σᵢ and η_mix for diagnostic output
                eta_mix = float(np.sum(y_arr))
                emit(f"#   η_mix = {eta_mix:.6f}  (total packing fraction Σ_i yᵢ)")
                for gi, grp in enumerate(self.sys.groups):
                    nmol_g = grp.nmol if self._molecular else grp.natom
                    V_g = result.volume[gi]   # already stored as V_total for default case
                    rho_i = nmol_g / V_g if V_g > 0 else 0.0
                    sig_i = bmcsl_sigma_from_y(float(y_arr[gi]), rho_i)
                    emit(f"#   Group {gi+1}: yᵢ = {float(y_arr[gi]):.6f}, "
                         f"ρᵢ = {rho_i:.4e} Å⁻³, σᵢ = {sig_i:.4f} Å")
            elif hs_eos == "cs" and ngrp > 1:
                eta_mix = float(np.sum(y_arr))
                emit(f"#   η_mix = {eta_mix:.6f}  (one-fluid CS for μ; individual yᵢ for S)")

        emit("")

