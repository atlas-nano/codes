# -*- coding: utf-8 -*-
"""
Velocity autocorrelation function (VACF) utilities — FFT-based Wiener-
Khinchin computation.

This module is the analysis-agnostic kernel used by every pyxpt
pipeline that needs the mass-weighted 3×3 VAC tensor: 2PT thermo
(``thermo.engine``), transport coefficients (``thermo.transport``),
mechanics (``thermo.mechanics``), IR/Raman (``spectra.engine``), and
the multi-resolution stitcher (``multires.runner``).  It depends only
on ``numpy`` and the GPU backend in :mod:`pyxpt.core.gpu`; *no*
analysis-pipeline knowledge.

Public API
----------
:func:`compute_vac_tensor`
    Pure FFT kernel.  Given a velocity time series for a set of
    particles and per-particle weights (typically masses), compute the
    full 3×3 mass-weighted autocorrelation tensor for each named
    group via the Wiener-Khinchin theorem
    (``IRFFT(F.conj() · F) / ns``), zero-padded to a user-supplied
    FFT length to avoid circular wrap.

:func:`subtract_system_com_velocity`
    Per-frame mass-weighted system-COM velocity removal — the right
    drift correction for any equilibrium-MD trajectory generated
    without LAMMPS ``fix momentum``.

Memory note
-----------
The kernel uses ``rfft + float32`` throughout (one factor-2 saving
from ``rfft`` over full ``fft``, another factor-2 from ``float32``
over ``float64``) to keep peak RSS bounded on long polymer
trajectories.  See commit ``aa1e718`` for the perf history.
"""
from __future__ import annotations

import gc
import logging
from typing import Mapping

import numpy as np

log = logging.getLogger(__name__)


# ── Drift correction ─────────────────────────────────────────────────────────


def subtract_system_com_velocity(
    velocities: np.ndarray,
    masses: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Subtract the per-frame mass-weighted system-COM velocity from each
    atom's velocity.

    Parameters
    ----------
    velocities : (n_atom, 3) np.ndarray
        Per-atom velocities at one frame.
    masses : (n_atom,) np.ndarray
        Per-atom masses (any consistent unit).

    Returns
    -------
    v_corrected : (n_atom, 3) np.ndarray
        ``velocities − v_sys_COM``.
    v_sys_COM : (3,) np.ndarray
        The subtracted COM velocity.

    Notes
    -----
    In an equilibrium MD trajectory ``⟨v_sys_COM⟩`` should be exactly
    zero, but NPT runs without ``fix momentum`` accumulate non-zero
    drift that contributes a constant ``M_total · ⟨|v_sys|²⟩`` offset
    to the mass-weighted VACF, integrating linearly with τ and
    inflating D by orders of magnitude on long trajectories.  This is
    the velocity-space analogue of LAMMPS's ``compute msd com=yes``.
    """
    M_total = float(np.asarray(masses).sum())
    if M_total <= 0:
        return velocities, np.zeros(velocities.shape[1], dtype=velocities.dtype)
    v_sys = (velocities * np.asarray(masses)[:, None]).sum(axis=0) / M_total
    return velocities - v_sys[None, :], v_sys


# ── VAC tensor (Wiener-Khinchin via FFT) ─────────────────────────────────────


def compute_vac_tensor(
    velocities: np.ndarray,
    weights: np.ndarray,
    group_indices: Mapping[int, np.ndarray],
    *,
    tot_N: int,
    ns: int | None = None,
    backend=None,
) -> dict[int, np.ndarray]:
    """Compute the mass-weighted 3×3 VAC tensor for each named group.

    Parameters
    ----------
    velocities : (ns, n_part, 3) np.ndarray
        Per-particle velocity time series.  Will be zero-padded to
        length ``tot_N`` along axis 0 before FFT.
    weights : (n_part,) np.ndarray
        Per-particle weights (typically atomic masses in g/mol).  The
        returned per-group tensor is ``Σ_p weight[p] · ⟨v_p_i v_p_j⟩``.
    group_indices : dict[int, np.ndarray]
        Mapping from group id → 1-D array of particle indices into the
        first axis of ``velocities`` belonging to that group.  Only
        groups present in the dict appear in the output.
    tot_N : int
        FFT length; must satisfy ``tot_N ≥ ns + 1`` so the linear
        autocorrelation is uncontaminated by circular wrap.  Should
        ideally be 5-smooth (``next_fast_len``) for performance.
    ns : int, optional
        Number of valid frames (``velocities.shape[0]`` if not given).
    backend : optional
        GPU backend (with ``to_gpu``, ``to_cpu``, ``rfft``, ``irfft``).
        Defaults to a CPU backend.

    Returns
    -------
    dict[int, (tot_N, 3, 3) np.ndarray]
        Per-group symmetric VAC tensor.  ``vac[gi][t, i, j]`` is the
        mass-weighted sum over the group's particles of
        ``⟨v_p_i(0) · v_p_j(t)⟩``.  The 3×3 sub-tensor at each lag is
        symmetric (we compute the upper triangle and copy).

    Notes
    -----
    Implements the Wiener-Khinchin identity
    ``c[k] = (1/ns) Σ_t v[t] · v[t + k]``
    via zero-padded FFT:

    1. ``F = rfft(v_padded)`` over the time axis (``rfft + float32``).
    2. ``pwr_ij = F_i.conj() * F_j / tot_N``       (cross power spectrum)
    3. ``acf_ij = irfft(pwr_ij, tot_N) * tot_N / ns``  (Wiener-Khinchin)
    4. Mass-weight and sum over each group's particles.

    The result for any lag ``k > ns - 1`` is the *circular* mirror of
    the forward lags (``c[tot_N - k] = c[k]`` by symmetry of
    ``|F|²``); zero between ``ns`` and ``tot_N - ns``.  Callers that
    only consume the forward lag should slice ``[:ns]`` (or
    ``[:vacmaxf]``) of the returned arrays.
    """
    velocities = np.asarray(velocities)
    weights = np.asarray(weights, dtype=velocities.dtype)
    if ns is None:
        ns = int(velocities.shape[0])
    n_part = int(velocities.shape[1])
    ncomp = int(velocities.shape[2])          # spatial dimension (3 = bulk; 4 = d=4 LJ)
    if ncomp < 1:
        raise ValueError(f"velocities must have shape (ns, n_part, d) with d>=1; "
                         f"got {velocities.shape}")
    if weights.shape != (n_part,):
        raise ValueError(f"weights must be 1-D with len = n_part = {n_part}; "
                         f"got {weights.shape}")
    if tot_N < ns + 1:
        raise ValueError(f"tot_N ({tot_N}) must be ≥ ns+1 ({ns + 1}) to "
                         f"avoid circular wrap of the linear ACF")

    if backend is None:
        # Lazy import to avoid a hard dep when the caller already has one
        from pyxpt.core.gpu import get_backend
        backend = get_backend(use_gpu=False)

    # Zero-pad to tot_N in-place (float32 keeps memory bounded)
    padded = np.zeros((tot_N, n_part, ncomp), dtype=np.float32)
    padded[:ns] = velocities.astype(np.float32, copy=False)
    padded_gpu = backend.to_gpu(padded)
    F = backend.rfft(padded_gpu, axis=0)           # complex64 (tot_N//2+1, n_part, ncomp)
    del padded
    gc.collect()

    # Pre-allocate per-group output (ncomp×ncomp sub-tensor per lag)
    vac: dict[int, np.ndarray] = {
        gi: np.zeros((tot_N, ncomp, ncomp), dtype=np.float64)
        for gi in group_indices
    }

    # Compute upper triangle (i ≤ j); copy to lower at the end of each (i, j) pass.
    for i in range(ncomp):
        F_i = F[:, :, i]
        for j in range(i, ncomp):
            F_j = F[:, :, j]
            pwr_ij  = F_i.conj() * F_j / tot_N
            acf_gpu = backend.irfft(pwr_ij, n=tot_N, axis=0) * tot_N / ns
            acf     = backend.to_cpu(acf_gpu.real)         # float32 (tot_N, n_part)
            del pwr_ij, acf_gpu
            acf_mw  = acf * weights[None, :]               # mass-weighted ACF
            for gi, idx in group_indices.items():
                if len(idx) == 0:
                    continue
                col = acf_mw[:, idx].sum(axis=1)
                vac[gi][:, i, j] += col
                if i != j:
                    vac[gi][:, j, i] = vac[gi][:, i, j]
            del acf, acf_mw
    del F
    gc.collect()
    return vac


# ── Cross-channel VAC tensor (MC1 diagnostic) ────────────────────────────────


def compute_cross_vac_tensor(
    velocities_a: np.ndarray,
    velocities_b: np.ndarray,
    weights: np.ndarray,
    group_indices: Mapping[int, np.ndarray],
    *,
    tot_N: int,
    ns: int | None = None,
    backend=None,
) -> dict[int, np.ndarray]:
    """Compute the mass-weighted 3×3 cross-VAC tensor between two channels.

    For two velocity time series :math:`v^A` and :math:`v^B` of the same
    particles (e.g., :math:`v^{trans}` and :math:`v^{rot}` from molecular
    decomposition), returns the per-group cross-correlator

    .. math::

       C^{AB}_{ij}(t) = \\sum_p w_p \\, \\langle v^A_{p,i}(0)\\, v^B_{p,j}(t)\\rangle

    Unlike :func:`compute_vac_tensor` the 3×3 sub-tensor is **not**
    symmetric — ``C^{AB}_{ij}`` and ``C^{AB}_{ji}`` are independent
    statistics — and ``C^{AB}(t) ≠ C^{AB}(−t)`` in general.  All 9
    components are computed.

    The cross power spectrum ``F_A^*(ν) · F_B(ν)`` is Hermitian-symmetric
    across ν → −ν when both inputs are real, so the rfft + irfft path
    yields a strictly real cross-correlation, matching the convention
    in :func:`compute_vac_tensor`.

    Parameters
    ----------
    velocities_a, velocities_b : (ns, n_part, 3) np.ndarray
        The two velocity time series.  Must share particle layout
        (same n_part) and frame count.
    weights : (n_part,) np.ndarray
        Per-particle weights (typically atomic masses in g/mol).
    group_indices : dict[int, np.ndarray]
        Mapping from group id → 1-D array of particle indices into the
        common first axis of the two velocity arrays.
    tot_N : int
        FFT length; ``tot_N ≥ ns + 1`` to avoid circular wrap.
    ns : int, optional
        Number of valid frames (``velocities_a.shape[0]`` if not given).
    backend : optional
        GPU backend.

    Returns
    -------
    dict[int, (tot_N, 3, 3) np.ndarray]
        Per-group non-symmetric cross-VAC tensor.
    """
    va = np.asarray(velocities_a)
    vb = np.asarray(velocities_b)
    if va.shape != vb.shape:
        raise ValueError(f"velocities_a {va.shape} and velocities_b {vb.shape} "
                         f"must have identical shapes")
    weights = np.asarray(weights, dtype=va.dtype)
    if ns is None:
        ns = int(va.shape[0])
    n_part = int(va.shape[1])
    ncomp = int(va.shape[2])
    if ncomp < 1:
        raise ValueError(f"velocities must have shape (ns, n_part, d) with d>=1; "
                         f"got {va.shape}")
    if weights.shape != (n_part,):
        raise ValueError(f"weights must be 1-D with len = n_part = {n_part}; "
                         f"got {weights.shape}")
    if tot_N < ns + 1:
        raise ValueError(f"tot_N ({tot_N}) must be ≥ ns+1 ({ns + 1}) to "
                         f"avoid circular wrap of the linear ACF")

    if backend is None:
        from pyxpt.core.gpu import get_backend
        backend = get_backend(use_gpu=False)

    pa = np.zeros((tot_N, n_part, ncomp), dtype=np.float32)
    pa[:ns] = va.astype(np.float32, copy=False)
    pa_gpu = backend.to_gpu(pa)
    F_A = backend.rfft(pa_gpu, axis=0)
    del pa
    gc.collect()

    pb = np.zeros((tot_N, n_part, ncomp), dtype=np.float32)
    pb[:ns] = vb.astype(np.float32, copy=False)
    pb_gpu = backend.to_gpu(pb)
    F_B = backend.rfft(pb_gpu, axis=0)
    del pb
    gc.collect()

    cross: dict[int, np.ndarray] = {
        gi: np.zeros((tot_N, ncomp, ncomp), dtype=np.float64)
        for gi in group_indices
    }

    # All d² (i, j) entries — cross-correlator has no transpose symmetry.
    for i in range(ncomp):
        F_Ai = F_A[:, :, i]
        for j in range(ncomp):
            F_Bj = F_B[:, :, j]
            pwr_ij  = F_Ai.conj() * F_Bj / tot_N
            acf_gpu = backend.irfft(pwr_ij, n=tot_N, axis=0) * tot_N / ns
            acf     = backend.to_cpu(acf_gpu.real)
            del pwr_ij, acf_gpu
            acf_mw  = acf * weights[None, :]
            for gi, idx in group_indices.items():
                if len(idx) == 0:
                    continue
                col = acf_mw[:, idx].sum(axis=1)
                cross[gi][:, i, j] += col
            del acf, acf_mw
    del F_A, F_B
    gc.collect()
    return cross
