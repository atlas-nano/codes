# -*- coding: utf-8 -*-
"""
RAM-budget driven atom batching (Phase MEM-R1).

The 2PT engine pre-allocates several arrays whose footprint scales as
``n_frames × n_atoms`` (or ``× n_molecules``).  On large HPC trajectories
this exceeds available memory.  The C++ predecessor solved this by letting
the user specify a RAM budget and processing atoms in batches with multiple
trajectory reads; this module ports that idea to the Python engine.

Public API
----------
:func:`estimate_engine_memory`
    Predict the engine's peak memory cost (bytes) for a given config /
    system / frame count.  Includes the per-atom/per-molecule time-series
    arrays *and* the FFT working-set during ``_compute_vac``.

:func:`get_ram_budget`
    Return the user-requested RAM budget in bytes.  Honours the explicit
    ``ram_budget_gb`` knob, otherwise falls back to ``ram_budget_fraction``
    of the available system memory (default 0.5 — half of free RAM).

:func:`compute_n_batches`
    How many atom batches the engine should split into to fit under the
    budget.  Honours ``single_pass_only`` by raising MemoryError when
    batching would otherwise be required.

:func:`partition_atoms`
    Split a system's atoms (or molecules, when ``molecular=True``) into
    roughly-equal contiguous batches.  Returns a list of ``Chunk`` objects
    holding the global atom and molecule indices belonging to each batch.

Design rationale
----------------
* The engine's hottest single allocation is ``_vacvv`` of shape
  ``(vactype, n_frames, n_atoms, 3)`` in float32.  All other per-atom or
  per-molecule arrays scale with the same product and add ~2× as much.
* The FFT working-set inside ``_compute_vac`` (``rfft`` over the padded
  velocity tensor → complex64 of shape ``(N//2+1, n_part, 3)``, plus a
  float32 ACF tensor of shape ``(N, n_part)`` reused per (i,j) component)
  adds another ~1.5× of the steady-state ``_vacvv`` cost.  We use
  ``2.5 × _vacvv`` as the engineering estimate for total peak.
* For molecular systems the per-molecule arrays (``_angvv``,
  ``_omega_lab``, ``_dipvv``, optional ``_inddip``) scale with
  ``n_frames × n_molecules × 3`` and are added to the cost.  They are kept
  full-size during batching only when the batch unit is the molecule
  (which is required so that ``decompose_velocities_batch`` always sees
  every atom of a molecule).
* Per-frame transport accumulators (``_frame_T``, ``_frame_V``, ...) are
  ``O(n_frames)`` each — small relative to the per-atom arrays — so we
  ignore them in the cost estimate.
"""
from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from pyxpt.config import Config
    from pyxpt.io.trajectory import System

log = logging.getLogger(__name__)


# ── Cost coefficients (bytes per element) ─────────────────────────────────────

_F32 = 4   # float32 element size (per-atom velocity arrays)
_F64 = 8   # float64 element size (per-frame scalar accumulators)
_C64 = 8   # complex64 element size (rfft output)

# FFT working-set multiplier.  rfft on the padded (N, n_part, 3) float32 array
# produces a complex64 of shape (N//2+1, n_part, 3) plus a per-(i,j) ACF of
# shape (N, n_part) float32; both are live simultaneously.  Combined with the
# original padded array (N≈n_frames after 5-smooth round-up) this peaks at
# roughly 2.5× the steady-state _vacvv cost — calibrated empirically against
# the May-2026 perf-audit measurements (engine.py:1480 comment).
_FFT_WORKING_MULTIPLIER = 2.5


@dataclass
class Chunk:
    """A subset of atoms (and, for molecular systems, the molecules whose atoms
    are contained entirely within this batch).

    Attributes
    ----------
    atom_ids
        Global atom indices belonging to this batch.  When the engine sets
        ``_atom_mask`` to this array, ``_vacvv`` is allocated with shape
        ``(vactype, n_frames, len(atom_ids), 3)``.
    mol_ids
        Global molecule indices whose atoms are exactly ``atom_ids``.  Empty
        for non-molecular runs.  Per-molecule arrays (``_angvv`` etc.) are
        sized to ``len(mol_ids)`` when the mask is active.
    """
    atom_ids: np.ndarray
    mol_ids: np.ndarray


def _per_frame_per_atom_bytes(cfg: "Config", system: "System", n_frames: int) -> int:
    """Bytes for arrays scaling with ``n_frames × n_atoms`` and
    ``n_frames × n_mols``.

    Mirrors the allocations in :meth:`xPTEngine._setup` so the estimate
    stays close to the actual footprint.
    """
    s = system
    natom = s.natom
    nmol  = s.nmol
    # vactype: 1 (atomic) or 5 (molecular: TRANS/ROTAT/IMVIB/ANGUL/TOTAL).
    vactype = 5 if cfg.molecular else 1

    # _vacvv = (vactype, n_frames, natom, 3) float32
    vacvv = vactype * n_frames * natom * 3 * _F32

    per_mol = 0
    if cfg.molecular:
        # _angvv + _omega_lab : 2 × (n_frames, nmol, 3) float32
        per_mol += 2 * n_frames * nmol * 3 * _F32
        # _dipvv : (n_frames, nmol, 3) float32 when system has charges
        if s.has_charges or getattr(s, "has_frame_charges", False):
            per_mol += n_frames * nmol * 3 * _F32
        # _inddip : (n_frames, nmol, 3) float32 when induced dipoles present
        if getattr(s, "has_induced_dipoles", False) and cfg.use_ind:
            per_mol += n_frames * nmol * 3 * _F32

    return vacvv + per_mol


def estimate_engine_memory(cfg: "Config", system: "System", n_frames: int) -> int:
    """Predict the engine's peak memory consumption (bytes).

    Returns
    -------
    int
        Conservative upper bound: ``_vacvv`` + per-mol arrays + an FFT
        working-set multiplier on top of the ``_vacvv`` portion.

    Notes
    -----
    * The estimate intentionally over-counts by ~10–20 % to leave headroom
      for short-lived numpy intermediates.  See :func:`compute_n_batches`,
      which also adds a 10 % safety margin.
    * Excludes per-frame transport accumulators (~1 MB) and the trajectory
      file descriptor — both sub-1 GB for any realistic run.
    """
    if cfg.disable_xpt:
        # MEM-B short-circuit: _vacvv is not allocated, only per-mol arrays
        # for IR/Raman remain — and even those are bypassed.  Engine fits
        # in transport-only memory regardless of n_frames or n_atoms.
        return 0

    per_atom = system.natom * 3 * _F32                         # one frame's worth
    n_per_frame = _per_frame_per_atom_bytes(cfg, system, n_frames)

    # Steady-state portion of _vacvv (the bigger of the two factors)
    vactype = 5 if cfg.molecular else 1
    vacvv_bytes = vactype * n_frames * system.natom * 3 * _F32

    # FFT peak: when _compute_vac runs, _vacvv is already allocated AND a
    # complex64 rfft output + the per-(i,j) ACF of shape (N, n_part) live on
    # top.  Empirical multiplier of 2.5 covers it.
    fft_peak = int(vacvv_bytes * _FFT_WORKING_MULTIPLIER)

    return max(n_per_frame, fft_peak) + per_atom


def get_ram_budget(cfg: "Config") -> int:
    """Return the user-requested RAM budget in bytes.

    Resolution order:
    1. Explicit ``cfg.ram_budget_gb`` (> 0)  →  exactly that many bytes.
    2. Otherwise, ``cfg.ram_budget_fraction × available_system_RAM``.
       *Available* RAM is queried via ``psutil`` when present; otherwise
       falls back to ``os.sysconf("SC_PHYS_PAGES") × SC_PAGE_SIZE × 0.7``
       (rough Linux estimate).

    The returned value is clamped to at least 256 MB so a
    misconfigured budget never causes an infinite batching loop.
    """
    if cfg.ram_budget_gb > 0:
        return max(int(cfg.ram_budget_gb * 1024**3), 256 * 1024 * 1024)

    fraction = max(min(cfg.ram_budget_fraction, 0.95), 0.05)
    avail = _system_available_bytes()
    return max(int(avail * fraction), 256 * 1024 * 1024)


def _system_available_bytes() -> int:
    """Best-effort estimate of available system RAM in bytes."""
    try:
        import psutil  # type: ignore[import-not-found]
        return int(psutil.virtual_memory().available)
    except Exception:
        try:
            pages = os.sysconf("SC_PHYS_PAGES")
            page  = os.sysconf("SC_PAGE_SIZE")
            return int(pages * page * 0.7)
        except (ValueError, OSError):
            return 8 * 1024**3   # 8 GB worst-case fallback


def compute_n_batches(cost: int, budget: int, single_pass_only: bool) -> int:
    """How many batches the engine should split into.

    Parameters
    ----------
    cost
        Estimated peak memory the engine would use in single-pass mode.
    budget
        Target RAM budget in bytes.
    single_pass_only
        When True, raise ``MemoryError`` instead of returning ``> 1``.

    Returns
    -------
    int
        Number of batches in 1..N.  ``1`` means single-pass (no batching).
    """
    if cost <= budget:
        return 1
    if single_pass_only:
        raise MemoryError(
            f"Engine memory estimate {cost / 1e9:.2f} GB exceeds RAM "
            f"budget {budget / 1e9:.2f} GB and single_pass_only=True. "
            f"Either raise [memory] ram_budget_gb, lower [memory] "
            f"single_pass_only, reduce the frame count via [frames] step, "
            f"or set [thermodynamics] disable_xpt=1 for transport/mechanics-"
            f"only runs."
        )
    # 10 % safety margin so we don't sit just under the budget; this also
    return int(math.ceil(cost * 1.10 / budget))


def partition_atoms(system: "System", cfg: "Config", n_batches: int) -> list[Chunk]:
    """Split a system into ``n_batches`` chunks for batched accumulation.

    For *molecular* systems the unit is the molecule: each chunk's atoms are
    exactly the union of its molecules' atoms, so
    ``decompose_velocities_batch`` always sees every atom of a molecule
    (required for the per-mol angular / dipole velocity arrays to be
    consistent).  For *non-molecular* systems atoms are partitioned
    directly.

    Splitting is contiguous and roughly equal-sized.  Returns ``n_batches``
    chunks even when ``len(units) < n_batches`` (extra chunks are empty;
    the engine should skip empty chunks).
    """
    if n_batches <= 1:
        return [Chunk(
            atom_ids=np.arange(system.natom, dtype=np.intp),
            mol_ids=np.arange(system.nmol, dtype=np.intp) if cfg.molecular else np.empty(0, dtype=np.intp),
        )]

    if cfg.molecular and system.nmol > 0:
        units = np.arange(system.nmol, dtype=np.intp)
        chunks = []
        # Roughly equal split; np.array_split handles non-divisible counts.
        for mol_chunk in np.array_split(units, n_batches):
            if len(mol_chunk) == 0:
                chunks.append(Chunk(
                    atom_ids=np.empty(0, dtype=np.intp),
                    mol_ids=np.empty(0, dtype=np.intp),
                ))
                continue
            atom_ids: list[int] = []
            for mid in mol_chunk:
                atom_ids.extend(system.mols[mid].atom_ids)
            chunks.append(Chunk(
                atom_ids=np.asarray(sorted(atom_ids), dtype=np.intp),
                mol_ids=mol_chunk,
            ))
        return chunks

    units = np.arange(system.natom, dtype=np.intp)
    chunks = [Chunk(
                atom_ids=np.asarray(c, dtype=np.intp),
                mol_ids=np.empty(0, dtype=np.intp),
              )
              for c in np.array_split(units, n_batches)]
    return chunks


def log_budget_decision(cost: int, budget: int, n_batches: int) -> None:
    """Emit a single info-level summary of the batching decision."""
    log.info(
        "Memory budget: estimated peak %.2f GB, budget %.2f GB → "
        "%s pass(es)",
        cost / 1e9, budget / 1e9, n_batches,
    )
