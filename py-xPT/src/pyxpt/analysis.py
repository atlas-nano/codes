# -*- coding: utf-8 -*-
"""
Top-level dispatcher and command-line entry point for py-xPT.

The [thermodynamics] config section drives the 2PT/3PT entropy analysis.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

from .constants import COPYRIGHT
from .config import Config, load, print_summary
from .io.trajectory import System
from .thermo.engine import xPTEngine, xPTResult

log = logging.getLogger(__name__)


def run(control_file: str | Path,
        log_level: int = logging.INFO,
        use_gpu: bool = False) -> xPTResult:
    """
    Execute a py-xPT analysis from a control file.

    Sections present in the file determine what is computed.
    """
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    t0 = time.perf_counter()
    print(COPYRIGHT)

    control_file = Path(control_file)
    cfg = _load_config(control_file)
    cfg.use_gpu = use_gpu
    print_summary(cfg)

    log.info("Building system from topology / trajectory ...")
    system = System.from_config(cfg)
    nframes = system.total_frames(cfg)

    # Phase MEM-R1: decide whether to run in single-pass mode or split into
    # multiple atom batches with re-reads of the trajectory.  See
    # ``pyxpt.memory_budget`` for cost estimation and partitioning logic.
    from pyxpt.memory_budget import (
        estimate_engine_memory, get_ram_budget, compute_n_batches,
        partition_atoms, log_budget_decision,
    )
    cost   = estimate_engine_memory(cfg, system, nframes)
    budget = get_ram_budget(cfg)
    n_batches = compute_n_batches(cost, budget, cfg.single_pass_only)
    log_budget_decision(cost, budget, n_batches)

    engine = xPTEngine(cfg, system)

    if n_batches == 1:
        engine._setup(nframes)
        log.info("Iterating trajectory frames ...")
        engine.accumulate(system.iter_frames(cfg))
        log.info("Running xPT computation ...")
        result = engine.compute()
    else:
        result = _run_batched(cfg, system, engine, nframes, n_batches,
                              partition_atoms(system, cfg, n_batches))

    log.info("Writing output files ...")
    engine.write(result)

    elapsed = time.perf_counter() - t0
    log.info("Total wall time: %.1f s", elapsed)
    return result


def _run_batched(cfg: Config, system: System, engine,
                 nframes: int, n_batches: int, chunks) -> xPTResult:
    """Multi-pass execution under a RAM budget (Phase MEM-R1).

    Re-reads the trajectory once per atom batch.  Per-frame scalars and
    per-molecule arrays are populated on the first pass only; ``_vacvv``
    is allocated to the batch-atom count on every pass and its FFT
    contribution is summed into ``vac_sum_total``.  Final
    ``_compute_postvac`` then runs against the accumulated ``vac_sum``
    plus the (full-size) per-frame and per-molecule state from pass 0.
    """
    import numpy as np
    log.info("Memory-budget batched execution: %d passes over the trajectory",
             n_batches)

    vac_sum_total = None
    nonempty_seen = 0

    for b, chunk in enumerate(chunks):
        if len(chunk.atom_ids) == 0:
            continue
        is_first = (nonempty_seen == 0)
        nonempty_seen += 1

        log.info("── Batch %d/%d ──  %d atoms%s%s",
                 b + 1, n_batches, len(chunk.atom_ids),
                 f", {len(chunk.mol_ids)} molecules" if cfg.molecular else "",
                 "  (filling per-frame + per-mol state)" if is_first
                 else "  (reusing pass-0 per-frame state)")

        engine.set_atom_mask(
            chunk.atom_ids,
            skip_full_arrays=not is_first,
            skip_per_mol_in_compute_vac=not is_first,
        )
        engine._setup(nframes)
        engine.accumulate(system.iter_frames(cfg))
        vac_sum_b = engine._compute_vac()

        if vac_sum_total is None:
            vac_sum_total = vac_sum_b.copy()
        else:
            vac_sum_total += vac_sum_b

        engine.free_batch_arrays()

    if vac_sum_total is None:
        raise RuntimeError("Memory-budget batching produced no non-empty "
                           "atom batches; check partition_atoms output.")

    # Clear the mask before the final post-vac pass so any code that inspects
    # _atom_mask sees the canonical full-system state.  ``_compute_postvac``
    # does not touch ``_vacvv`` directly (it consumes ``vac_sum`` plus the
    # full-size per-mol arrays populated on batch 0).
    engine.set_atom_mask(None)
    log.info("Running xPT computation on accumulated vac_sum ...")
    result = xPTResult(system.ngrp, engine._vactype,
                            engine._nused, engine._vacmaxf)
    return engine._compute_postvac(result, vac_sum_total)



def _load_config(path: Path) -> Config:
    return load(path)


def main() -> None:
    """Command-line entry point: ``pyxpt control.ini``"""
    import argparse
    parser = argparse.ArgumentParser(
        prog="pyxpt",
        description="py-xPT — Two- and Three-Phase Thermodynamics (2PT/3PT) entropy from MD trajectories",
    )
    parser.add_argument("control", help="INI-format control file")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Enable DEBUG logging")
    parser.add_argument("--gpu", action="store_true",
                        help="Enable GPU acceleration (requires CuPy)")
    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    try:
        run(args.control, log_level=level, use_gpu=args.gpu)
    except (FileNotFoundError, ValueError) as exc:
        log.error("%s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
