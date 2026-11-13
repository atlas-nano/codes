"""Runtime and strain convergence helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..config import ConvergenceSpec, DMAProductionSpec
from .dma import fit_dma_dataframe, parse_raw_name, read_raw_table


def mark_converged(summary: pd.DataFrame, spec: ConvergenceSpec | None = None) -> pd.DataFrame:
    spec = spec or ConvergenceSpec()
    df = summary.copy()
    h2 = df["harm2_over_harm1"].fillna(0.0)
    h3 = df["harm3_over_harm1"].fillna(0.0)
    drift = df["drift_fraction_per_cycle"].fillna(0.0)
    df["passes_dma2_thresholds"] = (
        (df["r2"] >= spec.min_r2)
        & (df["snr_first_harmonic"] >= spec.min_snr)
        & (h2 <= spec.max_harmonic_ratio)
        & (h3 <= spec.max_harmonic_ratio)
        & (drift <= spec.max_drift_fraction_per_cycle)
    )
    return df


def recommend_strains(summary: pd.DataFrame, spec: ConvergenceSpec | None = None) -> pd.DataFrame:
    spec = spec or ConvergenceSpec()
    df = mark_converged(summary, spec)
    lo, hi = spec.preferred_strain_window
    df["in_preferred_window"] = df["strain"].between(lo, hi)
    rows = []
    for (mode, signal, parser), group in df.groupby(["mode", "signal", "parser"]):
        candidates = group[group["passes_dma2_thresholds"] & group["in_preferred_window"]]
        if candidates.empty:
            candidates = group[group["passes_dma2_thresholds"]]
        if candidates.empty:
            candidates = group
        best = candidates.sort_values(
            ["in_preferred_window", "snr_first_harmonic", "r2"],
            ascending=[False, False, False],
        ).iloc[0]
        rows.append(best.to_dict() | {"recommendation": "preferred_strain"})
    return pd.DataFrame(rows)


def compare_parser_stability(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metrics = ["storage_modulus_MPa", "loss_modulus_MPa", "tan_delta"]
    for (mode, signal, parser), group in summary.groupby(["mode", "signal", "parser"]):
        row = {"mode": mode, "signal": signal, "parser": parser, "n": len(group)}
        for metric in metrics:
            vals = group[metric].replace([np.inf, -np.inf], np.nan).dropna()
            row[f"{metric}_mean"] = float(vals.mean()) if len(vals) else np.nan
            row[f"{metric}_cv"] = float(vals.std(ddof=1) / vals.mean()) if len(vals) > 1 and vals.mean() else np.nan
        row["median_r2"] = float(group["r2"].median())
        row["median_snr"] = float(group["snr_first_harmonic"].median())
        rows.append(row)
    return pd.DataFrame(rows)


def cycle_count_convergence(
    raw_file: str,
    frequency_ghz: float,
    signal: str,
    cycle_counts: list[int] | None = None,
    parser: str = "lockin",
    tolerance_r2: float = 0.01,
    tolerance_modulus: float = 0.05,
    consecutive_windows: int = 1,
    minimum_cycles: int = 2,
    discard_initial_cycles: int = 0,
    max_harmonic_ratio: float = 0.20,
    max_drift_fraction_per_cycle: float = 0.05,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate fits for 1..N cycles from a long trajectory.

    This follows the DMAx paper's cycle-count convergence idea: run the
    strain-selected 10-cycle trajectory once, fit segments containing increasing
    numbers of usable cycles, and choose the first cycle count whose moduli are
    close enough to the full-trajectory reference.
    """

    mode, strain = parse_raw_name(raw_file)
    full = read_raw_table(raw_file)
    if "t_ps" in full:
        t_fs = full["t_ps"].to_numpy(float) * 1000.0
    else:
        t_fs = full["step"].to_numpy(float)
    period_fs = 1_000_000.0 / frequency_ghz
    t0 = float(np.nanmin(t_fs)) + discard_initial_cycles * period_fs
    max_available = int(np.floor((float(np.nanmax(t_fs)) - t0) / period_fs))
    if cycle_counts is None:
        cycle_counts = list(range(1, max_available + 1))
    rows = []
    for ncycles in cycle_counts:
        if ncycles > max_available:
            continue
        keep = (t_fs >= t0) & (t_fs <= t0 + ncycles * period_fs + 1.0e-9)
        seg = full.loc[keep].copy()
        if len(seg) < 8:
            continue
        fit = fit_dma_dataframe(
            seg,
            mode=mode,
            signal=signal,
            strain=strain,
            frequency_ghz=frequency_ghz,
            discard_cycles=0,
            parser=parser,
        )
        row = fit.as_dict()
        row["cycles"] = int(ncycles)
        row["usable_cycles"] = int(ncycles)
        row["discarded_initial_cycles"] = int(discard_initial_cycles)
        row["total_cycles_including_discard"] = int(ncycles + discard_initial_cycles)
        rows.append(row)
    table = pd.DataFrame(rows).sort_values("cycles")
    if table.empty:
        return table, pd.DataFrame()
    table["delta_r2"] = table["r2"].diff().abs()
    table["rel_delta_storage"] = table["storage_modulus_MPa"].pct_change().abs()
    table["rel_delta_loss"] = table["loss_modulus_MPa"].pct_change().abs()
    table["rel_delta_tan_delta"] = table["tan_delta"].pct_change().abs()
    reference = table.iloc[-1]
    for metric in ["storage_modulus_MPa", "loss_modulus_MPa", "tan_delta"]:
        ref_value = float(reference[metric])
        err_col = f"{metric}_rel_error_to_reference"
        table[err_col] = np.nan if ref_value == 0 else (table[metric] - ref_value).abs() / abs(ref_value)
    table["within_reference_tolerance"] = (
        table["storage_modulus_MPa_rel_error_to_reference"].le(tolerance_modulus)
        & table["loss_modulus_MPa_rel_error_to_reference"].le(tolerance_modulus)
        & table["tan_delta_rel_error_to_reference"].le(tolerance_modulus)
    )
    table["diagnostics_clean"] = (
        table["harm2_over_harm1"].fillna(0.0).le(max_harmonic_ratio)
        & table["harm3_over_harm1"].fillna(0.0).le(max_harmonic_ratio)
        & table["drift_fraction_per_cycle"].fillna(0.0).le(max_drift_fraction_per_cycle)
    )
    plateau = (
        (table["cycles"] >= minimum_cycles)
        & (table["delta_r2"] <= tolerance_r2)
        & (table["rel_delta_storage"] <= tolerance_modulus)
        & (table["rel_delta_loss"] <= tolerance_modulus)
    )
    if consecutive_windows > 1:
        plateau = plateau.rolling(consecutive_windows, min_periods=consecutive_windows).sum() >= consecutive_windows
    candidates = table[
        (table["cycles"] >= minimum_cycles)
        & table["within_reference_tolerance"]
        & table["diagnostics_clean"]
    ]
    if not candidates.empty:
        selected = candidates.iloc[[0]].copy()
        selected["recommendation"] = "first_within_reference_tolerance"
    elif not table[plateau].empty:
        selected = table[plateau].iloc[[0]].copy()
        selected["recommendation"] = "first_plateau_cycle_count"
    else:
        selected = table.sort_values(["r2", "snr_first_harmonic"], ascending=[False, False]).iloc[[0]].copy()
        selected["recommendation"] = "best_available_cycle_count"
    return table, selected


def choose_cycle_convergence_targets(
    recommendations: pd.DataFrame,
    preferred_mode: str = "shear_sllod_remapv",
    preferred_signal: str = "sxy_vir_MPa",
    parser: str = "lockin",
) -> pd.DataFrame:
    """Select rows to use for cycle-count convergence."""

    df = recommendations.copy()
    if "parser" in df:
        df = df[df["parser"] == parser]
    preferred = df[(df["mode"] == preferred_mode) & (df["signal"] == preferred_signal)]
    if not preferred.empty:
        return preferred
    return df.sort_values(["snr_first_harmonic", "r2"], ascending=[False, False]).head(1)


def write_cycle_convergence_plots(
    table: pd.DataFrame,
    selected: pd.DataFrame,
    output_dir: str | Path,
) -> list[Path]:
    """Write cycle-count convergence plots for the selected strain."""

    output_dir = Path(output_dir)
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    if table.empty:
        return paths
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return paths

    mode = str(table["mode"].iloc[0])
    signal = str(table["signal"].iloc[0])
    parser = str(table["parser"].iloc[0])
    strain = float(table["strain"].iloc[0])
    selected_cycles = int(selected["cycles"].iloc[0]) if not selected.empty else None
    ref = table.iloc[-1]

    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.4), sharex=True)
    ax = axes[0, 0]
    ax.plot(table["cycles"], table["storage_modulus_MPa"], marker="o", label="storage")
    ax.plot(table["cycles"], table["loss_modulus_MPa"], marker="s", label="loss")
    ax.axhline(ref["storage_modulus_MPa"], color="#0f766e", linestyle=":", linewidth=1.0)
    ax.axhline(ref["loss_modulus_MPa"], color="#7c2d12", linestyle=":", linewidth=1.0)
    ax.set_ylabel("Modulus (MPa)")
    ax.legend(frameon=False)

    ax = axes[0, 1]
    ax.plot(table["cycles"], table["tan_delta"], marker="o", color="#9333ea")
    ax.axhline(ref["tan_delta"], color="#9333ea", linestyle=":", linewidth=1.0)
    ax.set_ylabel("tan(delta)")

    ax = axes[1, 0]
    ax.plot(
        table["cycles"],
        table["storage_modulus_MPa_rel_error_to_reference"],
        marker="o",
        label="storage",
    )
    ax.plot(
        table["cycles"],
        table["loss_modulus_MPa_rel_error_to_reference"],
        marker="s",
        label="loss",
    )
    ax.plot(table["cycles"], table["tan_delta_rel_error_to_reference"], marker="^", label="tan(delta)")
    ax.axhline(0.05, color="#64748b", linestyle="--", linewidth=0.9)
    ax.set_ylabel("Relative error vs full usable trajectory")
    ax.set_xlabel("Usable cycles fit")
    ax.legend(frameon=False)

    ax = axes[1, 1]
    ax.plot(table["cycles"], table["snr_first_harmonic"], marker="o", label="SNR")
    ax.plot(table["cycles"], table["harm2_over_harm1"], marker="s", label="2w/1w")
    ax.plot(table["cycles"], table["harm3_over_harm1"], marker="^", label="3w/1w")
    ax.plot(table["cycles"], table["drift_fraction_per_cycle"], marker="d", label="drift/cycle")
    ax.axhline(0.20, color="#64748b", linestyle="--", linewidth=0.9)
    ax.axhline(0.05, color="#94a3b8", linestyle=":", linewidth=0.9)
    ax.set_ylabel("Diagnostic ratio")
    ax.set_xlabel("Usable cycles fit")
    ax.legend(frameon=False, fontsize=8)

    for ax in axes.ravel():
        ax.grid(True, alpha=0.25)
        if selected_cycles is not None:
            ax.axvline(selected_cycles, color="#111827", linestyle="-.", linewidth=0.9)
    fig.suptitle(f"Cycle convergence: {mode}, {signal}, strain={strain:g}, parser={parser}")
    fig.tight_layout()
    token = f"{strain:.3f}".replace(".", "p")
    path = plot_dir / f"cycle_convergence__{mode}__{signal}__strain_{token}__{parser}.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    paths.append(path)
    return paths


def write_cycle_convergence_report(
    raw_file: str | Path,
    output_dir: str | Path,
    frequency_ghz: float,
    signal: str,
    convergence: ConvergenceSpec | None = None,
    production: DMAProductionSpec | None = None,
    parser: str = "lockin",
    discard_initial_cycles: int | None = None,
) -> dict[str, Path]:
    """Analyze cycle-count convergence and write tables, plots, and report."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    convergence = convergence or ConvergenceSpec()
    production = production or DMAProductionSpec()
    if discard_initial_cycles is None:
        discard_initial_cycles = production.discard_cycles
    table, selected = cycle_count_convergence(
        str(raw_file),
        frequency_ghz=frequency_ghz,
        signal=signal,
        cycle_counts=convergence.cycle_counts,
        parser=parser,
        tolerance_r2=convergence.cycle_r2_tolerance,
        tolerance_modulus=convergence.cycle_modulus_tolerance,
        consecutive_windows=convergence.cycle_consecutive_windows,
        minimum_cycles=convergence.minimum_cycles,
        discard_initial_cycles=discard_initial_cycles,
        max_harmonic_ratio=convergence.max_harmonic_ratio,
        max_drift_fraction_per_cycle=convergence.max_drift_fraction_per_cycle,
    )
    table_path = output_dir / "dma_cycle_convergence.csv"
    selected_path = output_dir / "dma_cycle_recommendations.csv"
    table.to_csv(table_path, index=False)
    selected.to_csv(selected_path, index=False)
    plot_paths = write_cycle_convergence_plots(table, selected, output_dir)

    report_path = output_dir / "dma_cycle_convergence_report.md"
    lines = [
        "# DMAx2 Cycle-Count Convergence",
        "",
        f"Raw trajectory: `{raw_file}`",
        f"Signal: `{signal}`",
        f"Parser: `{parser}`",
        f"Frequency: {frequency_ghz:g} GHz",
        f"Discarded startup cycles: {discard_initial_cycles}",
        "",
    ]
    if table.empty or selected.empty:
        lines.extend(["No cycle-count recommendation could be made from the available trajectory.", ""])
    else:
        row = selected.iloc[0]
        total_cycles = int(row["total_cycles_including_discard"])
        usable_cycles = int(row["usable_cycles"])
        savings = 1.0 - total_cycles / production.cycles if production.cycles else np.nan
        lines.extend(
            [
                f"Recommended usable analysis cycles: `{usable_cycles}`",
                f"Recommended total production cycles: `{total_cycles}`",
                f"Selection reason: {row['recommendation']}",
                f"Estimated cycle-count savings vs current {production.cycles:g}-cycle production: {100.0 * savings:.1f}%",
                "",
                "Recommended-cycle diagnostics:",
                "",
                f"- Storage modulus: {float(row['storage_modulus_MPa']):.6g} MPa",
                f"- Loss modulus: {float(row['loss_modulus_MPa']):.6g} MPa",
                f"- tan(delta): {float(row['tan_delta']):.6g}",
                f"- Storage relative error vs full usable trajectory: {float(row['storage_modulus_MPa_rel_error_to_reference']):.6g}",
                f"- Loss relative error vs full usable trajectory: {float(row['loss_modulus_MPa_rel_error_to_reference']):.6g}",
                f"- tan(delta) relative error vs full usable trajectory: {float(row['tan_delta_rel_error_to_reference']):.6g}",
                f"- R2: {float(row['r2']):.6g}",
                f"- First-harmonic SNR: {float(row['snr_first_harmonic']):.6g}",
                f"- 2w/1w: {float(row['harm2_over_harm1']):.6g}",
                f"- 3w/1w: {float(row['harm3_over_harm1']):.6g}",
                f"- Drift fraction per cycle: {float(row['drift_fraction_per_cycle']):.6g}",
                "",
                "Interpretation:",
                "",
                "The cycle-count recommendation uses the full retained trajectory as the local reference.",
                "It chooses the shortest cumulative segment whose storage modulus, loss modulus, and loss tangent are all within the configured tolerance of that reference while keeping nonlinear harmonic and drift diagnostics below the configured limits.",
                "This keeps the production protocol tied to accuracy in the quantities used downstream, instead of optimizing only the global sinusoidal R2.",
                "",
            ]
        )
    if plot_paths:
        lines.append("Plots:")
        lines.append("")
        for path in plot_paths:
            lines.append(f"- `{path.relative_to(output_dir)}`")
        lines.append("")
    report_path.write_text("\n".join(lines))
    return {
        "table": table_path,
        "recommendations": selected_path,
        "report": report_path,
        **{f"plot_{idx}": path for idx, path in enumerate(plot_paths, start=1)},
    }


def select_optimal_strain(
    summary: pd.DataFrame,
    spec: ConvergenceSpec | None = None,
    preferred_mode: str = "shear_sllod_remapv",
    preferred_signal: str = "sxy_vir_MPa",
    parser: str = "lockin",
) -> pd.DataFrame:
    """Select the strain amplitude for the next workflow stage.

    DMA trajectories can have very low global R2 when the instantaneous stress
    noise is much larger than the first harmonic. In that common case, the
    fallback selection prefers a strain in the configured window with small
    nonlinear harmonics and drift, then maximizes first-harmonic SNR.
    """

    spec = spec or ConvergenceSpec()
    df = mark_converged(summary, spec)
    primary = df[
        (df["mode"] == preferred_mode)
        & (df["signal"] == preferred_signal)
        & (df["parser"] == parser)
    ].copy()
    if primary.empty:
        return pd.DataFrame()

    lo, hi = spec.preferred_strain_window
    primary["in_preferred_window"] = primary["strain"].between(lo, hi)
    primary["harmonic_clean"] = (
        primary["harm2_over_harm1"].fillna(0.0).le(spec.max_harmonic_ratio)
        & primary["harm3_over_harm1"].fillna(0.0).le(spec.max_harmonic_ratio)
    )
    primary["drift_clean"] = primary["drift_fraction_per_cycle"].fillna(0.0).le(
        spec.max_drift_fraction_per_cycle
    )
    primary["analysis_score"] = (
        primary["snr_first_harmonic"].fillna(0.0)
        + 0.20 * primary["r2"].fillna(0.0)
        - 0.20 * primary["harm2_over_harm1"].fillna(0.0)
        - 0.20 * primary["harm3_over_harm1"].fillna(0.0)
        - 0.10 * primary["drift_fraction_per_cycle"].fillna(0.0)
    )

    tiers = [
        ("passes all DMAx2 thresholds in preferred strain window", primary["passes_dma2_thresholds"] & primary["in_preferred_window"]),
        ("passes all DMAx2 thresholds", primary["passes_dma2_thresholds"]),
        ("harmonic-clean and drift-clean in preferred strain window", primary["harmonic_clean"] & primary["drift_clean"] & primary["in_preferred_window"]),
        ("harmonic-clean and drift-clean", primary["harmonic_clean"] & primary["drift_clean"]),
        ("best available in preferred strain window", primary["in_preferred_window"]),
        ("best available", pd.Series(True, index=primary.index)),
    ]
    for reason, mask in tiers:
        candidates = primary[mask]
        if not candidates.empty:
            selected = candidates.sort_values(
                ["analysis_score", "snr_first_harmonic", "r2"],
                ascending=[False, False, False],
            ).iloc[[0]].copy()
            selected["selection_reason"] = reason
            selected["preferred_mode"] = preferred_mode
            selected["preferred_signal"] = preferred_signal
            return selected
    return pd.DataFrame()


def write_strain_convergence_plots(
    summary: pd.DataFrame,
    output_dir: str | Path,
    preferred_mode: str = "shear_sllod_remapv",
    preferred_signal: str = "sxy_vir_MPa",
    parser: str = "lockin",
) -> list[Path]:
    """Write compact plots used to choose the strain amplitude."""

    output_dir = Path(output_dir)
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return paths

    primary = summary[
        (summary["mode"] == preferred_mode)
        & (summary["signal"] == preferred_signal)
        & (summary["parser"] == parser)
    ].sort_values("strain")
    if primary.empty:
        return paths

    fig, axes = plt.subplots(2, 2, figsize=(8.4, 6.2), sharex=True)
    ax = axes[0, 0]
    ax.plot(primary["strain"], primary["storage_modulus_MPa"], marker="o", label="storage")
    ax.plot(primary["strain"], primary["loss_modulus_MPa"], marker="s", label="loss")
    ax.set_ylabel("Modulus (MPa)")
    ax.legend()

    ax = axes[0, 1]
    ax.plot(primary["strain"], primary["tan_delta"], marker="o", color="#7c2d12")
    ax.set_ylabel("tan(delta)")

    ax = axes[1, 0]
    ax.plot(primary["strain"], primary["snr_first_harmonic"], marker="o", label="SNR")
    ax.plot(primary["strain"], primary["r2"], marker="s", label="R2")
    ax.set_ylabel("Fit quality")
    ax.set_xlabel("Strain amplitude")
    ax.legend()

    ax = axes[1, 1]
    ax.plot(primary["strain"], primary["harm2_over_harm1"], marker="o", label="2w/1w")
    ax.plot(primary["strain"], primary["harm3_over_harm1"], marker="s", label="3w/1w")
    ax.plot(primary["strain"], primary["drift_fraction_per_cycle"], marker="^", label="drift/cycle")
    ax.axhline(0.20, color="#64748b", linestyle="--", linewidth=0.8)
    ax.axhline(0.05, color="#94a3b8", linestyle=":", linewidth=0.8)
    ax.set_ylabel("Diagnostic ratio")
    ax.set_xlabel("Strain amplitude")
    ax.legend()

    for ax in axes.ravel():
        ax.grid(True, alpha=0.25)
    fig.suptitle(f"{preferred_mode}: {preferred_signal} ({parser})")
    fig.tight_layout()
    path = plot_dir / f"strain_convergence__{preferred_mode}__{preferred_signal}__{parser}.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    paths.append(path)
    return paths


def write_strain_convergence_report(
    summary: pd.DataFrame,
    output_dir: str | Path,
    convergence: ConvergenceSpec | None = None,
    production: DMAProductionSpec | None = None,
) -> dict[str, Path]:
    """Write selection CSV, compact plots, and a Markdown report."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    convergence = convergence or ConvergenceSpec()
    production = production or DMAProductionSpec()
    mode = production.modes[0] if production.modes else "shear_sllod_remapv"
    signal = production.recommended_signal
    parser = "lockin"
    selection = select_optimal_strain(summary, convergence, mode, signal, parser)
    selection_path = output_dir / "dma_strain_selection.csv"
    selection.to_csv(selection_path, index=False)
    plot_paths = write_strain_convergence_plots(summary, output_dir, mode, signal, parser)

    primary = summary[
        (summary["mode"] == mode)
        & (summary["signal"] == signal)
        & (summary["parser"] == parser)
    ].sort_values("strain")
    observed = sorted(float(x) for x in primary["strain"].unique())
    expected = sorted(float(x) for x in production.strains)
    missing = [x for x in expected if not np.isclose(x, observed).any()]

    report_path = output_dir / "dma_strain_convergence_report.md"
    lines = [
        "# DMAx2 Strain-Size Convergence",
        "",
        f"Primary mode: `{mode}`",
        f"Primary signal: `{signal}`",
        f"Parser: `{parser}`",
        "",
        f"Observed strains: {', '.join(f'{x:g}' for x in observed) if observed else 'none'}",
        f"Missing configured strains: {', '.join(f'{x:g}' for x in missing) if missing else 'none'}",
        "",
    ]
    if selection.empty:
        lines.extend(["No optimal strain could be selected from the available summary.", ""])
    else:
        row = selection.iloc[0]
        lines.extend(
            [
                f"Recommended strain amplitude: `{float(row['strain']):g}`",
                f"Selection reason: {row['selection_reason']}",
                "",
                "Recommended-strain diagnostics:",
                "",
                f"- Storage modulus: {float(row['storage_modulus_MPa']):.6g} MPa",
                f"- Loss modulus: {float(row['loss_modulus_MPa']):.6g} MPa",
                f"- tan(delta): {float(row['tan_delta']):.6g}",
                f"- R2: {float(row['r2']):.6g}",
                f"- First-harmonic SNR: {float(row['snr_first_harmonic']):.6g}",
                f"- 2w/1w: {float(row['harm2_over_harm1']):.6g}",
                f"- 3w/1w: {float(row['harm3_over_harm1']):.6g}",
                f"- Drift fraction per cycle: {float(row['drift_fraction_per_cycle']):.6g}",
                f"- Status: `{row['status']}`",
                "",
            ]
        )
    lines.extend(
        [
            "Interpretation:",
            "",
            "DMAx2 prioritizes a harmonic-clean first-harmonic response in the configured strain window.",
            "The global R2 can remain low for noisy instantaneous stress traces even when the driven harmonic is measurable.",
            "For that reason the selection checks nonlinear harmonic content and drift before falling back to first-harmonic SNR.",
            "",
            "Difference from legacy DMAx:",
            "",
            "The legacy workflow used axial deformation and pressure-like scalar signals, then judged strain convergence mostly from modulus stability and fit quality.",
            "DMAx2 uses virial-only shear stress as the primary estimator, lock-in extraction at the imposed frequency, and explicit 2w/3w/drift diagnostics.",
            "Legacy axial-pressure results are still parsed for comparison, but they are not used as the default strain-selection signal.",
            "",
        ]
    )
    if plot_paths:
        lines.append("Plots:")
        lines.append("")
        for path in plot_paths:
            lines.append(f"- `{path.relative_to(output_dir)}`")
        lines.append("")
    report_path.write_text("\n".join(lines))
    return {
        "selection": selection_path,
        "report": report_path,
        **{f"plot_{idx}": path for idx, path in enumerate(plot_paths, start=1)},
    }
