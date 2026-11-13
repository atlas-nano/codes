"""Random-seed uncertainty analysis for DMAx2 production runs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


METRICS = [
    "storage_modulus_MPa",
    "loss_modulus_MPa",
    "tan_delta",
    "r2",
    "snr_first_harmonic",
    "harm2_over_harm1",
    "harm3_over_harm1",
    "drift_fraction_per_cycle",
]


def _ensure_replicate_columns(summary: pd.DataFrame) -> pd.DataFrame:
    data = summary.copy()
    if "replicate" not in data:
        data["replicate"] = np.arange(1, len(data) + 1)
    if "velocity_seed" not in data:
        data["velocity_seed"] = np.nan
    if "temperature_K" not in data:
        data["temperature_K"] = np.nan
    if "frequency_ghz" not in data:
        data["frequency_ghz"] = np.nan
    return data


def summarize_error_statistics(summary: pd.DataFrame) -> pd.DataFrame:
    """Summarize random-velocity replicate spread by parser/mode/signal."""

    data = _ensure_replicate_columns(summary)
    group_cols = ["mode", "signal", "parser", "strain", "frequency_ghz", "temperature_K"]
    rows: list[dict] = []
    for key, group in data.groupby(group_cols, dropna=False):
        base = dict(zip(group_cols, key, strict=True))
        base["n_replicates"] = int(group["replicate"].nunique())
        for metric in METRICS:
            if metric not in group:
                continue
            values = group[metric].to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            if len(values) == 0:
                mean = std = sem = cv = np.nan
            else:
                mean = float(np.mean(values))
                std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
                sem = float(std / np.sqrt(len(values))) if len(values) > 0 else np.nan
                cv = float(abs(std / mean)) if mean != 0 else np.nan
            base[f"{metric}_mean"] = mean
            base[f"{metric}_std"] = std
            base[f"{metric}_sem"] = sem
            base[f"{metric}_cv"] = cv
        rows.append(base)
    return pd.DataFrame(rows)


def _preferred_subset(
    summary: pd.DataFrame,
    signal: str,
    parser: str,
    mode: str | None = None,
) -> pd.DataFrame:
    data = _ensure_replicate_columns(summary)
    keep = (data["signal"] == signal) & (data["parser"] == parser)
    if mode is not None:
        keep &= data["mode"] == mode
    subset = data[keep].copy()
    if subset.empty and mode is None:
        subset = data[(data["signal"] == signal)].copy()
    return subset


def write_error_analysis_plots(
    summary: pd.DataFrame,
    statistics: pd.DataFrame,
    output_dir: str | Path,
    signal: str = "sxy_vir_MPa",
    parser: str = "lockin",
    mode: str | None = None,
) -> list[Path]:
    """Write compact plots of replicate uncertainty."""

    output_dir = Path(output_dir)
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return paths

    subset = _preferred_subset(summary, signal, parser, mode=mode)
    if subset.empty:
        return paths
    if mode is None:
        mode = str(subset["mode"].iloc[0])
        subset = subset[subset["mode"] == mode].copy()

    seed_labels = []
    for _, row in subset.sort_values("replicate").iterrows():
        rep = int(row["replicate"])
        seed = row.get("velocity_seed", np.nan)
        seed_labels.append(f"{rep}\n{int(seed)}" if np.isfinite(seed) else str(rep))

    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.6), sharex=True)
    for ax, metric, ylabel, color in [
        (axes[0], "storage_modulus_MPa", "Storage modulus (MPa)", "#0f766e"),
        (axes[1], "loss_modulus_MPa", "Loss modulus (MPa)", "#7c2d12"),
        (axes[2], "tan_delta", "tan(delta)", "#4338ca"),
    ]:
        values = subset.sort_values("replicate")[metric].to_numpy(dtype=float)
        x = np.arange(len(values))
        ax.scatter(x, values, s=46, color=color, edgecolor="#111827", linewidth=0.4)
        mean = float(np.nanmean(values))
        std = float(np.nanstd(values, ddof=1)) if np.isfinite(values).sum() > 1 else 0.0
        ax.axhline(mean, color="#111827", linewidth=1.2, label="mean")
        ax.fill_between([-0.5, len(values) - 0.5], mean - std, mean + std, color=color, alpha=0.13, label="1 std")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
    axes[0].set_xticks(np.arange(len(seed_labels)))
    axes[0].set_xticklabels(seed_labels, fontsize=8)
    axes[0].set_xlabel("replicate / seed")
    axes[1].set_xlabel("replicate / seed")
    axes[2].set_xlabel("replicate / seed")
    axes[0].legend(frameon=False, loc="best")
    strain = float(subset["strain"].iloc[0])
    freq = float(subset["frequency_ghz"].iloc[0]) if "frequency_ghz" in subset else np.nan
    fig.suptitle(f"DMAx2 error analysis: {mode}, {signal}, {parser}, strain={strain:g}, f={freq:g} GHz")
    fig.tight_layout()
    path = plot_dir / f"error_analysis_replicates__{mode}__{signal}__{parser}.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    paths.append(path)

    preferred_stats = statistics[
        (statistics["mode"] == mode)
        & (statistics["signal"] == signal)
        & (statistics["parser"] == parser)
    ].copy()
    if not preferred_stats.empty:
        row = preferred_stats.iloc[0]
        labels = ["Storage", "Loss", "tan(delta)"]
        means = [
            row["storage_modulus_MPa_mean"],
            row["loss_modulus_MPa_mean"],
            row["tan_delta_mean"],
        ]
        sems = [
            row["storage_modulus_MPa_sem"],
            row["loss_modulus_MPa_sem"],
            row["tan_delta_sem"],
        ]
        fig, ax = plt.subplots(figsize=(5.8, 3.8))
        ax.bar(labels, means, yerr=sems, color=["#0f766e", "#7c2d12", "#4338ca"], alpha=0.82, capsize=4)
        ax.set_ylabel("Mean +/- SEM")
        ax.set_title("Random-seed uncertainty summary")
        ax.grid(True, axis="y", alpha=0.25)
        fig.tight_layout()
        path = plot_dir / f"error_analysis_mean_sem__{mode}__{signal}__{parser}.png"
        fig.savefig(path, dpi=220)
        plt.close(fig)
        paths.append(path)

    diagnostic_cols = {
        "r2",
        "snr_first_harmonic",
        "harm2_over_harm1",
        "harm3_over_harm1",
        "drift_fraction_per_cycle",
    }
    if diagnostic_cols.issubset(subset.columns):
        ordered = subset.sort_values("replicate")
        x = np.arange(len(ordered))
        fig, axes = plt.subplots(2, 2, figsize=(9.5, 6.4), sharex=True)

        axes[0, 0].scatter(x, ordered["r2"], color="#0f766e", edgecolor="#111827", linewidth=0.4)
        axes[0, 0].axhline(0.70, color="#991b1b", linestyle="--", linewidth=1.0, label="target")
        axes[0, 0].set_ylabel("R2")
        axes[0, 0].set_title("Fit variance")

        axes[0, 1].scatter(
            x,
            ordered["snr_first_harmonic"],
            color="#7c2d12",
            edgecolor="#111827",
            linewidth=0.4,
        )
        axes[0, 1].axhline(2.0, color="#991b1b", linestyle="--", linewidth=1.0, label="target")
        axes[0, 1].set_ylabel("SNR")
        axes[0, 1].set_title("First harmonic")

        axes[1, 0].plot(x, ordered["harm2_over_harm1"], marker="o", label="2 omega / 1 omega")
        axes[1, 0].plot(x, ordered["harm3_over_harm1"], marker="s", label="3 omega / 1 omega")
        axes[1, 0].axhline(0.20, color="#991b1b", linestyle="--", linewidth=1.0, label="target")
        axes[1, 0].set_ylabel("Harmonic ratio")
        axes[1, 0].legend(frameon=False, fontsize=8)

        axes[1, 1].scatter(
            x,
            ordered["drift_fraction_per_cycle"],
            color="#4338ca",
            edgecolor="#111827",
            linewidth=0.4,
        )
        axes[1, 1].axhline(0.05, color="#991b1b", linestyle="--", linewidth=1.0, label="target")
        axes[1, 1].set_ylabel("Drift / cycle")
        axes[1, 1].set_title("Baseline drift")

        for ax in axes.ravel():
            ax.grid(True, alpha=0.25)
            ax.set_xticks(x)
            ax.set_xticklabels(seed_labels, fontsize=8)
            ax.set_xlabel("replicate / seed")
        fig.suptitle(f"DMAx2 fit diagnostics: {mode}, {signal}, {parser}")
        fig.tight_layout()
        path = plot_dir / f"error_analysis_diagnostics__{mode}__{signal}__{parser}.png"
        fig.savefig(path, dpi=220)
        plt.close(fig)
        paths.append(path)
    return paths


def write_error_analysis_report(
    summary_path: str | Path,
    output_dir: str | Path,
    signal: str = "sxy_vir_MPa",
    parser: str = "lockin",
    mode: str | None = None,
) -> dict[str, Path]:
    """Write random-seed uncertainty tables, plots, and a Markdown report."""

    summary_path = Path(summary_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = _ensure_replicate_columns(pd.read_csv(summary_path))
    replicates_path = output_dir / "dma_error_replicates.csv"
    summary.to_csv(replicates_path, index=False)

    statistics = summarize_error_statistics(summary)
    statistics_path = output_dir / "dma_error_statistics.csv"
    statistics.to_csv(statistics_path, index=False)
    plot_paths = write_error_analysis_plots(summary, statistics, output_dir, signal=signal, parser=parser, mode=mode)

    subset = _preferred_subset(summary, signal, parser, mode=mode)
    if mode is None and not subset.empty:
        mode = str(subset["mode"].iloc[0])
    preferred = statistics[
        (statistics["signal"] == signal)
        & (statistics["parser"] == parser)
        & ((statistics["mode"] == mode) if mode is not None else True)
    ].copy()

    report_path = output_dir / "dma_error_analysis_report.md"
    lines = [
        "# DMAx2 Error Analysis",
        "",
        "This stage repeats the converged DMA production from the same equilibrated structure with different initial velocity seeds.",
        "It follows the original DMAx paper's random-initial-condition uncertainty idea, while preserving DMAx2's lock-in parser diagnostics and legacy parser results for comparison.",
        "",
        f"Input summary table: `{summary_path}`",
        f"Replicate table: `{replicates_path.name}`",
        f"Statistics table: `{statistics_path.name}`",
        "",
    ]
    if preferred.empty:
        lines.extend(["No preferred signal/parser rows were found for the requested report settings.", ""])
    else:
        row = preferred.iloc[0]
        lines.extend(
            [
                "## Preferred Result",
                "",
                f"Mode: `{row['mode']}`",
                f"Signal: `{row['signal']}`",
                f"Parser: `{row['parser']}`",
                f"Strain amplitude: `{float(row['strain']):g}`",
                f"Frequency: `{float(row['frequency_ghz']):g} GHz`",
                f"Replicates: `{int(row['n_replicates'])}`",
                "",
                "| Quantity | Mean | Std | SEM | CV |",
                "| --- | ---: | ---: | ---: | ---: |",
                (
                    f"| Storage modulus (MPa) | {row['storage_modulus_MPa_mean']:.6g} | "
                    f"{row['storage_modulus_MPa_std']:.6g} | {row['storage_modulus_MPa_sem']:.6g} | "
                    f"{row['storage_modulus_MPa_cv']:.6g} |"
                ),
                (
                    f"| Loss modulus (MPa) | {row['loss_modulus_MPa_mean']:.6g} | "
                    f"{row['loss_modulus_MPa_std']:.6g} | {row['loss_modulus_MPa_sem']:.6g} | "
                    f"{row['loss_modulus_MPa_cv']:.6g} |"
                ),
                (
                    f"| tan(delta) | {row['tan_delta_mean']:.6g} | "
                    f"{row['tan_delta_std']:.6g} | {row['tan_delta_sem']:.6g} | "
                    f"{row['tan_delta_cv']:.6g} |"
                ),
                "",
            ]
        )

    if plot_paths:
        lines.extend(["## Plots", ""])
        for path in plot_paths:
            lines.append(f"- `{path.relative_to(output_dir)}`")
        lines.append("")

    lines.extend(
        [
            "## Notes",
            "",
            "- The reported standard deviation is the run-to-run spread from different Maxwell-Boltzmann velocity draws.",
            "- The SEM is useful for the uncertainty of the mean, but the standard deviation is the more direct estimate of single-run reproducibility.",
            "- DMAx2 keeps harmonic and drift diagnostics in the same table, so large random-seed spread can be separated from poor sinusoidal response quality.",
            "",
        ]
    )
    report_path.write_text("\n".join(lines))
    return {
        "replicates": replicates_path,
        "statistics": statistics_path,
        "report": report_path,
        **{f"plot_{i}": path for i, path in enumerate(plot_paths, start=1)},
    }
