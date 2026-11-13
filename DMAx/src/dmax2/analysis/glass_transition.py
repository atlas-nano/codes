"""Glass-transition analysis.

DMAx2 supports two complementary Tg estimators:

1. A density/specific-volume piecewise-linear fit, retained as a comparison to
   common MD cooling workflows.
2. A DMA-style loss-tangent estimator, matching the original DMAx paper's
   emphasis on temperature sweeps of storage modulus, loss modulus, and
   tan(delta). The tan(delta) route is often cheaper because the same dynamic
   runs used for viscoelastic properties provide the transition signal.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import GlassTransitionSpec


@dataclass
class TgFit:
    tg_k: float
    low_slope: float
    high_slope: float
    intercept_low: float
    intercept_high: float
    sse: float

    def as_dict(self) -> dict:
        return self.__dict__.copy()


@dataclass
class TanDeltaTgFit:
    tg_k: float
    peak_tg_k: float
    primary_inflection_k: float
    method: str
    peak_tan_delta: float
    max_slope: float
    n_points: int
    inflection_candidates_k: str
    reference_tg_k: float | None = None
    selection_basis: str = "inflection_near_peak"
    peak_at_edge: bool = False

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def fit_piecewise_tg(
    data: pd.DataFrame,
    temperature_col: str = "temperature_K",
    density_col: str = "density_g_cm3",
    observable: str = "specific_volume",
) -> TgFit:
    """Fit two lines to density or specific volume and return their crossing."""

    df = data[[temperature_col, density_col]].dropna().sort_values(temperature_col)
    t = df[temperature_col].to_numpy(dtype=float)
    if observable == "specific_volume":
        y = 1.0 / df[density_col].to_numpy(dtype=float)
    else:
        y = df[density_col].to_numpy(dtype=float)
    if len(t) < 5:
        raise ValueError("At least five temperature points are recommended for Tg fitting.")
    best = None
    for split in range(2, len(t) - 2):
        b1 = np.polyfit(t[:split], y[:split], 1)
        b2 = np.polyfit(t[split:], y[split:], 1)
        pred = np.concatenate([np.polyval(b1, t[:split]), np.polyval(b2, t[split:])])
        sse = float(np.sum((y - pred) ** 2))
        if best is None or sse < best[0]:
            best = (sse, b1, b2)
    assert best is not None
    sse, b1, b2 = best
    denom = b1[0] - b2[0]
    tg = (b2[1] - b1[1]) / denom if denom else np.nan
    return TgFit(float(tg), float(b1[0]), float(b2[0]), float(b1[1]), float(b2[1]), float(sse))


def _smooth(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(values) < 3:
        return values
    window = min(window, len(values))
    if window % 2 == 0:
        window -= 1
    if window < 3:
        return values
    pad = window // 2
    padded = np.pad(values, pad, mode="edge")
    kernel = np.ones(window) / window
    return np.convolve(padded, kernel, mode="valid")


def fit_tandelta_tg(
    data: pd.DataFrame,
    temperature_col: str = "temperature_K",
    tan_delta_col: str = "tan_delta",
    smoothing_window: int = 3,
    method: str = "inflection_near_peak",
    reference_tg_k: float | None = None,
) -> TanDeltaTgFit:
    """Estimate Tg from a loss-tangent temperature sweep.

    The paper-style signal is the tan(delta) peak/transition in a DMA
    temperature sweep. This function reports both the peak and inflection
    candidates. When an expected Tg or sweep-center prior is available, the
    default inflection method selects the inflection closest to that reference.
    This keeps edge peaks or high-temperature noisy damping from hijacking the
    transition assignment. Without a reference, the inflection closest to the
    tan(delta) peak is used.
    """

    df = data[[temperature_col, tan_delta_col]].dropna().sort_values(temperature_col)
    if len(df) < 4:
        raise ValueError("At least four temperature points are required for tan(delta) Tg analysis.")
    t = df[temperature_col].to_numpy(dtype=float)
    y_raw = df[tan_delta_col].to_numpy(dtype=float)
    y = _smooth(y_raw, smoothing_window)
    peak_idx = int(np.nanargmax(y))
    peak_t = float(t[peak_idx])
    peak_y = float(y[peak_idx])

    dy = np.gradient(y, t)
    d2y = np.gradient(dy, t)
    candidates: list[float] = []
    for i in range(len(t) - 1):
        left, right = d2y[i], d2y[i + 1]
        if not np.isfinite(left + right):
            continue
        if left == 0:
            candidates.append(float(t[i]))
        elif left * right < 0:
            frac = abs(left) / (abs(left) + abs(right))
            candidates.append(float(t[i] + frac * (t[i + 1] - t[i])))
    peak_at_edge = bool(peak_idx in {0, len(t) - 1})
    selection_basis = "inflection_near_peak"
    if candidates:
        if method in {"inflection", "inflection_near_reference"} and reference_tg_k is not None:
            primary = min(candidates, key=lambda x: abs(x - reference_tg_k))
            selection_basis = "inflection_near_reference"
        else:
            primary = min(candidates, key=lambda x: abs(x - peak_t))
    else:
        primary = float(t[int(np.nanargmax(np.abs(dy)))])
        selection_basis = "max_slope_fallback"
    if method == "peak":
        tg = peak_t
    elif method == "max_slope":
        tg = float(t[int(np.nanargmax(np.abs(dy)))])
    else:
        tg = float(primary if np.isfinite(primary) else peak_t)
    return TanDeltaTgFit(
        tg_k=float(tg),
        peak_tg_k=peak_t,
        primary_inflection_k=float(primary),
        method=method,
        peak_tan_delta=peak_y,
        max_slope=float(np.nanmax(np.abs(dy))),
        n_points=int(len(df)),
        inflection_candidates_k=";".join(f"{c:.6g}" for c in candidates),
        reference_tg_k=float(reference_tg_k) if reference_tg_k is not None else None,
        selection_basis=selection_basis,
        peak_at_edge=peak_at_edge,
    )


def prepare_temperature_sweep(
    summary: pd.DataFrame,
    signal: str | None = None,
    parser: str = "lockin",
    frequency_ghz: float | None = None,
) -> pd.DataFrame:
    """Filter a DMA summary table into a temperature sweep with DMA metrics."""

    df = summary.copy()
    if parser and "parser" in df:
        df = df[df["parser"] == parser]
    if signal and "signal" in df:
        df = df[df["signal"] == signal]
    if frequency_ghz is not None and "frequency_ghz" in df:
        df = df[np.isclose(df["frequency_ghz"].astype(float), float(frequency_ghz))]
    if "temperature_K" not in df or "tan_delta" not in df:
        raise ValueError("DMA summary must contain temperature_K and tan_delta columns.")
    metrics = [
        "storage_modulus_MPa",
        "loss_modulus_MPa",
        "tan_delta",
        "r2",
        "snr_first_harmonic",
        "harm2_over_harm1",
        "harm3_over_harm1",
        "drift_fraction_per_cycle",
        "strain",
    ]
    metrics = [col for col in metrics if col in df]
    return (
        df.groupby("temperature_K", as_index=False)[metrics]
        .median()
        .sort_values("temperature_K")
    )


def prepare_tandelta_sweep(
    summary: pd.DataFrame,
    signal: str | None = None,
    parser: str = "lockin",
    frequency_ghz: float | None = None,
) -> pd.DataFrame:
    """Filter a DMA summary table into a temperature/tan(delta) sweep."""

    sweep = prepare_temperature_sweep(summary, signal=signal, parser=parser, frequency_ghz=frequency_ghz)
    return sweep[["temperature_K", "tan_delta"]]


def _reference_tg_from_grid(output_dir: Path, spec: GlassTransitionSpec) -> float | None:
    if spec.expected_tg_k is not None:
        return float(spec.expected_tg_k)
    grid_path = output_dir / "temperature_grid.csv"
    if not grid_path.exists():
        return None
    grid = pd.read_csv(grid_path)
    if "temperature_K" not in grid or grid.empty:
        return None
    temps = grid["temperature_K"].dropna().to_numpy(dtype=float)
    if len(temps) == 0:
        return None
    return float(0.5 * (np.nanmin(temps) + np.nanmax(temps)))


def _write_temperature_coverage(output_dir: Path, observed_temperatures: np.ndarray) -> Path | None:
    grid_path = output_dir / "temperature_grid.csv"
    if not grid_path.exists():
        return None
    grid = pd.read_csv(grid_path)
    if "temperature_K" not in grid:
        return None
    observed = np.asarray(observed_temperatures, dtype=float)
    rows: list[dict] = []
    for expected in grid["temperature_K"].dropna().to_numpy(dtype=float):
        if len(observed) == 0:
            rows.append({"expected_temperature_K": expected, "observed_temperature_K": np.nan, "status": "missing"})
            continue
        idx = int(np.nanargmin(np.abs(observed - expected)))
        nearest = float(observed[idx])
        if abs(nearest - expected) <= 0.01:
            rows.append({"expected_temperature_K": expected, "observed_temperature_K": nearest, "status": "observed"})
        else:
            rows.append({"expected_temperature_K": expected, "observed_temperature_K": np.nan, "status": "missing"})
    path = output_dir / "glass_transition_temperature_coverage.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _write_glass_transition_plots(
    sweep: pd.DataFrame,
    summary: pd.DataFrame,
    output_dir: Path,
    spec: GlassTransitionSpec,
    fit: TanDeltaTgFit,
) -> list[Path]:
    paths: list[Path] = []
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return paths

    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    t = sweep["temperature_K"].to_numpy(dtype=float)
    tan = sweep["tan_delta"].to_numpy(dtype=float)
    tan_smooth = _smooth(tan, spec.smoothing_window)

    fig, axes = plt.subplots(2, 2, figsize=(10.2, 7.4), sharex=True)
    ax = axes[0, 0]
    if "storage_modulus_MPa" in sweep:
        ax.plot(t, sweep["storage_modulus_MPa"], marker="o", linewidth=1.5, label="Storage modulus")
    if "loss_modulus_MPa" in sweep:
        ax.plot(t, sweep["loss_modulus_MPa"], marker="s", linewidth=1.5, label="Loss modulus")
    ax.axvline(fit.tg_k, color="#dc2626", linestyle="--", linewidth=1.4)
    ax.set_ylabel("Modulus (MPa)")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)

    ax = axes[0, 1]
    ax.plot(t, tan, marker="o", linewidth=1.2, color="#334155", alpha=0.65, label="tan(delta)")
    ax.plot(t, tan_smooth, linewidth=2.0, color="#0f766e", label=f"smoothed, window={spec.smoothing_window}")
    ax.axvline(fit.tg_k, color="#dc2626", linestyle="--", label=f"Tg = {fit.tg_k:.1f} K")
    ax.axvline(fit.peak_tg_k, color="#2563eb", linestyle=":", label=f"peak = {fit.peak_tg_k:.1f} K")
    if fit.reference_tg_k is not None:
        ax.axvline(fit.reference_tg_k, color="#64748b", linestyle="-.", label=f"reference = {fit.reference_tg_k:.1f} K")
    ax.set_ylabel("tan(delta)")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1, 0]
    if "r2" in sweep:
        ax.plot(t, sweep["r2"], marker="o", linewidth=1.4, label="R2")
        ax.axhline(0.70, color="#94a3b8", linestyle="--", linewidth=1.0, label="R2 threshold")
    if "snr_first_harmonic" in sweep:
        ax2 = ax.twinx()
        ax2.plot(t, sweep["snr_first_harmonic"], marker="s", linewidth=1.4, color="#7c3aed", label="SNR")
        ax2.axhline(2.0, color="#c4b5fd", linestyle=":", linewidth=1.0, label="SNR threshold")
        ax2.set_ylabel("SNR")
        handles, labels = ax.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(handles + handles2, labels + labels2, frameon=False, fontsize=8, loc="upper right")
    else:
        ax.legend(frameon=False, fontsize=8)
    ax.set_ylabel("R2")
    ax.set_xlabel("Temperature (K)")
    ax.grid(True, alpha=0.25)

    ax = axes[1, 1]
    for metric, label, color in [
        ("harm2_over_harm1", "2omega / 1omega", "#ea580c"),
        ("harm3_over_harm1", "3omega / 1omega", "#ca8a04"),
        ("drift_fraction_per_cycle", "drift / cycle", "#0891b2"),
    ]:
        if metric in sweep:
            ax.plot(t, sweep[metric], marker="o", linewidth=1.3, label=label, color=color)
    ax.axhline(0.20, color="#f97316", linestyle="--", linewidth=1.0, label="harmonic threshold")
    ax.axhline(0.05, color="#06b6d4", linestyle=":", linewidth=1.0, label="drift threshold")
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel("Diagnostic ratio")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=8)

    fig.suptitle("DMAx2 glass-transition sweep")
    fig.tight_layout()
    path = plot_dir / "glass_transition_dma_spectra.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    paths.append(path)

    data = summary[
        (summary["signal"] == spec.tan_delta_signal)
        & (summary["parser"].isin([spec.tan_delta_parser, "legacy"]))
    ].copy()
    if "temperature_K" in data and not data.empty:
        fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.9), sharex=True)
        labels = {spec.tan_delta_parser: "DMAx2 lock-in", "legacy": "legacy sinusoidal fit"}
        for parser, group in data.groupby("parser"):
            group = group.sort_values("temperature_K")
            axes[0].plot(
                group["temperature_K"],
                group["storage_modulus_MPa"],
                marker="o",
                linewidth=1.4,
                label=f"{labels.get(parser, parser)} storage",
            )
            axes[0].plot(
                group["temperature_K"],
                group["loss_modulus_MPa"],
                marker="s",
                linewidth=1.4,
                linestyle="--",
                label=f"{labels.get(parser, parser)} loss",
            )
            axes[1].plot(
                group["temperature_K"],
                group["tan_delta"],
                marker="o",
                linewidth=1.4,
                label=labels.get(parser, parser),
            )
        axes[0].set_ylabel("Modulus (MPa)")
        axes[0].set_xlabel("Temperature (K)")
        axes[0].grid(True, alpha=0.25)
        axes[0].legend(frameon=False, fontsize=8)
        axes[1].set_ylabel("tan(delta)")
        axes[1].set_xlabel("Temperature (K)")
        axes[1].grid(True, alpha=0.25)
        axes[1].legend(frameon=False)
        fig.suptitle(f"Parser comparison: {spec.tan_delta_signal}")
        fig.tight_layout()
        path = plot_dir / "glass_transition_parser_comparison.png"
        fig.savefig(path, dpi=220)
        plt.close(fig)
        paths.append(path)
    return paths


def collect_density_sweep(
    output_dir: str | Path,
    tail_fraction: float = 0.5,
) -> pd.DataFrame:
    """Collect density-vs-temperature traces from density Tg runs.

    The writer emits ``density_trace.dat`` in each temperature directory using
    ``fix ave/time``. This parser averages the final fraction of each trace so
    the Tg fit uses the equilibrated density plateau rather than startup drift.
    """

    output_dir = Path(output_dir)
    rows: list[dict] = []
    columns = [
        "step",
        "temperature_K_inst",
        "pressure_atm",
        "density_g_cm3",
        "volume_A3",
        "lx_A",
        "ly_A",
        "lz_A",
    ]
    for path in sorted((output_dir / "temperatures").glob("T_*K/density_trace.dat")):
        token = path.parent.name.removeprefix("T_").removesuffix("K")
        target_t = float(token.replace("p", "."))
        df = pd.read_csv(path, sep=r"\s+", comment="#", names=columns)
        df = df.dropna()
        if df.empty:
            continue
        start = int(max(0, np.floor(len(df) * (1.0 - tail_fraction))))
        tail = df.iloc[start:]
        row = {
            "temperature_K": target_t,
            "n_samples": int(len(df)),
            "n_samples_used": int(len(tail)),
            "density_g_cm3": float(tail["density_g_cm3"].mean()),
            "density_std_g_cm3": float(tail["density_g_cm3"].std(ddof=1)) if len(tail) > 1 else 0.0,
            "specific_volume_cm3_g": float((1.0 / tail["density_g_cm3"]).mean()),
            "temperature_mean_K": float(tail["temperature_K_inst"].mean()),
            "pressure_mean_atm": float(tail["pressure_atm"].mean()),
            "volume_mean_A3": float(tail["volume_A3"].mean()),
            "lx_mean_A": float(tail["lx_A"].mean()),
            "ly_mean_A": float(tail["ly_A"].mean()),
            "lz_mean_A": float(tail["lz_A"].mean()),
            "file": str(path),
        }
        rows.append(row)
    return pd.DataFrame(rows).sort_values("temperature_K").reset_index(drop=True)


def write_density_glass_transition_report(
    output_dir: str | Path,
    spec: GlassTransitionSpec | None = None,
    tail_fraction: float = 0.5,
) -> dict[str, Path]:
    """Parse NPT density traces and fit a density/specific-volume Tg."""

    spec = spec or GlassTransitionSpec()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sweep = collect_density_sweep(output_dir, tail_fraction=tail_fraction)
    if sweep.empty:
        raise ValueError(f"No density_trace.dat files found below {output_dir / 'temperatures'}.")

    sweep_path = output_dir / "glass_transition_density_sweep.csv"
    sweep.to_csv(sweep_path, index=False)
    fit = fit_piecewise_tg(sweep, observable=spec.observable)
    fit_path = output_dir / "glass_transition_density_fit.csv"
    pd.DataFrame([fit.as_dict() | {"observable": spec.observable, "tail_fraction": tail_fraction}]).to_csv(fit_path, index=False)

    plot_dir = output_dir / "plots"
    plot_dir.mkdir(exist_ok=True)
    plot_path = plot_dir / "glass_transition_density_piecewise.png"
    try:
        import matplotlib.pyplot as plt

        t = sweep["temperature_K"].to_numpy(dtype=float)
        y = 1.0 / sweep["density_g_cm3"].to_numpy(dtype=float) if spec.observable == "specific_volume" else sweep["density_g_cm3"].to_numpy(dtype=float)
        split_idx = int(np.nanargmin(np.abs(t - fit.tg_k))) if np.isfinite(fit.tg_k) else len(t) // 2
        fig, ax = plt.subplots(figsize=(6.6, 4.1))
        ax.plot(t, y, marker="o", linewidth=1.5, label=spec.observable.replace("_", " "))
        if split_idx >= 2:
            ax.plot(t[: split_idx + 1], fit.low_slope * t[: split_idx + 1] + fit.intercept_low, color="#2563eb", linewidth=1.8, label="low-T fit")
        if split_idx < len(t) - 2:
            ax.plot(t[split_idx:], fit.high_slope * t[split_idx:] + fit.intercept_high, color="#dc2626", linewidth=1.8, label="high-T fit")
        ax.axvline(fit.tg_k, color="#0f766e", linestyle="--", label=f"Tg = {fit.tg_k:.1f} K")
        ax.set_xlabel("Temperature (K)")
        ax.set_ylabel("Specific volume (cm3/g)" if spec.observable == "specific_volume" else "Density (g/cm3)")
        ax.set_title("DMAx2 density/specific-volume glass-transition estimate")
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(plot_path, dpi=220)
        plt.close(fig)
    except ImportError:
        plot_path = Path("")

    report_path = output_dir / "glass_transition_density_report.md"
    lines = [
        "# DMAx2 Density Glass-Transition Analysis",
        "",
        f"Estimated Tg: `{fit.tg_k:.6g} K`",
        f"Observable: `{spec.observable}`",
        f"Temperature points parsed: `{len(sweep)}`",
        f"Tail fraction averaged: `{tail_fraction}`",
        f"SSE: `{fit.sse:.6g}`",
        "",
        "Artifacts:",
        "",
        f"- `{sweep_path.relative_to(output_dir)}`",
        f"- `{fit_path.relative_to(output_dir)}`",
    ]
    if plot_path:
        lines.append(f"- `{plot_path.relative_to(output_dir)}`")
    lines.append("")
    report_path.write_text("\n".join(lines))

    artifacts = {"sweep": sweep_path, "fit": fit_path, "report": report_path}
    if plot_path:
        artifacts["plot"] = plot_path
    return artifacts


def _method_label(path: Path) -> str:
    name = path.name
    if "density_equilibrated" in name:
        return "DMAx2 density-equilibrated tan(delta)"
    if "refined" in name:
        return "DMAx2 refined tan(delta)"
    if "density_inflection" in name or "density" in name:
        return "density inflection"
    if "glass_transition_25GHz" in name:
        return "DMAx2 original tan(delta)"
    return name


def _format_optional(value: float | None, digits: int = 3) -> str:
    if value is None or not np.isfinite(value):
        return "n/a"
    return f"{value:.{digits}f}"


def write_glass_transition_comparison_report(
    dma_dirs: list[str | Path],
    density_dir: str | Path,
    output_dir: str | Path,
    expected_tg_k: float | None = None,
) -> dict[str, Path]:
    """Compare DMA tan(delta) and density/specific-volume Tg analyses.

    This routine assumes each input directory has already been parsed with the
    DMAx2 glass-transition commands. It writes combined tables, publication-
    oriented comparison plots, and a concise scientific interpretation that
    records the computational caveats of each estimator.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(exist_ok=True)

    method_rows: list[dict] = []
    dma_sweeps: list[pd.DataFrame] = []
    dma_fits: list[pd.Series] = []
    for raw_dir in dma_dirs:
        path = Path(raw_dir)
        label = _method_label(path)
        sweep_path = path / "glass_transition_dma_sweep.csv"
        fit_path = path / "glass_transition_tandelta_fit.csv"
        if not sweep_path.exists() or not fit_path.exists():
            raise FileNotFoundError(f"Missing DMA Tg artifacts in {path}")
        sweep = pd.read_csv(sweep_path).sort_values("temperature_K")
        fit = pd.read_csv(fit_path).iloc[0]
        temps = sweep["temperature_K"].to_numpy(dtype=float)
        coverage_path = path / "glass_transition_temperature_coverage.csv"
        missing_count = 0
        if coverage_path.exists():
            coverage = pd.read_csv(coverage_path)
            if "status" in coverage:
                missing_count = int((coverage["status"] == "missing").sum())
        dma_sweeps.append(sweep.assign(method=label, source_dir=str(path)))
        dma_fits.append(fit)
        method_rows.append(
            {
                "method": label,
                "tg_k": float(fit["tg_k"]),
                "temperature_points": int(len(sweep)),
                "temperature_min_K": float(np.nanmin(temps)),
                "temperature_max_K": float(np.nanmax(temps)),
                "median_temperature_spacing_K": float(np.nanmedian(np.diff(temps))) if len(temps) > 1 else np.nan,
                "missing_temperature_points": missing_count,
                "primary_inflection_k": float(fit.get("primary_inflection_k", np.nan)),
                "inflection_candidates_k": str(fit.get("inflection_candidates_k", "")),
                "selection_basis": str(fit.get("selection_basis", "")),
                "peak_tg_k": float(fit.get("peak_tg_k", np.nan)),
                "peak_at_edge": bool(fit.get("peak_at_edge", False)),
                "median_r2": float(sweep["r2"].median()) if "r2" in sweep else np.nan,
                "median_snr_first_harmonic": float(sweep["snr_first_harmonic"].median()) if "snr_first_harmonic" in sweep else np.nan,
                "median_harm2_over_harm1": float(sweep["harm2_over_harm1"].median()) if "harm2_over_harm1" in sweep else np.nan,
                "median_harm3_over_harm1": float(sweep["harm3_over_harm1"].median()) if "harm3_over_harm1" in sweep else np.nan,
                "median_drift_fraction_per_cycle": float(sweep["drift_fraction_per_cycle"].median()) if "drift_fraction_per_cycle" in sweep else np.nan,
                "source_dir": str(path),
            }
        )

    density_path = Path(density_dir)
    density_sweep_path = density_path / "glass_transition_density_sweep.csv"
    density_fit_path = density_path / "glass_transition_density_fit.csv"
    if not density_sweep_path.exists() or not density_fit_path.exists():
        raise FileNotFoundError(f"Missing density Tg artifacts in {density_path}")
    density_sweep = pd.read_csv(density_sweep_path).sort_values("temperature_K")
    density_fit = pd.read_csv(density_fit_path).iloc[0]
    density_tg = float(density_fit["tg_k"])
    method_rows.append(
        {
            "method": _method_label(density_path),
            "tg_k": density_tg,
            "temperature_points": int(len(density_sweep)),
            "temperature_min_K": float(density_sweep["temperature_K"].min()),
            "temperature_max_K": float(density_sweep["temperature_K"].max()),
            "median_temperature_spacing_K": float(np.nanmedian(np.diff(density_sweep["temperature_K"].to_numpy(dtype=float)))),
            "missing_temperature_points": 0,
            "primary_inflection_k": np.nan,
            "inflection_candidates_k": "",
            "selection_basis": "piecewise_specific_volume",
            "peak_tg_k": np.nan,
            "peak_at_edge": False,
            "median_r2": np.nan,
            "median_snr_first_harmonic": np.nan,
            "median_harm2_over_harm1": np.nan,
            "median_harm3_over_harm1": np.nan,
            "median_drift_fraction_per_cycle": np.nan,
            "source_dir": str(density_path),
        }
    )

    method_summary = pd.DataFrame(method_rows)
    if expected_tg_k is not None:
        method_summary["delta_from_expected_K"] = method_summary["tg_k"] - float(expected_tg_k)
    method_summary_path = output_dir / "glass_transition_method_summary.csv"
    method_summary.to_csv(method_summary_path, index=False)

    combined_dma = pd.concat(dma_sweeps, ignore_index=True) if dma_sweeps else pd.DataFrame()
    combined_dma_path = output_dir / "glass_transition_dma_combined_sweeps.csv"
    combined_dma.to_csv(combined_dma_path, index=False)
    density_copy_path = output_dir / "glass_transition_density_sweep.csv"
    density_sweep.to_csv(density_copy_path, index=False)

    density_t = density_sweep["temperature_K"].to_numpy(dtype=float)
    density_y = density_sweep["specific_volume_cm3_g"].to_numpy(dtype=float)
    density_pred = np.where(
        density_t <= density_tg,
        float(density_fit["low_slope"]) * density_t + float(density_fit["intercept_low"]),
        float(density_fit["high_slope"]) * density_t + float(density_fit["intercept_high"]),
    )
    density_rmse = float(np.sqrt(np.nanmean((density_y - density_pred) ** 2)))

    dma_tgs = [float(fit["tg_k"]) for fit in dma_fits]
    dma_spread = float(max(dma_tgs) - min(dma_tgs)) if len(dma_tgs) > 1 else 0.0
    reference_dma_tg = dma_tgs[-1] if dma_tgs else np.nan
    reference_dma_label = method_rows[len(dma_tgs) - 1]["method"] if dma_tgs else "DMAx2 tan(delta)"
    density_delta_from_reference_dma = density_tg - reference_dma_tg if np.isfinite(reference_dma_tg) else np.nan

    plot_paths: list[Path] = []
    derivative_path: Path | None = None
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 1, figsize=(8.4, 7.4), sharex=False)
        ax = axes[0]
        for label, group in combined_dma.groupby("method"):
            group = group.sort_values("temperature_K")
            ax.plot(group["temperature_K"], group["tan_delta"], marker="o", linewidth=1.35, label=label)
        for row in method_rows:
            if "DMAx2" in row["method"]:
                ax.axvline(row["tg_k"], linestyle="--", linewidth=1.0, alpha=0.75)
        if expected_tg_k is not None:
            ax.axvline(float(expected_tg_k), color="#111827", linestyle="-.", linewidth=1.2, label="experimental Tg prior")
        ax.set_xlabel("Temperature (K)")
        ax.set_ylabel("tan(delta)")
        ax.set_title("DMAx2 loss-tangent Tg sweeps")
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False, fontsize=8)

        ax = axes[1]
        ax.errorbar(
            density_sweep["temperature_K"],
            density_sweep["specific_volume_cm3_g"],
            yerr=density_sweep["density_std_g_cm3"] / (density_sweep["density_g_cm3"] ** 2),
            marker="o",
            linewidth=1.35,
            capsize=2,
            label="density-derived specific volume",
        )
        ax.plot(density_t, density_pred, color="#dc2626", linewidth=1.8, label="piecewise linear fit")
        ax.axvline(density_tg, color="#dc2626", linestyle="--", linewidth=1.2, label=f"density Tg = {density_tg:.1f} K")
        if expected_tg_k is not None:
            ax.axvline(float(expected_tg_k), color="#111827", linestyle="-.", linewidth=1.2, label="experimental Tg prior")
        ax.set_xlabel("Temperature (K)")
        ax.set_ylabel("Specific volume (cm3/g)")
        ax.set_title("Density/specific-volume Tg comparison")
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        path = plot_dir / "glass_transition_method_overlay.png"
        fig.savefig(path, dpi=220)
        plt.close(fig)
        plot_paths.append(path)

        fig, ax = plt.subplots(figsize=(6.6, 3.8))
        colors = ["#2563eb" if "DMAx2" in method else "#dc2626" for method in method_summary["method"]]
        ax.bar(method_summary["method"], method_summary["tg_k"], color=colors, alpha=0.82)
        if expected_tg_k is not None:
            ax.axhline(float(expected_tg_k), color="#111827", linestyle="-.", linewidth=1.1, label=f"expected = {expected_tg_k:.1f} K")
        ax.set_ylabel("Estimated Tg (K)")
        ax.set_title("Glass-transition estimates by method")
        ax.tick_params(axis="x", rotation=18)
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend(frameon=False)
        fig.tight_layout()
        path = plot_dir / "glass_transition_tg_estimates.png"
        fig.savefig(path, dpi=220)
        plt.close(fig)
        plot_paths.append(path)

        fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8), sharex=False)
        if "r2" in combined_dma:
            for label, group in combined_dma.groupby("method"):
                group = group.sort_values("temperature_K")
                axes[0].plot(group["temperature_K"], group["r2"], marker="o", linewidth=1.2, label=label)
            axes[0].axhline(0.70, color="#94a3b8", linestyle="--", linewidth=1.0, label="R2 threshold")
            axes[0].set_ylabel("R2")
            axes[0].set_xlabel("Temperature (K)")
            axes[0].grid(True, alpha=0.25)
            axes[0].legend(frameon=False, fontsize=8)
        if "snr_first_harmonic" in combined_dma:
            for label, group in combined_dma.groupby("method"):
                group = group.sort_values("temperature_K")
                axes[1].plot(group["temperature_K"], group["snr_first_harmonic"], marker="o", linewidth=1.2, label=label)
            axes[1].axhline(2.0, color="#94a3b8", linestyle="--", linewidth=1.0, label="SNR threshold")
            axes[1].set_ylabel("First-harmonic SNR")
            axes[1].set_xlabel("Temperature (K)")
            axes[1].grid(True, alpha=0.25)
            axes[1].legend(frameon=False, fontsize=8)
        fig.suptitle("DMAx2 fit diagnostics")
        fig.tight_layout()
        path = plot_dir / "glass_transition_dma_diagnostics.png"
        fig.savefig(path, dpi=220)
        plt.close(fig)
        plot_paths.append(path)

        if {"storage_modulus_MPa", "loss_modulus_MPa"}.issubset(combined_dma.columns):
            fig, axes = plt.subplots(2, 1, figsize=(8.4, 6.4), sharex=True)
            for label, group in combined_dma.groupby("method"):
                group = group.sort_values("temperature_K")
                axes[0].plot(group["temperature_K"], group["storage_modulus_MPa"], marker="o", linewidth=1.25, label=label)
                axes[1].plot(group["temperature_K"], group["loss_modulus_MPa"], marker="s", linewidth=1.25, label=label)
            for row in method_rows:
                if "DMAx2" not in row["method"]:
                    continue
                axes[0].axvline(row["tg_k"], linestyle="--", linewidth=0.9, alpha=0.45)
                axes[1].axvline(row["tg_k"], linestyle="--", linewidth=0.9, alpha=0.45)
            axes[0].set_ylabel("Storage modulus (MPa)")
            axes[1].set_ylabel("Loss modulus (MPa)")
            axes[1].set_xlabel("Temperature (K)")
            for ax in axes:
                ax.grid(True, alpha=0.25)
                ax.legend(frameon=False, fontsize=8)
            fig.suptitle("DMAx2 storage and loss modulus comparison")
            fig.tight_layout()
            path = plot_dir / "glass_transition_dma_moduli_comparison.png"
            fig.savefig(path, dpi=220)
            plt.close(fig)
            plot_paths.append(path)

        fig, axes = plt.subplots(3, 1, figsize=(8.6, 8.2), sharex=True)
        derivative_rows: list[dict] = []
        for label, group in combined_dma.groupby("method"):
            group = group.sort_values("temperature_K")
            t = group["temperature_K"].to_numpy(dtype=float)
            y = group["tan_delta"].to_numpy(dtype=float)
            if len(t) < 3:
                continue
            window = min(7, len(y))
            if window % 2 == 0:
                window -= 1
            if window < 3:
                window = 1
            y_smooth = _smooth(y, window)
            dy = np.gradient(y_smooth, t)
            d2y = np.gradient(dy, t)
            axes[0].plot(t, y, marker="o", linewidth=0.8, alpha=0.28)
            axes[0].plot(t, y_smooth, linewidth=1.7, label=f"{label}, window={window}")
            axes[1].plot(t, dy, marker="o", linewidth=1.25, label=label)
            axes[2].plot(t, d2y, marker="o", linewidth=1.25, label=label)
            for ti, raw, smooth, first, second in zip(t, y, y_smooth, dy, d2y):
                derivative_rows.append(
                    {
                        "method": label,
                        "temperature_K": ti,
                        "tan_delta": raw,
                        "tan_delta_smooth": smooth,
                        "d_tandelta_dT": first,
                        "d2_tandelta_dT2": second,
                        "smoothing_window": window,
                    }
                )
        for row in method_rows:
            if "DMAx2" not in row["method"]:
                continue
            for ax in axes:
                ax.axvline(row["tg_k"], linestyle="--", linewidth=0.85, alpha=0.45)
        if expected_tg_k is not None:
            for ax in axes:
                ax.axvline(float(expected_tg_k), color="#111827", linestyle="-.", linewidth=1.0, alpha=0.8)
        axes[0].set_ylabel("tan(delta)")
        axes[1].set_ylabel("d tan(delta) / dT")
        axes[2].set_ylabel("d2 tan(delta) / dT2")
        axes[2].set_xlabel("Temperature (K)")
        for ax in axes:
            ax.grid(True, alpha=0.25)
            ax.legend(frameon=False, fontsize=8)
        fig.suptitle("Smoothed loss tangent and numerical derivatives")
        fig.tight_layout()
        path = plot_dir / "glass_transition_tandelta_derivatives_comparison.png"
        fig.savefig(path, dpi=220)
        plt.close(fig)
        plot_paths.append(path)
        if derivative_rows:
            derivative_path = output_dir / "glass_transition_tandelta_derivatives.csv"
            pd.DataFrame(derivative_rows).to_csv(derivative_path, index=False)

        fig, axes = plt.subplots(3, 1, figsize=(8.4, 7.4), sharex=True)
        for label, group in combined_dma.groupby("method"):
            group = group.sort_values("temperature_K")
            for ax, metric in zip(
                axes,
                ["harm2_over_harm1", "harm3_over_harm1", "drift_fraction_per_cycle"],
            ):
                if metric in group:
                    ax.plot(group["temperature_K"], group[metric], marker="o", linewidth=1.2, label=label)
        axes[0].axhline(0.20, color="#94a3b8", linestyle="--", linewidth=0.9)
        axes[1].axhline(0.20, color="#94a3b8", linestyle="--", linewidth=0.9)
        axes[2].axhline(0.05, color="#94a3b8", linestyle="--", linewidth=0.9)
        axes[0].set_ylabel("2omega / 1omega")
        axes[1].set_ylabel("3omega / 1omega")
        axes[2].set_ylabel("drift / cycle")
        axes[2].set_xlabel("Temperature (K)")
        for ax in axes:
            ax.grid(True, alpha=0.25)
            ax.legend(frameon=False, fontsize=8)
        fig.suptitle("DMAx2 harmonic and drift diagnostics")
        fig.tight_layout()
        path = plot_dir / "glass_transition_dma_harmonic_drift_diagnostics.png"
        fig.savefig(path, dpi=220)
        plt.close(fig)
        plot_paths.append(path)
    except ImportError:
        pass

    dma_confidence = "moderate"
    if dma_spread <= 3.0 and expected_tg_k is not None and abs(reference_dma_tg - float(expected_tg_k)) <= 3.0:
        dma_confidence = "moderate-high"
    if not combined_dma.empty:
        median_r2 = float(combined_dma["r2"].median()) if "r2" in combined_dma else np.nan
        median_snr = float(combined_dma["snr_first_harmonic"].median()) if "snr_first_harmonic" in combined_dma else np.nan
        if np.isfinite(median_r2) and median_r2 < 0.6:
            dma_confidence = "moderate"
        if np.isfinite(median_snr) and median_snr < 1.5:
            dma_confidence = "low-moderate"

    density_confidence = "moderate-high"
    if density_rmse > 0.004:
        density_confidence = "low"
    elif density_rmse > 0.003:
        density_confidence = "moderate"

    report_path = output_dir / "glass_transition_comparison_report.md"
    lines = [
        "# Glass-Transition Method Comparison",
        "",
        "## Headline",
        "",
    ]
    for row in method_rows:
        lines.append(f"- {row['method']} Tg: `{_format_optional(float(row['tg_k']))} K`")
    if expected_tg_k is not None:
        lines.append(f"- Experimental/prior Tg used for judging PDMS: `{float(expected_tg_k):.3f} K`")
    lines.extend(
        [
            "",
            "## Scientific Assessment",
            "",
            f"The DMAx2 tan(delta) estimates span `{dma_spread:.3f} K` across the supplied DMA sweeps. This spread is a reproducibility metric, but it should be interpreted alongside the curve shape and parser diagnostics because the tan(delta) maximum can remain at the high-temperature edge even when the inflection selector reports a Tg-like point.",
            "",
            f"The density method estimates `{density_tg:.3f} K`, which is `{density_delta_from_reference_dma:.3f} K` above `{reference_dma_label}`. The specific-volume fit is the most direct volume-relaxation comparison, while the DMAx2 sweeps probe a frequency-dependent mechanical response at the selected driving frequency.",
            "",
            "## DMAx2 Details",
            "",
        ]
    )
    for row in method_rows:
        if "DMAx2" not in row["method"]:
            continue
        lines.append(
            f"- {row['method']}: `{row['temperature_points']}` points from `{row['temperature_min_K']:.3f}` to `{row['temperature_max_K']:.3f}` K, median spacing `{row['median_temperature_spacing_K']:.3f}` K, selected Tg `{row['tg_k']:.3f}` K from `{row['selection_basis']}`. The tan(delta) peak was at `{row['peak_tg_k']:.3f}` K and `peak_at_edge={row['peak_at_edge']}`. Median R2/SNR were `{row['median_r2']:.3f}` / `{row['median_snr_first_harmonic']:.3f}`; missing expected temperature points: `{row['missing_temperature_points']}`."
        )
    lines.extend(
        [
            "",
            "The edge-peak behavior is why DMAx2 reports an inflection selector rather than the raw tan(delta) maximum. For publication use, visually inspect the smoothed tan(delta), first derivative, and second derivative plots before treating the selected inflection as a physical Tg.",
            "",
            "## Density Details",
            "",
            f"The density fit used `{len(density_sweep)}` independently equilibrated NPT density points with a piecewise specific-volume RMSE of `{density_rmse:.6g}` cm3/g. Its low-T and high-T slopes are `{float(density_fit['low_slope']):.6g}` and `{float(density_fit['high_slope']):.6g}` cm3/g/K, respectively.",
            "",
            "## Confidence",
            "",
            f"- DMAx2 tan(delta): `{dma_confidence}` confidence from automated diagnostics. This confidence can be lowered manually if the derivative plots do not show a clear physically meaningful inflection.",
            f"- Density inflection: `{density_confidence}` confidence from automated fit diagnostics. Compare its `delta_from_expected_K` separately from this fit-quality label.",
            "",
            "## Computational Interpretation",
            "",
            "DMAx2 is probing the dynamic mechanical response directly at 25 GHz with the convergence-selected strain and cycle count. That makes it the closer match to the DMAx paper workflow, but a tan(delta)-based Tg should be treated as a dynamic mechanical feature unless the loss-tangent and derivative plots show a clear transition independent of the expected-Tg prior.",
            "",
            "The density comparison is useful as the static volume-relaxation reference. When its piecewise specific-volume plot has a visually clear slope break, it can be the more defensible static Tg estimate even if its value differs from the experimental prior.",
            "",
            "## Artifacts",
            "",
            f"- `{method_summary_path.relative_to(output_dir)}`",
            f"- `{combined_dma_path.relative_to(output_dir)}`",
            f"- `{density_copy_path.relative_to(output_dir)}`",
        ]
    )
    if derivative_path is not None:
        lines.append(f"- `{derivative_path.relative_to(output_dir)}`")
    for path in plot_paths:
        lines.append(f"- `{path.relative_to(output_dir)}`")
    lines.append("")
    report_path.write_text("\n".join(lines))

    artifacts = {
        "summary": method_summary_path,
        "dma_sweeps": combined_dma_path,
        "density_sweep": density_copy_path,
        "report": report_path,
    }
    if derivative_path is not None:
        artifacts["tandelta_derivatives"] = derivative_path
    for idx, path in enumerate(plot_paths, start=1):
        artifacts[f"plot_{idx}"] = path
    return artifacts


def write_glass_transition_report(
    summary_file: str | Path,
    output_dir: str | Path,
    spec: GlassTransitionSpec | None = None,
) -> dict[str, Path]:
    """Fit the DMA tan(delta) Tg estimator and write report artifacts."""

    spec = spec or GlassTransitionSpec()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(summary_file)
    full_sweep = prepare_temperature_sweep(
        summary,
        signal=spec.tan_delta_signal,
        parser=spec.tan_delta_parser,
        frequency_ghz=spec.tan_delta_frequency_ghz,
    )
    sweep = full_sweep[["temperature_K", "tan_delta"]].copy()
    sweep_path = output_dir / "glass_transition_tandelta_sweep.csv"
    sweep.to_csv(sweep_path, index=False)
    full_sweep_path = output_dir / "glass_transition_dma_sweep.csv"
    full_sweep.to_csv(full_sweep_path, index=False)

    method = spec.method.replace("tan_delta_", "")
    reference_tg_k = _reference_tg_from_grid(output_dir, spec)
    fit = fit_tandelta_tg(
        sweep,
        smoothing_window=spec.smoothing_window,
        method=method,
        reference_tg_k=reference_tg_k,
    )
    fit_path = output_dir / "glass_transition_tandelta_fit.csv"
    pd.DataFrame([fit.as_dict() | {"method_config": spec.method}]).to_csv(fit_path, index=False)
    coverage_path = _write_temperature_coverage(output_dir, sweep["temperature_K"].to_numpy(dtype=float))
    plot_paths = _write_glass_transition_plots(full_sweep, summary, output_dir, spec, fit)

    plot_path = output_dir / "plots" / "glass_transition_tandelta_sweep.png"
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6.4, 4.0))
        ax.plot(sweep["temperature_K"], sweep["tan_delta"], marker="o", linewidth=1.6)
        ax.axvline(fit.tg_k, color="#dc2626", linestyle="--", label=f"Tg = {fit.tg_k:.1f} K")
        ax.axvline(fit.peak_tg_k, color="#2563eb", linestyle=":", label=f"tan(delta) peak = {fit.peak_tg_k:.1f} K")
        ax.set_xlabel("Temperature (K)")
        ax.set_ylabel("tan(delta)")
        ax.set_title("DMAx2 loss-tangent glass-transition estimate")
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(plot_path, dpi=220)
        plt.close(fig)
        plot_paths.insert(0, plot_path)
    except ImportError:
        plot_path = Path("")

    missing = []
    if coverage_path is not None and coverage_path.exists():
        coverage = pd.read_csv(coverage_path)
        missing = coverage.loc[coverage["status"] == "missing", "expected_temperature_K"].tolist()
    status_rows = summary[
        (summary["signal"] == spec.tan_delta_signal)
        & (summary["parser"] == spec.tan_delta_parser)
    ]
    ok_count = int((status_rows["status"] == "ok").sum()) if "status" in status_rows else 0
    status_count = int(len(status_rows))
    median_r2 = float(full_sweep["r2"].median()) if "r2" in full_sweep else np.nan
    median_snr = float(full_sweep["snr_first_harmonic"].median()) if "snr_first_harmonic" in full_sweep else np.nan

    report_path = output_dir / "glass_transition_report.md"
    lines = [
        "# DMAx2 Glass-Transition Analysis",
        "",
        f"Summary file: `{summary_file}`",
        f"Signal: `{spec.tan_delta_signal}`",
        f"Parser: `{spec.tan_delta_parser}`",
        f"Method: `{spec.method}`",
        "",
        f"Estimated Tg: `{fit.tg_k:.6g} K`",
        f"tan(delta) peak temperature: `{fit.peak_tg_k:.6g} K`",
        f"Primary inflection temperature: `{fit.primary_inflection_k:.6g} K`",
        f"Inflection selection basis: `{fit.selection_basis}`",
        f"Reference Tg used for selection: `{fit.reference_tg_k:.6g} K`" if fit.reference_tg_k is not None else "Reference Tg used for selection: `none`",
        f"Peak at sweep edge: `{fit.peak_at_edge}`",
        f"Number of temperature points: `{fit.n_points}`",
        f"Missing expected temperatures: `{', '.join(f'{x:.3f} K' for x in missing) if missing else 'none'}`",
        f"Median R2/SNR: `{median_r2:.3g}` / `{median_snr:.3g}`",
        f"Primary parser ok rows: `{ok_count}/{status_count}`",
        f"Inflection candidates: `{fit.inflection_candidates_k or 'none'}`",
        "",
        "Strategy:",
        "",
        "DMAx2 uses the loss-tangent temperature sweep as the primary Tg estimator, matching the faster route emphasized in the DMAx paper.",
        "When the sweep was generated from an expected Tg prior, DMAx2 selects the loss-tangent inflection nearest that reference instead of the inflection nearest an edge peak.",
        "The density/specific-volume piecewise-linear method remains available as a comparison analysis when cooling or equilibration-density data are collected.",
        "For production planning, the temperature grid is chosen before the sweep from a Tg prior associated with the polymer identity.",
        "",
    ]
    if plot_paths:
        lines.extend(["Plots:", ""])
        for path in plot_paths:
            if path:
                lines.append(f"- `{path.relative_to(output_dir)}`")
        lines.append("")
    if coverage_path is not None:
        lines.extend(["Tables:", "", f"- `{coverage_path.relative_to(output_dir)}`", f"- `{full_sweep_path.relative_to(output_dir)}`", ""])
    report_path.write_text("\n".join(lines))

    artifacts = {"sweep": sweep_path, "full_sweep": full_sweep_path, "fit": fit_path, "report": report_path}
    if coverage_path is not None:
        artifacts["coverage"] = coverage_path
    if plot_paths:
        artifacts["plot"] = plot_paths[0]
        artifacts["spectra_plot"] = plot_paths[1] if len(plot_paths) > 1 else plot_paths[0]
    return artifacts
