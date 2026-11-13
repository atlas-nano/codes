"""DMA trace parsers.

The modern parser is a lock-in estimator at the imposed frequency with drift
and harmonic diagnostics. The legacy parser reproduces the simpler first
sin/cos fit used by early DMAx pressure workflows.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


RAW_RE = re.compile(
    r"raw__(?P<mode>.+)__strain_(?P<strain>[0-9]+p[0-9]+)"
    r"(?:__f_(?P<frequency>[0-9]+p[0-9]+)GHz)?"
    r"(?:__T_(?P<temperature>[0-9]+p[0-9]+)K)?"
    r"(?:__rep_(?P<replicate>[0-9]+))?"
    r"(?:__seed_(?P<seed>[0-9]+))?\.dat$"
)


@dataclass
class FitResult:
    mode: str
    signal: str
    strain: float
    parser: str
    storage_modulus_MPa: float
    loss_modulus_MPa: float
    tan_delta: float
    r2: float
    snr_first_harmonic: float
    harm2_over_harm1: float
    harm3_over_harm1: float
    drift_fraction_per_cycle: float
    n_points: int
    status: str

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def read_raw_table(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path, sep=r"\s+", comment="#")


def _decode_token(value: str) -> float:
    return float(value.replace("p", "."))


def parse_raw_metadata(path: str | Path) -> tuple[str, float, float | None]:
    match = RAW_RE.search(Path(path).name)
    if not match:
        raise ValueError(f"Cannot parse DMA raw file name: {path}")
    temperature = match.group("temperature")
    return (
        match.group("mode"),
        _decode_token(match.group("strain")),
        _decode_token(temperature) if temperature is not None else None,
    )


def parse_raw_run_metadata(path: str | Path) -> dict[str, float | int | str | None]:
    match = RAW_RE.search(Path(path).name)
    if not match:
        raise ValueError(f"Cannot parse DMA raw file name: {path}")
    temperature = match.group("temperature")
    frequency = match.group("frequency")
    replicate = match.group("replicate")
    seed = match.group("seed")
    return {
        "mode": match.group("mode"),
        "strain": _decode_token(match.group("strain")),
        "temperature_K": _decode_token(temperature) if temperature is not None else None,
        "frequency_ghz": _decode_token(frequency) if frequency is not None else None,
        "replicate": int(replicate) if replicate is not None else None,
        "velocity_seed": int(seed) if seed is not None else None,
    }


def parse_raw_frequency(path: str | Path) -> float | None:
    match = RAW_RE.search(Path(path).name)
    if not match:
        raise ValueError(f"Cannot parse DMA raw file name: {path}")
    frequency = match.group("frequency")
    return _decode_token(frequency) if frequency is not None else None


def parse_raw_name(path: str | Path) -> tuple[str, float]:
    mode, strain, _temperature = parse_raw_metadata(path)
    return mode, strain


def _design(t_fs: np.ndarray, period_fs: float, include_harmonics: bool, include_drift: bool) -> np.ndarray:
    omega_t = 2.0 * np.pi * t_fs / period_fs
    cols = [np.ones_like(t_fs)]
    if include_drift:
        centered = (t_fs - t_fs.mean()) / period_fs
        cols.append(centered)
    max_h = 3 if include_harmonics else 1
    for h in range(1, max_h + 1):
        cols.append(np.sin(h * omega_t))
        cols.append(np.cos(h * omega_t))
    return np.column_stack(cols)


def _fit_signal(
    df: pd.DataFrame,
    signal: str,
    strain: float,
    period_fs: float,
    discard_cycles: int,
    parser: str,
) -> tuple[float, float, float, float, float, float, float, int]:
    if "t_ps" in df:
        t_fs = df["t_ps"].to_numpy(dtype=float) * 1000.0
    else:
        t_fs = df["step"].to_numpy(dtype=float)
    keep = t_fs >= discard_cycles * period_fs
    t_fs = t_fs[keep]
    y = df.loc[keep, signal].to_numpy(dtype=float)
    if len(y) < 8:
        raise ValueError(f"Too few points after discard for {signal}.")
    t_fit = t_fs - t_fs.min()
    include_harmonics = parser == "lockin"
    include_drift = parser == "lockin"
    x = _design(t_fit, period_fs, include_harmonics, include_drift)
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    yhat = x @ beta
    residual = y - yhat
    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    offset = 2 if include_drift else 1
    a1, b1 = float(beta[offset]), float(beta[offset + 1])
    first = float(np.hypot(a1, b1))
    noise = float(np.std(residual, ddof=max(1, min(5, len(residual) - 1))))
    snr = first / noise if noise > 0 else np.inf
    if include_harmonics:
        a2, b2 = float(beta[offset + 2]), float(beta[offset + 3])
        a3, b3 = float(beta[offset + 4]), float(beta[offset + 5])
        harm2 = float(np.hypot(a2, b2) / first) if first else np.inf
        harm3 = float(np.hypot(a3, b3) / first) if first else np.inf
        drift = abs(float(beta[1])) / first if first else np.inf
    else:
        harm2 = np.nan
        harm3 = np.nan
        drift = np.nan
    storage = a1 / strain
    loss = b1 / strain
    tan_delta = loss / storage if storage != 0 else np.nan
    return storage, loss, tan_delta, r2, snr, harm2, harm3, drift, len(y)


def classify_fit(r2: float, snr: float, harm2: float, harm3: float, drift: float) -> str:
    flags: list[str] = []
    if np.isfinite(r2) and r2 < 0.70:
        flags.append("low_r2")
    if np.isfinite(snr) and snr < 2.0:
        flags.append("low_snr")
    if np.isfinite(harm2) and harm2 > 0.20:
        flags.append("large_2omega")
    if np.isfinite(harm3) and harm3 > 0.20:
        flags.append("large_3omega")
    if np.isfinite(drift) and drift > 0.05:
        flags.append("drift")
    return "ok" if not flags else ";".join(flags)


def default_signals_for_mode(mode: str) -> list[str]:
    if mode.startswith("shear"):
        return ["sxy_vir_MPa", "sxy_tot_MPa"]
    if mode == "legacy_axial_pressure":
        return ["press_MPa", "szz_vir_MPa", "sdev_MPa"]
    return ["szz_vir_MPa", "sdev_MPa", "press_MPa"]


def parse_dma_file(
    path: str | Path,
    frequency_ghz: float,
    discard_cycles: int = 2,
    parser: str = "lockin",
    signals: Iterable[str] | None = None,
) -> list[FitResult]:
    mode, strain = parse_raw_name(path)
    df = read_raw_table(path)
    period_fs = 1_000_000.0 / frequency_ghz
    if signals is None:
        signals = default_signals_for_mode(mode)
    rows: list[FitResult] = []
    for signal in signals:
        if signal not in df:
            continue
        try:
            if parser == "fourier_projection":
                rows.append(fit_dma_dataframe(df, mode, signal, strain, frequency_ghz, discard_cycles, parser))
                continue
            storage, loss, tan_delta, r2, snr, h2, h3, drift, n = _fit_signal(
                df, signal, strain, period_fs, discard_cycles, parser
            )
            status = classify_fit(r2, snr, h2, h3, drift)
        except ValueError as exc:
            storage = loss = tan_delta = r2 = snr = h2 = h3 = drift = np.nan
            n = len(df)
            status = f"parse_failed:{exc}"
        rows.append(
            FitResult(
                mode=mode,
                signal=signal,
                strain=strain,
                parser=parser,
                storage_modulus_MPa=storage,
                loss_modulus_MPa=loss,
                tan_delta=tan_delta,
                r2=r2,
                snr_first_harmonic=snr,
                harm2_over_harm1=h2,
                harm3_over_harm1=h3,
                drift_fraction_per_cycle=drift,
                n_points=n,
                status=status,
            )
        )
    return rows


def fit_dma_dataframe(
    df: pd.DataFrame,
    mode: str,
    signal: str,
    strain: float,
    frequency_ghz: float,
    discard_cycles: int = 0,
    parser: str = "lockin",
) -> FitResult:
    """Fit one DMA signal from an already-loaded data frame."""

    if parser == "fourier_projection":
        from .fourier import project_periodic_response
        return project_periodic_response(df, mode, signal, frequency_ghz, discard_cycles)[0]
    period_fs = 1_000_000.0 / frequency_ghz
    storage, loss, tan_delta, r2, snr, h2, h3, drift, n = _fit_signal(
        df, signal, strain, period_fs, discard_cycles, parser
    )
    return FitResult(
        mode=mode,
        signal=signal,
        strain=strain,
        parser=parser,
        storage_modulus_MPa=storage,
        loss_modulus_MPa=loss,
        tan_delta=tan_delta,
        r2=r2,
        snr_first_harmonic=snr,
        harm2_over_harm1=h2,
        harm3_over_harm1=h3,
        drift_fraction_per_cycle=drift,
        n_points=n,
        status=classify_fit(r2, snr, h2, h3, drift),
    )


def parse_dma_directory(
    output_dir: str | Path,
    frequency_ghz: float,
    discard_cycles: int = 2,
    parsers: Iterable[str] = ("lockin", "legacy"),
) -> pd.DataFrame:
    output_dir = Path(output_dir)
    raw_dir = output_dir / "raw"
    files = sorted(raw_dir.glob("raw__*__strain_*.dat"))
    records: list[dict] = []
    for parser in parsers:
        for path in files:
            metadata = parse_raw_run_metadata(path)
            temperature = metadata["temperature_K"]
            file_frequency = metadata["frequency_ghz"] or frequency_ghz
            for fit in parse_dma_file(path, file_frequency, discard_cycles, parser):
                record = fit.as_dict() | {"file": str(path)}
                if temperature is not None:
                    record["temperature_K"] = temperature
                if metadata["replicate"] is not None:
                    record["replicate"] = metadata["replicate"]
                if metadata["velocity_seed"] is not None:
                    record["velocity_seed"] = metadata["velocity_seed"]
                record["frequency_ghz"] = file_frequency
                records.append(record)
    return pd.DataFrame.from_records(records)


def write_dma_plots(summary: pd.DataFrame, output_dir: str | Path) -> list[Path]:
    output_dir = Path(output_dir)
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return paths
    metrics = ["storage_modulus_MPa", "loss_modulus_MPa", "tan_delta", "r2", "snr_first_harmonic"]
    for (mode, signal), group in summary.groupby(["mode", "signal"]):
        x_col = "temperature_K" if "temperature_K" in group and group["temperature_K"].nunique() > 1 else "strain"
        x_label = "Temperature (K)" if x_col == "temperature_K" else "Strain amplitude"
        for metric in metrics:
            fig, ax = plt.subplots(figsize=(5.5, 3.6))
            for parser, pg in group.groupby("parser"):
                pg = pg.sort_values(x_col)
                ax.plot(pg[x_col], pg[metric], marker="o", label=parser)
            ax.set_xlabel(x_label)
            ax.set_ylabel(metric)
            ax.set_title(f"{mode}: {signal}")
            ax.legend()
            fig.tight_layout()
            path = plot_dir / f"summary__{mode}__{signal}__{x_col}__{metric}.png"
            fig.savefig(path, dpi=180)
            plt.close(fig)
            paths.append(path)
    return paths


def write_parser_comparison_plots(
    summary: pd.DataFrame,
    output_dir: str | Path,
    mode: str = "shear_sllod_remapv",
    signal: str = "sxy_vir_MPa",
    current_parser: str = "lockin",
    legacy_parser: str = "legacy",
) -> list[Path]:
    """Write publication-style comparisons of current and legacy DMA parsers."""

    output_dir = Path(output_dir)
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return paths

    data = summary[
        (summary["mode"] == mode)
        & (summary["signal"] == signal)
        & (summary["parser"].isin([current_parser, legacy_parser]))
    ].copy()
    if data.empty:
        return paths

    labels = {
        current_parser: "DMAx2 lock-in parser",
        legacy_parser: "legacy sinusoidal fit",
    }
    colors = {
        current_parser: "#0f766e",
        legacy_parser: "#7c3aed",
    }

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8), sharex=True)
    for ax, metric, ylabel in [
        (axes[0], "storage_modulus_MPa", "Storage modulus (MPa)"),
        (axes[1], "loss_modulus_MPa", "Loss modulus (MPa)"),
    ]:
        for parser, group in data.groupby("parser"):
            group = group.sort_values("strain")
            ax.plot(
                group["strain"],
                group[metric],
                marker="o",
                linewidth=1.6,
                label=labels.get(parser, parser),
                color=colors.get(parser),
            )
        ax.set_xlabel("Strain amplitude")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
    axes[0].legend(frameon=False)
    fig.suptitle(f"{mode}: {signal}")
    fig.tight_layout()
    path = plot_dir / f"parser_comparison_moduli__{mode}__{signal}.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    paths.append(path)

    fig, ax = plt.subplots(figsize=(5.8, 3.8))
    for parser, group in data.groupby("parser"):
        group = group.sort_values("strain")
        ax.plot(
            group["strain"],
            group["tan_delta"],
            marker="o",
            linewidth=1.6,
            label=labels.get(parser, parser),
            color=colors.get(parser),
        )
    ax.set_xlabel("Strain amplitude")
    ax.set_ylabel("Loss tangent, tan(delta)")
    ax.set_title(f"{mode}: {signal}")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    path = plot_dir / f"parser_comparison_tan_delta__{mode}__{signal}.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    paths.append(path)
    return paths


def _fit_trace_for_plot(
    df: pd.DataFrame,
    signal: str,
    frequency_ghz: float,
    discard_cycles: int,
    parser: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    if parser == "fourier_projection":
        raise ValueError("Historical trace plotting does not use measured-strain Fourier phase; use compare-estimators instead.")
    period_fs = 1_000_000.0 / frequency_ghz
    if "t_ps" in df:
        t_fs_all = df["t_ps"].to_numpy(dtype=float) * 1000.0
    else:
        t_fs_all = df["step"].to_numpy(dtype=float)
    keep = t_fs_all >= discard_cycles * period_fs
    if int(np.count_nonzero(keep)) < 8:
        keep = np.ones_like(t_fs_all, dtype=bool)
    t_fs = t_fs_all[keep]
    y = df.loc[keep, signal].to_numpy(dtype=float)
    t_fit = t_fs - t_fs.min()
    include_harmonics = parser == "lockin"
    include_drift = parser == "lockin"
    x = _design(t_fit, period_fs, include_harmonics, include_drift)
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    yhat = x @ beta
    offset = 2 if include_drift else 1
    a1 = float(beta[offset])
    b1 = float(beta[offset + 1])
    phase_deg = float(np.degrees(np.arctan2(b1, a1)))
    amplitude = float(np.hypot(a1, b1))
    return t_fs, t_fit, y, yhat, beta, phase_deg, amplitude


def write_legacy_pressure_phase_plot(
    raw_file: str | Path,
    output_dir: str | Path,
    frequency_ghz: float,
    signal: str = "press_MPa",
    strain_signal: str = "strain_applied",
    discard_cycles: int = 2,
    cycles_to_plot: int = 2,
) -> Path | None:
    """Plot legacy pressure sinusoid, fit, applied strain, and phase angle."""

    raw_file = Path(raw_file)
    if not raw_file.exists():
        return None
    output_dir = Path(output_dir)
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    df = read_raw_table(raw_file)
    if signal not in df or strain_signal not in df:
        return None
    period_fs = 1_000_000.0 / frequency_ghz
    _t_fs, t_fit, y, yhat, beta, phase_deg, amplitude = _fit_trace_for_plot(
        df, signal, frequency_ghz, discard_cycles, "legacy"
    )
    keep = t_fit <= cycles_to_plot * period_fs + 1.0e-9
    t_ps = t_fit[keep] / 1000.0
    y_fluct = y[keep] - float(beta[0])
    yhat_fluct = yhat[keep] - float(beta[0])

    if "t_ps" in df:
        t_all_fs = df["t_ps"].to_numpy(dtype=float) * 1000.0
    else:
        t_all_fs = df["step"].to_numpy(dtype=float)
    strain_keep = t_all_fs >= discard_cycles * period_fs
    strain = df.loc[strain_keep, strain_signal].to_numpy(dtype=float)[keep]

    fig, ax = plt.subplots(figsize=(7.2, 4.1))
    ax.plot(t_ps, y_fluct, color="#334155", linewidth=1.0, alpha=0.70, label="pressure fluctuation")
    ax.plot(t_ps, yhat_fluct, color="#dc2626", linewidth=2.0, label="legacy sinusoidal fit")
    ax.set_xlabel("Simulation runtime after discarded cycles (ps)")
    ax.set_ylabel("Pressure fluctuation (MPa)")
    ax.grid(True, alpha=0.25)

    ax2 = ax.twinx()
    ax2.plot(t_ps, strain, color="#2563eb", linewidth=1.5, linestyle="--", label="applied strain")
    ax2.set_ylabel("Applied strain")

    handles, labels = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(handles + handles2, labels + labels2, frameon=False, loc="upper right")
    ax.text(
        0.02,
        0.95,
        f"phase angle = {phase_deg:.1f} deg\nfit amplitude = {amplitude:.3g} MPa",
        transform=ax.transAxes,
        va="top",
        ha="left",
        bbox={"facecolor": "white", "edgecolor": "#cbd5e1", "alpha": 0.9},
    )
    fig.suptitle(f"Legacy DMA pressure fit: {raw_file.stem}")
    fig.tight_layout()
    path = plot_dir / f"legacy_pressure_phase__{raw_file.stem}.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def _first_harmonic_prediction(
    t_fit_fs: np.ndarray,
    beta: np.ndarray,
    period_fs: float,
    parser: str,
) -> np.ndarray:
    """Reconstruct the baseline plus first harmonic used for moduli."""

    include_drift = parser == "lockin"
    offset = 2 if include_drift else 1
    baseline = np.full_like(t_fit_fs, float(beta[0]), dtype=float)
    if include_drift:
        baseline = baseline + float(beta[1]) * ((t_fit_fs - t_fit_fs.mean()) / period_fs)
    omega_t = 2.0 * np.pi * t_fit_fs / period_fs
    return baseline + float(beta[offset]) * np.sin(omega_t) + float(beta[offset + 1]) * np.cos(omega_t)


def write_nemd_first_harmonic_diagnostics(
    raw_file: str | Path,
    output_dir: str | Path,
    frequency_ghz: float,
    signals: Iterable[str] = ("press_MPa", "szz_vir_MPa", "sdev_MPa", "sxy_vir_MPa", "sxy_tot_MPa"),
    strain_signal: str = "strain_applied",
    discard_cycles: int = 2,
    parser: str = "lockin",
    cycles_to_plot: float | None = None,
    plot_subdir: str | Path = "plots",
) -> list[Path]:
    """Plot raw NEMD signals and the first-harmonic fit used for DMA moduli.

    The lock-in parser fits drift and higher harmonics for diagnostics, but the
    reported storage modulus, loss modulus, and loss tangent are computed from
    the first sin/cos coefficients. These plots expose that first-harmonic
    component directly.
    """

    raw_file = Path(raw_file)
    output_dir = Path(output_dir)
    if not raw_file.exists():
        return []
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return []

    df = read_raw_table(raw_file)
    metadata = parse_raw_run_metadata(raw_file)
    mode = str(metadata["mode"])
    strain = float(metadata["strain"])
    period_fs = 1_000_000.0 / frequency_ghz
    available = [signal for signal in signals if signal in df]
    if not available:
        return []

    plot_dir = output_dir / plot_subdir
    plot_dir.mkdir(parents=True, exist_ok=True)

    nrows = len(available)
    fig, axes = plt.subplots(nrows, 1, figsize=(8.4, 2.6 * nrows), sharex=True)
    if nrows == 1:
        axes = [axes]
    written: list[Path] = []

    for ax, signal in zip(axes, available):
        t_fs, t_fit, y, _yhat, beta, phase_deg, amplitude = _fit_trace_for_plot(
            df,
            signal,
            frequency_ghz,
            discard_cycles,
            parser,
        )
        first_harmonic = _first_harmonic_prediction(t_fit, beta, period_fs, parser)
        keep = np.ones_like(t_fit, dtype=bool)
        if cycles_to_plot is not None:
            keep = t_fit <= cycles_to_plot * period_fs + 1.0e-9
        t_ps = t_fit[keep] / 1000.0

        try:
            fit = fit_dma_dataframe(
                df,
                mode=mode,
                signal=signal,
                strain=strain,
                frequency_ghz=frequency_ghz,
                discard_cycles=discard_cycles,
                parser=parser,
            )
        except ValueError:
            fit = fit_dma_dataframe(
                df,
                mode=mode,
                signal=signal,
                strain=strain,
                frequency_ghz=frequency_ghz,
                discard_cycles=0,
                parser=parser,
            )
            fit.status = f"{fit.status};diagnostic_discard0"

        ax.plot(t_ps, y[keep], color="#334155", linewidth=0.8, alpha=0.62, label="raw signal")
        ax.plot(
            t_ps,
            first_harmonic[keep],
            color="#dc2626",
            linewidth=1.8,
            label="baseline + first harmonic",
        )
        ax.set_ylabel(f"{signal} (MPa)")
        ax.grid(True, alpha=0.25)
        ax.text(
            0.012,
            0.96,
            (
                f"E'={fit.storage_modulus_MPa:.3g} MPa, "
                f"E''={fit.loss_modulus_MPa:.3g} MPa, "
                f"tan(delta)={fit.tan_delta:.3g}\n"
                f"R2={fit.r2:.3f}, SNR={fit.snr_first_harmonic:.3g}, "
                f"2omega/1omega={fit.harm2_over_harm1:.3g}, "
                f"phase={phase_deg:.1f} deg, status={fit.status}"
            ),
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8.6,
            bbox={"facecolor": "white", "edgecolor": "#cbd5e1", "alpha": 0.88},
        )
        if strain_signal in df:
            t_all_fs = df["t_ps"].to_numpy(dtype=float) * 1000.0 if "t_ps" in df else df["step"].to_numpy(dtype=float)
            strain_values = np.interp(t_fs, t_all_fs, df[strain_signal].to_numpy(dtype=float))
            ax2 = ax.twinx()
            ax2.plot(t_ps, strain_values[keep], color="#2563eb", linewidth=1.0, alpha=0.7, linestyle="--")
            ax2.set_ylabel("strain")
        ax.legend(frameon=False, loc="lower right")

        single_fig, single_ax = plt.subplots(figsize=(8.2, 3.8))
        single_ax.plot(t_ps, y[keep], color="#334155", linewidth=0.8, alpha=0.62, label="raw signal")
        single_ax.plot(
            t_ps,
            first_harmonic[keep],
            color="#dc2626",
            linewidth=1.8,
            label="baseline + first harmonic",
        )
        single_ax.set_xlabel("Simulation runtime after discarded cycles (ps)")
        single_ax.set_ylabel(f"{signal} (MPa)")
        single_ax.grid(True, alpha=0.25)
        single_ax.legend(frameon=False, loc="best")
        single_ax.set_title(f"{raw_file.stem}: {signal}")
        single_fig.tight_layout()
        single_path = plot_dir / f"nemd_first_harmonic__{signal}.png"
        single_fig.savefig(single_path, dpi=220)
        plt.close(single_fig)
        written.append(single_path)

    axes[-1].set_xlabel("Simulation runtime after discarded cycles (ps)")
    temp = metadata.get("temperature_K")
    temp_label = f", T={float(temp):.3g} K" if temp is not None else ""
    fig.suptitle(f"NEMD first-harmonic diagnostics: {mode}, strain={strain:g}{temp_label}, f={frequency_ghz:g} GHz")
    fig.tight_layout()
    combined_path = plot_dir / "nemd_first_harmonic_diagnostics.png"
    fig.savefig(combined_path, dpi=220)
    plt.close(fig)
    written.insert(0, combined_path)
    return written


def _diagnostic_output_for_raw(raw_file: Path, root_dir: Path, local: bool) -> tuple[Path, Path]:
    """Choose a plot destination for one raw file without overwriting siblings."""

    if local and raw_file.parent.name == "raw":
        output_dir = raw_file.parent.parent
        raw_files = list(raw_file.parent.glob("raw__*.dat"))
        if len(raw_files) == 1:
            return output_dir, Path("plots")
        return output_dir, Path("plots") / "nemd_first_harmonic" / raw_file.stem
    return root_dir, Path("plots") / "nemd_first_harmonic" / raw_file.stem


def write_nemd_diagnostics_for_directory(
    output_dir: str | Path,
    frequency_ghz: float,
    discard_cycles: int = 2,
    parser: str = "lockin",
    signals: Iterable[str] = ("press_MPa", "szz_vir_MPa", "sdev_MPa", "sxy_vir_MPa", "sxy_tot_MPa"),
    recursive: bool = False,
    local: bool = False,
    skip_root_raw: bool = False,
    skip_existing: bool = False,
) -> pd.DataFrame:
    """Write first-harmonic diagnostics for raw traces in a DMAx2 run tree.

    Parameters
    ----------
    output_dir
        Stage directory containing a ``raw`` folder, or a parent directory when
        ``recursive=True``.
    local
        If true, plots are written next to each raw folder's parent run
        directory. This is useful for temperature, strain, replicate, and
        frequency subdirectories. When a raw folder has multiple traces, each
        trace gets a stem-named subdirectory to avoid overwriting plots.
    skip_root_raw
        If true, skip raw traces directly inside ``output_dir/raw``. This avoids
        duplicating the aggregate diagnostics produced by ``dmax2 parse`` when
        doing a second recursive pass to populate per-run directories.
    skip_existing
        If true, leave traces with an existing combined diagnostic plot in place.
    """

    output_dir = Path(output_dir)
    pattern = "**/raw/raw__*.dat" if recursive else "raw/raw__*.dat"
    rows: list[dict] = []
    for raw_file in sorted(output_dir.glob(pattern)):
        if skip_root_raw and raw_file.parent == output_dir / "raw":
            continue
        metadata = parse_raw_run_metadata(raw_file)
        file_frequency = metadata["frequency_ghz"] or frequency_ghz
        destination, plot_subdir = _diagnostic_output_for_raw(raw_file, output_dir, local)
        try:
            existing_combined = destination / plot_subdir / "nemd_first_harmonic_diagnostics.png"
            if skip_existing and existing_combined.exists():
                paths = [existing_combined]
                status = "existing"
            else:
                paths = write_nemd_first_harmonic_diagnostics(
                    raw_file,
                    destination,
                    file_frequency,
                    signals=signals,
                    discard_cycles=discard_cycles,
                    parser=parser,
                    plot_subdir=plot_subdir,
                )
                status = "ok" if paths else "no_plots"
            error = ""
        except Exception as exc:  # pragma: no cover - batch plotting should not block parsing
            paths = []
            status = "failed"
            error = str(exc)
        rows.append(
            {
                **metadata,
                "raw_file": str(raw_file),
                "output_dir": str(destination),
                "plot_subdir": str(plot_subdir),
                "frequency_ghz_used": file_frequency,
                "n_plots": len(paths),
                "combined_plot": str(paths[0]) if paths else "",
                "status": status,
                "error": error,
            }
        )
    index = pd.DataFrame.from_records(rows)
    if rows:
        index_dir = output_dir / "plots" / "nemd_first_harmonic"
        index_dir.mkdir(parents=True, exist_ok=True)
        index.to_csv(index_dir / "nemd_first_harmonic_diagnostics_index.csv", index=False)
    return index
