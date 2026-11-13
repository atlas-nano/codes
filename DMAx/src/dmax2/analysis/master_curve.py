"""Master-curve construction utilities.

The built-in path applies explicit WLF/Arrhenius shifts. An optional backend
uses Kyle Lennon's `mastercurves` package for Gaussian-process assisted
superposition.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import MasterCurveSpec


@dataclass
class ShiftParameters:
    model: str
    reference_temperature_k: float
    c1: float
    c2: float

    def as_dict(self) -> dict:
        return self.__dict__.copy()


@dataclass
class MastercurvesResult:
    master_curve: pd.DataFrame
    shift_table: pd.DataFrame
    loss: float | None


PEAK_STATUS_COLUMNS = [
    "temperature_K",
    "peak_frequency_ghz",
    "peak_value",
    "peak_status",
    "interior",
    "lower_edge",
    "upper_edge",
    "failed_qc",
    "included_in_fit",
    "fit_eligible",
    "used_in_fit",
    "exclusion_reason",
]


def assess_frequency_sweep_peaks(
    summary: pd.DataFrame,
    metric: str = "loss_modulus_MPa",
    temperature_col: str = "temperature_K",
    frequency_col: str = "frequency_ghz",
    status_col: str = "status",
    accepted_statuses: tuple[str, ...] = ("ok", "scratch_refit"),
    minimum_points: int = 3,
) -> pd.DataFrame:
    """Classify each temperature's sampled maximum without extrapolating it.

    A maximum at either sampled frequency limit is censored and is never
    included in a temperature-dependent peak fit. Failed-QC rows are retained
    in the audit table so their exclusion is traceable.
    """

    required = [temperature_col, frequency_col, metric]
    missing = [column for column in required if column not in summary]
    if missing:
        raise ValueError(f"Peak assessment is missing required columns: {missing}")

    rows: list[dict] = []
    for temperature, raw_group in summary.groupby(temperature_col, sort=True):
        group = raw_group.replace([np.inf, -np.inf], np.nan).dropna(subset=[frequency_col, metric]).copy()
        group = group[group[frequency_col] > 0]
        if status_col in group:
            qc_mask = group[status_col].astype(str).isin(accepted_statuses)
        else:
            qc_mask = pd.Series(True, index=group.index)
        valid = group[qc_mask].sort_values(frequency_col)

        failed_qc = bool((~qc_mask).any())
        peak_frequency = np.nan
        peak_value = np.nan
        lower_edge = False
        upper_edge = False
        interior = False
        reasons: list[str] = []

        if len(valid) < minimum_points:
            reasons.append(f"fewer_than_{minimum_points}_qc_points")
        else:
            peak_row = valid.loc[valid[metric].idxmax()]
            peak_frequency = float(peak_row[frequency_col])
            peak_value = float(peak_row[metric])
            min_frequency = float(valid[frequency_col].min())
            max_frequency = float(valid[frequency_col].max())
            lower_edge = bool(np.isclose(peak_frequency, min_frequency))
            upper_edge = bool(np.isclose(peak_frequency, max_frequency))
            interior = not (lower_edge or upper_edge)
            if lower_edge:
                reasons.append("maximum_at_lower_frequency_edge")
            if upper_edge:
                reasons.append("maximum_at_upper_frequency_edge")
        if failed_qc:
            reasons.append("one_or_more_points_failed_qc")

        included = bool(interior and not failed_qc and len(valid) >= minimum_points)
        if included:
            peak_status = "interior"
            exclusion_reason = ""
        elif len(valid) < minimum_points:
            peak_status = "failed_qc"
            exclusion_reason = ";".join(reasons)
        elif lower_edge:
            peak_status = "lower_edge"
            exclusion_reason = ";".join(reasons)
        elif upper_edge:
            peak_status = "upper_edge"
            exclusion_reason = ";".join(reasons)
        else:
            peak_status = "failed_qc"
            exclusion_reason = ";".join(reasons) or "failed_qc"

        rows.append(
            {
                "temperature_K": float(temperature),
                "peak_frequency_ghz": peak_frequency,
                "peak_value": peak_value,
                "peak_status": peak_status,
                "interior": interior,
                "lower_edge": lower_edge,
                "upper_edge": upper_edge,
                "failed_qc": failed_qc,
                "included_in_fit": included,
                # `included_in_fit` is retained as a legacy alias for
                # eligibility.  It does not claim that a fit was performed.
                "fit_eligible": included,
                "used_in_fit": False,
                "exclusion_reason": exclusion_reason,
                "n_points_total": int(len(group)),
                "n_points_qc": int(len(valid)),
                "metric": metric,
            }
        )
    return pd.DataFrame(rows)


def require_interior_peaks(peak_table: pd.DataFrame, minimum_interior_peaks: int = 3) -> pd.DataFrame:
    """Return fit-ready peaks or fail with an explicit censoring message."""

    eligibility = "fit_eligible" if "fit_eligible" in peak_table else "included_in_fit"
    if eligibility not in peak_table:
        raise ValueError("Peak table is missing the fit_eligible/included_in_fit field.")
    included = peak_table[peak_table[eligibility].astype(bool)].copy()
    if len(included) < minimum_interior_peaks:
        raise ValueError(
            "Temperature-dependent loss-peak fitting requires at least "
            f"{minimum_interior_peaks} interior, QC-passing maxima; found {len(included)}. "
            "Edge-censored and failed-QC maxima were excluded."
        )
    return included


def _coerce_backend_loss(loss: object) -> float | None:
    if loss is None:
        return None
    try:
        arr = np.asarray(loss, dtype=float).ravel()
    except (TypeError, ValueError):
        return None
    if arr.size == 0:
        return None
    return float(arr[-1])


def wlf_log10_shift(temperature_k: np.ndarray, reference_temperature_k: float, c1: float, c2: float) -> np.ndarray:
    dt = temperature_k - reference_temperature_k
    return -c1 * dt / (c2 + dt)


def arrhenius_log10_shift(temperature_k: np.ndarray, reference_temperature_k: float, ea_kj_mol: float) -> np.ndarray:
    r = 8.314462618e-3
    return ea_kj_mol / (2.303 * r) * (1.0 / temperature_k - 1.0 / reference_temperature_k)


def build_master_curve(
    summary: pd.DataFrame,
    reference_temperature_k: float,
    frequency_col: str = "frequency_ghz",
    temperature_col: str = "temperature_K",
    log10_shift_col: str = "log10_aT",
) -> pd.DataFrame:
    """Apply precomputed shifts and return reduced-frequency data."""

    df = summary.copy()
    if log10_shift_col not in df:
        df[log10_shift_col] = 0.0
    df["reduced_frequency_ghz"] = df[frequency_col] * np.power(10.0, df[log10_shift_col])
    df["reference_temperature_K"] = reference_temperature_k
    return df.sort_values(["reduced_frequency_ghz", temperature_col])


def assign_wlf_shifts(
    summary: pd.DataFrame,
    reference_temperature_k: float,
    c1: float = 8.86,
    c2: float = 101.6,
    temperature_col: str = "temperature_K",
) -> tuple[pd.DataFrame, ShiftParameters]:
    df = summary.copy()
    df["log10_aT"] = wlf_log10_shift(df[temperature_col].to_numpy(float), reference_temperature_k, c1, c2)
    return df, ShiftParameters("wlf", reference_temperature_k, c1, c2)


def _clean_summary(
    summary: pd.DataFrame,
    metric: str,
    signal: str = "sxy_vir_MPa",
    parser: str = "lockin",
) -> pd.DataFrame:
    required = ["temperature_K", "frequency_ghz", metric]
    missing = [col for col in required if col not in summary]
    if missing:
        raise ValueError(f"Master-curve analysis is missing required columns: {missing}")
    df = summary.copy()
    if "signal" in df:
        df = df[df["signal"] == signal]
    if "parser" in df:
        df = df[df["parser"] == parser]
    if "status" in df:
        df = df[df["status"] == "ok"]
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=required)
    df = df[(df["frequency_ghz"] > 0) & (df[metric] > 0)]
    if df.empty:
        raise ValueError(f"No positive {metric} rows are available for master-curve analysis.")
    return df.sort_values(["temperature_K", "frequency_ghz"]).reset_index(drop=True)


def _score_horizontal_shift(ref_x: np.ndarray, ref_y: np.ndarray, x: np.ndarray, y: np.ndarray, shift: float) -> float:
    shifted_min = float(np.nanmin(x + shift))
    shifted_max = float(np.nanmax(x + shift))
    # `np.arange` can represent its nominal zero shift as a tiny nonzero
    # value.  Preserve exact curve overlap at the default search grid.
    tolerance = 1.0e-10
    common = ref_x[(ref_x >= shifted_min - tolerance) & (ref_x <= shifted_max + tolerance)]
    if len(common) < 3:
        return np.inf
    shifted_y = np.interp(common - shift, x, y)
    ref_common_y = np.interp(common, ref_x, ref_y)
    return float(np.nanmean((shifted_y - ref_common_y) ** 2))


def assign_tts_shifts(
    summary: pd.DataFrame,
    reference_temperature_k: float,
    metric: str = "storage_modulus_MPa",
    max_shift_decades: float = 18.0,
    shift_step_decades: float = 0.025,
    signal: str = "sxy_vir_MPa",
    parser: str = "lockin",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Assign free horizontal shifts by directly optimizing curve collapse.

    The reference curve is the simulated temperature nearest
    ``reference_temperature_k``. This method is intentionally simple and
    deterministic so it can serve as a transparent TTS baseline alongside WLF
    and the optional :mod:`mastercurves` backend.
    """

    # Keep the caller-selected signal/parser.  In particular, the axial
    # `szz_vir_MPa` pathway must not be silently re-filtered as shear.
    df = _clean_summary(summary, metric, signal=signal, parser=parser)
    log_df = df.assign(log10_frequency=np.log10(df["frequency_ghz"]), log10_metric=np.log10(df[metric]))
    temperatures = np.array(sorted(log_df["temperature_K"].unique()), dtype=float)
    if len(temperatures) < 2:
        raise ValueError("TTS shifting requires at least two temperatures.")
    ref_temp = float(temperatures[int(np.argmin(np.abs(temperatures - reference_temperature_k)))])
    ref = log_df[log_df["temperature_K"] == ref_temp].sort_values("log10_frequency")
    ref_x = ref["log10_frequency"].to_numpy(float)
    ref_y = ref["log10_metric"].to_numpy(float)
    grid = np.arange(-max_shift_decades, max_shift_decades + shift_step_decades / 2.0, shift_step_decades)

    rows: list[dict] = []
    for temp, group in log_df.groupby("temperature_K"):
        group = group.sort_values("log10_frequency")
        x = group["log10_frequency"].to_numpy(float)
        y = group["log10_metric"].to_numpy(float)
        if np.isclose(float(temp), ref_temp):
            shift = 0.0
            rmse = 0.0
        else:
            scores = np.array([_score_horizontal_shift(ref_x, ref_y, x, y, s) for s in grid])
            finite = np.flatnonzero(np.isfinite(scores))
            if finite.size == 0:
                raise ValueError(
                    "Free TTS shifting found no admissible frequency overlap for "
                    f"{temp:g} K relative to {ref_temp:g} K; no shift was assigned."
                )
            idx = int(finite[np.argmin(scores[finite])])
            shift = float(grid[idx])
            rmse = float(np.sqrt(scores[idx])) if np.isfinite(scores[idx]) else np.nan
        rows.append(
            {
                "temperature_K": float(temp),
                "reference_temperature_K": reference_temperature_k,
                "reference_state_temperature_K": ref_temp,
                "log10_aT": shift,
                "rmse_log10_metric_vs_reference": rmse,
                "shift_model": "tts_free",
                "metric": metric,
            }
        )
    shifts = pd.DataFrame(rows)
    shifted = df.merge(shifts[["temperature_K", "log10_aT"]], on="temperature_K", how="left")
    shifted["shift_model"] = "tts_free"
    shifted["reference_temperature_K"] = reference_temperature_k
    shifted["reference_state_temperature_K"] = ref_temp
    return shifted, shifts


def fit_arrhenius_shift_from_table(
    shift_table: pd.DataFrame,
    reference_temperature_k: float,
) -> float:
    """Fit an Arrhenius activation energy to free log10(aT) shifts."""

    data = shift_table[["temperature_K", "log10_aT"]].replace([np.inf, -np.inf], np.nan).dropna()
    x = 1.0 / data["temperature_K"].to_numpy(float) - 1.0 / float(reference_temperature_k)
    y = data["log10_aT"].to_numpy(float)
    mask = np.abs(x) > 1e-12
    if mask.sum() < 2:
        return float("nan")
    slope = float(np.sum(x[mask] * y[mask]) / np.sum(x[mask] * x[mask]))
    r = 8.314462618e-3
    return slope * 2.303 * r


def build_master_curve_with_mastercurves(
    summary: pd.DataFrame,
    reference_temperature_k: float,
    metric: str = "storage_modulus_MPa",
    frequency_col: str = "frequency_ghz",
    temperature_col: str = "temperature_K",
    include_vertical_shift: bool = False,
    fixed_noise: float = 0.04,
) -> MastercurvesResult:
    """Build a master curve with the optional `mastercurves` package.

    Data are supplied in log10(frequency) and log10(metric) coordinates because
    the package's default `Multiply()` transform assumes logarithmic data for
    multiplicative horizontal shifts.
    """

    try:
        from mastercurves import MasterCurve
        from mastercurves.transforms import Multiply
    except ImportError as exc:
        raise RuntimeError(
            "The mastercurves backend requires `pip install mastercurves` "
            "or `pip install -e .[mastercurves]`."
        ) from exc

    df = summary[[temperature_col, frequency_col, metric]].replace([np.inf, -np.inf], np.nan).dropna()
    df = df[df[metric] > 0]
    df = df[df[frequency_col] > 0]
    if df[temperature_col].nunique() < 2:
        raise ValueError("mastercurves backend requires at least two temperature states.")
    xdata: list[np.ndarray] = []
    ydata: list[np.ndarray] = []
    states: list[float] = []
    for temp, group in df.groupby(temperature_col):
        if len(group) < 2:
            continue
        group = group.sort_values(frequency_col)
        xdata.append(np.log10(group[frequency_col].to_numpy(float)))
        ydata.append(np.log10(group[metric].to_numpy(float)))
        states.append(float(temp))
    if len(states) < 2:
        raise ValueError("mastercurves backend requires at least two states with two points each.")

    mc = MasterCurve(fixed_noise=fixed_noise)
    mc.add_data(xdata, ydata, states)
    mc.add_htransform(Multiply())
    if include_vertical_shift:
        mc.add_vtransform(Multiply())
    loss = mc.superpose()
    mc.change_ref(reference_temperature_k)

    rows: list[dict] = []
    for temp, xs, ys in zip(states, mc.xtransformed, mc.ytransformed):
        for x, y in zip(xs, ys):
            rows.append(
                {
                    temperature_col: temp,
                    "log10_reduced_frequency": float(x),
                    "reduced_frequency_ghz": float(10.0**x),
                    f"log10_{metric}": float(y),
                    metric: float(10.0**y),
                    "reference_temperature_K": reference_temperature_k,
                    "shift_model": "mastercurves",
                }
            )
    try:
        shift_table = mc.output_table()
    except Exception:
        shift_table = pd.DataFrame()
    if not shift_table.empty:
        shift_table["reference_temperature_K"] = reference_temperature_k
        shift_table["shift_model"] = "mastercurves"
    return MastercurvesResult(pd.DataFrame(rows), shift_table, _coerce_backend_loss(loss))


def _linear_extrapolate(x: np.ndarray, y: np.ndarray, x_new: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    unique_x, unique_idx = np.unique(x, return_index=True)
    x = unique_x
    y = y[unique_idx]
    if len(x) < 2:
        return np.full_like(x_new, y[0] if len(y) else np.nan, dtype=float)
    out = np.interp(x_new, x, y)
    low = x_new < x[0]
    high = x_new > x[-1]
    low_slope = (y[1] - y[0]) / (x[1] - x[0])
    high_slope = (y[-1] - y[-2]) / (x[-1] - x[-2])
    out[low] = y[0] + low_slope * (x_new[low] - x[0])
    out[high] = y[-1] + high_slope * (x_new[high] - x[-1])
    return out


def _write_extrapolated_hz(
    master: pd.DataFrame,
    output_dir: Path,
    model: str,
    metric: str,
    min_hz: float = 1.0e-3,
    n_points: int = 360,
    anchor_reference_source: str = "unspecified",
    enabled: bool = False,
) -> Path | None:
    if not enabled:
        return None
    data = master[["reduced_frequency_ghz", metric]].replace([np.inf, -np.inf], np.nan).dropna()
    data = data[(data["reduced_frequency_ghz"] > 0) & (data[metric] > 0)]
    if len(data) < 2:
        return None
    x = np.log10(data["reduced_frequency_ghz"].to_numpy(float) * 1.0e9)
    y = np.log10(data[metric].to_numpy(float))
    x_min = float(np.log10(min_hz))
    x_max = float(np.nanmax(x))
    if x_max <= x_min:
        x_max = x_min + 1.0
    x_new = np.linspace(x_min, x_max, n_points)
    y_new = _linear_extrapolate(x, y, x_new)
    out = pd.DataFrame(
        {
            "shift_model": model,
            "metric": metric,
            "frequency_Hz": np.power(10.0, x_new),
            "frequency_GHz": np.power(10.0, x_new) / 1.0e9,
            metric: np.power(10.0, y_new),
            "log10_frequency_Hz": x_new,
            f"log10_{metric}": y_new,
            "extrapolated_below_data": x_new < float(np.nanmin(x)),
            "frequency_domain_status": np.where(x_new < float(np.nanmin(x)), "extrapolation", "interpolation"),
            "anchor_reference_source": anchor_reference_source,
            "data_min_frequency_Hz": float(np.power(10.0, np.nanmin(x))),
            "data_max_frequency_Hz": float(np.power(10.0, np.nanmax(x))),
        }
    )
    path = output_dir / f"master_curve_{model}_{metric}_extrapolated_to_hz.csv"
    out.to_csv(path, index=False)
    return path


def _write_experimental_window_plot(
    output_dir: Path,
    metric: str,
    min_hz: float,
    max_hz: float,
) -> Path | None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    files = sorted(output_dir.glob(f"master_curve_*_{metric}_extrapolated_to_hz.csv"))
    frames: list[pd.DataFrame] = []
    for path in files:
        df = pd.read_csv(path)
        if "shift_model" not in df or metric not in df:
            continue
        frames.append(df)
    if not frames:
        return None
    data = pd.concat(frames, ignore_index=True)
    data = data.replace([np.inf, -np.inf], np.nan).dropna(subset=["frequency_Hz", metric])
    data = data[(data["frequency_Hz"] > 0) & (data[metric] > 0)]
    if data.empty:
        return None

    plot_dir = output_dir / "plots"
    plot_dir.mkdir(exist_ok=True)
    label = {
        "storage_modulus_MPa": "Storage modulus (M′/E′/G′ depends on loading mode; MPa)",
        "loss_modulus_MPa": "Loss modulus, E'' or G'' (MPa)",
        "tan_delta": "Loss tangent, tan(delta)",
    }.get(metric, metric)
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    for model, group in data.groupby("shift_model"):
        group = group.sort_values("frequency_Hz")
        ax.plot(group["frequency_Hz"], group[metric], linewidth=1.7, label=model)
    ax.axvspan(min_hz, max_hz, color="#dbeafe", alpha=0.35, label="experimental DMA window")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(min_hz, max_hz)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel(label)
    ax.set_title(f"Low-frequency extrapolated master curves: {label}")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    path = plot_dir / f"master_curve_experimental_window_{metric}.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def write_master_curve_analysis_report(
    summary_file: str | Path,
    output_dir: str | Path,
    spec: MasterCurveSpec | None = None,
    signal: str = "sxy_vir_MPa",
    parser: str = "lockin",
) -> dict[str, Path]:
    """Write master curves using WLF, free TTS, Arrhenius, and mastercurves.

    The output includes raw frequency sweeps, shifted master curves, shift
    tables, backend diagnostics, and Hz-extrapolated CSV files. Extrapolated
    values are clearly marked so they can be separated from interpolated data
    when comparing with experimental low-frequency measurements.
    """

    spec = spec or MasterCurveSpec()
    summary = pd.read_csv(summary_file)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(exist_ok=True)

    metrics = ["storage_modulus_MPa", "loss_modulus_MPa", "tan_delta"]
    available_metrics = [metric for metric in metrics if metric in summary]
    if spec.metric in summary and spec.metric not in available_metrics:
        available_metrics.insert(0, spec.metric)
    if not available_metrics:
        raise ValueError("No modulus columns are available for master-curve construction.")

    filtered_by_metric: dict[str, pd.DataFrame] = {}
    for metric in available_metrics:
        filtered_by_metric[metric] = _clean_summary(summary, metric, signal=signal, parser=parser)

    artifacts: dict[str, Path] = {}
    model_frames: list[pd.DataFrame] = []
    shift_frames: list[pd.DataFrame] = []
    diagnostics: list[dict] = []
    extrapolated_paths: list[Path] = []

    if "loss_modulus_MPa" in summary:
        peak_source = summary.copy()
        if "signal" in peak_source:
            peak_source = peak_source[peak_source["signal"] == signal]
        if "parser" in peak_source:
            peak_source = peak_source[peak_source["parser"] == parser]
        peak_table = assess_frequency_sweep_peaks(peak_source)
        peak_path = output_dir / "loss_peak_status_by_temperature.csv"
        peak_table.to_csv(peak_path, index=False)
        artifacts["loss_peak_status"] = peak_path
        try:
            included_peaks = require_interior_peaks(peak_table)
            diagnostics.append(
                {
                    "shift_model": "temperature_dependent_loss_peaks",
                    "metric": "loss_modulus_MPa",
                    "n_points": len(included_peaks),
                    "fit_status": "ready",
                    "note": "Only interior, QC-passing sampled maxima are fit-eligible.",
                }
            )
        except ValueError as exc:
            diagnostics.append(
                {
                    "shift_model": "temperature_dependent_loss_peaks",
                    "metric": "loss_modulus_MPa",
                    "n_points": int(peak_table["included_in_fit"].sum()),
                    "fit_status": "failed_insufficient_interior_peaks",
                    "backend_error": str(exc),
                    "note": "No peak fit was attempted; censored maxima remain raw-response data only.",
                }
            )

    for metric, clean in filtered_by_metric.items():
        wlf_shifted, wlf_params = assign_wlf_shifts(clean, spec.reference_temperature_k)
        wlf_master = build_master_curve(wlf_shifted, spec.reference_temperature_k)
        wlf_master["shift_model"] = "wlf_default"
        wlf_master["metric"] = metric
        model_frames.append(wlf_master)
        shift_frames.append(pd.DataFrame([{**wlf_params.as_dict(), "metric": metric, "shift_model": "wlf_default"}]))
        diagnostics.append(
            {
                "shift_model": "wlf_default",
                "metric": metric,
                "n_points": len(wlf_master),
                "min_reduced_frequency_Hz": float(wlf_master["reduced_frequency_ghz"].min() * 1.0e9),
                "max_reduced_frequency_Hz": float(wlf_master["reduced_frequency_ghz"].max() * 1.0e9),
                "note": "Default WLF constants; inspect very large shifts below Tg before using extrapolations.",
            }
        )
        path = _write_extrapolated_hz(
            wlf_master,
            output_dir,
            "wlf_default",
            metric,
            min_hz=spec.experimental_min_frequency_hz,
            n_points=spec.extrapolation_points,
            anchor_reference_source=spec.anchor_reference_source,
            enabled=spec.enable_low_frequency_extrapolation,
        )
        if path and spec.enable_low_frequency_extrapolation:
            extrapolated_paths.append(path)

        try:
            tts_shifted, tts_shifts = assign_tts_shifts(
                clean, spec.reference_temperature_k, metric=metric, signal=signal, parser=parser
            )
        except ValueError as exc:
            diagnostics.append({"shift_model": "tts_free", "metric": metric, "n_points": 0,
                                "fit_status": "failed_no_admissible_overlap", "backend_error": str(exc),
                                "note": "Raw response is retained; no invalid free-TTS shifts were used."})
            continue
        tts_master = build_master_curve(tts_shifted, spec.reference_temperature_k)
        tts_master["shift_model"] = "tts_free"
        tts_master["metric"] = metric
        model_frames.append(tts_master)
        shift_frames.append(tts_shifts)
        diagnostics.append(
            {
                "shift_model": "tts_free",
                "metric": metric,
                "n_points": len(tts_master),
                "median_rmse_log10_metric_vs_reference": float(tts_shifts["rmse_log10_metric_vs_reference"].median()),
                "reference_state_temperature_K": float(tts_shifts["reference_state_temperature_K"].iloc[0]),
                "min_reduced_frequency_Hz": float(tts_master["reduced_frequency_ghz"].min() * 1.0e9),
                "max_reduced_frequency_Hz": float(tts_master["reduced_frequency_ghz"].max() * 1.0e9),
                "note": "Free horizontal shifts are the most data-driven built-in TTS baseline.",
            }
        )
        path = _write_extrapolated_hz(
            tts_master,
            output_dir,
            "tts_free",
            metric,
            min_hz=spec.experimental_min_frequency_hz,
            n_points=spec.extrapolation_points,
            anchor_reference_source=spec.anchor_reference_source,
            enabled=spec.enable_low_frequency_extrapolation,
        )
        if path and spec.enable_low_frequency_extrapolation:
            extrapolated_paths.append(path)

        ea = fit_arrhenius_shift_from_table(tts_shifts, spec.reference_temperature_k)
        if np.isfinite(ea):
            arr = clean.copy()
            arr["log10_aT"] = arrhenius_log10_shift(
                arr["temperature_K"].to_numpy(float), spec.reference_temperature_k, ea
            )
            arr_master = build_master_curve(arr, spec.reference_temperature_k)
            arr_master["shift_model"] = "arrhenius_fit_to_tts"
            arr_master["metric"] = metric
            model_frames.append(arr_master)
            shift_frames.append(
                pd.DataFrame(
                    [
                        {
                            "shift_model": "arrhenius_fit_to_tts",
                            "metric": metric,
                            "reference_temperature_K": spec.reference_temperature_k,
                            "ea_kj_mol": ea,
                        }
                    ]
                )
            )
            diagnostics.append(
                {
                    "shift_model": "arrhenius_fit_to_tts",
                    "metric": metric,
                    "n_points": len(arr_master),
                    "ea_kj_mol": ea,
                    "min_reduced_frequency_Hz": float(arr_master["reduced_frequency_ghz"].min() * 1.0e9),
                    "max_reduced_frequency_Hz": float(arr_master["reduced_frequency_ghz"].max() * 1.0e9),
                    "note": "Arrhenius fit smooths free TTS shifts but may be poor through Tg.",
                }
            )
            path = _write_extrapolated_hz(
                arr_master,
                output_dir,
                "arrhenius_fit_to_tts",
                metric,
                min_hz=spec.experimental_min_frequency_hz,
                n_points=spec.extrapolation_points,
                anchor_reference_source=spec.anchor_reference_source,
                enabled=spec.enable_low_frequency_extrapolation,
            )
            if path and spec.enable_low_frequency_extrapolation:
                extrapolated_paths.append(path)

        try:
            mc_result = build_master_curve_with_mastercurves(
                clean,
                reference_temperature_k=spec.reference_temperature_k,
                metric=metric,
                include_vertical_shift=spec.include_vertical_shift,
                fixed_noise=spec.fixed_noise,
            )
            mc_master = mc_result.master_curve.copy()
            mc_master["shift_model"] = "mastercurves"
            mc_master["metric"] = metric
            model_frames.append(mc_master)
            if not mc_result.shift_table.empty:
                table = mc_result.shift_table.copy()
                table["metric"] = metric
                shift_frames.append(table)
            diagnostics.append(
                {
                    "shift_model": "mastercurves",
                    "metric": metric,
                    "n_points": len(mc_master),
                    "backend_loss": mc_result.loss,
                    "min_reduced_frequency_Hz": float(mc_master["reduced_frequency_ghz"].min() * 1.0e9),
                    "max_reduced_frequency_Hz": float(mc_master["reduced_frequency_ghz"].max() * 1.0e9),
                    "note": "Automated mastercurves backend; inspect shifts and loss for overfitting.",
                }
            )
            path = _write_extrapolated_hz(
                mc_master,
                output_dir,
                "mastercurves",
                metric,
                min_hz=spec.experimental_min_frequency_hz,
                n_points=spec.extrapolation_points,
                anchor_reference_source=spec.anchor_reference_source,
                enabled=spec.enable_low_frequency_extrapolation,
            )
            if path and spec.enable_low_frequency_extrapolation:
                extrapolated_paths.append(path)
        except Exception as exc:
            diagnostics.append(
                {
                    "shift_model": "mastercurves",
                    "metric": metric,
                    "n_points": 0,
                    "backend_error": str(exc),
                    "note": "mastercurves backend failed or is not installed.",
                }
            )

    combined = pd.concat(model_frames, ignore_index=True) if model_frames else pd.DataFrame()
    shifts = pd.concat(shift_frames, ignore_index=True, sort=False) if shift_frames else pd.DataFrame()
    diagnostic_df = pd.DataFrame(diagnostics)

    combined_path = output_dir / "master_curve_all_methods.csv"
    shifts_path = output_dir / "master_curve_shift_parameters_all_methods.csv"
    diagnostics_path = output_dir / "master_curve_method_diagnostics.csv"
    combined.to_csv(combined_path, index=False)
    shifts.to_csv(shifts_path, index=False)
    diagnostic_df.to_csv(diagnostics_path, index=False)
    artifacts.update({"master_curves": combined_path, "shifts": shifts_path, "diagnostics": diagnostics_path})

    try:
        import matplotlib.pyplot as plt
        from matplotlib import colors as mpl_colors
        from matplotlib.cm import ScalarMappable

        label_map = {
            "storage_modulus_MPa": "Storage modulus (M′/E′/G′ depends on loading mode; MPa)",
            "loss_modulus_MPa": "Loss modulus (M″/E″/G″ depends on loading mode; MPa)",
            "tan_delta": "Loss tangent, tan(delta)",
        }

        fig, axes = plt.subplots(len(available_metrics), 1, figsize=(8.4, 3.8 * len(available_metrics)), squeeze=False)
        all_temps = pd.concat(filtered_by_metric.values(), ignore_index=True)["temperature_K"].dropna()
        temp_norm = mpl_colors.Normalize(vmin=float(all_temps.min()), vmax=float(all_temps.max()))
        temp_cmap = plt.get_cmap("viridis")
        for ax, metric in zip(axes[:, 0], available_metrics):
            data = filtered_by_metric[metric]
            for temp, group in data.groupby("temperature_K"):
                group = group.sort_values("frequency_ghz")
                ax.plot(
                    group["frequency_ghz"] * 1.0e9,
                    group[metric],
                    marker="o",
                    linewidth=0.9,
                    alpha=0.62,
                    color=temp_cmap(temp_norm(float(temp))),
                )
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel("Frequency (Hz)")
            ax.set_ylabel(label_map.get(metric, metric))
            ax.set_title(f"Raw {label_map.get(metric, metric)} sweeps by temperature")
            ax.grid(True, which="both", alpha=0.25)
        sm = ScalarMappable(norm=temp_norm, cmap=temp_cmap)
        sm.set_array([])
        fig.subplots_adjust(right=0.84, hspace=0.55)
        cax = fig.add_axes([0.87, 0.18, 0.025, 0.66])
        cbar = fig.colorbar(sm, cax=cax)
        cbar.set_label("Temperature (K)")
        path = plot_dir / "master_curve_raw_sweeps.png"
        fig.savefig(path, dpi=220)
        plt.close(fig)
        artifacts["raw_sweeps_plot"] = path

        for metric in available_metrics:
            fig, ax = plt.subplots(figsize=(8.4, 5.0))
            for model, group in combined[combined.get("metric", metric) == metric].groupby("shift_model"):
                if metric not in group:
                    continue
                group = group.replace([np.inf, -np.inf], np.nan).dropna(subset=["reduced_frequency_ghz", metric])
                group = group[(group["reduced_frequency_ghz"] > 0) & (group[metric] > 0)]
                ax.scatter(group["reduced_frequency_ghz"] * 1.0e9, group[metric], s=18, alpha=0.62, label=model)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel("Reduced frequency (Hz)")
            ax.set_ylabel(label_map.get(metric, metric))
            ax.set_title(f"Master-curve method comparison: {label_map.get(metric, metric)}")
            ax.grid(True, which="both", alpha=0.25)
            ax.legend(frameon=False, fontsize=8)
            fig.tight_layout()
            path = plot_dir / f"master_curve_methods_{metric}.png"
            fig.savefig(path, dpi=220)
            plt.close(fig)
            artifacts[f"method_plot_{metric}"] = path

            exp_path = _write_experimental_window_plot(
                output_dir,
                metric,
                spec.experimental_min_frequency_hz,
                spec.experimental_max_frequency_hz,
            )
            if exp_path:
                artifacts[f"experimental_window_plot_{metric}"] = exp_path

        span_metrics = [metric for metric in available_metrics if metric in combined]
        if span_metrics:
            fig, axes = plt.subplots(len(span_metrics), 1, figsize=(9.2, 3.9 * len(span_metrics)), squeeze=False)
            sim_freq_hz = summary["frequency_ghz"].replace([np.inf, -np.inf], np.nan).dropna() * 1.0e9
            sim_min_hz = max(float(sim_freq_hz.min()), spec.experimental_min_frequency_hz)
            sim_max_hz = float(sim_freq_hz.max())
            xmax = max(sim_max_hz, spec.experimental_max_frequency_hz)
            for idx, (ax, metric) in enumerate(zip(axes[:, 0], span_metrics)):
                metric_curves = combined[combined.get("metric", metric) == metric]
                for model, group in metric_curves.groupby("shift_model"):
                    group = group.replace([np.inf, -np.inf], np.nan).dropna(subset=["reduced_frequency_ghz", metric])
                    group = group[(group["reduced_frequency_ghz"] > 0) & (group[metric] > 0)]
                    group = group.sort_values("reduced_frequency_ghz")
                    x_hz = group["reduced_frequency_ghz"] * 1.0e9
                    visible = (x_hz >= spec.experimental_min_frequency_hz) & (x_hz <= xmax)
                    if visible.any():
                        ax.plot(x_hz[visible], group.loc[visible, metric], linewidth=1.5, alpha=0.78, label=model)
                raw = filtered_by_metric[metric]
                ax.scatter(
                    raw["frequency_ghz"] * 1.0e9,
                    raw[metric],
                    c=raw["temperature_K"],
                    cmap=temp_cmap,
                    norm=temp_norm,
                    s=20,
                    marker="x",
                    linewidths=0.9,
                    label="raw OK simulations",
                    zorder=4,
                )
                ax.axvspan(
                    spec.experimental_min_frequency_hz,
                    spec.experimental_max_frequency_hz,
                    color="#dbeafe",
                    alpha=0.32,
                    label="experimental DMA window" if idx == 0 else None,
                )
                ax.axvspan(
                    sim_min_hz,
                    sim_max_hz,
                    color="#fef3c7",
                    alpha=0.22,
                    label="simulated frequency window" if idx == 0 else None,
                )
                for freq_hz in sorted(set(sim_freq_hz.round(6))):
                    ax.axvline(freq_hz, color="0.78", linewidth=0.45, alpha=0.25, zorder=0)
                ax.set_xscale("log")
                ax.set_yscale("log")
                ax.set_xlim(spec.experimental_min_frequency_hz, xmax)
                ax.set_xlabel("Frequency or reduced frequency (Hz)")
                ax.set_ylabel(label_map.get(metric, metric))
                ax.set_title(f"Experimental-to-simulated frequency span: {label_map.get(metric, metric)}")
                ax.grid(True, which="both", alpha=0.25)
                ax.legend(frameon=False, fontsize=7.5, ncols=2)
            sm = ScalarMappable(norm=temp_norm, cmap=temp_cmap)
            sm.set_array([])
            fig.subplots_adjust(right=0.84, hspace=0.52)
            cax = fig.add_axes([0.87, 0.18, 0.025, 0.66])
            cbar = fig.colorbar(sm, cax=cax)
            cbar.set_label("Raw simulation temperature (K)")
            path = plot_dir / "master_curve_full_frequency_span_storage_loss_tandelta.png"
            fig.savefig(path, dpi=220)
            plt.close(fig)
            artifacts["full_frequency_span_plot"] = path
    except ImportError:
        pass

    for path in extrapolated_paths:
        artifacts[f"extrapolated_{path.stem}"] = path

    report_path = output_dir / "master_curve_scientific_computational_assessment.md"
    n_raw = int(sum(len(df) for df in filtered_by_metric.values()))
    temp_count = int(summary["temperature_K"].nunique()) if "temperature_K" in summary else 0
    freq_count = int(summary["frequency_ghz"].nunique()) if "frequency_ghz" in summary else 0
    lines = [
        "# Master-Curve Method Assessment",
        "",
        f"Parsed DMA rows used for master-curve analysis: `{n_raw}` across `{temp_count}` temperatures and `{freq_count}` frequencies.",
        f"Reference temperature: `{spec.reference_temperature_k:.3f} K`.",
        "",
        "## Method Notes",
        "",
        "- `tts_free` is the most data-driven built-in method: each temperature receives an independently optimized horizontal shift against the nearest simulated reference-temperature curve.",
        "- `wlf_default` applies fixed WLF constants. It is useful as a polymer-physics baseline, but it can produce enormous shifts outside the WLF validity window, especially below Tg.",
        "- `arrhenius_fit_to_tts` smooths the free TTS shifts into one activation-energy-like trend. It is usually less appropriate through a glass transition but is a stable extrapolation comparison.",
        "- `mastercurves` delegates the collapse to the optional Kyle Lennon `mastercurves` backend. It can be powerful, but the fitted shifts should be checked for overfitting and nonphysical long extrapolations.",
        "",
        "## Hz Extrapolation",
        "",
        f"CSV files named `*_extrapolated_to_hz.csv` contain interpolated/extrapolated master curves down to `{spec.experimental_min_frequency_hz:g} Hz`. Rows include an `extrapolated_below_data` flag. Values with that flag are predictions, not directly simulated DMA observables. The experimental-window plots focus on `{spec.experimental_min_frequency_hz:g}` to `{spec.experimental_max_frequency_hz:g} Hz`.",
        "",
        "## Diagnostics",
        "",
    ]
    for row in diagnostics:
        metric = row.get("metric", "")
        model = row.get("shift_model", "")
        note = row.get("note", "")
        lines.append(f"- `{model}` / `{metric}`: {note}")
    lines.extend(["", "## Artifacts", ""])
    for key, path in artifacts.items():
        lines.append(f"- `{key}`: `{Path(path).relative_to(output_dir)}`")
    report_path.write_text("\n".join(lines) + "\n")
    artifacts["report"] = report_path
    return artifacts
