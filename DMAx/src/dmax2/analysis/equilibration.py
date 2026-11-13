"""Equilibration trace parsing and plotting."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd


DEFAULT_COLUMNS = [
    "step",
    "time_ps",
    "temp_K",
    "press_MPa",
    "pxx_MPa",
    "pyy_MPa",
    "pzz_MPa",
    "pxy_MPa",
    "pxz_MPa",
    "pyz_MPa",
    "szz_vir_MPa",
    "sdev_vir_MPa",
    "lx_A",
    "ly_A",
    "lz_A",
    "density_g_cm3",
    "vol_A3",
]


def _columns_from_trace(path: Path) -> list[str]:
    with path.open() as fh:
        for line in fh:
            if not line.startswith("#"):
                break
            text = line[1:].strip()
            if text.startswith("step "):
                return text.split()
    return DEFAULT_COLUMNS


def read_equilibration_trace(path: str | Path) -> pd.DataFrame:
    """Read a DMAx2/LAMMPS `fix ave/time` equilibration trace."""

    path = Path(path)
    columns = _columns_from_trace(path)
    df = pd.read_csv(path, sep=r"\s+", comment="#", names=columns)
    if "time_ps" not in df and "step" in df:
        df["time_ps"] = df["step"].astype(float) / 1000.0
    return df


def stage_boundaries_from_config(config: Mapping) -> dict[str, tuple[float, float]]:
    """Return approximate stage windows in ps from an equilibration config dict."""

    eq = config.get("equilibration", config)
    dt = float(eq.get("timestep_fs", 1.0)) / 1000.0
    heat = int(eq.get("heat_steps", 0)) * dt
    target = int(eq.get("target_npt_steps", 0)) * dt
    anneal = int(eq.get("anneal_steps", eq.get("npt_anneal_steps", 0))) * dt
    cool = int(eq.get("cool_steps", eq.get("npt_anneal_steps", 0))) * dt
    relax = int(eq.get("relax_npt_steps", eq.get("npt_relax_steps", 0))) * dt
    final = int(eq.get("final_nvt_steps", eq.get("nvt_final_steps", 0))) * dt
    starts = np.cumsum([0.0, heat, target, anneal, cool, relax])
    lengths = [heat, target, anneal, cool, relax, final]
    names = ["heat_nvt", "target_npt", "anneal_npt", "cool_npt", "relax_npt", "final_nvt"]
    return {name: (float(start), float(start + length)) for name, start, length in zip(names, starts, lengths)}


def assign_stages(df: pd.DataFrame, stage_windows: Mapping[str, tuple[float, float]] | None) -> pd.DataFrame:
    out = df.copy()
    out["stage"] = "unassigned"
    if not stage_windows:
        return out
    for name, (start, stop) in stage_windows.items():
        mask = (out["time_ps"] >= start) & (out["time_ps"] <= stop)
        out.loc[mask, "stage"] = name
    return out


def summarize_equilibration(
    df: pd.DataFrame,
    stage_windows: Mapping[str, tuple[float, float]] | None = None,
    tail_ps: float = 100.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return final-window and per-stage summary tables."""

    staged = assign_stages(df, stage_windows)
    final_time = float(staged["time_ps"].max())
    tail = staged[staged["time_ps"] >= final_time - tail_ps].copy()
    metrics = [
        "temp_K",
        "press_MPa",
        "pxx_MPa",
        "pyy_MPa",
        "pzz_MPa",
        "sdev_vir_MPa",
        "lx_A",
        "ly_A",
        "lz_A",
        "density_g_cm3",
        "vol_A3",
    ]
    rows = []
    for metric in metrics:
        if metric not in tail:
            continue
        values = tail[metric].replace([np.inf, -np.inf], np.nan).dropna()
        rows.append(
            {
                "window": f"last_{tail_ps:g}_ps",
                "metric": metric,
                "mean": float(values.mean()) if len(values) else np.nan,
                "std": float(values.std(ddof=1)) if len(values) > 1 else np.nan,
                "min": float(values.min()) if len(values) else np.nan,
                "max": float(values.max()) if len(values) else np.nan,
                "n": int(len(values)),
            }
        )
    final_summary = pd.DataFrame(rows)

    stage_rows = []
    for stage, group in staged.groupby("stage", sort=False):
        row = {"stage": stage, "start_ps": float(group["time_ps"].min()), "stop_ps": float(group["time_ps"].max()), "n": int(len(group))}
        for metric in metrics:
            if metric in group:
                row[f"{metric}_mean"] = float(group[metric].mean())
                row[f"{metric}_std"] = float(group[metric].std(ddof=1)) if len(group) > 1 else np.nan
        stage_rows.append(row)
    stage_summary = pd.DataFrame(stage_rows)
    return final_summary, stage_summary


def _shade_stages(ax, stage_windows: Mapping[str, tuple[float, float]] | None) -> None:
    if not stage_windows:
        return
    colors = ["#eef2ff", "#f0fdf4", "#fff7ed", "#eff6ff", "#f8fafc", "#fef2f2"]
    for idx, (stage, (start, stop)) in enumerate(stage_windows.items()):
        ax.axvspan(start, stop, color=colors[idx % len(colors)], alpha=0.42, linewidth=0)
        ax.axvline(stop, color="#94a3b8", linewidth=0.55, alpha=0.8)
        mid = start + 0.5 * (stop - start)
        ymax = ax.get_ylim()[1]
        ax.text(mid, ymax, stage.replace("_", "\n"), ha="center", va="top", fontsize=7, color="#475569")


def write_equilibration_plots(
    df: pd.DataFrame,
    output_dir: str | Path,
    stage_windows: Mapping[str, tuple[float, float]] | None = None,
    prefix: str = "equilibration",
) -> list[Path]:
    """Write publication-oriented equilibration plots."""

    output_dir = Path(output_dir)
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return paths

    def save(fig, name: str) -> None:
        path = plot_dir / f"{prefix}_{name}.png"
        fig.tight_layout()
        fig.savefig(path, dpi=220)
        plt.close(fig)
        paths.append(path)

    fig, axes = plt.subplots(3, 1, figsize=(8.0, 7.0), sharex=True)
    axes[0].plot(df["time_ps"], df["temp_K"], color="#0f766e", linewidth=1.0)
    axes[0].set_ylabel("Temperature (K)")
    axes[1].plot(df["time_ps"], df["density_g_cm3"], color="#7c2d12", linewidth=1.0)
    axes[1].set_ylabel("Density (g/cm3)")
    axes[2].plot(df["time_ps"], df["press_MPa"], color="#1d4ed8", linewidth=0.8)
    axes[2].axhline(0.101325, color="#334155", linewidth=0.7, linestyle="--")
    axes[2].set_ylabel("Pressure (MPa)")
    axes[2].set_xlabel("Simulation time after minimization (ps)")
    for ax in axes:
        _shade_stages(ax, stage_windows)
        ax.grid(True, alpha=0.25)
    save(fig, "thermo_density_pressure")

    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    for col, label in [("lx_A", "Lx"), ("ly_A", "Ly"), ("lz_A", "Lz")]:
        ax.plot(df["time_ps"], df[col], linewidth=1.0, label=label)
    ax.set_xlabel("Simulation time after minimization (ps)")
    ax.set_ylabel("Box length (A)")
    ax.legend(ncol=3)
    ax.grid(True, alpha=0.25)
    _shade_stages(ax, stage_windows)
    save(fig, "box_lengths")

    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    for col in ["pxx_MPa", "pyy_MPa", "pzz_MPa"]:
        ax.plot(df["time_ps"], df[col], linewidth=0.8, label=col.replace("_MPa", ""))
    ax.axhline(0.101325, color="#334155", linewidth=0.7, linestyle="--")
    ax.set_xlabel("Simulation time after minimization (ps)")
    ax.set_ylabel("Pressure tensor (MPa)")
    ax.legend(ncol=3)
    ax.grid(True, alpha=0.25)
    _shade_stages(ax, stage_windows)
    save(fig, "pressure_tensor")

    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    ax.plot(df["time_ps"], df["sdev_vir_MPa"], color="#be123c", linewidth=0.8, label="virial deviatoric axial")
    if "szz_vir_MPa" in df:
        ax.plot(df["time_ps"], df["szz_vir_MPa"], color="#64748b", linewidth=0.7, alpha=0.7, label="virial axial")
    ax.axhline(0.0, color="#334155", linewidth=0.7, linestyle="--")
    ax.set_xlabel("Simulation time after minimization (ps)")
    ax.set_ylabel("Virial stress diagnostic (MPa)")
    ax.legend()
    ax.grid(True, alpha=0.25)
    _shade_stages(ax, stage_windows)
    save(fig, "virial_stress_diagnostics")

    return paths


def analyze_equilibration_trace(
    trace_path: str | Path,
    output_dir: str | Path,
    stage_windows: Mapping[str, tuple[float, float]] | None = None,
    tail_ps: float = 100.0,
    prefix: str = "equilibration",
) -> dict[str, Path]:
    """Parse, summarize, and plot an equilibration trace."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = read_equilibration_trace(trace_path)
    final_summary, stage_summary = summarize_equilibration(df, stage_windows, tail_ps=tail_ps)
    trace_out = output_dir / f"{prefix}_trace.csv"
    final_out = output_dir / f"{prefix}_final_window_summary.csv"
    stage_out = output_dir / f"{prefix}_stage_summary.csv"
    df.to_csv(trace_out, index=False)
    final_summary.to_csv(final_out, index=False)
    stage_summary.to_csv(stage_out, index=False)
    paths = write_equilibration_plots(df, output_dir, stage_windows, prefix=prefix)
    return {
        "trace_csv": trace_out,
        "final_summary": final_out,
        "stage_summary": stage_out,
        "plots_dir": output_dir / "plots",
        **{f"plot_{idx}": path for idx, path in enumerate(paths, start=1)},
    }
