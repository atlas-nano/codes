"""Independent periodic projection, with measured strain and explicit geometry.

For eps=A*sin(omega*t+phi), stress=S*sin(theta)+L*cos(theta),
the signed complex response is (S+i*L)/A. Positive loss dissipates work.
Discrete projection equals constant-plus-harmonic OLS only on an integer
number of uniformly sampled periods. Drift-adjusted OLS is a different model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .dma import FitResult, classify_fit, fit_dma_dataframe, parse_raw_run_metadata, read_raw_table


@dataclass
class FourierFitResult(FitResult):
    measured_strain_amplitude: float
    strain_phase_deg: float
    phase_deg: float
    response_label: str
    conjugacy: str
    sampling_flags: str
    complete_cycles: int
    samples_per_cycle: int
    harm5_over_harm1: float
    strain_residual_fraction: float
    work_per_cycle_MPa: float
    first_harmonic_work_MPa: float
    equivalent_ols_max_abs_error_MPa: float
    equivalent_ols_storage_MPa: float
    equivalent_ols_loss_MPa: float


def periodic_window(
    df: pd.DataFrame, frequency_ghz: float, columns: list[str],
    discard_cycles: int = 0, timestep_fs: float | None = None,
    incomplete: str = "reject",
) -> tuple[pd.DataFrame, int, int, list[str]]:
    """Select a half-open uniform integer-period window, without interpolation.

    Times are ps; step-only records require an explicit fs timestep. Discard
    is relative to the first recorded timestamp. 1e-7 cadence / 1e-6 period
    relative tolerances accommodate printed timestamps, not irregular data.
    `trim` explicitly drops trailing partial periods; the default rejects them.
    """
    if not np.isfinite(frequency_ghz) or frequency_ghz <= 0 or discard_cycles < 0:
        raise ValueError("Positive finite frequency and nonnegative discard are required.")
    if incomplete not in {"reject", "trim"}:
        raise ValueError("incomplete must be 'reject' or 'trim'.")
    data = df.copy()
    missing = [c for c in columns if c not in data]
    if missing:
        raise ValueError(f"Missing measured columns: {missing}")
    if "t_ps" not in data:
        if "time_ps" in data:
            data["t_ps"] = data["time_ps"]
        elif "step" in data and timestep_fs is not None and timestep_fs > 0:
            data["t_ps"] = data["step"] * timestep_fs / 1000.0
        else:
            raise ValueError("Physical time missing: supply t_ps or an explicit timestep_fs.")
    values = data[["t_ps", *columns]].to_numpy(float)
    if len(data) < 8 or not np.isfinite(values).all():
        raise ValueError("At least eight finite samples are required; missing values are not imputed.")
    flags = []
    duplicate = np.flatnonzero(np.diff(values[:, 0]) == 0) + 1
    if len(duplicate):
        if not np.all(values[duplicate] == values[duplicate - 1]):
            raise ValueError("Duplicate timestamps have conflicting measured values.")
        data = data.iloc[np.setdiff1d(np.arange(len(data)), duplicate)].copy()
        flags.append(f"duplicate_timestamps_removed:{len(duplicate)}")
    t = data["t_ps"].to_numpy(float)
    dt = float(np.median(np.diff(t)))
    if dt <= 0 or np.any(np.diff(t) <= 0):
        raise ValueError("Time must be strictly increasing; timestamps are not sorted automatically.")
    if not np.allclose(np.diff(t), dt, rtol=1e-7, atol=1e-10 * dt):
        raise ValueError("Irregular sampling or missing points: use generalized harmonic regression.")
    period = 1000.0 / frequency_ghz
    spp = int(round(period / dt))
    if spp < 3 or not np.isclose(spp * dt, period, rtol=1e-6, atol=0):
        raise ValueError("Noninteger samples per cycle: use generalized harmonic regression.")
    data = data.loc[t >= t[0] + discard_cycles * period - 1e-7 * dt].copy()
    if len(data) >= 2 and (len(data) - 1) % spp == 0:
        data = data.iloc[:-1].copy()
        flags.append("right_endpoint_removed")
    partial = len(data) % spp
    if partial:
        if incomplete == "reject":
            raise ValueError(f"Incomplete cycles ({partial} trailing samples): explicitly trim or use regression.")
        data = data.iloc[:len(data) - partial].copy()
        flags.append(f"partial_cycle_trimmed:{partial}")
    cycles = len(data) // spp
    if cycles < 1:
        raise ValueError("No complete cycles remain after discard.")
    return data.reset_index(drop=True), cycles, spp, flags


def conjugate_stress(data: pd.DataFrame, mode: str, signal: str, strain_signal: str) -> tuple[np.ndarray, str, str]:
    """Stress conjugate to measured engineering strain per reference volume.

    Tensile/shear-positive signals are already -P*0.101325 MPa. Simple shear
    uses sigma_xy; fixed transverse area uses sigma_zz. For prescribed
    volume conservation lambda_x=lambda_y=lambda_z**(-1/2), use
    [sigma_zz-(sigma_xx+sigma_yy)/2]/lambda_z. Its linear isotropic limit
    is 3G. Lateral-NPT axial nominal stress includes transverse area change;
    it is E only if transverse stress-free conditions are independently shown.
    """
    y = data[signal].to_numpy(float)
    eps = data[strain_signal].to_numpy(float)
    if mode in {"shear_sllod_remapv", "shear_nvt_remapx"}:
        if signal not in {"sxy_vir_MPa", "sxy_tot_MPa"}:
            raise ValueError("Simple shear requires shear-positive sxy stress.")
        return y, "G", "simple_shear_sigma_xy"
    if "volcons" in mode or mode == "legacy_axial_pressure":
        if signal != "sdev_MPa":
            raise ValueError("Volume-conserving drive requires sdev_MPa, not axial or hydrostatic pressure.")
        lam = 1.0 + eps
        if np.any(lam <= 0):
            raise ValueError("Nonpositive axial stretch.")
        for col in ("eps_x", "eps_y"):
            if col not in data or not np.allclose(1.0 + data[col], lam**-0.5, rtol=1e-6, atol=1e-8):
                raise ValueError("Measured transverse strains do not verify the prescribed volume-conserving path.")
        return y / lam, "volume_path (3G small-strain isotropic limit)", "sdev_divided_by_axial_stretch"
    if "fixed_lateral" in mode:
        if signal not in {"szz_vir_MPa", "szz_tot_MPa"}:
            raise ValueError("Fixed-lateral axial drive requires szz stress; other signals are diagnostics.")
        for col in ("eps_x", "eps_y"):
            if col in data and not np.allclose(data[col], data[col].iloc[0], rtol=0, atol=1e-8):
                raise ValueError("Fixed-lateral mode has changing transverse strain.")
        return y, "M", "fixed_transverse_area_sigma_zz"
    if mode in {"axial_npt_lateral", "tension_npt_z_lateral", "tension_pretension_npt_z_lateral"}:
        if signal not in {"szz_vir_MPa", "szz_tot_MPa"} or not {"eps_x", "eps_y"}.issubset(data):
            raise ValueError("Lateral-NPT axial component needs szz and measured eps_x/eps_y.")
        area = (1.0 + data["eps_x"].to_numpy(float)) * (1.0 + data["eps_y"].to_numpy(float))
        if np.any(area <= 0):
            raise ValueError("Nonpositive transverse area ratio.")
        return y * area, "axial nominal response (E only if transverse stress-free)", "axial_component_only_sigma_zz_times_area_ratio"
    raise ValueError(f"Unsupported or ambiguous deformation mode: {mode}")


def _project(y: np.ndarray, theta: np.ndarray, harmonics: tuple[int, ...]) -> np.ndarray:
    centered = y - y.mean()
    return np.array([2.0 * np.mean(centered * np.sin(h * theta)) +
                     2j * np.mean(centered * np.cos(h * theta)) for h in harmonics])


def _harmonic_design(theta: np.ndarray, harmonics: tuple[int, ...], drift: np.ndarray | None = None) -> np.ndarray:
    columns = [np.ones(len(theta))]
    if drift is not None:
        columns.append(drift)
    for h in harmonics:
        columns.extend((np.sin(h * theta), np.cos(h * theta)))
    return np.column_stack(columns)


def project_periodic_response(
    df: pd.DataFrame, mode: str, signal: str, frequency_ghz: float,
    discard_cycles: int = 0, strain_signal: str = "strain_box",
    timestep_fs: float | None = None, incomplete: str = "reject",
    harmonics: tuple[int, ...] = (1, 2, 3, 4, 5),
) -> tuple[FourierFitResult, pd.DataFrame, pd.DataFrame]:
    """Project stress and measured strain; return fit, spectrum and used window.

    Signed harmonics are rotated by n*phi of the measured strain fundamental.
    No drift subtraction, absolute-loss conversion, or LS fallback is hidden.
    Drift-adjusted LS is used only to diagnose trend contamination. Work is
    the spectral quadrature over the retained harmonics (MPa = MJ/m^3).
    """
    if not harmonics or harmonics[0] != 1 or len(set(harmonics)) != len(harmonics) or any(not isinstance(h, (int, np.integer)) or h < 1 for h in harmonics):
        raise ValueError("Distinct positive integer harmonics starting with 1 are required.")
    columns = [signal, strain_signal, *[c for c in ("eps_x", "eps_y") if c in df]]
    data, cycles, spp, flags = periodic_window(df, frequency_ghz, columns, discard_cycles, timestep_fs, incomplete)
    if 2 * max(harmonics) >= spp:
        raise ValueError("Harmonics at/above Nyquist are unresolved; increase samples per cycle.")
    t = data["t_ps"].to_numpy(float)
    theta = 2.0 * np.pi * (t - t[0]) * frequency_ghz / 1000.0
    strain = data[strain_signal].to_numpy(float)
    ec = _project(strain, theta, harmonics)
    amp = float(abs(ec[0]))
    if amp <= 100.0 * np.finfo(float).eps * max(1.0, float(np.max(abs(strain)))):
        raise ValueError("Measured strain fundamental is zero/unresolved; a filename amplitude is not a replacement.")
    stress, label, conjugacy = conjugate_stress(data, mode, signal, strain_signal)
    sc = _project(stress, theta, harmonics)
    response = sc[0] / ec[0]
    design = _harmonic_design(theta, harmonics)
    beta = np.linalg.lstsq(design, stress, rcond=None)[0]
    ols = beta[1::2] + 1j * beta[2::2]
    parity = float(np.max(abs(ols - sc)))
    drift_time = (t - t.mean()) * frequency_ghz / 1000.0
    drift_beta = np.linalg.lstsq(_harmonic_design(theta, harmonics, drift_time), stress, rcond=None)[0]
    first = float(abs(sc[0]))
    drift_fraction = abs(float(drift_beta[1])) / first if first > 0 else np.nan
    if drift_fraction > 0.05:
        flags.append("drift_detected_use_harmonic_regression")
    yhat = stress.mean() + sum(c.real * np.sin(h * theta) + c.imag * np.cos(h * theta) for c, h in zip(sc, harmonics))
    residual = stress - yhat
    ss_tot = float(np.sum((stress - stress.mean())**2))
    r2 = 1.0 - float(residual @ residual) / ss_tot if ss_tot > 0 else np.nan
    noise = float(np.std(residual, ddof=1))
    snr = first / noise if noise > 0 else np.inf
    ratios = {h: float(abs(c) / first) if first > 0 else np.nan for h, c in zip(harmonics, sc)}
    strain_first = ec[0].real * np.sin(theta) + ec[0].imag * np.cos(theta)
    strain_residual = float(np.sqrt(np.mean((strain - strain.mean() - strain_first)**2)) / amp)
    if strain_residual > 0.01:
        flags.append("nonsinusoidal_strain_fundamental_response_only")
    if "npt_lateral" in mode or "npt_z_lateral" in mode:
        flags.append("transverse_stress_free_boundary_not_verified")
    work = float(sum(np.pi * h * (c.imag * e.real - c.real * e.imag) for h, c, e in zip(harmonics, sc, ec)))
    spectrum = pd.DataFrame([{
        "harmonic": h, "stress_sin_MPa": float(c.real), "stress_cos_MPa": float(c.imag),
        "strain_sin": float(e.real), "strain_cos": float(e.imag),
        "stress_in_strain_phase_MPa": float((c * np.exp(-1j * h * np.angle(ec[0]))).real),
        "stress_quadrature_MPa": float((c * np.exp(-1j * h * np.angle(ec[0]))).imag),
        "harmonic_over_first": ratios[h], "ols_abs_error_MPa": float(abs(c - o)),
    } for h, c, e, o in zip(harmonics, sc, ec, ols)])
    status = classify_fit(r2, snr, ratios.get(2, np.nan), ratios.get(3, np.nan), drift_fraction)
    if ratios.get(5, 0.0) > 0.20:
        status = ("" if status == "ok" else status + ";") + "large_5omega"
    if flags and any("drift_detected" in f or "nonsinusoidal" in f or "boundary_not_verified" in f for f in flags):
        status = ("" if status == "ok" else status + ";") + ";".join(flags)
    fit = FourierFitResult(mode, signal, amp, "fourier_projection", float(response.real), float(response.imag),
        float(response.imag / response.real) if response.real != 0 else np.nan,
        r2, snr, ratios.get(2, np.nan), ratios.get(3, np.nan), drift_fraction, len(data), status,
        amp, float(np.degrees(np.angle(ec[0]))), float(np.degrees(np.angle(response))), label,
        conjugacy, ";".join(flags), cycles, spp, ratios.get(5, np.nan), strain_residual,
        work, float(np.pi * amp**2 * response.imag), parity,
        float((ols[0] / ec[0]).real), float((ols[0] / ec[0]).imag))
    return fit, spectrum, data


def compare_estimators(
    raw_file: str, frequency_ghz: float, signal: str, discard_cycles: int = 0,
    strain_signal: str = "strain_box", timestep_fs: float | None = None, incomplete: str = "reject",
    reference_length_a: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare identical samples, retaining both historical normalization and
    measured-strain/conjugate-stress normalization. Does not alter source data.
    """
    metadata = parse_raw_run_metadata(raw_file)
    mode = str(metadata["mode"])
    df = read_raw_table(raw_file)
    if reference_length_a is not None:
        if not np.isfinite(reference_length_a) or reference_length_a <= 0 or strain_signal not in df:
            raise ValueError("Box-length reconstruction requires a positive reference_length_a and measured length column.")
        if "shear" in mode:
            raise ValueError("Axial length reconstruction cannot supply a measured shear strain.")
        # Explicit reconstruction from saved box lengths, never the filename
        # amplitude. The caller must verify the pre-drive reference length.
        df["reconstructed_measured_strain"] = df[strain_signal] / reference_length_a - 1.0
        strain_signal = "reconstructed_measured_strain"
    fit, spectrum, data = project_periodic_response(df, mode, signal, frequency_ghz, discard_cycles,
        strain_signal, timestep_fs, incomplete)
    records = [fit.as_dict() | {"normalization": "measured_strain_conjugate_stress"}]
    stress, _, _ = conjugate_stress(data, mode, signal, strain_signal)
    corrected = data.copy()
    corrected[signal] = stress
    phi = np.radians(fit.strain_phase_deg)
    for parser in ("lockin", "legacy"):
        historical = fit_dma_dataframe(data, mode, signal, float(metadata["strain"]), frequency_ghz, parser=parser)
        records.append(historical.as_dict() | {"normalization": "historical_nominal_strain_time_origin",
            "phase_deg": float(np.degrees(np.arctan2(historical.loss_modulus_MPa, historical.storage_modulus_MPa)))})
        normalized = fit_dma_dataframe(corrected, mode, signal, fit.strain, frequency_ghz, parser=parser)
        z = complex(normalized.storage_modulus_MPa, normalized.loss_modulus_MPa) * np.exp(-1j * phi)
        records.append(normalized.as_dict() | {"normalization": "measured_strain_conjugate_stress",
            "storage_modulus_MPa": z.real, "loss_modulus_MPa": z.imag,
            "tan_delta": z.imag / z.real if z.real != 0 else np.nan, "phase_deg": float(np.degrees(np.angle(z))),
            "response_label": fit.response_label, "projection_difference_MPa": float(abs(z - complex(fit.storage_modulus_MPa, fit.loss_modulus_MPa)))})
    # Exact comparator: OLS on the same constant + 1..5 basis, without drift.
    records.append({**fit.as_dict(), "parser": "equivalent_harmonic_ols", "normalization": "measured_strain_conjugate_stress",
                    "storage_modulus_MPa": fit.equivalent_ols_storage_MPa,
                    "loss_modulus_MPa": fit.equivalent_ols_loss_MPa,
                    "tan_delta": fit.equivalent_ols_loss_MPa / fit.equivalent_ols_storage_MPa if fit.equivalent_ols_storage_MPa != 0 else np.nan,
                    "projection_difference_MPa": fit.equivalent_ols_max_abs_error_MPa / fit.strain})
    return pd.DataFrame(records), spectrum
