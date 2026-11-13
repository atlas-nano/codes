"""Synthetic truths in MPa/ps; no MD or physical-material validation."""

import numpy as np
import pandas as pd
import pytest

from dmax2.analysis.dma import fit_dma_dataframe, parse_dma_file, read_raw_table
from dmax2.analysis.fourier import compare_estimators, project_periodic_response
from dmax2.cli import main


def trace(storage=1000.0, loss=300.0, amp=0.03, phase=0.37, cycles=6, spp=80,
          mode="tension_nvt_z_fixed_lateral", offset=13.0):
    t = np.arange(cycles * spp) * 40.0 / spp + 7.5
    theta = 2 * np.pi * (t - t[0]) / 40.0 + phase
    eps = amp * np.sin(theta)
    y = offset + amp * (storage * np.sin(theta) + loss * np.cos(theta))
    return pd.DataFrame({"t_ps": t, "strain_box": eps, "strain_applied": eps,
                         "szz_vir_MPa": y, "szz_tot_MPa": y + 2.0,
                         "sxy_vir_MPa": y, "sxy_tot_MPa": y + 2.0,
                         "eps_x": np.zeros(len(t)), "eps_y": np.zeros(len(t))}), theta


def project(df, **kwargs):
    return project_periodic_response(df, "tension_nvt_z_fixed_lateral", "szz_vir_MPa", 25.0, **kwargs)


@pytest.mark.parametrize("storage,loss", [(1000, 0), (0, 300), (1000, 300), (-1000, -300)])
def test_elastic_viscous_phase_offset_and_signed_moduli(storage, loss):
    df, _ = trace(storage, loss)
    fit, _, _ = project(df)
    assert fit.storage_modulus_MPa == pytest.approx(storage, abs=1e-9)
    assert fit.loss_modulus_MPa == pytest.approx(loss, abs=1e-9)
    assert fit.measured_strain_amplitude == pytest.approx(0.03)
    assert fit.equivalent_ols_max_abs_error_MPa < 1e-10
    assert fit.first_harmonic_work_MPa == pytest.approx(np.pi * 0.03**2 * loss, abs=1e-12)


@pytest.mark.parametrize("omega_tau", [0.01, 1, 100])
def test_maxwell_truth_and_cycle_work(omega_tau):
    # Maxwell element: G*=G*i*w*tau/(1+i*w*tau); eta=G*tau.
    g = 2000.0
    storage = g * omega_tau**2 / (1 + omega_tau**2)
    loss = g * omega_tau / (1 + omega_tau**2)
    df, theta = trace(storage, loss)
    fit, _, _ = project(df)
    assert fit.storage_modulus_MPa == pytest.approx(storage, abs=1e-9)
    assert fit.loss_modulus_MPa == pytest.approx(loss, abs=1e-9)
    # Independent direct quadrature of stress times analytic dstrain/dtheta.
    direct_work = 2 * np.pi * np.mean(df.szz_vir_MPa * 0.03 * np.cos(theta))
    assert fit.work_per_cycle_MPa == pytest.approx(direct_work, abs=1e-11)


def test_second_third_fifth_harmonics_are_phase_referenced():
    df, theta = trace()
    df["szz_vir_MPa"] += 8 * np.sin(2 * theta) + 12 * np.cos(3 * theta) + 10 * np.sin(5 * theta)
    fit, spectrum, _ = project(df)
    assert fit.storage_modulus_MPa == pytest.approx(1000, abs=1e-9)
    assert fit.loss_modulus_MPa == pytest.approx(300, abs=1e-9)
    assert fit.harm5_over_harm1 == pytest.approx(10 / (0.03 * np.hypot(1000, 300)))
    fifth = spectrum.loc[spectrum.harmonic.eq(5)].iloc[0]
    assert fifth.stress_in_strain_phase_MPa == pytest.approx(10, abs=1e-10)
    assert fifth.stress_quadrature_MPa == pytest.approx(0, abs=1e-10)
    assert "large_3omega" in fit.status
    assert "large_5omega" in fit.status


def test_cubic_elastic_response_is_not_called_linear():
    df, _ = trace(storage=500, loss=0)
    df["szz_vir_MPa"] += 2e6 * df.strain_box**3
    fit, _, _ = project(df)
    expected = 500 + 0.75 * 2e6 * 0.03**2
    assert fit.storage_modulus_MPa == pytest.approx(expected, abs=1e-9)
    assert fit.harm3_over_harm1 > 0.20
    assert "large_3omega" in fit.status


def test_offset_closed_endpoint_and_exact_duplicate_timestamp():
    df, _ = trace(offset=2e6, phase=0)
    # Closed interval has one endpoint at the next period; do not double weight.
    endpoint = df.iloc[[0]].copy(); endpoint["t_ps"] += 6 * 40
    closed = pd.concat([df, endpoint], ignore_index=True)
    closed = pd.concat([closed.iloc[:30], closed.iloc[[29]], closed.iloc[30:]], ignore_index=True)
    fit, _, _ = project(closed)
    assert fit.n_points == len(df)
    assert "right_endpoint_removed" in fit.sampling_flags
    assert "duplicate_timestamps_removed:1" in fit.sampling_flags
    assert fit.storage_modulus_MPa == pytest.approx(1000, abs=1e-7)


@pytest.mark.parametrize("damage,message", [
    ("missing", "Irregular"), ("irregular", "Irregular"),
    ("partial", "Incomplete cycles"), ("conflict", "conflicting"),
    ("reversed", "strictly increasing"), ("nan", "finite"),
    ("zero_strain", "zero/unresolved"), ("no_time", "Physical time"),
])
def test_invalid_sampling_and_strain_are_explicit(damage, message):
    df, _ = trace()
    if damage == "missing": df = df.drop(index=30)
    if damage == "irregular": df.loc[30, "t_ps"] += 0.05
    if damage == "partial": df = df.iloc[:-10]
    if damage == "conflict": df.loc[30, "t_ps"] = df.loc[29, "t_ps"]
    if damage == "reversed": df = df.iloc[::-1]
    if damage == "nan": df.loc[30, "szz_vir_MPa"] = np.nan
    if damage == "zero_strain": df["strain_box"] = 0.0
    if damage == "no_time": df = df.drop(columns="t_ps"); df["step"] = np.arange(len(df))
    with pytest.raises(ValueError, match=message): project(df)


def test_explicit_trim_timestep_discard_and_nyquist():
    df, _ = trace(cycles=6)
    fit, _, _ = project(df.iloc[:-10], incomplete="trim", discard_cycles=1)
    assert fit.complete_cycles == 4
    assert "partial_cycle_trimmed" in fit.sampling_flags
    step_only = df.drop(columns="t_ps"); step_only["step"] = df.t_ps * 1000 / 0.5
    timed, _, _ = project(step_only, timestep_fs=0.5)
    assert timed.storage_modulus_MPa == pytest.approx(1000, abs=1e-9)
    sparse, _ = trace(spp=10)
    with pytest.raises(ValueError, match="Nyquist"): project(sparse)


def test_drift_is_diagnosed_instead_of_hidden_fallback():
    df, _ = trace(phase=0)
    df["szz_vir_MPa"] += 6 * df.t_ps / 40
    fit, _, _ = project(df)
    assert "drift_detected" in fit.sampling_flags
    assert fit.drift_fraction_per_cycle > 0.05
    assert abs(fit.loss_modulus_MPa - 300) > 1
    old = fit_dma_dataframe(df, "tension_nvt_z_fixed_lateral", "szz_vir_MPa", 0.03, 25, parser="lockin")
    assert old.loss_modulus_MPa == pytest.approx(300, abs=1e-9)


@pytest.mark.parametrize("correlated", [False, True])
def test_seeded_noise_projection_ols_equivalence_and_mean(correlated):
    rng = np.random.default_rng(77123)
    estimates = []
    for _ in range(48):
        df, _ = trace(cycles=12)
        noise = rng.normal(0, 2, len(df))
        if correlated:
            for i in range(1, len(noise)): noise[i] += 0.65 * noise[i - 1]
        df["szz_vir_MPa"] += noise
        fit, spectrum, _ = project(df)
        estimates.append([fit.storage_modulus_MPa, fit.loss_modulus_MPa])
        assert spectrum.ols_abs_error_MPa.max() < 1e-10
    values = np.asarray(estimates)
    # Replicate means, with independent synthetic seeds, are unbiased within
    # four SEM. This is not an MD uncertainty/coverage claim.
    assert np.all(abs(values.mean(axis=0) - [1000, 300]) < 4 * values.std(axis=0, ddof=1) / np.sqrt(len(values)))


def test_geometry_conjugacy_and_pressure_rejection():
    df, _ = trace()
    shear, _, _ = project_periodic_response(df, "shear_sllod_remapv", "sxy_vir_MPa", 25)
    assert shear.response_label == "G"
    assert shear.storage_modulus_MPa == pytest.approx(1000)
    lam = 1 + df.strain_box
    df["eps_x"] = df["eps_y"] = lam**-0.5 - 1
    df["sdev_MPa"] = df.szz_vir_MPa * lam
    vol, _, _ = project_periodic_response(df, "tension_nvt_z_volcons", "sdev_MPa", 25)
    assert vol.storage_modulus_MPa == pytest.approx(1000, abs=1e-9)
    assert "3G" in vol.response_label
    with pytest.raises(ValueError, match="requires sdev"):
        project_periodic_response(df, "tension_nvt_z_volcons", "szz_vir_MPa", 25)
    area = (1 + df.eps_x) * (1 + df.eps_y)
    df["szz_vir_MPa"] /= area
    lateral, _, _ = project_periodic_response(df, "tension_npt_z_lateral", "szz_vir_MPa", 25)
    assert lateral.storage_modulus_MPa == pytest.approx(1000, abs=1e-9)
    assert "boundary_not_verified" in lateral.status


def test_compare_command_and_opt_in_parser_preserve_archival_estimators(tmp_path):
    df, _ = trace(phase=0.37)
    raw = tmp_path / "raw__tension_nvt_z_fixed_lateral__strain_0p060.dat"
    df.to_csv(raw, sep=" ", index=False)
    paired, _ = compare_estimators(str(raw), 25, "szz_vir_MPa")
    normalized = paired[paired.normalization.eq("measured_strain_conjugate_stress")]
    assert np.allclose(normalized.storage_modulus_MPa, 1000, atol=1e-9)
    assert np.allclose(normalized.loss_modulus_MPa, 300, atol=1e-9)
    # Filename strain intentionally differs; old results stay exactly old.
    old = fit_dma_dataframe(read_raw_table(raw), "tension_nvt_z_fixed_lateral", "szz_vir_MPa", 0.06, 25)
    parsed_old = parse_dma_file(raw, 25, discard_cycles=0, signals=("szz_vir_MPa",))[0]
    assert parsed_old.as_dict() == old.as_dict()
    new = parse_dma_file(raw, 25, discard_cycles=0, parser="fourier_projection", signals=("szz_vir_MPa",))[0]
    assert new.storage_modulus_MPa == pytest.approx(1000)
    assert main(["compare-estimators", str(raw), "--freq-ghz", "25", "--signal", "szz_vir_MPa",
                 "--output-dir", str(tmp_path / "comparison")]) == 0
    assert (tmp_path / "comparison" / "estimator_comparison.csv").exists()
    with pytest.raises(FileExistsError):
        main(["compare-estimators", str(raw), "--freq-ghz", "25", "--signal", "szz_vir_MPa",
              "--output-dir", str(tmp_path / "comparison")])


def test_box_length_reconstruction_is_explicit_and_verified(tmp_path):
    df, _ = trace()
    df["lz"] = 42.0 * (1 + df.strain_box)
    df["strain_box"] = 0.0  # Historical live reference expression.
    raw = tmp_path / "raw__tension_nvt_z_fixed_lateral__strain_0p060.dat"
    df.to_csv(raw, sep=" ", index=False)
    with pytest.raises(ValueError, match="zero/unresolved"):
        compare_estimators(str(raw), 25, "szz_vir_MPa")
    table, _ = compare_estimators(str(raw), 25, "szz_vir_MPa", strain_signal="lz", reference_length_a=42.0)
    response = table[table.parser.eq("fourier_projection")].iloc[0]
    assert response.storage_modulus_MPa == pytest.approx(1000, abs=1e-8)
    assert response.loss_modulus_MPa == pytest.approx(300, abs=1e-8)


def test_invalid_fourier_record_remains_an_explicit_failed_row(tmp_path):
    df, _ = trace()
    df["strain_box"] = 0.0
    raw = tmp_path / "raw__tension_nvt_z_fixed_lateral__strain_0p060.dat"
    df.to_csv(raw, sep=" ", index=False)
    failed = parse_dma_file(raw, 25, discard_cycles=0, parser="fourier_projection", signals=("szz_vir_MPa",))[0]
    assert failed.status.startswith("parse_failed:")
    assert np.isnan(failed.storage_modulus_MPa)


def test_historical_plotting_cannot_silently_fit_fourier_as_legacy():
    from dmax2.analysis.dma import _fit_trace_for_plot
    df, _ = trace()
    with pytest.raises(ValueError, match="compare-estimators"):
        _fit_trace_for_plot(df, "szz_vir_MPa", 25, 0, "fourier_projection")


@pytest.mark.parametrize("mode", ["shear_sllod_remapv", "axial_sllod_remapv_volcons", "tension_nvt_z_fixed_lateral"])
def test_opt_in_future_writer_references_and_pressure(mode):
    from dmax2.config import DMAProductionSpec
    from dmax2.lammps.inputs import _mode_block
    original = "\n".join(_mode_block(mode, 0.06, DMAProductionSpec(), "raw", "pppm 1e-5"))
    repaired = "\n".join(_mode_block(mode, 0.06, DMAProductionSpec(validated_mechanics=True), "raw", "pppm 1e-5"))
    assert "variable lz0 equal lz" in original
    assert "variable lz0 equal $(lz:%.17g)" in repaired
    if "sllod" in mode:
        assert "tchain 3" in original
        assert "tchain 1" in repaired
        assert "compute Ptot all pressure TDEF" in repaired
        assert repaired.index("compute TDEF") < repaired.index("compute Ptot")
        assert "variable temp_K equal c_TDEF" in repaired
        assert "kick" not in repaired  # 2024/Apr2025 binaries do not support it.
    else:
        assert "compute Ptot all pressure thermo_temp" in repaired
        assert original.index("fix DEF ") < original.index("fix OUT ")
        assert repaired.index("fix OUT ") < repaired.index("fix DEF ")
        assert "variable press_MPa equal c_Ptot*v_atm2mpa" in repaired
