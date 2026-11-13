import pandas as pd
import pytest

from dmax2.analysis.master_curve import assess_frequency_sweep_peaks, assign_tts_shifts, require_interior_peaks, write_master_curve_analysis_report
from dmax2.config import MasterCurveSpec


def test_peak_assessment_classifies_interior_and_edges():
    summary = pd.DataFrame(
        {
            "temperature_K": [300] * 3 + [320] * 3 + [340] * 3,
            "frequency_ghz": [10, 25, 50] * 3,
            "loss_modulus_MPa": [1, 3, 2, 3, 2, 1, 1, 2, 3],
            "status": ["ok"] * 9,
        }
    )
    result = assess_frequency_sweep_peaks(summary)

    assert result["peak_status"].tolist() == ["interior", "lower_edge", "upper_edge"]
    assert result["included_in_fit"].tolist() == [True, False, False]
    assert result["fit_eligible"].tolist() == [True, False, False]
    assert not result["used_in_fit"].any()


def test_failed_qc_is_traceable_and_excluded():
    summary = pd.DataFrame(
        {
            "temperature_K": [300, 300, 300],
            "frequency_ghz": [10, 25, 50],
            "loss_modulus_MPa": [1, 3, 2],
            "status": ["ok", "bad_fit", "ok"],
        }
    )
    result = assess_frequency_sweep_peaks(summary)

    assert bool(result.loc[0, "failed_qc"])
    assert not bool(result.loc[0, "included_in_fit"])
    assert "failed_qc" in result.loc[0, "exclusion_reason"]


def test_peak_fit_fails_clearly_when_interior_series_is_too_small():
    peaks = pd.DataFrame({"included_in_fit": [True, False, False]})
    with pytest.raises(ValueError, match="at least 3 interior"):
        require_interior_peaks(peaks, minimum_interior_peaks=3)


def test_free_tts_preserves_axial_signal_selection():
    rows = []
    for temperature in (300.0, 320.0):
        for frequency, value in ((10.0, 1000.0), (25.0, 1200.0), (50.0, 1500.0)):
            rows.append({"temperature_K": temperature, "frequency_ghz": frequency,
                         "storage_modulus_MPa": value, "signal": "szz_vir_MPa",
                         "parser": "lockin", "status": "ok"})
    shifted, shifts = assign_tts_shifts(pd.DataFrame(rows), 300.0, signal="szz_vir_MPa")
    assert len(shifted) == 6
    assert shifts["log10_aT"].notna().all()


def test_free_tts_rejects_nonoverlapping_curves():
    data = pd.DataFrame({
        "temperature_K": [300.0] * 3 + [320.0] * 3,
        "frequency_ghz": [1.0, 2.0, 3.0, 1.0e30, 2.0e30, 3.0e30],
        "storage_modulus_MPa": [1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
        "signal": ["sxy_vir_MPa"] * 6, "parser": ["lockin"] * 6, "status": ["ok"] * 6,
    })
    with pytest.raises(ValueError, match="no admissible frequency overlap"):
        assign_tts_shifts(data, 300.0)


def test_axial_report_opt_in_extrapolation_and_labels(tmp_path, monkeypatch):
    import dmax2.analysis.master_curve as master_curve
    rows = []
    for temperature in (300.0, 320.0):
        for frequency in (10.0, 25.0, 50.0):
            rows.append({"temperature_K": temperature, "frequency_ghz": frequency,
                         "storage_modulus_MPa": 1000.0 + frequency, "loss_modulus_MPa": 100.0 + frequency,
                         "tan_delta": 0.1, "signal": "szz_vir_MPa", "parser": "lockin", "status": "ok"})
    source = tmp_path / "summary.csv"; pd.DataFrame(rows).to_csv(source, index=False)
    monkeypatch.setattr(master_curve, "build_master_curve_with_mastercurves", lambda *a, **k: (_ for _ in ()).throw(ImportError("disabled")))
    artifacts = write_master_curve_analysis_report(source, tmp_path / "default", MasterCurveSpec(), signal="szz_vir_MPa")
    assert not list((tmp_path / "default").glob("*_extrapolated_to_hz.csv"))
    master = pd.read_csv(artifacts["master_curves"])
    assert master["signal"].eq("szz_vir_MPa").all()
    opt = MasterCurveSpec(enable_low_frequency_extrapolation=True, anchor_reference_source="sensitivity_anchor")
    write_master_curve_analysis_report(source, tmp_path / "opt", opt, signal="szz_vir_MPa")
    out = pd.read_csv(next((tmp_path / "opt").glob("*_extrapolated_to_hz.csv")))
    assert {"frequency_domain_status", "anchor_reference_source"}.issubset(out.columns)
    assert out["anchor_reference_source"].eq("sensitivity_anchor").all()
