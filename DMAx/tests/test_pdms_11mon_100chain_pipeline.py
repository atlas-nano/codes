from pathlib import Path

import pandas as pd

from dmax2.config import DMAx2WorkflowSpec
from dmax2.workflow import DMAx2Workflow


def test_pdms_11mon_100chain_pipeline(tmp_path):
    config = Path(__file__).resolve().parents[1] / "examples" / "pdms_11mon_100chain.yaml"
    spec = DMAx2WorkflowSpec.from_file(config)
    result = DMAx2Workflow(spec).run_all(tmp_path, dry_run=True, synthetic=True)

    assert (tmp_path / "in.PDMS_dmax2_equil").exists()
    assert (tmp_path / "in.PDMS_dmax2_dma").exists()
    assert (tmp_path / "raw").is_dir()
    assert "dma_summary" in result.artifacts

    summary = pd.read_csv(tmp_path / "dma_summary.csv")
    assert {"lockin", "legacy"}.issubset(set(summary["parser"]))
    best = summary[
        (summary["mode"] == "shear_sllod_remapv")
        & (summary["signal"] == "sxy_vir_MPa")
        & (summary["parser"] == "lockin")
    ]
    assert not best.empty
    assert best["snr_first_harmonic"].median() > 2.0

    recs = pd.read_csv(tmp_path / "dma_strain_recommendations.csv")
    assert not recs.empty
    cycle = pd.read_csv(tmp_path / "dma_cycle_convergence.csv")
    assert not cycle.empty
    cycle_recs = pd.read_csv(tmp_path / "dma_cycle_recommendations.csv")
    assert not cycle_recs.empty
    assert cycle_recs["cycles"].iloc[0] <= spec.production.cycles

    assert (tmp_path / "glass_transition_fit.csv").exists()
    tg = pd.read_csv(tmp_path / "glass_transition_fit.csv")
    assert {"tan_delta", "density_piecewise"}.issubset(set(tg["estimator"]))
    assert (tmp_path / "glass_transition_tandelta_fit.csv").exists()

    assert (tmp_path / "master_curve.csv").exists()
