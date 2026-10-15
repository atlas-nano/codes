"""End-to-end regression test: 2PT/3PT entropy from a bundled mini LJ-argon
trajectory.  Verifies the full pipeline (trajectory -> VACF -> DoS -> fluidicity
-> gas/cage/solid partition -> entropy) runs and reproduces a recorded value.

Run from the repo root after `pip install -e .` (or with PYTHONPATH=src):
    pytest
"""
import re, sys, shutil, subprocess
from pathlib import Path

FIX = Path(__file__).parent / "fixtures"

def _run(tmp_path, cage_ws=None, quadrature=None):
    shutil.copy(FIX / "lj_argon_mini.lammpstrj", tmp_path / "lj_argon_mini.lammpstrj")
    ini = (FIX / "lj_argon_mini.ini").read_text()
    for key, val in (("cage_ws", cage_ws), ("cage_quadrature", quadrature)):
        if val is not None:
            ini = ini.replace("[thermodynamics]", f"[thermodynamics]\n{key} = {val}", 1)
    (tmp_path / "lj_argon_mini.ini").write_text(ini)
    r = subprocess.run([sys.executable, "-m", "pyxpt", "lj_argon_mini.ini"],
                       cwd=tmp_path, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return (tmp_path / "mini.thermo").read_text()

def _val(thermo, key):
    m = re.search(re.escape(key) + r"\s+([-\d.eE+]+)", thermo)
    assert m, f"{key!r} not found in .thermo"
    return float(m.group(1))

def test_lj_3pt_endtoend(tmp_path):
    thermo = _run(tmp_path)
    # recorded reference values for the bundled 150-frame fixture
    # default: trapezoidal (second-order) product integration in the kernel inversion
    assert abs(_val(thermo, "S_q (S*/atom)")   - 7.048) < 0.001
    assert abs(_val(thermo, "S_cage (S*/atom)") - 0.216) < 0.001
    assert abs(_val(thermo, "Fluidicity (trans)") - 0.35169) < 1e-3


def test_lj_3pt_legacy_cage_weight(tmp_path):
    # cage_ws = legacy + the first-order rule reproduce the 1.0.1/2.0.x values
    # (S_cage now in S*, 2.795 J/mol/K / R)
    thermo = _run(tmp_path, cage_ws="legacy", quadrature="rectangle")
    assert abs(_val(thermo, "S_q (S*/atom)")   - 7.172) < 0.001
    assert abs(_val(thermo, "S_cage (S*/atom)") - 0.340) < 0.001


def test_lj_3pt_rectangle_quadrature(tmp_path):
    # the released 2.0.x default (quantum weight, first-order rule)
    thermo = _run(tmp_path, quadrature="rectangle")
    assert abs(_val(thermo, "S_q (S*/atom)")   - 7.170) < 0.001
    assert abs(_val(thermo, "S_cage (S*/atom)") - 0.338) < 0.001


def test_cage_guard_zeroes_unphysical_cages():
    """A negative, non-finite, or oversized cage is set to 0."""
    import math
    from pyxpt.thermo.utility import cage_guard
    assert cage_guard((0.2, 0.25), 3) == (0.2, 0.25)
    assert cage_guard((-0.01, 0.02), 3) == (0.0, 0.0)
    assert cage_guard((math.nan, 0.1), 3) == (0.0, 0.0)
    assert cage_guard((9.5, 9.0), 3) == (0.0, 0.0)          # > 3 k_B per DoF x 3 DoF


def test_cage_uses_baseline_gas_weight(tmp_path):
    """The translational cage's W_g is the baseline's own per-DoF
    gas weight (hs_weighting wsp), saved with the cage inputs as wg_base."""
    import os
    import numpy as np
    shutil.copy(FIX / "lj_argon_mini.lammpstrj", tmp_path / "lj_argon_mini.lammpstrj")
    (tmp_path / "lj_argon_mini.ini").write_text((FIX / "lj_argon_mini.ini").read_text())
    env = dict(os.environ, CAGE_SAVE_INPUTS="cin_t53")
    r = subprocess.run([sys.executable, "-m", "pyxpt", "lj_argon_mini.ini"],
                       cwd=tmp_path, capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr
    wg = float(np.load(tmp_path / "cin_t53_grp1.npz")["wg_base"])
    assert np.isfinite(wg) and abs(wg - 3.13208) < 1e-4

