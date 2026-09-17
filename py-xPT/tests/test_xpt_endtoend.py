"""End-to-end regression test: 2PT/3PT entropy from a bundled mini LJ-argon
trajectory.  Verifies the full pipeline (trajectory -> VACF -> DoS -> fluidicity
-> gas/cage/solid partition -> entropy) runs and reproduces a recorded value.

Run from the repo root after `pip install -e .` (or with PYTHONPATH=src):
    pytest
"""
import re, sys, shutil, subprocess
from pathlib import Path

FIX = Path(__file__).parent / "fixtures"

def _run(tmp_path, cage_ws=None):
    shutil.copy(FIX / "lj_argon_mini.lammpstrj", tmp_path / "lj_argon_mini.lammpstrj")
    ini = (FIX / "lj_argon_mini.ini").read_text()
    if cage_ws is not None:
        ini = ini.replace("[thermodynamics]", f"[thermodynamics]\ncage_ws = {cage_ws}", 1)
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
    assert abs(_val(thermo, "S_q (S*/atom)")   - 7.166) < 0.001
    assert abs(_val(thermo, "S_cage (S*/atom)") - 0.334) < 0.001
    assert abs(_val(thermo, "Fluidicity (trans)") - 0.35169) < 1e-3


def test_lj_3pt_legacy_cage_weight(tmp_path):
    # cage_ws = legacy reproduces the 1.0.1 values (S_cage now in S*, 2.795 J/mol/K / R)
    thermo = _run(tmp_path, cage_ws="legacy")
    assert abs(_val(thermo, "S_q (S*/atom)")   - 7.168) < 0.001
    assert abs(_val(thermo, "S_cage (S*/atom)") - 0.336) < 0.001
