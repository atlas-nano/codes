"""Fast checks of the d != 3 branch (2.0.1 fixes) and of its d = 3 invariance.

The trajectory tests derive a small input (the 150 fixture frames, 64 atoms)
from the bundled fixture, so they need no extra data and run in a few seconds.
"""
import logging
import math
import types
from pathlib import Path

import numpy as np
import pytest

from pyxpt.constants import PI, KB, NA, R, VLIGHT
from pyxpt.thermo.utility import (
    enskog_K_parameter, chapman_enskog_cd, dos0_to_diffusivity, trapz_dos,
)

FIX = Path(__file__).parent / "fixtures"
NFRAMES, NATOMS = 150, 64
AREA = 28.752 ** 2          # in-plane area of the fixture box [A^2]


def _k_pre201(s0, T, m, n, vol, d):
    """The K expression of releases up to 2.0.0 (2/(3d) for every d)."""
    v_d = {1: 1.0, 2: PI / 4.0, 3: PI / 6.0, 4: PI * PI / 32.0}[d]
    return (s0 / VLIGHT * 1e-2) / n * math.sqrt(
        PI * NA * KB * T / (m * 1e-3)
    ) * 2.0 / (3.0 * d) * (n / vol) ** (1.0 / d) * 1e10 \
        * v_d ** (-(d - 1.0) / d)


# ── (a) Enskog K parameter ─────────────────────────────────────────────────────

def test_chapman_enskog_prefactors():
    assert chapman_enskog_cd(2) == 0.5
    assert abs(chapman_enskog_cd(3) - 3.0 / 8.0) < 1e-15
    assert abs(chapman_enskog_cd(4) - 1.0 / PI) < 1e-15


def test_enskog_K_d3_unchanged_d2_corrected():
    args = (2.7, 97.3, 39.948, 512, 23768.63)
    assert enskog_K_parameter(*args, dimension=3) == _k_pre201(*args, 3)
    args2 = (2.7, 97.3, 39.948, 512, 826.7)
    ratio = enskog_K_parameter(*args2, dimension=2) / _k_pre201(*args2, 2)
    assert abs(ratio - 0.75) < 1e-12
    ratio4 = enskog_K_parameter(*args2, dimension=4) / _k_pre201(*args2, 4)
    assert abs(ratio4 - (PI / 16.0) / (2.0 / 12.0)) < 1e-12


# ── (b) diffusivity from S(0) ──────────────────────────────────────────────────

def test_dos0_to_diffusivity():
    s0, T, m = 3.1, 120.0, 20453.376
    assert dos0_to_diffusivity(s0, T, m, 3) == s0 * R * T / (12.0 * VLIGHT * m) * 1e5
    assert abs(dos0_to_diffusivity(s0, T, m, 2) / dos0_to_diffusivity(s0, T, m, 3)
               - 1.5) < 1e-14
    # round trip through S(0) = 4 d m N D / (k_B T), everything in SI
    for d in (2, 3):
        D_cm2 = 1.7e-5
        mN_kg = m * 1e-3 / NA
        s0_si = 4.0 * d * mN_kg * (D_cm2 * 1e-4) / (KB * T)      # s
        s0_cm = s0_si * VLIGHT * 100.0                           # DoS per cm^-1
        assert abs(dos0_to_diffusivity(s0_cm, T, m, d) / D_cm2 - 1.0) < 1e-12


# ── D2: packing fraction in slit mode ──────────────────────────────────────────

def _fake_engine(dof_axes, n, mass, vol, nu, nused):
    grp = types.SimpleNamespace(nmol=n, mass=mass, volume=vol)
    cfg = types.SimpleNamespace(dimension=2, fixed_fluidicity=[],
                                finite_size_correction="off", d0_hs_scale=1.0)
    return types.SimpleNamespace(
        sys=types.SimpleNamespace(ngrp=1, groups=[grp]), cfg=cfg,
        _pwrfreq=nu, _nused=nused, _apply_xpt=True, _do_desjarlais=False,
        _molecular=False, _dof_axes=dof_axes)


def test_slit_packing_matches_strict_2d():
    """A bound third axis must not change the packing of the in-plane gas."""
    from pyxpt.thermo._xpt_compute import compute_xpt
    n, T, mass, nu, nused = 100, 100.0, 100 * 39.948, 0.5, 40000
    s0 = 0.4 / enskog_K_parameter(1.0, T, mass / n, n, AREA, dimension=2)
    freq = np.arange(nused) * nu
    inplane = s0 / (1.0 + (PI * s0 * freq / (2.0 * 2 * n)) ** 2)   # f = 1 shape, integral 2N
    zpeak = np.exp(-0.5 * ((freq - 150.0) / 10.0) ** 2)
    zpeak *= n / trapz_dos(zpeak, nu)                                # integral N
    vacT = np.array([[T]])
    y = {}
    for label, dos, axes in (("2d", inplane, 2), ("slit", inplane + zpeak, 3)):
        eng = _fake_engine({0: axes}, n, mass, AREA, nu, nused)
        f, _, y_arr, hsd, _ = compute_xpt(eng, dos[None, None, :], vacT,
                                          None, AREA, T)
        y[label] = (f[0, 0], y_arr[0], hsd[0, 0], trapz_dos(dos, nu))
    assert y["2d"][0] == y["slit"][0]                    # same fluidicity
    assert 0.0 < y["2d"][1] < 0.9
    assert abs(y["slit"][1] / y["2d"][1] - 1.0) < 2e-3
    # the 2.0.0 form, (f/K)^2 * hsd/tdf, gave 2/3 of it
    f0, _, hsd0, tdf0 = y["slit"]
    assert abs((hsd0 / y["slit"][3]) / (hsd0 / y["2d"][3]) - 2.0 / 3.0) < 2e-3


# ── (c)/(d) a 3-column dump read with dimension = 2 ────────────────────────────

def _tiny_dump(dst, zero_vz=False):
    with open(FIX / "lj_argon_mini.lammpstrj") as fh, open(dst, "w") as out:
        nf = 0
        while nf < NFRAMES:
            hdr = [fh.readline() for _ in range(9)]
            nat = int(hdr[3])
            rows = [fh.readline() for _ in range(nat)]
            hdr[3] = f"{NATOMS}\n"
            out.writelines(hdr)
            for ln in rows:
                t = ln.split()
                if int(t[0]) <= NATOMS:
                    if zero_vz:
                        t[7] = "0"
                    out.write(" ".join(t) + "\n")
            nf += 1


def _tiny_ini(tmp_path, zero_vz=False):
    _tiny_dump(tmp_path / "tiny.lammpstrj", zero_vz)
    ini = (FIX / "lj_argon_mini.ini").read_text()
    ini = ini.replace("lj_argon_mini.lammpstrj", "tiny.lammpstrj")
    ini = ini.replace("[thermodynamics]", "[thermodynamics]\ndimension = 2", 1)
    ini = ini.replace("[system_properties]",
                      f"[system_properties]\nvolume = {AREA:.4f}", 1)
    ini = ini.replace("prefix          = mini", "prefix          = tiny")
    (tmp_path / "tiny.ini").write_text(ini)
    return tmp_path / "tiny.ini"


@pytest.mark.parametrize("zero_vz, kept", [(False, 3), (True, 2)])
def test_dimension2_reads_three_column_dump(tmp_path, monkeypatch, caplog,
                                            zero_vz, kept):
    from pyxpt.analysis import run
    ini = _tiny_ini(tmp_path, zero_vz)
    monkeypatch.chdir(tmp_path)
    with caplog.at_level(logging.INFO, logger="pyxpt"):
        res = run(ini, log_level=logging.WARNING)
    assert f"{kept} velocity component(s) kept" in caplog.text
    assert res.vac.shape[-1] == kept
    assert np.isfinite(res.S_quantum[0, 0]) and res.S_quantum[0, 0] > 0
    assert 0.0 < res.fluidicity[0, 0] < 1.0
    assert res.packing_fraction[0] > 0.0
    assert (tmp_path / "tiny.thermo").exists()


def test_dimension2_low_ram_path(tmp_path, monkeypatch):
    import pyxpt.memory_budget as mb
    from pyxpt.analysis import run
    ini = _tiny_ini(tmp_path)
    monkeypatch.chdir(tmp_path)
    single = run(ini, log_level=logging.WARNING)
    s_single = single.S_quantum.copy()
    monkeypatch.setattr(mb, "compute_n_batches", lambda cost, budget, spo: 3)
    batched = run(ini, log_level=logging.WARNING)
    assert batched.vac.shape[-1] == 3
    # float32 VACF sums change order across batches (~1e-7 relative, as at d = 3)
    np.testing.assert_allclose(batched.S_quantum, s_single, rtol=1e-6)


# ── configuration warnings ─────────────────────────────────────────────────────

def test_config_warnings(tmp_path, caplog):
    import pyxpt.config.base as base
    from pyxpt.config import load
    (tmp_path / "w.ini").write_text(
        "[files]\ntrajectory = x.lammpstrj\ntrajectory_format = LAMMPSDUMP\n"
        "[system_properties]\ntimestep = 0.008\nconstraints = 3\n"
        "[thermodynamics]\nmode = 2PT\ndimension = 2\n")
    base._EXPERIMENTAL_DIM_WARNED = False
    with caplog.at_level(logging.WARNING):
        cfg = load(tmp_path / "w.ini")
    assert cfg.constraints == 0
    assert "constraints = 3 is IGNORED" in caplog.text
    assert "EXPERIMENTAL" in caplog.text
