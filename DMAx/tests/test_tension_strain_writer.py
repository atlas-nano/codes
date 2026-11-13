from pathlib import Path

from dmax2.config import (
    DMAProductionSpec,
    ForceFieldSpec,
    PolymerSpec,
    PreTensionProtocolSpec,
    TensionProtocolSpec,
)
from dmax2.lammps.inputs import (
    write_pretensioned_tension_strain_convergence_inputs,
    write_tension_strain_convergence_inputs,
)


def test_tension_strain_writer_creates_paired_nvt_npt_inputs(tmp_path: Path):
    data = tmp_path / "equilibrated.data"
    data.write_text("LAMMPS data placeholder\n")
    prod = DMAProductionSpec(
        frequency_ghz=25.0,
        strains=[0.01, 0.02],
        cycles=2,
        discard_cycles=1,
        output_every=50,
        modes=["tension_nvt_z_fixed_lateral", "tension_npt_z_lateral"],
        legacy_modes=[],
        recommended_signal="sdev_MPa",
    )
    protocol = TensionProtocolSpec(npt_equilibration_ps=1.0, nvt_hold_ps=1.0)

    paths = write_tension_strain_convergence_inputs(
        PolymerSpec(name="PDMS_test"),
        ForceFieldSpec(),
        prod,
        protocol,
        tmp_path / "tension",
        data,
    )

    assert len(paths) == 2
    text = paths[0].read_text()
    assert "DMAx2 mode=tension_nvt_z_fixed_lateral" in text
    assert "DMAx2 mode=tension_npt_z_lateral" in text
    assert "fix TENSION_RELAX_NPT all npt" in text
    assert "fix TENSION_HOLD_NVT all nvt" in text
    assert "run 1000" in text
    assert (tmp_path / "tension" / "slurm_array.sh").exists()
    assert (tmp_path / "tension" / "collect_and_parse.sh").exists()


def test_pretensioned_tension_writer_handles_sub_percent_strains(tmp_path: Path):
    data = tmp_path / "equilibrated.data"
    data.write_text("LAMMPS data placeholder\n")
    prod = DMAProductionSpec(
        frequency_ghz=25.0,
        strains=[0.0005, 0.0010],
        cycles=2,
        discard_cycles=1,
        timestep_fs=0.5,
        output_every=50,
        modes=["tension_pretension_nvt_z_fixed_lateral"],
        legacy_modes=[],
        recommended_signal="szz_vir_MPa",
    )
    protocol = PreTensionProtocolSpec(
        ramp_ps=1.0,
        hold_ps=1.0,
        static_stress_to_dynamic_stress_ratio=2.0,
    )

    paths = write_pretensioned_tension_strain_convergence_inputs(
        PolymerSpec(name="PVDF_test"),
        ForceFieldSpec(),
        prod,
        protocol,
        tmp_path / "pretension",
        data,
    )

    assert len(paths) == 2
    assert paths[0].parent.name == "strain_0p0005"
    assert paths[1].parent.name == "strain_0p001"
    text = paths[0].read_text()
    assert "variable static_bias_factor equal 2" in text
    assert "variable static_strain equal v_static_bias_factor*v_amp" in text
    assert "fix PRETENSION_DEFORM all deform 1 z erate" in text
    assert "raw__tension_pretension_nvt_z_fixed_lateral__strain_0p0005.dat" in text
    assert "raw__tension_pretension_npt_z_lateral__strain_0p0005.dat" in text
    assert (tmp_path / "pretension" / "slurm_array.sh").exists()
    assert (tmp_path / "pretension" / "collect_and_parse.sh").exists()
