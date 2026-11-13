"""LAMMPS input writers for DMAx2."""

from __future__ import annotations

from dataclasses import replace
from math import ceil, log10
from pathlib import Path

from ..config import (
    DMAProductionSpec,
    EquilibrationSpec,
    ErrorAnalysisSpec,
    ForceFieldSpec,
    GlassTransitionSpec,
    MasterCurveSpec,
    PolymerSpec,
    PreTensionProtocolSpec,
    TensionProtocolSpec,
)
from ..polymer_properties import glass_temperature_grid


ATM_TO_MPA = 0.101325


def _header(polymer: PolymerSpec, ff: ForceFieldSpec) -> list[str]:
    return [
        "units real",
        "atom_style full",
        "boundary p p p",
        "dielectric 1.0",
        f"pair_style {ff.pair_style}",
        f"bond_style {ff.bond_style}",
        f"angle_style {ff.angle_style}",
        f"dihedral_style {ff.dihedral_style}",
        f"improper_style {ff.improper_style}",
        f"kspace_style {ff.kspace_style}",
        f"pair_modify {ff.pair_modify}",
        f"special_bonds {ff.special_bonds}",
        f"read_data data.{polymer.name}",
        "neighbor 2.0 bin",
        "neigh_modify every 1 delay 0 check yes",
    ]


def write_equilibration_input(
    polymer: PolymerSpec,
    ff: ForceFieldSpec,
    eq: EquilibrationSpec,
    workdir: str | Path,
    filename: str | None = None,
) -> Path:
    """Write conservative amorphous-polymer equilibration input.

    The template is intentionally explicit: it produces restart files, staged
    data snapshots, and an equilibration trace suitable for the paper-style
    density/pressure/box-length plots.
    """

    workdir = Path(workdir)
    path = workdir / (filename or f"in.{polymer.name}_dmax2_equil")
    data_file = eq.data_file or polymer.existing_data_file or f"data.{polymer.name}"
    outdir = eq.output_directory or f"{polymer.name}_dmax2_equil"
    anneal_t = eq.anneal_temperature_k or max(eq.temperature_k + 100.0, 450.0)
    cool_t = eq.cool_temperature_k or eq.temperature_k
    pdamp_short = eq.pdamp_short_fs or max(eq.pdamp_fs, 10000.0)
    pdamp_long = eq.pdamp_long_fs or max(eq.pdamp_fs, 50000.0)
    anneal_steps = eq.anneal_steps if eq.anneal_steps is not None else eq.npt_anneal_steps
    cool_steps = eq.cool_steps if eq.cool_steps is not None else eq.npt_anneal_steps
    relax_steps = eq.relax_npt_steps if eq.relax_npt_steps is not None else eq.npt_relax_steps
    final_nvt_steps = eq.final_nvt_steps if eq.final_nvt_steps is not None else eq.nvt_final_steps
    lines = [
        "# DMAx2 conservative amorphous-polymer equilibration input.",
        "clear",
        "units real",
        "atom_style full",
        "boundary p p p",
        "dielectric 1.0",
        f"variable sname string {polymer.name}",
        f"variable datafile string {data_file}",
        f"variable outdir string {outdir}",
        "variable restartdir string ${outdir}/restart",
        f"variable rtemp equal {eq.temperature_k:g}",
        f"variable ptarget equal {eq.pressure_atm:g}",
        f"variable anneal_temp equal {anneal_t:g}",
        f"variable dt_fs equal {eq.timestep_fs:g}",
        f"variable tdamp equal {eq.tdamp_fs:g}",
        f"variable pdamp_short equal {pdamp_short:g}",
        f"variable pdamp_long equal {pdamp_long:g}",
        "variable seed equal 274637",
        f"variable heat_steps equal {eq.heat_steps}",
        f"variable target_npt_steps equal {eq.target_npt_steps}",
        f"variable anneal_steps equal {anneal_steps}",
        f"variable cool_steps equal {cool_steps}",
        f"variable relax_npt_steps equal {relax_steps}",
        f"variable final_nvt_steps equal {final_nvt_steps}",
        f"variable min_steps equal {eq.minimize_steps}",
        f"variable min_evals equal {eq.minimize_evaluations}",
        "variable thermo_every equal 2000",
        f"variable trace_every equal {eq.trace_every}",
        f"variable dump_every equal {eq.dump_every}",
        f"variable restart_every equal {eq.restart_every}",
        "",
        "shell mkdir -p ${outdir}",
        "shell mkdir -p ${restartdir}",
        "log ${outdir}/${sname}.dmax2_equil.lammps.log",
        "",
        f"pair_style {ff.pair_style}",
        f"bond_style {ff.bond_style}",
        f"angle_style {ff.angle_style}",
        f"dihedral_style {ff.dihedral_style}",
        f"improper_style {ff.improper_style}",
        f"kspace_style {ff.kspace_style}",
        f"pair_modify {ff.pair_modify}",
        f"special_bonds {ff.special_bonds}",
        "read_data ${datafile}",
        "",
        "neighbor 2.0 multi",
        "neigh_modify every 1 delay 0 check yes",
        "comm_modify vel yes",
        "timestep ${dt_fs}",
        "reset_timestep 0",
        "",
        "compute Pvir all pressure NULL virial",
        f"variable atm2mpa equal {ATM_TO_MPA}",
        "variable time_ps equal time/1000.0",
        "variable temp_K equal temp",
        "variable press_MPa equal press*v_atm2mpa",
        "variable pxx_MPa equal pxx*v_atm2mpa",
        "variable pyy_MPa equal pyy*v_atm2mpa",
        "variable pzz_MPa equal pzz*v_atm2mpa",
        "variable pxy_MPa equal pxy*v_atm2mpa",
        "variable pxz_MPa equal pxz*v_atm2mpa",
        "variable pyz_MPa equal pyz*v_atm2mpa",
        "variable szz_vir_MPa equal -c_Pvir[3]*v_atm2mpa",
        "variable sdev_vir_MPa equal -(c_Pvir[3]-0.5*(c_Pvir[1]+c_Pvir[2]))*v_atm2mpa",
        "variable lx_A equal lx",
        "variable ly_A equal ly",
        "variable lz_A equal lz",
        "variable density_g_cm3 equal density",
        "variable vol_A3 equal vol",
        "",
        "thermo ${thermo_every}",
        "thermo_style custom step time temp press pxx pyy pzz pxy pxz pyz lx ly lz density pe ke etotal",
        "thermo_modify line one format float %20.10g flush yes",
        "",
        "fix EQTRACE all ave/time ${trace_every} 1 ${trace_every} &",
        "  v_time_ps v_temp_K v_press_MPa v_pxx_MPa v_pyy_MPa v_pzz_MPa &",
        "  v_pxy_MPa v_pxz_MPa v_pyz_MPa v_szz_vir_MPa v_sdev_vir_MPa &",
        "  v_lx_A v_ly_A v_lz_A v_density_g_cm3 v_vol_A3 &",
        "  file ${outdir}/${sname}.equil_trace.dat &",
        '  title1 "# step time_ps temp_K press_MPa pxx_MPa pyy_MPa pzz_MPa pxy_MPa pxz_MPa pyz_MPa szz_vir_MPa sdev_vir_MPa lx_A ly_A lz_A density_g_cm3 vol_A3"',
        "restart ${restart_every} ${restartdir}/${sname}.*.restart",
        "",
        'print "============================================================"',
        'print "DMAx2 equilibration: minimization"',
        'print "============================================================"',
        "dump MIN all custom ${dump_every} ${outdir}/${sname}.min.lammpstrj id mol type q xu yu zu",
        "dump_modify MIN sort id",
        "min_style sd",
        "minimize 0.0 1.0e-4 ${min_steps} ${min_evals}",
        "min_style cg",
        "minimize 0.0 1.0e-4 ${min_steps} ${min_evals}",
        "undump MIN",
        "",
        'print "============================================================"',
        'print "DMAx2 equilibration: gentle NVT heat"',
        'print "============================================================"',
        "velocity all create 1.0 ${seed} mom yes rot yes dist gaussian",
        "dump HEAT all custom ${dump_every} ${outdir}/${sname}.heat.lammpstrj id mol type q xu yu zu vx vy vz",
        "dump_modify HEAT sort id",
        "fix HEAT all nvt temp 1.0 ${rtemp} ${tdamp}",
        "run ${heat_steps}",
        "unfix HEAT",
        "undump HEAT",
        "write_data ${outdir}/${sname}.after_heat.data",
        "",
        'print "============================================================"',
        'print "DMAx2 equilibration: target-temperature NPT relaxation"',
        'print "============================================================"',
        "dump NPT0 all custom ${dump_every} ${outdir}/${sname}.target_npt.lammpstrj id mol type q xu yu zu vx vy vz",
        "dump_modify NPT0 sort id",
        "fix NPT0 all npt temp ${rtemp} ${rtemp} ${tdamp} aniso ${ptarget} ${ptarget} ${pdamp_short} drag 1.0",
        "run ${target_npt_steps}",
        "unfix NPT0",
        "undump NPT0",
        "write_data ${outdir}/${sname}.after_target_npt.data",
        "",
        'print "============================================================"',
        'print "DMAx2 equilibration: high-temperature NPT anneal"',
        'print "============================================================"',
        "dump ANNEAL all custom ${dump_every} ${outdir}/${sname}.anneal.lammpstrj id mol type q xu yu zu vx vy vz",
        "dump_modify ANNEAL sort id",
        "fix ANNEAL all npt temp ${rtemp} ${anneal_temp} ${tdamp} aniso ${ptarget} ${ptarget} ${pdamp_short} drag 1.0",
        "run ${anneal_steps}",
        "unfix ANNEAL",
        "undump ANNEAL",
        "write_data ${outdir}/${sname}.after_anneal.data",
        "",
        'print "============================================================"',
        'print "DMAx2 equilibration: slow NPT cooling"',
        'print "============================================================"',
        "dump COOL all custom ${dump_every} ${outdir}/${sname}.cool.lammpstrj id mol type q xu yu zu vx vy vz",
        "dump_modify COOL sort id",
        "fix COOL all npt temp ${anneal_temp} " + f"{cool_t:g}" + " ${tdamp} aniso ${ptarget} ${ptarget} ${pdamp_short} drag 1.0",
        "run ${cool_steps}",
        "unfix COOL",
        "undump COOL",
        "write_data ${outdir}/${sname}.after_cool.data",
        "",
        'print "============================================================"',
        'print "DMAx2 equilibration: long target-temperature NPT relaxation"',
        'print "============================================================"',
        "dump RELAX all custom ${dump_every} ${outdir}/${sname}.relax_npt.lammpstrj id mol type q xu yu zu vx vy vz",
        "dump_modify RELAX sort id",
        "fix RELAX all npt temp ${rtemp} ${rtemp} ${tdamp} aniso ${ptarget} ${ptarget} ${pdamp_long} drag 1.0",
        "run ${relax_npt_steps}",
        "unfix RELAX",
        "undump RELAX",
        "write_data ${outdir}/${sname}.after_relax_npt.data",
        "",
        'print "============================================================"',
        'print "DMAx2 equilibration: final fixed-cell NVT hold"',
        'print "============================================================"',
        "dump FINAL all custom ${dump_every} ${outdir}/${sname}.final_nvt.lammpstrj id mol type q xu yu zu vx vy vz",
        "dump_modify FINAL sort id",
        "fix FINAL all nvt temp ${rtemp} ${rtemp} ${tdamp}",
        "run ${final_nvt_steps}",
        "unfix FINAL",
        "undump FINAL",
        "",
        "write_data ${outdir}/${sname}.dmax2.equilibrated.data",
        "write_restart ${outdir}/${sname}.dmax2.equilibrated.restart",
        "write_data ${sname}.dmax2.equilibrated.data",
    ]
    path.write_text("\n".join(lines) + "\n")
    return path


def _common_stress_output(temperature_compute: str = "thermo_temp") -> list[str]:
    return [
        f"compute Ptot all pressure {temperature_compute}",
        "compute Pvir all pressure NULL virial",
        f"variable atm2mpa equal {ATM_TO_MPA}",
        "variable istep equal step",
        "variable temp_K equal temp",
        "variable lxv equal lx",
        "variable lyv equal ly",
        "variable lzv equal lz",
        "variable densityv equal density",
        "variable szz_tot_MPa equal -c_Ptot[3]*v_atm2mpa",
        "variable szz_vir_MPa equal -c_Pvir[3]*v_atm2mpa",
        "variable sdev_MPa equal -(c_Pvir[3]-0.5*(c_Pvir[1]+c_Pvir[2]))*v_atm2mpa",
        "variable sxy_tot_MPa equal -c_Ptot[4]*v_atm2mpa",
        "variable sxy_vir_MPa equal -c_Pvir[4]*v_atm2mpa",
        "variable press_MPa equal press*v_atm2mpa",
        "variable t_ps equal step*dt/1000.0",
    ]


def _temperature_token(temperature_k: float) -> str:
    return f"{temperature_k:07.3f}".replace(".", "p")


def _frequency_token(frequency_ghz: float) -> str:
    return f"{frequency_ghz:07.3f}".replace(".", "p")


def _strain_token(strain: float) -> str:
    text = f"{strain:.6f}".rstrip("0").rstrip(".")
    if "." not in text:
        text += ".0"
    whole, frac = text.split(".", 1)
    frac = frac.ljust(3, "0")
    return f"{whole}p{frac}"


def _print_line(
    mode: str,
    strain: float,
    output_every: int,
    raw_dir: str,
    temperature_k: float | None = None,
    frequency_ghz: float | None = None,
    replicate_index: int | None = None,
    velocity_seed: int | None = None,
) -> str:
    token = _strain_token(strain)
    freq_suffix = f"__f_{_frequency_token(frequency_ghz)}GHz" if frequency_ghz is not None else ""
    suffix = f"__T_{_temperature_token(temperature_k)}K" if temperature_k is not None else ""
    rep_suffix = f"__rep_{replicate_index:03d}" if replicate_index is not None else ""
    seed_suffix = f"__seed_{velocity_seed}" if velocity_seed is not None else ""
    file = f"{raw_dir}/raw__{mode}__strain_{token}{freq_suffix}{suffix}{rep_suffix}{seed_suffix}.dat"
    title = (
        '"step t_ps strain_applied strain_box eps_x eps_y szz_tot_MPa szz_vir_MPa '
        'sdev_MPa sxy_tot_MPa sxy_vir_MPa temp_K press_MPa lx ly lz density"'
    )
    values = (
        "${istep} ${t_ps} ${strain_applied} ${strain_box} ${eps_x} ${eps_y} "
        "${szz_tot_MPa} ${szz_vir_MPa} ${sdev_MPa} ${sxy_tot_MPa} ${sxy_vir_MPa} "
        "${temp_K} ${press_MPa} ${lxv} ${lyv} ${lzv} ${densityv}"
    )
    return f'fix OUT all print {output_every} "{values}" file {file} screen no title {title}'


def _dma_read_header(ff: ForceFieldSpec, data: str, timestep_fs: float) -> list[str]:
    return [
        "clear",
        "units real",
        "atom_style full",
        "boundary p p p",
        "dielectric 1.0",
        f"pair_style {ff.pair_style}",
        f"bond_style {ff.bond_style}",
        f"angle_style {ff.angle_style}",
        f"dihedral_style {ff.dihedral_style}",
        f"improper_style {ff.improper_style}",
        f"kspace_style {ff.kspace_style}",
        f"pair_modify {ff.pair_modify}",
        f"special_bonds {ff.special_bonds}",
        f"read_data {data}",
        f"timestep {timestep_fs:g}",
        "thermo 1000",
        "thermo_style custom step temp press density lx ly lz pxx pyy pzz pxy",
        "thermo_modify flush yes",
        "shell mkdir -p raw",
    ]


def _mode_block(
    mode: str,
    strain: float,
    prod: DMAProductionSpec,
    raw_dir: str,
    kspace_style: str,
    temperature_k: float | None = None,
    include_frequency_in_raw_name: bool = False,
    replicate_index: int | None = None,
    velocity_seed: int | None = None,
) -> list[str]:
    period = prod.period_fs
    run_steps = prod.run_steps
    tdamp = prod.tdamp_fs
    temp = prod.temperature_k
    pdamp = prod.pdamp_fs
    common = [
        f"variable amp equal {strain:g}",
        f"variable period_fs equal {period:g}",
        "variable lx0 equal lx",
        "variable ly0 equal ly",
        "variable lz0 equal lz",
        "variable strain_applied equal v_amp*sin(2.0*PI*step*dt/v_period_fs)",
        "variable eps_x equal (lx-v_lx0)/v_lx0",
        "variable eps_y equal (ly-v_ly0)/v_ly0",
    ]
    if prod.validated_mechanics:
        # Freeze pre-drive reference lengths at input evaluation. `equal lz`
        # is live and makes (lz-v_lz0)/v_lz0 vanish as the box moves.
        frozen = {f"variable {axis}0 equal {axis}": f"variable {axis}0 equal $({axis}:%.17g)"
                  for axis in ("lx", "ly", "lz")}
        common = [frozen.get(line, line) for line in common]

    uses_temp_deform = False
    if mode == "shear_sllod_remapv":
        uses_temp_deform = True
        setup = [
            "change_box all triclinic",
            f"kspace_style {kspace_style}",
            "variable amp_tilt equal v_amp*v_ly0",
            "variable strain_box equal xy/v_ly0",
            "compute TDEF all temp/deform",
            f"fix TH all nvt/sllod temp {temp:g} {temp:g} {tdamp:g} tchain {1 if prod.validated_mechanics else 3}",
            "fix_modify TH temp TDEF",
            "fix DEF all deform 1 xy wiggle ${amp_tilt} ${period_fs} remap v units box flip no",
        ]
    elif mode == "shear_nvt_remapx":
        setup = [
            "change_box all triclinic",
            f"kspace_style {kspace_style}",
            "variable amp_tilt equal v_amp*v_ly0",
            "variable strain_box equal xy/v_ly0",
            f"fix TH all nvt temp {temp:g} {temp:g} {tdamp:g}",
            "fix DEF all deform 1 xy wiggle ${amp_tilt} ${period_fs} remap x units box flip no",
        ]
    elif mode in {"axial_nvt_remapx_volcons", "tension_nvt_z_volcons"}:
        setup = [
            "variable amp_z equal v_amp*v_lz0",
            "variable strain_box equal (lz-v_lz0)/v_lz0",
            f"fix TH all nvt temp {temp:g} {temp:g} {tdamp:g}",
            "fix DEF all deform 1 z wiggle ${amp_z} ${period_fs} x volume y volume remap x units box",
        ]
    elif mode in {"tension_nvt_z_fixed_lateral", "tension_pretension_nvt_z_fixed_lateral"}:
        setup = [
            "variable amp_z equal v_amp*v_lz0",
            "variable strain_box equal (lz-v_lz0)/v_lz0",
            f"fix TH all nvt temp {temp:g} {temp:g} {tdamp:g}",
            "fix DEF all deform 1 z wiggle ${amp_z} ${period_fs} remap x units box",
        ]
    elif mode == "axial_sllod_remapv_volcons":
        uses_temp_deform = True
        setup = [
            "variable amp_z equal v_amp*v_lz0",
            "variable strain_box equal (lz-v_lz0)/v_lz0",
            "compute TDEF all temp/deform",
            f"fix TH all nvt/sllod temp {temp:g} {temp:g} {tdamp:g} tchain {1 if prod.validated_mechanics else 3}",
            "fix_modify TH temp TDEF",
            "fix DEF all deform 1 z wiggle ${amp_z} ${period_fs} x volume y volume remap v units box",
        ]
    elif mode in {"axial_npt_lateral", "tension_npt_z_lateral", "tension_pretension_npt_z_lateral"}:
        setup = [
            "variable amp_z equal v_amp*v_lz0",
            "variable strain_box equal (lz-v_lz0)/v_lz0",
            f"fix TH all npt temp {temp:g} {temp:g} {tdamp:g} x 1.0 1.0 {pdamp:g} y 1.0 1.0 {pdamp:g} couple xy",
            "fix DEF all deform 1 z wiggle ${amp_z} ${period_fs} remap x units box",
        ]
    elif mode == "legacy_axial_pressure":
        setup = [
            "variable amp_z equal v_amp*v_lz0",
            "variable strain_box equal (lz-v_lz0)/v_lz0",
            f"fix TH all nvt temp {temp:g} {temp:g} {tdamp:g}",
            "fix DEF all deform 1 z wiggle ${amp_z} ${period_fs} x volume y volume remap x units box",
        ]
    else:
        raise ValueError(f"Unknown production mode: {mode}")

    cleanup = [
        "unfix OUT",
        "unfix DEF",
        "unfix TH",
    ]
    if uses_temp_deform:
        cleanup.append("uncompute TDEF")
    cleanup.extend(["uncompute Ptot", "uncompute Pvir"])

    stress_output = _common_stress_output("TDEF" if uses_temp_deform and prod.validated_mechanics else "thermo_temp")
    if uses_temp_deform and prod.validated_mechanics:
        stress_output = [line.replace("variable temp_K equal temp", "variable temp_K equal c_TDEF")
            .replace("variable press_MPa equal press*v_atm2mpa", "variable press_MPa equal (c_Ptot[1]+c_Ptot[2]+c_Ptot[3])/3.0*v_atm2mpa")
            for line in stress_output]
        cleanup.remove("uncompute TDEF")
        cleanup.append("uncompute TDEF")
    # Pressure needs its temperature compute to exist when it is defined.
    initialization = [*common, *setup, *stress_output] if prod.validated_mechanics else [*stress_output, *common, *setup]
    delayed_deform = []
    if prod.validated_mechanics and mode == "tension_nvt_z_fixed_lateral":
        # Apr-2025 Verlet caches force virial before END_OF_STEP deform.
        # The paired Stage-B MD logs validate logging before box remapping:
        # measured strain, pressure volume, and cached forces then align.
        # Limit this opt-in repair to the tested ordinary-NVT fixed-area mode;
        # legacy/default inputs and untested SLLOD/NPT paths remain unchanged.
        delayed_deform = [line for line in initialization if line.startswith("fix DEF ")]
        initialization = [line for line in initialization if not line.startswith("fix DEF ")]
        initialization = [line.replace("variable press_MPa equal press*v_atm2mpa",
            "variable press_MPa equal c_Ptot*v_atm2mpa") for line in initialization]

    return [
        "",
        f'print "DMAx2 mode={mode} strain={strain:g}"',
        "reset_timestep 0",
        *initialization,
        _print_line(
            mode,
            strain,
            prod.output_every,
            raw_dir,
            temperature_k=temperature_k,
            frequency_ghz=prod.frequency_ghz if include_frequency_in_raw_name else None,
            replicate_index=replicate_index,
            velocity_seed=velocity_seed,
        ),
        *delayed_deform,
        f"run {run_steps}",
        *cleanup,
    ]


def write_dma_input(
    polymer: PolymerSpec,
    ff: ForceFieldSpec,
    prod: DMAProductionSpec,
    workdir: str | Path,
    filename: str | None = None,
    equilibrated_data: str | None = None,
) -> Path:
    """Write a multi-mode DMA production input."""

    workdir = Path(workdir)
    path = workdir / (filename or f"in.{polymer.name}_dmax2_dma")
    data = equilibrated_data or f"{polymer.name}.dmax2.equilibrated.data"
    modes = list(dict.fromkeys([*prod.modes, *prod.legacy_modes]))
    lines = []
    for mode in modes:
        for strain in prod.strains:
            lines.extend(_dma_read_header(ff, data, prod.timestep_fs))
            lines.extend(_mode_block(mode, strain, prod, "raw", ff.kspace_style))
    lines.append(f"write_data {polymer.name}.dmax2.final.data")
    path.write_text("\n".join(lines) + "\n")
    return path


def write_strain_convergence_inputs(
    polymer: PolymerSpec,
    ff: ForceFieldSpec,
    prod: DMAProductionSpec,
    workdir: str | Path,
    equilibrated_data: str | Path,
) -> list[Path]:
    """Write one DMA production input per strain amplitude.

    The tree is designed for embarrassingly parallel execution now and for a
    future atomate2 mapping layer later: each strain directory is self-contained
    except for a shared read-only equilibrated data file.
    """

    workdir = Path(workdir)
    strains_dir = workdir / "strains"
    data_dir = workdir / "equilibrated"
    strains_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    data_path = Path(equilibrated_data)
    shared_data = data_dir / data_path.name
    if not shared_data.exists() and data_path.exists():
        shared_data.write_bytes(data_path.read_bytes())

    paths: list[Path] = []
    for strain in prod.strains:
        token = _strain_token(strain)
        run_dir = strains_dir / f"strain_{token}"
        run_dir.mkdir(parents=True, exist_ok=True)
        single = replace(prod, strains=[strain])
        input_path = write_dma_input(
            polymer,
            ff,
            single,
            run_dir,
            filename=f"in.{polymer.name}_dmax2_strain_{token}",
            equilibrated_data=f"../../equilibrated/{shared_data.name}",
        )
        (run_dir / "run.sh").write_text(
            "#!/usr/bin/env bash\n"
            "set -euo pipefail\n\n"
            'LMP_BIN="${LMP_BIN:-lmp}"\n'
            f'"${{LMP_BIN}}" -in "{input_path.name}" -log "{polymer.name}.strain_{token}.driver.log"\n'
        )
        (run_dir / "run.sh").chmod(0o775)
        paths.append(input_path)

    strain_tokens = [_strain_token(s) for s in prod.strains]
    (workdir / "run_all_local.sh").write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n\n"
        "for d in strains/strain_*; do\n"
        '  (cd "$d" && ./run.sh)\n'
        "done\n"
    )
    (workdir / "run_all_local.sh").chmod(0o775)

    (workdir / "slurm_array.sh").write_text(
        "#!/usr/bin/env bash\n"
        f"#SBATCH -J {polymer.name}_strain\n"
        "#SBATCH -A m5056\n"
        "#SBATCH -C cpu\n"
        "#SBATCH -q regular\n"
        "#SBATCH -t 08:00:00\n"
        "#SBATCH -N 1\n"
        "#SBATCH --ntasks-per-node=128\n"
        f"#SBATCH --array=0-{len(strain_tokens) - 1}\n\n"
        "set -euo pipefail\n"
        'STRAINS=("' + '" "'.join(strain_tokens) + '")\n'
        'TOKEN="${STRAINS[$SLURM_ARRAY_TASK_ID]}"\n'
        'cd "${SLURM_SUBMIT_DIR}/strains/strain_${TOKEN}"\n'
        'LMP_BIN="${LMP_BIN:-lmp}"\n'
        'srun -n "${SLURM_NTASKS}" "${LMP_BIN}" -in '
        f'"in.{polymer.name}_dmax2_strain_${{TOKEN}}" -log "{polymer.name}.strain_${{TOKEN}}.driver.log"\n'
    )
    (workdir / "slurm_array.sh").chmod(0o775)

    (workdir / "collect_and_parse.sh").write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n\n"
        "mkdir -p raw\n"
        "for f in strains/strain_*/raw/raw__*__strain_*.dat; do\n"
        '  [ -e "$f" ] || continue\n'
        '  cp "$f" raw/\n'
        "done\n"
        'CONFIG_ARG=""\n'
        'if [ -f strain_convergence_config.yaml ]; then\n'
        '  CONFIG_ARG="--config strain_convergence_config.yaml"\n'
        "fi\n"
        f"dmax2 parse . --freq-ghz {prod.frequency_ghz:g} --discard-cycles {prod.discard_cycles} ${{CONFIG_ARG}}\n"
        f"dmax2 plot-nemd-directory . --freq-ghz {prod.frequency_ghz:g} --discard-cycles {prod.discard_cycles} --recursive --local --skip-root-raw --skip-existing\n"
        "dmax2 analyze-master-curve . ${CONFIG_ARG}\n"
    )
    (workdir / "collect_and_parse.sh").chmod(0o775)

    return paths


def _steps_from_ps(duration_ps: float, timestep_fs: float) -> int:
    return int(round(duration_ps * 1000.0 / timestep_fs))


def _tension_interstage_equilibration_block(prod: DMAProductionSpec, protocol: TensionProtocolSpec) -> list[str]:
    temp = prod.temperature_k
    tdamp = prod.tdamp_fs
    pdamp = prod.pdamp_fs
    pressure = protocol.pressure_atm
    npt_steps = _steps_from_ps(protocol.npt_equilibration_ps, prod.timestep_fs)
    nvt_steps = _steps_from_ps(protocol.nvt_hold_ps, prod.timestep_fs)
    return [
        "",
        'print "DMAx2 tension protocol: interstage NPT density/stress relaxation"',
        f"fix TENSION_RELAX_NPT all npt temp {temp:g} {temp:g} {tdamp:g} aniso {pressure:g} {pressure:g} {pdamp:g} drag 1.0",
        f"run {npt_steps}",
        "unfix TENSION_RELAX_NPT",
        "write_data tension_after_interstage_npt.data",
        "",
        'print "DMAx2 tension protocol: fixed-cell NVT hold before NPT-NEMD"',
        f"fix TENSION_HOLD_NVT all nvt temp {temp:g} {temp:g} {tdamp:g}",
        f"run {nvt_steps}",
        "unfix TENSION_HOLD_NVT",
        "write_data tension_after_interstage_nvt.data",
    ]


def write_tension_strain_convergence_inputs(
    polymer: PolymerSpec,
    ff: ForceFieldSpec,
    prod: DMAProductionSpec,
    protocol: TensionProtocolSpec,
    workdir: str | Path,
    equilibrated_data: str | Path,
) -> list[Path]:
    """Write paired NVT and NPT axial-tension strain-convergence runs.

    Each strain directory contains one LAMMPS input that performs, in order:
    fixed-cell NVT NEMD, 100 ps-scale NPT relaxation, 100 ps-scale NVT hold,
    and lateral-NPT NEMD. This intentionally keeps both tension thermostat/
    barostat variants in the same run so they start from the same pre-relaxed
    structure and can be compared strain by strain.
    """

    workdir = Path(workdir)
    strains_dir = workdir / "strains"
    data_dir = workdir / "equilibrated"
    strains_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    data_path = Path(equilibrated_data)
    shared_data = data_dir / data_path.name
    if not shared_data.exists() and data_path.exists():
        shared_data.write_bytes(data_path.read_bytes())

    modes = [protocol.nvt_mode, protocol.npt_mode]
    if protocol.include_legacy_modes:
        modes.extend(prod.legacy_modes)
    modes = list(dict.fromkeys(modes))
    strain_tokens = [_strain_token(s) for s in prod.strains]
    paths: list[Path] = []
    manifest_rows = [
        "strain,run_dir,input,log,nvt_mode,npt_mode,frequency_ghz,temperature_K,total_cycles,"
        "discard_cycles,nvt_production_steps,npt_equilibration_steps,nvt_hold_steps,npt_production_steps"
    ]

    npt_equil_steps = _steps_from_ps(protocol.npt_equilibration_ps, prod.timestep_fs)
    nvt_hold_steps = _steps_from_ps(protocol.nvt_hold_ps, prod.timestep_fs)
    for strain, token in zip(prod.strains, strain_tokens, strict=True):
        run_dir = strains_dir / f"strain_{token}"
        run_dir.mkdir(parents=True, exist_ok=True)
        run_prod = replace(prod, strains=[strain], modes=modes, legacy_modes=[])
        input_name = f"in.{polymer.name}_dmax2_tension_strain_{token}"
        log_name = f"{polymer.name}.tension_strain_{token}.driver.log"
        lines = [
            "# DMAx2 paired axial-tension strain-convergence input.",
            "# Protocol: NVT NEMD -> NPT relaxation -> NVT hold -> lateral-NPT NEMD.",
            *_dma_read_header(ff, f"../../equilibrated/{shared_data.name}", prod.timestep_fs),
        ]
        lines.extend(_mode_block(protocol.nvt_mode, strain, run_prod, "raw", ff.kspace_style))
        lines.extend(_tension_interstage_equilibration_block(run_prod, protocol))
        lines.extend(_mode_block(protocol.npt_mode, strain, run_prod, "raw", ff.kspace_style))
        for mode in modes:
            if mode not in {protocol.nvt_mode, protocol.npt_mode}:
                lines.extend(_mode_block(mode, strain, run_prod, "raw", ff.kspace_style))
        lines.append(f"write_data {polymer.name}.tension_strain_{token}.dmax2.final.data")
        input_path = run_dir / input_name
        input_path.write_text("\n".join(lines) + "\n")

        (run_dir / "run.sh").write_text(
            "#!/usr/bin/env bash\n"
            "set -euo pipefail\n\n"
            'LMP_BIN="${LMP_BIN:-lmp}"\n'
            f'"${{LMP_BIN}}" -in "{input_name}" -log "{log_name}"\n'
        )
        (run_dir / "run.sh").chmod(0o775)

        (run_dir / "run.slurm").write_text(
            "#!/usr/bin/env bash\n"
            f"#SBATCH -J {polymer.name}_ten_{token}\n"
            f"#SBATCH -A {protocol.slurm_account}\n"
            f"#SBATCH -C {protocol.slurm_constraint}\n"
            f"#SBATCH -q {protocol.slurm_queue}\n"
            f"#SBATCH -t {protocol.slurm_time}\n\n"
            f"srun -N {protocol.slurm_nodes} --ntasks-per-node={protocol.slurm_ntasks_per_node} "
            f"-c {protocol.slurm_cpus_per_task} --cpu-bind=cores --module=mpich "
            f"shifter --image {protocol.shifter_image} lmp -in {input_name} -log {log_name}\n"
        )
        (run_dir / "run.slurm").chmod(0o775)

        manifest_rows.append(
            f"{strain:.8g},{run_dir.relative_to(workdir)},{input_path.relative_to(workdir)},{log_name},"
            f"{protocol.nvt_mode},{protocol.npt_mode},{prod.frequency_ghz:.8g},{prod.temperature_k:.8g},"
            f"{prod.cycles},{prod.discard_cycles},{prod.run_steps},{npt_equil_steps},{nvt_hold_steps},{prod.run_steps}"
        )
        paths.append(input_path)

    (workdir / "run_manifest.csv").write_text("\n".join(manifest_rows) + "\n")
    (workdir / "run_all_local.sh").write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n\n"
        "for d in strains/strain_*; do\n"
        '  (cd "$d" && ./run.sh)\n'
        "done\n"
    )
    (workdir / "run_all_local.sh").chmod(0o775)

    (workdir / "slurm_array.sh").write_text(
        "#!/usr/bin/env bash\n"
        f"#SBATCH -J {polymer.name}_tension_strain\n"
        f"#SBATCH -A {protocol.slurm_account}\n"
        f"#SBATCH -C {protocol.slurm_constraint}\n"
        f"#SBATCH -q {protocol.slurm_queue}\n"
        f"#SBATCH -t {protocol.slurm_time}\n"
        f"#SBATCH -N {protocol.slurm_nodes}\n"
        f"#SBATCH --ntasks-per-node={protocol.slurm_ntasks_per_node}\n"
        f"#SBATCH --array=0-{len(strain_tokens) - 1}\n\n"
        "set -euo pipefail\n"
        'STRAINS=("' + '" "'.join(strain_tokens) + '")\n'
        'TOKEN="${STRAINS[$SLURM_ARRAY_TASK_ID]}"\n'
        'cd "${SLURM_SUBMIT_DIR}/strains/strain_${TOKEN}"\n'
        f"srun -N {protocol.slurm_nodes} --ntasks-per-node={protocol.slurm_ntasks_per_node} "
        f"-c {protocol.slurm_cpus_per_task} --cpu-bind=cores --module=mpich "
        f"shifter --image {protocol.shifter_image} lmp -in "
        f'"in.{polymer.name}_dmax2_tension_strain_${{TOKEN}}" -log "{polymer.name}.tension_strain_${{TOKEN}}.driver.log"\n'
    )
    (workdir / "slurm_array.sh").chmod(0o775)

    (workdir / "collect_and_parse.sh").write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n\n"
        "mkdir -p raw\n"
        "for f in strains/strain_*/raw/raw__*__strain_*.dat; do\n"
        '  [ -e "$f" ] || continue\n'
        '  cp "$f" raw/\n'
        "done\n"
        'CONFIG_ARG=""\n'
        'if [ -f tension_strain_convergence_config.yaml ]; then\n'
        '  CONFIG_ARG="--config tension_strain_convergence_config.yaml"\n'
        "elif [ -f strain_convergence_config.yaml ]; then\n"
        '  CONFIG_ARG="--config strain_convergence_config.yaml"\n'
        "fi\n"
        f"dmax2 parse . --freq-ghz {prod.frequency_ghz:g} --discard-cycles {prod.discard_cycles} ${{CONFIG_ARG}}\n"
        f"dmax2 plot-nemd-directory . --freq-ghz {prod.frequency_ghz:g} --discard-cycles {prod.discard_cycles} --recursive --local --skip-root-raw --skip-existing\n"
    )
    (workdir / "collect_and_parse.sh").chmod(0o775)

    (workdir / "README.md").write_text(
        f"# DMAx2 Tension Strain-Size Convergence\n\n"
        "This directory tests whether axial tension/compression produces cleaner DMAx2 signals than the shear protocol.\n\n"
        "Each strain amplitude is one independent Slurm-array task. The strain grid is one run per integer percent from "
        f"`{min(prod.strains):g}` to `{max(prod.strains):g}`.\n\n"
        "For each strain, the generated input runs:\n\n"
        f"1. `{protocol.nvt_mode}` NEMD under fixed-cell NVT.\n"
        f"2. `{protocol.npt_equilibration_ps:g} ps` NPT relaxation at `{protocol.pressure_atm:g} atm`.\n"
        f"3. `{protocol.nvt_hold_ps:g} ps` fixed-cell NVT hold.\n"
        f"4. `{protocol.npt_mode}` NEMD with lateral NPT relaxation.\n\n"
        "The preferred parser signal in the config is `sdev_MPa`, while `szz_vir_MPa` and `press_MPa` are also parsed for comparison.\n\n"
        "Run with:\n\n"
        "```bash\n"
        "sbatch slurm_array.sh\n"
        "./collect_and_parse.sh\n"
        "```\n"
    )
    return paths


def _pre_tension_block(
    mode: str,
    strain: float,
    token: str,
    prod: DMAProductionSpec,
    protocol: PreTensionProtocolSpec,
    *,
    lateral_relax: bool,
) -> list[str]:
    temp = prod.temperature_k
    tdamp = prod.tdamp_fs
    pdamp = prod.pdamp_fs
    ramp_steps = max(1, _steps_from_ps(protocol.ramp_ps, prod.timestep_fs))
    hold_steps = max(0, _steps_from_ps(protocol.hold_ps, prod.timestep_fs))
    pressure = protocol.pressure_atm
    if lateral_relax:
        ramp_thermostat = (
            f"fix PRETENSION_TH all npt temp {temp:g} {temp:g} {tdamp:g} "
            f"x {pressure:g} {pressure:g} {pdamp:g} y {pressure:g} {pressure:g} {pdamp:g} couple xy"
        )
        hold_thermostat = ramp_thermostat.replace("PRETENSION_TH", "PRETENSION_HOLD_TH")
        lateral_note = "lateral NPT"
    else:
        ramp_thermostat = f"fix PRETENSION_TH all nvt temp {temp:g} {temp:g} {tdamp:g}"
        hold_thermostat = f"fix PRETENSION_HOLD_TH all nvt temp {temp:g} {temp:g} {tdamp:g}"
        lateral_note = "fixed lateral cell"

    lines = [
        "",
        f'print "DMAx2 pre-tension: mode={mode} dynamic_strain={strain:g} ({lateral_note})"',
        "reset_timestep 0",
        f"variable amp equal {strain:g}",
        f"variable static_bias_factor equal {protocol.static_stress_to_dynamic_stress_ratio:g}",
        "variable static_strain equal v_static_bias_factor*v_amp",
        "variable pretension_lz0 equal lz",
        f"variable pretension_ramp_steps equal {ramp_steps}",
        "variable pretension_erate equal v_static_strain/(v_pretension_ramp_steps*dt)",
        "variable pretension_step equal step",
        "variable pretension_t_ps equal step*dt/1000.0",
        "variable pretension_strain_box equal (lz-v_pretension_lz0)/v_pretension_lz0",
        "variable pretension_lz equal lz",
        "compute PRETENSION_Pvir all pressure NULL virial",
        f"variable atm2mpa equal {ATM_TO_MPA}",
        "variable pretension_szz_vir_MPa equal -c_PRETENSION_Pvir[3]*v_atm2mpa",
        (
            f'fix PRETENSION_TRACE all print {prod.output_every} '
            '"${pretension_step} ${pretension_t_ps} ${amp} ${static_bias_factor} '
            '${static_strain} ${pretension_strain_box} ${pretension_szz_vir_MPa} ${pretension_lz}" '
            f"file raw/pretension__{mode}__strain_{token}.dat screen no "
            'title "step t_ps dynamic_strain static_bias_factor static_strain '
            'strain_box szz_vir_MPa lz"'
        ),
        ramp_thermostat,
        "fix PRETENSION_DEFORM all deform 1 z erate ${pretension_erate} remap x units box",
        f"run {ramp_steps}",
        "unfix PRETENSION_DEFORM",
        "unfix PRETENSION_TH",
    ]
    if hold_steps:
        lines.extend(
            [
                hold_thermostat,
                f"run {hold_steps}",
                "unfix PRETENSION_HOLD_TH",
            ]
        )
    lines.extend(
        [
            "unfix PRETENSION_TRACE",
            "uncompute PRETENSION_Pvir",
            f"write_data {mode}.pretensioned_strain_{token}.data",
        ]
    )
    return lines


def write_pretensioned_tension_strain_convergence_inputs(
    polymer: PolymerSpec,
    ff: ForceFieldSpec,
    prod: DMAProductionSpec,
    protocol: PreTensionProtocolSpec,
    workdir: str | Path,
    equilibrated_data: str | Path,
) -> list[Path]:
    """Write pre-tensioned axial NEMD strain-convergence runs.

    The experiment maps the paper-style static-stress rule onto strain-
    controlled periodic-cell NEMD by applying a static tensile pre-strain equal
    to ``static_stress_to_dynamic_stress_ratio * dynamic_strain`` before the
    sinusoidal production deformation. In the 0.05-0.1% linear-response window,
    this keeps the total axial strain tensile throughout the cycle.
    """

    workdir = Path(workdir)
    strains_dir = workdir / "strains"
    data_dir = workdir / "equilibrated"
    strains_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    data_path = Path(equilibrated_data)
    shared_data = data_dir / data_path.name
    if not shared_data.exists() and data_path.exists():
        shared_data.write_bytes(data_path.read_bytes())

    modes: list[tuple[str, bool]] = [(protocol.nvt_mode, False)]
    if protocol.include_npt_lateral_mode:
        modes.append((protocol.npt_mode, True))

    strain_tokens = [_strain_token(s) for s in prod.strains]
    paths: list[Path] = []
    manifest_rows = [
        "strain,static_strain,run_dir,input,log,modes,frequency_ghz,temperature_K,total_cycles,"
        "discard_cycles,pretension_ramp_steps,pretension_hold_steps,production_steps,"
        "static_stress_to_dynamic_stress_ratio"
    ]
    ramp_steps = _steps_from_ps(protocol.ramp_ps, prod.timestep_fs)
    hold_steps = _steps_from_ps(protocol.hold_ps, prod.timestep_fs)

    for strain, token in zip(prod.strains, strain_tokens, strict=True):
        run_dir = strains_dir / f"strain_{token}"
        run_dir.mkdir(parents=True, exist_ok=True)
        run_prod = replace(
            prod,
            strains=[strain],
            modes=[mode for mode, _lateral_relax in modes],
            legacy_modes=[],
        )
        input_name = f"in.{polymer.name}_dmax2_pretension_strain_{token}"
        log_name = f"{polymer.name}.pretension_strain_{token}.driver.log"
        lines = [
            "# DMAx2 pre-tensioned axial strain-convergence input.",
            "# Protocol per mode: read equilibrated cell -> static tensile pre-strain -> hold -> dynamic NEMD.",
            "# Static pre-strain is set to static_stress_to_dynamic_stress_ratio * dynamic_strain.",
        ]
        for mode, lateral_relax in modes:
            lines.extend(_dma_read_header(ff, f"../../equilibrated/{shared_data.name}", prod.timestep_fs))
            lines.extend(_pre_tension_block(mode, strain, token, run_prod, protocol, lateral_relax=lateral_relax))
            lines.extend(_mode_block(mode, strain, run_prod, "raw", ff.kspace_style))
            lines.append(f"write_data {polymer.name}.{mode}.pretension_strain_{token}.dmax2.final.data")

        input_path = run_dir / input_name
        input_path.write_text("\n".join(lines) + "\n")

        (run_dir / "run.sh").write_text(
            "#!/usr/bin/env bash\n"
            "set -euo pipefail\n\n"
            'LMP_BIN="${LMP_BIN:-lmp}"\n'
            f'"${{LMP_BIN}}" -in "{input_name}" -log "{log_name}"\n'
        )
        (run_dir / "run.sh").chmod(0o775)

        (run_dir / "run.slurm").write_text(
            "#!/usr/bin/env bash\n"
            f"#SBATCH -J {polymer.name}_pt_{token}\n"
            f"#SBATCH -A {protocol.slurm_account}\n"
            f"#SBATCH -C {protocol.slurm_constraint}\n"
            f"#SBATCH -q {protocol.slurm_queue}\n"
            f"#SBATCH -t {protocol.slurm_time}\n\n"
            f"srun -N {protocol.slurm_nodes} --ntasks-per-node={protocol.slurm_ntasks_per_node} "
            f"-c {protocol.slurm_cpus_per_task} --cpu-bind=cores --module=mpich "
            f"shifter --image {protocol.shifter_image} lmp -in {input_name} -log {log_name}\n"
        )
        (run_dir / "run.slurm").chmod(0o775)

        mode_names = "|".join(mode for mode, _lateral_relax in modes)
        manifest_rows.append(
            f"{strain:.8g},{protocol.static_stress_to_dynamic_stress_ratio * strain:.8g},"
            f"{run_dir.relative_to(workdir)},{input_path.relative_to(workdir)},{log_name},{mode_names},"
            f"{prod.frequency_ghz:.8g},{prod.temperature_k:.8g},{prod.cycles},{prod.discard_cycles},"
            f"{ramp_steps},{hold_steps},{prod.run_steps},"
            f"{protocol.static_stress_to_dynamic_stress_ratio:.8g}"
        )
        paths.append(input_path)

    (workdir / "run_manifest.csv").write_text("\n".join(manifest_rows) + "\n")
    (workdir / "run_all_local.sh").write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n\n"
        "for d in strains/strain_*; do\n"
        '  (cd "$d" && ./run.sh)\n'
        "done\n"
    )
    (workdir / "run_all_local.sh").chmod(0o775)

    (workdir / "slurm_array.sh").write_text(
        "#!/usr/bin/env bash\n"
        f"#SBATCH -J {polymer.name}_pretension\n"
        f"#SBATCH -A {protocol.slurm_account}\n"
        f"#SBATCH -C {protocol.slurm_constraint}\n"
        f"#SBATCH -q {protocol.slurm_queue}\n"
        f"#SBATCH -t {protocol.slurm_time}\n"
        f"#SBATCH -N {protocol.slurm_nodes}\n"
        f"#SBATCH --ntasks-per-node={protocol.slurm_ntasks_per_node}\n"
        f"#SBATCH --array=0-{len(strain_tokens) - 1}\n\n"
        "set -euo pipefail\n"
        'STRAINS=("' + '" "'.join(strain_tokens) + '")\n'
        'TOKEN="${STRAINS[$SLURM_ARRAY_TASK_ID]}"\n'
        'cd "${SLURM_SUBMIT_DIR}/strains/strain_${TOKEN}"\n'
        f"srun -N {protocol.slurm_nodes} --ntasks-per-node={protocol.slurm_ntasks_per_node} "
        f"-c {protocol.slurm_cpus_per_task} --cpu-bind=cores --module=mpich "
        f"shifter --image {protocol.shifter_image} lmp -in "
        f'"in.{polymer.name}_dmax2_pretension_strain_${{TOKEN}}" -log "{polymer.name}.pretension_strain_${{TOKEN}}.driver.log"\n'
    )
    (workdir / "slurm_array.sh").chmod(0o775)

    (workdir / "collect_and_parse.sh").write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n\n"
        "mkdir -p raw\n"
        "for f in strains/strain_*/raw/raw__*__strain_*.dat; do\n"
        '  [ -e "$f" ] || continue\n'
        '  cp "$f" raw/\n'
        "done\n"
        'CONFIG_ARG=""\n'
        'if [ -f pretension_strain_convergence_config.yaml ]; then\n'
        '  CONFIG_ARG="--config pretension_strain_convergence_config.yaml"\n'
        "fi\n"
        f"dmax2 parse . --freq-ghz {prod.frequency_ghz:g} --discard-cycles {prod.discard_cycles} ${{CONFIG_ARG}}\n"
        f"dmax2 plot-nemd-directory . --freq-ghz {prod.frequency_ghz:g} --discard-cycles {prod.discard_cycles} --recursive --local --skip-root-raw --skip-existing\n"
    )
    (workdir / "collect_and_parse.sh").chmod(0o775)

    strain_min_pct = min(prod.strains) * 100.0
    strain_max_pct = max(prod.strains) * 100.0
    static_min_pct = min(prod.strains) * protocol.static_stress_to_dynamic_stress_ratio * 100.0
    static_max_pct = max(prod.strains) * protocol.static_stress_to_dynamic_stress_ratio * 100.0
    mode_list = ", ".join(mode for mode, _lateral_relax in modes)
    (workdir / "README.md").write_text(
        "# DMAx2 Pre-Tensioned Tension Strain-Size Convergence\n\n"
        "This directory tests axial NEMD in an already-tensioned PVDF unit cell.\n\n"
        "The dynamic strain grid follows the quoted experimental protocol: "
        f"`{strain_min_pct:.3g}%` to `{strain_max_pct:.3g}%`. Before each production branch, "
        "the cell is stretched in `z` by "
        f"`{protocol.static_stress_to_dynamic_stress_ratio:g} * dynamic_strain`, "
        f"giving a static pre-strain range of `{static_min_pct:.3g}%` to `{static_max_pct:.3g}%`.\n\n"
        "Because these are strain-controlled periodic-cell simulations, the 200% static-stress rule is implemented "
        "as the corresponding small-strain static-bias ratio. The pre-tension trace files record the actual virial "
        "axial stress reached during the ramp and hold.\n\n"
        f"Generated production modes: `{mode_list}`.\n\n"
        "Run with:\n\n"
        "```bash\n"
        "sbatch slurm_array.sh\n"
        "./collect_and_parse.sh\n"
        "```\n"
    )
    return paths


def _error_analysis_seeds(error: ErrorAnalysisSpec) -> list[int]:
    if error.seeds:
        return [int(seed) for seed in error.seeds]
    return [int(error.seed_start + 104729 * i) for i in range(int(error.replicates))]


def write_error_analysis_inputs(
    polymer: PolymerSpec,
    ff: ForceFieldSpec,
    prod: DMAProductionSpec,
    error: ErrorAnalysisSpec,
    workdir: str | Path,
    equilibrated_data: str | Path,
) -> list[Path]:
    """Write random-velocity replicate jobs for DMA uncertainty estimates.

    This stage follows the original DMAx paper's random-initial-condition
    uncertainty step, but keeps the DMAx2 default production signal and parser
    available by tagging raw traces with replicate and velocity-seed metadata.
    """

    workdir = Path(workdir)
    replicates_dir = workdir / "replicates"
    data_dir = workdir / "equilibrated"
    replicates_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    data_path = Path(equilibrated_data)
    shared_data = data_dir / data_path.name
    if not shared_data.exists() and data_path.exists():
        shared_data.write_bytes(data_path.read_bytes())

    strain = float(error.strain if error.strain is not None else prod.strains[0])
    cycles = int(error.cycles if error.cycles is not None else prod.cycles)
    discard_cycles = int(error.discard_cycles if error.discard_cycles is not None else prod.discard_cycles)
    frequency_ghz = float(error.frequency_ghz if error.frequency_ghz is not None else prod.frequency_ghz)
    temperature_k = float(error.temperature_k if error.temperature_k is not None else prod.temperature_k)
    modes = list(prod.modes) or ["shear_sllod_remapv"]
    if error.include_legacy_modes:
        modes = list(dict.fromkeys([*modes, *prod.legacy_modes]))

    run_prod = replace(
        prod,
        strains=[strain],
        cycles=cycles,
        discard_cycles=discard_cycles,
        frequency_ghz=frequency_ghz,
        temperature_k=temperature_k,
        modes=modes,
        legacy_modes=[],
    )
    strain_token = _strain_token(strain)
    seeds = _error_analysis_seeds(error)

    paths: list[Path] = []
    manifest_rows = [
        "replicate,velocity_seed,run_dir,input,log,strain,frequency_ghz,temperature_K,total_cycles,discard_cycles,usable_cycles,production_steps"
    ]
    replicate_dirs: list[str] = []
    for index, seed in enumerate(seeds, start=1):
        rep_token = f"{index:03d}"
        run_dir = replicates_dir / f"replicate_{rep_token}_seed_{seed}"
        replicate_dirs.append(run_dir.name)
        run_dir.mkdir(parents=True, exist_ok=True)
        input_name = f"in.{polymer.name}_dmax2_error_rep_{rep_token}"
        log_name = f"{polymer.name}.error_rep_{rep_token}.driver.log"
        lines: list[str] = [
            "# DMAx2 random-velocity error-analysis replicate.",
            "# Each production branch re-reads the same equilibrated structure and creates velocities from this replicate seed.",
        ]
        for mode in modes:
            lines.extend(_dma_read_header(ff, f"../../equilibrated/{shared_data.name}", prod.timestep_fs))
            lines.extend(
                [
                    "neighbor 2.0 multi",
                    "neigh_modify every 1 delay 0 check yes",
                    "comm_modify vel yes",
                    f"variable target_temp equal {temperature_k:.8g}",
                    f"variable velocity_seed equal {seed}",
                    "velocity all create ${target_temp} ${velocity_seed} mom yes rot yes dist gaussian",
                ]
            )
            lines.extend(
                _mode_block(
                    mode,
                    strain,
                    run_prod,
                    "raw",
                    ff.kspace_style,
                    replicate_index=index,
                    velocity_seed=seed,
                )
            )
        lines.append(f"write_data {polymer.name}.error_rep_{rep_token}.dmax2.final.data")
        input_path = run_dir / input_name
        input_path.write_text("\n".join(lines) + "\n")

        (run_dir / "run.sh").write_text(
            "#!/usr/bin/env bash\n"
            "set -euo pipefail\n\n"
            'LMP_BIN="${LMP_BIN:-lmp}"\n'
            f'"${{LMP_BIN}}" -in "{input_name}" -log "{log_name}"\n'
        )
        (run_dir / "run.sh").chmod(0o775)

        (run_dir / "run.slurm").write_text(
            "#!/usr/bin/env bash\n"
            f"#SBATCH -J {polymer.name}_err_{rep_token}\n"
            f"#SBATCH -A {error.slurm_account}\n"
            f"#SBATCH -C {error.slurm_constraint}\n"
            f"#SBATCH -q {error.slurm_queue}\n"
            f"#SBATCH -t {error.slurm_time}\n\n"
            f"srun -N {error.slurm_nodes} --ntasks-per-node={error.slurm_ntasks_per_node} "
            f"-c {error.slurm_cpus_per_task} --cpu-bind=cores --module=mpich "
            f"shifter --image {error.shifter_image} lmp -in {input_name} -log {log_name}\n"
        )
        (run_dir / "run.slurm").chmod(0o775)
        manifest_rows.append(
            f"{index},{seed},{run_dir.relative_to(workdir)},{input_path.relative_to(workdir)},{log_name},"
            f"{strain:.8g},{frequency_ghz:.8g},{temperature_k:.8g},{cycles},{discard_cycles},"
            f"{max(cycles - discard_cycles, 0)},{run_prod.run_steps}"
        )
        paths.append(input_path)

    (workdir / "run_manifest.csv").write_text("\n".join(manifest_rows) + "\n")
    (workdir / "run_all_local.sh").write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n\n"
        "for d in replicates/replicate_*; do\n"
        '  (cd "$d" && ./run.sh)\n'
        "done\n"
    )
    (workdir / "run_all_local.sh").chmod(0o775)

    (workdir / "slurm_array.sh").write_text(
        "#!/usr/bin/env bash\n"
        f"#SBATCH -J {polymer.name}_err\n"
        f"#SBATCH -A {error.slurm_account}\n"
        f"#SBATCH -C {error.slurm_constraint}\n"
        f"#SBATCH -q {error.slurm_queue}\n"
        f"#SBATCH -t {error.slurm_time}\n"
        f"#SBATCH -N {error.slurm_nodes}\n"
        f"#SBATCH --ntasks-per-node={error.slurm_ntasks_per_node}\n"
        f"#SBATCH --array=0-{len(replicate_dirs) - 1}\n\n"
        "set -euo pipefail\n"
        'REPLICATES=("' + '" "'.join(replicate_dirs) + '")\n'
        'RUN_DIR="${REPLICATES[$SLURM_ARRAY_TASK_ID]}"\n'
        'cd "${SLURM_SUBMIT_DIR}/replicates/${RUN_DIR}"\n'
        'INPUT="$(ls in.*_dmax2_error_rep_*)"\n'
        f"srun -N {error.slurm_nodes} --ntasks-per-node={error.slurm_ntasks_per_node} "
        f"-c {error.slurm_cpus_per_task} --cpu-bind=cores --module=mpich "
        f"shifter --image {error.shifter_image} lmp -in \"${{INPUT}}\" -log \"${{RUN_DIR}}.driver.log\"\n"
    )
    (workdir / "slurm_array.sh").chmod(0o775)

    (workdir / "collect_and_parse.sh").write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n\n"
        "mkdir -p raw\n"
        "for f in replicates/replicate_*/raw/raw__*__strain_*__rep_*__seed_*.dat; do\n"
        '  [ -e "$f" ] || continue\n'
        '  cp "$f" raw/\n'
        "done\n"
        'CONFIG_ARG=""\n'
        'if [ -f error_analysis_config.yaml ]; then\n'
        '  CONFIG_ARG="--config error_analysis_config.yaml"\n'
        "fi\n"
        f"dmax2 parse . --freq-ghz {frequency_ghz:g} --discard-cycles {discard_cycles} ${{CONFIG_ARG}}\n"
        f"dmax2 plot-nemd-directory . --freq-ghz {frequency_ghz:g} --discard-cycles {discard_cycles} --recursive --local --skip-root-raw --skip-existing\n"
        "dmax2 analyze-error . ${CONFIG_ARG}\n"
    )
    (workdir / "collect_and_parse.sh").chmod(0o775)

    (workdir / "README.md").write_text(
        f"# DMAx2 error analysis for {polymer.name}\n\n"
        "This step estimates random-initial-condition uncertainty after strain-size and cycle-count convergence.\n\n"
        f"- Strain amplitude: `{strain:g}`\n"
        f"- Frequency: `{frequency_ghz:g} GHz`\n"
        f"- Total production cycles: `{cycles}`\n"
        f"- Discarded startup cycles: `{discard_cycles}`\n"
        f"- Usable fitted cycles: `{max(cycles - discard_cycles, 0)}`\n"
        f"- Replicates: `{len(seeds)}`\n"
        f"- Velocity seeds: `{', '.join(str(seed) for seed in seeds)}`\n\n"
        "Run with:\n\n"
        "```bash\n"
        "sbatch slurm_array.sh\n"
        "./collect_and_parse.sh\n"
        "```\n"
    )

    return paths


def write_glass_transition_inputs(
    polymer: PolymerSpec,
    ff: ForceFieldSpec,
    prod: DMAProductionSpec,
    glass: GlassTransitionSpec,
    workdir: str | Path,
    equilibrated_data: str | Path,
) -> list[Path]:
    """Write one temperature-sweep DMA input per target temperature.

    Each temperature directory first relaxes a copy of the equilibrated structure
    at the target temperature, then re-reads that temperature-equilibrated data
    for each DMA production mode so branches remain independent.
    """

    workdir = Path(workdir)
    temperatures_dir = workdir / "temperatures"
    data_dir = workdir / "equilibrated"
    temperatures_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    data_path = Path(equilibrated_data)
    shared_data = data_dir / data_path.name
    if not shared_data.exists() and data_path.exists():
        shared_data.write_bytes(data_path.read_bytes())

    if glass.temperatures_k:
        temperatures = [float(t) for t in glass.temperatures_k]
        prior = None
        match_reason = "explicit_temperatures"
    else:
        grid, prior, match_reason = glass_temperature_grid(polymer, glass)
        temperatures = [float(t) for t in grid]

    modes = list(prod.modes)
    if glass.include_legacy_modes:
        modes = list(dict.fromkeys([*modes, *prod.legacy_modes]))
    if not modes:
        modes = ["shear_sllod_remapv"]

    paths: list[Path] = []
    temp_tokens: list[str] = []
    for temperature in temperatures:
        token = _temperature_token(temperature)
        temp_tokens.append(token)
        run_dir = temperatures_dir / f"T_{token}K"
        run_dir.mkdir(parents=True, exist_ok=True)
        temp_data = f"{polymer.name}.T_{token}K.equilibrated.data"
        temp_prod = replace(prod, temperature_k=temperature)
        lines: list[str] = [
            "# DMAx2 glass-transition temperature point.",
            "# Strategy: target-temperature relaxation followed by selected DMA production.",
            *_dma_read_header(ff, f"../../equilibrated/{shared_data.name}", prod.timestep_fs),
            f"variable target_temp equal {temperature:.8g}",
            f"variable ptarget equal {glass.pressure_atm:g}",
            f"variable tdamp equal {prod.tdamp_fs:g}",
            f"variable pdamp equal {prod.pdamp_fs:g}",
            f"variable prep_steps equal {glass.temperature_equilibration_steps}",
            "variable seed equal 928371",
            "shell mkdir -p equil",
            "",
            'print "DMAx2 glass transition: target-temperature relaxation"',
            "velocity all create ${target_temp} ${seed} mom yes rot yes dist gaussian",
            "fix GT_EQ all npt temp ${target_temp} ${target_temp} ${tdamp} aniso ${ptarget} ${ptarget} ${pdamp} drag 1.0",
            "run ${prep_steps}",
            "unfix GT_EQ",
            f"write_data equil/{temp_data}",
        ]
        for mode in modes:
            for strain in temp_prod.strains:
                lines.extend(_dma_read_header(ff, f"equil/{temp_data}", prod.timestep_fs))
                lines.extend(_mode_block(mode, strain, temp_prod, "raw", ff.kspace_style, temperature_k=temperature))
        lines.append(f"write_data {polymer.name}.T_{token}K.dmax2.final.data")
        input_path = run_dir / f"in.{polymer.name}_dmax2_gt_T_{token}K"
        input_path.write_text("\n".join(lines) + "\n")
        (run_dir / "run.sh").write_text(
            "#!/usr/bin/env bash\n"
            "set -euo pipefail\n\n"
            'LMP_BIN="${LMP_BIN:-lmp}"\n'
            f'"${{LMP_BIN}}" -in "{input_path.name}" -log "{polymer.name}.gt_T_{token}K.driver.log"\n'
        )
        (run_dir / "run.sh").chmod(0o775)
        paths.append(input_path)

    (workdir / "temperature_grid.csv").write_text(
        "temperature_K\n" + "\n".join(f"{t:.8g}" for t in temperatures) + "\n"
    )
    if prior is not None:
        (workdir / "tg_prior.json").write_text(
            "{\n"
            f'  "matched_by": "{match_reason}",\n'
            f'  "key": "{prior.key}",\n'
            f'  "name": "{prior.name}",\n'
            f'  "tg_k": {prior.tg_k:.8g},\n'
            f'  "source": "{prior.source}",\n'
            f'  "notes": "{prior.notes}"\n'
            "}\n"
        )

    (workdir / "run_all_local.sh").write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n\n"
        "for d in temperatures/T_*K; do\n"
        '  (cd "$d" && ./run.sh)\n'
        "done\n"
    )
    (workdir / "run_all_local.sh").chmod(0o775)

    (workdir / "slurm_array.sh").write_text(
        "#!/usr/bin/env bash\n"
        f"#SBATCH -J {polymer.name}_gt\n"
        "#SBATCH -A m5056\n"
        "#SBATCH -C cpu\n"
        "#SBATCH -q regular\n"
        "#SBATCH -t 08:00:00\n"
        "#SBATCH -N 1\n"
        "#SBATCH --ntasks-per-node=128\n"
        f"#SBATCH --array=0-{len(temp_tokens) - 1}\n\n"
        "set -euo pipefail\n"
        'TEMPS=("' + '" "'.join(temp_tokens) + '")\n'
        'TOKEN="${TEMPS[$SLURM_ARRAY_TASK_ID]}"\n'
        'cd "${SLURM_SUBMIT_DIR}/temperatures/T_${TOKEN}K"\n'
        'LMP_BIN="${LMP_BIN:-lmp}"\n'
        'srun -n "${SLURM_NTASKS}" "${LMP_BIN}" -in '
        f'"in.{polymer.name}_dmax2_gt_T_${{TOKEN}}K" -log "{polymer.name}.gt_T_${{TOKEN}}K.driver.log"\n'
    )
    (workdir / "slurm_array.sh").chmod(0o775)

    (workdir / "collect_and_parse.sh").write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n\n"
        "mkdir -p raw\n"
        "for f in temperatures/T_*K/raw/raw__*__strain_*__T_*K.dat; do\n"
        '  [ -e "$f" ] || continue\n'
        '  cp "$f" raw/\n'
        "done\n"
        'CONFIG_ARG=""\n'
        'if [ -f glass_transition_config.yaml ]; then\n'
        '  CONFIG_ARG="--config glass_transition_config.yaml"\n'
        "fi\n"
        f"dmax2 parse . --freq-ghz {prod.frequency_ghz:g} --discard-cycles {prod.discard_cycles} ${{CONFIG_ARG}}\n"
        f"dmax2 plot-nemd-directory . --freq-ghz {prod.frequency_ghz:g} --discard-cycles {prod.discard_cycles} --recursive --local --skip-root-raw --skip-existing\n"
        "dmax2 analyze-glass-transition . ${CONFIG_ARG}\n"
    )
    (workdir / "collect_and_parse.sh").chmod(0o775)

    return paths


def write_density_glass_transition_inputs(
    polymer: PolymerSpec,
    ff: ForceFieldSpec,
    prod: DMAProductionSpec,
    glass: GlassTransitionSpec,
    workdir: str | Path,
    equilibrated_data: str | Path,
) -> list[Path]:
    """Write independent NPT density-equilibration jobs for Tg comparison."""

    workdir = Path(workdir)
    temperatures_dir = workdir / "temperatures"
    data_dir = workdir / "equilibrated"
    temperatures_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    data_path = Path(equilibrated_data)
    shared_data = data_dir / data_path.name
    if not shared_data.exists() and data_path.exists():
        shared_data.write_bytes(data_path.read_bytes())

    if glass.temperatures_k:
        temperatures = [float(t) for t in glass.temperatures_k]
        prior = None
        match_reason = "explicit_temperatures"
    else:
        grid, prior, match_reason = glass_temperature_grid(polymer, glass)
        temperatures = [float(t) for t in grid]

    paths: list[Path] = []
    temp_tokens: list[str] = []
    for index, temperature in enumerate(temperatures):
        token = _temperature_token(temperature)
        temp_tokens.append(token)
        run_dir = temperatures_dir / f"T_{token}K"
        run_dir.mkdir(parents=True, exist_ok=True)
        seed = 341771 + index
        lines = [
            "# DMAx2 density glass-transition comparison point.",
            "# Strategy: independent target-temperature NPT equilibration and density sampling.",
            "clear",
            "units real",
            "atom_style full",
            "boundary p p p",
            "dielectric 1.0",
            f"pair_style {ff.pair_style}",
            f"bond_style {ff.bond_style}",
            f"angle_style {ff.angle_style}",
            f"dihedral_style {ff.dihedral_style}",
            f"improper_style {ff.improper_style}",
            f"kspace_style {ff.kspace_style}",
            f"pair_modify {ff.pair_modify}",
            f"special_bonds {ff.special_bonds}",
            f"read_data ../../equilibrated/{shared_data.name}",
            "neighbor 2.0 multi",
            "neigh_modify every 1 delay 0 check yes",
            "comm_modify vel yes",
            f"timestep {prod.timestep_fs:g}",
            f"variable target_temp equal {temperature:.8g}",
            f"variable ptarget equal {glass.pressure_atm:g}",
            f"variable tdamp equal {prod.tdamp_fs:g}",
            f"variable pdamp equal {glass.density_pdamp_fs:g}",
            f"variable eq_steps equal {glass.density_equilibration_steps}",
            f"variable sample_steps equal {glass.density_sampling_steps}",
            f"variable trace_every equal {glass.density_output_every}",
            f"variable seed equal {seed}",
            "variable temp_inst equal temp",
            "variable press_inst equal press",
            "variable density_inst equal density",
            "variable volume_inst equal vol",
            "variable lx_inst equal lx",
            "variable ly_inst equal ly",
            "variable lz_inst equal lz",
            "thermo ${trace_every}",
            "thermo_style custom step temp press density vol lx ly lz pe etotal",
            "thermo_modify flush yes",
            "shell mkdir -p traces data",
            "",
            'print "DMAx2 density Tg: NPT equilibration"',
            "velocity all create ${target_temp} ${seed} mom yes rot yes dist gaussian",
            "fix DENS_NPT all npt temp ${target_temp} ${target_temp} ${tdamp} aniso ${ptarget} ${ptarget} ${pdamp} drag 1.0",
            "run ${eq_steps}",
            "",
            'print "DMAx2 density Tg: density sampling"',
            "reset_timestep 0",
            (
                "fix DENS_TRACE all ave/time ${trace_every} 1 ${trace_every} "
                "v_temp_inst v_press_inst v_density_inst v_volume_inst "
                "v_lx_inst v_ly_inst v_lz_inst file density_trace.dat"
            ),
            "run ${sample_steps}",
            "unfix DENS_TRACE",
            "unfix DENS_NPT",
            f"write_data data/{polymer.name}.density_T_{token}K.final.data",
        ]
        input_path = run_dir / f"in.{polymer.name}_dmax2_density_T_{token}K"
        input_path.write_text("\n".join(lines) + "\n")
        (run_dir / "run.sh").write_text(
            "#!/usr/bin/env bash\n"
            "set -euo pipefail\n\n"
            'LMP_BIN="${LMP_BIN:-lmp}"\n'
            f'"${{LMP_BIN}}" -in "{input_path.name}" -log "{polymer.name}.density_T_{token}K.driver.log"\n'
        )
        (run_dir / "run.sh").chmod(0o775)
        paths.append(input_path)

    (workdir / "temperature_grid.csv").write_text(
        "temperature_K\n" + "\n".join(f"{t:.8g}" for t in temperatures) + "\n"
    )
    if prior is not None:
        (workdir / "tg_prior.json").write_text(
            "{\n"
            f'  "matched_by": "{match_reason}",\n'
            f'  "key": "{prior.key}",\n'
            f'  "name": "{prior.name}",\n'
            f'  "tg_k": {prior.tg_k:.8g},\n'
            f'  "source": "{prior.source}",\n'
            f'  "notes": "{prior.notes}"\n'
            "}\n"
        )

    (workdir / "run_all_local.sh").write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n\n"
        "for d in temperatures/T_*K; do\n"
        '  (cd "$d" && ./run.sh)\n'
        "done\n"
    )
    (workdir / "run_all_local.sh").chmod(0o775)

    (workdir / "slurm_array.sh").write_text(
        "#!/usr/bin/env bash\n"
        f"#SBATCH -J {polymer.name}_density_tg\n"
        "#SBATCH -A m5056\n"
        "#SBATCH -C cpu\n"
        "#SBATCH -q regular\n"
        "#SBATCH -t 08:00:00\n"
        "#SBATCH -N 1\n"
        "#SBATCH --ntasks-per-node=128\n"
        f"#SBATCH --array=0-{len(temp_tokens) - 1}\n\n"
        "set -euo pipefail\n"
        'TEMPS=("' + '" "'.join(temp_tokens) + '")\n'
        'TOKEN="${TEMPS[$SLURM_ARRAY_TASK_ID]}"\n'
        'cd "${SLURM_SUBMIT_DIR}/temperatures/T_${TOKEN}K"\n'
        'LMP_BIN="${LMP_BIN:-lmp}"\n'
        'srun -n "${SLURM_NTASKS}" "${LMP_BIN}" -in '
        f'"in.{polymer.name}_dmax2_density_T_${{TOKEN}}K" -log "{polymer.name}.density_T_${{TOKEN}}K.driver.log"\n'
    )
    (workdir / "slurm_array.sh").chmod(0o775)

    (workdir / "collect_and_parse.sh").write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n\n"
        "CONFIG_ARG=\"\"\n"
        "if [ -f density_glass_transition_config.yaml ]; then\n"
        "  CONFIG_ARG=\"--config density_glass_transition_config.yaml\"\n"
        "fi\n"
        "dmax2 analyze-density-glass-transition . ${CONFIG_ARG}\n"
    )
    (workdir / "collect_and_parse.sh").chmod(0o775)
    return paths


def write_density_equilibrated_glass_transition_inputs(
    polymer: PolymerSpec,
    ff: ForceFieldSpec,
    prod: DMAProductionSpec,
    glass: GlassTransitionSpec,
    workdir: str | Path,
    equilibrated_data: str | Path,
) -> list[Path]:
    """Write DMA Tg jobs that first equilibrate density at each temperature.

    Each job mirrors the density-Tg comparison protocol, then switches to a
    fixed-cell NVT hold before the DMA production branch. This tests whether
    DMA tan(delta) becomes more physically resolved when the starting
    structure/cell has been equilibrated at the same temperature as the NEMD
    measurement instead of inherited from a reference-temperature structure.
    """

    workdir = Path(workdir)
    temperatures_dir = workdir / "temperatures"
    data_dir = workdir / "equilibrated"
    temperatures_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    data_path = Path(equilibrated_data)
    shared_data = data_dir / data_path.name
    if not shared_data.exists() and data_path.exists():
        shared_data.write_bytes(data_path.read_bytes())

    if glass.temperatures_k:
        temperatures = [float(t) for t in glass.temperatures_k]
        prior = None
        match_reason = "explicit_temperatures"
    else:
        grid, prior, match_reason = glass_temperature_grid(polymer, glass)
        temperatures = [float(t) for t in grid]

    modes = list(prod.modes)
    if glass.include_legacy_modes:
        modes = list(dict.fromkeys([*modes, *prod.legacy_modes]))
    if not modes:
        modes = ["shear_sllod_remapv"]

    paths: list[Path] = []
    temp_tokens: list[str] = []
    manifest_rows = [
        "temperature_K,run_dir,input,log,density_equilibration_steps,density_sampling_steps,nvt_steps,production_steps,total_steps"
    ]
    for index, temperature in enumerate(temperatures):
        token = _temperature_token(temperature)
        temp_tokens.append(token)
        run_dir = temperatures_dir / f"T_{token}K"
        run_dir.mkdir(parents=True, exist_ok=True)
        seed = 451771 + index
        temp_prod = replace(prod, temperature_k=temperature)
        density_data = f"{polymer.name}.density_T_{token}K.after_npt.data"
        nvt_data = f"{polymer.name}.density_T_{token}K.after_nvt.data"
        lines: list[str] = [
            "# DMAx2 density-equilibrated glass-transition temperature point.",
            "# Strategy: target-temperature NPT density equilibration/sampling, short NVT hold, then DMA NEMD production.",
            "clear",
            "units real",
            "atom_style full",
            "boundary p p p",
            "dielectric 1.0",
            f"pair_style {ff.pair_style}",
            f"bond_style {ff.bond_style}",
            f"angle_style {ff.angle_style}",
            f"dihedral_style {ff.dihedral_style}",
            f"improper_style {ff.improper_style}",
            f"kspace_style {ff.kspace_style}",
            f"pair_modify {ff.pair_modify}",
            f"special_bonds {ff.special_bonds}",
            f"read_data ../../equilibrated/{shared_data.name}",
            "neighbor 2.0 multi",
            "neigh_modify every 1 delay 0 check yes",
            "comm_modify vel yes",
            f"timestep {prod.timestep_fs:g}",
            f"variable target_temp equal {temperature:.8g}",
            f"variable ptarget equal {glass.pressure_atm:g}",
            f"variable tdamp equal {prod.tdamp_fs:g}",
            f"variable density_pdamp equal {glass.density_pdamp_fs:g}",
            f"variable density_eq_steps equal {glass.density_equilibration_steps}",
            f"variable nvt_steps equal {glass.density_dma_nvt_steps}",
            f"variable trace_every equal {glass.density_output_every}",
            f"variable seed equal {seed}",
            "variable temp_inst equal temp",
            "variable press_inst equal press",
            "variable density_inst equal density",
            "variable volume_inst equal vol",
            "variable lx_inst equal lx",
            "variable ly_inst equal ly",
            "variable lz_inst equal lz",
            "thermo ${trace_every}",
            "thermo_style custom step temp press density vol lx ly lz pe etotal",
            "thermo_modify flush yes",
            "shell mkdir -p raw traces equil data",
            "",
            'print "DMAx2 density-equilibrated Tg: NPT density equilibration"',
            "velocity all create ${target_temp} ${seed} mom yes rot yes dist gaussian",
            "fix DENS_NPT all npt temp ${target_temp} ${target_temp} ${tdamp} aniso ${ptarget} ${ptarget} ${density_pdamp} drag 1.0",
            "run ${density_eq_steps}",
            "",
            'print "DMAx2 density-equilibrated Tg: write post-NPT density-equilibrated structure"',
            f"write_data equil/{density_data}",
            "unfix DENS_NPT",
            "",
            'print "DMAx2 density-equilibrated Tg: short fixed-cell NVT hold"',
            "reset_timestep 0",
            (
                "fix NVT_TRACE all ave/time ${trace_every} 1 ${trace_every} "
                "v_temp_inst v_press_inst v_density_inst v_volume_inst "
                "v_lx_inst v_ly_inst v_lz_inst file nvt_trace.dat"
            ),
            "fix DENS_NVT all nvt temp ${target_temp} ${target_temp} ${tdamp}",
            "run ${nvt_steps}",
            "unfix NVT_TRACE",
            "unfix DENS_NVT",
            f"write_data equil/{nvt_data}",
        ]
        for mode in modes:
            for strain in temp_prod.strains:
                lines.extend(_dma_read_header(ff, f"equil/{nvt_data}", prod.timestep_fs))
                lines.extend(_mode_block(mode, strain, temp_prod, "raw", ff.kspace_style, temperature_k=temperature))
        lines.append(f"write_data {polymer.name}.density_equilibrated_T_{token}K.dmax2.final.data")
        input_name = f"in.{polymer.name}_dmax2_density_equil_gt_T_{token}K"
        input_path = run_dir / input_name
        input_path.write_text("\n".join(lines) + "\n")
        log_name = f"{polymer.name}.density_equil_gt_T_{token}K.driver.log"
        (run_dir / "run.sh").write_text(
            "#!/usr/bin/env bash\n"
            "set -euo pipefail\n\n"
            'LMP_BIN="${LMP_BIN:-lmp}"\n'
            f'"${{LMP_BIN}}" -in "{input_name}" -log "{log_name}"\n'
        )
        (run_dir / "run.sh").chmod(0o775)
        production_steps = temp_prod.run_steps * max(1, len(modes) * len(temp_prod.strains))
        total_steps = int(
            glass.density_equilibration_steps
            + glass.density_sampling_steps
            + glass.density_dma_nvt_steps
            + production_steps
        )
        manifest_rows.append(
            f"{temperature:.8g},{run_dir.relative_to(workdir)},{input_path.relative_to(workdir)},{log_name},"
            f"{glass.density_equilibration_steps},0,{glass.density_dma_nvt_steps},"
            f"{production_steps},{total_steps}"
        )
        paths.append(input_path)

    (workdir / "temperature_grid.csv").write_text(
        "temperature_K\n" + "\n".join(f"{t:.8g}" for t in temperatures) + "\n"
    )
    if prior is not None:
        (workdir / "tg_prior.json").write_text(
            "{\n"
            f'  "matched_by": "{match_reason}",\n'
            f'  "key": "{prior.key}",\n'
            f'  "name": "{prior.name}",\n'
            f'  "tg_k": {prior.tg_k:.8g},\n'
            f'  "source": "{prior.source}",\n'
            f'  "notes": "{prior.notes}"\n'
            "}\n"
        )
    (workdir / "run_manifest.csv").write_text("\n".join(manifest_rows) + "\n")

    (workdir / "run_all_local.sh").write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n\n"
        "for d in temperatures/T_*K; do\n"
        '  (cd "$d" && ./run.sh)\n'
        "done\n"
    )
    (workdir / "run_all_local.sh").chmod(0o775)

    (workdir / "collect_and_parse.sh").write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n\n"
        "mkdir -p raw\n"
        "for f in temperatures/T_*K/raw/raw__*__strain_*__T_*K.dat; do\n"
        '  [ -e "$f" ] || continue\n'
        '  cp "$f" raw/\n'
        "done\n"
        'CONFIG_ARG=""\n'
        'if [ -f density_equilibrated_glass_transition_config.yaml ]; then\n'
        '  CONFIG_ARG="--config density_equilibrated_glass_transition_config.yaml"\n'
        'elif [ -f glass_transition_config.yaml ]; then\n'
        '  CONFIG_ARG="--config glass_transition_config.yaml"\n'
        "fi\n"
        f"dmax2 parse . --freq-ghz {prod.frequency_ghz:g} --discard-cycles {prod.discard_cycles} ${{CONFIG_ARG}}\n"
        f"dmax2 plot-nemd-directory . --freq-ghz {prod.frequency_ghz:g} --discard-cycles {prod.discard_cycles} --recursive --local --skip-root-raw --skip-existing\n"
        "dmax2 analyze-glass-transition . ${CONFIG_ARG}\n"
    )
    (workdir / "collect_and_parse.sh").chmod(0o775)
    return paths


def _master_curve_frequencies(master: MasterCurveSpec) -> list[float]:
    if master.frequencies_ghz:
        return [float(f) for f in master.frequencies_ghz]
    n = int(master.frequency_points)
    if n <= 1:
        return [float(master.min_frequency_ghz)]
    if master.frequency_spacing.lower() == "linear":
        step = (master.max_frequency_ghz - master.min_frequency_ghz) / (n - 1)
        return [float(master.min_frequency_ghz + i * step) for i in range(n)]
    lo = log10(float(master.min_frequency_ghz))
    hi = log10(float(master.max_frequency_ghz))
    return [float(10 ** (lo + i * (hi - lo) / (n - 1))) for i in range(n)]


def _slurm_time_for_steps(steps: int, master: MasterCurveSpec) -> str:
    seconds = steps / max(float(master.timesteps_per_second), 1.0)
    hours = max(float(master.slurm_min_hours), seconds * float(master.slurm_safety_factor) / 3600.0)
    minutes = int(ceil(hours * 4.0) * 15)
    return f"{minutes // 60:02d}:{minutes % 60:02d}:00"


def write_master_curve_inputs(
    polymer: PolymerSpec,
    ff: ForceFieldSpec,
    prod: DMAProductionSpec,
    glass: GlassTransitionSpec,
    master: MasterCurveSpec,
    workdir: str | Path,
    equilibrated_data: str | Path,
) -> list[Path]:
    """Write independent temperature/frequency DMA jobs for master curves."""

    workdir = Path(workdir)
    runs_dir = workdir / "runs"
    data_dir = workdir / "equilibrated"
    runs_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    data_path = Path(equilibrated_data)
    shared_data = data_dir / data_path.name
    if not shared_data.exists() and data_path.exists():
        shared_data.write_bytes(data_path.read_bytes())

    if glass.temperatures_k:
        temperatures = [float(t) for t in glass.temperatures_k]
        prior = None
        match_reason = "explicit_temperatures"
    else:
        grid, prior, match_reason = glass_temperature_grid(polymer, glass)
        temperatures = [float(t) for t in grid]
    frequencies = _master_curve_frequencies(master)

    modes = list(prod.modes) or ["shear_sllod_remapv"]
    paths: list[Path] = []
    manifest_rows = ["frequency_ghz,temperature_K,run_dir,input,log,production_steps,total_steps,slurm_time"]
    freq_to_temp_tokens: dict[str, list[str]] = {}
    for frequency in frequencies:
        freq_token = _frequency_token(frequency)
        freq_to_temp_tokens[freq_token] = []
        freq_dir = runs_dir / f"f_{freq_token}GHz"
        for temperature in temperatures:
            temp_token = _temperature_token(temperature)
            freq_to_temp_tokens[freq_token].append(temp_token)
            run_dir = freq_dir / f"T_{temp_token}K"
            run_dir.mkdir(parents=True, exist_ok=True)
            temp_data = f"{polymer.name}.f_{freq_token}GHz.T_{temp_token}K.equilibrated.data"
            temp_prod = replace(prod, temperature_k=temperature, frequency_ghz=frequency)
            production_steps = temp_prod.run_steps
            total_steps = int(glass.temperature_equilibration_steps + production_steps)
            slurm_time = _slurm_time_for_steps(total_steps, master)
            lines: list[str] = [
                "# DMAx2 master-curve temperature/frequency point.",
                "# Strategy: target-temperature relaxation followed by one dynamic production frequency.",
                *_dma_read_header(ff, f"../../../equilibrated/{shared_data.name}", prod.timestep_fs),
                f"variable target_temp equal {temperature:.8g}",
                f"variable ptarget equal {glass.pressure_atm:g}",
                f"variable tdamp equal {prod.tdamp_fs:g}",
                f"variable pdamp equal {prod.pdamp_fs:g}",
                f"variable prep_steps equal {glass.temperature_equilibration_steps}",
                "variable seed equal 928371",
                "shell mkdir -p equil",
                "",
                'print "DMAx2 master curve: target-temperature relaxation"',
                "velocity all create ${target_temp} ${seed} mom yes rot yes dist gaussian",
                "fix MC_EQ all npt temp ${target_temp} ${target_temp} ${tdamp} aniso ${ptarget} ${ptarget} ${pdamp} drag 1.0",
                "run ${prep_steps}",
                "unfix MC_EQ",
                f"write_data equil/{temp_data}",
            ]
            for mode in modes:
                for strain in temp_prod.strains:
                    lines.extend(_dma_read_header(ff, f"equil/{temp_data}", prod.timestep_fs))
                    lines.extend(
                        _mode_block(
                            mode,
                            strain,
                            temp_prod,
                            "raw",
                            ff.kspace_style,
                            temperature_k=temperature,
                            include_frequency_in_raw_name=True,
                        )
                    )
            lines.append(f"write_data {polymer.name}.f_{freq_token}GHz.T_{temp_token}K.dmax2.final.data")
            input_name = f"in.{polymer.name}_dmax2_mc_f_{freq_token}GHz_T_{temp_token}K"
            input_path = run_dir / input_name
            input_path.write_text("\n".join(lines) + "\n")
            log_name = f"{polymer.name}.mc_f_{freq_token}GHz_T_{temp_token}K.driver.log"
            (run_dir / "run.sh").write_text(
                "#!/usr/bin/env bash\n"
                "set -euo pipefail\n\n"
                'LMP_BIN="${LMP_BIN:-lmp}"\n'
                f'"${{LMP_BIN}}" -in "{input_name}" -log "{log_name}"\n'
            )
            (run_dir / "run.sh").chmod(0o775)
            manifest_rows.append(
                f"{frequency:.8g},{temperature:.8g},{run_dir.relative_to(workdir)},{input_path.relative_to(workdir)},{log_name},{production_steps},{total_steps},{slurm_time}"
            )
            paths.append(input_path)

    (workdir / "frequency_grid.csv").write_text(
        "frequency_ghz\n" + "\n".join(f"{f:.8g}" for f in frequencies) + "\n"
    )
    (workdir / "temperature_grid.csv").write_text(
        "temperature_K\n" + "\n".join(f"{t:.8g}" for t in temperatures) + "\n"
    )
    (workdir / "run_manifest.csv").write_text("\n".join(manifest_rows) + "\n")
    if prior is not None:
        (workdir / "tg_prior.json").write_text(
            "{\n"
            f'  "matched_by": "{match_reason}",\n'
            f'  "key": "{prior.key}",\n'
            f'  "name": "{prior.name}",\n'
            f'  "tg_k": {prior.tg_k:.8g},\n'
            f'  "source": "{prior.source}",\n'
            f'  "notes": "{prior.notes}"\n'
            "}\n"
        )

    (workdir / "collect_and_parse.sh").write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n\n"
        "mkdir -p raw\n"
        "for f in runs/f_*GHz/temperatures/T_*K/raw/raw__*__strain_*__f_*GHz__T_*K.dat runs/f_*GHz/T_*K/raw/raw__*__strain_*__f_*GHz__T_*K.dat; do\n"
        '  [ -e "$f" ] || continue\n'
        '  cp "$f" raw/\n'
        "done\n"
        'if ! compgen -G "raw/raw__*.dat" > /dev/null; then\n'
        '  echo "No master-curve raw traces found under runs/f_*GHz/T_*K/raw." >&2\n'
        "  exit 2\n"
        "fi\n"
        'CONFIG_ARG=""\n'
        'if [ -f master_curve_config.yaml ]; then\n'
        '  CONFIG_ARG="--config master_curve_config.yaml"\n'
        "fi\n"
        f"dmax2 parse . --freq-ghz {prod.frequency_ghz:g} --discard-cycles {prod.discard_cycles} ${{CONFIG_ARG}}\n"
        f"dmax2 plot-nemd-directory . --freq-ghz {prod.frequency_ghz:g} --discard-cycles {prod.discard_cycles} --recursive --local --skip-root-raw --skip-existing\n"
    )
    (workdir / "collect_and_parse.sh").chmod(0o775)

    scripts_dir = workdir / "slurm_arrays"
    scripts_dir.mkdir(exist_ok=True)
    submit_lines = ["#!/usr/bin/env bash", "set -euo pipefail", ""]
    for frequency in frequencies:
        freq_token = _frequency_token(frequency)
        temp_tokens = freq_to_temp_tokens[freq_token]
        temp_prod = replace(prod, frequency_ghz=frequency)
        total_steps = int(glass.temperature_equilibration_steps + temp_prod.run_steps)
        slurm_time = _slurm_time_for_steps(total_steps, master)
        script = scripts_dir / f"slurm_array_f_{freq_token}GHz.sh"
        script.write_text(
            "#!/usr/bin/env bash\n"
            f"#SBATCH -J {polymer.name}_mc_f{freq_token}\n"
            "#SBATCH -A m5056\n"
            "#SBATCH -C cpu\n"
            "#SBATCH -q regular\n"
            f"#SBATCH -t {slurm_time}\n"
            "#SBATCH -N 1\n"
            "#SBATCH --ntasks-per-node=128\n"
            f"#SBATCH --array=0-{len(temp_tokens) - 1}\n\n"
            "set -euo pipefail\n"
            f'FREQ_TOKEN="{freq_token}"\n'
            'TEMPS=("' + '" "'.join(temp_tokens) + '")\n'
            'TEMP_TOKEN="${TEMPS[$SLURM_ARRAY_TASK_ID]}"\n'
            'cd "${SLURM_SUBMIT_DIR}/runs/f_${FREQ_TOKEN}GHz/T_${TEMP_TOKEN}K"\n'
            f'INPUT_FILENAME="in.{polymer.name}_dmax2_mc_f_${{FREQ_TOKEN}}GHz_T_${{TEMP_TOKEN}}K"\n'
            f'LOG_FILENAME="{polymer.name}.mc_f_${{FREQ_TOKEN}}GHz_T_${{TEMP_TOKEN}}K.driver.log"\n'
            'LMP_BIN="${LMP_BIN:-lmp}"\n'
            'srun -n "${SLURM_NTASKS}" "${LMP_BIN}" -in "${INPUT_FILENAME}" -log "${LOG_FILENAME}"\n'
        )
        script.chmod(0o775)
        submit_lines.append(f"sbatch slurm_arrays/{script.name}")
    (workdir / "submit_all_frequency_arrays.sh").write_text("\n".join(submit_lines) + "\n")
    (workdir / "submit_all_frequency_arrays.sh").chmod(0o775)
    return paths
