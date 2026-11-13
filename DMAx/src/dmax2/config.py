"""Configuration dataclasses for DMAx2 workflows."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any


def _coerce_dataclass(cls: type, data: dict[str, Any]):
    names = {f.name for f in fields(cls)}
    kwargs = {}
    for name in names:
        if name in data:
            kwargs[name] = data[name]
    return cls(**kwargs)


@dataclass
class PolymerSpec:
    name: str = "PDMS"
    smiles: str = "[*]O[Si](C)(C)O[*]"
    monomers_per_chain: int = 11
    chains: int = 100
    conformers: int = 1
    cyclic: bool = False
    initial_density_g_cm3: float = 0.65
    structure_method: str = "psp"
    existing_data_file: str | None = None


@dataclass
class ForceFieldSpec:
    provider: str = "psp_oplsaa"
    forcefield: str = "oplsaa"
    neutralize: bool = True
    foyer_xml: str | None = None
    pair_style: str = "lj/cut/coul/long 8.0"
    bond_style: str = "harmonic"
    angle_style: str = "harmonic"
    dihedral_style: str = "opls"
    improper_style: str = "none"
    kspace_style: str = "pppm 1.0e-5"
    pair_modify: str = "mix geometric"
    special_bonds: str = "lj/coul 0.0 0.0 0.5"


@dataclass
class EquilibrationSpec:
    data_file: str | None = None
    output_directory: str | None = None
    temperature_k: float = 300.0
    pressure_atm: float = 1.0
    timestep_fs: float = 1.0
    tdamp_fs: float = 100.0
    pdamp_fs: float = 2000.0
    pdamp_short_fs: float | None = None
    pdamp_long_fs: float | None = None
    minimize_steps: int = 5000
    minimize_evaluations: int = 20000
    heat_steps: int = 10000
    target_npt_steps: int = 200000
    anneal_steps: int | None = None
    cool_steps: int | None = None
    npt_anneal_steps: int = 250000
    npt_relax_steps: int = 250000
    nvt_final_steps: int = 50000
    relax_npt_steps: int | None = None
    final_nvt_steps: int | None = None
    anneal_temperature_k: float | None = None
    cool_temperature_k: float | None = None
    trace_every: int = 1000
    dump_every: int = 20000
    restart_every: int = 200000


@dataclass
class DMAProductionSpec:
    frequency_ghz: float = 25.0
    strains: list[float] = field(default_factory=lambda: [0.02, 0.04, 0.06, 0.08])
    cycles: int = 10
    discard_cycles: int = 2
    timestep_fs: float = 1.0
    temperature_k: float = 300.0
    tdamp_fs: float = 100.0
    pdamp_fs: float = 120000.0
    output_every: int = 100
    modes: list[str] = field(default_factory=lambda: ["shear_sllod_remapv"])
    legacy_modes: list[str] = field(default_factory=lambda: ["legacy_axial_pressure"])
    recommended_signal: str = "sxy_vir_MPa"
    # Explicit future-run repair; archival template behavior stays the default.
    validated_mechanics: bool = False

    @property
    def period_fs(self) -> float:
        return 1_000_000.0 / self.frequency_ghz

    @property
    def steps_per_cycle(self) -> int:
        return int(round(self.period_fs / self.timestep_fs))

    @property
    def run_steps(self) -> int:
        return self.cycles * self.steps_per_cycle


@dataclass
class ConvergenceSpec:
    min_r2: float = 0.70
    min_snr: float = 2.0
    max_harmonic_ratio: float = 0.20
    max_drift_fraction_per_cycle: float = 0.05
    preferred_strain_window: tuple[float, float] = (0.04, 0.08)
    cycle_counts: list[int] = field(default_factory=lambda: list(range(1, 11)))
    cycle_r2_tolerance: float = 0.01
    cycle_modulus_tolerance: float = 0.05
    cycle_consecutive_windows: int = 1
    minimum_cycles: int = 2


@dataclass
class ErrorAnalysisSpec:
    enabled: bool = True
    replicates: int = 5
    seeds: list[int] = field(default_factory=list)
    seed_start: int = 917531
    strain: float | None = None
    cycles: int | None = None
    discard_cycles: int | None = None
    frequency_ghz: float | None = None
    temperature_k: float | None = None
    include_legacy_modes: bool = False
    slurm_account: str = "m4600"
    slurm_constraint: str = "cpu"
    slurm_queue: str = "regular"
    slurm_time: str = "01:00:00"
    slurm_nodes: int = 1
    slurm_ntasks_per_node: int = 128
    slurm_cpus_per_task: int = 2
    shifter_image: str = "docker:nersc/lammps_lite:25.04"


@dataclass
class TensionProtocolSpec:
    enabled: bool = True
    nvt_mode: str = "tension_nvt_z_fixed_lateral"
    npt_mode: str = "tension_npt_z_lateral"
    npt_equilibration_ps: float = 100.0
    nvt_hold_ps: float = 100.0
    pressure_atm: float = 1.0
    include_legacy_modes: bool = False
    slurm_account: str = "m4600"
    slurm_constraint: str = "cpu"
    slurm_queue: str = "regular"
    slurm_time: str = "02:00:00"
    slurm_nodes: int = 1
    slurm_ntasks_per_node: int = 128
    slurm_cpus_per_task: int = 2
    shifter_image: str = "docker:nersc/lammps_lite:25.04"


@dataclass
class PreTensionProtocolSpec:
    enabled: bool = True
    nvt_mode: str = "tension_pretension_nvt_z_fixed_lateral"
    npt_mode: str = "tension_pretension_npt_z_lateral"
    static_stress_to_dynamic_stress_ratio: float = 2.0
    ramp_ps: float = 50.0
    hold_ps: float = 100.0
    pressure_atm: float = 1.0
    include_npt_lateral_mode: bool = True
    slurm_account: str = "m4600"
    slurm_constraint: str = "cpu"
    slurm_queue: str = "regular"
    slurm_time: str = "02:00:00"
    slurm_nodes: int = 1
    slurm_ntasks_per_node: int = 128
    slurm_cpus_per_task: int = 2
    shifter_image: str = "docker:nersc/lammps_lite:25.04"


@dataclass
class GlassTransitionSpec:
    enabled: bool = True
    temperatures_k: list[float] = field(default_factory=list)
    expected_tg_k: float | None = None
    default_reference_polymer: str = "PVDF"
    temperature_window_k: float = 200.0
    temperature_points: int = 40
    min_temperature_k: float = 1.0
    max_temperature_k: float | None = None
    temperature_equilibration_steps: int = 100000
    density_equilibration_steps: int = 500000
    density_sampling_steps: int = 250000
    density_output_every: int = 1000
    density_pdamp_fs: float = 5000.0
    density_dma_nvt_steps: int = 100000
    pressure_atm: float = 1.0
    include_legacy_modes: bool = False
    method: str = "tan_delta_inflection"
    comparison_methods: list[str] = field(default_factory=lambda: ["density_piecewise"])
    observable: str = "specific_volume"
    tan_delta_frequency_ghz: float | None = None
    tan_delta_signal: str = "sxy_vir_MPa"
    tan_delta_parser: str = "lockin"
    smoothing_window: int = 3


@dataclass
class MasterCurveSpec:
    enabled: bool = True
    reference_temperature_k: float = 300.0
    frequencies_ghz: list[float] = field(default_factory=list)
    min_frequency_ghz: float = 1.0
    max_frequency_ghz: float = 100.0
    frequency_points: int = 10
    frequency_spacing: str = "log"
    timesteps_per_second: float = 577.395
    slurm_safety_factor: float = 1.35
    slurm_min_hours: float = 1.0
    shift_model: str = "wlf"
    metric: str = "storage_modulus_MPa"
    include_vertical_shift: bool = False
    fixed_noise: float = 0.04
    experimental_min_frequency_hz: float = 1.0e-3
    experimental_max_frequency_hz: float = 1.0e3
    extrapolation_points: int = 360
    enable_low_frequency_extrapolation: bool = False
    anchor_reference_source: str = "unspecified"


@dataclass
class DMAx2WorkflowSpec:
    polymer: PolymerSpec = field(default_factory=PolymerSpec)
    forcefield: ForceFieldSpec = field(default_factory=ForceFieldSpec)
    equilibration: EquilibrationSpec = field(default_factory=EquilibrationSpec)
    production: DMAProductionSpec = field(default_factory=DMAProductionSpec)
    convergence: ConvergenceSpec = field(default_factory=ConvergenceSpec)
    error_analysis: ErrorAnalysisSpec = field(default_factory=ErrorAnalysisSpec)
    tension_protocol: TensionProtocolSpec = field(default_factory=TensionProtocolSpec)
    pretension_protocol: PreTensionProtocolSpec = field(default_factory=PreTensionProtocolSpec)
    glass_transition: GlassTransitionSpec = field(default_factory=GlassTransitionSpec)
    master_curve: MasterCurveSpec = field(default_factory=MasterCurveSpec)
    project_name: str = "PDMS_dma2"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DMAx2WorkflowSpec":
        return cls(
            polymer=_coerce_dataclass(PolymerSpec, data.get("polymer", {})),
            forcefield=_coerce_dataclass(ForceFieldSpec, data.get("forcefield", {})),
            equilibration=_coerce_dataclass(EquilibrationSpec, data.get("equilibration", {})),
            production=_coerce_dataclass(DMAProductionSpec, data.get("production", {})),
            convergence=_coerce_dataclass(ConvergenceSpec, data.get("convergence", {})),
            error_analysis=_coerce_dataclass(ErrorAnalysisSpec, data.get("error_analysis", {})),
            tension_protocol=_coerce_dataclass(TensionProtocolSpec, data.get("tension_protocol", {})),
            pretension_protocol=_coerce_dataclass(
                PreTensionProtocolSpec, data.get("pretension_protocol", {})
            ),
            glass_transition=_coerce_dataclass(GlassTransitionSpec, data.get("glass_transition", {})),
            master_curve=_coerce_dataclass(MasterCurveSpec, data.get("master_curve", {})),
            project_name=data.get("project_name", "PDMS_dma2"),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_file(cls, path: str | Path) -> "DMAx2WorkflowSpec":
        path = Path(path)
        text = path.read_text()
        if path.suffix.lower() in {".yaml", ".yml"}:
            try:
                import yaml
            except ImportError as exc:
                raise RuntimeError("Install pyyaml or use JSON config files.") from exc
            data = yaml.safe_load(text)
        else:
            import json
            data = json.loads(text)
        return cls.from_dict(data)

    def write(self, path: str | Path) -> Path:
        path = Path(path)
        if path.suffix.lower() in {".yaml", ".yml"}:
            import yaml
            path.write_text(yaml.safe_dump(self.to_dict(), sort_keys=False))
        else:
            import json
            path.write_text(json.dumps(self.to_dict(), indent=2))
        return path


def dataclass_to_plain(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    return value
