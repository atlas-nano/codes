"""High-level standalone DMAx2 workflow."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .analysis.convergence import (
    choose_cycle_convergence_targets,
    compare_parser_stability,
    cycle_count_convergence,
    recommend_strains,
    write_cycle_convergence_plots,
)
from .analysis.dma import parse_dma_directory, write_dma_plots, write_nemd_diagnostics_for_directory
from .analysis.glass_transition import fit_piecewise_tg, fit_tandelta_tg, prepare_tandelta_sweep
from .analysis.master_curve import (
    assign_wlf_shifts,
    build_master_curve,
    build_master_curve_with_mastercurves,
)
from .config import DMAx2WorkflowSpec
from .forcefields import plan_forcefield
from .lammps.inputs import write_dma_input, write_equilibration_input
from .polymer_properties import glass_temperature_grid
from .schemas import WorkflowResult, WorkflowStep
from .structures import plan_structure_generation


class DMAx2Workflow:
    def __init__(self, spec: DMAx2WorkflowSpec):
        self.spec = spec

    def build_plan(self, workdir: str | Path) -> list[WorkflowStep]:
        workdir = Path(workdir)
        return [
            WorkflowStep("structure_generation", "structure", outputs={"data_file": str(workdir / f"data.{self.spec.polymer.name}")}),
            WorkflowStep("forcefield_parameterization", "forcefield", depends_on=["structure_generation"]),
            WorkflowStep("equilibration_input", "lammps_input", depends_on=["forcefield_parameterization"]),
            WorkflowStep("dma_input", "lammps_input", depends_on=["equilibration_input"]),
            WorkflowStep("dma_parse", "analysis", depends_on=["dma_input"]),
            WorkflowStep("strain_runtime_convergence", "analysis", depends_on=["dma_parse"]),
            WorkflowStep("error_analysis", "analysis", depends_on=["strain_runtime_convergence"]),
            WorkflowStep("glass_transition", "analysis", depends_on=["error_analysis", "equilibration_input"]),
            WorkflowStep("master_curve", "analysis", depends_on=["dma_parse", "glass_transition"]),
        ]

    def prepare(self, workdir: str | Path) -> dict[str, Path]:
        workdir = Path(workdir)
        workdir.mkdir(parents=True, exist_ok=True)
        structure_plan = plan_structure_generation(self.spec.polymer, workdir)
        forcefield_plan = plan_forcefield(self.spec.polymer, self.spec.forcefield, workdir)
        eq_input = write_equilibration_input(
            self.spec.polymer, self.spec.forcefield, self.spec.equilibration, workdir
        )
        dma_input = write_dma_input(
            self.spec.polymer, self.spec.forcefield, self.spec.production, workdir
        )
        return {
            "structure_script": structure_plan.script or Path(""),
            "data_file": structure_plan.data_file,
            "forcefield_script": forcefield_plan.script or Path(""),
            "equilibration_input": eq_input,
            "dma_input": dma_input,
        }

    def write_synthetic_dma(self, workdir: str | Path) -> Path:
        """Write synthetic traces for smoke-testing the full pipeline."""

        workdir = Path(workdir)
        raw = workdir / "raw"
        raw.mkdir(parents=True, exist_ok=True)
        prod = self.spec.production
        rng = np.random.default_rng(12345)
        cols = [
            "step", "t_ps", "strain_applied", "strain_box", "eps_x", "eps_y",
            "szz_tot_MPa", "szz_vir_MPa", "sdev_MPa", "sxy_tot_MPa", "sxy_vir_MPa",
            "temp_K", "press_MPa", "lx", "ly", "lz", "density",
        ]
        steps = np.arange(0, prod.run_steps + 1, prod.output_every)
        t_fs = steps * prod.timestep_fs
        omega = 2.0 * np.pi * t_fs / prod.period_fs
        for mode in [*prod.modes, *prod.legacy_modes]:
            for strain in prod.strains:
                gamma = strain * np.sin(omega)
                if mode.startswith("shear"):
                    g_storage, g_loss = 110.0, 165.0
                    stress = strain * (g_storage * np.sin(omega) + g_loss * np.cos(omega))
                    stress += rng.normal(0.0, 2.0, size=len(steps))
                    sxy_vir = stress
                    sxy_tot = stress + rng.normal(0.0, 1.5, size=len(steps))
                    szz_vir = rng.normal(0.0, 8.0, size=len(steps))
                    press = rng.normal(0.0, 12.0, size=len(steps))
                else:
                    e_storage, e_loss = 3.5, 14.0
                    stress = strain * (e_storage * np.sin(omega) + e_loss * np.cos(omega))
                    stress += rng.normal(0.0, 20.0, size=len(steps))
                    szz_vir = stress
                    press = -0.33 * stress + rng.normal(0.0, 25.0, size=len(steps))
                    sxy_vir = rng.normal(0.0, 4.0, size=len(steps))
                    sxy_tot = sxy_vir + rng.normal(0.0, 2.0, size=len(steps))
                data = pd.DataFrame({
                    "step": steps,
                    "t_ps": t_fs / 1000.0,
                    "strain_applied": gamma,
                    "strain_box": gamma,
                    "eps_x": 0.0,
                    "eps_y": 0.0,
                    "szz_tot_MPa": szz_vir + rng.normal(0.0, 5.0, size=len(steps)),
                    "szz_vir_MPa": szz_vir,
                    "sdev_MPa": szz_vir - rng.normal(0.0, 5.0, size=len(steps)),
                    "sxy_tot_MPa": sxy_tot,
                    "sxy_vir_MPa": sxy_vir,
                    "temp_K": prod.temperature_k + rng.normal(0.0, 1.5, size=len(steps)),
                    "press_MPa": press,
                    "lx": 60.0,
                    "ly": 60.0,
                    "lz": 60.0,
                    "density": 0.97,
                }, columns=cols)
                token = f"{strain:.3f}".replace(".", "p")
                data.to_csv(raw / f"raw__{mode}__strain_{token}.dat", sep=" ", index=False)
        return raw

    def analyze(self, workdir: str | Path) -> dict[str, Path]:
        workdir = Path(workdir)
        summary = parse_dma_directory(
            workdir,
            frequency_ghz=self.spec.production.frequency_ghz,
            discard_cycles=self.spec.production.discard_cycles,
            parsers=("lockin", "legacy"),
        )
        summary["frequency_ghz"] = self.spec.production.frequency_ghz
        if "temperature_K" in summary:
            summary["temperature_K"] = summary["temperature_K"].fillna(self.spec.production.temperature_k)
        else:
            summary["temperature_K"] = self.spec.production.temperature_k
        summary_path = workdir / "dma_summary.csv"
        summary.to_csv(summary_path, index=False)
        write_dma_plots(summary, workdir)
        write_nemd_diagnostics_for_directory(
            workdir,
            frequency_ghz=self.spec.production.frequency_ghz,
            discard_cycles=self.spec.production.discard_cycles,
        )

        recs = recommend_strains(summary, self.spec.convergence)
        rec_path = workdir / "dma_strain_recommendations.csv"
        recs.to_csv(rec_path, index=False)

        stability = compare_parser_stability(summary)
        stability_path = workdir / "dma_parser_stability.csv"
        stability.to_csv(stability_path, index=False)

        cycle_table_path, cycle_rec_path = self.analyze_cycle_convergence(workdir, recs)
        return {
            "dma_summary": summary_path,
            "recommendations": rec_path,
            "parser_stability": stability_path,
            "cycle_convergence": cycle_table_path,
            "cycle_recommendations": cycle_rec_path,
        }

    def analyze_cycle_convergence(self, workdir: str | Path, recommendations: pd.DataFrame) -> tuple[Path, Path]:
        workdir = Path(workdir)
        conv = self.spec.convergence
        prod = self.spec.production
        target_rows = choose_cycle_convergence_targets(
            recommendations,
            preferred_mode=prod.modes[0] if prod.modes else "shear_sllod_remapv",
            preferred_signal=prod.recommended_signal,
            parser="lockin",
        )
        all_tables = []
        all_recs = []
        for _, target in target_rows.iterrows():
            token = f"{float(target['strain']):.3f}".replace(".", "p")
            raw_file = workdir / "raw" / f"raw__{target['mode']}__strain_{token}.dat"
            if not raw_file.exists():
                continue
            table, rec = cycle_count_convergence(
                str(raw_file),
                frequency_ghz=prod.frequency_ghz,
                signal=str(target["signal"]),
                cycle_counts=conv.cycle_counts,
                parser="lockin",
                tolerance_r2=conv.cycle_r2_tolerance,
                tolerance_modulus=conv.cycle_modulus_tolerance,
                consecutive_windows=conv.cycle_consecutive_windows,
                minimum_cycles=conv.minimum_cycles,
                discard_initial_cycles=prod.discard_cycles,
                max_harmonic_ratio=conv.max_harmonic_ratio,
                max_drift_fraction_per_cycle=conv.max_drift_fraction_per_cycle,
            )
            table["raw_file"] = str(raw_file)
            rec["raw_file"] = str(raw_file)
            all_tables.append(table)
            all_recs.append(rec)
        table_out = workdir / "dma_cycle_convergence.csv"
        rec_out = workdir / "dma_cycle_recommendations.csv"
        pd.concat(all_tables, ignore_index=True).to_csv(table_out, index=False) if all_tables else pd.DataFrame().to_csv(table_out, index=False)
        if all_recs:
            recommendations = pd.concat(all_recs, ignore_index=True)
            recommendations.to_csv(rec_out, index=False)
            write_cycle_convergence_plots(pd.concat(all_tables, ignore_index=True), recommendations, workdir)
        else:
            pd.DataFrame().to_csv(rec_out, index=False)
        return table_out, rec_out

    def analyze_glass_transition(self, workdir: str | Path) -> Path:
        workdir = Path(workdir)
        if self.spec.glass_transition.temperatures_k:
            temps = np.array(self.spec.glass_transition.temperatures_k, dtype=float)
        else:
            temps, _prior, _match_reason = glass_temperature_grid(
                self.spec.polymer, self.spec.glass_transition
            )
        tg_true = 220.0
        density = np.where(temps < tg_true, 1.02 - 2.0e-4 * (temps - tg_true), 1.02 - 7.0e-4 * (temps - tg_true))
        df = pd.DataFrame({"temperature_K": temps, "density_g_cm3": density})
        density_fit = fit_piecewise_tg(df, observable=self.spec.glass_transition.observable)
        density_out = workdir / "glass_transition_density_fit.csv"
        pd.DataFrame([density_fit.as_dict() | {"method": "density_piecewise"}]).to_csv(density_out, index=False)

        tan_delta_input = self._load_or_synthesize_tandelta_sweep(workdir)
        tan_delta_input.to_csv(workdir / "glass_transition_tandelta_input.csv", index=False)
        tan_delta_fit = fit_tandelta_tg(
            tan_delta_input,
            smoothing_window=self.spec.glass_transition.smoothing_window,
            method=self.spec.glass_transition.method.replace("tan_delta_", ""),
        )
        tan_out = workdir / "glass_transition_tandelta_fit.csv"
        pd.DataFrame([tan_delta_fit.as_dict() | {"method": self.spec.glass_transition.method}]).to_csv(tan_out, index=False)

        out = workdir / "glass_transition_fit.csv"
        pd.DataFrame(
            [
                {"estimator": "tan_delta", **tan_delta_fit.as_dict()},
                {"estimator": "density_piecewise", "tg_k": density_fit.tg_k, **density_fit.as_dict()},
            ]
        ).to_csv(out, index=False)
        return out

    def _load_or_synthesize_tandelta_sweep(self, workdir: Path) -> pd.DataFrame:
        summary_path = workdir / "dma_summary.csv"
        gt = self.spec.glass_transition
        if summary_path.exists():
            summary = pd.read_csv(summary_path)
            try:
                sweep = prepare_tandelta_sweep(
                    summary,
                    signal=gt.tan_delta_signal,
                    parser=gt.tan_delta_parser,
                    frequency_ghz=gt.tan_delta_frequency_ghz,
                )
                if len(sweep) >= 4:
                    return sweep
            except ValueError:
                pass
        temps = np.array(gt.temperatures_k, dtype=float)
        if len(temps) == 0:
            temps, _prior, _match_reason = glass_temperature_grid(self.spec.polymer, gt)
        tg_true = 220.0
        width = 38.0
        tan_delta = 0.055 + 0.26 * np.exp(-0.5 * ((temps - tg_true) / width) ** 2)
        tan_delta += 0.0004 * (temps - temps.mean())
        return pd.DataFrame({"temperature_K": temps, "tan_delta": tan_delta})

    def analyze_master_curve(self, workdir: str | Path) -> Path:
        workdir = Path(workdir)
        summary = pd.read_csv(workdir / "dma_summary.csv")
        mc_spec = self.spec.master_curve
        if mc_spec.shift_model == "mastercurves":
            result = build_master_curve_with_mastercurves(
                summary,
                reference_temperature_k=mc_spec.reference_temperature_k,
                metric=mc_spec.metric,
                include_vertical_shift=mc_spec.include_vertical_shift,
                fixed_noise=mc_spec.fixed_noise,
            )
            master = result.master_curve
            result.shift_table.to_csv(workdir / "master_curve_shift_parameters.csv", index=False)
            pd.DataFrame([{"shift_model": "mastercurves", "loss": result.loss}]).to_csv(
                workdir / "master_curve_backend_summary.csv", index=False
            )
        else:
            shifted, params = assign_wlf_shifts(summary, mc_spec.reference_temperature_k)
            master = build_master_curve(shifted, mc_spec.reference_temperature_k)
            pd.DataFrame([params.as_dict()]).to_csv(workdir / "master_curve_shift_parameters.csv", index=False)
        out = workdir / "master_curve.csv"
        master.to_csv(out, index=False)
        return out

    def run_all(self, workdir: str | Path, dry_run: bool = True, synthetic: bool = False) -> WorkflowResult:
        workdir = Path(workdir)
        artifacts = {k: str(v) for k, v in self.prepare(workdir).items()}
        if synthetic:
            artifacts["synthetic_raw"] = str(self.write_synthetic_dma(workdir))
        if not dry_run:
            raise NotImplementedError("External LAMMPS execution is intentionally left to the caller for now.")
        if synthetic:
            artifacts.update({k: str(v) for k, v in self.analyze(workdir).items()})
            artifacts["glass_transition"] = str(self.analyze_glass_transition(workdir))
            artifacts["master_curve"] = str(self.analyze_master_curve(workdir))
        return WorkflowResult(str(workdir), self.build_plan(workdir), artifacts)
