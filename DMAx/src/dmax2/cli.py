"""Command-line entry points for DMAx2."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from .analysis.equilibration import analyze_equilibration_trace, stage_boundaries_from_config
from .analysis.error_analysis import write_error_analysis_report
from .analysis.glass_transition import (
    write_density_glass_transition_report,
    write_glass_transition_comparison_report,
    write_glass_transition_report,
)
from .analysis.dma import (
    parse_dma_directory,
    write_dma_plots,
    write_legacy_pressure_phase_plot,
    write_nemd_diagnostics_for_directory,
    write_nemd_first_harmonic_diagnostics,
    write_parser_comparison_plots,
)
from .analysis.convergence import (
    compare_parser_stability,
    recommend_strains,
    select_optimal_strain,
    write_cycle_convergence_report,
    write_strain_convergence_report,
)
from .analysis.master_curve import write_master_curve_analysis_report
from .config import DMAx2WorkflowSpec
from .lammps.inputs import (
    write_equilibration_input,
    write_pretensioned_tension_strain_convergence_inputs,
    write_strain_convergence_inputs,
    write_tension_strain_convergence_inputs,
)
from .lammps.inputs import (
    write_density_equilibrated_glass_transition_inputs,
    write_density_glass_transition_inputs,
    write_error_analysis_inputs,
    write_glass_transition_inputs,
    write_master_curve_inputs,
)
from .workflow import DMAx2Workflow


def _load_plain_config(path: str | Path) -> dict:
    path = Path(path)
    text = path.read_text()
    if path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError("Install pyyaml to load YAML config files.") from exc
        return yaml.safe_load(text) or {}
    return json.loads(text)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="dmax2")
    sub = parser.add_subparsers(dest="cmd", required=True)

    run_p = sub.add_parser("run")
    run_p.add_argument("config")
    run_p.add_argument("--workdir", default="dmax2_run")
    run_p.add_argument("--dry-run", action="store_true")
    run_p.add_argument("--synthetic", action="store_true")

    parse_p = sub.add_parser("parse")
    parse_p.add_argument("output_dir")
    parse_p.add_argument("--freq-ghz", type=float, required=True)
    parse_p.add_argument("--discard-cycles", type=int, default=2)
    parse_p.add_argument("--config")

    trace_plot_p = sub.add_parser("plot-nemd-diagnostics")
    trace_plot_p.add_argument("raw_file")
    trace_plot_p.add_argument("--output-dir", default=".")
    trace_plot_p.add_argument("--freq-ghz", type=float, required=True)
    trace_plot_p.add_argument("--discard-cycles", type=int, default=2)
    trace_plot_p.add_argument("--parser", default="lockin")
    trace_plot_p.add_argument("--cycles-to-plot", type=float)
    trace_plot_p.add_argument("--signal", action="append")
    trace_plot_p.add_argument("--plot-subdir", default="plots")

    trace_dir_plot_p = sub.add_parser("plot-nemd-directory")
    trace_dir_plot_p.add_argument("output_dir")
    trace_dir_plot_p.add_argument("--freq-ghz", type=float, required=True)
    trace_dir_plot_p.add_argument("--discard-cycles", type=int, default=2)
    trace_dir_plot_p.add_argument("--parser", default="lockin")
    trace_dir_plot_p.add_argument("--signal", action="append")
    trace_dir_plot_p.add_argument("--recursive", action="store_true")
    trace_dir_plot_p.add_argument("--local", action="store_true")
    trace_dir_plot_p.add_argument("--skip-root-raw", action="store_true")
    trace_dir_plot_p.add_argument("--skip-existing", action="store_true")

    eq_p = sub.add_parser("write-equilibration")
    eq_p.add_argument("config")
    eq_p.add_argument("--workdir", default=".")
    eq_p.add_argument("--data-file")
    eq_p.add_argument("--output-directory")
    eq_p.add_argument("--filename")

    eq_plot_p = sub.add_parser("plot-equilibration")
    eq_plot_p.add_argument("trace")
    eq_plot_p.add_argument("--output-dir", default="equilibration_analysis")
    eq_plot_p.add_argument("--config")
    eq_plot_p.add_argument("--tail-ps", type=float, default=100.0)
    eq_plot_p.add_argument("--prefix", default="equilibration")

    strain_p = sub.add_parser("write-strain-convergence")
    strain_p.add_argument("config")
    strain_p.add_argument("--workdir", default="strain_convergence")
    strain_p.add_argument("--equilibrated-data", required=True)

    tension_strain_p = sub.add_parser("write-tension-strain-convergence")
    tension_strain_p.add_argument("config")
    tension_strain_p.add_argument("--workdir", default="tension_strain_convergence")
    tension_strain_p.add_argument("--equilibrated-data", required=True)

    pretension_strain_p = sub.add_parser("write-pretensioned-tension-strain-convergence")
    pretension_strain_p.add_argument("config")
    pretension_strain_p.add_argument("--workdir", default="pretensioned_tension_strain_convergence")
    pretension_strain_p.add_argument("--equilibrated-data", required=True)

    cycle_p = sub.add_parser("cycle-convergence")
    cycle_p.add_argument("config")
    cycle_p.add_argument("--workdir", default="cycle_convergence")
    cycle_p.add_argument("--raw-file", required=True)
    cycle_p.add_argument("--signal")
    cycle_p.add_argument("--parser", default="lockin")
    cycle_p.add_argument("--freq-ghz", type=float)
    cycle_p.add_argument("--discard-cycles", type=int)
    cycle_p.add_argument("--copy-raw", action=argparse.BooleanOptionalAction, default=True)

    error_write_p = sub.add_parser("write-error-analysis")
    error_write_p.add_argument("config")
    error_write_p.add_argument("--workdir", default="error_analysis")
    error_write_p.add_argument("--equilibrated-data", required=True)

    error_analyze_p = sub.add_parser("analyze-error")
    error_analyze_p.add_argument("output_dir")
    error_analyze_p.add_argument("--config")
    error_analyze_p.add_argument("--summary", default="dma_summary.csv")
    error_analyze_p.add_argument("--signal")
    error_analyze_p.add_argument("--parser", default="lockin")
    error_analyze_p.add_argument("--mode")

    gt_write_p = sub.add_parser("write-glass-transition")
    gt_write_p.add_argument("config")
    gt_write_p.add_argument("--workdir", default="glass_transition")
    gt_write_p.add_argument("--equilibrated-data", required=True)

    gt_analyze_p = sub.add_parser("analyze-glass-transition")
    gt_analyze_p.add_argument("output_dir")
    gt_analyze_p.add_argument("--config")
    gt_analyze_p.add_argument("--summary", default="dma_summary.csv")

    dgt_write_p = sub.add_parser("write-density-glass-transition")
    dgt_write_p.add_argument("config")
    dgt_write_p.add_argument("--workdir", default="density_glass_transition")
    dgt_write_p.add_argument("--equilibrated-data", required=True)

    degt_write_p = sub.add_parser("write-density-equilibrated-glass-transition")
    degt_write_p.add_argument("config")
    degt_write_p.add_argument("--workdir", default="density_equilibrated_glass_transition")
    degt_write_p.add_argument("--equilibrated-data", required=True)

    dgt_analyze_p = sub.add_parser("analyze-density-glass-transition")
    dgt_analyze_p.add_argument("output_dir")
    dgt_analyze_p.add_argument("--config")
    dgt_analyze_p.add_argument("--tail-fraction", type=float, default=0.5)

    gt_compare_p = sub.add_parser("compare-glass-transition")
    gt_compare_p.add_argument("output_dir")
    gt_compare_p.add_argument("--dma-dir", action="append", required=True)
    gt_compare_p.add_argument("--density-dir", required=True)
    gt_compare_p.add_argument("--expected-tg-k", type=float)

    mc_write_p = sub.add_parser("write-master-curve")
    mc_write_p.add_argument("config")
    mc_write_p.add_argument("--workdir", default="master_curve")
    mc_write_p.add_argument("--equilibrated-data", required=True)

    mc_analyze_p = sub.add_parser("analyze-master-curve")
    mc_analyze_p.add_argument("output_dir")
    mc_analyze_p.add_argument("--config")
    mc_analyze_p.add_argument("--summary", default="dma_summary.csv")
    mc_analyze_p.add_argument("--signal", default="sxy_vir_MPa")
    mc_analyze_p.add_argument("--parser", default="lockin")

    compare_p = sub.add_parser("compare-estimators", help="Analysis-only, matched-window Fourier/OLS/parser comparison")
    compare_p.add_argument("raw_file")
    compare_p.add_argument("--freq-ghz", type=float, required=True)
    compare_p.add_argument("--signal", required=True)
    compare_p.add_argument("--strain-signal", default="strain_box")
    compare_p.add_argument("--discard-cycles", type=int, default=0)
    compare_p.add_argument("--timestep-fs", type=float)
    compare_p.add_argument("--reference-length-a", type=float, help="Explicit axial length reference for --strain-signal lz")
    compare_p.add_argument("--incomplete", choices=("reject", "trim"), default="reject")
    compare_p.add_argument("--output-dir", required=True)

    args = parser.parse_args(argv)
    if args.cmd == "compare-estimators":
        from .analysis.fourier import compare_estimators
        # A separate destination protects historical result tables.
        destination = Path(args.output_dir)
        if destination.exists():
            raise FileExistsError(f"Comparison output must be a new directory: {destination}")
        paired, harmonics = compare_estimators(args.raw_file, args.freq_ghz, args.signal,
            args.discard_cycles, args.strain_signal, args.timestep_fs, args.incomplete, args.reference_length_a)
        destination.mkdir(parents=True)
        paired.to_csv(destination / "estimator_comparison.csv", index=False)
        harmonics.to_csv(destination / "harmonic_comparison.csv", index=False)
        manifest = {"source_file": str(Path(args.raw_file).resolve()),
            "source_sha256": hashlib.sha256(Path(args.raw_file).read_bytes()).hexdigest(),
            "frequency_ghz": args.freq_ghz, "signal": args.signal, "strain_source": args.strain_signal,
            "reference_length_A": args.reference_length_a, "timestep_fs": args.timestep_fs,
            "discard_cycles": args.discard_cycles, "incomplete_policy": args.incomplete,
            "normalizations": list(paired.normalization.unique()),
            "window_policy": "shared half-open uniform complete cycles",
            "module_path": str(Path(__file__).resolve()),
            "fourier_source_sha256": hashlib.sha256((Path(__file__).parent / "analysis/fourier.py").read_bytes()).hexdigest()}
        (destination / "comparison_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(destination / "estimator_comparison.csv")
        return 0
    if args.cmd == "run":
        spec = DMAx2WorkflowSpec.from_file(args.config)
        result = DMAx2Workflow(spec).run_all(args.workdir, dry_run=args.dry_run, synthetic=args.synthetic)
        print(json.dumps(result.to_dict(), indent=2))
        return 0
    if args.cmd == "parse":
        out = Path(args.output_dir)
        spec = DMAx2WorkflowSpec.from_file(args.config) if args.config else DMAx2WorkflowSpec()
        summary = parse_dma_directory(out, args.freq_ghz, args.discard_cycles)
        if "frequency_ghz" in summary:
            summary["frequency_ghz"] = summary["frequency_ghz"].fillna(args.freq_ghz)
        else:
            summary["frequency_ghz"] = args.freq_ghz
        if "temperature_K" in summary:
            summary["temperature_K"] = summary["temperature_K"].fillna(spec.production.temperature_k)
        else:
            summary["temperature_K"] = spec.production.temperature_k
        path = out / "dma_summary.csv"
        summary.to_csv(path, index=False)
        write_dma_plots(summary, out)
        write_nemd_diagnostics_for_directory(
            out,
            args.freq_ghz,
            discard_cycles=args.discard_cycles,
            parser="lockin",
            recursive=False,
            local=False,
        )
        recs = recommend_strains(summary, spec.convergence)
        recs.to_csv(out / "dma_strain_recommendations.csv", index=False)
        stability = compare_parser_stability(summary)
        stability.to_csv(out / "dma_parser_stability.csv", index=False)
        write_strain_convergence_report(summary, out, spec.convergence, spec.production)
        mode = spec.production.modes[0] if spec.production.modes else "shear_sllod_remapv"
        write_parser_comparison_plots(summary, out, mode=mode, signal=spec.production.recommended_signal)
        selection = select_optimal_strain(
            summary,
            spec.convergence,
            mode,
            spec.production.recommended_signal,
            "lockin",
        )
        if not selection.empty:
            strain_token = f"{float(selection.iloc[0]['strain']):.3f}".replace(".", "p")
            write_legacy_pressure_phase_plot(
                out / "raw" / f"raw__legacy_axial_pressure__strain_{strain_token}.dat",
                out,
                args.freq_ghz,
                discard_cycles=args.discard_cycles,
            )
        print(path)
        return 0
    if args.cmd == "plot-nemd-diagnostics":
        paths = write_nemd_first_harmonic_diagnostics(
            args.raw_file,
            args.output_dir,
            args.freq_ghz,
            signals=tuple(args.signal) if args.signal else (
                "press_MPa",
                "szz_vir_MPa",
                "sdev_MPa",
                "sxy_vir_MPa",
                "sxy_tot_MPa",
            ),
            discard_cycles=args.discard_cycles,
            parser=args.parser,
            cycles_to_plot=args.cycles_to_plot,
            plot_subdir=args.plot_subdir,
        )
        print(json.dumps([str(path) for path in paths], indent=2))
        return 0
    if args.cmd == "plot-nemd-directory":
        index = write_nemd_diagnostics_for_directory(
            args.output_dir,
            args.freq_ghz,
            discard_cycles=args.discard_cycles,
            parser=args.parser,
            signals=tuple(args.signal) if args.signal else (
                "press_MPa",
                "szz_vir_MPa",
                "sdev_MPa",
                "sxy_vir_MPa",
                "sxy_tot_MPa",
            ),
            recursive=args.recursive,
            local=args.local,
            skip_root_raw=args.skip_root_raw,
            skip_existing=args.skip_existing,
        )
        print(json.dumps({"raw_traces": int(len(index))}, indent=2))
        return 0
    if args.cmd == "write-equilibration":
        spec = DMAx2WorkflowSpec.from_file(args.config)
        if args.data_file:
            spec.equilibration.data_file = args.data_file
        if args.output_directory:
            spec.equilibration.output_directory = args.output_directory
        path = write_equilibration_input(
            spec.polymer,
            spec.forcefield,
            spec.equilibration,
            args.workdir,
            filename=args.filename,
        )
        print(path)
        return 0
    if args.cmd == "plot-equilibration":
        stage_windows = None
        if args.config:
            stage_windows = stage_boundaries_from_config(_load_plain_config(args.config))
        artifacts = analyze_equilibration_trace(
            args.trace,
            args.output_dir,
            stage_windows=stage_windows,
            tail_ps=args.tail_ps,
            prefix=args.prefix,
        )
        print(json.dumps({k: str(v) for k, v in artifacts.items()}, indent=2))
        return 0
    if args.cmd == "write-strain-convergence":
        spec = DMAx2WorkflowSpec.from_file(args.config)
        paths = write_strain_convergence_inputs(
            spec.polymer,
            spec.forcefield,
            spec.production,
            args.workdir,
            args.equilibrated_data,
        )
        print(json.dumps([str(path) for path in paths], indent=2))
        return 0
    if args.cmd == "write-tension-strain-convergence":
        spec = DMAx2WorkflowSpec.from_file(args.config)
        paths = write_tension_strain_convergence_inputs(
            spec.polymer,
            spec.forcefield,
            spec.production,
            spec.tension_protocol,
            args.workdir,
            args.equilibrated_data,
        )
        print(json.dumps([str(path) for path in paths], indent=2))
        return 0
    if args.cmd == "write-pretensioned-tension-strain-convergence":
        spec = DMAx2WorkflowSpec.from_file(args.config)
        paths = write_pretensioned_tension_strain_convergence_inputs(
            spec.polymer,
            spec.forcefield,
            spec.production,
            spec.pretension_protocol,
            args.workdir,
            args.equilibrated_data,
        )
        print(json.dumps([str(path) for path in paths], indent=2))
        return 0
    if args.cmd == "cycle-convergence":
        spec = DMAx2WorkflowSpec.from_file(args.config)
        workdir = Path(args.workdir)
        workdir.mkdir(parents=True, exist_ok=True)
        raw_file = Path(args.raw_file)
        analysis_raw = raw_file
        if args.copy_raw:
            raw_dir = workdir / "raw"
            raw_dir.mkdir(parents=True, exist_ok=True)
            analysis_raw = raw_dir / raw_file.name
            if raw_file.resolve() != analysis_raw.resolve():
                shutil.copy2(raw_file, analysis_raw)
        artifacts = write_cycle_convergence_report(
            analysis_raw,
            workdir,
            frequency_ghz=args.freq_ghz or spec.production.frequency_ghz,
            signal=args.signal or spec.production.recommended_signal,
            convergence=spec.convergence,
            production=spec.production,
            parser=args.parser,
            discard_initial_cycles=(
                args.discard_cycles if args.discard_cycles is not None else spec.production.discard_cycles
            ),
        )
        print(json.dumps({k: str(v) for k, v in artifacts.items()}, indent=2))
        return 0
    if args.cmd == "write-error-analysis":
        spec = DMAx2WorkflowSpec.from_file(args.config)
        paths = write_error_analysis_inputs(
            spec.polymer,
            spec.forcefield,
            spec.production,
            spec.error_analysis,
            args.workdir,
            args.equilibrated_data,
        )
        print(json.dumps([str(path) for path in paths], indent=2))
        return 0
    if args.cmd == "analyze-error":
        out = Path(args.output_dir)
        spec = DMAx2WorkflowSpec.from_file(args.config) if args.config else DMAx2WorkflowSpec()
        artifacts = write_error_analysis_report(
            out / args.summary,
            out,
            signal=args.signal or spec.production.recommended_signal,
            parser=args.parser,
            mode=args.mode,
        )
        print(json.dumps({k: str(v) for k, v in artifacts.items()}, indent=2))
        return 0
    if args.cmd == "write-glass-transition":
        spec = DMAx2WorkflowSpec.from_file(args.config)
        paths = write_glass_transition_inputs(
            spec.polymer,
            spec.forcefield,
            spec.production,
            spec.glass_transition,
            args.workdir,
            args.equilibrated_data,
        )
        print(json.dumps([str(path) for path in paths], indent=2))
        return 0
    if args.cmd == "analyze-glass-transition":
        out = Path(args.output_dir)
        spec = DMAx2WorkflowSpec.from_file(args.config) if args.config else DMAx2WorkflowSpec()
        artifacts = write_glass_transition_report(
            out / args.summary,
            out,
            spec.glass_transition,
        )
        print(json.dumps({k: str(v) for k, v in artifacts.items()}, indent=2))
        return 0
    if args.cmd == "write-density-glass-transition":
        spec = DMAx2WorkflowSpec.from_file(args.config)
        paths = write_density_glass_transition_inputs(
            spec.polymer,
            spec.forcefield,
            spec.production,
            spec.glass_transition,
            args.workdir,
            args.equilibrated_data,
        )
        print(json.dumps([str(path) for path in paths], indent=2))
        return 0
    if args.cmd == "write-density-equilibrated-glass-transition":
        spec = DMAx2WorkflowSpec.from_file(args.config)
        paths = write_density_equilibrated_glass_transition_inputs(
            spec.polymer,
            spec.forcefield,
            spec.production,
            spec.glass_transition,
            args.workdir,
            args.equilibrated_data,
        )
        print(json.dumps([str(path) for path in paths], indent=2))
        return 0
    if args.cmd == "analyze-density-glass-transition":
        out = Path(args.output_dir)
        spec = DMAx2WorkflowSpec.from_file(args.config) if args.config else DMAx2WorkflowSpec()
        artifacts = write_density_glass_transition_report(
            out,
            spec.glass_transition,
            tail_fraction=args.tail_fraction,
        )
        print(json.dumps({k: str(v) for k, v in artifacts.items()}, indent=2))
        return 0
    if args.cmd == "compare-glass-transition":
        artifacts = write_glass_transition_comparison_report(
            args.dma_dir,
            args.density_dir,
            args.output_dir,
            expected_tg_k=args.expected_tg_k,
        )
        print(json.dumps({k: str(v) for k, v in artifacts.items()}, indent=2))
        return 0
    if args.cmd == "write-master-curve":
        spec = DMAx2WorkflowSpec.from_file(args.config)
        paths = write_master_curve_inputs(
            spec.polymer,
            spec.forcefield,
            spec.production,
            spec.glass_transition,
            spec.master_curve,
            args.workdir,
            args.equilibrated_data,
        )
        print(json.dumps([str(path) for path in paths], indent=2))
        return 0
    if args.cmd == "analyze-master-curve":
        out = Path(args.output_dir)
        spec = DMAx2WorkflowSpec.from_file(args.config) if args.config else DMAx2WorkflowSpec()
        artifacts = write_master_curve_analysis_report(
            out / args.summary,
            out,
            spec.master_curve,
            signal=args.signal,
            parser=args.parser,
        )
        print(json.dumps({k: str(v) for k, v in artifacts.items()}, indent=2))
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
