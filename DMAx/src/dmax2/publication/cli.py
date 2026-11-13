"""Public, explicit DMAx workflow interface; no implicit scheduler submission."""
from __future__ import annotations
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

from .build import build_packing,read_repeat,sha,validate_data
from .protocol import study,load_study,write_plan,continuation_input,paper_grid_study
from .analysis import analyze_trace,aggregate_responses,conditional_wlf_temperature


def _json_arg(value):
    if value is None:return None
    if value.lstrip().startswith(('{','[')):return json.loads(value)
    p=Path(value)
    return json.loads(p.read_text()) if p.is_file() else json.loads(value)


def main(argv=None):
    parser=argparse.ArgumentParser(prog='dmax',description='Source-bound high-frequency mechanical workflow; finite-age and finite-amplitude limits explicit.')
    sub=parser.add_subparsers(dest='command',required=True)
    init=sub.add_parser('init',help='Create a supported-chemistry or explicit-data study from minimal polymer input.')
    init.add_argument('--polymer',choices=['PVDF','PDMS','custom']);init.add_argument('--smiles');init.add_argument('--monomer-file')
    init.add_argument('--repeats',type=int);init.add_argument('--chains',type=int);init.add_argument('--packings',type=int,default=3)
    init.add_argument('--seed',type=int,default=20261113);init.add_argument('--output',type=Path,required=True)
    init.add_argument('--parameterized-data',type=Path);init.add_argument('--temperatures',type=float,nargs='+');init.add_argument('--frequencies',type=float,nargs='+')
    init.add_argument('--paper-grid',action='store_true',help='Explicit archival-style 1%,2.4ns grid; new statistical realizations, not historical checkpoint reconstruction.')
    init.add_argument('--architecture',choices=['short','intermediate','long'],default='intermediate',help='Architecture for --paper-grid only; otherwise use --repeats/--chains.')
    build=sub.add_parser('build',help='Generate independent starting packings, not equilibrated structures.')
    build.add_argument('study',type=Path);build.add_argument('--packing',type=int);build.add_argument('--packmol',default='packmol')
    plan=sub.add_parser('plan',help='Write hot/target conditioning and mechanical LAMMPS inputs without execution.')
    plan.add_argument('study',type=Path);plan.add_argument('--workdir',type=Path);plan.add_argument('--amplitude-screen',action='store_true')
    plan.add_argument('--screen-frequencies',type=float,nargs='+',default=[25.])
    run=sub.add_parser('run',help='Execute generated inputs using an explicit executable/MPI argument list.')
    run.add_argument('plan',type=Path);run.add_argument('--command-json',required=True);run.add_argument('--resume',action='store_true')
    run.add_argument('--workers',type=int,default=1);run.add_argument('--worker-logical-cpus',type=int)
    analyze=sub.add_parser('analyze',help='Measured-strain Fourier H1..H5, tensor candidates and conditional runtime diagnostics.')
    analyze.add_argument('trace',type=Path);analyze.add_argument('--frequency',type=float,required=True)
    analyze.add_argument('--operator',choices=['total','virial'],default='total');analyze.add_argument('--output',type=Path,required=True)
    analyze.add_argument('--case-json',type=Path);analyze.add_argument('--start-ps',type=float);analyze.add_argument('--discard-cycles',type=int,default=0)
    analyze.add_argument('--block-ps',type=float,default=200.);analyze.add_argument('--precision-targets');analyze.add_argument('--consistency-targets')
    aggregate=sub.add_parser('aggregate',help='Sample uncertainty across distinct independent packings, preserving each value.')
    aggregate.add_argument('responses',type=Path,nargs='+');aggregate.add_argument('--output',type=Path,required=True)
    prep=sub.add_parser('assess-preparation',help='Stage-specific density/energy/backbone diagnostics; no universal equilibrium certificate.')
    prep.add_argument('--trace',type=Path,required=True);prep.add_argument('--data',type=Path,required=True)
    prep.add_argument('--stages',type=Path,required=True);prep.add_argument('--polymer',choices=['PVDF','PDMS'],required=True)
    prep.add_argument('--geometry',type=Path);prep.add_argument('--targets');prep.add_argument('--output',type=Path,required=True)
    extension=sub.add_parser('write-continuation',help='Continue the original positive mechanical checkpoint without reseeding/resetting phase.')
    extension.add_argument('--case-json',type=Path,required=True);extension.add_argument('--checkpoint',type=Path,required=True)
    extension.add_argument('--references',type=Path,required=True);extension.add_argument('--study',type=Path,required=True)
    extension.add_argument('--additional-ps',type=float,required=True);extension.add_argument('--output',type=Path,required=True)
    wlf=sub.add_parser('wlf-marker',help='Explicit experiment-anchored marker translation; not independent Tg prediction.')
    for option in ['reference-temperature','reference-frequency','target-frequency','c1','c2']:
        wlf.add_argument('--'+option,type=float,required=True)
    wlf.add_argument('--output',type=Path)
    reproduce=sub.add_parser('reproduce-data',help='Verify and export packaged paper source tables and their provenance.')
    reproduce.add_argument('--manifest',type=Path);reproduce.add_argument('--output',type=Path,required=True)
    figures=sub.add_parser('reproduce-figures',help='Regenerate paper figures from the verified portable reduced-data bundle.')
    figures.add_argument('--data',type=Path);figures.add_argument('--output',type=Path,required=True)
    tts=sub.add_parser('tts-diagnostics',help='Compare independent storage/loss and common horizontal shifts within native coverage.')
    tts.add_argument('summary',type=Path);tts.add_argument('--reference-temperature',type=float,required=True)
    tts.add_argument('--output',type=Path,required=True)
    plotting=sub.add_parser('plot',help='Plot already reduced harmonic and cumulative/temporal diagnostics.')
    plotting.add_argument('analysis',type=Path)
    plotting_summary=sub.add_parser('plot-summary',help='Plot independent-packing means and sample SD.')
    plotting_summary.add_argument('summary',type=Path);plotting_summary.add_argument('--output',type=Path,required=True)
    sampling=sub.add_parser('sampling-check',help='Compare one dense trajectory with every phase of integer-stride downsampling.')
    sampling.add_argument('trace',type=Path);sampling.add_argument('--frequency',type=float,required=True)
    sampling.add_argument('--strides',type=int,nargs='+',required=True);sampling.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv)
    if args.command=='init':
        if args.paper_grid:
            if args.smiles or args.monomer_file or args.repeats or args.chains or args.parameterized_data or args.temperatures or args.frequencies:
                parser.error('--paper-grid uses the recorded supported architecture; use a regular init study for altered chemistry/counts.')
            c=paper_grid_study(args.polymer,args.architecture,args.seed)
            c['packings']=args.packings
        else:
            if args.architecture!='intermediate':parser.error('--architecture selects a paper-grid preset; otherwise use --repeats and --chains.')
            c=study(args.polymer,args.smiles,args.monomer_file,args.repeats,args.chains,args.packings,args.seed)
        if args.parameterized_data:c['polymer']['parameterized_data']=str(args.parameterized_data.resolve())
        if c['polymer']['name']=='custom' and not args.parameterized_data:
            parser.error('SMILES identifies chemistry, not OPLS parameters. Supply validated --parameterized-data for other polymers.')
        if args.temperatures:c['measurement']['temperatures_K']=args.temperatures
        if args.frequencies:c['measurement']['frequencies_GHz']=args.frequencies
        if args.output.exists():raise FileExistsError(args.output)
        args.output.mkdir(parents=True);p=args.output/'study.json';p.write_text(json.dumps(c,indent=2)+'\n');print(p)
    elif args.command=='build':
        c=load_study(args.study);packings=[args.packing] if args.packing else range(1,c['packings']+1)
        for packing in packings:
            output=args.study.resolve().parent/'structures'/f'packing_{packing:02d}'
            result=build_packing(c,output,packing,args.packmol);print(json.dumps(result,indent=2))
    elif args.command=='plan':
        c=load_study(args.study)
        if args.amplitude_screen:
            c['measurement']['strain_amplitudes']=c['measurement']['amplitude_screen']
            c['measurement']['frequencies_GHz']=args.screen_frequencies
        print(write_plan(c,args.workdir or args.study.resolve().parent/'runs'))
    elif args.command=='run':
        from .execution import execute_plan,execute_parallel_plan
        if args.workers>1:
            if args.worker_logical_cpus is None:parser.error('Declare --worker-logical-cpus to bound concurrent MPI workers.')
            result=execute_parallel_plan(args.plan,_json_arg(args.command_json),args.workers,args.worker_logical_cpus,args.resume)
        else:result=execute_plan(args.plan,_json_arg(args.command_json),args.resume)
        print(json.dumps(result,indent=2))
    elif args.command=='analyze':
        metadata=json.loads(args.case_json.read_text()) if args.case_json else {}
        result=analyze_trace(args.trace,args.frequency,args.output,args.operator,args.start_ps,args.discard_cycles,
                             args.block_ps,_json_arg(args.precision_targets),_json_arg(args.consistency_targets),metadata)
        print(json.dumps(result,indent=2))
    elif args.command=='aggregate':print(aggregate_responses(args.responses,args.output))
    elif args.command=='assess-preparation':
        from .preparation import assess_preparation
        print(json.dumps(assess_preparation(args.trace,args.data,json.loads(args.stages.read_text()),args.output,
                                          args.polymer,args.geometry,_json_arg(args.targets)),indent=2))
    elif args.command=='write-continuation':
        previous=json.loads(args.case_json.read_text());c=load_study(args.study)
        refs=args.references.read_text().splitlines()
        if len(refs)!=2 or refs[0].split()!=['RX_A','RY_A','RZ_A','RXY_A']:raise ValueError('Original measured reference_state.dat is required.')
        previous['reference_lengths_A']=dict(zip(['RX','RY','RZ','RXY'],map(float,refs[1].split())))
        if args.output.exists():raise FileExistsError(args.output)
        args.output.mkdir(parents=True)
        lines=continuation_input(previous,args.checkpoint.resolve(),c,args.additional_ps)
        (args.output/'in.lammps').write_text('\n'.join(lines)+'\n')
        updated=dict(previous,start_step=previous['stop_step'],stop_step=previous['stop_step']+round(args.additional_ps*1000/c['measurement']['timestep_fs']),
                     predecessor_case_sha256=sha(args.case_json),duration_ps=previous.get('duration_ps',0)+args.additional_ps)
        (args.output/'case.json').write_text(json.dumps(updated,indent=2)+'\n')
        (args.output/'plan.json').write_text(json.dumps(dict(schema_version=1,generated_only=True,MD_executed=False,
            execution_order=[dict(directory=str(args.output.resolve()),kind='measurement_continuation',depends_on=[],expected_output='final.data',input_sha256=sha(args.output/'in.lammps'),
                                 source_path=str(args.checkpoint.resolve()),source_sha256=sha(args.checkpoint))],cases=[updated]),indent=2)+'\n')
        (args.output/'continuation_provenance.json').write_text(json.dumps(dict(predecessor=previous,
            checkpoint=str(args.checkpoint.resolve()),checkpoint_sha256=sha(args.checkpoint),references_sha256=sha(args.references),
            MD_executed=False,numerical_continuity_check_required=True,velocity_reseeded=False),indent=2)+'\n')
        print(args.output/'in.lammps')
    elif args.command=='wlf-marker':
        result=conditional_wlf_temperature(args.reference_temperature,args.reference_frequency,args.target_frequency,args.c1,args.c2)
        text=json.dumps(result,indent=2)+'\n'
        if args.output:
            if args.output.exists():raise FileExistsError(args.output)
            args.output.write_text(text)
        print(text)
    elif args.command=='reproduce-data':
        from .paper import export_data
        print(export_data(args.output,args.manifest))
    elif args.command=='reproduce-figures':
        from .paper import reproduce_figures
        print(reproduce_figures(args.output,args.data))
    elif args.command=='tts-diagnostics':
        from .tts import diagnostics
        print(diagnostics(args.summary,args.reference_temperature,args.output))
    elif args.command=='plot':
        from .plots import plot_analysis
        print(plot_analysis(args.analysis))
    elif args.command=='plot-summary':
        from .plots import plot_packing_summary
        print(plot_packing_summary(args.summary,args.output))
    elif args.command=='sampling-check':
        from .sampling import compare_sampling
        print(compare_sampling(args.trace,args.frequency,args.strides,args.output))
    return 0


if __name__=='__main__':raise SystemExit(main())
