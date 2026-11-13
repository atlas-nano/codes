"""Portable input generation for the documented finite-age axial workflow.

Inputs are plans, not completed or equilibrated states. Production keeps the
actual measured box strain, total/virial tensors, dense synchronous acquisition,
and the original deformation reference. No low-frequency extrapolation runs
automatically. The user executes inputs in an appropriate compute allocation.
"""
from __future__ import annotations
from copy import deepcopy
import json
import math
from pathlib import Path
import shlex

from .build import SMILES, data_sections, read_repeat, sha


def study(polymer=None,smiles=None,monomer_file=None,repeats=None,chains=None,packings=3,seed=20261113):
    repeat=read_repeat(smiles,monomer_file,polymer)
    name=repeat['polymer']
    defaults={'PVDF':(100,20,450.,10.,.65),'PDMS':(30,40,400.,11.,.1)}
    nominal,nchains,hot,cutoff,rho=defaults.get(name,(20,20,450.,10.,.1))
    actual_repeats=nominal if repeats is None else int(repeats)
    actual_chains=nchains if chains is None else int(chains)
    if min(actual_repeats,actual_chains,int(packings),int(seed))<1:raise ValueError('Positive architecture, packing count and seed required.')
    return dict(schema_version=1,polymer=dict(name=name,repeat_smiles=repeat['repeat_smiles'],
        repeat_units=actual_repeats,chains=actual_chains,initial_density_g_cm3=rho),
        packings=int(packings),seed=int(seed),
        forcefield=dict(name='paper_OPLS_AA' if name in SMILES else 'explicit_user_parameterized_data',
                       cutoff_A=cutoff,pppm_accuracy=1e-5,special_bonds='lj/coul 0 0 0.5',mixing='geometric'),
        preparation=dict(timestep_fs=.5,hot_temperature_K=hot,soft_ps=75.,limited_ps=25.,
            hot_NPT_ps=1000.,hot_NVT_ps=200.,ramp_ps=150.,NPT_stage_ps=[750.,750.],NVT_hold_ps=500.,
            pressure_atm=1.,thermostat_damping_fs=100.,barostat_damping_fs=5000.,
            neighbor_check='yes',scalar_interval_ps=.5,coordinate_interval_ps=10.,
            policy='fixed_duration_with_diagnostics',equilibrium_established=False,
            adaptive_extension_ps=500.,maximum_adaptive_extensions=4,stopping_targets=None),
        measurement=dict(temperatures_K=[300.,400.],frequencies_GHz=[10.,25.,50.,100.,250.,500.],
            strain_amplitudes=[.02],candidate_amplitude=.02,amplitude_qualification_required=True,
            amplitude_screen=[.0025,.005,.01,.015,.02,.04,.06],timestep_fs=.5,logging_fs=2.,
            thermostat_damping_fs=100.,thermostat_chain=3,mode='axial_fixed',stress_operator='total',
            harmonics=[1,2,3,4,5],startup_discard_cycles=0,
            runtime=dict(policy='bounded_adaptive',initial_minimum_cycles=15,initial_minimum_ps=400.,
                         extension_ps=200.,maximum_ps=2400.,block_ps=200.,minimum_blocks=6,
                         precision_targets=None,temporal_consistency_targets=None),
            physical_linearity_established=False),
        scientific_scope=dict(default_architecture='benchmarked starting point; not system-size or entanglement convergence',
            default_amplitude='finite-amplitude candidate; not universal linear-response certification',
            runtime='conditional precision and late-window consistency; not equilibrium or absence of aging',
            TTS='diagnostic and opt-in; no automatic Tg prediction'))


def load_study(path):
    path=Path(path);c=json.loads(path.read_text())
    if c.get('schema_version')!=1:raise ValueError('Unsupported study schema.')
    if c['polymer']['name'] in SMILES:
        read_repeat(c['polymer']['repeat_smiles'],polymer=c['polymer']['name'])
    for key in ['repeat_units','chains']:
        if not isinstance(c['polymer'][key],int) or c['polymer'][key]<1:raise ValueError('Positive integer architecture required.')
    if c['packings']<1:raise ValueError('Positive packing count required.')
    m=c['measurement'];p=c['preparation']
    if m['mode']!='axial_fixed' or m['harmonics']!=[1,2,3,4,5]:
        raise ValueError('The publication CLI implements fixed-lateral axial H1–H5; other archived modes use the explicit legacy API.')
    if p.get('neighbor_check','yes') not in ('yes','no'):raise ValueError('Explicit yes/no neighbor policy required.')
    if m['runtime']['policy'] not in ('bounded_adaptive','fixed_duration'):raise ValueError('Unknown finite-runtime policy.')
    if not math.isclose(m['timestep_fs'],p['timestep_fs'],rel_tol=0,abs_tol=1e-12):
        raise ValueError('This generator requires one consistent preparation/measurement timestep.')
    if any(not math.isfinite(x) or x<=0 for x in m['temperatures_K']+m['frequencies_GHz']):
        raise ValueError('Positive finite target temperatures and frequencies required.')
    return c


def paper_grid_study(polymer,architecture='intermediate',seed=20261113):
    """Statistical replay preset; source tables preserve the actual measured grid.

    New coordinates do not recreate archived binary checkpoints or exceptional
    neighbor-control ancestry. Do not relabel archival 1% data as the 2% candidate.
    """
    choices={'PVDF':{'short':(10,200),'intermediate':(100,20),'long':(400,5)},
             'PDMS':{'short':(10,100),'intermediate':(30,40),'long':(100,12)}}
    if polymer not in choices or architecture not in choices[polymer]:raise ValueError('Known paper polymer/architecture required.')
    repeats,chains=choices[polymer][architecture];c=study(polymer,repeats=repeats,chains=chains,seed=seed)
    c['measurement'].update(temperatures_K=[338.15,377.,400.,453.15] if polymer=='PVDF' else [175.,225.,250.,345.],
                            strain_amplitudes=[.01],candidate_amplitude=.01,recommended_analysis_start_ps=400.)
    c['measurement']['runtime'].update(policy='fixed_duration',maximum_ps=2400.)
    c['reproduction']=dict(route='new independently generated MD realization of the paper protocol; not bitwise checkpoint replay',
        measured_source='paper_data/sources/papergrid_complex432.csv',
        source_table_sha256=sha(Path(__file__).parent/'paper_data/sources/papergrid_complex432.csv'),
        original_input_amplitude=.01,retained_start_ps=400.,retained_stop_ps=2400.,
        exceptions='Exact archived seeds/checkpoints and any case-specific neighbor-rebuild ancestry require the external input archive.')
    return c


def _steps(ps,dt):
    n=ps*1000/dt
    if ps<=0 or dt<=0 or not math.isclose(n,round(n),abs_tol=1e-7):raise ValueError('Positive integral MD step count required.')
    return round(n)


def header(source,c,read_restart=False):
    ff=c['forcefield'];dt=c['preparation']['timestep_fs']
    if any(ch in str(source) for ch in '\n\r"$'):raise ValueError('LAMMPS source paths cannot contain quotes, newlines or variable substitutions.')
    return ['units real','atom_style full','boundary p p p',
        f'pair_style lj/cut/coul/long {ff["cutoff_A"]:g}','bond_style harmonic','angle_style harmonic',
        'dihedral_style opls','improper_style none',f'special_bonds {ff["special_bonds"]}',
        f'pair_modify mix {ff["mixing"]}',('read_restart ' if read_restart else 'read_data ')+f'"{source}"',
        f'kspace_style pppm {ff["pppm_accuracy"]:g}',f'timestep {dt:g}',
        'neighbor 2.0 multi','neigh_modify every 1 delay 0 check yes','comm_modify vel yes',
        'thermo 10000','thermo_style custom step time temp press density lx ly lz pxx pyy pzz pe ke etotal',
        'thermo_modify format float %.16g flush yes']


def _thermal_logging(c,filename='thermal.dat'):
    p=c['preparation'];every=_steps(p['scalar_interval_ps'],p['timestep_fs']);dump=_steps(p['coordinate_interval_ps'],p['timestep_fs'])
    return ['variable PS equal step','variable PT equal time/1000','variable TT equal temp',
            'variable PP equal press','variable RR equal density','variable UU equal pe','variable KK equal ke',
            f'fix PREP all print {every} "${{PS}} ${{PT}} ${{TT}} ${{PP}} ${{RR}} ${{UU}} ${{KK}}" file {filename} screen no title "step time_ps temperature pressure density pe ke"',
            f'dump GEO all custom {dump} geometry.lammpstrj id mol type xu yu zu','dump_modify GEO sort id',
            'restart 200000 thermal_checkpoint.*.restart']


def hot_input(source,c,seed):
    p=c['preparation'];t=p['hot_temperature_K'];dt=p['timestep_fs'];pairs=data_sections(source)['Pair Coeffs']
    if not pairs:raise ValueError('Actual diagonal pair coefficients required for charge-preserving restoration.')
    push,limited,npt,nvt=[_steps(p[k],dt) for k in ['soft_ps','limited_ps','hot_NPT_ps','hot_NVT_ps']]
    lines=header(Path(source).resolve(),c)+['reset_timestep 0','kspace_style none','pair_style soft 6.0','pair_coeff * * 25.0',
        f'velocity all create {t:g} {seed} mom yes rot yes dist gaussian','variable SOFT equal ramp(0.0,25.0)',
        'fix RAMP all adapt 1 pair soft a * * v_SOFT','fix PUSH all nve/limit 0.02',
        f'fix BATH all langevin {t:g} {t:g} 100 {seed+37}']
    if c['polymer']['name']=='PDMS':
        d=data_sections(source);masses={int(x[0]):float(x[1]) for x in d['Masses']}
        mass=sum(masses[int(x[2])] for x in d['Atoms']);length=(mass/6.02214076e23/.65*1e24)**(1/3)
        lines.append(f'fix BOX all deform 1 x final 0 {length:.17g} y final 0 {length:.17g} z final 0 {length:.17g} remap x units box')
    lines += [f'run {push} start 0 stop {push}']
    if c['polymer']['name']=='PDMS':lines += ['unfix BOX']
    lines += ['unfix RAMP','unfix BATH','unfix PUSH',f'pair_style lj/cut/coul/long {c["forcefield"]["cutoff_A"]:g}',
              f'pair_modify mix {c["forcefield"]["mixing"]}']
    lines += [f'pair_coeff {int(v[0])} {int(v[0])} {float(v[1]):.17g} {float(v[2]):.17g}' for v in pairs]
    lines += [f'kspace_style pppm {c["forcefield"]["pppm_accuracy"]:g}','run 0',
        'fix PUSH all nve/limit 0.02',f'fix BATH all langevin {t:g} {t:g} 100 {seed+71}',
        f'run {limited}','unfix BATH','unfix PUSH']
    lines += _thermal_logging(c,'hot_preparation.dat')
    lines += [f'fix EQ all npt temp {t:g} {t:g} {p["thermostat_damping_fs"]:g} aniso {p["pressure_atm"]:g} {p["pressure_atm"]:g} {p["barostat_damping_fs"]:g}',f'run {npt}','unfix EQ',
              f'fix EQ all nvt temp {t:g} {t:g} {p["thermostat_damping_fs"]:g}',f'run {nvt}',
              'write_data prepared.data','write_restart prepared.restart']
    return lines


def target_input(hot_source,c,temperature):
    p=c['preparation'];dt=p['timestep_fs'];t=float(temperature);hot=p['hot_temperature_K']
    check=p.get('neighbor_check','yes')
    if check not in ('yes','no'):raise ValueError('neighbor_check must be yes or no; this is a recorded numerical policy.')
    lines=header(hot_source,c)+['reset_timestep 0']+_thermal_logging(c)
    durations=[p['ramp_ps'],*p['NPT_stage_ps'],p['NVT_hold_ps']]
    if len(durations)!=4:raise ValueError('Two explicit density stages required.')
    total=0
    for index,duration in enumerate(durations):
        temp0=hot if index==0 else t
        thermostat=(f'fix EQ all npt temp {temp0:g} {t:g} {p["thermostat_damping_fs"]:g} aniso {p["pressure_atm"]:g} {p["pressure_atm"]:g} {p["barostat_damping_fs"]:g}' if index<3
                    else f'fix EQ all nvt temp {t:g} {t:g} {p["thermostat_damping_fs"]:g}')
        steps=_steps(duration,dt);total+=steps
        lines += [f'neigh_modify every 1 delay 0 check {check if index<3 else "yes"}',thermostat,f'run {steps}',
                  f'write_data stage{index+1}.data',f'write_restart stage{index+1}.restart','unfix EQ']
    lines += ['write_data prepared.data','write_restart prepared.restart']
    return lines


def duration_for_cycles(frequency,minimum_cycles=15,minimum_ps=400.):
    if frequency<=0:raise ValueError('Positive frequency required.')
    cycles=max(int(minimum_cycles),math.ceil(minimum_ps*frequency/1000-1e-12))
    return cycles,cycles*1000/frequency


def measurement_input(source,c,temperature,frequency,strain,seed,duration_ps=None):
    m=c['measurement'];dt=m['timestep_fs'];f=float(frequency)
    if not 0<strain<1 or f<=0:raise ValueError('Positive frequency and0<engineering strain<1 required.')
    period=1e6/f;steps_per_cycle=period/dt;samples_per_cycle=period/m['logging_fs']
    if not math.isclose(steps_per_cycle,round(steps_per_cycle),abs_tol=1e-7) or not math.isclose(samples_per_cycle,round(samples_per_cycle),abs_tol=1e-7):
        raise ValueError('This writer requires commensurate cycle/step/logging clocks; choose a documented near frequency or use generalized regression explicitly.')
    if samples_per_cycle<=2*max(m['harmonics']):raise ValueError('Acquisition cannot resolve the requested harmonics.')
    ratio=m['logging_fs']/dt
    if not math.isclose(ratio,round(ratio),abs_tol=1e-9):raise ValueError('Logging must be an integral number of MD steps.')
    rt=m['runtime']
    if duration_ps is None:
        if rt['policy']=='fixed_duration':duration_ps=rt['maximum_ps']
        else:_,duration_ps=duration_for_cycles(f,rt['initial_minimum_cycles'],rt['initial_minimum_ps'])
    steps=_steps(duration_ps,dt)
    if steps%round(steps_per_cycle):raise ValueError('Production must contain complete cycles.')
    lines=header(source,c)+['reset_timestep 0',f'velocity all create {temperature:g} {seed} mom yes rot yes dist gaussian',
        f'fix TH all nvt temp {temperature:g} {temperature:g} {m["thermostat_damping_fs"]:g} tchain {m["thermostat_chain"]}',
        'variable RX equal $(lx:%.17g)','variable RY equal $(ly:%.17g)','variable RZ equal $(lz:%.17g)',
        'variable RXY equal $(xy:%.17g)','compute DT all temp','compute DP all pressure DT',
        'compute DV all pressure NULL virial','thermo_modify temp DT press DP']
    names=['step','t_ps','eps_x','eps_y','eps_z','volume_A3','density_g_cm3','temperature_K','pe','ke','thermostat_energy',
           'tot_xx','vir_xx','tot_yy','vir_yy','tot_zz','vir_zz','tot_xy','vir_xy','eps_xy']
    expressions=['step','time/1000','(lx-v_RX)/v_RX','(ly-v_RY)/v_RY','(lz-v_RZ)/v_RZ','vol','density','c_DT','pe','ke','f_TH']
    for index in (1,2,3,4):expressions += [f'-c_DP[{index}]*0.101325',f'-c_DV[{index}]*0.101325']
    expressions += ['(xy-v_RXY)/v_RY']
    for i,e in enumerate(expressions):lines.append(f'variable V{i} equal {e}')
    values=' '.join('${V'+str(i)+'}' for i in range(len(expressions)))
    # Ordering is intentional: logging occurs before end-of-step deformation.
    lines += [f'fix LOG all print {round(ratio)} "{values}" file trace.dat screen no title "'+ ' '.join(names)+'"',
        'print "RX_A RY_A RZ_A RXY_A" file reference_state.dat screen no',
        'print "$(v_RX:%.17g) $(v_RY:%.17g) $(v_RZ:%.17g) $(v_RXY:%.17g)" append reference_state.dat screen no',
        f'variable AMP equal {strain:.17g}*v_RZ',f'fix DEF all deform 1 z wiggle ${{AMP}} {period:.17g} remap x units box',
        'restart 200000 checkpoint.*.restart',f'run {steps} start 0 stop {steps}',
        'write_data final.data','write_restart final.restart']
    return lines


def continuation_input(previous_manifest,checkpoint,c,additional_ps):
    """Preserve drive phase/reference and TH/DEF restart state; never reseed."""
    old=deepcopy(previous_manifest);start=int(old['stop_step']);dt=c['measurement']['timestep_fs']
    refs=old.get('reference_lengths_A')
    if not isinstance(refs,dict) or set(refs)!={'RX','RY','RZ','RXY'}:
        raise ValueError('Continuation needs actual original reference lengths from reference_state.dat.')
    steps=_steps(additional_ps,dt);logical_stop=start+steps
    lines=measurement_input(checkpoint,c,old['temperature_K'],old['frequency_GHz'],old['strain_amplitude'],old['velocity_seed'],additional_ps)
    output=[]
    for line in lines:
        if line.startswith(('reset_timestep ','velocity all create ')):continue
        if line.startswith('read_data '):line='read_restart "'+str(checkpoint)+'"'
        for name,val in refs.items():
            if line.startswith('variable '+name+' equal '):line=f'variable {name} equal {val:.17g}'
        if line.startswith('run '):line=f'run {steps} start 0 stop {logical_stop}'
        output.append(line)
    output.insert(output.index('kspace_style pppm '+f'{c["forcefield"]["pppm_accuracy"]:g}'),f'if "$(step) != {start}" then "quit 9"')
    return output


def write_plan(c,workdir):
    workdir=Path(workdir).resolve()
    if workdir.exists():raise FileExistsError('New plan directory required.')
    workdir.mkdir(parents=True);cases=[];commands=[];execution_order=[]
    p=c['polymer'];m=c['measurement'];seed=int(c['seed'])
    for packing in range(1,c['packings']+1):
        builddir=workdir.parent/'structures'/f'packing_{packing:02d}'
        initial=builddir/'initial.data'
        if not initial.exists():raise FileNotFoundError('Build this packing first: '+str(initial))
        build_manifest=json.loads((builddir/'build_manifest.json').read_text())
        if sha(initial)!=build_manifest['data_sha256']:raise ValueError('Initial typed data changed since its build manifest: '+str(initial))
        hot=workdir/f'p{packing:02d}_hot';hot.mkdir()
        (hot/'in.lammps').write_text('\n'.join(hot_input(initial,c,seed+packing*101))+'\n')
        execution_order.append(dict(directory=str(hot),kind='hot_preparation',depends_on=[],expected_output='prepared.data',input_sha256=sha(hot/'in.lammps'),
                                    source_path=str(initial.resolve()),source_sha256=build_manifest['data_sha256']))
        commands.append(f'(cd {shlex.quote(str(hot))} && "${{LAMMPS_EXECUTABLE:-lmp}}" -in in.lammps)')
        for temperature in m['temperatures_K']:
            target=workdir/f'p{packing:02d}_T{temperature:g}_preparation';target.mkdir()
            (target/'in.lammps').write_text('\n'.join(target_input(hot/'prepared.data',c,temperature))+'\n')
            execution_order.append(dict(directory=str(target),kind='target_preparation',depends_on=[str(hot)],expected_output='prepared.data',input_sha256=sha(target/'in.lammps'),
                                        source_path=str(hot/'prepared.data'),source_dependency=str(hot)))
            t0=0.;stages=[]
            for stage_name,ensemble,duration in [('ramp','NPT',c['preparation']['ramp_ps']),
                ('density1','NPT',c['preparation']['NPT_stage_ps'][0]),('density2','NPT',c['preparation']['NPT_stage_ps'][1]),
                ('hold','NVT',c['preparation']['NVT_hold_ps'])]:
                stages.append(dict(stage=stage_name,ensemble=ensemble,start_ps=t0,stop_ps=t0+duration,
                                   fixed_temperature=(stage_name!='ramp'),target_temperature_K=temperature));t0+=duration
            (target/'stages.json').write_text(json.dumps(stages,indent=2)+'\n')
            commands.append(f'(cd {shlex.quote(str(target))} && "${{LAMMPS_EXECUTABLE:-lmp}}" -in in.lammps)')
            for frequency in m['frequencies_GHz']:
                for amplitude in m['strain_amplitudes']:
                    folder=workdir/f'p{packing:02d}_T{temperature:g}_f{frequency:g}_a{amplitude:g}';folder.mkdir()
                    v_seed=seed+packing*1009+round(temperature*10)
                    lines=measurement_input(target/'prepared.data',c,temperature,frequency,amplitude,v_seed)
                    (folder/'in.lammps').write_text('\n'.join(lines)+'\n')
                    rt=m['runtime'];duration=rt['maximum_ps'] if rt['policy']=='fixed_duration' else duration_for_cycles(frequency,rt['initial_minimum_cycles'],rt['initial_minimum_ps'])[1]
                    record=dict(case_id=folder.name,polymer=p['name'],repeat_units=p['repeat_units'],chains=p['chains'],packing=packing,
                                temperature_K=temperature,frequency_GHz=frequency,strain_amplitude=amplitude,velocity_seed=v_seed,
                                preparation_age_ps=c['preparation']['hot_NPT_ps']+c['preparation']['hot_NVT_ps']+t0,source=str(target/'prepared.data'),mode='axial_fixed',
                                source_build_manifest=str(builddir/'build_manifest.json'),start_step=0,stop_step=_steps(duration,m['timestep_fs']),
                                duration_ps=duration,logging_fs=m['logging_fs'],timestep_fs=m['timestep_fs'],
                                runtime_policy=rt['policy'],linearity_established=False,equilibrium_established=False)
                    record['initial_packing_sha256']=json.loads((builddir/'build_manifest.json').read_text())['data_sha256']
                    record['independent_packing_id']=f'{p["name"]}:n{p["repeat_units"]}:c{p["chains"]}:p{packing:02d}'
                    (folder/'case.json').write_text(json.dumps(record,indent=2)+'\n');cases.append(record)
                    execution_order.append(dict(directory=str(folder),kind='measurement',depends_on=[str(target)],expected_output='final.data',input_sha256=sha(folder/'in.lammps'),
                                                source_path=str(target/'prepared.data'),source_dependency=str(target)))
                    commands.append(f'(cd {shlex.quote(str(folder))} && "${{LAMMPS_EXECUTABLE:-lmp}}" -in in.lammps)')
    (workdir/'study.json').write_text(json.dumps(c,indent=2)+'\n')
    (workdir/'plan.json').write_text(json.dumps(dict(schema_version=1,generated_only=True,MD_executed=False,
                                                   execution_order=execution_order,cases=cases),indent=2)+'\n')
    (workdir/'run_serial.sh').write_text('#!/usr/bin/env bash\nset -euo pipefail\n# Run in an allocation; set LAMMPS_EXECUTABLE to an executable wrapper for srun/MPI.\n'+'\n'.join(commands)+'\n')
    return workdir/'plan.json'
