"""Small implementation regressions, never physical polymer validation."""
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd
import pytest

from dmax2.publication.build import read_repeat,build_pvdf,validate_data,sha
from dmax2.publication.protocol import study,measurement_input,continuation_input,duration_for_cycles,write_plan
from dmax2.publication.analysis import analyze_trace,temporal_diagnostics,tensor_candidates,acoustic_candidates,aggregate_responses,conditional_wlf_temperature
from dmax2.publication.execution import execute_plan,execute_parallel_plan
from dmax2.publication.tts import fit_shifts


FIXTURES=Path(__file__).parent/'fixtures'


def test_smiles_minimal_and_missing_forcefield():
    assert read_repeat('[*:1]CC(F)(F)[*:2]')['polymer']=='PVDF'
    assert read_repeat('[*:1]O[Si](C)(C)[*:2]')['polymer']=='PDMS'
    try:
        import rdkit
    except ImportError:
        with pytest.raises(ImportError):read_repeat('[*:1]CO[*:2]')
    else:
        assert read_repeat('[*:1]CO[*:2]')['force_field_available'] is False
        assert read_repeat('[*:1]C(F)C(F)[*:2]')['polymer']=='custom'
        assert read_repeat('[*:1]C(F)=C(F)[*:2]')['polymer']=='custom'
        with pytest.raises(ValueError):read_repeat('CC')


@pytest.mark.parametrize('polymer,repeats,atoms',[('PVDF',3,40),('PDMS',2,94)])
def test_real_exported_typed_topology_caps_neutrality(polymer,repeats,atoms):
    result=validate_data(FIXTURES/f'tiny_{polymer.lower()}.data',polymer,repeats,2)
    assert result['atoms']==atoms and result['chains']==2
    assert max(abs(x) for x in result['charge_per_chain_e'])<1e-6
    assert not result['physical_equilibrium_established']


def test_actual_pvdf_build_and_reproducibility(tmp_path):
    paths=[]
    for name in ['a','b']:
        p=tmp_path/name;p.mkdir();build_pvdf(p,3,2,123);paths.append(p/'initial.data')
    assert sha(paths[0])==sha(paths[1])
    assert validate_data(paths[0],'PVDF',3,2)['bonds']==38


def test_fixed_lateral_measurement_and_phase_preserving_continuation():
    with pytest.raises(ValueError):study('PVDF',repeats=0)
    c=study('PVDF',packings=1);text='\n'.join(measurement_input(FIXTURES/'tiny_pvdf.data',c,300,25,.02,123))
    assert 'compute DP all pressure DT' in text and 'compute DV all pressure NULL virial' in text
    assert text.index('fix LOG')<text.index('fix DEF') and 'fix LOG all print 4' in text
    assert 'run 1200000 start 0 stop 1200000' in text
    old=dict(stop_step=1200000,temperature_K=300,frequency_GHz=25,strain_amplitude=.02,velocity_seed=123,
             reference_lengths_A=dict(RX=40.,RY=41.,RZ=42.,RXY=0.))
    tail='\n'.join(continuation_input(old,'old.restart',c,400))
    assert 'read_restart "old.restart"' in tail and 'reset_timestep' not in tail and 'velocity all create' not in tail
    assert 'variable RZ equal 42' in tail and 'run 800000 start 0 stop 2000000' in tail
    assert duration_for_cycles(10)==(15,1500.) and duration_for_cycles(500)==(200,400.)
    with pytest.raises(ValueError):measurement_input('x.data',c,300,7.15,.01,123)


def test_plan_is_executable_dag_and_preserves_candidate_scope(tmp_path):
    c=study('PVDF',repeats=3,chains=2,packings=1);c['measurement'].update(temperatures_K=[300],frequencies_GHz=[25])
    p=tmp_path/'structures'/'packing_01';p.mkdir(parents=True)
    shutil.copy2(FIXTURES/'tiny_pvdf.data',p/'initial.data')
    (p/'build_manifest.json').write_text(json.dumps(dict(data_sha256=sha(p/'initial.data'))))
    value=json.loads(write_plan(c,tmp_path/'runs').read_text());jobs=value['execution_order']
    assert len(jobs)==3 and jobs[1]['depends_on']==[jobs[0]['directory']] and jobs[2]['depends_on']==[jobs[1]['directory']]
    assert value['cases'][0]['preparation_age_ps']==3350
    assert value['cases'][0]['strain_amplitude']==.02 and not value['cases'][0]['linearity_established']


def synthetic_trace(path,cycles=40,amplitude=.01,phase=.31,noise=0.):
    spp=40;t=np.arange(cycles*spp)*1.;theta=2*np.pi*t/40+phase
    e=amplitude*np.sin(theta);rng=np.random.default_rng(77)
    stress=amplitude*(2000*np.sin(theta)+400*np.cos(theta))+2*np.sin(2*theta)+noise*rng.normal(size=len(t))
    pd.DataFrame(dict(t_ps=t,eps_z=e,eps_x=0.,eps_y=0.,tot_zz=stress,vir_zz=stress-300*e,
                      tot_xx=stress-2*amplitude*(300*np.sin(theta)+100*np.cos(theta)),
                      tot_yy=stress-2*amplitude*(300*np.sin(theta)+100*np.cos(theta)))).to_csv(path,sep=' ',index=False)


def test_measured_strain_phase_harmonics_and_cumulative_tables(tmp_path):
    source=tmp_path/'trace.dat';synthetic_trace(source)
    fit=analyze_trace(source,25,tmp_path/'analysis')
    assert fit['storage_modulus_MPa']==pytest.approx(2000,abs=1e-8)
    assert fit['loss_modulus_MPa']==pytest.approx(400,abs=1e-8)
    h=pd.read_csv(tmp_path/'analysis'/'harmonics.csv')
    assert h.loc[h.harmonic==2,'harmonic_over_first'].iloc[0]==pytest.approx(2/np.hypot(20,4))
    assert not fit['physical_linearity_established']
    t=fit['isotropic_equivalent_candidates'];assert t['G']['real']==pytest.approx(300)
    assert len(pd.read_csv(tmp_path/'analysis'/'cumulative_cycles.csv'))==40


def test_temporal_precision_does_not_hide_aging_or_implicit_tolerances():
    c=np.tile([.01,0,20,4.],(40,1));targets=dict(storage_MPa=.05,loss_MPa=.05,tan_delta=.05)
    a=temporal_diagnostics(c,25,200,draws=100,precision_targets=targets,consistency_targets=targets)
    assert a.iloc[-1].accepted and not a.iloc[-1].aging_disproved
    c[20:,2:]*=1.3
    a=temporal_diagnostics(c,25,200,draws=100,precision_targets=targets,consistency_targets=targets)
    assert not a.iloc[-1].accepted and not a.iloc[-1].late_window_consistency_targets_met
    assert not temporal_diagnostics(c,25,200,draws=100).iloc[-1].accepted
    with pytest.raises(ValueError):temporal_diagnostics(c,25,precision_targets={})


def test_tensor_and_acoustic_identities_and_wlf_explicit_anchor():
    K=1500+200j;G=400+100j;M=K+4*G/3
    t=tensor_candidates(M,2*G)
    assert t['E']==pytest.approx(2*G*(1+t['nu']))
    ac=acoustic_candidates(2000+400j,1.5,25)
    k=2*np.pi*25e9*np.sqrt(1500/(2e9+4e8j))
    assert ac['phase_speed_m_s']==pytest.approx(2*np.pi*25e9/k.real)
    assert ac['attenuation_per_m']==pytest.approx(-k.imag)
    w=conditional_wlf_temperature(238.15,.01,25e9,17.44,51.6)
    assert w['temperature_K']==pytest.approx(365.029431333,abs=1e-6)
    assert not w['independent_Tg_prediction']


def test_independent_packings_reject_velocity_seed_clones(tmp_path):
    paths=[]
    for p in range(2):
        path=tmp_path/f'{p}.json';value=dict(metadata=dict(polymer='PVDF',repeat_units=100,chains=20,packing=p,
             temperature_K=300,strain_amplitude=.01,initial_packing_sha256='same'),frequency_GHz=25,operator='total',
             storage_modulus_MPa=2000,loss_modulus_MPa=400,tan_delta=.2)
        path.write_text(json.dumps(value));paths.append(path)
    with pytest.raises(ValueError,match='velocity'):aggregate_responses(paths,tmp_path/'aggregation')


def test_tts_recovers_shared_synthetic_horizontal_shift():
    f=np.geomspace(1,1000,25)
    def curve(f):
        x=f/20
        return dict(storage_MPa=100+2000*x*x/(1+x*x),loss_MPa=2000*x/(1+x*x))
    reference=pd.DataFrame(dict(frequency_GHz=f,**curve(f)))
    moving=pd.DataFrame(dict(frequency_GHz=f,**curve(f*2)))
    s=fit_shifts(reference,moving)
    assert np.allclose(s.log10_a,np.log10(2),atol=.01)
    assert not s.scientific_TTS_accepted.any()


@pytest.mark.parametrize('parallel',[False,True])
def test_actual_execution_entrypoint_with_fake_md_no_physical_claim(tmp_path,monkeypatch,parallel):
    monkeypatch.setenv('SLURM_JOB_ID','unit-test-no-MD');monkeypatch.setenv('SLURM_CPUS_ON_NODE','4')
    exe=tmp_path/'fake.py';exe.write_text('from pathlib import Path\nimport shutil\nshutil.copy2('+repr(str(FIXTURES/'tiny_pvdf.data'))+', "final.data")\nPath("driver.log").write_text("LAMMPS (synthetic implementation fixture; no MD)\\n")\n')
    jobs=[]
    for i in range(2):
        p=tmp_path/f'job{i}';p.mkdir();(p/'in.lammps').write_text('# no MD\n')
        jobs.append(dict(directory=str(p),depends_on=[] if i==0 else [str(tmp_path/'job0')],expected_output='final.data',input_sha256=sha(p/'in.lammps')))
    plan=tmp_path/'plan.json';plan.write_text(json.dumps(dict(execution_order=jobs)))
    if parallel:execute_parallel_plan(plan,[sys.executable,str(exe)],2,1)
    else:execute_plan(plan,[sys.executable,str(exe)])
    for job in jobs:
        receipt=json.loads((Path(job['directory'])/'execution.json').read_text())
        assert receipt['status']=='numerically_completed' and not receipt['scientific_validation']
    with pytest.raises(FileExistsError):execute_plan(plan,[sys.executable,str(exe)])
