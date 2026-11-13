import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd
import pytest

from dmax2.publication.build import sha
from dmax2.publication.paper import export_data
from dmax2.publication.sampling import compare_sampling
from dmax2.publication.preparation import assess_preparation
from dmax2.publication.execution import execute_plan
from dmax2.publication.cli import main
from dmax2.publication.protocol import paper_grid_study


FIXTURE=Path(__file__).parent/'fixtures/tiny_pvdf.data'


def test_minimal_smiles_cli_produces_honest_defaults(tmp_path):
    output=tmp_path/'pvdf';main(['init','--smiles','[*:1]CC(F)(F)[*:2]','--output',str(output)])
    c=json.loads((output/'study.json').read_text())
    assert c['polymer']['repeat_units']==100 and c['polymer']['chains']==20
    assert c['measurement']['candidate_amplitude']==.02
    assert not c['measurement']['physical_linearity_established']
    assert c['measurement']['runtime']['precision_targets'] is None


def test_archival_style_presets_match_actual_packaged_grid_and_do_not_retag_strain():
    d=pd.read_csv(Path(__file__).parents[1]/'src/dmax2/publication/paper_data/sources/papergrid_complex432.csv')
    for polymer in ['PVDF','PDMS']:
        for architecture in ['short','intermediate','long']:
            c=paper_grid_study(polymer,architecture)
            internal=c['polymer']['repeat_units']+(-1 if polymer=='PVDF' else 1)
            system=f'{polymer}_'+('n' if polymer=='PVDF' else 'u')+str(internal)+'_c'+str(c['polymer']['chains'])
            g=d.loc[d.system==system]
            assert len(g)==72 and sorted(g['T'].unique())==c['measurement']['temperatures_K']
            assert sorted(g.frequency_GHz.unique())==c['measurement']['frequencies_GHz']
            assert c['measurement']['strain_amplitudes']==[.01]
            assert c['measurement']['runtime']['policy']=='fixed_duration' and c['measurement']['runtime']['maximum_ps']==2400


def test_sampling_same_history_can_alias_unresolved_high_frequency_noise(tmp_path):
    theta=2*np.pi*np.arange(4000)/400;t=np.arange(4000)*.1
    e=.01*np.sin(theta);s=20*np.sin(theta)+4*np.cos(theta)+2*np.sin(19*theta)
    path=tmp_path/'trace.dat';pd.DataFrame(dict(t_ps=t,eps_z=e,tot_zz=s,vir_zz=s-300*e)).to_csv(path,sep=' ',index=False)
    compare_sampling(path,25,[20],tmp_path/'sampling')
    result=pd.read_csv(tmp_path/'sampling/all_sampling_offsets.csv')
    dense=result.loc[result.stride==1];sparse=result.loc[(result.stride==20)&(result.offset==0)]
    assert np.max(abs(dense.storage_MPa_offset))<1e-7
    assert np.allclose(sparse.storage_MPa_offset,-200,atol=1e-6)
    assert (tmp_path/'sampling/paired_sampling_offsets.pdf').exists()


def test_preparation_fixed_density_is_not_equilibrium_and_missing_samples_rejected(tmp_path):
    t=np.arange(0,1000.5,.5);d=pd.DataFrame(dict(step=t*2000,time_ps=t,temperature=300,pressure=1,density=1.5,pe=-100,ke=100))
    path=tmp_path/'thermal.dat';d.to_csv(path,sep=' ',index=False)
    stages=[dict(stage='hold',ensemble='NVT',start_ps=0,stop_ps=1000,fixed_temperature=True,target_temperature_K=300)]
    result=assess_preparation(path,FIXTURE,stages,tmp_path/'prep','PVDF')
    assert not result['physical_equilibrium_established'] and result['stages'][0]['NVT_density_imposed']
    d=d.drop(42);d.to_csv(path,sep=' ',index=False)
    with pytest.raises(ValueError,match='missing'):assess_preparation(path,FIXTURE,stages,tmp_path/'bad','PVDF')


def test_reduced_evidence_export_checksum_and_paths(tmp_path):
    source=tmp_path/'source';source.mkdir();data=source/'values.csv';data.write_text('x,y\n1,2\n')
    manifest=source/'manifest.json';manifest.write_text(json.dumps(dict(files=[dict(path='values.csv',sha256=sha(data))])))
    export_data(tmp_path/'export',manifest)
    assert (tmp_path/'export/values.csv').read_text()==data.read_text()
    data.write_text('x,y\n3,4\n')
    with pytest.raises(ValueError,match='Changed'):export_data(tmp_path/'changed',manifest)


def test_execution_launch_failure_is_preserved(tmp_path,monkeypatch):
    monkeypatch.setenv('SLURM_JOB_ID','fake-allocation-no-MD')
    (tmp_path/'in.lammps').write_text('# test no MD\n')
    plan=tmp_path/'plan.json';plan.write_text(json.dumps(dict(execution_order=[dict(directory=str(tmp_path),expected_output='final.data',input_sha256=sha(tmp_path/'in.lammps'))])))
    with pytest.raises(RuntimeError,match='Launcher failed'):execute_plan(plan,['/nonexistent/dmax-test-lammps'])
    receipt=json.loads((tmp_path/'execution.json').read_text())
    assert receipt['status']=='failed' and receipt['returncode'] is None
    with pytest.raises(FileExistsError):execute_plan(plan,['/nonexistent/dmax-test-lammps'])


def test_changed_initial_data_rejected_before_launch_even_when_input_text_unchanged(tmp_path,monkeypatch):
    monkeypatch.setenv('SLURM_JOB_ID','fake-allocation-no-MD')
    data=tmp_path/'initial.data';shutil.copy2(FIXTURE,data)
    (tmp_path/'in.lammps').write_text('read_data "initial.data"\n')
    job=dict(directory=str(tmp_path),expected_output='final.data',input_sha256=sha(tmp_path/'in.lammps'),
             source_path=str(data),source_sha256=sha(data))
    plan=tmp_path/'plan.json';plan.write_text(json.dumps(dict(execution_order=[job])))
    data.write_text(data.read_text()+'\n# mutation after plan generation\n')
    with pytest.raises(ValueError,match='source checksum changed'):execute_plan(plan,['/nonexistent/never-launched'])
    assert not (tmp_path/'execution.json').exists() and not (tmp_path/'launcher.log').exists()


@pytest.mark.parametrize('mutate',[False,True])
def test_dependency_source_bound_to_positive_parent_output_receipt(tmp_path,monkeypatch,mutate):
    monkeypatch.setenv('SLURM_JOB_ID','fake-allocation-no-MD')
    exe=tmp_path/'fake.py';exe.write_text('from pathlib import Path\nimport shutil\nshutil.copy2('+repr(str(FIXTURE))+',"final.data")\nPath("driver.log").write_text("LAMMPS (synthetic; no MD)\\n")\n')
    parent=tmp_path/'parent';parent.mkdir();(parent/'in.lammps').write_text('# implementation fixture no MD\n')
    parent_job=dict(directory=str(parent),expected_output='final.data',input_sha256=sha(parent/'in.lammps'))
    p=tmp_path/'parent_plan.json';p.write_text(json.dumps(dict(execution_order=[parent_job])))
    execute_plan(p,[sys.executable,str(exe)])
    child=tmp_path/'child';child.mkdir();(child/'in.lammps').write_text('read_data "'+str(parent/'final.data')+'"\n')
    child_job=dict(directory=str(child),expected_output='final.data',input_sha256=sha(child/'in.lammps'),
                   source_path=str(parent/'final.data'),source_dependency=str(parent))
    p=tmp_path/'child_plan.json';p.write_text(json.dumps(dict(execution_order=[child_job])))
    if mutate:
        (parent/'final.data').write_text((parent/'final.data').read_text()+'\n# post-receipt mutation\n')
        with pytest.raises(ValueError,match='does not match'):execute_plan(p,[sys.executable,str(exe)])
        assert not (child/'execution.json').exists()
    else:
        execute_plan(p,[sys.executable,str(exe)])
        receipt=json.loads((child/'execution.json').read_text())
        assert receipt['consumed_source']['sha256']==sha(parent/'final.data')
        assert receipt['consumed_source']['binding']=='successful_dependency_output' and receipt['source_binding_validated']
