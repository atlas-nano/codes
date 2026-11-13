"""Explicit local/allocation execution of a previously generated plan.

No scheduler submissions, credentials or site accounts are hidden here. A plan
is executed only by the user's `dmax run` command. Failed inputs/receipts/logs
are retained and never overwritten or automatically retried.
"""
from __future__ import annotations
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor,wait,FIRST_COMPLETED

from .build import sha,validate_data


def execute_plan(plan_path,command,resume=False):
    plan_path=Path(plan_path).resolve();plan=json.loads(plan_path.read_text())
    if not isinstance(command,list) or not command or any(not isinstance(x,str) or not x for x in command):
        raise ValueError('Supply an explicit nonempty command argument list, not a shell string.')
    if socket.gethostname().startswith('login') and not os.environ.get('SLURM_JOB_ID'):
        raise ValueError('MD is not permitted on a NERSC login node. Execute dmax run in a compute allocation.')
    outcomes=[]
    for job in plan['execution_order']:
        directory=Path(job['directory']);source=directory/'in.lammps';receipt_path=directory/'execution.json'
        if sha(source)!=job['input_sha256']:raise ValueError('Generated input changed: '+str(source))
        consumed=None
        if job.get('source_path'):
            artifact=Path(job['source_path']).resolve()
            if not artifact.is_file():raise ValueError('Planned read_data/read_restart source is missing: '+str(artifact))
            consumed=dict(path=str(artifact),sha256=sha(artifact))
            if job.get('source_dependency'):
                parent=Path(job['source_dependency']).resolve();parent_receipt=parent/'execution.json'
                if not parent_receipt.is_file():raise ValueError('Source dependency lacks a successful execution receipt: '+str(parent))
                value=json.loads(parent_receipt.read_text())
                if (value.get('status')!='numerically_completed' or value.get('output_path')!=str(artifact)
                        or value.get('output_sha256')!=consumed['sha256']):
                    raise ValueError('Source dependency endpoint does not match its positive receipt: '+str(artifact))
                consumed.update(parent_receipt_sha256=sha(parent_receipt),binding='successful_dependency_output')
            elif job.get('source_sha256')==consumed['sha256']:
                consumed['binding']='fixed_source_sha256'
            else:raise ValueError('Planned source checksum changed: '+str(artifact))
        if receipt_path.exists():
            old=json.loads(receipt_path.read_text())
            if (resume and old.get('status')=='numerically_completed' and old.get('input_sha256')==sha(source)
                    and old.get('consumed_source')==consumed
                    and (directory/job['expected_output']).exists()
                    and old.get('output_sha256')==sha(directory/job['expected_output'])):
                outcomes.append(old);continue
            raise FileExistsError('Existing execution evidence must be reviewed, not overwritten: '+str(receipt_path))
        if (directory/job['expected_output']).exists():
            raise FileExistsError('Unreceipted endpoint already exists; preserve and review it before execution: '+str(directory))
        argv=[*command,'-in','in.lammps','-log','driver.log'];start=time.time()
        receipt=dict(status='executing',command=argv,input_sha256=sha(source),start_epoch=start,
                     job_id=os.environ.get('SLURM_JOB_ID'),step_id=os.environ.get('SLURM_STEP_ID'),
                     host=socket.gethostname(),scientific_validation=False,consumed_source=consumed,
                     source_binding_validated=consumed is not None)
        receipt_path.write_text(json.dumps(receipt,indent=2)+'\n')
        try:
            with (directory/'launcher.log').open('x') as log:
                run=subprocess.run(argv,cwd=directory,stdout=log,stderr=subprocess.STDOUT,check=False)
        except OSError as exc:
            receipt.update(status='failed',errors=['launcher_error:'+repr(exc)],returncode=None,
                           elapsed_seconds=time.time()-start,physical_equilibrium_established=False,linearity_established=False)
            receipt_path.write_text(json.dumps(receipt,indent=2)+'\n')
            raise RuntimeError('Launcher failed; failure receipt preserved: '+str(receipt_path)) from exc
        error=[];output=directory/job['expected_output'];driver=directory/'driver.log'
        logtext=driver.read_text() if driver.exists() else ''
        if run.returncode:error.append('nonzero_process_returncode')
        if not output.exists():error.append('missing_expected_data_endpoint')
        if re.search(r'(^|\n)ERROR|Lost atoms',logtext):error.append('LAMMPS_error_or_lost_atoms')
        topology=None
        if not error:
            try:topology=validate_data(output)
            except ValueError as exc:error.append('endpoint_topology:'+str(exc))
        receipt.update(status='failed' if error else 'numerically_completed',returncode=run.returncode,
                       elapsed_seconds=time.time()-start,errors=error,
                       output_sha256=sha(output) if output.exists() else None,output_path=str(output.resolve()),endpoint_topology=topology,
                       LAMMPS_version=next((x for x in logtext.splitlines() if x.startswith('LAMMPS (')),None),
                       physical_equilibrium_established=False,linearity_established=False)
        receipt_path.write_text(json.dumps(receipt,indent=2)+'\n');outcomes.append(receipt)
        if error:raise RuntimeError('Execution failed; original evidence preserved: '+str(receipt_path))
    result=dict(plan=str(plan_path),plan_sha256=sha(plan_path),jobs=len(outcomes),
                numerical_completions=len(outcomes),MD_executed=True,scientific_validation=False,
                note='Run preparation, amplitude and runtime diagnostics before claiming convergence or linearity.')
    target=plan_path.parent/'execution_summary.json'
    if target.exists():
        target=plan_path.parent/f'execution_summary_{int(time.time())}.json'
    target.write_text(json.dumps(result,indent=2)+'\n');return result


def execute_parallel_plan(plan_path,command,workers,worker_logical_cpus,resume=False):
    """Bounded independent DAG jobs; MPI/resource binding stays explicit."""
    if workers<1 or worker_logical_cpus<1:raise ValueError('Positive worker count and declared per-worker CPU usage required.')
    capacity=os.environ.get('SLURM_CPUS_ON_NODE')
    if capacity and workers*worker_logical_cpus>int(re.match(r'\d+',capacity).group()):
        raise ValueError('Concurrent workers would exceed this node CPU allocation.')
    if workers>1 and any(str(x)=='srun' for x in command) and not {'--exact','--exclusive'}.issubset(command):
        raise ValueError('Packed srun workers require explicit --exact --exclusive resource isolation.')
    plan_path=Path(plan_path).resolve();plan=json.loads(plan_path.read_text())
    jobs={j['directory']:j for j in plan['execution_order']};pending=set(jobs);done=set();running={};failures=[]
    if len(jobs)!=len(plan['execution_order']):raise ValueError('Duplicate execution directory.')
    if any(set(j.get('depends_on',[]))-set(jobs) for j in jobs.values()):raise ValueError('Missing DAG dependency.')
    def execute_one(job):
        p=Path(job['directory'])/'execution_plan.json'
        value=dict(parent_plan=str(plan_path),parent_plan_sha256=sha(plan_path),execution_order=[job])
        if p.exists() and json.loads(p.read_text())!=value:raise ValueError('Per-case execution plan changed.')
        if not p.exists():p.write_text(json.dumps(value,indent=2)+'\n')
        return execute_plan(p,command,resume)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        while pending or running:
            if not failures:
                ready=[key for key in jobs if key in pending and set(jobs[key].get('depends_on',[]))<=done]
                for key in ready[:workers-len(running)]:
                    pending.remove(key);running[pool.submit(execute_one,jobs[key])]=key
            if not running:
                if failures:break
                if pending:raise ValueError('Cyclic/unresolved execution dependency.')
                break
            finished,_=wait(running,return_when=FIRST_COMPLETED)
            for future in finished:
                key=running.pop(future)
                try:future.result();done.add(key)
                except Exception as exc:failures.append(dict(directory=key,error=repr(exc)))
    result=dict(completed_jobs=len(done),failed_jobs=failures,unstarted_jobs=sorted(pending),
                workers=workers,declared_worker_logical_cpus=worker_logical_cpus,
                MD_executed=bool(done or failures),scientific_validation=False)
    p=plan_path.parent/f'parallel_execution_summary_{int(time.time())}.json';p.write_text(json.dumps(result,indent=2)+'\n')
    if failures:raise RuntimeError('Failure evidence preserved; independent running workers finalized, no further jobs started: '+str(p))
    return result
