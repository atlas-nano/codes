"""Stage-specific finite-window preparation checks and backbone dimensions.

Stable controls or arrested chain motion do not certify polymer equilibrium.
Any stopping tolerances are explicit user choices. Density in NVT is imposed.
"""
from __future__ import annotations
from collections import defaultdict
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from .build import data_sections,sha,validate_data


def backbone_paths(data_file,polymer):
    d=data_sections(data_file);atoms=sorted(d['Atoms'],key=lambda a:int(a[0]));ids=[int(a[0]) for a in atoms]
    index={a:i for i,a in enumerate(ids)};adj=defaultdict(set)
    for b in d['Bonds']:
        a,z=map(int,b[2:4]);adj[a].add(z);adj[z].add(a)
    backbone_types={'PVDF':{1,2,4,5},'PDMS':{3,4}}[polymer];paths=[]
    for molecule in sorted({int(a[1]) for a in atoms}):
        sites={int(a[0]) for a in atoms if int(a[1])==molecule and int(a[2]) in backbone_types}
        edges={a:adj[a]&sites for a in sites};ends=sorted(a for a in sites if len(edges[a])==1)
        if len(ends)!=2 or any(len(v) not in (1,2) for v in edges.values()):raise ValueError('Expected a verified linear backbone.')
        path=[ends[0]];previous=None
        while len(path)<len(sites):
            nxt=list(edges[path[-1]]-({previous} if previous is not None else set()))
            if len(nxt)!=1:raise ValueError('Disconnected/cyclic backbone.')
            previous=path[-1];path.append(nxt[0])
        paths.append(np.array([index[a] for a in path]))
    return np.array(ids),paths


def iter_frames(path):
    with Path(path).open() as stream:
        previous=None
        while True:
            line=stream.readline()
            if not line:return
            if line.strip()!='ITEM: TIMESTEP':raise ValueError('Invalid geometry dump header.')
            step=int(stream.readline())
            if previous is not None and step<=previous:raise ValueError('Duplicate/nonmonotonic geometry clock.')
            if stream.readline().strip()!='ITEM: NUMBER OF ATOMS':raise ValueError('Invalid atom-count header.')
            n=int(stream.readline())
            if stream.readline().split()!=['ITEM:','BOX','BOUNDS','pp','pp','pp']:raise ValueError('Orthorhombic periodic boxes required.')
            bounds=np.array([list(map(float,stream.readline().split())) for _ in range(3)])
            names=stream.readline().split()[2:]
            if not {'id','mol','type','xu','yu','zu'}.issubset(names):raise ValueError('Identities and unwrapped coordinates required.')
            lines=[stream.readline() for _ in range(n)]
            if any(not s.strip() for s in lines):raise ValueError('Incomplete geometry frame.')
            values=np.array([list(map(float,s.split())) for s in lines]);order=np.argsort(values[:,names.index('id')])
            values=values[order]
            if not np.isfinite(values).all():raise ValueError('Nonfinite coordinates.')
            yield step,values[:,names.index('id')].astype(int),values[:,[names.index(k) for k in ['xu','yu','zu']]],np.diff(bounds,axis=1).ravel()
            previous=step


def chain_dimensions(geometry,data_file,polymer,dt_fs=.5,start_ps=-np.inf,stop_ps=np.inf):
    validate_data(data_file);ids,paths=backbone_paths(data_file,polymer);rows=[]
    for step,actual,xyz,L in iter_frames(geometry):
        t=step*dt_fs/1000
        if t<start_ps or t>stop_ps:continue
        if not np.array_equal(actual,ids) or np.any(L<=0):raise ValueError('Topology identities or box changed incorrectly.')
        rgs=[];res=[];tangents=[]
        for path in paths:
            delta=np.diff(xyz[path],axis=0);delta-=np.rint(delta/L)*L
            chain=np.vstack([np.zeros(3),np.cumsum(delta,axis=0)])
            rgs.append(float(np.sqrt(np.mean(np.sum((chain-chain.mean(axis=0))**2,axis=1)))))
            res.append(float(np.linalg.norm(chain[-1]-chain[0])));tangents.extend(delta/np.linalg.norm(delta,axis=1)[:,None])
        unit=np.array(tangents);Q=1.5*(unit.T@unit)/len(unit)-.5*np.eye(3)
        rows.append(dict(step=step,t_ps=t,mean_Rg_A=np.mean(rgs),mean_Re_A=np.mean(res),
                         rms_Rg_A=np.sqrt(np.mean(np.array(rgs)**2)),rms_Re_A=np.sqrt(np.mean(np.array(res)**2)),
                         Q_max_eigenvalue=np.linalg.eigvalsh(Q)[-1],chains=len(paths),
                         lx_A=L[0],ly_A=L[1],lz_A=L[2],equilibrium_established=False))
    if not rows:raise ValueError('No coordinates in the requested preparation window.')
    return pd.DataFrame(rows)


def assess_preparation(trace,data_file,stages,output,polymer,geometry=None,targets=None,window_ps=250.,dt_fs=.5):
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    d=pd.read_csv(trace,sep=r'\s+',comment='#')
    required={'step','time_ps','temperature','pressure','density','pe','ke'}
    if not required.issubset(d):raise ValueError('Expected seven-column preparation scalar record.')
    if not np.isfinite(d[list(required)].to_numpy()).all() or d.step.duplicated().any():raise ValueError('Nonfinite or duplicated scalar data.')
    if len(d)<3 or (d.density<=0).any() or (d.temperature<=0).any():raise ValueError('A positive regular preparation record is required.')
    intervals=np.diff(d.time_ps)
    if (intervals<=0).any() or not np.allclose(intervals,intervals[0],rtol=1e-7,atol=1e-9):
        raise ValueError('Preparation scalar record has missing, irregular or nonmonotonic samples.')
    if not np.allclose(d.time_ps,d.step*dt_fs/1000,rtol=1e-9,atol=1e-8):
        raise ValueError('Physical time does not match the declared MD timestep.')
    if targets is not None and (not targets or any(not np.isfinite(v) or v<=0 for v in targets.values())):
        raise ValueError('Preparation stopping targets must be an explicit nonempty set of positive tolerances.')
    sections=data_sections(data_file);masses={int(v[0]):float(v[1]) for v in sections['Masses']}
    mass=sum(masses[int(v[2])] for v in sections['Atoms']);d['PE_kJ_kg']=d.pe*4184/mass
    d['pressure_MPa']=d.pressure*.101325;rows=[];geometry_tables=[]
    for phase in stages:
        if not phase['fixed_temperature']:continue
        start,stop=phase['start_ps'],phase['stop_ps']
        if stop-start<2*window_ps:continue
        early=d.loc[d.time_ps.gt(stop-2*window_ps)&d.time_ps.le(stop-window_ps)]
        late=d.loc[d.time_ps.gt(stop-window_ps)&d.time_ps.le(stop)]
        expected=window_ps/intervals[0]
        if not math.isclose(expected,round(expected),abs_tol=1e-6) or len(early)!=round(expected) or len(late)!=round(expected):
            raise ValueError('Incomplete phase windows; no stationarity inference.')
        row=dict(phase,window_ps=window_ps,temperature_early_K=early.temperature.mean(),temperature_late_K=late.temperature.mean(),
                 density_relative_change=(late.density.mean()/early.density.mean()-1),
                 potential_energy_change_kJ_kg=late.PE_kJ_kg.mean()-early.PE_kJ_kg.mean(),
                 pressure_change_MPa=late.pressure_MPa.mean()-early.pressure_MPa.mean(),
                 NVT_density_imposed=(phase['ensemble']=='NVT'),equilibrium_established=False)
        if geometry:
            g=chain_dimensions(geometry,data_file,polymer,dt_fs,stop-2*window_ps,stop);g['stage']=phase['stage'];geometry_tables.append(g)
            a=g.loc[g.t_ps.gt(stop-2*window_ps)&g.t_ps.le(stop-window_ps)]
            z=g.loc[g.t_ps.gt(stop-window_ps)&g.t_ps.le(stop)]
            for name in ['mean_Rg_A','mean_Re_A']:
                row[name+'_relative_change']=z[name].mean()/a[name].mean()-1 if len(a)>1 and len(z)>1 else None
        accepted=targets is not None
        if targets is not None:
            requested={'potential_energy_change_kJ_kg':abs(row['potential_energy_change_kJ_kg'])}
            if phase['ensemble']=='NPT':requested['density_relative_change']=abs(row['density_relative_change'])
            for name in ['mean_Rg_A_relative_change','mean_Re_A_relative_change']:
                if name in targets:requested[name]=abs(row.get(name,np.nan)) if row.get(name) is not None else np.nan
            for name,value in requested.items():
                accepted &= name in targets and np.isfinite(value) and value<=targets[name]
        row.update(finite_window_targets_met=bool(accepted),status='finite_window_targets_met' if accepted else
                   'diagnostic_only_no_targets' if targets is None else 'extend_or_report_remaining_change')
        rows.append(row)
    if not rows:raise ValueError('No fixed-temperature stage supports two complete comparison windows.')
    output.mkdir();pd.DataFrame(rows).to_csv(output/'preparation_windows.csv',index=False)
    if geometry_tables:pd.concat(geometry_tables).to_csv(output/'backbone_dimensions.csv',index=False)
    result=dict(stages=rows,explicit_targets=targets,physical_equilibrium_established=False,
                interpretation='Finite-window descriptive changes. Density is not a convergence test in NVT; stable/frozen chains do not prove equilibrium.',
                sources={str(Path(v).resolve()):sha(v) for v in [trace,data_file,*([geometry] if geometry else [])]})
    (output/'preparation_assessment.json').write_text(json.dumps(result,indent=2)+'\n')
    return result
