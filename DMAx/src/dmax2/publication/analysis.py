"""Measured-strain harmonics and finite-window qualification diagnostics.

Temporal bootstrap intervals condition on one history. They do not replace
independent-packing uncertainty or establish equilibrium, linearity or absence
of aging. No engineering tolerance is imposed as a universal physical law.
"""
from __future__ import annotations
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from ..analysis.fourier import periodic_window, project_periodic_response


def read_trace(path):
    d=pd.read_csv(path,sep=r'\s+',comment='#')
    aliases={'time_ps':'t_ps','epsz':'eps_z','epsx':'eps_x','epsy':'eps_y',
             'szz_tot_MPa':'tot_zz','szz_vir_MPa':'vir_zz',
             'sxx_tot_MPa':'tot_xx','sxx_vir_MPa':'vir_xx',
             'syy_tot_MPa':'tot_yy','syy_vir_MPa':'vir_yy'}
    d=d.rename(columns={k:v for k,v in aliases.items() if k in d and v not in d})
    if 'strain_box' not in d and 'eps_z' in d:d['strain_box']=d.eps_z
    if 'strain_box' not in d:raise ValueError('Actual measured axial engineering strain is required.')
    return d


def tensor_candidates(M,D):
    """Linear-isotropic identities applied conditionally to complex H1 values.

    D=(sigma_zz-(sigma_xx+sigma_yy)/2)/eps_zz for fixed lateral strains.
    Finite-amplitude/nonisotropic values are candidates, not certified E or G.
    """
    M=np.asarray(M,dtype=complex);D=np.asarray(D,dtype=complex)
    G=D/2;K=M-4*G/3;den=3*K+G
    if np.any(abs(den)<=np.finfo(float).eps*np.maximum(1.,abs(K)+abs(G))):
        raise ValueError('Isotropic conversion has an unresolved/zero denominator.')
    E=9*K*G/den;nu=(3*K-2*G)/(2*den)
    return dict(M=M,G=G,K=K,E=E,nu=nu,
                interpretation='finite-amplitude isotropic-equivalent candidates; isotropy/LVE not established')


def acoustic_candidates(M_MPa,density_g_cm3,frequency_GHz):
    """Complex longitudinal continuum k=omega*sqrt(rho/M*), weak-loss not assumed.

    Convention exp(i omega t-i k x), M*=M'+iM''; attenuation=-Im(k).
    Homogeneous-strain q→0 response is not itself a finite-q BLS simulation.
    """
    M=np.asarray(M_MPa,complex)*1e6;rho=np.asarray(density_g_cm3)*1000.;omega=2*np.pi*frequency_GHz*1e9
    if np.any(rho<=0) or np.any(abs(M)==0):raise ValueError('Positive density and nonzero longitudinal response required.')
    k=omega*np.sqrt(rho/M);k=np.where(k.real<0,-k,k)
    v=omega/k.real;alpha=-k.imag
    return dict(phase_speed_m_s=v,attenuation_per_m=alpha,attenuation_per_wavelength=alpha*2*np.pi/k.real,
                loss_tangent=M.imag/M.real,
                interpretation='continuum acoustic candidates; same material/state/observable and linearity assumptions still require validation')


def _complex_metrics(coefficients):
    e=coefficients[...,0]+1j*coefficients[...,1]
    if np.any(abs(e)<1e-12):raise ValueError('Unresolved measured-strain denominator.')
    s=coefficients[...,2]+1j*coefficients[...,3];M=s/e
    tangent=np.divide(M.imag,M.real,out=np.full_like(M.real,np.nan),where=abs(M.real)>100*np.finfo(float).eps*np.maximum(1.,abs(M)))
    return np.stack([M.real,M.imag,tangent,np.degrees(np.angle(M))],axis=-1)


METRICS=['storage_MPa','loss_MPa','tan_delta','phase_deg']


def _json_safe(value):
    if isinstance(value,dict):return {k:_json_safe(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [_json_safe(v) for v in value]
    if isinstance(value,(float,np.floating)) and not np.isfinite(value):return None
    return value


def temporal_diagnostics(cycle_coefficients,frequency_GHz,block_ps=200.,draws=2000,minimum_blocks=6,
                         precision_targets=None,consistency_targets=None,seed=77113):
    """Nonoverlapping physical-time blocks and disjoint-half equivalence targets.

    A block rounds UP to complete cycles. Bootstrap stress/strain jointly.
    Relative precision/consistency tolerances must be explicit dictionaries,
    keyed by storage_MPa/loss_MPa/tan_delta. For phase use absolute degrees.
    Crossing a target describes this finite record, not material equilibration.
    """
    c=np.asarray(cycle_coefficients,float)
    if c.ndim!=2 or c.shape[1]!=4 or not np.isfinite(c).all():raise ValueError('Four finite stress/strain coefficient columns per cycle required.')
    for targets in (precision_targets,consistency_targets):
        if targets is not None and (not {'storage_MPa','loss_MPa','tan_delta'}.issubset(targets)
                or any(not np.isfinite(x) or x<=0 for x in targets.values())):
            raise ValueError('Explicit positive storage/loss/tangent tolerances are required; an empty dictionary is not a qualification.')
    b=max(1,math.ceil(block_ps*frequency_GHz/1000.-1e-10));n=len(c)//b
    if n<2:return pd.DataFrame([dict(cycles=len(c),elapsed_ps=len(c)*1000/frequency_GHz,
                                   status='insufficient_temporal_blocks',accepted=False,block_cycles=b,blocks=n)])
    blocks=c[:n*b].reshape(n,b,4).mean(axis=1);rows=[];rng=np.random.default_rng(seed)
    for count in range(2,n+1):
        part=blocks[:count];point=_complex_metrics(part.mean(axis=0));boot=_complex_metrics(part[rng.integers(count,size=(draws,count))].mean(axis=1))
        lo,hi=np.quantile(boot,[.025,.975],axis=0);half=count//2
        early=part[:half];late=part[-half:]
        a=_complex_metrics(early.mean(axis=0));z=_complex_metrics(late.mean(axis=0))
        contrast=(_complex_metrics(late[rng.integers(half,size=(draws,half))].mean(axis=1))-
                  _complex_metrics(early[rng.integers(half,size=(draws,half))].mean(axis=1)))
        clo,chi=np.quantile(contrast,[.025,.975],axis=0)
        row=dict(cycles=count*b,elapsed_ps=count*b*1000/frequency_GHz,block_cycles=b,
                 actual_block_ps=b*1000/frequency_GHz,blocks=count,blocks_per_half=half,
                 conditional_temporal_interval=True,independence_established=False,
                 aging_disproved=False,physical_equilibrium_established=False)
        precision_ok=precision_targets is not None;consistency_ok=consistency_targets is not None
        for j,name in enumerate(METRICS):
            reach=max(point[j]-lo[j],hi[j]-point[j]);scale=abs(point[j])
            row.update({name:point[j],name+'_CI_low':lo[j],name+'_CI_high':hi[j],
                        name+'_CI_larger_reach':reach,name+'_CI_relative_reach':reach/scale if scale>1e-12 else np.nan,
                        name+'_late_minus_early':z[j]-a[j],name+'_difference_CI_low':clo[j],name+'_difference_CI_high':chi[j]})
            if precision_targets is not None:
                tolerance=precision_targets.get(name)
                if tolerance is not None:
                    allowed=tolerance if name=='phase_deg' else tolerance*scale
                    precision_ok &= reach<=allowed
            if consistency_targets is not None:
                tolerance=consistency_targets.get(name)
                if tolerance is not None:
                    allowed=tolerance if name=='phase_deg' else tolerance*scale
                    consistency_ok &= clo[j]>=-allowed and chi[j]<=allowed
        row.update(precision_targets_met=bool(precision_ok),late_window_consistency_targets_met=bool(consistency_ok),
                   accepted=bool(count>=minimum_blocks and precision_ok and consistency_ok))
        row['status']='finite_window_targets_met' if row['accepted'] else ('insufficient_temporal_blocks' if count<minimum_blocks else
                       'explicit_targets_required' if precision_targets is None or consistency_targets is None else 'extend_or_report_finite_record_limit')
        rows.append(row)
    return pd.DataFrame(rows)


def analyze_trace(path,frequency_GHz,output,operator='total',start_ps=None,discard_cycles=0,block_ps=200.,
                  precision_targets=None,consistency_targets=None,metadata=None):
    output=Path(output)
    if output.exists():raise FileExistsError('Analysis results require a new output directory.')
    d=read_trace(path)
    if start_ps is not None:d=d.loc[d.t_ps>=start_ps-1e-10].copy()
    label='tot' if operator=='total' else 'vir'
    signal=label+'_zz'
    if signal not in d:raise ValueError('Requested stress operator missing: '+signal)
    prepared=d.rename(columns={signal:'szz_tot_MPa' if label=='tot' else 'szz_vir_MPa'})
    canonical='szz_tot_MPa' if label=='tot' else 'szz_vir_MPa'
    fit,spectrum,used=project_periodic_response(prepared,'tension_nvt_z_fixed_lateral',canonical,frequency_GHz,
                                             discard_cycles=discard_cycles,incomplete='reject')
    t=used.t_ps.to_numpy();theta=2*np.pi*(t-t[0])*frequency_GHz/1000
    e=used.strain_box.to_numpy();s=used[canonical].to_numpy();spp=fit.samples_per_cycle
    cycles=fit.complete_cycles
    coeff=np.column_stack([2*np.mean((a*np.sin(theta)).reshape(cycles,spp),axis=1) for a in [e,s]])
    ecos=2*np.mean((e*np.cos(theta)).reshape(cycles,spp),axis=1)
    scos=2*np.mean((s*np.cos(theta)).reshape(cycles,spp),axis=1)
    per_cycle=np.column_stack([coeff[:,0],ecos,coeff[:,1],scos])
    prefix=np.cumsum(per_cycle,axis=0)/np.arange(1,cycles+1)[:,None]
    values=_complex_metrics(prefix);cycle_table=pd.DataFrame(values,columns=METRICS)
    cycle_table.insert(0,'cycle',np.arange(1,cycles+1));cycle_table.insert(1,'elapsed_ps',np.arange(1,cycles+1)*1000/frequency_GHz)
    for name in METRICS:cycle_table[name+'_change_after_cycle']=cycle_table[name].diff()
    diagnostics=temporal_diagnostics(per_cycle,frequency_GHz,block_ps,
                                    precision_targets=precision_targets,consistency_targets=consistency_targets)
    M=complex(fit.storage_modulus_MPa,fit.loss_modulus_MPa);tensor=None;tensor_error=None
    if {label+'_xx',label+'_yy'}.issubset(used):
        dev=used[canonical].to_numpy()-.5*(used[label+'_xx'].to_numpy()+used[label+'_yy'].to_numpy())
        ec=2*np.mean(e*np.sin(theta))+2j*np.mean(e*np.cos(theta))
        dc=2*np.mean(dev*np.sin(theta))+2j*np.mean(dev*np.cos(theta))
        try:tensor=tensor_candidates(M,dc/ec)
        except ValueError as exc:tensor_error=str(exc)
    output.mkdir(parents=True);spectrum.to_csv(output/'harmonics.csv',index=False)
    cycle_table.to_csv(output/'cumulative_cycles.csv',index=False)
    diagnostics.to_csv(output/'runtime_diagnostics.csv',index=False)
    pd.DataFrame(per_cycle,columns=['strain_sin','strain_cos','stress_sin_MPa','stress_cos_MPa']).to_csv(output/'cycle_coefficients.csv',index=False)
    result=dict(fit.as_dict(),operator=operator,frequency_GHz=frequency_GHz,
                mean_stress_MPa=float(s.mean()),harmonic_distortion_2to5=float(np.sqrt(sum(float(v)**2 for v in spectrum.loc[spectrum.harmonic.gt(1),'harmonic_over_first']))),
                physical_linearity_established=False,physical_equilibrium_established=False,
                metadata=metadata or {},uncertainty_scope='conditional temporal blocks; independent packings remain separate')
    if tensor is not None:
        result['isotropic_equivalent_candidates']={k:dict(real=float(np.real(v)),imag=float(np.imag(v))) for k,v in tensor.items() if k!='interpretation'}
        result['tensor_interpretation']=tensor['interpretation']
    if tensor_error is not None:result['tensor_conversion_unresolved']=tensor_error
    if 'density_g_cm3' in used:
        rho0=float(np.mean(used.density_g_cm3.to_numpy()*(1+e)))
        ac=acoustic_candidates(M,rho0,frequency_GHz)
        result.update(reference_density_g_cm3=rho0,acoustic_candidates={k:float(v) if k!='interpretation' else v for k,v in ac.items()})
    result=_json_safe(result)
    (output/'response.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    from .build import sha
    manifest=dict(source=str(Path(path).resolve()),source_sha256=sha(path),operator=operator,
                  frequency_GHz=frequency_GHz,start_ps=start_ps,discard_cycles=discard_cycles,
                  complete_cycles=cycles,sampling_policy='complete uniformly sampled half-open cycles, actual measured strain',
                  estimator='Fourier projection plus equivalent OLS numerical check; no hidden drift subtraction',
                  source_code_sha256=sha(__file__),outputs={p.name:sha(p) for p in output.iterdir() if p.is_file()})
    (output/'analysis_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return result


def aggregate_responses(paths,output):
    """Keep independent packings distinct from repeated velocities on one box."""
    rows=[]
    for path in paths:
        value=json.loads(Path(path).read_text());meta=value.get('metadata',{})
        if 'packing' not in meta or not meta.get('initial_packing_sha256'):
            raise ValueError('Every response needs explicit independent-packing metadata and initial-coordinate SHA256.')
        rows.append(dict(polymer=meta['polymer'],repeat_units=meta['repeat_units'],chains=meta['chains'],
                         packing=meta['packing'],temperature_K=meta['temperature_K'],frequency_GHz=value['frequency_GHz'],
                         strain_amplitude=meta['strain_amplitude'],operator=value['operator'],
                         initial_packing_sha256=meta['initial_packing_sha256'],
                         storage_MPa=value['storage_modulus_MPa'],loss_MPa=value['loss_modulus_MPa'],tan_delta=value['tan_delta']))
    frame=pd.DataFrame(rows);keys=['polymer','repeat_units','chains','temperature_K','frequency_GHz','strain_amplitude','operator']
    if frame.duplicated(keys+['packing']).any():raise ValueError('Repeated velocities/analyses of one packing are not independent packing replicas.')
    summaries=[]
    for key,g in frame.groupby(keys):
        if g.initial_packing_sha256.nunique()!=len(g):
            raise ValueError('Different velocity seeds or labels on one initial packing are not independent packing replicas.')
        row=dict(zip(keys,key));row['independent_packings']=len(g)
        for name in ['storage_MPa','loss_MPa','tan_delta']:
            row[name+'_mean']=g[name].mean();row[name+'_packing_SD']=g[name].std(ddof=1)
            row[name+'_packing_SEM']=g[name].std(ddof=1)/np.sqrt(len(g))
        row['ratio_of_mean_moduli']=g.loss_MPa.mean()/g.storage_MPa.mean()
        row['uncertainty_scope']='sample SD/SEM across recorded independent packings, not temporal CI'
        summaries.append(row)
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    output.mkdir();frame.to_csv(output/'per_packing.csv',index=False);pd.DataFrame(summaries).to_csv(output/'packing_summary.csv',index=False)
    return output/'packing_summary.csv'


def conditional_wlf_temperature(reference_temperature_K,reference_frequency_Hz,target_frequency_Hz,C1,C2_K):
    """Solve log10(f/f_ref)=C1*(T-Tref)/(C2+T-Tref), retaining explicit anchors."""
    inputs=[reference_temperature_K,reference_frequency_Hz,target_frequency_Hz,C1,C2_K]
    if not np.isfinite(inputs).all() or min(inputs)<=0:raise ValueError('Positive finite WLF temperatures/frequencies/constants required.')
    L=np.log10(target_frequency_Hz/reference_frequency_Hz)
    if C1-L<=0:raise ValueError('Target shift exceeds the finite branch of this WLF parameterization.')
    T=reference_temperature_K+C2_K*L/(C1-L)
    if T<=0:raise ValueError('Nonphysical extrapolated temperature.')
    return dict(temperature_K=float(T),log10_frequency_ratio=float(L),C1=float(C1),C2_K=float(C2_K),
                reference_temperature_K=float(reference_temperature_K),reference_frequency_Hz=float(reference_frequency_Hz),
                target_frequency_Hz=float(target_frequency_Hz),independent_Tg_prediction=False,
                interpretation='conditional experiment-anchored marker translation; observable, material/state and validity range must be supplied')
