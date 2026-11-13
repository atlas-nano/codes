"""Descriptive horizontal superposition within measured frequency support.

Independent storage/loss and shared shifts are compared in log modulus. A
numerical optimum is not a physical TTS certificate or a relaxation-time fit.
No Gaussian process or extrapolated support is used by this diagnostic.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..analysis.master_curve import assess_frequency_sweep_peaks
from .build import sha


def normalize_summary(frame):
    d=frame.copy()
    aliases={'frequency_ghz':'frequency_GHz','storage_modulus_MPa':'storage_MPa',
             'loss_modulus_MPa':'loss_MPa','storage_MPa_mean':'storage_MPa',
             'loss_MPa_mean':'loss_MPa','strain':'strain_amplitude'}
    d=d.rename(columns={k:v for k,v in aliases.items() if k in d and v not in d})
    required=['temperature_K','frequency_GHz','storage_MPa','loss_MPa']
    if any(k not in d for k in required):raise ValueError('TTS input requires '+str(required))
    if not np.isfinite(d[required].to_numpy(float)).all() or (d.frequency_GHz<=0).any():
        raise ValueError('Finite moduli/temperature and positive frequencies are required; exclusions must be explicit upstream.')
    return d


def shift_scores(reference,moving,metric,shifts,minimum_overlap_points=3):
    """log10(f_reduced)=log10(f)+shift; only interpolation of reference data."""
    r=reference.sort_values('frequency_GHz');m=moving.sort_values('frequency_GHz')
    if r.frequency_GHz.duplicated().any() or m.frequency_GHz.duplicated().any():
        raise ValueError('Partition repeated packings/velocities or aggregate them explicitly before superposition.')
    if (r[metric]<=0).any() or (m[metric]<=0).any():
        raise ValueError('Log-modulus diagnostics require positive values; signed/noisy loss must remain excluded explicitly.')
    rx=np.log10(r.frequency_GHz.to_numpy());ry=np.log10(r[metric].to_numpy())
    mx=np.log10(m.frequency_GHz.to_numpy());my=np.log10(m[metric].to_numpy())
    rows=[]
    for shift in shifts:
        x=mx+shift;use=(x>=rx[0]-1e-12)&(x<=rx[-1]+1e-12)
        n=int(use.sum());width=float(np.ptp(x[use])) if n>1 else 0.
        score=float(np.mean((my[use]-np.interp(x[use],rx,ry))**2)) if n>=minimum_overlap_points else np.inf
        rows.append((shift,score,n,width))
    return pd.DataFrame(rows,columns=['log10_a','mean_squared_log10_residual','overlap_points','overlap_log10_span'])


def fit_shifts(reference,moving,minimum_overlap_points=3,grid_points=2001):
    if minimum_overlap_points<3:raise ValueError('At least three measured overlap points required for this descriptive shape diagnostic.')
    rx=np.log10(reference.frequency_GHz);mx=np.log10(moving.frequency_GHz)
    span=max(float(np.ptp(rx)),float(np.ptp(mx)))
    if span<=0:raise ValueError('A frequency curve needs distinct measured frequencies.')
    shifts=np.linspace(-span,span,grid_points)
    curves={metric:shift_scores(reference,moving,metric,shifts,minimum_overlap_points) for metric in ['storage_MPa','loss_MPa']}
    curves['common']=curves['storage_MPa'].copy()
    curves['common']['mean_squared_log10_residual']=(curves['storage_MPa'].mean_squared_log10_residual+curves['loss_MPa'].mean_squared_log10_residual)/2
    result=[]
    for metric,table in curves.items():
        legal=np.isfinite(table.mean_squared_log10_residual)
        if not legal.any():
            result.append(dict(metric=metric,status='insufficient_measured_overlap',log10_a=np.nan,scientific_TTS_accepted=False));continue
        candidates=table.loc[legal];i=candidates.mean_squared_log10_residual.idxmin();row=table.loc[i].to_dict()
        # A support-limited optimum cannot establish a resolved temperature shift.
        edge=i in (candidates.index[0],candidates.index[-1])
        row.update(metric=metric,status='support_boundary_optimum' if edge else 'interior_numerical_optimum',
                   scientific_TTS_accepted=False,shift_grid_spacing=float(shifts[1]-shifts[0]),
                   support_min_log10_a=float(candidates.log10_a.min()),support_max_log10_a=float(candidates.log10_a.max()))
        result.append(row)
    return pd.DataFrame(result)


def diagnostics(summary,reference_temperature,output,minimum_overlap_points=3):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    source=Path(summary);d=normalize_summary(pd.read_csv(source));out=Path(output)
    if out.exists():raise FileExistsError(out)
    keys=[k for k in ['polymer','repeat_units','chains','strain_amplitude','operator','packing'] if k in d]
    out.mkdir(parents=True);all_shifts=[];all_peaks=[];outputs=[]
    groups=d.groupby(keys,dropna=False,sort=True) if keys else [((),d)]
    for number,(identity,g) in enumerate(groups,1):
        if not isinstance(identity,tuple):identity=(identity,)
        meta=dict(zip(keys,identity));reference=g.loc[np.isclose(g.temperature_K,reference_temperature,rtol=0,atol=1e-7)]
        if len(reference)<minimum_overlap_points:raise ValueError('Reference temperature lacks a measured frequency curve in '+str(meta))
        shifts=[]
        for temperature,moving in g.groupby('temperature_K'):
            if np.isclose(temperature,reference_temperature):
                s=pd.DataFrame([dict(metric=metric,log10_a=0.,status='reference',scientific_TTS_accepted=False) for metric in ['storage_MPa','loss_MPa','common']])
            else:s=fit_shifts(reference,moving,minimum_overlap_points)
            for k,v in meta.items():s[k]=v
            s['temperature_K']=temperature;shifts.append(s)
        s=pd.concat(shifts,ignore_index=True);all_shifts.append(s)
        peak=assess_frequency_sweep_peaks(g,metric='loss_MPa',frequency_col='frequency_GHz')
        for k,v in meta.items():peak[k]=v
        all_peaks.append(peak)
        fig,axes=plt.subplots(2,3,figsize=(13,7.7),constrained_layout=True)
        for temperature,h in g.groupby('temperature_K'):
            h=h.sort_values('frequency_GHz');label=f'{temperature:g} K'
            for j,metric in enumerate(['storage_MPa','loss_MPa']):
                axes[0,j].loglog(h.frequency_GHz,h[metric],'-o',label=label)
                fitted=s.loc[(s.temperature_K==temperature)&(s.metric==metric),'log10_a'].iloc[0]
                shared=s.loc[(s.temperature_K==temperature)&(s.metric=='common'),'log10_a'].iloc[0]
                if np.isfinite(fitted):axes[1,j].loglog(h.frequency_GHz*10**fitted,h[metric],'-o',label=label+' independent')
                if np.isfinite(shared):axes[1,j].loglog(h.frequency_GHz*10**shared,h[metric],':',alpha=.7)
            axes[0,2].semilogx(h.frequency_GHz,h.loss_MPa/h.storage_MPa,'-o',label=label)
        for metric,marker in [('storage_MPa','o'),('loss_MPa','s'),('common','^')]:
            h=s.loc[s.metric==metric].sort_values('temperature_K')
            axes[1,2].plot(h.temperature_K,h.log10_a,'-'+marker,label=metric)
        for j,label in enumerate(["M′ (MPa)","M″ (MPa)",'M″ / M′']):
            axes[0,j].set(xlabel='Native frequency (GHz)',ylabel=label)
        axes[1,0].set(xlabel='Reduced frequency (GHz)',ylabel="M′ (MPa)")
        axes[1,1].set(xlabel='Reduced frequency (GHz)',ylabel="M″ (MPa)")
        axes[1,2].set(xlabel='Temperature (K)',ylabel='Descriptive log₁₀ a')
        axes[0,0].legend(fontsize=7);axes[1,2].legend(fontsize=8)
        for ax in axes.flat:ax.grid(alpha=.2)
        fig.suptitle(str(meta)+'\nNumerical horizontal shifts only: solid independent; dotted common; no low-frequency prediction',fontsize=10)
        name=f'native_and_shift_diagnostics_{number:02d}.pdf';fig.savefig(out/name);plt.close(fig);outputs.append(name)
    pd.concat(all_shifts,ignore_index=True).to_csv(out/'horizontal_shifts.csv',index=False)
    pd.concat(all_peaks,ignore_index=True).to_csv(out/'frequency_peak_status.csv',index=False)
    report=dict(source=str(source.resolve()),source_sha256=sha(source),reference_temperature_K=reference_temperature,
                method='log-modulus interpolation on native support; independent and common horizontal shifts, no vertical shift',
                minimum_overlap_points=minimum_overlap_points,scientific_TTS_accepted=False,
                WLF_fit_performed=False,VFT_fit_performed=False,Tg_prediction_performed=False,
                limitations='A numerical optimum is not a TTS or relaxation assignment. Boundary maxima are censored. Shift uncertainty and common-shift equivalence require independent validation.',
                outputs={p.name:sha(p) for p in out.iterdir() if p.is_file()})
    (out/'tts_manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    return out
