"""Paired downsampling of exactly the same dense trajectory; no new MD."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

from .analysis import read_trace
from .build import sha
from ..analysis.fourier import project_periodic_response


def compare_sampling(trace,frequency_GHz,strides,output):
    d=read_trace(trace);out=Path(output)
    if out.exists():raise FileExistsError(out)
    rows=[]
    for operator,prefix in [('total','tot'),('virial','vir')]:
        signal=prefix+'_zz'
        if signal not in d:continue
        canonical='szz_'+prefix+'_MPa';source=d.rename(columns={signal:canonical})
        fit,_,used=project_periodic_response(source,'tension_nvt_z_fixed_lateral',canonical,frequency_GHz)
        baseline=np.array([fit.storage_modulus_MPa,fit.loss_modulus_MPa,fit.tan_delta])
        dense_dt=float(np.median(np.diff(used.t_ps)))
        for stride in sorted(set([1,*map(int,strides)])):
            if stride<1:raise ValueError('Positive integer stride required.')
            if fit.samples_per_cycle%stride:raise ValueError('Paired sampling comparison requires a divisor of samples per complete cycle.')
            if fit.samples_per_cycle//stride<=10:
                rows.append(dict(operator=operator,stride=stride,logging_fs=dense_dt*1000*stride,
                                 status='H1_to_H5_unresolved_at_this_cadence',offset=0));continue
            for offset in range(stride):
                sampled=used.iloc[offset::stride]
                f,_,_=project_periodic_response(sampled,'tension_nvt_z_fixed_lateral',canonical,frequency_GHz)
                values=np.array([f.storage_modulus_MPa,f.loss_modulus_MPa,f.tan_delta]);diff=values-baseline
                row=dict(operator=operator,stride=stride,logging_fs=dense_dt*1000*stride,offset=offset,
                         status='paired_numerical_comparison',complete_cycles=f.complete_cycles,
                         complex_modulus_offset_MPa=float(np.hypot(diff[0],diff[1])),
                         complex_modulus_relative_offset=float(np.hypot(diff[0],diff[1])/np.hypot(*baseline[:2])))
                for j,key in enumerate(['storage_MPa','loss_MPa','tan_delta']):
                    row[key]=values[j];row[key+'_offset']=diff[j]
                    row[key+'_relative_offset']=diff[j]/abs(baseline[j]) if abs(baseline[j])>1e-12 else np.nan
                rows.append(row)
    if not rows:raise ValueError('No total/virial stress available.')
    out.mkdir(parents=True);table=pd.DataFrame(rows);table.to_csv(out/'all_sampling_offsets.csv',index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(10,7),constrained_layout=True)
    for operator,g in table.loc[table.status=='paired_numerical_comparison'].groupby('operator'):
        for actual in [True,False]:
            h=g.loc[g.offset==0] if actual else g
            for ax,key,label in zip(axes.flat,['complex_modulus_relative_offset','storage_MPa_relative_offset','loss_MPa_relative_offset','tan_delta_relative_offset'],['|ΔM₁*| / |M₁*, dense|',"|ΔM′| / |M′dense|","|ΔM″| / |M″dense|",'|Δtanδ| / |tanδdense|']):
                values=h.assign(value=abs(h[key])).groupby('logging_fs').value.max()
                ax.plot(values.index,100*values.values,('-o' if actual else '--s'),label=operator+(': offset 0' if actual else ': worst recorded phase offset'))
                ax.set(xlabel='Saved stress interval (fs)',ylabel=label+' (%)',xscale='log');ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=7);fig.suptitle(f'{frequency_GHz:g} GHz: same dense history; no independent trajectory comparison',fontsize=10)
    fig.savefig(out/'paired_sampling_offsets.pdf');plt.close(fig)
    (out/'sampling_manifest.json').write_text(json.dumps(dict(source_sha256=sha(trace),frequency_GHz=frequency_GHz,
        method='Identical complete cycles; phase offsets of integer-stride decimation; actual measured strain',
        dense_interval_fs=dense_dt*1000,independent_uncertainty_estimate=False,physical_validation=False),indent=2)+'\n')
    return out
