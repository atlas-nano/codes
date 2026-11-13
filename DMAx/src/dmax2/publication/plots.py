"""Plots from reduced outputs, keeping temporal and packing uncertainty apart."""
from pathlib import Path
import json
import numpy as np
import pandas as pd


def plot_analysis(directory):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    d=Path(directory);value=json.loads((d/'response.json').read_text())
    cycles=pd.read_csv(d/'cumulative_cycles.csv');runtime=pd.read_csv(d/'runtime_diagnostics.csv');h=pd.read_csv(d/'harmonics.csv')
    output=d/'mechanical_diagnostics.pdf'
    if output.exists():raise FileExistsError(output)
    fig,axes=plt.subplots(2,2,figsize=(10,7),constrained_layout=True)
    for ax,name,label,scale in zip(axes.flat[:3],['storage_MPa','loss_MPa','tan_delta'],["M′ (GPa)","M″ (GPa)",'M″ / M′'],[.001,.001,1.]):
        ax.plot(cycles.cycle,cycles[name]*scale,label='Cumulative first harmonic')
        if name+'_CI_low' in runtime:
            ax.fill_between(runtime.cycles,runtime[name+'_CI_low']*scale,runtime[name+'_CI_high']*scale,alpha=.2,label='Conditional 95% temporal interval')
        ax.set(xlabel='Complete drive cycles',ylabel=label);ax.grid(alpha=.2)
    axes[1,1].stem(h.harmonic,h.harmonic_over_first*100)
    axes[1,1].set(xlabel='Harmonic order',ylabel='Stress harmonic / H1 (%)',xticks=h.harmonic)
    axes[0,0].legend(fontsize=7)
    fig.suptitle(f"{value['frequency_GHz']:g} GHz; {value['operator']} stress; fixed-lateral M₁*\nFinite-amplitude/finite-age response; point harmonic magnitudes do not prove linearity",fontsize=10)
    fig.savefig(output);fig.savefig(output.with_suffix('.png'),dpi=180);plt.close(fig)
    return output


def plot_packing_summary(summary,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    d=pd.read_csv(summary);output=Path(output)
    if output.exists():raise FileExistsError(output)
    output.mkdir(parents=True)
    keys=[k for k in ['polymer','repeat_units','chains','operator'] if k in d]
    for number,(identity,g) in enumerate(d.groupby(keys,dropna=False),1):
        for x,curves in [('strain_amplitude',['temperature_K','frequency_GHz']),('temperature_K',['frequency_GHz','strain_amplitude']),('frequency_GHz',['temperature_K','strain_amplitude'])]:
            if g[x].nunique()<2:continue
            fig,axes=plt.subplots(1,3,figsize=(12,3.6),constrained_layout=True)
            for label,h in g.groupby(curves):
                h=h.sort_values(x)
                for ax,name,title,scale in zip(axes,['storage_MPa','loss_MPa','tan_delta'],["M′ (GPa)","M″ (GPa)",'M″ / M′'],[.001,.001,1.]):
                    xx=h[x]*100 if x=='strain_amplitude' else h[x]
                    ax.errorbar(xx,h[name+'_mean']*scale,yerr=h[name+'_packing_SD']*scale,fmt='-o',label=str(label))
                    ax.set(xlabel={'strain_amplitude':'Engineering strain amplitude (%)','temperature_K':'Temperature (K)','frequency_GHz':'Native frequency (GHz)'}[x],ylabel=title)
                    if x=='frequency_GHz':ax.set_xscale('log')
                    ax.grid(alpha=.2)
            axes[0].legend(fontsize=7);fig.suptitle(str(identity)+'; bars = sample SD across independent packings',fontsize=10)
            fig.savefig(output/f'packing_{number:02d}_vs_{x}.pdf');plt.close(fig)
    return output
