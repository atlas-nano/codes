"""Reproduce publication figures from bundled reduced CSV tables.

No MD, absolute storage paths, raw trace processing, new harmonic fitting or
bootstrap resampling. The original scientific exclusions remain in the tables.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import importlib.util
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

OUT = None

SIZES = {'PVDF_n9_c200': (0, 'Short: 200 × 10 VDF'), 'PVDF_n99_c20': (1, 'Intermediate: 20 × 100 VDF'), 'PVDF_n399_c5': (2, 'Long: 5 × 400 VDF'), 'PDMS_u11_c100': (0, 'Short: 100 × 10 internal D'), 'PDMS_u31_c40': (1, 'Intermediate: 40 × 30 internal D'), 'PDMS_u101_c12': (2, 'Long: 12 × 100 internal D')}
COLORS = ['#0072b2', '#d55e00', '#009e73']
ORDER = {key: value[0] for key, value in SIZES.items()}

def save(fig,stem):
    fig.savefig(OUT/(stem+'.pdf'),facecolor='white')
    fig.savefig(OUT/(stem+'.png'),dpi=170,facecolor='white')
    plt.close(fig)

def finish(axes):
    for ax in np.asarray(axes).ravel():
        ax.grid(alpha=.2,lw=.6)
        ax.spines[['top','right']].set_visible(False)
        ax.tick_params(labelsize=8)

def temperature(temp):
    data=temp.loc[~temp.legacy_reference].copy()
    fig,axes=plt.subplots(2,3,figsize=(12.4,7.8),squeeze=False)
    for row,material in enumerate(['PVDF','PDMS']):
        part=data.loc[data.material.eq(material)]
        for system,g in part.groupby('system'):
            g=g.sort_values('temperature_K');idx,label=SIZES[system]
            for col,(mean,sd,scale) in enumerate([('storage_MPa_mean','storage_MPa_sample_SD',1000),
                                                 ('loss_MPa_mean','loss_MPa_sample_SD',1000),
                                                 ('tan_delta_mean','tan_delta_sample_SD',1)]):
                ax=axes[row,col]
                good=~g.any_storage_interval_contains_zero if col==2 else np.ones(len(g),bool)
                ax.errorbar(g.loc[good,'temperature_K'],g.loc[good,mean]/scale,
                            yerr=g.loc[good,sd]/scale,color=COLORS[idx],marker='o',ms=3,
                            lw=1,capsize=2,label=label)
                if col==2 and np.any(~good):
                    ax.scatter(g.loc[~good,'temperature_K'],g.loc[~good,mean],marker='x',s=40,
                               color=COLORS[idx],zorder=4)
            axes[row,0].text(.03,.96,material,transform=axes[row,0].transAxes,va='top',fontsize=11,fontweight='bold')
        axes[row,0].legend(fontsize=7,loc='lower left')
        for col in range(3): axes[row,col].set_xlabel('Target temperature (K)')
    for ax,label in zip(axes[0],[r'Storage $M_1^\prime$ (GPa)',r'Loss $M_1^{\prime\prime}$ (GPa)',r'$\tan\delta_1=M_1^{\prime\prime}/M_1^\prime$']):ax.set_title(label,fontsize=10)
    finish(axes)
    fig.suptitle('Native 25 GHz constrained longitudinal response at 1% strain',fontsize=13)
    fig.text(.07,.025,'Means ± sample SD across three prepared packings; 400–2400 ps retained. Lines guide the eye.\n'
             'Crosses flag a group containing a storage interval crossing zero; its tangent is unresolved. No fitted peak or transition assignment.',fontsize=8.5)
    fig.subplots_adjust(left=.08,right=.98,top=.88,bottom=.15,hspace=.38,wspace=.27)
    save(fig,'fig03_native_temperature_modern')

def amplitude(amp,warnings):
    part=amp.loc[amp.role.eq('papergrid_packing')]
    warning_keys={(row.system,float(row['T']),float(row.strain)) for _,row in warnings.loc[warnings.replication_kind.eq('packing')].iterrows()}
    for T in [300.,400.]:
        fig,axes=plt.subplots(2,3,figsize=(12.4,7.6),squeeze=False)
        for row,material in enumerate(['PVDF','PDMS']):
            for system,g in part.loc[part['T'].eq(T)].groupby('system'):
                if not system.startswith(material):continue
                g=g.sort_values('strain');idx,label=SIZES[system]
                for col,(field,sd,scale) in enumerate([('M_storage_MPa_mean','M_storage_MPa_SD',1000),
                                                      ('M_loss_MPa_mean','M_loss_MPa_SD',1000),
                                                      ('tan_delta_mean','tan_delta_SD',1)]):
                    bad=np.array([(system,T,float(a)) in warning_keys for a in g.strain]) if col==2 else np.zeros(len(g),bool)
                    axes[row,col].errorbar(100*g.loc[~bad,'strain'],g.loc[~bad,field]/scale,yerr=g.loc[~bad,sd]/scale,
                                          color=COLORS[idx],marker='o',ms=3,lw=1,capsize=2,label=label)
                    if np.any(bad):
                        axes[row,col].scatter(100*g.loc[bad,'strain'],g.loc[bad,field]/scale,color=COLORS[idx],marker='x',s=40,zorder=5)
            axes[row,0].text(.03,.96,material,transform=axes[row,0].transAxes,va='top',fontsize=11,fontweight='bold')
            axes[row,0].legend(fontsize=7,loc='lower right')
            for ax in axes[row]:
                ax.set_xlabel('Peak engineering strain (%)')
                ax.axvline(2,color='.4',ls=':',lw=.8)
        for ax,label in zip(axes[0],[r'Storage $M_1^\prime$ (GPa)',r'Loss $M_1^{\prime\prime}$ (GPa)',r'$\tan\delta_1$']):ax.set_title(label,fontsize=10)
        finish(axes)
        fig.suptitle(f'25 GHz amplitude assessment at {T:g} K',fontsize=13)
        fig.text(.07,.025,'Means ± sample SD across three packings; matched first 15 cycles (600 ps), total stress.\n'
                 'The 2% guide is a candidate operating amplitude, not a linearity boundary. Crosses flag groups with unresolved tangent denominators.',fontsize=8.5)
        fig.subplots_adjust(left=.08,right=.98,top=.88,bottom=.15,hspace=.38,wspace=.27)
        save(fig,f'fig05_amplitude_modern_T{int(T)}')
    fig,axes=plt.subplots(4,2,figsize=(9.6,11.5),squeeze=False)
    for row,(material,T) in enumerate([('PVDF',300.),('PVDF',400.),('PDMS',300.),('PDMS',400.)]):
        for system,g in part.loc[part['T'].eq(T)].groupby('system'):
            if not system.startswith(material):continue
            idx,label=SIZES[system];g=g.sort_values('strain')
            for col,n in enumerate([2,3]):
                axes[row,col].errorbar(100*g.strain,100*g[f'H{n}_over_H1_mean'],
                                      yerr=100*g[f'H{n}_over_H1_SD'],marker='o',ms=3,
                                      color=COLORS[idx],capsize=2,lw=1,label=label)
        for col,n in enumerate([2,3]):
            axes[row,col].set_ylabel(f'{material}, {T:g} K\n$H_{n}/H_1$ (%)')
            axes[row,col].set_xlabel('Peak engineering strain (%)')
            axes[row,col].axvline(2,color='.4',ls=':',lw=.8)
    axes[0,0].legend(fontsize=7)
    finish(axes);fig.suptitle('Stress higher harmonics in the matched amplitude series',fontsize=13)
    fig.text(.08,.015,'Three-packing mean ± sample SD, total-stress Fourier coefficients, first 15 cycles.\n'
             'Harmonic magnitudes include a positive noise floor; ratios alone do not establish a resolved nonlinear mechanism.',fontsize=8.5)
    fig.subplots_adjust(left=.12,right=.98,top=.94,bottom=.08,hspace=.55,wspace=.3)
    save(fig,'figS_amplitude_harmonics_modern')

def runtime(frame):
    fig,axes=plt.subplots(2,2,figsize=(10.3,7.8),squeeze=False)
    for row,material in enumerate(['PVDF','PDMS']):
        part=frame.loc[frame.material.eq(material)]
        for col,component in enumerate(['storage','loss']):
            for block,color,ls in [(80,'.5','--'),(200,'#0072b2','-'),(400,'#d55e00',':')]:
                g=part.loc[part.block_duration_ps.eq(block)].groupby('retained_cycles')[component+'_relative_larger_reach_pct'].median()
                axes[row,col].plot(g.index,g.values,'o',ls=ls,color=color,lw=1.2,ms=4,label=f'{block} ps blocks')
            axes[row,col].axhline(30,color='.4',lw=.8,ls='-.')
            axes[row,col].set_ylabel(f'{material}: median interval reach (%)')
            axes[row,col].set_xlabel('Retained cycles at 25 GHz')
            axes[row,col].set_title(f'{component.capitalize()} component',fontsize=10)
    axes[0,0].legend(fontsize=8)
    finish(axes);fig.suptitle('Runtime precision diagnostics with block-length sensitivity',fontsize=13)
    fig.text(.08,.025,'Median across 36 histories per polymer; farther conditional 95% interval endpoint divided by the point magnitude.\n'
             'Retained time = cycles/25 ns; add 0.4 ns for the archived initial aperture. The 30% line is a prospective screening target.\n'
             'Precision is not stationarity, experimental accuracy, or validated shorter-duration convergence.',fontsize=8.5)
    fig.subplots_adjust(left=.1,right=.98,top=.9,bottom=.18,hspace=.48,wspace=.32)
    save(fig,'fig06_runtime_precision')

def packing_uncertainty(grid):
    grid=grid.copy();grid['phase_deg']=np.degrees(np.arctan2(grid.imag,grid.real))
    assert grid.phase_deg.min()>-90 and grid.phase_deg.max()<90
    grouped=grid.groupby(['system','T','frequency_GHz']).agg(samples=('case_id','nunique'),
                storage_mean=('real','mean'),storage_SD=('real','std'),loss_mean=('imag','mean'),loss_SD=('imag','std'),
                phase_mean=('phase_deg','mean'),phase_SD=('phase_deg','std')).reset_index()
    assert grouped.samples.eq(3).all()
    grouped['storage_relative_SD_pct']=100*grouped.storage_SD/abs(grouped.storage_mean)
    grouped['loss_relative_SD_pct']=100*grouped.loss_SD/abs(grouped.loss_mean)
    grouped.to_csv(OUT/'papergrid_three_packing_statistics144.csv',index=False)
    summary=[]
    for system,g in grouped.groupby('system'):
        row=dict(system=system,coordinates=len(g),replicas=3)
        for field in ['storage_relative_SD_pct','loss_relative_SD_pct','phase_SD']:
            for suffix,number in [('median',g[field].median()),('min',g[field].min()),
                                  ('max',g[field].max()),('p90',g[field].quantile(.9))]:
                row[field+'_'+suffix]=number
        summary.append(row)
    pd.DataFrame(summary).to_csv(OUT/'repeatability_by_architecture.csv',index=False)
    fig,axes=plt.subplots(2,3,figsize=(12.2,7.3),squeeze=False)
    for row,(material,T) in enumerate([('PVDF',400.),('PDMS',345.)]):
        for system,g in grouped.loc[grouped['T'].eq(T)].groupby('system'):
            if not system.startswith(material):continue
            idx,label=SIZES[system];g=g.sort_values('frequency_GHz')
            for col,field in enumerate(['storage_relative_SD_pct','loss_relative_SD_pct','phase_SD']):
                axes[row,col].plot(g.frequency_GHz,g[field],'-o',ms=3,lw=1,color=COLORS[idx],label=label)
        axes[row,0].text(.03,.96,f'{material}, {T:g} K',transform=axes[row,0].transAxes,va='top',fontweight='bold',fontsize=10)
        axes[row,0].legend(fontsize=7,loc='lower left')
        for ax in axes[row]: ax.set(xscale='log',xlabel='Frequency (GHz)')
    for ax,title in zip(axes[0],['Storage: sample SD / |mean| (%)','Loss: sample SD / |mean| (%)','Phase: sample SD (degrees)']):ax.set_title(title,fontsize=10)
    finish(axes);fig.suptitle('Variation across three independently prepared packings',fontsize=13)
    fig.text(.08,.025,'1% strain, total stress, 400–2400 ps retained. Each point summarizes three packing-specific estimates.\n'
             'These sample standard deviations include structural/initial-condition differences and are not 95% population confidence intervals.',fontsize=8.5)
    fig.subplots_adjust(left=.08,right=.98,top=.88,bottom=.15,hspace=.4,wspace=.3)
    save(fig,'fig10_packing_uncertainty')

def figure2(grid,points):
    fig=plt.figure(figsize=(8.2,6.05))
    layout=fig.add_gridspec(2,5,width_ratios=[1,1,1,.07,1.35],left=.095,right=.985,
                           bottom=.205,top=.88,wspace=.52,hspace=.58)
    loop_rows=[];maximum_replay=0.
    for row,(material,Tref) in enumerate([('PVDF',400.),('PDMS',345.)]):
        systems=sorted([s for s in grid.system.unique() if s.startswith(material)],key=lambda s:ORDER[s])
        materialgrid=grid.loc[grid.system.isin(systems)]
        lo,hi=materialgrid.M_storage_GPa.min(),materialgrid.M_storage_GPa.max()
        axes=[]
        for col,system in enumerate(systems):
            ax=fig.add_subplot(layout[row,col]);axes.append(ax)
            g=grid.loc[grid.system.eq(system)];temperatures=sorted(g['T'].unique());frequencies=sorted(g.frequency_GHz.unique())
            matrix=g.pivot(index='T',columns='frequency_GHz',values='M_storage_GPa').loc[temperatures,frequencies]
            im=ax.imshow(matrix,origin='lower',aspect='auto',cmap='viridis',vmin=lo,vmax=hi)
            ax.set_xticks(np.arange(len(frequencies)),[f'{f:g}' for f in frequencies],rotation=55,ha='right',fontsize=7.6)
            ax.set_yticks(np.arange(len(temperatures)),[f'{t:g}' for t in temperatures],fontsize=7.6)
            if col!=0:ax.tick_params(labelleft=False)
            else:
                ax.set_ylabel('Temperature (K)',fontsize=8.4)
                ax.text(.04,.96,material,transform=ax.transAxes,va='top',color='white',fontsize=8.2,fontweight='bold')
            counts=['200 × 10','20 × 100','5 × 400'] if material=='PVDF' else ['100 × 10','40 × 30','12 × 100']
            ax.set_title(['Short','Intermediate','Long'][col]+'\n'+counts[col],fontsize=8.5)
            ax.set_xlabel('Frequency (GHz)',fontsize=8.2)
        colorax=fig.add_subplot(layout[row,3]);cb=fig.colorbar(im,cax=colorax);cb.ax.set_title('GPa',fontsize=7.5,pad=5);cb.ax.tick_params(labelsize=7)
        ax=fig.add_subplot(layout[row,4]);theta=np.linspace(0,2*np.pi,241);strain=.01*np.sin(theta)
        for system in systems:
            g=points.loc[points.system.eq(system)&points['T'].eq(Tref)&points.frequency_GHz.eq(25)]
            assert len(g)==3 and g.strain.eq(.01).all()
            real,imag=g.M_real_MPa.mean(),g.M_imag_MPa.mean()
            mean=grid.loc[grid.system.eq(system)&grid['T'].eq(Tref)&grid.frequency_GHz.eq(25)].iloc[0].M_storage_GPa
            maximum_replay=max(maximum_replay,abs(real/1000-mean))
            sigma=.01*(real*np.sin(theta)+imag*np.cos(theta));idx=ORDER[system]
            ax.plot(strain,sigma,lw=1.2,color=COLORS[idx],label=['Short','Intermediate','Long'][idx])
            ax.plot([-.01,.01],[-.01*real,.01*real],ls=':',lw=.8,color=COLORS[idx])
            loop_rows.extend(dict(material=material,system=system,T_K=Tref,frequency_GHz=25,
                                  strain_dimensionless=float(e),stress_H1_MPa=float(s),
                                  M_mean_storage_MPa=real,M_mean_loss_MPa=imag,
                                  reconstruction=True,offset_removed_by_construction=True) for e,s in zip(strain,sigma))
        ax.set_title(f'{Tref:g} K · 25 GHz\nH1 stress (MPa)',fontsize=8.1)
        ax.set_xlabel('Engineering strain\n'+r'$\epsilon_{zz}$ (dimensionless)',fontsize=7.8)
        ax.set_xticks([-.01,0,.01],['−0.01','0','0.01'])
        ax.tick_params(labelsize=7.5);ax.grid(alpha=.2)
        ax.legend(fontsize=7,loc='lower right',framealpha=.85)
    assert maximum_replay<1e-8
    fig.suptitle('Constrained longitudinal storage and first-harmonic response',fontsize=11.2,y=.985)
    fig.text(.065,.035,'Heatmaps: three-packing means at measured coordinates; categorical cell spacing, no interpolation.\n'
             'Loops: 1% nominal strain, zero-offset constructions from packing-mean M*; dotted lines have slope M\N{PRIME}.\n'
             'The ellipse major axis is not a Young modulus. All source estimates retain 400–2400 ps; these are not raw full waveforms.',fontsize=7.5)
    save(fig,'fig02_constrained_M_heatmaps_and_H1')
    pd.DataFrame(loop_rows).to_csv(OUT/'fig02_constructed_longitudinal_H1_points.csv',index=False)
    grid[['system','T','frequency_GHz','packings','M_storage_GPa','M_storage_packing_SD_GPa']].to_csv(OUT/'fig02_measured_Mprime_grid144.csv',index=False)
    return maximum_replay

def figure7(modern,context):
    m=modern.loc[modern['T'].isin([300.,400.])].copy()
    assert len(m)==18 and m.mechanical_strain.eq(.01).all()
    rows=[]
    for _,r in m.iterrows():
        rows.append(dict(system=r.system,T_K=float(r['T']),packing=int(r.packing),
                         nominal_VDF_units={0:10,1:100,2:400}[ORDER[r.system]],
                         chains={0:200,1:20,2:5}[ORDER[r.system]],
                         density_g_cm3=r.density_g_cm3,M_storage_GPa=r.mechanical_storage_MPa/1000,
                         preparation_ps=r.source_preparation_ps,legacy=False,
                         prepared_case_id=r.case_id,mechanical_case_id=r.mechanical_case_id,
                         density_role='prepared endpoint before NEMD',
                         source_table='matched_PVDF_common_temperature_source_rows.csv'))
    legacy=context.loc[context.system.eq('large')&context.operator.eq('tot')&context['T'].isin([300.,400.])]
    assert len(legacy)==2
    for _,r in legacy.iterrows():
        rows.append(dict(system='legacy_two_long_chains',T_K=float(r['T']),packing=int(r.packing),
                         nominal_VDF_units=1601,chains=2,density_g_cm3=r.density_g_cm3,
                         M_storage_GPa=r.M_storage_MPa/1000,preparation_ps=r.source_preparation_ps,
                         legacy=True,prepared_case_id=r.parent_case,mechanical_case_id=r.case_id,
                         density_role='reference-cell density from saved mechanical state',
                         source_table='mechanical_acoustic_context.csv'))
    data=pd.DataFrame(rows);data.to_csv(OUT/'fig07_architecture_state_source_points20.csv',index=False)
    summary=data.groupby(['system','T_K','nominal_VDF_units','chains','legacy']).agg(
        samples=('packing','count'),density_mean=('density_g_cm3','mean'),density_SD=('density_g_cm3','std'),
        M_storage_mean_GPa=('M_storage_GPa','mean'),M_storage_SD_GPa=('M_storage_GPa','std')).reset_index()
    summary.to_csv(OUT/'fig07_architecture_state_summary8.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(7.8,4.25))
    for T,color in [(300.,'#0072b2'),(400.,'#d55e00')]:
        for ax,mean,sd in zip(axes,['density_mean','M_storage_mean_GPa'],['density_SD','M_storage_SD_GPa']):
            g=summary.loc[summary.T_K.eq(T)&~summary.legacy].sort_values('nominal_VDF_units')
            ax.errorbar(g.nominal_VDF_units,g[mean],yerr=g[sd],marker='o',ms=4,lw=1.2,capsize=3,color=color,label=f'Modern, {T:g} K')
            old=summary.loc[summary.T_K.eq(T)&summary.legacy]
            ax.scatter(old.nominal_VDF_units,old[mean],marker='D',s=35,facecolors='none',edgecolors=color,
                       label=f'Legacy 2-chain, {T:g} K',zorder=5)
            ax.set_xscale('log');ax.set_xticks([10,100,400,1601],['10\n200 chains','100\n20 chains','400\n5 chains','1601\n2 chains'],fontsize=8)
            ax.set_xlabel('Nominal VDF units per chain',fontsize=9);ax.grid(alpha=.2);ax.tick_params(labelsize=8)
    axes[0].set_ylabel(r'Reference density (g cm$^{-3}$)',fontsize=9)
    axes[1].set_ylabel(r'25 GHz $M_1^\prime$ (GPa)',fontsize=9)
    for ax in axes:ax.set_xlim(7,2200)
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,.125),ncol=2,fontsize=8,frameon=False)
    fig.suptitle('PVDF architecture and prepared-state dependence',fontsize=11.5)
    fig.text(.065,.018,'Modern: mean ± sample SD of three packings, 3.35 ns preparation. Legacy: one packing, ~5.5 ns history; no SD assigned.\n'
             '1% strain; 400–2400 ps retained. Chain count, end concentration and preparation co-vary; this is not an entanglement-convergence test.',fontsize=7.1)
    fig.subplots_adjust(left=.095,right=.985,bottom=.35,top=.87,wspace=.3)
    save(fig,'fig07_architecture_density_storage')

def sampling(frame):
    frame=frame[(frame.signal=='tot_zz') & (frame.logging_interval_fs>2)].copy()
    assert set(frame.strain_percent)=={.25,.5,1.,2.,4.,6.}
    plt.rcParams.update({'font.size':8.3, 'axes.titlesize':9, 'axes.labelsize':8.5,
                         'font.family':'DejaVu Sans', 'pdf.fonttype':42})
    fig,axes=plt.subplots(2,2,figsize=(7.3,5.35))
    metrics=[('storage_modulus',r'$|\Delta M_1^{\prime}|$ (MPa)'),
             ('loss_modulus',r'$|\Delta M_1^{\prime\prime}|$ (MPa)'),
             ('loss_tangent',r'$|\Delta\tan\delta_1|$'),
             ('phase',r'$|\Delta\delta_1|$ (degrees)')]
    colors={4.:'#009E73',10.:'#E69F00',100.:'#D55E00'}
    for ax,(metric,label) in zip(axes.flat,metrics):
        for cadence,color in colors.items():
            g=frame[(frame.metric==metric)&(frame.logging_interval_fs==cadence)].sort_values('strain_percent')
            assert len(g)==6
            for key,style,marker in [('actual_absolute_offset','-','o'),('worst_absolute_offset','--',None)]:
                ax.plot(g.strain_percent,g[key],color=color,ls=style,marker=marker,ms=3,lw=1.25)
        ax.set_yscale('log');ax.set_ylabel(label);ax.set_xlabel('Strain amplitude (%)')
        ax.set_xticks([.25,1,2,4,6]);ax.grid(alpha=.22,which='both')
    handles=[Line2D([],[],color=c,lw=1.6,label=f'{int(k)} fs') for k,c in colors.items()]
    handles += [Line2D([],[],color='.25',marker='o',ms=3,label='Actual acquisition clock'),
                Line2D([],[],color='.25',ls='--',label='Worst tested clock offset')]
    fig.suptitle('Acquisition offsets depend on strain and the reported component', y=.985,fontsize=11)
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.944),ncol=3,frameon=False,fontsize=8)
    fig.subplots_adjust(left=.12,right=.98,bottom=.18,top=.82,wspace=.31,hspace=.35)
    fig.text(.025,.022,'Same saved historical short-PVDF trajectories: 300 K, 25 GHz, 50 cycles, total stress.\n'
             'Offsets from each trajectory’s full 2 fs record; 2 fs is the zero reference, not an absolute-error estimate.\n'
             'Clock offsets are correlated subsamples, not independent replicas or confidence limits.',fontsize=8)
    for suffix in ['png','pdf']:
        fig.savefig(OUT/f'fig09_sampling_component_offsets.{suffix}',dpi=300)
    plt.close(fig)
    frame.to_csv(OUT/'fig09_sampling_offsets_source_subset.csv',index=False)

def acoustic(out, hold):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.labelsize':8.2,
                         'axes.titlesize':8.8,'xtick.labelsize':7.6,'ytick.labelsize':7.6,
                         'axes.spines.top':False,'axes.spines.right':False,
                         'pdf.fonttype':42,'ps.fonttype':42})
    fig, axes = plt.subplots(1,3,figsize=(7.0,2.92),gridspec_kw={'width_ratios':[1,1,1]})
    fig.subplots_adjust(left=.075,right=.985,bottom=.19,top=.72,wspace=.42)
    colors = {'historical':'#626262','native':'#2878B5','imposed':'#D87815'}
    style = {}
    for q in out:
        if q['source_kind'] == 'historical':
            color,marker = colors['historical'],'s'
        elif 'imposed density' in q['category']:
            color,marker = colors['imposed'],'^'
        else:
            color,marker = colors['native'],'o'
        style[q['case_id']] = (color,marker)
        f = float(q['frequency_GHz'])
        for ax,key,lo,hi,scale in [
            (axes[0],'storage_MPa','storage_CI_low_MPa','storage_CI_high_MPa',1000),
            (axes[1],'phase_velocity_m_s','phase_velocity_m_s_low','phase_velocity_m_s_high',1)]:
            value = float(q[key])/scale
            err = [[value-float(q[lo])/scale],[float(q[hi])/scale-value]]
            ax.errorbar(f,value,yerr=err,color=color,marker=marker,ms=4.8,
                        capsize=2.2,lw=1.0,linestyle='none',zorder=3)
    for ax in axes[:2]:
        ax.set_xlabel('Frequency (GHz)')
        ax.set_xscale('log')
        ax.set_xlim(5.8,29)
        ax.set_xticks([7.15,10,25],labels=['7.15','10','25'])
        ax.minorticks_off()
    axes[0].set_ylabel(r"$M_1'$ (GPa)")
    axes[0].set_ylim(.45,3.13)
    axes[0].axhline(2.217716,color='#222222',ls='--',lw=.95,zorder=1)
    axes[1].set_ylabel(r'$c_{\mathrm{phase}}$ equivalent (m s$^{-1}$)')
    axes[1].set_ylim(760,1460)
    axes[1].axhline(1220,color='#222222',ls='--',lw=.95,zorder=1)

    for q in hold:
        role = q['role']
        color,marker = ((colors['imposed'],'^') if role == 'density_matched' else (colors['native'],'o'))
        x = float(q['density_mean_g_cm3'])*1000
        y = float(q['pressure_mean_MPa'])
        axes[2].plot(x,y,marker=marker,color=color,ms=5.5,ls='none')
    axes[2].set_xlabel(r'Reference $\rho_0$ (kg m$^{-3}$)')
    axes[2].set_ylabel('Conditioning pressure (MPa)')
    axes[2].set_xlim(1315,1518)
    axes[2].set_xticks([1343,1490])
    axes[2].set_ylim(-8,113)
    axes[2].axhline(0,color='.8',lw=.6,zorder=0)
    for index, ax in enumerate(axes):
        ax.text(-.20,1.04,'abc'[index],transform=ax.transAxes,fontweight='bold',fontsize=10)
        ax.grid(axis='y',alpha=.18,lw=.6)

    legend = [Line2D([],[],color=colors['historical'],marker='s',ls='none',ms=4.5,label='Earlier native state: 10/25 GHz'),
              Line2D([],[],color=colors['native'],marker='o',ls='none',ms=4.5,label='Equal-age native density: 7.15 GHz'),
              Line2D([],[],color=colors['imposed'],marker='^',ls='none',ms=4.5,label='Imposed density: 7.15 GHz'),
              Line2D([],[],color='#222222',ls='--',lw=.95,label='Sato melt reference; bulk frequency unspecified')]
    fig.legend(handles=legend,loc='upper center',bbox_to_anchor=(.51,1.005),ncol=2,
               frameon=False,fontsize=7.1,columnspacing=1.5,handletextpad=.55)
    for extension in ['png','pdf']:
        fig.savefig(OUT/('Figure11_native_acoustic_comparison.'+extension),dpi=350,
                    metadata={'Title':'PVDF native longitudinal acoustic comparison'} if extension=='pdf' else None)
    plt.close(fig)


def shared_clock_counterexample(points, fits):
    """Display saved shifts on saved means; this performs no optimization."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), layout='constrained')
    for column, (variant, label) in enumerate([
            ('storage_H', 'Shift chosen using storage only'),
            ('loss_H', 'Shift chosen using loss only'),
            ('shared_H', 'One shift chosen using both')]):
        selected = fits.loc[fits.variant.eq(variant)]
        assert len(selected) == 1
        q = selected.iloc[0].q_log10_a_other_over_ref
        for temperature, color in [(225., '0.25'), (250., 'C2')]:
            part = points.loc[points.variant.eq(variant) & points.T_K.eq(temperature)].sort_values('frequency_GHz')
            assert len(part) == 6
            for row, (field, symbol) in enumerate([('storage_MPa', 'M′'), ('loss_MPa', 'M″')]):
                axes[row, column].semilogx(part.reduced_frequency_GHz, part[field]/1000,
                    'o-', color=color, label=f'{temperature:g} K'+(' reference' if temperature == 225 else ' shifted'))
                axes[row, column].set(ylabel=f'{symbol} (GPa)', xlabel='Reduced frequency aT f (GHz)')
                axes[row, column].grid(alpha=.2)
        axes[0, column].set_title(f'{label}\nq = {q:.4f}; aT = {10**q:.3f}')
    axes[0, 0].legend()
    fig.suptitle('Why separate successful alignments do not establish a master curve\n'
        'Intermediate PDMS (40 chains × 30 internal siloxane units): 225 → 250 K; 1% axial strain; total M*', fontsize=14)
    fig.supxlabel('Fits use the same fixed reference anchors at 25, 50 and 100 GHz, b = 1. Lines connect measured points only.\n'
        'All three optima are interior to measured support. These are conditional finite-age responses, not accepted TTS or a Tg fit.', fontsize=10)
    for extension in ['pdf', 'png']:
        fig.savefig(OUT/f'PDMS_shared_clock_counterexample.{extension}', dpi=180, bbox_inches='tight')
    plt.close(fig)


def native_loss_censoring(grid):
    """Plot measured means and their sample SD, without fitting a peak."""
    means = grid.groupby(['system', 'T', 'frequency_GHz']).agg(
        loss_mean_MPa=('imag', 'mean'), loss_sample_SD_MPa=('imag', 'std'),
        packings=('case_id', 'nunique')).reset_index()
    assert len(means) == 144 and means.packings.eq(3).all()
    individual = grid.loc[grid.groupby(['system', 'T', 'packing']).imag.idxmax()]
    maxima = means.loc[means.groupby(['system', 'T']).loss_mean_MPa.idxmax()]
    assert len(individual) == 72 and individual.frequency_GHz.eq(500).all()
    assert len(maxima) == 24 and maxima.frequency_GHz.eq(500).all()
    means['sampled_frequency_peak_at_upper_edge'] = True
    means['fitted_alpha_peak'] = False
    means.to_csv(OUT/'native_frequency_loss_summary144.csv', index=False)
    fig, axes = plt.subplots(2, 3, figsize=(9.0, 5.7))
    for row, material in enumerate(['PVDF', 'PDMS']):
        systems = sorted([s for s in means.system.unique() if s.startswith(material)], key=lambda s:ORDER[s])
        for col, system in enumerate(systems):
            ax = axes[row, col]
            part = means.loc[means.system.eq(system)]
            for color, (temperature, group) in zip(plt.cm.viridis(np.linspace(.1, .8, 4)), part.groupby('T')):
                group = group.sort_values('frequency_GHz')
                ax.errorbar(group.frequency_GHz, group.loss_mean_MPa/1000,
                    yerr=group.loss_sample_SD_MPa/1000, color=color, marker='o', ms=3,
                    capsize=2, lw=1, label=f'{temperature:g} K')
            ax.axvline(500, color='.4', ls=':', lw=.8)
            ax.set(xscale='log', xlim=(8, 620), xlabel='Frequency (GHz)')
            ax.set_title(f'{material}: '+['short','intermediate','long'][col], fontsize=9)
            ax.grid(alpha=.18);ax.legend(fontsize=6.5, loc='upper left')
            if col == 0:ax.set_ylabel(r'Loss $M_1^{\prime\prime}$ (GPa)')
            ax.tick_params(labelsize=7.5)
    fig.suptitle('Native loss-frequency maxima remain upper-edge censored', fontsize=12)
    fig.text(.07, .025, 'Mean ± sample SD of three packings; 1% strain, 400–2400 ps retained, total stress.\n'
        'All 24 mean curves and all 72 packing-specific curves have their sampled maximum at 500 GHz. No fitted α peak.', fontsize=8)
    fig.subplots_adjust(left=.08, right=.99, top=.88, bottom=.16, hspace=.47, wspace=.30)
    save(fig, 'figS_native_frequency_loss_censoring')


def verify_sources(data):
    """Verify frozen source bytes before plotting; the full bundle is not MD."""
    manifest = json.loads((data / 'source_manifest.json').read_text())
    for entry in manifest['files']:
        path = data / entry['file']
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry['sha256']:
            raise ValueError(f"Checksum mismatch: {entry['file']}")
    return manifest


def reproduce(data, output):
    global OUT
    data = Path(data).resolve()
    OUT = Path(output).resolve()
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = verify_sources(data)
    source = data / 'sources'
    def load(name):
        return pd.read_csv(source / name, low_memory=False)
    grid = load('papergrid_complex432.csv')
    assert len(grid) == 432 and grid.strain.eq(.01).all()
    assert grid.case_id.nunique() == 432
    with plt.rc_context({'font.size':9, 'pdf.fonttype':42}):
        replay = figure2(load('papergrid_means144.csv'), load('papergrid_complex_tensor432.csv'))
        temperature(load('temperature_summary.csv'))
        amplitude(load('amplitude_summary126.csv'), load('amplitude_tangent_warnings.csv'))
        runtime(load('runtime_prefix25GHz.csv'))
        figure7(load('architecture_modern63.csv'), load('architecture_legacy_context.csv'))
        sampling(load('logging_metrics.csv'))
        packing_uncertainty(grid)
        shared_clock_counterexample(load('PDMS_shared_clock_counterexample_points.csv'), load('PDMS_shared_clock_counterexample_fits.csv'))
        native_loss_censoring(grid)
        with (source / 'Figure11_source_rows.csv').open() as handle:
            acoustic_rows = list(csv.DictReader(handle))
        with (source / 'SI_715GHz_premeasurement_final_hold.csv').open() as handle:
            hold_rows = list(csv.DictReader(handle))
        acoustic(acoustic_rows, hold_rows)
    aux_spec = importlib.util.spec_from_file_location('dmax_paper_aux', Path(__file__).with_name('reproduce_aux_figures.py'))
    aux = importlib.util.module_from_spec(aux_spec)
    aux_spec.loader.exec_module(aux)
    with plt.rc_context({'font.size':8.5, 'pdf.fonttype':42, 'ps.fonttype':42}):
        aux.prep(source, OUT)
        aux.shifts(source, OUT)
        aux.workflow(OUT)
        aux.sampling(source, OUT)
    figures = sorted(OUT.glob('*.png'))
    report = {
        'source_manifest_sha256': hashlib.sha256((data / 'source_manifest.json').read_bytes()).hexdigest(),
        'source_tables_verified': len(manifest['files']),
        'maximum_Mprime_mean_replay_error_GPa': replay,
        'figure_pngs': [p.name for p in figures],
        'generated_tables': [p.name for p in sorted(OUT.glob('*.csv'))],
        'raw_trace_reads': False, 'new_fits_or_bootstrap': False,
        'MD_performed': False, 'new_scientific_acceptance': False,
    }
    (OUT / 'reproduction_report.json').write_text(json.dumps(report, indent=2)+'\n')
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path(__file__).resolve().parent,
                        help='Exported bundle directory containing sources and source_manifest.json')
    parser.add_argument('--output', type=Path, required=True,
                        help='Directory for regenerated PNG/PDF figures and reduced plot tables')
    args = parser.parse_args(argv)
    report = reproduce(args.data, args.output)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
