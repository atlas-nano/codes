"""Portable figures from reduced source tables; no fitting or raw MD reads."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

SYSTEMS={'PVDF':['PVDF_n9_c200','PVDF_n99_c20','PVDF_n399_c5'],
         'PDMS':['PDMS_u11_c100','PDMS_u31_c40','PDMS_u101_c12']}
COLORS=['#0072b2','#d55e00','#009e73']

def save(fig,out,stem):
 for ext in ['pdf','png']:fig.savefig(out/f'{stem}.{ext}',dpi=300,facecolor='white')
 plt.close(fig)

def prep(data,out):
 t=pd.read_csv(data/'preparation_thermo_50ps.csv');g=pd.read_csv(data/'preparation_geometry.csv')
 fig,axes=plt.subplots(2,3,figsize=(8.2,5.65),squeeze=False)
 for row,(material,T) in enumerate([('PVDF',338.15),('PDMS',325.)]):
  for i,system in enumerate(SYSTEMS[material]):
   part=t.loc[t.system.eq(system)&t['T'].eq(T)]
   for cid,c in part.groupby('case_id'):
    c=c.sort_values('mid_age_ps')
    axes[row,0].plot(c.mid_age_ps/1000,c.density_g_cm3,color=COLORS[i],alpha=.3,lw=.65)
    n=c.loc[c.phase.eq('NVT')];base=n.loc[n.stop_age_ps.le(3100)].potential_energy_specific_kJ_kg.mean()
    axes[row,1].plot(n.mid_age_ps-2850,n.potential_energy_specific_kJ_kg-base,color=COLORS[i],alpha=.35,lw=.8)
    a=g.loc[g.case_id.eq(cid)].sort_values('physical_age_ps')
    ref=a.loc[a.physical_age_ps.gt(2850)&a.physical_age_ps.le(3100)].mean_Rg_A.mean()
    axes[row,2].plot(a.physical_age_ps-2850,100*(a.mean_Rg_A/ref-1),color=COLORS[i],alpha=.35,lw=.7)
   for ax in axes[row]:ax.plot([],[],color=COLORS[i],label=['Short','Intermediate','Long'][i])
  axes[row,0].axvspan(2.1,2.85,color='.5',alpha=.06);axes[row,0].axvline(2.85,color='.4',ls=':',lw=.7)
  axes[row,0].set_xlabel('Preparation age (ns)')
  for ax in axes[row,1:]:ax.set_xlabel('Final NVT hold (ps)');ax.axhline(0,color='.5',ls='--',lw=.6)
  axes[row,0].set_ylabel(f'{material}, {T:g} K\nDensity (g cm$^{{-3}}$)')
  axes[row,1].set_ylabel('Potential energy change\n(kJ kg$^{-1}$)')
  axes[row,2].set_ylabel('Mean $R_g$ change (%)')
 axes[0,0].legend(fontsize=6.8,loc='lower left')
 for ax,title in zip(axes[0],['(a) NPT to fixed-cell NVT','(b) Configurational energy','(c) Backbone dimensions']):ax.set_title(title,fontsize=9)
 for ax in axes.flat:ax.grid(alpha=.18);ax.tick_params(labelsize=7.5)
 fig.subplots_adjust(left=.10,right=.99,bottom=.11,top=.93,wspace=.47,hspace=.52)
 save(fig,out,'Figure01_preparation')

def shifts(data,out):
 fits=pd.read_csv(data/'tts_fits.csv');ints=pd.read_csv(data/'tts_intervals.csv')
 fits=fits.loc[fits.cohort.eq('mean_3_packings')&fits.anchor_window.eq(1)&fits.status.eq('fit_completed')]
 ints=ints.loc[ints.cohort.eq('mean_3_packings')&ints.anchor_window.eq(1)&ints.status.eq('propagation_completed')]
 fig,axes=plt.subplots(2,3,figsize=(8.2,5.6),squeeze=False)
 for row,material in enumerate(['PVDF','PDMS']):
  for col,system in enumerate(SYSTEMS[material]):
   ax=axes[row,col];part=fits.loc[fits.system.eq(system)]
   for variant,color,marker,label in [('storage_H',COLORS[0],'o','Storage only'),('loss_H',COLORS[1],'s','Loss only'),('shared_H','.15','D','Shared')]:
    p=part.loc[part.variant.eq(variant)].sort_values('other_T_K')
    ax.plot(p.other_T_K,p.q_log10_a_other_over_ref,color=color,marker=marker,lw=.9,ms=3,label=label)
    if variant=='shared_H':
     q=ints.loc[ints.system.eq(system)&ints.variant.eq(variant)]
     ax.vlines(q.other_T_K,q.q025,q.q975,color=color,lw=.9,alpha=.55)
   ax.axhline(0,color='.6',ls=':',lw=.7);ax.grid(alpha=.18);ax.tick_params(labelsize=7.5)
   ax.set_title(f'{material}: '+['short','intermediate','long'][col],fontsize=9)
   ax.set_xlabel('Other isotherm temperature (K)',fontsize=8)
   if col==0:ax.set_ylabel(r'$\log_{10} a_T$ (candidate)')
 axes[0,0].legend(fontsize=7)
 fig.subplots_adjust(left=.1,right=.99,bottom=.12,top=.94,hspace=.52,wspace=.32)
 save(fig,out,'Figure04_TTS_diagnostics')

def workflow(out):
 fig,ax=plt.subplots(figsize=(8.2,5.25));ax.set_xlim(0,12);ax.set_ylim(-.2,8);ax.axis('off')
 nodes=[(0.2,6,'User inputs','Repeat SMILES / monomer\nor parameterized structure'),
        (4.2,6,'Chemical model','Explicit caps, types, charges\nand force-field requirements'),
        (8.2,6,'Independent structures','Build / pack seeds\narchitecture records'),
        (8.2,4,'Conditioning','Hot + target NPT/NVT\nthermodynamic / chain checks'),
        (4.2,4,'Amplitude assessment','Fundamental precision\nand higher harmonics'),
        (0.2,4,'Duration assessment','Complete cycles + physical time\nuncertainty / window consistency'),
        (0.2,2,'Temperature–frequency grid','Same declared loading protocol\nindependent packing identities'),
        (4.2,2,'Measured observables','M′, M″, tanδ, phase\ntotal / virial; H2, H3, …'),
        (8.2,2,'Native comparisons','Longitudinal acoustic scale\ndistinct dielectric observables'),
        (8.2,0,'Conditional extrapolation','Shared-shift / overlap tests\nno accepted Tg if support fails')]
 for i,(x,y,title,body) in enumerate(nodes):
  ax.add_patch(FancyBboxPatch((x,y),3.5,1.45,boxstyle='round,pad=.05',fc='#edf4fa' if i<9 else '#fff2df',ec='#284d73',lw=1))
  ax.text(x+1.75,y+1.1,title,ha='center',va='center',fontsize=8.6,fontweight='bold')
  ax.text(x+1.75,y+.52,body,ha='center',va='center',fontsize=8,linespacing=1.4)
 links=[((3.7,6.72),(4.15,6.72)),((7.7,6.72),(8.15,6.72)),((9.95,6),(9.95,5.5)),
        ((8.15,4.72),(7.75,4.72)),((4.15,4.72),(3.75,4.72)),((1.95,4),(1.95,3.5)),
        ((3.7,2.72),(4.15,2.72)),((7.7,2.72),(8.15,2.72)),((9.95,2),(9.95,1.5))]
 for a,b in links:ax.annotate('',xy=b,xytext=a,arrowprops=dict(arrowstyle='->',lw=1.2,color='#284d73'))
 ax.text(.2,.6,'Every stage records inputs, seeds, versions, source identity,\nand separate numerical / scientific qualification status.',fontsize=9,ha='left')
 fig.subplots_adjust(left=.01,right=.99,bottom=.02,top=.99);save(fig,out,'Figure08_workflow')

def sampling(data,out):
 d=pd.read_csv(data/'sampling_pilot6.csv');d=d.sort_values(['system','frequency_GHz'])
 fig,axes=plt.subplots(1,2,figsize=(8.2,3.5));x=np.arange(len(d));w=.34
 labels=[f'{r.system}\n{r.frequency_GHz:g} GHz' for r in d.itertuples()]
 for ax,field,unit,title in [(axes[0],'complex_MPa','Complex-modulus offset (MPa)','(a) Modulus acquisition sensitivity'),(axes[1],'phase_deg','Phase offset (degrees)','(b) Phase acquisition sensitivity')]:
  ax.bar(x-w/2,d['max_2fs_'+field],w,color=COLORS[0],label='2 fs')
  ax.bar(x+w/2,d['max_100fs_'+field],w,color=COLORS[1],label='100 fs')
  ax.set_yscale('log');ax.set_xticks(x,labels,rotation=45,ha='right',fontsize=6.7)
  ax.set_ylabel(unit,fontsize=8.2);ax.set_title(title,fontsize=8.8);ax.grid(axis='y',alpha=.2)
  ax.tick_params(axis='y',labelsize=7.5)
 axes[0].legend(fontsize=7)
 fig.subplots_adjust(left=.1,right=.99,bottom=.31,top=.9,wspace=.36);save(fig,out,'Figure09_sampling')

def main():
 p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
 a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
 with plt.rc_context({'font.size':8.5,'pdf.fonttype':42,'ps.fonttype':42}):
  prep(a.data,a.output);shifts(a.data,a.output);workflow(a.output);sampling(a.data,a.output)

if __name__=='__main__':main()
