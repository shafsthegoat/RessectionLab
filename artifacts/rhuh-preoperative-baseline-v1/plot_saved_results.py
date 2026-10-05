"""Render retained predictions/metrics; never import a model or fit data."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

HERE = Path(__file__).resolve().parent
data = json.loads((HERE/'compact-summary.json').read_text())
plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':10,
                     'pdf.fonttype':42, 'svg.fonttype':'none', 'axes.titleweight':'bold'})
fig, (left, right) = plt.subplots(1, 2, figsize=(11.5, 6.8), gridspec_kw={'width_ratios':[1,1.18]})
fig.subplots_adjust(left=.18, right=.96, top=.79, bottom=.24, wspace=.62)
fig.suptitle('Preoperative volume did not improve the primary Brier score',
             x=.06, y=.96, ha='left', fontsize=16, weight='bold')
fig.text(.06,.902,'RHUH-GBM · fixed leave-one-out analysis · 40 patients (14 recorded deficits, 26 “No”)',
         fontsize=11, color='#435467')
colors = ['#7B8794','#007F87','#AD6D25']
keys = ['FOLD_PREVALENCE','KPS','KPS_CE_VOLUME']
labels = ['Training-fold prevalence','Preoperative KPS','KPS + enhancing volume']
values = [data['observed']['models'][key]['brier'] for key in keys]
for y,(v,c) in enumerate(zip(values,colors)):
    left.hlines(y,0,v,color=c,alpha=.20,lw=4)
    left.scatter(v,y,s=70,color=c,zorder=3)
    left.text(v+.006,y,f'{v:.5f}',va='center',fontsize=10)
left.set(yticks=range(3),yticklabels=labels,ylim=(2.5,-.5),xlim=(0,.295),
         xlabel='Brier score (lower is better)',title='A   Primary metric · all 40 patients')
left.xaxis.set_major_locator(MultipleLocator(.05))
left.text(0,-.17,'Volume − KPS: +0.00120647\nPositive difference means worse with volume.',
          transform=left.transAxes,fontsize=10,color='#435467',va='top')
records=data['influence']['records']
delta=data['observed']['paired_brier']['KPS_CE_VOLUME_minus_KPS']
right.axvline(0,color='#65717E',lw=.8)
right.axvline(delta,color='#AD6D25',lw=1,ls='--',alpha=.85)
right.scatter([delta],[-1],s=60,marker='D',color='#172B4D',zorder=4)
for y,row in enumerate(records):
    value=row['paired_difference']
    right.scatter(value,y,s=32,color='#007F87' if value<0 else '#AD6D25',zorder=3)
right.set(yticks=[-1]+list(range(14)),
          yticklabels=['All 40 patients']+[r['deleted_patient_id'] for r in records],
          ylim=(13.8,-1.8),xlim=(-.025,.009),
          xlabel='Brier difference: volume + KPS minus KPS',
          title='B   Positive-patient deletion sensitivity')
right.set_xticks([-.02,-.01,0]);right.set_xticklabels(['−0.02','−0.01','0'])
right.tick_params(axis='y',labelsize=8.5)
right.text(0,-.17,'Each deletion: 39-patient leave-one-out analysis.\n3 negative / 11 positive differences; no intervals.',
           transform=right.transAxes,fontsize=10,color='#435467',va='top')
for ax in (left,right):
    ax.spines[['top','right','left']].set_visible(False)
    ax.spines['bottom'].set_color('#CBD2D9');ax.tick_params(axis='y',length=0)
    ax.grid(axis='x',color='#E7EBF0',lw=.7);ax.set_axisbelow(True)
fig.text(.06,.035,'Exploratory internal association with a recorded postoperative category; new/worsened status and assessment timing are unverified.\nNo external validation, clinical calibration, causal effect, or route-specific harm estimate.',
         fontsize=9,color='#435467',linespacing=1.55)
for ext in ('png','pdf','svg'):
    fig.savefig(HERE/f'comparison.{ext}',dpi=220,facecolor='white')
plt.close(fig)
