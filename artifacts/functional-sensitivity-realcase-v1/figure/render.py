#!/usr/bin/env python3
"""Plot saved real-case sensitivity results only; never invoke a simulator."""
from pathlib import Path
import hashlib
import json
import math
from statistics import NormalDist

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
import numpy as np

OUT=Path(__file__).resolve().parent
SOURCE=OUT.parent/'analysis'
FILES=('gaussian_1mm_1deg_support_0.5.json','gaussian_2mm_2deg_support_0.5.json')
IDS=('native_native-fine-aspiration','native_native-wide-aspiration')
COLORS=('#19759B','#C66A29')
LABELS=('Fine tool','Wide tool')
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()


def wilson(k,n):
    z=NormalDist().inv_cdf(.975);p=k/n;den=1+z*z/n
    midpoint=(p+z*z/(2*n))/den;radius=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return max(0.,midpoint-radius),min(1.,midpoint+radius)


def tail_mean(values,alpha=.95):
    values=np.sort(values)[::-1];mass=len(values)*(1-alpha);whole=int(mass);fraction=mass-whole
    return float((values[:whole].sum()+(fraction*values[whole] if fraction else 0.))/mass)


def extract():
    geometry=json.loads((SOURCE/'geometry.json').read_text())
    tools={r['id']:r for r in geometry['alternatives'] if r['id'] in IDS}
    assert tools[IDS[0]]['entry_mm']==tools[IDS[1]]['entry_mm']
    assert tools[IDS[0]]['target_mm']==tools[IDS[1]]['target_mm']
    rows=[]
    for scale,filename in enumerate(FILES,1):
        panel=json.loads((SOURCE/filename).read_text())
        assert panel['event_configuration']['motor_threshold']==.5
        for plan_id in IDS:
            candidate=next(r for r in panel['candidates'] if r['plan_id']==plan_id)
            event=next(r for r in candidate['model_events'] if r['event']=='motor_supplied_map_contact')
            cost=candidate['surrogate_costs']['motor_contact_surrogate'];raw=candidate['world_outcomes']
            counts=sum(row['events']['motor_supplied_map_contact'] is True for row in raw)
            values=np.array([row['costs']['motor_contact_surrogate'] for row in raw])
            assert event['unknown_worlds']==cost['unknown_worlds']==0
            assert event['denominator']==len(raw)==64 and counts==event['numerator']
            np.testing.assert_allclose(event['monte_carlo_interval'],wilson(counts,len(raw)),rtol=0,atol=1e-14)
            np.testing.assert_allclose([cost['mean'],cost['upper_tail_cvar']],
                                       [values.mean(),tail_mean(values)],rtol=0,atol=1e-10)
            assert cost['cvar_alpha']==.95 and candidate['clinical_deficit_probability'] is None
            rows.append({'scale_mm_and_deg':scale,'tool':LABELS[IDS.index(plan_id)],'plan_id':plan_id,
                'encounter_count':counts,'world_count':64,'frequency':event['model_conditioned_frequency'],
                'conditional_95_wilson':event['monte_carlo_interval'],'mean_contact_surrogate':cost['mean'],
                'worst_5_percent_contact_surrogate':cost['upper_tail_cvar'],
                'tool_geometry':tools[plan_id]['tool'],'source_panel':filename,'world_partition_hash':panel['world_partition_hash']})
    return rows


def main():
    rows=extract()
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.labelsize':10,
        'axes.titlesize':12,'axes.titleweight':'bold','axes.spines.top':False,'axes.spines.right':False,
        'axes.edgecolor':'#657583','axes.linewidth':.7,'xtick.color':'#314554','ytick.color':'#314554',
        'text.color':'#172F40','axes.labelcolor':'#172F40','svg.fonttype':'none','svg.hashsalt':'real-case-motor-sensitivity-v1'})
    fig,axes=plt.subplots(1,3,figsize=(11.8,6.0))
    fig.subplots_adjust(left=.075,right=.982,bottom=.34,top=.75,wspace=.40)
    fig.text(.075,.963,'Motor prior sensitivity in one real glioma case',fontsize=16,weight='bold')
    fig.text(.075,.915,'UCSF-PDGM-0004  ·  Same fixed axis, alternative generic aspiration tools',fontsize=11)
    fig.legend([Line2D([],[],color=c,marker='o',ls='',markersize=7) for c in COLORS],LABELS,
        loc='upper center',bbox_to_anchor=(.53,.89),ncol=2,frameon=False,columnspacing=2.8)
    titles=('A  Motor-prior encounters','B  Mean contact surrogate','C  Worst 5% contact surrogate')
    for axis,title in zip(axes,titles):
        axis.set_title(title,loc='left',pad=16)
        axis.set_xticks([0,1],['1 mm / 1°','2 mm / 2°'])
        axis.set_xlabel('Assumed registration SD',labelpad=10)
        axis.set_xlim(-.53,1.53);axis.grid(axis='y',color='#E6EBEF',lw=.7);axis.set_axisbelow(True)
        axis.tick_params(axis='both',length=3)
    axes[0].set_ylim(0,1.16);axes[0].set_yticks([0,.25,.5,.75,1]);axes[0].yaxis.set_major_formatter(PercentFormatter(1,decimals=0))
    axes[0].set_ylabel('Conditional model-world frequency',labelpad=8)
    for axis in axes[1:]:
        axis.set_ylim(0,405);axis.set_yticks([0,100,200,300,400]);axis.set_ylabel('Map support × contacted-cell mm³',labelpad=8)
    for row in rows:
        tool=LABELS.index(row['tool']);x=row['scale_mm_and_deg']-1+(-.17 if tool==0 else .17);color=COLORS[tool]
        lo,hi=row['conditional_95_wilson'];freq=row['frequency']
        axes[0].errorbar(x,freq,yerr=[[freq-lo],[hi-freq]],fmt='o',color=color,ms=7,
            capsize=4,elinewidth=1.8,markeredgecolor='white',markeredgewidth=.5)
        axes[0].text(x,hi+.040,f"{row['encounter_count']}/64",ha='center',va='bottom',fontsize=10,color=color,weight='bold')
        for index,key in ((1,'mean_contact_surrogate'),(2,'worst_5_percent_contact_surrogate')):
            value=row[key]
            axes[index].bar(x,value,width=.27,color=color,zorder=3)
            axes[index].text(x,value+9,f'{value:.1f}',ha='center',va='bottom',fontsize=10,color=color,weight='bold')
    fig.text(.075,.196,'A: motor map support ≥ 0.5 in any contacted native cell; whiskers are conditional 95% Wilson intervals.',fontsize=9.5)
    fig.text(.075,.151,'Intervals describe finite Monte Carlo error only. B–C use continuous map support; contact is not tissue removal.',fontsize=9.5)
    fig.text(.075,.106,'Per-axis Gaussian translation / rotation SD; 64 worlds per setting. Worst 5% = empirical CVaR₉₅ with a fractional tail.',fontsize=9.5)
    fig.text(.075,.052,'Unreviewed population prior · Uncalibrated sensitivity · One consulted patient · No clinical injury probabilities',fontsize=9.6,weight='bold')
    fig.savefig(OUT/'motor-prior-sensitivity.png',dpi=300,facecolor='white')
    fig.savefig(OUT/'motor-prior-sensitivity.svg',facecolor='white',metadata={'Date':None,'Description':'One consulted development patient; unreviewed population-prior sensitivity, not clinical injury probabilities.'})
    plt.close(fig)
    receipt={'schema_version':1,'kind':'static_figure_from_saved_results_no_new_simulation',
        'inputs_sha256':{name:sha(SOURCE/name) for name in (*FILES,'geometry.json','declaration.json')},
        'script_sha256':sha(Path(__file__)),'rows':rows,'human_patients':1,'worlds_per_setting':64,
        'same_fixed_axis_verified':True,'source_threshold':.5,'interval':'conditional95%Wilson; finite Monte Carlo error only',
        'tail_definition':'Top5%empirical probability mass; for64equal worlds, top3 +0.2of4th, divided by3.2.',
        'source_count_interval_mean_tail_recomputed_and_verified':True,'no_new_patient_or_simulation_run':True,
        'clinical_injury_probability':None,'patient_function_assessed':False,'population_prior_alignment_reviewed':False,
        'interpretation':'Broader assumed registration error reduces these binary encounter counts while increasing mean/tail motor contact surrogates; a lower count is not a better-plan claim.',
        'matplotlib':matplotlib.__version__,'numpy':np.__version__,
        'outputs_sha256':{name:sha(OUT/name) for name in ('motor-prior-sensitivity.png','motor-prior-sensitivity.svg')}}
    (OUT/'figure-metadata.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({'outputs':list(receipt['outputs_sha256']),'numeric_checks':'counts,Wilson,mean,fractional-tail and same-axis verified'}))

if __name__=='__main__':main()
