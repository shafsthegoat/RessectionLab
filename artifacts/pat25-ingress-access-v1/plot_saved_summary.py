"""Fixed four-panel comparison from terminal scalar/ledger summaries only."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
summary = json.loads((HERE/'compact-summary.json').read_text())
if summary['schema'] != 'pat25-ingress-access-saved-summary-v1':
    raise ValueError('Unexpected saved reporting schema')
phases = ('original','screened_selected')
phase_labels = ('Original baseline','Screened selection')
colors = ('#73808D','#007E87')
eligibility_colors = {'eligible':'#007E87','rejected':'#AC6840','unknown':'#AAB2BB'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,'svg.fonttype':'none'})
fig,axes=plt.subplots(2,2,figsize=(12.8,9.3))
fig.subplots_adjust(left=.13,right=.965,top=.78,bottom=.18,wspace=.48,hspace=.56)
fig.suptitle('PAT25 ingress diagnostic: original versus deterministic screened access',
             x=.055,y=.958,ha='left',fontsize=15,weight='bold')
fig.text(.055,.914,f"Recorded attempt status: {summary['status']} · six fixed source-axis exits · initial uncommitted inventories only",
         fontsize=10.5,color='#465364')
fig.legend(handles=[Line2D([],[],color=c,marker='o',linestyle='',label=l)
           for c,l in zip(colors,phase_labels)],loc='upper left',bbox_to_anchor=(.05,.888),
           ncol=2,frameon=False,fontsize=10)

ax=axes[0,0]
ax.set_title('A   All six exits: static ingress eligibility',loc='left',weight='bold',fontsize=11)
for i,row in enumerate(summary['exits']):
    x=row['distance_mm'];color=eligibility_colors[row['eligibility']]
    suffix=(' · baseline' if row['original'] else '')+(' · chosen' if row['chosen'] else '')
    if x is None:
        ax.text(.02,i,'Not assessed',transform=ax.get_yaxis_transform(),va='center',color=color)
    else:
        ax.hlines(i,0,x,color=color,lw=4,alpha=.65)
        ax.scatter(x,i,s=42,color=color,marker='D' if row['original'] else 'o',zorder=3)
    ax.text(1.03,i,row['eligibility']+suffix,transform=ax.get_yaxis_transform(),va='center',fontsize=8)
ax.set(yticks=range(6),yticklabels=[f"Axis {r['axis']}, sign {r['outward_sign']:+d}" for r in summary['exits']],
       ylim=(5.6,-.6),xlabel='Existing local exit distance (mm)')
unknowns=sum(row['geometry_unknown_pose_count'] for row in summary['exits'])
ax.text(0,-.27,f'{unknowns} returned poses carry geometry unknowns (reuse included).\nEligibility does not erase those unknowns.',
        transform=ax.transAxes,va='top',fontsize=8,color='#465364')

ax=axes[0,1]
ax.set_title('B   Full native previews: acceptance by tool',loc='left',weight='bold',fontsize=11)
tools=sorted({tool for phase in phases for tool in (summary['phases'][phase].get('inventory') or {}).get('per_tool',{})})
for j,phase in enumerate(phases):
    record=summary['phases'][phase]; inventory=record.get('inventory')
    if inventory is None:
        ax.text(.02,.9-j*.15,phase_labels[j]+': '+record['status'],transform=ax.transAxes,color=colors[j],fontsize=9)
        continue
    for i,tool in enumerate(tools):
        value=inventory['per_tool'].get(tool)
        if value is None:continue
        y=i+(-.17 if j==0 else .17)
        ax.barh(y,value['accepted'],height=.26,color=colors[j],alpha=.85)
        ax.text(value['accepted']+.3,y,f"{value['accepted']}/{value['emitted']}",va='center',fontsize=9)
ax.set(yticks=range(len(tools)),yticklabels=[name.replace('native-','').replace('-',' ') for name in tools],
       xlabel='Accepted previews / emitted denominator (labels)')
ax.set_xlim(left=0,right=max([1]+[r['declared_slots'] for phase in phases for r in
    (summary['phases'][phase].get('inventory') or {}).get('per_tool',{}).values()])*1.2)

ax=axes[1,0]
ax.set_title('C   Actor nominal target and initial cavity',loc='left',weight='bold',fontsize=11)
for i,phase in enumerate(phases):
    actor=summary['phases'][phase].get('actor')
    if actor is None:
        ax.text(50,i,'Not assessed',ha='center',va='center',color='#697582');continue
    value=actor['nominal_target_mass_fraction_visible']
    if value is not None:
        ax.scatter(100*value,i,s=65,color=colors[i]);ax.text(100*value,i-.17,f'{100*value:.2f}%',ha='center',fontsize=9)
    else:ax.text(50,i,'Target unknown',ha='center',va='center')
    cavity=(f"Cavity: {actor['cavity_visible_cells']} visible / {actor['cavity_source_cells']} source cells"
            if actor['cavity_channel_available'] else 'Cavity channel unavailable')
    ax.text(0,i+.27,cavity,va='center',fontsize=8,color='#465364')
ax.set(yticks=range(2),yticklabels=phase_labels,ylim=(1.65,-.55),xlim=(-3,105),
       xlabel='Permitted nominal target mass retained in actor crop (%)')

ax=axes[1,1]
ax.set_title('D   Actor grid: continuous centerline coverage',loc='left',weight='bold',fontsize=11)
segments=('entry_to_tip','approach_shaft_centerline','deepest_shaft_centerline')
for j,phase in enumerate(phases):
    visibility=summary['phases'][phase].get('centerline_visibility')
    if visibility is None:
        ax.text(.02,.94-j*.11,phase_labels[j]+': not assessed',transform=ax.transAxes,color=colors[j]);continue
    emitted=visibility['emitted']
    for i,segment in enumerate(segments):
        metrics=emitted['segments'][segment];y=i+(-.14 if j==0 else .14)
        for domain,marker,face in (('center_domain','o',colors[j]),('fullcell_extent','s','none')):
            v=metrics[f'mean_continuous_{domain}_fraction']
            if v is not None:ax.scatter(100*v,y,s=35,marker=marker,facecolors=face,edgecolors=colors[j],zorder=3)
ax.set(yticks=range(3),yticklabels=['Entry to tip','Approach shaft','Deepest shaft'],
       ylim=(2.6,-.6),xlim=(-3,105),xlabel='Mean fraction of emitted centerline within grid domain (%)')
fig.text(.628,.105,'Filled circle: center interpolation domain. Open square: full-cell extent.\nAll emitted previews; accepted-subset summaries retained separately.',
         va='top',fontsize=8,color='#465364')
for ax in axes.flat:
    ax.spines[['top','right','left']].set_visible(False);ax.tick_params(axis='y',length=0)
    ax.spines['bottom'].set_color('#CBD2DA');ax.grid(axis='x',color='#E7EBEF',lw=.7);ax.set_axisbelow(True)
fig.text(.055,.025,'Preparation diagnostic only: ingress eligibility is not full-stroke feasibility; centerline visibility is not complete-tool clearance.\nNo committed resection, clinical benefit, model accuracy or learned-policy improvement is established.',fontsize=9,color='#465364',linespacing=1.5)
for ext in ('png','pdf','svg'):
    fig.savefig(HERE/f'ingress-comparison.{ext}',dpi=220,facecolor='white')
plt.close(fig)
