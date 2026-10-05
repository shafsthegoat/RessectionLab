"""Plot saved scalar coverage summaries only; no image or model imports."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
data = json.loads((HERE/'compact-summary.json').read_text())
rows = data['subjects']
names = ('legacy_access_crop', 'target_local', 'whole_source')
colors = ('#778391', '#007E87', '#8D572A')
offsets = (-.18, 0, .18)
labels = ['PAT05 · failed', 'PAT16 · blocked', 'PAT20 · blocked',
          'PAT22 · 76/78 accepted', 'PAT25 · 0/78 accepted', 'PAT28 · 70/78 accepted']
plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':9,
                     'pdf.fonttype':42, 'svg.fonttype':'none'})
fig, axes = plt.subplots(2,2,figsize=(12.8,8.5))
fig.subplots_adjust(left=.17,right=.965,top=.78,bottom=.12,hspace=.39,wspace=.61)
fig.suptitle('Three coverage records; the fixed six-case run remains incomplete',
             x=.055,y=.962,ha='left',fontsize=16,weight='bold')
fig.text(.055,.915,'Four new attempts · one task/frame check failure · two historical support blocks · no learning or committed cuts',
         fontsize=10.5,color='#465364')
legend = [Line2D([],[],marker='o',linestyle='',color=c,label=l) for c,l in zip(colors,
          ('Legacy access crop','Target-local crop','Whole-source coarse (union matches)'))]
legend += [Line2D([],[],marker='o',linestyle='',color='#25394C',label='Center interpolation domain'),
           Line2D([],[],marker='s',linestyle='',markerfacecolor='none',color='#25394C',label='Full-cell extent')]
fig.legend(handles=legend,loc='upper left',bbox_to_anchor=(.05,.885),ncol=3,frameon=False,
           columnspacing=1.8,handletextpad=.5,fontsize=9)
panels = [(None,'A   Nominal target mass retained'),
          ('entry_to_tip','B   Entry-to-tip centerline'),
          ('approach_shaft_centerline','C   Shaft centerline at approach'),
          ('deepest_shaft_centerline','D   Shaft centerline at deepest pose')]
for ax,(segment,title) in zip(axes.flat,panels):
    ax.set_title(title,loc='left',fontsize=11,weight='bold',pad=10)
    ax.set(ylim=(5.65,-.65),xlim=(0,104),yticks=range(6),yticklabels=labels,
           xlabel='Retained nominal mass (%)' if segment is None else 'Mean continuous domain coverage (%)')
    ax.set_xticks([0,20,40,60,80,100])
    ax.spines[['top','right','left']].set_visible(False)
    ax.spines['bottom'].set_color('#CCD3DA');ax.tick_params(axis='y',length=0,pad=7)
    ax.grid(axis='x',color='#E7EBEF',lw=.7);ax.set_axisbelow(True)
    for i,row in enumerate(rows):
        rep=row['representation']
        if rep is None:
            ax.axhspan(i-.45,i+.45,color='#F2F4F6',zorder=0)
            ax.text(52,i,'Not measured in this attempt',ha='center',va='center',
                    color='#697582',fontsize=8)
            continue
        for view,color,offset in zip(names,colors,offsets):
            if segment is None:
                x=100*rep['views'][view]['nominal_mass_fraction']
                ax.scatter(x,i+offset,s=31,color=color,zorder=3)
            else:
                metric=rep['visibility_summary']['emitted']['views'][view][segment]
                for domain,marker,face in (('center_domain','o',color),('fullcell_extent','s','none')):
                    x=100*metric[f'mean_{domain}_fraction']
                    ax.scatter(x,i+offset,s=30,marker=marker,facecolors=face,edgecolors=color,zorder=3)
fig.text(.055,.040,'Panels B–D average all 78 emitted proposals per completed case, including rejected proposals. Accepted counts are shown separately.\nGrid-domain visibility does not establish tissue evidence, complete-tool visibility, clearance, resection feasibility, or model accuracy.',
         fontsize=9,color='#465364',linespacing=1.5)
for extension in ('png','pdf','svg'):
    fig.savefig(HERE/f'coverage-comparison.{extension}',dpi=220,facecolor='white')
plt.close(fig)
