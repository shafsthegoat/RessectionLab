"""Plot the fixed, exposed-specimen comparison; not independent validation."""
import os
from pathlib import Path
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'src/resectionlab').is_dir())
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'build/hbe-ogden-shape-v1/plot-cache'))
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent
archived=HERE/'execution/comparison.json'
source=archived if archived.exists() else ROOT/'outputs/mechanics/hbe-ogden-shape-comparison-v1/postprocess-01/comparison.json'
report=json.loads(source.read_text())
fig,axes=plt.subplots(1,2,figsize=(10.8,4.8),layout='constrained')
fig.suptitle('Tissue-law comparison: improvement with a remaining tension mismatch',fontsize=13)
for ax,branch in zip(axes,('compression','tension')):
    observed=report['candidates']['2']['fit']['modes'][branch]
    x=[v*1000 for v in observed['coordinate']]
    ax.scatter(x,[v*1000 for v in observed['observed']],color='#17232d',s=17,zorder=3,label='Observed cycle averages')
    for key,color,style,label in [('2','#8c939f','--','Retained model'),('-8','#007f89','-','Alternative shape')]:
        metrics=report['candidates'][key]['metrics']['branches'][branch]
        ax.plot(x,[v*1000 for v in metrics['predicted_at_observed_coordinates']],style,color=color,lw=2,
                label=f"{label} (RMSE {1000*metrics['RMSE']:.2f} mN)")
    ax.set_title(branch.capitalize(),loc='left',fontsize=12)
    ax.set_xlabel('Signed displacement (mm)')
    ax.set_ylabel('Signed force (mN)')
    ax.grid(alpha=.18);ax.spines[['top','right']].set_visible(False)
    ax.legend(loc='best',fontsize=8,frameon=False)
fig.supxlabel('One exposed ex vivo specimen; each model fitted on these same measurements.\nNumerical checks passed; no held-out specimen, cutting, or surgical validation.',fontsize=9)
fig.savefig(HERE/'force-comparison.png',dpi=150)
