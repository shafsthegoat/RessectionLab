"""Plot saved diagnostic outputs; no model, patient arrays or simulations."""
from pathlib import Path
import hashlib,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
root=Path(__file__).resolve().parents[3]
source=root/'artifacts/native-opening-length-sensitivity-v1'
rows={name:json.loads((source/(name+'.json')).read_text()) for name in ('initial','RL256')}
fig,axes=plt.subplots(1,2,figsize=(11,4.8),gridspec_kw={'width_ratios':[1.2,1]})
fig.patch.set_facecolor('#f6f8fa')
for ax in axes:
 ax.set_facecolor('#f6f8fa');ax.spines[['top','right']].set_visible(False);ax.spines[['left','bottom']].set_color('#bbc7cf');ax.tick_params(colors='#314757')
x=np.arange(2);w=.33
before=[100*rows[n]['saved_original_stop_probability'] for n in rows]
after=[100*rows[n]['stop_probability'] for n in rows]
for offset,y,color,label in [(-w/2,before,'#7b8fa6','Original descriptors'),(w/2,after,'#007e86','120 mm descriptors')]:
 bars=axes[0].bar(x+offset,y,w,color=color,label=label)
 axes[0].bar_label(bars,labels=[f'{v:.1f}%' for v in y],padding=4,color='#183748',fontsize=11)
axes[0].set(xticks=x,xticklabels=['Initial weights','After 256 RL updates'],ylabel='STOP preference within five supplied choices (%)',ylim=(0,112),title='The preference shift appears after training')
axes[0].legend(frameon=False,loc='upper left',fontsize=9)
r=rows['RL256'];stop=r['logits'][0];original_best=stop-r['saved_original_margin'];changed_best=max(r['logits'][1:])
axes[1].plot([0,1],[stop,stop],color='#a55e36',marker='o',linewidth=2,label='STOP score')
axes[1].plot([0,1],[original_best,changed_best],color='#007e86',marker='o',linewidth=2,label='Best movement score')
axes[1].set(xticks=[0,1],xticklabels=['Original','120 mm'],ylabel='Raw model score (logit)',title='Movement scores fall; STOP stays fixed',ylim=(-9,1))
axes[1].legend(frameon=False,loc='lower left',fontsize=9)
fig.suptitle('Tool-length input sensitivity in the frozen policy',fontsize=17,color='#183748',x=.06,ha='left',y=.98)
fig.text(.06,.89,'Same generated image, state, five choices and weights. Only four length descriptors changed.',fontsize=10,color='#314757')
fig.text(.06,.055,'Descriptor intervention only: longer tools were not physically recertified.\nNo actions, new training, patient outcomes or clinical risk probabilities were evaluated.',fontsize=9,color='#314757')
fig.subplots_adjust(left=.08,right=.98,top=.77,bottom=.24,wspace=.34)
out=Path(__file__).parent;fig.savefig(out/'comparison.png',dpi=160,facecolor=fig.get_facecolor());plt.close(fig)
index={str((source/(n+'.json')).relative_to(root)):hashlib.sha256((source/(n+'.json')).read_bytes()).hexdigest() for n in rows}
(out/'figure.json').write_text(json.dumps({'scope':'Saved-output visualization only','inputs_sha256':index,'plot_sha256':hashlib.sha256((out/'comparison.png').read_bytes()).hexdigest()},indent=2)+'\n')
