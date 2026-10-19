from pathlib import Path
import nibabel as nib,numpy as np,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent;QC=ROOT.parents[1]/'artifacts/desktop-renderer/medivis-fidelity-v2/source-qc'
ct=nib.load(ROOT/'originals/topcow_ct_001_0000.nii.gz');lab=nib.load(ROOT/'originals/topcow_ct_001.nii.gz');a=ct.get_fdata(dtype=np.float32);labels=np.asanyarray(lab.dataobj);s=np.array(ct.header.get_zooms());arteries=(labels>0)&(labels<35);veins=labels>=35
fig,axes=plt.subplots(2,3,figsize=(15,9),facecolor='#11141b');fig.suptitle('TopBrain CTA001 · original source intensity and same-grid published labels',color='white',fontsize=17,y=.985)
for ax in axes.ravel():ax.set_facecolor('#11141b');ax.tick_params(colors='#9baac2',labelsize=8);[sp.set_visible(False) for sp in ax.spines.values()]
def title(ax,text):ax.set_title(text,color='#e6edf7',fontsize=11)
for col,z in enumerate([100,145,195]):
 ax=axes[0,col];img=a[:,:,z].T;extent=[0,a.shape[0]*s[0],0,a.shape[1]*s[1]];ax.imshow(img,cmap='gray',vmin=0,vmax=300,origin='lower',extent=extent);rgba=np.zeros((*img.shape,4));am=arteries[:,:,z].T;vm=veins[:,:,z].T;rgba[am]=[.96,.30,.25,.95];rgba[vm]=[.35,.55,1,.95];ax.imshow(rgba,origin='lower',extent=extent,interpolation='nearest');title(ax,f'Native axial z={z} · source labels over CTA');ax.set_xlabel('Native x distance, mm',color='#9baac2');ax.set_ylabel('Native y distance, mm',color='#9baac2')
ax=axes[1,0];ax.imshow(a.max(1).T,cmap='gray',vmin=0,vmax=1500,origin='lower',extent=[0,a.shape[0]*s[0],0,a.shape[2]*s[2]]);title(ax,'CTA coronal maximum projection · bone + contrast overlap');ax.set_xlabel('Native x distance, mm',color='#9baac2');ax.set_ylabel('Native z distance, mm',color='#9baac2')
for ax,m,color,text in [(axes[1,1],arteries,[.96,.30,.25,1],'Published arterial label projection'),(axes[1,2],veins,[.35,.55,1,1],'Published venous / sinus label projection')]:
 p=m.max(1).T;rgba=np.zeros((*p.shape,4));rgba[p]=color;ax.imshow(rgba,origin='lower',extent=[0,a.shape[0]*s[0],0,a.shape[2]*s[2]],interpolation='nearest');title(ax,text);ax.set_xlabel('Native x distance, mm',color='#9baac2');ax.set_ylabel('Native z distance, mm',color='#9baac2')
fig.text(.02,.048,'284 × 327 × 243 · 0.507813 × 0.507813 × 0.625 mm · 152 mm slab · 28 supplied classes present\nPublisher labels unreviewed locally; missing left ICA/MCA label classes preserved. Braincase crop; not full head/neck.',color='#c3cede',fontsize=9.5)
fig.text(.02,.015,'Source: TopBrain Challenge Organizers, zenodo.org/records/16623496 · Noncommercial with attribution; commercial use requires permission from USZ.',color='#9baac2',fontsize=8.5)
fig.subplots_adjust(left=.05,right=.98,top=.93,bottom=.16,hspace=.35,wspace=.22);fig.savefig(QC/'topbrain-cta001-source-qa.png',dpi=150,facecolor=fig.get_facecolor());plt.close(fig)
print(QC/'topbrain-cta001-source-qa.png')
