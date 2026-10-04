from pathlib import Path
import json
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from resectionlab.imaging import load_fractional_annotation_case, file_sha256

out = Path(__file__).parent
root = Path.cwd()
t1 = root / "data/diffusion_source/ds001226-v5.0.1/sub-PAT05/ses-preop/anat/sub-PAT05_ses-preop_T1w.nii.gz"
annotation = root / "data/diffusion_source/ds001226-v5.0.1/derivatives/tumor_masks/sub-PAT05/anat/sub-PAT05_space_T1_label-tumor.nii"
case = load_fractional_annotation_case(t1, annotation, threshold=0.5, annotation_interpretation="source fractional annotation, not probability")
image = nib.load(t1).get_fdata(dtype=np.float32)
tumor = next(iter(case.compartments.values()))
masks = {name: nib.load(out / (name + "_mask.nii.gz")).get_fdata().astype(bool) for name in ("nocsf", "main")}
masks["baseline"] = nib.load(out / "intensity_core_baseline.nii.gz").get_fdata().astype(bool)
masks["tumor"] = tumor
centers = [np.argwhere(masks["main"]).mean(axis=0).astype(int), np.argwhere(tumor).mean(axis=0).astype(int)]
colors = {"nocsf":"#31deff", "main":"#ffb557", "baseline":"#64d56f", "tumor":"#ff405c"}
fig, axes = plt.subplots(2, 3, figsize=(14, 9), facecolor="#101922")
vmax = np.percentile(image[image>0], 99.5)
for row, center in enumerate(centers):
 for axis in range(3):
  index = int(center[axis]); ax = axes[row, axis]
  ax.imshow(np.take(image, index, axis=axis).T, origin="lower", cmap="gray", vmin=0, vmax=vmax)
  for name, mask in masks.items():
   plane=np.take(mask,index,axis=axis).T
   if plane.any() and not plane.all():
    ax.contour(plane, levels=[.5], colors=[colors[name]], linewidths=.75 if name != "tumor" else 1.4, linestyles="dashed" if name=="baseline" else "solid")
  ax.set_title(f"{['Brain-envelope centre','Tumor-annotation centre'][row]} | native {['X','Y','Z'][axis]} = {index}",color="white",fontsize=10)
  ax.set_axis_off()
fig.suptitle("BTC PAT05 | estimated extraction, human QC pending", color="white", fontsize=17)
handles=[Line2D([0],[0],color=colors[n],linestyle="--" if n=="baseline" else "-",label=l) for n,l in [("nocsf","No-CSF model"),("main","Main model"),("baseline","Intensity baseline"),("tumor","Source annotation ≥ 0.5")]]
fig.legend(handles=handles,loc="lower center",ncol=4,labelcolor="white",facecolor="#101922",edgecolor="none",bbox_to_anchor=(.5,.055))
fig.text(.5,.025,"Native oblique voxel planes. Extraction contours do not establish pial surface, safe entry, or clinical accuracy.",ha="center",color="#b9c7d6",fontsize=10)
fig.subplots_adjust(top=.92,bottom=.12,hspace=.15,wspace=.05)
fig.savefig(out/"extraction_qc.png",dpi=150,facecolor=fig.get_facecolor())
plt.close(fig)
