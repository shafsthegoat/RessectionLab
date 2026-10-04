from pathlib import Path
import hashlib,json
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import map_coordinates
root=Path.cwd(); src=root/'data/remind_source/converted-native/ReMIND-001';out=root/'artifacts/remind-native-independent-qc-v1'
rows=[]
for mr_name,seg_name in [('structural_t1ce','cerebrum_annotation'),('structural_t2','tumor_annotation')]:
 image=nib.load(src/f'{mr_name}.nii.gz'); anatomy=image.get_fdata(dtype=np.float32)
 seg=nib.load(src/f'{seg_name}.nii.gz'); mask=np.asarray(seg.dataobj,dtype=np.uint8)
 count=int(mask.sum()); centroid=np.asarray([np.dot(np.arange(mask.shape[a]),mask.sum(axis=tuple(j for j in range(3) if j!=a),dtype=np.float64))/count for a in range(3)])
 world=(seg.affine @ np.r_[centroid,1])[:3]; location=(np.linalg.inv(image.affine)@np.r_[world,1])[:3]
 print(mr_name,'annotation centroid in MR voxels',location,flush=True)
 fig,axes=plt.subplots(1,3,figsize=(15,5));counts=[]
 for axis,ax in enumerate(axes):
  plane=int(np.rint(location[axis])); other=[j for j in range(3) if j!=axis]
  if not 0<=plane<anatomy.shape[axis]: raise ValueError('Source annotation centroid outside named MRI')
  mesh=np.indices(tuple(anatomy.shape[j] for j in other),dtype=np.float64)
  coords=np.empty((3,*mesh.shape[1:]));coords[axis]=plane;coords[other[0]]=mesh[0];coords[other[1]]=mesh[1]
  flat=coords.reshape(3,-1); transformation=np.linalg.inv(seg.affine)@image.affine
  segvox=transformation[:3,:3]@flat+transformation[:3,3,None]
  covered=np.all((segvox>=0)&(segvox<=np.asarray(mask.shape)[:,None]-1),axis=0)
  overlay=map_coordinates(mask,segvox,order=0,mode='constant',cval=0).reshape(mesh.shape[1:]);coverage=covered.reshape(mesh.shape[1:])
  selected=np.take(anatomy,plane,axis=axis);vmin,vmax=np.percentile(anatomy,[2,99.5])
  ax.imshow(selected.T,origin='lower',cmap='gray',vmin=vmin,vmax=vmax)
  if overlay.any(): ax.contour(overlay.T,[.5],colors=['#ffb02e'],linewidths=.9)
  if coverage.any() and not coverage.all(): ax.contour(coverage.T,[.5],colors=['#59d1e8'],linewidths=.6,linestyles='--')
  ax.set_title(f'MRI native axis {axis}, index {plane}');ax.set_xlabel(f'voxel axis {other[0]}');ax.set_ylabel(f'voxel axis {other[1]}')
  counts.append({'axis':axis,'index':plane,'seg_grid_covered_pixels':int(coverage.sum()),'positive_overlay_pixels':int(overlay.sum())})
 fig.suptitle(f'{mr_name} + {seg_name}: orange source annotation; dashed cyan source SEG field of view\nNative-frame QC only; outside SEG coverage is unassessed; no anatomical approval',fontsize=11)
 fig.tight_layout();path=out/f'{mr_name}-native-annotation-overlay.png';fig.savefig(path,dpi=140);plt.close(fig)
 rows.append({'mri':mr_name,'segmentation':seg_name,'annotation_centroid_ras_mm':world.tolist(),'annotation_centroid_mri_voxel':location.tolist(),'planes':counts,'png':path.name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
(out/'overlay-qc.json').write_text(json.dumps({'overlays':rows,'cross_frame_registration':False,'expert_anatomical_review':False},indent=2)+'\n')
