from pathlib import Path
import json,hashlib
import nibabel as nib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path.cwd();out=root/'artifacts/remind-rigid-feasibility-v1';names=['fixed_T1_3mm','T2_identity_control_3mm','T2_rigid_proposal_3mm']
images=[nib.load(out/(n+'.nii.gz')) for n in names]; arrays=[i.get_fdata(dtype=np.float32) for i in images]
fig,axes=plt.subplots(3,3,figsize=(12,11))
for row,array in enumerate(arrays):
 for axis in range(3):
  k=array.shape[axis]//2;plane=np.take(array,k,axis=axis)
  axes[row,axis].imshow(plane.T,cmap='gray',origin='lower',vmin=0,vmax=1)
  axes[row,axis].set_title(f'{names[row]}\nfixed native axis {axis}, index {k}',fontsize=9)
  axes[row,axis].set_xlabel('fixed-grid voxel axis');axes[row,axis].set_ylabel('fixed-grid voxel axis')
fig.suptitle('ReMIND-001 multimodal geometry diagnostic at 3 mm\nIdentity is an unaccepted control; fitted proposal worsened diagnostic MI and remains unselected',fontsize=12)
fig.tight_layout();fig.savefig(out/'before-after-native-planes.png',dpi=140)
print(str(out/'before-after-native-planes.png'))
