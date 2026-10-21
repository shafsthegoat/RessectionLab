import sys,json
from pathlib import Path
import numpy as np
from scipy import ndimage as ndi
sys.path.insert(0,str(Path.cwd()/'src'))
from resectionlab.surgical_environment import load_environment
scene=load_environment('testcases/cases/meningioma-sub-06-standard-v2.ressectionlab');lab=scene.labels
rows=[]
for axis in range(3):
 for sign in (1,-1):
  v=np.moveaxis(lab,axis,-1)
  if sign<0:v=v[...,::-1]
  target=v==2; present=target.any(-1);first=np.max(np.where(target,np.arange(v.shape[-1]),-1),-1)
  # toward index max boundary; signed reversal handles min boundary
  bad=np.maximum.reduce([np.max(np.where(v==r,np.arange(v.shape[-1]),-1),-1) for r in (1,4)])
  candidates=present&(bad<first)
  distance=ndi.distance_transform_edt(candidates)
  # Rank multiple maximally interior disjoint access regions
  cc,n=ndi.label(candidates);regions=[]
  for component in range(1,n+1):
   sel=cc==component; at=np.unravel_index(np.argmax(distance*sel),sel.shape);ij=list(map(int,at));k=int(first[at]);ij.insert(axis,k if sign>0 else lab.shape[axis]-1-k);xyz=scene.points([np.ravel_multi_index(tuple(ij),lab.shape)])[0]
   ray=v[at];above=ray[k+1:];bone=np.where(ray==3)[0];bone=bone[bone>k]
   regions.append({'columnCount':int(sel.sum()),'maxLateralColumnClearanceMm':float(distance[at]),'index':ij,'targetEntryRAS':xyz.tolist(),'aboveTargetCounts':{str(x):int(np.count_nonzero(above==x)) for x in (0,1,2,3,4)},'boneRangeFromTargetEntryMm':None if len(bone)==0 else [int(bone.min()-k),int(bone.max()-k)]})
  rows.append({'indexAxis':axis,'boundarySign':sign,'worldDirectionOut':(scene.affine[:3,axis]*sign).tolist(),'candidateColumnsWithoutSoftTissueOrVesselAbove':int(candidates.sum()),'regions':sorted(regions,key=lambda r:(-r['maxLateralColumnClearanceMm'],-r['columnCount']))[:15]})
Path('testcases/outputs/full-case-rehearsal-20261020/geometry/access_routes.json').write_text(json.dumps(rows,indent=2));print(json.dumps(rows,indent=2))
