import sys,json
from pathlib import Path
import numpy as np
from scipy import ndimage as ndi
sys.path.insert(0,str(Path.cwd()/'src'))
from resectionlab.surgical_environment import load_environment
scene=load_environment('testcases/cases/meningioma-sub-06-standard-v2.ressectionlab');v=scene.labels;z=np.arange(v.shape[-1]);target=v==2
bad=np.maximum.reduce([np.max(np.where(v==r,z,-1),-1) for r in (1,4)])
# no owned normal brain/vessel from any target cell toward superior boundary
potential=target&(z[None,None,:]>bad[:,:,None]);first=np.max(np.where(target,z,-1),-1);admit=(first>bad)&target.any(-1)
dist=ndi.distance_transform_edt(admit)
rows=[]
for inset in (0,1,2,3,4,5,6,8,10):
 mask=admit&(dist>inset);counts=potential[mask].sum()
 rows.append({'lateralInsetMm':inset,'columnCount':int(mask.sum()),'ownedTargetUpperBoundCells':int(counts)})
# centerline column runs across recommended area, very small table
columns=[]
for i in range(108,140,4):
 for j in range(136,164,4):
  if not admit[i,j]:continue
  ray=v[i,j];ids=np.where(potential[i,j])[0];above=ray[max(ids)+1:];columns.append({'indexXY':[i,j],'rasXY':(scene.affine@np.array([i,j,0,1]))[:2].tolist(),'targetMinMaxZ':[float(scene.affine[2,3]+min(ids)),float(scene.affine[2,3]+max(ids))],'targetCount':len(ids),'aboveBone':int((above==3).sum()),'columnWindowClearanceMm':float(dist[i,j])})
r={'interpretation':'Geometric upper bounds; ignore exact tip, shaft, guards, bone access feasibility. No source or state edits. Superior path admits bone as drillable and forbids owned normal brain/vessels. Root must exact-preview each action.','superiorPotentialCells':int(potential.sum()),'superiorPotentialFractionOwned':float(potential.sum()/target.sum()),'insetCapacity':rows,'columns':columns}
Path('testcases/outputs/full-case-rehearsal-20261020/geometry/frontier_capacity.json').write_text(json.dumps(r,indent=2));print(json.dumps({k:v for k,v in r.items() if k!='columns'},indent=2))
