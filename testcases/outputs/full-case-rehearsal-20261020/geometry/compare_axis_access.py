"""Bounded read-only six-axis source/residual label analysis; no engine run."""
import sys,json,hashlib,time
from pathlib import Path
import numpy as np
from scipy import ndimage as ndi
sys.path.insert(0,str(Path.cwd()/'src'))
from resectionlab.surgical_environment import load_environment
from resectionlab.workspace_bundle import load_workspace
out=Path('testcases/outputs/full-case-rehearsal-20261020/geometry');source=Path('testcases/cases/meningioma-sub-06-standard-v2.ressectionlab');latest=Path('testcases/outputs/full-case-rehearsal-20261020/primary-v2/primary.ressectionlab');s=load_environment(source);case,a,w=load_workspace(latest);tb=a['surgical_toolbox'];sn=tb['branches'][tb['branchId']];assert sn['sceneId']==s.identity;state=sn['checkpoint']['state'];removed=np.array(state['removed'],dtype=np.int64);residual=s.labels.copy();residual.ravel()[removed]=0
rows=[];initial_masks={};residual_masks={}
names={(0,1):'left-lateral (-X outward)',(0,-1):'right-lateral (+X outward)',(1,1):'anterior (+Y outward)',(1,-1):'posterior (-Y outward)',(2,1):'superior (+Z outward)',(2,-1):'inferior (-Z outward)'}
def analysis(lab,axis,sign):
 v=np.moveaxis(lab,axis,-1)
 if sign<0:v=v[...,::-1]
 z=np.arange(v.shape[-1]);target=v==2;bad=np.max(np.where((v!=0)&(v!=2)&(v!=3),z,-1),2);top=np.max(np.where(target,z,-1),2);potential=target&(z[None,None,:]>bad[:,:,None]);window=(top>bad)&(top>=0);clear=ndi.distance_transform_edt(window);at=np.unravel_index(np.argmax(clear),window.shape)
 xyz=None;ijk=None;bone=None;count=0
 if window.any():
  ijk=list(map(int,at));k=int(top[at]);ijk.insert(axis,k if sign>0 else lab.shape[axis]-1-k);xyz=s.points([np.ravel_multi_index(tuple(ijk),s.shape)])[0];ray=v[at];b=np.flatnonzero((ray==3)&(z>k));bone={'countCells':int(len(b)),'outwardDistanceFromFirstTargetMm':None if not len(b) else [int(b.min()-k),int(b.max()-k)]};count=int(potential[at].sum())
 mask=potential[...,::-1] if sign<0 else potential;mask=np.moveaxis(mask,-1,axis)
 return {'candidateColumnCount':int(window.sum()),'pointRayOwnedTargetCapacityMm3':int(potential.sum()),'maxWindowLateralClearanceMm':float(clear.max()),'bestWindowIndex':ijk,'bestWindowTargetEntryRAS':None if xyz is None else xyz.tolist(),'bestRayTargetCells':count,'bestRayOverlyingBone':bone,'insetCapacity':[{ 'lateralInsetMm':d,'columnCount':int((window&(clear>d)).sum()),'pointRayTargetCapacityMm3':int(potential[window&(clear>d)].sum())} for d in (1,2,3,4,5)]},mask
for axis in range(3):
 for sign in (1,-1):
  initial,im=analysis(s.labels,axis,sign);current,cm=analysis(residual,axis,sign);key=(axis,sign);initial_masks[key]=im;residual_masks[key]=cm;direction=(s.affine[:3,axis]*sign).tolist();rows.append({'name':names[key],'axis':axis,'boundarySign':sign,'outwardRASDirection':direction,'initial':initial,'currentResidual':current})
sup=initial_masks[(2,1)];sres=residual_masks[(2,1)]
for r in rows:
 key=(r['axis'],r['boundarySign']);r['initial']['capacityOutsideSuperiorPotentialMm3']=int((initial_masks[key]&~sup).sum());r['currentResidual']['capacityOutsideSuperiorPotentialMm3']=int((residual_masks[key]&~sres).sum())
union=np.logical_or.reduce(list(initial_masks.values()));ures=np.logical_or.reduce(list(residual_masks.values()));receipt={'schema':'resectionlab.axis-access-geometry-comparison.v1','readOnlyGeometry':True,'physicsExecution':False,'sourcePath':str(source.resolve()),'sourceSha256':hashlib.sha256(source.read_bytes()).hexdigest(),'sceneId':s.identity,'residualCheckpoint':str(latest),'residualStateId':sn['stateId'],'residualActionCount':len(sn['actions']),'sourceOwnedTargetDenominatorMm3':int((s.labels==2).sum()),'residualOwnedTargetMm3':int((residual==2).sum()),'removedOwnedTargetMm3':int(np.count_nonzero(s.labels.ravel()[removed]==2)),'noCurrentHolds':not bool(state.get('holds')),'currentBrainActiveNodes':len(state.get('deformation',{}).get('nodes',{})),'interpretation':'Point-ray and binary lateral-window upper bounds only. Bone/free allowed; every normal brain, vessel and other nontarget material blocks. Physical tool/shaft, guards, bone opening, spatially differing target caps still require exact engine previews. No anatomy beyond supplied categorical masks inferred.','rows':rows,'allAxisUnionInitialCapacityMm3':int(union.sum()),'allAxisUnionInitialOutsideSuperiorMm3':int((union&~sup).sum()),'allAxisUnionResidualCapacityMm3':int(ures.sum()),'allAxisUnionResidualOutsideSuperiorMm3':int((ures&~sres).sum())};(out/'axis_access_comparison.json').write_text(json.dumps(receipt,indent=2));print(json.dumps({k:v for k,v in receipt.items() if k!='rows'},indent=2))
for r in rows:print(json.dumps(r,indent=2))
