"""Bounded source-cell fixed-endpoint burr-cap diagnosis; no engine or replay."""
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path.cwd(); sys.path.insert(0,str(ROOT/'src'))
from resectionlab.core import array_digest
from resectionlab.surgical_environment import load_environment
base=ROOT/'testcases/outputs/full-case-rehearsal-20261020'
spec=importlib.util.spec_from_file_location('read_only_residual_records',base/'geometry/check_current_residual.py')
inspector=importlib.util.module_from_spec(spec);spec.loader.exec_module(inspector)
scene,snapshot,meta,identity=inspector.load_stable_records(base/'primary-v2/primary.ressectionlab')
original=ROOT/'testcases/cases/meningioma-sub-06-standard-v2.ressectionlab'
import hashlib
original_hash=hashlib.sha256(original.read_bytes()).hexdigest()
assert original_hash=='ff5a49b8706a8ccda08d8c3e82dad3e079f85dea98cb6bd40783d2262eec9838'
source=load_environment(original)
assert source.identity==scene.identity
assert array_digest(source.labels)==array_digest(scene.labels)
np.testing.assert_array_equal(source.affine,scene.affine)
assert source.source['caseHash']=='sha256:6ca9c1dee58b3635d92d576ad5b355d0a6d500892f30f07968a671b1b67a51ea'
removed=np.zeros(scene.shape,dtype=bool);removed.ravel()[np.asarray(snapshot['checkpoint']['state']['removed'],dtype=np.int64)]=True
regions={r['id']:r for r in scene.regions}
target_ids=[r['id'] for r in scene.regions if r.get('target') and r['material']=='tumor']
bone_ids=[r['id'] for r in scene.regions if r['material']=='bone']
vessel_ids=[r['id'] for r in scene.regions if r['material']=='vessel']
target=np.isin(source.labels,target_ids);bone=np.isin(source.labels,bone_ids)
normal=(source.labels!=0)&~target&~bone
remaining_target=target&~removed;remaining_bone=bone&~removed

def span(indices):
 a=np.asarray(indices)
 return None if not len(a) else [int(a.min()),int(a.max())]

def column_runs(values,retained):
 result=[]
 for k in range(len(values)-1,-1,-1):
  label=int(values[k])
  if not label:continue
  status='retained' if retained[k] else 'removed'
  if result and result[-1]['label']==label and result[-1]['status']==status and result[-1]['nativeDepthRange'][0]==k+1:
   result[-1]['nativeDepthRange'][0]=k
  else:
   result.append({'label':label,'material':regions[label]['material'],'target':bool(regions[label].get('target')),'status':status,'nativeDepthRange':[k,k]})
 return result

def analyze(access):
 axis=access.get('nativeAxis',2);assert axis in (0,2) and access.get('outwardIndexSign',1)==1
 index=np.asarray(access['targetEntryIndex'],dtype=float)
 np.testing.assert_allclose(scene.affine[:3,:3]@index+scene.affine[:3,3],access['entryPose']['position'],atol=1e-8)
 assert int(source.labels[tuple(index.astype(int))]) in target_ids
 transverse=[a for a in range(3) if a!=axis];center=index[transverse];entry=float(index[axis]);radius=float(access['drillRadiusMm'])
 labels=np.moveaxis(source.labels,axis,-1);keep=np.moveaxis(~removed,axis,-1)
 targets=np.moveaxis(remaining_target,axis,-1);bones=np.moveaxis(remaining_bone,axis,-1);norm=np.moveaxis(normal,axis,-1)
 k=np.arange(labels.shape[-1]);xx,yy=np.ogrid[:labels.shape[0],:labels.shape[1]]
 # Exact distance from axis ray to each closed transverse voxel rectangle.
 lateral2=np.maximum(np.abs((xx-center[0])*scene.spacing[transverse[0]])-scene.spacing[transverse[0]]/2.,0.)**2+np.maximum(np.abs((yy-center[1])*scene.spacing[transverse[1]])-scene.spacing[transverse[1]]/2.,0.)**2
 support=lateral2<=radius**2+1e-12
 cap=np.sqrt(np.maximum(radius**2-lateral2,0.))
 # Any earlier tool-tip center is outward of entry. A cell is wholly beyond
 # that entire swept burr capsule if its outward face is inward of the cap.
 deep=k[None,None,:] < entry-(cap[:,:,None]+scene.spacing[axis]/2.)/scene.spacing[axis]-1e-9
 unreachable_bones=bones&deep&support[:,:,None]
 bad_top=np.max(np.where(norm,k,-1),axis=2)
 eligible_targets=targets&(k[None,None,:]>bad_top[:,:,None])
 lowest_target=np.min(np.where(eligible_targets,k,labels.shape[-1]),axis=2)
 unreachable_bones &= k[None,None,:]>lowest_target[:,:,None]
 highest_deep_bone=np.max(np.where(unreachable_bones,k,-1),axis=2)
 original_outer_target=np.max(np.where(np.isin(labels,target_ids),k,-1),axis=2)
 interleaved_bones=unreachable_bones&(k[None,None,:]<original_outer_target[:,:,None])
 highest_interleaved=np.max(np.where(interleaved_bones,k,-1),axis=2)
 behind_interleaved=eligible_targets&(k[None,None,:]<highest_interleaved[:,:,None])&support[:,:,None]
 behind=targets&(k[None,None,:]<highest_deep_bone[:,:,None])&support[:,:,None]
 behind_normal_free=behind&(k[None,None,:]>bad_top[:,:,None])
 normal_between=norm&(k[None,None,:]<entry)&support[:,:,None]
 vessel_between=np.isin(labels,vessel_ids)&(k[None,None,:]<entry)&support[:,:,None]
 ij=tuple(map(int,center));center_bone=np.flatnonzero(unreachable_bones[ij]);center_targets=np.flatnonzero(targets[ij])
 chosen=None
 candidates=np.argwhere(behind_normal_free.any(axis=2))
 if len(candidates):
  at=max((tuple(map(int,p)) for p in candidates),key=lambda p:int(behind_normal_free[p].sum()))
  chosen={'transverseNativeIndex':list(at),'retainedTargetBehindBoneDepthRange':span(np.flatnonzero(behind_normal_free[at])),
          'unreachableBoneDepths':np.flatnonzero(unreachable_bones[at]).tolist(),
          'deepestFixedBurrReachCellCenterThreshold':float(entry-(cap[at]+scene.spacing[axis]/2.)/scene.spacing[axis]),
          'sourceColumnRuns':column_runs(labels[at],keep[at])}
 row={'id':access['id'],'nativeAxis':axis,'outwardIndexSign':1,'transverseNativeAxes':transverse,'originalTargetEntryIndex':list(map(int,index)),
      'entryRAS':access['entryPose']['position'],'drillRadiusMm':radius,'currentCenterTargetDepthRange':span(center_targets),
      'sourceCenterTargetDepthRange':span(np.flatnonzero(np.isin(labels[ij],target_ids))),
      'savedCenterEntryTargetStillRetained':bool(targets[ij][int(entry)]),'retainedEligibleTargetCellsInFootprint':int((eligible_targets&support[:,:,None]).sum()),'interleavedRemainingBoneCellsInFootprint':int(interleaved_bones.sum()),'targetBehindTrulyInterleavedBoneCells':int(behind_interleaved.sum()),'centerDeepBoneDepths':center_bone.tolist(),'centerTargetBehindDeepBoneNormalFreeCells':int(behind_normal_free[ij].sum()),
      'burrFootprintColumns':int(support.sum()),'unreachableRemainingBoneCellsInFootprint':int(unreachable_bones.sum()),
      'retainedTargetBehindDeepBoneCellsInFootprint':int(behind.sum()),
      'retainedTargetBehindDeepBoneAndNoNormalAboveCellsInFootprint':int(behind_normal_free.sum()),
      'sourceNormalCellsInwardOfEntryInFootprint':int(normal_between.sum()),'sourceVesselCellsInwardOfEntryInFootprint':int(vessel_between.sum()),
      'representativePositiveColumn':chosen}
 return row,np.moveaxis(behind_normal_free,-1,axis),np.moveaxis(unreachable_bones,-1,axis),np.moveaxis(behind_interleaved,-1,axis)

left=meta['accesses'];assert len(left)==8 and all(a.get('nativeAxis')==0 for a in left)
deferred=meta['deferredAccessPass']['pass'];superior=deferred['accesses'][deferred['accessIndex']:]
assert len(superior)==77
rows=[];groups={}
for name,accesses in [('configuredLeft8',left),('unvisitedLeftIncludingCurrent',left[meta['accessIndex']:]),('deferredSuperior77',superior)]:
 target_union=np.zeros(scene.shape,dtype=bool);bone_union=target_union.copy();interleaved_union=target_union.copy();stats=[]
 for access in accesses:
  row,hidden,bone_hidden,interleaved=analyze(access);stats.append(row);target_union|=hidden;bone_union|=bone_hidden;interleaved_union|=interleaved
 groups[name]={'accesses':len(accesses),'positiveAccesses':sum(r['retainedTargetBehindDeepBoneAndNoNormalAboveCellsInFootprint']>0 for r in stats),
               'distinctRetainedTargetBehindUnreachableBoneNormalFreeCells':int(target_union.sum()),'distinctRemainingBoneBeyondFixedBurrCapsules':int(bone_union.sum()),
               'distinctRetainedTargetBehindTrulyInterleavedBone':int(interleaved_union.sum()),'centerOriginalEntryTargetStillRetainedAccesses':sum(r['savedCenterEntryTargetStillRetained'] for r in stats),'centerColumnPositiveAccesses':sum(r['centerTargetBehindDeepBoneNormalFreeCells']>0 for r in stats),
               'rows':[{k:v for k,v in r.items() if k not in ['representativePositiveColumn','sourceNormalCellsInwardOfEntryInFootprint','sourceVesselCellsInwardOfEntryInFootprint']} for r in stats], 'representativePositiveColumns':[r['representativePositiveColumn'] for r in sorted(stats,key=lambda r:-r['retainedTargetBehindDeepBoneAndNoNormalAboveCellsInFootprint'])[:2] if r['representativePositiveColumn']]}
report={'readOnly':True,'noEnginePreviewStepReplay':True,'position':len(snapshot['actions']),'stateId':snapshot['stateId'],
        'originalSourcePath':str(original),'originalSourceSha256':original_hash,'originalCaseHash':source.source['caseHash'],
        'sceneId':source.identity,'sourceLabelsDigest':array_digest(source.labels),'currentNativeIdentity':identity,
        'remainingOwnedTargetCells':int(remaining_target.sum()),'currentLeftIndex':meta['accessIndex'],'currentLeftId':left[meta['accessIndex']]['id'],
        'deferredSuperiorIndex':deferred['accessIndex'],'groups':groups,
        'method':'Native +index is outward. For each actual route footprint, exact axis-aligned cell-box distance to the radius-R tip sweep bounds the deepest bone cell reachable by its fixed original endpoint. Retained target below a bone cell wholly beyond that capsule is counted only when no original nontarget/nonbone cell lies outward in the same column. Unique unions remove overlap within each configured list.',
        'qualifications':['This is a sufficient geometric endpoint-reach limitation only, not evidence that it causes current poor target yield.','It does not prove a deeper drill pose is legal: intervening target must first be removed, and full physical shaft/working checks, guards and exact public preview remain mandatory.','Target could be accessed around bone through already cleared cavities; these counts do not establish global unreachability.','Source normal/vessel depth totals are descriptive support-cell counts, not claims all block every retained target.','No protection/physics/source/state or route record was changed.']}
out=base/f'review/fixed-entry-bone-summary-{len(snapshot["actions"])}.json'
with out.open('x') as f:json.dump(report,f,indent=2)
print(json.dumps({k:v for k,v in report.items() if k not in ['groups','qualifications']},indent=2))
for name,group in groups.items():
 print(name,json.dumps({k:v for k,v in group.items() if k not in ['rows','representativePositiveColumns']}))
 print('top',json.dumps(sorted(group['rows'],key=lambda r:-r['retainedTargetBehindDeepBoneAndNoNormalAboveCellsInFootprint'])[:2],indent=2))
print(out)
