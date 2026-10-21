"""Frozen-source recorded-path accounting and residual configured-route ranking.
No engine construction, replay, preview or production mutation.
"""
import sys,json,math,hashlib
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
sys.path.insert(0,str(Path.cwd()/'src'))
from resectionlab.workspace_bundle import load_workspace
from resectionlab.surgical_environment import load_environment
out=Path('testcases/outputs/full-case-rehearsal-20261020/geometry');p=Path('testcases/outputs/full-case-rehearsal-20261020/primary-v2/primary.ressectionlab');c,a,w=load_workspace(p);tb=a['surgical_toolbox'];sn=tb['branches'][tb['branchId']];meta=tb['fullCase'][tb['branchId']];scene=load_environment(p);actions=sn['actions'];phases=meta['phases'];records=[]
def phase_at(pos):
 return next((r['id'] for r in phases if r['startPosition']<pos<=r['endPosition']),'prefix')
def micros(poses,duration):
 if not len(poses):return 0
 # Initial current->first segment is unavailable for old partial strokes;
 # use one initial checked fraction, then exact recorded interior segments.
 result=1
 for first,last in zip(poses,poses[1:]):
  d=float(np.linalg.norm(np.array(first['position'])-last['position']));angle=float((Rotation.from_quat(last['quaternion'])*Rotation.from_quat(first['quaternion']).inv()).magnitude());result+=max(1,math.ceil((d+40*angle)/(1/3)),math.ceil(duration/max(1,len(poses))/.1))
 return result
for pos,action in enumerate(actions,1):
 phase=phase_at(pos);path=action.get('path',[])
 if not phase.startswith('core-') or action['operation']!='apply' or not path:continue
 xyz=np.array([x['position'] for x in path]);arc=float(np.linalg.norm(np.diff(xyz,axis=0),axis=1).sum()) if len(path)>1 else 0
 ang=float((Rotation.from_quat(path[-1]['quaternion'])*Rotation.from_quat(path[0]['quaternion']).inv()).magnitude()) if len(path)>1 else 0
 kind='fan' if ang>1e-5 else ('straight' if arc>.01 else 'static')
 estimate=micros(path,action.get('duration',1));coarse=[]
 # Geometry-only resampling estimate, NOT an admitted changed action.
 if len(path)>1:
  selected=[path[0]];distance=0.;last=xyz[0]
  for idx in range(1,len(path)):
   distance+=float(np.linalg.norm(xyz[idx]-last));last=xyz[idx]
   if distance>=.3-1e-8:selected.append(path[idx]);distance=0.
  if selected[-1]!=path[-1]:selected.append(path[-1])
  coarse=selected
 else:coarse=path
 records.append({'position':pos,'phase':phase,'kind':kind,'poses':len(path),'tipTravelMm':arc,'angleRad':ang,'microstepsInteriorEstimate':estimate,'coarse03PoseEstimate':len(coarse),'coarse03MicrostepsInteriorEstimate':micros(coarse,action.get('duration',1))})
summary={}
for kind in ['fan','straight','static']:
 rows=[r for r in records if r['kind']==kind];summary[kind]={'actions':len(rows),'poses':sum(r['poses'] for r in rows),'microstepsInteriorEstimate':sum(r['microstepsInteriorEstimate'] for r in rows),'coarse03PoseEstimate':sum(r['coarse03PoseEstimate'] for r in rows),'coarse03MicrostepsInteriorEstimate':sum(r['coarse03MicrostepsInteriorEstimate'] for r in rows),'medianPoses':float(np.median([r['poses'] for r in rows])) if rows else 0}
res=scene.labels.copy();removed=np.array(sn['checkpoint']['state']['removed'],dtype=np.int64);res.ravel()[removed]=0;z=np.arange(scene.shape[2]);normal=(scene.labels!=0)&(scene.labels!=2)&(scene.labels!=3);bad=np.max(np.where(normal,z,-1),axis=2);eligible=(scene.labels==2)&(z[None,None,:]>bad[:,:,None]);remaining=(res==2)&eligible;xx,yy=np.ogrid[:scene.shape[0],:scene.shape[1]];rank=[]
for index,access in enumerate(meta['accesses']):
 i,j,_=access['targetEntryIndex'];disk=(xx-i)**2+(yy-j)**2<=max(1.,access['drillRadiusMm']-1.)**2;initial=int(eligible[disk].sum());retained=int(remaining[disk].sum());target=np.flatnonzero(remaining[i,j]);rank.append({'index':index,'id':access['id'],'entryRAS':access['entryPose']['position'],'initialDiskTarget':initial,'retainedEligibleDiskTarget':retained,'retainedFraction':retained/max(1,initial),'retainedCenterColumnTarget':len(target),'drillRadiusMm':access['drillRadiusMm']})
logs=[]
for line in (p.parent/'execution.jsonl').read_text().splitlines():
 try:r=json.loads(line)
 except:continue
 if r.get('phase','').startswith('core-') and r.get('wallSeconds') is not None:logs.append(r)
recent=logs[-12:]
receipt={'readOnly':True,'noPhysics':True,'sourceStateId':sn['stateId'],'position':len(actions),'targetRemovedMm3':int(np.count_nonzero(scene.labels.ravel()[removed]==2)),'accessIndex':meta['accessIndex'],'stage':meta['stage'],'pathSummary':summary,'pathRecords':records,'configuredAccessesInOriginalOrder':rank,'topResidualRemainingAccesses':sorted([r for r in rank if r['index']>meta['accessIndex']],key=lambda r:-r['retainedEligibleDiskTarget'])[:10],'recentCoreSliceWallSeconds':[r['wallSeconds'] for r in recent],'recentCoreSlicePositions':[r['position'] for r in recent],'recentCoreSliceTargetMm3':[r['metrics']['targetRemovedMm3'] for r in recent],'methodLimit':'Resampled paths are hypothetical counts only, not collision-admitted or state-equivalent actions. Counts exclude unknown original current-to-first fractions and stopped-action truncation. Residual disk rank is optimistic configured-column capacity, not predicted physical removal.','productionSourceSha256':{str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in [Path('src/resectionlab/full_case_debulking.py'),Path('src/resectionlab/surgical_motion.py'),Path('src/resectionlab/geometry.py')]}}
(out/'frozen_core_roi.json').write_text(json.dumps(receipt,indent=2));print(json.dumps({k:v for k,v in receipt.items() if k not in ['pathRecords','configuredAccessesInOriginalOrder','productionSourceSha256']},indent=2));print('originalNext',rank[meta['accessIndex']:meta['accessIndex']+8])
