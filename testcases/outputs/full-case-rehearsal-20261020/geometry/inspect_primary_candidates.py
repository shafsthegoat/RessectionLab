"""READ ONLY checkpoint geometry inspection, NOT an execution or replay proof."""
import sys,json,copy,time
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path.cwd()/'src'))
from resectionlab.workspace_bundle import load_workspace
from resectionlab.surgical_environment import load_environment
from resectionlab.surgical_engine import SurgicalEngine
from resectionlab.surgical_rl_protection import restore_protection_contract
from resectionlab.full_case_geometry import next_removal_candidates
out=Path('testcases/outputs/full-case-rehearsal-20261020/geometry');p=Path('testcases/outputs/full-case-rehearsal-20261020/primary-v2/primary.ressectionlab');c,a,w=load_workspace(p);tb=a['surgical_toolbox'];sn=tb['branches'][tb['branchId']];m=tb['fullCase'][tb['branchId']];s=load_environment(p);pr,g=restore_protection_contract(s,sn['protectionContract']);e=SurgicalEngine(s,protection=pr,guard_config=g)
# Explicitly authorized read-only geometry reconstruction from root-verified
# persisted checkpoint. No step/preview/restore/execution claims are made.
e.state=copy.deepcopy(sn['checkpoint']['state']);e.actions=copy.deepcopy(sn['actions'])
e.state['slots']['left'].update(instrument='cusa', active=False, inserted=False, grip=[], parameters={'radiusMm':.6,'forceN':0.})
# Hypothetical future outside CUSA setup, strictly proposal inspection only.
removed=set(e.state['removed']);lab=s.labels.copy().ravel();lab[list(removed)]=0;lab=lab.reshape(s.shape);z=np.arange(s.shape[2]);top=np.max(np.where(lab!=0,z,-1),2);first=np.take_along_axis(lab,np.maximum(top,0)[...,None],2)[...,0];exposed=np.argwhere(first==2)
r={'geometryInspectionOnly':True,'checkpointNotReplayed':True,'sceneId':s.identity,'checkpointStateId':sn['stateId'],'position':len(e.actions),'stage':m['stage'],'accessIndex':m['accessIndex'],'currentSlot':e.state['slots']['left'],'accesses':[]}
for idx in [1,2,m['accessIndex']]:
 if any(x['index']==idx for x in r['accesses']):continue
 access=copy.deepcopy(m['accesses'][idx]);center=np.array(access['entryPose']['position']);near=[]
 for ij in exposed:
  xyz=s.points([np.ravel_multi_index((*ij,top[tuple(ij)]),s.shape)])[0];distance=np.linalg.norm(xyz[:2]-center[:2])
  if distance<=7:near.append({'ijk':[*map(int,ij),int(top[tuple(ij)])],'ras':xyz.tolist(),'distanceFromAccessCenterMm':float(distance)})
 t=time.perf_counter();candidates=next_removal_candidates(e,access,limit=8,radii=[.6,1.,1.3,1.5,2.]);r['accesses'].append({'index':idx,'access':access,'exposedTargetColumns':sorted(near,key=lambda x:x['distanceFromAccessCenterMm']),'candidateCount':len(candidates),'candidates':candidates,'wallSeconds':time.perf_counter()-t})
(out/'primary_v2_candidate_inspection.json').write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
