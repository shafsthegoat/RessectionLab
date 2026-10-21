"""Read-only source-checkpoint/predicted-core geometry; no patient execution."""
import sys,json,copy,time
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path.cwd()/'src'))
from resectionlab.workspace_bundle import load_workspace
from resectionlab.surgical_environment import load_environment
from resectionlab.surgical_engine import SurgicalEngine
from resectionlab.surgical_rl_protection import restore_protection_contract
from resectionlab.full_case_debulking import local_debulking_candidates
out=Path('testcases/outputs/full-case-rehearsal-20261020/geometry');rows=[]
for name,path in [('current','testcases/outputs/full-case-rehearsal-20261020/primary-v2/primary.ressectionlab'),('verified516',str(out/'subvoxel-frontier-516.ressectionlab'))]:
 c,a,w=load_workspace(path);tb=a['surgical_toolbox'];sn=tb['branches'][tb['branchId']];scene=load_environment(path);p,g=restore_protection_contract(scene,sn['protectionContract']);e=SurgicalEngine(scene,protection=p,guard_config=g);e.state=copy.deepcopy(sn['checkpoint']['state']);e.actions=copy.deepcopy(sn['actions']);assert e.state_id==sn['stateId'];before=e.state_id;started=time.perf_counter();proposals=local_debulking_candidates(e,limit=6);rows.append({'case':name,'geometryInspectionOnly':True,'checkpointNotReplayed':True,'physicsExecution':False,'sourceStateId':before,'sourcePosition':len(e.actions),'slot':e.state['slots']['left'],'wallSeconds':time.perf_counter()-started,'proposals':proposals,'stateUnchanged':e.state_id==before})
 if name=='verified516' and proposals and proposals[0]['kind']=='core-positioning':
  positioning=proposals[0];future=copy.deepcopy(e.state['slots']['left']);future.update(position=positioning['action']['path'][-1]['position'],quaternion=positioning['action']['path'][-1]['quaternion']);e.state['slots']['left']=future
  started=time.perf_counter();fan=local_debulking_candidates(e,limit=6);rows.append({'case':'unexecuted-after-checked-positioning','geometryInspectionOnly':True,'checkpointNotReplayed':True,'physicsExecution':False,'sourceStateId':before,'requiredPositioningAction':positioning['action'],'hypotheticalSlot':future,'wallSeconds':time.perf_counter()-started,'proposals':fan})
(out/'pivot_fan_geometry.json').write_text(json.dumps(rows,indent=2))
for row in rows:
 print(json.dumps({'case':row['case'],'position':row.get('sourcePosition'),'wallSeconds':row['wallSeconds'],'proposals':[{k:v for k,v in x.items() if k not in ('action','assumptions')} for x in row['proposals']]},indent=2))
