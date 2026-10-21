"""Read-only actual saved-checkpoint withdrawal geometry; no execution/replay."""
import sys,json,copy,time
from pathlib import Path
sys.path.insert(0,str(Path.cwd()/'src'))
from resectionlab.workspace_bundle import load_workspace
from resectionlab.surgical_environment import load_environment
from resectionlab.surgical_engine import SurgicalEngine
from resectionlab.surgical_rl_protection import restore_protection_contract
from resectionlab.full_case_withdrawal import withdrawal_candidate
out=Path('testcases/outputs/full-case-rehearsal-20261020/geometry');path=Path('testcases/outputs/full-case-rehearsal-20261020/primary-v2/primary.ressectionlab');c,a,w=load_workspace(path);tb=a['surgical_toolbox'];sn=tb['branches'][tb['branchId']];s=load_environment(path);p,g=restore_protection_contract(s,sn['protectionContract']);e=SurgicalEngine(s,protection=p,guard_config=g);e.state=copy.deepcopy(sn['checkpoint']['state']);e.actions=copy.deepcopy(sn['actions']);assert e.state_id==sn['stateId'];before=e.state_id;started=time.perf_counter();proposal=withdrawal_candidate(e,'left');receipt={'geometryInspectionOnly':True,'checkpointNotReplayed':True,'physicsExecution':False,'sourcePosition':len(e.actions),'sourceStateId':before,'sourceSceneId':s.identity,'slot':e.state['slots']['left'],'wallSeconds':time.perf_counter()-started,'proposal':proposal,'stateUnchanged':e.state_id==before};(out/'withdrawal_proposal.json').write_text(json.dumps(receipt,indent=2));print(json.dumps({k:v for k,v in receipt.items() if k!='proposal'},indent=2));print('proposal',None if proposal is None else {'operation':proposal['operation'],'poses':len(proposal.get('path',[])),'endpoint':proposal.get('path',[proposal.get('pose')])[-1]})
