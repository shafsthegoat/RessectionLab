"""Read-only saved-checkpoint assistant field geometry, not execution/replay."""
import sys,json,copy,time,hashlib
from pathlib import Path
sys.path.insert(0,str(Path.cwd()/'src'))
from resectionlab.workspace_bundle import load_workspace
from resectionlab.surgical_environment import load_environment
from resectionlab.surgical_engine import SurgicalEngine
from resectionlab.surgical_rl_protection import restore_protection_contract
from resectionlab.full_case_assist import assistant_field_candidate
out=Path('testcases/outputs/full-case-rehearsal-20261020/geometry');path=Path('testcases/outputs/full-case-rehearsal-20261020/primary-v2/primary.ressectionlab');c,a,w=load_workspace(path);t=a['surgical_toolbox'];sn=t['branches'][t['branchId']];m=t['fullCase'][t['branchId']];s=load_environment(path);p,g=restore_protection_contract(s,sn['protectionContract']);e=SurgicalEngine(s,protection=p,guard_config=g);e.state=copy.deepcopy(sn['checkpoint']['state']);e.actions=copy.deepcopy(sn['actions']);assert e.state_id==sn['stateId'];access=m['accesses'][min(m['accessIndex'],len(m['accesses'])-1)];before=e.state_id;now=time.perf_counter();candidate=assistant_field_candidate(e,access);receipt={'geometryInspectionOnly':True,'checkpointNotReplayed':True,'physicsExecution':False,'sourceNative':str(path),'sourceStateId':before,'sourceSceneId':s.identity,'sourcePosition':len(e.actions),'currentAccess':access,'wallSeconds':time.perf_counter()-now,'candidate':candidate,'stateUnchanged':e.state_id==before,'sourceCaseHash':c.semantic_hash};(out/'assistant_field_proposal.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt,indent=2))
