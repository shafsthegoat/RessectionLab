"""Read-only local proposal inspection, not transition execution/replay."""
import sys,json,copy,time
from pathlib import Path
sys.path.insert(0,str(Path.cwd()/'src'))
from resectionlab.surgical_environment import load_environment
from resectionlab.surgical_engine import SurgicalEngine
from resectionlab.surgical_rl_protection import restore_protection_contract
from resectionlab.full_case_debulking import local_debulking_candidates
out=Path('testcases/outputs/full-case-rehearsal-20261020/geometry');sn=json.loads((out/'taper_snapshot.json').read_text());s=load_environment('testcases/cases/meningioma-sub-06-standard-v2.ressectionlab');p,g=restore_protection_contract(s,sn['protectionContract']);e=SurgicalEngine(s,protection=p,guard_config=g);e.state=copy.deepcopy(sn['checkpoint']['state']);e.actions=copy.deepcopy(sn['actions']);assert e.state_id==sn['stateId'];t=time.perf_counter();proposals=local_debulking_candidates(e);receipt={'geometryInspectionOnly':True,'checkpointNotReplayed':True,'stateId':sn['stateId'],'sourceSceneId':s.identity,'physicsExecution':False,'wallSeconds':time.perf_counter()-t,'proposals':proposals};(out/'local_debulking_proposals.json').write_text(json.dumps(receipt,indent=2));print(json.dumps({'count':len(proposals),'wallSeconds':receipt['wallSeconds'],'proposals':[{k:v for k,v in r.items() if k not in ('action','assumptions')} for r in proposals]},indent=2))
