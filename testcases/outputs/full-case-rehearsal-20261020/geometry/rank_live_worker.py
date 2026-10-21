"""Read-only proposal reconstruction of exact saved live-worker checkpoint."""
import sys,json,copy,time
from pathlib import Path
sys.path.insert(0,str(Path.cwd()/'src'))
from resectionlab.surgical_environment import load_environment
from resectionlab.surgical_engine import SurgicalEngine
from resectionlab.surgical_rl_protection import restore_protection_contract
from resectionlab.full_case_geometry import next_removal_candidates
out=Path('testcases/outputs/full-case-rehearsal-20261020/geometry');sn=json.loads((out/'taper_snapshot.json').read_text());s=load_environment('testcases/cases/meningioma-sub-06-standard-v2.ressectionlab');p,g=restore_protection_contract(s,sn['protectionContract']);e=SurgicalEngine(s,protection=p,guard_config=g);e.state=copy.deepcopy(sn['checkpoint']['state']);e.actions=copy.deepcopy(sn['actions']);access=json.loads((out/'dense_access_candidates.json').read_text())[1];t=time.perf_counter();a=next_removal_candidates(e,access,limit=6,radii=[.3,.45,.6,1.,1.3,1.5]);result={'geometryInspectionOnly':True,'checkpointNotReplayed':True,'snapshotStateId':sn['stateId'],'snapshotActionCount':len(e.actions),'access':access,'wallSeconds':time.perf_counter()-t,'actions':a};(out/'live_worker_candidates.json').write_text(json.dumps(result,indent=2));print(json.dumps({'count':len(a),'wallSeconds':result['wallSeconds'],'actions':[(x['parameters'],x['path'][-1],len(x['path'])) for x in a]},indent=2))
