"""One focused public-engine R6 preview on preserved exact verified prefix."""
import sys,json,time,hashlib
from pathlib import Path
sys.path.insert(0,str(Path.cwd()/'src'))
from resectionlab.workspace_bundle import load_workspace
from resectionlab.surgical_environment import load_environment
from resectionlab.surgical_engine import SurgicalEngine
from resectionlab.full_case_debulking import local_debulking_candidates
out=Path('testcases/outputs/full-case-rehearsal-20261020/geometry');path=out/'subvoxel-frontier-516.ressectionlab';case,art,w=load_workspace(path);tb=art['surgical_toolbox'];sn=tb['branches'][tb['branchId']];scene=load_environment(path);rows=[]
def emit(row):
 print(json.dumps(row),flush=True);rows.append(row);(out/'r6_core_public_preview.json').write_text(json.dumps(rows,indent=2))
start=time.perf_counter();e=SurgicalEngine.restore(scene,sn);emit({'stage':'public-restore','wallSeconds':time.perf_counter()-start,'stateId':e.state_id,'snapshotStateId':sn['stateId'],'actions':len(e.actions),'metrics':e.metrics(),'sourceCaseHash':case.semantic_hash})
start=time.perf_counter();candidates=local_debulking_candidates(e,radius_mm=6.,limit=1);emit({'stage':'positioning-proposal','wallSeconds':time.perf_counter()-start,'records':candidates})
assert candidates and candidates[0]['kind']=='core-positioning'
action=candidates[0]['action'];before=e.state_id;start=time.perf_counter();p=e.preview(action);emit({'stage':'positioning-preview','wallSeconds':time.perf_counter()-start,'result':{k:v for k,v in p.items() if k!='view'},'committedStateUnchanged':e.state_id==before});assert p['status']=='accepted' and not p['metrics']['removedByLabelMm3'].get('brain',0) and not p['metrics']['removedByLabelMm3'].get('vessels',0)
start=time.perf_counter();result=e.step(action);emit({'stage':'positioning-step','wallSeconds':time.perf_counter()-start,'result':result,'previewEqualsCommitStateId':result['stateId']==p['stateId']});assert result['stateId']==p['stateId']
start=time.perf_counter();candidates=local_debulking_candidates(e,radius_mm=6.,limit=1);emit({'stage':'core-proposal','wallSeconds':time.perf_counter()-start,'records':candidates});assert candidates and candidates[0]['kind']=='core-removal'
before=e.state_id;start=time.perf_counter();p=e.preview(candidates[0]['action']);emit({'stage':'r6-core-preview','wallSeconds':time.perf_counter()-start,'action':candidates[0]['action'],'result':{k:v for k,v in p.items() if k!='view'},'committedStateUnchanged':e.state_id==before,'previewOnly':True,'targetDeltaMm3':p['metrics']['targetRemovedMm3']-e.metrics()['targetRemovedMm3']})
