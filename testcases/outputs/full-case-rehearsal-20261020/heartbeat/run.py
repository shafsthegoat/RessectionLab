import sys,json,copy,time,hashlib
from pathlib import Path
sys.path.insert(0,str(Path.cwd()/'src'))
from resectionlab.workspace_bundle import load_workspace,save_workspace
from resectionlab.surgical_environment import load_environment
from resectionlab.surgical_engine import SurgicalEngine
from resectionlab.surgical_rl_protection import GuardConfig,load_protection
src=Path('testcases/cases/meningioma-sub-06-standard-v2.ressectionlab')
out=Path('testcases/outputs/full-case-rehearsal-20261020/heartbeat')
case,artifacts,workspace=load_workspace(src);scene=load_environment(src)
snapshot=artifacts['surgical_toolbox']['branches']['main']
assert len(snapshot['actions'])==0
engine=SurgicalEngine(scene,protection=load_protection(scene,Path("testcases/environments/meningioma-sub-06/rl-protection.json"),strict=True),guard_config=GuardConfig())
assert engine.guard_report()['configured']
actions=json.load(open('artifacts/integrated-rehearsal-v1/protocol.json'))['cases'][0]['actions'][:16]
rows=[]
for i,a in enumerate(actions):
 t=time.perf_counter();p=engine.preview(a);r=engine.step(a)
 assert p['stateId']==r['stateId']
 rows.append({'index':i,'action':a,'status':r['status'],'reason':r.get('reason'),'stateId':r['stateId'],'metrics':r['metrics'],'wallSeconds':time.perf_counter()-t})
 print(i,a['operation'],r['status'],r.get('reason'),r['metrics']['targetRemovedMm3'],flush=True)
 assert r['status']!='rejected'
record=copy.deepcopy(artifacts['surgical_toolbox']);record.update(branches={'heartbeat':engine.snapshot()},branchId='heartbeat',position=len(engine.actions),parents={'heartbeat':{'branchId':'main','position':0}},timeline={'heartbeat':engine.history})
artifacts['surgical_toolbox']=record
save_workspace(case,out/'heartbeat.ressectionlab',sources=workspace['sources'],imaging=workspace['imagingState'],episode=workspace['episode'],selection=workspace['selection'],originals=workspace['originals'],artifacts=artifacts)
(out/'actions.json').write_text(json.dumps(rows,indent=2))
print(json.dumps({'stateId':engine.state_id,'metrics':engine.metrics(),'originalSha256':hashlib.sha256(src.read_bytes()).hexdigest()}),flush=True)
