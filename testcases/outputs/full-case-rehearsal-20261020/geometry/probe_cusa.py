import sys,json,time,pickle
from pathlib import Path
sys.path.insert(0,str(Path.cwd()/'src'))
from resectionlab.surgical_environment import load_environment
from resectionlab.surgical_engine import SurgicalEngine
from resectionlab.full_case_geometry import access_candidates,next_removal_candidates
out=Path('testcases/outputs/full-case-rehearsal-20261020/geometry');snap=Path('testcases/outputs/full-case-rehearsal-20261020/heartbeat/broad-probe-snapshot.json')
log=[]
def emit(v):
 print(json.dumps(v),flush=True);log.append(v);(out/'cusa_probe_results.json').write_text(json.dumps(log,indent=2))
s=load_environment('testcases/cases/meningioma-sub-06-standard-v2.ressectionlab');t=time.perf_counter();e=SurgicalEngine.restore(s,json.loads(snap.read_text()));emit({'stage':'restored','wall':time.perf_counter()-t,'metrics':e.metrics()})
access=access_candidates(e)[0]
def commit(a):
 t=time.perf_counter();r=e.step(a);emit({'action':a,'result':r,'wall':time.perf_counter()-t});(out/'cusa_probe_snapshot.json').write_text(json.dumps(e.snapshot()));return r
for a in [{'slot':'left','operation':'withdraw','pose':access['outsidePose'],'duration':.1},{'slot':'left','operation':'exchange','instrument':'cusa','duration':.1},{'slot':'left','operation':'place','pose':access['outsidePose'],'parameters':{'radiusMm':1.5},'duration':.1}]:
 if commit(a)['status']=='rejected':sys.exit(2)
for index in range(12):
 t=time.perf_counter();c=next_removal_candidates(e,access,slot='left',limit=4);emit({'stage':'candidates','count':len(c),'wall':time.perf_counter()-t,'actions':c})
 if not c:break
 progressed=False
 for action in c:
  t=time.perf_counter();r=e.preview(action);emit({'stage':'preview','action':action,'wall':time.perf_counter()-t,'result':{k:v for k,v in r.items() if k!='view'}})
  before=e.metrics()['targetRemovedMm3']
  if r['status']!='rejected' and r['metrics']['targetRemovedMm3']>before and not r['metrics']['removedByLabelMm3'].get('brain',0):
   commit(action);progressed=True;break
 if not progressed:break
 if e.metrics()['targetRemovedMm3']>1000:break
emit({'stage':'finished','targetRemovedMm3':e.metrics()['targetRemovedMm3'],'metrics':e.metrics(),'stateId':e.state_id})
