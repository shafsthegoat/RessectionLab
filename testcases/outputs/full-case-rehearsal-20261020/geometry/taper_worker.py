import sys,json,time
from pathlib import Path
sys.path.insert(0,str(Path.cwd()/'src'))
from resectionlab.surgical_environment import load_environment
from resectionlab.surgical_engine import SurgicalEngine
from resectionlab.full_case_geometry import access_candidates,next_removal_candidates
out=Path('testcases/outputs/full-case-rehearsal-20261020/geometry');rows=[]
def emit(v):
 print(json.dumps(v),flush=True);rows.append(v);(out/'taper_results.json').write_text(json.dumps(rows,indent=2))
s=load_environment('testcases/cases/meningioma-sub-06-standard-v2.ressectionlab');t=time.perf_counter();e=SurgicalEngine.restore(s,json.loads((out/'cusa_probe_snapshot.json').read_text()));emit({'stage':'restored','wall':time.perf_counter()-t,'metrics':e.metrics()})
access=access_candidates(e)[0]
def command(action,preview=False):
 t=time.perf_counter();r=e.preview(action) if preview else e.step(action)
 emit({'stage':'preview' if preview else 'step','action':action,'wall':time.perf_counter()-t,'result':{k:v for k,v in r.items() if k not in ('view','deltas')}})
 if not preview:
  (out/'taper_snapshot.json').write_text(json.dumps(e.snapshot()))
 return r
for action in [{'slot':'left','operation':'withdraw','pose':access['outsidePose'],'duration':.1},{'slot':'left','operation':'exchange','instrument':'cranial_drill','parameters':{'radiusMm':3.,'forceN':20.,'rpm':80000.,'duty':1.},'duration':.1},{'slot':'left','operation':'place','pose':access['outsidePose'],'duration':.1},{'slot':'left','operation':'apply','pose':access['entryPose'],'duration':10.}]:
 r=command(action)
 if r['status']=='rejected':break
else:
 for action in [{'slot':'left','operation':'withdraw','pose':access['outsidePose'],'duration':.1},{'slot':'left','operation':'exchange','instrument':'cusa','parameters':{'radiusMm':1.3},'duration':.1},{'slot':'left','operation':'place','pose':access['outsidePose'],'duration':.1},{'slot':'left','operation':'apply','pose':access['entryPose'],'duration':2.}]:
  p=command(action,preview=True)
  if p['status']=='rejected' or p['metrics']['removedByLabelMm3'].get('brain',0) or p['metrics']['removedByLabelMm3'].get('vessels',0):break
  command(action)
 for iteration in range(8):
  t=time.perf_counter();candidates=next_removal_candidates(e,access,slot='left',limit=3,radii=[1.3,2.,2.5,3.,4.,6.]);emit({'stage':'candidates','count':len(candidates),'wall':time.perf_counter()-t,'actions':candidates})
  if not candidates:break
  before=e.metrics()['targetRemovedMm3'];progress=False
  for action in candidates:
   p=command(action,preview=True)
   if p['status']!='rejected' and p['metrics']['targetRemovedMm3']>before and not p['metrics']['removedByLabelMm3'].get('brain',0) and not p['metrics']['removedByLabelMm3'].get('vessels',0):command(action);progress=True;break
  if not progress or e.metrics()['targetRemovedMm3']>1000:break
emit({'stage':'ready','metrics':e.metrics(),'stateId':e.state_id})
for line in sys.stdin:
 try:
  request=json.loads(line)
  if request.get('op')=='quit':break
  if request.get('op')=='candidates':
   t=time.perf_counter();a=next_removal_candidates(e,request.get('access',access),slot=request.get('slot','left'),limit=request.get('limit',3),radii=request.get('radii'));emit({'stage':'candidates','wall':time.perf_counter()-t,'actions':a})
  else:command(request['action'],preview=request.get('preview',False))
 except Exception as error:emit({'stage':'error','error':str(error)})
