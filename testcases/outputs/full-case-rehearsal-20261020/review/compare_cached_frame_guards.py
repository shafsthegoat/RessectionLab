"""Exactly3 existing vascular cached frames under each guard; no physics replay."""
from pathlib import Path
import sys,json,time,hashlib,ast
from contextlib import contextmanager
from collections import Counter
ROOT=Path(__file__).resolve().parents[4];sys.path.insert(0,str(ROOT/'src'))
from resectionlab.surgical_engine import SurgicalEngine
from resectionlab.surgical_environment import load_environment
from resectionlab.surgical_frame_replay import SurgicalFrameCache
from resectionlab.surgical_replay_cache import SurgicalReplayCache
from resectionlab.surgical_verified_cache import VerifiedSurgicalCache,implementation_fingerprint
from resectionlab.workspace_bundle import load_workspace
from resectionlab import surgical_telemetry
from resectionlab.core import array_digest
from readonly_step_monitor import ReadOnlyStepMonitor,ForbiddenPhysicsStep
B=ROOT/'testcases/outputs/full-case-rehearsal-20261020';V=B/'review'
native=B/'gui/vascular-observability-28962e66.ressectionlab';receipt=V/'vascular-observability-28962e66-canonical.json'
canonical=json.loads(receipt.read_text());fingerprint=implementation_fingerprint();assert fingerprint==canonical['implementationFingerprint']
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(native)==canonical['nativeSha256']
source=ROOT/'testcases/cases/meningioma-sub-06-standard-v2.ressectionlab';assert sha(source)==canonical['originalSha256']
_,artifacts,_=load_workspace(native);snapshot=artifacts['surgical_toolbox']['branches']['full-vascular'];scene=load_environment(native)
verified=VerifiedSurgicalCache();frames=SurgicalFrameCache();replay=SurgicalReplayCache(verified_cache=verified)
with ReadOnlyStepMonitor(SurgicalEngine) as guard:
 loaded=verified.restore(SurgicalEngine,scene,snapshot,include_history=True)
 assert loaded is not None,'Cache-only admission missing; no replay fallback'
 current,history=loaded
 admission=frames.admit(current,snapshot,replay,verified_history=history)
assert not guard.attempts and verified.hits==2 and verified.publications==0
selections=[]
for position in (47,55,68):
 item=frames.store.load(admission,position);assert item is not None
 index=len(item['frames'])//2
 selections.append((position,index,item['frames'][index]))
@contextmanager
def legacy():
 calls=Counter()
 def profile(frame,event,arg):
  if event=='call' and frame.f_code.co_name=='step' and isinstance(frame.f_locals.get('self'),SurgicalEngine):
   calls['step']+=1;raise ForbiddenPhysicsStep('legacy public step attempt')
 sys.setprofile(profile)
 try:yield calls
 finally:sys.setprofile(None)
rows=[];stateids={};sampletimes={}
for label in ('legacy-profile','monitoring'):
 samples=[];ids=[]
 monitor=ReadOnlyStepMonitor(SurgicalEngine) if label=='monitoring' else None
 with (monitor if monitor else legacy()) as active:
  for position,index,expected in selections:
   if monitor:monitor.check()
   started=time.perf_counter()
   observed,ancestry,playback=frames.seek(current,admission.snapshot,position,index,replay,
    verified_history=admission.history,admission=admission,display_only=True)
   samples.append(time.perf_counter()-started)
   assert observed.state_id==expected['stateId'] and observed.scene is scene
   assert ancestry is admission.history and playback['frameIndex']==index
   assert playback['observedTimeSeconds']==expected['observedTimeSeconds']
   ids.append(observed.state_id)
   if monitor:monitor.check()
 assert not (monitor.attempts if monitor else active)
 stateids[label]=ids;sampletimes[label]=samples
 rows.append({'guard':label,'seeks':len(samples),'seekSeconds':samples,'stateIds':ids,'publicStepAttempts':{},
  'nonStepCallbacks':monitor.other_callbacks if monitor else None})
assert stateids['legacy-profile']==stateids['monitoring']
assert verified.publications==frames.store.publications==frames.store.misses==0
assert sha(native)==canonical['nativeSha256'] and sha(source)==canonical['originalSha256']
assert implementation_fingerprint()==fingerprint and array_digest(scene.labels)==canonical['labelArrayHash']
# Structural check that EVERY old main() assertion survives verbatim in the new main().
def assertions(path):
 tree=ast.parse(path.read_text());main=next(v for v in tree.body if isinstance(v,ast.FunctionDef) and v.name=='main')
 return Counter(ast.dump(v.test,include_attributes=False) for v in ast.walk(main) if isinstance(v,ast.Assert))
old_assertions=assertions(V/'verify_every_cached_frame.py');new_assertions=assertions(V/'verify_every_cached_frame_monitoring.py')
assert not old_assertions-new_assertions
out={'result':'passed:3 cached frames under each guard','nativeSha256':canonical['nativeSha256'],'implementationFingerprint':fingerprint,
 'selections':[{'actionIndex':p,'frameIndex':i,'stateId':e['stateId']} for p,i,e in selections],
 'results':rows,'allLegacyMainAssertionAstNodesRetained':sum(old_assertions.values()),
 'zeroPublicStepAttempts':True,'zeroCacheMissesAndPublications':True,'sourceAndNativeUnchanged':True,
 'scope':'Exactly3 existing authenticated vascular frames sought with each guard. No physical replay or full patient coverage; no GUI/renderer or episode throughput claim. Legacy running verifier not touched.'}
p=V/'monitoring-guard-three-cached-frames-comparison.json';p.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
