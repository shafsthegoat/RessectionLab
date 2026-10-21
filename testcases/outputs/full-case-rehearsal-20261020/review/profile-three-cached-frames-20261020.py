"""Exactly three authenticated cached frame seeks; no physics fallback permitted."""
from pathlib import Path
import sys,json,time,hashlib,cProfile,pstats
ROOT=Path(__file__).resolve().parents[4];sys.path.insert(0,str(ROOT/'src'))
from resectionlab.surgical_engine import SurgicalEngine
from resectionlab.surgical_environment import load_environment
from resectionlab.surgical_frame_replay import SurgicalFrameCache
from resectionlab.surgical_replay_cache import SurgicalReplayCache
from resectionlab.surgical_verified_cache import VerifiedSurgicalCache,implementation_fingerprint
from resectionlab.workspace_bundle import load_workspace
from resectionlab import surgical_telemetry
from resectionlab.core import array_digest
from readonly_step_monitor import ReadOnlyStepMonitor
B=ROOT/'testcases/outputs/full-case-rehearsal-20261020';V=B/'review'
native=B/'gui/vascular-observability-28962e66.ressectionlab';receipt=V/'vascular-observability-28962e66-canonical.json'
canonical=json.loads(receipt.read_text());fingerprint=implementation_fingerprint();assert fingerprint==canonical['implementationFingerprint']
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
source=ROOT/'testcases/cases/meningioma-sub-06-standard-v2.ressectionlab'
assert sha(native)==canonical['nativeSha256'] and sha(source)==canonical['originalSha256']
_,artifacts,_=load_workspace(native);snapshot=artifacts['surgical_toolbox']['branches']['full-vascular'];scene=load_environment(native)
verified=VerifiedSurgicalCache();frames=SurgicalFrameCache();replay=SurgicalReplayCache(verified_cache=verified)
rows=[]
def ranking(stats,which,n=24):
 entries=[]
 for key,value in sorted(stats.stats.items(),key=lambda kv:kv[1][3 if which=='cumulative' else 2],reverse=True)[:n]:
  cc,nc,tt,ct,_=value;file,line,name=key
  entries.append({'file':file,'line':line,'name':name,'primitiveCalls':cc,'totalCalls':nc,'selfSeconds':tt,'cumulativeSeconds':ct})
 return entries
with ReadOnlyStepMonitor(SurgicalEngine) as guard:
 loaded=verified.restore(SurgicalEngine,scene,snapshot,include_history=True)
 assert loaded is not None,'Cache-only admission missing; no replay fallback'
 current,history=loaded
 admission=frames.admit(current,snapshot,replay,verified_history=history)
 original_state=current.state_id
 for position in (2,30,51):
  item=frames.store.load(admission,position);assert item is not None
  index=len(item['frames'])//2;expected=item['frames'][index]
  guard.check();profile=cProfile.Profile();started=time.perf_counter()
  # Approved instrumentation only between guard checks; sys.monitoring remains
  # globally armed throughout and prevents every legacy-predicate public step.
  profile.enable()
  try:
   observed,ancestry,playback=frames.seek(current,admission.snapshot,position,index,replay,
    verified_history=admission.history,admission=admission,display_only=True)
  finally:profile.disable()
  elapsed=time.perf_counter()-started;guard.check()
  assert observed.state_id==expected['stateId'] and observed.scene is scene
  assert ancestry is admission.history and playback['frameIndex']==index
  assert playback['observedTimeSeconds']==expected['observedTimeSeconds']
  output=V/f'cached-frame-{position}-instrumented-20261020.pstats';profile.dump_stats(str(output));stats=pstats.Stats(profile)
  rows.append({'actionIndex':position,'frameIndex':index,'stateId':observed.state_id,'instrumentedWallSeconds':elapsed,
               'pstats':str(output),'totalCalls':stats.total_calls,'primitiveCalls':stats.prim_calls,
               'topCumulative':ranking(stats,'cumulative'),'topSelf':ranking(stats,'self')})
 assert current.state_id==original_state
assert not guard.attempts and verified.hits==2 and verified.publications==verified.misses==0
assert frames.store.publications==frames.store.misses==0 and frames.store.hits==6
assert sha(native)==canonical['nativeSha256'] and sha(source)==canonical['originalSha256']
assert implementation_fingerprint()==fingerprint and array_digest(scene.labels)==canonical['labelArrayHash']
result={'scope':'Exactly3 current authenticated cached frame seeks under global ReadOnlyStepMonitor plus bounded cProfile. Instrumented Python attribution, not normal GUI/seek speed. No physics replay/publicstep/cachepublication or full frame pass.',
 'nativeSha256':canonical['nativeSha256'],'sourceSha256':canonical['originalSha256'],'implementationFingerprint':fingerprint,
 'publicStepAttempts':dict(guard.attempts),'verifiedStateCache':verified.statistics,'observedFrameCache':frames.store.statistics,
 'sourceNativeAndCurrentEngineUnchanged':True,'rows':rows}
out=V/'profile-three-cached-frames-20261020.json'
with out.open('x') as f:json.dump(result,f,indent=2)
print(json.dumps({'output':str(out),'publicStepAttempts':result['publicStepAttempts'],
 'rows':[{'action':r['actionIndex'],'instrumentedWallSeconds':r['instrumentedWallSeconds'],'totalCalls':r['totalCalls'],'topCumulative':r['topCumulative'][:12]} for r in rows]},indent=2))
