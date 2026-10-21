from pathlib import Path
import sys,json,time,subprocess,hashlib
sys.path.insert(0,str(Path.cwd()/'src'))
from resectionlab.surgical_environment import load_environment
from resectionlab.imaging import read_case_artifacts
from resectionlab.surgical_engine import SurgicalEngine
from resectionlab.surgical_verified_cache import VerifiedSurgicalCache,implementation_fingerprint
base=Path('testcases/outputs/full-case-rehearsal-20261020/vascular')
path=base/'patient-vascular-rescue.ressectionlab'
scene=load_environment(path);toolbox=read_case_artifacts(path)['surgical_toolbox'];record=toolbox['branches'][toolbox['branchId']]
cache=VerifiedSurgicalCache()
start=time.perf_counter()
e=SurgicalEngine.restore(scene,record,use_cache=True,force_verify='--read' not in sys.argv,verified_cache=cache)
elapsed=time.perf_counter()-start
report={'mode':'fresh_cached_read' if '--read' in sys.argv else 'independent_full_replay_and_seal','wallSeconds':elapsed,'source':str(path.resolve()),'sourceSha256':hashlib.sha256(path.read_bytes()).hexdigest(),'sceneId':scene.identity,'stateId':e.state_id,'actions':len(e.actions),'snapshotExactEquality':e.snapshot()==record,'cache':cache.statistics,'implementation':implementation_fingerprint()}
assert report['snapshotExactEquality']
if '--read' in sys.argv:
 assert cache.hits==1 and cache.publications==0,report
 print(json.dumps(report));sys.exit(0)
assert cache.publications==1,{'report':report,'skip':cache.last_skip}
reopened=subprocess.run([sys.executable,__file__,'--read'],capture_output=True,text=True,check=True)
report['freshProcess']=json.loads(reopened.stdout)
assert report['freshProcess']['stateId']==e.state_id
(base/'verified-cache-patient-proof.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
