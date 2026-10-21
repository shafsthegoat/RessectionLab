from pathlib import Path
import sys,cProfile,pstats,time,json
sys.path.insert(0,'src')
from resectionlab.surgical_engine import SurgicalEngine
from resectionlab.surgical_environment import load_environment
from resectionlab.surgical_rl_protection import restore_protection_contract
base=Path('testcases/outputs/full-case-rehearsal-20261020/vascular/performance')
snapshot=json.load(open('testcases/outputs/full-case-rehearsal-20261020/heartbeat/broad-probe-snapshot.json'))
scene=load_environment('testcases/cases/meningioma-sub-06-standard-v2.ressectionlab')
protection,config=restore_protection_contract(scene,snapshot['protectionContract']);e=SurgicalEngine(scene,protection=protection,guard_config=config)
e.step(snapshot['actions'][0])
p=cProfile.Profile();start=time.perf_counter();p.enable();r=e.step(snapshot['actions'][1]);p.disable();elapsed=time.perf_counter()-start
p.dump_stats(str(base/'baseline.prof'))
with (base/'baseline-profile.txt').open('w') as f:pstats.Stats(p,stream=f).strip_dirs().sort_stats('cumtime').print_stats(60)
(base/'baseline-result.json').write_text(json.dumps({'wallSeconds':elapsed,'action':snapshot['actions'][1],'result':r,'stateId':e.state_id}))
(base/'baseline-snapshot.json').write_text(json.dumps(e.snapshot()))
print(elapsed,r['status'],r.get('reason'))
