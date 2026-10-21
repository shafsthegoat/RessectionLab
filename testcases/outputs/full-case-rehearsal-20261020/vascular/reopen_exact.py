from pathlib import Path
import sys,json
sys.path.insert(0,'src')
from resectionlab.surgical_environment import load_environment
from resectionlab.surgical_engine import SurgicalEngine
from resectionlab.imaging import read_case_artifacts
path=Path(sys.argv[1]);scene=load_environment(path)
r=read_case_artifacts(path)['surgical_toolbox'];snapshot=r['branches'][r['branchId']]
e=SurgicalEngine.restore(scene,snapshot)
assert e.guard_report()['configured']
print(json.dumps({'stateId':e.state_id,'sceneId':scene.identity,'actions':len(e.actions),'strictProtection':True}))
