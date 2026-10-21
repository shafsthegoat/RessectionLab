from pathlib import Path
import sys,json,copy,hashlib
sys.path[:0]=['src','scripts']
from resectionlab.imaging import load_case
from resectionlab.surgical_environment import load_environment
from resectionlab.surgical_rl_protection import GuardConfig,load_protection
from resectionlab.surgical_engine import SurgicalEngine
from resectionlab import surgical_vessels_v2 as vessel
from verify_patient_physics_v2 import _step,_conserved_fluid
base=Path('testcases/outputs/full-case-rehearsal-20261020/vascular')
source=Path('testcases/cases/meningioma-sub-06-standard-v2.ressectionlab')
before=hashlib.sha256(source.read_bytes()).hexdigest()
scene=load_environment(source)
e=SurgicalEngine(scene,protection=load_protection(scene,Path("testcases/environments/meningioma-sub-06/rl-protection.json"),strict=True),guard_config=GuardConfig()); records=[]
recipe=[copy.deepcopy(x['action']) for x in json.loads(Path('testcases/history/surgical-patient-sub06/vessel-verification.json').read_text())['actions'][:3]]
recipe[1]['parameters'].update(jawOpeningMm=2,forceN=0)
for a in recipe: assert _step(e,a,records)['status']=='accepted'
exposures=[]
for force,duration in ((1,.25),(20,.25),(20,1.5),(20,1),(20,1)):
    assert _step(e,{'slot':'left','operation':'apply','duration':duration,'parameters':{'forceN':force,'jawOpeningMm':2}},records)['status']=='accepted'
    severity=max((f['damage'] for f in e.state['vesselsV2']['faces'].values()),default=0)
    exposures.append({'forceN':force,'duration':duration,'maxInjury':severity,'metrics':vessel.metrics(e)})
    assert abs(_conserved_fluid(e))<1e-8
assert exposures[0]['maxInjury']==0
assert any(0<x['maxInjury']<1 for x in exposures)
assert exposures[-1]['maxInjury']==1
assert not e.state['transectedEdges']
assert before==hashlib.sha256(source.read_bytes()).hexdigest()
report={'status':'passed','sceneId':e.scene.identity,'exposures':exposures,'actions':records,'sourceUnchanged':True,'transectedEdges':len(e.state['transectedEdges']),'fluidResidualMm3':_conserved_fluid(e)}
(base/'mechanical-contact.json').write_text(json.dumps(report,indent=2)+'\n')
print([(x['forceN'],x['duration'],x['maxInjury']) for x in exposures])
