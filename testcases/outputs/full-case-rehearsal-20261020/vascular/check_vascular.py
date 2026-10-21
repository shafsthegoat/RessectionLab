from pathlib import Path
import sys, json, copy, hashlib, time
import numpy as np
sys.path[:0] = ['src','scripts']
from resectionlab.imaging import load_case, save_case, read_case_artifacts
from resectionlab.surgical_environment import load_environment
from resectionlab.surgical_rl_protection import GuardConfig,load_protection
from resectionlab.surgical_scene import from_case
from resectionlab.surgical_engine import SurgicalEngine
from resectionlab import surgical_vessels_v2 as vessel
from verify_patient_physics_v2 import _step, _conserved_fluid
BASE=Path('testcases/outputs/full-case-rehearsal-20261020/vascular').resolve()
SOURCE=Path('testcases/cases/meningioma-sub-06-standard-v2.ressectionlab').resolve()
sha=lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
original_hash=sha(SOURCE)
case=load_case(SOURCE); scene=load_environment(SOURCE)
protection=load_protection(scene,Path("testcases/environments/meningioma-sub-06/rl-protection.json"),strict=True)
def fresh(): return SurgicalEngine(scene,protection=protection,guard_config=GuardConfig())
recipe=[copy.deepcopy(x['action']) for x in json.loads(Path('testcases/history/surgical-patient-sub06/vessel-verification.json').read_text())['actions']]
report={'source':str(SOURCE),'sourceSha256':original_hash,'sceneId':scene.identity,'scenarios':{},'limitations':['Declared uncalibrated engineering model, not clinical validation.','Candidate historic peripheral vessel access is retested against full exact case; it is not a tumor dissection trajectory.']}
def step(e,a,r):
    result=_step(e,a,r)
    assert result['status']=='accepted', result
    assert abs(vessel.metrics(e)['conservationResidualMm3'])<1e-8
    assert abs(_conserved_fluid(e))<1e-8
    return result
def capture(e,r):
    return {'actions':copy.deepcopy(r),'metrics':vessel.metrics(e),'fluid':{'presentMm3':sum(e.state['fluid'].values()),'addedMm3':e.state['fluidAddedMm3'],'removedMm3':e.state['fluidRemovedMm3'],'escapedMm3':e.state['fluidEscapedMm3'],'reservoirMm3':sum(s.get('reservoirMm3',0) for s in e.state['sources']),'residualMm3':_conserved_fluid(e)},'stateId':e.state_id,'faceRecords':copy.deepcopy(e.state['vesselsV2']['faces']),'sources':copy.deepcopy(e.state['sources']),'removedCells':len(e.state['removed']),'transectedEdges':len(e.state['transectedEdges'])}
def persist(e,name):
    snapshot=e.snapshot()
    (BASE/(name+'-snapshot.json')).write_text(json.dumps(snapshot,indent=2)+'\n')
    artifact=copy.deepcopy(read_case_artifacts(SOURCE)['surgical_toolbox']); artifact.update(mapping=scene.regions,branches={'main':snapshot},branchId='main',position=len(e.actions),parents={'main':None})
    path=save_case(case,BASE/(name+'.ressectionlab'),artifacts={'surgical_toolbox':artifact})
    return str(path)
# Intact spontaneous bleed and force-free actual contact.
e=fresh(); r=[]
step(e,{'slot':'left','operation':'inspect','duration':5},r)
assert vessel.metrics(e)['generatedMm3']==0
report['scenarios']['uninjured_idle']=capture(e,r)
e=fresh(); r=[]
contact=copy.deepcopy(recipe[:3]); contact[1]['parameters'].update(jawOpeningMm=2,forceN=0)
for a in contact: step(e,a,r)
step(e,{'slot':'left','operation':'apply','duration':3,'parameters':{'jawOpeningMm':2,'forceN':0}},r)
assert not e.state['transectedEdges'] and not e.state['vesselsV2']['faces']
assert vessel.metrics(e)['generatedMm3']==0
assert e.state['vesselsV2']['contactDose'], 'Must have actual vessel contact'
report['scenarios']['mere_contact']=capture(e,r)
# Injury, true idle untreated leak, then tool rescue on same live branch.
e=fresh(); r=[]
for a in recipe[:3]: step(e,a,r)
step(e,{'slot':'left','operation':'deactivate','duration':.2},r)
injured=capture(e,r)
step(e,{'slot':'left','operation':'inspect','duration':2},r)
short=capture(e,r)
step(e,{'slot':'left','operation':'inspect','duration':5},r)
long=capture(e,r)
assert injured['metrics']['generatedMm3'] < short['metrics']['generatedMm3'] < long['metrics']['generatedMm3']
report['scenarios']['injury']=injured; report['scenarios']['untreated_short']=short; report['scenarios']['untreated_long']=long
report['untreatedNative']=persist(e,'patient-untreated-vascular')
print('FLUID',e.state['fluid'],file=sys.stderr,flush=True)
for a in recipe[3:8]: step(e,a,r)
sealed=capture(e,r)
step(e,{'slot':'left','operation':'deactivate','duration':.2},r)
step(e,{'slot':'left','operation':'inspect','duration':5},r)
after=capture(e,r)
assert sealed['metrics']['activeWounds']==0
assert after['metrics']['generatedMm3']==sealed['metrics']['generatedMm3']
assert all(not f['patent'] for f in e.state['vesselsV2']['faces'].values())
report['scenarios']['sealed']=sealed; report['scenarios']['sealed_after_idle']=after
# Same legal access path supports fluid tools, with arbitrary in-material field tip allowed by public API.
step(e,{'slot':'left','operation':'withdraw','duration':.2},r)
for tool in ('irrigation','frazier_suction'):
    step(e,{'slot':'left','operation':'exchange','duration':.2,'instrument':tool},r)
    place=copy.deepcopy(recipe[1]); place['parameters']={'radiusMm':3,'depthMm':.1,'flowMm3PerSecond':2,'forceN':0}
    step(e,place,r)
    approach=copy.deepcopy(recipe[2]); approach['pose']['position'][0] += 2.; approach['parameters']={'radiusMm':3,'depthMm':.1,'flowMm3PerSecond':2,'forceN':0}
    step(e,approach,r)
    step(e,{'slot':'left','operation':'apply','duration':2},r)
    report['scenarios'][tool]=capture(e,r)
    step(e,{'slot':'left','operation':'deactivate','duration':.2},r)
    step(e,{'slot':'left','operation':'withdraw','duration':.2},r)
assert e.state['fluidRemovedMm3']>0
report['rescueNative']=persist(e,'patient-vascular-rescue')
report['final']=capture(e,r)
assert sha(SOURCE)==original_hash
report.update(sourceUnchanged=True,status='passed',publicActionOnly=True,initialRemovedCells=0)
(BASE/'vascular-report.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:report[k] for k in ('status','sourceUnchanged','sceneId','rescueNative')}))
