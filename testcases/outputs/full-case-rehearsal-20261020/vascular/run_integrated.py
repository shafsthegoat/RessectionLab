from pathlib import Path
import sys,json,copy,hashlib
sys.path[:0]=['src','scripts']
from resectionlab.workspace_bundle import load_workspace,save_workspace
from resectionlab.surgical_environment import load_environment
from resectionlab.surgical_engine import SurgicalEngine
from resectionlab.full_case_vascular import rescue_plan
from resectionlab import surgical_vessels_v2 as vessel
from verify_patient_physics_v2 import _step,_conserved_fluid
base=Path('testcases/outputs/full-case-rehearsal-20261020/vascular')
source=Path('testcases/cases/meningioma-sub-06-standard-v2.ressectionlab')
heartbeat=base.parent/'heartbeat/heartbeat.ressectionlab'
original_hash=hashlib.sha256(source.read_bytes()).hexdigest()
case,artifacts,workspace=load_workspace(heartbeat);scene=load_environment(heartbeat)
toolbox=artifacts['surgical_toolbox'];e=SurgicalEngine.restore(scene,toolbox['branches'][toolbox['branchId']])
assert e.guard_report()['configured']
prefix=len(e.actions);before=e.state_id;plan=rescue_plan(e)
assert before==e.state_id
records=[];phases=[]
for phase in plan['phases']:
    initial=vessel.metrics(e)
    for a in phase['actions']:
        r=_step(e,a,records); assert r['status']=='accepted',r.get('reason')
        assert abs(_conserved_fluid(e))<1e-8
        assert abs(vessel.metrics(e)['conservationResidualMm3'])<1e-8
    phases.append({'id':phase['id'],'initial':initial,'final':vessel.metrics(e),'stateId':e.state_id,'fluidPresentMm3':sum(e.state['fluid'].values()),'fluidRemovedMm3':e.state['fluidRemovedMm3'],'elapsedSeconds':e.state['elapsedSeconds']})
assert phases[1]['final']['generatedMm3']>phases[0]['final']['generatedMm3']>0
assert phases[1]['final']['activeWounds']>0
assert phases[2]['final']['activeWounds']==0
assert phases[-1]['final']['generatedMm3']==phases[2]['final']['generatedMm3']
assert all(not f['patent'] for f in e.state['vesselsV2']['faces'].values())
assert e.state['fluidRemovedMm3']>0
record=copy.deepcopy(toolbox);record.update(branches={'vascular_rescue':e.snapshot()},branchId='vascular_rescue',position=len(e.actions),parents={'vascular_rescue':{'branchId':'heartbeat','position':prefix}},timeline={'vascular_rescue':e.history})
artifacts['surgical_toolbox']=record
native=save_workspace(case,base/'patient-integrated-vascular-rescue.ressectionlab',sources=workspace['sources'],imaging=workspace['imagingState'],episode=workspace['episode'],selection=workspace['selection'],originals=workspace['originals'],artifacts=artifacts)
assert original_hash==hashlib.sha256(source.read_bytes()).hexdigest()
report={'status':'passed','sceneId':scene.identity,'sourceSha256':original_hash,'sourceUnchanged':True,'strictProtection':e.guard_report(),'prefixActions':prefix,'totalActions':len(e.actions),'plan':plan,'phases':phases,'actions':records,'stateId':e.state_id,'fluidResidualMm3':_conserved_fluid(e),'native':str((base/'patient-integrated-vascular-rescue.ressectionlab').resolve()),'nativeSha256':native}
(base/'integrated-report.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'status':report['status'],'stateId':report['stateId'],'sceneId':scene.identity,'totalActions':len(e.actions),'native':str((base/'patient-integrated-vascular-rescue.ressectionlab').resolve()),'nativeSha256':native}))
