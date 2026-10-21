"""Tiny constructed V2 fixture only: no patient/native loading or restore."""
from pathlib import Path
import sys,json,math,copy,hashlib
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[4];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from resectionlab.surgical_scene import SurgicalScene
from resectionlab.surgical_engine import SurgicalEngine
from resectionlab.surgical_physics_v2 import default_profile
from resectionlab.full_case_path_variants import cusa_path_variants
from resectionlab.core import array_digest
labels=np.zeros((32,24,24),np.uint16); labels[17:20,7:10,8]=1;labels[8,8,8]=2
scene=SurgicalScene(labels,np.eye(4),[{'id':1,'name':'bone','material':'bone'},
 {'id':2,'name':'target','material':'tumor','target':True}],
 {'kind':'independent-cusa-variants-V2-shared-clock-fixture'}, {'tissuePhysicsV2':default_profile()})
e=SurgicalEngine(scene); labelhash=array_digest(scene.labels)
pose=lambda x,z:{'position':[float(x),8.,float(z)],'quaternion':[0.,0.,0.,1.]}
def step(op,slot='left',**values):
 a={'slot':slot,'operation':op,'duration':.01,**values};r=e.step(a)
 assert r['status'] in ('accepted','partial'),r
 return r
step('place',pose=pose(18,4),parameters={'radiusMm':1.5,'rpm':80000.,'forceN':20.})
for _ in range(3):
 r=step('apply',pose=pose(18,10),duration=3.)
 if r['status']=='accepted':break
assert len(e.state['removed'])==9 and math.isclose(e.state['debrisMm3'],9.)
step('withdraw');step('exchange',instrument='cusa');step('place',pose=pose(8,4),parameters={'radiusMm':1.5,'forceN':0.})
step('exchange',slot='right',instrument='frazier_suction')
step('place',slot='right',pose=pose(18,4),parameters={'radiusMm':.5,'forceN':0.,'flowMm3PerSecond':1.})
step('move',slot='right',pose=pose(18,8),duration=.2)
step('activate',slot='right',parameters={'radiusMm':2.})
assert e.state['slots']['right']['active'] and e.state['slots']['right']['inserted']
assert e.state['debrisMm3']>8.
assert max(e.state['drillingV2']['temperatureC'].values())>37.
fine={'slot':'left','operation':'apply','duration':2.,'parameters':{'radiusMm':1.5},
 'path':[pose(8,z) for z in np.linspace(4.,10.,151)]}
coarse,fallback=cusa_path_variants(fine);assert fallback is fine and len(coarse['path'])==2
before=e.snapshot();d0=copy.deepcopy(e.state['drillingV2']);rows=[]
for label,action in [('coarse',coarse),('fine',fine)]:
 trial=e.clone();preview=trial.preview(action,include_view=False);assert trial.snapshot()==before
 outcome=trial.step(action)
 assert outcome['status']=='accepted' and preview['stateId']==outcome['stateId']
 assert preview['deltas']==outcome['deltas'] and trial.metrics()['targetRemovedMm3']==1.
 elapsed=trial.state['elapsedSeconds']-before['checkpoint']['state']['elapsedSeconds']
 assert math.isclose(elapsed,2.,rel_tol=0,abs_tol=2e-12)
 vessel_elapsed=trial.state['vesselsV2']['elapsedSeconds']-before['checkpoint']['state']['vesselsV2']['elapsedSeconds']
 assert math.isclose(vessel_elapsed,2.,rel_tol=0,abs_tol=2e-12)
 aspirated=trial.state['debrisRemovedMm3']-before['checkpoint']['state']['debrisRemovedMm3']
 assert math.isclose(aspirated,2.,rel_tol=0,abs_tol=2e-12)
 debris_total=trial.state['debrisMm3']+trial.state['debrisRemovedMm3']+trial.state.get('debrisEscapedMm3',0.)
 assert math.isclose(debris_total,9.,rel_tol=0,abs_tol=2e-12)
 d=trial.state['drillingV2'];p=scene.properties['tissuePhysicsV2']['bone']
 capacity=p['heatCapacityJPerMm3K']*scene.voxel_mm3
 expected_cooling=sum(capacity*(d0['temperatureC'][k]-d['temperatureC'][k]) for k in d0['temperatureC'])
 actual_cooling=d['coolingLossJ']-d0['coolingLossJ']
 assert actual_cooling>0 and math.isclose(actual_cooling,expected_cooling,rel_tol=1e-11,abs_tol=1e-12)
 for k,t in d0['temperatureC'].items():
  expected=p['ambientTemperatureC']+(t-p['ambientTemperatureC'])*math.exp(-2./p['coolingTimeSeconds'])
  assert math.isclose(d['temperatureC'][k],expected,rel_tol=1e-13,abs_tol=1e-11)
 assert math.isclose(d['inputEnergyJ'],d['cuttingWorkJ']+d['unusedEnergyJ'],rel_tol=1e-12,abs_tol=1e-12)
 assert array_digest(scene.labels)==labelhash
 rows.append({'variant':label,'pathPoses':len(action['path']),'status':outcome['status'],
  'ownPreviewMatchesCommit':True,'simulationSeconds':elapsed,'vesselClockSeconds':vessel_elapsed,
  'actualOtherHandDebrisAspiratedMm3':aspirated,'generatedDebrisConservedMm3':debris_total,
  'passiveCoolingLossJ':actual_cooling,'thermalCoolingEnergyAndAnalyticTemperatureCheck':True,
  'removedTargetCells':trial.metrics()['targetRemovedMm3'],'nonTargetRemovedThisAction':len(outcome['deltas']['removedCells'])-1,
  'stateId':trial.state_id})
assert e.snapshot()==before and array_digest(scene.labels)==labelhash
out=Path(__file__).with_name('cusa-variants-independent-v2-clock-probe.json')
out.write_text(json.dumps({'utc':datetime.now(timezone.utc).isoformat(),'result':'pass','fixture':scene.source,
 'sourceLabelsSha256':labelhash,'baselineUnchanged':True,'actualBoneGeneratedDebrisMm3':9.,
 'initialTemperatureCells':len(d0['temperatureC']),'rows':rows,
 'limits':'Tiny explicit engineering fixture, public actions and clone only; no patient files/scene/engine/replay/live/GUI. Coarse and fine have independently checked outcomes; no bitwise equivalence or action/episode speed claim.'},indent=2)+'\n')
print(json.dumps({'receipt':str(out.resolve()),'rows':rows}))
