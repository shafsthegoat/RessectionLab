"""Scripted generated skill fixtures; no policy execution or learning."""
import copy
import json

import numpy as np
import pytest

pytest.importorskip('gymnasium')

from resectionlab.core import array_digest
from resectionlab.surgical_engine import SurgicalEngine
from resectionlab.surgical_cusa_v3 import default_profile as cusa_default_profile
from resectionlab.surgical_physics_v2 import default_profile
from resectionlab.surgical_rl_env import SurgicalRLEnv
from resectionlab.surgical_scene import SurgicalScene
from resectionlab.surgical_skills import SkillExecutor, SkillTask, propose_skill


def pose(x=9.,y=9.,z=-5.):
    return {'position':[x,y,z],'quaternion':[0.,0.,0.,1.]}


def fixture(*,rescue=False,causal_cusa=False):
    labels=np.zeros((24,24,20),np.uint16);labels[8:11,8:11,8:11]=1
    properties={'tissuePhysicsV2':default_profile()}
    if causal_cusa:properties['cusaV3']=cusa_default_profile()
    if rescue:
        properties['bleedingSources']=[{'id':'declared-preinjured-fixture',
            'cell':int(np.ravel_multi_index((9,9,9),labels.shape)),
            'rateMm3PerSecond':.1,'coagulationExposure':.02,'patent':False}]
    scene=SurgicalScene(labels,np.eye(4),[{'id':1,'name':'Synthetic target',
        'material':'tumor','target':True}],{'kind':'generated skill fixture'},properties)
    task=SkillTask('resection','preinjured_rescue' if rescue else 'prevention')
    env=SurgicalRLEnv(scene,config=task.config(require_critical_coverage=False))
    env.reset(seed=17)
    return env,task


def execute(executor,name,**values):
    receipt=executor.execute(name,**values)
    assert receipt['status'] in ('completed','episode_finished'),receipt
    return receipt


def test_task_horizons_and_preinjured_rescue_are_explicit():
    assert SkillTask('access').config().max_steps == 128
    assert SkillTask('resection').config().max_steps == 4096
    assert SkillTask('field_management').config().max_seconds == 300.
    e,task=fixture(rescue=True)
    with pytest.raises(ValueError,match='preinjured_rescue'):SkillExecutor(e,SkillTask('resection'))
    assert SkillExecutor(e,task).task.mode == 'preinjured_rescue'
    with pytest.raises(ValueError):SkillTask('unbounded')


def test_proposals_use_only_observation_and_explicit_endpoint_with_no_feasibility_oracle():
    e,_=fixture();obs=e._observe();before=e.engine.state_id
    original=copy.deepcopy(obs)
    plan=propose_skill(obs,'approach',instrument='cusa',pose=pose())
    assert [a['operation'] for a in plan['actions']] == ['exchange','move']
    assert plan['nominal'] and plan['feasibility']=='unknown_until_public_execution'
    assert plan['actions'][-1]['pose'] == pose()
    assert all(a['operation']!='place' for a in plan['actions'])
    assert e.engine.state_id==before and not e.attempts
    for key in original:np.testing.assert_array_equal(original[key],obs[key])
    plan['actions'][-1]['pose']['position'][0]=100.
    assert pose()['position'][0]==9.
    obs['local'][:]=0
    assert propose_skill(obs,'drill')['support']=='unknown_local_support'
    with pytest.raises(TypeError):propose_skill(obs,'drill',engine=e.engine)
    with pytest.raises(ValueError,match='published'):
        propose_skill({**obs,'hidden_feasibility_mask':np.ones(5)},'drill')


def test_skill_cancellation_is_forwarded_to_current_public_primitive():
    e,task=fixture();executor=SkillExecutor(e,task);before=e.engine.state_id
    calls=0
    def cancel():
        nonlocal calls
        calls+=1
        if calls>=12:raise TimeoutError('test cancellation')
    with pytest.raises(TimeoutError,match='test cancellation'):
        executor.execute('approach',pose=pose(z=7.),check=cancel)
    assert e.engine.state_id==before and not e.attempts


def test_partial_checked_approach_stops_skill_without_inventing_route():
    e,task=fixture();executor=SkillExecutor(e,task)
    receipt=executor.execute('approach',pose=pose(z=9.))
    assert receipt['status'] in ('partial','rejected')
    assert not e.engine.state['removed']
    assert receipt['plan']['actions'][0]['operation']=='move'


def test_stop_with_residual_target_cleans_controls_without_claiming_success():
    e,task=fixture();executor=SkillExecutor(e,task)
    execute(executor,'approach',instrument='cusa',pose=pose())
    receipt=execute(executor,'stop')
    assert receipt['stopped'] and not receipt['success']
    assert receipt['cleanup']['pending']==[]
    with pytest.raises(RuntimeError,match='stopped'):executor.execute('inspect')


def test_parameterized_drill_uses_public_strokes_and_conserves_generated_debris():
    labels=np.zeros((24,24,24),np.uint16)
    labels[8:11,8:11,8]=2;labels[9,9,12]=1
    scene=SurgicalScene(labels,np.eye(4),[
        {'id':1,'name':'Synthetic target','material':'tumor','target':True},
        {'id':2,'name':'Synthetic access bone','material':'bone'}],
        {'kind':'generated drilling skill coupon'},{'tissuePhysicsV2':default_profile()})
    task=SkillTask('access');e=SurgicalRLEnv(scene,config=task.config(require_critical_coverage=False))
    e.reset(seed=17);executor=SkillExecutor(e,task)
    execute(executor,'approach',pose=pose(),parameters={'radiusMm':1.5,'forceN':20.,'rpm':80000.})
    for _ in range(3):
        result=executor.execute('drill',pose=pose(z=10.),duration=5.)
        assert result['status'] in ('completed','partial')
        if len(e.engine.state['removed'])==9:break
    state=e.engine.state
    assert len(state['removed'])==9 and e.measurements['tumor_removed_mm3']==0
    assert state['drillingV2']['removedVolumeMm3']==pytest.approx(
        state['debrisMm3']+state.get('debrisRemovedMm3',0.)+state.get('debrisEscapedMm3',0.))
    assert state['debrisMm3']==9.


def test_scripted_rescue_resection_field_cleanup_stop_and_native_snapshot_agreement(tmp_path):
    e,task=fixture(rescue=True);executor=SkillExecutor(e,task)
    source=array_digest(e.scene.labels)
    receipts=[]
    def skill(name,**values):
        result=execute(executor,name,**values);receipts.append(result);return result
    skill('inspect')
    skill('approach',instrument='bipolar_forceps',pose=pose(),parameters={'radiusMm':1.})
    skill('coagulate',pose=pose(z=8.5),parameters={'power':2.},duration=.2)
    assert e._observe()['cleanup'][2]==0
    skill('withdraw')
    skill('approach',instrument='cusa',pose=pose(x=8.),parameters={'radiusMm':1.5})
    skill('resect',pose=pose(x=8.,z=11.),duration=1.)
    assert 0<e.measurements['tumor_removed_mm3']<27.
    skill('withdraw')
    skill('approach',pose=pose(x=10.),parameters={'radiusMm':1.5})
    skill('resect',pose=pose(x=10.,z=11.),duration=1.)
    assert e._target_reached() and not e._done
    skill('withdraw')
    skill('approach',instrument='frazier_suction',pose=pose(),parameters={'radiusMm':1.5})
    skill('aspirate',pose=pose(z=9.),parameters={'flowMm3PerSecond':100.},duration=.2)
    final=skill('stop')
    assert final['stopped'] and final['success'] and final['terminated'] and not final['truncated']
    assert final['cleanup']['pending']==[]
    assert e.measurements['tumor_removed_mm3']==27.
    assert e.measurements['healthy_removed_mm3']==0.
    state=e.engine.state
    assert state['fluidAddedMm3']>0 and state['fluidRemovedMm3']>0
    assert state['fluidAddedMm3']==pytest.approx(sum(state['fluid'].values())
        +sum(s.get('reservoirMm3',0.) for s in state['sources'])
        +state['fluidRemovedMm3']+state['fluidEscapedMm3'])
    assert array_digest(e.scene.labels)==source
    native_record=json.loads(json.dumps(e.engine.snapshot(),allow_nan=False))
    native=SurgicalEngine.restore(e.scene,native_record)
    assert native.state_id==e.engine.state_id and native.state==state
    replay,_=fixture(rescue=True)
    snapshot=json.loads(json.dumps(e.snapshot(),allow_nan=False));replay.restore(snapshot)
    assert replay.engine.state_id==native.state_id and replay.measurements==e.measurements
    (tmp_path/'scripted-skills-native-snapshot.json').write_text(json.dumps(native_record,indent=2)+'\n')
    (tmp_path/'scripted-skills-rl-snapshot.json').write_text(json.dumps(snapshot,indent=2)+'\n')
    (tmp_path/'scripted-skills-receipt.json').write_text(json.dumps({
        'generated_fixture':True,'learning_steps':0,'policy_executions':0,
        'task':final['task'],'receipts':receipts,'nativeStateId':native.state_id,
        'rlStateId':replay.engine.state_id,'metrics':e.measurements},indent=2)+'\n')


def test_skill_nominal_tip_checks_match_fixed_catalog_execution_contract():
    e,task=fixture();executor=SkillExecutor(e,task)
    with pytest.raises(ValueError,match='catalog'):
        executor.execute('approach',instrument='cusa',pose=pose(),parameters={'radiusMm':3.})
    assert not e.attempts
    execute(executor,'approach',instrument='cusa',pose=pose(),parameters={'radiusMm':.5})
    before=e.engine.state_id
    with pytest.raises(ValueError,match='physical radius'):
        executor.execute('resect',parameters={'radiusMm':1.})
    assert e.engine.state_id==before


def test_causal_cusa_work_partial_remains_honest_and_fixed_next_request_can_continue():
    e,task=fixture(causal_cusa=True);executor=SkillExecutor(e,task)
    execute(executor,'approach',instrument='cusa',pose=pose(z=5.9),parameters={'radiusMm':1.5})
    first=executor.execute('resect',pose=pose(z=11.),duration=4.9)
    assert first['status']=='partial'
    assert first['transitions'][-1]['reasonCode']=='TISSUE_ASPIRATION_IN_PROGRESS'
    assert 0<e.measurements['tumor_removed_mm3']<=2.*4.9
    assert e.engine.state['cusaV3']['progressExact']
    assert not e._done and e.invalid_actions==0
    removed=e.measurements['tumor_removed_mm3']
    second=executor.execute('resect',pose=pose(z=11.),duration=5.)
    assert second['status']=='partial'
    assert second['transitions'][-1]['reasonCode']=='TISSUE_ASPIRATION_IN_PROGRESS'
    assert e.measurements['tumor_removed_mm3']>removed and e.invalid_actions==0
