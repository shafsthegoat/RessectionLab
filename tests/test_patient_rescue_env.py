"""Real generated task transitions; patient task admission is checked separately."""
import copy

import numpy as np
import pytest

from resectionlab.surgical_scene import SurgicalScene
from resectionlab.surgical_physics_v2 import default_profile
from resectionlab.surgical_rl_env import RLConfig
from resectionlab.surgical_rl_protection import ProtectionSet, ProtectionAnnotation
from scripts.patient_rescue_env import (PreparedRescueRLEnv, RescueOptionEnv, OPTIONS,
    TaskProtectedEngine, option_requests, policy_features)


def episode():
    labels=np.zeros((24,24,24),np.uint16)
    labels[3:6,3:6,8:11]=1;labels[11,10:13,10]=2
    scene=SurgicalScene(labels,np.eye(4),[
        dict(id=1,name='Generated target',material='tumor',target=True),
        dict(id=2,name='Generated vessel',material='vessel',target=False)],
        dict(kind='generated rescue adapter regression'),{'tissuePhysicsV2':default_profile()})
    mask=np.zeros(scene.shape,bool);mask[0,0,0]=True
    protection=ProtectionSet(scene,[ProtectionAnnotation.from_mask(scene,'brainstem',mask,provenance='generated')])
    engine=TaskProtectedEngine(scene,protection,RLConfig())
    pose=lambda z:dict(position=[11.,11.,z],quaternion=[0.,0.,0.,1.])
    actions=[dict(slot='left',operation='exchange',instrument='micro_scissors',duration=.2),
        dict(slot='left',operation='place',pose=pose(-4.),parameters=dict(radiusMm=1.,forceN=0.,jawOpeningMm=0.),duration=.2),
        dict(slot='left',operation='apply',pose=pose(10.),duration=1.),
        dict(slot='left',operation='withdraw',duration=.2),
        dict(slot='left',operation='exchange',instrument='bipolar_forceps',duration=.2)]
    for action in actions:assert engine.step(action)['status']=='accepted'
    assert engine.metrics()['activeVesselWounds']>0
    points=dict(outside=pose(-4.),contact=pose(10.),coverage_lower=pose(10.),
        coverage_forward=pose(10.),suction_outside=pose(-4.),suction_pool=pose(8.5))
    return RescueOptionEnv(PreparedRescueRLEnv(engine),points)


def test_task_resets_from_real_prepared_state_and_keeps_historical_burden():
    env=episode();obs,info=env.reset(seed=17);before=env.native_snapshot()
    assert env.observation_space.contains(obs)
    assert info['historicalPrefixRawRlAdmitted'] is False
    assert info['entryMetrics']['vessel_damage_mm2']>0
    obs,reward,terminated,truncated,info=env.step(OPTIONS.index('wait'))
    assert not terminated and not truncated and not info['success']
    assert info['modeledField'][3]>0 and len(env.raw.attempts)==1
    assert env.raw.engine.state['elapsedSeconds']>before['checkpoint']['state']['elapsedSeconds']
    reset,_=env.reset(seed=17)
    np.testing.assert_array_equal(reset,env.reset(seed=17)[0])
    assert env.native_snapshot()==before and env.raw.attempts==[]


def test_success_requires_real_coagulation_suction_withdrawal_and_delayed_observation():
    env=episode();env.reset(seed=17)
    for option in ('coagulate_center','aspirate_pool','finish_verify'):
        obs,reward,terminated,truncated,info=env.step(OPTIONS.index(option))
        assert env.observation_space.contains(obs)
        assert not truncated
    assert terminated and info['success'],info
    assert info['verifiedCleanSeconds']>=5.-1e-9
    assert info['cleanup']['pending']==[]
    assert info['metrics']['vessel_damage_mm2']==info['entryMetrics']['vessel_damage_mm2']
    assert len(env.raw.engine.actions)>len(env.raw._prepared.actions)
    assert any(a['operation']=='exchange' for a in env.raw.engine.actions[len(env.raw._prepared.actions):])
    with pytest.raises(RuntimeError,match='Episode finished'):env.step(0)


def test_agent_functions_cannot_receive_hidden_state_or_mutate_published_observation():
    env=episode();obs=env.raw._observe();before=copy.deepcopy(obs)
    requests=option_requests(obs,'coagulate_center',env.waypoints)
    assert all(r['skill'] not in ('place','preview') for r in requests)
    assert [r['skill'] for r in requests[-3:]]==['coagulate','withdraw','withdraw']
    features=policy_features(obs,env.raw.field(),np.zeros(6),0.,env.waypoints['contact']['position'])
    assert features.shape==(60,)
    for key in obs:np.testing.assert_array_equal(obs[key],before[key])
    with pytest.raises(ValueError):option_requests({**obs,'world':env.raw.engine.state},'wait',env.waypoints)
    with pytest.raises(ValueError):policy_features({**obs,'future_reward':1},env.raw.field(),np.zeros(6),0.,[0,0,0])


def test_nominal_refusal_is_an_rl_attempt_without_a_public_engine_call():
    env=episode();env.reset(seed=17)
    calls=env.raw.engine.public_calls
    env.waypoints['contact']['quaternion']=[0.,0.,0.,0.]
    _,_,terminated,truncated,info=env.step(OPTIONS.index('coagulate_center'))
    assert not terminated and truncated and info['termination_reason']=='option_refused'
    assert len(env.raw.attempts)==2
    assert env.raw.engine.public_calls-calls==1
    assert env.raw.attempts[-1]['decodeError'].startswith('NOMINAL_SKILL_REFUSAL')


def test_interrupted_option_keeps_committed_actions_and_input_provenance():
    env=episode();env.reset(seed=17);start=env.native_snapshot()
    def check():
        if env.raw.attempts:raise TimeoutError('bounded cancellation')
    env.check=check
    with pytest.raises(TimeoutError):env.step(OPTIONS.index('coagulate_center'))
    assert env.raw.attempts and len(env.native_snapshot()['actions'])>len(start['actions'])
    assert env.inflightOption['option']=='coagulate_center'
    assert env.inflightOption['currentSkill']['skill']=='approach'
    assert env.inflightOption['policyObservationSha256']
    env.reset(seed=17)
    assert env.inflightOption is None and env.native_snapshot()==start
