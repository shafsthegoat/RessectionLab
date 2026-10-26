"""Generic observed tool-frame and precommit safety contracts."""
import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation
import importlib.util
from pathlib import Path
from types import SimpleNamespace

from resectionlab.patient_cusa_learning import (
    ACTION_IDS, Actor, Budget, Critic, _cleanup_boundary, _commit,
    _retry_durations, observed_options, parameter_sha256, ppo_update,
)
from resectionlab.surgical_rl_env import OBS_METRICS


def _observation():
    q = Rotation.from_rotvec([0, np.pi / 2, 0]).as_quat()
    slots = np.zeros((2, 12), np.float32)
    slots[0, :3] = [10, 20, 30]
    slots[0, 3:7] = q
    tools = np.ones((2, 11), np.float64)
    metrics = np.zeros(len(OBS_METRICS), np.float64)
    return {'local': np.zeros((2, 8, 9, 9, 9), np.float32),
            'slots': slots, 'tool_parameters': tools,
            'metrics': metrics, 'goal': np.zeros(4),
            'coverage': np.zeros(3), 'local_spacing_mm': np.array([1.]),
            'cleanup': np.zeros(3), 'budgets': np.ones(3)}


def test_observed_cusa_options_follow_current_oblique_tool_frame_and_task_progress():
    observation = _observation()
    features, actions = observed_options(observation)
    assert ACTION_IDS[1] == 'inward-1' and features.shape == (9, 22)
    assert actions[1]['pose']['position'] == pytest.approx([11, 20, 30], abs=1e-6)
    assert actions[3]['pose']['position'] == pytest.approx([12, 20, 30], abs=1e-6)
    assert actions[8] is None
    observation['metrics'][OBS_METRICS.index('tumor_removed_mm3')] = 30.
    progressed, _ = observed_options(observation)
    assert np.all(features[:, 20] == 1) and np.all(progressed[:, 20] == 0)


def test_cleanup_checkpoint_comes_from_recorded_phase_and_requires_checked_tail():
    actions = [{'slot': 'left', 'operation': 'apply'}] * 7 + [
        {'slot': 'left', 'operation': 'deactivate'},
        {'slot': 'left', 'operation': 'withdraw'}]
    plan = {'directorPhases': [{'id': 'corridor-withdrawal', 'startPosition': 7}]}
    assert _cleanup_boundary(plan, actions) == 7
    with pytest.raises(ValueError, match='pair'):
        _cleanup_boundary(plan, actions[:-1] + [{'slot':'left','operation':'inspect'}])


def test_non_target_cap_refuses_preview_without_a_committed_engine_step():
    class Engine:
        state_id = 'same-source-state'
        def preview(self, action, **kwargs):
            assert action['operation'] == 'apply'
            return {'status': 'accepted', 'stateId':self.state_id,
                    'metrics': {'targetRemovedMm3':0.,'nonTargetRemovedMm3': 121.}}
        def step(self, action, **kwargs):
            raise AssertionError('cap crossing must never commit')
    budget = Budget(max_wall_seconds=10, max_calls=4)
    result = _commit(Engine(), None, {'operation':'apply'}, 100., budget)
    assert result[-1] == 'incremental_non_target_cap'
    assert budget.calls == 1


def test_checked_collision_recovery_preserves_endpoint_and_every_refusal(monkeypatch):
    import resectionlab.patient_cusa_learning as learning
    pose={'position':[1.,2.,3.],'quaternion':[0.,0.,0.,1.]}
    class Engine:
        state_id='before'
        def __init__(self): self.seen=[]
        def preview(self,action,**kwargs):
            self.seen.append(('preview',action.copy()))
            accepted=action['duration']>=1.
            return {'status':'accepted' if accepted else 'rejected',
                'reason':None if accepted else 'SHAFT_COLLISION',
                'stateId':'after' if accepted else 'before',
                'metrics':{'targetRemovedMm3':10. if accepted else 0.,
                           'nonTargetRemovedMm3':2. if accepted else 0.}}
        def step(self,action,**kwargs):
            self.seen.append(('step',action.copy()))
            self.state_id='after'
            return {'status':'accepted','reason':None,'stateId':'after',
                    'metrics':self.metrics()}
        def metrics(self):
            return {'targetRemovedMm3':10.,'nonTargetRemovedMm3':2.}
    class Tracker:
        def measure(self,*args): return {}
    monkeypatch.setattr(learning,'_protection_measurements',lambda engine:{})
    monkeypatch.setattr(learning,'compute_reward',lambda *args:(0.,{}))
    monkeypatch.setattr(learning,'_observer',lambda *args:({},{}))
    engine=Engine()
    action={'slot':'left','operation':'apply','pose':pose,'duration':.2}
    observation,reward,info,refusal=_commit(engine,Tracker(),action,0.,Budget(max_calls=5))
    assert refusal is None and info['stateId']=='after' and reward==0.
    assert observation=={}
    assert [a['durationSeconds'] for a in info['attempts']]==[.2,.6,1.]
    assert [a['reason'] for a in info['attempts']]==['SHAFT_COLLISION','SHAFT_COLLISION',None]
    assert all(a['previewElapsedSeconds']>=0 for a in info['attempts'])
    assert [kind for kind,_ in engine.seen]==['preview','preview','preview','step']
    assert all(seen['pose']==pose for _,seen in engine.seen)
    assert action['duration']==.2 and info['action']['duration']==1.


def test_withdrawal_recovery_uses_same_line_and_bounded_slow_duration():
    class Engine:
        state={'slots':{'left':{'position':[0.,0.,0.]}}}
    action={'slot':'left','operation':'withdraw',
            'pose':{'position':[60.,0.,0.],'quaternion':[0.,0.,0.,1.]},
            'duration':2.}
    assert _retry_durations(Engine(),action)==(30.,)
    action['pose']['position']=[20.,0.,0.]
    assert _retry_durations(Engine(),action)==(10.,30.)


def test_clipped_policy_update_changes_actor_weights_independently_of_critic():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(17)
        actor, critic = Actor(), Critic()
    before = parameter_sha256(actor)
    features = np.zeros((9, 22), np.float32)
    features[:, 19] = np.arange(9)
    episodes = [{'training': [(features.copy(), 3, .02),
                              (features.copy(), 8, 0.)],
                 'receipt': {'status': 'success'}}]
    update = ppo_update(actor, critic, episodes, seed=17)
    assert update['actorBeforeSha256'] == before
    assert update['actorAfterSha256'] == parameter_sha256(actor)
    assert update['actorChanged'] and update['sampledDecisions'] == 2


def test_frozen_ppo_loader_requires_actual_actor_update_receipts(tmp_path):
    source=Path(__file__).resolve().parents[1]/'scripts/run_patient_cusa_learning.py'
    spec=importlib.util.spec_from_file_location('patient_cusa_runner',source)
    runner=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    actor=Actor()
    identity=parameter_sha256(actor)
    binding={'sceneId':'source-scene','stateId':'source-prefix'}
    producer=runner.producer_identity()
    physics={'schema':'resectionlab.tissue-physics.v2','profileHash':'sha256:example',
             'sceneId':'source-scene'}
    invalid={'schema':runner.SCHEMA,'stage':'bc-initialized-ppo',
        'controllerRevision':runner.CONTROLLER_REVISION,
        'actorSha256':identity,'state_dict':actor.state_dict(),
        'sourceBinding':binding,'producerIdentity':producer,'fitPhysicsAudit':physics,
        'observationSchema':'surgical-rl.v2-tool-frame-task-v1',
        'actionIds':ACTION_IDS,
        'taskGate':{'incrementalTargetGoalMm3':runner.TARGET_GOAL_MM3,
            'minAdmittedNonStopActions':runner.MIN_ADMITTED_ACTIONS,
            'incrementalNonTargetCapMm3':runner.NON_TARGET_CAP_MM3},
        'trainingReceipt':{'actualPublicTransitions':2,'sourceBinding':binding,
            'controllerRevision':runner.CONTROLLER_REVISION,
            'producerIdentity':producer,'fitPhysicsAudit':physics,
            'sampledEpisodes':[{'status':'success'}],'optimizerUpdates':[]}}
    path=tmp_path/'claimed-ppo.pt'
    torch.save(invalid,path)
    with pytest.raises(ValueError,match='actual actor updates'):
        runner.frozen_actor(path,identity)
    scene=SimpleNamespace(identity='transfer-scene',properties={
        'tissuePhysicsV2':{'schema':'resectionlab.tissue-physics.v2'}})
    with pytest.raises(ValueError,match='physics profile differs'):
        runner.frozen_actor(path,identity,scene=scene)


def test_frozen_actor_rejects_unbound_or_changed_producer_before_weights(tmp_path):
    source=Path(__file__).resolve().parents[1]/'scripts/run_patient_cusa_learning.py'
    spec=importlib.util.spec_from_file_location('patient_cusa_runner',source)
    runner=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    actor=Actor()
    identity=parameter_sha256(actor)
    bad={'schema':runner.SCHEMA,'stage':'bc','actorSha256':identity,
         'state_dict':actor.state_dict(),'producerIdentity':{'runnerSourceSha256':'false'},
         'trainingReceipt':{'producerIdentity':{'runnerSourceSha256':'false'}}}
    path=tmp_path/'unbound.pt'
    torch.save(bad,path)
    with pytest.raises(ValueError,match='policy, runner, engine, or runtime'):
        runner.frozen_actor(path,identity)
