"""Bounded learned CUSA continuation of a verified native access branch.

The selected DEVELOPMENT source is replayed through the public engine. A
policy receives only SurgicalRLEnv observations. Privileged teacher previews
are labeled as such and never presented to the frozen actor. This is an
engineering execution proof, not a treatment plan or generalization result.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import resource
import sys
import time

import numpy as np
from scipy.spatial.transform import Rotation
import torch

from .core import array_digest, semantic_digest
from .surgical_engine import SurgicalEngine
from .surgical_environment import load_environment
from .surgical_planning import _observer, _protection_measurements, validate_saved_comparison
from .surgical_rl_env import OBS_METRICS
from .surgical_rl_rewards import RewardConfig, RewardTracker, compute_reward
from .workspace_bundle import load_workspace

SCHEMA = 'resectionlab.patient-cusa-learning.v1'
CONTROLLER_REVISION = 'checked-duration-recovery.v1-post-frozen-0005-failure'
TARGET_GOAL_MM3 = 30.
NON_TARGET_CAP_MM3 = 20.
MIN_ADMITTED_ACTIONS = 2
DECISION_CAP = 6
APPLY_RETRY_DURATIONS_SECONDS = (.6, 1., 2.)
ACTION_IDS = ('inward-0.5', 'inward-1', 'inward-1.5', 'inward-2',
              'left-0.5', 'right-0.5', 'apply-here', 'inspect', 'STOP')
_OBSERVED_OFFSETS = ((0., 0., .5), (0., 0., 1.), (0., 0., 1.5), (0., 0., 2.),
                     (-.5, 0., 0.), (.5, 0., 0.), (0., 0., 0.),
                     (0., 0., 0.), (0., 0., 0.))


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def observation_hash(observation):
    return semantic_digest({key: array_digest(np.asarray(value))
                            for key, value in sorted(observation.items())})


def observed_options(observation, *, baseline_target=0., baseline_non_target=0.):
    """Nine tool-frame options and pilot-style features, observation only.

    The source route can be oblique. World-axis 2 mm moves from the older pilot
    violate its real shaft corridor, so offsets follow the *observed* CUSA pose.
    No source labels, preview, hidden engine state or receipt enters selection.
    """
    local = np.asarray(observation['local'][0], float)
    slots = np.asarray(observation['slots'], float)
    spacing = float(np.asarray(observation['local_spacing_mm'])[0])
    if local.shape != (8, 9, 9, 9) or slots.shape != (2, 12) or not .1 <= spacing <= 10.:
        raise ValueError('Invalid published local surgical observation')
    quaternion = slots[0, 3:7] / np.linalg.norm(slots[0, 3:7])
    rotation = Rotation.from_quat(quaternion)
    points = np.stack(np.meshgrid(*([np.arange(-4,5)*spacing]*3), indexing='ij'), axis=-1).reshape(-1,3)
    channels = local.reshape(8,-1)
    common = channels.mean(axis=1)
    radius = float(observation['tool_parameters'][0,0])
    metrics = np.asarray(observation['metrics'], float)
    target_progress = metrics[OBS_METRICS.index('tumor_removed_mm3')]-baseline_target
    non_target_progress = metrics[OBS_METRICS.index('healthy_removed_mm3')]-baseline_non_target
    remaining_goal = max(0., 1.-target_progress/TARGET_GOAL_MM3)
    remaining_cap = max(0., 1.-non_target_progress/NON_TARGET_CAP_MM3)
    rows, actions = [], []
    for index, offset in enumerate(_OBSERVED_OFFSETS):
        delta = rotation.apply(offset)
        squared = float(delta @ delta)
        fraction = np.clip(points @ delta / squared, 0, 1) if squared else np.zeros(len(points))
        selected = np.linalg.norm(points-fraction[:,None]*delta,axis=1) <= radius + .5*spacing
        mean = channels[:,selected].mean(axis=1) if selected.any() else np.zeros(8)
        healthy = ((channels[0,selected]>0)&(channels[1,selected]==0)).mean() if selected.any() else 0.
        rows.append([*mean,*common,float(np.linalg.norm(delta)/2.),float(index==7),float(healthy),
                     float(index==8),remaining_goal,remaining_cap])
        action = {'slot':'left','operation':'inspect' if index==7 else 'apply','duration':.2}
        if index < 7:
            action['pose'] = {'position': (slots[0,:3]+delta).tolist(), 'quaternion':quaternion.tolist()}
        actions.append(None if index==8 else action)
    return np.asarray(rows,np.float32), actions


def _rss_bytes():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value * (1 if sys.platform == 'darwin' else 1024))


@dataclass
class Budget:
    max_wall_seconds: float = 600.
    max_calls: int = 200
    max_rss_bytes: int = 3 * 1024**3
    started: float = 0.
    calls: int = 0

    def __post_init__(self):
        if not 0 < self.max_wall_seconds <= 600 or not 0 < self.max_calls <= 200 or not 0 < self.max_rss_bytes <= 3 * 1024**3:
            raise ValueError('Learning budget exceeds the bounded prototype limits')
        self.started = time.perf_counter()

    def check(self):
        if time.perf_counter() - self.started > self.max_wall_seconds:
            raise TimeoutError('Patient CUSA wall budget exhausted')
        if self.calls > self.max_calls:
            raise TimeoutError('Patient CUSA transition budget exhausted')
        if _rss_bytes() > self.max_rss_bytes:
            raise MemoryError('Patient CUSA memory budget exhausted')

    def account(self):
        self.calls += 1
        self.check()

    def reserve(self,count):
        if type(count) is not int or count<0 or self.calls+count>self.max_calls:
            raise TimeoutError('Patient CUSA transition budget exhausted')
        self.calls+=count
        self.check()

    def report(self):
        return {'wallSeconds': time.perf_counter()-self.started,
                'transitionCalls': self.calls, 'peakRssBytes': _rss_bytes(),
                'maxWallSeconds': self.max_wall_seconds, 'maxTransitionCalls': self.max_calls,
                'maxRssBytes': self.max_rss_bytes}


def _cleanup_boundary(plan, actions):
    phases = [p for p in plan.get('directorPhases', []) if p.get('id') == 'corridor-withdrawal']
    if len(phases) != 1:
        raise ValueError('Selected access branch lacks an explicit recorded withdrawal phase')
    position = phases[0].get('startPosition')
    if type(position) is not int or not 1 <= position < len(actions):
        raise ValueError('Recorded withdrawal boundary is invalid')
    tail = actions[position:]
    if [a.get('operation') for a in tail] != ['deactivate', 'withdraw'] or any(a.get('slot') != 'left' for a in tail):
        raise ValueError('Selected branch does not end in a checked CUSA deactivate/withdraw pair')
    return position


def prepare(source_path, branch_id, *, budget, seed=17):
    """Replay a recorded prefix into the existing RL adapter; inject no state."""
    source_path = Path(source_path).resolve()
    case, artifacts, workspace = load_workspace(source_path)
    scene = load_environment(source_path)
    record = artifacts.get('surgical_toolbox')
    if not record:
        raise ValueError('A saved native source access case is required')
    comparison = record.get('planning')
    if not comparison or comparison.get('task') != 'anatomy-access':
        raise ValueError('Saved source branch lacks access comparison evidence')
    validate_saved_comparison(scene, record, comparison)
    plans = [p for p in comparison['alternatives'] if p.get('branchId') == branch_id]
    if len(plans) != 1:
        raise ValueError('Select exactly one saved source access alternative')
    plan = plans[0]
    if plan.get('accessCompleteness') != 'unsupported_skull_access_brain_corridor':
        raise ValueError('Prototype requires a checked modeled-brain source corridor')
    budget.reserve(len(record['branches'][branch_id]['actions']))
    full = SurgicalEngine.restore(scene, record['branches'][branch_id], check=budget.check)
    position = _cleanup_boundary(plan, full.actions)
    budget.reserve(position)
    prefix = full.branch(position, check=budget.check)
    if prefix.state_id != plan['decisions'][position-1]['stateId'] or prefix.state_id != full.history[position-1]['stateAfter']:
        raise ValueError('Saved access prefix differs from the sealed action history')
    left = prefix.state['slots']['left']
    if left['instrument'] != 'cusa' or not left['inserted']:
        raise ValueError('Selected prefix is not an inserted left CUSA contact')
    target = prefix.metrics()['targetRemovedMm3']
    non_target = prefix.metrics()['nonTargetRemovedMm3']
    tracker = RewardTracker(prefix, prefix._protected_ids)
    observation, _ = _observer(prefix, tracker)
    binding = {'sourcePath': str(source_path), 'sourceSha256': sha256(source_path),
        'caseHash': case.semantic_hash, 'sceneId': scene.identity,
        'sourceLabelsHash': comparison['sourceLabelsHash'],
        'guardContractId': comparison['guardContractId'],
        'branchId': branch_id, 'planSeal': plan['planSeal'],
        'comparisonSeal': comparison['comparisonSeal'],
        'position': position, 'stateId': prefix.state_id,
        'actionsSha256': semantic_digest(full.actions[:position]),
        'targetRemovedMm3': target, 'nonTargetRemovedMm3': non_target,
        'sourceBrainLayers': plan['route'].get('sourceBrainLayers', []),
        'sourceTargetLayers': plan['route'].get('sourceTargetLayers', [])}
    return {'case': case, 'artifacts': artifacts, 'workspace': workspace,
            'scene': scene, 'record': record, 'plan': plan,
            'prefix': prefix, 'observation': observation,
            'binding': binding}


def _permitted_preview(engine, action, baseline_non_target, budget):
    budget.check()
    if budget.calls>=budget.max_calls:
        raise TimeoutError('Patient CUSA transition budget exhausted')
    preview = engine.preview(action, check=budget.check, include_view=False)
    budget.account()
    if preview['status'] == 'rejected' or (preview['status'] == 'partial' and
            preview.get('reason') != 'TISSUE_ASPIRATION_IN_PROGRESS'):
        return preview, 'physical_refusal'
    if preview['metrics']['nonTargetRemovedMm3']-baseline_non_target > NON_TARGET_CAP_MM3+1e-9:
        return preview, 'incremental_non_target_cap'
    return preview, None


def _retry_durations(engine, action):
    """Declared longer timing for the same pose after a swept-shaft refusal."""
    original = float(action.get('duration', 1.))
    if action['operation'] == 'apply' and 'pose' in action:
        candidates = APPLY_RETRY_DURATIONS_SECONDS
    elif action['operation'] == 'withdraw' and 'pose' in action:
        distance = float(np.linalg.norm(np.subtract(
            action['pose']['position'], engine.state['slots'][action['slot']]['position'])))
        candidates = (min(30., max(original, distance / 2.)), 30.)
    else:
        candidates = ()
    return tuple(dict.fromkeys(float(value) for value in candidates
        if original + 1e-9 < value <= 30.))


def _commit(engine, tracker, action, baseline_non_target, budget):
    original = copy.deepcopy(action)
    attempts = []
    candidate_durations = (float(action.get('duration', 1.)), *_retry_durations(engine, action))
    for index, duration in enumerate(candidate_durations):
        if index and (attempts[-1]['refusal'] != 'physical_refusal' or
                      attempts[-1]['reason'] != 'SHAFT_COLLISION'):
            break
        candidate = copy.deepcopy(original)
        candidate['duration'] = duration
        started = time.perf_counter()
        preview, refusal = _permitted_preview(engine, candidate, baseline_non_target, budget)
        attempts.append({'durationSeconds':duration,
            'previewElapsedSeconds':time.perf_counter()-started,
            'publicTransitionIndex':budget.calls,
            'previewStatus':preview['status'],'reason':preview.get('reason'),
            'refusal':refusal,'previewStateId':preview['stateId'],
            'previewTargetRemovedMm3':preview['metrics'].get('targetRemovedMm3'),
            'previewNonTargetRemovedMm3':preview['metrics']['nonTargetRemovedMm3']})
        if not refusal:
            action = candidate
            break
    if refusal:
        return None, 0., {'outcome': preview, 'stateId': engine.state_id,
                          'action':candidate,'attempts':attempts}, refusal
    before = tracker.measure(engine, _protection_measurements(engine))
    if budget.calls>=budget.max_calls:
        raise TimeoutError('Patient CUSA transition budget exhausted')
    outcome = engine.step(action, check=budget.check)
    budget.account()
    if (outcome['status'] != preview['status'] or
            engine.state_id != preview['stateId'] or
            engine.metrics()['nonTargetRemovedMm3']-baseline_non_target > NON_TARGET_CAP_MM3+1e-9):
        raise ValueError('CUSA preview, commit, or removal cap differed')
    attempts[-1]['commitStatus'] = outcome['status']
    attempts[-1]['commitStateId'] = engine.state_id
    attempts[-1]['commitTransitionIndex'] = budget.calls
    after = tracker.measure(engine, _protection_measurements(engine))
    reward, _ = compute_reward(before, after, outcome, RewardConfig())
    observation, _ = _observer(engine, tracker)
    return observation, float(reward), {'outcome': outcome, 'stateId': engine.state_id,
                                      'action':action,'attempts':attempts}, None


def execute(prepared, choose, *, budget, method, max_decisions=DECISION_CAP):
    """Run an observed-only actor with a fixed checked cleanup after its STOP."""
    engine = prepared['prefix'].clone()
    tracker = RewardTracker(engine, engine._protected_ids)
    obs, _ = _observer(engine, tracker)
    baseline = prepared['binding']['nonTargetRemovedMm3']
    base_target = prepared['binding']['targetRemovedMm3']
    rows, training = [], []
    admitted_non_stop = 0
    reason = 'decision_limit'
    for index in range(max_decisions):
        budget.check()
        features, actions = observed_options(obs, baseline_target=base_target,
                                            baseline_non_target=baseline)
        selected, scores = choose(features, obs, index)
        if type(selected) is not int or not 0 <= selected < len(ACTION_IDS):
            raise ValueError('Policy selected an invalid observed option')
        feature_hash = array_digest(features)
        before = observation_hash(obs)
        if selected == ACTION_IDS.index('STOP'):
            reason = 'policy_stop'
            rows.append({'index': index, 'observationHash': before,
                         'featureHash': feature_hash, 'actionId': 'STOP',
                         'scores': list(map(float, scores)), 'action': None,
                         'status': 'selected', 'reward': 0., 'stateId': engine.state_id})
            training.append((features.copy(), selected, 0.))
            break
        action = actions[selected]
        next_obs, reward, info, refusal = _commit(engine, tracker, action, baseline, budget)
        rows.append({'index': index, 'observationHash': before,
                     'featureHash': feature_hash, 'actionId': ACTION_IDS[selected],
                     'scores': list(map(float, scores)), 'actorAction':action,
                     'action': info['action'],'attempts':info['attempts'],
                     'controllerRecoveryApplied':len(info['attempts'])>1,
                     'previewStatus': info['outcome']['status'],
                     'status': 'refused' if refusal else info['outcome']['status'],
                     'reason': refusal or info['outcome'].get('reason'),
                     'engineReason': info['outcome'].get('reason'),
                     'reward': float(reward), 'stateId': info['stateId']})
        training.append((features.copy(), selected, float(reward)))
        if next_obs is not None: obs = next_obs
        if refusal:
            reason = refusal; break
        admitted_non_stop += 1
    # STOP is an explicit cleanup request. A budget/refusal also attempts a
    # safe checked withdrawal, but never upgrades that run to policy success.
    cleanup = []
    if reason:
        for action in [{'slot':'left','operation':'deactivate','duration':.2},
                {'slot':'left','operation':'withdraw','pose':copy.deepcopy(prepared['plan']['route']['outsidePose']),
                 'duration':2.}]:
            before = observation_hash(obs)
            next_obs, reward, info, refusal = _commit(engine, tracker, action, baseline, budget)
            if next_obs is not None: obs = next_obs
            cleanup.append({'observationHash':before,'selectedAction':action,
                            'action':info['action'],'attempts':info['attempts'],
                            'controllerRecoveryApplied':len(info['attempts'])>1,
                            'status':'refused' if refusal else info['outcome']['status'],
                            'reason':refusal or info['outcome'].get('reason'),
                            'engineReason':info['outcome'].get('reason'),
                            'reward':float(reward),'stateId':info['stateId']})
            if refusal or info['outcome']['status']=='rejected':
                reason = 'cleanup_refused_'+str(refusal or info['outcome'].get('reason') or 'episode_limit')
                break
    delta_target = engine.metrics()['targetRemovedMm3']-base_target
    delta_non_target = engine.metrics()['nonTargetRemovedMm3']-baseline
    state = engine.state
    cleanup_complete = all(not state['slots'][hand]['active'] and not state['slots'][hand]['inserted'] and
                           not state['slots'][hand]['grip'] for hand in ('left','right')) and not state['holds']
    gate_met = delta_target >= TARGET_GOAL_MM3-1e-9 and admitted_non_stop >= MIN_ADMITTED_ACTIONS
    status = ('success' if reason == 'policy_stop' and gate_met and cleanup_complete
              else 'abstained' if reason == 'policy_stop' and cleanup_complete else 'incomplete')
    return {'engine': engine, 'training': training,
            'receipt': {'method': method, 'status': status, 'stopReason': reason,
                        'controllerRevision':CONTROLLER_REVISION,
                        'decisions': rows, 'cleanup': cleanup,
                        'deltaTargetMm3': float(delta_target),
                        'deltaNonTargetMm3': float(delta_non_target),
                        'admittedNonStopActions': admitted_non_stop,
                        'taskGateMet': gate_met,
                        'cleanupComplete': cleanup_complete,
                        'finalStateId': engine.state_id,
                        'sourceStateId': prepared['binding']['stateId'],
                        'nativeActions': len(engine.actions)-prepared['binding']['position']}}


class Actor(torch.nn.Module):
    """Small categorical scorer over the nine observation-derived options."""
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(22,32),torch.nn.Tanh(),
                                       torch.nn.Linear(32,1))

    def forward(self, features):
        return self.net(torch.as_tensor(features,dtype=torch.float32)).squeeze(-1)


class Critic(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(22,32),torch.nn.Tanh(),
                                       torch.nn.Linear(32,1))

    def forward(self, features):
        return self.net(torch.as_tensor(features,dtype=torch.float32).mean(dim=0)).squeeze(-1)


def parameter_sha256(model):
    digest=hashlib.sha256()
    for name,value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def scripted_choice(features, observation, index):
    """Declared control: two 2 mm tool-frame strokes, then explicit STOP."""
    del observation
    return (3 if index < MIN_ADMITTED_ACTIONS else 8), [0.]*len(features)


def actor_choice(model, *, sample=False, generator=None):
    def choose(features, observation, index):
        del observation,index
        with torch.no_grad():
            logits=model(features)
            selected=int(torch.multinomial(logits.softmax(-1),1,generator=generator)) if sample else int(logits.argmax())
        return selected, logits.tolist()
    return choose


def fit_bc(demonstration, *, seed=17, epochs=300):
    """Freeze one measured DEVELOPMENT demonstration; no hidden-state actor input."""
    if demonstration['receipt']['status']!='success':
        raise ValueError('Cannot imitate a failed CUSA source demonstration')
    if len(demonstration['training']) < MIN_ADMITTED_ACTIONS+1:
        raise ValueError('BC source lacks two real strokes and STOP')
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        actor=Actor()
    initial=parameter_sha256(actor)
    optimizer=torch.optim.Adam(actor.parameters(),lr=.01)
    for _ in range(epochs):
        loss=torch.stack([-torch.log_softmax(actor(features),dim=0)[selected]
                          for features,selected,_ in demonstration['training']]).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    final=parameter_sha256(actor)
    if final==initial:
        raise ValueError('BC did not change actor weights')
    actor.eval()
    predictions=[int(actor(features).argmax()) for features,_,_ in demonstration['training']]
    expected=[selected for _,selected,_ in demonstration['training']]
    return actor, {'seed':seed,'epochs':epochs,'initialActorSha256':initial,
                   'finalActorSha256':final,'demonstrationActions':expected,
                   'fittedActions':predictions,'fittedExact':predictions==expected,
                   'finalLoss':float(loss.detach())}


def ppo_update(actor, critic, episodes, *, seed, epochs=2):
    """Two clipped on-policy actor/critic epochs; hashes prove actor learning."""
    if not episodes or not any(ep['training'] for ep in episodes):
        raise ValueError('PPO needs real sampled transitions')
    actor_before=parameter_sha256(actor)
    critic_before=parameter_sha256(critic)
    optimizer=torch.optim.Adam([{'params':actor.parameters(),'lr':.001},
                                {'params':critic.parameters(),'lr':.001}])
    samples=[]
    for episode in episodes:
        rewards=[float(r) for _,_,r in episode['training']]
        if episode['receipt']['status']=='success': rewards[-1]+=.02
        elif episode['receipt']['status']=='abstained': rewards[-1]-=.01
        else: rewards[-1]-=.02
        future=0.
        for (features,selected,_),reward in reversed(list(zip(episode['training'],rewards))):
            future=reward+.95*future
            with torch.no_grad():
                old_logprob=torch.log_softmax(actor(features),dim=0)[selected]
                old_value=critic(features)
            samples.append((features.copy(),selected,float(future),float(old_logprob),float(old_value)))
    actor.train();critic.train()
    for _ in range(epochs):
        losses=[]
        for features,selected,total,old_logprob,old_value in samples:
            logits=actor(features)
            logprob=torch.log_softmax(logits,dim=0)[selected]
            ratio=torch.exp(logprob-old_logprob)
            advantage=torch.as_tensor(total-old_value,dtype=torch.float32)
            actor_loss=-torch.minimum(ratio*advantage,
                torch.clamp(ratio,.8,1.2)*advantage)
            value_loss=.5*(critic(features)-total).square()
            entropy=torch.distributions.Categorical(logits=logits).entropy()
            losses.append(actor_loss+value_loss-.001*entropy)
        loss=torch.stack(losses).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(actor.parameters())+list(critic.parameters()),1.)
        optimizer.step()
    actor.eval();critic.eval()
    actor_after=parameter_sha256(actor)
    critic_after=parameter_sha256(critic)
    if actor_after==actor_before:
        raise ValueError('PPO actor weights/bias did not change')
    return {'seed':seed,'epochs':epochs,'sampledDecisions':len(samples),
            'actorBeforeSha256':actor_before,'actorAfterSha256':actor_after,
            'criticBeforeSha256':critic_before,'criticAfterSha256':critic_after,
            'actorChanged':True,'criticChanged':critic_after!=critic_before,
            'loss':float(loss.detach())}
