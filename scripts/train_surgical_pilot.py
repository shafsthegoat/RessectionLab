#!/usr/bin/env python3
"""One bounded, explicitly admitted pilot on two already-exposed development cases.

Preparation executes real transitions but never fits a policy. Training requires
a separate lead-owned launch receipt after the two-case desktop demonstration.
No historical learning pathway or checkpoint is opened by this runner.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import resource
import sys
import time

import numpy as np
import torch
from scipy.ndimage import maximum_filter

from resectionlab.core import array_digest
from resectionlab.surgical_environment import load_environment
from resectionlab.surgical_rl_env import SurgicalRLEnv, RLConfig
from resectionlab.surgical_rl_protection import load_protection

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 'resectionlab.surgical-pilot.v1'
SETTINGS = dict(seeds=[17, 29, 43], updates=4, episodes_per_update=2,
                decisions=6, learning_rate=.001, hidden=32, entropy_weight=.01,
                gamma=.95, return_scale=100., max_grad_norm=1.,
                max_wall_seconds=900., max_transition_calls=1000,
                max_action_wall_seconds=10., max_rss_bytes=6 * 1024**3,
                policy_healthy_budget_mm3=5., setup_healthy_budget_mm3=100.,
                local_spacing_mm=1., step_mm=2., action_seconds=.2)
SOURCES = {
    'sub06': dict(case='outputs/surgical-patient-v2/patient.ressectionlab',
                  protection='outputs/surgical-patient-v2/rl-protection.json',
                  identity='ZENODO-17486320-meningioma-sub-06-surgical-research',
                  role='already exposed development; acquired MRI with unreviewed model estimates'),
    'ucsf0004': dict(case='outputs/poc-cases/UCSF-PDGM-0004-standard-v2.ressectionlab',
                    protection='outputs/standard-environments/UCSF-PDGM-0004/rl-protection.json',
                    identity='UCSF-PDGM-0004-surgical-research',
                    role='development in manifests/cohort_registry.json; source annotated enhancing target'),
}
ACTION_IDS = ('x-', 'x+', 'y-', 'y+', 'z-', 'z+', 'apply_here', 'inspect', 'STOP')
OFFSETS = np.array([[-2, 0, 0], [2, 0, 0], [0, -2, 0], [0, 2, 0],
                    [0, 0, -2], [0, 0, 2], [0, 0, 0], [0, 0, 0], [0, 0, 0]], float)
LIMITATIONS = [
    'Simulated experience on acquired development anatomy; no recorded operations or clinical validation.',
    'Both anatomies are used in optimization and assessment; neither is held out.',
    'Setup uses target and anatomical maps to choose a reachable local task; identical prefixes for all methods.',
    'Source voxel arrays, source roles and affine geometry are preserved; no tissue is pre-removed.',
    'Observation-only local geometric greedy is an adaptation of the existing observed-greedy strategy, not the native search implementation.',
    'The learned decisions are CUSA-only. A matched fixed Frazier suction tail is checked separately; it is not learned instrument selection.',
    'No hyperparameter sweep, best-seed selection, final-patient access, or supported treatment claim.',
]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''): h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def source_hashes():
    paths = [Path(__file__), *sorted((ROOT / 'src/resectionlab').glob('surgical_*.py')),
             *[ROOT / 'src/resectionlab' / name for name in ('core.py', 'geometry.py', 'imaging.py')],
             ROOT / 'manifests/cohort_registry.json',
             ROOT / 'outputs/surgical-patient-v2/drill-verification.json',
             ROOT / 'outputs/surgical-patient-sub06/verification.json']
    for source in SOURCES.values():
        paths.extend([ROOT / source['case'], ROOT / source['protection'],
                      (ROOT / source['protection']).with_suffix('.npz')])
    return {str(p.relative_to(ROOT)): file_hash(p) for p in paths}


class Budget:
    def __init__(self):
        self.started = time.perf_counter()
        self.calls = 0
        self.seconds = {}
        self.max_action_seconds = 0.

    def check(self):
        if time.perf_counter() - self.started > SETTINGS['max_wall_seconds']:
            raise RuntimeError('WALL_TIME_BUDGET')
        if self.calls >= SETTINGS['max_transition_calls']:
            raise RuntimeError('TRANSITION_CALL_BUDGET')
        if self.rss() > SETTINGS['max_rss_bytes']:
            raise RuntimeError('MEMORY_BUDGET')

    @staticmethod
    def rss():
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024)

    def step(self, env, action, phase):
        self.check()
        started = time.perf_counter()
        result = env.step_action(action)
        elapsed = time.perf_counter() - started
        self.calls += 1
        self.seconds[phase] = self.seconds.get(phase, 0.) + elapsed
        self.max_action_seconds = max(self.max_action_seconds, elapsed)
        if elapsed > SETTINGS['max_action_wall_seconds']:
            raise RuntimeError(f'ACTION_LATENCY_BUDGET: {elapsed:.3f} seconds')
        self.check()
        return result, elapsed

    def report(self):
        return dict(transition_calls=self.calls, transition_wall_seconds_by_phase=self.seconds.copy(),
                    wall_seconds=time.perf_counter() - self.started,
                    max_action_wall_seconds=self.max_action_seconds, process_peak_rss_bytes=self.rss())


def config():
    # Exposure costs stay in the cumulative ledger. The stricter incremental
    # local damage bound is checked after every policy decision below.
    return RLConfig(max_steps=80, max_seconds=120., max_healthy_removed_mm3=100.,
                    local_spacing_mm=SETTINGS['local_spacing_mm'])


def load_source(name):
    if name not in SOURCES: raise ValueError('Only the two declared development sources are admitted')
    source = SOURCES[name]
    scene = load_environment(ROOT / source['case'])
    if scene.source['caseId'] != source['identity']: raise ValueError('Source identity differs')
    protection = load_protection(scene, ROOT / source['protection'], strict=True)
    return SurgicalRLEnv(scene, protection=protection, config=config())


def exposure_actions(name, env):
    """Privileged task construction only; outputs are frozen before any fitting."""
    if name == 'sub06':
        def recipe(path):
            return [copy.deepcopy(r['action']) for r in json.loads((ROOT / path).read_text())['actions']]
        drill = recipe('outputs/surgical-patient-v2/drill-verification.json')
        cusa = recipe('outputs/surgical-patient-sub06/verification.json')
        return [drill[0]] + [copy.deepcopy(drill[1]) for _ in range(11)] + cusa[2:]
    if name != 'ucsf0004': raise ValueError('Undeclared development source')
    scene = env.scene
    target = np.isin(scene.labels, [r['id'] for r in scene.regions if r.get('target')])
    risk = np.isin(scene.labels, [r['id'] for r in scene.regions if r['material'] not in ('tumor', 'soft_tissue')])
    risk.ravel()[env.protection.cells] = True
    risk = maximum_filter(risk, size=(5, 5, 5))
    candidates = []
    for x, y in np.argwhere(target.any(axis=2)):
        z = int(np.flatnonzero(target[x, y])[-1])
        if risk[x, y, z:].any(): continue
        healthy = int(np.count_nonzero((scene.labels[x-1:x+2, y-1:y+2, z:] != 0)
                                      & ~target[x-1:x+2, y-1:y+2, z:]))
        candidates.append((healthy, -z, int(x), int(y), z))
    if not candidates: raise RuntimeError('NO_DECLARED_VESSEL_CLEAR_EXPOSURE')
    _, _, x, y, z = min(candidates)
    goal = scene.points([np.ravel_multi_index((x, y, z), scene.shape)])[0]
    start = goal.copy(); start[2] = scene.points([scene.labels.size - 1])[0, 2] + 5
    return [dict(slot='left', operation='exchange', instrument='cusa', duration=.1),
            dict(slot='left', operation='place', duration=.1, parameters={'radiusMm': .6},
                 pose=dict(position=start.tolist(), quaternion=[0., 1., 0., 0.])),
            dict(slot='left', operation='apply', duration=1.,
                 pose=dict(position=goal.tolist(), quaternion=[0., 1., 0., 0.]))]


def fork_prepared(base):
    """Copy an actually executed prefix; never inject removed cells/tool poses.

    SurgicalEngine.clone is the same rollback clone used by step(). Sharing the
    immutable scene/protection avoids copying full anatomy for every rollout.
    This optimization is checked against full public-action replay in preparation.
    """
    return copy.deepcopy(base, {id(base.scene): base.scene, id(base.protection): base.protection,
                               id(base.engine): base.engine.clone()})


def observation_hash(obs):
    return digest({key: array_digest(np.asarray(value)) for key, value in sorted(obs.items())})


def candidates(obs):
    """Candidate actions and scores depend ONLY on the published observation.

    No info, environment, source label maps, occupancy, snapshot, or preview is
    accepted. Samples are an imperfect nominal estimate, not a feasibility oracle.
    """
    local = np.asarray(obs['local'][0], dtype=float)
    points = np.stack(np.meshgrid(*([np.arange(-4, 5) * SETTINGS['local_spacing_mm']] * 3),
                                 indexing='ij'), axis=-1).reshape(-1, 3)
    channels = local.reshape(8, -1)
    radius = float(obs['tool_parameters'][0, 0])
    common = channels.mean(axis=1)
    rows, actions = [], []
    for i, delta in enumerate(OFFSETS):
        fraction = np.clip(points @ delta / (delta @ delta), 0, 1) if delta @ delta else np.zeros(len(points))
        distance = np.linalg.norm(points - fraction[:, None] * delta, axis=1)
        selected = distance <= radius + .5 * SETTINGS['local_spacing_mm']
        mean = channels[:, selected].mean(axis=1) if selected.any() else np.zeros(8)
        healthy = ((channels[0, selected] > 0) & (channels[1, selected] == 0)).mean() if selected.any() else 0.
        rows.append([*mean, *common, np.linalg.norm(delta) / SETTINGS['step_mm'], float(i == 7), healthy, float(i == 8)])
        action = dict(slot='left', operation='deactivate' if i == 8 else 'inspect' if i == 7 else 'apply',
                      duration=SETTINGS['action_seconds'])
        if i < 7:
            q = np.asarray(obs['slots'][0, 3:7], float); q = q / np.linalg.norm(q)
            action['pose'] = dict(position=(np.asarray(obs['slots'][0, :3], float) + delta).tolist(), quaternion=q.tolist())
        actions.append(action)
    return np.asarray(rows, np.float32), actions


def observed_greedy(features):
    # Existing native observed greedy ranks permitted nominal gain minus burden.
    # Its NativeSpatialTask geometry is incompatible with SurgicalRLEnv. This
    # explicitly labeled local adaptation scores the same nine observed options
    # supplied to the policy, with zero hidden-state transition previews.
    target, protected, healthy = features[:, 1], features[:, 2], features[:, 18]
    scores = target - 2. * healthy - 5. * protected - .01 * features[:, 16] - .0012
    scores[7:] = -.0012
    # Explicit nominal no-positive-progress STOP threshold, not hidden truth.
    selected = 8 if float(scores.max()) <= float(scores[8]) else int(np.argmax(scores))
    return selected, scores.tolist()


def suction_tail_actions(obs, exterior_pose):
    """Four fixed public actions; endpoint/settings come from the same observation."""
    q = np.asarray(obs['slots'][0, 3:7], float); q /= np.linalg.norm(q)
    current = dict(position=obs['slots'][0, :3].astype(float).tolist(), quaternion=q.tolist())
    return [dict(slot='left', operation='withdraw', duration=.2, pose=copy.deepcopy(exterior_pose)),
            dict(slot='left', operation='exchange', instrument='frazier_suction', duration=.1),
            dict(slot='left', operation='move', duration=.2, pose=current,
                 parameters={'radiusMm': float(obs['tool_parameters'][0, 0])}),
            dict(slot='left', operation='apply', duration=.2)]


class Policy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(20, SETTINGS['hidden']), torch.nn.Tanh(),
                                       torch.nn.Linear(SETTINGS['hidden'], 1))

    def forward(self, features):
        return self.net(torch.as_tensor(features, dtype=torch.float32)).squeeze(-1)


def parameter_hash(model):
    h = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        h.update(key.encode()); h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def rollout(base, observation, method, seed, budget, model=None, generator=None, exterior_pose=None):
    started = time.perf_counter()
    env = fork_prepared(base)
    obs = copy.deepcopy(observation)
    initial = env.measurements.copy()
    rows, samples = [], []
    rng = np.random.default_rng(seed)
    reason = 'local_decision_limit'
    for decision in range(SETTINGS['decisions']):
        features, actions = candidates(obs)
        scores = None
        if method == 'random': selected = int(rng.integers(len(actions)))
        elif method == 'scripted': selected = 4  # fixed superior approach advances inferiorly
        elif method == 'observed_greedy_local': selected, scores = observed_greedy(features)
        elif method in ('untrained', 'trained', 'training'):
            with torch.no_grad(): logits = model(features)
            selected = int(torch.multinomial(logits.softmax(-1), 1, generator=generator)) if method == 'training' else int(logits.argmax())
        else: raise ValueError('Unknown method')
        before_hash = observation_hash(obs)
        (obs, reward, terminated, truncated, info), elapsed = budget.step(env, actions[selected], method)
        samples.append((features.copy(), selected, float(reward)))
        rows.append(dict(decision=decision, observation_sha256=before_hash, action_id=ACTION_IDS[selected],
                         action=actions[selected], reward=reward, reward_terms=info['reward_terms'],
                         state_id=info['stateId'], status=info['outcome']['status'],
                         reason=info['outcome'].get('reason'), wall_seconds=elapsed, nominal_scores=scores))
        if terminated or truncated:
            reason = info['termination_reason']; break
        if selected == 8:
            reason = 'policy_stop'; break
        if env.measurements['healthy_removed_mm3'] - initial['healthy_removed_mm3'] > SETTINGS['policy_healthy_budget_mm3']:
            reason = 'local_healthy_damage_budget'; break
    cusa_return = sum(r['reward'] for r in rows)
    tail_rows = []
    # A fixed measured instrument continuation, never a privileged action option.
    if method != 'training' and exterior_pose is not None and not env._done and reason not in ('local_healthy_damage_budget', 'policy_stop'):
        for action in suction_tail_actions(obs, exterior_pose):
            (obs, reward, terminated, truncated, info), elapsed = budget.step(env, action, 'suction_tail')
            tail_rows.append(dict(action=action, reward=reward, reward_terms=info['reward_terms'],
                                  status=info['outcome']['status'], reason=info['outcome'].get('reason'),
                                  state_id=info['stateId'], wall_seconds=elapsed,
                                  debris_removed_mm3=info['outcome'].get('metrics', {}).get('debrisRemovedMm3', 0.)))
            if terminated or truncated or info['outcome']['status'] != 'accepted': break
    delta = {key: float(env.measurements[key] - value) for key, value in initial.items()}
    report = dict(method=method, seed=seed, decisions=rows, stop_reason=reason,
                  local_task_terminated=reason == 'policy_stop' or env.terminated,
                  local_task_truncated=reason in ('local_decision_limit', 'local_healthy_damage_budget') or env.truncated,
                  cusa_return=cusa_return, local_return=cusa_return + sum(r['reward'] for r in tail_rows),
                  suction_tail=tail_rows, cumulative_return=env.total_reward,
                  metrics=env.measurements.copy(), local_metrics=delta, prefix_metrics=initial,
                  wall_seconds=time.perf_counter() - started, state_id=env.engine.state_id,
                  native_success=env.termination_reason == 'target_fraction_reached')
    return samples, report, env.snapshot()


def optimizer_update(model, optimizer, episodes):
    """Actual on-policy REINFORCE, fitted only to this batch's real transitions."""
    started = time.perf_counter(); before = parameter_hash(model)
    losses, entropies = [], []
    for episode in episodes:
        tail = 0.
        returns = []
        for _, _, reward in reversed(episode):
            tail = reward + SETTINGS['gamma'] * tail; returns.append(tail)
        for (features, action, _), value in zip(episode, reversed(returns)):
            distribution = torch.distributions.Categorical(logits=model(features))
            losses.append(-distribution.log_prob(torch.tensor(action)) * value * SETTINGS['return_scale'])
            entropies.append(distribution.entropy())
    loss = torch.stack(losses).mean() - SETTINGS['entropy_weight'] * torch.stack(entropies).mean()
    optimizer.zero_grad(set_to_none=True); loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), SETTINGS['max_grad_norm'])
    if not torch.isfinite(loss) or not torch.isfinite(norm): raise RuntimeError('NONFINITE_OPTIMIZER_UPDATE')
    optimizer.step()
    after = parameter_hash(model)
    if before == after: raise RuntimeError('OPTIMIZER_DID_NOT_CHANGE_PARAMETERS')
    return dict(loss=float(loss.detach()), gradient_norm=float(norm), before_sha256=before,
                after_sha256=after, parameters_changed=True, transition_count=sum(map(len, episodes)),
                optimizer_wall_seconds=time.perf_counter() - started)


def replay(env, record, budget):
    if record['environmentId'] != env.environment_id or record['config'] != asdict(env.config):
        raise RuntimeError('FINAL_REPLAY_ENVIRONMENT_MISMATCH')
    obs, _ = env.reset(seed=0)
    for item in record['attempts']:
        (obs, *_), _ = budget.step(env, item['action'], 'replay')
        if env.attempts[-1] != item: raise RuntimeError('FINAL_REPLAY_ATTEMPT_MISMATCH')
    if (env.engine.state_id != record['stateId'] or env.measurements != record['measurements']
            or env.total_reward != record['totalReward'] or env.terminated != record['terminated']
            or env.truncated != record['truncated']):
        raise RuntimeError('FINAL_REPLAY_LEDGER_MISMATCH')
    return dict(state_id=env.engine.state_id, reward_ledger_equal=True, measurements_equal=True), obs


def prepare(output):
    output.mkdir(parents=True, exist_ok=True)
    refuse_repeat_fit(output)
    budget = Budget()
    document = dict(schema=SCHEMA, status='preparing', settings=SETTINGS, sources=SOURCES,
                    source_hashes=source_hashes(), cases={}, limitations=LIMITATIONS, optimizer_updates=0)
    document['runtime'] = dict(python=sys.version, numpy=np.__version__, torch=torch.__version__, device='cpu')
    try:
        for name in SOURCES:
            env = load_source(name); labels_hash = array_digest(env.scene.labels)
            obs, _ = env.reset(seed=17)
            actions = exposure_actions(name, env)
            rows = []
            for action in actions:
                (obs, reward, term, trunc, info), elapsed = budget.step(env, action, 'exposure')
                rows.append(dict(action=action, status=info['outcome']['status'], reason=info['outcome'].get('reason'),
                                 reward=reward, metrics=info['metrics'], wall_seconds=elapsed))
                if term or trunc or info['outcome']['status'] == 'rejected':
                    raise RuntimeError(f'EXPOSURE_BLOCKED: {name}: {info["outcome"]}')
            if env.measurements['tumor_removed_mm3'] <= 0 or not obs['local'][0, 1].any():
                raise RuntimeError(f'NO_REACHABLE_LOCAL_TARGET: {name}')
            prefix = env.snapshot()
            # One real bounded forward option must remove additional target.
            # Equivalent native replay proves the cached fork has not injected state.
            fork = fork_prepared(env); _, actions_local = candidates(obs)
            (next_obs, _, _, _, probe_info), probe_elapsed = budget.step(fork, actions_local[4], 'preflight_probe')
            if probe_info['metrics']['tumor_removed_mm3'] <= env.measurements['tumor_removed_mm3']:
                raise RuntimeError(f'LOCAL_TARGET_ACTION_INEFFECTIVE: {name}')
            verification, _ = replay(load_source(name), fork.snapshot(), budget)
            exterior = next(a['pose'] for a in actions if a['operation'] == 'place')
            tail_rows = []
            for action in suction_tail_actions(next_obs, exterior):
                (next_obs, reward, term, trunc, info), elapsed = budget.step(fork, action, 'second_instrument_probe')
                tail_rows.append(dict(action=action, status=info['outcome']['status'], reason=info['outcome'].get('reason'),
                                      reward=reward, wall_seconds=elapsed,
                                      debris_removed_mm3=info['outcome'].get('metrics', {}).get('debrisRemovedMm3', 0.)))
                if term or trunc or info['outcome']['status'] != 'accepted': break
            supported = len(tail_rows) == 4 and all(r['status'] == 'accepted' for r in tail_rows)
            supported = supported and tail_rows[-1]['debris_removed_mm3'] > 0
            if array_digest(env.scene.labels) != labels_hash: raise RuntimeError('SOURCE_LABELS_CHANGED')
            write_json(output / f'{name}-prefix.json', prefix)
            document['cases'][name] = dict(scene_id=env.scene.identity, config=asdict(env.config),
                source_label_sha256=labels_hash, prefix_snapshot=f'{name}-prefix.json',
                prefix_sha256=file_hash(output / f'{name}-prefix.json'), prefix_actions=rows,
                exterior_pose=exterior, second_instrument_supported=bool(supported), second_instrument_probe=tail_rows,
                prefix_metrics=env.measurements, observation_sha256=observation_hash(obs),
                local_forward_probe=dict(metrics=probe_info['metrics'], wall_seconds=probe_elapsed,
                                         status=probe_info['outcome']['status']),
                cached_fork_vs_full_replay=verification, critical_coverage=env.protection.report())
        if document['source_hashes'] != source_hashes():
            raise RuntimeError('SOURCE_CHANGED_DURING_PREPARATION; prepare after shared edits settle')
        document['status'] = 'prepared_awaiting_desktop_and_guard_gate'
        document['estimated_run_bound_seconds'] = SETTINGS['max_wall_seconds']
    except Exception as error:
        document['status'] = 'blocked'; document['error'] = str(error)
        raise
    finally:
        document['cost'] = budget.report()
        history_path = output / 'preparation-history.json'
        history = json.loads(history_path.read_text()) if history_path.exists() else []
        previous_path = output / 'preparation.json'
        if not history and previous_path.exists():
            previous = json.loads(previous_path.read_text())
            history.append(dict(status=previous['status'], cost=previous['cost'],
                                source_hashes=previous['source_hashes']))
        history.append(dict(status=document['status'], cost=document['cost'], source_hashes=document['source_hashes']))
        write_json(history_path, history)
        document['cumulative_preparation_cost'] = dict(attempts=len(history),
            transition_calls=sum(r['cost']['transition_calls'] for r in history),
            wall_seconds=sum(r['cost']['wall_seconds'] for r in history))
        document['preparation_sha256'] = digest(document)
        write_json(output / 'preparation.json', document)
    return document


def refuse_repeat_fit(output):
    result = output / 'result.json'
    if result.exists() and json.loads(result.read_text()).get('optimizer_updates', 0):
        raise ValueError('This declared run has already updated parameters; no implicit repeat fitting or sweep')


def summarize(episodes):
    """Report every seed and paired delta; do not select a winning checkpoint."""
    assessed = [r for r in episodes if r['method'] != 'training']
    methods = ('untrained', 'random', 'scripted', 'observed_greedy_local', 'trained')
    summary = {}
    for name in SOURCES:
        table = {}
        by_method = {method: {r['seed']: r for r in assessed if r['case'] == name and r['method'] == method}
                     for method in methods}
        for method, rows in by_method.items():
            table[method] = dict(seeds=sorted(rows), count=len(rows),
                mean_local_return=float(np.mean([r['local_return'] for r in rows.values()])) if rows else None,
                mean_cusa_return=float(np.mean([r['cusa_return'] for r in rows.values()])) if rows else None,
                mean_local_target_mm3=float(np.mean([r['local_metrics']['tumor_removed_mm3'] for r in rows.values()])) if rows else None,
                mean_local_healthy_mm3=float(np.mean([r['local_metrics']['healthy_removed_mm3'] for r in rows.values()])) if rows else None,
                stops={str(seed): r['stop_reason'] for seed, r in rows.items()})
        differences = {}
        for method in methods[:-1]:
            differences[method] = [dict(seed=seed,
                cusa_return=trained['cusa_return'] - by_method[method][seed]['cusa_return'],
                target_mm3=trained['local_metrics']['tumor_removed_mm3'] - by_method[method][seed]['local_metrics']['tumor_removed_mm3'],
                healthy_mm3=trained['local_metrics']['healthy_removed_mm3'] - by_method[method][seed]['local_metrics']['healthy_removed_mm3'])
                for seed, trained in by_method['trained'].items() if seed in by_method[method]]
        summary[name] = dict(methods=table, paired_trained_minus_baseline=differences,
                             meaning='three development seeds; descriptive only, no held-out or statistical claim')
    return summary


def validate_launch(preparation, gate):
    expected = copy.deepcopy(preparation); claimed = expected.pop('preparation_sha256')
    if digest(expected) != claimed: raise ValueError('Preparation integrity differs')
    if preparation['status'] != 'prepared_awaiting_desktop_and_guard_gate': raise ValueError('Preparation not ready')
    if preparation['settings'] != SETTINGS or preparation['source_hashes'] != source_hashes():
        raise ValueError('Source, settings or shared engine changed; prepare again before launching')
    if gate.get('preparation_sha256') != claimed or gate.get('authorized') is not True:
        raise ValueError('Separate lead-owned launch receipt required')
    for field in ('desktop_sub06_demonstrated', 'desktop_ucsf0004_demonstrated',
                  'shared_guards_verified', 'performance_and_telemetry_verified'):
        if gate.get(field) is not True: raise ValueError('Launch prerequisite missing: ' + field)


def train(output, gate_path):
    refuse_repeat_fit(output)
    preparation = json.loads((output / 'preparation.json').read_text())
    gate = json.loads(Path(gate_path).read_text())
    validate_launch(preparation, gate)
    result = dict(schema=SCHEMA, status='running', preparation_sha256=preparation['preparation_sha256'],
                  settings=SETTINGS, limitations=LIMITATIONS, updates=[], episodes=[], final_replay=[],
                  setup_cost=preparation['cumulative_preparation_cost'], optimizer_updates=0)
    budget = Budget(); previous_threads = torch.get_num_threads(); torch.set_num_threads(1)
    try:
        prepared = {}
        for name, case in preparation['cases'].items():
            path = output / case['prefix_snapshot']
            if file_hash(path) != case['prefix_sha256']: raise ValueError('Prefix snapshot changed')
            env = load_source(name)
            _, obs = replay(env, json.loads(path.read_text()), budget)
            if observation_hash(obs) != case['observation_sha256']: raise ValueError('Prefix observation differs')
            prepared[name] = env, obs
        for seed in SETTINGS['seeds']:
            torch.manual_seed(seed); model = Policy()
            initial_hash = parameter_hash(model)
            optimizer = torch.optim.Adam(model.parameters(), lr=SETTINGS['learning_rate'])
            generator = torch.Generator().manual_seed(seed)
            for method in ('untrained', 'random', 'scripted', 'observed_greedy_local'):
                for name, (env, obs) in prepared.items():
                    case = preparation['cases'][name]
                    _, row, _ = rollout(env, obs, method, seed, budget, model,
                                        exterior_pose=case['exterior_pose'] if case['second_instrument_supported'] else None)
                    result['episodes'].append(dict(case=name, **row))
            for update in range(SETTINGS['updates']):
                batch = []
                for name, (env, obs) in prepared.items():
                    samples, row, _ = rollout(env, obs, 'training', seed, budget, model, generator)
                    batch.append(samples); result['episodes'].append(dict(case=name, update=update, **row))
                stats = optimizer_update(model, optimizer, batch)
                result['updates'].append(dict(seed=seed, update=update, **stats))
                result['optimizer_updates'] += 1
                result['cost'] = budget.report(); write_json(output / 'result.json', result)
            checkpoint = output / f'policy-seed-{seed}.pt'
            torch.save(dict(state_dict=model.state_dict(), settings=SETTINGS, seed=seed,
                            initial_sha256=initial_hash, final_sha256=parameter_hash(model),
                            preparation_sha256=preparation['preparation_sha256']), checkpoint)
            for name, (env, obs) in prepared.items():
                case = preparation['cases'][name]
                _, row, snapshot = rollout(env, obs, 'trained', seed, budget, model,
                                           exterior_pose=case['exterior_pose'] if case['second_instrument_supported'] else None)
                result['episodes'].append(dict(case=name, **row))
                path = output / f'{name}-seed-{seed}-final-replay.json'; write_json(path, snapshot)
                verified, _ = replay(load_source(name), snapshot, budget)
                result['final_replay'].append(dict(case=name, seed=seed, path=path.name, **verified))
        if preparation['source_hashes'] != source_hashes(): raise RuntimeError('SOURCE_CHANGED_DURING_RUN')
        result['status'] = 'completed'
    except Exception as error:
        result['status'] = 'blocked'; result['error'] = str(error)
        raise
    finally:
        result['cost'] = budget.report()
        result['cumulative_cost'] = dict(
            wall_seconds=result['setup_cost']['wall_seconds'] + result['cost']['wall_seconds'],
            transition_calls=result['setup_cost']['transition_calls'] + result['cost']['transition_calls'],
            scope='all recorded runner preparation attempts and this fit/assessment/replay run; earlier manual exploration is separate')
        result['summary'] = summarize(result['episodes'])
        write_json(output / 'result.json', result)
        torch.set_num_threads(previous_threads)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('prepare', 'train'), default='prepare')
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/surgical-pilot-v1')
    parser.add_argument('--launch-gate', type=Path)
    args = parser.parse_args()
    if args.mode == 'train' and not args.launch_gate: parser.error('--launch-gate is required for training')
    result = prepare(args.output) if args.mode == 'prepare' else train(args.output, args.launch_gate)
    print(json.dumps({k: result[k] for k in ('status', 'optimizer_updates', 'cost')}, indent=2))


if __name__ == '__main__': main()
