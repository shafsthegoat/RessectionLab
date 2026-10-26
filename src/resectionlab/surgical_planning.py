"""Bounded observation-only continuations in the existing surgical workbench.

This is an explicitly opted-in development comparison, not TRAIN admission,
whole-target planning, retained native search, or clinical validation. A frozen
failed learner remains useful for inspecting why its actual choices fail.
"""
from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import time

import numpy as np

from .core import array_digest, semantic_digest
from .surgical_observed_policy import ACTION_IDS, candidates, observed_greedy
from .surgical_rl_rewards import RewardTracker, RewardConfig, compute_reward

SCHEMA = 'resectionlab.surgical-planning.v1'
ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT = ROOT / 'outputs/surgical-pilot-v1/policy-seed-17.pt'
CHECKPOINT_SHA256 = '554c3c743dd8ede2b3aa9f630ca25e7977abf864d38402b7b50fa5959455ebba'
PARAMETER_SHA256 = '29382995eb3aee03af3be77a3eae10fa1aaed414b6b4834770c1c8cf010dc30f'
SOURCE_SCENES = {
    'sha256:e292ba41b650b221c90fb582bae49cd069597b119f9635e3f881b951e1138c56': 'meningioma sub-06 · development',
    'sha256:e32dca89c598eeb2ee4d5b3808ef00e9ed9bee35624d7ce2bcaf6b86edcfb225': 'UCSF-PDGM-0004 glioma · development',
}
FROZEN_SCENES = frozenset(SOURCE_SCENES)
SOURCE_CONTRACTS = {
    'sha256:e292ba41b650b221c90fb582bae49cd069597b119f9635e3f881b951e1138c56': 'sha256:c90de8fa14c77f2d2623b4e2e0070cefde8ea29e28189ba18d41865287bd6025',
    'sha256:e32dca89c598eeb2ee4d5b3808ef00e9ed9bee35624d7ce2bcaf6b86edcfb225': 'sha256:2900f709c06ac759fe7666b817edd9133602067e7e6aecd61a75e359f3391b29',
}
INPUT_CODE = {
    'surgical_observed_policy.py': 'fc53cd09a3ca87392e4bc7daea74491f7185369e3042fb26fbf2dfb869a53842',
    'surgical_rl_env.py': '01ed51dfa6a4bb9a889904a6aa377dc0a4b86c2a0bbef8477a3482b0226d2251',
}
DECISIONS = 6
HEALTHY_LIMIT_MM3 = 5.


def validate_saved_comparison(scene, record, comparison):
    """Bind stored receipts to the immutable branches whose replay they checked.

    This is content-integrity checking, not cryptographic author authentication.
    Fresh engine replay still checks the actual state when a branch is opened.
    """
    document = dict(comparison)
    seal = document.pop('comparisonSeal', None)
    if (document.get('schema') != SCHEMA or semantic_digest(document) != seal or
            document.get('sourceSceneId') != scene.identity or
            document.get('sourceLabelsHash') != array_digest(scene.labels)):
        raise ValueError('Saved research comparison source or receipt integrity differs')
    alternatives = document.get('alternatives', [])
    task = document.get('task')
    expected_count = 3 if task == 'vascular-rescue' else 2
    if task == 'anatomy-access':
        expected_count = len(alternatives)
        if (document.get('requiredAlternatives') != 3 or not 0 <= expected_count <= 3
                or document.get('status') not in ('three_executable_source_approaches',
                    'three_source_tissue_routes_skull_unsupported', 'insufficient_distinct_routes')):
            raise ValueError('Saved access comparison has an invalid completion contract')
        complete = document['status'] in ('three_executable_source_approaches',
                                           'three_source_tissue_routes_skull_unsupported')
        if complete != (expected_count == 3):
            raise ValueError('Saved access comparison completion differs from its branches')
        methods = {2: 'superior-access', 0: 'axis0-access', 1: 'axis1-access'}
        skull = any(r['material'] == 'bone' and np.any(scene.labels == r['id'])
                    for r in scene.regions)
        source_layers = (scene.source.get('adapter') or {}).get('sourceLayers', {})
        seen_axes = set()
        for plan in alternatives:
            route = plan.get('route', {})
            axis = route.get('nativeAxis')
            if (type(axis) is not int or axis not in methods or axis in seen_axes or
                    plan.get('method') != methods[axis] or
                    route.get('sourceSceneId') != scene.identity or
                    plan.get('prefixActions') != 0 or
                    route.get('accessCompleteness') not in ('source_skull_and_target',
                        'unsupported_skull_access_source_tissue_only',
                        'unsupported_skull_access_brain_corridor') or
                    (route.get('accessCompleteness') == 'unsupported_skull_access_brain_corridor'
                     and document.get('allowModeledBrainCorridor') is not True)):
                raise ValueError('Saved access comparison route binding differs')
            if (skull and route['accessCompleteness'] != 'source_skull_and_target') or (
                    not skull and route['accessCompleteness'] == 'source_skull_and_target'):
                raise ValueError('Saved access comparison skull evidence differs')
            if route['accessCompleteness'] == 'unsupported_skull_access_brain_corridor':
                named = {r['name']: r for r in scene.regions if r['material'] == 'soft_tissue'
                         and (r['name'].strip().lower() in ('brain', 'whole brain') or
                              r['name'].strip().lower().startswith('mapped brain'))}
                evidence = route.get('sourceBrainLayers')
                if not evidence or {row.get('regionName') for row in evidence} != set(named):
                    raise ValueError('Saved access comparison brain layer evidence differs')
                for row in evidence:
                    source = source_layers.get(row['regionName'], {})
                    if (row.get('layerId') != source.get('layerId') or
                            row.get('reviewStatus') != source.get('reviewStatus', 'not supplied')):
                        raise ValueError('Saved access comparison brain provenance differs')
            seen_axes.add(axis)
        failures = document.get('directionFailures')
        if (type(failures) is not list or len(failures) != 3-expected_count or
                {row.get('nativeAxis') for row in failures} != set(methods)-seen_axes):
            raise ValueError('Saved access comparison missing-direction record differs')
    if len(alternatives) != expected_count or len({p.get('branchId') for p in alternatives}) != expected_count:
        raise ValueError('Saved research comparison requires distinct task branches')
    if document.get('task') == 'vascular-rescue' and {p.get('method') for p in alternatives} != {'prompt-coagulation','suction-first','recorded-delay'}:
        raise ValueError('Saved rescue comparison strategies differ')
    prefixes = []
    for plan in alternatives:
        branch = record['branches'].get(plan.get('branchId'))
        if (branch is None or branch.get('sceneId') != scene.identity or
                branch.get('stateId') != plan.get('finalStateId') or
                plan.get('startStateId') != document['startStateId']):
            raise ValueError('Saved research comparison branch identity differs')
        count = plan['prefixActions']
        if type(count) is not int or not 0 <= count <= len(branch['actions']):
            raise ValueError('Saved research comparison prefix differs')
        rows = plan['decisions']
        if branch['actions'][count:] != [r['action'] for r in rows if r['status'] != 'rejected']:
            raise ValueError('Saved research comparison decisions differ from native history')
        expected = semantic_digest({'sourceSceneId': scene.identity, 'startStateId': document['startStateId'], 'decisions': rows})
        if plan.get('planSeal') != expected:
            raise ValueError('Saved research comparison decision seal differs')
        if (branch.get('protectionContract') or {}).get('contentHash') != document['guardContractId']:
            raise ValueError('Saved research comparison exclusion contract differs')
        prefixes.append(branch['actions'][:count])
    if any(prefix != prefixes[0] for prefix in prefixes[1:]):
        raise ValueError('Saved alternatives do not share the same native source prefix')


def availability(engine):
    """Fail visibly before execution; unknown coverage never becomes complete."""
    reasons = []
    if importlib.util.find_spec('gymnasium') is None:
        reasons.append('The local observation runtime requires the installed RL extra.')
    if engine.scene.identity not in SOURCE_SCENES:
        reasons.append('This source has no admission for the development continuation comparison.')
    if not engine._protection_contract:
        reasons.append('Bind the source exclusion contract before planning.')
    guard_report = engine.guard_report()
    if guard_report['missingRequired'] or not guard_report['structures']:
        reasons.append('Required source exclusion masks are missing: ' + ', '.join(guard_report['missingRequired']))
    if guard_report['contractId'] != SOURCE_CONTRACTS.get(engine.scene.identity):
        reasons.append('The exclusion contract differs from the admitted source evidence.')
    if not any(r.get('target') and r['material'] == 'tumor' and
               np.any(engine.scene.labels == r['id']) for r in engine.scene.regions):
        reasons.append('No supplied tumor target is available.')
    left = engine.state['slots']['left']
    if left['instrument'] != 'cusa' or not left['inserted']:
        reasons.append('Prepare a checked post-exposure approach with left-hand CUSA first.')
    checkpoint_reason = None
    if engine.scene.identity not in FROZEN_SCENES:
        checkpoint_reason = 'The frozen learner is limited to its two exact development source scenes.'
    elif not CHECKPOINT.is_file():
        checkpoint_reason = 'The hash-bound seed-17 checkpoint is unavailable on this computer.'
    return {'available': not reasons, 'reasons': reasons,
            'checkpointAvailable': not reasons and checkpoint_reason is None,
            'checkpointReason': checkpoint_reason,
            'sourceRole': SOURCE_SCENES.get(engine.scene.identity, 'not admitted for this frozen learner'),
            'unknown': ['complete vessel coverage', 'functional cortex and tracts', 'clinical injury'],
            'protection': guard_report,
            'scope': 'Six left-CUSA decisions from the current post-exposure state; no training runs.'}


def _parameter_hash(model):
    digest = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        digest.update(key.encode()); digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def load_frozen_policy(scene_id):
    """No renderer-selected paths, pickle objects, refitting or best-seed choice."""
    import torch
    if scene_id not in FROZEN_SCENES:
        raise ValueError('Frozen checkpoint source scene differs')
    if not CHECKPOINT.is_file() or hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest() != CHECKPOINT_SHA256:
        raise ValueError('Frozen checkpoint is unavailable or its bytes changed')
    for name, expected in INPUT_CODE.items():
        if hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() != expected:
            raise ValueError('Frozen learner observation/action implementation changed: ' + name)
    record = torch.load(CHECKPOINT, map_location='cpu', weights_only=True)
    with torch.random.fork_rng(devices=[]):
        model = torch.nn.Sequential(torch.nn.Linear(20, 32), torch.nn.Tanh(), torch.nn.Linear(32, 1))
    model.load_state_dict({key.removeprefix('net.'): value for key, value in record['state_dict'].items()}, strict=True)
    # Hash the original named architecture, exactly as the training runner did.
    class NamedPolicy(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.net = model
    if (record['seed'] != 17 or record['final_sha256'] != PARAMETER_SHA256 or
            _parameter_hash(NamedPolicy()) != PARAMETER_SHA256):
        raise ValueError('Frozen parameter identity differs')
    model.eval()
    def score(features):
        with torch.no_grad():
            return model(torch.as_tensor(features, dtype=torch.float32)).squeeze(-1).numpy()
    return score


def _observer(engine, tracker):
    # Reuse the deployed local sampler without resetting/injecting tissue or
    # replaying through different RL placement/termination semantics. Only the
    # returned observation crosses the action-selection boundary.
    from .surgical_rl_env import SurgicalRLEnv, RLConfig
    guard = engine.guard_config
    env = SurgicalRLEnv(engine.scene, protection=engine.protection, config=RLConfig(
        require_critical_coverage=guard.require_critical_coverage,
        critical_max_displacement_mm=guard.critical_max_displacement_mm,
        local_spacing_mm=1.))
    env.engine = engine
    env.measurements = tracker.measure(engine, _protection_measurements(engine))
    env.attempts = []; env.invalid_actions = 0
    return env._observe(), env.measurements


def _protection_measurements(engine):
    result = {'protected_damage_mm3': engine.protection.damage_mm3(engine.state['removed'])}
    if hasattr(engine, 'rl_near_exposure'):
        result['near_contact_exposure_mm3_seconds'] = engine.rl_near_exposure
    if hasattr(engine, 'rl_strain_peaks'):
        result['deformation_cell_peaks'] = engine.rl_strain_peaks
    return result


def select_observed(observation, method, scorer=None):
    """The complete selector interface: no engine, previews, masks or info."""
    features, options = candidates(observation)
    if method == 'greedy':
        selected, scores = observed_greedy(features)
    elif method == 'stop':
        selected, scores = 8, [0.] * 8 + [1.]
    elif method == 'checkpoint' and scorer is not None:
        scores = np.asarray(scorer(features), float)
        if scores.shape != (len(ACTION_IDS),) or not np.isfinite(scores).all():
            raise ValueError('Invalid frozen actor logits')
        selected, scores = int(scores.argmax()), scores.tolist()
    else:
        raise ValueError('Unknown observation-only method')
    return selected, options[selected], scores


def _run(base, method, scorer, check):
    engine = base.clone()
    tracker = RewardTracker(engine, engine._protected_ids)
    observation, initial = _observer(engine, tracker)
    rows = []; total_reward = 0.; reason = 'decision_limit'; truncated = True
    started = time.perf_counter()
    reward_config = RewardConfig()
    for index in range(DECISIONS):
        check()
        selected, action, scores = select_observed(observation, method, scorer)
        observation_hash = semantic_digest({key: array_digest(value) for key, value in observation.items()})
        before = tracker.measure(engine, _protection_measurements(engine))
        outcome = engine.step(action, check=check)
        observation, after = _observer(engine, tracker)
        reward, terms = compute_reward(before, after, outcome, reward_config)
        total_reward += reward
        rows.append({'index': index, 'actionId': ACTION_IDS[selected], 'action': action,
                     'scores': scores, 'observationHash': observation_hash,
                     'status': outcome['status'], 'reason': outcome.get('reason'),
                     'stateId': engine.state_id, 'reward': reward, 'rewardTerms': terms})
        if (after['healthy_removed_mm3'] - initial['healthy_removed_mm3'] > HEALTHY_LIMIT_MM3 or
                any(after[k] > initial[k] for k in ('vessel_damage_mm2', 'protected_damage_mm3', 'bleeding_generated_mm3'))):
            reason = 'constraint_budget_failure'; truncated = False; break
        if selected == 8:
            reason = 'policy_stop'; truncated = False; break
        if outcome['status'] == 'rejected':
            reason = 'exact_execution_rejected'; truncated = False; break
    planning_seconds = time.perf_counter() - started
    # Seal choices before a separate fresh-engine deterministic replay. No replay
    # outcome is returned to a selector or used to choose a different action.
    seal = semantic_digest({'sourceSceneId': base.scene.identity, 'startStateId': base.state_id, 'decisions': rows})
    replay_started = time.perf_counter()
    replay = base.fresh()
    replay_calls = 0
    for action in base.actions:
        check(); result = replay.step(action, check=check); replay_calls += 1
        if result['status'] == 'rejected': raise ValueError('Source prefix replay rejected')
    if replay.state_id != base.state_id:
        raise ValueError('Source prefix replay identity differs')
    replay_tracker = RewardTracker(replay, replay._protected_ids)
    replay_tracker.measure(replay, _protection_measurements(replay))
    for row in rows:
        check(); result = replay.step(row['action'], check=check); replay_calls += 1
        if result['status'] != row['status'] or result.get('reason') != row['reason'] or replay.state_id != row['stateId']:
            raise ValueError('Sealed continuation replay differs')
        replay_tracker.measure(replay, _protection_measurements(replay))
    if replay.state != engine.state:
        raise ValueError('Replayed final anatomy differs')
    final = tracker.measure(engine, _protection_measurements(engine))
    independently_measured = replay_tracker.measure(replay, _protection_measurements(replay))
    if independently_measured != final:
        raise ValueError('Replayed endpoint vector differs')
    unknown_endpoints = []
    if not hasattr(engine, 'rl_near_exposure'):
        unknown_endpoints.append('near_contact_exposure_mm3_seconds')
    if not hasattr(engine, 'rl_strain_peaks'):
        unknown_endpoints.append('deformation_injury_mm3')
    return engine, {
        'method': method, 'checkpointSha256': CHECKPOINT_SHA256 if method == 'checkpoint' else None,
        'startStateId': base.state_id, 'finalStateId': engine.state_id, 'prefixActions': len(base.actions),
        'decisions': rows, 'stopReason': reason, 'truncated': truncated,
        'earlyStop': reason == 'policy_stop' and final['tumor_removed_mm3'] == initial['tumor_removed_mm3'],
        'localReward': total_reward,
        'cumulativeEndpoints': {key: None if key in unknown_endpoints else value for key, value in final.items()},
        'incrementalEndpoints': {key: None if key in unknown_endpoints else value - initial[key] for key, value in final.items()},
        'unassessedEndpoints': unknown_endpoints,
        'fullTargetFractionRemoved': engine.metrics()['targetRemovedMm3'] / float(sum(
            np.count_nonzero(engine.scene.labels == r['id']) * engine.scene.voxel_mm3
            for r in engine.scene.regions if r.get('target') and r['material'] == 'tumor')),
        'reachableTargetFractionRemoved': None, 'neurologicalHarm': 'unknown; no calibrated outcome model',
        'invalidActions': sum(row['status'] == 'rejected' for row in rows),
        'travelMm': engine.state['travelMm'] - base.state['travelMm'],
        'instrumentChanges': 0, 'planSeal': seal,
        'cost': {'planningSeconds': planning_seconds, 'freshReplaySeconds': time.perf_counter() - replay_started,
                 'executionCalls': len(rows), 'freshReplayCalls': replay_calls, 'previewCalls': 0, 'optimizerUpdates': 0},
        'verification': {'exactStateMatch': True, 'endpointVectorMatch': True,
                         'scope': 'fresh deterministic replay in the same shared engine; not external physical validation'},
    }


def compare_continuations(engine, *, acknowledge_unknowns=False, opponent='stop', check=lambda: None, progress=lambda *_: None):
    """Build alternatives atomically; leave the live source branch untouched."""
    started = time.perf_counter()
    cancellation = check
    def check():
        cancellation()
        if time.perf_counter() - started > 100:
            raise ValueError('Research planning exceeded its 100-second budget; no branches installed')
    if acknowledge_unknowns is not True:
        raise ValueError('Explicitly acknowledge missing anatomy and uncalibrated development simulation')
    readiness = availability(engine)
    if not readiness['available']:
        raise ValueError('Research planning refused: ' + ' '.join(readiness['reasons']))
    if opponent not in ('stop', 'checkpoint'):
        raise ValueError('Choose STOP or the pinned development checkpoint')
    if opponent == 'checkpoint' and not readiness['checkpointAvailable']:
        raise ValueError(readiness['checkpointReason'])
    scorer = load_frozen_policy(engine.scene.identity) if opponent == 'checkpoint' else None
    source_hash = array_digest(engine.scene.labels)
    branches = []
    for index, method in enumerate(('greedy', opponent)):
        check(); progress(.1 + index * .4, 'Executing and checking ' + method + ' continuation')
        candidate, report = _run(engine, method, scorer, check)
        branches.append((candidate, report))
    if source_hash != array_digest(engine.scene.labels):
        raise ValueError('Source labels changed during planning')
    comparison = {'schema': SCHEMA, 'sourceSceneId': engine.scene.identity, 'startStateId': engine.state_id,
        'sourceRole': readiness['sourceRole'], 'sourceLabelsHash': source_hash,
        'guardContractId': engine.guard_report()['contractId'], 'availability': readiness,
        'totalWallSeconds': time.perf_counter() - started, 'optimizerUpdates': 0,
        'limits': {'decisions': DECISIONS, 'incrementalHealthyMm3': HEALTHY_LIMIT_MM3,
                   'budgetSemantics': 'checked after each action; within-action overshoot fails'},
        'limitations': ['Development continuation comparison; not admitted TRAIN, SELECT or EVAL evidence.',
            'Actions, rewards and tissue changes are simulated; images and supplied annotations are acquired inputs.',
            'Local observation-only greedy is not the retained native bounded-search planner.',
            'The seed-17 learner previously failed both development assessments and is not promoted.',
            'No whole-tumor/reachable-target success claim; missing vessel and functional coverage remains unknown.',
            'Inspection does not reveal new patient anatomy in this model.'],
        'alternatives': [report for _, report in branches]}
    return branches, comparison
