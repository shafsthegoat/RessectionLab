"""Observed-state high-level CUSA choices over checked native sector skills.

The actor sees only the detached published strategy DTO. The option executor
owns source-sector membership, previews and native action checks. This is a
bounded DEVELOPMENT experiment, not a clinical policy or held-out validation.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import resource
import time

import numpy as np
import torch

from .core import array_digest, semantic_digest
from .patient_cusa_learning import sha256
from .surgical_engine import SurgicalEngine
from .surgical_environment import load_environment
from .surgical_planning import validate_saved_comparison
from .surgical_sector_planning import (execute_strategy_option,
    intact_strategy_observation, source_sector_binding, strategy_observation)
from .workspace_bundle import load_workspace


SCHEMA = 'resectionlab.patient-sector-learning.v1'
OBSERVATION_SCHEMA = 'published-surgical-rl-v2-plus-sector-ledger-v1'
ACTION_IDS = ('corridor:left', 'corridor:right', 'near:left', 'near:right',
              'far:left', 'far:right', 'STOP')
DEVELOPMENT_ACCESS_METHOD = 'superior-access'
MAX_NATIVE_ACTIONS = 48
MAX_HIGH_LEVEL_DECISIONS = 5
MAX_ROLLOUT_SECONDS = 120.
MAX_ROLLOUT_RSS_BYTES = 3 * 1024**3

_OBS_KEYS = frozenset(('local', 'slots', 'tool_parameters', 'metrics', 'goal',
                       'coverage', 'local_spacing_mm', 'cleanup', 'budgets'))
_LEDGER_KEYS = frozenset(('additionalTargetMm3', 'additionalNonTargetMm3',
                          'additionalPerSectorRemovedMm3', 'remainingNativeActions',
                          'activeHand', 'thresholds'))
_THRESHOLD_KEYS = frozenset(('additionalTargetMm3', 'eachSectorMm3',
                             'additionalNonTargetMm3', 'totalNonTargetMm3'))
_OPTION_TYPES = ('corridor', 'near', 'far', 'STOP')
_OPTION_WIDTH = 3 + 1 + 2 + len(_OPTION_TYPES) + 1
_GLOBAL_WIDTH = 24 + 14 + 22 + 4 + 3 + 3 + 3 + 1 + 5


@dataclass(frozen=True)
class PolicyInput:
    """Only actor-visible current sensors and candidate descriptors."""
    local: np.ndarray
    global_features: np.ndarray
    options: np.ndarray
    available: np.ndarray
    option_ids: tuple[str, ...]
    boundary: str


def _finite(value, shape, name):
    array = np.asarray(value, dtype=np.float32)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError('Invalid published strategy '+name)
    return np.array(array, copy=True)


def policy_input(dto) -> PolicyInput:
    """Exact whitelist: no engine, source arrays, future preview, state ID or history."""
    if not isinstance(dto, dict) or dto.get('boundary') not in (
            'intact_before_access', 'post_corridor'):
        raise ValueError('Unknown strategy observation boundary')
    intact = dto['boundary'] == 'intact_before_access'
    if set(dto) != ({'boundary', 'observation', 'options', 'remainingNativeActions'}
                    if intact else {'boundary', 'observation', 'options', 'ledger'}):
        raise ValueError('Strategy DTO contains unsupported actor fields')
    observed = dto['observation']
    if not isinstance(observed, dict) or set(observed) != _OBS_KEYS:
        raise ValueError('Strategy DTO differs from published surgical observation')
    local = _finite(observed['local'], (2, 8, 9, 9, 9), 'local field')
    slots = _finite(observed['slots'], (2, 12), 'slots')
    parameters = _finite(observed['tool_parameters'], (2, 11), 'tool parameters')
    metrics = _finite(observed['metrics'], (14,), 'metrics')
    goal = _finite(observed['goal'], (4,), 'goal')
    coverage = _finite(observed['coverage'], (3,), 'coverage')
    spacing = _finite(observed['local_spacing_mm'], (1,), 'spacing')
    cleanup = _finite(observed['cleanup'], (3,), 'cleanup')
    budgets = _finite(observed['budgets'], (3,), 'budgets')
    if intact:
        remaining = dto['remainingNativeActions']
        additional = [0., 0., 0., 0.]
    else:
        ledger = dto['ledger']
        if not isinstance(ledger, dict) or set(ledger) != _LEDGER_KEYS or \
                not isinstance(ledger['additionalPerSectorRemovedMm3'], dict) or \
                set(ledger['additionalPerSectorRemovedMm3']) != {'near', 'far'} or \
                not isinstance(ledger['thresholds'], dict) or \
                set(ledger['thresholds']) != _THRESHOLD_KEYS or \
                ledger['activeHand'] not in ('left', 'right'):
            raise ValueError('Strategy task ledger differs')
        remaining = ledger['remainingNativeActions']
        additional = [ledger['additionalTargetMm3']/30.,
                      ledger['additionalNonTargetMm3']/20.,
                      ledger['additionalPerSectorRemovedMm3']['near']/10.,
                      ledger['additionalPerSectorRemovedMm3']['far']/10.]
        if ledger['thresholds'] != {'additionalTargetMm3': 30.,
                'eachSectorMm3': 10., 'additionalNonTargetMm3': 20.,
                'totalNonTargetMm3': 200.}:
            raise ValueError('Strategy task thresholds changed')
    if not isinstance(remaining, int) or not 0 <= remaining <= MAX_NATIVE_ACTIONS:
        raise ValueError('Strategy native action budget differs')
    scalar = np.concatenate((np.tanh(slots.reshape(-1)/200.),
        np.tanh(metrics/200.), np.tanh(parameters.reshape(-1)/100.),
        np.r_[np.tanh(goal[:3]/200.), goal[3]], coverage,
        np.tanh(cleanup/100.), budgets, spacing/10.,
        np.asarray([*additional, remaining/MAX_NATIVE_ACTIONS], np.float32)))
    if scalar.shape != (_GLOBAL_WIDTH,) or not np.isfinite(scalar).all():
        raise ValueError('Strategy actor context is malformed')
    rows = dto['options']
    if not isinstance(rows, list) or len(rows) != (3 if intact else 5):
        raise ValueError('Strategy option inventory differs')
    expected = (ACTION_IDS[:2]+('STOP',) if intact else ACTION_IDS[2:])
    if tuple(row.get('id') for row in rows) != expected:
        raise ValueError('Strategy option identity/order differs')
    features, available = [], []
    for row in rows:
        option_id = row['id']
        kind = option_id.split(':')[0] if option_id != 'STOP' else 'STOP'
        hand = row.get('hand')
        if (not isinstance(row.get('structurallyAvailable'), bool) or
                hand != (None if option_id == 'STOP' else option_id.split(':')[1])):
            raise ValueError('Strategy option hand/availability differs')
        keys = ({'id', 'hand', 'entryRelativeToToolMm', 'structurallyAvailable'}
                if intact else {'id', 'sector', 'hand', 'centerRelativeToToolMm',
                                'structurallyAvailable'})
        if set(row) != keys or (not intact and row['sector'] !=
                (None if option_id == 'STOP' else kind)):
            raise ValueError('Strategy option contains unsupported actor fields')
        vector = row['entryRelativeToToolMm'] if intact else row['centerRelativeToToolMm']
        if vector is None:
            if option_id != 'STOP' and row['structurallyAvailable']:
                raise ValueError('Available sector has no declared center')
            vector = np.zeros(3, np.float32)
        vector = _finite(vector, (3,), 'candidate geometry')
        relation = np.tanh(vector/20.)
        distance = np.tanh(np.linalg.norm(vector)/20.)
        hand_onehot = [float(hand == 'left'), float(hand == 'right')]
        kind_onehot = [float(kind == label) for label in _OPTION_TYPES]
        features.append([*relation, distance, *hand_onehot, *kind_onehot,
                         float(row['structurallyAvailable'])])
        available.append(row['structurallyAvailable'])
    if not any(available):
        raise ValueError('Strategy has no structurally available option')
    return PolicyInput(local.reshape(16, 9, 9, 9), scalar,
                       np.asarray(features, np.float32),
                       np.asarray(available, bool), expected, dto['boundary'])


def input_digest(inp: PolicyInput):
    return semantic_digest({'schema': OBSERVATION_SCHEMA,
        'local': array_digest(inp.local), 'global': array_digest(inp.global_features),
        'options': array_digest(inp.options), 'available': inp.available.tolist(),
        'ids': inp.option_ids, 'boundary': inp.boundary})


class OptionActor(torch.nn.Module):
    """Shared option scorer; spatial voxels and candidate vectors both matter."""
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv3d(16, 8, 3, padding=1), torch.nn.ReLU(),
            torch.nn.Conv3d(8, 8, 3, padding=1), torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool3d((2, 2, 2)))
        self.actor = torch.nn.Sequential(
            torch.nn.Linear(64+_GLOBAL_WIDTH+_OPTION_WIDTH, 64),
            torch.nn.Tanh(), torch.nn.Linear(64, 1))

    def forward(self, inp: PolicyInput):
        local = torch.as_tensor(inp.local, dtype=torch.float32).unsqueeze(0)
        context = torch.cat((self.encoder(local).flatten(),
                             torch.as_tensor(inp.global_features, dtype=torch.float32)))
        rows = torch.cat((context.expand(len(inp.option_ids), -1),
                          torch.as_tensor(inp.options, dtype=torch.float32)), dim=1)
        logits = self.actor(rows).flatten()
        return logits.masked_fill(~torch.as_tensor(inp.available), -1.e9)


class OptionCritic(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv3d(16, 8, 3, padding=1), torch.nn.ReLU(),
            torch.nn.Conv3d(8, 8, 3, padding=1), torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool3d((2, 2, 2)))
        self.critic = torch.nn.Sequential(
            torch.nn.Linear(64+_GLOBAL_WIDTH, 64), torch.nn.Tanh(),
            torch.nn.Linear(64, 1))

    def forward(self, inp: PolicyInput):
        local = torch.as_tensor(inp.local, dtype=torch.float32).unsqueeze(0)
        context = torch.cat((self.encoder(local).flatten(),
                             torch.as_tensor(inp.global_features, dtype=torch.float32)))
        return self.critic(context).squeeze()


def parameter_hash(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def actor_choice(actor, *, sample=False, generator=None):
    def choose(inp):
        with torch.no_grad():
            logits = actor(inp)
            selected = (int(torch.multinomial(logits.softmax(0), 1,
                            generator=generator)) if sample else int(logits.argmax()))
        return selected, logits.tolist()
    return choose


@dataclass
class PreparedCase:
    source: Path
    source_sha256: str
    case_id: str
    scene: object
    intact: SurgicalEngine
    access: dict
    access_method: str
    sector_cells: dict
    sector_evidence: dict
    access_comparison_seal: str


def prepare_case(path, access_method=DEVELOPMENT_ACCESS_METHOD) -> PreparedCase:
    source = Path(path).resolve()
    case, artifacts, _ = load_workspace(source)
    scene = load_environment(source)
    record = artifacts.get('surgical_toolbox')
    if not isinstance(record, dict) or record.get('planning', {}).get('task') != 'anatomy-access':
        raise ValueError('Verified anatomy-access source required')
    validate_saved_comparison(scene, record, record['planning'])
    if not isinstance(access_method, str) or not access_method:
        raise ValueError('A saved access method is required')
    matches = [row for row in record['planning']['alternatives']
               if row['method'] == access_method]
    if len(matches) != 1:
        raise ValueError('Selected saved access route missing or ambiguous')
    access = matches[0]
    intact = SurgicalEngine.restore(scene, record['branches'][record['branchId']])
    if intact.actions or intact.state_id != access['startStateId']:
        raise ValueError('Source did not reopen at the common intact boundary')
    cells, evidence = source_sector_binding(scene, access['route'])
    if any(row['sourceVolumeMm3'] < 10. for row in evidence.values()):
        raise ValueError('Declared source sector lacks 10 mm3 capacity')
    return PreparedCase(source, sha256(source), case.case_id, scene, intact,
                        access, access_method, cells, evidence,
                        record['planning']['comparisonSeal'])


def _rss_bytes():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value * (1 if __import__('sys').platform == 'darwin' else 1024))


def _state_observation(current: SurgicalEngine, prepared: PreparedCase, binding):
    if binding is None:
        return intact_strategy_observation(current, prepared.access,
                                           max_actions=MAX_NATIVE_ACTIONS)
    return strategy_observation(current, prepared.sector_evidence,
        prepared.sector_cells, prefix_metrics=binding['prefixMetrics'],
        prefix_sector_removed_mm3=binding['prefixSectorRemovedMm3'],
        active_hand=binding['hand'], max_actions=MAX_NATIVE_ACTIONS)


def _option_call_cost(option):
    """Count retained native work and discarded escape probes separately."""
    attempts = option['attempts']
    appended = option['nativeActionsAppended']
    if not isinstance(attempts, list) or type(appended) is not int or appended < 0:
        raise ValueError('Malformed high-level option physical-call receipt')
    admitted_previews = sum('previewStatus' in row for row in attempts)
    probe = option.get('escapePreflight')
    if option['id'].startswith(('near:', 'far:')):
        if not isinstance(probe, dict) or not {
                'totalPreviewCalls', 'totalCommittedCalls', 'physicalCalls'} <= set(option):
            raise ValueError('Sector option lacks checked escape and call receipts')
    discarded_previews = 0 if probe is None else probe.get('previewCalls')
    discarded_commits = 0 if probe is None else probe.get('committedCalls')
    if any(type(value) is not int or value < 0 for value in
           (discarded_previews, discarded_commits)):
        raise ValueError('Malformed discarded escape physical-call count')
    previews = admitted_previews+discarded_previews
    commits = appended+discarded_commits
    if (option.get('totalPreviewCalls', previews) != previews or
            option.get('totalCommittedCalls', commits) != commits or
            option.get('physicalCalls', previews+commits) != previews+commits):
        raise ValueError('High-level option physical-call totals disagree')
    if probe is not None and option['status'] == 'refused' and \
            (attempts or appended or option['startStateId'] != option['finalStateId']):
        raise ValueError('Refused escape probe changed the accepted native prefix')
    return {'previewCalls': previews, 'committedCalls': commits,
            'discardedProbePreviewCalls': discarded_previews,
            'discardedProbeCommittedCalls': discarded_commits}


def rollout(prepared: PreparedCase, choose, *, method, max_wall_seconds=MAX_ROLLOUT_SECONDS):
    """Choose fresh options from current sensors and verify the full native replay."""
    started = time.perf_counter()
    def check():
        if time.perf_counter()-started > max_wall_seconds:
            raise TimeoutError('Hierarchical option rollout wall budget exhausted')
        if _rss_bytes() > MAX_ROLLOUT_RSS_BYTES:
            raise MemoryError('Hierarchical option rollout memory budget exhausted')
    current = prepared.intact.clone()
    binding = None
    decisions, training = [], []
    preview_calls = commit_calls = 0
    discarded_preview_calls = discarded_commit_calls = 0
    forced_cleanup_resource_penalty = 0.
    stop_selected = False
    failure = None
    admitted_sectors = 0
    for index in range(MAX_HIGH_LEVEL_DECISIONS):
        check()
        dto = _state_observation(current, prepared, binding)
        inp = policy_input(dto)
        selected, scores = choose(inp)
        if type(selected) is not int or not 0 <= selected < len(inp.option_ids) or \
                not inp.available[selected] or len(scores) != len(inp.option_ids) or \
                not np.isfinite(np.asarray(scores, float)).all():
            raise ValueError('Actor selected an invalid current observed option')
        option_id = inp.option_ids[selected]
        before = current.metrics()
        next_engine, option = execute_strategy_option(current, prepared.access,
            option_id, corridor_binding=binding, max_actions=MAX_NATIVE_ACTIONS,
            check=check)
        physical = _option_call_cost(option)
        option_previews = physical['previewCalls']
        option_commits = physical['committedCalls']
        preview_calls += option_previews
        commit_calls += option_commits
        discarded_preview_calls += physical['discardedProbePreviewCalls']
        discarded_commit_calls += physical['discardedProbeCommittedCalls']
        if next_engine.actions[:len(current.actions)] != current.actions:
            raise ValueError('High-level option rewrote an executed native prefix')
        current = next_engine
        if option_id.startswith('corridor:') and option['status'] == 'accepted':
            binding = option['corridorBinding']
        if option_id.startswith(('near:', 'far:')) and option['status'] == 'accepted':
            admitted_sectors += 1
        target_gain = current.metrics()['targetRemovedMm3']-before['targetRemovedMm3']
        healthy_gain = current.metrics()['nonTargetRemovedMm3']-before['nonTargetRemovedMm3']
        if option_id.startswith('corridor:'):
            # Access exposure is bounded by the intact-source 200 mm3 cap.
            # The 20 mm3 continuation cap starts only after this prefix.
            reward = (-max(healthy_gain, 0.)/200. -
                      .002*(option_previews+option_commits))
        else:
            reward = (target_gain/30. - healthy_gain/20. -
                      .002*(option_previews+option_commits))
        if option['status'] != 'accepted':
            reward -= 1.
        if option_id == 'STOP':
            stop_selected = True
        decisions.append({'index': index, 'boundary': inp.boundary,
                          'observationHash': input_digest(inp),
                          'optionId': option_id, 'scores': list(map(float, scores)),
                          'reward': float(reward), 'physicalCallCost': physical,
                          'nativeOption': option,
                          'stateId': current.state_id})
        training.append((inp, selected, float(reward)))
        if option['status'] != 'accepted':
            failure = option['reason'] or 'option_refused'
            break
        if stop_selected:
            break
    if not stop_selected and binding is not None and current.state['slots'][binding['hand']]['inserted']:
        # Checked cleanup is still attempted, but never upgrades an omitted STOP.
        previous = current
        current, cleanup = execute_strategy_option(current, prepared.access, 'STOP',
            corridor_binding=binding, max_actions=MAX_NATIVE_ACTIONS, check=check)
        cleanup_physical = _option_call_cost(cleanup)
        preview_calls += cleanup_physical['previewCalls']
        commit_calls += cleanup_physical['committedCalls']
        discarded_preview_calls += cleanup_physical['discardedProbePreviewCalls']
        discarded_commit_calls += cleanup_physical['discardedProbeCommittedCalls']
        forced_cleanup_resource_penalty = .002*(
            cleanup_physical['previewCalls']+cleanup_physical['committedCalls'])
        if current.actions[:len(previous.actions)] != previous.actions:
            raise ValueError('Forced checked cleanup rewrote the policy trajectory')
        decisions.append({'index': len(decisions), 'boundary': 'forced_cleanup',
                          'optionId': 'STOP', 'nativeOption': cleanup,
                          'physicalCallCost': cleanup_physical,
                          'resourcePenalty': forced_cleanup_resource_penalty,
                          'stateId': current.state_id})
    if binding is None:
        delta_target = delta_non_target = 0.
        sectors = {'near': 0., 'far': 0.}
    else:
        prefix_metrics = binding['prefixMetrics']
        metrics = current.metrics()
        delta_target = metrics['targetRemovedMm3']-prefix_metrics['targetRemovedMm3']
        delta_non_target = metrics['nonTargetRemovedMm3']-prefix_metrics['nonTargetRemovedMm3']
        removed = set(current.state['removed'])
        sectors = {name: len(removed & prepared.sector_cells[name])*prepared.scene.voxel_mm3-
                   binding['prefixSectorRemovedMm3'][name] for name in ('near', 'far')}
    total_non_target = (current.metrics()['nonTargetRemovedMm3']-
                        prepared.intact.metrics()['nonTargetRemovedMm3'])
    cleanup_complete = decisions[-1]['nativeOption']['cleanupComplete'] if decisions else False
    gate = (stop_selected and failure is None and cleanup_complete and
            delta_target >= 30.-1e-9 and delta_non_target <= 20.+1e-9 and
            total_non_target <= 200.+1e-9 and admitted_sectors >= 2 and
            all(value >= 10.-1e-9 for value in sectors.values()) and
            len(current.actions) <= MAX_NATIVE_ACTIONS)
    status = 'success' if gate else 'abstained' if stop_selected and cleanup_complete else 'incomplete'
    terminal_reward = 4. if gate else -2.
    if training:
        inp, selected, reward = training[-1]
        # A checked two-sector completion must outrank intact abstention.
        training[-1] = (inp, selected,
                        reward+terminal_reward-forced_cleanup_resource_penalty)
        for decision, (_, _, actual_reward) in zip(decisions, training):
            decision['trainingReward'] = actual_reward
    check()
    replay = SurgicalEngine.restore(prepared.scene, current.snapshot(), check=check)
    if replay.state_id != current.state_id or replay.state != current.state:
        raise ValueError('Independent full native strategy replay differs')
    report = {'method': method, 'status': status, 'taskGateMet': gate,
              'stopSelected': stop_selected, 'stopReason': failure or
                  ('policy_stop' if stop_selected else 'decision_limit'),
              'deltaTargetMm3': float(delta_target),
              'deltaNonTargetMm3': float(delta_non_target),
              'totalNonTargetMm3': float(total_non_target),
              'perSectorRemovedMm3': sectors, 'admittedSectorOptions': admitted_sectors,
              'cleanupComplete': bool(cleanup_complete),
              'sourceSceneId': prepared.scene.identity,
              'sourceAccessPlanSeal': prepared.access['planSeal'],
              'sourceAccessMethod': prepared.access_method,
              'sourceArchiveSha256': prepared.source_sha256,
              'finalStateId': current.state_id, 'nativeActions': len(current.actions),
              'decisions': decisions, 'cost': {'previewCalls': preview_calls,
                  'committedCalls': commit_calls, 'independentReplayCalls': len(current.actions),
                  'discardedProbePreviewCalls': discarded_preview_calls,
                  'discardedProbeCommittedCalls': discarded_commit_calls,
                  'retainedBranchCommittedCalls': commit_calls-discarded_commit_calls,
                  'wallSeconds': time.perf_counter()-started,
                  'peakRssBytes': _rss_bytes()},
              'exactNativeReplay': True,
              'terminalReward': terminal_reward,
              'forcedCleanupResourcePenalty': forced_cleanup_resource_penalty,
              'terminalAppliedDecisionIndex': len(training)-1 if training else None}
    return {'engine': current, 'report': report, 'training': training}


def fixed_choice(option_ids):
    """Predeclared sequence for physical scripted/search controls."""
    ordered = iter(option_ids)
    def choose(inp):
        selected_id = next(ordered)
        if selected_id not in inp.option_ids:
            raise ValueError('Fixed strategy requested a different option boundary')
        index = inp.option_ids.index(selected_id)
        return index, [float(i == index) for i in range(len(inp.option_ids))]
    return choose


def search_control(prepared: PreparedCase):
    """Four fixed hand/order runs; teacher selection and all cost stay visible."""
    variants = []
    for hand in ('left', 'right'):
        for order in (('near', 'far'), ('far', 'near')):
            sequence = ('corridor:'+hand, order[0]+':'+hand,
                        order[1]+':'+hand, 'STOP')
            trial = rollout(prepared, fixed_choice(sequence),
                            method='bounded-search-'+hand+'-'+order[0]+'-first')
            variants.append(trial)
    successful = [item for item in variants if item['report']['taskGateMet']]
    if not successful:
        return {'variants': variants, 'teacher': None,
                'selectionRule': 'successful gate, then minimum additional non-target, '
                    'maximum additional target, left before right, near before far'}
    def rank(item):
        report = item['report']
        sequence = [d['optionId'] for d in report['decisions']
                    if d['boundary'] != 'forced_cleanup']
        return (report['deltaNonTargetMm3'], -report['deltaTargetMm3'],
                ACTION_IDS.index(sequence[0]), ACTION_IDS.index(sequence[1]))
    teacher = min(successful, key=rank)
    return {'variants': variants, 'teacher': teacher,
            'selectionRule': 'successful gate, then minimum additional non-target, '
                'maximum additional target, left before right, near before far'}


def fit_bc(demonstration, *, seed, epochs=300):
    """Fit only current-sensor high-level decisions from a native success."""
    report = demonstration['report']
    samples = demonstration['training']
    if not report['taskGateMet'] or len(samples) < 4 or \
            len({item[0].option_ids[item[1]] for item in samples}) < 3:
        raise ValueError('BC requires a successful varied native skill sequence')
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        actor = OptionActor()
    before = parameter_hash(actor)
    optimizer = torch.optim.Adam(actor.parameters(), lr=.01)
    actor.train()
    for _ in range(epochs):
        losses = [-torch.log_softmax(actor(inp), dim=0)[selected]
                  for inp, selected, _ in samples]
        loss = torch.stack(losses).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    actor.eval()
    after = parameter_hash(actor)
    predictions = [int(actor(inp).argmax()) for inp, _, _ in samples]
    expected = [selected for _, selected, _ in samples]
    if after == before or predictions != expected:
        raise ValueError('Frozen BC did not exactly fit the actual strategy choices')
    return actor, {'seed': seed, 'epochs': epochs,
        'initialActorSha256': before, 'finalActorSha256': after,
        'demonstrationActions': [inp.option_ids[selected] for inp, selected, _ in samples],
        'fittedActions': [inp.option_ids[selected] for (inp, _, _), selected in zip(samples, predictions)],
        'fittedExact': True, 'finalLoss': float(loss.detach())}


def ppo_update(actor, critic, episodes, *, epochs=2):
    """Small clipped on-policy update; returns separate actor/critic identities."""
    samples = []
    for episode in episodes:
        sequence = episode['training']
        future = 0.
        for inp, selected, reward in reversed(sequence):
            future = float(reward) + .95*future
            with torch.no_grad():
                old_logprob = torch.log_softmax(actor(inp), dim=0)[selected]
                old_value = critic(inp)
            samples.append((inp, selected, future, float(old_logprob), float(old_value)))
    if not samples:
        raise ValueError('PPO needs actual on-policy high-level choices')
    actor_before = parameter_hash(actor)
    critic_before = parameter_hash(critic)
    optimizer = torch.optim.Adam([{'params': actor.parameters(), 'lr': .001},
                                  {'params': critic.parameters(), 'lr': .001}])
    actor.train(); critic.train()
    for _ in range(epochs):
        losses = []
        for inp, selected, total, old_logprob, old_value in samples:
            logits = actor(inp)
            logprob = torch.log_softmax(logits, dim=0)[selected]
            ratio = torch.exp(logprob-old_logprob)
            advantage = torch.as_tensor(total-old_value, dtype=torch.float32)
            actor_loss = -torch.minimum(ratio*advantage,
                torch.clamp(ratio, .8, 1.2)*advantage)
            value_loss = .5*(critic(inp)-total).square()
            entropy = torch.distributions.Categorical(logits=logits).entropy()
            losses.append(actor_loss + value_loss - .005*entropy)
        loss = torch.stack(losses).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    actor.eval(); critic.eval()
    actor_after = parameter_hash(actor)
    critic_after = parameter_hash(critic)
    if actor_after == actor_before:
        raise ValueError('PPO optimizer changed no actor parameter')
    return {'epochs': epochs, 'sampledChoices': len(samples),
        'actorBeforeSha256': actor_before, 'actorAfterSha256': actor_after,
        'actorChanged': True, 'criticBeforeSha256': critic_before,
        'criticAfterSha256': critic_after,
        'criticChanged': critic_after != critic_before,
        'finalLoss': float(loss.detach())}


def model_binding(prepared: PreparedCase):
    return {'sourceArchiveSha256': prepared.source_sha256,
        'sceneId': prepared.scene.identity,
        'accessComparisonSeal': prepared.access_comparison_seal,
        'accessPlanSeal': prepared.access['planSeal'],
        'accessMethod': prepared.access_method,
        'sectorCellSha256': {name: row['sourceCellSha256']
                            for name, row in prepared.sector_evidence.items()},
        'taskGate': {'additionalTargetMm3': 30., 'eachSectorMm3': 10.,
                     'additionalNonTargetMm3': 20., 'totalNonTargetMm3': 200.,
                     'minAdmittedSectorOptions': 2, 'maxNativeActions': MAX_NATIVE_ACTIONS}}


def seal_actor(actor, path, *, stage, prepared, producer, training_receipt):
    """Keep training identity in checkpoint; current source binding is audit only."""
    if stage not in ('bc', 'bc-initialized-ppo'):
        raise ValueError('Unknown hierarchical fitting stage')
    actor_hash = parameter_hash(actor)
    payload = {'schema': SCHEMA, 'observationSchema': OBSERVATION_SCHEMA,
        'actionIds': ACTION_IDS, 'stage': stage, 'actorSha256': actor_hash,
        'stateDict': actor.state_dict(), 'sourceBinding': model_binding(prepared),
        'producerIdentity': producer, 'trainingReceipt': training_receipt}
    torch.save(payload, path)
    loaded = frozen_actor(path, expected_producer=producer)
    if parameter_hash(loaded) != actor_hash:
        raise ValueError('Saved hierarchical actor changed on cold load')
    return {'path': str(Path(path).resolve()), 'fileSha256': sha256(path),
            'actorSha256': actor_hash, 'stage': stage,
            'sourceBinding': payload['sourceBinding'],
            'trainingReceiptSha256': semantic_digest(training_receipt)}


def frozen_actor(path, *, expected_producer):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if (checkpoint.get('schema') != SCHEMA or
            checkpoint.get('observationSchema') != OBSERVATION_SCHEMA or
            checkpoint.get('actionIds') != ACTION_IDS or
            checkpoint.get('producerIdentity') != expected_producer or
            checkpoint.get('stage') not in ('bc', 'bc-initialized-ppo') or
            not isinstance(checkpoint.get('sourceBinding'), dict) or
            checkpoint['sourceBinding'].get('taskGate') != model_binding_task_gate() or
            not isinstance(checkpoint.get('trainingReceipt'), dict)):
        raise ValueError('Frozen hierarchical actor metadata differs')
    proof = checkpoint['trainingReceipt']
    if proof.get('sourceBinding') != checkpoint['sourceBinding'] or \
            proof.get('producerIdentity') != expected_producer:
        raise ValueError('Frozen hierarchy fitting receipt differs')
    if checkpoint['stage'] == 'bc':
        fit = proof.get('bcFit', {})
        if not fit.get('fittedExact') or fit.get('finalActorSha256') != checkpoint['actorSha256']:
            raise ValueError('Frozen BC lacks actual fitted option choices')
    else:
        updates = proof.get('ppoUpdates', [])
        if not updates or updates[-1].get('actorAfterSha256') != checkpoint['actorSha256'] or \
                any(not row.get('actorChanged') for row in updates):
            raise ValueError('Frozen PPO lacks actor updates')
    actor = OptionActor()
    actor.load_state_dict(checkpoint['stateDict'], strict=True)
    actor.eval()
    if parameter_hash(actor) != checkpoint['actorSha256']:
        raise ValueError('Frozen hierarchical actor parameters differ')
    return actor


def model_binding_task_gate():
    return {'additionalTargetMm3': 30., 'eachSectorMm3': 10.,
            'additionalNonTargetMm3': 20., 'totalNonTargetMm3': 200.,
            'minAdmittedSectorOptions': 2, 'maxNativeActions': MAX_NATIVE_ACTIONS}
