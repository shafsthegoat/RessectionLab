"""Checkpoint-reloaded SELECT planning and matched public SEARCH replay.

The owner qualifies public013/037 sources and binds the original TRAIN terminal
receipt before entry. EVAL and private-reference inputs have no interface here.
The existing TRAIN collector, trace capability and optimizer admission are unchanged.
"""
from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Mapping

import torch

from .core import freeze_json, semantic_digest, thaw_json
from .native_proposals import SUPPLIED_GOAL_REGION
from .patient_planning_admission import (PatientPlanningContext, COHORT_SHA256,
    SELECT_SUBJECTS, SELECT_INITIALIZATION, validate_select_checkpoint_lineage)
from .patient_planning_cohort_io import load_cohort_checkpoint
from . import patient_planning_preflight as preflight
from .spatial_policy import SpatialPolicy, SpatialTransition, parameter_hash

_LOADED = object()
_INFERENCE_TRACE = object()


def require_select_context(context):
    if type(context) is not PatientPlanningContext:
        raise TypeError('Exact admitted SELECT context required')
    record = context.record()
    if (record['subject'] not in SELECT_SUBJECTS or record['role'] != 'SELECT'
            or record['cohort_sha256'] != COHORT_SHA256
            or record['patient_group'] != 'ReMIND:'+record['subject'].rsplit('-', 1)[1]
            or record['scope'] != 'patient_native_planning_experiment'
            or type(record['max_optimizer_updates']) is not int or record['max_optimizer_updates'] != 0
            or record.get('initialization') != SELECT_INITIALIZATION
            or record['private_reference_in_task'] is not False):
        raise ValueError('Only fixed SELECT checkpoint inference with zero gradients is admitted')
    validate_select_checkpoint_lineage(record['checkpoint_lineage'],
        learning_protocol_hash=record['learning_protocol_hash'])
    return record


@dataclass(frozen=True)
class FrozenSelectCheckpoint:
    """A verified weight-only reload, separate from every TRAIN-session registry."""
    policy: SpatialPolicy
    _lineage: Mapping
    _protocol: Mapping
    _capability: object = field(repr=False)
    _seal: str = field(init=False, repr=False)

    def __post_init__(self):
        if self._capability is not _LOADED or type(self.policy) is not SpatialPolicy:
            raise ValueError('Only the bounded checkpoint loader may admit SELECT weights')
        lineage = validate_select_checkpoint_lineage(self._lineage)
        object.__setattr__(self, '_lineage', lineage)
        object.__setattr__(self, '_protocol', freeze_json(self._protocol))
        object.__setattr__(self, '_seal', semantic_digest({'lineage': lineage, 'protocol': self._protocol}))
        self.require()

    def lineage_record(self):
        self.require()
        return thaw_json(self._lineage)

    def require(self, context=None):
        if (self._capability is not _LOADED
                or semantic_digest({'lineage': self._lineage, 'protocol': self._protocol}) != self._seal
                or semantic_digest(self._protocol) != self._lineage['learning_protocol_hash']
                or parameter_hash(self.policy) != self._lineage['parameter_hash']
                or self.policy.architecture_hash != self._lineage['architecture_hash']
                or self.policy.public_target_context_variant != self._lineage['public_target_context_variant']
                or any(module.training for module in self.policy.modules())
                or any(p.requires_grad or p.grad is not None for p in self.policy.parameters())):
            raise ValueError('Frozen SELECT checkpoint weights, lineage or gradient state changed')
        if context is not None:
            record = require_select_context(context)
            if (semantic_digest(record['checkpoint_lineage']) != semantic_digest(self._lineage)
                    or record.get('public_target_context_variant') != self._lineage['public_target_context_variant']):
                raise ValueError('SELECT context does not bind these exact reloaded weights')
        return self

    def require_task(self, base, context):
        self.require(context); context.require_task(base)
        if base.case.public_target_context_variant != self._lineage['public_target_context_variant']:
            raise ValueError('SELECT source observation variant differs from reloaded checkpoint')
        execution = self._protocol.get('cohort_execution')
        if execution is not None and (
                base.max_steps != execution['max_steps']
                or base.case.proposal_mode != 'nominal_cavity_v1'
                or base.case.proposal_config.fingerprint != execution['proposal_rule_hash']
                or base.case.target_semantics != SUPPLIED_GOAL_REGION
                or context.record()['budgets']['search'] != thaw_json(execution['search'])):
            raise ValueError('SELECT task horizon, proposals or search differ from frozen TRAIN condition')
        return self


def load_select_checkpoint(path, *, expected_sha256, expected_learning_protocol,
        expected_context_hashes, expected_method, training_release_sha256):
    """Reuse the bounded NPZ/ZIP loader with external TRAIN provenance pins.

No task, observation, forward, optimizer or patient file is opened. The owner
must authenticate training_release_sha256 and the supplied endpoint pins against
the completed TRAIN evidence. Metadata alone is not proof of training execution.
"""
    if (type(training_release_sha256) is not str or len(training_release_sha256) != 64
            or any(c not in '0123456789abcdef' for c in training_release_sha256)):
        raise ValueError('Exact original TRAIN release SHA required')
    policy, metadata = load_cohort_checkpoint(path, expected_sha256=expected_sha256,
        expected_learning_protocol=expected_learning_protocol,
        expected_context_hashes=expected_context_hashes, expected_method=expected_method)
    policy.eval(); policy.requires_grad_(False)
    lineage = {'version': 'frozen-TRAIN-checkpoint-lineage-v1',
        'checkpoint_sha256': expected_sha256, 'method': metadata['method'],
        'training_release_sha256': training_release_sha256,
        'learning_protocol_hash': metadata['learning_protocol_hash'],
        'initial_parameter_hash': metadata['initial_parameter_hash'],
        'parameter_hash': metadata['parameter_hash'], 'architecture_hash': metadata['architecture_hash'],
        'completed_updates': metadata['completed_updates'],
        'training_context_hashes': {row['patient_group']: row['context_hash'] for row in metadata['contexts']},
        'public_target_context_variant': expected_learning_protocol.get('public_target_context_variant'),
        'optimizer_updates_on_SELECT': 0}
    return FrozenSelectCheckpoint(policy, lineage, expected_learning_protocol, _LOADED)


@dataclass(frozen=True)
class PatientInferenceTrace:
    """A complete SELECT trace. This is never a PatientTrainingTrace capability."""
    context: PatientPlanningContext
    transitions: tuple
    history: tuple
    checkpoint_lineage: Mapping
    method: str
    behavior_parameter_hash: str | None
    seal_hash: str
    _capability: object = field(repr=False)

    def require(self):
        record = require_select_context(self.context)
        if (self._capability is not _INFERENCE_TRACE
                or semantic_digest(self.checkpoint_lineage) != semantic_digest(record['checkpoint_lineage'])
                or self.method not in ('SEARCH', self.checkpoint_lineage['method'])
                or self.behavior_parameter_hash != (None if self.method == 'SEARCH' else self.checkpoint_lineage['parameter_hash'])):
            raise ValueError('Owned SELECT inference trace and exact checkpoint required')
        rows = self.transitions
        if (not rows or len(rows) > self.context.max_steps or len(rows) != len(self.history)
                or not rows[-1].terminated or any(row.terminated for row in rows[:-1])):
            raise ValueError('Complete SELECT STOP-or-horizon trajectory required')
        self.context.require_observations(row.observation for row in rows)
        for step, (row, native) in enumerate(zip(rows, self.history)):
            if (type(row) is not SpatialTransition or row.observation.state_features[0] != step
                    or not math.isfinite(row.reward) or row.action_id != native['action_id']
                    or row.reward != native['reward'] or row.action_id not in row.observation.action_ids
                    or not row.observation.action_mask[row.observation.action_ids.index(row.action_id)]):
                raise ValueError('SELECT observation/action/reward differs from complete native history')
        stops = [i for i, row in enumerate(rows) if row.action_id == 'STOP']
        if ((stops and stops != [len(rows)-1]) or (not stops and len(rows) != self.context.max_steps)
                or any(row.action_id == 'STOP' and row.reward != 0. for row in rows)):
            raise ValueError('Incomplete or invalid SELECT termination')
        expected = semantic_digest({'context': self.context.fingerprint,
            'observations': [row.observation.fingerprint for row in rows], 'history': self.history,
            'checkpoint_lineage': self.checkpoint_lineage, 'method': self.method,
            'behavior_parameter_hash': self.behavior_parameter_hash})
        if expected != self.seal_hash: raise ValueError('SELECT inference trace identity changed')
        return self


def collect_select_greedy(base, context, checkpoint, *, output, guard):
    """Greedy zero-shot SELECT only; no teacher override, stochastic draw or update."""
    return _collect_select(base, context, checkpoint, output=output, guard=guard)


def collect_select_search(base, context, checkpoint, *, actions, accounting, output, guard):
    """Seal a complete uncapped public search return, with no policy forward.

The checkpoint/context fixes the comparison world and task condition only.
SEARCH actions are never attributed to that checkpoint or used as gradients.
The owned caller runs the unchanged observed_beam_search and supplies its return.
"""
    accounting = freeze_json(accounting); actions = tuple(actions)
    if (accounting.get('call_cap_reached') is not False or accounting.get('time_cap_reached') is not False
            or not actions or any(type(action) is not str for action in actions)):
        raise ValueError('Complete uncapped SELECT SEARCH return required')
    return _collect_select(base, context, checkpoint, actions=actions,
        search_accounting=accounting, output=output, guard=guard)


def _collect_select(base, context, checkpoint, *, output, guard, actions=None, search_accounting=None):
    if type(checkpoint) is not FrozenSelectCheckpoint: raise TypeError('Reloaded SELECT checkpoint required')
    checkpoint.require_task(base, context)
    if base.terminated or base.observation().state_features[0] != 0:
        raise ValueError('Fresh SELECT task required')
    output = Path(output); worker = base.planning_clone(); rows = []; decisions = []
    method = checkpoint._lineage['method'] if actions is None else 'SEARCH'
    before = checkpoint._lineage['parameter_hash'] if actions is None else None
    if actions is not None:
        preflight._write(output/'search-return.json', {'actions': actions, 'accounting': search_accounting,
            'checkpoint_lineage_scope': 'comparison_reference_only_not_action_author',
            'behavior_parameter_hash': None})
    while not worker.terminated:
        guard(); checkpoint.require_task(worker, context)
        observation = worker.observation(); context.require_observations((observation,))
        if actions is None:
            with torch.no_grad():
                logits, _ = checkpoint.policy(observation)
                action = observation.action_ids[int(logits.argmax())]
        else:
            if len(rows) >= len(actions): raise ValueError('Incomplete SELECT SEARCH plan')
            action = actions[len(rows)]
        checkpoint.require(context)
        decision = {'step': len(rows), 'observation_hash': observation.fingerprint,
            'action_ids': list(observation.action_ids), 'action_mask': observation.action_mask.tolist(),
            'action_id': action, 'behavior_parameter_hash': before}
        preflight._write(output/('attempt-%02d.json' % len(rows)), decision)
        try: outcome = worker.step(action)
        except BaseException as error:
            preflight._write(output/('transition-failure-%02d.json' % len(rows)), {
                'error_type': type(error).__name__, 'message': str(error),
                'committed': bool(getattr(error, 'committed', False)),
                'committed_info': getattr(error, 'info', None), 'retained_metrics': worker.metrics()})
            raise
        rows.append(SpatialTransition(observation, action, outcome.reward, outcome.terminated))
        decisions.append({**decision, 'reward': outcome.reward, 'terminated': outcome.terminated})
        preflight._write(output/('returned-%02d.json' % (len(rows)-1)), decisions[-1])
    checkpoint.require_task(worker, context)
    if actions is not None and len(rows) != len(actions):
        raise ValueError('SELECT SEARCH actions continue after termination')
    if worker.metrics().get('planning_estimator_only') is not True:
        raise ValueError('SELECT plan must use the admitted public nominal world')
    history = freeze_json(worker.metrics()['history']); lineage = checkpoint._lineage
    seal = semantic_digest({'context': context.fingerprint,
        'observations': [row.observation.fingerprint for row in rows], 'history': history,
        'checkpoint_lineage': lineage, 'method': method, 'behavior_parameter_hash': before})
    trace = PatientInferenceTrace(context, tuple(rows), history, lineage, method, before, seal, _INFERENCE_TRACE).require()
    preflight._write(output/'complete-trace.json', {'trace_seal': trace.seal_hash,
        'decisions': decisions, 'metrics': worker.metrics(), 'context_hash': context.fingerprint,
        'method': method, 'checkpoint_lineage': lineage, 'optimizer_updates_on_SELECT': 0,
        'checkpoint_lineage_scope': 'executed_greedy_policy' if actions is None else 'comparison_reference_only_not_action_author'})
    return trace


def seal_and_replay_select(base, context, trace, checkpoint, *, output, guard):
    """Reuse unchanged native replay/full geometry checks with frozen-weight guards."""
    if type(trace) is not PatientInferenceTrace or type(checkpoint) is not FrozenSelectCheckpoint:
        raise TypeError('Exact SELECT inference trace and reloaded checkpoint required')
    checkpoint.require_task(base, context); trace.require()
    if trace.context.fingerprint != context.fingerprint:
        raise ValueError('SELECT replay context differs from collected trace')
    def frozen_guard():
        guard(); checkpoint.require(context)
    frozen_guard()
    is_search = trace.method == 'SEARCH'
    result = preflight._seal_and_replay(base, context, trace, method=trace.method,
        policy=None if is_search else checkpoint.policy,
        updates=0 if is_search else checkpoint._lineage['completed_updates'],
        output=Path(output), guard=frozen_guard)
    checkpoint.require_task(base, context); trace.require(); frozen_guard()
    receipt = {'version': 'checkpoint-reloaded-SELECT-native-replay-v1',
        'status': 'complete', 'subject': context.record()['subject'], 'role': 'SELECT',
        'context_hash': context.fingerprint, 'trace_seal': trace.seal_hash,
        'plan_seal': result['plan_seal'], 'checkpoint_lineage': checkpoint._lineage,
        'method': trace.method, 'behavior_parameter_hash': trace.behavior_parameter_hash,
        'checkpoint_lineage_scope': 'comparison_reference_only_not_action_author' if is_search else 'executed_greedy_policy',
        'optimizer_updates_on_SELECT': 0, 'patient_adaptation': False,
        'learning_updates_scope': 'SEARCH has zero learning updates' if is_search else 'plan count is upstream TRAIN updates; no SELECT updates',
        'private_reference_reads': 0, 'EVAL_opened': False, 'clinical_claim': False}
    preflight._write(Path(output)/'select-replay.json', receipt)
    return {**result, 'select_inference': freeze_json(receipt)}
