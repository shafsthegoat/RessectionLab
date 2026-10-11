"""Logged-action continuation value on detached frozen spatial-policy features.

Targets force EACH recorded first action, then allow stopping at its best
remaining recorded prefix. STOP itself is fixed zero. They are neither optimal values nor evidence
of successful preparation absent such trajectories. Future outcomes are labels
only. This module does not reconstruct sources, query geometry, select actions,
or authorize fitting; the owned runner binds saved evidence and resource limits.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, asdict
import math
import time
from collections import Counter

import numpy as np
import torch
from torch import nn

from .core import array_digest, freeze_json, immutable_array, semantic_digest, thaw_json
from .patient_planning_admission import PatientPlanningContext
from .patient_planning_learning import PatientTrainingTrace
from .spatial_policy import SpatialPolicy, parameter_hash

VERSION = 'frozen-spatial-logged-action-continuation-v1'
TRAIN_SUBJECTS = ('ReMIND-002', 'ReMIND-015', 'ReMIND-018', 'ReMIND-045')
TARGET = 'mandatory_logged_action_then_best_recorded_prefix_per_trace'
WEIGHTING = 'equal_subject_then_equal_trace_then_equal_state'


def _finite(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number)):
        raise ValueError(name + ' must be a finite real scalar')
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(name + ' must be finite')
    return value


def recorded_prefix_targets(rewards):
    """y[i] = max(0, max_j>=i sum(rewards[i:j+1])); no cross-trace max."""
    rewards = tuple(_finite(r, 'reward') for r in rewards)
    if not rewards:
        raise ValueError('A nonempty complete recorded continuation is required')
    result = [0.] * len(rewards)
    tail = 0.
    for i in range(len(rewards) - 1, -1, -1):
        tail = max(0., _finite(rewards[i] + tail, 'prefix target'))
        result[i] = tail
    return tuple(result)


def recorded_action_targets(rewards):
    """q[i] forces action i: r[i] + max(0, best recorded prefix from i+1).

Unlike the state target, a negative mandatory action remains negative when no
recorded continuation recovers its cost. STOP is a distinct unchanged action.
"""
    rewards = tuple(_finite(r, 'reward') for r in rewards)
    values = recorded_prefix_targets(rewards)
    return tuple(_finite(r + (values[i+1] if i+1 < len(values) else 0.),
                         'logged action target') for i, r in enumerate(rewards))


def _world(context, *, training):
    if type(context) is not PatientPlanningContext:
        raise TypeError('Exact admitted patient planning context required')
    if training:
        context.require_training()
    record = context.record()
    if record.get('role') != 'TRAIN' or record.get('subject') not in TRAIN_SUBJECTS:
        raise ValueError('Only the declared original four TRAIN subjects are admitted')
    return freeze_json({key: record[key] for key in ('subject', 'source_hash',
        'decision_model_hash', 'public_observation_binding', 'observation_track', 'max_steps')})


def trace_descriptor(key, kind, trace):
    """Stable payload join; the runner separately pins original files/certificates.

A new admitted collector context can have a new protocol seal; observations,
actions, complete nominal history, world and historical behavior provenance must
still match exactly. This function does not admit or reconstruct a trace.
"""
    if type(trace) is not PatientTrainingTrace:
        raise TypeError('Exact owned complete training trace required')
    trace.require()
    if not isinstance(key, str) or not key or kind not in ('teacher', 'RL'):
        raise ValueError('Explicit unique trace key and teacher/RL kind required')
    if (kind == 'teacher') != (trace.behavior_parameter_hash is None):
        raise ValueError('Teacher/RL behavior provenance differs')
    for row, history in zip(trace.transitions, trace.history):
        if row.action_id != 'STOP' and history.get('outcome_scope') != 'permitted_nominal_model':
            raise ValueError('Only complete permitted-nominal model rewards are labels')
    return freeze_json({'key': key, 'kind': kind, 'world': _world(trace.context, training=True),
        'behavior_parameter_hash': trace.behavior_parameter_hash,
        'history_hash': semantic_digest(trace.history),
        'decisions': [{'observation_hash': row.observation.fingerprint,
            'action_id': row.action_id, 'reward': row.reward, 'terminated': row.terminated}
            for row in trace.transitions]})


@dataclass(frozen=True)
class ContinuationLimits:
    max_traces: int = 36
    max_states: int = 497
    max_steps: int = 24
    max_feature_bytes: int = 4 * 1024**2
    seconds: float = 3600.

    def __post_init__(self):
        for name, high in (('max_traces', 36), ('max_states', 864), ('max_steps', 24),
                           ('max_feature_bytes', 16 * 1024**2)):
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= high:
                raise ValueError('Invalid bounded ' + name)
        if not 0 < _finite(self.seconds, 'seconds') <= 7200:
            raise ValueError('Invalid bounded seconds')


@dataclass(frozen=True)
class ContinuationFitConfig:
    updates: int = 64
    learning_rate: float = .001
    max_gradient_norm: float = 5.
    seconds: float = 120.

    def __post_init__(self):
        if (type(self.updates) is not int or not 1 <= self.updates <= 64
                or self.learning_rate != .001 or self.max_gradient_norm != 5.
                or not 0 < _finite(self.seconds, 'fit seconds') <= 300):
            raise ValueError('Fixed bounded head-only Adam protocol required')


class _FrozenPolicy:
    def __init__(self, policy):
        if type(policy) is not SpatialPolicy or any(m.training for m in policy.modules()):
            raise ValueError('Exact preloaded eval SpatialPolicy required')
        if any(p.device.type != 'cpu' or p.grad is not None for p in policy.parameters()):
            raise ValueError('CPU policy with clean gradient buffers required')
        self.policy = policy
        self.architecture = policy.architecture_hash
        self.parameters = parameter_hash(policy)
        self.identity = self._identity()
        self.checks = 0
        self.seconds = 0.

    def _identity(self):
        return (tuple((n, id(p), p._version, p.requires_grad) for n, p in self.policy.named_parameters()),
                tuple((n, id(m), m.training) for n, m in self.policy.named_modules()))

    def require(self):
        start = time.perf_counter()
        try:
            if (self.policy.architecture_hash != self.architecture or self._identity() != self.identity
                    or any(p.grad is not None for p in self.policy.parameters())
                    or parameter_hash(self.policy) != self.parameters):
                raise ValueError('Frozen encoder/actor/STOP/base critic or buffers changed')
        finally:
            self.checks += 1
            self.seconds += time.perf_counter() - start


def _feature(policy, observation, costs):
    """Capture exact existing actor input; count the FULL policy forward paid.

Temporary hook is always removed. Captured rows own detached CPU bytes and do
not retain observations, native tasks, graphs or policy tensors.
"""
    captured = []
    def capture(module, args):
        if len(args) != 1 or args[0].ndim != 2:
            raise ValueError('Existing batched actor feature interface changed')
        captured.append(args[0].detach().cpu().clone())
    handle = policy.actor.register_forward_pre_hook(capture)
    start = time.perf_counter()
    costs['policy_forwards_started'] += 1
    try:
        with torch.no_grad():
            policy(observation)
        if (len(captured) != 1 or captured[0].shape[0] != len(observation.action_ids)
                or not torch.isfinite(captured[0]).all()):
            raise ValueError('Exactly one finite actor-input matrix required')
        costs['policy_forwards_completed'] += 1
        return immutable_array(captured[0].numpy())
    finally:
        handle.remove()
        costs['feature_seconds'] += time.perf_counter() - start


def _costs():
    return {'policy_forwards_started': 0, 'policy_forwards_completed': 0,
            'feature_seconds': 0., 'geometry_queries': 0, 'observations_constructed': 0}


@dataclass(frozen=True)
class ContinuationCorpus:
    features: np.ndarray
    targets: np.ndarray
    weights: np.ndarray
    stop_rows: np.ndarray
    metadata: object
    seal: str
    _guard: _FrozenPolicy

    def require(self, policy):
        if policy is not self._guard.policy:
            raise ValueError('Corpus belongs to a different frozen policy object')
        self._guard.require()
        if self.seal != semantic_digest({'metadata': self.metadata,
                'features': array_digest(self.features), 'targets': array_digest(self.targets),
                'weights': array_digest(self.weights), 'stop_rows': array_digest(self.stop_rows)}):
            raise ValueError('Detached corpus changed')
        return self


class ContinuationCorpusBuilder:
    """Incremental complete-trace ingestion; only compact features/scalars survive.

Manifest order fixes fitting order. Metadata includes every declared trace and
row; duplicate public states retain their distinct recorded-continuation labels.
On failure this builder is consumed, so a partial corpus cannot be fitted.
"""
    def __init__(self, policy, manifest, *, manifest_hash, limits=None, cancelled=None):
        self.limits = limits or ContinuationLimits()
        self.manifest = freeze_json(manifest)
        if semantic_digest(self.manifest) != manifest_hash:
            raise ValueError('Frozen trace manifest hash mismatch')
        self.manifest_hash = manifest_hash
        if not 1 <= len(self.manifest) <= self.limits.max_traces:
            raise ValueError('Declared complete-trace limit exceeded')
        keys = [r['key'] for r in self.manifest]
        if len(set(keys)) != len(keys):
            raise ValueError('Duplicate declared trace key')
        counts = Counter((r['world']['subject'], r['kind']) for r in self.manifest)
        if (set(s for s, _ in counts) != set(TRAIN_SUBJECTS)
                or any(counts[(s, 'teacher')] != 1 or not 1 <= counts[(s, 'RL')] <= 8
                       for s in TRAIN_SUBJECTS)):
            raise ValueError('All four TRAIN subjects need one teacher and declared RL traces')
        for row in self.manifest:
            if (set(row) != {'key', 'kind', 'world', 'decisions', 'history_hash', 'behavior_parameter_hash'}
                    or row['kind'] not in ('teacher', 'RL') or not row['decisions']
                    or len(row['decisions']) > self.limits.max_steps):
                raise ValueError('Malformed bounded trace declaration')
        if sum(len(r['decisions']) for r in self.manifest) > self.limits.max_states:
            raise ValueError('Declared feature-row limit exceeded')
        self.guard = _FrozenPolicy(policy)
        self.cancelled = cancelled
        self.started = time.perf_counter()
        self.costs = _costs()
        self._rows = {}
        self._dead = False
        self._worlds = {}
        self._bytes = 0

    def _check(self):
        if self._dead:
            raise ValueError('Failed/finalized corpus builder is consumed')
        if time.perf_counter() - self.started > self.limits.seconds or (self.cancelled and self.cancelled()):
            raise RuntimeError('Continuation corpus time/cancellation cap')

    def add_trace(self, key, trace):
        try:
            self._check()
            if key in self._rows:
                raise ValueError('Duplicate contributed trace')
            expected = next((r for r in self.manifest if r['key'] == key), None)
            if expected is None:
                raise ValueError('Undeclared trace')
            actual = trace_descriptor(key, expected['kind'], trace)
            if actual != expected:
                raise ValueError('Trace payload differs from frozen saved manifest')
            self.guard.require()
            world = actual['world']; subject = world['subject']
            if subject in self._worlds and self._worlds[subject] != world:
                raise ValueError('One public world per admitted TRAIN subject required')
            features = []
            for row in trace.transitions:
                self._check()
                matrix = _feature(self.guard.policy, row.observation, self.costs)
                feature = immutable_array(matrix[row.observation.action_ids.index(row.action_id)])
                self._bytes += feature.nbytes
                if self._bytes > self.limits.max_feature_bytes:
                    raise RuntimeError('Detached feature-byte cap')
                features.append(feature)
            targets = recorded_action_targets(r.reward for r in trace.transitions)
            self._rows[key] = (features, targets, freeze_json({'context_hash': trace.context.fingerprint,
                'trace_seal': trace.seal_hash, 'world': world}))
            self._worlds[subject] = world
        except BaseException:
            self._dead = True
            raise
        finally:
            self.guard.require()

    def finalize(self):
        try:
            self._check()
            if set(self._rows) != {r['key'] for r in self.manifest}:
                raise ValueError('Every declared complete trace is required')
            self.guard.require()
            counts = Counter(r['world']['subject'] for r in self.manifest)
            features, targets, weights, stops, provenance = [], [], [], [], []
            positive_after_negative = 0
            for declaration in self.manifest:
                rows, y, binding = self._rows[declaration['key']]
                features.extend(rows); targets.extend(y)
                stops.extend(d['action_id'] == 'STOP' for d in declaration['decisions'])
                weights.extend([1. / (len(TRAIN_SUBJECTS) * counts[declaration['world']['subject']] * len(rows))] * len(rows))
                positive_after_negative += sum(d['reward'] < 0 and value > 0
                    for d, value in zip(declaration['decisions'], y))
                provenance.append({'key': declaration['key'], **thaw_json(binding), 'rows': len(rows)})
            x = immutable_array(np.stack(features), np.float32)
            y = immutable_array(targets, np.float32)
            w = immutable_array(weights, np.float32)
            stop_rows = immutable_array(stops, np.bool_)
            if not np.isfinite(x).all() or not np.isfinite(y).all() or not np.isclose(float(w.sum()), 1.):
                raise ValueError('Nonfinite corpus or invalid fixed mass')
            metadata = freeze_json({'version': VERSION, 'target': TARGET, 'weighting': WEIGHTING,
                'manifest': self.manifest, 'manifest_hash': self.manifest_hash, 'trace_bindings': provenance,
                'base_parameter_hash': self.guard.parameters, 'architecture_hash': self.guard.architecture,
                'worlds': self._worlds, 'states': len(y), 'feature_bytes': x.nbytes, 'STOP_rows': int(stop_rows.sum()),
                'STOP_prediction': 'fixed_zero_no_head_gradient',
                'negative_immediate_positive_recorded_continuation': positive_after_negative,
                'costs': {**self.costs, 'freeze_checks': self.guard.checks,
                    'freeze_check_seconds': self.guard.seconds, 'elapsed_seconds': time.perf_counter() - self.started}})
            seal = semantic_digest({'metadata': metadata, 'features': array_digest(x),
                'targets': array_digest(y), 'weights': array_digest(w), 'stop_rows': array_digest(stop_rows)})
            return ContinuationCorpus(x, y, w, stop_rows, metadata, seal, self.guard)
        finally:
            self._dead = True
            self._rows.clear()


class FittedContinuationValue:
    """Public-observation scoring only; final selection still uses actual return."""
    def __init__(self, guard, head, worlds, report, *, max_scores, seconds, cancelled):
        self._guard, self._head = guard, head
        self._worlds = freeze_json(worlds)
        self.report = freeze_json(report)
        self._head_hash = parameter_hash(head)
        self._max_scores, self._seconds, self._cancelled = max_scores, seconds, cancelled
        self._started = time.perf_counter()
        self.costs = {**_costs(), 'head_forwards': 0, 'score_calls': 0, 'candidate_rows_scored': 0,
                      'score_seconds': 0., 'freeze_check_seconds': 0.}

    def require(self):
        start = time.perf_counter()
        self._guard.require()
        if (parameter_hash(self._head) != self._head_hash or any(m.training for m in self._head.modules())
                or any(p.requires_grad or p.grad is not None for p in self._head.parameters())):
            raise ValueError('Frozen fitted continuation head changed')
        self.costs['freeze_check_seconds'] += time.perf_counter() - start

    def bind_context(self, context):
        world = _world(context, training=False)
        if self._worlds.get(world['subject']) != world:
            raise ValueError('Foreign public world/source/model/observation context')
        return context

    def score_actions(self, observation, context):
        """One frozen policy call, one head batch; STOP0, illegal entries None.

The returned values ALREADY include the candidate's logged-action reward model.
They must not be added to that action's actual immediate reward a second time.
The integrator owns native legality, actual-return incumbents and completion.
"""
        start = time.perf_counter()
        try:
            self.require(); self.bind_context(context)
            if (self.costs['score_calls'] >= self._max_scores
                    or time.perf_counter() - self._started > self._seconds
                    or (self._cancelled and self._cancelled())):
                raise RuntimeError('Continuation score call/time/cancellation cap')
            self.costs['score_calls'] += 1
            context.require_observations((observation,))
            if observation.action_ids[0] != 'STOP' or not observation.action_mask[0]:
                raise ValueError('Unchanged legal STOP0 alternative required')
            features = _feature(self._guard.policy, observation, self.costs)
            with torch.no_grad():
                raw = self._head(torch.tensor(np.array(features, copy=True))).squeeze(-1)
            self.costs['head_forwards'] += 1
            if not torch.isfinite(raw).all():
                raise FloatingPointError('Nonfinite logged-action predictions')
            result = tuple(None if not legal else 0. if i == 0 else float(raw[i])
                for i, legal in enumerate(observation.action_mask))
            self.costs['candidate_rows_scored'] += len(result)
            return result
        finally:
            self.costs['score_seconds'] += time.perf_counter() - start
            self.require()


def fit_continuation_value(policy, corpus, *, config=None, cancelled=None,
        max_scores=12000, score_seconds=3600.):
    """One bounded head fit; exact original IL actor initialization, fresh Adam.

No encoder/actor/STOP/base-critic parameter participates in this optimizer.
Feature forwards are already paid once during corpus extraction. Full-batch
head calls, backward and optimizer steps are reported separately.
"""
    if type(corpus) is not ContinuationCorpus:
        raise TypeError('Exact detached authenticated continuation corpus required')
    config = config or ContinuationFitConfig()
    if (type(max_scores) is not int or not 1 <= max_scores <= 50000
            or not 0 < _finite(score_seconds, 'score seconds') <= 7200):
        raise ValueError('Bounded score declaration required')
    corpus.require(policy)
    guard = corpus._guard
    started = time.perf_counter()
    rows = []
    work = {name + suffix: 0 for name in ('head_training_forwards',
        'head_readout_forwards', 'head_backward_calls', 'head_optimizer_steps')
        for suffix in ('_attempted', '_completed')}
    try:
        head = copy.deepcopy(policy.actor)
        head.train()
        for p in head.parameters():
            p.requires_grad_(True)
        initial = parameter_hash(head)
        x = torch.tensor(np.array(corpus.features, copy=True))
        y = torch.tensor(np.array(corpus.targets, copy=True))
        w = torch.tensor(np.array(corpus.weights, copy=True))
        stops = torch.tensor(np.array(corpus.stop_rows, copy=True))
        optimizer = torch.optim.Adam(head.parameters(), lr=config.learning_rate,
            betas=(.9, .999), eps=1e-8, weight_decay=0., amsgrad=False, foreach=False, fused=False)
        for update in range(config.updates):
            if time.perf_counter() - started > config.seconds or (cancelled and cancelled()):
                raise RuntimeError('Continuation fit time/cancellation cap')
            guard.require()
            optimizer.zero_grad(set_to_none=True)
            work['head_training_forwards_attempted'] += 1
            prediction = head(x).squeeze(-1).masked_fill(stops, 0.)
            work['head_training_forwards_completed'] += 1
            loss = ((prediction - y).square() * w).sum()
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite continuation objective')
            work['head_backward_calls_attempted'] += 1
            loss.backward()
            work['head_backward_calls_completed'] += 1
            norm = float(nn.utils.clip_grad_norm_(head.parameters(), config.max_gradient_norm,
                error_if_nonfinite=True))
            work['head_optimizer_steps_attempted'] += 1
            optimizer.step()
            work['head_optimizer_steps_completed'] += 1
            rows.append({'update': update + 1, 'weighted_mse_before_update': float(loss.detach()),
                         'gradient_norm_before_clip': norm})
        work['head_readout_forwards_attempted'] += 1
        with torch.no_grad():
            final_loss = float(((head(x).squeeze(-1).masked_fill(stops, 0.) - y).square() * w).sum())
        work['head_readout_forwards_completed'] += 1
        if not math.isfinite(final_loss):
            raise FloatingPointError('Nonfinite fitted continuation objective')
        head.eval()
        for p in head.parameters():
            p.grad = None; p.requires_grad_(False)
        guard.require()
        report = {'version': VERSION, 'target': TARGET, 'weighting': WEIGHTING,
            'corpus_seal': corpus.seal, 'base_parameter_hash': guard.parameters,
            'architecture_hash': guard.architecture, 'config': asdict(config),
            'initial_head_hash': initial, 'fitted_head_hash': parameter_hash(head),
            'initialization': 'exact_clone_of_frozen_IL64_actor_MLP',
            'updates': rows, 'final_weighted_mse': final_loss,
            'costs': {'head_training_forwards': len(rows), 'head_readout_forwards': 1,
                'head_backward_calls': len(rows), 'head_optimizer_steps': len(rows),
                'base_policy_forwards': 0, 'base_optimizer_steps': 0,
                'elapsed_seconds': time.perf_counter() - started,
                'freeze_checks': guard.checks, 'freeze_check_seconds': guard.seconds}}
        return FittedContinuationValue(guard, head, corpus.metadata['worlds'], report,
            max_scores=max_scores, seconds=score_seconds, cancelled=cancelled)
    except BaseException as error:
        # Never replace the original failure with a secondary integrity error.
        # A step which raised after mutating its head is attempted, not known to
        # have completed; the separate counters preserve that uncertainty.
        base_unchanged = None
        integrity_error = None
        try:
            guard.require()
            base_unchanged = True
        except BaseException as check_error:
            base_unchanged = False
            integrity_error = type(check_error).__name__ + ': ' + str(check_error)[:200]
        accounting = {**work, 'completed_updates': work['head_optimizer_steps_completed'],
            'optimizer_step_completion_unknown': work['head_optimizer_steps_attempted']
                != work['head_optimizer_steps_completed'],
            'update_rows': [dict(row) for row in rows],
            'elapsed_seconds': time.perf_counter() - started,
            'base_parameters_unchanged': base_unchanged,
            'base_integrity_error': integrity_error,
            'base_policy_forwards': 0, 'base_optimizer_steps': 0}
        try:
            error.continuation_fit_accounting = accounting
        except BaseException:
            pass  # Unusual immutable exception objects must still propagate.
        raise
