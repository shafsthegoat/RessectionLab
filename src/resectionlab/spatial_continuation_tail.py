"""Recorded continuation-tail readout with exact immediate reward at deployment.

This fits the best *recorded subsequent* prefix, including stopping after the
current action. It is not an optimal or policy-independent continuation value.
No images, tasks, native previews or reward evaluation enter this fitter. The
planner supplies its already-certified current nominal reward separately.
"""
from __future__ import annotations

import copy
import math
import time

import numpy as np
import torch
from torch import nn

from .core import array_digest, freeze_json, immutable_array, semantic_digest
from .spatial_policy import parameter_hash
from .spatial_continuation_value import (
    ContinuationCorpus, FittedContinuationValue, TRAIN_SUBJECTS, WEIGHTING,
    _finite, recorded_prefix_targets,
)
from .spatial_continuation_calibration import (
    VERSION as CALIBRATION_VERSION, SVD_RELATIVE_CUTOFF, MAX_SECONDS,
    _minimum_change, _mse,
)

VERSION = 'frozen-recorded-continuation-tail-readout-v1'
TARGET = 'best_recorded_subsequent_prefix_including_STOP0_after_mandatory_action'
PREDICTION_SEMANTICS = 'raw_recorded_continuation_tail_excluding_current_reward'
DEPLOYMENT = 'exact_immediate_nominal_reward_plus_max_zero_predicted_tail'


class FittedContinuationTail(FittedContinuationValue):
    """Distinct exact inference type; raw tails are never exposed as direct Q.

The inherited frozen feature/head lifecycle is reused. The direct-Q planner's
existing exact-type boundary rejects this type; callers must use score_tails.
"""
    def __init__(self, guard, head, worlds, report, **kwargs):
        if (report.get('version') != VERSION or report.get('target') != TARGET
                or report.get('prediction_semantics') != PREDICTION_SEMANTICS
                or report.get('deployment') != DEPLOYMENT):
            raise ValueError('Explicit raw-tail prediction/deployment declaration required')
        super().__init__(guard, head, worlds, report, **kwargs)
        self._tail_report_hash = semantic_digest(self.report)

    def require(self):
        super().require()
        if semantic_digest(self.report) != self._tail_report_hash:
            raise ValueError('Frozen tail declaration changed')

    def score_actions(self, observation, context):
        raise TypeError('Raw continuation tails are not direct Q; use score_tails')

    def score_tails(self, observation, context):
        """One public-state batch of raw tails; STOP0/illegal None unchanged."""
        return super().score_actions(observation, context)


def recorded_tail_targets(rewards):
    """Tail[i] = max(0, max_j>i sum(r[i+1:j+1])); tail[-1] = 0."""
    values = recorded_prefix_targets(rewards)
    return tuple(values[1:]) + (0.,)


def _bound_targets(corpus):
    rewards, tails, stops, subjects = [], [], [], []
    for trace in corpus.metadata['manifest']:
        decisions = trace['decisions']
        current = tuple(_finite(row['reward'], 'recorded reward') for row in decisions)
        tail = recorded_tail_targets(current)
        for i, row in enumerate(decisions):
            stop = row['action_id'] == 'STOP'
            if stop and (i != len(decisions) - 1 or current[i] != 0. or tail[i] != 0.):
                raise ValueError('STOP must be terminal with exact zero reward and tail')
            rewards.append(current[i]); tails.append(tail[i]); stops.append(stop)
            subjects.append(trace['world']['subject'])
    rewards = immutable_array(rewards, np.float64)
    tails = immutable_array(tails, np.float64)
    stops = np.asarray(stops, dtype=np.bool_)
    # Preserve float64 recurrence values until this single rounding. Subtracting
    # rounded cached Q targets could invent a negative tail or double rounding.
    joined = np.asarray(rewards + tails, dtype=np.float32)
    if (rewards.shape != corpus.targets.shape or not np.array_equal(stops, corpus.stop_rows)
            or not np.array_equal(joined, corpus.targets)
            or not np.isfinite(tails).all() or np.any(tails < 0)
            or set(subjects) != set(TRAIN_SUBJECTS)):
        raise ValueError('Original mandatory-action target, STOP or TRAIN row join differs')
    declaration = freeze_json({
        'version': VERSION, 'target': TARGET, 'source_corpus_seal': corpus.seal,
        'source_manifest_hash': corpus.metadata['manifest_hash'],
        'original_Q_targets_hash': array_digest(corpus.targets),
        'current_rewards_hash': array_digest(rewards), 'tail_targets_hash': array_digest(tails),
        'weight_hash': array_digest(corpus.weights), 'STOP_hash': array_digest(corpus.stop_rows),
        'rows': len(tails), 'STOP_rows': int(stops.sum()),
        'tail_precision': 'float64_manifest_recurrence',
        'original_target_join': 'exact_float32_current_reward_plus_float64_tail',
    })
    return rewards, tails, tuple(subjects), declaration


def _metrics(before, after, solved, rewards, tails, q_targets, weights, stops, subjects):
    clipped = np.maximum(0., after)
    metrics = {
        'direct_calibrated_Q_weighted_mse': _mse(before, q_targets, weights, stops),
        'initial_raw_tail_weighted_mse': _mse(before, tails, weights, stops),
        'raw_tail_weighted_mse': _mse(after, tails, weights, stops),
        'clipped_tail_weighted_mse': _mse(clipped, tails, weights, stops),
        'float64_solution_tail_weighted_mse': _mse(solved, tails, weights, stops),
        'reconstructed_Q_weighted_mse': _mse(rewards + clipped, q_targets, weights, stops),
        'negative_raw_tail_rows': int(np.sum((after < 0) & ~stops)),
    }
    per_case = {}
    subjects = np.asarray(subjects)
    for subject in TRAIN_SUBJECTS:
        take = subjects == subject
        mass = float(weights[take].sum())
        per_case[subject] = {'rows': int(take.sum()), 'weight_mass': mass}
        for key, prediction, target in (
            ('direct_calibrated_Q_weighted_mse', before, q_targets),
            ('raw_tail_weighted_mse', after, tails),
            ('clipped_tail_weighted_mse', clipped, tails),
            ('reconstructed_Q_weighted_mse', rewards + clipped, q_targets)):
            per_case[subject][key] = _mse(prediction[take], target[take], weights[take], stops[take]) / mass
    return metrics, per_case


def fit_continuation_tail(policy, corpus, calibrated, *, cancelled=None,
        max_scores=12000, score_seconds=3600.):
    """One fixed SVD solve on the calibrated head's frozen hidden features.

    The old corpus and calibrated scorer remain unchanged. Returned score_tails
    are RAW tails: only tail_guided_nominal_search clips and adds immediate reward.
Failures carry continuation_tail_accounting with attempted/completed work.
"""
    if type(corpus) is not ContinuationCorpus or type(calibrated) is not FittedContinuationValue:
        raise TypeError('Exact authenticated corpus and calibrated scorer required')
    if (type(max_scores) is not int or not 1 <= max_scores <= 50000
            or not 0 < _finite(score_seconds, 'score seconds') <= 7200):
        raise ValueError('Bounded scoring declaration required')
    started = time.perf_counter()
    work = {name + suffix: 0 for name in ('hidden_forwards', 'head_readout_forwards', 'svd_calls')
            for suffix in ('_attempted', '_completed')}
    work.update(base_policy_forwards=0, base_optimizer_steps=0,
                head_optimizer_steps=0, head_backward_calls=0)
    def check():
        if time.perf_counter() - started > MAX_SECONDS or (cancelled and cancelled()):
            raise RuntimeError('Continuation tail time/cancellation cap')
        corpus.require(policy); calibrated.require()
    try:
        check()
        prior = calibrated.report
        if (calibrated._guard is not corpus._guard or prior['version'] != CALIBRATION_VERSION
                or prior['corpus_seal'] != corpus.seal
                or prior['base_parameter_hash'] != corpus.metadata['base_parameter_hash']
                or prior['architecture_hash'] != corpus.metadata['architecture_hash']
                or calibrated._worlds != corpus.metadata['worlds']
                or prior['weighting'] != WEIGHTING or prior['fitted_head_hash'] != calibrated._head_hash
                or prior['solver']['relative_cutoff'] != SVD_RELATIVE_CUTOFF):
            raise ValueError('Tail requires the same corpus and fixed direct-calibrated head')
        old = calibrated._head
        if (type(old) is not nn.Sequential or len(old) != 3
                or type(old[0]) is not nn.Linear or type(old[1]) is not nn.ReLU
                or type(old[2]) is not nn.Linear or old[2].out_features != 1
                or old[2].bias is None or old[0].out_features > 512
                or any(p.device.type != 'cpu' or p.dtype != torch.float32 for p in old.parameters())):
            raise ValueError('Exact bounded float32 Linear/ReLU/Linear head required')
        x = np.asarray(corpus.features)
        yq = np.asarray(corpus.targets, dtype=np.float64)
        w = np.asarray(corpus.weights, dtype=np.float64)
        stops = np.asarray(corpus.stop_rows)
        if (x.ndim != 2 or x.shape[1] != old[0].in_features or not 1 <= len(x) <= 864
                or yq.shape != (len(x),) or w.shape != yq.shape or stops.shape != yq.shape
                or stops.dtype != np.bool_ or not all(np.isfinite(a).all() for a in (x, yq, w))
                or np.any(w <= 0) or not np.isclose(w.sum(), 1., rtol=1e-6, atol=1e-7)):
            raise ValueError('Invalid finite corpus/weights/shape')
        rewards, tails, subjects, declaration = _bound_targets(corpus)
        tensor = torch.tensor(np.array(x, copy=True), dtype=torch.float32)
        work['hidden_forwards_attempted'] += 1
        with torch.no_grad():
            hidden = old[1](old[0](tensor)).numpy().astype(np.float64)
        work['hidden_forwards_completed'] += 1
        work['head_readout_forwards_attempted'] += 1
        with torch.no_grad():
            before = old(tensor).squeeze(-1).numpy().astype(np.float64)
        work['head_readout_forwards_completed'] += 1
        if not np.isfinite(hidden).all() or not np.isfinite(before).all():
            raise FloatingPointError('Nonfinite frozen hidden features/readout')
        design = np.column_stack((hidden, np.ones(len(hidden), dtype=np.float64)))
        initial = np.concatenate((old[2].weight.detach().numpy().ravel().astype(np.float64),
                                  old[2].bias.detach().numpy().astype(np.float64)))
        check()
        solution, solver = _minimum_change(design, tails, w, stops, initial, work)
        solver = {**solver,
            'solver': 'float64_single_SVD_minimum_change_about_direct_calibrated_readout'}
        check()
        head = copy.deepcopy(old)
        with torch.no_grad():
            head[2].weight.copy_(torch.from_numpy(solution[:-1].astype(np.float32)[None, :]))
            head[2].bias.copy_(torch.from_numpy(solution[-1:].astype(np.float32)))
        head.eval()
        for p in head.parameters():
            p.requires_grad_(False); p.grad = None
        if parameter_hash(head[0]) != parameter_hash(old[0]):
            raise ValueError('Frozen calibrated hidden layer changed')
        work['head_readout_forwards_attempted'] += 1
        with torch.no_grad():
            after = head(tensor).squeeze(-1).numpy().astype(np.float64)
        work['head_readout_forwards_completed'] += 1
        if not np.isfinite(after).all():
            raise FloatingPointError('Nonfinite deployed float32 tail')
        metrics, per_case = _metrics(before, after, design @ solution, rewards, tails,
                                    yq, w, stops, subjects)
        if (not all(math.isfinite(v) for v in metrics.values())
                or metrics['raw_tail_weighted_mse'] > metrics['initial_raw_tail_weighted_mse']
                   + 1e-6 * max(1., metrics['initial_raw_tail_weighted_mse'])):
            raise FloatingPointError('Deployed tail objective nonfinite or regresses')
        check()
        report = {
            'version': VERSION, 'target': TARGET, 'weighting': WEIGHTING,
            'prediction_semantics': PREDICTION_SEMANTICS, 'deployment': DEPLOYMENT,
            'corpus_seal': corpus.seal, 'manifest_hash': corpus.metadata['manifest_hash'],
            'target_declaration': declaration, 'target_seal': semantic_digest(declaration),
            'base_parameter_hash': corpus._guard.parameters,
            'architecture_hash': corpus._guard.architecture,
            'initial_head_hash': calibrated._head_hash, 'fitted_head_hash': parameter_hash(head),
            'calibrated_report_hash': semantic_digest(prior),
            'frozen_hidden_hash': parameter_hash(old[0]), 'hidden_feature_hash': array_digest(hidden),
            'float64_solution_hash': array_digest(solution), 'solver': solver,
            'weight_sum': float(w.sum()), 'active_weight_sum': float(w[~stops].sum()),
            **metrics, 'per_case': per_case,
            'config': {'relative_svd_cutoff': SVD_RELATIVE_CUTOFF, 'solves': 1,
                       'maximum_seconds': MAX_SECONDS,
                       'initialization': 'minimum_change_about_direct_calibrated_readout'},
            'STOP_prediction': 'fixed_zero',
            'interpretation': 'recorded_continuation_only_unlogged_action_extrapolation_unresolved',
            'costs': {**work, 'elapsed_seconds': time.perf_counter() - started},
        }
        return FittedContinuationTail(corpus._guard, head, corpus.metadata['worlds'], report,
            max_scores=max_scores, seconds=score_seconds, cancelled=cancelled)
    except BaseException as error:
        intact, detail = None, None
        try:
            corpus.require(policy); calibrated.require(); intact = True
        except BaseException as integrity_error:
            intact = False
            detail = type(integrity_error).__name__ + ': ' + str(integrity_error)[:200]
        try:
            error.continuation_tail_accounting = {**work,
                'elapsed_seconds': time.perf_counter() - started,
                'base_corpus_and_calibrated_head_unchanged': intact, 'integrity_error': detail}
        except BaseException:
            pass
        raise
