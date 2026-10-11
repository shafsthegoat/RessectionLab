"""One fixed readout calibration of an already fitted logged-continuation head.

The encoder and fitted hidden layer stay fixed. This is a truncated float64
weighted least-squares correction about the old readout, retaining its nullspace
component. It tests optimization error on recorded actions, not optimal action
values or reliability on unlogged actions. STOP remains identically zero.
"""
from __future__ import annotations

import copy
import math
import time

import numpy as np
import torch
from torch import nn

from .core import array_digest, semantic_digest, thaw_json
from .spatial_policy import parameter_hash
from .spatial_continuation_value import (
    ContinuationCorpus, FittedContinuationValue, TRAIN_SUBJECTS, TARGET, WEIGHTING,
    _finite,
)

VERSION = 'frozen-continuation-minimum-change-readout-v1'
SVD_RELATIVE_CUTOFF = 1e-8
MAX_SECONDS = 60.


def _mse(prediction, targets, weights, stops):
    error = np.where(stops, 0., prediction) - targets
    return float(np.dot(weights, error * error))


def _minimum_change(design, targets, weights, stops, initial, work):
    """Single truncated SVD; original global mass, unpenalized bias in design."""
    active = ~stops
    if not np.any(active):
        raise ValueError('No nonSTOP recorded action to calibrate')
    weighted = design[active] * np.sqrt(weights[active, None])
    residual = (targets[active] - design[active] @ initial) * np.sqrt(weights[active])
    work['svd_calls_attempted'] += 1
    u, singular, vt = np.linalg.svd(weighted, full_matrices=False)
    work['svd_calls_completed'] += 1
    if (not np.isfinite(singular).all() or not np.isfinite(u).all()
            or not np.isfinite(vt).all() or singular.size == 0 or singular[0] <= 0):
        raise FloatingPointError('Nonfinite or empty calibration decomposition')
    threshold = float(singular[0]) * SVD_RELATIVE_CUTOFF
    keep = singular > threshold
    rank = int(keep.sum())
    if rank == 0:
        raise FloatingPointError('Calibration has no numerically retained direction')
    condition = float(singular[0] / singular[keep][-1])
    if not math.isfinite(condition) or condition > 1. / SVD_RELATIVE_CUTOFF:
        raise FloatingPointError('Ill-conditioned retained calibration subspace')
    delta = vt[keep].T @ ((u[:, keep].T @ residual) / singular[keep])
    solution = initial + delta
    if not np.isfinite(solution).all():
        raise FloatingPointError('Nonfinite calibrated readout')
    remaining = weighted @ solution - targets[active] * np.sqrt(weights[active])
    # This is stationarity ONLY in the retained subspace, not a claim about
    # numerical directions deliberately omitted by the fixed cutoff.
    retained_gradient = vt[keep] @ (weighted.T @ remaining)
    report = {
        'solver': 'float64_single_SVD_minimum_change_about_fitted64_readout',
        'relative_cutoff': SVD_RELATIVE_CUTOFF, 'absolute_cutoff': threshold,
        'rank': rank, 'design_columns': int(design.shape[1]),
        'active_rows': int(active.sum()), 'STOP_rows_excluded': int(stops.sum()),
        'singular_values': singular.tolist(),
        'discarded_or_null_directions': int(design.shape[1] - rank),
        'retained_condition_number': condition,
        'all_returned_singular_condition_number': (
            float(singular[0] / singular[-1]) if singular[-1] > 0
            and math.isfinite(float(singular[0] / singular[-1])) else None),
        'readout_delta_l2': float(np.linalg.norm(delta)),
        'retained_gradient_l2': float(np.linalg.norm(retained_gradient)),
        'weighted_residual_l2_float64': float(np.linalg.norm(remaining)),
        'truncated_subspace_only': bool(rank < design.shape[1]),
    }
    if not all(math.isfinite(report[k]) for k in
               ('readout_delta_l2', 'retained_gradient_l2', 'weighted_residual_l2_float64')):
        raise FloatingPointError('Nonfinite calibration residual')
    return solution, report


def _case_metrics(corpus, before, after, solved):
    manifest = corpus.metadata['manifest']
    offset = 0
    groups = {s: [] for s in TRAIN_SUBJECTS}
    for trace in manifest:
        length = len(trace['decisions'])
        groups[trace['world']['subject']].extend(range(offset, offset + length))
        offset += length
    if offset != len(corpus.targets) or any(not indices for indices in groups.values()):
        raise ValueError('Complete fixed TRAIN row ownership differs')
    result = {}
    for subject, indices in groups.items():
        y = np.asarray(corpus.targets[indices], dtype=np.float64)
        w = np.asarray(corpus.weights[indices], dtype=np.float64)
        stop = corpus.stop_rows[indices]
        mass = float(w.sum())
        values = [_mse(p[indices], y, w, stop) for p in (before, after, solved)]
        result[subject] = {
            'rows': len(indices), 'weight_mass': mass,
            'weighted_mse_before': values[0] / mass,
            'weighted_mse_after_deployed_float32': values[1] / mass,
            'weighted_mse_after_float64': values[2] / mass,
            'weighted_SSE_before': values[0],
            'weighted_SSE_after_deployed_float32': values[1],
        }
    return result


def calibrate_continuation_readout(policy, corpus, fitted64, *, cancelled=None,
        max_scores=12000, score_seconds=3600.):
    """Return a separate frozen scorer, leaving corpus/base/fitted64 untouched.

No new geometry, observation, policy forward, gradient or optimizer is used.
One hidden-feature batch and two full-head validation batches are charged. A
failed solve raises with bounded ``continuation_calibration_accounting``; it
never silently returns the old scorer as an apparent successful calibration.
"""
    if type(corpus) is not ContinuationCorpus or type(fitted64) is not FittedContinuationValue:
        raise TypeError('Exact authenticated corpus and fitted64 scorer required')
    if (type(max_scores) is not int or not 1 <= max_scores <= 50000
            or not 0 < _finite(score_seconds, 'score seconds') <= 7200):
        raise ValueError('Bounded scoring declaration required')
    started = time.perf_counter()
    work = {key + suffix: 0 for key in ('hidden_forwards', 'head_readout_forwards', 'svd_calls')
            for suffix in ('_attempted', '_completed')}
    work.update(base_policy_forwards=0, base_optimizer_steps=0,
                head_optimizer_steps=0, head_backward_calls=0)
    def check():
        if time.perf_counter() - started > MAX_SECONDS or (cancelled and cancelled()):
            raise RuntimeError('Calibration time/cancellation cap')
        corpus.require(policy)
        fitted64.require()
    try:
        check()
        prior = fitted64.report
        if (fitted64._guard is not corpus._guard
                or prior['corpus_seal'] != corpus.seal
                or prior['base_parameter_hash'] != corpus.metadata['base_parameter_hash']
                or prior['architecture_hash'] != corpus.metadata['architecture_hash']
                or fitted64._worlds != corpus.metadata['worlds']
                or prior['target'] != TARGET or prior['weighting'] != WEIGHTING
                or prior['config']['updates'] != 64 or len(prior['updates']) != 64
                or prior['fitted_head_hash'] != fitted64._head_hash):
            raise ValueError('Calibration requires this corpus and exact completed64 head')
        old = fitted64._head
        if (type(old) is not nn.Sequential or len(old) != 3
                or type(old[0]) is not nn.Linear or type(old[1]) is not nn.ReLU
                or type(old[2]) is not nn.Linear or old[2].out_features != 1
                or old[2].bias is None or old[0].out_features > 512
                or any(p.device.type != 'cpu' or p.dtype != torch.float32 for p in old.parameters())):
            raise ValueError('Exact bounded float32 Linear/ReLU/Linear head required')
        x = np.asarray(corpus.features)
        y, w = (np.asarray(a, dtype=np.float64) for a in (corpus.targets, corpus.weights))
        stops = np.asarray(corpus.stop_rows)
        if (x.ndim != 2 or x.shape[1] != old[0].in_features or not 1 <= len(x) <= 864
                or y.shape != (len(x),) or w.shape != y.shape or stops.shape != y.shape
                or stops.dtype != np.bool_ or not np.isfinite(x).all()
                or not np.isfinite(y).all() or not np.isfinite(w).all()
                or np.any(w <= 0) or not np.isclose(w.sum(), 1., rtol=1e-6, atol=1e-7)
                or np.any(y[stops] != 0)):
            raise ValueError('Invalid finite corpus, weights, shape or STOP0 targets')
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
        solution, solver = _minimum_change(design, y, w, stops, initial, work)
        check()
        head = copy.deepcopy(old)
        with torch.no_grad():
            head[2].weight.copy_(torch.from_numpy(solution[:-1].astype(np.float32)[None, :]))
            head[2].bias.copy_(torch.from_numpy(solution[-1:].astype(np.float32)))
        head.eval()
        for p in head.parameters():
            p.requires_grad_(False)
            p.grad = None
        if parameter_hash(head[0]) != parameter_hash(old[0]):
            raise ValueError('Frozen hidden layer changed during calibration')
        work['head_readout_forwards_attempted'] += 1
        with torch.no_grad():
            after = head(tensor).squeeze(-1).numpy().astype(np.float64)
        work['head_readout_forwards_completed'] += 1
        if not np.isfinite(after).all():
            raise FloatingPointError('Nonfinite deployed float32 calibration')
        solved = design @ solution
        mse_before, mse_after, mse_solved = [_mse(p, y, w, stops) for p in (before, after, solved)]
        if (not all(math.isfinite(v) for v in (mse_before, mse_after, mse_solved))
                or mse_after > mse_before + 1e-6 * max(1., mse_before)):
            raise FloatingPointError('Deployed calibration objective is nonfinite or regresses')
        check()
        report = {
            'version': VERSION, 'target': TARGET, 'weighting': WEIGHTING,
            'corpus_seal': corpus.seal, 'manifest_hash': corpus.metadata['manifest_hash'],
            'base_parameter_hash': corpus._guard.parameters,
            'architecture_hash': corpus._guard.architecture,
            'initial_head_hash': fitted64._head_hash, 'fitted_head_hash': parameter_hash(head),
            'fitted64_report_hash': semantic_digest(prior),
            'frozen_hidden_hash': parameter_hash(old[0]),
            'hidden_feature_hash': array_digest(hidden),
            'float64_solution_hash': array_digest(solution),
            'solver': solver,
            'weight_sum': float(w.sum()), 'active_weight_sum': float(w[~stops].sum()),
            'initial_weighted_mse': mse_before, 'final_weighted_mse': mse_after,
            'float64_solution_weighted_mse': mse_solved,
            'float32_deployment_mse_minus_float64': mse_after - mse_solved,
            'per_case': _case_metrics(corpus, before, after, solved),
            'config': {'relative_svd_cutoff': SVD_RELATIVE_CUTOFF,
                       'maximum_seconds': MAX_SECONDS, 'solves': 1,
                       'initialization': 'minimum_change_about_exact_fitted64_final_readout'},
            'interpretation': 'recorded_action_fit_only_unlogged_action_extrapolation_unresolved',
            'costs': {**work, 'elapsed_seconds': time.perf_counter() - started},
        }
        return FittedContinuationValue(corpus._guard, head, corpus.metadata['worlds'], report,
            max_scores=max_scores, seconds=score_seconds, cancelled=cancelled)
    except BaseException as error:
        intact, detail = None, None
        try:
            corpus.require(policy)
            fitted64.require()
            intact = True
        except BaseException as integrity_error:
            intact = False
            detail = type(integrity_error).__name__ + ': ' + str(integrity_error)[:200]
        try:
            error.continuation_calibration_accounting = {
                **work, 'elapsed_seconds': time.perf_counter() - started,
                'base_corpus_and_original_head_unchanged': intact, 'integrity_error': detail}
        except BaseException:
            pass
        raise
