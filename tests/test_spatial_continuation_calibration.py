"""Tiny generated calibration controls; no acquired sources or saved weights."""
from dataclasses import replace
import copy

import numpy as np
import pytest
import torch

from resectionlab.core import semantic_digest, thaw_json
from resectionlab.spatial_policy import parameter_hash
from resectionlab.spatial_continuation_value import fit_continuation_value
from resectionlab.spatial_continuation_calibration import (
    _minimum_change, calibrate_continuation_readout, SVD_RELATIVE_CUTOFF,
)
from test_spatial_continuation_value import generated_fixture, corpus_for


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def counters():
    return {'svd_calls_attempted': 0, 'svd_calls_completed': 0}


def fitted_fixture():
    policy, traces, manifest = generated_fixture()
    corpus = corpus_for(policy, traces, manifest)
    fitted = fit_continuation_value(policy, corpus)
    return policy, traces, corpus, fitted


def test_weighted_solution_matches_normal_equation_oracle_stop_excluded():
    # No zero clipping of mandatory action targets. Huge STOP features cannot
    # influence the solve, and original mass is retained instead of normalized.
    design = np.array([[0., 1., 1.], [1., 0., 1.], [2., 2., 1.],
                       [-1., 3., 1.], [1e20, -1e20, 1.]])
    y = np.array([-2., 4., 3., 1., 0.])
    w = np.array([.1, .15, .2, .25, .3])
    stops = np.array([False] * 4 + [True])
    initial = np.array([3., -2., 4.])
    a = design[:4]
    oracle = np.linalg.solve(a.T @ (w[:4, None] * a), a.T @ (w[:4] * y[:4]))
    work = counters()
    solution, report = _minimum_change(design, y, w, stops, initial, work)
    np.testing.assert_allclose(solution, oracle, rtol=1e-12, atol=1e-12)
    assert report['rank'] == 3 and report['STOP_rows_excluded'] == 1
    assert report['weighted_residual_l2_float64'] ** 2 == pytest.approx(
        np.dot(w[:4], (a @ oracle - y[:4]) ** 2))
    assert work == {'svd_calls_attempted': 1, 'svd_calls_completed': 1}


def test_minimum_change_retains_duplicate_and_dead_column_nullspace():
    z = np.arange(5, dtype=np.float64)
    design = np.column_stack((z, z, np.zeros(5), np.ones(5)))
    initial = np.array([5., -3., 7., 2.])
    solution, report = _minimum_change(design, 6. * z - 1., np.ones(5) / 5.,
        np.zeros(5, dtype=bool), initial, counters())
    np.testing.assert_allclose(solution, [7., -1., 7., -1.], atol=1e-12)
    assert solution[0] - solution[1] == pytest.approx(initial[0] - initial[1])
    assert solution[2] == initial[2]
    assert report['rank'] == 2 and report['discarded_or_null_directions'] == 2
    assert report['truncated_subspace_only'] is True


def test_declared_cutoff_preserves_unresolved_tiny_direction():
    design = np.diag([1., 1e-10])
    initial = np.array([3., 7.])
    solution, report = _minimum_change(design, np.array([2., 4.]),
        np.array([.5, .5]), np.zeros(2, dtype=bool), initial, counters())
    np.testing.assert_allclose(solution, [2., 7.], atol=1e-12)
    assert report['rank'] == 1
    assert report['relative_cutoff'] == SVD_RELATIVE_CUTOFF == 1e-8
    assert report['retained_condition_number'] == 1.
    assert report['weighted_residual_l2_float64'] > 2.


def test_real_feature_corpus_calibration_freeze_scoring_and_case_metrics():
    policy, traces, corpus, baseline = fitted_fixture()
    before = (parameter_hash(policy), parameter_hash(baseline._head), corpus.seal,
              semantic_digest(baseline.report), parameter_hash(baseline._head[0]))
    original_parameters = [(id(p), p._version, p.requires_grad) for p in baseline._head.parameters()]
    result = calibrate_continuation_readout(policy, corpus, baseline, max_scores=1)
    report = result.report
    assert result is not baseline and result._head is not baseline._head
    assert result._guard is baseline._guard
    assert report['initial_head_hash'] == before[1]
    assert report['final_weighted_mse'] <= report['initial_weighted_mse'] + 1e-6
    assert report['float64_solution_weighted_mse'] <= report['initial_weighted_mse'] + 1e-10
    assert report['costs']['hidden_forwards_completed'] == 1
    assert report['costs']['head_readout_forwards_completed'] == 2
    assert report['costs']['svd_calls_completed'] == 1
    for name in ('base_policy_forwards', 'base_optimizer_steps', 'head_optimizer_steps', 'head_backward_calls'):
        assert report['costs'][name] == 0
    with torch.no_grad():
        prediction = result._head(torch.tensor(np.array(corpus.features, copy=True))).squeeze(-1).numpy()
    prediction[corpus.stop_rows] = 0.
    actual_mse = np.sum(corpus.weights.astype(np.float64) *
                        (prediction.astype(np.float64) - corpus.targets) ** 2)
    assert report['final_weighted_mse'] == pytest.approx(actual_mse, abs=1e-14)
    assert sum(r['weighted_SSE_after_deployed_float32'] for r in report['per_case'].values()) == pytest.approx(actual_mse)
    assert sum(r['rows'] for r in report['per_case'].values()) == len(corpus.targets)
    trace = traces[0][2]
    scores = result.score_actions(trace.transitions[0].observation, trace.context)
    assert scores[0] == 0. and all(np.isfinite(s) for s in scores)
    assert result.costs['policy_forwards_completed'] == 1
    with pytest.raises(RuntimeError, match='cap'):
        result.score_actions(trace.transitions[0].observation, trace.context)
    assert (parameter_hash(policy), parameter_hash(baseline._head), corpus.seal,
            semantic_digest(baseline.report), parameter_hash(baseline._head[0])) == before
    assert [(id(p), p._version, p.requires_grad) for p in baseline._head.parameters()] == original_parameters
    assert parameter_hash(result._head[0]) == before[-1]
    assert all(p.grad is None and not p.requires_grad for p in result._head.parameters())
    assert not any(m.training for m in result._head.modules())


def test_foreign_corpus_seal_and_mutated_head_rejected():
    policy, _, corpus, baseline = fitted_fixture()
    with pytest.raises(ValueError, match='Detached corpus'):
        calibrate_continuation_readout(policy, replace(corpus, seal='foreign'), baseline)
    with torch.no_grad():
        baseline._head[0].weight.add_(.1)
    with pytest.raises(ValueError, match='Frozen fitted'):
        calibrate_continuation_readout(policy, corpus, baseline)


def test_svd_failure_preserves_baseline_and_reports_attempted_work(monkeypatch):
    policy, _, corpus, baseline = fitted_fixture()
    before = parameter_hash(baseline._head)
    def failure(*args, **kwargs):
        raise np.linalg.LinAlgError('generated decomposition failure')
    monkeypatch.setattr(np.linalg, 'svd', failure)
    with pytest.raises(np.linalg.LinAlgError, match='generated decomposition') as caught:
        calibrate_continuation_readout(policy, corpus, baseline)
    costs = caught.value.continuation_calibration_accounting
    assert costs['svd_calls_attempted'] == 1 and costs['svd_calls_completed'] == 0
    assert costs['hidden_forwards_completed'] == 1
    assert costs['head_readout_forwards_completed'] == 1
    assert costs['base_corpus_and_original_head_unchanged'] is True
    assert parameter_hash(baseline._head) == before


def test_deployed_nonfinite_solution_is_refused_without_baseline_mutation(monkeypatch):
    import resectionlab.spatial_continuation_calibration as module
    policy, _, corpus, baseline = fitted_fixture()
    before = parameter_hash(baseline._head)
    original = module._minimum_change
    def overflowing(*args, **kwargs):
        solution, report = original(*args, **kwargs)
        solution[:] = 1e39  # Finite float64, unsafe deployed float32.
        return solution, report
    monkeypatch.setattr(module, '_minimum_change', overflowing)
    with np.errstate(over='ignore'):
        with pytest.raises(FloatingPointError, match='Nonfinite deployed') as caught:
            calibrate_continuation_readout(policy, corpus, baseline)
    assert caught.value.continuation_calibration_accounting['base_corpus_and_original_head_unchanged']
    assert parameter_hash(baseline._head) == before


def test_cancellation_has_no_hidden_or_solve_work():
    policy, _, corpus, baseline = fitted_fixture()
    with pytest.raises(RuntimeError, match='cancellation') as caught:
        calibrate_continuation_readout(policy, corpus, baseline, cancelled=lambda: True)
    costs = caught.value.continuation_calibration_accounting
    assert costs['hidden_forwards_attempted'] == costs['svd_calls_attempted'] == 0
    assert costs['base_corpus_and_original_head_unchanged'] is True
