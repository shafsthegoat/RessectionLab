"""Generated tail-decomposition algebra and real shared native planner controls."""
from dataclasses import replace
import itertools

import numpy as np
import pytest
import torch

from resectionlab.core import freeze_json, semantic_digest, thaw_json
from resectionlab.spatial_policy import parameter_hash
from resectionlab.spatial_continuation_value import fit_continuation_value, recorded_action_targets
from resectionlab.spatial_continuation_calibration import calibrate_continuation_readout
from resectionlab.spatial_continuation_tail import (
    VERSION, TARGET, PREDICTION_SEMANTICS, DEPLOYMENT, FittedContinuationTail,
    recorded_tail_targets, _bound_targets, fit_continuation_tail,
)
from resectionlab.spatial_continuation_planner import (
    _choose_positive, _combine_tail, q_guided_nominal_search, tail_guided_nominal_search,
)
from test_spatial_continuation_value import generated_fixture, corpus_for
from test_spatial_continuation_planner import generated_fixture as native_fixture, capture_clone


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def learned_fixture():
    policy, traces, manifest = generated_fixture()
    corpus = corpus_for(policy, traces, manifest)
    q64 = fit_continuation_value(policy, corpus)
    direct = calibrate_continuation_readout(policy, corpus, q64)
    return policy, traces, corpus, direct


def as_tail(direct):
    # Explicit generated constant-head interface, never acquired fitted evidence.
    report = {**thaw_json(direct.report), 'version': VERSION, 'target': TARGET,
        'prediction_semantics': PREDICTION_SEMANTICS, 'deployment': DEPLOYMENT}
    return FittedContinuationTail(direct._guard, direct._head, direct._worlds, report,
        max_scores=24, seconds=60., cancelled=None)


def test_tail_targets_match_exhaustive_subsequent_prefixes():
    for length in range(1, 6):
        for rewards in itertools.product((-2., 0., 3.), repeat=length):
            oracle = tuple(max([0.] + [sum(rewards[i+1:j+1]) for j in range(i+1, length)])
                           for i in range(length))
            tail = recorded_tail_targets(rewards)
            assert tail == oracle
            assert tuple(r + t for r, t in zip(rewards, tail)) == recorded_action_targets(rewards)
            assert tail[-1] == 0.
    assert recorded_tail_targets((-2., 3., -5.)) == (3., 0., 0.)
    assert recorded_tail_targets((-.1,)) == (0.,)  # Finite horizon, not invented continuation.


def test_original_manifest_single_rounding_join_and_wrong_target_refusal():
    policy, traces, manifest = generated_fixture()
    corpus = corpus_for(policy, traces, manifest)
    rewards, tails, subjects, declaration = _bound_targets(corpus)
    np.testing.assert_array_equal((rewards + tails).astype(np.float32), corpus.targets)
    assert tails.dtype == np.float64 and not tails.flags.writeable
    assert any(t != float(q) - r for t, q, r in zip(tails, corpus.targets, rewards))
    assert declaration['rows'] == len(corpus.targets) == len(subjects)
    changed = np.array(corpus.targets, copy=True); changed[0] += .01
    with pytest.raises(ValueError, match='target, STOP or TRAIN row join'):
        _bound_targets(replace(corpus, targets=changed))


@pytest.mark.parametrize('rewards, tails, choice', [
    ([0., 4., 1.], [0., 0., 4.], 2),  # Adding reward twice incorrectly selects action1.
    ([0., 2., 3.], [0., -100., -2.], 2),
    ([0., 3., 2.], [0., -100., -2.], 1),
    ([0., 2., 1.], [0., 1., 2.], 1),  # Stable inventory tie, no lexical sorting.
    ([0., -1., 2.], [0., 1e6, 0.], 2),  # Unchanged actual-positive guard.
])
def test_exact_reward_once_tail_clamp_and_guard(rewards, tails, choice):
    rows = [{'reward': r} for r in rewards]
    combined = _combine_tail(rows, tuple(tails))
    assert combined[0] == 0.
    assert _choose_positive(rows, combined)[0] == choice
    for i in range(1, len(rows)):
        assert combined[i] == rewards[i] + max(0., tails[i])


def test_fit_freezes_originals_and_reports_real_deployed_tail_objective():
    policy, traces, corpus, direct = learned_fixture()
    before = (parameter_hash(policy), parameter_hash(direct._head), parameter_hash(direct._head[0]),
              corpus.seal, semantic_digest(direct.report))
    fitted = fit_continuation_tail(policy, corpus, direct)
    assert type(fitted) is FittedContinuationTail
    assert fitted._head is not direct._head
    assert parameter_hash(fitted._head[0]) == before[2]
    rewards, targets, _, declaration = _bound_targets(corpus)
    assert fitted.report['target_declaration'] == declaration
    assert fitted.report['target_seal'] == semantic_digest(declaration)
    with torch.no_grad():
        predicted = fitted._head(torch.tensor(np.array(corpus.features, copy=True))).squeeze(-1).numpy().astype(np.float64)
    predicted[corpus.stop_rows] = 0.
    actual = float(np.dot(corpus.weights.astype(np.float64), (predicted - targets) ** 2))
    assert fitted.report['raw_tail_weighted_mse'] == pytest.approx(actual, abs=1e-14)
    combined = rewards + np.maximum(0., predicted)
    qerror = float(np.dot(corpus.weights.astype(np.float64), (combined - corpus.targets) ** 2))
    assert fitted.report['reconstructed_Q_weighted_mse'] == pytest.approx(qerror, abs=1e-14)
    assert fitted.report['costs']['svd_calls_completed'] == 1
    assert fitted.report['costs']['base_policy_forwards'] == 0
    assert fitted.report['costs']['head_optimizer_steps'] == 0
    assert fitted.report['costs']['head_backward_calls'] == 0
    trace = traces[0][2]
    with pytest.raises(TypeError, match='not direct Q'):
        fitted.score_actions(trace.transitions[0].observation, trace.context)
    assert fitted.costs['policy_forwards_started'] == 0
    values = fitted.score_tails(trace.transitions[0].observation, trace.context)
    assert values[0] == 0. and fitted.costs['policy_forwards_completed'] == 1
    assert (parameter_hash(policy), parameter_hash(direct._head), parameter_hash(direct._head[0]),
            corpus.seal, semantic_digest(direct.report)) == before
    assert all(not p.requires_grad and p.grad is None for p in fitted._head.parameters())


def test_tail_and_direct_types_and_semantic_declarations_cannot_cross():
    task, context, direct = native_fixture()
    tail = as_tail(direct)
    with pytest.raises(TypeError, match='matching fitted scorer'):
        q_guided_nominal_search(task, context, tail)
    with pytest.raises(TypeError, match='matching fitted scorer'):
        tail_guided_nominal_search(task, context, direct)
    bad = thaw_json(tail.report); bad['deployment'] = 'add_reward_twice'
    with pytest.raises(ValueError, match='declaration required'):
        FittedContinuationTail(direct._guard, direct._head, direct._worlds, bad,
            max_scores=24, seconds=60., cancelled=None)
    tail.report = freeze_json(bad)
    with pytest.raises(ValueError, match='tail declaration changed'):
        tail_guided_nominal_search(task, context, tail)
    assert tail.costs['policy_forwards_started'] == 0


def test_zero_or_negative_tail_reuses_greedy_route_no_extra_native_work():
    task, context, direct = native_fixture(q=-1.)
    old_actions, old = q_guided_nominal_search(task, context, direct)
    task2, context2, raw_tail = native_fixture(q=-1.)
    tail = as_tail(raw_tail)
    actions, report = tail_guided_nominal_search(task2, context2, tail)
    assert actions == old_actions
    assert report['estimated_incremental_return'] == old['estimated_incremental_return']
    for key in ('native_previews_started', 'native_previews_completed', 'immediate_scores_completed',
                'observation_requests', 'q_batches_started', 'q_batches_completed',
                'model_transition_calls', 'committed_transitions'):
        assert report[key] == old[key]
    assert report['scorer_costs']['policy_forwards_completed'] == old['scorer_costs']['policy_forwards_completed']
    assert report['additional_native_previews_for_tail'] == report['additional_observations_for_tail'] == 0
    for decision in report['decisions']:
        for row in decision['scores']:
            if row['raw_predicted_tail'] is not None:
                assert row['clipped_predicted_tail'] == max(0., row['raw_predicted_tail'])
                assert row['combined_nominal_q'] == row['reward'] + row['clipped_predicted_tail']
                assert row['predicted_q'] == row['combined_nominal_q']
    replay = task2.fresh()
    for action in actions: replay.step(action)
    assert replay.terminated and replay.independent_geometry_check().feasible
    assert replay.metrics()['total_reward'] == report['estimated_incremental_return']


def test_guard_forces_stop_without_forward_or_tail_credit():
    task, context, direct = native_fixture(q=1e6, costly=True)
    tail = as_tail(direct)
    actions, report = tail_guided_nominal_search(task, context, tail)
    assert actions == ('STOP',) and report['estimated_incremental_return'] == 0.
    assert report['q_batches_started'] == tail.costs['policy_forwards_started'] == 0
    stop = report['decisions'][0]['scores'][0]
    assert stop['raw_predicted_tail'] == stop['clipped_predicted_tail'] == stop['combined_nominal_q'] == 0.
    assert report['decisions'][0]['selected_q'] == 0.


def test_fit_source_and_solver_failures_retain_attempted_work(monkeypatch):
    policy, _, corpus, direct = learned_fixture()
    before = parameter_hash(direct._head)
    with pytest.raises(ValueError, match='Detached corpus') as changed:
        fit_continuation_tail(policy, replace(corpus, seal='foreign'), direct)
    assert changed.value.continuation_tail_accounting['svd_calls_attempted'] == 0
    def fail(*args, **kwargs):
        raise np.linalg.LinAlgError('generated tail solve failure')
    monkeypatch.setattr(np.linalg, 'svd', fail)
    with pytest.raises(np.linalg.LinAlgError, match='generated tail') as caught:
        fit_continuation_tail(policy, corpus, direct)
    costs = caught.value.continuation_tail_accounting
    assert costs['svd_calls_attempted'] == 1 and costs['svd_calls_completed'] == 0
    assert costs['hidden_forwards_completed'] == 1
    assert costs['base_corpus_and_calibrated_head_unchanged'] is True
    assert parameter_hash(direct._head) == before


def test_failed_tail_forward_is_charged_and_native_hook_restored(monkeypatch):
    task, context, direct = native_fixture()
    tail = as_tail(direct)
    clones = capture_clone(monkeypatch)
    def fail(*args, **kwargs):
        raise RuntimeError('generated tail forward failure')
    monkeypatch.setattr(tail._guard.policy, 'forward', fail)
    with pytest.raises(RuntimeError, match='generated tail forward') as caught:
        tail_guided_nominal_search(task, context, tail)
    report = caught.value.accounting
    assert not report['complete'] and report['selected_actions'] is None
    assert report['q_batches_started'] == tail.costs['policy_forwards_started'] == 1
    assert report['q_batches_completed'] == tail.costs['policy_forwards_completed'] == 0
    assert report['model_transition_calls'] == 0
    assert 'preview_stroke' not in vars(clones[0]._engine)
    assert not tail._guard.policy.actor._forward_pre_hooks
