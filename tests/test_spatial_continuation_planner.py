"""Generated native controls for guarded direct-Q inference; no fitting/data.

The explicit generated context below exercises the admitted interface. It does
not admit a patient or supply evidence of learned costly preparation. Its head
has constant declared outputs so the actual frozen forward path is exercised
without claiming that a generated head was trained.
"""
import copy
from dataclasses import replace

import numpy as np
import pytest
import torch

from resectionlab.core import freeze_json, semantic_digest
from resectionlab.native_spatial_task import NativeSpatialTask, make_native_opening_task
from resectionlab.observed_search import ObservedSearchLimit
from resectionlab.patient_planning_admission import (
    PatientPlanningContext, COHORT_SHA256, _public_observation_binding)
from resectionlab.spatial_policy import SpatialPolicy, SpatialPolicyConfig, parameter_hash
from resectionlab.spatial_continuation_value import (
    FittedContinuationValue, TRAIN_SUBJECTS, _FrozenPolicy, _world)
from resectionlab.spatial_continuation_planner import _choose_positive, q_guided_nominal_search


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def generated_fixture(*, q=1., max_steps=2, costly=False, cancelled=None):
    base = make_native_opening_task(max_steps=max_steps)
    support = base.case.observed_support
    # Same fixed physical opening; every represented support cell is now an
    # explicitly public target, encoded by the generated scan threshold too.
    case = replace(base.case, structural_intensity=np.where(support, .8, 0.).astype(np.float32),
        nominal_target=support, reference_target=support,
        target_derivation="generated public nonzero support signal .8 > .5")
    reward = replace(base.reward_spec, action_cost=100.) if costly else base.reward_spec
    task = NativeSpatialTask(case, max_steps=max_steps, reward=reward, cancelled=cancelled)
    observation = task.observation()
    subject = TRAIN_SUBJECTS[0]
    record = freeze_json({'scope': 'patient_native_planning_experiment', 'role': 'TRAIN',
        'cohort_sha256': COHORT_SHA256, 'subject': subject, 'patient_group': 'ReMIND:' + subject[-3:],
        'evidence_domain': 'generated_interface_control', 'source_hash': task.case.source_hash,
        'decision_model_hash': task.decision_model_hash, 'max_steps': max_steps,
        'max_optimizer_updates': 64, 'observation_track': observation.track,
        'public_observation_binding': _public_observation_binding(observation)})
    context = PatientPlanningContext(record, semantic_digest(record))
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(53)
        policy = SpatialPolicy(SpatialPolicyConfig(encoder_channels=(2, 2),
            hidden_features=4, ray_samples=2, critic_candidate_context=True)).eval()
    head = copy.deepcopy(policy.actor).eval()
    with torch.no_grad():
        for parameter in head.parameters():
            parameter.zero_()
            parameter.requires_grad_(False)
        head[-1].bias.fill_(q)
    guard = _FrozenPolicy(policy)
    report = {'scope': 'generated_constant_head_interface_control_not_fitted_evidence',
        'base_parameter_hash': parameter_hash(policy), 'architecture_hash': policy.architecture_hash,
        'fitted_head_hash': parameter_hash(head)}
    scorer = FittedContinuationValue(guard, head, {subject: _world(context, training=False)},
        report, max_scores=24, seconds=60., cancelled=None)
    return task, context, scorer


@pytest.mark.parametrize('rewards,qs,choice,reason', [
    ([0., -1., 2., 1.], [0., 1000., 1., 3.], 3, 'positive_q_among_actual_positive'),
    # A direct Q already includes immediate reward: adding 100 would choose wrong.
    ([0., 100., 1.], [0., 1., 2.], 2, 'positive_q_among_actual_positive'),
    ([0., 1., 9.], [0., 2., 2.], 1, 'positive_q_among_actual_positive'),
    ([0., 2., 3.], [0., 0., -4.], 2, 'actual_greedy_fallback_no_positive_q'),
    ([0., 3., 3.], [0., -2., -1.], 1, 'actual_greedy_fallback_no_positive_q'),
    ([0., 0., -1.], None, 0, 'no_actual_positive_action'),
])
def test_exact_guard_ranking_stop_and_original_order_ties(rewards, qs, choice, reason):
    # Deliberately reverse lexical IDs: ties must use existing inventory order.
    rows = [{'action_id': name, 'reward': reward}
            for name, reward in zip(('STOP', 'z-last-lexically', 'a-first-lexically', 'middle'), rewards)]
    assert _choose_positive(rows, qs) == (choice, reason)


def test_real_frozen_batch_complete_native_route_and_independent_replay():
    task, context, scorer = generated_fixture()
    before = task.metrics()
    base_hash = parameter_hash(scorer._guard.policy)
    actions, accounting = q_guided_nominal_search(task, context, scorer)
    assert accounting['complete'] and actions and actions[0] != 'STOP'
    productive = sum(d['actual_positive_candidates'] > 0 for d in accounting['decisions'])
    assert productive > 0
    assert accounting['q_batches_started'] == accounting['q_batches_completed'] == productive
    assert accounting['scorer_costs']['policy_forwards_completed'] == productive
    assert accounting['scorer_costs']['head_forwards'] == productive
    assert accounting['successor_CNN_calls'] == 0
    assert accounting['observations_completed'] == len(actions)
    assert accounting['model_transition_calls'] == accounting['committed_transitions'] == len(actions)
    for decision in accounting['decisions']:
        assert decision['all_current_legal_actions_scored']
        assert decision['legal_nonstop_actions'] == len(decision['scores']) - 1
        positive = [s for s in decision['scores'] if s['reward'] > 0.]
        assert decision['selected_action_id'] == (positive[0]['action_id'] if positive else 'STOP')
        assert decision['actual_reward'] >= 0.
        assert decision['transition_status'] == 'committed'
    assert accounting['immediate_scores_completed'] == sum(d['legal_nonstop_actions'] for d in accounting['decisions'])
    assert accounting['native_previews_started'] == accounting['native_previews_completed']
    assert accounting['immediate_reward_added_to_Q'] is False
    assert task.metrics() == before
    assert parameter_hash(scorer._guard.policy) == base_hash
    assert not scorer._guard.policy.actor._forward_pre_hooks
    replay = task.fresh()
    for action in actions:
        replay.step(action)
    assert replay.terminated
    assert replay.metrics()['total_reward'] == accounting['estimated_incremental_return']
    assert replay.independent_geometry_check().feasible


def test_nonpositive_q_fallback_is_exact_existing_greedy_route():
    task, context, scorer = generated_fixture(q=-1.)
    baseline, old = task.observed_greedy_search(seconds=30.)
    actions, accounting = q_guided_nominal_search(task, context, scorer)
    assert actions == baseline
    assert accounting['estimated_incremental_return'] == old['estimated_incremental_return']
    for decision, prior in zip(accounting['decisions'], old['decisions']):
        assert [{k: row[k] for k in prior['scores'][0]} for row in decision['scores']] == prior['scores']
        if decision['actual_positive_candidates']:
            assert decision['selection_reason'] == 'actual_greedy_fallback_no_positive_q'


def test_costly_candidates_cannot_be_rescued_by_large_q_and_stop_needs_no_forward():
    task, context, scorer = generated_fixture(q=1e6, costly=True)
    actions, accounting = q_guided_nominal_search(task, context, scorer)
    assert actions == ('STOP',) and accounting['complete']
    assert accounting['decisions'][0]['legal_nonstop_actions'] > 0
    assert accounting['decisions'][0]['actual_positive_candidates'] == 0
    assert accounting['estimated_incremental_return'] == 0.
    assert accounting['q_batches_started'] == accounting['scorer_costs']['policy_forwards_started'] == 0


def capture_clone(monkeypatch, after=None):
    clones = []
    original = NativeSpatialTask.planning_clone
    def capture(self):
        model = original(self)
        clones.append(model)
        if after is not None:
            after(model)
        return model
    monkeypatch.setattr(NativeSpatialTask, 'planning_clone', capture)
    return clones


def test_transition_cap_retains_cost_and_refuses_partial_plan_restores_preview(monkeypatch):
    task, context, scorer = generated_fixture()
    clones = capture_clone(monkeypatch)
    with pytest.raises(ObservedSearchLimit) as caught:
        q_guided_nominal_search(task, context, scorer, max_calls=1)
    report = caught.value.accounting
    assert not report['complete'] and report['selected_actions'] is None
    assert report['call_cap_reached']
    assert report['committed_transitions'] == report['model_transition_calls'] == 1
    assert len(report['explored_actions']) == 1 and report['explored_actions'][0] != 'STOP'
    assert caught.value.best_sequence == tuple(report['explored_actions'])
    assert 'preview_stroke' not in vars(clones[0]._engine)
    assert task._steps == 0 and not task._history


def test_post_commit_cancellation_is_charged_and_restores_preview(monkeypatch):
    cancelled = [False]
    task, context, scorer = generated_fixture(cancelled=lambda: cancelled[0])
    def after(model):
        commit = model._engine.commit_preview
        def cancel_after_commit(result):
            value = commit(result)
            cancelled[0] = True
            return value
        model._engine.commit_preview = cancel_after_commit
    clones = capture_clone(monkeypatch, after)
    with pytest.raises(InterruptedError) as caught:
        q_guided_nominal_search(task, context, scorer)
    report = caught.value.accounting
    assert report['cancelled'] and not report['complete'] and report['selected_actions'] is None
    assert report['committed_transitions'] == report['model_transition_calls'] == 1
    assert report['decisions'][0]['transition_status'] == 'committed'
    assert len(report['explored_actions']) == 1
    assert 'preview_stroke' not in vars(clones[0]._engine)
    assert task._steps == 0


def test_foreign_certificate_refused_before_inference_or_commit(monkeypatch):
    task, context, scorer = generated_fixture()
    def forge(model):
        preview = next(iter(model._inventory.values()))
        model._engine._preview_digests[id(preview)] = 'foreign-certificate'
    clones = capture_clone(monkeypatch, forge)
    with pytest.raises(ValueError, match='certificate') as caught:
        q_guided_nominal_search(task, context, scorer)
    report = caught.value.accounting
    assert not report['complete'] and report['model_transition_calls'] == report['q_batches_started'] == 0
    assert report['immediate_scores_started'] == 1 and report['immediate_scores_completed'] == 0
    assert 'preview_stroke' not in vars(clones[0]._engine)


@pytest.mark.parametrize('bad', ['foreign_context', 'changed_head'])
def test_deployment_binding_or_frozen_head_refused_before_work(bad):
    task, context, scorer = generated_fixture()
    if bad == 'foreign_context':
        record = context.record(); record['source_hash'] = 'foreign'
        context = PatientPlanningContext(freeze_json(record), semantic_digest(record))
    else:
        scorer._head.train()
    with pytest.raises((ValueError, RuntimeError)) as caught:
        q_guided_nominal_search(task, context, scorer)
    report = caught.value.accounting
    assert not report['complete'] and report['selected_actions'] is None
    assert report['q_batches_started'] == report['native_previews_started'] == report['model_transition_calls'] == 0


def test_failed_q_forward_charged_and_hooks_restored(monkeypatch):
    task, context, scorer = generated_fixture()
    clones = capture_clone(monkeypatch)
    def failed(*args, **kwargs):
        raise RuntimeError('generated forward failure')
    monkeypatch.setattr(scorer._guard.policy, 'forward', failed)
    with pytest.raises(RuntimeError, match='generated forward failure') as caught:
        q_guided_nominal_search(task, context, scorer)
    report = caught.value.accounting
    assert not report['complete'] and report['selected_actions'] is None
    assert report['q_batches_started'] == report['scorer_costs']['policy_forwards_started'] == 1
    assert report['q_batches_completed'] == report['scorer_costs']['policy_forwards_completed'] == 0
    assert report['model_transition_calls'] == 0
    assert 'preview_stroke' not in vars(clones[0]._engine)
    assert not scorer._guard.policy.actor._forward_pre_hooks
