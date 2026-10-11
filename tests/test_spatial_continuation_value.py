"""Generated public DTO algebra/lifecycle controls, not acquired trace evidence.

The explicit synthetic _TRACE capability below tests the module's owned-collector
interface. Actual runners must reconstruct native traces and join saved evidence.
"""
from dataclasses import replace
import copy
import gc
import itertools
import weakref

import numpy as np
import pytest
import torch

from resectionlab.core import freeze_json, semantic_digest, thaw_json
from resectionlab.geometry import AccessWindow, ToolGeometry
from resectionlab.patient_planning_admission import (
    PatientPlanningContext, COHORT_SHA256, _public_observation_binding)
from resectionlab.patient_planning_learning import PatientTrainingTrace, _TRACE
from resectionlab.spatial_observations import (ObservedChannel, SpatialInputs,
    ObservedProcedureState, SpatialAction, build_spatial_observation)
from resectionlab.spatial_policy import SpatialPolicy, SpatialPolicyConfig, SpatialTransition, parameter_hash
from resectionlab.spatial_continuation_value import (
    TRAIN_SUBJECTS, ContinuationCorpusBuilder, ContinuationFitConfig,
    recorded_prefix_targets, recorded_action_targets, trace_descriptor,
    fit_continuation_value)


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def generated_fixture():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(23)
        policy = SpatialPolicy(SpatialPolicyConfig(encoder_channels=(2, 2),
            hidden_features=4, ray_samples=2, critic_candidate_context=True)).eval()
    traces = []
    for index, subject in enumerate(TRAIN_SUBJECTS):
        shape = (4, 4, 4)
        source = semantic_digest({'generated_fixture': 'continuation-only', 'subject': subject})
        target = np.zeros(shape, np.float32); target[3, index, 3] = 1.
        channels = {
            'structural_intensity': ObservedChannel(np.arange(64).reshape(shape) / 64., source_kind='observed_scan'),
            'nominal_tissue': ObservedChannel(np.ones(shape), source_kind='supplied_annotation'),
            'nominal_target': ObservedChannel(target, source_kind='supplied_annotation'),
            'observed_cavity': ObservedChannel(np.zeros(shape), source_kind='observed_procedure_state')}
        inputs = SpatialInputs(channels, np.eye(4), 'annotation_assisted', source)
        access = AccessWindow((0., 1., 1.), (1., 0., 0.), 3.)
        tool = ToolGeometry('generated_tool', .4, .2, 20., tip_length_mm=.5)
        actions = (SpatialAction('STOP'), SpatialAction('candidate', (0., 1., 1.), (2., 1., 1.), tool),
                   SpatialAction('alternative', (0., 1., 1.), (3., 1., 1.), tool))
        base = build_spatial_observation(inputs, actions, ObservedProcedureState(access, 0, 3))
        record = freeze_json({'scope': 'patient_native_planning_experiment', 'role': 'TRAIN',
            'cohort_sha256': COHORT_SHA256, 'subject': subject, 'patient_group': 'ReMIND:' + subject[-3:],
            'evidence_domain': 'generated_interface_control', 'source_hash': source,
            'decision_model_hash': semantic_digest({'generated_world': subject}),
            'max_optimizer_updates': 64, 'max_steps': 3, 'observation_track': base.track,
            'public_observation_binding': _public_observation_binding(base)})
        context = PatientPlanningContext(record, semantic_digest(record))
        for kind, rewards in [('teacher', (.1, .2, 0.)), ('RL', (-.2, .1, 0.))]:
            rows = []
            for step, reward in enumerate(rewards):
                state = base.state_features.copy(); state[0] = step
                observation = replace(base, state_features=state)
                action = 'STOP' if step == 2 else 'candidate'
                rows.append(SpatialTransition(observation, action, reward, step == 2))
            history = freeze_json([{'action_id': r.action_id, 'reward': r.reward,
                **({} if r.action_id == 'STOP' else {'outcome_scope': 'permitted_nominal_model'})}
                for r in rows])
            behavior = None if kind == 'teacher' else semantic_digest({'generated_behavior': subject})
            seal = semantic_digest({'context': context.fingerprint,
                'observations': [r.observation.fingerprint for r in rows],
                'history': history, 'behavior_parameter_hash': behavior})
            trace = PatientTrainingTrace(context, tuple(rows), history, behavior, seal, _TRACE).require()
            traces.append((subject + '/' + kind, kind, trace))
    manifest = freeze_json([trace_descriptor(key, kind, trace) for key, kind, trace in traces])
    return policy, traces, manifest


def builder_for(policy, manifest, **kwargs):
    return ContinuationCorpusBuilder(policy, manifest, manifest_hash=semantic_digest(manifest), **kwargs)


def corpus_for(policy, traces, manifest):
    builder = builder_for(policy, manifest)
    # Ingest out of manifest order; fitting order is still fixed by declaration.
    for key, _, trace in reversed(traces):
        builder.add_trace(key, trace)
    return builder.finalize()


def test_targets_match_independent_exhaustive_prefix_oracle():
    for length in range(1, 5):
        for rewards in itertools.product((-2., 0., 3.), repeat=length):
            q = tuple(max(sum(rewards[i:j + 1]) for j in range(i, length)) for i in range(length))
            v = tuple(max(0., value) for value in q)
            assert recorded_action_targets(rewards) == q
            assert recorded_prefix_targets(rewards) == v
    assert recorded_action_targets((-2., .5, 0.)) == (-1.5, .5, 0.)
    assert recorded_action_targets((-1., 3., -5., 0.)) == (2., 3., -5., 0.)


@pytest.mark.parametrize('bad', [(), (True,), (float('nan'),), (float('inf'),), ('1',)])
def test_targets_reject_invalid_rewards(bad):
    with pytest.raises(ValueError):
        recorded_action_targets(bad)


def test_streaming_features_equal_real_actor_input_and_no_trace_retained():
    policy, traces, manifest = generated_fixture()
    captured = []
    hook = policy.actor.register_forward_pre_hook(lambda _, args: captured.append(args[0].detach().clone()))
    with torch.no_grad():
        policy(traces[0][2].transitions[0].observation)
    hook.remove()
    builder = builder_for(policy, manifest)
    weak = weakref.ref(traces[0][2])
    expected = captured[0][1].numpy()
    for key, _, trace in traces:
        builder.add_trace(key, trace)
    del trace
    traces.clear(); gc.collect()
    assert weak() is None
    corpus = builder.finalize()
    np.testing.assert_array_equal(corpus.features[0], expected)
    assert corpus.features.shape[0] == 24
    assert corpus.metadata['costs']['policy_forwards_completed'] == 24
    assert corpus.metadata['STOP_rows'] == 8
    assert corpus.targets[3] == pytest.approx(-.1)
    assert corpus.targets[0] == pytest.approx(.3)  # Same observation, different recorded labels.
    assert np.sum(corpus.weights) == pytest.approx(1.)
    for offset in range(0, 24, 6):
        assert np.sum(corpus.weights[offset:offset + 6]) == pytest.approx(.25)
    for array in (corpus.features, corpus.targets, corpus.weights, corpus.stop_rows):
        with pytest.raises(ValueError):
            array.setflags(write=True)
    assert not policy.actor._forward_pre_hooks
    with pytest.raises(ValueError, match='consumed'):
        builder.finalize()


def test_head_only_fit_scoring_freeze_and_costs():
    policy, traces, manifest = generated_fixture()
    before = parameter_hash(policy)
    flags = [(p.requires_grad, p._version) for p in policy.parameters()]
    corpus = corpus_for(policy, traces, manifest)
    fitted = fit_continuation_value(policy, corpus, max_scores=2)
    report = fitted.report
    assert report['initial_head_hash'] == parameter_hash(policy.actor)
    assert report['fitted_head_hash'] != report['initial_head_hash']
    assert report['final_weighted_mse'] < report['updates'][0]['weighted_mse_before_update']
    assert report['costs']['head_optimizer_steps'] == report['costs']['head_backward_calls'] == 64
    assert report['costs']['base_policy_forwards'] == report['costs']['base_optimizer_steps'] == 0
    trace = traces[0][2]; obs = trace.transitions[0].observation
    scores = fitted.score_actions(obs, trace.context)
    assert len(scores) == len(obs.action_ids) and scores[0] == 0.
    assert fitted.costs['policy_forwards_completed'] == fitted.costs['head_forwards'] == 1
    # Masks are public action-budget masks, never arbitrary geometric legality.
    masked = obs.action_mask.copy(); masked[2] = False
    with pytest.raises(ValueError, match='masking is restricted'):
        replace(obs, action_mask=masked)
    exhausted = obs.state_features.copy(); exhausted[0] = exhausted[1]
    masked[1:] = False
    terminal_observation = replace(obs, action_mask=masked, state_features=exhausted)
    scores = fitted.score_actions(terminal_observation, trace.context)
    assert scores == (0., None, None)
    with pytest.raises(RuntimeError, match='cap'):
        fitted.score_actions(obs, trace.context)
    assert fitted.costs['policy_forwards_completed'] == 2
    assert parameter_hash(policy) == before
    assert [(p.requires_grad, p._version) for p in policy.parameters()] == flags
    assert all(p.grad is None for p in policy.parameters())
    assert all(not m.training for m in policy.modules())


@pytest.mark.parametrize('bad', ['payload', 'private', 'incomplete', 'foreign', 'duplicate', 'missing'])
def test_binding_and_complete_denominator_refusal(bad):
    policy, traces, manifest = generated_fixture()
    builder = builder_for(policy, manifest)
    key, kind, trace = traces[0]
    if bad == 'payload':
        changed = thaw_json(manifest); changed[0]['decisions'][0]['reward'] += 1.
        builder = builder_for(policy, freeze_json(changed))
    elif bad == 'private':
        history = thaw_json(trace.history); history[0]['outcome_scope'] = 'evaluator_reference'
        history = freeze_json(history)
        seal = semantic_digest({'context': trace.context.fingerprint,
            'observations': [r.observation.fingerprint for r in trace.transitions],
            'history': history, 'behavior_parameter_hash': trace.behavior_parameter_hash})
        trace = replace(trace, history=history, seal_hash=seal)
    elif bad == 'incomplete':
        trace = replace(trace, transitions=trace.transitions[:-1], history=trace.history[:-1])
    elif bad == 'foreign':
        record = trace.context.record(); record['role'] = 'SELECT'
        context = PatientPlanningContext(freeze_json(record), semantic_digest(record))
        trace = replace(trace, context=context)
    elif bad == 'duplicate':
        builder.add_trace(key, trace)
    elif bad == 'missing':
        builder.add_trace(key, trace)
        with pytest.raises(ValueError, match='Every declared'):
            builder.finalize()
        return
    with pytest.raises(ValueError):
        builder.add_trace(key, trace)
    with pytest.raises(ValueError, match='consumed'):
        builder.finalize()
    assert all(p.grad is None for p in policy.parameters())


def test_changed_policy_and_foreign_scoring_context_fail_closed():
    policy, traces, manifest = generated_fixture()
    fitted = fit_continuation_value(policy, corpus_for(policy, traces, manifest),
        config=ContinuationFitConfig(updates=1))
    trace = traces[0][2]
    record = trace.context.record(); record['decision_model_hash'] = 'foreign-world'
    foreign = PatientPlanningContext(freeze_json(record), semantic_digest(record))
    with pytest.raises(ValueError, match='Foreign public world'):
        fitted.score_actions(trace.transitions[0].observation, foreign)
    assert fitted.costs['policy_forwards_started'] == 0
    with torch.no_grad():
        next(policy.encoder.parameters()).add_(.001)
    with pytest.raises(ValueError, match='Frozen encoder'):
        fitted.score_actions(trace.transitions[0].observation, trace.context)


def test_failure_removes_hook_and_cannot_fit_partial_features(monkeypatch):
    policy, traces, manifest = generated_fixture()
    builder = builder_for(policy, manifest)
    original = policy.forward
    def fail_after_feature(observation):
        original(observation)
        raise RuntimeError('generated post-forward failure')
    monkeypatch.setattr(policy, 'forward', fail_after_feature)
    with pytest.raises(RuntimeError, match='post-forward'):
        builder.add_trace(traces[0][0], traces[0][2])
    assert not policy.actor._forward_pre_hooks
    assert builder.costs['policy_forwards_started'] == 1
    assert builder.costs['policy_forwards_completed'] == 0
    with pytest.raises(ValueError, match='consumed'):
        builder.finalize()


def test_cancel_dirty_grad_and_manifest_refuse_before_feature_work():
    policy, traces, manifest = generated_fixture()
    builder = builder_for(policy, manifest, cancelled=lambda: True)
    with pytest.raises(RuntimeError, match='cancellation'):
        builder.add_trace(traces[0][0], traces[0][2])
    assert builder.costs['policy_forwards_started'] == 0
    with pytest.raises(ValueError, match='manifest hash'):
        ContinuationCorpusBuilder(policy, manifest, manifest_hash='changed')
    next(policy.parameters()).grad = torch.ones_like(next(policy.parameters()))
    with pytest.raises(ValueError, match='clean gradient'):
        builder_for(policy, manifest)


def test_unequal_length_forced_STOP_trace_keeps_declared_subject_and_trace_mass():
    policy, traces, _ = generated_fixture()
    key, kind, original = traces[0]
    row = SpatialTransition(original.transitions[0].observation, 'STOP', 0., True)
    history = freeze_json([{'action_id': 'STOP', 'reward': 0.}])
    seal = semantic_digest({'context': original.context.fingerprint,
        'observations': [row.observation.fingerprint], 'history': history,
        'behavior_parameter_hash': None})
    short = PatientTrainingTrace(original.context, (row,), history, None, seal, _TRACE).require()
    traces[0] = (key, kind, short)
    manifest = freeze_json([trace_descriptor(k, kind, t) for k, kind, t in traces])
    corpus = corpus_for(policy, traces, manifest)
    assert len(corpus.targets) == 22 and corpus.stop_rows[0] and corpus.targets[0] == 0.
    assert corpus.weights[0] == pytest.approx(.125)
    assert np.sum(corpus.weights[1:4]) == pytest.approx(.125)
    assert np.sum(corpus.weights[:4]) == pytest.approx(.25)
    assert np.sum(corpus.weights) == pytest.approx(1.)


@pytest.mark.parametrize('also_break_base_integrity', [False, True])
def test_partial_fit_costs_survive_original_error_and_secondary_integrity_failure(
        monkeypatch, also_break_base_integrity):
    policy, traces, manifest = generated_fixture()
    corpus = corpus_for(policy, traces, manifest)
    before = parameter_hash(policy)
    original_step = torch.optim.Adam.step
    original_error = RuntimeError('generated second optimizer call failure')
    attempts = 0
    def fail_second(optimizer, *args, **kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 2:
            if also_break_base_integrity:
                with torch.no_grad():
                    next(policy.encoder.parameters()).add_(.001)
            raise original_error
        return original_step(optimizer, *args, **kwargs)
    monkeypatch.setattr(torch.optim.Adam, 'step', fail_second)
    with pytest.raises(RuntimeError, match='second optimizer') as caught:
        fit_continuation_value(policy, corpus, config=ContinuationFitConfig(updates=3))
    assert caught.value is original_error
    costs = caught.value.continuation_fit_accounting
    assert costs['head_training_forwards_attempted'] == costs['head_training_forwards_completed'] == 2
    assert costs['head_backward_calls_attempted'] == costs['head_backward_calls_completed'] == 2
    assert costs['head_optimizer_steps_attempted'] == 2
    assert costs['head_optimizer_steps_completed'] == costs['completed_updates'] == 1
    assert costs['optimizer_step_completion_unknown'] is True
    assert costs['head_readout_forwards_attempted'] == costs['head_readout_forwards_completed'] == 0
    assert len(costs['update_rows']) == 1 and costs['update_rows'][0]['update'] == 1
    assert costs['elapsed_seconds'] >= 0.
    assert costs['base_policy_forwards'] == costs['base_optimizer_steps'] == 0
    assert costs['base_parameters_unchanged'] is (not also_break_base_integrity)
    assert (costs['base_integrity_error'] is not None) is also_break_base_integrity
    assert (parameter_hash(policy) == before) is (not also_break_base_integrity)
    assert all(p.grad is None for p in policy.parameters())
