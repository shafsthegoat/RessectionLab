"""Generated native/tensor controls only; no patient or checkpoint payloads."""
from dataclasses import replace

import pytest
import torch

from resectionlab.core import array_digest, semantic_digest, thaw_json
from resectionlab.native_spatial_task import make_native_opening_task
from resectionlab.patient_planning_admission import make_patient_planning_task
from resectionlab.patient_planning_learning import (PREFLIGHT_PROTOCOL, PatientImitationSample,
    PatientTrainSession, admit_collected_trace, common_patient_policies,
    patient_gradient_step, patient_imitation_loss, patient_reinforce_loss)
from resectionlab.spatial_policy import SpatialTransition, parameter_hash
from test_patient_planning_admission import fixture_contract, rebind


def bound_fixture(*, updates=1):
    _, args = fixture_contract()
    original = make_native_opening_task().case
    case = original
    source = args["source_binding"]
    source.update(source_hash=case.source_hash, image_array_hash=array_digest(case.structural_intensity),
        support_array_hash=array_digest(case.observed_support), target_array_hash=array_digest(case.nominal_target),
        affine_array_hash=array_digest(case.affine_ras_mm))
    protocol = thaw_json(PREFLIGHT_PROTOCOL); protocol["updates_per_method"] = updates
    args["protocol"]["learning_protocol_hash"] = semantic_digest(protocol)
    args["protocol"]["max_optimizer_updates"] = 2*updates
    rebind(args)
    task, context = make_patient_planning_task(case, **args)
    return task, context, protocol


def trace(task, context, *, policy=None, opening=False):
    worker = task.planning_clone(); transitions = []
    while not worker.terminated:
        observation = worker.observation()
        legal = [a for a, ok in zip(observation.action_ids, observation.action_mask) if a != "STOP" and ok]
        action = legal[0] if opening and not transitions else "STOP"
        result = worker.step(action)
        transitions.append(SpatialTransition(observation, action, result.reward, result.terminated))
    return admit_collected_trace(context, transitions, worker,
        behavior_parameter_hash=None if policy is None else parameter_hash(policy))


def sessions(context, protocol):
    il, rl = common_patient_policies((context,), protocol)
    initial = parameter_hash(il)
    return tuple(PatientTrainSession((context,), method, policy,
        initial_parameter_hash=initial, protocol=protocol) for method, policy in (("IL", il), ("RL", rl)))


def test_common_initial_parameters_and_rng_are_exact():
    _, context, protocol = bound_fixture()
    before = torch.random.get_rng_state().clone()
    il, rl = common_patient_policies((context,), protocol)
    assert torch.equal(before, torch.random.get_rng_state())
    assert parameter_hash(il) == parameter_hash(rl)
    assert all(torch.equal(a, b) for a, b in zip(il.parameters(), rl.parameters()))


def test_context_protocol_mismatch_refuses_before_policy_construction():
    _, context, protocol = bound_fixture()
    protocol["seed"] += 1
    with pytest.raises(ValueError, match="protocol differs"):
        common_patient_policies((context,), protocol)


def test_imitation_is_exact_mean_ce_and_update_is_single_use():
    task, context, protocol = bound_fixture()
    il, _ = sessions(context, protocol)
    teacher = trace(task, context, opening=True)
    samples = [PatientImitationSample(teacher, i) for i in range(len(teacher.transitions))]
    expected = torch.stack([-il.policy(row.observation)[0].log_softmax(-1)[row.observation.action_ids.index(row.action_id)]
                            for row in teacher.transitions]).mean()
    loss, record = patient_imitation_loss(il, samples)
    assert torch.allclose(loss, expected, atol=1e-7, rtol=0)
    assert record["loss_forward_calls"] == len(samples)
    receipt = patient_gradient_step(il, loss)
    assert receipt["optimizer_updates"] == 1 and receipt["parameters_changed"]
    assert receipt["gradient_norm_before_clip"] > 0
    with pytest.raises(ValueError): patient_gradient_step(il, loss)


def test_declared_two_update_budget_reuses_same_session():
    task, context, protocol = bound_fixture(updates=2)
    il, _ = sessions(context, protocol)
    teacher = trace(task, context)
    samples = [PatientImitationSample(teacher, 0)]
    for expected in (1, 2):
        loss, _ = patient_imitation_loss(il, samples)
        assert patient_gradient_step(il, loss)["completed_updates"] == expected
    with pytest.raises(ValueError): patient_imitation_loss(il, samples)


def test_rl_complete_episode_sum_matches_explicit_formula():
    task, context, protocol = bound_fixture()
    _, rl = sessions(context, protocol)
    trajectory = trace(task, context, policy=rl.policy, opening=True)
    targets = [sum(t.reward for t in trajectory.transitions[i:]) for i in range(len(trajectory.transitions))]
    actors = []; values = []; entropies = []
    for row, target in zip(trajectory.transitions, targets):
        logits, value = rl.policy(row.observation)
        distribution = torch.distributions.Categorical(logits=logits)
        idx = logits.new_tensor(row.observation.action_ids.index(row.action_id), dtype=torch.long)
        actors.append(-distribution.log_prob(idx)*(value.new_tensor(target)-value.detach()))
        values.append((value-target).square()); entropies.append(distribution.entropy())
    expected = torch.stack(actors).sum()+.5*torch.stack(values).mean()-.01*torch.stack(entropies).mean()
    loss, record = patient_reinforce_loss(rl, (trajectory,))
    assert torch.allclose(loss, expected, atol=1e-7, rtol=0)
    assert record["actor_reduction"] == "mean_episodes_sum_score_terms"
    assert record["loss_forward_calls"] == len(trajectory.transitions)


def test_changed_method_parameters_and_counter_refuse():
    task, context, protocol = bound_fixture()
    il, _ = sessions(context, protocol)
    teacher = trace(task, context)
    il.method = "RL"
    with pytest.raises(ValueError): patient_imitation_loss(il, [PatientImitationSample(teacher, 0)])
    il.method = "IL"; il.updates = -1
    with pytest.raises(ValueError): patient_imitation_loss(il, [PatientImitationSample(teacher, 0)])
    il.updates = 0
    with torch.no_grad(): next(il.policy.parameters()).add_(.001)
    with pytest.raises(ValueError): patient_imitation_loss(il, [PatientImitationSample(teacher, 0)])


def test_stale_or_changed_trace_and_incomplete_history_refuse():
    task, context, protocol = bound_fixture()
    _, rl = sessions(context, protocol)
    trajectory = trace(task, context, policy=rl.policy, opening=True)
    with pytest.raises(ValueError): replace(trajectory, behavior_parameter_hash="sha256:"+"0"*64).require()
    with pytest.raises(ValueError): replace(trajectory, transitions=trajectory.transitions[:-1]).require()
    with pytest.raises(ValueError): replace(trajectory, history=trajectory.history[:-1]).require()
    with pytest.raises(ValueError): replace(trajectory, _capability=object()).require()


def test_foreign_public_world_does_not_supply_loss():
    task, context, protocol = bound_fixture()
    il, _ = sessions(context, protocol)
    other = replace(task.case, target_derivation="different disclosed public target lineage")
    # Its observation is intact, but belongs to a different public source identity.
    from resectionlab.native_spatial_task import NativeSpatialTask
    observation = NativeSpatialTask(other, max_steps=2).observation()
    with pytest.raises(ValueError): context.require_observations((observation,))
