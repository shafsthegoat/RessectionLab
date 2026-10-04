"""Independent analytic and information-boundary checks; no training study."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from resectionlab.spatial_policy import SpatialPolicy, SpatialTransition, reinforce_loss
from resectionlab.spatial_task import SpatialTask, make_opening_task, make_spatial_case


class _TwoDecisionPolicy(torch.nn.Module):
    """Two independent logits permit an exact trajectory-gradient calculation."""

    def __init__(self, baseline):
        super().__init__()
        self.theta = torch.nn.Parameter(torch.tensor([.37, -.42], dtype=torch.float64))
        self.baseline = baseline

    def forward(self, observation):
        value = self.theta.new_tensor(self.baseline[observation.state])
        return torch.stack((self.theta.new_zeros(()), self.theta[observation.state])), value


def _all_short_task_trajectories():
    first = SimpleNamespace(state=0, action_ids=("STOP", "OPEN"), action_mask=(True, True))
    second = SimpleNamespace(state=1, action_ids=("STOP", "TAKE"), action_mask=(True, True))
    opening = SpatialTransition(first, "OPEN", -1., False)
    return (
        [SpatialTransition(first, "STOP", 0., True)],
        [opening, SpatialTransition(second, "STOP", 0., True)],
        [opening, SpatialTransition(second, "TAKE", 3., True)],
    )


@pytest.mark.parametrize("gamma", [0., .4, 1.])
@pytest.mark.parametrize("baseline", [(0., 0.), (2.3, -.8)])
def test_expected_score_gradient_equals_exact_variable_length_return(gamma, baseline):
    """Enumeration checks unbiased gradient, not a loss expression copied from code.

    J = p(OPEN) * [-1 + gamma * 3 * p(TAKE)]. Path likelihoods are
    detached when averaging score gradients; differentiating them again would
    double-count policy dependence. An arbitrary state-only critic must cancel.
    """
    model = _TwoDecisionPolicy(baseline)
    p, q = model.theta.sigmoid().detach()
    weights = (1 - p, p * (1 - q), p * q)
    losses = [reinforce_loss(model, [episode], gamma=gamma,
                            entropy_weight=0., value_weight=0.)[0]
              for episode in _all_short_task_trajectories()]
    actual = -torch.autograd.grad(sum(w * loss for w, loss in zip(weights, losses)), model.theta)[0]
    expected = torch.stack((p * (1 - p) * (-1 + gamma * 3 * q),
                            gamma * 3 * p * q * (1 - q)))
    torch.testing.assert_close(actual, expected, atol=2e-15, rtol=2e-14)


def test_monte_carlo_batch_gives_each_episode_equal_weight():
    model = _TwoDecisionPolicy((1.1, -.3))
    short, _, long = _all_short_task_trajectories()
    settings = dict(gamma=.6, entropy_weight=0., value_weight=0.)
    combined, receipt = reinforce_loss(model, [short, long], **settings)
    a, _ = reinforce_loss(model, [short], **settings)
    b, _ = reinforce_loss(model, [long], **settings)
    torch.testing.assert_close(combined, (a + b) / 2, atol=0, rtol=0)
    assert receipt["completed_episodes"] == 2
    assert receipt["loss_forward_calls"] == 3


def _assert_same_actor_evidence(first, second):
    for name in ("image_channels", "coverage", "channel_available", "affine_ras_mm",
                 "spacing_mm", "action_geometry", "action_mask", "state_features"):
        np.testing.assert_array_equal(getattr(first, name), getattr(second, name), err_msg=name)
    for name in ("action_ids", "action_tool_ids", "source_id", "track", "channel_provenance"):
        assert getattr(first, name) == getattr(second, name), name
    assert first.fingerprint == second.fingerprint


def _nominal_action_values(task):
    nominal = task.planning_clone()
    values = []
    for action_id in nominal.observation().action_ids:
        clone = nominal.clone()
        values.append((action_id, clone.step(action_id).reward))
    return values


def test_private_reference_swap_cannot_change_action_set_actor_or_search_inputs():
    first = make_opening_task()
    shape = first.case.structural_intensity.shape
    # Strong counterfactual: every cell changes target class, and unavailable
    # functional reference magnitudes change by two orders of magnitude.
    alternate_case = replace(first.case, reference_target=~first.case.reference_target,
                             reference_motor=np.full(shape, 7., np.float32),
                             reference_language=np.full(shape, 103., np.float32),
                             recipe={"private_metadata": "must not become actor evidence"})
    second = SpatialTask(alternate_case, world_seed=47, max_steps=first.max_steps)
    assert first.case.source_hash == second.case.source_hash
    assert first.case.anatomy_hash != second.case.anatomy_hash
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(31)
        model = SpatialPolicy().eval()
    saw_private_reward_difference = False
    for action in ("REMOVE:opening-tool:1,1,0", "REMOVE:opening-tool:0,1,0"):
        obs1, obs2 = first.observation(), second.observation()
        _assert_same_actor_evidence(obs1, obs2)
        assert first.candidate_inventory() == second.candidate_inventory()
        with torch.no_grad():
            logits1, value1 = model(obs1)
            logits2, value2 = model(obs2)
        torch.testing.assert_close(logits1, logits2, atol=0, rtol=0)
        torch.testing.assert_close(value1, value2, atol=0, rtol=0)
        assert _nominal_action_values(first) == _nominal_action_values(second)
        step1, step2 = first.step(action), second.step(action)
        saw_private_reward_difference |= step1.reward != step2.reward
        np.testing.assert_array_equal(step1.info["removed_indices"], step2.info["removed_indices"])
    _assert_same_actor_evidence(first.observation(), second.observation())
    assert _nominal_action_values(first) == _nominal_action_values(second)
    assert saw_private_reward_difference, "Counterfactual must change actual private outcome, not just metadata"


@pytest.mark.parametrize("changed", ["tool", "access", "horizon"])
def test_actor_geometry_cannot_diverge_from_frozen_simulator_geometry(changed):
    task = make_opening_task()
    if changed == "tool":
        task.case = replace(task.case, tools=(replace(task.case.tools[0], shaft_radius_mm=.4),))
    elif changed == "access":
        task.case = replace(task.case, access=replace(task.case.access, center_mm=(1.1, 1., -.5)))
    else:
        task.max_steps += 1
    with pytest.raises(RuntimeError, match="changed|frozen|geometry|horizon"):
        task.observation()


@pytest.mark.parametrize("field", ["reference_target", "reference_motor", "reference_language"])
def test_evaluator_identity_cannot_be_replaced_during_an_episode(field):
    task = make_opening_task()
    original = getattr(task.case, field)
    changed = ~original if original.dtype == np.bool_ else original + 1.
    task.case = replace(task.case, **{field: changed})
    with pytest.raises(RuntimeError, match="changed|frozen|reference|world"):
        task.metrics()


def test_exhaustive_opening_optimum_matches_direct_source_cell_accounting():
    """Independent breadth-first enumeration and score arithmetic on a 3³ case."""
    root = make_opening_task()
    queue = [(root, (), ())]
    terminal = []
    transitions = 0
    while queue:
        node, prefix, rewards = queue.pop(0)
        if node.terminated:
            terminal.append((sum(rewards), prefix, node))
            continue
        for action in node.observation().action_ids:
            child = node.clone()
            result = child.step(action)
            queue.append((child, prefix + (action,), rewards + (result.reward,)))
            transitions += 1
            assert transitions <= 64, "Development fixture unexpectedly exceeded its exact-audit bound"
    assert transitions == 38 and len(terminal) == 32
    best, path, winner = max(terminal, key=lambda row: row[0])
    expected_path = ("REMOVE:opening-tool:1,1,0", "REMOVE:opening-tool:0,1,0",
                     "REMOVE:opening-tool:0,1,1")
    assert path == expected_path
    assert sum(score == best for score, _, _ in terminal) == 1
    # Three physically located cells, not a score copied from task.metrics().
    tips = np.array([[1., 1., 0.], [0., 1., 0.], [0., 1., 1.]])
    travel = 2 * np.linalg.norm(tips - np.asarray(root.case.access.center_mm), axis=1).sum()
    assert best == pytest.approx(1. - .2 * 2 - .03 * 3 - .001 * travel)
    immediate = dict(_nominal_action_values(root))
    assert immediate["STOP"] == 0 and all(value < 0 for key, value in immediate.items() if key != "STOP")
    assert best > 0
    first_two = winner.metrics()["history"][:2]
    assert all(record["reward"] < 0 for record in first_two)
    removed = [tuple(index) for record in winner.metrics()["history"] for index in record["removed_indices"]]
    assert len(removed) == len(set(removed)) == 3
    assert set(removed) == set(map(tuple, tips.astype(int)))


def test_unobserved_functional_realizations_change_diagnostics_not_primary_objective():
    base = make_opening_task()
    case = replace(base.case, reference_motor=np.ones((3, 3, 3), np.float32),
                   reference_language=np.full((3, 3, 3), 2., np.float32))
    first = SpatialTask(case, max_steps=3, world_seed=11)
    second = SpatialTask(case, max_steps=3, world_seed=23)
    nominal = first.planning_clone()
    for action in ("REMOVE:opening-tool:1,1,0", "REMOVE:opening-tool:0,1,0",
                   "REMOVE:opening-tool:0,1,1"):
        _assert_same_actor_evidence(first.observation(), second.observation())
        assert first.step(action).reward == second.step(action).reward == nominal.step(action).reward
    a, b, estimated = first.metrics(), second.metrics(), nominal.metrics()
    assert a["world_hash"] != b["world_hash"]
    assert a["motor_surrogate_exposure"] != b["motor_surrogate_exposure"]
    assert a["total_reward"] == b["total_reward"]
    for metrics in (a, b, estimated):
        assert metrics["function_unassessed_removed_volume_mm3"] == 3
        assert metrics["clinical_deficit_probability"] is None
    assert estimated["motor_surrogate_exposure"] is None
    assert estimated["language_surrogate_exposure"] is None


def test_anatomy_group_cannot_be_split_using_scan_noise_tools_or_private_function():
    original = make_spatial_case(8)
    support = original.structural_intensity > .05
    noisier_scan = np.where(support, original.structural_intensity + .013, 0.)
    changed = replace(original, structural_intensity=noisier_scan,
                      reference_motor=original.reference_motor * 3.,
                      reference_language=original.reference_language * 4.,
                      tools=(replace(original.tools[0], shaft_radius_mm=.4),),
                      access=replace(original.access, center_mm=(3.1, 3., -.5)),
                      recipe={"anatomy_seed": 999, "split": "eval"})
    assert original.source_hash != changed.source_hash
    assert original.reference_hash != changed.reference_hash
    assert original.anatomy_hash == changed.anatomy_hash


def test_all_eight_xy_symmetries_share_group_but_keep_actual_reference_identity():
    original = make_spatial_case(8)
    references = set()
    for quarter_turns in range(4):
        for mirror in (False, True):
            def transform(array):
                rotated = np.rot90(array, quarter_turns, axes=(0, 1))
                return np.flip(rotated, axis=0) if mirror else rotated
            changed = replace(original, **{name: transform(getattr(original, name))
                for name in ("structural_intensity", "reference_target",
                             "reference_motor", "reference_language")})
            assert original.anatomy_hash == changed.anatomy_hash
            references.add(changed.reference_hash)
    assert len(references) == 8, "The fixture must exercise eight distinct orientations"


def test_changed_target_support_or_physical_lattice_remains_a_distinct_group():
    original = make_spatial_case(8)
    target = original.reference_target.copy()
    target[3, 3, 0] = ~target[3, 3, 0]
    assert replace(original, reference_target=target).anatomy_hash != original.anatomy_hash
    affine = original.affine_ras_mm.copy()
    affine[0, 0] *= 1.1
    assert replace(original, affine_ras_mm=affine).anatomy_hash != original.anatomy_hash
    scan = original.structural_intensity.copy()
    scan[0, 0, 0] = .2
    assert replace(original, structural_intensity=scan).anatomy_hash != original.anatomy_hash
