"""Search arithmetic and isolation tests; no anatomy generation or training."""
from dataclasses import dataclass, replace
import importlib
from itertools import product
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from resectionlab.observed_search import ObservedPolicyChanged, ObservedSearchLimit, observed_beam_search

search_module = importlib.import_module("resectionlab.observed_search")


@dataclass
class GraphTask:
    tree: dict
    max_steps: int = 2
    path: tuple = ()
    terminated: bool = False
    planning: bool = False
    private_reward: float = 0.
    masked: tuple = ()

    def planning_clone(self):
        return replace(self, planning=True, private_reward=0.)

    def clone(self):
        return replace(self)

    def metrics(self):
        return {"planning_estimator_only": self.planning, "steps": len(self.path)}

    def observation(self):
        ids = ("STOP", *self.tree.get(self.path, {}))
        return SimpleNamespace(action_ids=ids,
            action_mask=np.asarray([action not in self.masked for action in ids], dtype=bool))

    def step(self, action):
        assert not self.terminated
        assert action not in self.masked
        reward, terminal = self.tree[self.path][action]
        self.path = (*self.path, action)
        self.terminated = terminal or len(self.path) == self.max_steps
        return SimpleNamespace(reward=reward + self.private_reward, terminated=self.terminated)


def opening_graph():
    return GraphTask({(): {"opening": (-2., False), "dead_end": (-1., False)},
        ("opening",): {"finish": (5., True)}, ("dead_end",): {"finish": (1., True)}})


def test_complete_search_retains_costly_prefixes_and_reports_layers():
    task = opening_graph()
    path, row = observed_beam_search(task, max_calls=4, beam_width=2)
    assert path == ("opening", "finish")
    assert row["estimated_incremental_return"] == 3.
    assert row["model_transition_calls"] == 4
    assert row["completed_layers"] == 2
    assert not row["call_cap_reached"]
    assert row["negative_prefixes_evaluated"] == 2
    assert row["negative_prefixes_retained"] == 2
    assert row["negative_prefixes_pruned_by_beam"] == 0
    assert row["root_legal_nonstop_actions"] == 2
    assert row["implicit_stop_prefix_count"] == 5
    assert row["stop_baseline_return"] == 0.
    assert row["actor_forward_calls"] == 0
    assert task.path == () and not task.planning


def test_partial_first_layer_reports_weak_stop_teacher_and_pending_negative():
    path, row = observed_beam_search(opening_graph(), max_calls=1, beam_width=2)
    assert path == ("STOP",)
    assert row["call_cap_reached"] and row["completed_layers"] == 0
    assert row["layers"][0]["negative_prefixes_pending_at_cap"] == 1
    assert not row["layers"][0]["completed"]
    assert row["negative_prefixes_retained"] == 0


def test_partial_later_layer_and_beam_pruning_are_distinct():
    _, cap = observed_beam_search(opening_graph(), max_calls=3, beam_width=2)
    assert cap["completed_layers"] == 1 and cap["call_cap_reached"]
    assert cap["layers"][1]["model_transition_calls"] == 1
    path, beam = observed_beam_search(opening_graph(), max_calls=8, beam_width=1)
    assert path == ("STOP",)
    assert not beam["call_cap_reached"]
    assert beam["negative_prefixes_pruned_by_beam"] == 1
    assert beam["beam_pruned_prefixes"] == 1


class FixedActor(torch.nn.Module):
    def __init__(self, scores):
        super().__init__()
        self.scores = torch.nn.Parameter(torch.tensor(scores, dtype=torch.float32))
        self.grad_enabled = []

    def forward(self, observation):
        self.grad_enabled.append(torch.is_grad_enabled())
        return self.scores * 1., torch.tensor(float("nan"))  # Value must not guide search.


def test_actor_only_reorders_all_actions_and_charges_one_forward():
    task = GraphTask({(): {"first": (1., True), "second": (3., True)}})
    actor = FixedActor([100., 0., 2.])
    unguided, _ = observed_beam_search(task, max_calls=1, beam_width=2)
    guided, row = observed_beam_search(task, max_calls=1, beam_width=2, policy=actor)
    assert unguided == ("first",) and guided == ("second",)
    assert row["model_transition_calls"] == row["actor_forward_calls"] == 1
    assert row["policy_state_checks"] == 2
    assert row["initial_policy_state_hash"] == row["final_policy_state_hash"]
    assert 0 <= row["policy_state_check_seconds"] <= row["planning_seconds"]
    assert row["call_cap_reached"]
    complete, all_actions = observed_beam_search(task, max_calls=2, beam_width=2, policy=actor)
    assert complete == guided
    assert all_actions["model_transition_calls"] == 2 and not all_actions["call_cap_reached"]
    assert actor.grad_enabled == [False, False]
    assert actor.scores.grad is None


def test_guidance_buffer_mutation_fails_even_without_gradients():
    class MutatingActor(FixedActor):
        def __init__(self):
            super().__init__([0., 1., 2.])
            self.register_buffer("running_count", torch.zeros(()))
        def forward(self, observation):
            self.running_count.add_(1.)
            return super().forward(observation)
    actor = MutatingActor()
    with pytest.raises(ObservedPolicyChanged, match="state_dict") as caught:
        observed_beam_search(opening_graph(), max_calls=1, beam_width=2, policy=actor)
    row = caught.value.accounting
    assert row["policy_state_checks"] == 2 and row["actor_forward_calls"] == 1
    assert row["initial_policy_state_hash"] != row["final_policy_state_hash"]


def test_guided_timeout_still_checks_final_model_state(monkeypatch):
    clock = [0.]
    class SlowActor(FixedActor):
        def forward(self, observation):
            result = super().forward(observation)
            clock[0] = 2.
            return result
    monkeypatch.setattr(search_module.time, "perf_counter", lambda: clock[0])
    actor = SlowActor([0., 1., 2.])
    with pytest.raises(ObservedSearchLimit) as caught:
        observed_beam_search(opening_graph(), max_calls=4, beam_width=2, seconds=1., policy=actor)
    row = caught.value.accounting
    assert row["model_transition_calls"] == 0 and row["actor_forward_calls"] == 1
    assert row["policy_state_checks"] == 2 and row["time_cap_reached"]
    assert row["initial_policy_state_hash"] == row["final_policy_state_hash"]


def test_failed_actor_forward_is_charged_and_retains_partial_accounting():
    class FailingActor(FixedActor):
        def forward(self, observation):
            raise RuntimeError("intentional inference failure")
    actor = FailingActor([0., 1., 2.])
    with pytest.raises(RuntimeError, match="inference failure") as caught:
        observed_beam_search(opening_graph(), max_calls=4, beam_width=2, policy=actor)
    row = caught.value.accounting
    assert row["actor_forward_calls"] == 1 and row["model_transition_calls"] == 0
    assert row["policy_state_checks"] == 2
    assert row["root_legal_nonstop_actions"] == 2
    assert not row["layers"][0]["completed"]
    assert row["initial_policy_state_hash"] == row["final_policy_state_hash"]


def test_transition_failure_keeps_completed_guidance_accounting():
    class FailingTask(GraphTask):
        def step(self, action):
            raise RuntimeError("intentional transition failure")
    with pytest.raises(RuntimeError, match="transition failure") as caught:
        observed_beam_search(FailingTask(opening_graph().tree), max_calls=4, beam_width=2,
                             policy=FixedActor([0., 1., 2.]))
    row = caught.value.accounting
    assert row["actor_forward_calls"] == 1 and row["policy_state_checks"] == 2
    assert row["model_transition_calls"] == 1
    assert row["evaluated_transition_prefixes"] == 0
    assert row["implicit_stop_prefix_count"] == 1
    assert row["completed_layers"] == 0


def test_nonfinite_transition_charged_without_a_fabricated_stop_prefix():
    task = GraphTask({(): {"bad": (float("nan"), True)}})
    with pytest.raises(FloatingPointError, match="Nonfinite") as caught:
        observed_beam_search(task, max_calls=4, beam_width=2)
    row = caught.value.accounting
    assert row["model_transition_calls"] == 1
    assert row["evaluated_transition_prefixes"] == 0
    assert row["implicit_stop_prefix_count"] == 1


def test_actor_ties_and_masks_preserve_inventory_semantics():
    task = GraphTask({(): {"first": (1., True), "second": (99., True)}}, masked=("second",))
    actor = FixedActor([0., 0., 100.])
    path, row = observed_beam_search(task, max_calls=8, beam_width=2, policy=actor)
    assert path == ("first",) and row["root_legal_nonstop_actions"] == 1
    assert row["model_transition_calls"] == 1
    tied = replace(task, masked=())
    path, _ = observed_beam_search(tied, max_calls=1, beam_width=2, policy=FixedActor([0., 1., 1.]))
    assert path == ("first",)


def test_counterfactual_reference_does_not_change_plan_or_accounting():
    left = opening_graph()
    right = replace(left, private_reward=-1000.)
    plan_a, a = observed_beam_search(left, max_calls=8, beam_width=2)
    plan_b, b = observed_beam_search(right, max_calls=8, beam_width=2)
    assert plan_a == plan_b
    assert {k: v for k, v in a.items() if k != "planning_seconds"} == {
        k: v for k, v in b.items() if k != "planning_seconds"}


def test_timeout_preserves_failure_and_partial_trace(monkeypatch):
    times = iter([0., 0., 2., 2.])
    monkeypatch.setattr(search_module.time, "perf_counter", lambda: next(times))
    with pytest.raises(ObservedSearchLimit) as caught:
        observed_beam_search(opening_graph(), max_calls=8, beam_width=2, seconds=1.)
    failure = caught.value
    assert failure.accounting["model_transition_calls"] == 1
    assert failure.accounting["time_cap_reached"]
    assert failure.accounting["completed_layers"] == 0
    assert failure.best_sequence == ("STOP",)


def test_stop_only_and_zero_call_budget_have_explicit_accounting():
    path, row = observed_beam_search(GraphTask({}), max_calls=0, beam_width=1)
    assert path == ("STOP",) and row["root_legal_nonstop_actions"] == 0
    assert row["model_transition_calls"] == 0 and not row["call_cap_reached"]
    _, capped = observed_beam_search(opening_graph(), max_calls=0, beam_width=1,
        policy=FixedActor([0., 1., 2.]))
    assert capped["call_cap_reached"] and capped["actor_forward_calls"] == 0


def test_nonterminal_best_prefix_gets_stop_and_terminal_prefix_does_not():
    task = GraphTask({(): {"gain": (1., False)}, ("gain",): {"loss": (-2., True)}})
    path, _ = observed_beam_search(task, max_calls=8, beam_width=2)
    assert path == ("gain", "STOP")
    terminal = replace(task, tree={(): {"gain": (1., True)}})
    path, _ = observed_beam_search(terminal, max_calls=8, beam_width=2)
    assert path == ("gain",)


@pytest.mark.parametrize("kwargs", [{"max_calls": -1}, {"max_calls": True}, {"beam_width": 0},
    {"seconds": float("inf")}, {"seconds": -1.}, {"objective_source": ""}])
def test_invalid_budgets_fail_before_planning(kwargs):
    args = {"max_calls": 4, "beam_width": 2, **kwargs}
    with pytest.raises(ValueError):
        observed_beam_search(opening_graph(), **args)


def test_observed_clone_guard_is_required():
    class BadTask(GraphTask):
        def planning_clone(self):
            return self.clone()
    with pytest.raises(ValueError, match="observed-only"):
        observed_beam_search(BadTask({}), max_calls=4, beam_width=2)


def full_layer_oracle(task, cap, width):
    """Independent old full-materialization beam for arithmetic equivalence."""
    frontier = [(0., (), task.planning_clone())]
    best, path, best_terminal = 0., (), False
    calls = layers = negative_seen = negative_kept = negative_pruned = pruned = 0
    truncated = False
    pending = 0
    while frontier:
        all_children = []
        for value, prefix, node in frontier:
            for action in node.observation().action_ids[1:]:
                if calls == cap:
                    truncated = True
                    break
                next_node = node.clone()
                step = next_node.step(action)
                calls += 1
                score, candidate = value + step.reward, (*prefix, action)
                negative_seen += int(score < 0)
                if score > best:
                    best, path, best_terminal = score, candidate, next_node.terminated
                if not next_node.terminated:
                    all_children.append((score, candidate, next_node))
            if truncated:
                break
        if truncated:
            pending = sum(row[0] < 0 for row in all_children)
            break
        layers += 1
        all_children.sort(key=lambda row: (-row[0], row[1]))
        frontier = all_children[:width]
        negative_kept += sum(row[0] < 0 for row in frontier)
        negative_pruned += sum(row[0] < 0 for row in all_children[width:])
        pruned += len(all_children[width:])
    executable = path if best_terminal or len(path) == task.max_steps else (*path, "STOP")
    return executable, {"model_transition_calls": calls, "completed_layers": layers,
        "estimated_incremental_return": best, "call_cap_reached": truncated,
        "negative_prefixes_evaluated": negative_seen, "negative_prefixes_retained": negative_kept,
        "negative_prefixes_pruned_by_beam": negative_pruned, "beam_pruned_prefixes": pruned}, pending


def tied_graph(branches=4):
    actions = tuple(f"action-{index}" for index in range(branches))
    tree = {}
    for depth in range(3):
        for prefix in product(actions, repeat=depth):
            tree[prefix] = {action: (-1. if depth < 2 else float(index % 2 + 2), depth == 2)
                            for index, action in enumerate(reversed(actions))}
    return GraphTask(tree, max_steps=3)


@pytest.mark.parametrize("width", [1, 2, 3, 8, 16])
def test_incremental_retention_matches_full_layer_oracle_for_every_cap(width):
    task = tied_graph()
    for cap in range(86):
        expected_path, expected, pending = full_layer_oracle(task, cap, width)
        path, report = observed_beam_search(task, max_calls=cap, beam_width=width)
        assert path == expected_path
        assert {key: report[key] for key in expected} == expected
        assert report["layers"][-1]["negative_prefixes_pending_at_cap"] == pending
        assert report["peak_next_layer_states"] <= width + 1


def test_live_clone_count_is_bounded_without_weakrefs():
    counters = {"active": 0, "peak": 0, "created": 0}

    class CountedTask(GraphTask):
        def __init__(self, tree, max_steps=3, path=(), terminated=False, planning=False, *, cloned=False):
            super().__init__(tree, max_steps, path, terminated, planning)
            self.counted = cloned
            if cloned:
                counters["active"] += 1
                counters["created"] += 1
                counters["peak"] = max(counters["peak"], counters["active"])

        def __del__(self):
            if self.counted:
                counters["active"] -= 1

        def clone(self):
            return CountedTask(self.tree, self.max_steps, self.path, self.terminated,
                               self.planning, cloned=True)

        def planning_clone(self):
            result = self.clone()
            result.planning = True
            return result

    width = 3
    task = CountedTask(tied_graph(branches=12).tree)
    _, row = observed_beam_search(task, max_calls=1000, beam_width=width)
    assert row["model_transition_calls"] == 12 + 36 + 36
    assert counters["created"] == 1 + row["model_transition_calls"]
    # At most the current beam plus the next beam and one insertion candidate.
    # Direct __init__/__del__ counters do not rely on weakref-based measurement.
    assert counters["peak"] <= 2 * width + 1
    assert counters["active"] == 0
    assert row["peak_next_layer_states"] <= width + 1
