"""Independent analytic controls; no patient images, fitting, or optimizer steps."""
from dataclasses import dataclass, replace
import importlib
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from resectionlab.observed_search import ObservedSearchLimit, observed_beam_search
from resectionlab.real_patient_learning import TRAIN_GROUPS, patient_uniform_bc_indices
from resectionlab.spatial_policy import imitation_loss, parameter_hash


search_module = importlib.import_module("resectionlab.observed_search")
TREE = {
    (): {"OPEN": (-4., False), "SHORT": (1., True),
         "FORBIDDEN": (1000., True), "DUD": (-.5, False)},
    ("DUD",): {"FINISH": (-.1, True)},
    ("OPEN",): {"CONTINUE": (-1., False)},
    ("OPEN", "CONTINUE"): {"TAKE": (12., True)},
}


@dataclass
class ObservedTree:
    visited: list
    path: tuple = ()
    terminated: bool = False
    max_steps: int = 3
    clock: object = None

    def clone(self):
        return replace(self)

    def metrics(self):
        return {"planning_estimator_only": True, "steps": len(self.path)}

    def observation(self):
        ids = ("STOP", *TREE.get(self.path, {}))
        return SimpleNamespace(action_ids=ids,
            action_mask=np.array([name != "FORBIDDEN" for name in ids], dtype=bool))

    def step(self, action):
        assert action not in ("STOP", "FORBIDDEN") and not self.terminated
        self.visited.append((self.path, action))
        reward, done = TREE[self.path][action]
        self.path = (*self.path, action)
        self.terminated = done or len(self.path) == self.max_steps
        if self.clock is not None:
            self.clock.now += 2.
        return SimpleNamespace(reward=reward, terminated=self.terminated)


class PrivateSourceTrap:
    """The search caller exposes only a sanctioned clone and horizon."""
    def __init__(self, observed):
        self._observed = observed

    @property
    def max_steps(self):
        return self._observed.max_steps

    def planning_clone(self):
        return self._observed.clone()

    def __getattr__(self, name):
        raise AssertionError("Search accessed private/source task field: " + name)


class ReverseActor(torch.nn.Module):
    def __init__(self, critic):
        super().__init__()
        self.offset = torch.nn.Parameter(torch.tensor(0.))
        self.critic = critic
        self.forward_count = 0

    def forward(self, observation):
        assert not torch.is_grad_enabled()
        self.forward_count += 1
        scores = {"STOP": 10000., "DUD": 100., "SHORT": 50., "OPEN": -50.,
                  "FORBIDDEN": 1e8}
        logits = self.offset + torch.tensor([scores.get(name, 0.) for name in observation.action_ids])
        return logits, torch.tensor(self.critic)


def test_every_call_budget_has_auditable_inventory_and_monotonic_incumbent():
    """An adversarial critic/STOP score cannot prune the costly opening chain."""
    outcomes = []
    expected_edges = {(prefix, action) for prefix, choices in TREE.items()
                      for action in choices if action != "FORBIDDEN"}
    for critic in (None, -1e9, 1e9):
        values, traces = [], []
        for cap in range(8):
            visited = []
            actor = None if critic is None else ReverseActor(critic)
            before = None if actor is None else parameter_hash(actor)
            path, row = observed_beam_search(PrivateSourceTrap(ObservedTree(visited)),
                max_calls=cap, beam_width=2, policy=actor)
            assert row["model_transition_calls"] == len(visited) == min(cap, 6)
            assert row["model_transition_calls"] == sum(x["model_transition_calls"] for x in row["layers"])
            assert row["completed_layers"] == sum(x["completed"] for x in row["layers"])
            assert row["call_cap_reached"] == (cap < 6)
            assert row["root_legal_nonstop_actions"] == 3
            assert row["implicit_stop_prefix_count"] == len(visited) + 1
            assert row["stop_baseline_return"] == 0.
            assert row["negative_prefixes_pruned_by_beam"] == 0
            assert row["actor_forward_calls"] == (0 if actor is None else actor.forward_count)
            assert before is None or before == parameter_hash(actor)
            if cap >= 6:
                assert set(visited) == expected_edges and len(set(visited)) == len(visited)
                assert path == ("OPEN", "CONTINUE", "TAKE")
                assert row["estimated_incremental_return"] == 7.
                assert row["completed_layers"] == 3
                assert row["negative_prefixes_retained"] == 3
                if actor is not None:
                    assert actor.forward_count == 4
            values.append(row["estimated_incremental_return"])
            traces.append((path, visited, row["actor_forward_calls"]))
        assert values == sorted(values)
        outcomes.append(traces)
    assert outcomes[1] == outcomes[2]  # Critic perturbation never changes enumeration or route.


def test_last_expensive_transition_is_charged_and_cannot_return_within_budget(monkeypatch):
    clock = SimpleNamespace(now=0.)
    monkeypatch.setattr(search_module.time, "perf_counter", lambda: clock.now)
    visited = []
    # Start immediately before the only terminal edge; the expensive step itself crosses the limit.
    node = ObservedTree(visited, path=("OPEN", "CONTINUE"), clock=clock)
    with pytest.raises(ObservedSearchLimit) as failure:
        observed_beam_search(PrivateSourceTrap(node), max_calls=1, beam_width=1, seconds=1.)
    row = failure.value.accounting
    assert row["model_transition_calls"] == 1 and row["time_cap_reached"]
    assert row["planning_seconds"] == 2.
    assert row["estimated_incremental_return"] == 12.
    assert failure.value.best_sequence == ("TAKE",)
    assert visited == [(("OPEN", "CONTINUE"), "TAKE")]


def test_a_forward_that_mutates_registered_state_cannot_return_a_frozen_plan():
    class MutatingActor(ReverseActor):
        def __init__(self):
            super().__init__(0.)
            self.register_buffer("running_observations", torch.zeros(()))

        def forward(self, observation):
            self.running_observations.add_(1.)
            return super().forward(observation)

    with pytest.raises((RuntimeError, ValueError), match="[Ff]rozen|[Mm]utat|[Ss]tate|[Cc]hanged"):
        observed_beam_search(PrivateSourceTrap(ObservedTree([])), max_calls=1,
            beam_width=2, policy=MutatingActor())


def test_an_interrupted_mutating_forward_still_records_the_attempted_work():
    class InterruptedActor(ReverseActor):
        def forward(self, observation):
            self.offset.add_(1.)
            raise RuntimeError("Analytic forward interruption")

    with pytest.raises(RuntimeError, match="changed the frozen") as failure:
        observed_beam_search(PrivateSourceTrap(ObservedTree([])), max_calls=1,
            beam_width=2, policy=InterruptedActor(0.))
    row = failure.value.accounting
    assert row["actor_forward_calls"] == 1
    assert row["model_transition_calls"] == 0
    assert row["completed_layers"] == 0
    assert row["policy_state_checks"] == 2


def test_failed_transition_is_charged_without_inventing_an_evaluated_stop_prefix(monkeypatch):
    def failed_step(self, action):
        raise RuntimeError("Analytic transition failure")

    monkeypatch.setattr(ObservedTree, "step", failed_step)
    with pytest.raises(RuntimeError, match="Analytic transition failure") as failure:
        observed_beam_search(PrivateSourceTrap(ObservedTree([])), max_calls=1, beam_width=2)
    row = failure.value.accounting
    assert row["model_transition_calls"] == 1
    assert row["implicit_stop_prefix_count"] == 1  # Only the initial state exists.


def test_patient_uniform_draws_give_the_patient_mean_bc_gradient_without_fitting():
    """A 1000-state teacher must not reverse the six-patient objective's gradient."""
    assert TRAIN_GROUPS == tuple("BTC:sub-PAT" + value for value in ("05", "16", "20", "22", "25", "28"))
    counts = dict(zip(TRAIN_GROUPS, (1, 2, 3, 4, 5, 1000)))

    class BalancedDraws:
        def __init__(self):
            self.index = 0

        def integers(self, bound):
            patient = (self.index // 2) % 6
            assert bound == (6 if self.index % 2 == 0 else counts[TRAIN_GROUPS[patient]])
            result = patient if self.index % 2 == 0 else 0
            self.index += 1
            return result

    class ScalarActor(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.log_odds = torch.nn.Parameter(torch.tensor(0., dtype=torch.float64))

        def forward(self, observation):
            return torch.stack((self.log_odds * 0., self.log_odds)), self.log_odds * 0.

    draws = patient_uniform_bc_indices(counts, batch_size=12, generator=BalancedDraws())
    observation = SimpleNamespace(action_ids=("STOP", "TAKE"), action_mask=np.ones(2, dtype=bool))
    samples = [(observation, "TAKE" if group == TRAIN_GROUPS[-1] else "STOP") for group, _ in draws]
    actor = ScalarActor()
    frozen = parameter_hash(actor)
    loss, report = imitation_loss(actor, samples)
    loss.backward()  # Derivative check only; no optimizer exists or runs.
    assert actor.log_odds.grad.item() == pytest.approx(.5 - 1. / 6.)
    pooled_state_gradient = .5 - 1000. / sum(counts.values())
    assert pooled_state_gradient < 0. < actor.log_odds.grad.item()
    assert parameter_hash(actor) == frozen
    assert report["loss_forward_calls"] == 12
