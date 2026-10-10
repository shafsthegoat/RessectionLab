"""Compact generated search diagnostics; no models, anatomy or extra previews."""
from dataclasses import dataclass, field, replace
import importlib
import importlib.util
import json
import os
from types import SimpleNamespace

import numpy as np
import pytest


def load_file(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


search = (load_file(os.environ["OBSERVED_SEARCH_CANDIDATE"], "staged_observed_search")
          if "OBSERVED_SEARCH_CANDIDATE" in os.environ
          else importlib.import_module("resectionlab.observed_search"))


@dataclass
class PublicTree:
    tree: dict
    max_steps: int = 3
    path: tuple = ()
    terminated: bool = False
    planning: bool = False
    counters: dict = field(default_factory=lambda: dict(metrics=0, clones=0,
        observations=0, inventory_builds=0, transitions=0))
    inventory_ready: bool = False
    scope: str = "permitted_nominal_model"
    missing: bool = False
    bad_value: object = None
    fail_at: int | None = None

    @property
    def private_reference(self):
        raise AssertionError("Private reference must never be queried")

    def planning_clone(self):
        return replace(self, planning=True)

    def clone(self):
        self.counters["clones"] += 1
        return replace(self)

    def metrics(self):
        self.counters["metrics"] += 1
        assert self.counters["metrics"] == 1, "No extra metrics/history queries"
        return {"planning_estimator_only": self.planning, "steps": len(self.path)}

    def observation(self):
        self.counters["observations"] += 1
        if not self.inventory_ready:
            self.counters["inventory_builds"] += 1
            self.inventory_ready = True
        ids = ("STOP", *(self.tree.get(self.path, {}) if not self.terminated else {}))
        return SimpleNamespace(action_ids=ids, action_mask=np.ones(len(ids), dtype=bool))

    def _advance(self, action, eager):
        self.counters["transitions"] += 1
        if self.counters["transitions"] == self.fail_at:
            raise InterruptedError("generated transition refusal")
        reward, target, outside, distance, terminal = self.tree[self.path][action]
        self.path = (*self.path, action)
        self.terminated = terminal or len(self.path) == self.max_steps
        self.inventory_ready = False
        info = {"action_id": action, "outcome_scope": self.scope,
                "target_removed_mm3": target, "normal_removed_mm3": outside,
                "insertion_distance_mm": distance}
        if self.missing:
            info.pop("target_removed_mm3")
        if self.bad_value is not None:
            info["target_removed_mm3"] = self.bad_value
        return SimpleNamespace(reward=reward, info=info,
            observation=self.observation() if eager else None, terminated=self.terminated)

    def step(self, action):
        return self._advance(action, True)

    def advance_planning(self, action):
        assert self.planning
        return self._advance(action, False)


def opening_tree(**kwargs):
    return PublicTree({
        (): {"open": (-2., 0., 3., .5, False),
             "shallow": (-1., 0., 1., .25, False),
             "discarded": (-5., 0., 9., .8, False)},
        ("open",): {"gain": (5., 4., .5, 3., True),
                    "deepen": (-.5, 0., .25, 1.5, False)},
        ("shallow",): {"empty": (-1., 0., .1, .4, True)},
        ("open", "deepen"): {"finish": (1., 1., .25, 2., True)},
    }, **kwargs)


def without_diagnostics(report):
    return {key: ([{k: v for k, v in row.items() if k != "retained_prefix_diagnostics"}
                   for row in value] if key == "layers" else value)
            for key, value in report.items()
            if key not in {"retained_prefix_diagnostics", "planning_seconds"}}


@pytest.mark.parametrize("mode", ["eager", "lazy_planning"])
@pytest.mark.parametrize("cap", [0, 1, 3, 5, 20])
def test_enabled_is_exact_decision_accounting_and_query_parity(mode, cap):
    left, right = opening_tree(), opening_tree()
    args = dict(max_calls=cap, beam_width=2, seconds=20., transition_mode=mode)
    old_path, old = search.observed_beam_search(left, **args)
    new_path, new = search.observed_beam_search(right, **args, retained_prefix_diagnostics=True)
    assert old_path == new_path
    assert without_diagnostics(old) == without_diagnostics(new)
    assert left.counters == right.counters
    assert "retained_prefix_diagnostics" not in old
    assert all("retained_prefix_diagnostics" not in row for row in old["layers"])
    assert new["retained_prefix_diagnostics"]["extra_observations_or_previews"] == 0
    json.dumps(new, allow_nan=False)


def test_cumulative_public_scalars_depth_and_normal_expansion_availability():
    path, report = search.observed_beam_search(opening_tree(), max_calls=20, beam_width=2,
        seconds=20., transition_mode="lazy_planning", retained_prefix_diagnostics=True)
    assert path == ("open", "gain")
    first = report["layers"][0]["retained_prefix_diagnostics"]
    assert [row["actions"] for row in first["retained_nonterminal_prefixes"]] == [["shallow"], ["open"]]
    shallow, opened = first["retained_nonterminal_prefixes"]
    assert shallow["legal_nonstop_actions"] == 1 and opened["legal_nonstop_actions"] == 2
    assert all(row["legal_target_family_actions"] is None for row in (shallow, opened))
    second = report["layers"][1]["retained_prefix_diagnostics"]
    gain = second["best_terminal_prefixes"][0]
    assert gain["actions"] == ["open", "gain"]
    assert gain["target_removed_mm3"] == 4.
    assert gain["outside_supplied_target_removed_mm3"] == 3.5
    assert gain["max_insertion_distance_mm"] == 3.
    assert gain["estimated_incremental_return"] == 3.
    assert gain["legal_nonstop_actions"] is None
    assert gain["successor_inventory_status"] == "terminal_not_observed"
    final = report["layers"][2]["retained_prefix_diagnostics"]
    assert final["retained_nonterminal_prefixes"] == []
    finish = final["best_terminal_prefixes"][0]
    assert finish["actions"] == ["open", "deepen", "finish"]
    assert (finish["target_removed_mm3"], finish["outside_supplied_target_removed_mm3"],
            finish["max_insertion_distance_mm"]) == (1., 3.5, 2.)


def test_partial_layer_candidates_are_not_claimed_as_advanced_or_observed():
    _, report = search.observed_beam_search(opening_tree(), max_calls=1, beam_width=2,
        retained_prefix_diagnostics=True)
    diag = report["layers"][0]["retained_prefix_diagnostics"]
    assert diag["selection_status"] == "partial_layer_not_advanced"
    assert report["completed_layers"] == 0
    candidate = diag["retained_nonterminal_prefixes"][0]
    assert candidate["actions"] == ["open"]
    assert candidate["legal_nonstop_actions"] is None
    assert candidate["successor_inventory_status"] == "not_observed_by_search"


@pytest.mark.parametrize("kwargs", [dict(missing=True), dict(scope="separate_evaluator_reference"),
    dict(bad_value=float("nan")), dict(bad_value=-1.), dict(bad_value=True)])
def test_unavailable_or_private_progress_is_never_interpreted_as_zero(kwargs):
    _, report = search.observed_beam_search(opening_tree(**kwargs), max_calls=20, beam_width=2,
        retained_prefix_diagnostics=True)
    for layer in report["layers"]:
        diag = layer["retained_prefix_diagnostics"]
        for row in diag["retained_nonterminal_prefixes"] + diag["best_terminal_prefixes"]:
            assert row["progress_status"] == "nominal_record_unavailable"
            assert row["target_removed_mm3"] is None
            assert row["outside_supplied_target_removed_mm3"] is None
            assert row["max_insertion_distance_mm"] is None
    json.dumps(report, allow_nan=False)


def test_private_scoped_record_does_not_read_numeric_fields():
    class ForbiddenValues(dict):
        def get(self, key, default=None):
            if key != "outcome_scope":
                raise AssertionError("Attempted private numeric read")
            return "separate_evaluator_reference"
    row = search._prefix_progress({"progress_status": "available"}, ("x",), -1.,
        SimpleNamespace(info=ForbiddenValues()))
    assert row["target_removed_mm3"] is None


def test_summary_count_is_beam_bounded_even_for_wide_terminal_layer():
    tree = {(): {f"action-{i:03d}": (-1., 0., 1., .5, True) for i in range(100)}}
    _, report = search.observed_beam_search(PublicTree(tree), max_calls=100, beam_width=2,
        retained_prefix_diagnostics=True)
    diag = report["layers"][0]["retained_prefix_diagnostics"]
    assert len(diag["best_terminal_prefixes"]) == 2
    assert [r["actions"] for r in diag["best_terminal_prefixes"]] == [["action-000"], ["action-001"]]
    assert report["model_transition_calls"] == 100


def test_failure_retains_partial_diagnostic_and_same_charged_call():
    outcomes = []
    for enabled in (False, True):
        with pytest.raises(InterruptedError) as error:
            search.observed_beam_search(opening_tree(fail_at=2), max_calls=20, beam_width=2,
                retained_prefix_diagnostics=enabled)
        outcomes.append(error.value.accounting)
    assert without_diagnostics(outcomes[0]) == without_diagnostics(outcomes[1])
    diag = outcomes[1]["layers"][0]["retained_prefix_diagnostics"]
    assert diag["selection_status"] == "partial_layer_not_advanced"
    assert diag["retained_nonterminal_prefixes"][0]["actions"] == ["open"]
    assert outcomes[1]["model_transition_calls"] == 2


def test_diagnostic_work_remains_inside_deadline(monkeypatch):
    ticks = [0.]
    monkeypatch.setattr(search, "time", SimpleNamespace(perf_counter=lambda: ticks[0]))
    original = search._prefix_progress
    def delayed(*args):
        result = original(*args)
        ticks[0] += 2.
        return result
    monkeypatch.setattr(search, "_prefix_progress", delayed)
    with pytest.raises(search.ObservedSearchLimit) as error:
        search.observed_beam_search(opening_tree(), max_calls=20, beam_width=2,
            seconds=1., retained_prefix_diagnostics=True)
    assert error.value.accounting["time_cap_reached"]
    assert error.value.accounting["model_transition_calls"] == 1


@pytest.mark.parametrize("bad", [1, "yes", None])
def test_optional_switch_rejects_ambiguous_types_before_task_queries(bad):
    task = opening_tree()
    with pytest.raises(ValueError, match="explicit bool"):
        search.observed_beam_search(task, max_calls=1, beam_width=2, retained_prefix_diagnostics=bad)
    assert not any(task.counters.values())


def test_terminal_root_and_empty_inventory_have_no_fabricated_progress():
    for task in (PublicTree({}), PublicTree({}, terminated=True)):
        path, report = search.observed_beam_search(task, max_calls=0, beam_width=2,
            retained_prefix_diagnostics=True)
        for row in report["layers"]:
            diag = row["retained_prefix_diagnostics"]
            assert not diag["retained_nonterminal_prefixes"] and not diag["best_terminal_prefixes"]


@pytest.mark.skipif("OBSERVED_SEARCH_BASELINE" not in os.environ, reason="historical source only in candidate review")
def test_default_matches_frozen_baseline_for_all_control_budgets():
    baseline = load_file(os.environ["OBSERVED_SEARCH_BASELINE"], "historical_observed_search")
    for mode in ("eager", "lazy_planning"):
        for cap in (0, 1, 3, 5, 20):
            a, b = opening_tree(), opening_tree()
            args = dict(max_calls=cap, beam_width=2, seconds=20., transition_mode=mode)
            pa, ra = baseline.observed_beam_search(a, **args)
            pb, rb = search.observed_beam_search(b, **args)
            assert pa == pb and without_diagnostics(ra) == without_diagnostics(rb)
            assert a.counters == b.counters
