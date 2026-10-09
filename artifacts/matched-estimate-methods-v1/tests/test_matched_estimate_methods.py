"""Small generated behavioral tests using explicitly scripted policies only."""
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from test_research_estimate_planning import fixture_spec
from resectionlab.core import semantic_digest
from resectionlab.native_spatial_task import make_native_opening_task
from resectionlab.research_estimate_planning import _nominal_task
from matched_estimate_methods import (
    METHODS, MatchedBudget, evaluate_matched_estimates, plan_matched_estimates,
)

TRAINING = {method: {"lineage": "scripted_unit_test_policy_not_a_trained_model", "optimizer_updates": 0}
            for method in ("IL", "RL")}


class ScriptedSequence(torch.nn.Module):
    """Fixture rule reads observed geometry/state, never a target reference."""
    def __init__(self, *, stop_on_long=False, fail_after=None):
        super().__init__()
        self.register_buffer("frozen_identity", torch.tensor([float(stop_on_long)]))
        self.stop_on_long, self.fail_after, self.calls = stop_on_long, fail_after, 0
        self.eval()

    def forward(self, observation):
        self.calls += 1
        if self.fail_after is not None and self.calls > self.fail_after:
            raise RuntimeError("scripted forward failure")
        scores = torch.full((len(observation.action_ids),), -10.)
        scores[0] = 0.
        if (self.stop_on_long and len(observation.action_ids) > 1
                and np.max(observation.action_geometry[1:, 12]) > 50):
            scores[0] = 20.
        else:
            first = observation.state_features[0] == 0
            for i, (tool, geometry) in enumerate(zip(observation.action_tool_ids, observation.action_geometry)):
                if ((first and tool == "short-wide-opener" and geometry[6] == 1.)
                        or (not first and tool == "long-narrow-cutter" and geometry[6] == 5.)):
                    scores[i] = 10.
        return scores, torch.tensor(float("nan"))  # Value never scores search.


def run_fixture(spec, output, *, il=None, rl=None, budget=MatchedBudget()):
    il, rl = il or ScriptedSequence(), rl or ScriptedSequence(stop_on_long=True)
    suite = plan_matched_estimates(spec, il_policy=il, rl_policy=rl,
        training_accounting=TRAINING, output=output, budget=budget)
    return suite, il, rl


@pytest.fixture(scope="module")
def pair(tmp_path_factory):
    root = tmp_path_factory.mktemp("matched_generated")
    baseline = fixture_spec()
    long_tools = replace(baseline, tools=tuple(replace(tool, working_length_mm=120.) for tool in baseline.tools))
    rows = []
    for name, spec in (("baseline", baseline), ("actual_long_tools", long_tools)):
        suite, il, rl = run_fixture(spec, root / name)
        assert all(row["status"] == "complete" for row in suite["methods"].values()), suite["methods"]
        called = []
        def load_reference():
            # Durable method and full suite files predate private reference access.
            assert all((root / name / (method + "-strategy.json")).is_file() for method in METHODS)
            assert json.loads((root / name / "planning-suite.json").read_text())["seal_hash"] == suite["seal_hash"]
            called.append(True)
            return spec.target.mask
        counts = (il.calls, rl.calls)
        report = evaluate_matched_estimates(suite, spec, load_reference=load_reference,
            output=root / (name + "-evaluation.json"))
        assert called == [True] and (il.calls, rl.calls) == counts
        rows.append((spec, suite, report))
    return rows


def test_baseline_reuses_complete_sequence_and_hybrid_keeps_strong_search(pair):
    _, suite, report = pair[0]
    assert set(suite["methods"]) == set(METHODS)
    assert len({row["strategy"]["plan"]["initial_observation_hash"] for row in suite["methods"].values()}) == 1
    for method in METHODS:
        assert report["methods"][method]["evaluation_status"] == "accepted"
        assert report["methods"][method]["outcomes"]["target_removed_mm3"] == 2.
        assert len(suite["methods"][method]["strategy"]["physical_history"]) == 2
        assert suite["methods"][method]["online_budget"]["counting_reliable"]
        assert suite["methods"][method]["online_budget"]["native_preview_entries"] > 0
    for method in ("SEARCH", "HYBRID"):
        assert suite["methods"][method]["details"]["call_cap_reached"] is False
        assert suite["methods"][method]["details"]["beam_pruned_prefixes"] == 0
    assert report["methods"]["SEARCH"]["outcomes"]["total_reward"] == pytest.approx(1.1)
    assert report["methods"]["HYBRID"]["outcomes"] == report["methods"]["SEARCH"]["outcomes"]


def test_actual_long_tool_geometry_is_recertified_and_scripted_stop_does_not_poison_hybrid(pair):
    spec, suite, report = pair[1]
    # This controlled policy is deliberately scripted to STOP; it is not an RL result.
    assert all(tool.working_length_mm == 120. for tool in spec.tools)
    observed = _nominal_task(spec).observation()
    assert np.all(observed.action_geometry[1:, 12] == 120.)
    assert suite["methods"]["RL"]["strategy"]["plan"]["action_ids"] == ["STOP"]
    assert report["methods"]["RL"]["outcomes"]["target_removed_mm3"] == 0.
    assert report["methods"]["SEARCH"]["outcomes"]["target_removed_mm3"] > 0
    assert report["methods"]["HYBRID"]["outcomes"] == report["methods"]["SEARCH"]["outcomes"]
    assert suite["methods"]["HYBRID"]["details"]["guidance"] == "actor_expansion_order_only"
    assert suite["methods"]["HYBRID"]["details"]["actor_forward_calls"] > 0


def test_original_estimate_adapter_observations_match_training_fixture_arrays():
    permitted = _nominal_task(fixture_spec()).observation()
    original = make_native_opening_task().observation()
    for name in ("image_channels", "coverage", "channel_available", "affine_ras_mm", "spacing_mm",
                 "action_geometry", "action_mask", "state_features"):
        np.testing.assert_array_equal(getattr(permitted, name), getattr(original, name))
    assert permitted.fingerprint != original.fingerprint  # Honest distinct estimate/source provenance.


def test_generated_reference_change_alters_only_independent_endpoints(pair, tmp_path, monkeypatch):
    spec, suite, report = pair[0]
    frozen = semantic_digest(suite)
    def forbidden(*args, **kwargs):
        raise AssertionError("Evaluation must not request policy/search decisions")
    monkeypatch.setattr("matched_estimate_methods.observed_beam_search", forbidden)
    monkeypatch.setattr("matched_estimate_methods._rollout_actor", forbidden)
    changed = evaluate_matched_estimates(suite, spec,
        load_reference=lambda: np.zeros_like(spec.target.mask), output=tmp_path / "counterfactual.json")
    for method in METHODS:
        row = changed["methods"][method]
        assert row["evaluation_status"] == "accepted"
        assert row["outcomes"]["target_removed_mm3"] == 0
        assert row["outcomes"]["normal_removed_mm3"] == 6
        assert row["physical_history_hash"] == report["methods"][method]["physical_history_hash"]
    assert semantic_digest(suite) == frozen
    assert changed["patient_generalization"] is changed["clinical_validation"] is None


def test_unfinished_method_denominator_rejected_before_reference_load(pair, tmp_path):
    spec, suite, _ = pair[0]
    changed = json.loads(json.dumps(suite))
    changed["methods"]["RL"]["status"] = "started"
    changed.pop("seal_hash")
    changed["seal_hash"] = semantic_digest(changed)
    def forbidden():
        raise AssertionError("All method slots must be committed first")
    with pytest.raises(ValueError, match="Every method must finish"):
        evaluate_matched_estimates(changed, spec, load_reference=forbidden, output=tmp_path / "never.json")


def test_estimate_abstention_does_not_become_zero_score_or_execute_actor(tmp_path):
    spec = fixture_spec(qc_status="abstain")
    suite, il, rl = run_fixture(spec, tmp_path / "abstain")
    assert il.calls == rl.calls == 0
    assert all(row["status"] == "abstained" for row in suite["methods"].values())
    def forbidden():
        raise AssertionError("No sealed strategy exists to evaluate")
    report = evaluate_matched_estimates(suite, spec, load_reference=forbidden, output=tmp_path / "abstain.json")
    assert report["private_reference_loaded"] is False
    assert all(row["outcomes"] is None for row in report["methods"].values())


def test_actor_failure_retains_partial_sequence_and_method_denominator(tmp_path):
    spec = fixture_spec()
    suite, _, _ = run_fixture(spec, tmp_path / "failure", rl=ScriptedSequence(fail_after=1))
    assert set(suite["methods"]) == set(METHODS)
    row = suite["methods"]["RL"]
    assert row["status"] == "failed" and row["strategy"] is None and row["outcomes"] is None
    assert row["details"]["actor_forward_calls"] == 2
    assert len(row["partial_nominal_history"]) == 1
    report = evaluate_matched_estimates(suite, spec, load_reference=lambda: spec.target.mask,
        output=tmp_path / "failure-evaluation.json")
    assert report["methods"]["RL"]["outcomes"] is None
    assert report["methods"]["SEARCH"]["outcomes"]["target_removed_mm3"] == 2


def test_call_limited_search_stop_is_labelled_incomplete_search_not_optimality(tmp_path):
    spec = fixture_spec()
    suite, _, _ = run_fixture(spec, tmp_path / "budget", budget=MatchedBudget(search_calls=1))
    for method in ("SEARCH", "HYBRID"):
        row = suite["methods"][method]
        assert row["status"] == "complete"
        assert row["details"]["call_cap_reached"] is True
        assert row["details"]["completed_layers"] == 0
        assert row["strategy"]["plan"]["action_ids"] == ["STOP"]


def test_changed_actor_buffers_invalidate_its_route_instead_of_scoring_it(tmp_path):
    class MutatingPolicy(ScriptedSequence):
        def forward(self, observation):
            self.frozen_identity.add_(1.)
            return super().forward(observation)
    spec = fixture_spec()
    suite, _, _ = run_fixture(spec, tmp_path / "mutating", il=MutatingPolicy())
    assert suite["methods"]["IL"]["status"] == "failed"
    assert suite["methods"]["IL"]["strategy"] is None
    assert "Frozen actor changed" in suite["methods"]["IL"]["error"]
    assert suite["methods"]["SEARCH"]["status"] == "complete"


def test_native_preview_exhaustion_retains_failed_slots_without_stop_scores(tmp_path):
    spec = fixture_spec()
    suite, il, rl = run_fixture(spec, tmp_path / "preview_limit", budget=MatchedBudget(native_previews=1))
    assert il.calls == rl.calls == 0
    assert all(row["status"] == "failed" and row["strategy"] is None
               for row in suite["methods"].values())
    assert all(row["online_budget"]["failure"] == "native_preview_entry_limit"
               for row in suite["methods"].values())
    def forbidden():
        raise AssertionError("A failed partial attempt is not a STOP strategy")
    report = evaluate_matched_estimates(suite, spec, load_reference=forbidden, output=tmp_path / "failure.json")
    assert all(row["outcomes"] is None for row in report["methods"].values())
