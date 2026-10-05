"""Independent lazy-transition controls; analytic cells, no patient loading or fitting."""
from dataclasses import asdict, replace

import numpy as np
import pytest

from resectionlab.native_axis_simulation import CommittedTransitionInterrupted
from resectionlab.native_proposals import NominalCavityProposalConfig
from resectionlab.native_resection import NativeResectionEngine
from resectionlab.native_spatial_task import NativeSpatialTask, OPENING_TOOLS, make_native_opening_task
from resectionlab.observed_search import observed_beam_search


def analytic_source(provider):
    source = make_native_opening_task().case
    if provider == "nominal_cavity_v1":
        offsets = tuple((a, b) for a in (-1, 0, 1) for b in (-1, 0, 1))
        source = replace(source, proposal_mode=provider,
            proposal_config=NominalCavityProposalConfig(offsets))
    return source


def choose(task, tool, voxel):
    return next(row["action_id"] for row in task.candidate_inventory()["emitted"]
                if row["feasible"] and row["tool_id"] == tool and row["voxel"] == list(voxel))


def same_observation(first, second):
    assert first.fingerprint == second.fingerprint
    assert first.action_ids == second.action_ids and first.action_tool_ids == second.action_tool_ids
    for name in ("image_channels", "coverage", "channel_available", "affine_ras_mm",
                 "spacing_mm", "action_geometry", "action_mask", "state_features"):
        np.testing.assert_array_equal(getattr(first, name), getattr(second, name))


@pytest.mark.parametrize("provider", ["fixed_lattice", "nominal_cavity_v1"])
def test_lazy_branch_after_private_history_has_identical_public_state_and_independent_certificate(provider):
    source = analytic_source(provider)
    altered = replace(source, reference_target=np.ones(source.reference_target.shape, np.float32))
    left = NativeSpatialTask(source, max_steps=3)
    right = NativeSpatialTask(altered, max_steps=3)
    opening = choose(left, OPENING_TOOLS[0].tool_id, (4, 4, 1))
    assert left.step(opening).reward != right.step(opening).reward
    eager, lazy = left.planning_clone(), right.planning_clone()
    assert eager.metrics() == lazy.metrics()  # The old private reward is replaced too.
    distal = choose(eager, OPENING_TOOLS[1].tool_id, (4, 4, 5))
    expected, deferred = eager.step(distal), lazy.advance_planning(distal)
    assert deferred.observation is None and lazy._inventory is None
    assert deferred.reward == expected.reward and deferred.info == expected.info
    assert deferred.terminated == expected.terminated
    assert lazy.metrics() == eager.metrics()
    assert lazy._state_seal == eager._state_seal
    assert lazy._engine.history == eager._engine.history
    for field in ("remaining_mask", "removed_mask", "contact_mask", "connected_free_mask"):
        np.testing.assert_array_equal(getattr(lazy._engine, field), getattr(eager._engine, field))
    audit = lazy.independent_geometry_check()
    assert audit.feasible and asdict(audit) == asdict(eager.independent_geometry_check())
    assert lazy._inventory is None  # Metrics/audit do not secretly build the successor.
    branch = lazy.clone()
    same_observation(branch.observation(), expected.observation)
    assert branch.candidate_inventory() == eager.candidate_inventory()
    assert lazy._inventory is None and lazy._state_seal == eager._state_seal
    same_observation(lazy.observation(), expected.observation)
    assert lazy.metrics()["clinical_deficit_probability"] is None


@pytest.mark.parametrize("provider", ["fixed_lattice", "nominal_cavity_v1"])
def test_search_full_layers_keep_exact_results_while_avoiding_discarded_child_previews(provider, monkeypatch):
    task = NativeSpatialTask(analytic_source(provider), max_steps=3)
    original = NativeResectionEngine.preview_stroke
    counts = {"eager": 0, "lazy_planning": 0}
    active = {"mode": "eager"}

    def counted_preview(self, *args, **kwargs):
        counts[active["mode"]] += 1
        return original(self, *args, **kwargs)

    monkeypatch.setattr(NativeResectionEngine, "preview_stroke", counted_preview)
    results = {}
    for mode in counts:
        active["mode"] = mode
        results[mode] = observed_beam_search(task, max_calls=128, beam_width=1,
            seconds=10., transition_mode=mode)
    eager_path, eager = results["eager"]
    lazy_path, lazy = results["lazy_planning"]
    assert eager_path == lazy_path
    implementation_fields = {"planning_seconds", "transition_mode", "eager_transition_calls",
                             "lazy_planning_transition_calls"}
    assert {key: value for key, value in eager.items() if key not in implementation_fields} == {
        key: value for key, value in lazy.items() if key not in implementation_fields}
    assert not eager["call_cap_reached"] and eager["completed_layers"] >= 2
    assert eager["beam_pruned_prefixes"] > 0
    assert eager["eager_transition_calls"] == eager["model_transition_calls"]
    assert lazy["lazy_planning_transition_calls"] == lazy["model_transition_calls"]
    assert 0 < counts["lazy_planning"] < counts["eager"]
    assert task.metrics()["steps"] == 0 and not task._engine.removed_mask.any()


@pytest.mark.parametrize("provider", ["fixed_lattice", "nominal_cavity_v1"])
def test_search_preserves_a_postcommit_cancellation_as_failure_in_both_modes(provider, monkeypatch):
    canceled = {"value": False}
    task = NativeSpatialTask(analytic_source(provider), max_steps=3,
        cancelled=lambda: canceled["value"])
    original = NativeResectionEngine.commit_preview

    def commit_then_cancel(self, preview):
        result = original(self, preview)
        canceled["value"] = True
        return result

    monkeypatch.setattr(NativeResectionEngine, "commit_preview", commit_then_cancel)
    failures = []
    for mode in ("eager", "lazy_planning"):
        canceled["value"] = False
        with pytest.raises(CommittedTransitionInterrupted) as interrupted:
            observed_beam_search(task, max_calls=128, beam_width=1,
                seconds=10., transition_mode=mode)
        error = interrupted.value
        assert error.committed and error.info["removed_indices_native"]
        assert error.accounting["model_transition_calls"] == 1
        assert error.accounting["evaluated_transition_prefixes"] == 0
        assert error.accounting["implicit_stop_prefix_count"] == 1
        assert error.accounting["completed_layers"] == 0
        assert task.metrics()["steps"] == 0 and not task._engine.removed_mask.any()
        failures.append((error.reward, error.info))
    assert failures[0] == failures[1]
