"""Generated DEVELOPMENT controls for durable scan-estimate strategy records.

No patient files, stored model checkpoints, training, or acquired reference
anatomy are opened. Tests intentionally exercise a multi-action physical route.
"""
import json
from dataclasses import replace

import numpy as np
import pytest

from test_research_estimate_planning import fixture_spec, plan, stop_planner, target_reaching_planner
from resectionlab.research_estimate_planning import (
    _nominal_task, _physical_hash, evaluate_sealed_research_plan,
    research_strategy_from_record, research_strategy_to_record,
)
from resectionlab.core import semantic_digest


@pytest.fixture(scope="module")
def search_result():
    spec = fixture_spec()
    sealed = plan(spec, planner=target_reaching_planner)
    assert len(sealed.action_ids) == 2 and "STOP" not in sealed.action_ids
    record = research_strategy_to_record(sealed, spec)
    return spec, sealed, record


def test_multiaction_json_roundtrip_preserves_route_microsteps_and_resulting_state(search_result, tmp_path):
    spec, sealed, record = search_result
    destination = tmp_path / "generated-strategy.json"
    destination.write_text(json.dumps(record, allow_nan=False))
    loaded = research_strategy_from_record(json.loads(destination.read_text()), spec)
    assert loaded.seal_hash == sealed.seal_hash
    assert type(loaded.action_ids) is tuple and loaded.action_ids == sealed.action_ids
    assert _physical_hash(record["physical_history"]) == sealed.physical_history_hash
    assert all(row["microsteps"] for row in record["physical_history"])
    assert record["physical_history"][0]["tool_id"] != record["physical_history"][1]["tool_id"]
    replay = _nominal_task(spec)
    for action in sealed.action_ids:
        replay.advance_planning(action)
    # Compare sparse recorded state to engine arrays, independently of the
    # exporter, which derives these same sets by unioning committed history.
    state = record["terminal_state"]
    assert state["removed_indices_native"] == np.argwhere(replay._engine.removed_mask).tolist()
    assert state["contact_indices_native"] == np.argwhere(replay._engine.contact_mask).tolist()
    assert state["retained_contact_indices_native"] == np.argwhere(
        replay._engine.contact_mask & ~replay._engine.removed_mask).tolist()
    assert state["steps_taken"] == 2 and state["remaining_steps"] == 0
    assert state["terminal_reason"] == "HORIZON"
    assert state["current_tool_id"] == replay._current_tool
    assert state["source_shape"] == list(spec.sources[0].image.shape)
    assert state["affine_ras_mm"] == spec.sources[0].affine_ras_mm.tolist()
    assert state["lineage"] == "nominal_simulation_replay_not_observed_patient_state"
    assert record["recording_accounting"] == {"nominal_replay_transition_calls": 2}


def test_stop_after_opening_retains_committed_state(search_result):
    spec, search, _ = search_result
    def open_then_stop(task):
        return (search.action_ids[0], "STOP"), {"model_transition_calls": 0, "actor_forward_calls": 0}
    sealed = plan(spec, planner=open_then_stop)
    record = research_strategy_to_record(sealed, spec)
    state = record["terminal_state"]
    assert state["terminal_reason"] == "STOP" and state["steps_taken"] == 2
    assert state["current_tool_id"] == record["physical_history"][0]["tool_id"]
    assert state["removed_indices_native"] == sorted(record["physical_history"][0]["removed_indices_native"])
    assert record["physical_history"][-1] == {
        "action_id": "STOP", "insertion_distance_mm": 0., "complete_tool_path_length_mm": 0.}
    assert research_strategy_from_record(json.loads(json.dumps(record)), spec).seal_hash == sealed.seal_hash


def test_immediate_stop_records_empty_physical_delta():
    spec = fixture_spec()
    sealed = plan(spec, planner=stop_planner)
    record = research_strategy_to_record(sealed, spec)
    state = record["terminal_state"]
    assert state["terminal_reason"] == "STOP" and state["remaining_steps"] == 1
    assert state["current_tool_id"] is None
    assert state["removed_indices_native"] == state["contact_indices_native"] == []
    assert research_strategy_from_record(record, spec).seal_hash == sealed.seal_hash


@pytest.mark.parametrize("changed", ["cell", "microstep", "affine", "state", "outcome", "accounting"])
def test_changed_record_cannot_replay_as_the_sealed_strategy(search_result, changed):
    spec, _, original = search_result
    record = json.loads(json.dumps(original))
    if changed == "cell":
        record["physical_history"][0]["removed_indices_native"][0][0] += 1
    elif changed == "microstep":
        record["physical_history"][0]["microsteps"].pop()
    elif changed == "affine":
        record["terminal_state"]["affine_ras_mm"][0][3] += 1
    elif changed == "state":
        record["terminal_state"]["removed_indices_native"].pop()
    elif changed == "outcome":
        record["physical_history"][0]["reward"] = 999
    else:
        record["recording_accounting"]["nominal_replay_transition_calls"] = 0
    with pytest.raises(ValueError, match="differs from nominal replay"):
        research_strategy_from_record(record, spec)


def test_record_is_detached_and_contains_no_private_target_or_scores(search_result):
    spec, sealed, original = search_result
    record = research_strategy_to_record(sealed, spec)
    banned = {"reference_target", "reference_hash", "target_removed_mm3", "normal_removed_mm3",
              "total_reward", "reward", "function_unassessed_removed_volume_mm3", "outcome_scope"}
    def inspect(value):
        if isinstance(value, dict):
            assert not (set(value) & banned)
            for item in value.values(): inspect(item)
        elif isinstance(value, list):
            for item in value: inspect(item)
    inspect(record)
    record["terminal_state"]["removed_indices_native"].clear()
    record["physical_history"][0]["entry_mm"][0] += 50
    assert research_strategy_to_record(sealed, spec) == original
    sealed.assert_intact()


def test_roundtripped_strategy_scores_generated_worlds_only_after_it_is_fixed(search_result):
    spec, sealed, record = search_result
    loaded = research_strategy_from_record(json.loads(json.dumps(record)), spec)
    before = semantic_digest(record)
    calls = []
    def reference(value):
        def load():
            loaded.assert_intact()
            calls.append(loaded.seal_hash)
            return value
        return load
    result_a = evaluate_sealed_research_plan(loaded, spec, load_reference=reference(spec.target.mask))
    result_b = evaluate_sealed_research_plan(loaded, spec,
        load_reference=reference(np.zeros_like(spec.target.mask)))
    assert calls == [sealed.seal_hash, sealed.seal_hash]
    assert result_a["target_removed_mm3"] == 2. and result_b["target_removed_mm3"] == 0.
    assert result_a["physical_history_hash"] == result_b["physical_history_hash"]
    assert semantic_digest(record) == before
    assert research_strategy_to_record(loaded, spec) == record
    assert result_a["patient_generalization"] is result_b["clinical_deficit_probability"] is None


def test_cropped_native_roi_record_preserves_local_indices_and_physical_frame():
    coverage = np.ones((9, 9, 7), bool)
    coverage[0, 0, 0] = False
    spec = fixture_spec(coverage=coverage, roi_start=(2, 2, 0), roi_stop=(8, 8, 7))
    sealed = plan(spec, planner=target_reaching_planner)
    record = research_strategy_to_record(sealed, spec)
    state = record["terminal_state"]
    assert state["source_shape"] == [6, 6, 7]
    assert np.asarray(state["affine_ras_mm"])[:3, 3].tolist() == [2., 2., 0.]
    assert all(row["native_affine"] == state["affine_ras_mm"] for row in record["physical_history"])
    cells = np.asarray(state["removed_indices_native"])
    assert np.all(cells >= 0) and np.all(cells < [6, 6, 7])
    assert research_strategy_from_record(json.loads(json.dumps(record)), spec).seal_hash == sealed.seal_hash


@pytest.mark.parametrize("numeric_horizon", [2.0, True])
def test_export_and_import_reject_noninteger_horizon_without_changing_legacy_constructor(numeric_horizon):
    spec = fixture_spec()
    if type(numeric_horizon) is bool:
        spec = replace(spec, horizon=1)
    sealed = plan(spec, planner=stop_planner)
    malformed = replace(sealed, horizon=numeric_horizon)
    with pytest.raises(ValueError, match="exact integer horizon"):
        research_strategy_to_record(malformed, spec)
    record = research_strategy_to_record(sealed, spec)
    record["plan"] = {**malformed._payload(), "seal_hash": malformed.seal_hash}
    with pytest.raises(ValueError, match="exact integer horizon"):
        research_strategy_from_record(json.loads(json.dumps(record)), spec)
