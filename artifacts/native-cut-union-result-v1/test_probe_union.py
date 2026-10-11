"""Bounded generated controls; source only until root releases the test slot."""
from copy import deepcopy
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
LOADER = importlib.util.spec_from_file_location("fixed_generic_cut_union_probe", HERE / "probe_union.py")
P = importlib.util.module_from_spec(LOADER)
LOADER.loader.exec_module(P)


def reset_counts():
    P.RUN_COUNTS.update(native_strokes_started=0, native_strokes_returned=0, native_seconds=0.)
    P.COVERAGE_WORK.update({name: 0 for name in P.COVERAGE_WORK})
    P.GENERIC.RUN_COUNTS.update(native_strokes_started=0, native_strokes_returned=0, native_seconds=0.)


def default_projection(row):
    result = deepcopy(row)
    result.pop("timing_seconds", None)
    result.pop("independent_audit", None)
    for action in result["actions"]:
        action.pop("native_seconds", None)
    # Compare the same wire representation: live tuples become JSON lists in
    # the retained report. Keep every scalar, state hash and history hash exact.
    return json.loads(json.dumps(result))


def test_exact_retained_inputs_and_explicit_model_specification():
    spec = P.declaration()
    prior = P.retained_baseline(spec)
    assert len(prior["rows"]) == spec["expected_rows"] == 18
    assert spec["expected_action_slots"] == 180
    assert P.cut_spec(spec).to_record() == {
        "mode": "cumulative_affine_subcell_cut_union_v1", "max_depth": 3,
        "max_tracked_cells": 4096, "max_leaf_tests_per_preview": 4194304}
    for row in prior["rows"]:
        assert P.BASE.digest(P.physical_actions(row["grid"]["tool_id"], spec)) == row["physical_actions_sha256"]
    source, experimental = P.grid_case(1., spec["grids"][0], spec)
    old_source, default = P.GENERIC.grid_case(1., spec["grids"][0], spec)
    assert experimental.cut_union == P.cut_spec(spec)
    assert default.cut_union is None
    assert replace(experimental, cut_union=None).fingerprint == default.fingerprint
    assert source.semantic_hash == old_source.semantic_hash


def test_physical_treatment_or_saved_baseline_substitution_refused():
    spec = P.declaration()
    bad = deepcopy(spec); bad["target_tip_z_mm"] += .1
    with pytest.raises(ValueError, match="Matched physical baseline"):
        P.retained_baseline(bad)
    bad = deepcopy(spec); bad["retained_baseline"]["report_sha256"] = "0"*64
    with pytest.raises(ValueError, match="baseline report hash"):
        P.retained_baseline(bad)


@pytest.mark.parametrize("grid_index", [0, 1])
def test_current_default_exact_scalar_state_and_history_parity_with_retained_coarse_rows(grid_index):
    reset_counts()
    spec = P.declaration()
    prior = P.retained_baseline(spec)
    row, _, config, _ = P.GENERIC.numerical_row(1., spec["grids"][grid_index], spec)
    old = next(r for r in prior["rows"] if r["row_id"] == row["row_id"])
    assert config.cut_union is None
    assert default_projection(row) == default_projection(old)


def test_union_upper_bounds_allow_more_than_individual_capsule_lower():
    static = {"contained_physical_lower_mm3": 1., "contact_physical_upper_mm3": 4.}
    P.check_volume_bounds(3., .2, static, .4)
    with pytest.raises(AssertionError, match="active-contact upper"):
        P.check_volume_bounds(4.1, .2, static, .4)
    with pytest.raises(AssertionError, match="continuous target cap"):
        P.check_volume_bounds(3., .5, static, .4)


def test_one_real_coarse_union_sequence_keeps_denominator_and_independent_proof():
    from resectionlab.evaluation import independent_check_native_history
    reset_counts()
    spec = P.declaration()
    row, case, config, history = P.numerical_row(1., spec["grids"][2], spec)
    assert len(row["actions"]) == 10
    assert row["native_calls"] == P.RUN_COUNTS["native_strokes_started"] == P.RUN_COUNTS["native_strokes_returned"]
    assert row["native"]["accepted_actions"] == len(history)
    assert row["native"]["partial_cells_kept_occupied"]
    assert row["coverage"]["attempted_leaf_tests"] > 0
    assert row["coverage"]["coverage_only_accepted_actions"] > 0
    assert history
    audit = independent_check_native_history(case, config.tools, history,
        tissue_mask=config.tissue_mask, access=config.access,
        distance_backend="batch", distance_batch_size=256, cut_union_spec=config.cut_union.to_record())
    assert audit.feasible and audit.complete_tool_checked
    for i, action in enumerate(row["actions"]):
        if action["status"] == "rejected":
            assert action["source_state_hash"] == action["result_state_hash"]
            assert action["removed_source_cell_mm3"] == 0
            assert all(a["status"] == "unexecuted_prior_refusal" for a in row["actions"][i+1:])


def test_explicit_coverage_budget_refusal_has_zero_commit_and_charged_attempt(monkeypatch):
    from resectionlab.native_cut_union import CutUnionSpec
    original = P.grid_case
    def small(spacing, grid, spec):
        case, config = original(spacing, grid, spec)
        return case, replace(config, cut_union=CutUnionSpec(max_depth=3, max_tracked_cells=4096, max_leaf_tests_per_preview=1))
    monkeypatch.setattr(P, "grid_case", small)
    reset_counts()
    spec = P.declaration()
    row, _, _, history = P.numerical_row(1., spec["grids"][2], spec)
    assert not history and row["native_calls"] == 1
    assert row["actions"][0]["reason"] == "CUMULATIVE_CUT_COVERAGE_BUDGET"
    assert row["coverage"]["budget_refusals"] == 1
    assert P.COVERAGE_WORK["returned_previews_accounted"] == P.COVERAGE_WORK["budget_refusals"] == 1
    assert row["actions"][0]["source_state_hash"] == row["actions"][0]["result_state_hash"]
    assert row["native"]["removed_cells"] == 0
    assert all(a["status"] == "unexecuted_prior_refusal" for a in row["actions"][1:])


def test_missing_rows_unaccepted_and_explicit_checker_hook_restored(tmp_path, monkeypatch):
    from resectionlab import evaluation
    seen = []
    def fake_check(*args, **kwargs):
        seen.append(kwargs); return SimpleNamespace(feasible=True)
    monkeypatch.setattr(evaluation, "independent_check_native_history", fake_check)
    def capped(output):
        evaluation.independent_check_native_history("saved-prefix")
        P.BASE.write_json(output/"report.json", {"status": "cooperative_cap_reached", "rows": []})
        return 1
    monkeypatch.setattr(P, "_original_worker", capped)
    assert P.worker(tmp_path) == 1
    assert evaluation.independent_check_native_history is fake_check
    assert seen == [{"distance_backend": "batch", "distance_batch_size": 256,
                     "cut_union_spec": P.cut_spec(P.declaration()).to_record()}]
    record = json.loads((tmp_path/"report.json").read_text())
    assert len(record["missing_rows"]) == len(record["matched_comparison"]) == 18
    assert record["planned_action_slots"] == 180
    assert record["verified_complete_sequences"] == record["accepted_complete_sequences"] == 0
    assert all(r["experimental_status"] == "missing" for r in record["matched_comparison"])
    assert P.DEADLINE is P.LIVE_OUTPUT is None


def test_existing_output_never_overwritten(tmp_path):
    with pytest.raises(FileExistsError):
        P.launch(tmp_path)


def test_predeclared_criterion_requires_gain_no_regression_and_all_checks():
    prior = {"rows": [{"row_id": str(i), "native": {"complete_sequence": i == 0}} for i in range(18)]}
    rows = [{"row_id": str(i), "native": {"complete_sequence": i in (0, 1), "accepted_actions": 10 if i in (0, 1) else 0,
             "removed_physical_target_mm3": .1 if i in (0, 1) else 0}, "coverage": {"budget_refusals": 0},
             "independent_audit": {"status": "completed", "certificate": {"feasible": True}},
             "exact_continuous_target_cap_mm3": .4} for i in range(18)]
    report = {"rows": rows, "status": "completed", "source_unchanged": True}
    assert P.useful_criterion(prior, report)["meets_predeclared_useful_criterion"]
    for change in ("regression", "audit", "cap", "missing", "invented_volume", "source"):
        bad = deepcopy(report)
        if change == "regression": bad["rows"][0]["native"]["complete_sequence"] = False
        elif change == "audit": bad["rows"][1]["independent_audit"]["status"] = "cancelled_cap_reached"
        elif change == "cap": bad["rows"][1]["coverage"]["budget_refusals"] = 1
        elif change == "missing": bad["rows"].pop()
        elif change == "invented_volume": bad["rows"][1]["native"]["removed_physical_target_mm3"] = .5
        elif change == "source": bad["source_unchanged"] = False
        assert not P.useful_criterion(prior, bad)["meets_predeclared_useful_criterion"]
