"""Generated controls for the fixed-scene adapter; never acquired inputs."""
from copy import deepcopy
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
SPEC = importlib.util.spec_from_file_location("generic_resolution_probe", HERE / "probe.py")
P = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(P)


def test_actual_tools_reward_and_physically_fixed_sequence_across_grid_changes():
    from resectionlab.geometry import GENERIC_TOOLS
    from resectionlab.native_spatial_task import DEFAULT_NATIVE_SPATIAL_REWARD
    spec = P.declaration()
    assert spec["tools"] == [asdict(t) for t in GENERIC_TOOLS]
    assert spec["reward"] == asdict(DEFAULT_NATIVE_SPATIAL_REWARD)
    assert spec["expected_rows"] == 3*3*2 == 18
    assert spec["expected_action_slots"] == 18*10 == 180
    for tool in GENERIC_TOOLS:
        actions = P.physical_actions(tool.tool_id, spec)
        assert len(actions) == 10 and actions[0]["entry_mm"] == [0., 0., -2]
        assert actions[-1]["tip_mm"] == [0., 0., 2.2]
        assert all(a["tool_id"] == tool.tool_id for a in actions)
        assert len({tuple(a["tip_mm"]) for a in actions}) == 10
        for spacing in spec["spacings_mm"]:
            for grid in spec["grids"]:
                if grid["tool_id"] == tool.tool_id:
                    assert P.physical_actions(grid["tool_id"], spec) == actions
        proof = P.continuum_rationale(tool, spec)
        assert proof["disk_cover_margin_mm"] > .29
        assert proof["aperture_margin_mm"] > .75
        assert proof["preparation_shaft_highest_z_mm"] < -.29
        assert proof["target_shaft_highest_z_mm"] < 1.
        assert proof["preparation_to_target_plane_gap_mm"] > .09


def test_grid_phase_changes_cells_without_moving_physical_tool_access_or_target_plane():
    spec = P.declaration()
    a, ca = P.grid_case(1., spec["grids"][0], spec)
    b, cb = P.grid_case(1., spec["grids"][2], spec)
    c, cc = P.grid_case(1., spec["grids"][4], spec)
    assert ca.tools == cb.tools == cc.tools
    for other in (cb, cc):
        np.testing.assert_array_equal(ca.access.center_mm, other.access.center_mm)
        np.testing.assert_array_equal(ca.access.normal_inward, other.access.normal_inward)
        assert ca.max_tip_step_mm == other.max_tip_step_mm == .0625
        assert not np.array_equal(ca.affine, other.affine)
    assert len({ca.source_hash, cb.source_hash, cc.source_hash}) == 3
    for source, config in ((a, ca), (b, cb), (c, cc)):
        coords = np.indices(config.tissue_mask.shape).reshape(3, -1).T
        z = coords @ config.affine[2, :3] + config.affine[2, 3]
        expected = P.BASE.half_space_fractions(z-2.8, config.affine).astype(np.float32)
        expected[~config.tissue_mask.ravel()] = 0.
        np.testing.assert_array_equal(source.reference_target.ravel(), expected)


def test_tool_or_reward_change_is_refused_before_native_execution():
    spec = P.declaration()
    for field in ("tool", "reward"):
        bad = deepcopy(spec)
        if field == "tool": bad["tools"][0]["shaft_radius_mm"] = .45
        else: bad["reward"]["normal_per_mm3"] = 0.
        with pytest.raises(ValueError, match="tool or frozen reward"):
            P.grid_case(1., spec["grids"][0], bad)


@pytest.mark.parametrize("grid_index", [0, 1])
def test_one_coarse_actual_tool_row_preserves_refusal_and_audits_any_committed_prefix(grid_index):
    from resectionlab.evaluation import independent_check_native_history
    spec = P.declaration()
    P.RUN_COUNTS.update(native_strokes_started=0, native_strokes_returned=0, native_seconds=0.)
    row, case, config, history = P.numerical_row(1., spec["grids"][grid_index], spec)
    assert len(row["actions"]) == row["planned_actions"] == 10
    attempted = [r for r in row["actions"] if r["status"] != "unexecuted_prior_refusal"]
    assert len(attempted) == row["native_calls"] == P.RUN_COUNTS["native_strokes_started"]
    assert P.RUN_COUNTS["native_strokes_started"] == P.RUN_COUNTS["native_strokes_returned"]
    assert row["native"]["accepted_actions"] == len(history)
    for i, action in enumerate(row["actions"]):
        if action["status"] == "rejected":
            assert action["source_state_hash"] == action["result_state_hash"]
            assert action["removed_source_cell_mm3"] == 0
            assert all(r["status"] == "unexecuted_prior_refusal" for r in row["actions"][i+1:])
    assert row["native"]["partial_cells_kept_occupied"]
    assert row["native"]["removed_physical_mm3"] <= row["static_not_executable"]["contained_physical_lower_mm3"] + 1e-8
    assert row["static_not_executable"]["contained_physical_lower_mm3"] <= row["static_not_executable"]["contact_physical_upper_mm3"]
    assert all(r.get("removed_physical_target_mm3", 0.) <= 1e-10 for r in row["actions"][:-1])
    if history:
        audit = independent_check_native_history(case, config.tools, history,
            tissue_mask=config.tissue_mask, access=config.access, distance_backend="batch", distance_batch_size=256)
        assert audit.feasible


def test_missing_rows_remain_denominator_and_batch_hook_restores(tmp_path, monkeypatch):
    from resectionlab import evaluation
    old = evaluation.independent_check_native_history
    seen = []
    def fake_check(*args, **kwargs):
        seen.append(kwargs); return SimpleNamespace(feasible=True)
    monkeypatch.setattr(evaluation, "independent_check_native_history", fake_check)
    def incomplete(output):
        evaluation.independent_check_native_history("saved-prefix")
        P.BASE.write_json(output/"report.json", {"status": "cooperative_cap_reached", "rows": []})
        return 1
    monkeypatch.setattr(P, "_original_worker", incomplete)
    assert P.worker(tmp_path) == 1
    assert evaluation.independent_check_native_history is fake_check
    assert seen == [{"distance_backend": "batch", "distance_batch_size": 256}]
    report = json.loads((tmp_path/"report.json").read_text())
    assert len(report["planned_rows"]) == len(report["missing_rows"]) == 18
    assert report["planned_action_slots"] == 180
    assert report["verified_complete_sequences"] == report["accepted_complete_sequences"] == 0
    assert P.DEADLINE is None


def test_existing_output_is_never_retried_or_overwritten(tmp_path):
    with pytest.raises(FileExistsError): P._original_launch(tmp_path)
