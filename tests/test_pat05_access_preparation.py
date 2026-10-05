"""Pinned metadata and mocked source boundaries only; no patient/native task."""
import copy
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import pat05_access_preparation as adapter
from resectionlab.geometry import AccessWindow
from resectionlab.simulation import RewardSpec


def canonical_from_metadata(profile):
    """Construct a candidate metadata fixture, never infer a patient mask."""
    old = profile["access_derivation"]
    matrix = np.asarray(profile["expected_native_grid_binding"]["original_affine_ras_mm"])
    rows = []
    for row in old["six_axis_exit_distances_mm"]:
        boundary = np.asarray(old["representative_voxel"], float)
        axis, sign = row["axis"], row["outward_sign"]
        boundary[axis] += sign * row["distance_mm"] / np.linalg.norm(matrix[:3, axis])
        rows.append({**row, "boundary_voxel": boundary.tolist()})
    return {"rule": "annotation-centroid-nearest-cell-six-axis-shortest-exit-radius6-v1",
        **{key: copy.deepcopy(old[key]) for key in ("annotation_centroid_voxels", "representative_voxel",
            "selected_boundary_voxel", "depth_to_annotation_representative_mm", "cortical_access_permitted")},
        "six_axis_exit_distances_mm": rows, "selection_uses_reward_or_native_preview": False}


@pytest.fixture
def metadata():
    records = adapter.load_pat05_historical_records()
    profile, original, executed = records.decoded()
    return records, profile, original, executed, canonical_from_metadata(profile)


@pytest.fixture
def source_boundary(monkeypatch, metadata):
    """Typed boundary double with tiny stand-ins; never an actual task or scan."""
    records, profile, original, executed, derived = metadata
    class Task:
        def _assert_frozen(self): pass
    monkeypatch.setattr(adapter, "NativeSpatialTask", Task)
    task = Task()
    task.case = SimpleNamespace(source_hash=executed["initial_task_metrics"]["source_hash"],
        _grid_record=copy.deepcopy(executed["initial_task_metrics"]["native_grid_reconciliation"]),
        nominal_target=np.ones((2, 2, 2)), observed_support=np.ones((2, 2, 2), bool),
        affine_ras_mm=np.asarray(profile["expected_native_grid_binding"]["original_affine_ras_mm"]),
        access=AccessWindow(**original["member"]["access"]))
    task.decision_model_hash = executed["decision_model_hash"]
    task._state_seal = "unchanged-analytical-boundary"
    task._steps = 0; task.terminated = False; task._planning = False
    task._engine = SimpleNamespace(revision=0, history=[])
    task.reward_spec = RewardSpec(**original["objective"])
    task.max_steps = original["settings"]["max_steps"]
    calls = []
    def derive(target, support, affine, subject):
        calls.append((target, support, affine, subject))
        return copy.deepcopy(profile["access"]), copy.deepcopy(derived)
    monkeypatch.setattr(adapter, "derive_access", derive)
    return task, records, calls


def test_metadata_loader_pins_only_three_existing_records(metadata):
    records, profile, original, executed, _ = metadata
    assert profile["study_id"] == "pat05-real-spatial-profile-v3-nominal64"
    assert "access_derivation" not in original["member"]
    assert len(executed["initial_task_metrics"]["native_grid_reconciliation"]) == 20
    assert all("boundary_voxel" not in row for row in profile["access_derivation"]["six_axis_exit_distances_mm"])
    assert records.decoded()[0] == profile


def test_canonical_metadata_preserves_history_and_labels_new_fields(source_boundary):
    task, records, calls = source_boundary
    before = records.profile, records.learning, records.receipt
    result = adapter.canonical_pat05_access_metadata(task, records)
    assert result["expected_source_hash"] == task.case.source_hash
    assert result["expected_model_hash"] == task.decision_model_hash
    assert len(result["complete_native_grid_binding"]["complete_grid"]) == 20
    assert result["newly_rederived_fields"] == ("six_axis_exit_distances_mm[*].boundary_voxel",)
    assert set(result["current_shared_rule_metadata"]) == {"rule", "selection_uses_reward_or_native_preview"}
    assert not result["historical_records_rewritten"] and not result["new_access_selected"]
    assert before == (records.profile, records.learning, records.receipt)
    assert len(calls) == 1 and calls[0][1] is task.case.observed_support and calls[0][2] is task.case.affine_ras_mm
    assert calls[0][3] == "sub-PAT05" and calls[0][0].dtype == bool
    with pytest.raises(TypeError): result["access_derivation"]["rule"] = "changed"


@pytest.mark.parametrize("field", ["profile", "learning", "receipt"])
def test_changed_historical_bytes_reject_before_task_access(metadata, field):
    records = metadata[0]
    values = {key: getattr(records, key) for key in ("profile", "learning", "receipt")}
    values[field] += b" "
    with pytest.raises(ValueError, match="historical metadata bytes"):
        adapter.canonical_pat05_access_metadata(object(), adapter.PAT05HistoricalRecords(**values))


@pytest.mark.parametrize("field", ["annotation_centroid_voxels", "representative_voxel", "selected_boundary_voxel"])
def test_every_historical_coordinate_requires_exact_equality(metadata, field):
    _, profile, _, _, derived = metadata
    derived[field][0] = np.nextafter(float(derived[field][0]), np.inf).item()
    with pytest.raises(ValueError, match="derivation differs"):
        adapter.verify_historical_access(profile["access"], derived, profile)


@pytest.mark.parametrize("index", range(6))
@pytest.mark.parametrize("field", ["axis", "outward_sign", "distance_mm"])
def test_every_historical_exit_triple_requires_exact_equality(metadata, index, field):
    _, profile, _, _, derived = metadata
    row = derived["six_axis_exit_distances_mm"][index]
    row[field] = np.nextafter(float(row[field]), np.inf).item()
    with pytest.raises(ValueError, match="six ordered"):
        adapter.verify_historical_access(profile["access"], derived, profile)


@pytest.mark.parametrize("field", ["center_mm", "normal_inward", "radius_mm", "window_id"])
def test_historical_selected_access_cannot_change(metadata, field):
    _, profile, _, _, derived = metadata
    geometry = copy.deepcopy(profile["access"])
    if field in ("center_mm", "normal_inward"):
        geometry[field][0] = np.nextafter(geometry[field][0], np.inf).item()
    elif field == "radius_mm": geometry[field] = np.nextafter(geometry[field], np.inf).item()
    else: geometry[field] += "changed"
    with pytest.raises(ValueError, match="selected access"):
        adapter.verify_historical_access(geometry, derived, profile)


@pytest.mark.parametrize("field", sorted(adapter.coverage.PAT05_COMPLETE_GRID_KEYS))
def test_all_twenty_grid_fields_are_required(source_boundary, field):
    task, records, _ = source_boundary
    del task.case._grid_record[field]
    with pytest.raises(adapter.coverage.PAT05TaskBindingMismatch) as caught:
        adapter.canonical_pat05_access_metadata(task, records)
    assert "native_grid." + field in caught.value.details["mismatched_fields"]


def test_uses_executed_objective_not_old_profile_objective(source_boundary):
    task, records, _ = source_boundary
    _, original, _ = records.decoded()
    task.max_steps = original["settings"]["max_steps"] + 1
    with pytest.raises(adapter.coverage.PAT05TaskBindingMismatch, match="max_steps"):
        adapter.canonical_pat05_access_metadata(task, records)


def test_source_or_model_join_rejects_before_derivation(source_boundary):
    task, records, calls = source_boundary
    task.decision_model_hash += "changed"
    with pytest.raises(ValueError, match="source/model"):
        adapter.canonical_pat05_access_metadata(task, records)
    assert not calls


def test_post_derivation_state_change_rejects_metadata(source_boundary, monkeypatch):
    task, records, _ = source_boundary
    original = adapter.derive_access
    def changed(*args):
        result = original(*args)
        task._state_seal = "mutated"
        return result
    monkeypatch.setattr(adapter, "derive_access", changed)
    with pytest.raises(RuntimeError, match="task changed"):
        adapter.canonical_pat05_access_metadata(task, records)
