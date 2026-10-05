"""Analytical source fixtures and metadata; no patient decoding or experiments."""
from dataclasses import asdict, replace
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import prepare_real_training_cases as legacy
import resectionlab.native_access_preparation as access
from resectionlab.geometry import AccessWindow, ToolGeometry
from resectionlab.native_ingress import IngressScreeningResult
from resectionlab.native_resection import NativeResectionEngine
from resectionlab.native_spatial_task import NativeSpatialCase, NativeSpatialTask

COHORT = Path(__file__).resolve().parents[1] / legacy.COHORT_PATH


def fixture_source(*, reference=None, crop_shape=(7, 7, 7), affine=None):
    support = np.zeros((7, 7, 7), bool); support[1:6, 1:6, 1:6] = True
    nominal = np.zeros_like(support); nominal[2:5, 2:5, 2:5] = True
    frame = np.eye(4) if affine is None else affine
    geometry, derivation = access.derive_access(nominal, support, frame, "sub-PAT25")
    source = NativeSpatialCase(np.indices(support.shape).sum(axis=0), support,
        nominal if reference is None else reference, frame, AccessWindow(**geometry),
        (ToolGeometry("analytic-fine", 1.25, .45, 10., 35., 2.),
         ToolGeometry("analytic-wide", 2.25, 1.1, 10., 35., 3.)),
        nominal_target=nominal, target_derivation="permitted analytical annotation",
        crop_shape=crop_shape, proposal_mode="nominal_cavity_v1")
    return source, derivation


@pytest.fixture
def initial():
    source, derivation = fixture_source()
    task = NativeSpatialTask(source, max_steps=3)
    return task, {"subject": "sub-PAT25", "cohort_json": COHORT.read_bytes(),
        "expected_source_hash": source.source_hash, "expected_model_hash": task.decision_model_hash,
        "access_derivation": derivation}


def test_old_caller_reexports_single_unchanged_derivation():
    assert legacy.derive_access is access.derive_access
    assert legacy.ACCESS_RULE == access.ACCESS_RULE
    nominal = np.zeros((7, 7, 7), bool); nominal[3, 3, 3] = True
    geometry, row = legacy.derive_access(nominal, np.ones_like(nominal), np.diag([1., 2., 3., 1.]), "sub-PAT25")
    assert row["representative_voxel"] == [3, 3, 3]
    assert [x["distance_mm"] for x in row["six_axis_exit_distances_mm"]] == [3.5, 3.5, 7., 7., 10.5, 10.5]
    assert geometry == {"center_mm": [-.5, 6., 9.], "normal_inward": [1., 0., 0.],
        "radius_mm": 6., "window_id": "PAT25-main-provisional-annotation-axis-v1"}


def test_legacy_task_factory_still_uses_declared_access_and_default_path(monkeypatch):
    import resectionlab.native_spatial_task as native
    observed = {}
    def original(case, **kwargs): observed.update(kwargs); return "unchanged-result"
    monkeypatch.setattr(native, "native_spatial_task_from_case", original)
    monkeypatch.setattr(access, "prepare_native_access", lambda *a, **k: pytest.fail("No default opt-in"))
    geometry, _ = access.derive_access(np.ones((3, 3, 3), bool), np.ones((3, 3, 3), bool), np.eye(4), "sub-PAT25")
    common = {"tools": [asdict(ToolGeometry("x", 1., .5, 10., 30., 2.))],
        "max_steps": 3, "track": "annotation_assisted", "adapter_options": {"crop_shape": [64, 64, 64]}}
    assert legacy._construct_task(object(), {"access": geometry, "research_support_acknowledgment": {"opaque": "existing"}}, common, None) == "unchanged-result"
    assert observed["access"].window_id == geometry["window_id"]
    assert observed["crop_shape"] == [64, 64, 64] and observed["max_steps"] == 3


def test_opt_in_screens_only_and_preserves_original_task(initial, monkeypatch):
    task, kwargs = initial
    before = task._state_seal, task.decision_model_hash, id(task._inventory)
    for method in ("preview_stroke", "commit_preview", "execute_stroke"):
        monkeypatch.setattr(NativeResectionEngine, method, lambda *a, **k: pytest.fail("No full motion or commit"))
    monkeypatch.setattr(NativeSpatialTask, "__init__", lambda *a, **k: pytest.fail("No selected inventory"))
    result = access.prepare_native_access(task, **kwargs)
    record = result.to_dict()
    assert record["status"] == "selected", record
    assert record["selected_exit"] == [0, -1]
    assert before == (task._state_seal, task.decision_model_hash, id(task._inventory))
    assert record["native_preview_calls"] == 0 and not record["selected_inventory_constructed"]
    assert record["original_preparation"]["declared_slots"] == 78
    assert record["original_preparation"]["elapsed_seconds"] is None
    assert record["screening"]["screen_count"] <= 468
    assert record["reward"] == asdict(task.reward_spec) and record["horizon"] == task.max_steps
    with pytest.raises(TypeError): result.receipt["horizon"] = 99
    with pytest.raises(TypeError): result.receipt["derived_exits"][0]["distance_mm"] = 0


@pytest.mark.parametrize("field", ["expected_source_hash", "expected_model_hash"])
def test_upstream_source_join_is_required(initial, field):
    task, kwargs = initial; kwargs[field] = "sha256:" + "0" * 64
    with pytest.raises(ValueError, match="upstream"): access.prepare_native_access(task, **kwargs)


def test_saved_derivation_cannot_change_distance_or_support_rule(initial):
    task, kwargs = initial
    kwargs["access_derivation"]["six_axis_exit_distances_mm"][0]["distance_mm"] = np.nextafter(2.5, np.inf).item()
    with pytest.raises(ValueError, match="derivation"): access.prepare_native_access(task, **kwargs)


@pytest.mark.parametrize("subject", ["sub-PAT26", "sub-PAT29", "sub-PAT31", "invented"])
def test_nontrain_roles_fail_before_task_access(subject):
    with pytest.raises(ValueError, match="permitted"):
        access.prepare_native_access(object(), subject=subject, cohort_json=COHORT.read_bytes(),
            expected_source_hash=None, expected_model_hash=None, access_derivation=None)


def test_changed_cohort_fails_before_task_access():
    with pytest.raises(ValueError, match="pinned"):
        access.prepare_native_access(object(), subject="sub-PAT25", cohort_json=COHORT.read_bytes()+b" ",
            expected_source_hash=None, expected_model_hash=None, access_derivation=None)


def test_cancel_returns_no_selected_source(initial):
    task, kwargs = initial
    result = access.prepare_native_access(task, **kwargs, cancelled=lambda: True)
    assert result.selected_case is None and result.receipt["status"] == "interrupted"
    assert result.receipt["screening"] is None


@pytest.mark.parametrize("status", ["no_ingress_admissible_access", "failed", "interrupted"])
def test_unselected_or_incomplete_screen_never_falls_back(initial, monkeypatch, status):
    task, kwargs = initial
    monkeypatch.setattr(access, "screen_axis_accesses", lambda *a, **k: IngressScreeningResult(status, None, (), 0))
    result = access.prepare_native_access(task, **kwargs)
    assert result.selected_case is None and result.receipt["selected_exit"] is None
    assert result.receipt["status"] == status


def test_final_callback_source_mutation_refuses_selection(initial, monkeypatch):
    task, kwargs = initial
    def damaged(*args, **kw):
        object.__setattr__(task.case, "support_derivation", "replaced after checks")
        return IngressScreeningResult("selected", (0, -1), (), 0)
    monkeypatch.setattr(access, "screen_axis_accesses", damaged)
    result = access.prepare_native_access(task, **kwargs)
    assert result.selected_case is None and result.receipt["status"] == "failed"


def test_later_selected_motion_outcomes_cannot_influence_source_choice(initial, monkeypatch):
    task, kwargs = initial
    monkeypatch.setattr(access, "screen_axis_accesses", lambda *a, **k: IngressScreeningResult("selected", (2, 1), (), 0))
    result = access.prepare_native_access(task, **kwargs)
    assert result.receipt["selected_exit"] == (2, 1)
    assert result.selected_case.access.window_id == "PAT25-existing-exit-axis2-sign+1-v1"
    assert result.selected_case.tools == task.case.tools
    np.testing.assert_array_equal(result.selected_case.reference_target, task.case.reference_target)
    assert result.selected_case.support_provenance == task.case.support_provenance
    assert result.selected_case.crop_shape == task.case.crop_shape
    assert not result.receipt["automatic_fallback"]
