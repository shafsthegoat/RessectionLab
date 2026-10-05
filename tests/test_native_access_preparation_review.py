"""Independent tiny analytical controls; no patient arrays, learning or releases."""
from dataclasses import asdict
import copy
import inspect
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from resectionlab import native_access_preparation as p
from resectionlab.core import array_digest
from resectionlab.geometry import AccessWindow, ToolGeometry
from resectionlab.native_resection import NativeResectionEngine
from resectionlab.native_spatial_task import NativeSpatialCase, NativeSpatialTask

ROOT = Path(__file__).resolve().parents[1]
COHORT = (ROOT / "manifests/experiments/btc-spatial-development-cohort-v1.json").read_bytes()
SUBJECT = "sub-PAT05"  # Declared upstream subject; every image below is an analytic fixture.
TOOLS = (ToolGeometry("analytic-narrow", .7, .15, 8., 35., 1.),
         ToolGeometry("analytic-broad", .9, .2, 8., 35., 1.5))


def make_task(*, crop=(9, 9, 9), private_variant=False, affine=None, reconciliation="none"):
    shape = (9, 9, 9)
    support = np.zeros(shape, bool)
    support[3:6, 3:6, 3:6] = True
    nominal = np.zeros(shape, bool)
    nominal[4, 4, 4] = True
    reference = nominal.copy()
    if private_variant:
        reference[:] = False
        reference[0, 0, 0] = True
    affine = np.eye(4) if affine is None else affine
    access, derivation = p.derive_access(nominal, support, affine, SUBJECT)
    source = NativeSpatialCase(np.arange(np.prod(shape), dtype=np.float32).reshape(shape),
        support, reference, affine, AccessWindow(**access), TOOLS,
        nominal_target=nominal, target_derivation="explicit tiny analytical nominal annotation",
        support_derivation="tiny analytical support; no patient source", crop_shape=crop,
        native_grid_reconciliation=reconciliation, proposal_mode="nominal_cavity_v1")
    task = NativeSpatialTask(source)
    return task, derivation


@pytest.fixture(scope="module")
def initial():
    return make_task()


def arguments(task, derivation):
    return dict(subject=SUBJECT, cohort_json=COHORT, expected_source_hash=task.case.source_hash,
                expected_model_hash=task.decision_model_hash, access_derivation=copy.deepcopy(derivation))


def test_derive_access_extraction_and_legacy_reexport_are_exact():
    expected = (ROOT / "artifacts/native-access-preparation-review-v1/original-derive-access.txt").read_text()
    assert inspect.getsource(p.derive_access) == expected
    sys.path.insert(0, str(ROOT / "scripts"))
    import prepare_real_training_cases as legacy
    assert legacy.derive_access is p.derive_access and legacy.ACCESS_RULE == p.ACCESS_RULE


def test_lexicographic_nominal_cell_and_physical_axis_distances_under_reflection():
    support = np.zeros((7, 7, 7), bool)
    support[1:6, 1:6, 1:6] = True
    nominal = np.zeros_like(support)
    nominal[2, 3, 3] = nominal[4, 3, 3] = True
    affine = np.diag([-2., 3., 4., 1.]); affine[:3, 3] = [17., -4., 9.]
    access, record = p.derive_access(nominal, support, affine, SUBJECT)
    assert record["representative_voxel"] == [2, 3, 3]
    assert record["annotation_centroid_voxels"] == [3., 3., 3.]
    rows = record["six_axis_exit_distances_mm"]
    assert [(r["axis"], r["outward_sign"]) for r in rows] == [(a, s) for a in range(3) for s in (-1, 1)]
    assert [r["distance_mm"] for r in rows] == [3., 7., 7.5, 7.5, 10., 10.]
    assert access["center_mm"] == [16., 5., 21.] and access["normal_inward"] == [-1., 0., 0.]
    assert not record["cortical_access_permitted"] and not record["selection_uses_reward_or_native_preview"]


@pytest.mark.parametrize("subject", ["sub-PAT26", "sub-PAT27", "sub-PAT29", "sub-PAT31", "unknown"])
def test_nontrain_declared_role_rejected_before_task_access(subject):
    class ForbiddenTask:
        def __getattribute__(self, name):
            raise AssertionError("Role rejection must precede task-array access")
    with pytest.raises(ValueError):
        p.prepare_native_access(ForbiddenTask(), subject=subject, cohort_json=COHORT,
            expected_source_hash="unread", expected_model_hash="unread", access_derivation={})


def test_resealed_cohort_cannot_promote_select_patient_before_task_access():
    cohort = json.loads(COHORT)
    changed = json.dumps(cohort, sort_keys=True).encode()  # Exact pinned bytes, not caller resealing.
    with pytest.raises(ValueError, match="pinned"):
        p.prepare_native_access(object(), subject="sub-PAT26", cohort_json=changed,
            expected_source_hash="unread", expected_model_hash="unread", access_derivation={})


@pytest.mark.parametrize("key", ["expected_source_hash", "expected_model_hash"])
def test_upstream_subject_record_hash_mismatch_rejected_before_screen(initial, monkeypatch, key):
    task, derivation = initial
    kwargs = arguments(task, derivation); kwargs[key] += "changed"
    monkeypatch.setattr(p, "screen_axis_accesses", lambda *a, **k: pytest.fail("Wrong upstream hash reached screening"))
    with pytest.raises(ValueError, match="upstream"):
        p.prepare_native_access(task, **kwargs)


@pytest.mark.parametrize("index", range(6))
@pytest.mark.parametrize("field", ["distance_mm", "axis", "outward_sign"])
def test_all_six_saved_descriptors_are_exactly_reauthenticated(initial, monkeypatch, index, field):
    task, derivation = initial
    kwargs = arguments(task, derivation)
    row = kwargs["access_derivation"]["six_axis_exit_distances_mm"][index]
    row[field] = (float(np.nextafter(row[field], np.inf)) if field == "distance_mm"
                  else (row[field] + 1) % 3 if field == "axis" else -row[field])
    monkeypatch.setattr(p, "screen_axis_accesses", lambda *a, **k: pytest.fail("Altered derivation reached screening"))
    with pytest.raises(ValueError, match="derivation"):
        p.prepare_native_access(task, **kwargs)


def forbidden_during_helper(monkeypatch):
    def forbidden(*a, **k):
        pytest.fail("Preparation attempted a task/inventory, preview, cut or transition")
    for name in ("preview_stroke", "commit_preview", "execute_stroke"):
        monkeypatch.setattr(NativeResectionEngine, name, forbidden)
    monkeypatch.setattr(NativeSpatialTask, "__init__", forbidden)
    monkeypatch.setattr(NativeSpatialTask, "step", forbidden)


def physical_screen(receipt):
    return [{"axis": row["axis"], "outward_sign": row["outward_sign"],
             "distance_mm": row["distance_mm"], "eligible": row["eligible"],
             "poses": [{key: pose[key] for key in ("tool_id", "entry_mm", "tip_mm", "admissible", "blocked_cell_count")}
                       for pose in row["poses"]]} for row in receipt["screening"]["exits"]]


def test_real_analytic_screen_preserves_physical_inputs_without_preview_or_selected_task(initial, monkeypatch):
    task, derivation = initial
    kwargs = arguments(task, derivation)
    before = (task.case.source_hash, task.decision_model_hash, task._state_seal, task._engine.state_hash)
    forbidden_during_helper(monkeypatch)
    result = p.prepare_native_access(task, **kwargs)
    record = result.to_dict()
    assert record["status"] == "selected", record
    assert len(record["screening"]["exits"]) == 6 and record["screening"]["screen_count"] <= 468
    assert record["native_preview_calls"] == 0 and not record["selected_inventory_constructed"]
    assert not record["automatic_fallback"] and not record["complete_stroke_certified"]
    assert record["reward"] == asdict(task.reward_spec) and record["horizon"] == task.max_steps
    assert record["original_preparation"]["already_completed_by_caller"]
    assert record["original_preparation"]["elapsed_seconds"] is None
    assert before == (task.case.source_hash, task.decision_model_hash, task._state_seal, task._engine.state_hash)
    selected = result.selected_case
    for name in ("structural_intensity", "observed_support", "nominal_target", "reference_target", "affine_ras_mm", "_native_affine_ras_mm"):
        np.testing.assert_array_equal(getattr(selected, name), getattr(task.case, name))
    assert selected.tools == task.case.tools and selected.crop_shape == task.case.crop_shape
    assert selected.support_provenance == task.case.support_provenance
    assert selected.proposal_config == task.case.proposal_config
    with pytest.raises(TypeError):
        result.receipt["status"] = "forged"


@pytest.mark.parametrize("variant", ["private_reference", "actor_crop"])
def test_private_reference_and_actor_crop_cannot_choose_access(initial, monkeypatch, variant):
    task, derivation = initial
    other, other_derivation = make_task(private_variant=variant == "private_reference",
        crop=(3, 3, 3) if variant == "actor_crop" else (9, 9, 9))
    kwargs, other_kwargs = arguments(task, derivation), arguments(other, other_derivation)
    forbidden_during_helper(monkeypatch)
    first = p.prepare_native_access(task, **kwargs).to_dict()
    second = p.prepare_native_access(other, **other_kwargs).to_dict()
    assert first["status"] == second["status"] == "selected"
    assert first["selected_exit"] == second["selected_exit"]
    assert first["access_derivation"] == second["access_derivation"]
    assert physical_screen(first) == physical_screen(second)


def test_pre_cancelled_preparation_retains_failure_without_candidate_or_fallback(initial, monkeypatch):
    task, derivation = initial
    monkeypatch.setattr(p, "NativeResectionEngine", lambda *a, **k: pytest.fail("Cancelled before candidate construction"))
    result = p.prepare_native_access(task, **arguments(task, derivation), cancelled=lambda: True)
    assert result.selected_case is None
    assert result.receipt["status"] == "interrupted" and result.receipt["selected_exit"] is None
    assert not result.receipt["automatic_fallback"] and result.receipt["native_preview_calls"] == 0


@pytest.mark.parametrize("mode", ["failed", "none", "oversized"])
def test_screen_failure_or_absence_has_no_selected_source_or_fallback(initial, monkeypatch, mode):
    from types import SimpleNamespace
    task, derivation = initial
    calls = []
    def screen(candidates, **kw):
        calls.append(tuple(candidates))
        return SimpleNamespace(status="failed" if mode == "failed" else "selected", complete=mode != "failed",
            selected_exit=None if mode == "none" else (1, 1), screen_count=469 if mode == "oversized" else 1,
            to_dict=lambda: {"status": mode, "partial": "retained"})
    monkeypatch.setattr(p, "screen_axis_accesses", screen)
    forbidden_during_helper(monkeypatch)
    result = p.prepare_native_access(task, **arguments(task, derivation))
    assert len(calls) == 1 and len(calls[0]) == 6
    assert result.selected_case is None and result.receipt["selected_exit"] is None
    assert result.receipt["screening"]["partial"] == "retained"
    assert not result.receipt["automatic_fallback"]


@pytest.mark.parametrize("condition", ["advanced", "planning", "unprepared"])
def test_noninitial_or_missing_original_inventory_cannot_opt_in(condition, monkeypatch):
    task, derivation = make_task()
    kwargs = arguments(task, derivation)
    if condition == "advanced":
        task._steps = 1
        task._seal()  # Analytical state fixture only; no transition is executed.
    elif condition == "planning":
        task._planning = True
        task._seal()
    else:
        task._inventory = None
    monkeypatch.setattr(p, "screen_axis_accesses", lambda *a, **k: pytest.fail("Invalid task reached selection"))
    with pytest.raises(ValueError):
        p.prepare_native_access(task, **kwargs)


def test_original_and_reconciled_native_frames_remain_distinct_and_unchanged(monkeypatch):
    affine = np.eye(4); affine[0, 1] = 1e-9
    task, derivation = make_task(affine=affine, reconciliation="orthogonal_roundoff_1e-6mm")
    original, native = array_digest(task.case.affine_ras_mm), array_digest(task.case._native_affine_ras_mm)
    assert original != native
    forbidden_during_helper(monkeypatch)
    result = p.prepare_native_access(task, **arguments(task, derivation))
    assert result.receipt["status"] == "selected", result.to_dict()
    assert result.receipt["original_frame_hash"] == original
    assert result.receipt["native_frame_hash"] == native
    assert array_digest(result.selected_case.affine_ras_mm) == original
    assert array_digest(result.selected_case._native_affine_ras_mm) == native


def test_after_screen_horizon_mutation_refuses_selected_source(monkeypatch):
    from resectionlab.native_ingress import IngressScreeningResult
    task, derivation = make_task()
    kwargs = arguments(task, derivation)
    def damaged(*args, **kw):
        task.max_steps = 4
        return IngressScreeningResult("selected", (0, -1), (), 0)
    monkeypatch.setattr(p, "screen_axis_accesses", damaged)
    forbidden_during_helper(monkeypatch)
    result = p.prepare_native_access(task, **kwargs)
    assert result.selected_case is None and result.receipt["status"] == "failed"
    assert result.receipt["selected_exit"] is None and result.receipt["horizon"] == 3
    assert result.receipt["screening"]["selected_exit"] == (0, -1)
