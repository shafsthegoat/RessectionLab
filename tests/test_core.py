"""Core invariants: physical frames, provenance, timing and immutable versions."""

from dataclasses import replace
from datetime import datetime, timedelta, timezone
import json
import warnings

import numpy as np
import pytest

from resectionlab.core import (
    CaseData, ContextField, PatientContext, Plan, SourceRef, StalePlanError,
    semantic_digest,
)


def source() -> SourceRef:
    return SourceRef("synthetic", "generated://core-test", provenance="simulated")


def case(**updates) -> CaseData:
    values = {
        "case_id": "synthetic-001", "mri": np.zeros((4, 5, 6), dtype=np.float32),
        "compartments": {"enhancing": np.zeros((4, 5, 6), dtype=bool)},
        "affine": np.diag([1.0, 2.0, 3.0, 1.0]), "source_refs": (source(),),
        "unknowns": ("vascular_anatomy_unassessed", "motor_evidence_unavailable"),
    }
    values.update(updates)
    return CaseData(**values)


def plan(value: CaseData, **updates) -> Plan:
    values = {"plan_id": "route-1", "case_hash": value.semantic_hash,
              "route_points_mm": np.array([[0, 0, 0], [1, 2, 3]]), "tool_id": "generic-cannula"}
    values.update(updates)
    return Plan(**values)


def test_physical_affine_preserves_rotation_translation_and_anisotropy():
    affine = np.array([[0, -2, 0, 12], [1, 0, 0, -30], [0, 0, -3, 9], [0, 0, 0, 1.0]])
    value = case(affine=affine)
    points = np.array([[1, 2, 3], [0, 0, 0]])
    np.testing.assert_allclose(value.voxel_to_world(points), [[8, -29, 0], [12, -30, 9]])
    np.testing.assert_allclose(value.world_to_voxel(value.voxel_to_world(points)), points)
    assert value.voxel_volume_mm3 == pytest.approx(6)
    assert value.spacing_mm == (1, 2, 3)


def test_shear_voxel_volume_uses_determinant_not_spacing_product():
    affine = np.eye(4)
    affine[0, 1] = 1.0
    value = case(affine=affine)
    assert value.voxel_volume_mm3 == pytest.approx(1)
    assert np.prod(value.spacing_mm) > value.voxel_volume_mm3


@pytest.mark.parametrize("field,bad", [
    ("mri", np.zeros((4, 5))), ("mri", np.full((4, 5, 6), np.nan)),
    ("affine", np.zeros((4, 4))), ("affine", np.eye(3)),
    ("compartments", {"enhancing": np.zeros((2, 3, 4))}),
    ("compartments", {"enhancing": np.full((4, 5, 6), 2)}),
    ("brain_mask", np.full((4, 5, 6), np.nan)), ("frame", "unknown"),
    ("revision", 0), ("revision", True), ("source_refs", ()),
])
def test_rejects_ambiguous_or_invalid_input(field, bad):
    with pytest.raises((TypeError, ValueError)):
        case(**{field: bad})


def test_immutable_arrays_isolate_originals_and_cannot_reenable_writes():
    mri = np.zeros((4, 5, 6), dtype=np.float32)
    mask = np.zeros_like(mri, dtype=bool)
    affine = np.eye(4)
    value = case(mri=mri, compartments={"enhancing": mask}, affine=affine, brain_mask=mask)
    original_hash = value.semantic_hash
    mri[:] = 100
    mask[:] = True
    affine[0, 0] = 9
    assert value.mri.sum() == 0
    assert not value.compartments["enhancing"].any()
    assert value.affine[0, 0] == 1
    assert value.semantic_hash == original_hash
    for array in (value.mri, value.affine, value.compartments["enhancing"], value.source_compartments["enhancing"], value.brain_mask):
        with pytest.raises(ValueError):
            array.setflags(write=True)


def test_semantic_hash_changes_on_every_material_input():
    original = case()
    changed_image = original.mri.copy()
    changed_image[1, 2, 3] = 1
    changed_mask = original.compartments["enhancing"].copy()
    changed_mask[1, 2, 3] = True
    changed_affine = original.affine.copy()
    changed_affine[0, 3] = 1
    changes = [
        {"mri": changed_image}, {"compartments": {"enhancing": changed_mask}},
        {"affine": changed_affine}, {"metadata": {"access_window_version": 2}},
        {"frame": "LPS+"}, {"revision": 2}, {"unknowns": ("new_uncertainty",)},
        {"source_refs": (replace(source(), uri="generated://new-source"),)},
    ]
    for update in changes:
        assert replace(original, **update).semantic_hash != original.semantic_hash


def test_metadata_mutations_are_isolated_and_hash_stable():
    metadata = {"annotation": {"edits": [1, 2]}}
    value = case(metadata=metadata)
    old_hash = value.semantic_hash
    metadata["annotation"]["edits"].append(3)
    assert value.metadata["annotation"]["edits"] == (1, 2)
    with pytest.raises(TypeError):
        value.metadata["annotation"]["new"] = 1
    manifest = value.to_manifest()
    manifest["metadata"]["annotation"]["edits"].append(9)
    assert value.semantic_hash == old_hash
    assert semantic_digest({"a": 1, "b": [2]}) == semantic_digest({"b": [2], "a": 1})


def test_edit_retains_original_annotation_and_invalidates_plan():
    original = case()
    route = plan(original)
    corrected = original.compartments["enhancing"].copy()
    corrected[0, 0, 0] = True
    edited = original.revised(compartments={"enhancing": corrected})
    assert edited.revision == 2
    assert edited.compartments["enhancing"][0, 0, 0]
    assert not edited.source_compartments["enhancing"][0, 0, 0]
    route.assert_current(original)
    with pytest.raises(StalePlanError, match="obsolete"):
        route.assert_current(edited)
    with pytest.raises(ValueError, match="preserves"):
        original.revised(source_compartments={})


def test_context_availability_uses_timestamps_not_dataset_presence():
    cutoff = datetime(2026, 10, 4, 12, tzinfo=timezone.utc)
    records = (
        ContextField("idh_prior_biopsy", "mutant", source(), available_at=cutoff - timedelta(days=3)),
        ContextField("mgmt_resection", "methylated", source(), available_at=cutoff + timedelta(days=3)),
        ContextField("diagnosis_unknown_timing", "glioma", source()),
        ContextField("predicted_idh", {"label": "mutant"}, source(), available_at=cutoff, evidence_type="estimated"),
        ContextField("scenario", "aggressive", source(), available_at=cutoff, evidence_type="scenario_assumption"),
        ContextField("missing", None, source(), evidence_type="unknown"),
    )
    context = PatientContext(cutoff, records)
    assert context.planner_values() == {"idh_prior_biopsy": "mutant", "predicted_idh": {"label": "mutant"}}
    assert context.planning_view()["mgmt_resection"]["value"] is None
    assert context.planning_view()["diagnosis_unknown_timing"]["exclusion_reason"] == "availability_time_unknown"
    assert context.to_dict()["fields"][1]["value"] == "methylated"
    assert context.planning_view()["predicted_idh"]["evidence_type"] == "estimated"
    assert context.planner_values(include_scenarios=True)["scenario"] == "aggressive"
    assert len(context.withheld_fields()) == 4


def test_context_exact_cutoff_normalizes_timezones_and_roundtrips():
    cutoff = datetime(2026, 10, 4, 12, tzinfo=timezone.utc)
    eastern = timezone(timedelta(hours=-4))
    record = ContextField("motor_exam", "recorded", source(), available_at=cutoff.astimezone(eastern))
    context = PatientContext(cutoff, (record,))
    assert context.planner_values() == {"motor_exam": "recorded"}
    restored = PatientContext.from_dict(json.loads(json.dumps(context.to_dict())))
    assert restored.semantic_hash == context.semantic_hash
    assert restored.planner_values() == context.planner_values()


@pytest.mark.parametrize("name", ["available_at", "measurement_at"])
def test_rejects_naive_context_timestamps(name):
    with pytest.raises(ValueError, match="timezone-aware"):
        ContextField("idh", "mutant", source(), **{name: datetime(2026, 10, 4)})
    with pytest.raises(ValueError, match="timezone-aware"):
        PatientContext(datetime(2026, 10, 4))


def test_context_rejects_inverted_measurement_availability():
    moment = datetime(2026, 10, 4, tzinfo=timezone.utc)
    with pytest.raises(ValueError, match="precede"):
        ContextField("idh", "mutant", source(), available_at=moment, measurement_at=moment + timedelta(seconds=1))


def test_route_only_accessibility_is_not_removal_and_clinical_risk_is_null():
    value = case()
    route = plan(value, accessible_target_volume_mm3=20)
    encoded = route.to_dict()
    assert encoded["simulated_removed_target_volume_mm3"] is None
    assert encoded["clinical_deficit_probability"] is None
    assert encoded["clinical_risk_reason"] == "no_validated_clinical_outcome_model"
    assert Plan.from_dict(encoded).semantic_hash == route.semantic_hash
    with pytest.raises(ValueError, match="route-only"):
        plan(value, accessible_target_volume_mm3=20, simulated_removed_target_volume_mm3=20)
    with pytest.raises(ValueError, match="clinical deficit probability"):
        plan(value, clinical_deficit_probability=0)
    with pytest.raises(ValueError, match="removal_replay_hash"):
        plan(value, plan_type="simulated_resection", simulated_removed_target_volume_mm3=1)


def test_plan_geometry_and_metadata_are_immutable_and_versioned():
    value = case()
    route = plan(value, metadata={"objective": {"weights": [1, 2]}})
    with pytest.raises(ValueError):
        route.route_points_mm.setflags(write=True)
    revised = replace(route, world_model_version="world-2")
    assert revised.semantic_hash != route.semantic_hash
    assert replace(route, metadata={"objective": {"weights": [1, 3]}}).semantic_hash != route.semantic_hash


def test_stop_plan_has_empty_replay_geometry_and_survives_json():
    route = plan(case(), route_points_mm=np.empty((0, 3)), optimizer_mode="STOP")
    restored = Plan.from_dict(json.loads(json.dumps(route.to_dict())))
    assert restored.route_points_mm.shape == (0, 3)
    assert restored.semantic_hash == route.semantic_hash


def test_source_hash_validation_and_provenance_categories():
    normalized = SourceRef("public", "https://example.org/data", sha256="sha256:" + "A" * 64)
    assert normalized.sha256 == "a" * 64
    with pytest.raises(ValueError, match="64 hexadecimal"):
        SourceRef("invalid", "https://example.org/data", sha256="abc")
    with pytest.raises(ValueError, match="provenance"):
        SourceRef("invalid", "https://example.org/data", provenance="clinical_truth")


def test_array_view_metadata_changes_cannot_retain_cached_hash():
    value = case()
    previous = value.semantic_hash
    previous_planning = value.planning_hash
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        value.mri.shape = (20, 6)
    assert value.semantic_hash != previous
    assert value.planning_hash != previous_planning


def test_future_context_cannot_affect_planning_seed_identity():
    cutoff = datetime(2026, 10, 4, tzinfo=timezone.utc)
    future = ContextField("postoperative_idh", "mutant", source(), available_at=cutoff + timedelta(days=5))
    baseline = ContextField("motor_exam", "documented", source(), available_at=cutoff)
    original = case(context=PatientContext(cutoff, (baseline, future)))
    updated = original.revised(context=PatientContext(cutoff, (baseline, replace(future, value="wildtype")), version=2))
    assert updated.semantic_hash != original.semantic_hash
    assert updated.planning_hash == original.planning_hash
    available_update = original.revised(context=PatientContext(cutoff, (replace(baseline, value="different"), future)))
    assert available_update.planning_hash != original.planning_hash


def test_nested_source_and_units_cannot_retain_mutable_values():
    with pytest.raises(ValueError, match="license"):
        SourceRef("bad", "generated://bad", license=["mutable"])
    with pytest.raises(ValueError, match="unit"):
        ContextField("bad", 1, source(), unit={"mutable": True})
