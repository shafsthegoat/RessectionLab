"""Analytic code checks only: no human or synthetic model training/evaluation."""
from dataclasses import replace

import numpy as np
import pytest

from resectionlab.core import CaseData, SourceRef, array_digest
from resectionlab.data_policy import DataPolicyError
from resectionlab.geometry import AccessWindow
from resectionlab import native_spatial_task as native_task_module
from resectionlab.native_resection import NativeResectionEngine
from resectionlab.native_spatial_task import (
    DEFAULT_NATIVE_SPATIAL_REWARD, NativeSpatialCase, NativeSpatialTask,
    OPENING_TOOLS, make_native_opening_task, native_spatial_task_from_case,
    reconcile_native_grid_roundoff, _validate_provisional_proposal_binding,
)
from resectionlab.simulation import InvalidActionError
from resectionlab.structural_evidence import structural_frame_hash


def action(task, tool, voxel):
    return next(row["action_id"] for row in task.candidate_inventory()["ledger"]
                if row["feasible"] and row["tool_id"] == tool and row["voxel"] == list(voxel))


def exact(task):
    if task.terminated:
        return task.metrics()["total_reward"], (), 1
    best, route, leaves = -np.inf, (), 0
    for identifier in task.observation().action_ids:
        child = task.clone()
        child.step(identifier)
        score, tail, count = exact(child)
        leaves += count
        if score > best:
            best, route = score, (identifier, *tail)
    return best, route, leaves


def same_observations(first, second):
    assert first.action_ids == second.action_ids and first.action_tool_ids == second.action_tool_ids
    assert first.fingerprint == second.fingerprint and first.source_id == second.source_id
    for name in ("image_channels", "coverage", "channel_available", "affine_ras_mm", "spacing_mm",
                 "action_geometry", "action_mask", "state_features"):
        np.testing.assert_array_equal(getattr(first, name), getattr(second, name))


def annotated_source(*, support=True, assumption=True, frame="RAS+"):
    """Fabricated IO-contract fixture with MRI-like units; never a patient result."""
    native = make_native_opening_task().case
    image = np.full(native.structural_intensity.shape, 1500., np.float32)
    case = CaseData("analytic-native-io", image, {"provided_target": native.reference_target > 0},
        np.eye(4), (SourceRef("unit-io-source", "generated://native-io-contract", provenance="observed"),),
        frame=frame, brain_mask=native.observed_support if support else None,
        metadata={"structural_coverage": "full_head"})
    if support and assumption:
        declaration = {"scope": "hypothetical_tissue_support", "mask_hash": array_digest(case.brain_mask),
            "source_image_hash": array_digest(case.mri), "source_frame_hash": structural_frame_hash(case),
            "declared_by": "analytic test fixture", "declared_at": "2026-10-04T12:00:00+00:00",
            "rationale": "explicit artificial geometry contract, no medical approval", "cortical_access_permitted": False}
        case = case.revised(metadata={**case.metadata, "brain_mask_support_assumption": declaration})
    return case


def test_complete_unit_inventory_requires_costly_opening_then_distinct_tool():
    task = make_native_opening_task()
    inventory = task.candidate_inventory()
    assert inventory["declared_slots"] == inventory["evaluated_slots"] == 12
    assert inventory["omitted_count"] == 0 and inventory["complete"]
    assert inventory["accepted_count"] + inventory["rejected_count"] == 12
    best, sequence, leaves = exact(task.planning_clone())
    assert best == pytest.approx(1.1) and len(sequence) == 2 and leaves > 10
    assert max(task.clone().step(option).reward for option in task.observation().action_ids) == 0.
    rewards = [task.step(identifier).reward for identifier in sequence]
    assert rewards[0] < 0 and rewards[1] > 0
    metrics = task.metrics()
    assert metrics["target_removed_mm3"] == 2. and metrics["normal_removed_mm3"] == 4.
    assert metrics["simulated_removed_volume_mm3"] == 6.
    assert task.independent_geometry_check().feasible
    for tool in OPENING_TOOLS:
        assert exact(make_native_opening_task(tools=(tool,)))[0] == 0.


def test_exact_offset_entry_reaches_policy_and_independent_history():
    task = make_native_opening_task()
    identifier = action(task, OPENING_TOOLS[0].tool_id, (4, 4, 1))
    observed = task.observation()
    index = observed.action_ids.index(identifier)
    np.testing.assert_array_equal(observed.action_geometry[index, 1:4], [4, 4, .5])
    assert observed.state_features[3] == 3.9
    result = task.step(identifier)
    np.testing.assert_array_equal(result.info["entry_mm"], [4, 4, .5])
    assert task.independent_geometry_check().feasible


def test_partial_contacts_remain_occupied_and_have_no_history_dependent_cost():
    task = make_native_opening_task()
    task.step(action(task, OPENING_TOOLS[0].tool_id, (4, 4, 1)))
    retained = task._engine.contact_mask & ~task._engine.removed_mask
    assert retained.any() and task._engine.remaining_mask[retained].all()
    metrics = task.metrics()
    assert metrics["partial_contact_weight"] == 0.
    assert metrics["currently_retained_contacted_tissue_upper_bound_mm3"] == float(retained.sum())
    assert metrics["simulated_removed_volume_mm3"] == 3.
    assert metrics["motor_surrogate"] is metrics["language_surrogate"] is None
    assert metrics["clinical_deficit_probability"] is None


def test_hidden_references_never_change_tensors_inventory_or_teacher_even_outside_support():
    first = make_native_opening_task()
    reference = np.ones(first.case.structural_intensity.shape, np.float32)
    second = NativeSpatialTask(replace(first.case, reference_target=reference), max_steps=2)
    assert first.decision_model_hash == second.decision_model_hash
    same_observations(first.observation(), second.observation())
    assert first.candidate_inventory() == second.candidate_inventory()
    identifier = action(first, OPENING_TOOLS[0].tool_id, (4, 4, 1))
    assert first.step(identifier).reward != second.step(identifier).reward
    same_observations(first.observation(), second.observation())
    a, b = first.planning_clone(), second.planning_clone()
    assert a.metrics() == b.metrics()
    same_observations(a.observation(), b.observation())


def test_teacher_clone_does_not_replay_geometry_or_copy_full_source(monkeypatch):
    task = make_native_opening_task()
    task.step(action(task, OPENING_TOOLS[0].tool_id, (4, 4, 1)))
    task.observation()  # Prepare this state's shared immutable certificates.
    def forbidden(*args, **kwargs):
        raise AssertionError("A branch must reuse its prepared source/geometry")
    monkeypatch.setattr(NativeResectionEngine, "__init__", forbidden)
    monkeypatch.setattr(NativeResectionEngine, "preview_stroke", forbidden)
    teacher = task.planning_clone()
    same_observations(task.observation(), teacher.observation())
    assert teacher.case.structural_intensity is task.case.structural_intensity
    assert teacher.case._native_config is task.case._native_config
    assert teacher.case.reference_target is task.case.nominal_target
    teacher.step(action(teacher, OPENING_TOOLS[1].tool_id, (4, 4, 5)))
    assert teacher.terminated and not task.terminated
    assert task.metrics()["target_removed_mm3"] == 0.


def test_real_io_track_uses_supplied_targets_without_any_intensity_threshold():
    case = annotated_source()
    task = native_spatial_task_from_case(case, access=make_native_opening_task().case.access, tools=OPENING_TOOLS)
    observation = task.observation()
    assert observation.track == "annotation_assisted"
    assert np.all(observation.image_channels[0] == 1500.)
    np.testing.assert_array_equal(observation.image_channels[2], case.compartments["provided_target"])
    assert observation.channel_provenance["nominal_target"]["source_kind"] == "supplied_annotation"
    np.testing.assert_array_equal(task.planning_clone().case.reference_target, case.compartments["provided_target"])


@pytest.mark.parametrize("support,assumption", [(False, False), (True, False)])
def test_full_head_or_unreviewed_support_never_becomes_working_anatomy(support, assumption):
    case = annotated_source(support=support, assumption=assumption)
    with pytest.raises(ValueError, match="ESSENTIAL_EVIDENCE_MISSING|BRAIN_MASK_REVIEW_REQUIRED"):
        native_spatial_task_from_case(case, access=make_native_opening_task().case.access, tools=OPENING_TOOLS)


def test_inference_only_requires_explicit_nominal_target_for_teacher():
    case = annotated_source()
    # A research assumption alone cannot relabel supplied anatomy as an image
    # model. This unit source explicitly declares its artificial model output.
    with pytest.raises(ValueError, match="reference annotations"):
        native_spatial_task_from_case(case, access=make_native_opening_task().case.access,
            tools=OPENING_TOOLS, track="inference_only")
    source = NativeSpatialCase(case.mri, case.brain_mask, case.compartments["provided_target"], case.affine,
        make_native_opening_task().case.access, OPENING_TOOLS, track="inference_only",
        support_source_kind="derived_from_scan", support_derivation="explicit artificial unit-model output")
    task = NativeSpatialTask(source)
    assert not task.observation().channel_available[2]
    with pytest.raises(ValueError, match="no MRI threshold fallback"):
        task.planning_clone()
    estimate = .6 * case.compartments["provided_target"]
    actual = NativeSpatialTask(replace(source, nominal_target=estimate, target_source_kind="derived_from_scan",
        target_derivation="explicit unit estimate/checkpoint fixture"))
    np.testing.assert_allclose(actual.planning_clone().case.reference_target, estimate)


def test_fixed_access_crop_and_lattice_ignore_private_target_extent():
    image = np.full((40, 42, 44), 100., np.float32)
    support = np.ones(image.shape, bool)
    reference = np.zeros(image.shape, np.float32)
    source = NativeSpatialCase(image, support, reference, np.eye(4),
        AccessWindow((20, 21, -.5), (0, 0, 1), 3), OPENING_TOOLS, crop_shape=(16, 16, 16))
    moved = replace(source, reference_target=np.ones(image.shape, np.float32))
    assert source.source_hash == moved.source_hash
    assert source._crop_origin == moved._crop_origin and source._candidate_voxels == moved._candidate_voxels
    assert len(source._candidate_voxels) * len(source.tools) <= 96
    inputs = source.spatial_inputs(np.zeros(image.shape, bool))
    assert inputs.channels["structural_intensity"].data.shape == (16, 16, 16)
    np.testing.assert_array_equal(inputs.affine_ras_mm[:3, 3], source._crop_origin)
    assert source._native_config.tissue_mask.shape == image.shape


@pytest.mark.parametrize("change", ["source", "config", "cavity", "reward"])
def test_source_and_committed_state_tampering_fail_closed(change):
    task = make_native_opening_task()
    if change == "source":
        object.__setattr__(task.case, "structural_intensity", np.zeros(task.case.structural_intensity.shape))
    elif change == "config":
        object.__setattr__(task._config, "target_labels", task._config.target_labels.copy())
    elif change == "cavity":
        task._engine.removed_mask[0, 0, 0] = True
    else:
        task.reward_spec = replace(task.reward_spec, target_per_mm3=9.)
    with pytest.raises((ValueError, RuntimeError)):
        task.observation()


def test_invalid_or_stale_action_and_nonzero_world_seed_are_refused():
    task = make_native_opening_task()
    old = action(task, OPENING_TOOLS[0].tool_id, (4, 4, 1))
    task.step(old)
    with pytest.raises(InvalidActionError):
        task.step(old)
    with pytest.raises(ValueError, match="deterministic"):
        task.reset(seed=1)
    with pytest.raises(ValueError, match="function remains unassessed"):
        NativeSpatialTask(task.case, reward=replace(DEFAULT_NATIVE_SPATIAL_REWARD, motor_per_mm3=1.))


def test_oblique_reflected_grid_preserves_native_source_cells_and_audit():
    task = make_native_opening_task()
    angle = .4
    transform = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, -1.]])
    affine = np.eye(4)
    affine[:3, :3], affine[:3, 3] = transform, [14., -8., 12.]
    access = AccessWindow(transform @ task.case.access.center_mm + affine[:3, 3],
        transform @ task.case.access.normal_inward, task.case.access.radius_mm)
    moved = NativeSpatialTask(replace(task.case, affine_ras_mm=affine, access=access), max_steps=2)
    for tool, voxel in ((OPENING_TOOLS[0].tool_id, (4, 4, 1)), (OPENING_TOOLS[1].tool_id, (4, 4, 5))):
        task.step(action(task, tool, voxel))
        moved.step(action(moved, tool, voxel))
    np.testing.assert_array_equal(task._engine.removed_mask, moved._engine.removed_mask)
    assert moved.independent_geometry_check().feasible
    assert moved.metrics()["total_reward"] == pytest.approx(task.metrics()["total_reward"])


def test_cancellation_distinguishes_before_commit_and_after_committed_transition(monkeypatch):
    task = make_native_opening_task()
    identifier = action(task, OPENING_TOOLS[0].tool_id, (4, 4, 1))
    task._cancelled = lambda: True
    with pytest.raises(InterruptedError):
        task.step(identifier)
    assert task.metrics()["steps"] == 0 and not task._engine.removed_mask.any()
    flag = [False]
    task._cancelled = lambda: flag[0]
    commit = task._engine.commit_preview
    def commit_then_cancel(result):
        value = commit(result)
        flag[0] = True
        return value
    monkeypatch.setattr(task._engine, "commit_preview", commit_then_cancel)
    from resectionlab.native_axis_simulation import CommittedTransitionInterrupted
    with pytest.raises(CommittedTransitionInterrupted) as error:
        task.step(identifier)
    assert error.value.committed and error.value.info["action_id"] == identifier
    assert task.metrics()["steps"] == 1 and task.metrics()["simulated_removed_volume_mm3"] == 3.
    assert task.independent_geometry_check().feasible


def provisional_source():
    from resectionlab.structural_evidence import StructuralEvidence
    case = annotated_source(support=False)
    item = StructuralEvidence("unit_model_proposal", make_native_opening_task().case.observed_support,
        array_digest(case.mri), structural_frame_hash(case), None,
        "sha256:" + "1" * 64, "sha256:" + "2" * 64, "explicit analytic model output")
    case = case.revised(structural_evidence={item.evidence_id: item})
    acknowledgment = {"schema_version": 1, "scope": "hypothetical_tissue_support",
        "purpose": "native_spatial_provisional_research", "case_hash": case.semantic_hash,
        "planning_hash": case.planning_hash, "evidence_id": item.evidence_id,
        "evidence_hash": item.evidence_hash, "source_image_hash": item.source_image_hash,
        "source_frame_hash": item.source_frame_hash, "mask_hash": item.mask_hash,
        "model_sha256": item.model_sha256, "run_sha256": item.run_sha256,
        "declared_by": "unit-test research declaration", "declared_at": "2026-10-04T12:00:00+00:00",
        "rationale": "analytic provisional support boundary check; no expert review",
        "acknowledge_unreviewed": True, "cortical_access_permitted": False, "clinical_use_permitted": False}
    return case, acknowledgment


def test_provisional_support_rejects_unverified_model_before_task_creation(monkeypatch):
    case, acknowledgment = provisional_source()
    original_hash, original_planning_hash = case.semantic_hash, case.planning_hash
    access = make_native_opening_task().case.access
    item = case.structural_evidence[acknowledgment["evidence_id"]]
    assert _validate_provisional_proposal_binding(case, item, acknowledgment) is None
    def forbidden_task(*args, **kwargs):
        pytest.fail("No native task or candidate inventory may be created from an unverified model")
    monkeypatch.setattr(native_task_module, "NativeSpatialTask", forbidden_task)
    with pytest.raises(DataPolicyError, match="WEIGHT_LINEAGE_UNVERIFIED"):
        native_spatial_task_from_case(case, access=access, tools=OPENING_TOOLS,
            research_support_acknowledgment=acknowledgment)
    assert case.brain_mask is None and item.review is None and item.review_status == "review_required"
    assert case.semantic_hash == original_hash and case.planning_hash == original_planning_hash
    with pytest.raises(ValueError, match="ESSENTIAL_EVIDENCE_MISSING"):
        native_spatial_task_from_case(case, access=access, tools=OPENING_TOOLS)


@pytest.mark.parametrize("field", ["case_hash", "planning_hash", "evidence_hash", "source_image_hash",
    "source_frame_hash", "mask_hash", "model_sha256", "run_sha256"])
def test_provisional_support_rejects_every_stale_identity(field):
    case, acknowledgment = provisional_source()
    acknowledgment[field] = "sha256:" + "f" * 64
    item = case.structural_evidence[acknowledgment["evidence_id"]]
    with pytest.raises(ValueError, match="PROVISIONAL_SUPPORT_STALE"):
        _validate_provisional_proposal_binding(case, item, acknowledgment)


@pytest.mark.parametrize("change", ["missing_field", "expert_claim", "clinical", "cortical", "unacknowledged", "naive_time", "track"])
def test_provisional_support_refuses_missing_or_promotional_declarations(change):
    case, acknowledgment = provisional_source()
    if change == "missing_field":
        acknowledgment.pop("purpose")
    elif change == "expert_claim":
        acknowledgment["brain_reviewed"] = True
    elif change in {"clinical", "cortical"}:
        acknowledgment[change + "_use_permitted" if change == "clinical" else "cortical_access_permitted"] = True
    elif change == "unacknowledged":
        acknowledgment["acknowledge_unreviewed"] = False
    elif change == "naive_time":
        acknowledgment["declared_at"] = "2026-10-04T12:00:00"
    if change == "naive_time":
        item = case.structural_evidence[acknowledgment["evidence_id"]]
        with pytest.raises(ValueError, match="timezone-aware timestamp"):
            _validate_provisional_proposal_binding(case, item, acknowledgment)
        return
    expected = {
        "missing_field": "PROVISIONAL_SUPPORT_ACKNOWLEDGMENT_FIELDS",
        "expert_claim": "PROVISIONAL_SUPPORT_ACKNOWLEDGMENT_FIELDS",
        "clinical": "PROVISIONAL_SUPPORT_RESEARCH_ONLY",
        "cortical": "PROVISIONAL_SUPPORT_RESEARCH_ONLY",
        "unacknowledged": "PROVISIONAL_SUPPORT_RESEARCH_ONLY",
        "track": "PROVISIONAL_SUPPORT_TRACK",
    }
    with pytest.raises(ValueError, match=expected[change]):
        native_spatial_task_from_case(case, access=make_native_opening_task().case.access, tools=OPENING_TOOLS,
            track="inference_only" if change == "track" else "annotation_assisted",
            research_support_acknowledgment=acknowledgment)


def test_rejected_model_proposal_cannot_be_bypassed_by_research_acknowledgment():
    from resectionlab.structural_evidence import BrainEnvelopeReview
    case, acknowledgment = provisional_source()
    item = case.structural_evidence[acknowledgment["evidence_id"]]
    rejected = replace(item, review=BrainEnvelopeReview(item.evidence_hash, "unit reviewer",
        "2026-10-04T12:00:00+00:00", "rejected", "unit rejection"))
    case = case.revised(structural_evidence={rejected.evidence_id: rejected})
    acknowledgment["case_hash"], acknowledgment["planning_hash"] = case.semantic_hash, case.planning_hash
    with pytest.raises(ValueError, match="PROVISIONAL_SUPPORT_INELIGIBLE"):
        _validate_provisional_proposal_binding(case, rejected, acknowledgment)


def test_opt_in_normalization_uses_whole_permitted_support_and_preserves_raw_scan():
    source = make_native_opening_task().case
    image = np.full(source.structural_intensity.shape, 1e6, np.float32)
    image[source.observed_support] = np.array([100., 200., 300., 400., 500., 900.], np.float32)
    normalized = replace(source, structural_intensity=image, intensity_normalization="support_percentile_1_99")
    np.testing.assert_array_equal(normalized.structural_intensity, image)
    record = NativeSpatialTask(normalized).metrics()["intensity_normalization"]
    assert record["lower"] == pytest.approx(105.) and record["upper"] == pytest.approx(880.)
    assert record["statistics_voxels"] == 6 and record["statistics_scope"] == "entire_permitted_support"
    assert record["source_image_hash"] == array_digest(image) and record["support_hash"] == array_digest(source.observed_support)
    assert record["reference_labels_used"] is False and record["crop_used_for_statistics"] is False
    channel = normalized.spatial_inputs(np.zeros(image.shape, bool)).channels["structural_intensity"]
    np.testing.assert_allclose(channel.data, np.clip((image.astype(float) - 105.) / 775., 0., 1.), atol=1e-7)
    assert "support_percentile_1_99" in channel.derivation
    assert normalized._native_config.voxel_volume_mm3 == source._native_config.voxel_volume_mm3
    assert normalized._candidate_voxels == source._candidate_voxels


def test_normalization_bounds_ignore_reference_labels_and_actor_crop():
    source = make_native_opening_task().case
    image = np.arange(source.structural_intensity.size, dtype=np.float32).reshape(source.structural_intensity.shape)
    normalized = replace(source, structural_intensity=image, intensity_normalization="support_percentile_1_99")
    changed = replace(normalized, reference_target=np.ones(image.shape), crop_shape=(5, 5, 5))
    assert normalized._normalization_record == changed._normalization_record
    private_changed = replace(normalized, reference_target=np.ones(image.shape))
    assert normalized.source_hash == private_changed.source_hash
    same_observations(NativeSpatialTask(normalized).observation(), NativeSpatialTask(private_changed).observation())


def test_normalization_is_explicit_and_degenerate_bounds_are_refused():
    source = make_native_opening_task().case
    constant = np.full(source.structural_intensity.shape, 1500., np.float32)
    raw = replace(source, structural_intensity=constant)
    assert raw.intensity_normalization == "raw" and raw._normalization_record == {"method": "raw"}
    assert np.all(raw.spatial_inputs(np.zeros(constant.shape, bool)).channels["structural_intensity"].data == 1500.)
    with pytest.raises(ValueError, match="DEGENERATE_SCAN_INTENSITY_RANGE"):
        replace(raw, intensity_normalization="support_percentile_1_99")
    with pytest.raises(ValueError, match="Unknown native spatial"):
        replace(raw, intensity_normalization="target_percentiles")


def test_normalization_record_is_frozen_and_part_of_source_integrity():
    source = replace(make_native_opening_task().case, intensity_normalization="support_percentile_1_99")
    with pytest.raises(TypeError):
        source._normalization_record["lower"] = -100.
    object.__setattr__(source, "_normalization_record", {**source._normalization_record, "lower": -100.})
    with pytest.raises(RuntimeError, match="interpretation was replaced"):
        source.assert_intact()


def test_grid_roundoff_requires_opt_in_preserves_original_and_uses_derived_grid_everywhere():
    source = make_native_opening_task().case
    original = np.eye(4)
    original[0, 1] = 1e-9
    with pytest.raises(ValueError, match="orthogonal affine"):
        replace(source, affine_ras_mm=original)
    reconciled = replace(source, affine_ras_mm=original,
        native_grid_reconciliation="orthogonal_roundoff_1e-6mm")
    np.testing.assert_array_equal(reconciled.affine_ras_mm, original)
    np.testing.assert_array_equal(reconciled.structural_intensity, source.structural_intensity)
    np.testing.assert_array_equal(reconciled._native_config.affine, reconciled._native_affine_ras_mm)
    assert reconciled._grid_record["maximum_corner_displacement_mm"] < 1e-6
    assert reconciled._grid_record["resampled"] is False
    assert reconciled._crop_origin == source._crop_origin and reconciled._candidate_voxels == source._candidate_voxels
    inputs = reconciled.spatial_inputs(np.zeros(source.structural_intensity.shape, bool))
    np.testing.assert_array_equal(inputs.affine_ras_mm, reconciled._native_affine_ras_mm)
    task = NativeSpatialTask(reconciled, max_steps=2)
    for tool, voxel in ((OPENING_TOOLS[0].tool_id, (4, 4, 1)), (OPENING_TOOLS[1].tool_id, (4, 4, 5))):
        row = next(row for row in task.candidate_inventory()["ledger"]
                   if row["tool_id"] == tool and row["voxel"] == list(voxel))
        observation = task.observation()
        index = observation.action_ids.index(row["action_id"])
        np.testing.assert_allclose(observation.action_geometry[index, 4:7],
            reconciled._native_affine_ras_mm[:3, :3] @ voxel + reconciled._native_affine_ras_mm[:3, 3], rtol=0, atol=1e-14)
        task.step(row["action_id"])
    assert task.independent_geometry_check().feasible
    assert task.metrics()["native_grid_reconciliation"]["original_affine_hash"] == array_digest(original)


def test_roundoff_preserves_anisotropic_reflected_frame_origin_and_full_corner_bound():
    affine = np.array([[-.9, 1e-10, 0, 170], [0, 1.7, 0, -240], [0, 0, 2.5, 100], [0, 0, 0, 1.]])
    corrected, record = reconcile_native_grid_roundoff(affine, (160, 256, 256))
    np.testing.assert_array_equal(corrected[:3, 3], affine[:3, 3])
    np.testing.assert_allclose(np.linalg.norm(corrected[:3, :3], axis=0),
        np.linalg.norm(affine[:3, :3], axis=0), rtol=1e-14, atol=0)
    assert np.linalg.det(corrected[:3, :3]) < 0 and record["handedness_preserved"]
    corners = np.array([[x, y, z] for x in [-.5,159.5] for y in [-.5,255.5] for z in [-.5,255.5]])
    measured = np.linalg.norm(corners @ (corrected[:3, :3] - affine[:3, :3]).T, axis=1).max()
    assert measured == record["maximum_corner_displacement_mm"] and measured < 1e-6


@pytest.mark.parametrize("affine,shape,reason", [
    (np.array([[1., 1e-4, 0, 0], [0, 1., 0, 0], [0, 0, 1., 0], [0, 0, 0, 1.]]), (3,3,3), "MEANINGFUL_SHEAR"),
    (np.array([[1., 1e-9, 0, 0], [0, 1., 0, 0], [0, 0, 1., 0], [0, 0, 0, 1.]]), (2000,2000,3), "DISPLACEMENT_EXCEEDED"),
    (np.diag([1., 0., 1., 1.]), (3,3,3), "INVALID_SCALE"),
])
def test_roundoff_refuses_material_shear_large_extent_and_degenerate_scales(affine, shape, reason):
    with pytest.raises(ValueError, match=reason):
        reconcile_native_grid_roundoff(affine, shape)


def test_roundoff_rejects_svd_frame_changes_and_seals_record(monkeypatch):
    def reflected_svd(directions):
        return np.diag([-1., 1., 1.]), np.ones(3), np.eye(3)
    with monkeypatch.context() as patch:
        patch.setattr(np.linalg, "svd", reflected_svd)
        with pytest.raises(ValueError, match="FRAME_CHANGE"):
            reconcile_native_grid_roundoff(np.eye(4), (3,3,3))
    source = replace(make_native_opening_task().case, native_grid_reconciliation="orthogonal_roundoff_1e-6mm")
    with pytest.raises(TypeError):
        source._grid_record["resampled"] = True
    object.__setattr__(source, "_grid_record", {**source._grid_record, "maximum_corner_displacement_mm": 1.})
    with pytest.raises(RuntimeError, match="interpretation was replaced"):
        source.assert_intact()
