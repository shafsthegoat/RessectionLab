"""Provisional-support provenance checks on artificial arrays, never patient use."""
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone

import numpy as np
import pytest

from resectionlab.core import CaseData, SourceRef, array_digest
from resectionlab.data_policy import DataPolicyError
from resectionlab.geometry import AccessWindow
from resectionlab.native_spatial_task import (
    OPENING_TOOLS, NativeSpatialCase, NativeSpatialTask,
    _validate_provisional_proposal_binding, native_spatial_task_from_case,
)
from resectionlab.structural_evidence import BrainEnvelopeReview, StructuralEvidence, structural_frame_hash


def proposal_case(*, varied=False):
    image = np.full((9, 9, 7), 100., np.float32)
    support = np.zeros(image.shape, bool)
    support[4, 4, 1:6], support[5, 5, 1] = True, True
    if varied:
        image[:] = 1e6  # Outside-support values must not affect the quantiles.
        image[support] = [-50., 10., 200., 400., 1200., 9000.]
    target = np.zeros(image.shape, bool)
    target[4, 4, 4:6] = True
    source = CaseData("independent-provisional-io", image, {"supplied_target": target}, np.eye(4),
        (SourceRef("structural", "generated://independent-analytic-io", sha256="a" * 64, provenance="observed"),),
        metadata={"structural_coverage": "full_head", "is_synthetic": True})
    proposal = StructuralEvidence("independent-model-proposal", support,
        array_digest(source.mri), structural_frame_hash(source), "a" * 64, "b" * 64, "c" * 64,
        "constructed model-output fixture; not a patient extraction")
    return source.revised(structural_evidence={proposal.evidence_id: proposal}), proposal


def access():
    return AccessWindow((3.9, 4., .5), (0., 0., 1.), 2.4, "explicit-unit-aperture")


def acknowledgment(case, proposal):
    return {"schema_version": 1, "scope": "hypothetical_tissue_support",
        "purpose": "native_spatial_provisional_research", "case_hash": case.semantic_hash,
        "planning_hash": case.planning_hash, "evidence_id": proposal.evidence_id,
        "evidence_hash": proposal.evidence_hash, "source_image_hash": proposal.source_image_hash,
        "source_frame_hash": proposal.source_frame_hash, "mask_hash": proposal.mask_hash,
        "model_sha256": proposal.model_sha256, "run_sha256": proposal.run_sha256,
        "declared_by": "independent unit fixture", "declared_at": "2026-10-04T12:00:00+00:00",
        "rationale": "constructed contract only; no patient or real model reviewed",
        "acknowledge_unreviewed": True, "cortical_access_permitted": False,
        "clinical_use_permitted": False}


def generated_task(case, proposal, **kwargs):
    """Exercise native geometry on artificial arrays, without model admission."""
    reference = case.compartments["supplied_target"]
    native = NativeSpatialCase(case.mri, proposal.mask, reference, np.eye(4), access(), OPENING_TOOLS,
        track="annotation_assisted", support_source_kind="supplied_annotation",
        support_derivation="constructed unit geometry; no anatomical model",
        nominal_target=reference, target_source_kind="supplied_annotation",
        target_derivation="constructed unit reference; no scan estimator", **kwargs)
    return NativeSpatialTask(native)


def test_reviewed_model_support_still_requires_admitted_weight_lineage():
    source, proposal = proposal_case()
    review = BrainEnvelopeReview(proposal.evidence_hash, "unit-fixture-only", datetime(2026, 10, 4, tzinfo=timezone.utc),
        "accepted", "test data contract; no patient or real model reviewed")
    approved = replace(proposal, review=review)
    case = source.revised(brain_mask=approved.mask, structural_evidence={approved.evidence_id: approved})
    with pytest.raises(DataPolicyError, match="WEIGHT_LINEAGE_UNVERIFIED"):
        native_spatial_task_from_case(case, access=access(), tools=OPENING_TOOLS)
    assert case.structural_evidence[proposal.evidence_id].review_status == "accepted"


def test_explicit_selection_cannot_admit_unverified_model_or_change_source():
    case, proposal = proposal_case()
    before = deepcopy(case.to_manifest())
    declaration = acknowledgment(case, proposal)
    assert _validate_provisional_proposal_binding(case, proposal, declaration) is None
    with pytest.raises(DataPolicyError, match="WEIGHT_LINEAGE_UNVERIFIED"):
        native_spatial_task_from_case(case, access=access(), tools=OPENING_TOOLS,
            research_support_acknowledgment=declaration)
    assert case.to_manifest() == before and case.brain_mask is None and proposal.review is None
    with pytest.raises(ValueError, match="ESSENTIAL_EVIDENCE_MISSING"):
        native_spatial_task_from_case(case, access=access(), tools=OPENING_TOOLS)


@pytest.mark.parametrize("field", ["image", "frame", "source_file", "proposal_mask"])
def test_source_replacement_after_acknowledgment_cannot_reuse_proposal(field):
    case, proposal = proposal_case()
    declaration = acknowledgment(case, proposal)
    if field == "image":
        object.__setattr__(case, "mri", case.mri + 1.)
    elif field == "frame":
        changed = case.affine.copy()
        changed[0, 3] += .1
        object.__setattr__(case, "affine", changed)
    elif field == "source_file":
        object.__setattr__(case, "source_refs", (replace(case.source_refs[0], sha256="d" * 64),))
    else:
        changed = proposal.mask.copy()
        changed[5, 5, 1] = False
        object.__setattr__(proposal, "mask", changed)
    with pytest.raises(ValueError, match="Structural evidence"):
        _validate_provisional_proposal_binding(case, proposal, declaration)


@pytest.mark.parametrize("decision", ["accepted", "rejected"])
def test_fresh_declaration_cannot_override_a_recorded_review(decision):
    case, proposal = proposal_case()
    reviewed = replace(proposal, review=BrainEnvelopeReview(proposal.evidence_hash,
        "unit-review-only", "2026-10-04T12:00:00+00:00", decision, "artificial review fixture"))
    case = case.revised(structural_evidence={reviewed.evidence_id: reviewed})
    with pytest.raises(ValueError, match="PROVISIONAL_SUPPORT_INELIGIBLE"):
        _validate_provisional_proposal_binding(case, reviewed, acknowledgment(case, reviewed))


def test_identical_mask_from_another_model_run_needs_its_own_selected_identity():
    case, proposal = proposal_case()
    other = replace(proposal, evidence_id="other-model-run", model_sha256="d" * 64, run_sha256="e" * 64)
    case = case.revised(structural_evidence={proposal.evidence_id: proposal, other.evidence_id: other})
    declaration = acknowledgment(case, proposal)
    declaration["evidence_id"] = other.evidence_id
    assert other.mask_hash == proposal.mask_hash
    with pytest.raises(ValueError, match="PROVISIONAL_SUPPORT_STALE"):
        _validate_provisional_proposal_binding(case, other, declaration)


def test_annotation_outside_selected_proposal_is_not_silently_added_to_support():
    case, proposal = proposal_case()
    changed = case.compartments["supplied_target"].copy()
    changed[0, 0, 0] = True
    case = case.revised(compartments={"supplied_target": changed})
    with pytest.raises(ValueError, match="target estimates conflict"):
        generated_task(case, proposal)
    assert not proposal.mask[0, 0, 0] and case.brain_mask is None


def test_generated_support_quantiles_drive_only_actor_pixels_and_freeze_exact_provenance():
    case, proposal = proposal_case(varied=True)
    raw = generated_task(case, proposal, crop_shape=(5, 5, 5))
    normalized = generated_task(case, proposal, crop_shape=(5, 5, 5), intensity_normalization="support_percentile_1_99")
    # Manual linear quantiles of six ordered samples: .05 and 4.95 indices.
    lower, upper = -47., 8610.
    record = normalized.metrics()["intensity_normalization"]
    assert record["lower"] == pytest.approx(lower) and record["upper"] == pytest.approx(upper)
    assert record["statistics_voxels"] == 6 and record["statistics_scope"] == "entire_permitted_support"
    assert record["percentile_method"] == "linear" and record["percentiles"] == [1., 99.]
    assert record["source_image_hash"] == array_digest(case.mri)
    assert record["support_hash"] == proposal.mask_hash
    assert record["raw_source_preserved"] and not record["reference_labels_used"] and not record["crop_used_for_statistics"]
    np.testing.assert_array_equal(normalized.case.structural_intensity, case.mri)
    assert normalized.case.structural_intensity.flags.writeable is False
    raw_observation, observation = raw.observation(), normalized.observation()
    expected = np.clip((raw_observation.image_channels[0].astype(float) - lower) / (upper - lower), 0., 1.).astype(np.float32)
    np.testing.assert_allclose(observation.image_channels[0], expected, atol=1e-7)
    np.testing.assert_array_equal(observation.image_channels[1:], raw_observation.image_channels[1:])
    np.testing.assert_array_equal(observation.action_geometry, raw_observation.action_geometry)
    np.testing.assert_array_equal(observation.action_mask, raw_observation.action_mask)
    np.testing.assert_array_equal(observation.affine_ras_mm, raw_observation.affine_ras_mm)
    assert normalized.case._candidate_voxels == raw.case._candidate_voxels
    assert normalized.case.source_hash != raw.case.source_hash
    record["lower"] = 123.
    assert normalized.metrics()["intensity_normalization"]["lower"] == pytest.approx(lower)


def test_normalization_ignores_hidden_reference_and_crop_but_detects_record_replacement():
    case, proposal = proposal_case(varied=True)
    source = generated_task(case, proposal, intensity_normalization="support_percentile_1_99").case
    changed = replace(source, reference_target=np.ones(case.mri.shape), crop_shape=(5, 5, 5))
    assert changed._normalization_record == source._normalization_record
    private_only = replace(source, reference_target=np.ones(case.mri.shape))
    assert private_only.source_hash == source.source_hash
    np.testing.assert_array_equal(NativeSpatialTask(private_only).observation().image_channels,
        NativeSpatialTask(source).observation().image_channels)
    with pytest.raises(TypeError):
        source._normalization_record["lower"] = 0.
    object.__setattr__(source, "_normalization_record", {**source._normalization_record, "lower": 0.})
    with pytest.raises(RuntimeError, match="interpretation was replaced"):
        source.assert_intact()


def test_constant_support_requires_explicit_normalization_refusal_without_changing_raw_default():
    case, proposal = proposal_case()
    raw = generated_task(case, proposal)
    assert raw.metrics()["intensity_normalization"] == {"method": "raw"}
    np.testing.assert_array_equal(raw.observation().image_channels[0], case.mri)
    with pytest.raises(ValueError, match="DEGENERATE_SCAN_INTENSITY_RANGE"):
        generated_task(case, proposal, intensity_normalization="support_percentile_1_99")
