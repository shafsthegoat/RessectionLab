"""Proposal traceability never substitutes for anatomical review or cortex data."""
from dataclasses import replace
from datetime import datetime, timezone

import numpy as np
import pytest

from resectionlab.core import SourceRef, array_digest
from resectionlab.geometry import AccessWindow
from resectionlab.imaging import create_synthetic_case, load_case, save_case
from resectionlab.native_resection import native_config_from_case
from resectionlab.native_simulation import make_native_patient_simulator
from resectionlab.planning import generate_candidate_routes, generate_hypothetical_windows
from resectionlab.simulation import make_patient_simulator
from resectionlab.structural_evidence import (BrainEnvelopeReview, StructuralEvidence,
                                            planning_brain_support, structural_frame_hash)


@pytest.fixture
def source_case():
    # Observed provenance deliberately tests the patient gate; these numeric
    # arrays remain synthetic test data, never anatomy performance evidence.
    fixture = create_synthetic_case((24, 24, 24))
    return replace(fixture, source_refs=(SourceRef("structural", "fixture://source", sha256="a" * 64),),
                   metadata={"structural_coverage": "full_head", "allow_nonzero_mri_access_support": False})


def proposal(case, mask=None):
    return StructuralEvidence("model-envelope", case.brain_mask if mask is None else mask,
        array_digest(case.mri), structural_frame_hash(case), "a" * 64, "b" * 64,
        "c" * 64, "test model native-grid envelope", metadata={"anatomy_accuracy": "unknown"})


def review(evidence, **changes):
    values = dict(evidence_hash=evidence.evidence_hash, reviewer_id="fixture-reviewer",
                  reviewed_at=datetime(2026, 10, 4, tzinfo=timezone.utc), decision="accepted",
                  rationale="Synthetic fixture attestation of envelope only; cortical anatomy unavailable")
    return BrainEnvelopeReview(**{**values, **changes})


def test_unselected_proposal_is_immutable_separate_and_does_not_change_planning_seed(source_case, tmp_path):
    source = replace(source_case, brain_mask=None)
    mutable = source_case.brain_mask.copy()
    evidence = proposal(source, mutable)
    before = evidence.evidence_hash
    mutable[:] = False
    assert evidence.mask.any() and evidence.evidence_hash == before
    with pytest.raises(ValueError):
        evidence.mask.setflags(write=True)
    updated = source.revised(structural_evidence={evidence.evidence_id: evidence})
    assert updated.brain_mask is None
    assert updated.semantic_hash != source.semantic_hash
    assert updated.planning_hash == source.planning_hash
    assert evidence.review_status == "review_required" and evidence.cortical_access_permitted is False
    restored = load_case(save_case(updated, tmp_path / "evidence.ressectionlab"))
    assert restored.semantic_hash == updated.semantic_hash and restored.brain_mask is None
    assert restored.structural_evidence[evidence.evidence_id].to_manifest() == evidence.to_manifest()


def test_exact_review_does_not_select_mask_or_grant_cortical_permission(source_case):
    evidence = proposal(source_case)
    accepted = replace(evidence, review=review(evidence))
    case = replace(source_case, brain_mask=None, structural_evidence={accepted.evidence_id: accepted})
    assert case.brain_mask is None
    assert planning_brain_support(case) == (None, {})
    selected = case.revised(brain_mask=accepted.mask)
    mask, provenance = planning_brain_support(selected)
    assert np.array_equal(mask, accepted.mask)
    assert provenance["review_status"] == "accepted"
    assert provenance["cortical_access_permitted"] is False
    assert accepted.to_manifest()["clinical_deficit_probability"] is None
    changed = evidence.mask.copy()
    changed[0, 0, 0] = ~changed[0, 0, 0]
    with pytest.raises(ValueError, match="does not match"):
        replace(accepted, mask=changed)
    with pytest.raises(ValueError, match="timezone-aware"):
        review(evidence, reviewed_at=datetime(2026, 10, 4))
    with pytest.raises(ValueError, match="cannot grant"):
        review(evidence, scope="safe_cortical_entry")


def test_stale_source_frame_mask_and_forged_manifest_are_rejected(source_case):
    evidence = proposal(source_case)
    case = source_case.revised(structural_evidence={evidence.evidence_id: evidence})
    with pytest.raises(ValueError, match="another source image"):
        case.revised(mri=case.mri + 1)
    with pytest.raises(ValueError, match="physical frame"):
        case.revised(frame="LPS+")
    for key, forged in (("cortical_access_permitted", True), ("review_status", "accepted"),
                        ("clinical_deficit_probability", .1), ("mask_hash", "sha256:" + "d" * 64)):
        with pytest.raises(ValueError):
            StructuralEvidence.from_manifest({**evidence.to_manifest(), key: forged}, mask=evidence.mask)


@pytest.mark.parametrize("mode", ["unknown", "unreviewed_extraction", "rejected_extraction", "flat_review_boolean"])
def test_every_case_factory_blocks_unreviewed_working_brain_mask(source_case, mode):
    case = source_case
    if mode in {"unreviewed_extraction", "rejected_extraction"}:
        evidence = proposal(case)
        if mode == "rejected_extraction":
            evidence = replace(evidence, review=review(evidence, decision="rejected"))
        case = case.revised(structural_evidence={evidence.evidence_id: evidence})
    elif mode == "flat_review_boolean":
        case = case.revised(metadata={**case.metadata, "brain_reviewed": True})
    calls = [lambda: generate_hypothetical_windows(case), lambda: generate_candidate_routes(case),
             lambda: make_native_patient_simulator(case), lambda: make_patient_simulator(case),
             lambda: native_config_from_case(case, access=AccessWindow((0, 0, 0), (0, 0, 1), 4))]
    for call in calls:
        with pytest.raises(ValueError, match="BRAIN_MASK_REVIEW_REQUIRED"):
            call()


def test_explicit_hypothetical_assumption_is_source_bound_and_remains_unverified(source_case):
    case = source_case
    assumption = {"scope": "hypothetical_tissue_support", "mask_hash": array_digest(case.brain_mask),
                  "source_image_hash": array_digest(case.mri), "source_frame_hash": structural_frame_hash(case),
                  "declared_by": "test-researcher", "declared_at": "2026-10-04T12:00:00Z",
                  "rationale": "A declared synthetic planning scenario, not reviewed anatomy"}
    declared = case.revised(metadata={**case.metadata, "brain_mask_support_assumption": assumption})
    _, record = planning_brain_support(declared)
    assert record["review_status"] == "supplied_unverified"
    assert record["evidence_type"] == "estimated" and record["cortical_access_permitted"] is False
    assert generate_hypothetical_windows(declared)
    changed = declared.brain_mask.copy()
    changed[0, 0, 0] = ~changed[0, 0, 0]
    with pytest.raises(ValueError, match="ASSUMPTION_STALE"):
        planning_brain_support(declared.revised(brain_mask=changed))


def test_explicit_support_cannot_reuse_unreviewed_extraction(source_case):
    evidence = proposal(source_case)
    case = source_case.revised(brain_mask=None, structural_evidence={evidence.evidence_id: evidence})
    with pytest.raises(ValueError, match="BRAIN_MASK_REVIEW_REQUIRED"):
        generate_hypothetical_windows(case, support_mask=evidence.mask,
            support_provenance={"source": evidence.evidence_hash, "method": "extraction", "evidence_type": "estimated"})


def test_simulated_source_cannot_be_relabelled_patient_observation():
    source = create_synthetic_case((24, 24, 24))
    case = source.revised(brain_mask=None)
    result = generate_candidate_routes(case, support_mask=source.brain_mask,
        support_provenance={"source": "synthetic fixture", "method": "known box", "evidence_type": "observed"})
    assert result.access_support["evidence_type"] == "simulated"
    assert result.access_support["source_domain"] == "synthetic_fixture"


def test_desktop_displays_separate_proposal_without_planning_permission(source_case, tmp_path):
    from resectionlab.desktop_bridge import BridgeRuntime
    evidence = proposal(source_case)
    case = source_case.revised(structural_evidence={evidence.evidence_id: evidence})
    path = save_case(case, tmp_path / "unreviewed.ressectionlab")
    events = []
    runtime = BridgeRuntime(tmp_path / "transfers", events.append)
    try:
        runtime.submit({"id": "load", "op": "loadCase", "args": {"path": str(path)}})
        assert runtime.wait_idle(5)
        payload = events[-1]["result"]
        item = payload["structuralEvidence"][0]
        assert item["reviewStatus"] == "review_required" and item["modelHash"] == evidence.model_sha256
        assert item["corticalAccessPermitted"] is False
        assert payload["brainSupport"]["usableForResearchSimulation"] is False
        runtime.submit({"id": "plan", "op": "generateRoutes", "args": {"caseHash": case.semantic_hash}})
        assert runtime.wait_idle(5)
        assert events[-1]["event"] == "error" and "BRAIN_MASK_REVIEW_REQUIRED" in events[-1]["error"]["message"]
    finally:
        runtime.close()
