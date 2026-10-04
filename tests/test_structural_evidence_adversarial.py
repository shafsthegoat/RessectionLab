"""Independent attacks on structural proposal selection and source assumptions.

All numeric anatomy here is synthetic. Observed provenance exercises the patient
boundary and supplies no evidence of segmentation or clinical accuracy.
"""
from dataclasses import replace
import warnings

import numpy as np
import pytest

from resectionlab.core import SourceRef, array_digest
from resectionlab.geometry import AccessWindow
from resectionlab.imaging import create_synthetic_case
from resectionlab.native_resection import native_config_from_case
from resectionlab.native_simulation import make_native_patient_simulator
from resectionlab.planning import generate_hypothetical_windows
from resectionlab.simulation import make_patient_simulator
from resectionlab.structural_evidence import (
    BrainEnvelopeReview, StructuralEvidence, planning_brain_support,
    structural_frame_hash, validate_explicit_support,
)
from resectionlab.worlds import WorldGenerator, WorldGeneratorConfig, generate_partitions


@pytest.fixture
def observed_fixture():
    return replace(create_synthetic_case((24, 24, 24)),
                   source_refs=(SourceRef("structural", "fixture://observed", sha256="a" * 64),),
                   metadata={"structural_coverage": "full_head",
                             "allow_nonzero_mri_access_support": False})


def evidence(case, *, evidence_id="independent-envelope", mask=None):
    return StructuralEvidence(evidence_id, case.brain_mask if mask is None else mask,
                              array_digest(case.mri), structural_frame_hash(case),
                              "a" * 64, "b" * 64, "c" * 64, "test-only model proposal")


def accepted(item):
    return replace(item, review=BrainEnvelopeReview(
        item.evidence_hash, "test-only-reviewer", "2026-10-04T12:00:00Z",
        "accepted", "Synthetic research fixture; cortical access remains unavailable"))


def assumption(case, mask):
    return {"scope": "hypothetical_tissue_support", "mask_hash": array_digest(mask),
            "source_image_hash": array_digest(case.mri),
            "source_frame_hash": structural_frame_hash(case),
            "declared_by": "test-only-researcher", "declared_at": "2026-10-04T12:00:00Z",
            "rationale": "Explicit test-only scenario; anatomy remains unverified",
            "cortical_access_permitted": False}


@pytest.mark.parametrize("factory", ["coarse", "native", "native_config"])
@pytest.mark.parametrize("prohibition", [
    {"structural_coverage": "full_head"},
    {"allow_nonzero_mri_access_support": False},
    {"skull_stripped": False},
    {"structural_coverage": "full_head", "skull_stripped": True},
])
def test_explicit_full_head_prohibition_wins_over_collection_name(observed_fixture, factory, prohibition):
    # A known collection label must never override this particular volume's
    # explicit full-head classification or disabled MRI-support assumption.
    case = replace(observed_fixture, brain_mask=None,
                   metadata={"source_collection": {"name": "UCSF-PDGM"}, **prohibition})
    with pytest.raises(ValueError, match="(?i)(brain|support|full.head|skull)"):
        if factory == "coarse":
            make_patient_simulator(case)
        elif factory == "native":
            make_native_patient_simulator(case)
        else:
            native_config_from_case(case, access=AccessWindow((0, 0, 0), (0, 0, 1), 4))


def test_one_voxel_proposal_change_cannot_launder_unknown_support(observed_fixture):
    item = evidence(observed_fixture)
    case = observed_fixture.revised(brain_mask=None, structural_evidence={item.evidence_id: item})
    altered = item.mask.copy()
    altered[0, 0, 0] = ~altered[0, 0, 0]
    assert not np.array_equal(altered, item.mask)
    with pytest.raises(ValueError, match="BRAIN_MASK_REVIEW_REQUIRED"):
        generate_hypothetical_windows(case, support_mask=altered,
            support_provenance={"source": "uploaded", "method": "edited model mask",
                                "evidence_type": "observed"})
    # The documented opt-in remains possible only as a source-bound assumption,
    # including after an edit; it cannot silently become accepted anatomy.
    windows = generate_hypothetical_windows(case, support_mask=altered,
        support_provenance={"source": "uploaded", "method": "declared test scenario",
                            "evidence_type": "estimated", "assumption": assumption(case, altered)})
    assert windows


def test_exact_pending_proposal_cannot_be_overridden_by_formal_assumption(observed_fixture):
    item = evidence(observed_fixture)
    case = observed_fixture.revised(brain_mask=None, structural_evidence={item.evidence_id: item})
    with pytest.raises(ValueError, match="BRAIN_MASK_REVIEW_REQUIRED"):
        generate_hypothetical_windows(case, support_mask=item.mask,
            support_provenance={"source": item.evidence_hash, "method": "pending model",
                                "evidence_type": "estimated", "assumption": assumption(case, item.mask)})


@pytest.mark.parametrize("claimed_provenance", ["observed", "simulated"])
def test_reviewed_model_provenance_cannot_be_relabelled_by_support_caller(observed_fixture, claimed_provenance):
    item = accepted(evidence(observed_fixture))
    case = observed_fixture.revised(brain_mask=None, structural_evidence={item.evidence_id: item})
    record = validate_explicit_support(case, item.mask, {
        "source": "caller-supplied", "method": "caller description",
        "evidence_type": claimed_provenance})
    assert record["evidence_type"] == item.provenance == "estimated"
    assert record["source"] == item.evidence_hash
    assert record["cortical_access_permitted"] is False


def test_array_descriptor_change_cannot_reuse_accepted_review(observed_fixture):
    item = accepted(evidence(observed_fixture))
    selected = observed_fixture.revised(brain_mask=item.mask, structural_evidence={item.evidence_id: item})
    assert planning_brain_support(selected)[1]["review_status"] == "accepted"
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            item.mask.strides = tuple(reversed(item.mask.strides))
            selected.brain_mask.strides = tuple(reversed(selected.brain_mask.strides))
    except (AttributeError, TypeError, ValueError):
        return  # Refusing mutation itself also satisfies this boundary.
    with pytest.raises(ValueError, match="(?i)(metadata|layout|review|changed)"):
        planning_brain_support(selected)
    with pytest.raises(ValueError, match="(?i)(metadata|layout|review|changed)"):
        item.to_manifest()


@pytest.mark.parametrize("factory", ["coarse", "native", "native_config"])
def test_selected_review_does_not_authorize_adding_targets_outside_envelope(observed_fixture, factory):
    target = np.logical_or.reduce(list(observed_fixture.compartments.values()))
    mask = observed_fixture.brain_mask.copy()
    mask[target] = False
    item = accepted(evidence(observed_fixture, mask=mask))
    case = observed_fixture.revised(brain_mask=mask, structural_evidence={item.evidence_id: item})
    assert planning_brain_support(case)[1]["review_status"] == "accepted"
    with pytest.raises(ValueError, match="(?i)(outside|conflicting|target)"):
        if factory == "coarse":
            make_patient_simulator(case)
        elif factory == "native":
            make_native_patient_simulator(case)
        else:
            native_config_from_case(case, access=AccessWindow((0, 0, 0), (0, 0, 1), 4))


def test_unselected_proposals_and_reviews_leave_actual_worlds_unchanged(observed_fixture):
    source = observed_fixture.revised(brain_mask=None)
    first = evidence(source, mask=observed_fixture.brain_mask)
    second_mask = observed_fixture.brain_mask.copy()
    second_mask[0, 0, 0] = ~second_mask[0, 0, 0]
    second = accepted(evidence(source, evidence_id="other-model", mask=second_mask))
    updated = source.revised(structural_evidence={first.evidence_id: first, second.evidence_id: second})
    assert updated.semantic_hash != source.semantic_hash
    assert updated.planning_hash == source.planning_hash
    config = WorldGeneratorConfig(translation_scale_mm=(1., 2., 3.), rotation_scale_deg=(1., 1., 1.))
    before = generate_partitions(source.semantic_hash, config, 19, planning_hash=source.planning_hash)
    after = generate_partitions(updated.semantic_hash, config, 19, planning_hash=updated.planning_hash)
    for role in ("optimization", "selection", "final_evaluation", "stress"):
        original, revised = getattr(before, role), getattr(after, role)
        assert original.case_hash != revised.case_hash  # Trace still captures the evidence import.
        assert original.seeds == revised.seeds
        sampler = WorldGenerator(original.generator)
        for index in range(len(original.seeds)):
            np.testing.assert_array_equal(sampler.sample(original, index).anatomy_transform_mm,
                                          sampler.sample(revised, index).anatomy_transform_mm)
