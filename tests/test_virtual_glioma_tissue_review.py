"""Generated tiny arrays only, including fixtures exercising acquired-source rules.

No patient images, model or real-host solver execution. Fixture attestations
are test data, never a review of an acquired person or anatomical result.
"""
from dataclasses import replace
import numpy as np
import pytest

from resectionlab.core import CaseData, SourceRef, array_digest, semantic_digest, thaw_json
from resectionlab.structural_evidence import structural_frame_hash
import resectionlab.virtual_glioma as vg


def fixture(*, generated=False):
    shape = (6, 5, 4)
    brain = np.ones(shape, bool)
    unknown = np.zeros(shape, bool)
    unknown[0, 0, 0] = True
    support = brain & ~unknown
    wm = (np.indices(shape)[0] < 3) & support
    gm = support & ~wm
    case = CaseData("tiny-contract-fixture", np.arange(120, dtype=np.float32).reshape(shape),
        {"wm": wm, "gm": gm, "support": support, "unknown": unknown}, np.diag([2., 2., 2., 1.]),
        tuple(SourceRef(name, "generated://tiny-contract-fixture/" + name, sha256=digit * 64,
                        provenance="simulated" if generated else "observed", native_frame="RAS+")
              for name, digit in (("image", "1"), ("labels", "2"), ("method_receipt", "3"))),
        brain_mask=brain, metadata={"is_synthetic": generated, "role": "TRAIN", "patient_group": "fixture:pair"})
    layers = {}
    for name, kind in (("wm", "white_matter"), ("gm", "gray_matter"),
                       ("support", "classification_support"), ("unknown", "unknown")):
        layers[name] = {"tissue_class": kind, "source_role": "generated_numerical_control" if generated else "fitted_intensity_class",
            "source_sha256": "2" * 64, "run_sha256": "3" * 64, "method": "tiny generated test partition",
            "provenance": "simulated" if generated else "estimated", "source_image_hash": array_digest(case.mri),
            "source_frame_hash": structural_frame_hash(case)}
    return replace(case, metadata={**case.metadata, "anatomy_layers": layers})


def add_fixture_review(case):
    """Independently construct the explicit review format; production cannot mint it."""
    layers = case.metadata["anatomy_layers"]
    evidence = {"case_id": case.case_id, "patient_group": "fixture:pair",
        "source_image_hash": array_digest(case.mri), "source_frame_hash": structural_frame_hash(case),
        "source_refs_hash": semantic_digest([ref.to_dict() for ref in case.source_refs]),
        "brain_envelope_hash": array_digest(case.brain_mask), "classification_support_layer": "support",
        "unknown_layer": "unknown", "representation": "disjoint_binary_tissue_classes",
        "physical_partial_volume_claim": False,
        "layers": {kind: {"name": name, "array_hash": array_digest(case.compartments[name]), "provenance": thaw_json(layers[name])}
                   for kind, name in (("white_matter", "wm"), ("gray_matter", "gm"),
                                      ("classification_support", "support"), ("unknown", "unknown"))}}
    digest = semantic_digest(evidence)
    record = {"schema": vg.TISSUE_REVIEW_SCHEMA, "evidence": evidence, "evidence_hash": digest,
        "review": {"evidence_hash": digest, "reviewer_id": "generated-test-reviewer", "reviewed_at": "2026-01-01T00:00:00+00:00",
            "decision": "accepted", "scope": vg.TISSUE_REVIEW_SCOPE, "rationale": "Generated fixture; no human anatomy reviewed"}}
    return replace(case, metadata={**case.metadata, "fk_tissue_pair_review": record})


def qualify(case, *, uniform=False, grid_shape=None, grid_affine=None):
    generated = case.metadata.get("is_synthetic") is True
    return vg.qualify_host(case, patient_group="fixture:pair", role="TRAIN",
        healthy_source_evidence={"status": "generated_numerical_control" if generated else "source_reported_healthy",
            "patient_group": "fixture:pair", "source_url": "https://example.invalid/generated-contract-fixture",
            "source_sha256": "4" * 64, "statement": "Generated test text; never a statement about an acquired person"},
        qualified_by="generated-test", grid_shape=list(case.mri.shape) if grid_shape is None else grid_shape,
        grid_affine=case.affine if grid_affine is None else grid_affine,
        tissue_mode=vg.UNIFORM_DIFFUSION_DOMAIN if uniform else vg.GM_WM_FRACTIONS,
        source_wm_layer=None if uniform else "wm", source_gm_layer=None if uniform else "gm")


def forbid_resampling(monkeypatch):
    monkeypatch.setattr(vg, "_resample", lambda *a, **k: pytest.fail("Unreviewed/stale input reached resampling"))
    monkeypatch.setattr(vg, "load_pinned_kernel", lambda *a, **k: pytest.fail("Guard test reached solver loading"))


def test_metadata_class_and_blanket_acceptance_cannot_grant_review(monkeypatch):
    case = fixture()
    case = replace(case, metadata={**case.metadata, "accepted": True, "review_status": "accepted"})
    forbid_resampling(monkeypatch)
    with pytest.raises(ValueError, match="pair-bound"):
        qualify(case)


def test_exact_pair_review_admits_only_declared_categorical_support():
    case = add_fixture_review(fixture())
    q = qualify(case)
    _, _, _, domain, wm, gm, mapping = vg._validate_qualification(case, q)
    assert np.array_equal(domain, case.compartments["support"])
    assert np.array_equal(wm > 0, case.compartments["wm"])
    assert np.array_equal(gm > 0, case.compartments["gm"])
    assert not domain[0, 0, 0]
    assert mapping["tissuePairReview"] == thaw_json(case.metadata["fk_tissue_pair_review"])
    assert mapping["tissueInterpolationSemantics"].endswith("not_physical_PVE")


def test_shifted_oblique_interpolation_cannot_reenter_unknown_source_support():
    case = fixture()
    angle = .4
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0.], [np.sin(angle), np.cos(angle), 0.], [0., 0., 1.]])
    affine = np.eye(4); affine[:3, :3] = rotation @ np.diag([2., 1.5, 2.5]); affine[:3, 3] = [7., -2., 3.]
    case = replace(case, affine=affine)
    metadata = thaw_json(case.metadata)
    for layer in metadata["anatomy_layers"].values(): layer["source_frame_hash"] = structural_frame_hash(case)
    case = add_fixture_review(replace(case, metadata=metadata))
    target = affine.copy(); target[:3, 3] = (affine @ np.array([.25, .25, .25, 1.]))[:3]
    q = qualify(case, grid_shape=[5, 4, 3], grid_affine=target)
    _, _, _, domain, wm, gm, mapping = vg._validate_qualification(case, q)
    assert wm[0, 0, 0] + gm[0, 0, 0] == pytest.approx(1 - .75 ** 3)
    assert wm[0, 0, 0] + gm[0, 0, 0] > q["mapping"]["matterThreshold"]
    # Independently derived nearest-center mapping: every .25-shifted target
    # center rounds back to the same integer source index on these bounds.
    known = case.compartments["support"][:5, :4, :3]
    unknown = case.compartments["unknown"][:5, :4, :3]
    assert not domain[0, 0, 0]
    assert not np.any(domain & unknown)
    assert not np.any(domain & ~known)
    coverage = mapping["reviewedCoverageMapping"]
    assert coverage["classificationSupportHash"] == array_digest(known)
    assert coverage["unknownHash"] == array_digest(unknown)
    assert coverage["interpolation"] == "nearest_neighbor_order0_constant0_voxel_centers"
    assert mapping["domainRule"] == coverage["domainRule"]


@pytest.mark.parametrize("change", ["class_cell", "class_swap", "unknown_coverage", "image", "frame", "brain", "method", "source", "patient"])
def test_review_cannot_be_replayed_after_pair_or_source_changes(monkeypatch, change):
    case = add_fixture_review(fixture())
    masks = {key: value.copy() for key, value in case.compartments.items()}
    metadata = thaw_json(case.metadata)
    if change == "class_cell":
        masks["wm"][1, 1, 1] = False
        case = replace(case, compartments=masks)
    elif change == "class_swap":
        case = replace(case, compartments={**masks, "wm": masks["gm"], "gm": masks["wm"]})
    elif change == "unknown_coverage":
        masks["wm"][1, 1, 1] = masks["support"][1, 1, 1] = False
        masks["unknown"][1, 1, 1] = True
        case = replace(case, compartments=masks)
    elif change == "image":
        image = case.mri.copy(); image[1, 1, 1] += 1
        case = replace(case, mri=image)
    elif change == "frame":
        affine = case.affine.copy(); affine[0, 3] += 1
        case = replace(case, affine=affine)
    elif change == "brain":
        brain = case.brain_mask.copy(); brain[1, 1, 1] = False
        case = replace(case, brain_mask=brain)
    elif change == "method":
        metadata["anatomy_layers"]["gm"]["method"] = "different class method"
        case = replace(case, metadata=metadata)
    elif change == "source":
        case = replace(case, source_refs=tuple(replace(ref, uri=ref.uri + "/different") for ref in case.source_refs))
    else:
        case = replace(case, case_id="different-generated-person")
    forbid_resampling(monkeypatch)
    with pytest.raises(ValueError):
        qualify(case)


@pytest.mark.parametrize("change", ["scope", "rejected", "naive_time", "empty_reviewer", "unbound_digest", "pve_claim"])
def test_review_contract_fields_are_not_generic_acceptance(monkeypatch, change):
    case = add_fixture_review(fixture()); metadata = thaw_json(case.metadata)
    record = metadata["fk_tissue_pair_review"]
    if change == "scope": record["review"]["scope"] = "research_brain_envelope_only"
    elif change == "rejected": record["review"]["decision"] = "rejected"
    elif change == "naive_time": record["review"]["reviewed_at"] = "2026-01-01T00:00:00"
    elif change == "empty_reviewer": record["review"]["reviewer_id"] = " "
    elif change == "unbound_digest": record["review"]["evidence_hash"] = "sha256:" + "0" * 64
    else:
        record["evidence"]["physical_partial_volume_claim"] = True
        digest = semantic_digest(record["evidence"])
        record["evidence_hash"] = record["review"]["evidence_hash"] = digest
    forbid_resampling(monkeypatch)
    with pytest.raises(ValueError): qualify(replace(case, metadata=metadata))


@pytest.mark.parametrize("change", ["overlap", "unknown_tissue", "missing_unknown", "missing_provenance", "missing_run_ref"])
def test_fresh_fixture_attestation_cannot_override_support_or_provenance(monkeypatch, change):
    case = fixture(); masks = {key: value.copy() for key, value in case.compartments.items()}
    metadata = thaw_json(case.metadata)
    if change == "overlap":
        masks["gm"][1, 1, 1] = True; case = replace(case, compartments=masks)
    elif change == "unknown_tissue":
        masks["wm"][0, 0, 0] = True; case = replace(case, compartments=masks)
    elif change == "missing_unknown":
        masks["unknown"][:] = False; case = replace(case, compartments=masks)
    elif change == "missing_provenance":
        del metadata["anatomy_layers"]["support"]["method"]; case = replace(case, metadata=metadata)
    else: case = replace(case, source_refs=case.source_refs[:2])
    case = add_fixture_review(case)
    forbid_resampling(monkeypatch)
    with pytest.raises(ValueError): qualify(case)


def test_legacy_strict_receipt_and_resealed_outer_qualification_cannot_bypass_review(monkeypatch):
    reviewed = add_fixture_review(fixture()); q = qualify(reviewed)
    no_review = fixture()
    # Recompute the outer binding/hash: the required inner review is still absent.
    q["sourceBinding"] = vg._source_binding(no_review)
    q["qualificationHash"] = semantic_digest({key: value for key, value in q.items() if key != "qualificationHash"})
    forbid_resampling(monkeypatch)
    with pytest.raises(ValueError, match="pair-bound"):
        vg._validate_qualification(no_review, q)


def test_generated_exemption_preserves_existing_strict_fixture_mapping():
    case = fixture(generated=True); q = qualify(case)
    assert "tissuePairReview" not in q["mapping"]
    assert q["qualificationHash"] == "sha256:411ee7131f27c6261ed75a2e49c8b5ad01e7ca03da5e50ce42bb17d74e81c067"
    assert vg._validate_qualification(case, q)[-1] == q["mapping"]


@pytest.mark.parametrize("change", ["one_acquired_ref", "one_real_layer", "synthetic_flag_only"])
def test_generated_exemption_requires_every_lineage_condition(monkeypatch, change):
    case = fixture(generated=True); metadata = thaw_json(case.metadata)
    if change == "one_acquired_ref":
        case = replace(case, source_refs=(replace(case.source_refs[0], provenance="observed"), *case.source_refs[1:]))
    elif change == "one_real_layer":
        metadata["anatomy_layers"]["wm"]["source_role"] = "fitted_intensity_class"
        case = replace(case, metadata=metadata)
    else:
        case = fixture(); case = replace(case, metadata={**case.metadata, "is_synthetic": True})
    forbid_resampling(monkeypatch)
    with pytest.raises(ValueError): qualify(case)


@pytest.mark.parametrize("generated,expected_hash", [
    (False,"sha256:86b4d0bff3b7844240f2c7e2d30532ba50a808f8231c161a1b9cb7c407255c82"),
    (True,"sha256:83bdea3fc1bdebeb3a1aada6c36e3b3f4d1fef81ebf0f3e78ce656c81e5b3498")])
def test_uniform_qualification_and_reconstruction_are_identical_to_before(generated, expected_hash):
    case = fixture(generated=generated); q = qualify(case, uniform=True)
    assert q["qualificationHash"] == expected_hash
    assert "tissuePairReview" not in q["mapping"]
    assert vg._validate_qualification(case, q)[-1] == q["mapping"]
