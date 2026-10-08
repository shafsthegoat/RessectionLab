"""Actual acquired TOF/manual labels only; no generated patient or route data."""
import base64
from dataclasses import replace
from datetime import datetime, timezone
import gzip
import json

import numpy as np
import pytest

from resectionlab.core import PatientContext, array_digest, semantic_digest, thaw_json
from resectionlab.critical_evidence import (
    canonical_hard_exclusion, nifti1_header_record, resolve_critical_evidence,
    source_reference_grid_metrics,
)
from resectionlab.desktop_bridge import BridgeSession, _Request
from resectionlab.imaging import file_sha256, load_case, save_case
from resectionlab.lausanne_sub476_component import (
    DATA, EVIDENCE_ID, MANIFEST, MASK_SHA256, METHODS, ROOT,
    build_sub476_component, component_receipt,
)


@pytest.fixture(scope="module")
def actual_component():
    m = json.loads((ROOT/MANIFEST).read_bytes())
    paths = [ROOT/DATA/m["file"]["path"], ROOT/DATA/m["original_tof"]["path"], ROOT/METHODS]
    if not all(p.is_file() for p in paths):
        pytest.skip("Requires the actual pinned sub476 TOF, manual mask and archived methods")
    originals = {path: file_sha256(path) for path in paths}
    case = build_sub476_component()
    yield case
    assert {path: file_sha256(path) for path in paths} == originals


@pytest.fixture(scope="module")
def actual_bundle(actual_component, tmp_path_factory):
    return save_case(actual_component, tmp_path_factory.mktemp("sub476-component")/"actual.ressectionlab")


def test_actual_positive_support_is_consumed_without_negative_domain(actual_component):
    hard, constraints = canonical_hard_exclusion(actual_component)
    evidence = actual_component.critical_evidence[EVIDENCE_ID]
    assert np.count_nonzero(hard) == 193
    assert np.array_equal(hard, evidence.mask)
    assert np.array_equal(evidence.annotation_coverage, hard)
    assert constraints.receipt["missing"] == ("motor", "language")
    assert constraints.receipt["objective_structures"] == ()
    assert constraints.receipt["clinical_clearance"] is False
    assert constraints.receipt["time_scope"] == "retrospective_image_geometry_availability_unassessed"
    assert not actual_component.compartments and actual_component.brain_mask is None
    assert actual_component.metadata["scanner_frame_admitted"] is False
    assert actual_component.metadata["spatial_planning_admitted"] is False
    assert evidence.source.sha256 == MASK_SHA256
    assert evidence.source_binding["positive_class"] == "source_manual_voxelwise_aneurysm_region"
    receipt = component_receipt(actual_component)
    assert receipt["negative_annotation_voxels"] == 0
    assert receipt["background_unknown_voxels"] == 18800640-193
    # Evaluate actual source-positive indices, not a generated trajectory.
    coverage = constraints.route_coverage(np.argwhere(evidence.mask))
    assert coverage["vessels"]["covered_cells"] == 193
    assert coverage["vessels"]["meaning"] == "annotation_domain_only"
    assert coverage["motor"]["covered_cells"] is None


def test_untouched_headers_and_precision_proof_are_retained(actual_component):
    e = actual_component.critical_evidence[EVIDENCE_ID]
    d = e.derivation
    assert d["method"] == "source_reference_grid_normalization"
    raw, ref = d["annotation_raw_grid"], d["reference_raw_grid"]
    assert raw["raw_grid"]["spatial_units"] == "unknown"
    assert (raw["raw_grid"]["qform_code"], raw["raw_grid"]["sform_code"]) == (0, 2)
    assert ref["raw_grid"]["spatial_units"] == "mm"
    assert (ref["raw_grid"]["qform_code"], ref["raw_grid"]["sform_code"]) == (1, 1)
    assert not np.array_equal(raw["raw_grid"]["sform_numeric"], ref["raw_grid"]["sform_numeric"])
    assert np.array_equal(actual_component.affine, ref["raw_grid"]["sform_numeric"])
    proof = source_reference_grid_metrics(raw, ref)
    assert proof["maximum_observed_float32_steps"] == 3
    assert proof["maximum_allowed_float32_steps"] == 4
    assert proof["all_voxel_centres_keep_reference_index"] is True
    assert max(proof["serialization_component_bound_voxels"]) < .5
    m = json.loads((ROOT/MANIFEST).read_bytes())
    for entry, path in [(raw, m["file"]["path"]), (ref, m["original_tof"]["path"])]:
        with gzip.open(ROOT/DATA/path, "rb") as stream:
            assert base64.b64decode(entry["header_base64"]) == stream.read(348)
    assert e.available_at is None and e.review["available_at"] is None
    assert e.review["binding_kind"] == "software_provenance_binding_not_clinician_signature"


@pytest.mark.parametrize("method", ["identity_grid", "registration"])
def test_normalization_is_not_relabeled_or_replaced(actual_component, method):
    e = actual_component.critical_evidence[EVIDENCE_ID]
    changed = thaw_json(e.derivation)
    changed["method"] = method
    with pytest.raises(ValueError, match="identity_grid|registration/resampling"):
        replace(e, derivation=changed, review=None)


@pytest.mark.parametrize("change", ["raw_file", "sidecar", "precision", "scope"])
def test_rehashed_metadata_cannot_bypass_normalization_checks(actual_component, change):
    e = actual_component.critical_evidence[EVIDENCE_ID]
    d = thaw_json(e.derivation)
    if change == "raw_file":
        d["annotation_raw_grid"]["source_file_sha256"] = d["source_image_sha256"]
    elif change == "sidecar":
        # A contradictory declared filename; no image/label bytes are changed.
        d["reference_filename"] = "different_acquisition.nii.gz"
    elif change == "precision":
        d["precision_equivalence"]["maximum_allowed_float32_steps"] = 100
    else:
        d["scanner_frame_admitted"] = True
    d.pop("normalization_record_hash")
    d["normalization_record_hash"] = semantic_digest(d)
    with pytest.raises(ValueError):
        replace(e, derivation=d, review=None)


def test_another_actual_acquisition_cannot_supply_normalized_grid(actual_component):
    m = json.loads((ROOT/MANIFEST).read_bytes())
    acquired = json.loads((ROOT/m["original_acquisition"]["path"]).read_bytes())
    t1 = next(row for row in acquired["files"] if row["path"].endswith("_T1w.nii.gz"))
    assert file_sha256(ROOT/DATA/t1["path"]) == t1["sha256"]
    with gzip.open(ROOT/DATA/t1["path"], "rb") as stream:
        other = nifti1_header_record(stream.read(348), t1["sha256"])
    raw = actual_component.critical_evidence[EVIDENCE_ID].derivation["annotation_raw_grid"]
    with pytest.raises(ValueError, match="dimensions|qform/sform"):
        source_reference_grid_metrics(raw, other)


def test_unknown_availability_excludes_timestamped_use(actual_component):
    timed = replace(actual_component, context=PatientContext(datetime(2026, 10, 8, tzinfo=timezone.utc)))
    constraints = resolve_critical_evidence(timed)
    assert constraints.hard_exclusion is None and constraints.planning_binding is None
    assert constraints.receipt["records"][EVIDENCE_ID]["exclusion_reason"] == "availability_time_unknown"
    assert constraints.receipt["missing"] == ("motor", "language", "vessels")


def test_provenance_change_invalidates_review_and_planning_binding(actual_component):
    original = actual_component.critical_evidence[EVIDENCE_ID]
    d = thaw_json(original.derivation)
    d["inference_rationale"] += " The interpretation is limited to a component check."
    with pytest.raises(ValueError, match="normalization record hash"):
        replace(original, derivation=d, review=None)
    d.pop("normalization_record_hash")
    d["normalization_record_hash"] = semantic_digest(d)
    with pytest.raises(ValueError, match="Review does not bind"):
        replace(original, derivation=d)
    updated = replace(original, derivation=d, review=None)
    review = {**thaw_json(original.review), "content_hash": updated.content_hash}
    updated = replace(updated, review=review)
    changed = replace(actual_component, critical_evidence={EVIDENCE_ID: updated})
    assert array_digest(changed.mri) == array_digest(actual_component.mri)
    assert array_digest(updated.mask) == array_digest(original.mask)
    assert changed.semantic_hash != actual_component.semantic_hash
    assert resolve_critical_evidence(changed).fingerprint != resolve_critical_evidence(actual_component).fingerprint


def test_actual_receipt_and_bundle_roundtrip(actual_component, actual_bundle):
    reopened = load_case(actual_bundle)
    assert reopened.semantic_hash == actual_component.semantic_hash
    assert reopened.planning_hash == actual_component.planning_hash
    assert component_receipt(reopened) == component_receipt(actual_component)
    assert np.count_nonzero(canonical_hard_exclusion(reopened)[0]) == 193


def test_desktop_inspects_actual_positive_evidence_without_planning(actual_component, actual_bundle, tmp_path):
    bridge = BridgeSession(tmp_path/"transfers")
    progress = lambda *_: None
    try:
        loaded = bridge.execute("loadCase", {"path": str(actual_bundle)}, _Request("open"), progress)
        inspected = bridge.execute("inspectEvidence", {"caseHash": loaded["caseHash"]}, _Request("inspect"), progress)
        receipt = inspected["criticalEvidence"]
        assert receipt == thaw_json(resolve_critical_evidence(actual_component).receipt)
        record = receipt["records"][EVIDENCE_ID]
        assert record["annotated_cells"] == 193 and record["exclusion_reason"] is None
        assert record["source_binding"]["background_meaning"] == "unknown_for_vascular_anatomy"
        derivation = record["derivation"]
        assert derivation == thaw_json(actual_component.critical_evidence[EVIDENCE_ID].derivation)
        assert derivation["method"] == "source_reference_grid_normalization"
        assert derivation["annotation_raw_grid"]["raw_grid"]["spatial_units"] == "unknown"
        assert derivation["reference_raw_grid"]["raw_grid"]["spatial_units"] == "mm"
        assert derivation["scanner_frame_admitted"] is False
        assert derivation["spatial_planning_admitted"] is False
        assert receipt["missing"] == ["motor", "language"] and not receipt["objective_structures"]
        assert bridge.cases[loaded["caseHash"]].routes is None
    finally:
        bridge.close()
