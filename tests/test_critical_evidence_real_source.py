"""Real acquired source and metadata-refusal checks; no patient fixtures generated.

These checks intentionally cannot establish positive vascular integration: the
Lausanne original has no admitted vessel annotation. Local data are required.
"""
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from resectionlab.core import array_digest
from resectionlab.critical_evidence import (
    availability_exclusion, canonical_hard_exclusion, require_canonical_masks,
    resolve_critical_evidence,
)
from resectionlab.desktop_bridge import BridgeError, BridgeSession, _Request
from resectionlab.evaluation import independent_check_native_history
from resectionlab.imaging import load_case, load_nifti_case, save_case
from resectionlab.native_refinement import _validate_critical_replay_binding
from resectionlab.simulation import _rank_search_children

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def actual_case():
    record = json.loads((ROOT / "artifacts/lausanne-original-pilot-v1/acquisition.json").read_bytes())
    original = next(row for row in record["files"] if row["path"].endswith("_T1w.nii.gz"))
    path = ROOT / "data/anatomy/ds003949-v1.0.1" / original["path"]
    if not path.is_file():
        pytest.skip("Requires the hash-verified actual Lausanne TRAIN sub-000 original")
    assert record["subject"] == "sub-000" and record["role"] == "TRAIN"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == original["sha256"]
    case = load_nifti_case(path, case_id="Lausanne-sub-000-ses-20110101", license="CC0",
        source_url=original["source_url"], metadata={"structural_coverage": "full_head",
        "evidence_identity": {"dataset": "ds003949-v1.0.1", "participant": "sub-000", "timepoint": "ses-20110101"}})
    yield case
    assert hashlib.sha256(path.read_bytes()).hexdigest() == original["sha256"]


def test_actual_missing_evidence_stays_missing(actual_case):
    result = resolve_critical_evidence(actual_case)
    assert not actual_case.critical_evidence
    assert all(value is None for value in result.masks.values())
    assert all(value is None for value in result.annotation_coverage.values())
    assert result.hard_exclusion is None
    assert result.planning_binding is None
    assert list(result.receipt["missing"]) == ["motor", "language", "vessels"]
    assert not result.receipt["objective_structures"]
    assert result.receipt["clinical_clearance"] is False
    assert result.receipt["trajectory_coverage"] == "not_computed"
    assert "critical_evidence" not in actual_case.to_manifest()


def test_empty_registry_roundtrip_preserves_actual_source(actual_case, tmp_path):
    reopened = load_case(save_case(actual_case, tmp_path / "actual.ressectionlab"))
    assert reopened.semantic_hash == actual_case.semantic_hash
    assert reopened.planning_hash == actual_case.planning_hash
    assert reopened.prior_registration_input_hash == actual_case.prior_registration_input_hash
    assert array_digest(reopened.mri) == array_digest(actual_case.mri)
    assert reopened.source_refs == actual_case.source_refs
    assert not reopened.critical_evidence


def test_untyped_critical_record_is_rejected(actual_case):
    with pytest.raises(ValueError, match="IDs and typed records"):
        replace(actual_case, critical_evidence={"unbound": {"structure": "vessels"}})


@pytest.mark.parametrize("name", ["vessels", "unrecognized"])
def test_raw_override_without_source_is_rejected(actual_case, name):
    # Scalar protocol misuse, not a fabricated anatomical array.
    with pytest.raises(ValueError, match="source-bound"):
        require_canonical_masks(resolve_critical_evidence(actual_case), {name: True})
    with pytest.raises(ValueError, match="source-bound"):
        canonical_hard_exclusion(actual_case, True)


@pytest.mark.parametrize("indices", [np.array([[-1, 0, 0]]), np.array([[0., .5, 0.]]),
                                     np.array([0, 0, 0])])
def test_invalid_coverage_indices_are_not_wrapped_or_truncated(actual_case, indices):
    with pytest.raises(ValueError, match="indices"):
        resolve_critical_evidence(actual_case).route_coverage(indices)


def test_upper_grid_boundary_is_rejected(actual_case):
    with pytest.raises(ValueError, match="within the source image"):
        resolve_critical_evidence(actual_case).route_coverage(np.array([actual_case.mri.shape]))


def test_coordinate_accounting_does_not_create_missing_coverage(actual_case):
    # Repeated index identity on the acquired grid, not an authored route.
    result = resolve_critical_evidence(actual_case).route_coverage(np.array([[0, 0, 0], [0, 0, 0]]))
    assert all(row["in_image_swept_cells"] == 1 and row["covered_cells"] is None
               and row["uncovered_cells"] is None for row in result.values())


def test_unbound_replay_metadata_cannot_claim_absent_anatomy(actual_case):
    # Protocol declarations only: no trajectory, outcome or observation is made.
    _validate_critical_replay_binding(actual_case, None)
    _validate_critical_replay_binding(actual_case, {})
    for name in ("critical_evidence", "hard_exclusion"):
        with pytest.raises(ValueError, match="Native replay"):
            _validate_critical_replay_binding(actual_case, {"evidence_and_constraints": {name: {}}})


def test_independent_audit_rejects_unsourced_override_before_geometry(actual_case):
    with pytest.raises(ValueError, match="source-bound"):
        independent_check_native_history(actual_case, (), (), tissue_mask=None,
                                         access=None, hard_exclusion=True)


@pytest.mark.parametrize("available,review,cutoff,reason", [
    (None, None, None, None),
    (None, None, "2000-01-01T12:00:00Z", "availability_time_unknown"),
    ("2000-01-01T12:00:01Z", None, "2000-01-01T12:00:00Z", "available_after_planning_cutoff"),
    ("2000-01-01T11:00:00Z", None, "2000-01-01T12:00:00Z", "review_availability_time_unknown"),
    ("2000-01-01T11:00:00Z", "2000-01-01T12:00:01Z", "2000-01-01T12:00:00Z", "review_available_after_planning_cutoff"),
    ("2000-01-01T12:00:00Z", "2000-01-01T12:00:00Z", "2000-01-01T12:00:00Z", None),
    ("2000-01-01T12:00:00Z", "2000-01-01T07:00:00-05:00", "2000-01-01T12:00:00Z", None),
])
def test_clock_ordering_only(available, review, cutoff, reason):
    # Abstract timestamp identities; these do not describe patient events.
    assert availability_exclusion(available, review, cutoff) == reason


def test_naive_clock_is_rejected():
    with pytest.raises(ValueError, match="timezone"):
        availability_exclusion(None, None, "2000-01-01T12:00:00")


def test_equal_numeric_keys_ignore_opaque_identifier_renaming():
    # Sorting identity only. These scalars are not actions, rewards or episodes.
    for identifiers in (("z", "a", "m"), ("a", "z", "b")):
        entries = [(2, identifiers[0], 0), (2, identifiers[1], 1), (3, identifiers[2], 2)]
        assert [entry[2] for entry in _rank_search_children(entries)] == [2, 0, 1]
        assert [entry[2] for entry in entries] == [0, 1, 2]


def test_desktop_load_inspect_and_missing_support_refusal(actual_case, tmp_path):
    path = save_case(actual_case, tmp_path / "actual.ressectionlab")
    session = BridgeSession(tmp_path / "transfers")
    progress = lambda *_: None
    try:
        loaded = session.execute("loadCase", {"path": str(path)}, _Request("open"), progress)
        assert loaded["criticalEvidence"]["missing"] == ["motor", "language", "vessels"]
        inspected = session.execute("inspectEvidence", {"caseHash": loaded["caseHash"]}, _Request("inspect"), progress)
        assert inspected["criticalEvidence"] == loaded["criticalEvidence"]
        with pytest.raises(BridgeError) as error:
            session.execute("generateRoutes", {"caseHash": loaded["caseHash"]}, _Request("routes"), progress)
        assert error.value.code == "ACCESS_SUPPORT_REQUIRED"
        assert session.cases[loaded["caseHash"]].routes is None
    finally:
        session.close()
