"""Cohort leakage tests distinguish patient adaptation from population training."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from resectionlab.cohort import audit_registry, check_evidence, input_exclusion_reason


@pytest.fixture
def registry(tmp_path):
    payload = b"id,alias\npatient-A,public-A\npatient-B,public-B\n"
    (tmp_path / "identity.csv").write_bytes(payload)
    return {
        "schema_version": 1,
        "evidence": [{"id": "identity-A", "kind": "delimited_row", "path": "identity.csv",
                      "sha256": hashlib.sha256(payload).hexdigest(),
                      "supports_identities": ["collection:patient-A", "collection:public-A", "BraTS:case-A"],
                      "match": {"id": "patient-A"}, "expect": {"alias": "public-A"}}],
        "identity_links": [],
        "records": [],
    }


def record(record_id, identity="collection:patient-A", split="development", uses=None, **changes):
    return {
        "record_id": record_id, "identity": identity, "identity_status": "verified_primary",
        "identity_evidence_ids": ["identity-A"], "visit_id": "baseline",
        "derivative_of": None, "outer_split": split, "uses": uses or ["viewer_qc"], **changes,
    }


def codes(report):
    return {item["code"] for item in report["issues"]}


def test_repeated_visits_are_not_independent(registry, tmp_path):
    registry["records"] = [record("baseline"), record("followup", split="final_evaluation", visit_id="followup")]
    report = audit_registry(registry, tmp_path)
    assert report["registered_patient_groups"] == 1
    assert "CROSS_SPLIT_PATIENT_LEAKAGE" in codes(report)
    assert not report["registry_valid"]


def test_transitive_brats_alias_detects_pretraining_leak(registry, tmp_path):
    registry["identity_links"] = [
        {"id": "first", "identities": ["collection:patient-A", "collection:public-A"], "evidence_ids": ["identity-A"]},
        {"id": "second", "identities": ["collection:public-A", "BraTS:case-A"], "evidence_ids": ["identity-A"]},
    ]
    registry["records"] = [record("shared-model-training", "BraTS:case-A", uses=["population_pretraining"]),
                           record("outer-final", split="final_evaluation", uses=["final_scoring"])]
    report = audit_registry(registry, tmp_path)
    assert report["registered_patient_groups"] == 1
    assert "FINAL_PATIENT_USED_FOR_POPULATION_PRETRAINING" in codes(report)


def test_pretraining_flag_cannot_hide_in_final_split(registry, tmp_path):
    registry["records"] = [record("case", split="final_evaluation", uses=["population_pretraining", "final_scoring"])]
    assert "FINAL_PATIENT_USED_FOR_POPULATION_PRETRAINING" in codes(audit_registry(registry, tmp_path))


def test_case_adaptation_is_allowed_within_outer_final(registry, tmp_path):
    registry["records"] = [record("case", split="final_evaluation",
                                  uses=["case_optimization", "checkpoint_selection", "final_scoring"])]
    report = audit_registry(registry, tmp_path)
    assert report["registry_valid"]
    assert report["independence_claim_status"] == "known_identity_overlap_checks_passed"
    assert "Within-patient world isolation requires separate protocol checks" in report["limits"]


def test_global_development_cannot_be_renamed_final(registry, tmp_path):
    registry["records"] = [record("case", split="final_evaluation", uses=["global_development"])]
    assert "FINAL_PATIENT_USED_FOR_GLOBAL_DEVELOPMENT" in codes(audit_registry(registry, tmp_path))


def test_derivative_lineage_closes_identifier_loophole(registry, tmp_path):
    registry["records"] = [record("source", uses=["population_pretraining"]),
                           record("derived", identity=None, identity_status="unknown", derivative_of="source", split="final_evaluation")]
    report = audit_registry(registry, tmp_path)
    assert report["registered_patient_groups"] == 1
    assert "CROSS_SPLIT_PATIENT_LEAKAGE" in codes(report)
    assert "PATIENT_IDENTITY_NOT_VERIFIED" in codes(report)


def test_derivative_cannot_name_another_verified_patient(registry, tmp_path):
    registry["evidence"].append(dict(registry["evidence"][0], id="identity-B", supports_identities=["other:patient-B"],
                                     match={"id": "patient-B"}, expect={"alias": "public-B"}))
    registry["records"] = [record("source"), record("derived", identity="other:patient-B", derivative_of="source", identity_evidence_ids=["identity-B"])]
    assert "CONTRADICTORY_DERIVATIVE_IDENTITY" in codes(audit_registry(registry, tmp_path))


@pytest.mark.parametrize("state", ["unknown", "mirror_attributed"])
def test_unverified_identity_cannot_establish_final_independence(registry, tmp_path, state):
    registry["records"] = [record("case", split="final_evaluation", identity_status=state)]
    report = audit_registry(registry, tmp_path)
    assert not report["registry_valid"]
    assert report["groups_with_verified_primary_identity"] == 0
    assert report["independence_claim_status"] == "blocked_by_registry_errors"


def test_different_source_namespaces_are_not_silently_same_person(registry, tmp_path):
    registry["records"] = [record("A", "dataset-one:P01", identity_status="unknown"),
                           record("B", "dataset-two:P01", identity_status="unknown")]
    report = audit_registry(registry, tmp_path)
    assert report["registered_patient_groups"] == 2
    assert report["groups_with_verified_primary_identity"] == 0
    assert "No undeclared or undiscoverable cross-dataset identity overlap is ruled out" in report["limits"]


def test_missing_source_evidence_blocks_identity_claim(registry, tmp_path):
    registry["records"] = [record("case", split="final_evaluation")]
    (tmp_path / "identity.csv").unlink()
    report = audit_registry(registry, tmp_path)
    assert not report["registry_valid"]
    assert report["evidence"]["identity-A"]["reason"] == "source_evidence_missing"


def test_evidence_for_another_patient_cannot_certify_identity(registry, tmp_path):
    registry["records"] = [record("B", identity="collection:patient-B", split="final_evaluation")]
    report = audit_registry(registry, tmp_path)
    assert not report["registry_valid"]
    assert report["groups_with_verified_primary_identity"] == 0


def test_source_corruption_does_not_trust_verified_label(registry, tmp_path):
    registry["records"] = [record("case", split="final_evaluation")]
    (tmp_path / "identity.csv").write_text("id,alias\npatient-A,wrong-patient\n")
    report = audit_registry(registry, tmp_path)
    assert not report["registry_valid"]
    assert report["evidence"]["identity-A"]["reason"] == "source_evidence_hash_mismatch"


def test_changed_identity_assertion_fails_with_intact_bytes(registry, tmp_path):
    registry["evidence"][0]["expect"]["alias"] = "invented-alias"
    result = check_evidence(registry["evidence"][0], tmp_path)
    assert result == {"verified": False, "reason": "identity_assertion_mismatch"}


def test_unknown_link_is_conservatively_grouped(registry, tmp_path):
    registry["identity_links"] = [{"id": "unresolved", "identities": ["collection:patient-A", "alias:unknown"], "evidence_ids": []}]
    registry["records"] = [record("source"), record("aliased", "alias:unknown", split="final_evaluation")]
    report = audit_registry(registry, tmp_path)
    assert "UNVERIFIED_IDENTITY_LINK" in codes(report)
    assert "CROSS_SPLIT_PATIENT_LEAKAGE" in codes(report)


def test_missing_and_cyclic_derivative_parentage_are_rejected(registry, tmp_path):
    registry["records"] = [record("a", derivative_of="missing")]
    assert "UNRESOLVED_DERIVATIVE_PARENT" in codes(audit_registry(registry, tmp_path))
    registry["records"] = [record("a", derivative_of="b"), record("b", derivative_of="a")]
    assert "CYCLIC_DERIVATIVE_LINEAGE" in codes(audit_registry(registry, tmp_path))


def test_unsafe_evidence_path_is_rejected(registry, tmp_path):
    spec = dict(registry["evidence"][0], path="../identity.csv")
    assert check_evidence(spec, tmp_path)["reason"] == "unsafe_evidence_path"


def test_release_counts_do_not_become_eligible_patients(registry, tmp_path):
    registry["collections"] = [{"published_unique_patients": 99999, "eligible_patient_count": None}]
    report = audit_registry(registry, tmp_path)
    assert report["registered_patient_groups"] == 0
    assert report["final_patient_groups"] == 0
    assert report["independence_claim_status"] == "no_final_patients_registered"


def test_postoperative_or_timing_unknown_metadata_is_excluded():
    base = {"kind": "molecular", "source_evidence_ids": ["source"], "operative_phase": "preoperative",
            "availability_basis": "source_declares_preoperative_images"}
    assert input_exclusion_reason(base, None) == "availability_time_unknown"
    assert input_exclusion_reason(dict(base, operative_phase="postoperative"), None) == "postoperative_information_excluded"
    assert input_exclusion_reason(dict(base, kind="outcome"), None) == "postoperative_information_excluded"


def test_preoperative_imaging_phase_does_not_invent_a_calendar_time():
    base = {"kind": "imaging", "source_evidence_ids": ["source"], "operative_phase": "preoperative",
            "availability_basis": "source_declares_preoperative_images"}
    assert input_exclusion_reason(base, None) is None
    assert input_exclusion_reason(dict(base, kind="source_annotation"), None) == "source_annotation_requires_annotation_assisted_track"
    assert input_exclusion_reason(dict(base, kind="source_annotation", track="annotation_assisted"), None) is None


def test_available_before_cutoff_is_required_for_molecular_context():
    item = {"kind": "molecular", "source_evidence_ids": ["source"], "available_at": "2026-01-02T12:00:00Z"}
    assert input_exclusion_reason(item, "2026-01-02T07:00:00-05:00") is None
    assert input_exclusion_reason(item, "2026-01-01T12:00:00Z") == "available_after_planning_cutoff"
    assert input_exclusion_reason(item, None) == "planning_cutoff_unknown"
    assert input_exclusion_reason(dict(item, available_at="2026-01-02T12:00:00"), "2026-01-02T12:00:00Z") == "invalid_or_naive_timestamp"
    assert input_exclusion_reason(dict(item, measurement_at="2026-01-03T12:00:00Z"), "2026-01-04T12:00:00Z") == "availability_precedes_measurement"


def test_unavailable_field_in_optimizer_is_a_registry_error(registry, tmp_path):
    item = {"input_id": "IDH", "kind": "molecular", "source_evidence_ids": ["identity-A"],
            "available_at": None, "used_by_primary_optimizer": True}
    registry["records"] = [record("case", inputs=[item])]
    report = audit_registry(registry, tmp_path)
    assert "UNAVAILABLE_INPUT_USED_BY_OPTIMIZER" in codes(report)
    assert report["inputs"][0]["exclusion_reason"] == "availability_time_unknown"


def test_registry_is_not_mutated_by_audit(registry, tmp_path):
    registry["records"] = [record("case")]
    original = deepcopy(registry)
    audit_registry(registry, tmp_path)
    assert registry == original


def test_checked_in_registry_has_only_explicit_development_records():
    root = Path(__file__).resolve().parents[1]
    data = json.loads((root / "manifests/cohort_registry.json").read_text())
    assert {r["outer_split"] for r in data["records"]} == {"development"}
    assert len(data["records"]) == 2
    assert all(c["eligible_patient_count"] is None for c in data["collections"])
    assert next(c for c in data["collections"] if c["collection_id"] == "UPENN-GBM")["role"] == "reserved_final_candidate"
