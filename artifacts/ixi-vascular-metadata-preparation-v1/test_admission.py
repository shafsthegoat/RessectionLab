"""Generated metadata controls: no image/header/archive/checkpoint bytes opened."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("ixi_admission_candidate", Path(__file__).with_name("ixi_vascular_admission.py"))
a = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = a
SPEC.loader.exec_module(a)

COHORT = (ROOT / "manifests/experiments/ixi-component-person-cohort-v1.json").read_bytes()
PEOPLE = json.loads(COHORT)["members"]
DOMAIN = "generated_metadata_control"


def digest(text):
    return "sha256:" + hashlib.sha256(text.encode()).hexdigest()


def grid(shift=0.):
    affine = np.eye(4)
    affine[0, 3] = shift
    fields = {"shape": [9, 9, 9], "affine_ras_mm": affine.tolist()}
    return {**fields, "frame_sha256": a.semantic_digest({**fields, "frame": "RAS+"})}


def checked(tag):
    scope_key = "acquired_" + tag if tag in ("T1", "MRA") else (
        "derived_model_assisted_manually_refined_reference" if tag == "vessel_annotation" else tag)
    return {"evidence_domain": DOMAIN, "qc_status": "pass", "qc_scope": a.QC_SCOPES[scope_key],
            "qc_record_sha256": digest("GENERATED-QC-" + tag)}


def fixture(role="TRAIN"):
    members = [m for m in PEOPLE if m["role"] == role and m["annotation_available"]]
    member = min(members, key=lambda m: (m["split_sha256"], m["subject"]))
    sources = {}
    for name in ("T1", "MRA", "vessel_annotation"):
        t1 = name == "T1"
        sources[name] = {**checked(name), "kind": "acquired_" + name if name != "vessel_annotation"
            else "derived_model_assisted_manually_refined_reference",
            "member": member["raw_files"][name]["name"] if name != "vessel_annotation" else member["vessel_annotation"]["member"],
            "source_file_sha256": digest("GENERATED-BYTES-" + name), "array_sha256": digest("GENERATED-ARRAY-" + name),
            "grid": grid(0. if t1 else 1.), "acquired_at": None,
            "available_at": "2026-10-09T00:00:00Z" if t1 else None,
            "availability_record_sha256": digest("GENERATED-T1-AVAILABLE") if t1 else None}
    t1, mra, label = (sources[k] for k in ("T1", "MRA", "vessel_annotation"))
    support = {**checked("support"), "kind": "T1_derived_brain_envelope", "source_file_sha256": t1["source_file_sha256"],
        "frame_sha256": t1["grid"]["frame_sha256"], "output_sha256": digest("GENERATED-SUPPORT"),
        "coverage_sha256": digest("GENERATED-SUPPORT-DOMAIN"), "estimator_record_sha256": digest("GENERATED-ESTIMATOR"),
        "estimator_model_sha256": digest("GENERATED-MODEL-NO-WEIGHTS"), "project_fit_roles": ["TRAIN"], "pretrained_exposure": "unknown",
        "available_at": "2026-10-09T00:01:00Z", "availability_record_sha256": digest("GENERATED-SUPPORT-AVAILABLE")}
    task = {**checked("task"), "kind": a.TASK_KIND, "target_semantics": a.TARGET_MEANING,
        "source_file_sha256": t1["source_file_sha256"], "support_output_sha256": support["output_sha256"],
        "frame_sha256": t1["grid"]["frame_sha256"], "definition_sha256": digest("GENERATED-WAYPOINTS-ACCESS-TOOLS-PROCEDURE"),
        "waypoint_count": 2, "horizon": 1, "available_at": "2026-10-09T00:02:00Z",
        "availability_record_sha256": digest("GENERATED-TASK-AVAILABLE")}
    registration = {**checked("registration"), "from_source_sha256": t1["source_file_sha256"], "to_source_sha256": mra["source_file_sha256"],
        "from_frame_sha256": t1["grid"]["frame_sha256"], "to_frame_sha256": mra["grid"]["frame_sha256"],
        "direction": "T1_RAS_mm_to_MRA_RAS_mm", "matrix_ras_mm": np.eye(4).tolist(),
        "valid_domain_sha256": digest("GENERATED-REGISTRATION-DOMAIN")}
    coverage = {**checked("coverage"), "annotation_source_sha256": label["source_file_sha256"], "mask_sha256": label["array_sha256"],
        "annotation_domain_sha256": digest("GENERATED-ANNOTATION-DOMAIN"), "mra_valid_domain_sha256": digest("GENERATED-MRA-DOMAIN"),
        "registration_valid_domain_sha256": registration["valid_domain_sha256"], "coverage_sha256": digest("GENERATED-INTERSECTION-DOMAIN"),
        "frame_sha256": label["grid"]["frame_sha256"],
        "meaning": "intersection_of_annotation_MRA_and_registration_domains_not_vessel_completeness",
        "unlabelled_meaning": "unknown_biological_vessel_status", "positive_outside_domain_cells": 0}
    return {"schema": a.VERSION, "evidence_domain": DOMAIN, "cohort_sha256": a.COHORT_SHA256,
        "person_id": member["person_group"], "role": role, "observation_regime": "retrospective_dataset",
        "decision_cutoff": "2026-10-09T00:03:00Z", "sources": sources, "support": support, "task": task,
        "registration": registration, "coverage": coverage}


def prepare(value=None, purpose="frozen_scoring", **kwargs):
    value = fixture() if value is None else value
    return a.prepare_ixi_vascular_case(cohort_bytes=COHORT, evidence=value,
        reviewed_evidence_sha256=a.semantic_digest(value), purpose=purpose, **kwargs)


def test_retrospective_unknown_acquisition_remains_unknown_without_preoperative_claim():
    prepared = prepare()
    assert prepared.actor_contract["T1"]["acquired_at"] is None
    assert prepared.actor_contract["preoperative_claim"] is False
    assert prepared.evaluator_contract["MRA"]["available_at"] is None
    assert prepared.actor_contract["evidence_domain"] == DOMAIN
    assert prepared.execution_status()["execution_admitted"] is False
    assert prepared.execution_status()["patient_payloads_validated"] is False
    assert prepared.actor_contract["method_order"] == a.METHODS


@pytest.mark.parametrize("role,purpose", [("TRAIN", "training_component_check"), ("SELECT", "declared_selection"),
    ("MEASUREMENT_EVAL", "frozen_scoring")])
def test_roles_support_only_declared_uses_without_reassignment(role, purpose):
    assert prepare(fixture(role), purpose).actor_contract["role"] == role


@pytest.mark.parametrize("role,purpose", [("SELECT", "training_component_check"), ("MEASUREMENT_EVAL", "declared_selection"),
    ("MEASUREMENT_EVAL", "training_component_check"), ("TRAIN", "optimizer_update")])
def test_role_misuse_refuses(role, purpose):
    with pytest.raises(ValueError, match="role_forbids"):
        prepare(fixture(role), purpose)


def test_frozen_cohort_and_person_role_cannot_be_rewritten():
    value = fixture()
    value["role"] = "SELECT"
    with pytest.raises(ValueError, match="frozen_person_role"):
        prepare(value)
    with pytest.raises(ValueError, match="frozen_IXI_cohort"):
        a.prepare_ixi_vascular_case(cohort_bytes=COHORT + b" ", evidence=fixture(),
            reviewed_evidence_sha256=a.semantic_digest(fixture()), purpose="frozen_scoring")


def test_generated_receipts_do_not_become_real_by_changing_envelope_domain():
    value = fixture()
    value["evidence_domain"] = "real_source_receipts"
    with pytest.raises(ValueError, match="generated_evidence_cannot_be_relabelled"):
        prepare(value)


def test_no_private_reference_metadata_in_actor_and_private_swap_preserves_actor():
    value = fixture()
    original = prepare(value)
    private_digest = value["sources"]["MRA"]["source_file_sha256"]
    actor_json = json.dumps(a.thaw_json(original.actor_contract), sort_keys=True)
    assert private_digest not in actor_json and '"MRA"' not in actor_json
    assert '"registration"' not in actor_json and '"coverage"' not in actor_json
    value["sources"]["MRA"]["source_file_sha256"] = digest("GENERATED-SECOND-MRA")
    value["registration"]["to_source_sha256"] = value["sources"]["MRA"]["source_file_sha256"]
    changed = prepare(value)
    assert original.actor_contract == changed.actor_contract
    assert original.evaluator_contract != changed.evaluator_contract


def test_detached_contracts_and_reviewed_evidence_fixity():
    value = fixture()
    before = a.semantic_digest(value)
    prepared = prepare(value)
    value["sources"]["T1"]["grid"]["shape"][0] = 8
    assert prepared.actor_contract["T1"]["grid"]["shape"] == (9, 9, 9)
    with pytest.raises(TypeError):
        prepared.actor_contract["role"] = "SELECT"
    with pytest.raises(ValueError, match="reviewed_evidence_changed"):
        a.prepare_ixi_vascular_case(cohort_bytes=COHORT, evidence=value, reviewed_evidence_sha256=before, purpose="frozen_scoring")


@pytest.mark.parametrize("part", ["support", "task", "registration", "coverage"])
def test_missing_or_unreviewed_prerequisite_refuses(part):
    value = fixture()
    value[part]["qc_status"] = "not_run"
    with pytest.raises(ValueError, match="reviewed_QC"):
        prepare(value)
    del value[part]
    with pytest.raises(ValueError, match="case_evidence_fields"):
        prepare(value)


@pytest.mark.parametrize("part", ["T1", "MRA", "vessel_annotation", "support", "task", "registration", "coverage"])
def test_header_only_or_different_intended_use_QC_never_meets_case_prerequisite(part):
    value = fixture()
    row = value["sources"][part] if part in value["sources"] else value[part]
    for scope in ("header_only", a.QC_SCOPES["task"] if part != "task" else a.QC_SCOPES["support"]):
        row["qc_scope"] = scope
        with pytest.raises(ValueError, match="intended_use_QC_scope"):
            prepare(value)


@pytest.mark.parametrize("part", ["support", "task"])
def test_private_ancestry_cannot_enter_nominal_support_or_target(part):
    value = fixture()
    value[part]["source_file_sha256"] = value["sources"]["MRA"]["source_file_sha256"]
    with pytest.raises(ValueError, match="(support|task)_must"):
        prepare(value)


@pytest.mark.parametrize("part", ["T1", "support", "task"])
def test_actor_availability_unknown_or_after_cutoff_refuses(part):
    value = fixture()
    row = value["sources"]["T1"] if part == "T1" else value[part]
    row["available_at"] = "2026-10-09T00:04:00Z"
    with pytest.raises(ValueError, match="actor_unavailable"):
        prepare(value)
    row["available_at"] = None
    if part == "T1":
        row["availability_record_sha256"] = None
    with pytest.raises(ValueError, match="actor_unavailable"):
        prepare(value)


def test_scan_known_acquisition_and_derived_availability_cannot_run_backward():
    value = fixture()
    value["sources"]["T1"]["acquired_at"] = "2026-10-09T00:01:00Z"
    with pytest.raises(ValueError, match="availability_precedes"):
        prepare(value)
    value = fixture()
    value["task"]["available_at"] = "2026-10-09T00:00:00Z"
    with pytest.raises(ValueError, match="task_precedes"):
        prepare(value)


@pytest.mark.parametrize("mutation,reason", [("direction", "direction"), ("reflection", "rigid"),
    ("scale", "rigid"), ("private_frame", "frame"), ("coverage_domain", "coverage_binding")])
def test_registration_direction_frame_rigidity_and_domain_binding(mutation, reason):
    value = fixture()
    if mutation == "direction":
        value["registration"]["direction"] = "MRA_RAS_mm_to_T1_RAS_mm"
    elif mutation == "reflection":
        value["registration"]["matrix_ras_mm"][0][0] = -1
    elif mutation == "scale":
        value["registration"]["matrix_ras_mm"][0][0] = 2
    elif mutation == "private_frame":
        value["registration"]["to_frame_sha256"] = value["sources"]["T1"]["grid"]["frame_sha256"]
    else:
        value["coverage"]["registration_valid_domain_sha256"] = digest("WRONG-DOMAIN")
    with pytest.raises(ValueError, match=reason):
        prepare(value)


@pytest.mark.parametrize("field,bad", [("positive_outside_domain_cells", 1), ("positive_outside_domain_cells", False),
    ("unlabelled_meaning", "vessel_free"), ("meaning", "complete_vascular_truth")])
def test_coverage_preserves_unknowns_and_rejects_positive_outside_domain(field, bad):
    value = fixture()
    value["coverage"][field] = bad
    with pytest.raises(ValueError, match="qualified_annotation_domain"):
        prepare(value)


@pytest.mark.parametrize("field,bad", [("kind", "whole_tumor_candidate"), ("horizon", True), ("waypoint_count", 0)])
def test_component_task_cannot_masquerade_as_tumor_or_change_horizon(field, bad):
    value = fixture()
    value["task"][field] = bad
    with pytest.raises(ValueError, match="(healthy_component|bounded_waypoint)"):
        prepare(value)


def test_estimator_fit_does_not_use_protected_roles_and_unknown_exposure_is_retained():
    value = fixture("MEASUREMENT_EVAL")
    assert prepare(value).actor_contract["support"]["pretrained_exposure"] == "unknown"
    value["support"]["project_fit_roles"] = ["TRAIN", "SELECT"]
    with pytest.raises(ValueError, match="estimator_fit_role"):
        prepare(value)


def test_preparation_needs_no_files_or_model_loader(monkeypatch):
    import builtins
    import io
    value = fixture()
    def forbidden(*args, **kwargs):
        pytest.fail("No file reader belongs in metadata preparation")
    monkeypatch.setattr(builtins, "open", forbidden)
    monkeypatch.setattr(io, "open", forbidden)
    assert prepare(value).execution_status()["execution_admitted"] is False
