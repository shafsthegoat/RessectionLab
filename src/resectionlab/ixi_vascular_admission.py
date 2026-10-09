"""Metadata preparation for a future retrospective IXI vascular component task.

This reads no files, constructs no patient arrays and grants no runtime release.
The caller supplies separately reviewed metadata bytes/hashes. A checksum binds
those assertions; it does not authenticate them or perform image/anatomical QC.
Generated controls remain generated. Existing generated planner/evaluator gates
are deliberately unchanged; neither returned contract is a current spec/seal.
"""
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import re

import numpy as np

from resectionlab.core import freeze_json, semantic_digest, thaw_json

VERSION = "ixi-vascular-metadata-preparation-v1"
COHORT_SHA256 = "b27f28c21089e61f6c4899f172f608cd0f81391588777443939196b1700cb127"
METHODS = ("SEARCH", "IL", "RL", "HYBRID")
ROLE_USE = {"TRAIN": {"training_component_check", "frozen_scoring"},
            "SELECT": {"declared_selection", "frozen_scoring"},
            "MEASUREMENT_EVAL": {"frozen_scoring"}}
TASK_KIND = "retrospective_healthy_vascular_corridor"
TARGET_MEANING = "hypothetical_T1_derived_waypoint_not_pathology"
QC_SCOPES = {
    "acquired_T1": "T1_header_anatomy_and_frame_for_component_use",
    "acquired_MRA": "MRA_header_anatomy_and_frame_for_vascular_reference",
    "derived_model_assisted_manually_refined_reference": "derived_annotation_anatomy_and_MRA_grid_link",
    "support": "T1_support_estimate_and_qualified_coverage",
    "task": "hypothetical_T1_corridor_definition_and_tool_coverage",
    "registration": "directed_T1_to_MRA_alignment_and_valid_domain",
    "coverage": "annotation_MRA_registration_domain_intersection",
}
EXECUTION_GAPS = (
    "retrospective_component_planning_spec_and_waypoint_objective_not_implemented",
    "real_person_matched_suite_and_vascular_seal_versions_not_implemented",
    "actual_array_fixity_coverage_and_registration_validation_required_at_load",
)


def _need(condition, reason):
    if not condition:
        raise ValueError(reason)


def _sha(value):
    _need(isinstance(value, str) and re.fullmatch(r"(?:sha256:)?[a-f0-9]{64}", value), "exact_sha256_required")
    return value.removeprefix("sha256:")


def _fields(value, names, reason):
    _need(isinstance(value, Mapping) and set(value) == set(names.split()), reason)
    return value


def _time(value):
    _need(isinstance(value, str), "availability_time_required")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    _need(parsed.tzinfo is not None and parsed.utcoffset() is not None, "timezone_required")
    return parsed


def _grid(value):
    _fields(value, "shape affine_ras_mm frame_sha256", "grid_fields")
    shape, affine = value["shape"], np.asarray(value["affine_ras_mm"], dtype=float)
    _need(isinstance(shape, (list, tuple)) and len(shape) == 3
          and all(type(n) is int and 3 <= n <= 4096 for n in shape)
          and int(np.prod(shape)) <= 32_000_000, "bounded_3d_grid_required")
    _need(affine.shape == (4, 4) and np.isfinite(affine).all()
          and np.array_equal(affine[3], [0, 0, 0, 1]), "RAS_mm_affine_required")
    basis = affine[:3, :3]
    spacing = np.linalg.norm(basis, axis=0)
    _need(np.all(spacing >= .01) and np.all(spacing <= 100)
          and np.allclose((basis / spacing).T @ (basis / spacing), np.eye(3), atol=1e-9, rtol=0),
          "orthogonal_mm_grid_required")
    expected = semantic_digest({"shape": list(shape), "affine_ras_mm": affine.tolist(), "frame": "RAS+"})
    _need(_sha(value["frame_sha256"]) == _sha(expected), "frame_hash_mismatch")
    return value["frame_sha256"]


def _checked(row, domain, scope):
    _need(row.get("evidence_domain") == domain, "generated_evidence_cannot_be_relabelled")
    _need(row.get("qc_status") == "pass", "reviewed_QC_required")
    _need(row.get("qc_scope") == QC_SCOPES[scope], "intended_use_QC_scope_required")
    _sha(row.get("qc_record_sha256"))


def _source(row, *, domain, member, kind):
    _fields(row, "evidence_domain kind member source_file_sha256 array_sha256 grid acquired_at available_at availability_record_sha256 qc_status qc_scope qc_record_sha256",
            "source_receipt_fields")
    _checked(row, domain, kind)
    _need(row["kind"] == kind and row["member"] == member, "source_kind_or_member_mismatch")
    _sha(row["source_file_sha256"])
    _sha(row["array_sha256"])
    _grid(row["grid"])
    if row["acquired_at"] is not None:
        _time(row["acquired_at"])
    if row["available_at"] is not None:
        available = _time(row["available_at"])
        _sha(row["availability_record_sha256"])
        _need(row["acquired_at"] is None or _time(row["acquired_at"]) <= available,
              "availability_precedes_acquisition")
    else:
        _need(row["availability_record_sha256"] is None, "unknown_availability_must_remain_unknown")


def _actor_available(row, cutoff):
    _need(row["available_at"] is not None and _time(row["available_at"]) <= cutoff,
          "actor_unavailable_at_decision")


@dataclass(frozen=True, slots=True)
class PreparedIXIVascularCase:
    """Separate snapshots: only actor_contract is suitable for upstream planning.

    The enclosing object is evaluator-owned. Do not pass it to an actor. It is
    an accidental-misuse boundary, not a Python or OS isolation mechanism.
    """
    actor_contract: Mapping
    evaluator_contract: Mapping
    evidence_sha256: str

    def __post_init__(self):
        object.__setattr__(self, "actor_contract", freeze_json(self.actor_contract))
        object.__setattr__(self, "evaluator_contract", freeze_json(self.evaluator_contract))

    def execution_status(self):
        return {"execution_admitted": False, "patient_payloads_validated": False,
                "required_bridges": list(EXECUTION_GAPS)}


def prepare_ixi_vascular_case(*, cohort_bytes, evidence, reviewed_evidence_sha256, purpose):
    """Join reviewed receipts to frozen roles, preserving retrospective unknowns.

    This prepares one person for a later versioned component task. It does not
    authorize extraction, fitting, selection, actor inference or private loading.
    Role restrictions apply to the declared downstream use; no role is changed.
    """
    _need(type(cohort_bytes) is bytes and len(cohort_bytes) <= 2 * 1024**2
          and hashlib.sha256(cohort_bytes).hexdigest() == COHORT_SHA256, "frozen_IXI_cohort_required")
    cohort = json.loads(cohort_bytes)
    _need(cohort["schema"] == "ixi-component-person-cohort-v1", "cohort_schema")
    value = thaw_json(freeze_json(evidence))
    _need(_sha(semantic_digest(value)) == _sha(reviewed_evidence_sha256), "reviewed_evidence_changed")
    _fields(value, "schema evidence_domain cohort_sha256 person_id role observation_regime decision_cutoff sources support task registration coverage",
            "case_evidence_fields")
    _need(value["schema"] == VERSION and _sha(value["cohort_sha256"]) == COHORT_SHA256, "case_schema_or_cohort")
    domain = value["evidence_domain"]
    _need(domain in {"real_source_receipts", "generated_metadata_control"}, "evidence_domain_required")
    _need(value["observation_regime"] == "retrospective_dataset", "retrospective_scope_required")
    member = [m for m in cohort["members"] if m["person_group"] == value["person_id"]]
    _need(len(member) == 1 and member[0]["role"] == value["role"], "frozen_person_role_mismatch")
    member = member[0]
    _need(purpose in ROLE_USE[value["role"]], "role_forbids_requested_use")
    _need(set(member["raw_files"]) == {"T1", "MRA"} and member["annotation_available"] is True,
          "paired_annotated_person_required_no_reassignment")
    _need(member["vessel_annotation"]["original_MRA_link"] == member["raw_files"]["MRA"]["name"], "annotation_person_link")
    sources = _fields(value["sources"], "T1 MRA vessel_annotation", "exact_source_roles_required")
    for modality in ("T1", "MRA"):
        _source(sources[modality], domain=domain, member=member["raw_files"][modality]["name"], kind="acquired_" + modality)
    _source(sources["vessel_annotation"], domain=domain, member=member["vessel_annotation"]["member"],
            kind="derived_model_assisted_manually_refined_reference")
    t1, mra, annotation = (sources[k] for k in ("T1", "MRA", "vessel_annotation"))
    _need(annotation["acquired_at"] is None, "derived_annotation_is_not_an_acquisition")
    _need(annotation["grid"] == mra["grid"], "annotation_to_MRA_grid_validation_required")
    cutoff = _time(value["decision_cutoff"])
    _actor_available(t1, cutoff)
    support = _fields(value["support"], "evidence_domain kind source_file_sha256 frame_sha256 output_sha256 coverage_sha256 estimator_record_sha256 estimator_model_sha256 project_fit_roles pretrained_exposure available_at availability_record_sha256 qc_status qc_scope qc_record_sha256",
                      "support_receipt_fields")
    _checked(support, domain, "support")
    _need(support["kind"] == "T1_derived_brain_envelope" and support["source_file_sha256"] == t1["source_file_sha256"]
          and support["frame_sha256"] == t1["grid"]["frame_sha256"], "support_must_derive_only_from_actor_T1")
    for key in ("output_sha256", "coverage_sha256", "estimator_record_sha256", "estimator_model_sha256", "availability_record_sha256"):
        _sha(support[key])
    _need(support["project_fit_roles"] in ([], ["TRAIN"])
          and support["pretrained_exposure"] in {"unknown", "known_overlap", "audited_no_overlap"},
          "estimator_fit_role_or_exposure_required")
    _actor_available(support, cutoff)
    _need(_time(support["available_at"]) >= _time(t1["available_at"]), "support_precedes_actor_source")
    task = _fields(value["task"], "evidence_domain kind target_semantics source_file_sha256 support_output_sha256 frame_sha256 definition_sha256 waypoint_count horizon available_at availability_record_sha256 qc_status qc_scope qc_record_sha256",
                   "component_task_fields")
    _checked(task, domain, "task")
    _need(task["kind"] == TASK_KIND and task["target_semantics"] == TARGET_MEANING, "healthy_component_task_required")
    _need(task["source_file_sha256"] == t1["source_file_sha256"] and task["support_output_sha256"] == support["output_sha256"]
          and task["frame_sha256"] == t1["grid"]["frame_sha256"], "task_must_use_only_actor_T1_and_support")
    _need(type(task["waypoint_count"]) is int and 1 <= task["waypoint_count"] <= 2
          and type(task["horizon"]) is int and task["horizon"] == 1, "bounded_waypoint_task_required")
    _sha(task["definition_sha256"])
    _sha(task["availability_record_sha256"])
    _actor_available(task, cutoff)
    _need(_time(task["available_at"]) >= _time(support["available_at"]), "task_precedes_support")
    registration = _fields(value["registration"], "evidence_domain from_source_sha256 to_source_sha256 from_frame_sha256 to_frame_sha256 direction matrix_ras_mm valid_domain_sha256 qc_status qc_scope qc_record_sha256",
                           "registration_receipt_fields")
    _checked(registration, domain, "registration")
    _sha(registration["valid_domain_sha256"])
    _need(registration["direction"] == "T1_RAS_mm_to_MRA_RAS_mm"
          and registration["from_source_sha256"] == t1["source_file_sha256"]
          and registration["to_source_sha256"] == mra["source_file_sha256"]
          and registration["from_frame_sha256"] == t1["grid"]["frame_sha256"]
          and registration["to_frame_sha256"] == mra["grid"]["frame_sha256"], "registration_source_frame_direction")
    matrix = np.asarray(registration["matrix_ras_mm"], dtype=float)
    _need(matrix.shape == (4, 4) and np.isfinite(matrix).all() and np.array_equal(matrix[3], [0, 0, 0, 1])
          and np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-9, rtol=0)
          and np.isclose(np.linalg.det(matrix[:3, :3]), 1., atol=1e-9, rtol=0)
          and np.max(np.abs(matrix[:3, 3])) <= 10000, "reviewed_rigid_transform_required")
    coverage = _fields(value["coverage"], "evidence_domain annotation_source_sha256 mask_sha256 annotation_domain_sha256 mra_valid_domain_sha256 registration_valid_domain_sha256 coverage_sha256 frame_sha256 meaning unlabelled_meaning positive_outside_domain_cells qc_status qc_scope qc_record_sha256",
                       "coverage_receipt_fields")
    _checked(coverage, domain, "coverage")
    _need(coverage["annotation_source_sha256"] == annotation["source_file_sha256"]
          and coverage["mask_sha256"] == annotation["array_sha256"]
          and coverage["frame_sha256"] == annotation["grid"]["frame_sha256"], "coverage_reference_binding")
    for key in ("coverage_sha256", "annotation_domain_sha256", "mra_valid_domain_sha256", "registration_valid_domain_sha256"):
        _sha(coverage[key])
    _need(coverage["registration_valid_domain_sha256"] == registration["valid_domain_sha256"], "registration_coverage_binding")
    _need(coverage["meaning"] == "intersection_of_annotation_MRA_and_registration_domains_not_vessel_completeness"
          and coverage["unlabelled_meaning"] == "unknown_biological_vessel_status"
          and type(coverage["positive_outside_domain_cells"]) is int
          and coverage["positive_outside_domain_cells"] == 0, "qualified_annotation_domain_required")
    actor = {"schema": VERSION, "evidence_domain": domain, "person_id": value["person_id"], "role": value["role"],
        "cohort_sha256": COHORT_SHA256, "purpose": purpose, "observation_regime": value["observation_regime"],
        "decision_cutoff": value["decision_cutoff"], "T1": t1, "support": support, "task": task,
        "method_order": list(METHODS), "preoperative_claim": False, "clinical_use": False}
    private = {"schema": VERSION, "evidence_domain": domain, "actor_contract_sha256": semantic_digest(actor),
        "person_id": value["person_id"], "role": value["role"], "MRA": mra, "vessel_annotation": annotation,
        "registration": registration, "coverage": coverage, "private_until_all_methods_sealed": True,
        "annotation_lineage": "SynthStrip_Frangi_manual_refinement_with_incomplete_sensitivity_and_publisher_subset_dependence",
        "independent_external_site_or_unseen_pretrained_claim": False,
        "biological_vessel_free_claim": False, "clinical_injury_claim": False}
    return PreparedIXIVascularCase(actor, private, reviewed_evidence_sha256)
