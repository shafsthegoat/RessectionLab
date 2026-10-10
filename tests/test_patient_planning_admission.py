"""Small generated arrays exercise admission; no patient payload is read."""
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from resectionlab.core import array_digest, semantic_digest
from resectionlab.geometry import AccessWindow, ToolGeometry
from resectionlab.native_spatial_task import NativeSpatialCase
from resectionlab.patient_planning_admission import (
    VERSION, COHORT_SHA256, QC_SCOPE, make_patient_planning_task)


def fixture_contract(subject="ReMIND-008", role="TRAIN"):
    # Patient IDs exercise only the frozen metadata join; these are fabricated
    # software arrays, not generated replacements for any person's anatomy.
    root = next(p for p in Path(__file__).resolve().parents if (p / "manifests").is_dir())
    cohort = (root / "manifests/experiments/remind-component-cohort-v1.json").read_bytes()
    support = np.zeros((7, 7, 7), bool)
    support[3, 3, 1:5] = True
    target = np.zeros(support.shape, np.float32)
    target[3, 3, 3] = 1
    case = NativeSpatialCase(support.astype(np.float32), support, target, np.eye(4),
        AccessWindow((3., 3., .5), (0., 0., 1.), 2.5, "generated-interface-only"),
        (ToolGeometry("generated-tool", .6, .2, 6., 30., .5),), track="synthetic_scan",
        support_source_kind="derived_from_scan", support_derivation="generated support interface control",
        nominal_target=target, target_source_kind="derived_from_scan",
        target_derivation="generated target interface control", crop_shape=(7, 7, 7))
    source = {"version": VERSION, "evidence_domain": "generated_interface_control",
        "subject": subject, "patient_group": "ReMIND:"+subject.split("-")[1],
        "cohort_sha256": COHORT_SHA256, "t1_source_sha256": "1"*64,
        "support_source_sha256": "2"*64, "target_source_sha256": "3"*64,
        "image_array_hash": array_digest(case.structural_intensity),
        "support_array_hash": array_digest(case.observed_support),
        "target_array_hash": array_digest(case.nominal_target),
        "affine_array_hash": array_digest(case.affine_ras_mm), "source_hash": case.source_hash,
        "availability_basis": "retrospective_annotation_assisted_source_preop",
        "acquired_at": None, "annotation_available_at": None,
        "support_semantics": "generated_support_interface_control",
        "target_semantics": "generated_target_interface_control"}
    qc = {"version": VERSION, "evidence_domain": source["evidence_domain"], "subject": subject,
        "public_source_binding_hash": semantic_digest(source), "status": "pass", "scope": QC_SCOPE,
        "source_linkage_checked": True, "frame_geometry_checked": True, "coverage_checked": True,
        "annotation_meaning_checked": True, "public_support_assumption": True,
        "native_domain_fully_covered": True, "hypothetical_access_assumption": True,
        "evidence_record_sha256": "4"*64}
    protocol = {"version": VERSION, "scope": "patient_native_planning_experiment",
        "subject": subject, "role": role, "evidence_domain": source["evidence_domain"],
        "public_source_binding_hash": semantic_digest(source), "qc_receipt_hash": semantic_digest(qc),
        "max_steps": 2, "max_optimizer_updates": 2 if role == "TRAIN" else 0,
        "search": {"max_calls": 64, "beam_width": 4, "seconds": 2},
        "max_native_previews": 256, "max_policy_forwards": 16,
        "worker_seconds": 10, "memory_bytes": 256*1024**2, "threads": 1,
        "initialization": "fresh_seeded_shared_initialization", "private_reference_used": False,
        "clinical_claim": False, "split_changes": False, "runtime_release_sha256": "5"*64,
        "learning_protocol_hash": "sha256:"+"6"*64}
    return case, dict(cohort_bytes=cohort, source_binding=source, qc_receipt=qc, protocol=protocol)


def rebind(args):
    args["qc_receipt"]["public_source_binding_hash"] = semantic_digest(args["source_binding"])
    args["protocol"]["public_source_binding_hash"] = semantic_digest(args["source_binding"])
    args["protocol"]["qc_receipt_hash"] = semantic_digest(args["qc_receipt"])


def test_generated_public_context_native_stop_and_clone():
    case, args = fixture_contract()
    task, context = make_patient_planning_task(case, **args)
    context.require_training()
    context.require_task(task)
    context.require_task(task.planning_clone())
    context.require_observations([task.observation()])
    assert context.record()["real_patient_count"] == 0
    assert context.record()["ventricular_evidence"] == "unavailable_to_actor_and_search"
    before = task._engine.state_hash
    result = task.step("STOP")
    assert result.terminated and task._engine.state_hash == before and result.reward == 0
    context.require_task(task)


def test_select_planning_context_never_gradient_admission():
    case, args = fixture_contract("ReMIND-013", "SELECT")
    task, context = make_patient_planning_task(case, **args)
    context.require_task(task)
    with pytest.raises(ValueError, match="TRAIN_gradient"):
        context.require_training()
    args["protocol"]["max_optimizer_updates"] = 1
    with pytest.raises(ValueError, match="role_and_budget"):
        make_patient_planning_task(case, **args)


@pytest.mark.parametrize("subject,role", [("ReMIND-067", "TRAIN"), ("ReMIND-013", "TRAIN"), ("ReMIND-001", "TRAIN")])
def test_caller_cannot_relabel_protected_or_development_roles(subject, role):
    case, args = fixture_contract(subject, role)
    with pytest.raises(ValueError, match="frozen_TRAIN_or_SELECT"):
        make_patient_planning_task(case, **args)


@pytest.mark.parametrize("mutation", ["cohort", "header_only", "coverage", "uncovered_domain", "private_field", "private_reward", "fake_time", "generated_relabel", "bool_budget"])
def test_invalid_receipts_refuse_without_anatomical_execution(mutation):
    case, args = fixture_contract()
    if mutation == "cohort": args["cohort_bytes"] += b" "
    if mutation == "header_only": args["qc_receipt"]["scope"] = "header_only"
    if mutation == "coverage": args["qc_receipt"]["coverage_checked"] = False
    if mutation == "uncovered_domain": args["qc_receipt"]["native_domain_fully_covered"] = False
    if mutation == "private_field": args["source_binding"]["private_ventricular_hash"] = "7"*64
    if mutation == "private_reward": args["protocol"]["private_reference_used"] = True
    if mutation == "fake_time": args["source_binding"]["annotation_available_at"] = "2026-01-01T00:00:00Z"
    if mutation == "generated_relabel":
        for key in ("source_binding", "qc_receipt", "protocol"):
            args[key]["evidence_domain"] = "acquired_patient"
    if mutation == "bool_budget": args["protocol"]["search"]["max_calls"] = True
    rebind(args)
    with pytest.raises(ValueError):
        make_patient_planning_task(case, **args)


def test_private_reference_cannot_enter_existing_native_target_slot():
    case, args = fixture_contract()
    other = np.zeros(case.nominal_target.shape, np.float32)
    changed = replace(case, reference_target=other)
    assert changed.source_hash == case.source_hash  # inherited separation
    with pytest.raises(ValueError, match="public_source_target"):
        make_patient_planning_task(changed, **args)


def test_detached_observation_from_other_public_goal_is_refused():
    case, args = fixture_contract()
    task, context = make_patient_planning_task(case, **args)
    target = case.nominal_target.copy()
    target[3, 3, 3], target[3, 3, 2] = 0, 1
    changed = replace(case, reference_target=target, nominal_target=target)
    from resectionlab.native_spatial_task import NativeSpatialTask
    with pytest.raises(ValueError, match="observation_source"):
        context.require_observations([NativeSpatialTask(changed, max_steps=2).observation()])


def test_receipt_snapshot_not_changed_by_caller_mutation():
    case, args = fixture_contract()
    _, context = make_patient_planning_task(case, **args)
    fingerprint = context.fingerprint
    args["source_binding"]["target_source_sha256"] = "8"*64
    args["protocol"]["max_optimizer_updates"] = 999
    assert context.fingerprint == fingerprint and context.record()["max_optimizer_updates"] == 2


def test_same_source_tag_cannot_hide_changed_static_target_bytes():
    case, args = fixture_contract()
    task, context = make_patient_planning_task(case, **args)
    observation = task.observation()
    images = observation.image_channels.copy()
    images[2, 3, 3, 3], images[2, 3, 3, 2] = 0., 1.
    changed = replace(observation, image_channels=images)
    assert changed.source_id == observation.source_id
    with pytest.raises(ValueError, match="observation_source"):
        context.require_observations([changed])
