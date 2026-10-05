#!/usr/bin/env python3
"""Prepare the remaining five TRAIN cases under the unchanged PAT05 task.

The input registry is existing source/QC receipts, not newly acquired images.
Preparation preserves support conflicts and performs no action or optimization.
Callers must durably save the returned record before comparing methods and count
preparation in their supervised wall/memory budget. The PAT05 loader is untouched.
"""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from preflight_real_spatial_policy import sha256
from resectionlab.real_patient_learning import (
    COHORT_SHA256, read_development_cohort, require_development_role,
)

VERSION = "remaining-real-training-preparation-v1"
SUBJECTS = ("sub-PAT16", "sub-PAT20", "sub-PAT22", "sub-PAT25", "sub-PAT28")
COHORT_PATH = "manifests/experiments/btc-spatial-development-cohort-v1.json"
COMMON_PATH = "artifacts/pat05-real-geometric-learning-v1/declaration-input.json"
ACCESS_RULE = "annotation-centroid-nearest-cell-six-axis-shortest-exit-radius6-v1"
RECEIPT_PATHS = (
    "docs/brain-extraction-pat16-pat20-bundle-integration.json",
    "artifacts/brain-extraction/BTC-spatial-main-v1-integration/attempt-02/integration-record.json",
    "artifacts/brain-extraction/PAT28-structural-evidence-bridge-v1.json",
    "artifacts/electron-btc-inventory-validation.json",
)


def _json(path):
    return json.loads((ROOT / path).read_text())


def binding_hash(binding):
    encoded = json.dumps(binding, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def known_members():
    """Metadata only: frozen source byte and proposal identities for all five."""
    members = {}
    for index in (0, 1):
        for row in _json(RECEIPT_PATHS[index])["cases"]:
            subject = row["subject"]
            if subject not in SUBJECTS:
                continue
            evidence = (next(item for item in row["evidence"] if item["variant"] == "main")
                        if index == 0 else row)
            members[subject] = {
                "subject": subject, "role": "TRAIN", "case_bundle": row["derived_case"],
                "case_bundle_sha256": row["derived_case_sha256"],
                "case_semantic_hash": row["derived_semantic_hash"],
                "planning_hash": row["planning_hash"], "evidence_id": evidence["evidence_id"],
                "evidence_hash": evidence["evidence_hash"],
            }
    source, bundle = _json(RECEIPT_PATHS[2]), _json(RECEIPT_PATHS[3])
    evidence = next(item for item in source["structural_evidence"]
                    if item["evidenceId"].startswith("synthstrip_main_"))
    members["sub-PAT28"] = {
        "subject": "sub-PAT28", "role": "TRAIN", "case_bundle": bundle["case_bundle"],
        "case_bundle_sha256": bundle["case_bundle_sha256"],
        "case_semantic_hash": source["case_hash"], "planning_hash": source["planning_hash"],
        "evidence_id": evidence["evidenceId"], "evidence_hash": evidence["evidenceHash"],
    }
    if set(members) != set(SUBJECTS):
        raise ValueError("Every predeclared TRAIN case must remain in the input registry")
    for subject, member in members.items():
        if member["case_bundle"] != f"outputs/cases/BTC-{subject}-structural-evidence.ressectionlab":
            raise ValueError("Unexpected bundle path in retained preparation receipt")
    return members


def common_task_definition():
    """Use final executed PAT05 geometry, not its obsolete initial 32-cube profile."""
    original = _json(COMMON_PATH)
    if (original["track"] != "annotation_assisted" or original["settings"]["max_steps"] != 3
            or original["adapter_options"]["crop_shape"] != [64, 64, 64]
            or original["adapter_options"]["proposal_mode"] != "nominal_cavity_v1"):
        raise ValueError("Expected the final PAT05 annotation-assisted native task")
    return {"definition_path": COMMON_PATH, "definition_sha256": sha256(ROOT / COMMON_PATH),
            "track": original["track"], "max_steps": original["settings"]["max_steps"],
            **{key: original[key] for key in ("tools", "objective", "adapter_options")}}


def preparation_inputs():
    """Small prospective input closure for the comparison runner's declaration."""
    read_development_cohort(ROOT / COHORT_PATH)
    return {"version": VERSION, "subjects": list(SUBJECTS), "cohort_sha256": COHORT_SHA256,
            "members": known_members(), "common_task": common_task_definition(),
            "access_rule": ACCESS_RULE,
            "receipt_sha256": {path: sha256(ROOT / path) for path in RECEIPT_PATHS},
            "preparation_script_sha256": sha256(__file__)}


def derive_access(target, support, affine_ras_mm, subject):
    """PAT05's exact source-index rule, without reward/feasibility selection."""
    import numpy as np
    from resectionlab.geometry import AccessWindow

    target, support = np.asarray(target), np.asarray(support)
    affine = np.asarray(affine_ras_mm, dtype=float)
    if (target.dtype != bool or support.dtype != bool or target.shape != support.shape
            or target.ndim != 3 or not target.any() or not support.any()
            or np.any(target & ~support)):
        raise ValueError("Access needs a nonempty target wholly inside the unchanged support")
    if (affine.shape != (4, 4) or not np.isfinite(affine).all()
            or not np.array_equal(affine[3], [0, 0, 0, 1])
            or abs(np.linalg.det(affine[:3, :3])) < 1e-12):
        raise ValueError("Access needs a finite invertible physical source affine")
    cells = np.argwhere(target)  # Lexicographic order supplies the exact tie break.
    centroid = cells.mean(axis=0)
    representative = cells[np.argmin(np.square(cells - centroid).sum(axis=1))]
    exits = []
    for axis in range(3):
        spacing = float(np.linalg.norm(affine[:3, axis]))
        for sign in (-1, 1):
            point = representative.copy()
            while np.all(point >= 0) and np.all(point < target.shape) and support[tuple(point)]:
                point[axis] += sign
            boundary = point.astype(float)
            boundary[axis] -= .5 * sign
            distance = abs(float(boundary[axis] - representative[axis])) * spacing
            exits.append({"axis": axis, "outward_sign": sign, "distance_mm": distance,
                          "boundary_voxel": boundary.tolist()})
    chosen = min(exits, key=lambda row: (row["distance_mm"], row["axis"], row["outward_sign"]))
    axis, sign = chosen["axis"], chosen["outward_sign"]
    normal = -sign * affine[:3, axis] / np.linalg.norm(affine[:3, axis])
    center = affine[:3, :3] @ chosen["boundary_voxel"] + affine[:3, 3]
    access = AccessWindow(center, normal, 6., subject.removeprefix("sub-") + "-main-provisional-annotation-axis-v1")
    record = {"rule": ACCESS_RULE, "annotation_centroid_voxels": centroid.tolist(),
              "representative_voxel": representative.tolist(),
              "selected_boundary_voxel": chosen["boundary_voxel"],
              "depth_to_annotation_representative_mm": chosen["distance_mm"],
              "six_axis_exit_distances_mm": exits, "cortical_access_permitted": False,
              "selection_uses_reward_or_native_preview": False}
    geometry = {"center_mm": access.center_mm.tolist(), "normal_inward": access.normal_inward.tolist(),
                "radius_mm": access.radius_mm, "window_id": access.window_id}
    return geometry, record


def _decode_case(path):
    from resectionlab.imaging import load_case
    return load_case(path)


def _construct_task(case, member, common, cancelled):
    from resectionlab.geometry import AccessWindow, ToolGeometry
    from resectionlab.native_spatial_task import native_spatial_task_from_case
    return native_spatial_task_from_case(
        case, access=AccessWindow(**member["access"]),
        tools=tuple(ToolGeometry(**row) for row in common["tools"]),
        max_steps=common["max_steps"], track=common["track"], cancelled=cancelled,
        research_support_acknowledgment=member["research_support_acknowledgment"],
        **common["adapter_options"])


def prepare_training_case(subject, *, declared_at, cancelled=None):
    """Return ``(initial_task, record)``; support conflicts return ``(None, record)``.

    Only initial native previews are evaluated. No cut, teacher, optimizer,
    selection case or unopened case is executed here. Other failures raise and
    must remain failures in the supervising runner's five-case denominator.
    """
    started = time.perf_counter()
    if subject not in SUBJECTS:
        raise ValueError("Preparation allows only the five remaining TRAIN subjects")
    cohort = read_development_cohort(ROOT / COHORT_PATH)
    role = require_development_role(cohort, subject, role="TRAIN")
    if not isinstance(declared_at, str) or datetime.fromisoformat(declared_at).utcoffset() is None:
        raise ValueError("A prospectively declared aware timestamp is required")

    def check():
        if cancelled is not None and cancelled():
            raise InterruptedError("Training case preparation cancelled; no partial success")

    check()
    member = known_members()[subject]
    common = common_task_definition()
    path = ROOT / member["case_bundle"]
    actual_digest = sha256(path)
    if actual_digest != member["case_bundle_sha256"]:
        raise ValueError("Prepared source bundle differs from the retained byte identity")
    check()
    case = _decode_case(path)
    collection = case.metadata.get("source_collection", {})
    if (case.case_id != f"BTC-ds001226-{subject}-preop" or case.metadata.get("is_synthetic")
            or case.semantic_hash != member["case_semantic_hash"]
            or case.planning_hash != member["planning_hash"]
            or collection.get("accession") != "ds001226" or collection.get("release") != "5.0.1"
            or collection.get("git_commit") != cohort["source"]["git_commit"]
            or not any(ref.source_id == "structural" and ref.provenance == "observed" for ref in case.source_refs)):
        raise ValueError("Bundle does not match its real preoperative TRAIN source binding")
    evidence = case.structural_evidence.get(member["evidence_id"])
    if evidence is None or evidence.evidence_hash != member["evidence_hash"]:
        raise ValueError("The prospectively fixed main support proposal is absent or changed")
    evidence.assert_matches(case)
    if (evidence.provenance != "estimated" or evidence.review_status != "review_required"
            or evidence.model_sha256 is None or evidence.review is not None):
        raise ValueError("Expected the same explicitly unreviewed model-support scope")
    check()

    import numpy as np
    from resectionlab.core import thaw_json

    target = np.logical_or.reduce(tuple(case.compartments.values()))
    if not target.any():
        raise ValueError("The complete supplied target annotation is empty")
    support = np.asarray(evidence.mask, dtype=bool)
    outside = int(np.count_nonzero(target & ~support))
    count = int(np.count_nonzero(target))
    member = {**member, "case_bundle_sha256": actual_digest}
    acknowledgment = {
        "schema_version": 1, "scope": "hypothetical_tissue_support",
        "purpose": "native_spatial_provisional_research", "case_hash": case.semantic_hash,
        "planning_hash": case.planning_hash, "evidence_id": evidence.evidence_id,
        **{key: getattr(evidence, key) for key in ("evidence_hash", "source_image_hash", "source_frame_hash",
                                                   "mask_hash", "model_sha256", "run_sha256")},
        "declared_by": "RessectionLab user-authorized annotation-assisted research workflow",
        "declared_at": declared_at,
        "rationale": "Use the fixed main estimated envelope and unchanged supplied annotation; retain every support conflict. No anatomical review or cortical approval is claimed.",
        "acknowledge_unreviewed": True, "cortical_access_permitted": False, "clinical_use_permitted": False,
    }
    member["research_support_acknowledgment"] = acknowledgment
    binding = {"subject": subject, "patient_group": role.patient_group, "role": "TRAIN",
               "cohort_sha256": COHORT_SHA256, "common_task": common, "member": member}
    coverage = {"full_target_source_cells": count, "target_outside_support_source_cells": outside,
                "target_inside_support_source_cells": count - outside,
                "reference_or_nominal_clipped": False, "support_expanded": False}
    record = {"version": VERSION, "subject": subject, "role": "TRAIN", "binding": binding,
              "coverage": coverage, "executed_transitions": 0, "optimizer_updates": 0,
              "clinical_deficit_probability": None, "cortical_access_permitted": False}
    task = None
    if outside:
        record.update(status="blocked_support_conflict", failure_code="TARGET_OUTSIDE_SUPPORT",
                      reason="Supplied nominal target conflicts with fixed support; no native inventory constructed.")
    else:
        affine = np.asarray(case.affine)
        if case.frame == "LPS+":
            affine = np.diag([-1., -1., 1., 1.]) @ affine
        elif case.frame != "RAS+":
            raise ValueError("Source frame must be RAS+ or LPS+")
        member["access"], member["access_derivation"] = derive_access(target, support, affine, subject)
        check()
        task = _construct_task(case, member, common, cancelled)
        if asdict(task.reward_spec) != common["objective"]:
            raise ValueError("Executed objective differs from the unchanged PAT05 objective")
        check()
        observation = task.observation()
        inventory = task.candidate_inventory()
        from resectionlab.spatial_policy_diagnostics import spatial_coverage, runtime_proposal_coverage
        native = task.case._native_affine_ras_mm
        coverage["actor"] = spatial_coverage(observation, source_shape=target.shape, source_affine=affine,
                                             nominal_target=task.case.nominal_target, native_affine=native)
        coverage["proposals"] = runtime_proposal_coverage(task.case, inventory, observation, native_affine=native)
        coverage["target_outside_actor_crop_source_cells"] = count - coverage["actor"]["nominal_target_positive_voxels_in_crop"]
        coverage["full_target_volume_mm3"] = float(count * abs(np.linalg.det(native[:3, :3])))
        coverage["native_geometry_uses_full_source_grid"] = True
        member["expected_native_grid_binding"] = thaw_json(task.case._grid_record)
        binding["decision_model_hash"] = task.decision_model_hash
        binding["native_source_hash"] = task.case.source_hash
        record.update(status="prepared", initial_inventory=inventory,
                      initial_observation_hash=observation.fingerprint,
                      limitations=["Provisional unreviewed support and hypothetical aperture; no cortical approval.",
                                   "Bounded source-column catalog; whole-target reachability unproved.",
                                   "Search sees full permitted annotation; policy sees the fixed crop, whose omissions remain explicit.",
                                   "Function, vessels, mechanics and clinical utility unvalidated."])
    check()
    # The caller-controlled cancellation boundary must not alter source bytes.
    if sha256(path) != actual_digest:
        raise ValueError("Source bundle changed during preparation")
    if task is not None:
        task._assert_frozen()
    record["binding_hash"] = binding_hash(binding)
    record["preparation_seconds"] = time.perf_counter() - started
    return task, record


def load_prepared_training_case(record, *, cancelled=None):
    """Reload a frozen successful binding; fail rather than silently re-prepare it."""
    if (record.get("version") != VERSION or record.get("status") != "prepared"
            or record.get("role") != "TRAIN" or record.get("subject") not in SUBJECTS):
        raise ValueError("Only a successful remaining-TRAIN preparation can be loaded")
    binding = record["binding"]
    if (binding.get("subject") != record["subject"] or binding.get("role") != "TRAIN"
            or binding_hash(binding) != record.get("binding_hash")):
        raise ValueError("Frozen training binding changed")
    # Validate the supplied source/task records against the independent registry
    # before decoding; recomputation below also binds access, grid and model.
    expected = known_members()[record["subject"]]
    if (any(binding["member"].get(key) != value for key, value in expected.items())
            or binding["common_task"] != common_task_definition()):
        raise ValueError("Frozen source/task binding differs from the authoritative input records")
    timestamp = binding["member"]["research_support_acknowledgment"]["declared_at"]
    task, rebuilt = prepare_training_case(record["subject"], declared_at=timestamp, cancelled=cancelled)
    if (rebuilt["status"] != "prepared" or rebuilt["binding"] != binding
            or rebuilt["coverage"] != record["coverage"]
            or rebuilt["initial_inventory"] != record["initial_inventory"]
            or rebuilt["initial_observation_hash"] != record["initial_observation_hash"]):
        raise ValueError("Reconstructed preparation differs from the frozen source/access/grid/coverage binding")
    return task


if __name__ == "__main__":
    # Actual execution is owned by the supervised comparison runner. This entry
    # point only prints the small, image-free input closure for a root checkpoint.
    print(json.dumps(preparation_inputs(), sort_keys=True, indent=2, allow_nan=False))
