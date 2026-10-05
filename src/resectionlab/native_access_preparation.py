"""Explicit initial-source ingress opt-in; unchanged default access derivation.

No patient I/O, inventory construction, execution, reward ranking or policy use.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
import time
from typing import Mapping

import numpy as np

from .core import array_digest, freeze_json, semantic_digest, thaw_json
from .geometry import AccessWindow
from .native_ingress import IngressCandidate, MAX_SCREENS, screen_axis_accesses
from .native_proposals import NominalCavityProposalConfig
from .native_resection import NativeResectionEngine
from .native_spatial_task import NativeSpatialCase, NativeSpatialTask
from .real_patient_learning import COHORT_SHA256, require_development_role

ACCESS_RULE = "annotation-centroid-nearest-cell-six-axis-shortest-exit-radius6-v1"
PREPARATION_VERSION = "native-initial-screened-access-preparation-v1"


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



def _access_record(access):
    return {"center_mm": access.center_mm.tolist(), "normal_inward": access.normal_inward.tolist(),
            "radius_mm": access.radius_mm, "window_id": access.window_id}


@dataclass(frozen=True)
class NativeAccessPreparation:
    """Selected immutable source plus diagnostic metadata, never a path certificate."""
    selected_case: NativeSpatialCase | None
    receipt: Mapping

    def __post_init__(self):
        object.__setattr__(self, "receipt", freeze_json(self.receipt))

    def to_dict(self):
        return thaw_json(self.receipt)


def _assert_initial(task):
    task._assert_frozen()
    if (task._steps != 0 or task.terminated or task._engine.revision != 0
            or task._engine.history or task._history or task._planning):
        raise ValueError("Screened access preparation needs the original unexecuted source task")
    source = task.case
    if (source.track != "annotation_assisted" or source.proposal_mode != "nominal_cavity_v1"
            or source.proposal_config != NominalCavityProposalConfig() or len(source.tools) != 2):
        raise ValueError("The fixed annotation-assisted 13-entry/two-tool/three-family contract is required")
    if task._inventory is None or task._proposal_batch is None:
        raise ValueError("The unchanged original inventory must already have been prepared")
    if (len(task._proposal_batch.ledger) != 78
            or any(slot.reason == "CANDIDATE_CAP" for slot in task._proposal_batch.ledger)):
        raise ValueError("Original preparation needs its complete 78-slot catalog")
    source._nominal_proposer.validate_batch(task._proposal_batch, task._engine)


def _existing_exits(source, subject, geometry, derivation):
    """Convert the authenticated existing six boundaries into physical windows."""
    matrix = source.affine_ras_mm
    result = []
    for index, row in enumerate(derivation["six_axis_exit_distances_mm"]):
        axis, sign = row["axis"], row["outward_sign"]
        original = row["boundary_voxel"] == derivation["selected_boundary_voxel"]
        if original:
            access = source.access
        else:
            access = AccessWindow(matrix[:3, :3] @ row["boundary_voxel"] + matrix[:3, 3],
                -sign * matrix[:3, axis] / np.linalg.norm(matrix[:3, axis]), 6.,
                f"{subject.removeprefix('sub-')}-existing-exit-axis{axis}-sign{sign:+d}-v1")
        result.append({"exit_index": index, **row, "selected_original": original,
                       "access": _access_record(access)})
    if _access_record(source.access) != _access_record(AccessWindow(**geometry)):
        raise ValueError("Original task access differs from the unchanged shortest-exit rule")
    return result


def prepare_native_access(task, *, subject, cohort_json, expected_source_hash,
                          expected_model_hash, access_derivation, cancelled=None):
    """Opt in after original preparation; return no task, preview or fallback.

    Expected hashes and derivation must come from the upstream subject-specific
    preparation record. The task has no patient identity field: this function
    validates the declared TRAIN role and supplied source join, not biological
    identity. It does not read a cohort file, patient bundle or evaluator target.
    """
    started = time.perf_counter()
    if type(cohort_json) is not bytes or hashlib.sha256(cohort_json).hexdigest() != COHORT_SHA256:
        raise ValueError("Exact pinned development cohort bytes are required")
    role = require_development_role(json.loads(cohort_json), subject, role="TRAIN")
    if not isinstance(task, NativeSpatialTask):
        raise TypeError("An already validated initial NativeSpatialTask is required")
    _assert_initial(task)
    source = task.case
    if source.source_hash != expected_source_hash or task.decision_model_hash != expected_model_hash:
        raise ValueError("Original source/model differs from the upstream subject-specific preparation")
    declared = freeze_json(access_derivation)
    nominal = source.nominal_target
    if nominal is None or not np.isin(nominal, (0., 1.)).all():
        raise ValueError("The unchanged access rule needs a binary permitted nominal annotation")
    geometry, derivation = derive_access(nominal.astype(bool), source.observed_support,
                                         source.affine_ras_mm, subject)
    if semantic_digest(declared) != semantic_digest(derivation):
        raise ValueError("Saved access derivation no longer matches the permitted source arrays")
    exits = _existing_exits(source, subject, geometry, derivation)
    original_identity = (source.source_hash, task.decision_model_hash, task._state_seal)
    report = {"version": PREPARATION_VERSION, "status": "preparing", "subject": subject,
        "patient_group": role.patient_group, "role": role.role, "cohort_sha256": COHORT_SHA256,
        "original_source_hash": source.source_hash, "original_model_hash": task.decision_model_hash,
        "original_state_hash": task._engine.state_hash, "original_access": _access_record(source.access),
        "reward": asdict(task.reward_spec), "horizon": task.max_steps,
        "access_derivation": derivation, "derived_exits": exits,
        "original_frame_hash": array_digest(source.affine_ras_mm),
        "native_frame_hash": array_digest(source._native_affine_ras_mm),
        "original_preparation": {"already_completed_by_caller": True, "declared_slots": 78,
            "emitted_proposals": len(task._proposal_batch.proposals), "elapsed_seconds": None,
            "cost_scope": "prior caller preparation; never counted as zero or waived"},
        "native_preview_calls": 0, "selected_inventory_constructed": False,
        "selected_exit": None, "selected_source_hash": None, "screening": None,
        "complete_stroke_certified": False, "clinical_use_permitted": False,
        "selection": "any_entry_then_distance_axis_outward_sign", "automatic_fallback": False}
    report["validation_seconds"] = time.perf_counter() - started
    setup_started = time.perf_counter()
    selected_source = None

    def check():
        if cancelled is not None and cancelled():
            raise InterruptedError("Access preparation cancelled")

    try:
        candidates, sources = [], {}
        for row in exits:
            check()
            key = row["axis"], row["outward_sign"]
            derived = source if row["selected_original"] else replace(source, access=AccessWindow(**row["access"]))
            derived.assert_intact()
            sources[key] = derived
            candidates.append(IngressCandidate(*key, row["distance_mm"],
                NativeResectionEngine(derived._native_config), derived._nominal_proposer))
        report["candidate_setup_seconds"] = time.perf_counter() - setup_started
        screen_started = time.perf_counter()
        screening = screen_axis_accesses(candidates, cancelled=cancelled)
        report["screening_seconds"] = time.perf_counter() - screen_started
        report["screening"] = screening.to_dict()
        check()
        _assert_initial(task)
        if original_identity != (source.source_hash, task.decision_model_hash, task._state_seal):
            raise RuntimeError("Original task changed during access preparation")
        if not screening.complete or screening.screen_count > MAX_SCREENS:
            report["status"] = "interrupted" if screening.status == "interrupted" else "failed"
        elif screening.selected_exit is None:
            report["status"] = "no_ingress_admissible_access"
        else:
            selected_source = sources[screening.selected_exit]
            selected_source.assert_intact()
            report.update(status="selected", selected_exit=list(screening.selected_exit),
                selected_source_hash=selected_source.source_hash,
                selected_access=_access_record(selected_source.access))
    except InterruptedError as error:
        report.update(status="interrupted", error=str(error))
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
    report["elapsed_seconds"] = time.perf_counter() - started
    report["timing_scope"] = "function entry through result metadata preparation; final immutable copy excluded"
    if report["status"] != "selected":
        selected_source = None
        report.update(selected_exit=None, selected_source_hash=None)
    return NativeAccessPreparation(selected_source, report)
