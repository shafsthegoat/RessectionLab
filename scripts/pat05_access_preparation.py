"""Canonicalize the existing PAT05 access metadata without changing its task.

Only pinned small JSON records are read. Patient loading and ingress screening
remain outside this adapter and require their own authenticated preparation.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys
from typing import Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import diagnose_training_observation_coverage as coverage
from resectionlab.core import freeze_json
from resectionlab.geometry import AccessWindow
from resectionlab.native_access_preparation import derive_access
from resectionlab.native_spatial_task import NativeSpatialTask

VERSION = "pat05-canonical-access-metadata-v1"
PROFILE_PATH = "artifacts/pat05-real-spatial-profile-v3/nominal64/declaration-input.json"
LEARNING_PATH = "artifacts/pat05-real-geometric-learning-v1/declaration-input.json"
RECEIPT_PATH = coverage.PAT05_GRID_RECEIPT
RECORD_SHA256 = {
    PROFILE_PATH: "5a27d1815a728d05c9d6d69d3f04e7ac9b87edb61c5f7ff752ab9bcb0b00df64",
    LEARNING_PATH: coverage.ANCHOR_SHA256["sub-PAT05"],
    RECEIPT_PATH: coverage.PAT05_GRID_RECEIPT_SHA256,
}
MAX_RECORD_BYTES = 2 * 1024**2
HISTORICAL_DERIVATION_FIELDS = frozenset({"annotation_centroid_voxels", "representative_voxel",
    "selected_boundary_voxel", "depth_to_annotation_representative_mm", "six_axis_exit_distances_mm",
    "procedure", "track", "source_grid_only_no_native_engine_execution", "cortical_access_permitted"})


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


@dataclass(frozen=True)
class PAT05HistoricalRecords:
    """Original immutable bytes, reauthenticated on every use."""
    profile: bytes
    learning: bytes
    receipt: bytes

    def decoded(self):
        values = (self.profile, self.learning, self.receipt)
        result = []
        for path, raw in zip((PROFILE_PATH, LEARNING_PATH, RECEIPT_PATH), values, strict=True):
            if (type(raw) is not bytes or len(raw) > MAX_RECORD_BYTES
                    or hashlib.sha256(raw).hexdigest() != RECORD_SHA256[path]):
                raise ValueError("Pinned PAT05 historical metadata bytes changed: " + path)
            row = json.loads(raw)
            if not isinstance(row, dict):
                raise ValueError("Historical metadata must be a JSON object")
            result.append(row)
        return tuple(result)


def load_pat05_historical_records():
    """Read three fixed small metadata files, never a patient/checkpoint bundle."""
    values = []
    for path in (PROFILE_PATH, LEARNING_PATH, RECEIPT_PATH):
        with (ROOT / path).open("rb") as stream:
            values.append(stream.read(MAX_RECORD_BYTES + 1))
    records = PAT05HistoricalRecords(*values)
    records.decoded()
    return records


def _access_record(access):
    return {"center_mm": access.center_mm.tolist(), "normal_inward": access.normal_inward.tolist(),
            "radius_mm": access.radius_mm, "window_id": access.window_id}


def verify_historical_access(geometry: Mapping, canonical_derivation: Mapping, profile: Mapping):
    """Exact projection onto every field actually saved in the older schema."""
    old = profile["access_derivation"]
    if set(old) != HISTORICAL_DERIVATION_FIELDS:
        raise ValueError("Historical PAT05 derivation fields changed")
    for name in ("annotation_centroid_voxels", "representative_voxel", "selected_boundary_voxel",
                 "depth_to_annotation_representative_mm", "cortical_access_permitted"):
        if _canonical(old[name]) != _canonical(canonical_derivation[name]):
            raise ValueError("Historical PAT05 access derivation differs: " + name)
    projected = [{key: row[key] for key in ("axis", "outward_sign", "distance_mm")}
                 for row in canonical_derivation["six_axis_exit_distances_mm"]]
    if _canonical(projected) != _canonical(old["six_axis_exit_distances_mm"]):
        raise ValueError("Historical PAT05 six ordered exit distance records differ")
    if _canonical(geometry) != _canonical(profile["access"]):
        raise ValueError("Historical PAT05 selected access differs")
    if (old["track"] != "annotation_assisted" or old["source_grid_only_no_native_engine_execution"] is not True
            or not isinstance(old["procedure"], str) or not old["procedure"].strip()):
        raise ValueError("Historical access provenance changed")


def canonical_pat05_access_metadata(task, records: PAT05HistoricalRecords):
    """Authenticate the initial task, rederive, and return copied metadata only.

    The caller must already authenticate the real source bundle and its support
    through the existing PAT05 loader. No loader, task constructor, observation,
    native preview, screen, mutation, policy or reference-target lookup occurs.
    """
    if not isinstance(records, PAT05HistoricalRecords):
        raise TypeError("Pinned raw PAT05 historical records are required")
    profile, original, executed = records.decoded()
    if (profile["subject"] != "sub-PAT05" or profile["study_id"] != "pat05-real-spatial-profile-v3-nominal64"
            or original["member"]["subject"] != "sub-PAT05" or original["member"]["role"] != "TRAIN"
            or executed["patient"] != "sub-PAT05" or executed["role"] != "TRAIN"):
        raise ValueError("Original PAT05 TRAIN metadata identity changed")
    for name in ("case_bundle", "case_bundle_sha256", "case_semantic_hash", "access", "expected_native_grid_binding"):
        if _canonical(profile[name]) != _canonical(original["member"][name]):
            raise ValueError("Profile and executed PAT05 source/access join changed: " + name)
    if not isinstance(task, NativeSpatialTask):
        raise TypeError("An already authenticated initial NativeSpatialTask is required")
    task._assert_frozen()
    if task._steps or task.terminated or task._engine.revision or task._engine.history or task._planning:
        raise ValueError("PAT05 canonical metadata requires an unexecuted original task")
    identity = (task.case.source_hash, task.decision_model_hash, task._state_seal)
    metrics = executed["initial_task_metrics"]
    if identity[:2] != (metrics["source_hash"], executed["decision_model_hash"]):
        raise ValueError("PAT05 task differs from the executed learning source/model")
    binding = coverage.pat05_complete_grid_binding(original)
    if _canonical(binding["complete_grid"]) != _canonical(metrics["native_grid_reconciliation"]):
        raise ValueError("Provided executed receipt differs from the complete grid anchor")
    coverage.verify_pat05_task_binding(task, original, binding)
    source = task.case
    nominal = source.nominal_target
    if nominal is None or not np.isin(nominal, (0., 1.)).all():
        raise ValueError("PAT05 canonical derivation needs the unchanged binary permitted annotation")
    geometry, derived = derive_access(nominal.astype(bool), source.observed_support, source.affine_ras_mm, "sub-PAT05")
    verify_historical_access(geometry, derived, profile)
    if _canonical(_access_record(source.access)) != _canonical(_access_record(AccessWindow(**original["member"]["access"]))):
        raise ValueError("The actual task no longer uses the original selected access")
    coverage.verify_pat05_task_binding(task, original, binding)
    task._assert_frozen()
    if identity != (task.case.source_hash, task.decision_model_hash, task._state_seal):
        raise RuntimeError("PAT05 initial task changed while preparing canonical metadata")
    return freeze_json({"version": VERSION, "subject": "sub-PAT05", "role": "TRAIN",
        "expected_source_hash": identity[0], "expected_model_hash": identity[1],
        "access_derivation": derived, "access": geometry, "complete_native_grid_binding": binding,
        "historical_records": [{"path": path, "sha256": RECORD_SHA256[path]} for path in RECORD_SHA256],
        "historically_verified_fields": ["annotation_centroid_voxels", "representative_voxel",
            "selected_boundary_voxel", "depth_to_annotation_representative_mm", "cortical_access_permitted",
            "six_ordered_axis_sign_distance_records", "selected_access"],
        "current_shared_rule_metadata": ["rule", "selection_uses_reward_or_native_preview"],
        "newly_rederived_fields": ["six_axis_exit_distances_mm[*].boundary_voxel"],
        "historical_provenance": {key: profile["access_derivation"][key] for key in
            ("procedure", "track", "source_grid_only_no_native_engine_execution")},
        "historical_records_rewritten": False, "new_access_selected": False,
        "complete_stroke_certified": False, "clinical_use_permitted": False})
