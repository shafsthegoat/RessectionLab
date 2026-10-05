"""Population functional priors, explicit registration, and alignment review.

Import validates source bytes and spatial headers. Resampling an atlas never
personalizes its functional evidence: every output remains a population prior.
Registration acceptance records a user's alignment review, not clinical validity.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
from scipy.ndimage import affine_transform

from .core import SourceRef, array_digest, freeze_json, immutable_array, semantic_digest, thaw_json
from .imaging import file_sha256, inspect_nifti


TEMPLATE_FRAME = "FSL_MNI152_RAS_mm"
COMPONENTS = frozenset({"motor", "phonology", "semantics", "speech_articulation"})
MAP_KINDS = frozenset({"functional_concordance", "structural_mask"})


def _affine(matrix: Any, name: str) -> np.ndarray:
    value = np.asarray(matrix, dtype=float)
    if (value.shape != (4, 4) or not np.isfinite(value).all()
            or not np.allclose(value[3], [0, 0, 0, 1])
            or abs(np.linalg.det(value[:3, :3])) < 1e-12):
        raise ValueError(f"{name} must be a finite invertible physical affine")
    return value


@dataclass(frozen=True)
class FunctionalPrior:
    map_id: str
    component: str
    map_kind: str
    data: np.ndarray
    affine_ras_mm: np.ndarray
    source: SourceRef
    header_qc: Mapping[str, Any]
    template_frame: str = TEMPLATE_FRAME

    def __post_init__(self) -> None:
        if not self.map_id or self.component not in COMPONENTS or self.map_kind not in MAP_KINDS:
            raise ValueError("Unknown functional component or map kind")
        if self.source.provenance != "prior" or self.template_frame != TEMPLATE_FRAME:
            raise ValueError("Functional atlas evidence must retain its population/template provenance")
        data = np.asarray(self.data)
        if (data.ndim != 3 or 0 in data.shape or not np.isfinite(data).all()
                or np.any(data < 0) or np.any(data > 1)):
            raise ValueError("Functional priors require finite three-dimensional values in [0,1]")
        if self.map_kind == "structural_mask" and not np.all((data == 0) | (data == 1)):
            raise ValueError("Structural masks must be binary")
        object.__setattr__(self, "data", immutable_array(data, np.float32))
        object.__setattr__(self, "affine_ras_mm", immutable_array(_affine(self.affine_ras_mm, "source affine")))
        object.__setattr__(self, "header_qc", freeze_json(self.header_qc))

    @property
    def semantic_hash(self) -> str:
        return semantic_digest({"map_id": self.map_id, "component": self.component,
                                "map_kind": self.map_kind, "data": array_digest(self.data),
                                "affine": self.affine_ras_mm.tolist(),
                                "source": self.source.to_dict(), "template_frame": self.template_frame})

    def inventory_item(self) -> dict[str, Any]:
        return {"map_id": self.map_id, "title": self.component.replace("_", " ").title(),
                "map_kind": self.map_kind, "qc_state": "unregistered_prior",
                "reason": "Population prior; patient alignment has not been reviewed",
                "source": self.source.to_dict(), "template_frame": self.template_frame,
                "shape": list(self.data.shape), "affine_ras_mm": self.affine_ras_mm.tolist(),
                "header_qc": thaw_json(self.header_qc),
                "patient_specific": False, "clinical_deficit_probability": None,
                "overlay_enabled": False}


def load_functional_prior(path: str | Path, specification: Mapping[str, Any]) -> FunctionalPrior:
    """Load one hash-pinned release member, respecting active qform/sform semantics."""
    actual_hash = file_sha256(path)
    if actual_hash != specification["sha256"].removeprefix("sha256:"):
        raise ValueError("PRIOR_HASH_MISMATCH: source bytes differ from the pinned manifest")
    qc = inspect_nifti(path)
    source = SourceRef(
        source_id=f"zenodo:{specification['record_id']}:{specification['archive_member']}",
        uri=f"https://zenodo.org/records/{specification['record_id']}", sha256=actual_hash,
        license=specification["license"], native_frame=specification["template_frame"], provenance="prior")
    return FunctionalPrior(specification["map_id"], specification["component"],
                           specification["map_kind"], nib.load(str(path)).get_fdata(dtype=np.float32),
                           np.asarray(qc["affine_ras_mm"]), source, qc,
                           specification["template_frame"])


def load_prior_collection(cache_dir: str | Path, manifest_path: str | Path) -> dict[str, FunctionalPrior]:
    """Load the seven inspectable motor/language maps from a local verified cache."""
    manifest = json.loads(Path(manifest_path).read_text())
    if manifest.get("schema_version") != 1 or manifest.get("evidence_type") != "prior":
        raise ValueError("Unsupported functional prior manifest")
    root = Path(cache_dir).resolve()
    result = {}
    for specification in manifest["maps"]:
        path = (root / specification["cache_file"]).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Prior cache path escapes its declared directory")
        prior = load_functional_prior(path, specification)
        if prior.map_id in result:
            raise ValueError("Duplicate functional map identifier")
        result[prior.map_id] = prior
    return result


@dataclass(frozen=True)
class RegisteredPrior:
    """Candidate patient-space overlay; alignment review gates planning use."""
    prior: FunctionalPrior
    case_hash: str
    data: np.ndarray
    sampling_coverage: np.ndarray
    affine_ras_mm: np.ndarray
    mni_ras_to_patient_ras_mm: np.ndarray
    registration_method: str
    registration_version: str
    qc_state: str = "alignment_review_required"
    review: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if not self.case_hash or not self.registration_method or not self.registration_version:
            raise ValueError("Case hash, registration method, and version are required")
        if self.qc_state not in {"alignment_review_required", "alignment_accepted", "alignment_rejected"}:
            raise ValueError("Unknown alignment review state")
        data, coverage = np.asarray(self.data), np.asarray(self.sampling_coverage, bool)
        if (data.ndim != 3 or 0 in data.shape or data.shape != coverage.shape
                or not np.isfinite(data).all() or np.any(data < 0) or np.any(data > 1)):
            raise ValueError("Resampled evidence and sampling coverage must share a finite 3D grid")
        object.__setattr__(self, "data", immutable_array(data, np.float32))
        object.__setattr__(self, "sampling_coverage", immutable_array(coverage, bool))
        object.__setattr__(self, "affine_ras_mm", immutable_array(_affine(self.affine_ras_mm, "target affine")))
        transform = _affine(self.mni_ras_to_patient_ras_mm, "MNI-to-patient transform")
        if np.linalg.det(transform[:3, :3]) <= 0:
            raise ValueError("LEFT_RIGHT_REFLECTION: physical registration may not reflect anatomy")
        object.__setattr__(self, "mni_ras_to_patient_ras_mm", immutable_array(transform))
        object.__setattr__(self, "review", None if self.review is None else freeze_json(self.review))
        if self.qc_state != "alignment_review_required":
            if self.review is None or self.review.get("registration_hash") != self.registration_hash:
                raise ValueError("Alignment review is absent or stale for these source/transform/case bytes")

    @property
    def registration_hash(self) -> str:
        return semantic_digest({"prior": self.prior.semantic_hash, "case": self.case_hash,
                                "affine": self.affine_ras_mm.tolist(), "shape": self.data.shape,
                                "transform": self.mni_ras_to_patient_ras_mm.tolist(),
                                "resampled_data": array_digest(self.data),
                                "coverage": array_digest(self.sampling_coverage),
                                "method": self.registration_method, "version": self.registration_version})

    def require_current(self, case_hash: str, *, require_accepted: bool = True) -> None:
        if case_hash != self.case_hash:
            raise ValueError("STALE_FUNCTIONAL_EVIDENCE: case inputs changed; repeat registration/review")
        if require_accepted and self.qc_state != "alignment_accepted":
            raise ValueError("FUNCTIONAL_ALIGNMENT_UNREVIEWED: inspect the overlay and record alignment review")

    def inventory_item(self) -> dict[str, Any]:
        return {**self.prior.inventory_item(), "qc_state": self.qc_state,
                "reason": "Population prior; " + self.qc_state.replace("_", " "),
                "case_hash": self.case_hash, "registration_hash": self.registration_hash,
                "shape": list(self.data.shape), "affine_ras_mm": self.affine_ras_mm.tolist(),
                "mni_ras_to_patient_ras_mm": self.mni_ras_to_patient_ras_mm.tolist(),
                "registration_method": self.registration_method,
                "registration_version": self.registration_version,
                "interpolation": "nearest_neighbor" if self.prior.map_kind == "structural_mask" else "linear",
                "overlay_enabled": self.qc_state == "alignment_accepted",
                "sampling_coverage_fraction": float(self.sampling_coverage.mean()),
                "coverage_meaning": "atlas field of view, not patient functional coverage",
                "review": None if self.review is None else thaw_json(self.review)}


def register_prior(prior: FunctionalPrior, *, target_shape: tuple[int, int, int],
                   target_affine: Any, case_hash: str, mni_ras_to_patient_ras_mm: Any,
                   method: str, version: str, target_frame: str = "RAS+") -> RegisteredPrior:
    """Resample using an explicitly supplied physical MNI-RAS -> patient-RAS map.

    This applies a transform; it does not estimate or validate registration.
    `sampling_coverage` records where the atlas was sampled. Zero atlas intensity
    inside that field of view is never evidence of absent patient function.
    """
    if len(target_shape) != 3 or any(not isinstance(n, (int, np.integer)) or isinstance(n, bool) or n <= 0
                                      for n in target_shape):
        raise ValueError("Target shape must contain three positive integer dimensions")
    target = _affine(target_affine, "target affine")
    if target_frame == "LPS+":
        target = np.diag([-1., -1., 1., 1.]) @ target
    elif target_frame != "RAS+":
        raise ValueError("Target frame must explicitly be RAS+ or LPS+")
    transform = _affine(mni_ras_to_patient_ras_mm, "MNI-to-patient transform")
    if np.linalg.det(transform[:3, :3]) <= 0:
        raise ValueError("LEFT_RIGHT_REFLECTION: physical registration may not reflect anatomy")
    pull = np.linalg.inv(prior.affine_ras_mm) @ np.linalg.inv(transform) @ target
    order = 0 if prior.map_kind == "structural_mask" else 1
    values = affine_transform(prior.data, pull[:3, :3], pull[:3, 3], output_shape=target_shape,
                              order=order, mode="constant", cval=0., prefilter=False)
    coverage = affine_transform(np.ones(prior.data.shape, dtype=np.uint8), pull[:3, :3],
                                pull[:3, 3], output_shape=target_shape, order=0,
                                mode="constant", cval=0, prefilter=False).astype(bool)
    return RegisteredPrior(prior, case_hash, values, coverage, target, transform, method, version)


def review_alignment(evidence: RegisteredPrior, *, case_hash: str, accepted: bool,
                     reviewer: str, notes: str, reviewed_at: datetime) -> RegisteredPrior:
    """Record explicit visual alignment review tied to immutable registration inputs."""
    evidence.require_current(case_hash, require_accepted=False)
    if not isinstance(accepted, bool) or not reviewer.strip() or not notes.strip():
        raise ValueError("An explicit decision, reviewer, and alignment notes are required")
    if reviewed_at.tzinfo is None or reviewed_at.utcoffset() is None:
        raise ValueError("Alignment review timestamp must include a timezone")
    if accepted and not evidence.sampling_coverage.any():
        raise ValueError("NO_ATLAS_OVERLAP: this registration cannot be accepted")
    review = {"registration_hash": evidence.registration_hash, "reviewer": reviewer,
              "notes": notes, "reviewed_at": reviewed_at.astimezone(timezone.utc).isoformat(),
              "scope": "visual_alignment_only_not_patient_function_or_clinical_validation"}
    return replace(evidence, qc_state="alignment_accepted" if accepted else "alignment_rejected", review=review)


def prior_proximity_halo(evidence: RegisteredPrior, *, case_hash: str, scale_mm: float,
                         threshold: float):
    """H0 to a declared atlas threshold; threshold and scale are research settings."""
    from .worlds import physical_proximity_halo

    evidence.require_current(case_hash)
    if not np.isfinite(threshold) or not 0 < threshold <= 1:
        raise ValueError("A declared threshold in (0,1] is required")
    return physical_proximity_halo(evidence.data >= threshold, evidence.affine_ras_mm, scale_mm,
                                   source=f"population_prior:{evidence.registration_hash}:threshold={threshold}",
                                   coverage_mask=evidence.sampling_coverage,
                                   frame="RAS+", qc_state=evidence.qc_state)
