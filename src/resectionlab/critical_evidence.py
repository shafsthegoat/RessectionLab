"""Source-bound critical annotations for retrospective geometric planning.

Bindings validate declared identity, not the truth of a source's annotations.
Coverage describes the annotation domain, never sensitivity to unseen vessels.
This module neither performs registration nor supplies learning observations.
"""
from __future__ import annotations

from collections.abc import Mapping
import base64
from dataclasses import dataclass, field
from datetime import datetime
import hashlib
from itertools import product
import json
from pathlib import PurePosixPath
from types import MappingProxyType
from typing import Any
from urllib.parse import unquote, urlsplit

import numpy as np

from .core import SourceRef, array_digest, freeze_json, immutable_array, semantic_digest, thaw_json
from .data_policy import DataPolicyError, require_admitted_model
from .structural_evidence import _hash, _text, _time

STRUCTURES = ("motor", "language", "vessels")

# A bounded NIfTI-1 serialization envelope, not an anatomical/clinical tolerance.
# No caller may enlarge it. Both the coefficient bound and voxel-index bound
# must pass; this does not license registration or a change of acquisition.
_SERIALIZATION_ULPS = 4


def nifti1_header_record(raw: bytes, source_file_sha256: str) -> dict:
    """Retain the untouched 348-byte header and its original file binding."""
    if len(raw) != 348:
        raise ValueError("Normalization requires the original NIfTI-1 header")
    import nibabel as nib

    header = nib.Nifti1Header(binaryblock=raw, check=False)
    return {"source_file_sha256": _hash(source_file_sha256, "header source file"),
            "header_base64": base64.b64encode(raw).decode("ascii"),
            "header_sha256": hashlib.sha256(raw).hexdigest(),
            "raw_grid": _header_summary(header)}


def _header_summary(header) -> dict:
    return {"shape": list(header.get_data_shape()), "spatial_units": header.get_xyzt_units()[0],
            "xyzt_units_code": int(header["xyzt_units"]), "pixdim": header["pixdim"].tolist(),
            "qform_code": int(header["qform_code"]), "sform_code": int(header["sform_code"]),
            "qform_numeric_including_inactive": header.get_qform().tolist(),
            "sform_numeric": header.get_sform().tolist()}


def _raw_grid(record: Mapping):
    import nibabel as nib

    _hash(record.get("source_file_sha256"), "raw grid source file")
    try:
        raw = base64.b64decode(record["header_base64"], validate=True)
    except (KeyError, ValueError, TypeError) as error:
        raise ValueError("Raw grid needs an intact encoded NIfTI-1 header") from error
    if (len(raw) != 348 or _hash(hashlib.sha256(raw).hexdigest(), "raw header")
            != _hash(record.get("header_sha256"), "declared header")):
        raise ValueError("Raw grid header hash differs")
    header = nib.Nifti1Header(binaryblock=raw, check=False)
    if thaw_json(record.get("raw_grid")) != _header_summary(header):
        raise ValueError("Raw grid summary differs from preserved header bytes")
    if int(header["sizeof_hdr"]) != 348 or bytes(header["magic"]) != b"n+1\x00":
        raise ValueError("Normalization requires a single-file NIfTI-1 source")
    shape = header.get_data_shape()
    affine = header.get_sform()
    if (len(shape) != 3 or any(n <= 0 for n in shape) or int(header["sform_code"]) not in (1, 2)
            or not np.isfinite(affine).all() or not np.array_equal(affine[3], [0, 0, 0, 1])
            or abs(np.linalg.det(affine[:3, :3])) < 1e-12):
        raise ValueError("Normalization requires a finite, coded native sform")
    if int(header["qform_code"]) and _float32_steps(header.get_qform(), affine).max() > _SERIALIZATION_ULPS:
        raise ValueError("Active raw qform/sform disagree beyond serialization precision")
    return header, shape, affine


def _float32_steps(first, second) -> np.ndarray:
    # Main sform entries are stored as float32. Rounding is only relevant when
    # checking the quaternion-derived qform against those serialized entries.
    a, b = np.asarray(first, dtype=np.float32), np.asarray(second, dtype=np.float32)
    if not np.isfinite(a).all() or not np.isfinite(b).all() or np.any(np.signbit(a) != np.signbit(b)):
        raise ValueError("Normalization cannot change an affine coefficient sign")
    return np.abs(a.view(np.int32).astype(np.int64) - b.view(np.int32).astype(np.int64))


def source_reference_grid_metrics(annotation_grid: Mapping, reference_grid: Mapping) -> dict:
    """Check precision equivalence from raw headers without fitting a transform.

    The infinity norm of an affine displacement reaches its maximum at a box
    corner. Bounding the full voxel-support box therefore proves that every
    voxel centre retains its nearest reference index, without a resampling.
    """
    label, shape, a = _raw_grid(annotation_grid)
    reference, reference_shape, b = _raw_grid(reference_grid)
    if shape != reference_shape:
        raise ValueError("Normalization cannot change dimensions or voxel order")
    if reference.get_xyzt_units()[0] != "mm" or label.get_xyzt_units()[0] not in {"unknown", "mm"}:
        raise ValueError("Only explicit reference-mm inheritance is supported")
    steps = _float32_steps(a, b)
    if steps.max() > _SERIALIZATION_ULPS:
        raise ValueError("Raw grids differ beyond float32 serialization precision")
    corners = np.array([(*point, 1.) for point in product(*[(-.5, n-.5) for n in shape])])
    displacement = ((a-b) @ corners.T)[:3]
    voxel_displacement = (np.linalg.inv(b)[:3, :3] @ displacement)
    # Independently propagate the allowed coefficient rounding envelope. This
    # bound depends on stored precision and extent, never on clinical distance.
    ulp = np.maximum(np.abs(np.spacing(a.astype(np.float32))).astype(float),
                     np.abs(np.spacing(b.astype(np.float32))).astype(float))
    extent = np.array([*(n-.5 for n in shape), 1.])
    world_bound = (_SERIALIZATION_ULPS * ulp[:3]) @ extent
    voxel_bound = np.abs(np.linalg.inv(b)[:3, :3]) @ world_bound
    if (np.any(np.max(np.abs(displacement), axis=1) > world_bound)
            or np.max(voxel_bound) >= .5 or np.max(np.abs(voxel_displacement)) >= .5):
        raise ValueError("Serialization envelope cannot guarantee unchanged voxel indices")
    return {"rule": "nifti1_float32_four_ulp_and_unchanged_indices_v1",
            "maximum_allowed_float32_steps": _SERIALIZATION_ULPS,
            "maximum_observed_float32_steps": int(steps.max()),
            "maximum_support_displacement_reference_mm": float(np.linalg.norm(displacement, axis=0).max()),
            "maximum_support_displacement_voxels": float(np.linalg.norm(voxel_displacement, axis=0).max()),
            "maximum_support_component_voxels": float(np.abs(voxel_displacement).max()),
            "serialization_component_bound_reference_mm": world_bound.tolist(),
            "serialization_component_bound_voxels": voxel_bound.tolist(),
            "all_voxel_centres_keep_reference_index": True}


def _validate_reference_normalization(item, mask, coverage, affine) -> None:
    record = item.derivation
    proof = thaw_json(record)
    declared_hash = proof.pop("normalization_record_hash", None)
    if declared_hash != semantic_digest(proof):
        raise ValueError("Source-reference normalization record hash differs")
    if (item.structure != "vessels" or item.lineage["kind"] != "manual"
            or record.get("array_operation") != "unchanged_binary_positive_support"
            or not np.array_equal(mask, coverage)
            or record.get("positive_mask_hash") != array_digest(mask)):
        raise ValueError("Normalization requires unchanged manual positive-only vessel support")
    semantics = item.source_binding
    if (semantics.get("positive_class") != "source_manual_voxelwise_aneurysm_region"
            or semantics.get("coverage_policy") != "positive_support_only"
            or semantics.get("background_meaning") != "unknown_for_vascular_anatomy"):
        raise ValueError("Normalization requires explicit aneurysm-only positive semantics")
    if (record.get("unit_interpretation") != "reference_mm_from_explicit_RawSources_orig"
            or record.get("scope") != "retrospective_source_coded_frame_component"
            or record.get("scanner_frame_admitted") is not False
            or record.get("spatial_planning_admitted") is not False):
        raise ValueError("Normalization cannot claim scanner or spatial planning admission")
    linkage = SourceRef.from_dict(record.get("sidecar_source", {}))
    _source(linkage, "normalization linkage")
    if linkage.to_dict() != SourceRef.from_dict(semantics["linkage_source"]).to_dict():
        raise ValueError("Normalization linkage differs from annotation binding")
    try:
        sidecar = base64.b64decode(record["sidecar_base64"], validate=True)
        declaration = json.loads(sidecar)
    except (KeyError, ValueError, TypeError) as error:
        raise ValueError("Normalization needs the authentic source sidecar") from error
    if _hash(hashlib.sha256(sidecar).hexdigest(), "sidecar") != _hash(linkage.sha256, "sidecar source"):
        raise ValueError("Normalization sidecar hash differs")
    if declaration != {"Type": "Lesion", "RawSources": record.get("reference_filename"), "Space": "orig"}:
        raise ValueError("Normalization needs an explicit matching RawSources/orig declaration")
    _text(record.get("reference_filename"), "normalization reference filename")
    _text(record.get("inference_rationale"), "normalization inference rationale")
    annotation_grid, reference_grid = record["annotation_raw_grid"], record["reference_raw_grid"]
    if (_hash(annotation_grid["source_file_sha256"], "annotation grid file") != _hash(item.source.sha256, "annotation")
            or _hash(reference_grid["source_file_sha256"], "reference grid file") != item.reference_source_sha256):
        raise ValueError("Raw grids are bound to different original files")
    _, shape, _ = _raw_grid(annotation_grid)
    _, _, reference_affine = _raw_grid(reference_grid)
    if shape != mask.shape or not np.array_equal(affine, reference_affine):
        raise ValueError("Normalized arrays must retain the exact reference source grid")
    if thaw_json(record.get("precision_equivalence")) != source_reference_grid_metrics(annotation_grid, reference_grid):
        raise ValueError("Normalization precision receipt differs from original headers")


def availability_exclusion(available_at, review_available_at, cutoff) -> str | None:
    """Clock comparison only; unknown dates are never inferred from file dates."""
    if cutoff is None:
        return None
    cutoff = _time(cutoff, "planning cutoff")
    if available_at is None:
        return "availability_time_unknown"
    if _time(available_at, "annotation availability") > cutoff:
        return "available_after_planning_cutoff"
    if review_available_at is None:
        return "review_availability_time_unknown"
    if _time(review_available_at, "review availability") > cutoff:
        return "review_available_after_planning_cutoff"
    return None


def _source(value: SourceRef, name: str) -> None:
    if not isinstance(value, SourceRef) or value.sha256 is None or value.license is None:
        raise ValueError(f"{name} needs a typed, hash-pinned source and explicit rights")
    if value.provenance not in {"observed", "estimated"}:
        raise ValueError(f"{name} cannot be simulated or a population prior")


@dataclass(frozen=True, slots=True, eq=False)
class CriticalStructureEvidence:
    evidence_id: str
    structure: str
    mask: np.ndarray
    annotation_coverage: np.ndarray
    affine_ras_mm: np.ndarray
    case_id: str
    reference_image_hash: str
    reference_source_id: str
    reference_source_sha256: str
    source: SourceRef
    source_binding: Mapping[str, Any]
    derivation: Mapping[str, Any]
    lineage: Mapping[str, Any]
    acquisition_time: datetime | None = None
    available_at: datetime | None = None
    review: Mapping[str, Any] | None = None
    _array_state: tuple = field(init=False, repr=False)

    def __post_init__(self) -> None:
        for name in ("evidence_id", "case_id", "reference_source_id"):
            _text(getattr(self, name), name)
        if self.structure not in STRUCTURES:
            raise ValueError("Unsupported critical structure")
        for name in ("reference_image_hash", "reference_source_sha256"):
            object.__setattr__(self, name, _hash(getattr(self, name), name))
        _source(self.source, "annotation")
        for name in ("source_binding", "derivation", "lineage"):
            if not isinstance(getattr(self, name), Mapping):
                raise ValueError(f"{name} must be a source-linked record")
            object.__setattr__(self, name, freeze_json(getattr(self, name)))
        binding = self.source_binding
        for name in ("dataset", "participant", "timepoint"):
            _text(binding.get(name), f"source_binding.{name}")
        _source(SourceRef.from_dict(binding.get("linkage_source", {})), "patient/acquisition linkage")
        _source(SourceRef.from_dict(binding.get("coverage_source", {})), "annotation coverage")
        if binding.get("coverage_meaning") != "source_documented_annotation_domain":
            raise ValueError("Coverage must describe a documented annotation domain")
        # Precision normalization has its own source-bound proof below. It must
        # never be described as equality of the untouched source headers.
        if self.derivation.get("method") not in {"identity_grid", "source_reference_grid_normalization"}:
            raise ValueError("Critical annotation registration/resampling is not yet admitted")
        if self.derivation["method"] == "identity_grid" and any(
                key in self.derivation for key in ("annotation_raw_grid", "reference_raw_grid", "normalization_record_hash")):
            raise ValueError("Normalization provenance cannot be relabeled identity_grid")
        if _hash(self.derivation.get("source_image_sha256"), "derivation image") != self.reference_source_sha256:
            raise ValueError("Annotation derivation refers to another acquired image")
        if _hash(self.derivation.get("annotation_sha256"), "derivation annotation") != _hash(self.source.sha256, "annotation hash"):
            raise ValueError("Annotation derivation and source disagree")
        _source(SourceRef.from_dict(self.lineage.get("source", {})), "annotation lineage")
        if self.lineage.get("kind") not in {"manual", "manual_with_nonlearned_interpolation", "model_assisted", "unknown"}:
            raise ValueError("Unsupported annotation ancestry")
        models = self.lineage.get("model_sha256", ())
        if not isinstance(models, tuple):
            raise ValueError("model_sha256 must list every learned initializer")
        for digest in models:
            _hash(digest, "initializer hash")
        if models and self.lineage["kind"] != "model_assisted":
            raise ValueError("Learned initializer cannot be relabeled as manual ancestry")
        for name in ("acquisition_time", "available_at"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _time(value, name))
        if self.acquisition_time is not None and self.available_at is not None and self.available_at < self.acquisition_time:
            raise ValueError("Annotation cannot be available before acquisition")
        mask, coverage = np.asarray(self.mask), np.asarray(self.annotation_coverage)
        if mask.ndim != 3 or not mask.size or mask.dtype != np.bool_:
            raise ValueError("Critical mask must be a nonempty Boolean 3D array")
        if coverage.shape != mask.shape or coverage.dtype != np.bool_:
            raise ValueError("Annotation coverage must be Boolean and share the source grid")
        if np.any(mask & ~coverage):
            raise ValueError("Positive labels cannot lie outside the documented annotation domain")
        affine = np.asarray(self.affine_ras_mm, dtype=float)
        if (affine.shape != (4, 4) or not np.isfinite(affine).all()
                or not np.allclose(affine[3], [0, 0, 0, 1], atol=1e-12, rtol=0)
                or abs(np.linalg.det(affine[:3, :3])) < 1e-12):
            raise ValueError("Evidence affine must be an invertible RAS+ millimeter transform")
        if self.derivation["method"] == "source_reference_grid_normalization":
            _validate_reference_normalization(self, mask, coverage, affine)
        for name, value in (("mask", mask), ("annotation_coverage", coverage), ("affine_ras_mm", affine)):
            object.__setattr__(self, name, immutable_array(value))
        object.__setattr__(self, "_array_state", self._layout())
        if self.review is not None:
            if not isinstance(self.review, Mapping):
                raise ValueError("Review must be a source-linked record")
            object.__setattr__(self, "review", freeze_json(self.review))
            if self.review.get("status") not in {"source_human_reviewed", "rejected"}:
                raise ValueError("Review status must describe the actual source review")
            _source(SourceRef.from_dict(self.review.get("source", {})), "review")
            _text(self.review.get("scope"), "review scope")
            if "available_at" not in self.review:
                raise ValueError("Review availability must be recorded or explicitly unknown")
            if self.review["available_at"] is not None:
                _time(self.review["available_at"], "review availability")
            if self.review.get("content_hash") != self.content_hash:
                raise ValueError("Review does not bind the exact annotation, coverage and provenance")

    def _layout(self) -> tuple:
        return tuple((a.shape, a.dtype.str, a.strides, a.__array_interface__["data"][0])
                     for a in (self.mask, self.annotation_coverage, self.affine_ras_mm))

    def _check_layout(self) -> None:
        if self._layout() != self._array_state:
            raise ValueError("Critical evidence array metadata changed")

    def _content(self) -> dict:
        self._check_layout()
        return {"schema": "critical-structure-evidence-v1", "evidence_id": self.evidence_id,
                "structure": self.structure, "case_id": self.case_id,
                "reference_image_hash": self.reference_image_hash,
                "reference_source_id": self.reference_source_id,
                "reference_source_sha256": self.reference_source_sha256,
                "shape": list(self.mask.shape), "mask_hash": array_digest(self.mask),
                "annotation_coverage_hash": array_digest(self.annotation_coverage),
                "affine_ras_mm": self.affine_ras_mm.tolist(), "source": self.source.to_dict(),
                "source_binding": thaw_json(self.source_binding), "derivation": thaw_json(self.derivation),
                "lineage": thaw_json(self.lineage),
                "acquisition_time": None if self.acquisition_time is None else self.acquisition_time.isoformat(),
                "available_at": None if self.available_at is None else self.available_at.isoformat()}

    @property
    def content_hash(self) -> str:
        return semantic_digest(self._content())

    @property
    def evidence_hash(self) -> str:
        return semantic_digest({**self._content(), "review": thaw_json(self.review)})

    def to_manifest(self) -> dict:
        return {**self._content(), "review": thaw_json(self.review), "evidence_hash": self.evidence_hash}

    @classmethod
    def from_manifest(cls, value: Mapping, *, mask: np.ndarray, annotation_coverage: np.ndarray):
        names = ("evidence_id", "structure", "case_id", "reference_image_hash", "reference_source_id",
                 "reference_source_sha256", "affine_ras_mm", "source_binding", "derivation", "lineage",
                 "acquisition_time", "available_at", "review")
        result = cls(mask=mask, annotation_coverage=annotation_coverage,
                     source=SourceRef.from_dict(value["source"]), **{key: value[key] for key in names})
        if result.to_manifest() != dict(value):
            raise ValueError("Critical evidence manifest contradicts its source arrays")
        return result

    def assert_matches(self, case: Any) -> None:
        self._check_layout()
        identity = case.metadata.get("evidence_identity", {})
        if any(identity.get(key) != self.source_binding[key] for key in ("dataset", "participant", "timepoint")):
            raise ValueError("Critical evidence patient/acquisition differs from source-derived case identity")
        ras = case.affine if case.frame == "RAS+" else np.diag([-1., -1., 1., 1.]) @ case.affine
        if (self.case_id != case.case_id or self.mask.shape != case.mri.shape
                or self.reference_image_hash != array_digest(case.mri)
                or not np.array_equal(self.affine_ras_mm, ras)):
            raise ValueError("Critical evidence differs from the selected image or physical frame")
        refs = [ref for ref in case.source_refs if ref.source_id == self.reference_source_id]
        if (len(refs) != 1 or refs[0].sha256 is None or refs[0].provenance != "observed"
                or _hash(refs[0].sha256, "case image SHA") != self.reference_source_sha256):
            raise ValueError("Critical reference acquisition is absent from case provenance")
        if self.derivation["method"] == "source_reference_grid_normalization":
            filename = PurePosixPath(unquote(urlsplit(refs[0].uri).path)).name
            if filename != self.derivation["reference_filename"]:
                raise ValueError("Normalized source filename differs from reference acquisition")
            if (case.metadata.get("evidence_scope") != "retrospective_source_coded_frame_component"
                    or case.metadata.get("scanner_frame_admitted") is not False
                    or case.metadata.get("spatial_planning_admitted") is not False):
                raise ValueError("Normalized evidence requires explicit source-frame component limits")

    def exclusion_reason(self, case: Any) -> str | None:
        self.assert_matches(case)
        reason = availability_exclusion(self.available_at,
            None if self.review is None else self.review["available_at"],
            None if case.context is None else case.context.planning_as_of)
        if reason is not None:
            return reason
        for model in self.lineage.get("model_sha256", ()):
            try:
                require_admitted_model(model, "critical_annotation_consumption")
            except DataPolicyError as error:
                return error.code
        if self.lineage["kind"] not in {"manual", "manual_with_nonlearned_interpolation"}:
            return "annotation_ancestry_unresolved"
        if self.review is None or self.review["status"] != "source_human_reviewed":
            return "source_annotation_review_unaccepted"
        if not np.any(self.annotation_coverage):
            return "annotation_domain_empty"
        return None


@dataclass(frozen=True, slots=True)
class CriticalConstraints:
    masks: Mapping[str, np.ndarray | None]
    annotation_coverage: Mapping[str, np.ndarray | None]
    receipt: Mapping[str, Any]
    shape: tuple[int, int, int]
    _array_state: tuple = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_array_state", self._layout())

    def _layout(self) -> tuple:
        return tuple((a.shape, a.dtype.str, a.strides, a.__array_interface__["data"][0])
                     for values in (self.masks, self.annotation_coverage)
                     for a in values.values() if a is not None)

    def _check_layout(self) -> None:
        if self._layout() != self._array_state:
            raise ValueError("Resolved critical evidence array metadata changed")

    @property
    def fingerprint(self) -> str:
        # Withheld observations cannot influence planner seeds/model identity.
        return semantic_digest(self.planning_receipt)

    @property
    def planning_receipt(self) -> dict:
        self._check_layout()
        record = thaw_json(self.receipt)
        record["records"] = {key: value for key, value in record["records"].items()
                             if value["exclusion_reason"] is None}
        return record

    @property
    def planning_binding(self) -> dict | None:
        """Absent and entirely excluded registries have the same input binding."""
        record = self.planning_receipt
        return record if record["records"] else None

    @property
    def hard_exclusion(self) -> np.ndarray | None:
        self._check_layout()
        return self.masks["vessels"]

    def route_coverage(self, indices: np.ndarray) -> dict:
        self._check_layout()
        points = np.asarray(indices)
        if points.ndim != 2 or points.shape[1] != 3 or points.dtype.kind not in "iu":
            raise ValueError("Swept cell indices must be an integer N by 3 array")
        if np.any(points < 0) or np.any(points >= np.asarray(self.shape)):
            raise ValueError("Swept coverage indices must lie within the source image")
        points = np.unique(points, axis=0)
        result = {}
        for name, coverage in self.annotation_coverage.items():
            covered = None if coverage is None else int(np.count_nonzero(coverage[tuple(points.T)]))
            result[name] = {"in_image_swept_cells": len(points), "covered_cells": covered,
                            "uncovered_cells": None if covered is None else len(points) - covered,
                            "outside_image": "unassessed", "meaning": "annotation_domain_only"}
        return result


def resolve_critical_evidence(case: Any) -> CriticalConstraints:
    masks = {name: None for name in STRUCTURES}
    coverage = {name: None for name in STRUCTURES}
    records, seen = {}, set()
    for key, item in sorted(getattr(case, "critical_evidence", {}).items()):
        if not isinstance(item, CriticalStructureEvidence) or key != item.evidence_id:
            raise ValueError("Critical evidence must map IDs to typed records")
        if item.structure in seen:
            raise ValueError("Only one selected critical record per structure is supported")
        seen.add(item.structure)
        reason = item.exclusion_reason(case)
        records[key] = {"evidence_hash": item.evidence_hash, "structure": item.structure,
                        "exclusion_reason": reason, "source": item.source.to_dict(),
                        "source_binding": thaw_json(item.source_binding), "review": thaw_json(item.review),
                        "derivation": thaw_json(item.derivation),
                        "lineage": thaw_json(item.lineage),
                        "acquisition_time": None if item.acquisition_time is None else item.acquisition_time.isoformat(),
                        "available_at": None if item.available_at is None else item.available_at.isoformat(),
                        "annotated_cells": int(np.count_nonzero(item.annotation_coverage)),
                        "image_cells": item.mask.size, "coverage_hash": array_digest(item.annotation_coverage)}
        if reason is None:
            masks[item.structure], coverage[item.structure] = item.mask, item.annotation_coverage
    receipt = {"schema": "critical-constraints-v1", "records": records,
               "missing": [name for name in STRUCTURES if masks[name] is None],
               "objective_structures": [name for name in STRUCTURES
                                         if coverage[name] is not None and bool(np.all(coverage[name]))],
               "time_scope": "retrospective_image_geometry_availability_unassessed" if case.context is None else "timestamp_filtered",
               "coverage_meaning": "Annotation domain only; unseen structures and outside-image anatomy remain unknown",
               "trajectory_coverage": "not_computed", "clinical_clearance": False}
    return CriticalConstraints(MappingProxyType(masks), MappingProxyType(coverage), freeze_json(receipt), tuple(case.mri.shape))


def require_canonical_masks(constraints: CriticalConstraints, supplied: Mapping | None) -> None:
    constraints._check_layout()
    for name, mask in (supplied or {}).items():
        expected = constraints.masks.get(name)
        if name not in constraints.masks or (mask is None) != (expected is None):
            raise ValueError("Explicit critical mask differs from source-bound case evidence")
        if mask is not None and (np.asarray(mask).dtype != np.bool_ or not np.array_equal(mask, expected)):
            raise ValueError("Explicit critical mask differs from source-bound case evidence")


def canonical_hard_exclusion(case: Any, supplied: np.ndarray | None = None):
    constraints = resolve_critical_evidence(case)
    if any(constraints.masks[name] is not None for name in ("motor", "language")):
        raise ValueError("UNSUPPORTED_NATIVE_CRITICAL_FUNCTION: native consumers do not yet evaluate these functional annotations")
    expected = constraints.hard_exclusion
    if supplied is not None and (expected is None or np.asarray(supplied).dtype != np.bool_
                                 or not np.array_equal(supplied, expected)):
        raise ValueError("Explicit hard exclusion differs from source-bound case evidence")
    return expected, constraints
