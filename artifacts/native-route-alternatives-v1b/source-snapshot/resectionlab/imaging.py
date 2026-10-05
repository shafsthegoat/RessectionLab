"""Local NIfTI import, explicit input QC, and portable research case bundles.

Arrays retain their acquired voxel order. Affines map voxel centres to RAS+
millimetres; no implicit registration, orientation fix, or segmentation occurs.
The original label masks remain independent from reviewed working annotations.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields, is_dataclass, replace
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path
import tempfile
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile

import nibabel as nib
import numpy as np
from scipy.ndimage import affine_transform

from .core import CaseData, PatientContext, SourceRef, array_digest


BUNDLE_SCHEMA = "resectionlab.case/1"
_UNIT_TO_MM = {"mm": 1.0, "meter": 1000.0, "micron": 0.001}


class ImagingError(ValueError):
    """Named, actionable import/QC failure suitable for desktop display."""

    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(f"{code}: {message}")


def file_sha256(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if is_dataclass(value):
        return {field.name: _jsonable(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Path):
        return str(value)
    return value


def inspect_nifti(path: str | Path, *, dimensions: int = 3) -> dict[str, Any]:
    """Validate spatial header semantics without choosing between conflicting forms.

    A valid qform-only or sform-only image is accepted. Both active forms must
    agree. Unknown units or an absent coded transform require an explicit source
    repair, never an undocumented assumption by the importer.
    """
    path = Path(path)
    try:
        image = nib.load(str(path))
    except Exception as exc:
        raise ImagingError("UNREADABLE_NIFTI", f"Cannot read {path.name}: {exc}") from exc
    if not isinstance(image, (nib.Nifti1Image, nib.Nifti2Image)):
        raise ImagingError("UNSUPPORTED_FORMAT", "Input must be NIfTI-1 or NIfTI-2.")
    if len(image.shape) != dimensions or any(size <= 0 for size in image.shape):
        raise ImagingError("INVALID_DIMENSIONS", f"Expected {dimensions}D data; found {image.shape}.")
    units = image.header.get_xyzt_units()[0]
    if units not in _UNIT_TO_MM:
        raise ImagingError("UNKNOWN_SPATIAL_UNITS", "Spatial units must be explicitly mm, meter, or micron.")
    factor = _UNIT_TO_MM[units]
    try:
        qform, qcode = image.get_qform(coded=True)
        sform, scode = image.get_sform(coded=True)
    except Exception as exc:
        raise ImagingError("INVALID_TRANSFORM", f"Invalid coded transform: {exc}") from exc
    if not qcode and not scode:
        raise ImagingError("TRANSFORM_UNRESOLVED", "Neither qform nor sform declares an anatomical frame.")
    if qcode and scode:
        scale_to_mm = np.diag([factor, factor, factor, 1.0])
        if _maximum_corner_displacement(scale_to_mm @ qform, scale_to_mm @ sform, image.shape[:3]) > 0.01:
            raise ImagingError("QFORM_SFORM_DISAGREEMENT", "Active qform and sform differ across the image extent; review the source transform.")
    affine = np.array(sform if scode else qform, dtype=float, copy=True)
    affine[:3, :] *= factor
    if (not np.isfinite(affine).all() or not np.allclose(affine[3], [0, 0, 0, 1])
            or abs(np.linalg.det(affine[:3, :3])) < 1e-12):
        raise ImagingError("INVALID_AFFINE", "Voxel-to-world affine must be finite and invertible.")
    spacing = np.linalg.norm(affine[:3, :3], axis=0)
    directions = affine[:3, :3] / spacing
    if not np.allclose(directions.T @ directions, np.eye(3), atol=1e-4):
        raise ImagingError("UNSUPPORTED_SHEAR", "Sheared voxel grids need an explicit, reviewed resampling transform.")
    header_spacing = np.array(image.header.get_zooms()[:3]) * factor
    if not np.allclose(spacing, header_spacing, atol=0.01, rtol=1e-4):
        raise ImagingError("SPACING_DISAGREEMENT", "Header voxel sizes disagree with the coded affine.")
    codes = nib.aff2axcodes(affine)
    if any(code is None for code in codes):
        raise ImagingError("ORIENTATION_UNRESOLVED", "Image axes do not define three spatial directions.")
    obliquity = np.rad2deg(nib.affines.obliquity(affine))
    return {
        "file": path.name, "shape": list(image.shape), "affine_ras_mm": affine.tolist(),
        "source_units": units, "physical_units": "mm", "unit_scale_to_mm": factor,
        "qform_code": int(qcode), "sform_code": int(scode),
        "qform_native": None if qform is None else qform.tolist(),
        "sform_native": None if sform is None else sform.tolist(),
        "orientation": list(codes), "spacing_mm": spacing.tolist(),
        "voxel_volume_mm3": float(abs(np.linalg.det(affine[:3, :3]))),
        "handedness": "left" if np.linalg.det(affine[:3, :3]) < 0 else "right",
        "anisotropic": bool(np.max(spacing) - np.min(spacing) > 0.01),
        "obliquity_degrees": obliquity.tolist(), "status": "passed_header_checks",
        "visual_alignment_review": "pending", "frame": "RAS+",
        "simulation_frame": "source_coded_frame_native_status_requires_source_provenance",
    }


def _read_finite(path: str | Path) -> np.ndarray:
    try:
        data = nib.load(str(path)).get_fdata(dtype=np.float32)
    except Exception as exc:
        raise ImagingError("UNREADABLE_DATA", f"Cannot read voxel data from {Path(path).name}: {exc}") from exc
    if not np.isfinite(data).all():
        raise ImagingError("NONFINITE_DATA", f"{Path(path).name} contains NaN or infinite voxels.")
    return data


def _maximum_corner_displacement(first: np.ndarray, second: np.ndarray, shape: tuple | list) -> float:
    """Maximum grid-corner displacement in the transforms' output coordinate units."""
    corners = np.array(np.meshgrid(*[(0, size - 1) for size in shape], indexing="ij")).reshape(3, -1).T
    delta = nib.affines.apply_affine(first, corners) - nib.affines.apply_affine(second, corners)
    return float(np.max(np.linalg.norm(delta, axis=1))) if np.isfinite(delta).all() else float("inf")


def _aligned_mask(path: str | Path, reference_qc: Mapping[str, Any]) -> tuple[np.ndarray, dict]:
    qc = inspect_nifti(path)
    if qc["shape"] != reference_qc["shape"]:
        raise ImagingError("MASK_SHAPE_MISMATCH", f"{Path(path).name} does not share the structural image grid.")
    if _maximum_corner_displacement(np.asarray(qc["affine_ras_mm"]), np.asarray(reference_qc["affine_ras_mm"]), qc["shape"]) > 0.01:
        raise ImagingError("MASK_AFFINE_MISMATCH", f"{Path(path).name} is not aligned to the structural image.")
    data = _read_finite(path)
    if (data < 0).any() or not np.equal(data, np.rint(data)).all():
        raise ImagingError("INVALID_LABELS", "Supplied annotations must contain nonnegative integer labels.")
    return data, qc


def load_nifti_case(
    structural_path: str | Path, tumor_mask_path: str | Path | None = None, *,
    case_id: str | None = None, label_map: Mapping[int, str] | None = None,
    brain_mask_path: str | Path | None = None, source_url: str | None = None,
    license: str | None = None, metadata: Mapping[str, Any] | None = None,
    context: PatientContext | None = None,
) -> CaseData:
    """Import supplied structural imaging and annotations without inferring function.

    Unknown positive labels retain neutral ``label_N`` names. An explicit map
    must cover every nonzero label; absent release labels are permitted and no
    radiological compartment is guessed.
    Header checks do not replace the required visual/source-provenance review.
    """
    structural_path = Path(structural_path)
    qc = inspect_nifti(structural_path)
    mri = _read_finite(structural_path)
    if float(np.ptp(mri)) == 0:
        raise ImagingError("CONSTANT_STRUCTURAL_IMAGE", "The structural image has no intensity variation.")
    compartments: dict[str, np.ndarray] = {}
    refs = [SourceRef(source_id="structural", uri=structural_path.resolve().as_uri(),
                      sha256=file_sha256(structural_path), license=license,
                      native_frame="RAS+", provenance="observed")]
    qc_inputs = {"structural": qc}
    if tumor_mask_path is not None:
        mask, mask_qc = _aligned_mask(tumor_mask_path, qc)
        labels = [int(value) for value in np.unique(mask) if value > 0]
        if not labels:
            raise ImagingError("EMPTY_TARGET_ANNOTATION", "Supplied target annotation has no positive labels.")
        mapping = {int(key): str(value) for key, value in (label_map or {}).items()}
        if label_map is not None and not set(labels).issubset(mapping):
            raise ImagingError("LABEL_CONVENTION_MISMATCH", f"Label map must cover observed labels {labels}.")
        names = [mapping.get(label, f"label_{label}") for label in labels]
        if len(set(names)) != len(names) or any(not name.strip() for name in names):
            raise ImagingError("INVALID_COMPARTMENT_NAMES", "Each source label needs a distinct, nonempty name.")
        compartments = {name: mask == label for name, label in zip(names, labels)}
        refs.append(SourceRef(source_id="supplied_target_annotation", uri=Path(tumor_mask_path).resolve().as_uri(),
                              sha256=file_sha256(tumor_mask_path), license=license,
                              native_frame="RAS+", provenance="observed"))
        qc_inputs["target_annotation"] = mask_qc
    brain_mask = None
    if brain_mask_path is not None:
        brain_data, brain_qc = _aligned_mask(brain_mask_path, qc)
        if set(np.unique(brain_data)) - {0, 1}:
            raise ImagingError("BRAIN_MASK_NOT_BINARY", "A supplied brain mask must contain only 0 and 1.")
        brain_mask = brain_data.astype(bool)
        if not brain_mask.any():
            raise ImagingError("EMPTY_BRAIN_MASK", "The supplied brain mask is empty.")
        refs.append(SourceRef(source_id="supplied_brain_mask", uri=Path(brain_mask_path).resolve().as_uri(),
                              sha256=file_sha256(brain_mask_path), license=license,
                              native_frame="RAS+", provenance="observed"))
        qc_inputs["brain_mask"] = brain_qc
    details = dict(metadata or {})
    details.update({
        "input_mode": "restricted_structural", "benchmark_track": "annotation_assisted" if compartments else "viewer_only",
        "imaging_qc": qc_inputs, "source_record_url": source_url,
        "label_conventions": {str(label): name for label, name in zip(labels, names)} if compartments else {},
        "functional_evidence": {"motor": "unassessed", "language": "unassessed"},
        "diffusion_status": "not_audited", "clinical_use_status": "research_only",
        "clinical_deficit_probability": None, "clinical_probability_reason": "no_validated_clinical_outcome_model",
        "annotation_review": "pending", "is_synthetic": False,
    })
    unknowns = ["motor_function_unassessed", "language_function_unassessed", "vascular_anatomy_unassessed",
                "skull_anatomy_unassessed", "diffusion_not_audited", "native_frame_provenance_requires_review"]
    if brain_mask is None:
        unknowns.append("brain_segmentation_unassessed")
    return CaseData(case_id=case_id or structural_path.name.removesuffix(".gz").removesuffix(".nii"),
                    mri=mri, compartments=compartments, affine=np.asarray(qc["affine_ras_mm"]),
                    source_refs=tuple(refs), context=context, brain_mask=brain_mask,
                    unknowns=tuple(unknowns), metadata=details)


def audit_diffusion(
    dwi_path: str | Path | None, bvals_path: str | Path | None, bvecs_path: str | Path | None, *,
    gradient_frame: str | None = None, registration_reviewed: bool = False,
    gradient_convention: str | None = None, preprocessing_reviewed: bool = False,
    gradient_rotation_reviewed: bool = False, preprocessing_reference: str | None = None,
) -> dict[str, Any]:
    """Audit raw directional input separately from any tract reconstruction claim.

    Valid direction magnitudes/counts alone cannot resolve rotation conventions,
    motion corrections, registration, or tumor-domain tract quality.
    """
    result: dict[str, Any] = {"usable_for_reconstruction": False, "tracts_verified": False,
                              "clinical_deficit_probability": None, "issues": []}
    if dwi_path is None or bvals_path is None or bvecs_path is None:
        result["issues"].append("MISSING_GRADIENTS" if dwi_path else "MISSING_RAW_DIFFUSION")
        return result
    try:
        result["imaging_qc"] = inspect_nifti(dwi_path, dimensions=4)
        volumes = result["imaging_qc"]["shape"][3]
        bvals = np.loadtxt(bvals_path, dtype=float).reshape(-1)
        bvecs = np.loadtxt(bvecs_path, dtype=float)
        if bvecs.shape == (3, volumes):
            bvecs = bvecs.T
        elif bvecs.shape != (volumes, 3):
            raise ImagingError("GRADIENT_COUNT_MISMATCH", "b-vectors must have shape 3×N or N×3.")
        if len(bvals) != volumes:
            raise ImagingError("GRADIENT_COUNT_MISMATCH", "b-value count differs from diffusion volume count.")
        if not np.isfinite(bvals).all() or not np.isfinite(bvecs).all() or (bvals < 0).any():
            raise ImagingError("INVALID_GRADIENT_VALUES", "Gradient files must be finite with nonnegative b-values.")
        weighted = bvals > 50
        norms = np.linalg.norm(bvecs, axis=1)
        if np.any(np.abs(norms[weighted] - 1.0) > 0.05):
            raise ImagingError("GRADIENT_NORM_INVALID", "Diffusion-weighted b-vectors must have unit norm within 0.05.")
        if not np.any(~weighted) or np.count_nonzero(weighted) < 6:
            raise ImagingError("INSUFFICIENT_DIRECTIONS", "Initial reconstruction needs a b0 and at least six weighted volumes.")
        g = bvecs[weighted]
        design = np.column_stack((g[:, 0] ** 2, g[:, 1] ** 2, g[:, 2] ** 2,
                                  2 * g[:, 0] * g[:, 1], 2 * g[:, 0] * g[:, 2], 2 * g[:, 1] * g[:, 2]))
        if np.linalg.matrix_rank(design) < 6:
            raise ImagingError("DEGENERATE_GRADIENT_DIRECTIONS", "Directions do not support even a full-rank tensor design.")
        if gradient_frame not in {"voxel", "world_RAS", "world_LPS"}:
            result["issues"].append("GRADIENT_FRAME_UNRESOLVED")
        if not registration_reviewed:
            result["issues"].append("DIFFUSION_REGISTRATION_UNREVIEWED")
        if gradient_convention not in {"BIDS_FSL", "image_axes", "world_RAS", "world_LPS"}:
            result["issues"].append("GRADIENT_CONVENTION_UNRESOLVED")
        if not preprocessing_reviewed or not preprocessing_reference:
            result["issues"].append("PREPROCESSING_UNREVIEWED")
        if not gradient_rotation_reviewed:
            result["issues"].append("GRADIENT_ROTATION_UNREVIEWED")
        result.update({"volume_count": volumes, "b0_count": int(np.count_nonzero(~weighted)),
                       "weighted_direction_count": int(np.count_nonzero(weighted)), "gradient_frame": gradient_frame,
                       "gradient_design_rank": int(np.linalg.matrix_rank(design)), "bvalue_units": "s/mm²",
                       "gradient_convention": gradient_convention, "preprocessing_reference": preprocessing_reference,
                       "source_hashes": {"dwi": file_sha256(dwi_path), "bvals": file_sha256(bvals_path),
                                         "bvecs": file_sha256(bvecs_path)},
                       "numerical_gradient_checks": "passed", "usable_for_reconstruction": not result["issues"]})
    except ImagingError as exc:
        result["issues"].append(exc.code)
        result["detail"] = str(exc)
    except (OSError, ValueError) as exc:
        result["issues"].append("UNREADABLE_GRADIENTS")
        result["detail"] = str(exc)
    return result


def load_fractional_annotation_case(
    structural_path: str | Path, annotation_path: str | Path, *, threshold: float,
    annotation_interpretation: str, compartment_name: str = "source_target_threshold_scenario",
    case_id: str | None = None, source_url: str | None = None, license: str | None = None,
    metadata: Mapping[str, Any] | None = None, context: PatientContext | None = None,
) -> CaseData:
    """Explicit threshold scenario for a fractional source annotation.

    Only physically identical grids expressed with different axis permutations
    or flips are accepted. Reindexing uses nearest-neighbour sampling at integer
    source indices, with no inferred image registration. The fractional source
    file is never rewritten. Its reference/hash and intensity range remain in
    the case; ``source_compartments`` retains the initial *derived* binary mask.
    Fractional source intensities are never labeled calibrated probabilities.
    """
    if not np.isfinite(threshold) or not 0 < threshold <= 1:
        raise ImagingError("INVALID_ANNOTATION_THRESHOLD", "Specify a finite threshold greater than zero and at most one.")
    if not annotation_interpretation.strip() or not compartment_name.strip():
        raise ImagingError("ANNOTATION_INTERPRETATION_REQUIRED", "Name the source annotation meaning and derived compartment.")
    base = load_nifti_case(structural_path, case_id=case_id, source_url=source_url,
                           license=license, metadata=metadata, context=context)
    source_qc = inspect_nifti(annotation_path)
    source = _read_finite(annotation_path)
    if source.min() < 0 or source.max() > 1 + 1e-6:
        raise ImagingError("FRACTIONAL_ANNOTATION_RANGE", "This explicit operation requires source intensities in [0,1].")
    transform = np.linalg.inv(np.asarray(source_qc["affine_ras_mm"])) @ base.affine
    rounded = np.rint(transform)
    # NIfTI-1 stores affines in float32; permit sub-0.001-voxel representation
    # noise while rejecting an actual image registration or resampling warp.
    linear = rounded[:3, :3]
    if (not np.allclose(transform, rounded, atol=1e-3, rtol=0)
            or not np.all(np.isin(linear, [-1, 0, 1]))
            or not np.all(np.sum(np.abs(linear), axis=0) == 1)
            or not np.all(np.sum(np.abs(linear), axis=1) == 1)):
        raise ImagingError("ANNOTATION_REGISTRATION_REQUIRED", "Annotation differs by more than a physical grid-axis reindexing.")
    maximum_rounding_error = _maximum_corner_displacement(transform, rounded, base.mri.shape)
    if maximum_rounding_error > 1e-3:
        raise ImagingError("ANNOTATION_REGISTRATION_REQUIRED", "Annotation grid drift exceeds 0.001 voxel across the image extent.")
    corners = np.array(np.meshgrid(*[(0, size - 1) for size in base.mri.shape], indexing="ij")).reshape(3, -1).T
    mapped = corners @ linear.T + rounded[:3, 3]
    if not (np.array_equal(mapped.min(axis=0), [0, 0, 0])
            and np.array_equal(mapped.max(axis=0), np.asarray(source.shape) - 1)):
        raise ImagingError("ANNOTATION_COVERAGE_MISMATCH", "Annotation and structural grids do not cover the same physical voxel centres.")
    reindexed = affine_transform(source, linear, offset=rounded[:3, 3], output_shape=base.mri.shape,
                                order=0, mode="constant", cval=0, prefilter=False)
    mask = reindexed >= threshold
    if not mask.any():
        raise ImagingError("EMPTY_TARGET_ANNOTATION", "The declared threshold leaves no target voxels.")
    details = _jsonable(base.metadata)
    details.update({"benchmark_track": "annotation_assisted_threshold_scenario",
                    "annotation_review": "pending", "label_conventions": {"1": compartment_name},
                    "fractional_annotation": {
                        "interpretation": annotation_interpretation, "provenance": "observed",
                        "intensity_meaning": "source annotation intensity; not a calibrated probability",
                        "source_min": float(source.min()), "source_max": float(source.max()),
                        "source_shape": list(source.shape), "source_affine_ras_mm": source_qc["affine_ras_mm"],
                        "target_voxel_to_source_voxel": transform.tolist(),
                        "applied_integer_reindex": rounded.tolist(),
                        "max_grid_rounding_error_voxels": maximum_rounding_error,
                        "grid_rounding_error_definition": "maximum Euclidean source-voxel displacement at the eight structural-grid corners",
                        "threshold": float(threshold), "threshold_rule": "source_intensity >= threshold",
                        "threshold_evidence": "declared_research_assumption", "derived_provenance": "estimated",
                        "source_compartments_meaning": "initial threshold-derived binary annotation, not original fractional values",
                        "raw_source_retention": "immutable external source file referenced by SHA256; not embedded in compact bundle",
                    }})
    details["imaging_qc"]["fractional_annotation"] = source_qc
    ref = SourceRef(source_id="fractional_source_annotation", uri=Path(annotation_path).resolve().as_uri(),
                    sha256=file_sha256(annotation_path), license=license, native_frame="RAS+", provenance="observed")
    return replace(base, compartments={compartment_name: mask}, source_compartments={compartment_name: mask},
                   source_refs=base.source_refs + (ref,), metadata=details,
                   unknowns=base.unknowns + ("annotation_threshold_is_research_assumption",))


def revise_compartment(case: CaseData, name: str, mask: np.ndarray, *, reason: str) -> CaseData:
    """Return a new case revision and retain immutable, originally supplied labels."""
    if name not in case.compartments:
        raise ImagingError("UNKNOWN_COMPARTMENT", f"Cannot correct missing compartment {name!r}.")
    if not reason.strip():
        raise ImagingError("CORRECTION_REASON_REQUIRED", "Record why the annotation changed.")
    mask = np.asarray(mask)
    if mask.shape != case.mri.shape or mask.dtype != np.bool_:
        raise ImagingError("INVALID_CORRECTION_MASK", "Correction must be a boolean array on the unchanged image grid.")
    active = dict(case.compartments)
    active[name] = mask
    metadata = _jsonable(case.metadata)
    history = list(metadata.get("annotation_history", []))
    history.append({"compartment": name, "reason": reason.strip(), "from_case_hash": case.semantic_hash,
                    "to_revision": case.revision + 1, "provenance": "estimated", "dependent_plans": "invalidated"})
    metadata["annotation_history"] = history
    metadata["annotation_review"] = "manually_corrected"
    return replace(case, compartments=active, revision=case.revision + 1, metadata=metadata)


def save_case(case: CaseData, path: str | Path, *, artifacts: Mapping[str, Any] | None = None) -> Path:
    """Atomically save a self-contained ZIP with JSON and non-pickled numeric arrays."""
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {"mri": case.mri, "affine": case.affine}
    index: dict[str, dict[str, str]] = {"compartments": {}, "source_compartments": {}}
    for group in index:
        for position, (name, values) in enumerate(getattr(case, group).items()):
            key = f"{group}_{position}"
            arrays[key] = values
            index[group][name] = key
    if case.brain_mask is not None:
        arrays["brain_mask"] = case.brain_mask
    structural_records = {}
    for position, (name, evidence) in enumerate(getattr(case, "structural_evidence", {}).items()):
        key = f"structural_evidence_{position}"
        arrays[key] = evidence.mask
        structural_records[name] = {"array_key": key, "manifest": evidence.to_manifest()}
    buffer = BytesIO()
    np.savez_compressed(buffer, **arrays)
    payload = buffer.getvalue()
    manifest = {
        "schema": BUNDLE_SCHEMA, "case_id": case.case_id, "revision": case.revision,
        "case_semantic_hash": case.semantic_hash, "frame": case.frame,
        "source_refs": _jsonable(case.source_refs), "context": None if case.context is None else case.context.to_dict(),
        "metadata": _jsonable(case.metadata), "unknowns": list(case.unknowns), "array_index": index,
        "array_sha256": sha256(payload).hexdigest(), "artifacts": _jsonable(artifacts or {}),
    }
    if structural_records:
        manifest["structural_evidence"] = structural_records
    encoded_manifest = json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False).encode("utf-8")
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=f".{destination.name}.", delete=False) as handle:
            temporary = Path(handle.name)
        with ZipFile(temporary, "w", compression=ZIP_DEFLATED) as archive:
            archive.writestr("manifest.json", encoded_manifest)
            archive.writestr("arrays.npz", payload)
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return destination


def _read_bundle(path: str | Path) -> tuple[dict[str, Any], bytes]:
    try:
        with ZipFile(path, "r") as archive:
            if sorted(archive.namelist()) != ["arrays.npz", "manifest.json"]:
                raise ImagingError("INVALID_BUNDLE_MEMBERS", "Bundle must contain exactly manifest.json and arrays.npz.")
            if archive.getinfo("manifest.json").file_size > 16 * 1024 * 1024:
                raise ImagingError("BUNDLE_SIZE_LIMIT", "Manifest exceeds the 16 MiB local import limit.")
            if archive.getinfo("arrays.npz").file_size > 2 * 1024 ** 3:
                raise ImagingError("BUNDLE_SIZE_LIMIT", "Array payload exceeds the 2 GiB local import limit.")
            manifest = json.loads(archive.read("manifest.json"))
            payload = archive.read("arrays.npz")
        if manifest.get("schema") != BUNDLE_SCHEMA:
            raise ImagingError("UNSUPPORTED_BUNDLE_VERSION", "This case uses an unsupported bundle schema.")
        if sha256(payload).hexdigest() != manifest.get("array_sha256"):
            raise ImagingError("BUNDLE_HASH_MISMATCH", "Stored imaging bytes do not match the manifest hash.")
        return manifest, payload
    except ImagingError:
        raise
    except Exception as exc:
        raise ImagingError("UNREADABLE_BUNDLE", f"Cannot read case bundle: {exc}") from exc


def load_case(path: str | Path) -> CaseData:
    """Reopen a bundle and verify bytes, model invariants, and semantic identity."""
    manifest, payload = _read_bundle(path)
    try:
        with ZipFile(BytesIO(payload)) as nested:
            if sum(info.file_size for info in nested.infolist()) > 2 * 1024 ** 3:
                raise ImagingError("BUNDLE_SIZE_LIMIT", "Expanded arrays exceed the 2 GiB local import limit.")
        with np.load(BytesIO(payload), allow_pickle=False) as arrays:
            groups = {group: {name: arrays[key] for name, key in manifest["array_index"][group].items()}
                      for group in ("compartments", "source_compartments")}
            structural = {}
            if manifest.get("structural_evidence"):
                from .structural_evidence import StructuralEvidence
                structural = {
                    name: StructuralEvidence.from_manifest(record["manifest"], mask=arrays[record["array_key"]])
                    for name, record in manifest["structural_evidence"].items()
                }
            extra = {"structural_evidence": structural} if structural else {}
            case = CaseData(case_id=manifest["case_id"], mri=arrays["mri"], affine=arrays["affine"],
                            compartments=groups["compartments"], source_compartments=groups["source_compartments"],
                            brain_mask=arrays["brain_mask"] if "brain_mask" in arrays.files else None,
                            source_refs=tuple(SourceRef(**reference) for reference in manifest["source_refs"]),
                            context=PatientContext.from_dict(manifest["context"]) if manifest["context"] is not None else None,
                            frame=manifest["frame"], revision=manifest["revision"], unknowns=tuple(manifest["unknowns"]),
                            metadata=manifest["metadata"], **extra)
        if case.semantic_hash != manifest["case_semantic_hash"]:
            raise ImagingError("CASE_HASH_MISMATCH", "Reopened case differs from its saved semantic identity.")
        return case
    except ImagingError:
        raise
    except Exception as exc:
        raise ImagingError("INVALID_BUNDLE_DATA", f"Case arrays or metadata are invalid: {exc}") from exc


def read_case_artifacts(path: str | Path) -> dict[str, Any]:
    """Read saved plan/settings JSON; callers must verify each plan's case hash."""
    return _read_bundle(path)[0]["artifacts"]


def import_brain_extraction_evidence(
    case: CaseData, *, source_image_path: str | Path, mask_path: str | Path,
    report_path: str | Path, variant: str, evidence_id: str | None = None,
) -> CaseData:
    """Attach a pinned, unreviewed extraction artifact as separate display evidence.

    Source-file bytes, current MRI values/frame, report identity, checkpoint and
    mask bytes/grid must agree. The operation never sets ``brain_mask``, accepts
    an anatomical review, or enables cortical access. Current target inclusion is
    recomputed; a source QC report is not silently reused after target edits.
    """
    from .brain_extraction import ASSETS
    from .structural_evidence import StructuralEvidence, structural_frame_hash

    if variant not in {"main", "nocsf"}:
        raise ImagingError("UNKNOWN_EXTRACTION_VARIANT", "Choose the recorded main or no-CSF extraction variant.")
    source_image_path, mask_path, report_path = map(Path, (source_image_path, mask_path, report_path))
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if not isinstance(report, dict):
            raise ValueError("Report must be a JSON object")
    except (OSError, ValueError) as exc:
        raise ImagingError("UNREADABLE_EXTRACTION_REPORT", f"Cannot read extraction report: {exc}") from exc
    source_hash, mask_file_hash, report_hash = map(file_sha256, (source_image_path, mask_path, report_path))
    source_refs = [item for item in case.source_refs if item.source_id == "structural"]
    if len(source_refs) != 1 or source_refs[0].sha256 != source_hash:
        raise ImagingError("EXTRACTION_SOURCE_MISMATCH", "Extraction source does not match the case's hashed structural input.")
    if report.get("source_t1_sha256") != source_hash:
        raise ImagingError("EXTRACTION_SOURCE_MISMATCH", "Extraction report belongs to another source image.")
    source_qc = inspect_nifti(source_image_path)
    source_data = _read_finite(source_image_path)
    source_affine = np.asarray(source_qc["affine_ras_mm"])
    if case.frame == "LPS+":
        source_affine = np.diag([-1, -1, 1, 1]) @ source_affine
    if (not np.array_equal(source_data, case.mri) or not np.allclose(source_affine, case.affine, rtol=0, atol=1e-5)):
        raise ImagingError("EXTRACTION_CASE_IMAGE_CHANGED", "Case MRI values or physical frame changed since the extracted source.")
    try:
        variant_record = report["variants"][variant]
        inference = variant_record["inference"]
        model = inference["model"]
        checkpoint_name = "synthstrip.1.pt" if variant == "main" else "synthstrip.nocsf.1.pt"
        checkpoint_hash = model["files"][checkpoint_name]["sha256"]
        expected_mask_hash = inference["artifact_hashes"][f"{variant}_mask.nii.gz"]
    except (KeyError, TypeError) as exc:
        raise ImagingError("EXTRACTION_REPORT_INCOMPLETE", "Report lacks the selected source/model/output identity.") from exc
    if inference.get("input_sha256") != source_hash or inference.get("failure") is not None or inference.get("exit_code") != 0:
        raise ImagingError("EXTRACTION_INFERENCE_UNVERIFIED", "Only a successful inference on this exact source may be attached.")
    if (model.get("name") != "SynthStrip" or model.get("variant") != variant
            or checkpoint_hash != ASSETS[checkpoint_name][1]):
        raise ImagingError("EXTRACTION_MODEL_MISMATCH", "Report checkpoint is not the declared pinned extraction model.")
    if expected_mask_hash != mask_file_hash:
        raise ImagingError("EXTRACTION_MASK_HASH_MISMATCH", "Mask bytes differ from the extraction report.")
    if any(inference.get(key) is not False for key in ("brain_reviewed", "cortical_access_permitted")):
        raise ImagingError("EXTRACTION_REPORT_REVIEW_CLAIM", "An inference report cannot supply an accepted anatomical or cortical review.")
    for scope in (report, variant_record.get("qc", {})):
        if not isinstance(scope, Mapping) or any(scope.get(key, False) is not False
                                                 for key in ("brain_reviewed", "cortex_localized", "cortical_access_permitted")):
            raise ImagingError("EXTRACTION_REPORT_REVIEW_CLAIM", "Extraction metadata cannot grant anatomical or cortical acceptance.")
    mask_data, _ = _aligned_mask(mask_path, source_qc)
    if not np.isin(mask_data, [0, 1]).all() or not mask_data.any():
        raise ImagingError("INVALID_EXTRACTION_MASK", "Extraction mask must be a nonempty binary native-grid array.")
    mask = mask_data.astype(bool)
    target = np.zeros(case.mri.shape, dtype=bool)
    for compartment in case.compartments.values():
        target |= compartment
    outside = int(np.count_nonzero(target & ~mask))
    flags = list(variant_record.get("qc", {}).get("flags", []))
    if outside:
        flags.append("CURRENT_TARGET_ANNOTATION_OUTSIDE_ESTIMATED_ENVELOPE")
    evidence = StructuralEvidence(
        evidence_id=evidence_id or f"synthstrip_{variant}_{report_hash[:12]}", mask=mask,
        source_image_hash=array_digest(case.mri), source_frame_hash=structural_frame_hash(case),
        source_file_sha256=source_hash, model_sha256=checkpoint_hash, run_sha256=report_hash,
        method=f"SynthStrip version 1 {variant}; native-grid model-estimated whole-brain envelope",
        provenance="estimated", metadata={
            "evidence_kind": "whole_brain_envelope", "variant": variant,
            "mask_source_uri": mask_path.resolve().as_uri(), "mask_file_sha256": mask_file_hash,
            "report_source_uri": report_path.resolve().as_uri(), "model": model,
            "inference_configuration": inference.get("configuration", {}),
            "runtime_versions": inference.get("runtime_versions", {}),
            "executed_runner_sha256": inference.get("executed_runner_sha256"),
            "source_qc": variant_record.get("qc", {}), "qc_flags": sorted(set(flags)),
            "current_target_union_hash": array_digest(target), "current_target_voxels": int(target.sum()),
            "current_target_annotation_outside_voxels": outside,
            "current_target_inclusion_meaning": "annotation inclusion only; not extraction accuracy or removal",
            "cortex_localized": False, "clinical_deficit_probability": None,
        },
    )
    existing = dict(case.structural_evidence)
    if evidence.evidence_id in existing:
        if existing[evidence.evidence_id].to_manifest() == evidence.to_manifest():
            return case
        raise ImagingError("EXTRACTION_EVIDENCE_ID_EXISTS", "Evidence ID already identifies different data or review; retain a separate version.")
    existing[evidence.evidence_id] = evidence
    return case.revised(structural_evidence=existing)


def create_synthetic_case(shape: tuple[int, int, int] = (64, 64, 64)) -> CaseData:
    """Deterministic, explicitly synthetic imaging fixture; never a patient case.

    Deliberately has a superficial target and separate radiological-like labels.
    No simulated target represents a biological prescription for removal.
    """
    if len(shape) != 3 or any(size < 24 or size > 256 for size in shape):
        raise ValueError("Synthetic fixture shape must contain three sizes between 24 and 256.")
    grid = np.indices(shape, dtype=np.float32)
    center = (np.asarray(shape) - 1) / 2
    normalized = [(grid[axis] - center[axis]) / (shape[axis] * fraction)
                  for axis, fraction in enumerate((0.43, 0.40, 0.45))]
    radius = sum(axis ** 2 for axis in normalized)
    brain = radius <= 1
    texture = 8 * np.sin(grid[0] * 0.40) * np.cos(grid[1] * 0.32) + 5 * np.sin(grid[2] * 0.48)
    mri = np.where(brain, 75 + 35 * (1 - radius) + texture, 0).astype(np.float32)
    lesion_center = np.asarray(shape) * np.array([0.60, 0.57, 0.74])
    lesion = sum(((grid[axis] - lesion_center[axis]) / (shape[axis] * scale)) ** 2
                 for axis, scale in enumerate((0.13, 0.15, 0.16)))
    core = (lesion < 0.26) & brain
    enhancing = (lesion >= 0.26) & (lesion < 0.64) & brain
    flair = (lesion >= 0.64) & (lesion < 1.0) & brain
    mri[core] = 48
    mri[enhancing] = 170
    mri[flair] += 20
    affine = np.eye(4)
    affine[:3, 3] = -center
    return CaseData(case_id="synthetic_geometry_demo", mri=mri,
                    compartments={"enhancing": enhancing, "nonenhancing_core": core, "flair_abnormality": flair},
                    affine=affine, brain_mask=brain,
                    source_refs=(SourceRef(source_id="synthetic_geometry_v1", uri="synthetic://resectionlab/geometry-v1",
                                           license="CC0-1.0", native_frame="RAS+", provenance="simulated"),),
                    unknowns=("not_a_patient", "vascular_anatomy_unmodeled", "functional_anatomy_simulated_only",
                              "uncalibrated_tissue_response", "skull_anatomy_unmodeled"),
                    metadata={"is_synthetic": True, "benchmark_track": "synthetic_geometry",
                              "input_mode": "synthetic", "clinical_use_status": "research_only",
                              "clinical_deficit_probability": None,
                              "description": "Analytic fixture for geometry and interface testing; no patient imaging.",
                              "annotation_review": "synthetic_ground_truth", "voxel_units": "mm"})
