#!/usr/bin/env python3
"""Independently check saved extraction artifacts; never approve anatomy or access.

This audit uses NiBabel orientation reindexing and independent connected-component
bookkeeping rather than the production fractional-annotation/QC implementations.
It does not rerun the network and does not measure segmentation accuracy.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import itertools
import json
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage


def digest(path: Path) -> str:
    result = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def verify_file(path: Path, expected: str) -> None:
    if digest(path) != expected:
        raise ValueError(f"Pinned file hash mismatch: {path.name}")


def _corners(shape: tuple[int, ...]) -> np.ndarray:
    return np.array(list(itertools.product(*[(0, size - 1) for size in shape])))


def read_native(path: Path, reference: nib.Nifti1Image, *, binary: bool) -> tuple[np.ndarray, dict]:
    volume = nib.load(path)
    if volume.shape != reference.shape:
        raise ValueError("Native artifact shape differs from source T1")
    if volume.header.get_xyzt_units()[0] != "mm":
        raise ValueError("Native artifact lacks explicit millimeter units")
    qform, qcode = volume.get_qform(coded=True)
    sform, scode = volume.get_sform(coded=True)
    if not (qcode or scode):
        raise ValueError("Native artifact has no coded spatial transform")
    if qcode and scode and not np.allclose(qform, sform, rtol=0, atol=0.01):
        raise ValueError("Native artifact qform/sform conflict")
    points = _corners(reference.shape)
    discrepancy = nib.affines.apply_affine(volume.affine, points) - nib.affines.apply_affine(reference.affine, points)
    maximum_error = float(np.max(np.linalg.norm(discrepancy, axis=1)))
    if not np.isfinite(maximum_error) or maximum_error > 0.01:
        raise ValueError("Native artifact physical voxel corners differ from source T1")
    values = volume.get_fdata(dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError("Native artifact contains nonfinite values")
    if binary and (not np.isin(values, [0, 1]).all() or not values.any()):
        raise ValueError("Mask must be binary and nonempty")
    return (values.astype(bool) if binary else values), {
        "shape": list(volume.shape), "qform_code": int(qcode), "sform_code": int(scode),
        "orientation": list(nib.aff2axcodes(volume.affine)), "physical_units": "mm",
        "maximum_native_corner_difference_mm": maximum_error,
    }


def source_annotation(path: Path, reference: nib.Nifti1Image, *, threshold: float) -> np.ndarray:
    """Independent axis reindexing: no inferred registration or hidden threshold."""
    source = nib.load(path)
    orientation = nib.orientations.ornt_transform(nib.io_orientation(source.affine), nib.io_orientation(reference.affine))
    reindexed = nib.orientations.apply_orientation(source.get_fdata(dtype=np.float32), orientation)
    affine = source.affine @ nib.orientations.inv_ornt_aff(orientation, source.shape)
    if reindexed.shape != reference.shape:
        raise ValueError("Source annotation shape differs after axis reindexing")
    points = _corners(reference.shape)
    delta = nib.affines.apply_affine(affine, points) - nib.affines.apply_affine(reference.affine, points)
    if np.max(np.linalg.norm(delta, axis=1)) > 0.001:
        raise ValueError("Source annotation needs registration beyond an axis permutation")
    if not np.isfinite(reindexed).all() or not 0 < threshold <= 1:
        raise ValueError("Invalid source annotation or threshold")
    return reindexed >= threshold


def measure(mask: np.ndarray, distance: np.ndarray, tumor: np.ndarray, affine: np.ndarray, *, border_mm: float) -> dict:
    """Recompute mask derivation, annotation discrepancy and geometry from arrays."""
    if mask.ndim != 3 or mask.dtype != bool or tumor.dtype != bool or mask.shape != distance.shape or mask.shape != tumor.shape:
        raise ValueError("Audit arrays must share a grid with boolean masks")
    if not mask.any() or not tumor.any() or not np.isfinite(distance).all():
        raise ValueError("Audit needs nonempty masks and finite predicted distance values")
    affine = np.asarray(affine)
    if (affine.shape != (4, 4) or not np.isfinite(affine).all()
            or abs(np.linalg.det(affine[:3, :3])) < 1e-12):
        raise ValueError("Audit requires a finite invertible physical affine")
    labels, components = ndimage.label(distance < border_mm)
    if components == 0:
        raise ValueError("Predicted distance threshold has no positive mask support")
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    reconstructed = ndimage.binary_fill_holes(labels == sizes.argmax())
    mismatch = int(np.count_nonzero(reconstructed != mask))
    if mismatch:
        raise ValueError(f"Saved mask differs from declared SDT/component/fill derivation at {mismatch} voxels")
    outside = tumor & ~mask
    positions = np.argwhere(outside)
    outside_world = nib.affines.apply_affine(affine, positions) if len(positions) else None
    faces = np.zeros_like(mask)
    for axis in range(3):
        for edge in (0, -1):
            selection = [slice(None)] * 3
            selection[axis] = edge
            faces[tuple(selection)] = True
    return {
        "mask_voxels": int(mask.sum()), "volume_ml": float(mask.sum() * abs(np.linalg.det(affine[:3, :3])) / 1000),
        "connected_components": int(ndimage.label(mask)[1]), "input_face_contact_voxels": int(np.count_nonzero(mask & faces)),
        "source_annotation_voxels": int(tumor.sum()), "source_annotation_outside_voxels": int(outside.sum()),
        "source_annotation_inclusion_fraction": float(np.count_nonzero(tumor & mask) / tumor.sum()),
        "outside_annotation_voxel_bbox": None if not len(positions) else [positions.min(axis=0).tolist(), positions.max(axis=0).tolist()],
        "outside_annotation_world_ras_mm_bbox": None if outside_world is None else [outside_world.min(axis=0).tolist(), outside_world.max(axis=0).tolist()],
        "reconstructed_mask_mismatch_voxels": mismatch,
        "predicted_distance_range_mm": [float(distance.min()), float(distance.max())],
        "flags": ["SOURCE_ANNOTATION_OUTSIDE_ESTIMATED_ENVELOPE"] if outside.any() else [],
        "meaning": "engineering geometry and source-annotation inclusion; not anatomical accuracy",
    }


def audit(repository: Path, experiment: Path) -> dict:
    source_manifest = json.loads((repository / "manifests/btc_acquisition.json").read_text())
    source_root = repository / "data/diffusion_source/ds001226-v5.0.1"
    source_files = {item["path"]: item for item in source_manifest["files"]}
    t1_relative = "sub-PAT28/ses-preop/anat/sub-PAT28_ses-preop_T1w.nii.gz"
    annotation_relative = "derivatives/tumor_masks/sub-PAT28/anat/sub-PAT28_space_T1_label-tumor.nii"
    t1_path, annotation_path = source_root / t1_relative, source_root / annotation_relative
    for relative in (t1_relative, annotation_relative):
        verify_file(source_root / relative, source_files[relative]["sha256"])
    report_path = experiment / "brain_extraction_report.json"
    report = json.loads(report_path.read_text())
    if report["source_t1_sha256"] != source_files[t1_relative]["sha256"]:
        raise ValueError("Extraction report belongs to a different source image")
    if report["source_annotation"]["sha256"] != source_files[annotation_relative]["sha256"]:
        raise ValueError("Extraction report belongs to a different source annotation")
    verify_file(experiment / "implementation_snapshot.py", report["implementation_sha256"])
    reference = nib.load(t1_path)
    threshold = report["source_annotation"]["geometry_and_threshold"]["threshold"]
    tumor = source_annotation(annotation_path, reference, threshold=threshold)
    model_manifest = json.loads((repository / "docs/synthstrip-model-manifest.json").read_text())
    variants = {}
    for name in ("main", "nocsf"):
        record = report["variants"][name]["inference"]
        if record["failure"] is not None or record["exit_code"] != 0 or record["input_sha256"] != digest(t1_path):
            raise ValueError("Inference failed or used a different input")
        if record["brain_reviewed"] or record["cortical_access_permitted"]:
            raise ValueError("Inference output must remain unreviewed and unavailable for cortical access")
        for filename, stated in record["model"]["files"].items():
            pinned = model_manifest["files"][filename]
            if stated["sha256"] != pinned["sha256"]:
                raise ValueError("Recorded model asset differs from the separately pinned model manifest")
            verify_file(repository / "data/models/synthstrip-v1" / filename, pinned["sha256"])
        verify_file(experiment / f"{name}_synthstrip_mps.py", record["executed_runner_sha256"])
        for filename, expected in record["artifact_hashes"].items():
            verify_file(experiment / filename, expected)
        mask, geometry = read_native(experiment / f"{name}_mask.nii.gz", reference, binary=True)
        distance, _ = read_native(experiment / f"{name}_distance_mm.nii.gz", reference, binary=False)
        values = measure(mask, distance, tumor, reference.affine, border_mm=record["configuration"]["border_mm"])
        repeated = []
        for run in ("PAT28-mps-v1", "PAT28-mps-v2", "PAT28-mps-v3"):
            if run == experiment.name:
                continue
            repeat_mask, _ = read_native(experiment.parent / run / f"{name}_mask.nii.gz", reference, binary=True)
            repeat_distance, _ = read_native(experiment.parent / run / f"{name}_distance_mm.nii.gz", reference, binary=False)
            repeated.append({"run": run, "identical_native_mask": bool(np.array_equal(mask, repeat_mask)),
                             "maximum_predicted_distance_difference_mm": float(np.max(np.abs(distance - repeat_distance)))})
        variants[name] = {"geometry": geometry, "independent_measurements": values, "repeats": repeated,
                          "output_hashes": record["artifact_hashes"], "source_model_and_runner_hashes_verified": True,
                          "review_status": "review_required", "brain_reviewed": False,
                          "cortex_localized": False, "cortical_access_permitted": False}
    return {
        "schema_version": 1, "audit_kind": "independent_saved_artifact_engineering_qc",
        "auditor_implementation_sha256": digest(Path(__file__)), "source_report_sha256": digest(report_path),
        "source_t1_sha256": digest(t1_path), "source_annotation_sha256": digest(annotation_path),
        "source_annotation_threshold": threshold, "variants": variants,
        "visual_qc_image": str((experiment / "extraction_qc.png").relative_to(repository)),
        "visual_qc_sha256": digest(experiment / "extraction_qc.png"),
        "brain_reviewed": False, "cortex_localized": False, "cortical_access_permitted": False,
        "clinical_deficit_probability": None, "review_status": "review_required",
        "limitations": ["No expert anatomical review or independent brain/cortex labels.",
                        "Stored repeats are reproducibility checks, not population accuracy or latency estimates.",
                        "Predicted SDT is a model output, not an independently measured surgical clearance field.",
                        "Whole-brain envelopes include inferior structures and cannot define a cerebral access window.",
                        "Source annotation is a threshold scenario; complete inclusion is not segmentation accuracy."]}


def main() -> None:
    repository = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, default=repository / "artifacts/brain-extraction/PAT28-mps-v4")
    parser.add_argument("--report", type=Path, default=repository / "docs/brain-extraction-independent-qc.json")
    args = parser.parse_args()
    result = audit(repository, args.experiment.resolve())
    if args.report.resolve().is_relative_to(args.experiment.resolve()) or args.report.resolve().is_relative_to(repository / "data"):
        parser.error("Write the independent audit outside immutable source and experiment directories")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"report": str(args.report), "review_status": "review_required",
                      "annotation_outside_voxels": {name: item["independent_measurements"]["source_annotation_outside_voxels"]
                                                   for name, item in result["variants"].items()}}))


if __name__ == "__main__":
    main()
