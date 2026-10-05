"""Convert the pinned real ReMIND-001 native MR/SEG grids without cross-frame registration.

This intentionally retains separate T1 and T2 frames. SEG source correspondence is
reported as matching frame plus release description, not invented SOP references.
No resampling, cortical-access approval, normalization or training is performed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pydicom


def regular_affine(orientation, pixel_spacing, positions):
    """Return sorted slice indices and XYZ-to-RAS affine from DICOM LPS geometry."""
    orientation = np.asarray(orientation, dtype=np.float64)
    pixel_spacing = np.asarray(pixel_spacing, dtype=np.float64)
    positions = np.asarray(positions, dtype=np.float64)
    if (orientation.shape != (6,) or pixel_spacing.shape != (2,) or positions.ndim != 2
            or positions.shape[1] != 3 or len(positions) < 2
            or not all(np.isfinite(a).all() for a in [orientation, pixel_spacing, positions])
            or np.any(pixel_spacing <= 0)):
        raise ValueError("Malformed physical grid")
    x, y = orientation[:3], orientation[3:]
    if abs(np.linalg.norm(x) - 1) > 1e-5 or abs(np.linalg.norm(y) - 1) > 1e-5 or abs(x @ y) > 1e-5:
        raise ValueError("Invalid DICOM orientation basis")
    normal = np.cross(x, y)
    normal /= np.linalg.norm(normal)
    order = np.argsort(positions @ normal, kind="stable")
    sorted_positions = positions[order]
    # Fit all decimal-serialized positions, rather than amplifying rounding in
    # two endpoint slices. The maximum physical residual is still strictly gated.
    design = np.c_[np.ones(len(positions)), np.arange(len(positions))]
    origin, increment = np.linalg.lstsq(design, sorted_positions, rcond=None)[0]
    if increment @ normal <= 0:
        raise ValueError("Duplicate or reversed image planes")
    residual = sorted_positions - (origin + np.arange(len(positions))[:, None] * increment)
    max_residual = float(np.max(np.linalg.norm(residual, axis=1)))
    if max_residual > 1e-3:
        raise ValueError("Nonuniform image planes exceed 0.001 mm tolerance")
    affine_lps = np.eye(4)
    affine_lps[:3, 0] = x * pixel_spacing[1]
    affine_lps[:3, 1] = y * pixel_spacing[0]
    affine_lps[:3, 2] = increment
    affine_lps[:3, 3] = origin
    affine_ras = np.diag([-1., -1., 1., 1.]) @ affine_lps
    return order, affine_ras, max_residual


def shared_identity(datasets, series):
    expected = {"PatientID": "ReMIND-001", "SeriesInstanceUID": series["SeriesInstanceUID"],
                "StudyInstanceUID": series["StudyInstanceUID"], "Modality": series["Modality"]}
    for ds in datasets:
        if any(str(getattr(ds, key, "")) != value for key, value in expected.items()):
            raise ValueError("DICOM source identity differs from pinned series")
    frames = {str(ds.FrameOfReferenceUID) for ds in datasets}
    sops = [str(ds.SOPInstanceUID) for ds in datasets]
    if len(frames) != 1 or len(set(sops)) != len(sops):
        raise ValueError("Mixed frames or duplicate SOP instances")
    return frames.pop(), sops


def convert_mr(datasets):
    first = datasets[0]
    orientation = np.asarray(first.ImageOrientationPatient, dtype=float)
    spacing = np.asarray(first.PixelSpacing, dtype=float)
    for ds in datasets:
        if (ds.Rows != first.Rows or ds.Columns != first.Columns
                or not np.allclose(ds.ImageOrientationPatient, orientation, rtol=0, atol=1e-6)
                or not np.allclose(ds.PixelSpacing, spacing, rtol=0, atol=1e-6)):
            raise ValueError("Inconsistent MR slice grid")
    order, affine, residual = regular_affine(orientation, spacing,
                                           [ds.ImagePositionPatient for ds in datasets])
    arrays = [datasets[i].pixel_array.astype(np.float32) * float(getattr(datasets[i], "RescaleSlope", 1))
              + float(getattr(datasets[i], "RescaleIntercept", 0)) for i in order]
    # DICOM samples are row, column; NIfTI here is column, row, slice (XYZ).
    return np.stack(arrays, axis=2).transpose(1, 0, 2), affine, {"slice_position_max_residual_mm": residual}


def convert_seg(ds):
    if ds.SegmentationType != "BINARY" or len(ds.SegmentSequence) != 1:
        raise ValueError("This bounded converter handles one binary source segment")
    shared = ds.SharedFunctionalGroupsSequence[0]
    orientation = shared.PlaneOrientationSequence[0].ImageOrientationPatient
    spacing = shared.PixelMeasuresSequence[0].PixelSpacing
    segment = ds.SegmentSequence[0]
    frames = ds.PerFrameFunctionalGroupsSequence
    if any(f.SegmentIdentificationSequence[0].ReferencedSegmentNumber != segment.SegmentNumber for f in frames):
        raise ValueError("Mixed SEG segment numbers")
    order, affine, residual = regular_affine(orientation, spacing,
                                           [f.PlanePositionSequence[0].ImagePositionPatient for f in frames])
    data = ds.pixel_array[order].transpose(2, 1, 0).astype(np.uint8)
    if not np.isin(data, [0, 1]).all() or not data.any():
        raise ValueError("Empty or nonbinary SEG")
    references = [str(e.value) for e in ds.iterall() if e.keyword == "ReferencedSOPInstanceUID"]
    return data, affine, {"slice_position_max_residual_mm": residual,
                         "segment_label": str(segment.SegmentLabel),
                         "segment_description": str(getattr(segment, "SegmentDescription", "")),
                         "algorithm_type": str(segment.SegmentAlgorithmType),
                         "algorithm_name": str(getattr(segment, "SegmentAlgorithmName", "")),
                         "referenced_sop_instance_uids": references,
                         "explicit_source_sop_references_present": bool(references),
                         "outside_native_seg_grid_coverage": "unassessed; not encoded as verified zero",
                         "positive_voxels": int(np.count_nonzero(data)),
                         "positive_volume_mm3": float(np.count_nonzero(data) * abs(np.linalg.det(affine[:3, :3])))}


def convert(manifest_path: Path, source_root: Path, receipt_path: Path, output: Path):
    manifest_raw = manifest_path.read_bytes()
    manifest = json.loads(manifest_raw)
    receipt = json.loads(receipt_path.read_bytes())
    if (receipt["status"] != "complete" or receipt["manifest_sha256"] != hashlib.sha256(manifest_raw).hexdigest()
            or receipt["actual_bytes_verified"] != 59_163_552):
        raise ValueError("A complete matching source acquisition is required")
    hashes = {r["key"]: r["sha256"] for r in receipt["objects"]}
    if (output / "native-conversion.json").exists():
        raise ValueError("Refusing to overwrite native conversion")
    output.mkdir(parents=True, exist_ok=True)
    results = []
    for series in manifest["series"]:
        datasets = []
        for obj in series["objects"]:
            path = source_root / obj["key"]
            if (path.stat().st_size != obj["bytes"] or hashlib.sha256(path.read_bytes()).hexdigest() != hashes[obj["key"]]):
                raise ValueError("Source bytes changed since acquisition")
            datasets.append(pydicom.dcmread(path))
        frame, sops = shared_identity(datasets, series)
        if series["Modality"] == "MR":
            data, affine, details = convert_mr(datasets)
        else:
            if len(datasets) != 1:
                raise ValueError("Unexpected SEG instance count")
            data, affine, details = convert_seg(datasets[0])
        image = nib.Nifti1Image(data, affine)
        image.header.set_xyzt_units("mm")
        image.set_sform(affine, code=1)
        image.set_qform(affine, code=1)
        path = output / (series["role"] + ".nii.gz")
        nib.save(image, path)
        loaded = nib.load(path)
        corners = np.array(np.meshgrid(*[(0, n - 1) for n in data.shape], indexing="ij")).reshape(3, -1).T
        points = np.c_[corners, np.ones(len(corners))]
        corner_error = float(np.max(np.linalg.norm((points @ (loaded.affine - affine).T)[:, :3], axis=1)))
        if corner_error > 1e-3 or not np.array_equal(np.asarray(loaded.dataobj), data):
            raise ValueError("NIfTI roundtrip changed samples or physical frame")
        results.append({"role": series["role"], "file": path.name,
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "series_instance_uid": series["SeriesInstanceUID"], "sop_instance_uids": sops,
                        "frame_of_reference_uid": frame, "shape_xyz": list(data.shape),
                        "affine_xyz_to_ras_mm": affine.tolist(), "roundtrip_max_corner_error_mm": corner_error,
                        "source_pixel_values_preserved": True, **details})
    by_role = {r["role"]: r for r in results}
    for seg, mr in [("cerebrum_annotation", "structural_t1ce"), ("tumor_annotation", "structural_t2")]:
        if by_role[seg]["frame_of_reference_uid"] != by_role[mr]["frame_of_reference_uid"]:
            raise ValueError("SEG and release-described source MRI have different frames")
        by_role[seg]["source_link_evidence"] = {
            "release_series_description_names": mr, "frame_uid_matches_named_mri": True,
            "explicit_sop_reference_verified": by_role[seg]["explicit_source_sop_references_present"],
            "status": "source description plus shared frame; absent explicit SOP links remain a limitation"}
    record = {"schema": "resectionlab.remind-native-conversion.v1", "patient_id": "ReMIND-001",
              "patient_group": "ReMIND:001", "role": "development_annotation_assisted_geometry",
              "manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(),
              "acquisition_receipt_sha256": hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
              "converter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "native_series": results, "cross_sequence_registration": "not performed",
              "cross_sequence_frame_uids_equal": by_role["structural_t1ce"]["frame_of_reference_uid"] == by_role["structural_t2"]["frame_of_reference_uid"],
              "expert_review": "unreviewed", "cortical_access": "unapproved",
              "functional_and_vascular_coverage": "absent; cannot imply zero risk",
              "planner_case_ready": False}
    (output / "native-conversion.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--acquisition-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = convert(args.manifest, args.source, args.acquisition_receipt, args.output)
    print(json.dumps({"planner_case_ready": result["planner_case_ready"],
                      "cross_sequence_frame_uids_equal": result["cross_sequence_frame_uids_equal"],
                      "series": [{k: r[k] for k in ["role", "shape_xyz", "roundtrip_max_corner_error_mm"]}
                                 for r in result["native_series"]]}, indent=2))


if __name__ == "__main__":
    main()
