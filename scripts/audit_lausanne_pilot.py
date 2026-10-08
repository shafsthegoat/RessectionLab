#!/usr/bin/env python3
"""Independently recompute source-byte and geometry metadata checks.

No importer transforms, image registration, model or label generation. Axis
agreement permits storage permutation/sign changes; it does not prove position
agreement or cross-scan registration. Inspect only the declared TRAIN pilot.
"""
from __future__ import annotations

import hashlib
from itertools import permutations, product
import json
from pathlib import Path

import nibabel as nib
import numpy as np

from acquire_lausanne_pilot import DATA, MANIFEST, RESULT, ROOT, preserve, json_bytes, sha256_file


def main() -> None:
    manifest = json.loads(MANIFEST.read_bytes())
    recorded = json.loads(RESULT.read_bytes())
    if sha256_file(MANIFEST) != recorded["source_manifest_sha256"]:
        raise ValueError("Acquisition declaration changed")
    if manifest["subject"] != "sub-000" or manifest["role"] != "TRAIN":
        raise ValueError("Audit is restricted to the declared TRAIN pilot")
    checks = []
    for expected, receipt in zip(manifest["files"], recorded["files"], strict=True):
        path = DATA / expected["path"]
        if receipt["path"] != expected["path"] or path.stat().st_size != expected["bytes"]:
            raise ValueError("Source identity or size mismatch")
        if sha256_file(path) != receipt["sha256"]:
            raise ValueError("Acquired bytes changed")
        if expected.get("expected_md5"):
            with path.open("rb") as handle:
                if hashlib.file_digest(handle, "md5").hexdigest() != expected["expected_md5"]:
                    raise ValueError("Published source MD5 mismatch")
        elif receipt["sha256"] != expected["sha256"]:
            raise ValueError("Source sidecar checksum mismatch")
        checks.append({"path": expected["path"], "sha256": receipt["sha256"], "source_byte_checks": "passed"})
    geometry = []
    for expected in manifest["files"][:2]:
        path = DATA / expected["path"]
        image = nib.load(path)
        if len(image.shape) != 3 or int(np.prod(image.shape)) * 4 > 512 * 1024**2:
            raise ValueError("Unexpected dimensions or decoded memory size")
        if image.header.get_xyzt_units()[0] != "mm":
            raise ValueError("Expected explicit millimetre units")
        affine = image.affine
        spacing = np.linalg.norm(affine[:3, :3], axis=0)
        axes = affine[:3, :3] / spacing
        if not np.isfinite(affine).all() or not np.allclose(axes.T @ axes, np.eye(3), atol=1e-4):
            raise ValueError("Invalid or sheared source affine")
        qform, qcode = image.get_qform(coded=True)
        sform, scode = image.get_sform(coded=True)
        corners = np.array([(*point, 1) for point in product(*[(0, n - 1) for n in image.shape])])
        corner_error = (float(np.linalg.norm(((qform - sform) @ corners.T)[:3], axis=0).max())
                        if qcode and scode else None)
        sidecar = json.loads(path.with_name(path.name.removesuffix(".nii.gz") + ".json").read_bytes())
        directions = np.array(sidecar["ImageOrientationPatientDICOM"], dtype=float).reshape(2, 3).T
        directions = np.diag([-1., -1., 1.]) @ directions  # DICOM LPS -> NIfTI RAS
        directions = np.column_stack((directions, np.cross(directions[:, 0], directions[:, 1])))
        directions /= np.linalg.norm(directions, axis=0)
        if not np.allclose(directions.T @ directions, np.eye(3), atol=1e-4):
            raise ValueError("Sidecar orientation directions are not orthogonal")
        angles = np.degrees(np.arccos(np.clip(np.abs(axes.T @ directions), 0, 1)))
        matching = min(permutations(range(3)), key=lambda order: (max(angles[i, order[i]] for i in range(3)), order))
        errors = [float(angles[i, matching[i]]) for i in range(3)]
        values = image.get_fdata(dtype=np.float32)
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite acquired image")
        item = {"path": expected["path"], "shape": list(image.shape), "orientation": list(nib.aff2axcodes(affine)),
                "spacing_mm": spacing.tolist(), "qform_code": int(qcode), "sform_code": int(scode),
                "maximum_qform_sform_corner_disagreement_mm": corner_error,
                "finite_voxels": int(values.size), "nonzero_voxels": int(np.count_nonzero(values)),
                "minimum": float(values.min()), "maximum": float(values.max()),
                "decoded_float32_bytes": int(values.nbytes),
                "sidecar_to_nifti_axis_disagreement_degrees": errors,
                "sidecar_orientation_consistent_at_0_05_degrees": max(errors) <= 0.05,
                "orientation_threshold_scope": "metadata-rounding screen, not clinical accuracy",
                "sidecar_spacing_between_slices": sidecar.get("SpacingBetweenSlices"),
                "sidecar_image_type": sidecar["ImageType"],
                "scanner_frame_admission": "unresolved",
                "cross_scan_registration": "unverified"}
        geometry.append(item)
        del values
    output = ROOT / "artifacts/lausanne-original-pilot-v1/independent-check.json"
    preserve(output, json_bytes({"status": "byte_and_scalar_checks_passed_geometry_provenance_unresolved",
        "source_acquisition_sha256": sha256_file(RESULT), "audit_script_sha256": sha256_file(Path(__file__)),
        "source_checks": checks, "geometry": geometry,
        "limits": ["No source header modified or missing geometry inferred",
            "Storage-axis agreement alone does not certify DICOM/scanner position",
            "Multislab spacing metadata is not silently substituted for voxel spacing",
            "No anatomical coverage, vascular annotation, clinical validation or learning acceptance"]}))
    print(json.dumps({"output": str(output), "orientation_errors_degrees": [item["sidecar_to_nifti_axis_disagreement_degrees"] for item in geometry]}))


if __name__ == "__main__":
    main()
