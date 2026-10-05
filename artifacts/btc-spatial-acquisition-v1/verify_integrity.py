"""Read-only source integrity and native-grid checks; no case preparation."""
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import itertools
import json
import os
from pathlib import Path
import resource
import sys
import time

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[name] = "1"

import nibabel as nib
import numpy as np


def digest(path, algorithm="sha256"):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, algorithm).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def image(path):
    volume = nib.load(path)
    require(len(volume.shape) == 3, "Expected a 3-D source image")
    require(volume.header.get_xyzt_units()[0] == "mm", "Source units must be explicit millimeters")
    require(np.isfinite(volume.affine).all() and abs(np.linalg.det(volume.affine[:3, :3])) > 1e-8,
            "Source affine must be finite and invertible")
    qform, qcode = volume.get_qform(coded=True)
    sform, scode = volume.get_sform(coded=True)
    require(bool(qcode or scode), "Source lacks a coded physical transform")
    corners = np.array(list(itertools.product(*[(0, size - 1) for size in volume.shape])))
    discrepancy = None
    if qcode and scode:
        discrepancy = float(np.max(np.linalg.norm(nib.affines.apply_affine(qform, corners)
                                                  - nib.affines.apply_affine(sform, corners), axis=1)))
        require(discrepancy <= 0.01, "Coded source transforms disagree at native corners")
    values = volume.get_fdata(dtype=np.float32)
    require(np.isfinite(values).all(), "Source contains nonfinite values")
    return volume, values, {"shape": list(volume.shape), "stored_dtype": str(volume.get_data_dtype()),
        "physical_units": "mm", "spacing_mm": list(map(float, volume.header.get_zooms())),
        "orientation": list(nib.aff2axcodes(volume.affine)), "affine_ras_mm": volume.affine.tolist(),
        "qform_code": int(qcode), "sform_code": int(scode), "maximum_qform_sform_corner_error_mm": discrepancy,
        "scale_slope": float(volume.dataobj.slope), "scale_intercept": float(volume.dataobj.inter),
        "all_voxels_finite": True, "scaled_intensity_min": float(values.min()), "scaled_intensity_max": float(values.max())}


def main():
    started = time.perf_counter()
    directory = Path(__file__).resolve().parent
    root = directory.parents[1]
    output = directory / "integrity-review.json"
    require(not output.exists(), "Do not overwrite an existing integrity review")
    release_path, batch_path = directory / "root-release.json", directory / "batch-record.json"
    require(digest(release_path) == "968bf5dad7dff71ff6a90e57674dba790ad335122a3872441f1a06b0da4eca0b", "Root release changed")
    require(digest(batch_path) == "c06636316525305d964bbca82829164ec45e4ece60e87082aa060c73baff20b8", "Completed acquisition record changed")
    release, batch = json.loads(release_path.read_text()), json.loads(batch_path.read_text())
    data = Path(release["output_root"])
    snapshot = root / release["source_snapshot"]
    declaration = json.loads((snapshot / "manifests/experiments/btc-spatial-development-cohort-v1.json").read_text())
    record = {"schema_version": 1, "recorded_at": datetime.now(timezone.utc).isoformat(),
              "root_release_sha256": digest(release_path), "batch_record_sha256": digest(batch_path),
              "auditor_sha256": digest(Path(__file__)), "status": "running", "cases": [], "failures": [],
              "runtime": {"python": sys.version, "numpy": np.__version__, "nibabel": nib.__version__},
              "numerical_threads_requested": 1, "downloads_performed": False, "inference_executed": False,
              "case_preparation_executed": False, "source_images_modified": False, "brain_reviewed": False,
              "cortex_localized": False, "cortical_access_permitted": False, "clinical_deficit_probability": None,
              "interpretation": "Source identity, finite arrays and coordinate consistency only; not anatomical accuracy, surgical feasibility or planning eligibility."}
    for attempt in batch["attempts"]:
        subject = attempt["subject"]
        case = {"subject": subject, "development_role": attempt["development_role"], "status": "checking"}
        record["cases"].append(case)
        try:
            manifest_path = directory / f"{subject}-source-manifest.json"
            require(digest(manifest_path) == attempt["source_manifest_sha256"], "Saved source-manifest bytes changed")
            manifest = json.loads(manifest_path.read_text())
            require(manifest["selection_cohort_sha256"] == "962d964e1d71427f3625cdbebc0f7e4759e5d2345d8f95cb211ed45810ed2985", "Wrong selection cohort")
            verified = []
            for entry in manifest["files"]:
                path = data / entry["path"]
                require(path.stat().st_size == entry["bytes"] and digest(path) == entry["sha256"], "Source length or SHA256 changed")
                if entry.get("expected_md5"):
                    require(digest(path, "md5") == entry["expected_md5"], "Source annex MD5 mismatch")
                verified.append({"path": entry["path"], "bytes": entry["bytes"], "sha256": entry["sha256"],
                                 "annex_md5": entry.get("expected_md5")})
            t1_path = data / f"{subject}/ses-preop/anat/{subject}_ses-preop_T1w.nii.gz"
            mask_path = data / f"derivatives/tumor_masks/{subject}/anat/{subject}_space_T1_label-tumor.nii"
            t1, mri, t1_info = image(t1_path)
            annotation, values, annotation_info = image(mask_path)
            require(t1.shape == annotation.shape == (160, 256, 256), "Unexpected source dimensions")
            require(values.min() >= 0 and values.max() <= 1.000001, "Fractional annotation range invalid")
            transform = nib.orientations.ornt_transform(nib.io_orientation(annotation.affine), nib.io_orientation(t1.affine))
            reindexed = nib.orientations.apply_orientation(values, transform)
            aligned_affine = annotation.affine @ nib.orientations.inv_ornt_aff(transform, annotation.shape)
            corners = np.array(list(itertools.product(*[(0, size - 1) for size in t1.shape])))
            error = float(np.max(np.linalg.norm(nib.affines.apply_affine(aligned_affine, corners)
                                               - nib.affines.apply_affine(t1.affine, corners), axis=1)))
            require(error <= 0.001 and reindexed.shape == t1.shape, "Annotation needs more than orientation reindexing")
            target_count = int(np.count_nonzero(reindexed >= 0.5))
            require(target_count > 0, "No target at declared research threshold")
            case.update(status="source_integrity_passed", verified_files=verified, structural=t1_info,
                source_annotation=annotation_info, orientation_reindex=transform.tolist(),
                maximum_reindexed_corner_error_mm=error, annotation_threshold=0.5,
                annotation_threshold_status="existing research assumption; no binary image saved",
                target_voxels_at_declared_threshold=target_count,
                source_annotation_meaning="fractional source annotation; not a clinical probability or observed removal",
                full_head_support_accepted=False, working_brain_mask=None)
            for entry in verified:
                require(digest(data / entry["path"]) == entry["sha256"], "Read-only inspection modified source bytes")
            del t1, mri, annotation, values, reindexed
        except Exception as error:
            case["status"] = "failed"
            record["failures"].append({"subject": subject, "type": type(error).__name__, "message": str(error)})
    for item in declaration["candidates"]:
        if item["subject"] in release["forbidden_subjects"]:
            require(not any((data / f["path"]).exists() for f in item["files"]), "Unopened transfer files appeared")
    record["unopened_PAT29_PAT31_files_absent"] = True
    record["frozen_source_unchanged"] = all(digest(snapshot / p) == h for p, h in release["frozen_source_files"].items())
    record["elapsed_seconds"] = time.perf_counter() - started
    record["peak_rss_bytes_macos"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    record["status"] = "passed" if not record["failures"] and record["frozen_source_unchanged"] else "failed"
    output.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": record["status"], "cases": len(record["cases"]), "failures": record["failures"],
                      "elapsed_seconds": record["elapsed_seconds"], "receipt_sha256": digest(output)}))
    return 0 if record["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
