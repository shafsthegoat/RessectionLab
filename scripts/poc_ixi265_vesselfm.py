#!/usr/bin/env python3
"""Bounded vesselFM on the exact IXI265 native MRA, never the source reference.

The frozen host/registration package is read-only. Outputs go to a new folder.
The checkpoint may overlap IXI training data; no held-out accuracy is claimed.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HOST = ROOT/"outputs/poc-ixi265"
DEFAULT_OUTPUT = HOST/"vesselfm-base-v1"
CASE = "IXI265-Guys-0845"
MRA_SHA = "a87b381b882242b7d61a1930a8332d44200fdb3c56c68fc09c24a057eea86dc4"
T1_SHA = "317faf612b8e8a4e319fe3ec67404561b928ea9090fc23d1da20ce0900257950"
WEIGHT_SHA = "6da4fbd06dd9c3b597e5d50cc628deb3026919ff126d01afbd13ac8fe22ee10c"
WORKER_SHA = "7446332d8e948624c055f3c19ad5d64c12018095d8e4b330535d62c47908c28b"
REGISTRATION_SHA = "30d79cf98d10386fad2af8eafaaf6a754661f0b98a1a8d3cb7ab23f8b6073b1b"
TRANSFORM_SHA = "5ac523e6fae83057fd55c75cffac7c3ef89561ce5a3208f0a56ff2d1f742ef0b"
NATIVE_COVERAGE_SHA = "a92bf82175e1208ebef037bf3117fc40f779c4cf5669f70422be7b0015bc863b"
T1_COVERAGE_SHA = "e0a18a9d7c18e0be5510b12a89aafd22b4297519577d7e08d5e6a7bb64b8352b"
COHORT_SHA = "b27f28c21089e61f6c4899f172f608cd0f81391588777443939196b1700cb127"
HOST_BUNDLE_SHA = "c735daaf6d60f5f0d1567760a2accc18d4bd4838a096d8e0b5f490ee88abaff1"
PARAMETERS = {"patch_size": 128, "batch_size": 1, "overlap": .5, "threshold": .5,
              "blend": "constant", "tta": False, "postprocessing": False, "retain_probabilities": True}


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")


def verify(path, expected):
    path = Path(path)
    if not path.is_file() or sha(path) != expected:
        raise ValueError(f"Pinned artifact mismatch: {path.name}")
    return path


def required_paths():
    return {
        "MRA": (HOST/"originals/IXI265-MRA.nii.gz", MRA_SHA),
        "T1": (HOST/"originals/IXI265-T1.nii.gz", T1_SHA),
        "weights": (ROOT/"data/models/poc-vesselfm-base/vesselFM_base.pt", WEIGHT_SHA),
        "worker": (ROOT/"src/resectionlab/poc_models.py", WORKER_SHA),
        "registration": (HOST/"registration/registration.json", REGISTRATION_SHA),
        "transform": (HOST/"registration/T1_to_MRA_sampling.tfm", TRANSFORM_SHA),
        "native_coverage": (HOST/"registration/MRA_native_sampling_coverage.nii.gz", NATIVE_COVERAGE_SHA),
        "T1_coverage": (HOST/"registration/MRA_coverage_on_T1_candidate.nii.gz", T1_COVERAGE_SHA),
        "cohort": (ROOT/"manifests/experiments/ixi-component-person-cohort-v1.json", COHORT_SHA),
    }


def make_request(image, weights, output):
    """The worker receives exactly one acquired angiogram and checkpoint."""
    return {"image": str(image), "case": CASE, "model_id": "vesselfm-base", "modality": "MRA", "device": "mps",
            "output": str(output), "weights": [{"path": str(weights.resolve()), "sha256": WEIGHT_SHA,
                                                  "bytes": weights.stat().st_size}], "parameters": dict(PARAMETERS)}


def prepare(output):
    if (output/"inference.json").exists():
        raise ValueError("Inference attempt already exists; do not overwrite its history")
    output.mkdir(parents=True, exist_ok=True)
    pinned = {name: verify(path, digest) for name, (path, digest) in required_paths().items()}
    acquisition = json.loads((ROOT/"data/models/poc-vesselfm-base/acquisition.json").read_text())
    for item in acquisition["files"]:
        verify(ROOT/item["path"], item["sha256"])
    image = nib.load(pinned["MRA"])
    spacing = np.linalg.norm(image.affine[:3, :3], axis=0)
    target_spacing = float(np.median(spacing)) if len(set(spacing)) == 2 else float(min(spacing))
    target_size = np.round(np.array(image.shape)*spacing/target_spacing).astype(int)
    count = int(np.prod(np.maximum(1, np.ceil((target_size-128)/64).astype(int)+1)))
    request = make_request(pinned["MRA"], pinned["weights"], output)
    write_json(output/"request.json", request)
    record = {
        "schema": "poc-ixi265-vesselfm-preparation-v1", "status": "prepared_not_executed", "case": CASE,
        "patient_group": "IXI:265", "split": {"role": "TRAIN", "patient_group": "IXI:265", "training_admitted": False},
        "cohort_sha256": COHORT_SHA, "request_sha256": sha(output/"request.json"),
        "pinned": {name: {"path": str(path), "sha256": required_paths()[name][1]} for name, path in pinned.items()},
        "input": {"path": str(pinned["MRA"]), "sha256": MRA_SHA, "modality": "MRA", "shape": list(image.shape),
                  "affine_ras_mm": image.affine.tolist(), "spacing_mm": spacing.tolist(), "source_license": "CC-BY-SA-3.0"},
        "preprocessing": {"isotropic_spacing_mm": target_spacing, "isotropic_shape_xyz": target_size.tolist(),
                          "estimated_tiles": count, "normalization": "1st to 99th percentile to [0,1]", "prediction_domain": "entire acquired native MRA FOV"},
        "model": {"name": "vesselFM", "version": "base / "+acquisition["revision"], "variant": "base", "revision": acquisition["revision"], "weights_sha256": WEIGHT_SHA,
                  "weights_bytes": pinned["weights"].stat().st_size,
                  "license": "vesselFM Open RAIL++-M; Attachment A: research purposes and non-commercial use only",
                  "license_url": "https://huggingface.co/bwittmann/vesselFM/blob/dd70bbdd2b9de06b1316b4d1105bda9766c24fd7/LICENSE",
                  "upstream_code_license": "GPL-3.0", "actual_execution": "Existing local MONAI DynUNet POC worker; CPU accumulation; MPS tiles"},
        "training_lineage": {"sources": ["D_real", "D_drand", "D_flow"], "IXI_collection_present_in_D_real_CSD": True,
            "IXI265_subject_overlap": "unknown", "held_out_accuracy_claim": False,
            "evidence_url": "https://raw.githubusercontent.com/bwittmann/vesselFM/main/vesselfm/d_real/README.md",
            "paper_url": "https://arxiv.org/html/2411.17386v1", "checkpoint_card_url": "https://huggingface.co/bwittmann/vesselFM/blob/dd70bbdd2b9de06b1316b4d1105bda9766c24fd7/README.md"},
        "parameters": PARAMETERS, "budgets": {"wall_seconds": 600, "process_family_rss_bytes": 4*1024**3,
            "minimum_system_available_bytes": int(.75*1024**3), "threads": 2, "MPS_cap": "Existing worker 0.55 fraction of recommended memory; unchanged"},
        "source_reference_read_or_used": False, "annotation_completeness": "unknown", "anatomical_review": "pending",
        "registration": {"status": "estimated_unreviewed", "report_sha256": REGISTRATION_SHA, "transform_sha256": TRANSFORM_SHA},
        "downloads": False,
    }
    write_json(output/"prepared.json", record)
    return record


def supervise(command, log_path, *, timeout, rss_gib=4):
    import psutil
    environment = dict(os.environ, PYTHONPATH=str(ROOT/"src"), PYTHONFAULTHANDLER="1", OMP_NUM_THREADS="2",
        OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2", ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS="2",
        PYTORCH_ENABLE_MPS_FALLBACK="1", MPLBACKEND="Agg")
    started = time.monotonic(); peak = 0; minimum_available = psutil.virtual_memory().available; failure = None
    with log_path.open("w") as log:
        process = subprocess.Popen(command, env=environment, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        while process.poll() is None:
            try:
                parent = psutil.Process(process.pid)
                peak = max(peak, sum(p.memory_info().rss for p in [parent, *parent.children(recursive=True)] if p.is_running())
                           + psutil.Process().memory_info().rss)
            except psutil.NoSuchProcess:
                pass
            minimum_available = min(minimum_available, psutil.virtual_memory().available)
            if time.monotonic()-started > timeout or peak > rss_gib*1024**3 or minimum_available < .75*1024**3:
                failure = "wall_budget_exceeded" if time.monotonic()-started > timeout else "RSS_budget_exceeded" if peak > rss_gib*1024**3 else "system_available_memory_below_0.75GiB"
                os.killpg(process.pid, signal.SIGTERM)
                try: process.wait(timeout=3)
                except subprocess.TimeoutExpired: os.killpg(process.pid, signal.SIGKILL)
                break
            time.sleep(.2)
        process.wait()
    return {"command": command, "exit_code": process.returncode, "failure": failure, "elapsed_seconds": time.monotonic()-started,
            "peak_process_family_plus_supervisor_rss_bytes_sampled": peak, "minimum_system_available_bytes_sampled": minimum_available,
            "sampling_interval_seconds": .2, "wall_limit_seconds": timeout, "rss_limit_bytes": int(rss_gib*1024**3)}


def validate_native(mask_path, probability_path, reference):
    mask_image, probability_image = nib.load(mask_path), nib.load(probability_path)
    for image in (mask_image, probability_image):
        if image.shape != reference.shape or not np.allclose(image.affine, reference.affine, rtol=0, atol=1e-5):
            raise ValueError("Model output did not preserve native MRA geometry")
    mask = mask_image.get_fdata(dtype=np.float32)
    if not np.all((mask == 0) | (mask == 1)) or not mask.any():
        raise ValueError("Model output is not a nonempty binary mask")
    probability = probability_image.get_fdata(dtype=np.float32)
    if not np.isfinite(probability).all() or probability.min() < 0 or probability.max() > 1:
        raise ValueError("Nonfinite or out-of-range model score")
    if not np.array_equal(mask > 0, probability > .5):
        raise ValueError("Native mask does not match the declared score threshold")
    return {"foreground_voxels": int(np.count_nonzero(mask)), "score_range": [float(probability.min()), float(probability.max())]}


def inference(output):
    prepared = json.loads((output/"prepared.json").read_text())
    if (output/"inference.json").exists():
        raise ValueError("Inference history exists; use a new output directory for a new explicit attempt")
    for name, (path, digest) in required_paths().items(): verify(path, digest)
    verify(output/"request.json", prepared["request_sha256"])
    request = json.loads((output/"request.json").read_text())
    expected = make_request(HOST/"originals/IXI265-MRA.nii.gz", ROOT/"data/models/poc-vesselfm-base/vesselFM_base.pt", output)
    if request != expected:
        raise ValueError("Unexpected model request or extra annotation inputs")
    runtime = supervise([sys.executable, "-m", "resectionlab.poc_models", "--worker", str(output/"request.json")], output/"inference.log", timeout=600)
    failure = runtime["failure"] or ("worker_failed" if runtime["exit_code"] else None)
    validation = None
    if not failure:
        try:
            validation = validate_native(output/"vessels_mask.nii.gz", output/"vessels_probability.nii.gz", nib.load(request["image"]))
        except Exception as error:
            failure = "native_output_validation_failed: " + str(error)
    identity = {"input_sha256": MRA_SHA, "weights_sha256": WEIGHT_SHA, "worker_sha256": WORKER_SHA,
                "parameters": PARAMETERS, "device": "mps"}
    record = {"schema": "resectionlab-poc-inference-v1", "status": "failed" if failure else "succeeded", "failure": failure,
        "created_at": datetime.now(timezone.utc).isoformat(), "case": CASE, "model_id": "vesselfm-base", "model": prepared["model"],
        "weights": request["weights"], "input": prepared["input"], "device": "mps", "parameters": PARAMETERS,
        "cache_key": hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:20], "identity": identity,
        "elapsed_seconds": runtime["elapsed_seconds"], "runtime": runtime, "outputs": [], "native_validation": validation,
        "training_lineage": prepared["training_lineage"], "split": prepared["split"], "source_reference_read_or_used": False,
        "clinical_validation": "none", "anatomical_review": "pending", "coverage": "Acquired MRA partial FOV only; absence of prediction does not certify absence of vessels"}
    for name, kind in [("vessels_mask.nii.gz", "estimated_binary_mask"), ("vessels_probability.nii.gz", "uncalibrated_model_score")]:
        path = output/name
        if path.exists(): record["outputs"].append({"path": str(path), "name": name, "sha256": sha(path), "kind": kind})
    if (output/"worker.json").exists(): record["worker"] = json.loads((output/"worker.json").read_text())
    record["memory_caveat"] = "Sampled process RSS and MPS tensor/driver allocations overlap and must not be summed; not exact peaks."
    write_json(output/"inference.json", record)
    if failure: raise RuntimeError(failure)
    return record


def save_native(values, reference, path, *, binary=False):
    header = reference.header.copy(); header.set_data_dtype(np.uint8 if binary else np.float32)
    image = nib.Nifti1Image(values.astype(np.uint8 if binary else np.float32), reference.affine, header)
    image.set_qform(reference.get_qform(), int(reference.header["qform_code"]))
    image.set_sform(reference.get_sform(), int(reference.header["sform_code"]))
    nib.save(image, path)


def map_to_target(values, source_affine, target_shape, target_affine, transform, *, nearest):
    import SimpleITK as sitk
    from resectionlab.preprocessing import to_sitk
    reference = to_sitk(np.zeros(target_shape, np.float32), target_affine)
    result = sitk.Resample(to_sitk(values, source_affine), reference, transform,
        sitk.sitkNearestNeighbor if nearest else sitk.sitkLinear, 0., sitk.sitkFloat32)
    return sitk.GetArrayFromImage(result).transpose(2, 1, 0)


def register_saved(output):
    import SimpleITK as sitk
    from resectionlab.preprocessing import sitk_transform_to_ras, validate_rigid
    record = json.loads((output/"inference.json").read_text())
    if record["status"] != "succeeded" or record["input"]["sha256"] != MRA_SHA:
        raise ValueError("Only successful inference on this exact MRA may be registered")
    pinned = {name: verify(path, digest) for name, (path, digest) in required_paths().items()}
    for row in record["outputs"]: verify(row["path"], row["sha256"])
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(2)
    transform = sitk.ReadTransform(str(pinned["transform"]))
    matrix = sitk_transform_to_ras(transform); validate_rigid(matrix)
    registration = json.loads(pinned["registration"].read_text())
    if not np.allclose(matrix, registration["T1_RAS_mm_to_MRA_RAS_mm"], rtol=0, atol=1e-8):
        raise ValueError("Saved transform direction/center differs from registration receipt")
    mra, t1 = nib.load(pinned["MRA"]), nib.load(pinned["T1"])
    coverage = nib.load(pinned["native_coverage"]).get_fdata(dtype=np.float32)
    mapped_coverage = map_to_target(coverage, mra.affine, t1.shape, t1.affine, transform, nearest=True) > 0
    frozen_coverage = nib.load(pinned["T1_coverage"]).get_fdata(dtype=np.float32) > 0
    if not np.array_equal(mapped_coverage, frozen_coverage):
        raise ValueError("Recomputed physical sampling domain differs from frozen registration")
    mask = nib.load(output/"vessels_mask.nii.gz").get_fdata(dtype=np.float32)
    native_border_prediction_voxels = int(np.count_nonzero((mask > 0) & (coverage == 0)))
    mapped_mask = map_to_target(mask, mra.affine, t1.shape, t1.affine, transform, nearest=True) > 0
    excluded = int(np.count_nonzero(mapped_mask & ~mapped_coverage)); mapped_mask &= mapped_coverage
    save_native(coverage, mra, output/"native_sampling_coverage.nii.gz", binary=True)
    save_native(mapped_coverage, t1, output/"T1_sampling_coverage.nii.gz", binary=True)
    save_native(mapped_mask, t1, output/"vessels_mask_on_T1_estimated.nii.gz", binary=True)
    del mask
    score = nib.load(output/"vessels_probability.nii.gz").get_fdata(dtype=np.float32)
    mapped_score = map_to_target(score, mra.affine, t1.shape, t1.affine, transform, nearest=False)
    mapped_score[~mapped_coverage] = 0
    save_native(mapped_score, t1, output/"vessels_score_on_T1_estimated.nii.gz")
    result = {"schema": "poc-ixi265-registered-vesselfm-v1", "case": CASE, "patient_group": "IXI:265", "split": record["split"],
        "inference_sha256": sha(output/"inference.json"), "native_MRA_sha256": MRA_SHA, "target_T1_sha256": T1_SHA,
        "registration_sha256": REGISTRATION_SHA, "transform_sha256": TRANSFORM_SHA,
        "T1_RAS_mm_to_MRA_RAS_mm": matrix.tolist(), "MRA_RAS_mm_to_T1_RAS_mm": np.linalg.inv(matrix).tolist(),
        "registration_status": "estimated_unreviewed", "anatomical_registration_accepted": False, "vessel_anatomy_review": "pending",
        "source_reference_read_or_used": False, "held_out_accuracy_claim": False,
        "interpolation": {"mask_and_coverage": "nearest neighbor", "uncalibrated_score": "linear"},
        "threshold_order": "Native score threshold0.5 before nearest-neighbor mask transfer; registered score is retained separately",
        "coverage_meaning": "Conservative native acquired FOV excluding one border voxel; geometric sampling only, not vascular completeness",
        "registered_foreground_voxels": int(mapped_mask.sum()), "native_predicted_border_voxels": native_border_prediction_voxels,
        "mapped_prediction_voxels_excluded_by_coverage": excluded, "prediction_outside_saved_coverage_voxels": int((mapped_mask & ~mapped_coverage).sum()),
        "outputs": [{"path": str(output/name), "sha256": sha(output/name)} for name in
            ["native_sampling_coverage.nii.gz", "T1_sampling_coverage.nii.gz", "vessels_mask_on_T1_estimated.nii.gz", "vessels_score_on_T1_estimated.nii.gz"]]}
    write_json(output/"registered.json", result)
    return result


def finish_receipt(output):
    prepared = json.loads((output/"prepared.json").read_text())
    previous = json.loads((output/"receipt.json").read_text()) if (output/"receipt.json").exists() else {}
    record = {"schema": "poc-ixi265-vesselfm-artifact-v1", "case": CASE, "prepared": prepared,
              "inference": json.loads((output/"inference.json").read_text()),
              "registration": json.loads((output/"registered.json").read_text()) if (output/"registered.json").exists() else None,
              "script_sha256": sha(Path(__file__)), "frozen_host_bundle_edited": False,
              "inference_runner_script_sha256_recorded_at_run": previous.get("inference_runner_script_sha256_recorded_at_run", previous.get("script_sha256", sha(Path(__file__)))),
              "script_revision_note": "Later additions only provide saved-output QC and separate candidate packaging; the actual model worker remains pinned by worker_sha256",
              "versions": {p: importlib.metadata.version(p) for p in ["torch", "monai", "numpy", "nibabel", "SimpleITK", "psutil"]}}
    for name in ["visual-qc.json", "bundle-receipt.json", "package-runtime.json"]:
        if (output/name).exists():
            record[name] = {"sha256": sha(output/name), "content": json.loads((output/name).read_text())}
    if (output/"software-controls.txt").exists():
        record["software_controls"] = {"path": str(output/"software-controls.txt"), "sha256": sha(output/"software-controls.txt"),
            "result": (output/"software-controls.txt").read_text(), "generated_controls_are_not_anatomical_validation": True}
    write_json(output/"receipt.json", record)
    destination = ROOT/"artifacts/poc-ixi265-vesselfm-v1"
    destination.mkdir(exist_ok=True)
    write_json(destination/"receipt.json", record)


def render_qc(output):
    """Display predictions against acquired intensities, never source annotations."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    registered = json.loads((output/"registered.json").read_text())
    for row in registered["outputs"]: verify(row["path"], row["sha256"])
    reference = nib.load(verify(HOST/"originals/IXI265-T1.nii.gz", T1_SHA))
    registration = json.loads(verify(HOST/"registration/registration.json", REGISTRATION_SHA).read_text())
    mra_row = next(row for row in registration["outputs"] if Path(row["path"]).name == "MRA_on_T1_candidate.nii.gz")
    mra_path = verify(mra_row["path"], mra_row["sha256"])
    def canonical(path):
        return nib.as_closest_canonical(nib.load(path)).get_fdata(dtype=np.float32)
    t1 = nib.as_closest_canonical(reference).get_fdata(dtype=np.float32)
    mra = canonical(mra_path)
    mask = canonical(output/"vessels_mask_on_T1_estimated.nii.gz") > 0
    coverage = canonical(output/"T1_sampling_coverage.nii.gz") > 0
    center = np.argwhere(mask).mean(axis=0).astype(int)
    for values in (t1, mra):
        high = np.percentile(values[values > 0], 99)
        values /= high
    fig, axes = plt.subplots(3, 3, figsize=(12, 11))
    for row, (dimension, plane) in enumerate([(2, "Axial"), (1, "Coronal"), (0, "Sagittal")]):
        take = lambda values: np.take(values, int(center[dimension]), axis=dimension).T
        for column, values in enumerate((t1, mra, t1)):
            axis = axes[row, column]
            axis.imshow(take(values), cmap="gray", origin="lower", vmin=0, vmax=1)
            if column < 2 and take(mask).any():
                axis.contour(take(mask), levels=[.5], colors=["#ff456a"], linewidths=.6)
            if column == 2:
                axis.imshow(np.ma.masked_where(~take(coverage), take(coverage)), cmap="winter", origin="lower", vmin=0, vmax=1, alpha=.25)
                axis.contour(take(coverage), levels=[.5], colors=["cyan"], linewidths=.6)
            axis.set_title(f"{plane}: "+["T1 + estimated vessels", "MRA + estimated vessels", "Acquired MRA sampling coverage"][column])
            axis.axis("off")
    fig.suptitle("IXI265 vesselFM base: estimates, unreviewed registration, partial FOV. No source annotation used.")
    fig.tight_layout()
    fig.savefig(output/"vesselfm-qc.png", dpi=140)
    plt.close(fig)


def package(output, *, host_bundle=None, host_digest=HOST_BUNDLE_SHA):
    """Append a separate estimate display to the immutable source host.

    Existing reference pixels are decoded only to preserve their attachments;
    they are never forwarded to model inference or compared as ground truth.
    """
    from resectionlab.core import SourceRef, thaw_json
    from resectionlab.workspace_bundle import OriginalFile, load_workspace, save_workspace, view_state
    from resectionlab.workspace_imaging import DisplaySource, source_registration
    host_bundle = verify(host_bundle or HOST/(CASE+".ressectionlab"), host_digest)
    destination = output/"IXI265-vesselfm-estimates.ressectionlab"
    if destination.exists():
        raise ValueError("Candidate bundle already exists; preserve its exact saved bytes")
    inference_path, registered_path = output/"inference.json", output/"registered.json"
    inference_record = json.loads(inference_path.read_text())
    registered = json.loads(registered_path.read_text())
    if (inference_record["status"] != "succeeded" or inference_record["model_id"] != "vesselfm-base"
            or inference_record["input"]["sha256"] != MRA_SHA
            or inference_record["identity"]["weights_sha256"] != WEIGHT_SHA
            or registered["inference_sha256"] != sha(inference_path)
            or registered["registration_sha256"] != REGISTRATION_SHA
            or registered["anatomical_registration_accepted"] is not False):
        raise ValueError("Candidate provenance is not the pinned successful inference and unreviewed registration")
    for row in [*inference_record["outputs"], *registered["outputs"]]:
        verify(row["path"], row["sha256"])
    primary, artifacts, saved = load_workspace(host_bundle)
    if (saved is None or len(saved["sources"]) != 2 or len(saved["originals"]) != 16
            or primary.compartments or primary.critical_evidence or primary.prior_proposals
            or primary.metadata["split"]["role"] != "TRAIN"
            or primary.metadata["split"]["training_admitted"] is not False
            or primary.metadata["patient_group"] != "IXI:265"
            or next(ref for ref in primary.source_refs if ref.source_id == "structural").sha256 != T1_SHA):
        raise ValueError("Frozen source host contract changed")
    sources = list(saved["sources"])
    native = next(source.case for source in sources if source.modality == "MRA")
    if next(ref for ref in native.source_refs if ref.source_id == "structural").sha256 != MRA_SHA:
        raise ValueError("Native display image is not the pinned acquired MRA")
    mask_path = output/"vessels_mask.nii.gz"
    mask_image = nib.load(mask_path)
    mask = mask_image.get_fdata(dtype=np.float32)
    if (mask_image.shape != native.mri.shape or not np.allclose(mask_image.affine, native.affine, rtol=0, atol=1e-5)
            or not np.all((mask == 0) | (mask == 1)) or not mask.any()):
        raise ValueError("Estimated native display mask has incompatible values or geometry")
    layer = {"name": "estimated_vessels", "label": "vesselFM estimated vessels (native MRA)",
        "kind": "vessels", "provenance": "estimated", "policy_use": "excluded",
        "source_role": "display_only_model_estimate", "model": inference_record["model"],
        "model_weights_sha256": WEIGHT_SHA, "inference_sha256": sha(inference_path),
        "coverage": "Partial acquired MRA FOV; vascular completeness unknown; prediction borders retained",
        "native_sampling_coverage_sha256": NATIVE_COVERAGE_SHA,
        "registration_status": "native_MRA_not_registered_for_fusion", "anatomical_review": "pending",
        "held_out_accuracy_claim": False, "training_lineage": inference_record["training_lineage"], "color": "#37d6df"}
    metadata = {**thaw_json(native.metadata), "anatomy_layers": {"estimated_vessels": layer},
        "model_identity": inference_record["model"], "model_lineage": inference_record["training_lineage"],
        "annotation_use": "display_only", "annotation_policy_use": "excluded",
        "workspace_registration": {"kind": "estimated-unreviewed", "overlayPermitted": False},
        "missing_anatomy": ["skull", "tissue_classes", "ventricles"],
        "source_reference_used_for_inference": False, "inference_cache_key": inference_record["cache_key"]}
    labels = {"estimated_vessels": mask > 0}
    estimate = replace(native, compartments=labels, source_compartments=labels, metadata=metadata,
        source_refs=tuple(ref for ref in native.source_refs if ref.source_id == "structural") +
            (SourceRef("supplied_target_annotation", mask_path.resolve().as_uri(), sha256=sha(mask_path),
                       license=inference_record["model"]["license"], native_frame="RAS+", provenance="estimated"),))
    sources.append(DisplaySource(estimate, "MRA", "estimated"))
    state = view_state(primary, sources)
    state["states"].update(deepcopy(saved["imagingState"]["states"]))
    state["selectedSeriesId"] = saved["imagingState"]["selectedSeriesId"]
    state["states"][sources[-1].identity(primary)]["viewer"] = {
        "layerOpacity": {"estimated_vessels": .65}, "layerColor": {"estimated_vessels": "#37d6df"}}
    originals = list(saved["originals"])
    for name in ["vessels_mask.nii.gz", "native_sampling_coverage.nii.gz", "T1_sampling_coverage.nii.gz",
                 "vessels_mask_on_T1_estimated.nii.gz", "inference.json", "registered.json", "prepared.json", "visual-qc.json"]:
        originals.append(OriginalFile.from_path(output/name, role="model-result" if name.endswith(".nii.gz") else "metadata"))
    session = save_workspace(primary, destination, sources=sources, imaging=state, episode=saved["episode"],
        selection=saved["selection"], artifacts=artifacts, originals=originals)
    reopened, restored_artifacts, restored = load_workspace(destination)
    if (reopened.semantic_hash != primary.semantic_hash or reopened.planning_hash != primary.planning_hash
            or not np.array_equal(reopened.mri, primary.mri) or not np.array_equal(reopened.affine, primary.affine)
            or not np.array_equal(reopened.brain_mask, primary.brain_mask) or restored_artifacts != artifacts
            or restored["sessionHash"] != session or restored["imagingState"] != state):
        raise ValueError("Primary arrays, identity, artifacts or state changed in candidate")
    for before, after in zip(sources, restored["sources"], strict=True):
        if before.manifest(primary) != after.manifest(reopened):
            raise ValueError("Saved source metadata changed")
        for name in ("mri", "affine"):
            if not np.array_equal(getattr(before.case, name), getattr(after.case, name)):
                raise ValueError("Saved source image or frame changed")
        for group in ("compartments", "source_compartments"):
            for name, values in getattr(before.case, group).items():
                if not np.array_equal(values, getattr(after.case, group)[name]):
                    raise ValueError("Saved reference or estimated source array changed")
    if originals != restored["originals"] or sha(host_bundle) != host_digest:
        raise ValueError("Original bytes or immutable parent changed")
    fusion = [source_registration(reopened, item.case)[1]["overlayPermitted"] for item in restored["sources"]]
    if any(fusion):
        raise ValueError("Unreviewed source was inadvertently admitted for fusion")
    report = {"schema": "poc-ixi265-vesselfm-candidate-bundle-v1", "bundle": str(destination),
        "sha256": sha(destination), "bytes": destination.stat().st_size, "session_hash": session,
        "parent_bundle": str(host_bundle), "parent_sha256": host_digest, "parent_unchanged": True,
        "primary_case_hash": reopened.semantic_hash, "primary_planning_hash": reopened.planning_hash,
        "primary_arrays_and_metadata_unchanged": True, "preserved_parent_originals": 16,
        "total_originals": len(originals), "preserved_reference_sources": 2, "display_series": [source.manifest(primary) for source in sources],
        "source_arrays_verified_after_reopen": True, "fusion_permitted": fusion, "policy_constraints_added": False,
        "source_reference_pixels_decoded_for_archive_preservation_only": True,
        "mapped_T1_estimate_display_omitted": "Both additional full-resolution displays would exceed the existing 384 MiB attachment bound",
        "mapped_T1_mask_and_coverage_archived": True, "anatomical_registration_accepted": False,
        "vessel_anatomy_review": "pending", "training_admitted": False}
    write_json(output/"bundle-receipt.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--action", choices=["prepare", "run", "register-worker", "qc", "package", "finalize"], default="prepare")
    args = parser.parse_args(); output = args.output.resolve()
    if args.action == "prepare":
        print(json.dumps(prepare(output), indent=2)); return
    if args.action == "register-worker":
        register_saved(output); return
    if args.action == "qc":
        render_qc(output); return
    if args.action == "package":
        print(json.dumps(package(output), indent=2)); return
    if args.action == "finalize":
        finish_receipt(output); return
    try:
        inference(output)
        runtime = supervise([sys.executable, str(Path(__file__).resolve()), "--action", "register-worker", "--output", str(output)],
                            output/"registration.log", timeout=120)
        write_json(output/"registration-runtime.json", runtime)
        if runtime["failure"] or runtime["exit_code"]:
            raise RuntimeError("Saved-mask registration failed; inspect registration.log")
    finally:
        if (output/"inference.json").exists(): finish_receipt(output)
    print(json.dumps({"status": "succeeded", "receipt": str(output/"receipt.json")}))


if __name__ == "__main__": main()
