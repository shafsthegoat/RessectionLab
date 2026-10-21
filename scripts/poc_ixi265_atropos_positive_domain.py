"""Prepared v2 missing-intensity protocol; no fit without a separate release.

Fit only original_brain AND unchanged_T1>0. The eight independently diagnosed
zero-intensity cells are UNKNOWN/ABSTAINED, never CSF or corrected anatomy.
This protocol preserves v1 evidence and uses the same immutable fitting kernel,
parameters and native-grid checks. Posteriors remain class probabilities.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PREPARATION = ROOT/"artifacts/poc-ixi265-atropos-positive-domain-preparation-v2"
RUNTIME = ROOT/".tools/poc-antspyx-0.6.3-py312-v1/runtime"
RUNTIME_AUDIT = ROOT/"artifacts/poc-ixi265-atropos-preparation-v1/installed-runtime-audit.json"
V1_PATH = ROOT/"scripts/poc_ixi265_atropos.py"
V1_SHA = "4a5c5b4b2f85153212ac27e6356d5f1dca5c8b42d77257fb55a2fee53253624b"
MARKER = "ixi265-n4-atropos-positive-domain-v2-attempt.json"
EXPECTED_UNKNOWN = ((60,125,75), (67,135,49), (69,131,40), (89,120,83),
                    (107,123,65), (127,219,68), (128,115,125), (185,127,119))
EXPECTED_SHAPE = (256, 256, 150)
EXPECTED_BRAIN_VOXELS = 1_451_509
DIAGNOSTIC_PATH = "artifacts/ixi265-quality-independent-v1/independent-qa.json"
DIAGNOSTIC_SHA = "7a51ca88a898b11eb7fc132180d5f77da4a0bffaf49ceb6e5b8eb229d6d162a6"
LINEAGE_RECORDS = {DIAGNOSTIC_PATH: DIAGNOSTIC_SHA,
    "artifacts/poc-ixi265-atropos-run-v1/outcome.json": "55f4dd46f7cc692d4a66799cefca21bfa37430d3f5cf2ebe3eda675b6e4c989e"}


def sha(path):
    with Path(path).open("rb") as stream: return hashlib.file_digest(stream, "sha256").hexdigest()


def reference():
    if sha(V1_PATH) != V1_SHA: raise ValueError("Immutable v1 fitting/checking source changed")
    spec = importlib.util.spec_from_file_location("ixi_atropos_v1_frozen", V1_PATH)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def method():
    original = reference().plan()
    return {**original, "schema": "resectionlab-ixi265-atropos-positive-domain-candidate-v2",
        "status": "prepared_fit_unreleased", "protocol": "positive_intensity_fitting_domain_with_explicit_missing_intensity_abstention",
        "fitting_domain": "unchanged_original_brain_mask AND unchanged_acquired_T1 > 0",
        "expected_unknown_indices_native": [list(p) for p in EXPECTED_UNKNOWN],
        "expected_original_shape": list(EXPECTED_SHAPE), "expected_original_brain_voxels": EXPECTED_BRAIN_VOXELS,
        "expected_fit_domain_voxels": EXPECTED_BRAIN_VOXELS-len(EXPECTED_UNKNOWN),
        "excluded_cell_semantics": "UNKNOWN_ABSTAINED_missing_intensity_not_CSF_not_anatomy_correction",
        "brain_mask_replaced": False, "intensity_offset_or_clipping": False, "source_crop": False,
        "domain_maps_retained_separately": ["fit_domain_native", "excluded_unknown_native"],
        "v1_source_sha256": V1_SHA, "single_attempt_marker": MARKER,
        "limitations": original["limitations"] + ["eight_missing_intensity_cells_are_unknown_not_tissue_predictions",
            "positive_intensity_domain_is_a_declared_fitting_support_not_a_corrected_brain_segmentation"]}


def positive_domain(values, brain, *, expected_unknown=EXPECTED_UNKNOWN):
    """Pure source-preserving support construction; no correction/normalization."""
    if (values.ndim != 3 or brain.shape != values.shape or brain.dtype != np.bool_
            or not brain.any() or not np.isfinite(values).all() or np.any(values < 0)):
        raise ValueError("Finite nonnegative image and original nonempty binary brain required")
    expected = np.zeros(brain.shape, bool)
    if len(expected_unknown) != 8 or len(set(expected_unknown)) != 8:
        raise ValueError("Exactly eight unique prospective unknown cells required")
    for index in expected_unknown:
        if (len(index) != 3 or any(type(v) is not int or not 0 <= v < brain.shape[a] for a, v in enumerate(index))):
            raise ValueError("Unknown cell index outside the prospective native frame")
        expected[index] = True
    actual = brain & (values == 0)
    if not np.array_equal(actual, expected):
        raise ValueError("Missing-intensity cells differ from the exact frozen diagnostic set")
    domain = brain & (values > 0)
    if not domain.any(): raise ValueError("No positive-intensity fitting support")
    return domain, actual.copy(), {"original_brain_voxels": int(brain.sum()),
        "fit_domain_voxels": int(domain.sum()), "excluded_unknown_voxels": int(actual.sum()),
        "unknown_indices_native": np.argwhere(actual).tolist(),
        "excluded_semantics": "UNKNOWN_ABSTAINED", "source_arrays_modified": False,
        "domain_is_anatomical_correction": False}


def require_unknown_partition(brain, domain, unknown, labels, probabilities):
    if (domain.shape != brain.shape or unknown.shape != brain.shape
            or not np.array_equal(domain | unknown, brain) or np.any(domain & unknown)):
        raise ValueError("Fitting and unknown supports must exactly partition original brain")
    if not np.array_equal((labels == 0) & brain, unknown):
        raise ValueError("Class zero inside original brain is allowed only at declared unknown cells")
    if any(np.any(p[unknown] != 0) for p in probabilities):
        raise ValueError("Unknown cells must not receive a class posterior")


def diagnostic_lineage():
    """Only frozen diagnostic metadata is inspected; no source image is read."""
    document = json.loads((PREPARATION/"diagnostic-lineage.json").read_text())
    if document["unknown_indices_native"] != [list(p) for p in EXPECTED_UNKNOWN]:
        raise ValueError("Prospective missing-intensity declaration changed")
    if document["independent_diagnostic"] != {"path": DIAGNOSTIC_PATH, "sha256": DIAGNOSTIC_SHA}:
        raise ValueError("Exact independent diagnostic identity required")
    if (len(document["records"]) != len(LINEAGE_RECORDS)
            or {record["path"]: record["sha256"] for record in document["records"]} != LINEAGE_RECORDS):
        raise ValueError("Only exact frozen diagnostic and historical outcome metadata paths are admitted")
    for record in document["records"]:
        path = ROOT/record["path"]
        if not path.is_relative_to(ROOT) or path.resolve() != path or sha(path) != record["sha256"]:
            raise ValueError("Frozen independent diagnostic lineage changed")
    if sha(ROOT/DIAGNOSTIC_PATH) != DIAGNOSTIC_SHA: raise ValueError("Independent diagnostic changed")
    diagnosis = json.loads((ROOT/DIAGNOSTIC_PATH).read_text())["failed_tissue_fit"]
    base = reference()
    if ([row["index_ijk"] for row in diagnosis["rejected_cells"]] != [list(p) for p in EXPECTED_UNKNOWN]
            or any(row["value"] != 0 for row in diagnosis["rejected_cells"])
            or diagnosis["source_native_shape"] != list(EXPECTED_SHAPE)
            or diagnosis["estimated_brain_voxels"] != EXPECTED_BRAIN_VOXELS
            or diagnosis["masked_zero_voxels"] != 8 or diagnosis["masked_negative_voxels"] != 0
            or diagnosis["source_bytes_unchanged"] is not True
            or {k: row["sha256"] for k, row in diagnosis["inputs"].items()}
                != {"T1": base.T1_SHA, "brain_mask": base.MASK_SHA, "host_bundle": base.HOST_SHA}):
        raise ValueError("Independent diagnostic does not support this frozen missing-intensity protocol")
    return document


def preparation_intact():
    document = json.loads((PREPARATION/"preparation.json").read_text())
    for row in document["frozen_files"]:
        if sha(ROOT/row["path"]) != row["sha256"]: raise ValueError("Frozen v2 preparation changed")
    diagnostic_lineage()
    return document


def worker(spec_path):
    spec = json.loads(Path(spec_path).read_text())
    if spec.get("released_single_attempt") is not True: raise ValueError("Separate v2 fitting release required")
    base = reference(); base.require_admission(spec["admission"])
    preparation_intact(); base.bound_inputs(spec["paths"], spec["admission"])
    if Path(spec["runtime"]).resolve() != RUNTIME.resolve(): raise ValueError("Exact previously verified runtime required")
    runtime = base.runtime_audit(spec["runtime"])
    retained_runtime = json.loads(RUNTIME_AUDIT.read_text())
    for key in ("packages", "runtime_plan_sha256", "verified_official_API_sources", "installed_native_binary_hashes"):
        if runtime[key] != retained_runtime[key]: raise ValueError("Previously verified runtime evidence changed")
    import nibabel as nib
    import ants
    image, brain_image = (nib.load(spec["paths"][name]) for name in ("T1", "brain_mask"))
    if image.shape != EXPECTED_SHAPE: raise ValueError("Frozen native source shape differs")
    for item in (image, brain_image):
        q, qc = item.get_qform(coded=True); s, sc = item.get_sform(coded=True)
        if not (qc or sc) or qc and sc and not np.allclose(q, s, rtol=0, atol=base.FRAME_TOLERANCE_MM):
            raise ValueError("Missing/conflicting source NIfTI physical frame")
    mask_frame = base.require_native_frame(image.shape, image.affine, brain_image.shape,
        **base.ras_to_lps_geometry(brain_image.affine))
    values = np.asarray(image.dataobj, dtype=np.float32)
    binary = np.asarray(brain_image.dataobj)
    if not np.isin(binary, [0, 1]).all(): raise ValueError("Binary source brain required")
    brain = binary.astype(bool)
    if int(brain.sum()) != EXPECTED_BRAIN_VOXELS: raise ValueError("Frozen original brain count differs")
    domain, unknown, support = positive_domain(values, brain)
    # Same N4/Atropos parameters and native-grid checks, on explicitly changed support.
    corrected, labels, posteriors, audit = base.fit_arrays(values, domain, image.affine, ants)
    require_unknown_partition(brain, domain, unknown, labels, posteriors)
    output = Path(spec["output"]); outputs = []
    rows = [("n4_corrected_estimated", corrected, "bias_corrected_intensity_not_acquired_MRI"),
        ("class_labels_raw", labels, "unmodified_fitted_class_ids_on_positive_intensity_domain"),
        ("fit_domain_native", domain.astype(np.uint8), "declared_fitting_support_not_replacement_brain_mask"),
        ("excluded_unknown_native", unknown.astype(np.uint8), "UNKNOWN_ABSTAINED_missing_intensity_not_CSF")]
    rows.extend((f"class_{i}_posterior", p, "unmodified_fitted_membership_probability_not_PVE") for i, p in enumerate(posteriors, 1))
    rows.extend((name, (labels == code).astype(np.uint8), "hard_class_mask_under_declared_T1_intensity_assumption")
                for code, name in base.ASSUMPTIONS.items())
    for name, data, semantics in rows:
        header = image.header.copy(); header.set_data_dtype(data.dtype)
        result = nib.Nifti1Image(data, image.affine, header)
        result.set_qform(image.get_qform(), int(image.header["qform_code"]))
        result.set_sform(image.get_sform(), int(image.header["sform_code"]))
        path = output/(name+".nii.gz"); nib.save(result, path)
        reopened = nib.load(path)
        base.require_native_frame(image.shape, image.affine, reopened.shape, **base.ras_to_lps_geometry(reopened.affine))
        if not np.array_equal(np.asarray(reopened.dataobj), data): raise ValueError("Saved candidate array changed")
        outputs.append({"name": name, "sha256": sha(path), "bytes": path.stat().st_size,
            "semantics": semantics, "native_shape": list(data.shape)})
    base.bound_inputs(spec["paths"], spec["admission"])
    base.write_json(output/"candidate.json", {**method(), "status": "fitted_candidate_unaccepted",
        "runner_sha256": sha(__file__), "runtime": runtime, "outputs": outputs, "support": support,
        "numerical_audit": audit, "source_mask_frame_check": mask_frame, "source_bytes_unchanged": True,
        "diagnostic_lineage_sha256": sha(PREPARATION/"diagnostic-lineage.json")})


def run(paths, metadata, runtime, output, *, released_single_attempt=False):
    if not released_single_attempt: raise ValueError("Positive-domain v2 fitting is not released")
    base = reference(); base.require_admission(metadata); preparation = preparation_intact()
    runtime, output = Path(runtime).resolve(), Path(output).resolve()
    if runtime != RUNTIME.resolve() or runtime == (ROOT/".venv").resolve() or not (runtime/"bin/python").is_file():
        raise ValueError("Previously verified isolated runtime required")
    if output.exists(): raise ValueError("New v2 output only; no retry/overwrite")
    import psutil
    minimum_available = psutil.virtual_memory().available
    reason = base.resource_violation(base.sampled_rss(psutil), minimum_available, 0, 0, psutil.disk_usage(output.parent).free)
    if reason: raise ValueError(reason)
    base.bound_inputs(paths, metadata)
    declaration = {**method(), "status": "declared_before_source_decode",
        "created_utc": datetime.now(timezone.utc).isoformat(), "runner_sha256": sha(__file__),
        "preparation_sha256": sha(PREPARATION/"preparation.json"),
        "diagnostic_lineage": diagnostic_lineage(), "frozen_preparation_files": preparation["frozen_files"],
        "sources": {key: {"path": str(path), "sha256": sha(path)} for key, path in paths.items()},
        "runtime": str(runtime), "output": str(output), "real_attempt_number_for_this_protocol": 1,
        "v1_attempt_retained": True, "automatic_retry_or_tuning": False}
    with (runtime/MARKER).open("x") as stream: json.dump(declaration, stream, indent=2)
    output.mkdir(); temporary = output/"temporary"; temporary.mkdir()
    base.write_json(output/"declaration.json", declaration)
    spec = {"released_single_attempt": True, "admission": metadata, "paths": {k: str(v) for k, v in paths.items()},
            "runtime": str(runtime), "output": str(output)}
    base.write_json(output/"worker-input.json", spec)
    record = {**method(), "status": "running", "runner_sha256": sha(__file__),
        "memory_scope": "supervising_Python_plus_worker_and_recursive_descendants",
        "memory_caveat": "Sampled combined RSS; between-sample peaks are not a hard reservation"}
    base.write_json(output/"attempt.json", record)
    env = {**os.environ, "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS": "2", "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2",
        "VECLIB_MAXIMUM_THREADS": "2", "MKL_NUM_THREADS": "2", "NUMEXPR_NUM_THREADS": "2", "PYTHONHASHSEED": "0",
        "PYTHONNOUSERSITE": "1", "TMPDIR": str(temporary), "MPLCONFIGDIR": str(temporary/"matplotlib")}
    env.pop("PYTHONPATH", None)
    process = None; peak = base.sampled_rss(psutil); started = time.monotonic()
    try:
        with (output/"worker.log").open("w") as log:
            process = subprocess.Popen([str(runtime/"bin/python"), "-I", str(Path(__file__).resolve()),
                "--worker", str(output/"worker-input.json")], env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            while process.poll() is None:
                try: peak = max(peak, base.sampled_rss(psutil, process.pid))
                except psutil.NoSuchProcess: pass
                minimum_available = min(minimum_available, psutil.virtual_memory().available)
                reason = base.resource_violation(peak, minimum_available, time.monotonic()-started,
                    base.directory_bytes(output), psutil.disk_usage(output).free)
                if reason: raise RuntimeError(reason)
                time.sleep(.2)
        minimum_available = min(minimum_available, psutil.virtual_memory().available)
        reason = base.resource_violation(peak, minimum_available, time.monotonic()-started,
            base.directory_bytes(output), psutil.disk_usage(output).free)
        if reason or process.returncode: raise RuntimeError(reason or "worker_failed_see_retained_log")
        if not (output/"candidate.json").is_file(): raise RuntimeError("Missing candidate receipt")
        base.bound_inputs(paths, metadata); record["status"] = "fitted_candidate_unaccepted"
    except BaseException as error:
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try: process.wait(timeout=5)
            except subprocess.TimeoutExpired: os.killpg(process.pid, signal.SIGKILL); process.wait()
        record.update(status="failed", failure=str(error)); raise
    finally:
        record.update(elapsed_seconds=time.monotonic()-started, peak_process_family_rss_bytes_sampled=peak,
            minimum_available_memory_bytes_sampled=minimum_available, output_and_temporary_bytes=base.directory_bytes(output),
            exit_code=None if process is None else process.returncode)
        base.write_json(output/"attempt.json", record)
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--released-single-attempt-spec", type=Path)
    args = parser.parse_args()
    if args.worker: worker(args.worker)
    elif args.released_single_attempt_spec:
        spec = json.loads(args.released_single_attempt_spec.read_text())
        if spec.pop("released_single_attempt", None) is not True: raise ValueError("Separate v2 release required")
        print(json.dumps(run(**spec, released_single_attempt=True), indent=2))
    else: print(json.dumps(method(), indent=2))
