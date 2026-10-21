"""One declared generated-only two-arm N4 mask dtype diagnostic.

There are no source-image arguments or image reads. Both subprocesses generate
the same 48-cubed phantom. The uint8 arm deliberately preserves the historical
adapter representation; the float32 arm tests the native N4 type contract.
Neither successful execution nor generated class accuracy accepts anatomy.
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
REFERENCE_PATH = ROOT / "scripts/poc_ixi265_atropos.py"
REFERENCE_SHA = "4a5c5b4b2f85153212ac27e6356d5f1dca5c8b42d77257fb55a2fee53253624b"
RUNTIME = ROOT / ".tools/poc-antspyx-0.6.3-py312-v1/runtime"
OUTPUT = ROOT / "outputs/poc-ants-mask-dtype-generated-v1"
ARTIFACT = ROOT / "artifacts/poc-ants-mask-dtype-generated-v1"
ARMS = ("uint8", "float32")
RESOURCES = {"threads": 2, "rss_limit_bytes": 1024**3,
    "available_memory_floor_bytes": 1024**3, "seconds_per_arm": 120.,
    "output_temporary_limit_bytes": 16*1024**2, "disk_floor_bytes": 50*1024**3,
    "sample_interval_seconds": .1, "attempts_per_arm": 1,
    "RSS_scope": "supervising_Python_plus_worker_and_recursive_descendants"}


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def reference():
    if sha(REFERENCE_PATH) != REFERENCE_SHA:
        raise ValueError("Frozen reference changed")
    spec = importlib.util.spec_from_file_location("frozen_atropos_v1", REFERENCE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def same_runtime(actual, frozen):
    """Checkout aliases may differ; all package/source/binary records must agree."""
    left, right = dict(actual), dict(frozen)
    left["prefix"] = str(Path(left["prefix"]).resolve())
    right["prefix"] = str(Path(right["prefix"]).resolve())
    return left == right


def phantom():
    """No random sampling: concentric classes under a smooth multiplicative field."""
    coordinates = np.indices((48, 48, 48), dtype=np.float64) - 23.5
    x, y, z = coordinates / 24.
    radius2 = np.sum(coordinates**2, axis=0)
    labels = np.zeros((48, 48, 48), dtype=np.uint8)
    labels[radius2 <= 20**2] = 1
    labels[radius2 <= 15**2] = 2
    labels[radius2 <= 9**2] = 3
    clean = np.choose(labels, [0., 40., 85., 140.])
    bias = np.exp(.12*x + .08*y*y + .04*np.sin(z))
    values = np.asarray(clean*bias, dtype=np.float32)
    return values, labels != 0, labels, np.eye(4)


def array_record(array):
    return {"shape": list(array.shape), "dtype": str(array.dtype),
        "C_order_bytes_sha256": hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()}


def generated_record():
    values, mask, labels, affine = phantom()
    return {"generator": "fixed_concentric_spheres_20_15_9_voxel_radii_center23.5",
        "formula": "class_intensity[0,40,85,140]*exp(0.12*x+0.08*y*y+0.04*sin(z)); xyz=(index-23.5)/24",
        "sampling_seed": None, "random_sampling": False,
        "values": array_record(values), "binary_mask": array_record(mask),
        "generated_class_truth": array_record(labels), "affine_ras_mm": affine.tolist(),
        "identity_physical_frame": "RAS; LPS conversion changes metadata signs only",
        "mask_voxels": int(mask.sum()), "class_counts": [int((labels == n).sum()) for n in (1, 2, 3)],
        "masked_intensity_range": [float(values[mask].min()), float(values[mask].max())]}


def boundary_checks(mask, native_mask, affine, arm, base):
    """Assert representation, exact support/values, and full native frame before N4."""
    if arm not in ARMS: raise ValueError("Unknown declared arm")
    required_type = {"uint8": "unsigned char", "float32": "float"}[arm]
    if native_mask.pixeltype != required_type:
        raise ValueError("N4 boundary mask native pixeltype differs from declared arm")
    actual = native_mask.numpy()
    if actual.dtype != np.dtype(arm) or not np.array_equal(actual, mask.astype(arm)):
        raise ValueError("N4 boundary mask values changed")
    frame = base.require_native_frame(mask.shape, affine, native_mask.shape,
        native_mask.origin, native_mask.spacing, native_mask.direction)
    return {"native_pixeltype": native_mask.pixeltype, "numpy_dtype": str(actual.dtype),
        "binary_values": np.unique(actual).tolist(), "values_exact": True,
        "support_exact": bool(np.array_equal(actual != 0, mask)), "frame": frame,
        "representation": array_record(actual)}


def violation(rss, available, elapsed, output_bytes, free_disk):
    if rss > RESOURCES["rss_limit_bytes"]: return "supervisor_and_process_family_RSS_limit"
    if available < RESOURCES["available_memory_floor_bytes"]: return "available_memory_floor"
    if elapsed > RESOURCES["seconds_per_arm"]: return "time_limit"
    if output_bytes > RESOURCES["output_temporary_limit_bytes"]: return "combined_output_and_temporary_limit"
    if free_disk < RESOURCES["disk_floor_bytes"]: return "disk_floor"
    return None


def worker(arm):
    if arm not in ARMS: raise ValueError("Unknown declared arm")
    base = reference()
    declaration = json.loads((OUTPUT/"declaration.json").read_text())
    if (declaration.get("released_generated_two_arm_diagnostic") is not True
            or declaration["runner_sha256"] != sha(__file__)
            or declaration["arms"] != list(ARMS)
            or declaration["generated_input"] != generated_record()
            or declaration["resources"] != RESOURCES):
        raise ValueError("Changed or missing prospective generated declaration")
    runtime = base.runtime_audit(RUNTIME)
    if runtime != declaration["runtime"]: raise ValueError("Runtime differs from declaration")
    output = OUTPUT/arm
    values, mask, truth, affine = phantom()
    before = [array_record(a) for a in (values, mask, truth, affine)]
    import ants
    geometry = base.ras_to_lps_geometry(affine)
    image = ants.from_numpy(values.copy(), **geometry)
    native_mask = ants.from_numpy(mask.astype(arm), **geometry)
    boundary = boundary_checks(mask, native_mask, affine, arm, base)
    base.require_native_frame(values.shape, affine, image.shape, image.origin, image.spacing, image.direction)
    if image.pixeltype != "float" or not np.array_equal(image.numpy(), values):
        raise ValueError("Generated intensity representation changed")
    stages = {"worker_pid": os.getpid(), "generated_input": generated_record(), "mask_boundary": boundary,
        "N4_started": True, "N4_completed": False, "Atropos_started": False, "Atropos_completed": False}
    base.write_json(output/"stages.json", stages)
    corrected = ants.n4_bias_field_correction(image, mask=native_mask, **base.N4_PARAMETERS)
    corrected_values = corrected.numpy()
    n4_frame = base.require_native_frame(values.shape, affine, corrected.shape,
        corrected.origin, corrected.spacing, corrected.direction)
    if not np.isfinite(corrected_values).all() or np.any(corrected_values[mask] <= 0):
        raise ValueError("Invalid corrected generated intensity")
    if not np.array_equal(corrected_values[~mask], values[~mask]):
        raise ValueError("N4 modified values outside generated mask")
    np.save(output/"n4_raw.npy", corrected_values, allow_pickle=False)
    stages.update(N4_completed=True, N4_frame=n4_frame)
    base.write_json(output/"stages.json", stages)
    result_record = {"N4_raw": array_record(corrected_values), "mask_boundary": boundary,
        "native_frame": n4_frame, "N4_parameters": base.N4_PARAMETERS,
        "input_unchanged": before == [array_record(a) for a in (values, mask, truth, affine)]}
    if arm == "float32":
        # Atropos's exact pinned wrapper makes its own unsigned-int mask clone.
        stages["Atropos_started"] = True; base.write_json(output/"stages.json", stages)
        result = ants.atropos(a=corrected, x=native_mask, **base.ATROPOS_PARAMETERS)
        candidates = [result["segmentation"], *result["probabilityimages"]]
        frames = [base.require_native_frame(values.shape, affine, c.shape, c.origin, c.spacing, c.direction) for c in candidates]
        labels, *posteriors = [c.numpy() for c in candidates]
        audit = base.validate_classes(corrected_values, mask, labels, posteriors)
        for name, array in [("classes_raw", labels), *[(f"posterior_{n}_raw", p) for n,p in enumerate(posteriors,1)]]:
            np.save(output/(name+".npy"), array, allow_pickle=False)
        stages["Atropos_completed"] = True; base.write_json(output/"stages.json", stages)
        result_record.update(Atropos_parameters=base.ATROPOS_PARAMETERS, raw_class_validation=audit,
            Atropos_frames=frames, generated_truth_agreement=float(np.mean(labels[mask] == truth[mask])),
            posteriors_semantics="unmodified_membership_probabilities_not_PVE",
            raw_class_representation=array_record(labels))
    if not result_record["input_unchanged"]: raise ValueError("Generated inputs mutated")
    result_record.update(status="generated_arm_completed", anatomy_accepted=False,
        patient_inputs_read=False, clinical_validity_inferred=False)
    base.write_json(output/"result.json", result_record)


def run_arm(arm, base, psutil):
    output = OUTPUT/arm; output.mkdir()
    temporary = output/"temporary"; temporary.mkdir()
    environment = {**os.environ, "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS": "2", "OMP_NUM_THREADS": "2",
        "OPENBLAS_NUM_THREADS": "2", "VECLIB_MAXIMUM_THREADS": "2", "MKL_NUM_THREADS": "2",
        "NUMEXPR_NUM_THREADS": "2", "PYTHONHASHSEED": "0", "PYTHONNOUSERSITE": "1",
        "TMPDIR": str(temporary), "MPLCONFIGDIR": str(temporary/"matplotlib")}
    environment.pop("PYTHONPATH", None)
    started = time.monotonic(); process = None; peak = base.sampled_rss(psutil)
    available = psutil.virtual_memory().available; minimum_available = available
    minimum_disk = psutil.disk_usage(OUTPUT).free
    record = {"arm": arm, "started_utc": datetime.now(timezone.utc).isoformat(),
        "supervisor_pid": os.getpid(), "resources": RESOURCES, "attempts": 1}
    reason = violation(peak, available, 0, base.directory_bytes(OUTPUT), minimum_disk)
    if reason:
        record.update(status="not_started_resource_guard", failure=reason)
        base.write_json(output/"attempt.json", record)
        return record
    try:
        with (output/"worker.log").open("x") as log:
            process = subprocess.Popen([str(RUNTIME/"bin/python"), "-I", str(Path(__file__).resolve()), "--worker", arm],
                env=environment, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            record["worker_pid"] = process.pid
            base.write_json(output/"attempt.json", {**record, "status": "running"})
            while process.poll() is None:
                try: peak = max(peak, base.sampled_rss(psutil, process.pid))
                except psutil.NoSuchProcess: pass
                minimum_available = min(minimum_available, psutil.virtual_memory().available)
                minimum_disk = min(minimum_disk, psutil.disk_usage(OUTPUT).free)
                reason = violation(peak, minimum_available, time.monotonic()-started,
                    base.directory_bytes(OUTPUT), minimum_disk)
                if reason: raise RuntimeError(reason)
                time.sleep(RESOURCES["sample_interval_seconds"])
        reason = violation(peak, minimum_available, time.monotonic()-started, base.directory_bytes(OUTPUT), minimum_disk)
        if reason or process.returncode: raise RuntimeError(reason or "worker_failed_see_retained_log")
        if not (output/"result.json").is_file(): raise RuntimeError("missing_result")
        record["status"] = "generated_arm_completed"
    except BaseException as error:
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try: process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL); process.wait()
        record.update(status="failed", failure=str(error))
    finally:
        record.update(elapsed_seconds=time.monotonic()-started, finished_utc=datetime.now(timezone.utc).isoformat(),
            peak_supervisor_and_process_family_rss_bytes_sampled=peak,
            minimum_available_memory_bytes_sampled=minimum_available, minimum_free_disk_bytes_sampled=minimum_disk,
            combined_output_and_temporary_bytes=base.directory_bytes(OUTPUT),
            exit_code=None if process is None else process.returncode,
            sampling_limitation="Between-sample peaks are not a hard memory reservation")
        base.write_json(output/"attempt.json", record)
    return record


def run(*, released=False):
    if not released: raise ValueError("Generated-only diagnostic release required")
    if OUTPUT.exists(): raise ValueError("No retry or overwrite of declared diagnostic")
    base = reference()
    basis = ARTIFACT/"source-basis.json"
    controls = ARTIFACT/"controls.json"
    runtime = base.runtime_audit(RUNTIME)
    frozen_runtime = json.loads((base.PREPARATION/"installed-runtime-audit.json").read_text())
    if not same_runtime(runtime, frozen_runtime): raise ValueError("Installed runtime changed")
    import psutil
    reason = violation(base.sampled_rss(psutil), psutil.virtual_memory().available,
        0, 0, psutil.disk_usage(OUTPUT.parent).free)
    if reason: raise ValueError(reason)
    declaration = {"schema": "resectionlab-generated-N4-mask-dtype-diagnostic-v1",
        "released_generated_two_arm_diagnostic": True, "created_utc": datetime.now(timezone.utc).isoformat(),
        "arms": list(ARMS), "runner_sha256": sha(__file__), "frozen_reference_sha256": REFERENCE_SHA,
        "source_basis_sha256": sha(basis), "generated_controls_sha256": sha(controls),
        "runtime": runtime, "generated_input": generated_record(), "resources": RESOURCES,
        "parameters": {"N4": base.N4_PARAMETERS, "Atropos": base.ATROPOS_PARAMETERS},
        "scope": "generated_arrays_only_no_patient_reads_no_install_or_runtime_changes",
        "interpretation": "uint8 may crash, error, or silently misbehave; float success tests adapter representation, not clinical validity or universal crash causation",
        "float_success_continuation": "same_child_fixed_3class_Atropos_raw_outputs_and_frame_guards",
        "automatic_patient_fit": False}
    OUTPUT.mkdir()
    base.write_json(OUTPUT/"declaration.json", declaration)
    base.write_json(ARTIFACT/"declaration.json", declaration)
    records = []
    for arm in ARMS:
        record = run_arm(arm, base, psutil); records.append(record)
        print(json.dumps(record), flush=True)
    unchanged = base.runtime_audit(RUNTIME) == runtime
    outcome = {"declaration_sha256": sha(OUTPUT/"declaration.json"), "arms": records,
        "runtime_unchanged": unchanged, "patient_inputs_read": False, "anatomy_accepted": False,
        "manifest": [{"path": str(p.relative_to(OUTPUT)), "sha256": sha(p), "bytes": p.stat().st_size}
            for p in sorted(OUTPUT.rglob("*")) if p.is_file()]}
    base.write_json(OUTPUT/"outcome.json", outcome); base.write_json(ARTIFACT/"outcome.json", outcome)
    return outcome


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", choices=ARMS)
    parser.add_argument("--released-generated-diagnostic", action="store_true")
    args = parser.parse_args()
    if args.worker: worker(args.worker)
    elif args.released_generated_diagnostic: run(released=True)
    else: print(json.dumps({"resources": RESOURCES, "generated_input": generated_record(), "released": False}, indent=2))
