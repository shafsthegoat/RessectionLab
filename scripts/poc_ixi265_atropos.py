"""Prepared, opt-in N4 + Atropos fitting candidate for one frozen IXI265 T1.

No pretrained model is involved. Intensity classes and their posterior
probabilities are preserved; a separate T1 intensity-to-CSF/GM/WM assumption
does not make the classes anatomical truth or the posteriors tissue fractions.
The default command only prints the plan. No host/growth bundle is modified.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import itertools
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PREPARATION = ROOT / "artifacts/poc-ixi265-atropos-preparation-v1"
T1_SHA = "317faf612b8e8a4e319fe3ec67404561b928ea9090fc23d1da20ce0900257950"
MASK_SHA = "98710d7d1fa578dd510bf40b953e6d34926d8ff0c766cba16c005ffb6dbff192"
HOST_SHA = "c735daaf6d60f5f0d1567760a2accc18d4bd4838a096d8e0b5f490ee88abaff1"
WHEEL_SHA = "e378329089f622c72eacc47e4e2c18503d0f3fbf2cc39ccb9df65880c0a7846f"
THREADS = 2
RSS_LIMIT = 3 * 1024**3
MEMORY_FLOOR = 1024**3
TIME_LIMIT = 600.
OUTPUT_LIMIT = 512 * 1024**2
DISK_FLOOR = 50 * 1024**3
MAX_VOXELS = 12_000_000
FRAME_TOLERANCE_MM = 1e-4
N4_PARAMETERS = {"shrink_factor": 4, "convergence": {"iters": [30, 20, 10], "tol": 1e-7},
                 "spline_param": 150, "rescale_intensities": False, "return_bias_field": False, "verbose": True}
ATROPOS_PARAMETERS = {"i": "KMeans[3]", "m": "[0.1,1x1x1]", "c": "[10,0]", "priorweight": 0., "r": 0, "v": 1}
ASSUMPTIONS = {1: "csf_like_intensity_estimated", 2: "gray_matter_like_intensity_estimated",
               3: "white_matter_like_intensity_estimated"}
LIMITATIONS = ["within_image_finite_mixture_fitting_not_pretrained_inference",
    "ordered_T1_intensity_to_CSF_GM_WM_is_an_unreviewed_assumption",
    "posterior_probabilities_are_not_physical_partial_volume_fractions",
    "CSF_like_class_does_not_identify_ventricles",
    "no_anatomical_acceptance_host_upgrade_or_automatic_growth_use",
    "two_thread_floating_point_execution_not_claimed_bitwise_reproducible"]


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, document):
    path = Path(path)
    temporary = path.with_suffix(".partial.json")
    temporary.write_text(json.dumps(document, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def plan():
    return {"schema": "resectionlab-ixi265-atropos-candidate-v1", "status": "prepared_not_run",
        "patient_group": "IXI:265", "role": "TRAIN", "training_admitted": False,
        "source_hashes": {"T1": T1_SHA, "brain_mask": MASK_SHA, "host_bundle": HOST_SHA},
        "package": {"name": "antspyx", "version": "0.6.3", "wheel_sha256": WHEEL_SHA},
        "parameters": {"N4": N4_PARAMETERS, "Atropos": ATROPOS_PARAMETERS,
            "grid": "full_native_grid_no_resampling_no_crop", "priors": None,
            "intensity_clipping_or_normalization": None, "Atropos_random_seed_option": "r=0 (fixed upstream seed)"},
        "resources": {"threads": THREADS, "rss_limit_bytes": RSS_LIMIT, "minimum_available_bytes": MEMORY_FLOOR,
            "timeout_seconds": TIME_LIMIT, "output_limit_bytes": OUTPUT_LIMIT, "free_disk_floor_bytes": DISK_FLOOR,
            "sample_interval_seconds": .2, "max_voxels": MAX_VOXELS, "attempts": 1},
        "frame_tolerance_mm": FRAME_TOLERANCE_MM, "hard_class_assumptions": ASSUMPTIONS,
        "posterior_semantics": "unmodified_fitted_class_membership_probabilities",
        "anatomy_accepted": False, "host_accepted": False, "growth_use_accepted": False, "limitations": LIMITATIONS}


def require_admission(metadata):
    """Metadata before any image/file read; no general or protected-host API."""
    if metadata.get("training_admitted") is not False or metadata != {"patient_group": "IXI:265", "role": "TRAIN", "training_admitted": False,
                    "T1_sha256": T1_SHA, "brain_mask_sha256": MASK_SHA, "host_sha256": HOST_SHA}:
        raise ValueError("Only the exact frozen IXI265 TRAIN source binding is admitted")


def bound_inputs(paths, metadata):
    require_admission(metadata)
    if set(paths) != {"T1", "brain_mask", "host_bundle"}:
        raise ValueError("Exactly the admitted T1, brain mask and host bundle are required")
    canonical = {"T1": ROOT/"outputs/poc-ixi265/originals/IXI265-T1.nii.gz",
                 "brain_mask": ROOT/"outputs/poc-ixi265/synthstrip/brain_mask.nii.gz",
                 "host_bundle": ROOT/"outputs/poc-ixi265/IXI265-Guys-0845.ressectionlab"}
    # Validate ALL lexical paths and redirections before hashing ANY payload.
    # These frozen local paths have no symlink components; a redirected leaf
    # or ancestor is refused even if the caller declares the expected hashes.
    for key, path in paths.items():
        supplied = Path(path)
        if supplied != canonical[key] or supplied.resolve() != canonical[key]:
            raise ValueError("Only canonical nonredirected IXI265 source paths are admitted")
    expected = {"T1": T1_SHA, "brain_mask": MASK_SHA, "host_bundle": HOST_SHA}
    for key, path in paths.items():
        if digest(path) != expected[key]:
            raise ValueError("Changed or foreign admitted source bytes: " + key)


def ras_to_lps_geometry(affine):
    affine = np.asarray(affine, dtype=np.float64)
    if affine.shape != (4, 4) or not np.isfinite(affine).all() or not np.array_equal(affine[3], [0, 0, 0, 1]):
        raise ValueError("Finite homogeneous source RAS affine required")
    lps = np.diag([-1., -1., 1., 1.]) @ affine
    spacing = np.linalg.norm(lps[:3, :3], axis=0)
    if np.any(spacing <= 0):
        raise ValueError("Positive source spacing required")
    direction = lps[:3, :3] / spacing
    if not np.allclose(direction.T @ direction, np.eye(3), rtol=0, atol=1e-5):
        raise ValueError("Source shear exceeds the bounded ITK direction contract")
    return {"origin": tuple(lps[:3, 3]), "spacing": tuple(spacing), "direction": direction}


def require_native_frame(shape, affine, candidate_shape, origin, spacing, direction):
    """Check affine AND eight physical voxel-cell corners; never rewrite source."""
    if tuple(shape) != tuple(candidate_shape):
        raise ValueError("Native output shape differs")
    candidate = np.eye(4)
    candidate[:3, :3] = np.asarray(direction) @ np.diag(spacing)
    candidate[:3, 3] = origin
    candidate = np.diag([-1., -1., 1., 1.]) @ candidate
    if not np.isfinite(candidate).all():
        raise ValueError("Nonfinite output frame")
    corners = np.array(list(itertools.product(*[(-.5, n-.5) for n in shape])))
    source_points = corners @ affine[:3, :3].T + affine[:3, 3]
    output_points = corners @ candidate[:3, :3].T + candidate[:3, 3]
    errors = np.linalg.norm(source_points-output_points, axis=1)
    affine_error = float(np.max(np.abs(candidate-affine)))
    if affine_error > FRAME_TOLERANCE_MM or float(errors.max()) > FRAME_TOLERANCE_MM:
        raise ValueError("Native output affine/physical corners differ")
    return {"candidate_affine_ras_mm": candidate.tolist(), "source_affine_ras_mm": affine.tolist(),
        "affine_max_abs_difference": affine_error, "eight_cell_corner_errors_mm": errors.tolist(),
        "output_saved_with_original_affine": True, "array_permutation_or_resampling": False}


def validate_classes(corrected, mask, labels, posteriors):
    if (labels.shape != mask.shape or not np.isfinite(labels).all()
            or not np.array_equal(labels, np.rint(labels)) or not np.isin(labels, [0, 1, 2, 3]).all()
            or np.any(labels[~mask] != 0) or np.any(labels[mask] == 0)):
        raise ValueError("Invalid or out-of-mask class labels")
    if len(posteriors) != 3:
        raise ValueError("Exactly three fitted class posterior maps required")
    total = np.zeros(mask.shape, np.float64)
    for posterior in posteriors:
        if (posterior.shape != mask.shape or not np.isfinite(posterior).all()
                or np.any(posterior < 0) or np.any(posterior > 1) or np.any(posterior[~mask] != 0)):
            raise ValueError("Invalid posterior probabilities; no clipping or renormalization")
        total += posterior
    error = float(np.max(np.abs(total[mask]-1)))
    if error > 2e-5:
        raise ValueError("Posterior normalization failed; no renormalization")
    rows = []
    for code, assumption in ASSUMPTIONS.items():
        selected = labels == code
        if not selected.any():
            raise ValueError("A fitted class is empty")
        rows.append({"class": code, "voxels": int(selected.sum()),
            "corrected_intensity_mean": float(corrected[selected].mean()),
            "declared_assumption": assumption, "anatomical_acceptance": False})
    if not rows[0]["corrected_intensity_mean"] < rows[1]["corrected_intensity_mean"] < rows[2]["corrected_intensity_mean"]:
        raise ValueError("Fitted class intensities violate declared order; no relabeling")
    return {"classes": rows, "posterior_sum_max_abs_error": error, "probabilities_modified": False}


def fit_arrays(values, mask, affine, ants):
    """Worker-only fitter; generated controls may inject a fake ANTs backend."""
    if (values.ndim != 3 or values.size > MAX_VOXELS or mask.shape != values.shape
            or mask.dtype != np.bool_ or not mask.any() or not np.isfinite(values).all()
            or np.any(values[mask] <= 0)):
        raise ValueError("Bounded finite 3D source with positive masked T1 intensities required")
    geometry = ras_to_lps_geometry(affine)
    image = ants.from_numpy(np.array(values, dtype=np.float32, copy=True), **geometry)
    brain = ants.from_numpy(np.array(mask, dtype=np.uint8, copy=True), **geometry)
    frames = {}
    def check(name, candidate):
        frames[name] = require_native_frame(values.shape, affine, candidate.shape,
            candidate.origin, candidate.spacing, candidate.direction)
    check("input", image)
    check("mask", brain)
    corrected = ants.n4_bias_field_correction(image, mask=brain, **N4_PARAMETERS)
    check("N4", corrected)
    corrected_values = corrected.numpy()
    if not np.isfinite(corrected_values).all() or np.any(corrected_values[mask] <= 0):
        raise ValueError("Invalid N4 corrected intensities")
    result = ants.atropos(a=corrected, x=brain, **ATROPOS_PARAMETERS)
    check("classes", result["segmentation"])
    labels = result["segmentation"].numpy()
    posteriors = []
    for index, probability in enumerate(result["probabilityimages"], 1):
        check("posterior_" + str(index), probability)
        posteriors.append(probability.numpy())
    audit = validate_classes(corrected_values, mask, labels, posteriors)
    return corrected_values, labels.astype(np.uint8), posteriors, {"frames": frames, "classes": audit}


def resource_violation(rss, available, elapsed, output_bytes, free_disk):
    if rss > RSS_LIMIT: return "sampled_process_family_RSS_limit"
    if available < MEMORY_FLOOR: return "available_memory_floor"
    if elapsed > TIME_LIMIT: return "time_limit"
    if output_bytes > OUTPUT_LIMIT: return "output_and_temporary_bytes_limit"
    if free_disk < DISK_FLOOR: return "free_disk_floor"
    return None


def directory_bytes(path):
    return sum(p.stat().st_size for p in Path(path).rglob("*") if p.is_file())


def sampled_rss(psutil, worker_pid=None):
    """Count this supervising Python process plus all worker descendants."""
    own = psutil.Process(os.getpid()).memory_info().rss
    if worker_pid is None: return own
    worker = psutil.Process(worker_pid)
    return own + sum(p.memory_info().rss for p in [worker, *worker.children(recursive=True)] if p.is_running())


def runtime_audit(runtime):
    if (sys.version_info[:2] != (3, 12) or platform.system() != "Darwin" or platform.machine() != "arm64"
            or Path(sys.prefix).resolve() != Path(runtime).resolve() or sys.prefix == sys.base_prefix
            or Path(runtime).resolve() == (ROOT/".venv").resolve()):
        raise ValueError("Separate native macOS arm64 CPython 3.12 venv required")
    cfg = (Path(runtime)/"pyvenv.cfg").read_text().lower().replace(" ", "")
    if "include-system-site-packages=false" not in cfg:
        raise ValueError("System/application package inheritance is forbidden")
    document = json.loads((PREPARATION/"runtime-plan.json").read_text())
    actual = []
    for package in document["packages"]:
        installed = importlib.metadata.distribution(package["name"])
        if installed.version != package["version"]:
            raise ValueError("Changed pinned runtime package: " + package["name"])
        metadata = installed.read_text("METADATA").encode()
        actual.append({"name": package["name"], "version": installed.version,
            "metadata_sha256": hashlib.sha256(metadata).hexdigest()})
    official = json.loads((PREPARATION/"official-provenance.json").read_text())
    distribution = importlib.metadata.distribution("antspyx")
    sources = []
    for entry in official["documents"]:
        relative = entry.get("installed_package_path")
        if relative is None: continue
        path = Path(distribution.locate_file(relative)).resolve()
        if not path.is_relative_to(Path(runtime).resolve()) or digest(path) != entry["sha256"]:
            raise ValueError("Installed ANTsPy API source differs from the frozen official release")
        sources.append({"path": relative, "sha256": digest(path)})
    binaries = [{"path": str(entry), "sha256": digest(distribution.locate_file(entry))}
                for entry in distribution.files if str(entry).endswith((".so", ".dylib"))]
    return {"packages": actual, "runtime_plan_sha256": digest(PREPARATION/"runtime-plan.json"),
            "python": sys.version, "prefix": sys.prefix, "verified_official_API_sources": sources,
            "installed_native_binary_hashes": binaries}


def worker(spec_path):
    spec = json.loads(Path(spec_path).read_text())
    if spec.get("released_single_attempt") is not True:
        raise ValueError("A released single fitting attempt is required")
    require_admission(spec["admission"])
    bound_inputs(spec["paths"], spec["admission"])
    runtime = runtime_audit(spec["runtime"])
    # Imports and all pixel decoding occur only inside the supervised worker.
    import nibabel as nib
    import ants
    image, brain = (nib.load(spec["paths"][name]) for name in ("T1", "brain_mask"))
    for item in (image, brain):
        q, qc = item.get_qform(coded=True); s, sc = item.get_sform(coded=True)
        if not (qc or sc) or qc and sc and not np.allclose(q, s, rtol=0, atol=FRAME_TOLERANCE_MM):
            raise ValueError("Missing/conflicting source NIfTI physical frame")
    mask_geometry = ras_to_lps_geometry(brain.affine)
    mask_frame = require_native_frame(image.shape, image.affine, brain.shape, **mask_geometry)
    values = np.array(image.dataobj, dtype=np.float32)
    binary = np.asarray(brain.dataobj)
    if not np.isin(binary, [0, 1]).all(): raise ValueError("Binary source brain mask required")
    mask = binary.astype(bool)
    corrected, labels, posteriors, audit = fit_arrays(values, mask, image.affine, ants)
    output = Path(spec["output"])
    outputs = []
    rows = [("n4_corrected_estimated", corrected, "bias_corrected_intensity_not_acquired_MRI"),
            ("class_labels_raw", labels, "unmodified_ordered_fitted_class_ids")]
    rows.extend((f"class_{i}_posterior", p, "fitted_membership_probability_not_PVE") for i, p in enumerate(posteriors, 1))
    rows.extend((name, (labels == code).astype(np.uint8), "hard_class_mask_under_declared_T1_intensity_assumption")
                for code, name in ASSUMPTIONS.items())
    for name, data, semantics in rows:
        header = image.header.copy(); header.set_data_dtype(data.dtype)
        saved = nib.Nifti1Image(data, image.affine, header)
        saved.set_qform(image.get_qform(), int(image.header["qform_code"]))
        saved.set_sform(image.get_sform(), int(image.header["sform_code"]))
        path = output/(name+".nii.gz"); nib.save(saved, path)
        reopened = nib.load(path)
        geom = ras_to_lps_geometry(reopened.affine)
        require_native_frame(image.shape, image.affine, reopened.shape, **geom)
        if not np.array_equal(np.asarray(reopened.dataobj), data):
            raise ValueError("Saved candidate array changed on reopen")
        outputs.append({"name": name, "sha256": digest(path), "bytes": path.stat().st_size,
                        "semantics": semantics, "native_shape": list(data.shape)})
    bound_inputs(spec["paths"], spec["admission"])
    write_json(output/"candidate.json", {**plan(), "status": "fitted_candidate_unaccepted",
        "runtime": runtime, "outputs": outputs, "numerical_audit": audit, "source_mask_frame_check": mask_frame,
        "runner_sha256": digest(__file__), "source_bytes_unchanged": True})


def run(paths, metadata, runtime, output, *, released_single_attempt=False):
    if not released_single_attempt:
        raise ValueError("Fitting is not released; preparation does not authorize source decoding")
    require_admission(metadata)
    runtime, output = Path(runtime).resolve(), Path(output).resolve()
    if runtime == (ROOT/".venv").resolve() or not (runtime/"bin/python").is_file():
        raise ValueError("Prepared isolated runtime required; never use shared .venv")
    if output.exists(): raise ValueError("New candidate artifact directory required; no retry/overwrite")
    import psutil
    minimum_available = psutil.virtual_memory().available
    reason = resource_violation(sampled_rss(psutil), minimum_available, 0, 0, psutil.disk_usage(output.parent).free)
    if reason: raise ValueError(reason)
    bound_inputs(paths, metadata)
    marker = runtime/"ixi265-n4-atropos-single-attempt.json"
    with marker.open("x") as stream:
        json.dump({"created_utc": datetime.now(timezone.utc).isoformat(), "output": str(output), "runner_sha256": digest(__file__)}, stream)
    output.mkdir()
    temporary = output/"temporary"; temporary.mkdir()
    specification = {"released_single_attempt": True, "admission": metadata,
        "paths": {name: str(Path(p).resolve()) for name, p in paths.items()}, "runtime": str(runtime), "output": str(output)}
    write_json(output/"worker-input.json", specification)
    record = {**plan(), "status": "running", "runner_sha256": digest(__file__), "started_utc": datetime.now(timezone.utc).isoformat(),
        "memory_scope": "supervising_Python_plus_worker_and_recursive_descendants",
        "memory_caveat": "Sampled combined RSS; between-sample peaks are not a hard memory reservation", "sources": specification["paths"]}
    write_json(output/"attempt.json", record)
    environment = {**os.environ, "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS": "2", "OMP_NUM_THREADS": "2",
        "OPENBLAS_NUM_THREADS": "2", "VECLIB_MAXIMUM_THREADS": "2", "MKL_NUM_THREADS": "2",
        "NUMEXPR_NUM_THREADS": "2", "PYTHONHASHSEED": "0", "PYTHONNOUSERSITE": "1",
        "TMPDIR": str(temporary), "MPLCONFIGDIR": str(temporary/"matplotlib")}
    environment.pop("PYTHONPATH", None)
    process = None; peak = sampled_rss(psutil); started = time.monotonic(); reason = None
    try:
        with (output/"worker.log").open("w") as log:
            process = subprocess.Popen([str(runtime/"bin/python"), "-I", str(Path(__file__).resolve()),
                "--worker", str(output/"worker-input.json")], env=environment, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            while process.poll() is None:
                try:
                    peak = max(peak, sampled_rss(psutil, process.pid))
                except psutil.NoSuchProcess: pass
                minimum_available = min(minimum_available, psutil.virtual_memory().available)
                reason = resource_violation(peak, minimum_available, time.monotonic()-started,
                    directory_bytes(output), psutil.disk_usage(output).free)
                if reason: raise RuntimeError(reason)
                time.sleep(.2)
        minimum_available = min(minimum_available, psutil.virtual_memory().available)
        reason = resource_violation(peak, minimum_available, time.monotonic()-started,
            directory_bytes(output), psutil.disk_usage(output).free)
        if reason or process.returncode: raise RuntimeError(reason or "worker_failed_see_retained_log")
        if not (output/"candidate.json").is_file(): raise RuntimeError("worker_missing_candidate_receipt")
        bound_inputs(paths, metadata)
        record["status"] = "fitted_candidate_unaccepted"
    except BaseException as error:
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try: process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL); process.wait()
        record.update(status="failed", failure=str(error))
        raise
    finally:
        record.update(elapsed_seconds=time.monotonic()-started, peak_process_family_rss_bytes_sampled=peak,
            minimum_available_memory_bytes_sampled=minimum_available,
            output_and_temporary_bytes=directory_bytes(output), exit_code=None if process is None else process.returncode)
        write_json(output/"attempt.json", record)
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--released-single-attempt-spec", type=Path,
                        help="Root-released JSON containing paths, admission, runtime and new output; no installation is performed")
    args = parser.parse_args()
    if args.worker:
        worker(args.worker)
    elif args.released_single_attempt_spec:
        spec = json.loads(args.released_single_attempt_spec.read_text())
        if spec.pop("released_single_attempt", None) is not True: raise ValueError("Explicit release required")
        print(json.dumps(run(**spec, released_single_attempt=True), indent=2))
    else:
        print(json.dumps(plan(), indent=2))
