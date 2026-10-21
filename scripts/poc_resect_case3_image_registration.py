#!/usr/bin/env python3
"""Prospective Case3 image-only rigid candidate; no landmark reader or evaluator.

Preflight reads compressed bytes for fixity and exactly 348 decompressed header
bytes per image. Only a later ROOT release may start the supervised fit. A fit
failure is a result, never a request to tune settings or substitute a case.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
from real_intake_io import atomic_preserve, check_deadline, termination_cleanup, verify_source_file
from run_resect_case4_sparse_update import header_record
import poc_ixi265_registration as rigid

SCHEMA = "resect-case3-image-registration-v1"
CASE = "RESECT:Case3"
COHORT = "manifests/resect-component-cohort-v1.json"
COHORT_SHA = "4fc51f01e42253d395660d850cda2296a3e7f23e5a5ade80bbd5a3ed411b6a77"
INPUTS = {
    "before": {"path": "data/anatomy/resect-v1/Case3/US/Case3-US-before.nii.gz",
               "bytes": 9822564, "sha256": "9fbb4d63cb26d324c7c8281a2b2b932a55a5f3abf4a8fd571a0e0518ac996192",
               "expected_md5": "16ea554e1deb97e12759d89c078bff34", "phase": "intraoperative_before_resection"},
    "during": {"path": "data/annotations/resect-seg-v1/originals/Case3-US-during.nii.gz",
               "bytes": 9156131, "sha256": "431ae1e23bd2984801c0b0d2720c9f4c240a404bab5fe37f5d8afdac521ddb20",
               "expected_md5": "5043491e6b4cea1fe3be1003a42a2cf5", "phase": "intraoperative_during_resection"},
}
PINS = {
    COHORT: COHORT_SHA,
    "manifests/resect-train-originals-v1.json": "1764a0ce75b449d542ef51578c54cab6202b66d72c71b811999cb51a55164135",
    "manifests/resect-case3-cavity-pilot-v1.json": "9765796da5abc4464a50e063703c60ef9d77387667475d4ccb317eb596e0f08f",
    "artifacts/resect-train-originals-execution-v1/completion-metadata/resect-original-Case3-US-before.nii.gz.json": "1c87d80c1ac8ba38599da8c1325e703031d57814a2ebd78c0e011a8c363195ff",
    "artifacts/resect-component-admission-v1/acquisition-complete-01/worker-result.json": "cc3e33c5a281b9567de89771fd4cd650ff32bc7b27ee9df3bfd4148be2194afd",
    "artifacts/resect-component-admission-v1/pair-qc-01/worker-result.json": "fddfe22f3e9c1db9ae949a21db9edb6d8a8115f1958586d06926ee43e1844982",
    "scripts/poc_ixi265_registration.py": "ea1b4f43b20b74d9a34d1458b39870011bf35f0495cb411f2eecce32963eb02b",
    "scripts/run_resect_case4_sparse_update.py": "0ab08e48899017707451d1d67528decbdf1452f0c5828c8c2ac2f3a1bbc797f6",
    "scripts/real_intake_io.py": "496821360b1daee79459952f970d299856baacad75fcc57ee589e55115036fd8",
    "src/resectionlab/preprocessing.py": "dbd2c4ba9551e2aa0d454b638e256ad8bc6cdf45dcdc8ec08e091e43c3eb14e5",
    "src/resectionlab/diffusion.py": "e7f57bf7467fb3cb5d6b9b3f1c80e4665710713124d0720aa84d31ccc03b8e21",
}
CONFIG = {"threads": 2, "metric": "Mattes_MI32", "sampling_seed": 41,
          "sampling_fraction": .2, "coarse_spacing_mm": 1.5, "iterations_per_level": 80,
          "shrink_levels": [4, 2, 1], "smoothing_sigmas_mm": [2., 1., 0.],
          "initial_pose": "physical_identity", "optimizer_seconds": 180.,
          "learning_rate": 1., "minimum_step": .002, "relaxation_factor": .5,
          "gradient_magnitude_tolerance": 1e-5, "support_erosion_coarse_voxels": 2,
          "minimum_metric_support_voxels": 1000}
LIMITS = {"preflight_seconds": 30., "worker_wall_seconds": 240., "workers": 1,
          "maximum_source_voxels_each": 32000000, "maximum_expanded_bytes_each": 128 * 1024**2,
          "maximum_compressed_pair_bytes": 32 * 1024**2, "maximum_record_bytes": 1024**2,
          "memory_planning_envelope_bytes": 4 * 1024**3, "sampled_family_rss_stop_bytes": 4 * 1024**3,
          "memory_sampling_interval_seconds": .2, "hard_rss_limit_enforced": False}
SUPPORT = ("Fixed positive ultrasound signal inside one-voxel native geometric coverage; "
           "intersect moving native geometric coverage at identity on the 1.5 mm grid; "
           "erode two coarse voxels. Image-derived metric support only, never a brain, "
           "tissue, tumor, cavity, vessel, or safe-access mask.")


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read_json(path):
    path = Path(path)
    if path.is_symlink() or path.stat().st_size > LIMITS["maximum_record_bytes"]:
        raise ValueError("METADATA_RECORD_SIZE_OR_LINK")
    return json.loads(path.read_text())


def publish(path, value):
    """Exclusive, atomic publication; content hashes detect later outside changes."""
    path = Path(path)
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    payload = encoded(value)
    if len(payload) > LIMITS["maximum_record_bytes"]:
        raise ValueError("OUTPUT_RECORD_LIMIT")
    atomic_preserve(path, payload)
    path.chmod(0o444)


def validate_identity(case_id, inputs, cohort, cohort_sha):
    if case_id != CASE or inputs != INPUTS or cohort_sha != COHORT_SHA:
        raise ValueError("WRONG_ID_SOURCE_OR_COHORT")
    members = [m for m in cohort.get("members", []) if m.get("patient_group") == CASE]
    if (len(members) != 1 or members[0].get("case_id") != "Case3"
            or members[0].get("role") != "TRAIN"
            or members[0].get("all_timepoints_and_derivatives_grouped") is not True):
        raise ValueError("CASE3_TRAIN_ROLE_REQUIRED")


def verify_metadata():
    for relative, expected in PINS.items():
        if digest(ROOT / relative) != expected:
            raise ValueError("PIN_CHANGED: " + relative)
    validate_identity(CASE, INPUTS, read_json(ROOT / COHORT), digest(ROOT / COHORT))


def inspect_header(raw):
    record = header_record(raw)
    header = nib.Nifti1Header.from_fileobj(io.BytesIO(raw), check=False)
    affine = np.asarray(record["selected_affine"])
    spacing = np.linalg.norm(affine[:3, :3], axis=0)
    axes = affine[:3, :3] / spacing
    if not np.allclose(axes.T @ axes, np.eye(3), atol=1e-4, rtol=0):
        raise ValueError("SHEARED_OR_NONORTHOGONAL_SFORM")
    voxels = math.prod(record["shape"])
    expanded = 352 + voxels * 4
    slope, intercept = float(header["scl_slope"]), float(header["scl_inter"])
    if (header.get_data_dtype() != np.dtype("float32") or float(header["vox_offset"]) != 352.
            or not np.isfinite([slope, intercept]).all() or slope <= 0
            or not np.allclose(header.get_zooms(), spacing, atol=1e-5, rtol=1e-5)):
        raise ValueError("SOURCE_STORAGE_OR_SCALING_UNEXPECTED")
    if voxels > LIMITS["maximum_source_voxels_each"] or expanded > LIMITS["maximum_expanded_bytes_each"]:
        raise ValueError("HEADER_RESOURCE_LIMIT")
    record.update({"dtype": "float32", "voxels": voxels, "float32_array_bytes": voxels * 4,
                   "expected_expanded_bytes": expanded, "vox_offset": 352,
                   "source_scaling_slope": slope, "source_scaling_intercept": intercept,
                   "spacing_from_sform_mm": spacing.tolist(), "orthogonality_verified": True,
                   "inactive_qform_reconstructed": False, "qform_qfac_raw": float(header["pixdim"][0]),
                   "full_payload_length_and_finiteness_verified": False})
    return record


def patient_read_guard(data_root, allowed_images):
    """Worker-local Python audit guard; no tag or other patient payload reads."""
    data_root = Path(data_root).resolve()
    allowed = {Path(p).resolve() for p in allowed_images}
    def guard(event, args):
        if event != "open" or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        path = Path(os.fsdecode(args[0])).resolve()
        if path.suffix.lower() == ".tag":
            raise PermissionError("LANDMARK_ACCESS_PROHIBITED")
        if path == data_root or data_root in path.parents:
            mode, flags = args[1:3]
            writing = (isinstance(mode, str) and any(c in mode for c in "wax+")) or (
                isinstance(flags, int) and bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC)))
            if path not in allowed or writing:
                raise PermissionError("NONALLOWLISTED_PATIENT_ACCESS")
    return guard


def runtime_record():
    import SimpleITK as sitk
    from SimpleITK import _SimpleITK, SimpleITK
    import psutil
    runtime_files = {"python_executable": Path(sys.executable).resolve(),
                     "SimpleITK_native": Path(_SimpleITK.__file__).resolve(),
                     "SimpleITK_wrapper": Path(SimpleITK.__file__).resolve()}
    return {"python_executable": str(runtime_files["python_executable"]),
            "python_version": sys.version.split()[0], "numpy": np.__version__,
            "nibabel": nib.__version__, "SimpleITK": sitk.Version_VersionString(), "psutil": psutil.__version__,
            "binary_and_wrapper_bindings": {key: {"path": str(path), "sha256": digest(path)}
                                            for key, path in runtime_files.items()}}


def preflight():
    started = time.monotonic()
    deadline = started + LIMITS["preflight_seconds"]
    verify_metadata()
    if sum(v["bytes"] for v in INPUTS.values()) > LIMITS["maximum_compressed_pair_bytes"]:
        raise ValueError("COMPRESSED_PAIR_BUDGET")
    headers = {}
    for name, entry in INPUTS.items():
        path = ROOT / entry["path"]
        verify_source_file(path, entry, deadline=deadline)
        with gzip.open(path, "rb") as handle:
            headers[name] = inspect_header(handle.read(348))
        verify_source_file(path, entry, deadline=deadline)
        headers[name]["resolved_path"] = str(path.resolve())
    pair_bytes = sum(h["float32_array_bytes"] for h in headers.values())
    estimated = 12 * pair_bytes
    if estimated > LIMITS["memory_planning_envelope_bytes"]:
        raise ValueError("MEMORY_PLANNING_ENVELOPE")
    check_deadline(deadline)
    return {"schema": SCHEMA + "-preflight", "status": "headers_verified_fit_not_run", "case_id": CASE,
            "role": "TRAIN", "cohort_sha256": COHORT_SHA, "inputs": INPUTS, "headers": headers,
            "pins": PINS, "runner_sha256": digest(__file__), "runtime": runtime_record(),
            "configuration": CONFIG, "limits": LIMITS, "metric_support": SUPPORT,
            "pair_float32_bytes": pair_bytes, "memory_planning_estimate_bytes": estimated,
            "memory_estimate_basis": "12 times native pair float32 bytes; conservative planning estimate, not measured peak RSS",
            "elapsed_seconds": time.monotonic() - started, "decoded_array_bytes": 0,
            "landmark_coordinates_read": False, "fit_run": False, "anatomical_alignment_accepted": False,
            "frame_claim": "Active NIfTI sform RAS mm; inactive qform preserved. Header convention only; scanner/tracker calibration and anatomy unverified.",
            "rights": "Original RESECT v1 CC-BY-4.0; no RESECT-SEG masks used. Retain source attribution and preprocessing lineage."}


def require_fit_release(preflight_path, release_path, output):
    before = read_json(preflight_path)
    release = read_json(release_path)
    if (release.get("schema") != SCHEMA + "-root-release" or release.get("decision") != "release_one_fit"
            or release.get("case_id") != CASE or release.get("preflight_sha256") != digest(preflight_path)
            or release.get("runner_sha256") != digest(__file__)
            or release.get("output_directory") != str(output.resolve())
            or not output.resolve().is_relative_to((ROOT / "outputs").resolve())
            or release.get("landmark_access_permitted") is not False):
        raise ValueError("EXACT_ROOT_FIT_RELEASE_REQUIRED")
    if (before.get("schema") != SCHEMA + "-preflight" or before.get("status") != "headers_verified_fit_not_run"
            or before.get("runner_sha256") != digest(__file__) or before.get("runtime") != runtime_record()
            or before.get("pins") != PINS or before.get("configuration") != CONFIG or before.get("limits") != LIMITS):
        raise ValueError("PREFLIGHT_IDENTITY_CHANGED")
    validate_identity(before.get("case_id"), before.get("inputs"), read_json(ROOT / COHORT), before.get("cohort_sha256"))
    verify_metadata()
    return before


def signal_support(array):
    if array.ndim != 3 or not np.isfinite(array).all():
        raise ValueError("INVALID_SIGNAL_ARRAY")
    support = (array > 0) & rigid.interior_coverage(array.shape).astype(bool)
    if not support.any():
        raise ValueError("NO_POSITIVE_SIGNAL_SUPPORT")
    return support


def fit_arrays(before_array, before_affine, during_array, during_affine):
    """Array-only adapter. No points, labels, learned masks, or file inputs."""
    support = signal_support(before_array)
    # Reuse the pinned algorithm verbatim. Its historical parameter name is
    # `brain`; this caller supplies only the explicitly non-anatomical support.
    fitted, report = rigid.rigid_candidate(before_array, before_affine, during_array, during_affine,
                                           support, timeout=CONFIG["optimizer_seconds"],
                                           iterations=CONFIG["iterations_per_level"])
    report["before_RAS_mm_to_during_RAS_mm"] = report.pop("T1_RAS_mm_to_MRA_RAS_mm")
    report["during_RAS_mm_to_before_RAS_mm"] = report.pop("MRA_RAS_mm_to_T1_RAS_mm")
    report["fixed"], report["moving"] = "before_resection_US", "during_resection_US"
    report["sampling_direction"] = "before-RAS output physical point -> during-RAS input physical point"
    windows = report.pop("normalization_windows")
    report["normalization_windows"] = {"before": windows["T1"], "during": windows["MRA"]}
    report["configuration"] = CONFIG
    report["metric_support"] = SUPPORT
    report["native_fixed_positive_signal_support_voxels"] = int(support.sum())
    report["landmarks_used"] = False
    report["identity_baseline_before_RAS_to_during_RAS"] = np.eye(4).tolist()
    return fitted, report


def worker(preflight_path, release_path, output):
    baseline = require_fit_release(preflight_path, release_path, output)
    sys.addaudithook(patient_read_guard(ROOT / "data", [ROOT / v["path"] for v in INPUTS.values()]))
    current = preflight()  # Repeat byte/header gates before any voxel decode.
    if current["headers"] != baseline["headers"]:
        raise ValueError("HEADERS_CHANGED")
    arrays, affines = {}, {}
    for name, entry in INPUTS.items():
        image, arrays[name], expanded = rigid.checked_source(ROOT / entry["path"], entry["sha256"])
        if (expanded != baseline["headers"][name]["expected_expanded_bytes"]
                or not np.array_equal(image.affine, baseline["headers"][name]["selected_affine"])):
            raise ValueError("DECODED_GEOMETRY_OR_LENGTH_CHANGED")
        verify_source_file(ROOT / entry["path"], entry)
        affines[name] = image.affine
    _, report = fit_arrays(arrays["before"], affines["before"], arrays["during"], affines["during"])
    final_fixity = {name: verify_source_file(ROOT / entry["path"], entry) for name, entry in INPUTS.items()}
    verify_metadata()
    final_runtime = runtime_record()
    if final_runtime != baseline["runtime"] or digest(__file__) != baseline["runner_sha256"]:
        raise ValueError("RUNTIME_OR_RUNNER_CHANGED_DURING_FIT")
    publish(output / "candidate.json", {"schema": SCHEMA + "-candidate", "case_id": CASE,
            "preflight_sha256": digest(preflight_path), "fit_release_sha256": digest(release_path),
            "runner_sha256": digest(__file__), "sources_rechecked_after_each_decode": True,
            "source_fixity_after_fit": final_fixity, "runtime": final_runtime, "report": report})


def seal_attempt(output, status, exit_code, preflight_sha, release_sha):
    """Only supervised success can publish a freeze; failed candidates stay ineligible."""
    freeze_sha = None
    supervision_sha = digest(output / "supervision.json") if (output / "supervision.json").exists() else None
    if status == "completed" and exit_code == 0:
        supervision = read_json(output / "supervision.json")
        if (supervision.get("status") != "completed" or supervision.get("exit_code") != 0
                or supervision.get("worker_reaped") is not True or supervision.get("samples", 0) < 1
                or supervision.get("sampled_family_rss_stop_bytes") != LIMITS["sampled_family_rss_stop_bytes"]
                or supervision.get("peak_family_rss_bytes_sampled", math.inf) > LIMITS["sampled_family_rss_stop_bytes"]):
            raise ValueError("SUCCESSFUL_MEMORY_SUPERVISION_REQUIRED")
        candidate = read_json(output / "candidate.json")
        from resectionlab.preprocessing import validate_rigid
        if (candidate.get("schema") != SCHEMA + "-candidate" or candidate.get("case_id") != CASE
                or candidate.get("preflight_sha256") != preflight_sha
                or candidate.get("fit_release_sha256") != release_sha
                or candidate.get("runner_sha256") != digest(__file__)
                or candidate.get("runtime") != runtime_record()
                or candidate.get("sources_rechecked_after_each_decode") is not True
                or candidate.get("source_fixity_after_fit") != {name: entry["sha256"] for name, entry in INPUTS.items()}
                or candidate["report"].get("landmarks_used") is not False
                or candidate["report"].get("configuration") != CONFIG
                or candidate["report"].get("fixed") != "before_resection_US"
                or candidate["report"].get("moving") != "during_resection_US"
                or candidate["report"].get("sampling_direction") != "before-RAS output physical point -> during-RAS input physical point"
                or candidate["report"].get("identity_baseline_before_RAS_to_during_RAS") != np.eye(4).tolist()):
            raise ValueError("CANDIDATE_IDENTITY_CHANGED")
        matrix = np.asarray(candidate["report"]["before_RAS_mm_to_during_RAS_mm"])
        inverse = np.asarray(candidate["report"].get("during_RAS_mm_to_before_RAS_mm"))
        validate_rigid(matrix)
        validate_rigid(inverse)
        if not np.allclose(inverse, np.linalg.inv(matrix), rtol=0, atol=1e-9):
            raise ValueError("INCONSISTENT_FORWARD_INVERSE")
        frozen = {**candidate, "schema": SCHEMA + "-frozen", "candidate_sha256": digest(output / "candidate.json"),
                  "supervision_sha256": supervision_sha,
                  "evaluation_requires_separate_root_release": True, "anatomical_alignment_accepted": False}
        publish(output / "frozen-transform.json", frozen)
        freeze_sha = digest(output / "frozen-transform.json")
    publish(output / "result.json", {"schema": SCHEMA + "-result", "case_id": CASE, "status": status,
            "exit_code": exit_code, "frozen_transform_sha256": freeze_sha, "preflight_sha256": preflight_sha,
            "fit_release_sha256": release_sha, "supervision_sha256": supervision_sha,
            "evaluation_run": False, "anatomical_alignment_accepted": False})


def evaluation_gate(output, release):
    """Metadata-only gate for a future separate evaluator; never opens a tag."""
    result = read_json(output / "result.json")
    if (result.get("schema") != SCHEMA + "-result" or result.get("status") != "completed"
            or result.get("exit_code") != 0 or not result.get("frozen_transform_sha256")):
        raise ValueError("SUCCESSFULLY_SUPERVISED_FREEZE_REQUIRED")
    if (release.get("schema") != SCHEMA + "-evaluation-release" or release.get("case_id") != CASE
            or release.get("decision") != "release_separate_evaluation"
            or release.get("fit_result_sha256") != digest(output / "result.json")
            or release.get("frozen_transform_sha256") != digest(output / "frozen-transform.json")
            or release.get("frozen_transform_sha256") != result["frozen_transform_sha256"]
            or result.get("supervision_sha256") != digest(output / "supervision.json")
            or release.get("full_tag_role") != "evaluation_only_all_points"
            or release.get("refitting_or_parameter_selection_permitted") is not False):
        raise ValueError("SEPARATE_ROOT_EVALUATION_RELEASE_REQUIRED")
    return read_json(output / "frozen-transform.json")


def sample_family(pid, tracked):
    """Reuse the IXI psutil sampling pattern; retain observed descendants."""
    import psutil
    try:
        parent = psutil.Process(pid)
        tracked[pid] = parent
        for child in parent.children(recursive=True):
            tracked[child.pid] = child
    except psutil.NoSuchProcess:
        pass
    rss, members = {}, []
    for member_pid, process in tracked.items():
        try:
            if process.is_running() and process.status() != psutil.STATUS_ZOMBIE:
                rss[member_pid] = process.memory_info().rss
                members.append(member_pid)
        except psutil.NoSuchProcess:
            pass
    return {"worker_rss_bytes": rss.get(pid, 0), "family_rss_bytes": sum(rss.values()),
            "available_bytes": psutil.virtual_memory().available, "live_members": members}


def supervise_sampled(command, log, *, deadline, on_start, rss_stop=None, observer=sample_family):
    with termination_cleanup():
        return _supervise_sampled(command, log, deadline=deadline, on_start=on_start,
                                  rss_stop=rss_stop, observer=observer)


def _supervise_sampled(command, log, *, deadline, on_start, rss_stop=None, observer=sample_family):
    """Sampled stop enforcement, not an OS memory limit or exact peak measure.

    Adapted from the pinned IXI watchdog; retain unconditional process-group
    cleanup from real_intake_io, including cancellation and observer failures.
    """
    cap = LIMITS["sampled_family_rss_stop_bytes"] if rss_stop is None else rss_stop
    record = {"schema": SCHEMA + "-supervision", "status": "supervisor_failed", "exit_code": None, "samples": 0,
              "peak_worker_rss_bytes_sampled": 0, "peak_family_rss_bytes_sampled": 0,
              "minimum_available_memory_bytes_sampled": None, "sampled_family_rss_stop_bytes": cap,
              "sampling_interval_seconds": LIMITS["memory_sampling_interval_seconds"],
              "memory_control": "sampled_stop_enforcement", "os_hard_memory_limit": False,
              "process_group_cleanup_attempted": False, "worker_reaped": False}
    process, tracked = None, {}
    started = time.monotonic()
    try:
        check_deadline(deadline)
        with log.open("xb") as stream:
            environment = dict(os.environ, OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2",
                               MKL_NUM_THREADS="2", ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS="2")
            process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                       env=environment, start_new_session=True)
            record["pid"] = process.pid
            on_start(process.pid)
            while True:
                sample = observer(process.pid, tracked)
                record["samples"] += 1
                record["peak_worker_rss_bytes_sampled"] = max(record["peak_worker_rss_bytes_sampled"], sample["worker_rss_bytes"])
                record["peak_family_rss_bytes_sampled"] = max(record["peak_family_rss_bytes_sampled"], sample["family_rss_bytes"])
                previous = record["minimum_available_memory_bytes_sampled"]
                record["minimum_available_memory_bytes_sampled"] = min(sample["available_bytes"], previous if previous is not None else sample["available_bytes"])
                code = process.poll()
                if sample["family_rss_bytes"] > cap:
                    record["status"] = "memory_stop"
                    break
                if time.monotonic() >= deadline:
                    record["status"] = "timeout"
                    break
                if code is not None:
                    record["status"] = ("descendants_outlived_worker" if any(p != process.pid for p in sample["live_members"])
                                        else "completed" if code == 0 else "worker_failed")
                    break
                if process.pid not in sample["live_members"]:
                    raise RuntimeError("Running worker memory sample unavailable")
                time.sleep(min(LIMITS["memory_sampling_interval_seconds"], max(0., deadline - time.monotonic())))
    except BaseException as error:
        record["status"] = "supervisor_failed"
        record["error"] = type(error).__name__ + ": " + str(error)
    finally:
        if process is not None:
            record["process_group_cleanup_attempted"] = True
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            except OSError as error:
                record["status"] = "cleanup_failed"
                record["cleanup_error"] = type(error).__name__ + ": " + str(error)
            try:
                if process.poll() is None:
                    process.kill()
            except OSError as error:
                record["status"] = "cleanup_failed"
                record["worker_cleanup_error"] = type(error).__name__ + ": " + str(error)
            # An observed child that changed groups must also be stopped.
            for member in tracked.values():
                try:
                    if member.is_running():
                        member.kill()
                except Exception as error:
                    import psutil
                    if not isinstance(error, psutil.NoSuchProcess):
                        record["status"] = "cleanup_failed"
                        record["cleanup_error"] = type(error).__name__ + ": " + str(error)
            try:
                record["exit_code"] = process.wait(timeout=2)
                record["worker_reaped"] = True
            except subprocess.TimeoutExpired:
                record["status"] = "cleanup_failed"
        record["elapsed_seconds"] = time.monotonic() - started
    return record


def run_fit(preflight_path, release_path, output):
    require_fit_release(preflight_path, release_path, output)
    output.mkdir(parents=True, exist_ok=False)
    preflight_sha, release_sha = digest(preflight_path), digest(release_path)
    status, exit_code = "supervisor_failed", None
    try:
        supervision = supervise_sampled([sys.executable, str(Path(__file__).resolve()), "worker", "--preflight",
            str(preflight_path.resolve()), "--release", str(release_path.resolve()), "--output", str(output.resolve())],
            output / "worker.log", deadline=time.monotonic() + LIMITS["worker_wall_seconds"],
            on_start=lambda pid: publish(output / "started.json", {"pid": pid, "preflight_sha256": preflight_sha,
                "fit_release_sha256": release_sha, "configuration": CONFIG, "limits": LIMITS}))
        publish(output / "supervision.json", supervision)
        status, exit_code = supervision["status"], supervision["exit_code"]
        if status == "completed" and exit_code == 0:
            require_fit_release(preflight_path, release_path, output)
        seal_attempt(output, status, exit_code, preflight_sha, release_sha)
    except BaseException:
        if not (output / "result.json").exists():
            seal_attempt(output, "supervisor_or_seal_failed", None, preflight_sha, release_sha)
        raise
    return 0 if status == "completed" and exit_code == 0 else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["preflight", "fit", "worker"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preflight", type=Path)
    parser.add_argument("--release", type=Path)
    args = parser.parse_args()
    if args.stage == "preflight":
        try:
            record = preflight()
        except Exception as error:
            publish(args.output, {"schema": SCHEMA + "-preflight", "status": "failed", "case_id": CASE,
                "error": type(error).__name__ + ": " + str(error), "fit_run": False})
            return 1
        publish(args.output, record)
        return 0
    if args.preflight is None or args.release is None:
        parser.error("fit/worker require exact preflight and ROOT release files")
    if args.stage == "worker":
        worker(args.preflight, args.release, args.output)
        return 0
    return run_fit(args.preflight, args.release, args.output)


if __name__ == "__main__":
    raise SystemExit(main())
