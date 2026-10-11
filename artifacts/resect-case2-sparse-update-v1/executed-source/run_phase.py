#!/usr/bin/env python3
"""Three explicit Case2 stages. Unreleased source preparation; no automatic phases.

Reuses real_intake_io's owned process-group timeout/cleanup. Memory is observed
ru_maxrss, not a hard RSS limit. No image array construction or new acquisition.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import resource
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "build/hybrid-real-observation-case2-v1"
OUT = BASE / "attempt-01"
PHASES = ("qualify", "fit", "evaluate")
CAP = 4 * 1024**2
SCHEMA = "hybrid-real-observation-case2-phase-v1"
PROTOCOL = BASE / "protocol.json"
INDEX = BASE / "source-index.json"
SOURCE_PATHS = (
    "build/hybrid-real-observation-case2-v1/run_phase.py",
    "build/hybrid-real-observation-case2-v1/case2_adapter.py",
    "build/hybrid-real-observation-case2-v1/protocol.json",
    "scripts/run_resect_case4_sparse_update.py", "scripts/real_intake_io.py",
    "scripts/mechanics_patient_comparison.py", "src/resectionlab/mechanics_landmarks.py",
    "src/resectionlab/core.py", "src/resectionlab/__init__.py",
    "manifests/resect-component-cohort-v1.json",
    "manifests/experiments/resect-case4-sparse-update-v1.json",
)
for path in (str(BASE), str(ROOT / "src"), str(ROOT)):
    sys.path.insert(0, path)
from scripts.real_intake_io import atomic_preserve, check_deadline, supervise, termination_cleanup, verify_source_file
from scripts.run_resect_case4_sparse_update import header_record
import case2_adapter as adapter


def encode(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def sha(payload):
    return hashlib.sha256(payload).hexdigest()


def relative(path):
    return str(path.relative_to(ROOT))


def safe_path(name):
    if not isinstance(name, str) or Path(name).is_absolute() or ".." in Path(name).parts:
        raise ValueError("EXACT_RELATIVE_PATH_REQUIRED")
    path = ROOT / name
    if path.resolve() != path.absolute():
        raise ValueError("PATH_ALIAS_REFUSED")
    return path


def read(path, maximum=CAP):
    if path.is_symlink() or path.stat().st_size > maximum:
        raise ValueError("BOUNDED_NONALIAS_FILE_REQUIRED")
    with path.open("rb") as handle:
        payload = handle.read(maximum + 1)
    if len(payload) > maximum:
        raise ValueError("READ_CAP")
    return payload


def checked(binding, expected):
    if set(binding) != {"path", "sha256"} or binding["path"] != relative(expected):
        raise ValueError("EXACT_PREREQUISITE_PATH_REQUIRED")
    payload = read(expected)
    if sha(payload) != binding["sha256"]:
        raise ValueError("PREREQUISITE_BYTES_CHANGED")
    return json.loads(payload)


def tree_bytes(path):
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) if path.exists() else 0


def save(path, value):
    payload = encode(value)
    if path.exists():
        raise ValueError("EXCLUSIVE_OUTPUT_ALREADY_EXISTS")
    if tree_bytes(OUT) + len(payload) > CAP:
        raise ValueError("TOTAL_OUTPUT_CAP")
    atomic_preserve(path, payload)
    return {"path": relative(path), "sha256": sha(payload)}


def source_hashes():
    return {name: sha(read(safe_path(name))) for name in SOURCE_PATHS}


def runtime_identity():
    executable = Path(sys.executable).resolve()
    with executable.open("rb") as handle:
        executable_sha = hashlib.file_digest(handle, "sha256").hexdigest()
    return {"executable": str(executable), "executable_sha256": executable_sha,
            "prefix": sys.prefix, "python": sys.version,
            "numpy": importlib.metadata.version("numpy"),
            "nibabel": importlib.metadata.version("nibabel")}


def dependency_paths(phase):
    result = {}
    if phase in ("fit", "evaluate"):
        for name in ("frame", "partition", "result", "supervision"):
            result["qualify_" + name] = OUT / "qualify" / (name + ".json")
    if phase == "evaluate":
        for name in ("result", "supervision"):
            result["fit_" + name] = OUT / "fit" / (name + ".json")
        result["fit_freeze"] = OUT / "fit/comparison/freeze.json"
    return result


def authenticate(phase, release_path, release_sha256):
    if phase not in PHASES or release_path != BASE / "releases" / (phase + ".json"):
        raise ValueError("FIXED_PHASE_RELEASE_PATH_REQUIRED")
    payload = read(release_path)
    if sha(payload) != release_sha256:
        raise ValueError("RELEASE_BYTES_CHANGED")
    release = json.loads(payload)
    if (set(release) != {"schema", "phase", "authorizer", "authorized", "source_index", "dependencies"}
            or release["schema"] != SCHEMA + "-release" or release["phase"] != phase
            or release["authorizer"] != "root" or release["authorized"] is not True):
        raise ValueError("EXPLICIT_ROOT_PHASE_RELEASE_REQUIRED")
    index = checked(release["source_index"], INDEX)
    if index["sources"] != source_hashes() or index["runtime"] != runtime_identity():
        raise ValueError("SOURCE_OR_RUNTIME_CHANGED")
    if (Path(sys.executable).absolute() != ROOT / ".venv/bin/python"
            or Path(sys.prefix).resolve() != (ROOT / ".venv").resolve()):
        raise ValueError("DECLARED_PROJECT_RUNTIME_REQUIRED")
    protocol = json.loads(read(PROTOCOL))
    if (protocol["patient_group"] != adapter.PATIENT or protocol["role"] != "TRAIN"
            or protocol["methods"] != list(adapter.comparison.METHODS)
            or protocol["coordinate_convention"] != adapter.CONVENTION
            or protocol["execution_released"] is not False):
        raise ValueError("FIXED_PROTOCOL_REQUIRED")
    cohort = checked(protocol["role_binding"], ROOT / "manifests/resect-component-cohort-v1.json")
    roles = [row["role"] for row in cohort["members"] if row["patient_group"] == adapter.PATIENT]
    if roles != ["TRAIN"]:
        raise ValueError("CASE2_TRAIN_REQUIRED")
    if set(release["dependencies"]) != set(dependency_paths(phase)):
        raise ValueError("EXACT_PHASE_PREREQUISITES_REQUIRED")
    dependencies = {key: checked(release["dependencies"][key], path)
                    for key, path in dependency_paths(phase).items()}
    validate_predecessors(phase, dependencies, release)
    return release, protocol, dependencies


def validate_predecessors(phase, dependencies, release):
    """Pure predecessor/result joins, after checked() authenticated each byte blob."""
    for previous in (() if phase == "qualify" else ("qualify",) if phase == "fit" else ("qualify", "fit")):
        receipt = dependencies[previous + "_supervision"]
        result = dependencies[previous + "_result"]
        if (receipt.get("status") != "completed" or receipt.get("exit_code") != 0
                or receipt.get("scientific_result_accepted") is not True
                or result.get("status") != "completed" or result.get("phase") != previous
                or result.get("source_index_sha256") != release["source_index"]["sha256"]
                or receipt.get("source_index_sha256") != release["source_index"]["sha256"]
                or result.get("release_sha256") != receipt.get("release_sha256")
                or receipt.get("result_sha256") != release["dependencies"][previous + "_result"]["sha256"]):
            raise ValueError("PRIOR_PHASE_NOT_AUTHENTICATED_COMPLETE")
    if phase in ("fit", "evaluate"):
        for key in ("frame", "partition"):
            if dependencies["qualify_result"][key] != release["dependencies"]["qualify_" + key]:
                raise ValueError("QUALIFY_OUTPUT_JOIN_CHANGED")
    if phase == "evaluate" and dependencies["fit_result"]["freeze"] != release["dependencies"]["fit_freeze"]:
        raise ValueError("FROZEN_FIELD_JOIN_CHANGED")


def patient_guard(phase, inputs):
    allowed = {safe_path(inputs[key]["path"]) for key in (inputs if phase == "qualify" else ("tag",))}
    data_root = (ROOT / "data").resolve()
    def guard(event, args):
        if event == "socket.connect":
            raise PermissionError("NETWORK_NOT_PERMITTED")
        if event != "open" or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        path = Path(os.fsdecode(args[0])).absolute()
        resolved = path.resolve()
        if path.is_relative_to(ROOT / "data") or resolved.is_relative_to(data_root):
            mode, flags = args[1:3]
            writes = ((isinstance(mode, str) and any(c in mode for c in "wax+"))
                      or (isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC)))
            if path not in allowed or resolved != path or writes:
                raise PermissionError("EXACT_READ_ONLY_PATIENT_SCOPE_REQUIRED")
    return guard


def qualify(protocol, deadline):
    inputs = protocol["inputs"]
    headers = {}
    for key in ("before", "during"):
        path = safe_path(inputs[key]["path"])
        verify_source_file(path, inputs[key], deadline=deadline)
        with gzip.open(path, "rb") as handle:
            headers[key] = header_record(handle.read(348))
    identity = adapter.np.eye(4).tolist()
    frame = {"schema": SCHEMA + "-frame", "status": "coordinate_convention_verified",
             "source_image_sha256": adapter.BEFORE_SHA, "destination_image_sha256": adapter.DURING_SHA,
             "source_world_to_ras_mm": identity, "destination_world_to_ras_mm": identity,
             "convention": adapter.CONVENTION, "headers": headers,
             "anatomical_alignment_accepted": False, "physical_clearance_mm": None,
             "cavity_support": None, "total_registration_uncertainty_mm": None}
    payload = tag_bytes(protocol, deadline)
    partition = adapter.source_partition(payload)
    binding = adapter.lm.LandmarkPairBinding(adapter.PATIENT, adapter.lm.DISPLACEMENT_ROLE,
        adapter.TAG_SHA, adapter.BEFORE_SHA, adapter.DURING_SHA)
    sources = adapter.lm.parse_tag_sources(payload, binding)
    coverage = adapter.native_coverage(sources.source_world_mm, sources.row_ids, headers["before"])
    return {"frame": save(OUT / "qualify/frame.json", frame),
            "partition": save(OUT / "qualify/partition.json", partition),
            "source_coverage": coverage, "destination_coordinates_parsed": 0,
            "image_arrays_accessed": False}


def tag_bytes(protocol, deadline):
    item = protocol["inputs"]["tag"]
    path = safe_path(item["path"])
    verify_source_file(path, item, deadline=deadline)
    return read(path, adapter.TAG_BYTES)


def scientific_stage(phase, protocol, release, deps, deadline):
    if phase == "qualify":
        return qualify(protocol, deadline)
    arguments = dict(payload=tag_bytes(protocol, deadline), partition_record=deps["qualify_partition"],
        frame=deps["qualify_frame"], frame_sha256=release["dependencies"]["qualify_frame"]["sha256"])
    if phase == "fit":
        return adapter.fit_and_freeze(ROOT, **arguments,
            protocol_binding={"path": relative(PROTOCOL), "sha256": sha(read(PROTOCOL))},
            output_directory=relative(OUT / "fit/comparison"))
    value = adapter.evaluate_once(ROOT, **arguments, freeze_binding=release["dependencies"]["fit_freeze"])
    source = deps["qualify_result"]["source_coverage"]
    destination = (deps["fit_result"]["B_destination_coverage"]["rows"]
                   + value["V_destination_coverage"]["rows"])
    destination.sort(key=lambda row: row["row_id"])
    expected = list(range(1, deps["qualify_partition"]["source_row_count"] + 1))
    if [row["row_id"] for row in destination] != expected:
        raise ValueError("ALL_ORIGINAL_DESTINATION_ROWS_REQUIRED")
    value["all_original_row_coverage"] = {"source": source,
        "destination": {"meaning": source["meaning"], "rows": destination, "excluded_rows": []}}
    return value


def worker(phase, release_path, release_sha256):
    start = time.monotonic()
    release, protocol, deps = authenticate(phase, release_path, release_sha256)
    directory = OUT / phase
    attempt = json.loads(read(directory / "attempt.json"))
    if attempt.get("release_sha256") != release_sha256 or attempt.get("phase") != phase:
        raise ValueError("SUPERVISED_ATTEMPT_REQUIRED")
    save(directory / "worker-started.json", {"phase": phase, "release_sha256": release_sha256})
    resource.setrlimit(resource.RLIMIT_FSIZE, (1024**2, 1024**2))
    sys.addaudithook(patient_guard(phase, protocol["inputs"]))
    result = {"schema": SCHEMA + "-result", "phase": phase, "status": "failed",
              "release_sha256": release_sha256, "source_index_sha256": release["source_index"]["sha256"]}
    try:
        result.update(scientific_stage(phase, protocol, release, deps, start + 30))
        authenticate(phase, release_path, release_sha256)
        for key in (protocol["inputs"] if phase == "qualify" else ("tag",)):
            item = protocol["inputs"][key]
            verify_source_file(safe_path(item["path"]), item, deadline=start + 30)
        check_deadline(start + 30)
        if tree_bytes(OUT) > CAP:
            raise ValueError("TOTAL_OUTPUT_CAP")
        result.update(status="completed", original_bytes_unchanged=True)
    except BaseException as error:
        result["error_type"] = type(error).__name__
        # Error messages can contain a source coordinate; never serialize tokens.
        raise
    finally:
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        result.update(elapsed_seconds=time.monotonic() - start,
            worker_peak_rss_bytes=peak if sys.platform == "darwin" else peak * 1024,
            memory_is_observed_not_hard_limit=True)
        save(directory / "result.json", result)


def launch(phase, release_path, release_sha256):
    release, _, _ = authenticate(phase, release_path, release_sha256)
    directory = OUT / phase
    directory.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    receipt = {"schema": SCHEMA + "-supervision", "phase": phase, "status": "not_started",
        "release_sha256": release_sha256, "source_index_sha256": release["source_index"]["sha256"],
        "started_utc": datetime.now(timezone.utc).isoformat(), "scientific_result_accepted": False}
    save(directory / "attempt.json", receipt)
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    try:
        with termination_cleanup():
            status, code = supervise([sys.executable, "-I", "-B", "-X",
                "pycache_prefix=" + str(directory / "fresh-pycache"), str(Path(__file__).resolve()), phase,
                "--release", relative(release_path), "--release-sha256", release_sha256, "--worker"],
                directory / "worker.log", deadline=start + 30,
                on_start=lambda pid: save(directory / "process.json", {"pid": pid, "pgid": pid}))
        receipt.update(status=status, exit_code=code)
        if status == "completed":
            payload = read(directory / "result.json")
            result = json.loads(payload)
            authenticate(phase, release_path, release_sha256)
            accepted = (result.get("status") == "completed" and result.get("phase") == phase
                and result.get("release_sha256") == release_sha256
                and result.get("source_index_sha256") == release["source_index"]["sha256"]
                and result.get("original_bytes_unchanged") is True
                and result.get("elapsed_seconds", math.inf) <= 30
                and time.monotonic() - start <= 30 and tree_bytes(OUT) <= CAP)
            receipt.update(scientific_result_accepted=accepted, result_sha256=sha(payload))
    except BaseException as error:
        receipt.update(status="supervision_failed", error_type=type(error).__name__)
        raise
    finally:
        receipt.update(elapsed_seconds=time.monotonic() - start,
            process_control="real_intake_io.supervise dedicated group killed/reaped in finally",
            output_bytes_before_receipt=tree_bytes(OUT), memory_is_observed_not_hard_limit=True)
        save(directory / "supervision.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=PHASES)
    parser.add_argument("--release", required=True)
    parser.add_argument("--release-sha256", required=True)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    path = safe_path(args.release)
    try:
        if args.check_only:
            authenticate(args.phase, path, args.release_sha256)
            print(json.dumps({"status": "metadata_checks_passed", "patient_reads": 0, "workers": 0}))
        elif args.worker:
            worker(args.phase, path, args.release_sha256)
        elif not launch(args.phase, path, args.release_sha256)["scientific_result_accepted"]:
            raise RuntimeError("STAGE_NOT_ACCEPTED")
    except BaseException as error:
        print(json.dumps({"status": "failed", "error_type": type(error).__name__}), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
