#!/usr/bin/env python3
"""Separately released header, six-B fit, and thirteen-V evaluation stages.

Sparse observed US correspondence only. No MRI, image arrays, domain, FEM,
training, acquisition, retry, alternative settings or anatomical acceptance.
The root-authored release is an authorization record, not an access sandbox.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import re
import resource
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "resect-case4-sparse-update-v1"
MANIFEST = "manifests/experiments/" + SCHEMA + ".json"
OUTPUT = "artifacts/" + SCHEMA
DATA = "data/mechanics/resect-case4-v1/RESECT/NIFTI/Case4/"
PHASES = ("header-qc", "fit-freeze", "evaluate")
B_IDS = (1, 14, 8, 17, 19, 7)
V_IDS = (2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 15, 16, 18)
PARTITION_SHA = "cc9d7776cbb4bb6ba745519e24f008c53e556bfe60d3c0509917377c9db02982"
SOURCE_NAMES = ("scripts/run_resect_case4_sparse_update.py",
                "scripts/mechanics_patient_comparison.py", "scripts/real_intake_io.py",
                "src/resectionlab/mechanics_landmarks.py", "src/resectionlab/core.py",
                "src/resectionlab/__init__.py")
INPUTS = {
    "before": {"path": DATA + "US/Case4-US-before.nii.gz", "bytes": 9955302,
               "sha256": "62f6b6217774653267702acd4a55b16ad5c8cc4980237afaa39e92b102240379",
               "expected_md5": "984ab4b314a55b014e770b32e623b16c"},
    "during": {"path": DATA + "US/Case4-US-during.nii.gz", "bytes": 9796312,
               "sha256": "90fd3e82fbf99f7d45ed25108829f8e61a632ecf6d9a8706b5e4ac6866760a8a",
               "expected_md5": "14b8b7acee3beccd4ee04e96e9201f0c"},
    "tag": {"path": DATA + "Landmarks/Case4-beforeUS-duringUS-full.tag", "bytes": 1304,
            "sha256": "9e2c3c2765bffd2914f8113f8b2e223c06b2f0640e561f03a94049752e8ddec1",
            "expected_md5": "14259fc0f9ab32dfecbe1f2b8830ae86"},
}
FRAME_PATH = OUTPUT + "/header-qc/frame-qc.json"
FREEZE_PATH = OUTPUT + "/fit-freeze/comparison/freeze.json"
DEPENDENCY_PATHS = {"frame_qc": FRAME_PATH, "freeze": FREEZE_PATH,
                    "header_supervision": OUTPUT + "/header-qc/supervision.json",
                    "fit_supervision": OUTPUT + "/fit-freeze/supervision.json"}


def encode(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def safe_path(relative):
    if not isinstance(relative, str) or Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise ValueError("RELATIVE_FIXED_PATH_REQUIRED")
    path = ROOT / relative
    if path.resolve() != path.absolute() or not path.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError("PATH_ALIAS_PROHIBITED")
    return path


def bounded_read(relative, maximum=1024 * 1024):
    with safe_path(relative).open("rb") as handle:
        payload = handle.read(maximum + 1)
    if len(payload) > maximum:
        raise ValueError("READ_LIMIT_EXCEEDED")
    return payload


def checked_binding(binding, expected_path):
    if set(binding) != {"path", "sha256"} or binding["path"] != expected_path:
        raise ValueError("EXACT_DEPENDENCY_REQUIRED")
    payload = bounded_read(expected_path)
    if digest(payload) != binding["sha256"]:
        raise ValueError("DEPENDENCY_CHANGED")
    return json.loads(payload)


def source_hashes():
    return {name: digest(bounded_read(name)) for name in SOURCE_NAMES}


def claim_worker(phase, release_sha):
    attempt = json.loads(bounded_read(OUTPUT + "/" + phase + "/attempt.json"))
    if attempt.get("phase") != phase or attempt.get("release_sha256") != release_sha:
        raise ValueError("SUPERVISED_ATTEMPT_REQUIRED")
    with safe_path(OUTPUT + "/" + phase + "/worker-started.json").open("xb") as handle:
        handle.write(encode({"phase": phase, "release_sha256": release_sha}))
        handle.flush()
        os.fsync(handle.fileno())


def validate_declaration(value):
    if (value.get("schema") != SCHEMA or value.get("patient_group") != "RESECT:Case4"
            or value.get("role") != "DEVELOPMENT" or value.get("inputs") != INPUTS
            or value.get("B_ids") != list(B_IDS) or value.get("V_ids") != list(V_IDS)
            or value.get("partition_sha256") != PARTITION_SHA
            or value.get("methods") != ["no_shift", "proper_rigid", "inverse_distance_squared"]
            or value.get("idw_power") != 2 or value.get("idw_neighbors") != "all_six"
            or value.get("external_field") is not None
            or value.get("output_directory") != OUTPUT
            or value.get("header_rule") != "nifti1_3d_mm_sform1_qform0_finite_invertible"
            or value.get("phase_wall_seconds") != 30
            or value.get("maximum_output_file_bytes") != 1048576):
        raise ValueError("DECLARATION_POLICY_CHANGED")
    return value


def authenticate(release_path, phase):
    """All source/declaration/release checks precede any original patient read."""
    if phase not in PHASES or release_path != OUTPUT + "/releases/" + phase + ".json":
        raise ValueError("EXACT_PHASE_RELEASE_REQUIRED")
    release_bytes = bounded_read(release_path)
    release = json.loads(release_bytes)
    expected_keys = {"schema", "phase", "authorized", "authorizer", "source_commit",
                     "declaration_sha256", "source_sha256", "dependencies"}
    if (set(release) != expected_keys or release["schema"] != SCHEMA + "-release"
            or release["phase"] != phase or release["authorized"] is not True
            or release["authorizer"] != "root"
            or re.fullmatch(r"[0-9a-f]{40}", release["source_commit"]) is None):
        raise ValueError("EXPLICIT_ROOT_RELEASE_REQUIRED")
    protocol = bounded_read(MANIFEST)
    if digest(protocol) != release["declaration_sha256"]:
        raise ValueError("DECLARATION_CHANGED")
    declaration = validate_declaration(json.loads(protocol))
    hashes = source_hashes()
    if hashes != release["source_sha256"]:
        raise ValueError("SOURCE_CLOSURE_CHANGED")
    expected_helpers = {k: v for k, v in hashes.items() if k != SOURCE_NAMES[0]}
    if declaration["reused_source_sha256"] != expected_helpers:
        raise ValueError("REUSED_HELPER_CHANGED")
    for name in (*SOURCE_NAMES, MANIFEST):
        saved = subprocess.run(["git", "show", release["source_commit"] + ":" + name],
                               cwd=ROOT, capture_output=True, check=True, timeout=5).stdout
        if digest(saved) != digest(bounded_read(name)):
            raise ValueError("COMMITTED_SOURCE_REQUIRED")
    required = (() if phase == "header-qc" else ("frame_qc", "header_supervision")
                if phase == "fit-freeze" else ("frame_qc", "header_supervision", "freeze", "fit_supervision"))
    if set(release["dependencies"]) != set(required):
        raise ValueError("EXACT_PHASE_DEPENDENCIES_REQUIRED")
    dependencies = {}
    for key in required:
        dependencies[key] = checked_binding(release["dependencies"][key], DEPENDENCY_PATHS[key])
        if key.endswith("_supervision") and (dependencies[key].get("status") != "completed"
                or dependencies[key].get("scientific_result_accepted") is not True):
            raise ValueError("PREVIOUS_STAGE_NOT_ACCEPTED")
    return declaration, release, dependencies, digest(release_bytes)


def patient_guard(phase):
    allowed = {safe_path(INPUTS[k]["path"]) for k in (("before", "during") if phase == "header-qc" else ("tag",))}
    data_root = (ROOT / "data").resolve()

    def guard(event, args):
        if event != "open" or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        path = Path(os.fsdecode(args[0])).absolute()
        resolved = path.resolve()
        if path.is_relative_to(ROOT / "data") or resolved.is_relative_to(data_root):
            mode, flags = args[1:3]
            writing = (isinstance(mode, str) and any(c in mode for c in "wax+")) or (
                isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC))
            if resolved not in allowed or resolved != path or writing:
                raise PermissionError("PHASE_PATIENT_ACCESS_PROHIBITED")
    return guard


def header_record(raw):
    """Pure bounded header interpretation; no image constructor or array proxy."""
    import nibabel as nib
    import numpy as np
    if len(raw) != 348:
        raise ValueError("NIFTI_HEADER_LENGTH")
    header = nib.Nifti1Header.from_fileobj(io.BytesIO(raw), check=False)
    shape = tuple(int(n) for n in header.get_data_shape())
    qform, qcode = header.get_qform(coded=True)
    sform, scode = header.get_sform(coded=True)
    if (int(header["sizeof_hdr"]) != 348 or bytes(header["magic"]) != b"n+1\0"
            or len(shape) != 3 or min(shape) < 1 or header.get_xyzt_units()[0] != "mm"
            or qcode != 0 or scode != 1 or sform is None
            or not np.isfinite(sform).all() or abs(float(np.linalg.det(sform[:3, :3]))) < 1e-12
            or not np.array_equal(sform[3], [0, 0, 0, 1])):
        raise ValueError("NATIVE_HEADER_CONVENTION_UNVERIFIED")
    import itertools
    corners = np.array(list(itertools.product(*[(-.5, n-.5) for n in shape])))
    world = nib.affines.apply_affine(sform, corners)
    return {"header_sha256": digest(raw), "decompressed_bytes_returned": 348,
            "shape": list(shape), "spatial_units": "mm", "qform_code": int(qcode),
            "sform_code": int(scode), "selected_affine": sform.tolist(),
            "native_axis_codes": list(nib.aff2axcodes(sform)),
            "full_cell_bounds_ras_mm": {"minimum": world.min(0).tolist(), "maximum": world.max(0).tolist()},
            "image_array_accessed": False, "affine_replaced": False}


def frame_qc(declaration, deadline):
    from real_intake_io import verify_source_file
    records = {}
    for key in ("before", "during"):
        item = declaration["inputs"][key]
        path = safe_path(item["path"])
        verify_source_file(path, item, deadline=deadline)
        with gzip.open(path, "rb") as handle:
            records[key] = header_record(handle.read(348))
        verify_source_file(path, item, deadline=deadline)
    identity = [[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]]
    return {"schema": SCHEMA + "-frame-qc", "status": "coordinate_convention_verified",
            "source_image_sha256": INPUTS["before"]["sha256"],
            "destination_image_sha256": INPUTS["during"]["sha256"],
            "source_world_to_ras_mm": identity, "destination_world_to_ras_mm": identity,
            "convention": declaration["coordinate_convention"], "headers": records,
            "anatomical_alignment_accepted": False, "total_registration_uncertainty_mm": None,
            "physical_clearance_mm": None, "cavity_support": None}


def load_helpers():
    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(ROOT))
    from resectionlab import core, mechanics_landmarks as lm
    from scripts import mechanics_patient_comparison as comparison
    for module, name in ((core, "src/resectionlab/core.py"), (lm, "src/resectionlab/mechanics_landmarks.py"),
                         (comparison, "scripts/mechanics_patient_comparison.py")):
        if Path(module.__file__).resolve() != safe_path(name):
            raise ValueError("HELPER_IMPORT_ORIGIN_CHANGED")
    return lm, comparison


def prepare_landmarks(declaration, release, frame):
    import numpy as np
    lm, comparison = load_helpers()
    identity = np.eye(4).tolist()
    if (frame.get("schema") != SCHEMA + "-frame-qc" or frame.get("status") != "coordinate_convention_verified"
            or frame.get("source_image_sha256") != INPUTS["before"]["sha256"]
            or frame.get("destination_image_sha256") != INPUTS["during"]["sha256"]
            or frame.get("source_world_to_ras_mm") != identity
            or frame.get("destination_world_to_ras_mm") != identity
            or frame.get("convention") != declaration["coordinate_convention"]
            or frame.get("anatomical_alignment_accepted") is not False
            or any(frame.get(key) is not None for key in ("physical_clearance_mm", "cavity_support",
                                                          "total_registration_uncertainty_mm"))):
        raise ValueError("FRAME_QC_CHANGED_OR_PROMOTED")
    qc = lm.LandmarkFrameQC(INPUTS["before"]["sha256"], INPUTS["during"]["sha256"],
                           np.eye(4), np.eye(4), release["dependencies"]["frame_qc"]["sha256"], frame["convention"])
    item = INPUTS["tag"]
    payload = bounded_read(item["path"], item["bytes"])
    if len(payload) != item["bytes"] or digest(payload) != item["sha256"]:
        raise ValueError("ORIGINAL_TAG_CHANGED")
    binding = lm.LandmarkPairBinding("RESECT:Case4", lm.DISPLACEMENT_ROLE, item["sha256"],
                                    INPUTS["before"]["sha256"], INPUTS["during"]["sha256"], qc)
    partition = lm.partition_displacement_sources(lm.parse_tag_sources(payload, binding))
    if (partition.boundary_ids != B_IDS or partition.validation_ids != V_IDS
            or partition.partition_hash.removeprefix("sha256:") != PARTITION_SHA):
        raise ValueError("PROTECTED_PARTITION_CHANGED")
    return lm, comparison, payload, binding, partition


def scientific_stage(phase, declaration, release, dependencies, deadline):
    if phase == "header-qc":
        return {"frame_qc": frame_qc(declaration, deadline)}
    lm, comparison, payload, binding, partition = prepare_landmarks(declaration, release, dependencies["frame_qc"])
    if phase == "fit-freeze":
        def forbidden(*args, **kwargs):
            raise PermissionError("VALIDATION_ACCESS_BEFORE_RELEASE")
        original_reveal = lm.reveal_validation_landmarks
        try:
            lm.reveal_validation_landmarks = forbidden
            forward = lm.build_forward_landmarks(payload, binding, partition)
            field = comparison.fit_baselines(forward, protocol_sha256=release["declaration_sha256"])
            freeze = comparison.freeze_comparison(ROOT, forward=forward, source_binding=binding,
                protocol_binding={"path": MANIFEST, "sha256": release["declaration_sha256"]},
                field=field, output_directory=OUTPUT + "/fit-freeze/comparison", external_field=None)
        finally:
            lm.reveal_validation_landmarks = original_reveal
        return {"freeze": freeze, "B_destinations_accessed": list(B_IDS), "V_destinations_accessed": [],
                "prediction_scope": "complete_evaluable_fields_not_presampled_V_points"}
    report_binding, report = comparison.evaluate_frozen_comparison(ROOT,
        freeze_binding=release["dependencies"]["freeze"], payload=payload,
        source_binding=binding, partition=partition, external_sampler=None)
    if report["validation_landmarks"] != 13:
        raise ValueError("ALL_THIRTEEN_VALIDATION_ROWS_REQUIRED")
    return {"evaluation": report_binding, "V_destinations_accessed": list(V_IDS),
            "physical_clearance_mm": None, "cavity_support": None, "brain_support": None,
            "MRI_transfer_performed": False, "FEM_performed": False,
            "coverage_meaning": "Thirteen source-linked correspondences; mathematical baseline support is not retained-tissue or whole-image coverage",
            "future_untouched_Case4_validation_available": False}


def worker(release_path, phase):
    sys.path.insert(0, str(ROOT / "scripts"))
    from real_intake_io import atomic_preserve, check_deadline, verify_source_file
    start = time.monotonic()
    declaration, release, dependencies, release_sha = authenticate(release_path, phase)
    claim_worker(phase, release_sha)
    before = source_hashes()
    sys.addaudithook(patient_guard(phase))
    resource.setrlimit(resource.RLIMIT_FSIZE, (declaration["maximum_output_file_bytes"],) * 2)
    result = {"schema": SCHEMA + "-stage-result", "phase": phase, "status": "failed",
              "release_sha256": release_sha, "source_sha256": before,
              "runtime": {"python": sys.version, "executable": str(Path(sys.executable).resolve())}}
    try:
        value = scientific_stage(phase, declaration, release, dependencies, start + 30)
        check_deadline(start + 30)
        if source_hashes() != before or digest(bounded_read(release_path)) != release_sha:
            raise ValueError("EXECUTION_BINDING_CHANGED")
        if digest(bounded_read(MANIFEST)) != release["declaration_sha256"]:
            raise ValueError("PROTOCOL_CHANGED")
        for name, binding in release["dependencies"].items():
            checked_binding(binding, DEPENDENCY_PATHS[name])
        for key in (("before", "during") if phase == "header-qc" else ("tag",)):
            item = declaration["inputs"][key]
            verify_source_file(safe_path(item["path"]), item, deadline=start + 30)
        result["original_bytes_unchanged"] = True
        if phase == "header-qc":
            atomic_preserve(safe_path(FRAME_PATH), encode(value.pop("frame_qc")))
        result.update(status="completed", **value)
    except BaseException as error:
        result["error_type"] = type(error).__name__  # Never print patient tokens from exceptions.
        raise
    finally:
        result["elapsed_seconds"] = time.monotonic() - start
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        result["worker_peak_rss_bytes"] = peak if sys.platform == "darwin" else peak * 1024
        result["peak_memory_is_observed_not_a_hard_limit"] = True
        atomic_preserve(safe_path(OUTPUT + "/" + phase + "/result.json"), encode(result))


def launch(release_path, phase):
    sys.path.insert(0, str(ROOT / "scripts"))
    from real_intake_io import atomic_preserve, supervise, termination_cleanup
    declaration, release, _, release_sha = authenticate(release_path, phase)
    attempt = safe_path(OUTPUT + "/" + phase)
    attempt.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    receipt = {"schema": SCHEMA + "-supervision", "phase": phase, "status": "not_started",
               "release_sha256": release_sha, "started_at_utc": datetime.now(timezone.utc).isoformat(),
               "source_sha256": release["source_sha256"], "scientific_result_accepted": False}
    atomic_preserve(attempt / "attempt.json", encode(receipt))
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[key] = "1"
    try:
        with termination_cleanup():
            status, code = supervise([sys.executable, "-B", str(safe_path(SOURCE_NAMES[0])),
                phase, "--release", release_path, "--worker"], attempt / "worker.log",
                deadline=started + declaration["phase_wall_seconds"],
                on_start=lambda pid: atomic_preserve(attempt / "process.json", encode({"pid": pid})))
        receipt.update(status=status, exit_code=code)
        if status == "completed":
            result = json.loads(bounded_read(OUTPUT + "/" + phase + "/result.json"))
            receipt["scientific_result_accepted"] = (result["status"] == "completed"
                and source_hashes() == release["source_sha256"]
                and digest(bounded_read(release_path)) == release_sha
                and time.monotonic() - started <= declaration["phase_wall_seconds"])
    except BaseException as error:
        receipt.update(status="supervision_failed", error_type=type(error).__name__)
        raise
    finally:
        receipt["elapsed_seconds"] = time.monotonic() - started
        atomic_preserve(attempt / "supervision.json", encode(receipt))
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=PHASES)
    parser.add_argument("--release", required=True, help="Exact separately root-authored phase release path")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    try:
        if args.worker:
            worker(args.release, args.phase)
        elif not launch(args.release, args.phase)["scientific_result_accepted"]:
            raise RuntimeError("STAGE_NOT_ACCEPTED")
    except BaseException as error:
        print(json.dumps({"status": "failed", "error_type": type(error).__name__}), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
