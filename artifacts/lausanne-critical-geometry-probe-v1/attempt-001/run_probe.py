"""Bounded existing-API probe of one acquired TRAIN annotation; no source edits.

Run from the repository with its venv:
  PYTHONDONTWRITEBYTECODE=1 .venv/bin/python build/lausanne-critical-geometry-probe-v1/run_probe.py
Use --attempt attempt-002 for a separately retained reproduction. Existing attempt
directories are never overwritten. The one scientific worker has 60 s and 1 GiB
RSS supervision. macOS ps sampling is an observed-RSS watchdog, not a promise of
instantaneous kernel enforcement. Worker ru_maxrss records the actual peak.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import shutil
import signal
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
LIMIT_SECONDS = 60.0
LIMIT_RSS_BYTES = 1024 ** 3
COMPONENT = "artifacts/lausanne-sub476-annotation-v1/component.json"
COMPONENT_SHA = "1c2bfeb33fb241b2766e734a762061e9582fa3ec6b90c8e827cb29547638c79e"
RECEIPTS = [COMPONENT,
    "artifacts/lausanne-sub476-annotation-v1/component-verification.json",
    "artifacts/lausanne-deferred-qc-execution-v1/summary.json",
    "artifacts/lausanne-train-acquisition-complete-v1/completion.json",
    "manifests/lausanne-sub476-manual-annotation-v1.json"]


def utc():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def peak_rss():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if platform.system() == "Darwin" else value * 1024)


def worker(attempt):
    started = time.monotonic()
    sys.path.insert(0, str(attempt / "source"))
    result = {"schema": "lausanne-critical-geometry-probe-v1", "status": "running",
              "started_at": utc(), "pid": os.getpid(), "scope": "existing_API_static_geometry_component_only"}

    def phase(name):
        with (attempt / "worker-phases.jsonl").open("a") as stream:
            stream.write(json.dumps({"phase": name, "utc": utc(), "elapsed_seconds": time.monotonic()-started,
                                     "peak_rss_bytes_so_far": peak_rss()}) + "\n")

    def alarm(_signal, _frame):
        raise TimeoutError("Declared 60-second worker deadline reached")

    signal.signal(signal.SIGALRM, alarm)
    signal.setitimer(signal.ITIMER_REAL, LIMIT_SECONDS)
    try:
        phase("importing_frozen_existing_source")
        import numpy as np
        import nibabel as nib
        import scipy
        from resectionlab.core import array_digest, semantic_digest, thaw_json
        from resectionlab.critical_evidence import canonical_hard_exclusion
        from resectionlab.geometry import GEOMETRY_VERSION, GeometryScene, ToolGeometry, ToolPose, check_pose
        from resectionlab.lausanne_sub476_component import DATA, EVIDENCE_ID, build_sub476_component, component_receipt

        assert digest(ROOT / COMPONENT) == COMPONENT_SHA, "Pinned component receipt changed"
        expected = json.loads((ROOT / COMPONENT).read_text())
        manifest = json.loads((ROOT / RECEIPTS[-1]).read_text())
        paths = [ROOT / DATA / manifest[key]["path"] for key in ("original_tof", "file")]
        before = {str(path.relative_to(ROOT)): {"sha256": digest(path), "bytes": path.stat().st_size} for path in paths}
        phase("building_actual_source_bound_case")
        case = build_sub476_component(root=ROOT)
        hard, constraints = canonical_hard_exclusion(case)
        e = case.critical_evidence[EVIDENCE_ID]
        assert component_receipt(case) == expected, "Current actual case differs from committed component receipt"
        assert hard is not None and np.count_nonzero(hard) == 193
        assert np.array_equal(hard, e.mask) and np.array_equal(e.annotation_coverage, hard)
        assert not case.compartments and case.brain_mask is None
        assert case.metadata["role"] == "TRAIN" and case.context is None
        assert case.metadata["scanner_frame_admitted"] is False
        assert case.metadata["spatial_planning_admitted"] is False
        assert not constraints.receipt["objective_structures"]
        before_binding = constraints.fingerprint
        positive_indices = np.argwhere(hard)
        chosen_index = positive_indices[0]
        chosen_mm = chosen_index @ case.affine[:3, :3].T + case.affine[:3, 3]
        axis = case.affine[:3, 0] / np.linalg.norm(case.affine[:3, 0])
        tool = ToolGeometry("source_grid_static_probe_v1", tip_radius_mm=0.25,
            shaft_radius_mm=0.25, working_length_mm=2.0, tip_length_mm=0.5,
            parameter_source="Declared mathematical two-capsule probe: 0.25 mm radii, 2 mm centerline, 0.5 mm distal segment; no measured commercial instrument or surgical claim")
        pose = ToolPose(chosen_mm, axis)
        unknowns = ("positive_support_only_background_unknown", "source_frame_not_scanner_validated",
                    "motor_language_unknown", "static_probe_without_access_or_target",
                    "geometry_only_not_surgical_safety")
        phase("checking_actual_canonical_exclusion")
        scene = GeometryScene(forbidden_mask=hard, affine=case.affine, unknowns=unknowns)
        assert array_digest(scene.forbidden_mask) == array_digest(hard)
        observed = check_pose(tool, pose, scene)
        cells = np.asarray(observed.swept_voxel_indices, dtype=np.int64).reshape(-1, 3)
        actual_contacts = cells[hard[tuple(cells.T)]]
        assert len(actual_contacts) > 0
        assert any(np.array_equal(chosen_index, row) for row in actual_contacts)
        assert not observed.feasible
        assert "FORBIDDEN_COLLISION" in {failure.reason for failure in observed.failures}
        coverage = constraints.route_coverage(cells)
        assert coverage["vessels"]["covered_cells"] == len(actual_contacts)
        assert coverage["motor"]["covered_cells"] is None
        assert coverage["language"]["covered_cells"] is None
        # This is an explicit software ablation of the checker scene, never a
        # replacement CaseData/annotation, negative anatomy or admissible route.
        phase("checking_zero_exclusion_software_control")
        empty = np.zeros_like(hard)
        control_scene = GeometryScene(forbidden_mask=empty, affine=case.affine,
            unknowns=unknowns + ("zero_exclusion_software_control_not_anatomy",))
        control = check_pose(tool, pose, control_scene)
        assert "FORBIDDEN_COLLISION" not in {failure.reason for failure in control.failures}
        assert np.array_equal(control.swept_voxel_indices, cells)
        assert np.array_equal(scene.forbidden_mask, hard)
        hard_after, after_constraints = canonical_hard_exclusion(case)
        assert after_constraints.fingerprint == before_binding
        assert np.array_equal(hard_after, hard)
        after = {str(path.relative_to(ROOT)): {"sha256": digest(path), "bytes": path.stat().st_size} for path in paths}
        assert before == after, "Original source files changed"
        phase("recording_result")
        query = {"selection_rule": "lexicographically first source-positive voxel, np.argwhere(canonical_mask)[0]",
                 "voxel_index": chosen_index.tolist(), "tip_source_reference_mm": pose.tip_mm.tolist(),
                 "axis_rule": "normalized first reference-affine column", "axis_unit": pose.axis_unit.tolist(),
                 "tool": asdict(tool), "geometry_version": GEOMETRY_VERSION,
                 "case_hash": case.semantic_hash, "critical_binding_hash": before_binding,
                 "scope": "retrospective_source_coded_frame_component", "access": None}
        result.update(status="passed", query=query, query_hash=semantic_digest(query),
            evidence=e.to_manifest(), critical_receipt=thaw_json(constraints.receipt),
            actual={"geometry": observed.to_dict(include_voxels=True),
                    "positive_contact_cells": actual_contacts.tolist(), "positive_contact_cell_count": len(actual_contacts),
                    "annotation_domain_coverage": coverage, "mask_hash": array_digest(hard),
                    "coverage_hash": array_digest(e.annotation_coverage)},
            software_control={"scope": "zero_exclusion_software_ablation_only_not_real_safe_tissue",
                "geometry": control.to_dict(include_voxels=True), "same_queried_cells": True,
                "clinical_interpretation": "none; feasible only denotes passing the deliberately empty modeled exclusions",
                "canonical_case_or_annotations_modified": False},
            source_files_before=before, source_files_after=after,
            original_sources_unchanged=True, committed_component_receipt_matched=True,
            component_receipt_sha256=COMPONENT_SHA,
            positive_annotation_cells=193, negative_annotation_cells=0,
            background_unknown_cells=int(hard.size-193),
            scalar_decode_scope="Existing canonical CaseData builder requires full structural-array validation/hash (18800640 float32 samples, 75202560 bytes) and original mask validation; no replacement lightweight case or inferred anatomy used. Streaming file SHA256 uses 1 MiB chunks; no extra scalar decode for control.",
            scanner_frame_admitted=False, spatial_planning_admitted=False, clinical_clearance=False,
            generated_routes=0, generated_targets=0, generated_access_support=0,
            recorded_surgical_transitions=0, simulator_episodes=0, optimizer_updates=0,
            runtime={"python": sys.version, "numpy": np.__version__, "nibabel": nib.__version__, "scipy": scipy.__version__},
            interpretation="Actual source-bound positive aneurysm labels changed the existing geometry checker response at a declared source-positive static probe. This is shared-kernel component consumption, not full planner/desktop acceptance, complete vessel coverage, anatomical validation or safety evidence.",
            remaining_gap="A bound planning/desktop probe entry is absent. generateRoutes still correctly requires radiological target and access evidence; no such evidence is created or admitted here.")
    except BaseException as error:
        result.update(status="failed", error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        result.update(completed_at=utc(), elapsed_seconds=time.monotonic()-started,
                      worker_peak_rss_bytes=peak_rss(), worker_peak_rss_source="getrusage(RUSAGE_SELF).ru_maxrss; bytes on Darwin")
        if result["worker_peak_rss_bytes"] > LIMIT_RSS_BYTES:
            result.update(status="failed", resource_failure="observed_worker_peak_rss_exceeded_1GiB")
        write_json(attempt / "worker-result.json", result)
    return 0 if result["status"] == "passed" else 1


def supervise(attempt):
    attempt.mkdir()  # immutable attempt; refuse a repeat overwrite
    snapshot = attempt / "source" / "resectionlab"
    source_hashes = {}
    for source in sorted((ROOT / "src/resectionlab").rglob("*.py")):
        rel = source.relative_to(ROOT / "src/resectionlab")
        dest = snapshot / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, dest)
        source_hashes[str(rel)] = digest(dest)
    assert all(digest(ROOT/"src/resectionlab"/rel) == sha for rel, sha in source_hashes.items()), "Source changed while snapshotting; retained attempt requires review"
    shutil.copyfile(__file__, attempt / "run_probe.py")
    receipt_hashes = {path: digest(ROOT / path) for path in RECEIPTS}
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", OPENBLAS_NUM_THREADS="1",
               OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
    intent = {"schema": "bounded-component-probe-intent-v1", "created_at": utc(),
        "source_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "script_sha256": digest(__file__), "source_snapshot_sha256": source_hashes,
        "receipt_sha256": receipt_hashes, "scientific_workers": 1,
        "worker_seconds_limit": LIMIT_SECONDS, "worker_rss_limit_bytes": LIMIT_RSS_BYTES,
        "rss_watchdog": "ps RSS KiB sampled approximately every 0.1s; kill process group on observed crossing; worker self-peak checked after completion",
        "platform": platform.platform(), "python_executable": sys.executable,
        "source_decoding_authorized": "Existing acquired sub-476/ses-20140519 TRAIN image and manual mask only",
        "zero_exclusion_control": "Explicit software control, not real anatomy or safe tissue",
        "no_source_writes": True, "native_thread_limits": {k: env[k] for k in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"]}}
    write_json(attempt / "intent.json", intent)
    began = time.monotonic()
    termination = None
    samples = []
    with (attempt / "worker.log").open("w") as log:
        process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--worker", "--attempt", attempt.name],
            cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            while process.poll() is None:
                elapsed = time.monotonic()-began
                sample = subprocess.run(["ps", "-o", "rss=", "-p", str(process.pid)], text=True,
                    capture_output=True, timeout=1)
                rss = int(sample.stdout.strip()) * 1024 if sample.returncode == 0 and sample.stdout.strip() else None
                samples.append({"elapsed_seconds": elapsed, "rss_bytes": rss})
                if elapsed >= LIMIT_SECONDS or (rss is not None and rss > LIMIT_RSS_BYTES):
                    termination = "wall_time_limit" if elapsed >= LIMIT_SECONDS else "rss_limit"
                    os.killpg(process.pid, signal.SIGKILL)
                    break
                time.sleep(0.1)
        except BaseException as error:
            termination = "supervisor_error: " + repr(error)
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=5)
    worker_file = attempt / "worker-result.json"
    wr = json.loads(worker_file.read_text()) if worker_file.exists() else None
    passed = process.returncode == 0 and termination is None and wr is not None and wr["status"] == "passed"
    outcome = {"schema": "bounded-component-probe-outcome-v1", "status": "passed" if passed else "failed",
        "completed_at": utc(), "elapsed_seconds": time.monotonic()-began, "worker_pid": process.pid,
        "worker_returncode": process.returncode, "termination": termination,
        "rss_samples": samples, "maximum_sampled_rss_bytes": max((s["rss_bytes"] or 0 for s in samples), default=0),
        "worker_peak_rss_bytes": None if wr is None else wr["worker_peak_rss_bytes"],
        "worker_result_sha256": digest(worker_file) if worker_file.exists() else None,
        "intent_sha256": digest(attempt/"intent.json"), "worker_log_sha256": digest(attempt/"worker.log"),
        "worker_reaped": True, "scientific_workers": 1}
    write_json(attempt / "outcome.json", outcome)
    print(json.dumps({k: v for k, v in outcome.items() if k != "rss_samples"}, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--attempt", default="attempt-001")
    args = parser.parse_args()
    if not args.attempt.startswith("attempt-") or not args.attempt.replace("-", "").isalnum():
        parser.error("attempt must be a simple attempt-name within this output directory")
    target = OUT / args.attempt
    raise SystemExit(worker(target) if args.worker else supervise(target))
