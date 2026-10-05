#!/usr/bin/env python3
"""One separately released offline RHUH inspection, with bounded supervision.

The execution manifest is root-owned and hash-bound by the caller. No decoder
is imported here. RSS is sampled, not an instantaneous OS memory ceiling.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import signal
import stat
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts import rhuh_public_transfer as shared

SOURCE = "/RHUH-GBM_nii_v1/RHUH-0001/0/RHUH-0001_0_t1.nii.gz"
TOKEN = "a5c579950d0431d04928e5b59e91ce9d"
LOG_CAP = 65_536
PAYLOAD_CAP = 64 * 1024**2
RSS_CAP = 2 * 1024**3
WALL_SECONDS = 60
SUCCESS = "format_inspected_scientific_use_unreleased"
SOURCE_PATHS = {
    "inspector": "scripts/inspect_rhuh_single_image.py",
    "launcher": "artifacts/rhuh-single-image-inspection-v1/run_once.py",
    "public_transfer": "scripts/rhuh_public_transfer.py",
}
HEX = re.compile(r"[0-9a-f]{64}\Z")
Rejected = shared.Rejected


def need(condition, code):
    if not condition:
        raise Rejected(code)


def local_path(root, text):
    need(isinstance(text, str), "invalid_local_path")
    rel = Path(text)
    need(not rel.is_absolute() and ".." not in rel.parts and str(rel) == text, "invalid_local_path")
    path = root / rel
    need(not any(p.is_symlink() for p in [path, *path.parents]), "local_path_symlink")
    return path


def bound_bytes(root, binding):
    need(isinstance(binding, dict) and set(binding) == {"path", "sha256"}
         and isinstance(binding["sha256"], str) and HEX.fullmatch(binding["sha256"]), "invalid_binding")
    raw = shared.bounded_regular_bytes(local_path(root, binding["path"]), 262_144)
    need(hashlib.sha256(raw).hexdigest() == binding["sha256"], "bound_input_hash_changed")
    return raw


def read_plan(root, manifest_binding, request_sha256, release_sha256):
    plan = json.loads(bound_bytes(root, manifest_binding))
    need(set(plan) == {"schema", "released", "source", "request", "release", "inspector",
                      "launcher", "public_transfer", "output_directory"}
         and plan["schema"] == "resectionlab.rhuh-image-inspection-launch.v1"
         and plan["released"] is True and plan["source"] == SOURCE, "launch_manifest_not_released")
    need(plan["request"]["sha256"] == request_sha256
         and plan["release"]["sha256"] == release_sha256, "explicit_request_release_pins_differ")
    for key, path in SOURCE_PATHS.items():
        need(plan[key]["path"] == path, "unexpected_source_path")
        bound_bytes(root, plan[key])
    request = json.loads(bound_bytes(root, plan["request"]))
    release = json.loads(bound_bytes(root, plan["release"]))
    need(request.get("schema") == "resectionlab.rhuh-image-inspection-request.v1"
         and request.get("source") == SOURCE, "request_source_mismatch")
    need(release == {"schema": "resectionlab.rhuh-image-inspection-release.v1", "released": True,
                    "action": "bounded_header_and_voxel_inspection_once", "request": plan["request"],
                    "inspector_source_sha256": plan["inspector"]["sha256"], "source": SOURCE},
         "inspection_release_mismatch")
    payload = request["payload"]
    path = local_path(root, payload["path"])
    rel = path.relative_to(root)
    need(rel.parts[:3] == ("outputs", "rhuh-single-image-v2", "quarantine")
         and len(rel.parts) == 5 and rel.parts[3].startswith("run-") and rel.name == Path(SOURCE).name
         and type(payload.get("measured_compressed_bytes")) is int
         and 0 < payload["measured_compressed_bytes"] <= PAYLOAD_CAP
         and isinstance(payload.get("sha256"), str) and HEX.fullmatch(payload["sha256"]), "payload_binding_invalid")
    output = local_path(root, plan["output_directory"]).relative_to(root)
    need(output.parts[:2] == ("outputs", "rhuh-single-image-inspection-v1") and len(output.parts) == 3
         and output.name.startswith("run-"), "output_path_invalid")
    return plan, request


def child_limits():
    resource.setrlimit(resource.RLIMIT_FSIZE, (LOG_CAP, LOG_CAP))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def sample_rss(group_id):
    """Combined child-group and launcher RSS; ps reports KiB on this Mac."""
    result = subprocess.run(["ps", "-axo", "pid=,pgid=,rss="], stdout=subprocess.PIPE,
                            stderr=subprocess.DEVNULL, timeout=1, check=True)
    need(len(result.stdout) <= 1_048_576, "process_snapshot_cap")
    total = launcher_rows = 0
    for line in result.stdout.splitlines():
        pid, group, rss = map(int, line.split())
        launcher_rows += pid == os.getpid()
        if group == group_id or pid == os.getpid():
            need(rss >= 0, "invalid_rss_snapshot")
            total += rss * 1024
    need(launcher_rows == 1, "launcher_rss_snapshot_missing_or_duplicate")
    return total


def reconcile(root, plan, request, result):
    need(result.get("schema") == "resectionlab.rhuh-single-image-inspection.v1"
         and result.get("status") == SUCCESS and result.get("binary_structure_valid") is True
         and result.get("scientific_use_released") is False and result.get("clinical_validation") is False,
         "inspection_result_not_accepted")
    need(result.get("bindings") == {"request": plan["request"], "release": plan["release"],
                                    "reconciliation": request["reconciliation"], "source": SOURCE},
         "inspection_result_bindings_differ")
    expected = request["payload"]
    need(result.get("compressed_sha256") == expected["sha256"]
         and type(result.get("compressed_bytes")) is int
         and result["compressed_bytes"] == expected["measured_compressed_bytes"]
         and result.get("candidate_compressed_md5") == TOKEN, "inspection_result_identity_differ")
    path = local_path(root, expected["path"])
    sha, count = hashlib.sha256(), 0
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW), "rb") as stream:
        before = os.fstat(stream.fileno())
        need(stat.S_ISREG(before.st_mode) and before.st_size == expected["measured_compressed_bytes"],
             "original_size_or_type_changed")
        while chunk := stream.read(min(1_048_576, PAYLOAD_CAP - count + 1)):
            count += len(chunk)
            need(count <= PAYLOAD_CAP, "original_byte_cap")
            sha.update(chunk)
        after = os.fstat(stream.fileno())
    identity = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
    need(identity(before) == identity(after) and count == before.st_size
         and sha.hexdigest() == expected["sha256"], "original_identity_changed")
    return {"compressed_bytes": count, "original_sha256": sha.hexdigest(), "parent_decoded_image": False}


def run(root, manifest_binding, request_sha256, release_sha256, started, group):
    output = None
    child = None
    result = {"status": "launch_failed", "accepted": False, "scientific_use_released": False,
              "launcher_exit_zero_required": True}
    peak, samples = 0, 0
    try:
        plan, request = read_plan(root, manifest_binding, request_sha256, release_sha256)
        output = local_path(root, plan["output_directory"])
        output.parent.mkdir(parents=True, exist_ok=True)
        output.mkdir(mode=0o700)  # Existing directory rejects any repeat attempt.
        env = {key: os.environ[key] for key in ("PATH", "LANG", "LC_ALL", "TMPDIR") if key in os.environ}
        env.update({key: "1" for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS")})
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        argv = [sys.executable, str(root / plan["inspector"]["path"]), "--execute",
                "--request", plan["request"]["path"], "--request-sha256", request_sha256,
                "--release", plan["release"]["path"], "--release-sha256", release_sha256]
        with open(output / "stdout.json", "xb") as stdout, open(output / "stderr.txt", "xb") as stderr:
            os.chmod(stdout.name, 0o600)
            os.chmod(stderr.name, 0o600)
            need(time.monotonic() < started + WALL_SECONDS - 3, "deadline_before_child")
            child = subprocess.Popen(argv, cwd=root, env=env, stdin=subprocess.DEVNULL, stdout=stdout,
                                     stderr=stderr, start_new_session=True, preexec_fn=child_limits)
            group.append(child.pid)
            while True:
                need(time.monotonic() < started + WALL_SECONDS - 3, "inspection_timeout")
                rss = sample_rss(child.pid)
                peak, samples = max(peak, rss), samples + 1
                need(rss <= RSS_CAP, "sampled_rss_cap")
                if child.poll() is not None:
                    break
                time.sleep(0.1)
        need(child.returncode == 0, "inspector_exit_nonzero")
        result["status"] = "child_completed_pending_reconciliation"
    except Exception as exc:
        result["status"] = str(exc) if isinstance(exc, Rejected) else "launch_failed_" + type(exc).__name__
    finally:
        if child is not None:
            try:
                os.killpg(child.pid, signal.SIGKILL)  # Also after an exited leader.
            except ProcessLookupError:
                pass
            else:
                result["status"] = "surviving_group_unaccepted"
            try:
                child.wait(timeout=min(1.0, max(0.001, started + WALL_SECONDS - time.monotonic())))
            except subprocess.TimeoutExpired:
                result["status"] = "group_reap_unconfirmed"
    try:
        if result["status"] == "child_completed_pending_reconciliation":
            final_plan, final_request = read_plan(root, manifest_binding, request_sha256, release_sha256)
            need(plan == final_plan and request == final_request, "metadata_changed_after_child")
            raw = shared.bounded_regular_bytes(output / "stdout.json", LOG_CAP)
            shared.bounded_regular_bytes(output / "stderr.txt", LOG_CAP)
            summary = json.loads(raw)
            result["original"] = reconcile(root, plan, request, summary)
            final_plan, final_request = read_plan(root, manifest_binding, request_sha256, release_sha256)
            need(plan == final_plan and request == final_request, "metadata_changed_during_final_read")
            result.update(status=SUCCESS, accepted=True, inspector_exit_code=child.returncode,
                          stdout_sha256=hashlib.sha256(raw).hexdigest(),
                          manifest=manifest_binding, request=plan["request"], release=plan["release"],
                          source_closure_unchanged=True)
        need(time.monotonic() < started + WALL_SECONDS - 1, "deadline_after_reconciliation")
    except Exception as exc:
        result.update(accepted=False, status=str(exc) if isinstance(exc, Rejected) else "reconciliation_failed_" + type(exc).__name__)
    result.update(elapsed_seconds=round(time.monotonic() - started, 6), peak_sampled_combined_rss_bytes=peak,
                  rss_samples=samples, rss_sampling_interval_seconds=0.1, source=SOURCE)
    try:
        if output is not None and output.is_dir() and child is not None:
            raw = (json.dumps(result, sort_keys=True) + "\n").encode()
            need(len(raw) <= LOG_CAP, "launcher_receipt_cap")
            with open(output / "launch.json", "xb") as stream:
                stream.write(raw)
        need(time.monotonic() < started + WALL_SECONDS - 1, "deadline_during_publication")
    except Exception as exc:
        result.update(accepted=False, status=str(exc) if isinstance(exc, Rejected) else "publication_failed_" + type(exc).__name__)
    print(json.dumps(result, sort_keys=True))
    # A provisional receipt alone never proves successful launcher completion.
    return 0 if result["accepted"] and time.monotonic() < started + WALL_SECONDS - 1 else 1


def launch(root, manifest_binding, request_sha256, release_sha256, started=None):
    started = time.monotonic() if started is None else started
    return shared.hard_watchdog(started, lambda began, group: run(
        root, manifest_binding, request_sha256, release_sha256, began, group), wall_seconds=WALL_SECONDS)


def main():
    started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--request-sha256", required=True)
    parser.add_argument("--release-sha256", required=True)
    args = parser.parse_args()
    return launch(ROOT, {"path": args.manifest, "sha256": args.manifest_sha256},
                  args.request_sha256, args.release_sha256, started)


if __name__ == "__main__":
    raise SystemExit(main())
