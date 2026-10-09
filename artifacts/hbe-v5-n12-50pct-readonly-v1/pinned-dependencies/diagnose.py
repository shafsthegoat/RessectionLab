"""Ignored read-only HBE preflight memory diagnostic; requires root authorization.

This script never writes a release, reserves an HBE output, runs FEBio, or reads
measured/patient data. Its supervisor executes one bounded validation worker.
"""
from __future__ import annotations

import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "build/hbe-v5-n12-preflight-memory-diagnostic-v1/attempt-01"
TEMPLATE = ROOT / "build/hbe-v5-tension-n12-s60-release-prep/RELEASE_TEMPLATE.json"
TEMPLATE_SHA = "32d7566d10d1a671fe1d7fcd2cb3bfc73dc70c3632d286480fd860e2e15d2feb"
HOST_SAMPLER = ROOT / "build/limited-input-guard-design/gliomoda-tile1-parity-64-v2/darwin_fast_sampler.py"
HOST_SAMPLER_SHA = "2ff8f3e4d30ed0a8976fb01da99403ff5b2b2ffd6e750791fe7a954946d0a00f"
WALL_CAP = 150
WORKER_WALL_CAP = 147
RSS_CAP = 3 * 1024**3
OUTPUT_CAP = 4 * 1024**2
THREAD_ENV = {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
              "MKL_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1"}


def command(argv: list[str], timeout: float = 3.0) -> dict:
    try:
        result = subprocess.run(argv, capture_output=True, text=True,
                                timeout=timeout, check=False)
        return {"exit_code": result.returncode,
                "stdout": result.stdout[:4096], "stderr": result.stderr[:1024]}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"error": type(exc).__name__, "detail": str(exc)[:300]}


def save_bounded(name: str, value: object) -> None:
    raw = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    if len(raw) > 1024**2:
        raise RuntimeError("One MiB worker JSON cap exceeded")
    destination = OUT / name
    temp = destination.with_suffix(destination.suffix + ".tmp")
    with temp.open("wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, destination)


def load_host_sampler():
    raw = HOST_SAMPLER.read_bytes()
    if hashlib.sha256(raw).hexdigest() != HOST_SAMPLER_SHA:
        raise RuntimeError("Reviewed direct macOS host sampler changed")
    spec = importlib.util.spec_from_file_location("hbe_preflight_direct_host_sampler", HOST_SAMPLER)
    if spec is None or spec.loader is None:
        raise RuntimeError("Reviewed direct host sampler cannot load")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.FastDarwinSampler()


def snapshot(label: str, samples: list[dict], *, started: float, host_sampler) -> None:
    direct_host = host_sampler.host()
    pressure = command(["/usr/bin/memory_pressure", "-Q"])
    vmstat = command(["/usr/bin/vm_stat"])
    ps = command(["/bin/ps", "-o", "rss=", "-p", str(os.getpid())])
    match = re.search(r"free percentage:\s*(\d+)%", pressure.get("stdout", ""), re.I)
    item = {
        "label": label, "elapsed_seconds": time.monotonic() - started,
        "direct_kernel_host": direct_host,
        "memory_pressure_q_free_percent": int(match.group(1)) if match else None,
        "process_rss_bytes": int(ps["stdout"].strip()) * 1024
            if ps.get("exit_code") == 0 and ps.get("stdout", "").strip().isdigit() else None,
        "memory_pressure": pressure, "vm_stat": vmstat,
    }
    samples.append(item)
    save_bounded("worker-samples.json", samples)


def require_selected_head(source_commit: str) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", source_commit):
        raise RuntimeError("Root must select one full current source commit")
    head = subprocess.run(["/usr/bin/git", "rev-parse", "HEAD"], cwd=ROOT,
                          capture_output=True, check=True, text=True,
                          timeout=5).stdout.strip()
    if head != source_commit:
        raise RuntimeError("HEAD changed from root-selected source commit")
    return head


def worker(source_commit: str) -> int:
    started = time.monotonic()
    samples: list[dict] = []
    host_sampler = load_host_sampler()
    snapshot("stdlib_host_before_hbe_import", samples, started=started,
             host_sampler=host_sampler)
    direct = samples[-1]["direct_kernel_host"]
    if direct["kernel_pressure_mask"] != 1 or direct["available_percent"] < 45:
        save_bounded("worker-result.json", {
            "status": "hold_before_validation_host_guard",
            "direct_kernel_host": direct,
            "native_calls": 0, "release_written": False, "target_reserved": False,
        })
        return 2
    from scripts import mechanics_hbe_backend as backend
    from scripts import mechanics_hbe_branch_calibration_v5 as v5
    from scripts import mechanics_hbe_v5_remaining_one_shot as remaining
    from scripts import mechanics_hbe_v5_source_bindings as sources
    snapshot("after_hbe_import", samples, started=started, host_sampler=host_sampler)

    head = require_selected_head(source_commit)
    if TEMPLATE.is_symlink() or TEMPLATE.stat().st_size > 1024**2:
        raise RuntimeError("Template absent, linked, or oversized")
    template_raw = TEMPLATE.read_bytes()
    if hashlib.sha256(template_raw).hexdigest() != TEMPLATE_SHA:
        raise RuntimeError("Exact reviewed template bytes changed")
    release = json.loads(template_raw)
    if (release["status"] != "TEMPLATE_NOT_RELEASED" or release["ordinal"] != 8
            or release["run_id"] != "tension:N12:S60:reference"):
        raise RuntimeError("Expected non-executable ordinal-8 template absent")
    release["status"] = "root_released_one_native_call"  # in memory only
    release["source_commit"] = source_commit
    for name, module in (
            ("source_manifest_topology", sources),
            ("adapter", v5),
            ("predecessor_chain_hashing", remaining),
            ("committed_source_hashing", remaining),
            ("runtime_profile_controls", backend)):
        attr = {"source_manifest_topology": "validate_binding_manifest",
                "adapter": "adapt_deck",
                "predecessor_chain_hashing": "validate_prior_chain",
                "committed_source_hashing": "source_hashes",
                "runtime_profile_controls": "verify_profile"}[name]
        original = getattr(module, attr)

        def traced(*args, _name=name, _original=original, **kwargs):
            snapshot("before_" + _name, samples, started=started,
                     host_sampler=host_sampler)
            try:
                return _original(*args, **kwargs)
            finally:
                snapshot("after_" + _name, samples, started=started,
                         host_sampler=host_sampler)

        setattr(module, attr, traced)
    snapshot("before_validate_release", samples, started=started,
             host_sampler=host_sampler)
    result = remaining.validate_release(release, root=ROOT)
    if result["index"] != 8 or result["run_id"] != release["run_id"]:
        raise RuntimeError("Validated row differs")
    require_selected_head(source_commit)
    snapshot("after_validate_release_result_retained", samples, started=started,
             host_sampler=host_sampler)
    collected_with_result = gc.collect()
    snapshot("after_gc_result_retained", samples, started=started,
             host_sampler=host_sampler)
    result_deck_sha = hashlib.sha256(result["deck"]).hexdigest()
    del result
    collected_after_drop = gc.collect()
    snapshot("after_gc_result_dropped", samples, started=started,
             host_sampler=host_sampler)
    require_selected_head(source_commit)
    save_bounded("worker-result.json", {
        "status": "read_only_validation_memory_diagnostic_complete",
        "source_commit": head, "adapted_deck_sha256": result_deck_sha,
        "expected_adapted_deck_sha256": release["adapted_deck_sha256"],
        "samples": len(samples), "elapsed_seconds": time.monotonic() - started,
        "gc_collected_with_result": collected_with_result,
        "gc_collected_after_drop": collected_after_drop,
        "native_calls": 0, "release_written": False, "target_reserved": False,
        "postvalidation_below_45_percent":
            samples[-3]["direct_kernel_host"]["available_percent"] < 45,
    })
    return 0


class OwnedWorker:
    """Retain exact child handle so failed group signaling cannot skip reaping."""

    def __init__(self):
        self.process = None

    def __call__(self, *args, **kwargs):
        if self.process is not None:
            raise RuntimeError("Second diagnostic worker forbidden")
        self.process = subprocess.Popen(*args, **kwargs)
        return self.process

    def cleanup(self, observer, deadline: float) -> dict:
        record = {"contained": self.process is None,
                  "direct_child_reaped": self.process is None,
                  "remaining_members": [], "errors": [], "fallback_used": False}
        if self.process is None:
            return record
        process = self.process

        def remaining(maximum: float) -> float:
            value = min(maximum, deadline - time.monotonic())
            if value <= 0:
                raise RuntimeError("Diagnostic cleanup deadline expired")
            return value

        members = []
        try:
            _, members = observer(process.pid, timeout_seconds=remaining(.5))
        except Exception as error:
            record["errors"].append("initial_observer:" + type(error).__name__)
        if process.poll() is None or members or record["errors"]:
            record["fallback_used"] = True
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            except Exception as error:
                record["errors"].append("killpg:" + type(error).__name__)
            try:
                if process.poll() is None:
                    process.kill()
            except Exception as error:
                record["errors"].append("direct_kill:" + type(error).__name__)
            # Do not signal cached member PIDs: they could have been reused.
            # The final group observation records any unresolved descendants.
        try:
            record["exit_code"] = process.wait(timeout=remaining(2.))
            record["direct_child_reaped"] = True
        except Exception as error:
            record["errors"].append("reap:" + type(error).__name__)
        try:
            _, record["remaining_members"] = observer(process.pid,
                                                       timeout_seconds=remaining(.5))
            record["contained"] = (record["direct_child_reaped"]
                                   and not record["remaining_members"])
        except Exception as error:
            record["errors"].append("final_observer:" + type(error).__name__)
        return record


def supervisor(source_commit: str) -> int:
    from scripts import febio_runtime
    from scripts import mechanics_hbe_v5_remaining_one_shot as remaining
    require_selected_head(source_commit)
    if OUT.exists():
        raise RuntimeError("One-shot diagnostic output already exists")
    OUT.mkdir(parents=True, exist_ok=False)
    environment = os.environ.copy()
    environment.update(THREAD_ENV)
    environment["PYTHONPATH"] = str(ROOT)
    receipt = {"schema": "hbe-v5-n12-read-only-memory-diagnostic-v1",
               "status": "running", "native_calls": 0, "hbe_readout_calls": 0,
               "release_written": False, "hbe_output_reserved": False,
               "head_required": source_commit, "template_sha256": TEMPLATE_SHA,
               "caps": {"outer_wall_seconds": WALL_CAP,
                        "worker_wall_seconds": WORKER_WALL_CAP,
                        "sampled_process_group_rss_bytes": RSS_CAP,
                        "aggregate_output_bytes": OUTPUT_CAP,
                        "numerical_threads": 1, "attempts": 1},
               "supervisor_stage_label": "readout_api_only_not_hbe_readout"}
    owner = OwnedWorker()
    started = time.monotonic()
    result = None
    supervision_error = None
    try:
        result = remaining.supervise_stage(
            "readout", [sys.executable, "-B", str(Path(__file__).resolve()),
                        "worker", source_commit],
            OUT, receipt, cwd=ROOT, environment=environment,
            wall_cap=WORKER_WALL_CAP, rss_cap=RSS_CAP, output_cap=OUTPUT_CAP,
            rss_observer=febio_runtime.process_group_rss, popen=owner)
    except BaseException as error:
        supervision_error = {"type": type(error).__name__, "message": str(error)[:500]}
    finally:
        try:
            receipt["owned_worker_cleanup"] = owner.cleanup(
                febio_runtime.process_group_rss, started + WALL_CAP)
        except BaseException as error:
            receipt["owned_worker_cleanup"] = {
                "contained": False, "direct_child_reaped": False,
                "errors": ["cleanup_exception:" + type(error).__name__],
                "remaining_members": [], "fallback_used": True}
        receipt["outer_elapsed_seconds"] = time.monotonic() - started
        receipt["status"] = "post_supervision_pending"
        remaining.io.durable_json(OUT / "receipt.json", receipt)
    worker_result_path = OUT / "worker-result.json"
    try:
        worker_result = (json.loads(worker_result_path.read_text())
                         if worker_result_path.is_file() and not worker_result_path.is_symlink()
                         and worker_result_path.stat().st_size <= 1024**2 else None)
    except (OSError, UnicodeError, json.JSONDecodeError):
        worker_result = None
    cleanup = receipt["owned_worker_cleanup"]
    good = (supervision_error is None and result is not None
            and result["status"] == "completed_within_caps"
            and cleanup["contained"] and cleanup["direct_child_reaped"]
            and not cleanup["fallback_used"] and not cleanup["errors"]
            and worker_result is not None
            and worker_result.get("status") == "read_only_validation_memory_diagnostic_complete"
            and worker_result.get("native_calls") == 0
            and worker_result.get("release_written") is False
            and worker_result.get("target_reserved") is False
            and receipt["outer_elapsed_seconds"] < WALL_CAP)
    receipt["status"] = ("completed_read_only_diagnostic" if good
                         else "failed_or_incomplete")
    if supervision_error is not None:
        receipt["supervision_error"] = supervision_error
    receipt["worker_result_status"] = (worker_result or {}).get("status")
    remaining.io.durable_json(OUT / "receipt.json", receipt)
    try:
        retained = remaining.active_bytes(OUT, OUTPUT_CAP)
        receipt["final_retained_output_bytes"] = retained
        if retained > OUTPUT_CAP - 64 * 1024:
            receipt["status"] = "failed_final_output_cap"
    except Exception as error:
        receipt["status"] = "failed_final_output_scan"
        receipt["final_scan_error_type"] = type(error).__name__
    receipt["outer_elapsed_seconds"] = time.monotonic() - started
    if receipt["outer_elapsed_seconds"] >= WALL_CAP:
        receipt["status"] = "failed_outer_wall_cap"
    remaining.io.durable_json(OUT / "receipt.json", receipt)
    try:
        if remaining.active_bytes(OUT, OUTPUT_CAP) > OUTPUT_CAP:
            receipt["status"] = "failed_final_output_cap_after_receipt"
            remaining.io.durable_json(OUT / "receipt.json", receipt)
    except Exception as error:
        receipt["status"] = "failed_final_output_scan_after_receipt"
        receipt["final_scan_error_type"] = type(error).__name__
        remaining.io.durable_json(OUT / "receipt.json", receipt)
    print(json.dumps({"status": receipt["status"],
                      "receipt": str(OUT / "receipt.json")}, sort_keys=True))
    return 0 if receipt["status"] == "completed_read_only_diagnostic" else 1


def worker_terminal(source_commit: str) -> int:
    try:
        return worker(source_commit)
    except Exception as error:
        if OUT.is_dir():
            save_bounded("worker-result.json", {
                "status": "failed_or_incomplete",
                "error_type": type(error).__name__,
                "error_message": str(error)[:500],
                "native_calls": 0, "release_written": False,
                "target_reserved": False,
            })
        raise


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "worker":
        raise SystemExit(worker_terminal(sys.argv[2]))
    if len(sys.argv) == 3 and sys.argv[1] == "supervisor":
        raise SystemExit(supervisor(sys.argv[2]))
    raise SystemExit("Expected worker or supervisor plus one full source commit")
