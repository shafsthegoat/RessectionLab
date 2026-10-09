"""Unreleased one-use read-only hinted HBE N12 host-feasibility probe.

This source does not execute unless root separately authorizes one exact HEAD.
It never reserves an HBE attempt, writes a release, calls FEBio, or reads
measured force/patient data. It reuses the pinned original validator and hint.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from types import ModuleType

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
TARGET = HERE / "attempt-01"
COMPARE = ROOT / "build/hbe-v5-n12-nocache-comparison-v1/compare.py"
COMPARE_SHA = "2113791dc16d7191b7c9c24f4148fb820450ecfff4c2d2f5c0af8ad8488d0b45"
DIAGNOSTIC = ROOT / "build/hbe-v5-n12-preflight-memory-diagnostic-v1/diagnose.py"
DIAGNOSTIC_SHA = "8684aa673e8e0189bd6c5fdfab144e7ff6d8a33af5e16047caa5a88b196dce16"
HINT = ROOT / "build/hbe-v5-n12-nocache-comparison-v1/hash_hint.py"
HINT_SHA = "96c71d0d9b8d896b713aef7305e3e86d5b12b9d2dca5fcc55515f6b2eb9cf3fc"
TEMPLATE = ROOT / "build/hbe-v5-tension-n12-s60-release-prep/RELEASE_TEMPLATE.json"
TEMPLATE_SHA = "32d7566d10d1a671fe1d7fcd2cb3bfc73dc70c3632d286480fd860e2e15d2feb"
INITIAL_FLOOR = 50
UNCHANGED_NATIVE_FLOOR = 45
NORMAL_MASK = 1
SERIES_LIMIT = 1000


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def self_sha() -> str:
    path = Path(__file__).absolute()
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 128 * 1024:
        raise RuntimeError("Probe source absent, linked or oversized")
    return sha(path.read_bytes())


def load_dependencies():
    def exact(path: Path, expected: str, name: str):
        before = path.lstat()
        if path.is_symlink() or not path.is_file() or before.st_size > 128 * 1024:
            raise RuntimeError("Pinned read-only source absent, linked or oversized")
        raw = path.read_bytes()
        after = path.lstat()
        identity = lambda item: (item.st_dev, item.st_ino, item.st_size, item.st_mtime_ns)
        if identity(before) != identity(after) or sha(raw) != expected:
            raise RuntimeError("Pinned read-only source changed: " + str(path))
        module = ModuleType(name)
        module.__file__ = str(path)
        exec(compile(raw, str(path), "exec"), module.__dict__)
        return module

    compare = exact(COMPARE, COMPARE_SHA, "hbe_probe50_reviewed_compare")
    diagnostic = exact(DIAGNOSTIC, DIAGNOSTIC_SHA, "hbe_probe50_reviewed_diagnostic")
    hint = exact(HINT, HINT_SHA, "hbe_probe50_reviewed_hint")
    return compare, diagnostic, hint


def selected_sources(source_commit: str) -> dict:
    """Cheap exact 20-source closure before the full read-only validation."""
    if not re.fullmatch(r"[0-9a-f]{40}", source_commit):
        raise RuntimeError("Root-selected full source commit required")
    head = subprocess.run(["/usr/bin/git", "rev-parse", "HEAD"], cwd=ROOT,
                          capture_output=True, check=True, text=True,
                          timeout=5).stdout.strip()
    if head != source_commit:
        raise RuntimeError("Checkout changed from selected probe commit")
    if TEMPLATE.is_symlink() or not TEMPLATE.is_file() or TEMPLATE.stat().st_size > 1024**2:
        raise RuntimeError("Original audited template absent, linked or oversized")
    raw = TEMPLATE.read_bytes()
    if sha(raw) != TEMPLATE_SHA:
        raise RuntimeError("Original audited template changed")
    template = json.loads(raw)
    bindings = template["source_bindings"]
    if (template["status"] != "TEMPLATE_NOT_RELEASED" or template["ordinal"] != 8
            or template["run_id"] != "tension:N12:S60:reference"
            or len(bindings) != 20):
        raise RuntimeError("Expected frozen ordinal-8 source closure absent")
    for relative, binding in bindings.items():
        path = ROOT / relative
        if (binding["path"] != relative or path.is_symlink()
                or not path.is_file() or path.stat().st_size > 1024**2):
            raise RuntimeError("Frozen source path changed: " + relative)
        current = path.read_bytes()
        committed = subprocess.run(
            ["/usr/bin/git", "show", source_commit + ":" + relative],
            cwd=ROOT, capture_output=True, check=True, timeout=5).stdout
        if current != committed or sha(current) != binding["sha256"]:
            raise RuntimeError("Frozen source differs from template/commit: " + relative)
    return {"source_commit": source_commit, "original_template_sha256": TEMPLATE_SHA,
            "source_count": len(bindings),
            "run_id": template["run_id"],
            "expected_adapted_deck_sha256": template["adapted_deck_sha256"],
            "adapter_receipt": template["adapter_receipt"],
            "old_source_hashes": {path: item["sha256"] for path, item in bindings.items()}}


class HostSeries:
    """Sample direct host at each old RSS tick and abort on native-floor breach."""

    def __init__(self, host_sampler, output: Path, started: float):
        self.host_sampler = host_sampler
        self.output = output
        self.started = started
        self.samples: list[dict] = []
        self.breach: dict | None = None

    def flush(self) -> None:
        from scripts import mechanics_hbe_v5_n8_one_shot as io
        raw = (json.dumps(self.samples, sort_keys=True, allow_nan=False) + "\n").encode()
        if len(raw) >= 1024**2:
            raise RuntimeError("Host series exceeded one MiB output cap")
        io.durable_json(self.output / "host-time-series.json", {
            "schema": "hbe-v5-n12-host-series-v1", "samples": self.samples,
            "breach": self.breach})

    def sample(self, label: str, *, minimum: int = UNCHANGED_NATIVE_FLOOR) -> dict:
        if len(self.samples) >= SERIES_LIMIT:
            raise RuntimeError("Bounded host series sample count exhausted")
        host = self.host_sampler.host()
        item = {"label": label, "elapsed_seconds": time.monotonic() - self.started,
                "available_percent": host["available_percent"],
                "kernel_pressure_mask": host["kernel_pressure_mask"],
                "swap_used_bytes": host["swap_used_bytes"]}
        self.samples.append(item)
        if (host["kernel_pressure_mask"] != NORMAL_MASK
                or host["available_percent"] < minimum):
            self.breach = item
            self.flush()
            raise RuntimeError("Read-only probe aborted below required "
                               + str(minimum) + "%/normal host floor")
        if label == "supervised_rss_tick" and len(self.samples) % 10 == 0:
            self.flush()
        return item

    def observer(self, original_observer, pgid: int, *, timeout_seconds: float):
        rss, members = original_observer(pgid, timeout_seconds=timeout_seconds)
        self.sample("supervised_rss_tick")
        return rss, members


def worker(source_commit: str, expected_self_sha: str) -> int:
    if self_sha() != expected_self_sha:
        raise RuntimeError("Probe source changed before worker validation")
    compare, diagnostic, hint = load_dependencies()
    compare.selected_head(source_commit)
    diagnostic.OUT = TARGET
    compare.ARMS = {"hinted": TARGET}
    original_snapshot = diagnostic.snapshot

    def guarded_snapshot(label, samples, *, started, host_sampler):
        original_snapshot(label, samples, started=started, host_sampler=host_sampler)
        direct = samples[-1]["direct_kernel_host"]
        floor = (INITIAL_FLOOR if label == "stdlib_host_before_hbe_import"
                 else UNCHANGED_NATIVE_FLOOR)
        if (direct["kernel_pressure_mask"] != NORMAL_MASK
                or direct["available_percent"] < floor):
            raise RuntimeError("Worker read-only validation crossed "
                               + str(floor) + "%/normal host floor")

    diagnostic.snapshot = guarded_snapshot
    try:
        status = compare.run_worker_arm("hinted", source_commit, diagnostic, hint)
        if self_sha() != expected_self_sha:
            raise RuntimeError("Probe source changed after worker validation")
        return status
    finally:
        diagnostic.snapshot = original_snapshot


def supervisor(source_commit: str) -> int:
    from scripts import febio_runtime
    from scripts import mechanics_hbe_v5_remaining_one_shot as remaining
    compare, diagnostic, _ = load_dependencies()
    compare.selected_head(source_commit)
    closure = selected_sources(source_commit)
    wrapper_sha = self_sha()
    if TARGET.exists() or TARGET.is_symlink():
        raise RuntimeError("Read-only 50%-start probe is one-use")
    host_sampler = diagnostic.load_host_sampler()
    before = host_sampler.host()
    if (before["kernel_pressure_mask"] != NORMAL_MASK
            or before["available_percent"] < INITIAL_FLOOR):
        raise RuntimeError("Read-only probe initial host below 50%/normal")
    TARGET.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    series = HostSeries(host_sampler, TARGET, started)
    environment = os.environ.copy()
    environment.update(diagnostic.THREAD_ENV)
    environment["PYTHONPATH"] = str(ROOT)
    receipt = {"schema": "hbe-v5-n12-read-only-50pct-hinted-probe-v1",
               "status": "running", "source_commit": source_commit,
               "probe_source_sha256": wrapper_sha,
               "compare_source_sha256": COMPARE_SHA,
               "diagnostic_source_sha256": DIAGNOSTIC_SHA,
               "hint_source_sha256": HINT_SHA,
               "original_source_closure": closure,
               "host_before": before,
               "host_policy": {"initial_available_percent_floor": INITIAL_FLOOR,
                               "during_validation_available_percent_floor": UNCHANGED_NATIVE_FLOOR,
                               "normal_kernel_pressure_mask": NORMAL_MASK},
               "native_calls": 0, "hbe_readout_calls": 0,
               "release_written": False, "hbe_output_reserved": False,
               "caps": {"outer_wall_seconds": diagnostic.WALL_CAP,
                        "worker_wall_seconds": diagnostic.WORKER_WALL_CAP,
                        "sampled_process_group_rss_bytes": diagnostic.RSS_CAP,
                        "aggregate_output_bytes": diagnostic.OUTPUT_CAP,
                        "numerical_threads": 1, "attempts": 1},
               "supervisor_stage_label": "readout_api_only_not_hbe_readout"}
    try:
        series.sample("before_worker_supervision", minimum=INITIAL_FLOOR)
    except BaseException as failure:
        receipt["status"] = "failed_or_incomplete"
        receipt["error"] = {"type": type(failure).__name__,
                            "message": str(failure)[:500]}
        receipt["host_breach"] = series.breach
        receipt["host_sample_count"] = len(series.samples)
        remaining.io.durable_json(TARGET / "receipt.json", receipt)
        return 1
    owner = diagnostic.OwnedWorker()
    result = None
    error = None
    try:
        result = remaining.supervise_stage(
            "readout", [sys.executable, "-B", str(Path(__file__).resolve()),
                        "worker", source_commit, wrapper_sha],
            TARGET, receipt, cwd=ROOT, environment=environment,
            wall_cap=diagnostic.WORKER_WALL_CAP, rss_cap=diagnostic.RSS_CAP,
            output_cap=diagnostic.OUTPUT_CAP,
            rss_observer=lambda pgid, *, timeout_seconds: series.observer(
                febio_runtime.process_group_rss, pgid, timeout_seconds=timeout_seconds),
            popen=owner)
    except BaseException as failure:
        error = {"type": type(failure).__name__, "message": str(failure)[:500]}
    finally:
        try:
            receipt["owned_worker_cleanup"] = owner.cleanup(
                febio_runtime.process_group_rss, started + diagnostic.WALL_CAP)
        except BaseException as failure:
            receipt["owned_worker_cleanup"] = {
                "contained": False, "direct_child_reaped": False,
                "errors": ["cleanup_exception:" + type(failure).__name__],
                "remaining_members": [], "fallback_used": True}
        try:
            series.sample("after_owned_cleanup")
        except BaseException as failure:
            error = error or {"type": type(failure).__name__,
                              "message": str(failure)[:500]}
        try:
            series.flush()
        except BaseException as failure:
            error = error or {"type": type(failure).__name__,
                              "message": "Host series persistence failed: " + str(failure)[:400]}
        receipt["outer_elapsed_seconds"] = time.monotonic() - started
        receipt["status"] = "post_supervision_pending"
        remaining.io.durable_json(TARGET / "receipt.json", receipt)
    def bounded_json(name: str):
        path = TARGET / name
        try:
            if path.is_symlink() or not path.is_file() or path.stat().st_size > 1024**2:
                return None
            value = json.loads(path.read_text())
            return value if isinstance(value, dict) else None
        except (OSError, UnicodeError, json.JSONDecodeError):
            return None

    worker_result = bounded_json("worker-result.json")
    identity = bounded_json("validation-identity.json")
    hints = bounded_json("hint-audit.json")
    cleanup = receipt["owned_worker_cleanup"]
    try:
        after_wrapper_sha = self_sha()
        after_head = compare.selected_head(source_commit)
        after_closure = selected_sources(source_commit)
    except BaseException as failure:
        after_wrapper_sha = None
        after_head = None
        after_closure = None
        error = error or {"type": type(failure).__name__, "message": str(failure)[:500]}
    receipt["probe_source_sha256_postguard"] = after_wrapper_sha
    receipt["observed_head_postguard"] = after_head
    receipt["original_source_closure_postguard"] = after_closure
    good = (error is None and result is not None
            and result["status"] == "completed_within_caps"
            and receipt.get("readout_calls_attempted") == 1
            and receipt.get("readout_stage", {}).get("status") == "completed_within_caps"
            and cleanup["contained"] and cleanup["direct_child_reaped"]
            and not cleanup["fallback_used"] and not cleanup["errors"]
            and series.breach is None and len(series.samples) >= 2
            and any(x["label"] == "supervised_rss_tick" for x in series.samples)
            and worker_result is not None
            and worker_result.get("status") == "read_only_validation_memory_diagnostic_complete"
            and worker_result.get("source_commit") == source_commit
            and worker_result.get("native_calls") == 0
            and worker_result.get("release_written") is False
            and worker_result.get("target_reserved") is False
            and worker_result.get("adapted_deck_sha256") ==
                worker_result.get("expected_adapted_deck_sha256")
            and worker_result.get("adapted_deck_sha256") ==
                closure["expected_adapted_deck_sha256"]
            and identity is not None and identity.get("index") == 8
            and identity.get("run_id") == closure["run_id"]
            and identity.get("deck_sha256") == closure["expected_adapted_deck_sha256"]
            and identity.get("source_hashes") == closure["old_source_hashes"]
            and identity.get("adapter_receipt") == closure["adapter_receipt"]
            and identity.get("observed_head_preflight") == source_commit
            and hints is not None and hints.get("eligible_files") == 14
            and hints.get("hinted_file_opens") == 14
            and hints.get("eligible_bytes") == 2801621755
            and after_wrapper_sha == wrapper_sha
            and after_head == source_commit and after_closure == closure
            and receipt["outer_elapsed_seconds"] < diagnostic.WALL_CAP)
    receipt["status"] = "completed_read_only_50pct_probe" if good else "failed_or_incomplete"
    receipt["worker_result_status"] = (worker_result or {}).get("status")
    receipt["host_sample_count"] = len(series.samples)
    receipt["host_breach"] = series.breach
    if error is not None:
        receipt["error"] = error
    remaining.io.durable_json(TARGET / "receipt.json", receipt)
    try:
        retained = remaining.active_bytes(TARGET, diagnostic.OUTPUT_CAP)
        receipt["final_retained_output_bytes_before_final_receipt"] = retained
        if retained > diagnostic.OUTPUT_CAP - 64 * 1024:
            receipt["status"] = "failed_final_output_cap"
    except BaseException as failure:
        receipt["status"] = "failed_final_output_scan"
        receipt["final_scan_error_type"] = type(failure).__name__
    receipt["outer_elapsed_seconds"] = time.monotonic() - started
    if receipt["outer_elapsed_seconds"] >= diagnostic.WALL_CAP:
        receipt["status"] = "failed_outer_wall_cap"
    remaining.io.durable_json(TARGET / "receipt.json", receipt)
    try:
        if remaining.active_bytes(TARGET, diagnostic.OUTPUT_CAP) > diagnostic.OUTPUT_CAP:
            receipt["status"] = "failed_final_output_cap_after_receipt"
            remaining.io.durable_json(TARGET / "receipt.json", receipt)
    except BaseException as failure:
        receipt["status"] = "failed_final_output_scan_after_receipt"
        receipt["final_scan_error_type"] = type(failure).__name__
        remaining.io.durable_json(TARGET / "receipt.json", receipt)
    print(json.dumps({"status": receipt["status"],
                      "receipt": str(TARGET / "receipt.json")}, sort_keys=True))
    return 0 if receipt["status"] == "completed_read_only_50pct_probe" else 1


def main() -> int:
    if len(sys.argv) not in (3, 4):
        raise SystemExit("Expected supervisor FULL_COMMIT or worker FULL_COMMIT WRAPPER_SHA")
    action, source_commit = sys.argv[1:3]
    if action == "supervisor" and len(sys.argv) == 3:
        return supervisor(source_commit)
    if (action == "worker" and len(sys.argv) == 4
            and re.fullmatch(r"[0-9a-f]{64}", sys.argv[3])):
        return worker(source_commit, sys.argv[3])
    raise SystemExit("Undeclared read-only probe action")


if __name__ == "__main__":
    raise SystemExit(main())
