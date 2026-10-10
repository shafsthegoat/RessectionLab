"""Versioned one-use read-only ordinal-9 hinted feasibility candidate.

Root must separately review the exact committed HEAD, template bytes, host and
this source before executing. The first probe is preserved as a negative; this
one loads the ordinal-8 admission by verified bytes without registering an
unlisted scripts module. No native solve, HBE readout or release occurs.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import time
from types import ModuleType

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
TARGET = HERE / "attempt-01"
TEMPLATE = ROOT / "build/hbe-v5-tension-n16-s60-continuation-release-prep/INNER_TEMPLATE_NOT_RELEASED.json"
COMPARE = ROOT / "build/hbe-v5-n12-nocache-comparison-v1/compare.py"
COMPARE_SHA = "2113791dc16d7191b7c9c24f4148fb820450ecfff4c2d2f5c0af8ad8488d0b45"
DIAGNOSTIC = ROOT / "build/hbe-v5-n12-preflight-memory-diagnostic-v1/diagnose.py"
DIAGNOSTIC_SHA = "8684aa673e8e0189bd6c5fdfab144e7ff6d8a33af5e16047caa5a88b196dce16"
HINT = ROOT / "build/hbe-v5-n12-nocache-comparison-v1/hash_hint.py"
HINT_SHA = "96c71d0d9b8d896b713aef7305e3e86d5b12b9d2dca5fcc55515f6b2eb9cf3fc"
INITIAL_FLOOR = 54
UNCHANGED_NATIVE_FLOOR = 45
NORMAL_MASK = 1
EXPECTED_HINT_OPENS = 16
EXPECTED_HINT_BYTES = 2_849_433_519
SERIES_LIMIT = 1000


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def self_sha() -> str:
    path = Path(__file__).absolute()
    before = path.lstat()
    if path.is_symlink() or not stat.S_ISREG(before.st_mode) or before.st_size > 128 * 1024:
        raise RuntimeError("Probe source absent, linked or oversized")
    raw = path.read_bytes()
    after = path.lstat()
    ident = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns)
    if ident(before) != ident(after):
        raise RuntimeError("Probe source changed while read")
    return sha(raw)


def exact_dependency(path: Path, expected: str, name: str):
    before = path.lstat()
    if path.is_symlink() or not stat.S_ISREG(before.st_mode) or before.st_size > 128 * 1024:
        raise RuntimeError("Pinned dependency absent, linked or oversized")
    raw = path.read_bytes()
    after = path.lstat()
    ident = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns)
    if ident(before) != ident(after) or sha(raw) != expected:
        raise RuntimeError("Pinned dependency changed: " + str(path))
    module = ModuleType(name)
    module.__file__ = str(path)
    exec(compile(raw, str(path), "exec"), module.__dict__)
    return module


def dependencies():
    return (exact_dependency(COMPARE, COMPARE_SHA, "ordinal9_bound_compare"),
            exact_dependency(DIAGNOSTIC, DIAGNOSTIC_SHA, "ordinal9_bound_diagnostic"),
            exact_dependency(HINT, HINT_SHA, "ordinal9_bound_hint"))


def selected_head(commit: str) -> None:
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise RuntimeError("Exact selected source commit required")
    observed = subprocess.run(["/usr/bin/git", "rev-parse", "HEAD"], cwd=ROOT,
                              capture_output=True, check=True, timeout=5).stdout.decode().strip()
    if observed != commit:
        raise RuntimeError("Checkout changed from selected source commit")


def exact_template(commit: str, expected_sha: str) -> dict:
    if not re.fullmatch(r"[0-9a-f]{64}", expected_sha):
        raise RuntimeError("Root-reviewed template SHA-256 required")
    before = TEMPLATE.lstat()
    if TEMPLATE.is_symlink() or not stat.S_ISREG(before.st_mode) or before.st_size > 1024**2:
        raise RuntimeError("Non-executable ordinal-9 template absent, linked or oversized")
    raw = TEMPLATE.read_bytes()
    after = TEMPLATE.lstat()
    ident = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns)
    if ident(before) != ident(after) or sha(raw) != expected_sha:
        raise RuntimeError("Reviewed ordinal-9 template bytes changed")
    item = json.loads(raw)
    if (item.get("status") != "TEMPLATE_NOT_RELEASED"
            or item.get("source_commit") != commit or item.get("ordinal") != 9
            or item.get("run_id") != "tension:N16:S60:reference"
            or len(item.get("source_bindings", {})) != 20
            or len(item.get("prior_receipts", [])) != 9):
        raise RuntimeError("Exact ordinal-9 non-executable identity differs")
    return item


def small_source_closure(commit: str, template: dict, template_sha: str,
                         outer_sha: str) -> dict:
    """Check 20 original + four extension source bytes without bulk output."""
    from launchers import hbe_v5_ordinal9_continuation_v1 as wrapper
    from scripts import mechanics_hbe_v5_remaining_one_shot as remaining
    selected_head(commit)
    if set(template["source_bindings"]) != set(remaining.SOURCE_PATHS):
        raise RuntimeError("Original frozen twenty-file source closure differs")
    expected = {name: binding["sha256"] for name, binding in template["source_bindings"].items()}
    for relative, binding in template["source_bindings"].items():
        if binding["path"] != relative:
            raise RuntimeError("Original source binding path differs")
    outer = ROOT / "build/hbe-v5-tension-n16-s60-continuation-release-prep/OUTER_TEMPLATE_NOT_RELEASED.json"
    if outer.is_symlink() or not outer.is_file() or outer.stat().st_size > 1024**2:
        raise RuntimeError("Ordinal-9 outer template absent or linked")
    outer_before = outer.lstat()
    outer_raw = outer.read_bytes()
    outer_after = outer.lstat()
    ident = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns)
    if ident(outer_before) != ident(outer_after):
        raise RuntimeError("Ordinal-9 outer template changed during read")
    if not re.fullmatch(r"[0-9a-f]{64}", outer_sha) or sha(outer_raw) != outer_sha:
        raise RuntimeError("Root-reviewed outer template SHA-256 differs")
    outer_value = json.loads(outer_raw)
    if (outer_value.get("schema") != "hbe-v5-ordinal9-continuation-envelope-v1"
            or outer_value.get("status") != "TEMPLATE_NOT_RELEASED"
            or outer_value.get("source_commit") != commit
            or outer_value.get("policy") != wrapper.POLICY
            or outer_value.get("sidecar_directory") != wrapper.SIDECAR
            or outer_value.get("inner_release") != {
                "path": str(TEMPLATE.relative_to(ROOT)), "sha256": template_sha}
            or set(outer_value) != {"schema", "status", "source_commit",
                                        "extension_source_bindings", "inner_release",
                                        "sidecar_directory", "policy"}
            or set(outer_value.get("extension_source_bindings", {})) != set(wrapper.SOURCES)):
        raise RuntimeError("Ordinal-9 policy template identity differs")
    old = {}
    for relative in (*remaining.SOURCE_PATHS, *wrapper.SOURCES):
        path = ROOT / relative
        before = path.lstat()
        if path.is_symlink() or not stat.S_ISREG(before.st_mode) or before.st_size > 1024**2:
            raise RuntimeError("Committed source missing/linked: " + relative)
        raw = path.read_bytes()
        after = path.lstat()
        committed = subprocess.run(["/usr/bin/git", "show", commit + ":" + relative],
                                   cwd=ROOT, capture_output=True, check=True,
                                   timeout=10).stdout
        ident = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns)
        if ident(before) != ident(after) or raw != committed:
            raise RuntimeError("Committed source changed: " + relative)
        digest = sha(raw)
        if relative in expected and digest != expected[relative]:
            raise RuntimeError("Original template source hash differs")
        if relative in wrapper.SOURCES and outer_value["extension_source_bindings"].get(relative) != digest:
            raise RuntimeError("Extension template source hash differs")
        old[relative] = digest
    admission = wrapper._load_bound_admission(ROOT, old[wrapper.SOURCES[1]])
    exact_v2 = admission.verify_exact(root=ROOT)
    if exact_v2["physical_validation_pass"] is not None:
        raise RuntimeError("Numerical-only ancestor changed physical status")
    return {"source_commit": commit, "source_hashes": old,
            "ordinal8_exact": exact_v2,
            "expected_deck_sha256": template["adapted_deck_sha256"],
            "adapter_receipt": template["adapter_receipt"],
            "outer_template_sha256": outer_sha}


class HostSeries:
    def __init__(self, sampler, output: Path, started: float):
        self.sampler, self.output, self.started = sampler, output, started
        self.samples: list[dict] = []
        self.breach = None

    def flush(self):
        from scripts import mechanics_hbe_v5_n8_one_shot as io
        payload = {"schema": "hbe-v5-ordinal9-host-series-v1",
                   "samples": self.samples, "breach": self.breach}
        raw = json.dumps(payload, sort_keys=True, allow_nan=False).encode()
        if len(raw) > 1024**2:
            raise RuntimeError("Host series exceeded one MiB")
        io.durable_json(self.output / "host-time-series.json", payload)

    def sample(self, label: str, floor: int = UNCHANGED_NATIVE_FLOOR):
        if len(self.samples) >= SERIES_LIMIT:
            raise RuntimeError("Host series limit exhausted")
        host = self.sampler.host()
        item = {"label": label, "elapsed_seconds": time.monotonic() - self.started,
                "available_percent": host["available_percent"],
                "kernel_pressure_mask": host["kernel_pressure_mask"],
                "swap_used_bytes": host["swap_used_bytes"]}
        self.samples.append(item)
        if host["available_percent"] < floor or host["kernel_pressure_mask"] != NORMAL_MASK:
            self.breach = item
            self.flush()
            raise RuntimeError("Read-only ordinal-9 host fell below declared floor")
        if label == "rss_tick" and len(self.samples) % 10 == 0:
            self.flush()
        return item

    def observe(self, original, pid: int, *, timeout_seconds: float):
        rss, members = original(pid, timeout_seconds=timeout_seconds)
        self.sample("rss_tick")
        return rss, members


def worker(commit: str, template_sha: str, outer_sha: str,
           expected_self_sha: str) -> int:
    if self_sha() != expected_self_sha:
        raise RuntimeError("Worker probe source changed before read-only validation")
    compare, diagnostic, hint = dependencies()
    selected_head(commit)
    template = exact_template(commit, template_sha)
    closure = small_source_closure(commit, template, template_sha, outer_sha)
    diagnostic.OUT = TARGET
    compare.ARMS = {"hinted": TARGET}
    from scripts import mechanics_hbe_v5_remaining_one_shot as remaining
    from launchers import hbe_v5_ordinal9_continuation_v1 as wrapper
    admission = wrapper._load_bound_admission(
        ROOT, closure["source_hashes"][wrapper.SOURCES[1]])
    started = time.monotonic()
    samples = []
    host_sampler = diagnostic.load_host_sampler()
    original_snapshot = diagnostic.snapshot

    def guarded_snapshot(label, items, *, started, host_sampler):
        original_snapshot(label, items, started=started, host_sampler=host_sampler)
        direct = items[-1]["direct_kernel_host"]
        floor = INITIAL_FLOOR if label == "worker_before_import" else UNCHANGED_NATIVE_FLOOR
        if direct["kernel_pressure_mask"] != NORMAL_MASK or direct["available_percent"] < floor:
            raise RuntimeError("Worker host below read-only ordinal-9 floor")

    def row9_terminal(source_commit: str) -> int:
        guarded_snapshot("worker_before_import", samples, started=started,
                         host_sampler=host_sampler)
        release = dict(template, status="root_released_one_native_call")  # memory only
        guarded_snapshot("before_validate_release", samples, started=started,
                         host_sampler=host_sampler)
        context = remaining.validate_release(release, root=ROOT)
        if context["index"] != 9 or context["run_id"] != remaining.ORDER[9]:
            raise RuntimeError("Read-only validation returned wrong row")
        exact = admission.verify_exact(root=ROOT)
        charged = admission.continuation_ledger(context["previous"], exact, 9)
        diagnostic.save_bounded("continuation-ledger.json", charged)
        guarded_snapshot("after_validate_release", samples, started=started,
                         host_sampler=host_sampler)
        selected_head(commit)
        diagnostic.save_bounded("worker-result.json", {
            "status": "read_only_ordinal9_full_chain_complete",
            "source_commit": commit, "run_id": context["run_id"],
            "index": context["index"],
            "adapted_deck_sha256": sha(context["deck"]),
            "expected_adapted_deck_sha256": template["adapted_deck_sha256"],
            "charged_previous": charged["charged_cumulative_ledger"],
            "elapsed_seconds": time.monotonic() - started,
            "native_calls": 0, "hbe_readout_calls": 0,
            "release_written": False, "target_reserved": False,
            "physical_validation_pass": None})
        return 0

    diagnostic.snapshot = guarded_snapshot
    diagnostic.worker_terminal = row9_terminal
    try:
        allowed = [ROOT / Path(remaining.receipt_path(i)).parent for i in range(9)]
        result = compare.run_worker_arm("hinted", commit, diagnostic, hint,
                                        allowed_directories=allowed)
        if (self_sha() != expected_self_sha
                or exact_template(commit, template_sha) != template
                or small_source_closure(commit, template, template_sha,
                                        outer_sha) != closure):
            raise RuntimeError("Worker source or ordinal-8 ancestry changed postvalidation")
        return result
    finally:
        diagnostic.snapshot = original_snapshot


def supervisor(commit: str, template_sha: str, outer_sha: str) -> int:
    from scripts import febio_runtime
    from scripts import mechanics_hbe_v5_remaining_one_shot as remaining
    compare, diagnostic, _ = dependencies()
    selected_head(commit)
    template = exact_template(commit, template_sha)
    closure = small_source_closure(commit, template, template_sha, outer_sha)
    wrapper_sha = self_sha()
    from launchers import hbe_v5_ordinal9_continuation_v1 as wrapper
    native_target = ROOT / remaining.output_directory(9)
    policy_target = ROOT / wrapper.SIDECAR
    if (native_target.exists() or native_target.is_symlink()
            or policy_target.exists() or policy_target.is_symlink()):
        raise RuntimeError("Ordinal-9 native or continuation policy target already consumed")
    if TARGET.exists() or TARGET.is_symlink():
        raise RuntimeError("Ordinal-9 read-only feasibility attempt is one-use")
    sampler = diagnostic.load_host_sampler()
    before = sampler.host()
    if before["kernel_pressure_mask"] != NORMAL_MASK or before["available_percent"] < INITIAL_FLOOR:
        raise RuntimeError("Initial host below prospective 54%/normal floor")
    TARGET.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    series = HostSeries(sampler, TARGET, started)
    env = os.environ.copy()
    env.update(diagnostic.THREAD_ENV)
    env["PYTHONPATH"] = str(ROOT)
    receipt = {"schema": "hbe-v5-ordinal9-read-only-feasibility-v2",
               "status": "running", "source_commit": commit,
               "template_sha256": template_sha,
               "outer_template_sha256": outer_sha,
               "probe_source_sha256": wrapper_sha,
               "compare_source_sha256": COMPARE_SHA,
               "diagnostic_source_sha256": DIAGNOSTIC_SHA,
               "hint_source_sha256": HINT_SHA,
               "source_closure": closure,
               "host_before": before,
               "host_policy": {"initial_percent": INITIAL_FLOOR,
                               "during_validation_percent": UNCHANGED_NATIVE_FLOOR,
                               "normal_mask": NORMAL_MASK},
               "native_calls": 0, "hbe_readout_calls": 0,
               "release_written": False, "hbe_output_reserved": False,
               "caps": {"outer_wall_seconds": diagnostic.WALL_CAP,
                        "worker_wall_seconds": diagnostic.WORKER_WALL_CAP,
                        "sampled_group_rss_bytes": diagnostic.RSS_CAP,
                        "total_output_bytes": diagnostic.OUTPUT_CAP,
                        "numerical_threads": 1, "attempts": 1},
               "supervisor_stage_label": "readout_api_only_not_HBE_readout"}
    try:
        series.sample("before_worker", INITIAL_FLOOR)
    except BaseException as error:
        receipt["status"] = "failed_or_incomplete"
        receipt["error"] = {"type": type(error).__name__, "message": str(error)[:500]}
        remaining.io.durable_json(TARGET / "receipt.json", receipt)
        return 1
    owner = diagnostic.OwnedWorker()
    result, error = None, None
    try:
        result = remaining.supervise_stage(
            "readout", [sys.executable, "-B", str(Path(__file__).resolve()),
                        "worker", commit, template_sha, outer_sha, wrapper_sha],
            TARGET, receipt, cwd=ROOT, environment=env,
            wall_cap=diagnostic.WORKER_WALL_CAP, rss_cap=diagnostic.RSS_CAP,
            output_cap=diagnostic.OUTPUT_CAP,
            rss_observer=lambda pid, *, timeout_seconds: series.observe(
                febio_runtime.process_group_rss, pid, timeout_seconds=timeout_seconds),
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
                "fallback_used": True, "remaining_members": [],
                "errors": ["cleanup_exception:" + type(failure).__name__]}
        try:
            series.sample("after_cleanup")
            series.flush()
        except BaseException as failure:
            error = error or {"type": type(failure).__name__, "message": str(failure)[:500]}
        receipt["outer_elapsed_seconds"] = time.monotonic() - started
        receipt["status"] = "post_supervision_pending"
        remaining.io.durable_json(TARGET / "receipt.json", receipt)

    def bounded_json(name: str):
        path = TARGET / name
        try:
            if path.is_symlink() or not path.is_file() or path.stat().st_size > 1024**2:
                return None
            item = json.loads(path.read_bytes())
            return item if isinstance(item, dict) else None
        except (OSError, UnicodeError, json.JSONDecodeError):
            return None

    worker_result = bounded_json("worker-result.json")
    identity = bounded_json("validation-identity.json")
    hints = bounded_json("hint-audit.json")
    ledger = bounded_json("continuation-ledger.json")
    cleanup = receipt["owned_worker_cleanup"]
    expected_ledger = None
    if identity is not None and isinstance(identity.get("previous"), dict):
        try:
            admission = wrapper._load_bound_admission(
                ROOT, closure["source_hashes"][wrapper.SOURCES[1]])
            expected_ledger = admission.continuation_ledger(
                identity["previous"], closure["ordinal8_exact"], 9)
        except BaseException as failure:
            error = error or {"type": type(failure).__name__,
                              "message": str(failure)[:500]}
    try:
        final_closure = small_source_closure(commit, template, template_sha,
                                             outer_sha)
        if exact_template(commit, template_sha) != template:
            raise RuntimeError("Inner ordinal-9 template changed after validation")
        selected_head(commit)
        final_wrapper_sha = self_sha()
    except BaseException as failure:
        final_closure, final_wrapper_sha = None, None
        error = error or {"type": type(failure).__name__, "message": str(failure)[:500]}
    good = (error is None and result is not None
            and result["status"] == "completed_within_caps"
            and cleanup["contained"] and cleanup["direct_child_reaped"]
            and not cleanup["fallback_used"] and not cleanup["errors"]
            and series.breach is None
            and receipt.get("readout_calls_attempted") == 1
            and worker_result is not None
            and worker_result.get("status") == "read_only_ordinal9_full_chain_complete"
            and worker_result.get("source_commit") == commit
            and worker_result.get("run_id") == "tension:N16:S60:reference"
            and worker_result.get("index") == 9
            and worker_result.get("adapted_deck_sha256") == template["adapted_deck_sha256"]
            and worker_result.get("native_calls") == 0
            and worker_result.get("hbe_readout_calls") == 0
            and worker_result.get("release_written") is False
            and worker_result.get("target_reserved") is False
            and worker_result.get("physical_validation_pass") is None
            and identity is not None and identity.get("index") == 9
            and identity.get("run_id") == "tension:N16:S60:reference"
            and identity.get("deck_sha256") == template["adapted_deck_sha256"]
            and identity.get("adapter_receipt") == closure["adapter_receipt"]
            and identity.get("source_hashes") == {
                name: closure["source_hashes"][name]
                for name in template["source_bindings"]}
            and identity.get("observed_head_preflight") == commit
            and ledger is not None and ledger == expected_ledger
            and ledger.get("old_native_receipt_ledger") == identity.get("previous")
            and ledger.get("authenticated_ordinal8_surcharge") ==
                closure["ordinal8_exact"]["surcharge"]
            and ledger.get("charged_cumulative_ledger") ==
                worker_result.get("charged_previous")
            and hints is not None and hints.get("eligible_files") == EXPECTED_HINT_OPENS
            and hints.get("hinted_file_opens") == EXPECTED_HINT_OPENS
            and hints.get("eligible_bytes") == EXPECTED_HINT_BYTES
            and not native_target.exists() and not native_target.is_symlink()
            and not policy_target.exists() and not policy_target.is_symlink()
            and any(item["label"] == "rss_tick" for item in series.samples)
            and final_closure == closure and final_wrapper_sha == wrapper_sha
            and receipt["outer_elapsed_seconds"] < diagnostic.WALL_CAP)
    receipt["status"] = "completed_read_only_ordinal9_feasibility_v2" if good else "failed_or_incomplete"
    receipt["worker_status"] = (worker_result or {}).get("status")
    receipt["host_breach"] = series.breach
    receipt["host_sample_count"] = len(series.samples)
    if error is not None:
        receipt["error"] = error
    remaining.io.durable_json(TARGET / "receipt.json", receipt)
    try:
        output = remaining.active_bytes(TARGET, diagnostic.OUTPUT_CAP)
        receipt["retained_output_snapshot_bytes"] = output
        if output > diagnostic.OUTPUT_CAP - 64 * 1024:
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
    return 0 if receipt["status"] == "completed_read_only_ordinal9_feasibility_v2" else 1


def main() -> int:
    if len(sys.argv) == 5 and sys.argv[1] == "supervisor":
        return supervisor(sys.argv[2], sys.argv[3], sys.argv[4])
    if len(sys.argv) == 6 and sys.argv[1] == "worker":
        return worker(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
    raise SystemExit("Expected supervisor FULL_COMMIT INNER_SHA OUTER_SHA or worker FULL_COMMIT INNER_SHA OUTER_SHA WRAPPER_SHA")


if __name__ == "__main__":
    raise SystemExit(main())
