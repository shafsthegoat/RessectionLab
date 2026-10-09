"""Prepared, unexecuted baseline/no-cache HBE predecessor validation comparison.

Both arms run the same exact read-only validate_release. The only hinted-arm
change is F_NOCACHE on eligible predecessor file descriptors during the
existing original SHA/stat guard. Root must approve each full-size read.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
DIAGNOSTIC = ROOT / "build/hbe-v5-n12-preflight-memory-diagnostic-v1/diagnose.py"
DIAGNOSTIC_SHA = "8684aa673e8e0189bd6c5fdfab144e7ff6d8a33af5e16047caa5a88b196dce16"
HINT = HERE / "hash_hint.py"
HINT_SHA = "96c71d0d9b8d896b713aef7305e3e86d5b12b9d2dca5fcc55515f6b2eb9cf3fc"
ARMS = {"baseline": HERE / "baseline-01", "hinted": HERE / "hinted-01"}


def comparison_source_sha() -> str:
    """Bind these ignored wrapper bytes across supervisor, child and postguard."""
    path = Path(__file__).absolute()
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 128 * 1024:
        raise RuntimeError("Comparison source absent, linked, or oversized")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_exact(path: Path, expected: str, name: str):
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 128 * 1024:
        raise RuntimeError("Pinned comparison source absent, linked, or oversized")
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise RuntimeError("Pinned comparison source changed")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Pinned comparison source cannot load")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def selected_head(source_commit: str) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", source_commit):
        raise RuntimeError("One root-selected full commit is required")
    head = subprocess.run(["/usr/bin/git", "rev-parse", "HEAD"], cwd=ROOT,
                          capture_output=True, text=True, check=True,
                          timeout=5).stdout.strip()
    if head != source_commit:
        raise RuntimeError("Checkout changed from selected comparison source")
    return head


def run_worker_arm(mode: str, source_commit: str, diagnostic, hint,
                   *, allowed_directories=None) -> int:
    from scripts import mechanics_hbe_v5_remaining_one_shot as remaining
    if mode not in ARMS:
        raise RuntimeError("Undeclared comparison arm")
    diagnostic.OUT = ARMS[mode]
    original_validation = remaining.validate_release

    def captured_validation(*args, **kwargs):
        checked = original_validation(*args, **kwargs)
        identity = {
            "status": "full_read_only_validation_identity",
            "run_id": checked["run_id"], "index": checked["index"],
            "deck_sha256": hashlib.sha256(checked["deck"]).hexdigest(),
            "previous": checked["previous"],
            "adapter_receipt": checked["adapter_receipt"],
            "source_hashes": checked["source_hashes"],
            "observed_head_preflight": checked["observed_head_preflight"],
        }
        diagnostic.save_bounded("validation-identity.json", identity)
        return checked

    remaining.validate_release = captured_validation
    if mode == "baseline":
        try:
            return diagnostic.worker_terminal(source_commit)
        finally:
            remaining.validate_release = original_validation
    if allowed_directories is None:
        allowed_directories = [ROOT / Path(remaining.receipt_path(i)).parent
                               for i in range(8)]
    audit = {"mode": mode, "eligible_bytes": 0, "eligible_files": 0,
             "hinted_file_opens": 0, "hinted_file_paths": []}
    original_chain = remaining.validate_prior_chain
    original_hash = remaining.io.file_hash
    chain_depth = 0

    def hinted_chain(*args, **kwargs):
        nonlocal chain_depth
        # The frozen N12 exception re-enters predecessor validation. Keep the
        # outer hint installed until its complete chain returns; restoring on
        # the inner return would leave later large predecessors unhinted.
        if chain_depth:
            return original_chain(*args, **kwargs)

        def selected_hash(path, *, maximum=128 * 1024**2):
            return hint.hinted_hash(
                path, maximum=maximum, original_hash=original_hash,
                allowed_directories=allowed_directories, audit=audit)

        previous_hash = remaining.io.file_hash
        chain_depth += 1
        remaining.io.file_hash = selected_hash
        try:
            return original_chain(*args, **kwargs)
        finally:
            remaining.io.file_hash = previous_hash
            chain_depth -= 1

    remaining.validate_prior_chain = hinted_chain
    try:
        return diagnostic.worker_terminal(source_commit)
    finally:
        remaining.validate_prior_chain = original_chain
        remaining.validate_release = original_validation
        diagnostic.save_bounded("hint-audit.json", audit)


def worker(mode: str, source_commit: str, expected_comparison_sha: str) -> int:
    if comparison_source_sha() != expected_comparison_sha:
        raise RuntimeError("Comparison wrapper changed before child validation")
    selected_head(source_commit)
    diagnostic = load_exact(DIAGNOSTIC, DIAGNOSTIC_SHA, "hbe_n12_saved_memory_diagnostic")
    hint = load_exact(HINT, HINT_SHA, "hbe_n12_per_fd_hint")
    diagnostic.OUT = ARMS[mode]
    return run_worker_arm(mode, source_commit, diagnostic, hint)


def supervisor(mode: str, source_commit: str) -> int:
    from scripts import febio_runtime
    from scripts import mechanics_hbe_v5_remaining_one_shot as remaining
    selected_head(source_commit)
    diagnostic = load_exact(DIAGNOSTIC, DIAGNOSTIC_SHA, "hbe_n12_saved_memory_diagnostic")
    load_exact(HINT, HINT_SHA, "hbe_n12_per_fd_hint")
    comparison_sha = comparison_source_sha()
    target = ARMS[mode]
    if target.exists() or target.is_symlink():
        raise RuntimeError("Comparison arm is one-use and already exists")
    host = diagnostic.load_host_sampler().host()
    if host["kernel_pressure_mask"] != 1 or host["available_percent"] < 55:
        raise RuntimeError("Comparison host preflight below matched 55%/normal threshold")
    target.mkdir(parents=True, exist_ok=False)
    environment = os.environ.copy()
    environment.update(diagnostic.THREAD_ENV)
    environment["PYTHONPATH"] = str(ROOT)
    receipt = {"schema": "hbe-v5-n12-nocache-comparison-arm-v1",
               "status": "running", "arm": mode, "source_commit": source_commit,
               "diagnostic_source_sha256": DIAGNOSTIC_SHA,
               "hint_source_sha256": HINT_SHA,
               "comparison_source_sha256": comparison_sha,
               "host_before": host,
               "native_calls": 0, "hbe_readout_calls": 0,
               "release_written": False, "hbe_output_reserved": False,
               "caps": {"outer_wall_seconds": diagnostic.WALL_CAP,
                        "worker_wall_seconds": diagnostic.WORKER_WALL_CAP,
                        "sampled_process_group_rss_bytes": diagnostic.RSS_CAP,
                        "aggregate_output_bytes": diagnostic.OUTPUT_CAP,
                        "numerical_threads": 1, "attempts": 1},
               "supervisor_stage_label": "readout_api_only_not_hbe_readout"}
    owner = diagnostic.OwnedWorker()
    started = time.monotonic()
    result = None
    supervision_error = None
    try:
        result = remaining.supervise_stage(
            "readout", [sys.executable, "-B", str(Path(__file__).resolve()),
                        "worker", mode, source_commit, comparison_sha],
            target, receipt, cwd=ROOT, environment=environment,
            wall_cap=diagnostic.WORKER_WALL_CAP, rss_cap=diagnostic.RSS_CAP,
            output_cap=diagnostic.OUTPUT_CAP,
            rss_observer=febio_runtime.process_group_rss, popen=owner)
    except BaseException as error:
        supervision_error = {"type": type(error).__name__, "message": str(error)[:500]}
    finally:
        try:
            receipt["owned_worker_cleanup"] = owner.cleanup(
                febio_runtime.process_group_rss, started + diagnostic.WALL_CAP)
        except BaseException as error:
            receipt["owned_worker_cleanup"] = {
                "contained": False, "direct_child_reaped": False,
                "errors": ["cleanup_exception:" + type(error).__name__],
                "remaining_members": [], "fallback_used": True}
        receipt["outer_elapsed_seconds"] = time.monotonic() - started
        receipt["status"] = "post_supervision_pending"
        remaining.io.durable_json(target / "receipt.json", receipt)
    worker_result_path = target / "worker-result.json"
    try:
        worker_result = (json.loads(worker_result_path.read_text())
                         if worker_result_path.is_file() and not worker_result_path.is_symlink()
                         and worker_result_path.stat().st_size <= 1024**2 else None)
    except (OSError, UnicodeError, json.JSONDecodeError):
        worker_result = None
    hint_audit_path = target / "hint-audit.json"
    try:
        hint_audit = (json.loads(hint_audit_path.read_text())
                      if hint_audit_path.is_file() and not hint_audit_path.is_symlink()
                      and hint_audit_path.stat().st_size <= 1024**2 else None)
    except (OSError, UnicodeError, json.JSONDecodeError):
        hint_audit = None
    identity_path = target / "validation-identity.json"
    try:
        identity = (json.loads(identity_path.read_text())
                    if identity_path.is_file() and not identity_path.is_symlink()
                    and identity_path.stat().st_size <= 1024**2 else None)
    except (OSError, UnicodeError, json.JSONDecodeError):
        identity = None
    if not isinstance(worker_result, dict):
        worker_result = None
    if not isinstance(hint_audit, dict):
        hint_audit = None
    if not isinstance(identity, dict):
        identity = None
    cleanup = receipt["owned_worker_cleanup"]
    try:
        post_comparison_sha = comparison_source_sha()
    except (OSError, RuntimeError):
        post_comparison_sha = None
    receipt["comparison_source_sha256_postguard"] = post_comparison_sha
    good = (supervision_error is None and result is not None
            and result["status"] == "completed_within_caps"
            and cleanup["contained"] and cleanup["direct_child_reaped"]
            and not cleanup["fallback_used"] and not cleanup["errors"]
            and worker_result is not None
            and worker_result.get("status") == "read_only_validation_memory_diagnostic_complete"
            and worker_result.get("native_calls") == 0
            and worker_result.get("release_written") is False
            and worker_result.get("target_reserved") is False
            and identity is not None
            and identity.get("index") == 8
            and identity.get("deck_sha256") == worker_result.get("adapted_deck_sha256")
            and identity.get("observed_head_preflight") == source_commit
            and post_comparison_sha == comparison_sha
            and (mode == "baseline" or (hint_audit is not None
                 and type(hint_audit.get("eligible_files")) is int
                 and hint_audit.get("eligible_files") == hint_audit.get("hinted_file_opens")
                 and hint_audit.get("hinted_file_opens") > 0))
            and receipt["outer_elapsed_seconds"] < diagnostic.WALL_CAP)
    receipt["status"] = "completed_read_only_arm" if good else "failed_or_incomplete"
    receipt["worker_result_status"] = (worker_result or {}).get("status")
    receipt["hint_audit_present"] = hint_audit is not None
    receipt["validation_identity_present"] = identity is not None
    if supervision_error is not None:
        receipt["supervision_error"] = supervision_error
    remaining.io.durable_json(target / "receipt.json", receipt)
    try:
        retained = remaining.active_bytes(target, diagnostic.OUTPUT_CAP)
        receipt["final_retained_output_bytes"] = retained
        if retained > diagnostic.OUTPUT_CAP - 64 * 1024:
            receipt["status"] = "failed_final_output_cap"
    except Exception as error:
        receipt["status"] = "failed_final_output_scan"
        receipt["final_scan_error_type"] = type(error).__name__
    receipt["outer_elapsed_seconds"] = time.monotonic() - started
    if receipt["outer_elapsed_seconds"] >= diagnostic.WALL_CAP:
        receipt["status"] = "failed_outer_wall_cap"
    remaining.io.durable_json(target / "receipt.json", receipt)
    try:
        if remaining.active_bytes(target, diagnostic.OUTPUT_CAP) > diagnostic.OUTPUT_CAP:
            receipt["status"] = "failed_final_output_cap_after_receipt"
            remaining.io.durable_json(target / "receipt.json", receipt)
    except Exception as error:
        receipt["status"] = "failed_final_output_scan_after_receipt"
        receipt["final_scan_error_type"] = type(error).__name__
        remaining.io.durable_json(target / "receipt.json", receipt)
    print(json.dumps({"status": receipt["status"], "arm": mode,
                      "receipt": str(target / "receipt.json")}, sort_keys=True))
    return 0 if receipt["status"] == "completed_read_only_arm" else 1


def main() -> int:
    if len(sys.argv) not in (4, 5) or sys.argv[2] not in ARMS:
        raise SystemExit("Expected (worker|supervisor) (baseline|hinted) full-source-commit [wrapper-sha]")
    action, mode, commit = sys.argv[1:4]
    if action == "worker":
        if len(sys.argv) != 5 or not re.fullmatch(r"[0-9a-f]{64}", sys.argv[4]):
            raise SystemExit("Worker requires supervisor-bound wrapper SHA-256")
        return worker(mode, commit, sys.argv[4])
    if action == "supervisor":
        if len(sys.argv) != 4:
            raise SystemExit("Supervisor does not accept worker wrapper SHA-256")
        return supervisor(mode, commit)
    raise SystemExit("Expected worker or supervisor")


if __name__ == "__main__":
    raise SystemExit(main())
