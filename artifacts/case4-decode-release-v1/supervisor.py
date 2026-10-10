"""Prospective one-shot Case4 DEVELOPMENT display-decode resource monitor.

One decode arm reuses the reviewed identity-bound resource guard.
It consumes only the separately accepted saved network output.
"""
import fcntl
import json
import os
import subprocess
import sys
import time

from darwin_fast_sampler import FastDarwinSampler
from monitor_primitives import (CONTRACT, HERE, ROOT, RUNTIME, WORKER, deferred_receipt,
                                last_phase, overlap, preflight, sha256_file, stop_reason,
                                utc_now)
from safe_finalize import closeout_supervision, write_json_once_durable
from slow_inventory import BackgroundInventory, collect, load_acquisition_allowlist


LAUNCH_GATE = HERE / "launch_gate.py"
MODEL_DIR = HERE.parent / "model_backend"
FORWARD_AUDIT = ROOT / "build/scan-target-estimator-independent/case4-forward-v1-actual-audit.md"


def checked_forward_for_decode(contract):
    """Require the exact independently audited forward before host preflight."""
    audit = contract["independent_forward_audit"]
    if (audit.get("path") != str(FORWARD_AUDIT.relative_to(ROOT)) or
            not FORWARD_AUDIT.is_file() or FORWARD_AUDIT.is_symlink() or
            sha256_file(FORWARD_AUDIT) != audit.get("sha256")):
        raise ValueError("independent forward audit missing or changed")
    expected = contract["accepted_forward_sha256"]
    paths = {
        "contract": MODEL_DIR / "pair-contract.json",
        "result": MODEL_DIR / "case4/result.json",
        "supervision": MODEL_DIR / "case4/supervision.json",
        "logits": MODEL_DIR / "case4/logits.npy",
        "input_receipt": MODEL_DIR / "case4-input/input-receipt.json",
        "input_npy": MODEL_DIR / "case4-input/input.npy",
    }
    for name, path in paths.items():
        if not path.is_file() or path.is_symlink() or sha256_file(path) != expected[name]:
            raise ValueError("accepted Case4 forward bytes changed: " + name)
    result = json.loads(paths["result"].read_text())
    supervision = json.loads(paths["supervision"].read_text())
    finalization = supervision.get("finalization", {})
    if (sha256_file(paths["result"]) != expected["result"] or
            sha256_file(paths["supervision"]) != expected["supervision"] or
            result.get("contract_sha256") != expected["contract"] or
            result.get("logits_npy_sha256") != expected["logits"] or
            result.get("output_nonfinite") != 0 or
            result.get("model_inference_performed") is not True or
            result.get("planning_admitted") is not False or
            result.get("clinical_evidence") is not False or
            supervision.get("accepted_for_feasibility") is not True or
            supervision.get("contract_sha256") != expected["contract"] or
            supervision.get("exit_code") != 0 or
            supervision.get("watchdog_reason") is not None or
            supervision.get("post_guard_reason") is not None or
            finalization.get("cleanup_errors") != [] or
            finalization.get("remaining_group_pids") != [] or
            finalization.get("remaining_detached_pids") != []):
        raise ValueError("Case4 forward did not produce a completed guarded diagnostic")


def abort_unreleased_gate(child, token_write_fd, fast, expected_identity, expected_pgid,
                          outdir, reason, contract_sha256, error=None):
    """Close the gate; only identity-bound cleanup is allowed if EOF is insufficient."""
    try:
        os.close(token_write_fd)
    except OSError:
        pass
    actions = []
    residual_possible = False
    try:
        exit_code = child.wait(timeout=5)
    except subprocess.TimeoutExpired:
        exit_code = None
        if expected_identity is not None and expected_pgid == child.pid:
            try:
                if os.getpgid(child.pid) == expected_pgid:
                    actions.append(fast.signal_if_same_process(child.pid, expected_identity, 9))
                    exit_code = child.wait(timeout=1)
            except Exception as stop_error:
                actions.append({"type": type(stop_error).__name__, "error": str(stop_error)})
        if exit_code is None:
            residual_possible = True
    except Exception as wait_error:
        exit_code = None
        residual_possible = True
        actions.append({"type": type(wait_error).__name__, "error": str(wait_error)})
    payload = {"accepted_for_feasibility": False, "no_retry": True,
               "child_started": True, "worker_released": False,
               "reason": reason, "error": error, "pid": child.pid,
               "launch_start_identity": list(expected_identity) if expected_identity else None,
               "expected_pgid": expected_pgid, "exit_code": exit_code,
               "cleanup_actions": actions, "residual_possible": residual_possible,
               "manual_attention_required": residual_possible,
               "contract_sha256": contract_sha256, "completed_at_utc": utc_now()}
    try:
        write_json_once_durable(outdir / "unreleased-launch-failure.json", payload)
    except Exception as receipt_error:
        payload["manual_attention_required"] = True
        payload["receipt_write_error"] = "%s: %s" % (type(receipt_error).__name__, receipt_error)
    return payload


def bind_launch_identity(child, fast, deadline_seconds=0.5):
    """Observe kernel PID/start and new-session PGID before releasing the gate."""
    deadline = time.monotonic() + deadline_seconds
    while time.monotonic() < deadline:
        identity = fast.process_start_identity(child.pid)
        if identity is not None:
            pgid = os.getpgid(child.pid)
            if pgid != child.pid:
                raise ValueError("gated worker did not form its own process group")
            return identity, pgid
        if child.poll() is not None:
            break
        time.sleep(0.01)
    raise RuntimeError("gated worker start identity unavailable before deadline")


def pre_release_inventory(supervisor_pid, pgid, acquisition_allowlist,
                          collect_fn=collect):
    """Start a synchronous inventory with the bound child PGID already known."""
    snapshot = collect_fn(supervisor_pid, pgid, acquisition_allowlist)
    if not isinstance(snapshot, dict) or "project_process_inventory" not in snapshot:
        raise ValueError("pre-release inventory is unavailable")
    if overlap(snapshot, pgid):
        raise RuntimeError("project compute overlap before input worker release")
    return snapshot


def emergency_after_closeout_failure(child, fast, pgid, launch_identity, slow,
                                     outdir, error, contract_sha256,
                                     reason="closeout_exception",
                                     detached_seen=None):
    """Fail closed if receipt closeout itself unexpectedly raises after release."""
    actions = []
    # A failed closeout or ambiguous release can lose a child before it is
    # observed; post-inventory cannot prove an unseen orphan never existed.
    residual_possible = True
    orphan_status_unverified = True
    remaining_group_pids = []
    remaining_detached_pids = []
    detached_identities = dict(detached_seen or {})
    try:
        before = fast.group(pgid, child.pid)
        for pid, identity in before["detached_descendant_start_identities"].items():
            previous = detached_identities.setdefault(int(pid), identity)
            if tuple(previous) != tuple(identity):
                raise RuntimeError("detached PID start identity changed before emergency cleanup")
        unbound_group_pids = sorted(set(before["pids"]) - {child.pid} - set(detached_identities))
        if unbound_group_pids:
            actions.append({"unbound_group_pids_not_signaled": unbound_group_pids})
            residual_possible = True
    except Exception as inventory_error:
        residual_possible = True
        actions.append({"type": type(inventory_error).__name__,
                        "error": "pre_cleanup_inventory: %s" % inventory_error})
    if child.poll() is None:
        try:
            if os.getpgid(child.pid) == pgid:
                actions.append(fast.signal_if_same_process(child.pid, launch_identity, 9))
            else:
                residual_possible = True
        except Exception as stop_error:
            residual_possible = True
            actions.append({"type": type(stop_error).__name__, "error": str(stop_error)})
    for pid, identity in sorted(detached_identities.items()):
        try:
            outcome = fast.signal_if_same_process(int(pid), tuple(identity), 9)
            actions.append(outcome)
            if not outcome.get("signaled") and fast.process_start_identity(int(pid)) == tuple(identity):
                residual_possible = True
        except Exception as stop_error:
            residual_possible = True
            actions.append({"pid": int(pid), "type": type(stop_error).__name__,
                            "error": str(stop_error)})
    try:
        exit_code = child.wait(timeout=5)
    except Exception as wait_error:
        exit_code = None
        residual_possible = True
        actions.append({"type": type(wait_error).__name__, "error": str(wait_error)})
    try:
        after = fast.group(pgid, child.pid)
        remaining_group_pids = sorted(set(after["pids"]))
        if remaining_group_pids:
            residual_possible = True
    except Exception as inventory_error:
        residual_possible = True
        actions.append({"type": type(inventory_error).__name__,
                        "error": "post_cleanup_inventory: %s" % inventory_error})
    for pid, identity in sorted(detached_identities.items()):
        try:
            if fast.process_start_identity(int(pid)) == tuple(identity):
                remaining_detached_pids.append(int(pid))
                residual_possible = True
        except Exception as liveness_error:
            residual_possible = True
            actions.append({"pid": int(pid), "type": type(liveness_error).__name__,
                            "error": "post_detached_liveness: %s" % liveness_error})
    try:
        slow.close()
    except Exception as close_error:
        actions.append({"type": type(close_error).__name__, "error": str(close_error)})
    payload = {"accepted_for_feasibility": False, "no_retry": True,
               "reason": reason, "error": "%s: %s" % (type(error).__name__, error),
               "pid": child.pid, "pgid": pgid, "launch_start_identity": list(launch_identity),
               "exit_code": exit_code, "cleanup_actions": actions,
               "residual_possible": residual_possible,
               "orphan_status_unverified": orphan_status_unverified,
               "remaining_group_pids": remaining_group_pids,
               "remaining_detached_pids": remaining_detached_pids,
               "manual_attention_required": True,
               "contract_sha256": contract_sha256, "completed_at_utc": utc_now()}
    try:
        write_json_once_durable(outdir / "closeout-failure.json", payload)
    except Exception as receipt_error:
        payload["receipt_write_error"] = "%s: %s" % (type(receipt_error).__name__, receipt_error)
    return payload


def run_one(arm):
    contract = json.loads(CONTRACT.read_text())
    if (contract["gate"] != "CASE4_DEVELOPMENT_DISPLAY_DECODE_V1" or
            contract["monitor_revision"] != "identity-bound-closeout-v2" or
            contract["arm_order"] != ["case4"] or
            arm != "case4" or contract["native_arm_enabled"] or
            contract.get("release_status") != "root_released_once" or
            contract.get("model_inference_performed_in_this_step") is not False or
            contract.get("planning_admitted") is not False or
            contract.get("evaluation_admitted") is not False or
            contract.get("clinical_evidence") is not False or
            contract.get("split_role") != "DEVELOPMENT" or
            contract.get("process_group_rss_cap_kib") != 3145728 or
            contract.get("wall_time_cap_seconds") != 120 or
            contract.get("host_preflight_seconds") != 30 or
            contract.get("host_preflight_min_free_percent") != 45 or
            contract.get("host_abort_min_free_percent") != 30 or
            contract.get("host_abort_free_drop_points") != 20 or
            contract.get("host_abort_swap_growth_mib") != 128 or
            contract.get("host_abort_kernel_pressure_masks") != [2, 4] or
            contract.get("host_sample_max_interval_seconds") != 0.2 or
            contract.get("host_sample_target_interval_seconds") != 0.05 or
            contract.get("one_attempt_per_arm") is not True or
            contract.get("fresh_preflight_per_arm") is not True or
            contract.get("arm_budget") != {
                name: {"process_group_rss_cap_kib": contract["process_group_rss_cap_kib"],
                       "wall_time_cap_seconds": contract["wall_time_cap_seconds"],
                       "host_abort_kernel_pressure_masks":
                       contract["host_abort_kernel_pressure_masks"]}
                for name in ("case4",)}):
        raise ValueError("unexpected Case4 display-decode contract")
    contract_sha256 = sha256_file(CONTRACT)
    for name, expected in contract["runner_source_sha256"].items():
        if sha256_file(HERE / name) != expected:
            raise ValueError("runner source changed after contract freeze: " + name)
    checked_forward_for_decode(contract)
    canonical_runner = ROOT / "src/resectionlab/scan_diagnostic_runner.py"
    if (not canonical_runner.is_file() or canonical_runner.is_symlink() or
            sha256_file(canonical_runner) != contract["decode_source_sha256"]["diagnostic_runner"]):
        raise ValueError("canonical Case4 decode source has not been promoted exactly")
    allowlist_path = contract.get("acquisition_allowlist_path")
    acquisition_allowlist = load_acquisition_allowlist(ROOT / allowlist_path if allowlist_path else None)
    if acquisition_allowlist is not None and acquisition_allowlist["manifest_sha256"] != contract["acquisition_allowlist_sha256"]:
        raise ValueError("acquisition allowlist changed after review")
    outdir = HERE / arm
    if outdir.exists():
        raise FileExistsError("arm output already exists; refusing retry")
    fast = FastDarwinSampler()
    try:
        accepted_preflight = preflight(fast, contract, acquisition_allowlist)
    except Exception as error:
        deferred_receipt(arm, {"accepted": False, "error_type": type(error).__name__,
                               "error": str(error), "child_started": False,
                               "timestamp_utc": utc_now()})
        raise SystemExit(2)
    if not accepted_preflight["accepted"]:
        deferred_receipt(arm, accepted_preflight)
        raise SystemExit(2)
    outdir.mkdir(mode=0o700)
    write_json_once_durable(outdir / "host-preflight.json", accepted_preflight)
    slow = BackgroundInventory(os.getpid(), contract["slow_inventory_interval_seconds"], acquisition_allowlist)
    slow.start()
    first_deadline = time.monotonic() + contract["slow_inventory_max_age_seconds"]
    while True:
        first, age, slow_errors = slow.status()
        if first is not None and age <= contract["slow_inventory_max_age_seconds"]:
            break
        if time.monotonic() > first_deadline:
            slow.close()
            write_json_once_durable(outdir / "no-child-failure.json", {
                "accepted_for_feasibility": False, "reason": "slow_inventory_not_ready",
                "errors": slow_errors, "child_started": False, "no_retry": True})
            raise SystemExit(2)
        time.sleep(0.02)
    if overlap(first) or slow.status()[2]:
        slow.close()
        write_json_once_durable(outdir / "no-child-failure.json", {
            "accepted_for_feasibility": False, "reason": "overlap_or_inventory_error_before_child",
            "child_started": False, "no_retry": True})
        raise SystemExit(2)
    env = os.environ.copy()
    env.update(PYTHONNOUSERSITE="1", CUDA_VISIBLE_DEVICES="",
               PYTORCH_ENABLE_MPS_FALLBACK="0", OMP_NUM_THREADS="2",
               MKL_NUM_THREADS="2", NNUNET_COMPILE="false",
               RESECTIONLAB_PAIR_ARM=arm)
    started = time.monotonic()
    log_path = outdir / "worker.log"
    with log_path.open("xb") as log:
        gate_read, gate_write = os.pipe()
        try:
            child = subprocess.Popen([str(RUNTIME), "-I", str(LAUNCH_GATE),
                                      str(gate_read), str(WORKER)],
                                     cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=subprocess.STDOUT,
                                     pass_fds=(gate_read,), start_new_session=True)
        except Exception as error:
            os.close(gate_read)
            os.close(gate_write)
            close_error = None
            try:
                slow.close()
            except Exception as problem:
                close_error = "%s: %s" % (type(problem).__name__, problem)
            write_json_once_durable(outdir / "no-child-failure.json", {
                "accepted_for_feasibility": False, "no_retry": True,
                "child_started": False, "worker_released": False,
                "reason": "gated_worker_spawn_failed",
                "error": "%s: %s" % (type(error).__name__, error),
                "slow_close_error": close_error,
                "contract_sha256": contract_sha256,
                "completed_at_utc": utc_now()})
            raise
        launch_identity = None
        pgid = None
        release_attempted = False
        try:
            os.close(gate_read)
            launch_identity, pgid = bind_launch_identity(child, fast)
            slow.set_child_pgid(pgid)
            pre_release_inventory(os.getpid(), pgid, acquisition_allowlist)
            if slow.status()[2]:
                raise RuntimeError("background inventory error before input worker release")
            before_release_reason = stop_reason(fast.host(), accepted_preflight["baseline"]["fast_host"], contract)
            if before_release_reason is not None:
                raise RuntimeError("host guard before input worker release: " + before_release_reason)
            write_json_once_durable(outdir / "launch-binding.json", {
                "accepted_for_feasibility": False, "status": "bound_not_yet_released",
                "pid": child.pid, "pgid": pgid,
                "kernel_start_identity": list(launch_identity),
                "contract_sha256": contract_sha256,
                "launch_gate_sha256": sha256_file(LAUNCH_GATE),
                "worker_sha256": sha256_file(WORKER),
                "completed_at_utc": utc_now()})
            release_attempted = True  # A failed write can have ambiguous delivery.
            if os.write(gate_write, b"G") != 1:
                raise IOError("worker release token was not written completely")
            os.close(gate_write)
        except Exception as error:
            if release_attempted:
                try:
                    os.close(gate_write)
                except OSError:
                    pass
                negative = emergency_after_closeout_failure(
                    child, fast, pgid, launch_identity, slow, outdir, error, contract_sha256,
                    reason="worker_release_attempt_failed_or_ambiguous")
                print(json.dumps(negative, sort_keys=True), flush=True)
                raise SystemExit(2)
            negative = abort_unreleased_gate(child, gate_write, fast, launch_identity, pgid,
                                             outdir, "launch_binding_or_release_failed",
                                             contract_sha256,
                                             "%s: %s" % (type(error).__name__, error))
            try:
                slow.close()
            except Exception as close_error:
                negative["slow_close_error"] = "%s: %s" % (type(close_error).__name__, close_error)
            print(json.dumps(negative, sort_keys=True), flush=True)
            raise SystemExit(2)

        fast_series = []
        slow_series = []
        slow_seen = None
        peak_rss = 0
        detached_seen = {}
        reason = None
        monitor_error = None
        previous_start = None
        phase = None
        baseline = accepted_preflight["baseline"]["fast_host"]
        try:
            while child.poll() is None:
                sample_start = time.monotonic()
                if previous_start is not None and sample_start - previous_start > contract["host_sample_max_interval_seconds"]:
                    reason = "fast_sample_gap_over_0_2_seconds"
                    break
                previous_start = sample_start
                sample = fast.sample(pgid, child.pid)
                for detached_pid, identity in sample["detached_descendant_start_identities"].items():
                    previous = detached_seen.setdefault(int(detached_pid), identity)
                    if tuple(previous) != tuple(identity):
                        raise RuntimeError("detached descendant PID start identity changed during run")
                if child.poll() is None and not sample["pids"]:
                    reason = "child_group_missing_during_run"
                    break
                latest, age, errors = slow.status()
                if errors:
                    reason = "slow_inventory_command_or_parse_failed"
                    break
                if latest is None or age is None or age > contract["slow_inventory_max_age_seconds"]:
                    reason = "slow_inventory_stale"
                    break
                if latest["sample_monotonic"] != slow_seen:
                    slow_seen = latest["sample_monotonic"]
                    slow_series.append(latest)
                phase = last_phase(log_path) or phase
                sample["timestamp_utc"] = utc_now()
                sample["elapsed_seconds"] = time.monotonic() - started
                sample["slow_inventory_age_seconds"] = age
                sample["slow_inventory_errors_so_far"] = len(errors)
                sample["last_flushed_phase"] = phase
                fast_series.append(sample)
                peak_rss = max(peak_rss, sample["process_group_resident_bytes"])
                reason = stop_reason(sample, baseline, contract)
                if reason is None and peak_rss > contract["process_group_rss_cap_kib"] * 1024:
                    reason = "process_group_rss_cap_exceeded"
                if reason is None and detached_seen:
                    reason = "detached_input_worker_descendant_forbidden"
                if reason is None and overlap(latest, pgid):
                    reason = "project_compute_overlap"
                if reason is None and time.monotonic() - started > contract["wall_time_cap_seconds"]:
                    reason = "wall_time_cap_exceeded"
                if reason is None and log_path.stat().st_size > contract["log_cap_bytes"]:
                    reason = "log_cap_exceeded"
                if reason is None and time.monotonic() - sample_start > contract["host_sample_max_interval_seconds"]:
                    reason = "fast_sample_duration_over_0_2_seconds"
                if reason:
                    break
                delay = contract["host_sample_target_interval_seconds"] - (time.monotonic() - sample_start)
                if delay > 0:
                    time.sleep(delay)
        except Exception as error:
            reason = "monitor_exception"
            monitor_error = "%s: %s" % (type(error).__name__, error)
        try:
            final_phase = last_phase(log_path) or phase
        except Exception as error:
            reason = "monitor_exception"
            monitor_error = "final_phase_%s: %s" % (type(error).__name__, error)
            final_phase = phase
        provisional = {
            "scope": "One unreviewed Case4 DEVELOPMENT input preparation; no model forward",
            "arm": arm, "process_group_id": pgid, "contract_sha256": contract_sha256,
            "watchdog_reason": reason, "monitor_error": monitor_error,
            "elapsed_seconds": time.monotonic() - started,
            "process_group_rss_cap_kib": contract["process_group_rss_cap_kib"],
            "sampled_peak_process_group_resident_bytes": peak_rss,
            "fast_series": fast_series, "slow_series": slow_series,
            "detached_descendant_pids_observed": sorted(detached_seen),
            "last_flushed_phase": final_phase,
            "wall_time_cap_seconds": contract["wall_time_cap_seconds"],
            "acquisition_allowlist_manifest_sha256": acquisition_allowlist["manifest_sha256"] if acquisition_allowlist else None,
            "memory_accounting": "sampled descendant-aware RSS, available percentage and swap are distinct"}
        try:
            report = closeout_supervision(
                child, fast, slow, pgid, launch_identity, outdir, provisional,
                baseline, contract,
                lambda target_pgid, allowlist: collect(os.getpid(), target_pgid, allowlist),
                stop_reason, lambda snapshot, target_pgid: bool(overlap(snapshot, target_pgid)),
                acquisition_allowlist=acquisition_allowlist,
                wait_seconds=5, detached_seen=detached_seen)
        except Exception as error:
            negative = emergency_after_closeout_failure(child, fast, pgid, launch_identity,
                                                        slow, outdir, error, contract_sha256,
                                                        detached_seen=detached_seen)
            print(json.dumps(negative, sort_keys=True), flush=True)
            raise SystemExit(1)
    print(json.dumps({key: report.get(key) for key in (
        "arm", "exit_code", "watchdog_reason", "post_guard_reason",
        "accepted_for_feasibility", "sampled_peak_process_group_resident_bytes",
        "elapsed_seconds")}, sort_keys=True), flush=True)
    if not report["accepted_for_feasibility"]:
        raise SystemExit(1)


if __name__ == "__main__":
    if (len(sys.argv) != 3 or sys.argv[1] != "case4" or
            not CONTRACT.is_file() or sha256_file(CONTRACT) != sys.argv[2]):
        raise SystemExit("require one Case4 arm and exact reviewed contract SHA-256")
    with (HERE / ".case4-diagnostic.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run_one(sys.argv[1])
