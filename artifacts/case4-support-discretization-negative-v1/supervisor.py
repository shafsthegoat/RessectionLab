"""One bounded saved-scan support-map replay; no model or planner execution."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
RELEASE = HERE / "support-map-release.json"
RECORD = HERE / "support-map-supervision-result.json"
LOG = HERE / "support-map-worker.log"
NEGATIVE = HERE / "support-map-provisional-negative.json"
EXPECTED_GENERATOR = "build/scan-target-estimator-research/case4-support-contract-v1/make_support_map.py"
EXPECTED_PREP_RESULT = "outputs/scan-target/resect-case4-prep-v1/attempt-01/preparation-result.json"
EXPECTED_MANIFEST = "manifests/experiments/resect-case4-scan-diagnostic-v1.json"
EXPECTED_SUPERVISOR = "build/scan-target-estimator-research/case4-support-cleanup-v2/support_map_supervise.py"
EXPECTED_IGNORED_HELPER = "build/scan-target-estimator-research/case4-support-contract-v1/support_contract.py"
EXPECTED_TRACKED_HELPER = "src/resectionlab/scan_support_contract.py"
EXPECTED_PYTHON = ".tools/scan-target-runtime/venv/bin/python"
EXPECTED_RUNTIME_LOCK = ".tools/scan-target-runtime/requirements-lock.txt"
EXPECTED_SAMPLER = "build/limited-input-guard-design/gliomoda-mps-gate-a-v3/darwin_fast_sampler.py"
EXPECTED_REVIEWED_HELPERS = "build/scan-target-estimator-research/case4-prep-v1/supervise.py"
EXPECTED_SOURCE_HASHES = {
    "generator_sha256": "82e205d2c47b46d0d65e15055a73ec15ba65f0010352fb236597623ccf2ddd6d",
    "identical_helper_sha256": "3c7a9bbce5042cc9ca0ab5c51086045ddebc4590ccbc508160681bc4111f335a",
    "source_preparation_result_sha256": "92fdc93f1240c81ac0ecbd7910a8256a18693480bd4bed48b19e109fe308d72c",
    "source_manifest_sha256": "d37948a4dc7acd16bdfac0244b926cfda5c684c60da1f6e4dfccb09ddcf21b26",
    "isolated_runtime_lock_sha256": "f78bda5e88157d0974a81b53c08f79895841db4e48934aeb4058b08f31c5fde1",
    "darwin_fast_sampler_sha256": "2ff8f3e4d30ed0a8976fb01da99403ff5b2b2ffd6e750791fe7a954946d0a00f",
    "reviewed_supervisor_helpers_sha256": "81ae9f7b6aa8a0ed7f714a22ab0e3ca98d699077bfc2862a8aa18c9e34ce593b",
}
EXPECTED_RESOURCE = {
    "host_physical_ram_gib": 16,
    "exclusive_heavy_compute_window_required": True,
    "preflight_seconds": 30,
    "preflight_sample_interval_seconds": 5,
    "require_kernel_pressure_mask": 1,
    "minimum_preflight_available_percent": 45,
    "maximum_preflight_available_spread_points": 5,
    "minimum_runtime_available_percent": 30,
    "maximum_runtime_drop_from_baseline_points": 20,
    "maximum_swap_used_growth_mib": 128,
    "maximum_process_group_rss_gib": 3,
    "maximum_output_bytes": 268435456,
    "maximum_wall_seconds": 180,
    "runtime_sample_interval_seconds": 0.2,
}
EXPECTED_OUTPUTS = [
    "build/scan-target-estimator-research/case4-support-contract-v1/support-bits-atlas.nii.gz",
    "build/scan-target-estimator-research/case4-support-contract-v1/support-map-result.json",
    "build/scan-target-estimator-research/case4-support-contract-v1/discretization-hold.json",
]


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def exact(path: Path, expected: str) -> None:
    if not path.is_file() or sha(path) != expected:
        raise ValueError(f"released source changed or unavailable: {path}")


def reviewed_helpers(path: Path, expected: str):
    exact(path, expected)
    spec = importlib.util.spec_from_file_location("case4_reviewed_supervisor_helpers", path)
    if spec is None or spec.loader is None:
        raise ValueError("cannot load reviewed resource helpers")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def output_bytes(paths: list[Path]) -> int:
    return sum(path.stat().st_size for path in paths if path.is_file())


def child_environment(ambient: dict[str, str]) -> dict[str, str]:
    """Keep the pinned interpreter from importing ambient Python packages."""
    result = {key: value for key, value in ambient.items() if not key.startswith("PYTHON")}
    result["PYTHONNOUSERSITE"] = "1"
    result["PYTHONDONTWRITEBYTECODE"] = "1"
    result["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = "2"
    return result


def validate_release(release: dict) -> None:
    """Reject a self-consistent but substituted release before any child starts."""
    if (release.get("schema_version") != "case4-source-support-map-release-v1"
            or release.get("status") != "prepared_not_executed"
            or release.get("one_shot_no_automatic_retry") is not True
            or release.get("split_role") != "DEVELOPMENT"
            or release.get("run_id") != "resect-case4-source-support-map-v1-attempt-01"
            or release.get("generator_path") != EXPECTED_GENERATOR
            or release.get("supervisor_path") != EXPECTED_SUPERVISOR
            or release.get("source_preparation_result_path") != EXPECTED_PREP_RESULT
            or release.get("source_manifest_path") != EXPECTED_MANIFEST
            or release.get("ignored_helper_path") != EXPECTED_IGNORED_HELPER
            or release.get("tracked_helper_path") != EXPECTED_TRACKED_HELPER
            or release.get("isolated_python_path") != EXPECTED_PYTHON
            or release.get("isolated_runtime_lock_path") != EXPECTED_RUNTIME_LOCK
            or release.get("darwin_fast_sampler_path") != EXPECTED_SAMPLER
            or release.get("reviewed_supervisor_helpers_path") != EXPECTED_REVIEWED_HELPERS
            or release.get("output_paths") != EXPECTED_OUTPUTS
            or any(release.get(key) != value for key, value in EXPECTED_SOURCE_HASHES.items())
            or release.get("resource_protocol") != EXPECTED_RESOURCE
            or release.get("anatomical_qc_admitted") is not False
            or release.get("tumor_inference_admitted") is not False
            or release.get("planning_admitted") is not False
            or release.get("clinical_use_admitted") is not False):
        raise ValueError("not the frozen Case4 DEVELOPMENT source-support release")


def retain_live_sample_identities(proc, sampler, sample: dict,
                                  leader_identity: tuple[int, int] | None,
                                  owned_seen: dict[int, tuple[int, int]]) -> None:
    """Promote sampled group PIDs only while the original leader is verified live."""
    def leader_matches() -> bool:
        return (leader_identity is not None and proc.poll() is None
                and sampler.process_start_identity(proc.pid) == leader_identity)

    if not leader_matches():
        raise RuntimeError("original worker exited or changed before group PID ownership check")
    staged = {}
    for pid in sample["pids"]:
        identity = sampler.process_start_identity(pid)
        if identity is None:
            raise RuntimeError(f"observed owned PID {pid} start identity unavailable")
        identity = tuple(identity)
        if pid in owned_seen and owned_seen[pid] != identity:
            raise RuntimeError(f"observed owned PID {pid} start identity changed")
        staged[int(pid)] = identity
    if not leader_matches():
        raise RuntimeError("original worker exited during group PID ownership check")
    owned_seen.update(staged)


def safe_stop_owned_group(proc, sampler, detached_seen: dict[int, tuple[int, int]],
                          owned_seen: dict[int, tuple[int, int]],
                          leader_identity: tuple[int, int] | None) -> dict:
    """Best-effort, identity-checked stop that never hides an OS cleanup failure."""
    report = {"actions": [], "errors": [], "unresolved_owned_pids": None,
              "leader_still_alive": None, "verification_complete": False}

    def failure(stage: str, error: Exception) -> None:
        report["errors"].append({"stage": stage, "error_type": type(error).__name__,
                                 "detail": str(error)})

    def snapshot(stage: str):
        try:
            return sampler.group(proc.pid, proc.pid)
        except Exception as error:
            failure(stage, error)
            return None

    def individual_signals(sig: signal.Signals, current: dict | None) -> None:
        pids = set(detached_seen)
        if current is not None:
            pids.update(current.get("pids", []))
        if leader_identity is not None and proc.poll() is None:
            pids.add(proc.pid)  # Known Popen leader survives a failed group enumeration.
        for pid in sorted(pids):
            if pid == proc.pid:
                if proc.poll() is not None:
                    continue
                if leader_identity is None or owned_seen.get(pid) != leader_identity:
                    report["errors"].append({"stage": "worker_identity",
                                             "error_type": "IdentityUnavailable",
                                             "detail": "worker PID not signaled without launch identity"})
                    continue
                identity = leader_identity
            else:
                identity = detached_seen.get(pid)
            if identity is None:
                try:
                    if os.getpgid(pid) != proc.pid:
                        continue
                except ProcessLookupError:
                    continue
                except Exception as error:
                    failure(f"pid_{pid}_membership_or_identity", error)
                    continue
                identity = owned_seen.get(pid)
                if identity is None:
                    if proc.poll() is not None:
                        report["errors"].append({"stage": f"pid_{pid}_provenance",
                                                 "error_type": "UnobservedAfterLeaderExit",
                                                 "detail": "new PID in old PGID is not proof of ownership"})
                        continue
                    try:
                        identity = sampler.process_start_identity(pid)
                    except Exception as error:
                        failure(f"pid_{pid}_identity", error)
                        continue
                    if identity is not None:
                        owned_seen[pid] = identity
            if identity is None:
                report["errors"].append({"stage": f"pid_{pid}_identity",
                                         "error_type": "IdentityUnavailable",
                                         "detail": "PID not signaled without a start identity"})
                continue
            try:
                action = sampler.signal_if_same_process(pid, identity, sig)
                report["actions"].append({"signal": sig.name, **action})
                if not action.get("signaled") and action.get("error"):
                    report["errors"].append({"stage": f"pid_{pid}_signal_{sig.name}",
                                             "error_type": "SignalFailure",
                                             "detail": action["error"]})
            except Exception as error:
                failure(f"pid_{pid}_signal_{sig.name}", error)

    for sig, timeout in ((signal.SIGTERM, 2), (signal.SIGKILL, 5)):
        current = snapshot("snapshot_before_" + sig.name)
        individual_signals(sig, current)
        if proc.poll() is None:
            try:
                proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                pass
            except Exception as error:
                failure("worker_wait_" + sig.name, error)
        current = snapshot("snapshot_after_" + sig.name)
        try:
            alive = proc.poll() is None
        except Exception as error:
            failure("worker_poll_" + sig.name, error)
            alive = True
        if current is not None and not current.get("pids") and not alive:
            report["unresolved_owned_pids"] = []
            report["leader_still_alive"] = False
            report["verification_complete"] = True
            break
    if not report["verification_complete"]:
        final = snapshot("final_owned_group")
        report["unresolved_owned_pids"] = None if final is None else final.get("pids")
        try:
            report["leader_still_alive"] = proc.poll() is None
        except Exception as error:
            failure("final_worker_poll", error)
            report["leader_still_alive"] = None
        report["verification_complete"] = (final is not None
                                             and report["leader_still_alive"] is False
                                             and not report["unresolved_owned_pids"])
    return report


def record_cleanup(record: dict, cleanup: dict) -> None:
    record["stop_actions"].extend(cleanup["actions"])
    record["cleanup_errors"].extend(cleanup["errors"])
    record["unresolved_owned_group_pids"] = cleanup["unresolved_owned_pids"]
    record["worker_liveness_after_cleanup"] = cleanup["leader_still_alive"]
    record["cleanup_verification_complete"] = cleanup["verification_complete"]


def write_provisional_negative(record: dict, stage: str) -> None:
    """Save a no-clobber negative before cleanup can fail or be interrupted."""
    if record.get("provisional_negative_written"):
        return
    payload = {
        "schema_version": "case4-support-map-provisional-negative-v1",
        "status": "provisional_negative_cleanup_pending",
        "run_id": record["run_id"],
        "release_sha256": record["release_sha256"],
        "at_utc": record.get("last_event_at_utc"),
        "stage": stage,
        "stop_reason": record["stop_reason"],
        "worker_pid": record.get("worker_pid"),
        "worker_start_identity": record.get("worker_start_identity"),
        "observed_owned_start_identities": record.get("observed_owned_start_identities", {}),
        "observed_detached_start_identities": record.get("observed_detached_start_identities", {}),
        "samples_recorded": len(record["samples"]),
        "model_inference_performed": False,
        "planning_performed": False,
    }
    try:
        fd = os.open(NEGATIVE, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        record["provisional_negative_written"] = True
    except Exception as error:
        record["receipt_write_errors"].append({"stage": stage,
                                               "error_type": type(error).__name__,
                                               "detail": str(error)})


def result_status(record: dict, exit_code: int | None, *, map_exists: bool,
                  result_exists: bool, hold_exists: bool) -> str:
    if (exit_code == 0 and record["stop_reason"] is None
            and not record["cleanup_errors"]
            and not record["receipt_write_errors"]
            and not record.get("finalization_errors")
            and record["cleanup_verification_complete"] is True
            and record["unresolved_owned_group_pids"] == []
            and record["worker_liveness_after_cleanup"] is False
            and map_exists and result_exists and not hold_exists):
        return "completed_source_support_only_qc_pending"
    if (hold_exists and record["stop_reason"] is None
            and not record["cleanup_errors"] and not record["receipt_write_errors"]):
        return "discretization_hold"
    return "failed"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--release-sha256", required=True)
    args = parser.parse_args()
    if not args.execute:
        raise ValueError("explicit reviewed one-shot execution required")
    exact(RELEASE, args.release_sha256)
    release = json.loads(RELEASE.read_text())
    validate_release(release)
    if sys.platform != "darwin":
        raise RuntimeError("this reviewed host monitor is macOS-only")
    exact(Path(__file__), release["supervisor_sha256"])
    exact(ROOT / release["generator_path"], release["generator_sha256"])
    for key in ("ignored_helper_path", "tracked_helper_path"):
        exact(ROOT / release[key], release["identical_helper_sha256"])
    for path_key, sha_key in (
        ("isolated_runtime_lock_path", "isolated_runtime_lock_sha256"),
        ("source_preparation_result_path", "source_preparation_result_sha256"),
        ("source_manifest_path", "source_manifest_sha256"),
        ("darwin_fast_sampler_path", "darwin_fast_sampler_sha256"),
    ):
        exact(ROOT / release[path_key], release[sha_key])
    helper = reviewed_helpers(ROOT / release["reviewed_supervisor_helpers_path"],
                              release["reviewed_supervisor_helpers_sha256"])
    sampler = helper.sampler_from_exact_file(release["darwin_fast_sampler_path"],
                                              release["darwin_fast_sampler_sha256"])
    python = ROOT / release["isolated_python_path"]
    if not python.is_file():
        raise FileNotFoundError("isolated runtime is unavailable")
    paths = [ROOT / relative for relative in release["output_paths"]]
    if any(path.exists() for path in [*paths, RECORD, LOG, NEGATIVE]):
        raise FileExistsError("support-map attempt already exists; no overwrite or retry")
    rule = release["resource_protocol"]
    record = {
        "schema_version": "case4-support-map-supervision-v1",
        "run_id": release["run_id"],
        "release_sha256": args.release_sha256,
        "supervisor_sha256": sha(Path(__file__)),
        "generator_sha256": release["generator_sha256"],
        "started_at_utc": helper.now(),
        "status": "preflight", "preflight": [], "samples": [],
        "stop_reason": None, "stop_actions": [], "cleanup_errors": [],
        "receipt_write_errors": [], "finalization_errors": [],
        "provisional_negative_written": False,
        "observed_owned_start_identities": {},
        "observed_detached_start_identities": {},
        "cleanup_verification_complete": None,
        "unresolved_owned_group_pids": None,
        "worker_liveness_after_cleanup": None,
        "model_inference_performed": False,
        "planning_performed": False,
        "anatomical_qc_status": "unreviewed_estimated_mask",
    }
    helper.atomic_json(RECORD, record)
    try:
        count = int(rule["preflight_seconds"] / rule["preflight_sample_interval_seconds"])
        for index in range(count + 1):
            host = sampler.host()
            record["preflight"].append({"at_utc": helper.now(), **host})
            helper.atomic_json(RECORD, record)
            if (host["kernel_pressure_mask"] != rule["require_kernel_pressure_mask"]
                    or host["available_percent"] < rule["minimum_preflight_available_percent"]):
                raise RuntimeError("host_pressure_or_availability_preflight_hold")
            if index < count:
                time.sleep(rule["preflight_sample_interval_seconds"])
        preflight = record["preflight"]
        available = [sample["available_percent"] for sample in preflight]
        if (max(available) - min(available) > rule["maximum_preflight_available_spread_points"]
                or any(b["swap_used_bytes"] > a["swap_used_bytes"]
                       for a, b in zip(preflight, preflight[1:]))):
            raise RuntimeError("unstable_availability_or_rising_swap_preflight_hold")
    except Exception as error:
        record["status"] = "deferred_before_worker"
        record["stop_reason"] = f"preflight:{type(error).__name__}:{error}"
        record["finished_at_utc"] = helper.now()
        helper.atomic_json(RECORD, record)
        return 2
    baseline = record["preflight"][-1]
    record["status"] = "launched"
    helper.atomic_json(RECORD, record)
    proc = None
    leader_identity = None
    detached_seen: dict[int, tuple[int, int]] = {}
    owned_seen: dict[int, tuple[int, int]] = {}
    started = time.monotonic()
    try:
        environment = child_environment(os.environ)
        with LOG.open("x") as log:
            proc = subprocess.Popen(
                [str(python), "-B", str(ROOT / release["generator_path"])],
                cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True)
            record["worker_pid"] = proc.pid
            try:
                leader_identity = sampler.process_start_identity(proc.pid)
            except Exception as error:
                record["cleanup_errors"].append({"stage": "worker_start_identity",
                                                 "error_type": type(error).__name__,
                                                 "detail": str(error)})
            record["worker_start_identity"] = leader_identity
            if leader_identity is not None:
                owned_seen[proc.pid] = tuple(leader_identity)
                record["observed_owned_start_identities"][str(proc.pid)] = list(leader_identity)
            if leader_identity is None and proc.poll() is None:
                record["stop_reason"] = "worker_start_identity_unavailable"
                record["last_event_at_utc"] = helper.now()
                write_provisional_negative(record, "worker_start_identity")
                record_cleanup(record, safe_stop_owned_group(proc, sampler, detached_seen,
                                                             owned_seen, leader_identity))
            while proc.poll() is None:
                if record["stop_reason"] is not None:
                    break
                elapsed = time.monotonic() - started
                try:
                    sample = sampler.sample(proc.pid, proc.pid)
                    retain_live_sample_identities(proc, sampler, sample,
                                                  leader_identity, owned_seen)
                    for pid, identity in owned_seen.items():
                        record["observed_owned_start_identities"][str(pid)] = list(identity)
                    record["samples"].append({"at_utc": helper.now(),
                                               "elapsed_seconds": elapsed,
                                               **sample,
                                               "output_bytes": output_bytes([*paths, LOG])})
                    for pid, identity in sample["detached_descendant_start_identities"].items():
                        old = detached_seen.setdefault(int(pid), tuple(identity))
                        if tuple(old) != tuple(identity):
                            raise RuntimeError("detached descendant PID identity changed")
                        record["observed_detached_start_identities"][str(pid)] = list(identity)
                    stop = helper.prospective_checks(sample, baseline=baseline, rule=rule)
                    if detached_seen:
                        stop = "detached_descendant_forbidden"
                    if (sample["process_group_resident_bytes"]
                            > rule["maximum_process_group_rss_gib"] * (1 << 30)):
                        stop = "process_group_rss_cap_exceeded"
                    if output_bytes([*paths, LOG]) > rule["maximum_output_bytes"]:
                        stop = "output_bytes_cap_exceeded"
                except Exception as error:
                    stop = f"resource_monitor_failed:{type(error).__name__}:{error}"
                if elapsed > rule["maximum_wall_seconds"]:
                    stop = "wall_time_cap_exceeded"
                if stop:
                    record["stop_reason"] = stop
                    record["last_event_at_utc"] = helper.now()
                    write_provisional_negative(record, "runtime_stop")
                    record_cleanup(record, safe_stop_owned_group(proc, sampler, detached_seen,
                                                                 owned_seen, leader_identity))
                    break
                if len(record["samples"]) % 25 == 0:
                    helper.atomic_json(RECORD, record)
                time.sleep(rule["runtime_sample_interval_seconds"])
            try:
                remaining = sampler.group(proc.pid, proc.pid)
            except Exception as error:
                record["cleanup_errors"].append({"stage": "final_group_enumeration",
                                                 "error_type": type(error).__name__,
                                                 "detail": str(error)})
                remaining = None
            record["final_owned_group_pids"] = None if remaining is None else remaining["pids"]
            if remaining is None or remaining["pids"] or proc.poll() is None:
                if record["stop_reason"] is None:
                    record["stop_reason"] = "owned_group_or_worker_not_verified_exited"
                record["last_event_at_utc"] = helper.now()
                write_provisional_negative(record, "final_group_cleanup")
                record_cleanup(record, safe_stop_owned_group(proc, sampler, detached_seen,
                                                             owned_seen, leader_identity))
            elif record["cleanup_verification_complete"] is None:
                record["cleanup_verification_complete"] = True
                record["unresolved_owned_group_pids"] = []
                record["worker_liveness_after_cleanup"] = False
    except BaseException as error:
        record["supervisor_error"] = f"{type(error).__name__}:{error}"
        if record["stop_reason"] is None:
            record["stop_reason"] = "supervisor_exception"
        record["last_event_at_utc"] = helper.now()
        write_provisional_negative(record, "supervisor_exception")
        if proc is not None:
            record_cleanup(record, safe_stop_owned_group(proc, sampler, detached_seen,
                                                         owned_seen, leader_identity))
    finally:
        record["finished_at_utc"] = helper.now()
        try:
            record["worker_exit_code"] = proc.poll() if proc is not None else None
            record["output_bytes_final"] = output_bytes([*paths, LOG])
            record["output_sha256"] = {
                str(path.relative_to(ROOT)): sha(path) if path.is_file() else None
                for path in paths}
            result, hold = paths[1], paths[2]
            record["status"] = result_status(
                record, record["worker_exit_code"], map_exists=paths[0].is_file(),
                result_exists=result.is_file(), hold_exists=hold.is_file())
        except Exception as error:
            record["finalization_errors"].append({"stage": "final_evidence",
                                                  "error_type": type(error).__name__,
                                                  "detail": str(error)})
            record["stop_reason"] = record["stop_reason"] or "final_evidence_unavailable"
            record["status"] = "failed"
            record["last_event_at_utc"] = helper.now()
            write_provisional_negative(record, "final_evidence")
        try:
            helper.atomic_json(RECORD, record)
        except Exception as error:
            record["receipt_write_errors"].append({"stage": "final_record",
                                                   "error_type": type(error).__name__,
                                                   "detail": str(error)})
            record["stop_reason"] = record["stop_reason"] or "final_receipt_write_failed"
            record["status"] = "failed"
            record["last_event_at_utc"] = helper.now()
            write_provisional_negative(record, "final_record")
    return 0 if record["status"] == "completed_source_support_only_qc_pending" else 1


if __name__ == "__main__":
    raise SystemExit(main())
