"""Candidate separate HBE v5 N12 I/O-policy launcher; no release is included.

Intended tracked location: launchers/hbe_v5_nocache_v1.py, with its two
siblings. This ignored source is for generated controls and independent review.
It does not alter the frozen HBE validator, solver, output inventory or caps.
"""
from __future__ import annotations

import argparse
import copy
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
from types import ModuleType

ROOT = Path(__file__).resolve().parents[1]  # Correct after copy to launchers/.
SOURCES = (
    "launchers/hbe_v5_nocache_v1.py",
    "launchers/hbe_v5_nocache_hash_v1.py",
    "launchers/hbe_v5_nocache_host_v1.py",
)
SIDECAR = ("outputs/mechanics/hbe-v5-nocache-policy-v1/"
           "08-tension-N12-S60-reference/attempt-01")
NATIVE_OUTPUT = ("outputs/mechanics/hbe-v5-remaining-one-shot-v1/"
                 "08-tension-N12-S60-reference/attempt-01")
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
HEX40 = re.compile(r"[0-9a-f]{40}\Z")
POLICY = {
    "io_policy": "darwin_F_NOCACHE_48_read_only_predecessor_hash_v1",
    "minimum_hint_bytes": 16 * 1024**2,
    "predecessor_ordinals": list(range(8)),
    "expected_preflight_hint_opens": 14,
    "expected_preflight_hint_bytes": 2801621755,
    "expected_completed_hint_opens": 28,
    "expected_completed_hint_bytes": 5603243510,
    "initial_available_percent_floor": 55,
    "pre_native_available_percent_floor": 45,
    "normal_kernel_pressure_mask": 1,
    "sidecar_output_cap_bytes": 1024**2,
    "extension_before_old_execute_wall_seconds": 45,
    "extension_after_old_execute_wall_seconds": 45,
    "extension_self_peak_rss_bytes": 3 * 1024**3,
    "owned_stage_cleanup_seconds": 3,
    "finalization_prep_reserve_seconds": 1,
}


def _self_peak_rss_bytes() -> int:
    if sys.platform != "darwin":
        raise RuntimeError("Versioned F_NOCACHE launcher requires macOS")
    # Darwin reports ru_maxrss in bytes; this is a self-process high-water mark.
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


class OwnedStage:
    """Retain the exact child handle even if the old group cleanup errors."""

    def __init__(self):
        self.process = None

    def __call__(self, *args, **kwargs):
        if self.process is not None:
            raise RuntimeError("Second process in one old stage forbidden")
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
                raise RuntimeError("Owned-stage cleanup deadline expired")
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
            # Never signal cached group-member PIDs; they may have been reused.
        try:
            record["exit_code"] = process.wait(timeout=remaining(2.))
            record["direct_child_reaped"] = True
        except Exception as error:
            record["errors"].append("reap:" + type(error).__name__)
        try:
            _, record["remaining_members"] = observer(
                process.pid, timeout_seconds=remaining(.5))
            record["contained"] = (record["direct_child_reaped"]
                                   and not record["remaining_members"])
        except Exception as error:
            record["errors"].append("final_observer:" + type(error).__name__)
        return record


def _local(root: Path, relative: str) -> Path:
    path = Path(relative)
    if not relative or path.is_absolute() or ".." in path.parts:
        raise ValueError("Repository-relative path required")
    current = root.resolve()
    for part in path.parts:
        current /= part
        if current.is_symlink():
            raise ValueError("Symlinked extension path refused")
    result = (root.resolve() / path).resolve()
    if not result.is_relative_to(root.resolve()):
        raise ValueError("Extension path escapes repository")
    return result


def _git(root: Path, *arguments: str) -> bytes:
    env = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    env.update(GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL="/dev/null",
               GIT_CONFIG_SYSTEM="/dev/null", GIT_OPTIONAL_LOCKS="0")
    return subprocess.run(["/usr/bin/git", *arguments], cwd=root, env=env,
                          capture_output=True, check=True, timeout=10).stdout


def verify_extension_sources(root: Path, commit: str, bindings: dict,
                             *, require_head: bool) -> dict:
    """Bind all new executing bytes to the same commit as old HBE sources."""
    if not isinstance(commit, str) or not HEX40.fullmatch(commit):
        raise ValueError("Full extension source commit required")
    if not isinstance(bindings, dict) or set(bindings) != set(SOURCES):
        raise ValueError("Exact three-file extension source closure required")
    head = _git(root, "rev-parse", "HEAD").decode().strip()
    if require_head and head != commit:
        raise ValueError("Checkout differs from released source commit")
    for relative in SOURCES:
        expected = bindings[relative]
        if not isinstance(expected, str) or not HEX64.fullmatch(expected):
            raise ValueError("Full extension source SHA-256 required")
        path = _local(root, relative)
        before = path.lstat()
        if not stat.S_ISREG(before.st_mode) or before.st_size > 128 * 1024:
            raise ValueError("Extension source absent, special or oversized")
        raw = path.read_bytes()
        after = path.lstat()
        identity = lambda x: (x.st_dev, x.st_ino, x.st_size, x.st_mtime_ns)
        if identity(before) != identity(after) or hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError("Extension source changed or hash differs")
        name = f"{commit}:{relative}"
        committed_size = int(_git(root, "cat-file", "-s", name))
        if committed_size != len(raw) or committed_size > 128 * 1024:
            raise ValueError("Committed extension source size differs")
        if _git(root, "cat-file", "blob", name) != raw:
            raise ValueError("Committed extension source bytes differ")
    return {"source_commit": commit, "observed_head": head,
            "source_hashes": dict(bindings)}


def _load_sibling(name: str, filename: str, expected_sha: str):
    path = Path(__file__).absolute().with_name(filename)
    before = path.lstat()
    if (not stat.S_ISREG(before.st_mode) or path.is_symlink()
            or before.st_size > 128 * 1024):
        raise RuntimeError("Bound extension helper is absent, linked or oversized")
    raw = path.read_bytes()
    after = path.lstat()
    identity = lambda x: (x.st_dev, x.st_ino, x.st_size, x.st_mtime_ns)
    if (identity(before) != identity(after)
            or hashlib.sha256(raw).hexdigest() != expected_sha):
        raise RuntimeError("Bound extension helper changed before import")
    module = ModuleType(name)
    module.__file__ = str(path)
    exec(compile(raw, str(path), "exec"), module.__dict__)
    return module


def _read_exact_release(root: Path, envelope_path: Path):
    from scripts import mechanics_hbe_v5_n8_one_shot as io
    if Path(__file__).resolve() != _local(root, SOURCES[0]):
        raise ValueError("Versioned launcher must execute from its bound tracked path")
    raw, identity = io.read_release(envelope_path)
    envelope = json.loads(raw)
    keys = {"schema", "status", "source_commit", "extension_source_bindings",
            "inner_release", "sidecar_directory", "policy"}
    if (not isinstance(envelope, dict) or set(envelope) != keys
            or envelope["schema"] != "hbe-v5-nocache-launch-envelope-v1"
            or envelope["status"] != "root_released_one_native_call_with_declared_io_policy"
            or envelope["sidecar_directory"] != SIDECAR
            or envelope["policy"] != POLICY):
        raise ValueError("Exact separately released N12 no-cache policy required")
    commit = envelope["source_commit"]
    before_sources = verify_extension_sources(
        root, commit, envelope["extension_source_bindings"], require_head=True)
    binding = envelope["inner_release"]
    if (not isinstance(binding, dict) or set(binding) != {"path", "sha256"}
            or not isinstance(binding["sha256"], str)
            or not HEX64.fullmatch(binding["sha256"])):
        raise ValueError("Exact old one-row release binding required")
    inner_path = _local(root, binding["path"])
    inner_raw, inner_identity = io.read_release(inner_path)
    if hashlib.sha256(inner_raw).hexdigest() != binding["sha256"]:
        raise ValueError("Old one-row release bytes differ")
    inner = json.loads(inner_raw)
    if (inner.get("ordinal") != 8 or inner.get("run_id") != "tension:N12:S60:reference"
            or inner.get("status") != "root_released_one_native_call"
            or inner.get("source_commit") != commit):
        raise ValueError("Old one-row release identity differs")
    return (envelope, raw, identity, before_sources,
            inner_path, inner_raw, inner_identity)


def _write_sidecar(directory: Path, receipt: dict) -> None:
    from scripts import mechanics_hbe_v5_n8_one_shot as io
    encoded = (json.dumps(receipt, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    if len(encoded) > POLICY["sidecar_output_cap_bytes"] - 4096:
        raise ValueError("Policy sidecar exceeds one MiB reserved output cap")
    io.durable_json(directory / "receipt.json", receipt)


def _bind_native_receipt(root: Path, record: dict) -> dict | None:
    """Bind any saved old receipt even on an extension failure path."""
    from scripts import mechanics_hbe_v5_n8_one_shot as io
    path = _local(root, NATIVE_OUTPUT + "/receipt.json")
    if not path.exists() and not path.is_symlink():
        return None
    raw, _ = io.read_release(path)  # Exact stat-checked read, one MiB maximum.
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("Old native receipt must be a JSON object")
    record["native_receipt_sha256"] = hashlib.sha256(raw).hexdigest()
    record["native_receipt_status"] = value.get("status")
    record["native_calls_attempted"] = value.get("native_calls_attempted")
    record["hbe_readout_calls_attempted"] = value.get("readout_calls_attempted")
    return value


def _require_matching_native_receipt(saved: dict | None, returned: dict) -> None:
    if saved is None or saved != returned:
        raise ValueError("Old returned receipt differs from saved native receipt")


def _rebind_native_receipt_stably(root: Path, record: dict) -> dict | None:
    first = record.get("native_receipt_sha256")
    saved = _bind_native_receipt(root, record)
    if first is not None and (saved is None or record.get("native_receipt_sha256") != first):
        record["native_receipt_sha256_first_binding"] = first
        raise ValueError("Old native receipt changed or disappeared after first extension binding")
    return saved


def _charge_full_preparation(remaining, root: Path, record: dict,
                             extension_started: float, native_receipt: dict) -> None:
    """Charge wrapper work and the reserved sidecar to frozen aggregate caps."""
    previous = record["old_validation_identity"]["previous"]
    native = native_receipt.get("native_stage", {}).get("elapsed_seconds", 0.)
    readout = native_receipt.get("readout_stage", {}).get("elapsed_seconds", 0.)
    if any(not isinstance(x, (int, float)) or x < 0 for x in (native, readout)):
        raise ValueError("Saved native/readout stage timing is invalid")
    total = time.monotonic() - extension_started
    prep = max(0., total - native - readout) + POLICY["finalization_prep_reserve_seconds"]
    if prep >= remaining.PREP_WALL:
        raise ValueError("Launcher-inclusive preparation wall cap exhausted")
    output = remaining.active_bytes(
        _local(root, NATIVE_OUTPUT), remaining.caps(8)["active_output_bytes"])
    charged_output = output + POLICY["sidecar_output_cap_bytes"]
    ledger = remaining._check_aggregate(
        previous, native, readout, prep, charged_output,
        native_receipt.get("native_calls_attempted", 0))
    record["extension_resource_charge"] = {
        "launcher_elapsed_seconds_at_check": total,
        "native_stage_seconds": native, "readout_stage_seconds": readout,
        "launcher_inclusive_prep_seconds_with_finalization_reserve": prep,
        "per_row_prep_wall_seconds": remaining.PREP_WALL,
        "native_closed_output_bytes_at_check": output,
        "sidecar_reserved_output_bytes": POLICY["sidecar_output_cap_bytes"],
        "charged_current_output_bytes": charged_output,
        "aggregate_with_extension": ledger,
        "interpretation": "One-second finalization reserve and full one-MiB sidecar are conservatively charged."}


def _actual_final_preparation(record: dict, extension_started: float) -> float:
    """Evaluate final persisted work against the prior charged reserve."""
    charge = record["extension_resource_charge"]
    spent = (time.monotonic() - extension_started
             - charge["native_stage_seconds"] - charge["readout_stage_seconds"])
    if (spent > charge["launcher_inclusive_prep_seconds_with_finalization_reserve"]
            or spent >= charge["per_row_prep_wall_seconds"]):
        raise ValueError("Final sidecar write exceeded charged preparation reserve")
    return spent


def _check_pre_native_charge(remaining, root: Path, record: dict,
                             extension_started: float, phase: str) -> None:
    """Reject an inclusive preparation overrun before the old native call."""
    previous = record["old_validation_identity"]["previous"]
    elapsed = time.monotonic() - extension_started
    prep = elapsed + POLICY["finalization_prep_reserve_seconds"]
    if prep >= remaining.PREP_WALL:
        raise ValueError("Launcher-inclusive preparation exhausted before native " + phase)
    directory = _local(root, NATIVE_OUTPUT)
    output = (remaining.active_bytes(directory, remaining.caps(8)["active_output_bytes"])
              if directory.exists() else 0)
    charged_output = output + POLICY["sidecar_output_cap_bytes"]
    ledger = remaining._check_aggregate(
        previous, 0., 0., prep, charged_output, 1)
    record.setdefault("pre_native_resource_checks", []).append({
        "phase": phase, "launcher_elapsed_seconds": elapsed,
        "launcher_inclusive_prep_seconds_with_finalization_reserve": prep,
        "native_output_bytes_at_check": output,
        "sidecar_reserved_output_bytes": POLICY["sidecar_output_cap_bytes"],
        "charged_current_output_bytes": charged_output,
        "aggregate_with_extension": ledger})


def scoped_execute(remaining, hint, allowed_directories, old_execute,
                   preflight_check, *, audit=None, native_stage_check=None,
                   stage_cleanup_callback=None):
    """Hint each original predecessor call, including the post-native recheck."""
    original_chain = remaining.validate_prior_chain
    original_validate = remaining.validate_release
    original_hash = remaining.io.file_hash
    original_supervise = (remaining.supervise_stage
                          if native_stage_check is not None else None)
    if audit is None:
        audit = {"eligible_bytes": 0, "eligible_files": 0,
                 "hinted_file_opens": 0, "hinted_file_paths": []}
    depth = 0
    validation_count = 0

    def hinted_chain(*args, **kwargs):
        nonlocal depth
        if depth:
            return original_chain(*args, **kwargs)

        def selected_hash(path, *, maximum=128 * 1024**2):
            return hint.hinted_hash(
                path, maximum=maximum, original_hash=original_hash,
                allowed_directories=allowed_directories, audit=audit)

        previous_hash = remaining.io.file_hash
        depth += 1
        remaining.io.file_hash = selected_hash
        try:
            return original_chain(*args, **kwargs)
        finally:
            remaining.io.file_hash = previous_hash
            depth -= 1

    def checked_validate(*args, **kwargs):
        nonlocal validation_count
        try:
            context = original_validate(*args, **kwargs)
            validation_count += 1
            if (audit["eligible_files"] != POLICY["expected_preflight_hint_opens"]
                    or audit["hinted_file_opens"] != audit["eligible_files"]
                    or audit["eligible_bytes"] != POLICY["expected_preflight_hint_bytes"]
                    or remaining.io.file_hash is not original_hash):
                raise ValueError("Complete declared preflight hint coverage required")
            preflight_check(context, audit)
            return context
        finally:
            # Old execute resumes its own code before output reservation/native.
            remaining.validate_release = original_validate

    def checked_supervise(stage, *args, **kwargs):
        if stage == "native":
            native_stage_check()
        if kwargs.get("popen") is not None:
            raise ValueError("Old stage cannot override bound owned-child launcher")
        from scripts import febio_runtime
        owner = OwnedStage()
        outcome = None
        stage_error = None
        try:
            outcome = original_supervise(stage, *args, **dict(kwargs, popen=owner))
        except BaseException as error:
            stage_error = error
        try:
            cleanup = owner.cleanup(
                febio_runtime.process_group_rss,
                time.monotonic() + POLICY["owned_stage_cleanup_seconds"])
        except BaseException as error:
            cleanup = {"contained": False, "direct_child_reaped": False,
                       "fallback_used": True, "remaining_members": [],
                       "errors": ["cleanup_exception:" + type(error).__name__]}
        if stage_cleanup_callback is not None:
            stage_cleanup_callback(stage, cleanup)
        if (not cleanup["contained"] or not cleanup["direct_child_reaped"]
                or cleanup["fallback_used"] or cleanup["errors"]):
            raise RuntimeError("Owned " + stage + " child cleanup incomplete")
        if stage_error is not None:
            raise stage_error
        return outcome

    remaining.validate_prior_chain = hinted_chain
    remaining.validate_release = checked_validate
    if native_stage_check is not None:
        remaining.supervise_stage = checked_supervise
    try:
        result = old_execute()
        if validation_count != 1 or remaining.io.file_hash is not original_hash:
            raise ValueError("One fully restored old validation required")
        return result, audit
    finally:
        remaining.validate_prior_chain = original_chain
        remaining.validate_release = original_validate
        remaining.io.file_hash = original_hash
        if original_supervise is not None:
            remaining.supervise_stage = original_supervise


def execute_envelope(envelope_path: Path, *, root: Path = ROOT) -> dict:
    """One-use prospective launcher; requires a separate reviewed envelope."""
    extension_started = time.monotonic()
    from scripts import mechanics_hbe_v5_n8_one_shot as io
    (envelope, raw, identity, before_sources,
     inner_path, inner_raw, inner_identity) = _read_exact_release(root, envelope_path)
    hint = _load_sibling("hbe_v5_nocache_bound_hint_v1",
                         "hbe_v5_nocache_hash_v1.py",
                         envelope["extension_source_bindings"][SOURCES[1]])
    host_module = _load_sibling("hbe_v5_nocache_bound_host_v1",
                                "hbe_v5_nocache_host_v1.py",
                                envelope["extension_source_bindings"][SOURCES[2]])
    host_sampler = host_module.FastDarwinSampler()
    sidecar = _local(root, SIDECAR)
    if sidecar.exists() or sidecar.is_symlink():
        raise ValueError("No-cache policy sidecar is one-use")
    native_output = _local(root, NATIVE_OUTPUT)
    if native_output.exists() or native_output.is_symlink():
        raise ValueError("Frozen native attempt directory is already consumed")
    sidecar.mkdir(parents=True, exist_ok=False)
    record = {
        "schema": "hbe-v5-nocache-policy-sidecar-v1", "status": "preflight_pending",
        "source_commit": envelope["source_commit"],
        "envelope_sha256": hashlib.sha256(raw).hexdigest(),
        "inner_release_sha256": hashlib.sha256(inner_raw).hexdigest(),
        "extension_sources_before": before_sources,
        "policy": POLICY, "native_calls_authorized": 1,
        "native_calls_attempted": 0, "hbe_readout_calls_attempted": 0,
        "native_output_directory": NATIVE_OUTPUT,
        "native_receipt_sha256": None,
        "scientific_interpretation": "Specimen numerical software only; no measured force or patient validation."}
    _write_sidecar(sidecar, record)
    audit = {"eligible_bytes": 0, "eligible_files": 0,
             "hinted_file_opens": 0, "hinted_file_paths": []}
    try:
        host_before = host_sampler.host()
        record["host_before_validation"] = host_before
        if (host_before["kernel_pressure_mask"] != 1
                or host_before["available_percent"] < 55):
            raise ValueError("Initial host below declared 55%/normal preflight floor")
        from scripts import mechanics_hbe_v5_remaining_one_shot as remaining
        if remaining.output_directory(8) != NATIVE_OUTPUT:
            raise ValueError("Frozen old native output directory differs")
        allowed = [_local(root, str(Path(remaining.receipt_path(i)).parent))
                   for i in POLICY["predecessor_ordinals"]]

        def preflight_check(context, current_audit):
            record["preflight_hint_audit"] = copy.deepcopy(current_audit)
            record["old_validation_identity"] = {
                "run_id": context["run_id"], "index": context["index"],
                "adapted_deck_sha256": hashlib.sha256(context["deck"]).hexdigest(),
                "adapter_receipt": context["adapter_receipt"],
                "previous": context["previous"],
                "source_hashes": context["source_hashes"],
                "observed_head_preflight": context["observed_head_preflight"]}
            host_after = host_sampler.host()
            record["host_before_native_reservation"] = host_after
            if (host_after["kernel_pressure_mask"] != 1
                    or host_after["available_percent"] < 45):
                raise ValueError("Host below unchanged 45%/normal native launch floor")
            verify_extension_sources(root, envelope["source_commit"],
                                     envelope["extension_source_bindings"],
                                     require_head=True)
            if (io.read_release(envelope_path) != (raw, identity)
                    or io.read_release(inner_path) != (inner_raw, inner_identity)):
                raise ValueError("Policy or old release changed before native reservation")
            _check_pre_native_charge(remaining, root, record, extension_started,
                                     "reservation")
            record["status"] = "preflight_passed_before_native_reservation"
            _write_sidecar(sidecar, record)

        def native_stage_check():
            host_at_launch = host_sampler.host()
            record["host_immediately_before_native_supervision"] = host_at_launch
            if (host_at_launch["kernel_pressure_mask"] != 1
                    or host_at_launch["available_percent"] < 45):
                raise ValueError("Host below unchanged 45%/normal native supervision floor")
            _check_pre_native_charge(remaining, root, record, extension_started,
                                     "supervision")
            _write_sidecar(sidecar, record)

        def stage_cleanup_callback(stage, cleanup):
            record.setdefault("owned_stage_cleanup", {})[stage] = cleanup
            _write_sidecar(sidecar, record)

        record["extension_before_old_execute_wall_seconds"] = (
            time.monotonic() - extension_started)
        record["extension_self_peak_rss_bytes_before"] = _self_peak_rss_bytes()
        if (record["extension_before_old_execute_wall_seconds"] >=
                POLICY["extension_before_old_execute_wall_seconds"]
                or record["extension_self_peak_rss_bytes_before"] >
                   POLICY["extension_self_peak_rss_bytes"]):
            raise ValueError("Extension pre-execute wall or self RSS cap exhausted")

        old_execute_finished = None
        record["native_calls_attempted"] = None
        record["hbe_readout_calls_attempted"] = None
        native_result, audit = scoped_execute(
            remaining, hint, allowed, lambda: remaining.execute(inner_path, root=root),
            preflight_check, audit=audit, native_stage_check=native_stage_check,
            stage_cleanup_callback=stage_cleanup_callback)
        old_execute_finished = time.monotonic()
        record["final_hint_audit"] = audit
        native_receipt = _bind_native_receipt(root, record)
        _require_matching_native_receipt(native_receipt, native_result)
        record["extension_sources_after"] = verify_extension_sources(
            root, envelope["source_commit"], envelope["extension_source_bindings"],
            require_head=False)
        if (io.read_release(envelope_path) != (raw, identity)
                or io.read_release(inner_path) != (inner_raw, inner_identity)):
            raise ValueError("Policy or old release changed after execution")
        record["extension_after_old_execute_wall_seconds"] = (
            time.monotonic() - old_execute_finished)
        record["extension_self_peak_rss_bytes_after"] = _self_peak_rss_bytes()
        if (record["extension_after_old_execute_wall_seconds"] >=
                POLICY["extension_after_old_execute_wall_seconds"]
                or record["extension_self_peak_rss_bytes_after"] >
                   POLICY["extension_self_peak_rss_bytes"]):
            raise ValueError("Extension post-execute wall or self RSS cap exhausted")
        _charge_full_preparation(remaining, root, record, extension_started,
                                 native_receipt)
        if (native_result.get("status") != "passed_numerical_software_only"
                or audit["eligible_files"] != POLICY["expected_completed_hint_opens"]
                or audit["hinted_file_opens"] != audit["eligible_files"]
                or audit["eligible_bytes"] != POLICY["expected_completed_hint_bytes"]
                or record["native_receipt_sha256"] is None):
            raise ValueError("Native result or complete post-native hint coverage failed")
        record["status"] = "passed_numerical_software_only_with_declared_io_policy"
    except BaseException as error:
        record["status"] = "failed_or_incomplete"
        record["failure"] = {"type": type(error).__name__, "message": str(error)[:500]}
    finally:
        if "final_hint_audit" not in record:
            record["final_hint_audit"] = audit
        try:
            record["extension_self_peak_rss_bytes_terminal"] = _self_peak_rss_bytes()
            if record["extension_self_peak_rss_bytes_terminal"] > POLICY["extension_self_peak_rss_bytes"]:
                raise ValueError("Extension self RSS cap exceeded")
        except BaseException as error:
            record["status"] = "failed_or_incomplete"
            record["terminal_resource_failure"] = type(error).__name__
        try:
            saved_receipt = _rebind_native_receipt_stably(root, record)
            if saved_receipt is not None and "old_validation_identity" in record:
                from scripts import mechanics_hbe_v5_remaining_one_shot as remaining
                _charge_full_preparation(remaining, root, record,
                                         extension_started, saved_receipt)
        except BaseException as error:
            record["status"] = "failed_or_incomplete"
            record["native_receipt_or_resource_binding_failure"] = {
                "type": type(error).__name__, "message": str(error)[:300]}
        if (record["native_calls_attempted"] is None
                and not native_output.exists()):
            record["native_calls_attempted"] = 0
            record["hbe_readout_calls_attempted"] = 0
        record["launcher_elapsed_seconds_terminal"] = time.monotonic() - extension_started
        _write_sidecar(sidecar, record)
        # Final durable writing is also preparation work.  A reserve overrun
        # invalidates a provisional success rather than silently spending it.
        charge = record.get("extension_resource_charge")
        if charge is not None:
            try:
                _actual_final_preparation(record, extension_started)
            except ValueError as error:
                record["status"] = "failed_or_incomplete"
                record["terminal_finalization_failure"] = {
                    "type": "ValueError",
                    "message": str(error)}
                record["launcher_elapsed_seconds_terminal"] = time.monotonic() - extension_started
                _write_sidecar(sidecar, record)
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--envelope", type=Path)
    args = parser.parse_args()
    if not args.execute or args.envelope is None:
        raise ValueError("Explicit --execute and separately reviewed policy envelope required")
    result = execute_envelope(args.envelope)
    print(json.dumps({"status": result["status"], "sidecar_directory": SIDECAR}))
    return 0 if result["status"] == "passed_numerical_software_only_with_declared_io_policy" else 1


if __name__ == "__main__":
    raise SystemExit(main())
