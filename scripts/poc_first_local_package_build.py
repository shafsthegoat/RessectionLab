#!/usr/bin/env python3
"""Supervise the two existing local package commands; never install or retry.

ROOT releases each stage against the committed sources and fixed plan. Worker
logs and the cumulative file ledger stay in private outputs; artifact receipts
contain scalar observations. Limits are sampled stop enforcement, not hard caps.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import stat
import subprocess
import sys
import time

import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from real_intake_io import atomic_preserve, termination_cleanup

SCHEMA = "poc-first-local-package-supervisor-v1"
PLAN = ROOT / "artifacts/poc-first-local-package-preflight-v1/build-plan.json"
PLAN_SHA = "a7a3e918e519882e115714d0624eb4ed8e9f6952598cca505029dcbe0940a7a5"
HELPER_SHA = "496821360b1daee79459952f970d299856baacad75fcc57ee589e55115036fd8"
SCOPES = ("build/electron-sidecar", "build/poc-first-local-package-v1", "desktop/sidecar", "desktop/release")
RECOVERY_STAGE = "sidecar-recovery-01"
RECOVERY_SESSION = "outputs/poc-first-local-package-build-v1"
RECOVERY_PREDECESSOR = {
    "receipt_sha256": "5c03bcfffb832de6a907a6e3f94044c815cdd0cd345bb66ed14c08c9f9dd42ac",
    "growth_ledger_sha256": "8ade65f73cd9c411256272f4037abc34a052a579142022c0fa1e3a6f413f73d3",
    "baseline_sha256": "34e8f9cdafdfac30f30b5621c1d51cfb569d71d7764cac0e74f2b5b62f5923c3",
}


def encode(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def value_sha(value):
    return hashlib.sha256(encode(value)).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def publish(path, value):
    path = Path(path)
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    atomic_preserve(path, encode(value))
    path.chmod(0o600)


def plan():
    if sha(PLAN) != PLAN_SHA or sha(ROOT / "scripts/real_intake_io.py") != HELPER_SHA:
        raise ValueError("PLAN_OR_HELPER_CHANGED")
    return read(PLAN)


def source_manifest():
    paths = {*list((ROOT / "src/resectionlab").rglob("*.py")), *list(ROOT.glob("*lock*")),
             ROOT / "scripts/build_electron_sidecar.py", ROOT / "packaging/sidecar_entry.py",
             ROOT / "packaging/python_notices.py", ROOT / "pyproject.toml", ROOT / "packaging/RessectionLab.icns"}
    for folder in ("desktop/src", "desktop/electron"):
        paths.update(p for p in (ROOT / folder).rglob("*") if not p.is_dir())
    paths.update(ROOT / "desktop" / p for p in ("index.html", "vite.config.ts", "tsconfig.json", "package.json", "pnpm-lock.yaml"))
    if any(p.is_symlink() for p in paths):
        raise ValueError("SOURCE_SYMLINK_UNSUPPORTED")
    files = {str(p.relative_to(ROOT)): sha(p) for p in sorted(paths)}
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    return {"commit": commit, "files": files}


def effective_environment(stage, document):
    env = dict(stage["environment"])
    if stage["stage"] == "electron":
        env["RESECTIONLAB_ELECTRON_ZIP_DIR"] = str(Path(document["offline_build"]["existing_verified_archive"]).parent)
    elif "PATH" not in env:
        env["PATH"] = os.environ.get("PATH", "/usr/bin:/bin:/usr/sbin:/sbin")
    for tool in ("git", "codesign", "otool", "lipo"):
        if shutil.which(tool, path=env["PATH"]) is None:
            raise ValueError("SYSTEM_TOOL_UNAVAILABLE: " + tool)
    return env


def archive_fixity(document):
    expected = document["offline_build"]
    path = Path(expected["existing_verified_archive"])
    if path.is_symlink() or path.stat().st_size != expected["bytes"] or sha(path) != expected["sha256"]:
        raise ValueError("LOCAL_ARCHIVE_CHANGED_OR_MISSING")
    return expected["sha256"]


def archive_binding(document):
    item = document["offline_build"]
    return {key: item[key] for key in ("existing_verified_archive", "sha256", "bytes")}


def disk_inventory(roots):
    """Logical file/link lengths, with no traversal through symbolic links."""
    roots = [Path(p).absolute() for p in roots]
    if any(a == b or a in b.parents or b in a.parents for i, a in enumerate(roots) for b in roots[i + 1:]):
        raise ValueError("OVERLAPPING_DISK_SCOPES")
    result = {}
    def visit(path):
        try:
            info = path.lstat()
            if stat.S_ISLNK(info.st_mode) or stat.S_ISREG(info.st_mode):
                result[str(path)] = info.st_size
            elif stat.S_ISDIR(info.st_mode):
                for item in path.iterdir():
                    visit(item)
            else:
                raise ValueError("UNSUPPORTED_BUILD_FILE_TYPE")
        except FileNotFoundError:
            pass  # A concurrently removed temporary contributes no current bytes.
    for path in roots:
        visit(path)
    return result


def accumulate_growth(baseline, high_water, current):
    for path, size in current.items():
        growth = max(0, size - baseline.get(path, 0))
        if growth:
            high_water[path] = max(high_water.get(path, 0), growth)
    return sum(high_water.values())


def sample_processes(process, tracked):
    """Track process identities, not reusable PID numbers, across samples."""
    if process.pid not in tracked:
        try:
            tracked[process.pid] = psutil.Process(process.pid)
        except psutil.NoSuchProcess:
            pass
    parent = tracked.get(process.pid)
    if parent is not None and parent.is_running():
        try:
            for child in parent.children(recursive=True):
                tracked.setdefault(child.pid, child)
        except psutil.NoSuchProcess:
            pass
    family, live = 0, []
    for member in tracked.values():
        try:
            if member.is_running() and member.status() != psutil.STATUS_ZOMBIE:
                family += member.memory_info().rss
                live.append(member.pid)
        except psutil.NoSuchProcess:
            pass
    return {"rss_bytes": family + psutil.Process().memory_info().rss,
            "available_bytes": psutil.virtual_memory().available, "live_members": live}


def cleanup_owned(process, tracked):
    """Kill only the created session and observed descendants; always reap."""
    errors = []
    # A still-live process identity confirms ownership of its new process group.
    owned = [p for p in tracked.values() if p.is_running()]
    group_owned = False
    for member in owned:
        try:
            if os.getpgid(member.pid) == process.pid:
                group_owned = True
        except ProcessLookupError:
            pass
    if group_owned:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except OSError as error:
            errors.append(type(error).__name__)
    for member in owned:
        try:
            member.kill()
        except psutil.NoSuchProcess:
            pass
        except psutil.Error as error:
            errors.append(type(error).__name__)
    try:
        if process.poll() is None:
            process.kill()  # Popen still owns an unreaped child PID.
        code = process.wait(timeout=3)
        reaped = True
    except (OSError, subprocess.TimeoutExpired) as error:
        errors.append(type(error).__name__)
        code, reaped = None, False
    _, alive = psutil.wait_procs(owned, timeout=2)
    if any(p.is_running() and p.status() != psutil.STATUS_ZOMBIE for p in alive):
        errors.append("observed_descendant_survived")
    return code, reaped, errors


def supervise(command, cwd, env, log, roots, baseline, high_water, limits, observer=sample_processes):
    """Small sampled watchdog adapted from the tested Case3 process controls."""
    result = {"status": "supervisor_failed", "exit_code": None, "samples": 0,
              "peak_family_plus_supervisor_rss_bytes": 0, "minimum_available_memory_bytes": None,
              "minimum_free_disk_bytes": None, "cumulative_new_bytes": sum(high_water.values()),
              "maximum_sample_gap_seconds": 0., "worker_reaped": False,
              "sampled_stop_enforcement": True, "os_hard_caps": False}
    process, tracked = None, {}
    started = previous = time.monotonic()
    with termination_cleanup():
        try:
            # Prelaunch observations prevent starting below memory/disk reserves.
            current = disk_inventory(roots)
            result["cumulative_new_bytes"] = accumulate_growth(baseline, high_water, current)
            available = psutil.virtual_memory().available
            free = shutil.disk_usage(cwd).free
            result["minimum_available_memory_bytes"], result["minimum_free_disk_bytes"] = available, free
            if available < limits["available_bytes"] or free < limits["free_disk_bytes"] or result["cumulative_new_bytes"] > limits["new_bytes"]:
                result["status"] = "prelaunch_resource_stop"
                return result
            with log.open("xb") as stream:
                process = subprocess.Popen(command, cwd=cwd, env=env, stdout=stream, stderr=subprocess.STDOUT,
                                           start_new_session=True, umask=0o077)
                try:
                    tracked[process.pid] = psutil.Process(process.pid)
                except psutil.NoSuchProcess:
                    pass
                while True:
                    now = time.monotonic()
                    result["maximum_sample_gap_seconds"] = max(result["maximum_sample_gap_seconds"], now - previous)
                    previous = now
                    sample = observer(process, tracked)
                    result["samples"] += 1
                    result["peak_family_plus_supervisor_rss_bytes"] = max(result["peak_family_plus_supervisor_rss_bytes"], sample["rss_bytes"])
                    result["minimum_available_memory_bytes"] = min(result["minimum_available_memory_bytes"], sample["available_bytes"])
                    result["minimum_free_disk_bytes"] = min(result["minimum_free_disk_bytes"], shutil.disk_usage(cwd).free)
                    result["cumulative_new_bytes"] = accumulate_growth(baseline, high_water, disk_inventory(roots))
                    if sample["rss_bytes"] > limits["rss_bytes"]: result["status"] = "rss_stop"
                    elif sample["available_bytes"] < limits["available_bytes"]: result["status"] = "available_memory_stop"
                    elif result["minimum_free_disk_bytes"] < limits["free_disk_bytes"]: result["status"] = "disk_reserve_stop"
                    elif result["cumulative_new_bytes"] > limits["new_bytes"]: result["status"] = "cumulative_bytes_stop"
                    elif time.monotonic() - started >= limits["wall_seconds"]: result["status"] = "timeout"
                    elif process.poll() is not None:
                        result["status"] = ("descendants_outlived_worker" if any(p != process.pid for p in sample["live_members"])
                                            else "completed_unverified" if process.returncode == 0 else "worker_failed")
                    else:
                        time.sleep(limits["interval_seconds"])
                        continue
                    break
        except BaseException as error:
            result["status"], result["error_type"] = "supervisor_failed", type(error).__name__
        finally:
            if process is not None:
                try:
                    result["exit_code"], result["worker_reaped"], errors = cleanup_owned(process, tracked)
                except BaseException as error:
                    errors = [type(error).__name__]
                    # Preserve the observation even if process inspection fails.
                    try:
                        if process.poll() is None: process.kill()
                        result["exit_code"] = process.wait(timeout=3)
                        result["worker_reaped"] = True
                    except (OSError, subprocess.TimeoutExpired) as cleanup_error:
                        errors.append(type(cleanup_error).__name__)
                if errors:
                    result["original_status"] = result["status"]
                    result["status"], result["cleanup_errors"] = "cleanup_failed", errors
            try:
                result["cumulative_new_bytes"] = accumulate_growth(baseline, high_water, disk_inventory(roots))
                free = shutil.disk_usage(cwd).free
                result["minimum_free_disk_bytes"] = min(result["minimum_free_disk_bytes"] or free, free)
                if result["status"] == "completed_unverified" and (result["cumulative_new_bytes"] > limits["new_bytes"] or free < limits["free_disk_bytes"]):
                    result["status"] = "terminal_disk_stop"
            except Exception as error:
                result["original_status"], result["status"] = result["status"], "terminal_observation_failed"
                result["error_type"] = type(error).__name__
            result["elapsed_seconds"] = time.monotonic() - started
    return result


def describe():
    document = plan()
    source = source_manifest()
    return {"schema": SCHEMA + "-description", "plan_sha256": PLAN_SHA, "runner_sha256": sha(__file__),
            "source_manifest": source, "source_manifest_sha256": value_sha(source), "helper_sha256": HELPER_SHA,
            "program_sha256": {stage["stage"]: sha(stage["program"]) for stage in document["build_sequence"]},
            "environment_sha256": {stage["stage"]: value_sha(effective_environment(stage, document)) for stage in document["build_sequence"]},
            "environment_values_logged": False, "status": "advisory_until_root_source_commit_and_release"}


def predecessor_state(session, subdirectory, binding):
    """Read immutable accounting from one explicitly selected sidecar attempt."""
    if subdirectory not in ("sidecar", RECOVERY_STAGE):
        raise ValueError("SIDECAR_SUBDIRECTORY_NOT_ALLOWED")
    prior_path = session / subdirectory
    paths = {"receipt_sha256": prior_path / "receipt.json", "growth_ledger_sha256": prior_path / "growth-ledger.json",
             "baseline_sha256": session / "baseline.json"}
    if not isinstance(binding, dict) or set(binding) != set(paths) or any(sha(path) != binding[key] for key, path in paths.items()):
        raise ValueError("PREDECESSOR_ACCOUNTING_BINDING_CHANGED")
    prior, baseline, ledger = read(paths["receipt_sha256"]), read(paths["baseline_sha256"]), read(paths["growth_ledger_sha256"])
    if (prior.get("schema") != SCHEMA + "-receipt" or prior.get("stage") != subdirectory
            or prior.get("plan_sha256") != PLAN_SHA or prior.get("attempts") != 1
            or prior.get("baseline_sha256") != binding["baseline_sha256"]
            or prior.get("growth_ledger_sha256") != binding["growth_ledger_sha256"]
            or any(type(value) is not int or value < 0 for value in ledger.values())
            or sum(ledger.values()) != prior.get("cumulative_new_bytes")
            or prior.get("limits", {}).get("new_bytes") != 8 * 1024**3
            or baseline.get("roots") != [str(ROOT / name) for name in SCOPES]):
        raise ValueError("PREDECESSOR_ACCOUNTING_INVALID")
    return prior, baseline, ledger


def recovery_state(session, release):
    if (session.resolve() != (ROOT / RECOVERY_SESSION).resolve()
            or release.get("predecessor") != RECOVERY_PREDECESSOR):
        raise ValueError("EXACT_FIRST_FAILURE_RECOVERY_REQUIRED")
    prior, baseline, ledger = predecessor_state(session, "sidecar", release["predecessor"])
    if prior.get("status") != "worker_failed" or prior.get("exit_code") != 1 or prior.get("worker_reaped") is not True:
        raise ValueError("FAILED_REAPED_FIRST_ATTEMPT_REQUIRED")
    return prior, baseline, ledger


def preserved_outputs_ready():
    archive = ROOT / "build/poc-first-local-package-v1/failed-first-sidecar"
    if archive.is_symlink():
        raise ValueError("ROOT_PRESERVED_FAILED_OUTPUTS_REQUIRED")
    for original, preserved in ((ROOT / "build/electron-sidecar", archive / "electron-sidecar"),
                                (ROOT / "desktop/sidecar/ressectionlab-engine", archive / "ressectionlab-engine")):
        if original.exists() or original.is_symlink() or not preserved.is_dir() or preserved.is_symlink():
            raise ValueError("ROOT_PRESERVED_FAILED_OUTPUTS_REQUIRED")


def require_successful_sidecar(prior, source_sha):
    if (prior.get("status") != "completed_unverified" or prior.get("exit_code") != 0
            or prior.get("worker_reaped") is not True or prior.get("source_manifest_sha256") != source_sha):
        raise ValueError("SUCCESSFUL_SIDECAR_WITH_CURRENT_SOURCE_REQUIRED")


def run_stage(stage_name, release_path):
    document = plan()
    build_stage = "sidecar" if stage_name == RECOVERY_STAGE else stage_name
    stage = next(x for x in document["build_sequence"] if x["stage"] == build_stage)
    release = read(release_path)
    session, artifact = Path(release["session_directory"]), Path(release["receipt_path"])
    if (release.get("schema") != SCHEMA + "-root-release" or release.get("issuer") != "ROOT"
            or release.get("decision") != "release_one_stage"
            or release.get("stage") != stage_name or release.get("plan_sha256") != PLAN_SHA
            or release.get("runner_sha256") != sha(__file__)
            or not session.is_absolute() or not session.resolve().is_relative_to((ROOT / "outputs").resolve())
            or not artifact.is_absolute() or not artifact.resolve().is_relative_to((ROOT / "artifacts").resolve())
            or artifact.exists()):
        raise ValueError("EXACT_ROOT_STAGE_RELEASE_REQUIRED")
    if stage_name == "sidecar": session.mkdir(mode=0o700, parents=True, exist_ok=False)
    output = session / stage_name
    output.mkdir(mode=0o700, exist_ok=False)
    record = {"schema": SCHEMA + "-receipt", "stage": stage_name, "status": "preflight_failed",
              "plan_sha256": PLAN_SHA, "release_sha256": sha(release_path), "runner_sha256": sha(__file__),
              "source_manifest_sha256": release.get("source_manifest_sha256"), "environment_values_logged": False,
              "build_verified": False, "attempts": 1}
    high_water = {}
    try:
        baseline_path = session / "baseline.json"
        if stage_name == RECOVERY_STAGE:
            prior, baseline, high_water = recovery_state(session, release)
            record["predecessor"] = release["predecessor"]
            record["baseline_sha256"] = sha(baseline_path)
            record["carried_new_bytes"] = sum(high_water.values())
            record["cumulative_new_bytes"] = sum(high_water.values())
        elif stage_name == "electron":
            prior, baseline, high_water = predecessor_state(session, release.get("sidecar_subdirectory"), release.get("sidecar_binding"))
            record["selected_sidecar_subdirectory"] = release["sidecar_subdirectory"]
            record["selected_sidecar_binding"] = release["sidecar_binding"]
            record["baseline_sha256"] = sha(baseline_path)
            record["carried_new_bytes"] = record["cumulative_new_bytes"] = sum(high_water.values())
        source = source_manifest()
        if (release.get("source_commit") != source["commit"] or release.get("source_manifest_sha256") != value_sha(source)
                or release.get("program_sha256") != sha(stage["program"])):
            raise ValueError("SOURCE_OR_PROGRAM_BINDING_CHANGED")
        names = [*source["files"], str(Path(__file__).relative_to(ROOT)), "scripts/real_intake_io.py"]
        tracked = set(subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode().split("\0"))
        dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=all", "--", *names], cwd=ROOT)
        if not set(names) <= tracked or dirty:
            raise ValueError("SOURCE_COMMIT_REQUIRED")
        env = effective_environment(stage, document)
        if release.get("environment_sha256") != value_sha(env):
            raise ValueError("ENVIRONMENT_BINDING_CHANGED")
        if stage_name == RECOVERY_STAGE:
            preserved_outputs_ready()  # ROOT moves these intact; this runner never moves or copies them.
        if stage_name == "electron":
            require_successful_sidecar(prior, value_sha(source))
            addition = {"RESECTIONLAB_ELECTRON_ZIP_DIR": env["RESECTIONLAB_ELECTRON_ZIP_DIR"]}
            if release.get("allowed_environment_addition") != addition or release.get("archive") != archive_binding(document):
                raise ValueError("EXACT_LOCAL_ARCHIVE_RELEASE_REQUIRED")
            archive_fixity(document)
        roots = [ROOT / name for name in SCOPES]
        if any(p.is_symlink() for path in roots for p in (path, *path.parents) if p.is_relative_to(ROOT)):
            raise ValueError("MONITORED_ROOT_SYMLINK")
        if stage_name == "sidecar":
            publish(baseline_path, {"roots": [str(p) for p in roots], "files": disk_inventory(roots)})
        baseline = read(baseline_path)
        if baseline["roots"] != [str(p) for p in roots]: raise ValueError("BASELINE_SCOPE_CHANGED")
        record["baseline_sha256"] = sha(baseline_path)
        before_report = sha(ROOT / stage["report"]) if (ROOT / stage["report"]).exists() else None
        for variable in ("TMPDIR", "PYINSTALLER_CONFIG_DIR"):
            if variable in env: Path(env[variable]).mkdir(mode=0o700, parents=True, exist_ok=True)
        shared = document["shared_resource_limits"]
        limits = {"rss_bytes": stage["stop_limits"]["family_plus_supervisor_rss_bytes"],
                  "wall_seconds": stage["stop_limits"]["wall_seconds"], "available_bytes": shared["available_memory_floor_bytes"],
                  "free_disk_bytes": shared["free_disk_reserve_bytes"], "new_bytes": shared["cumulative_new_build_and_temporary_bytes_limit"],
                  "interval_seconds": shared["sample_interval_seconds"]}
        record["limits"] = limits
        record.update(supervise([stage["program"], *stage["args"]], stage["cwd"], env, output / "worker.log", roots,
                                baseline["files"], high_water, limits))
        plan()  # Fixed plan/helper continuity is also required after the worker.
        if value_sha(source_manifest()) != value_sha(source) or sha(__file__) != release["runner_sha256"] or sha(stage["program"]) != release["program_sha256"]:
            record["original_status"], record["status"] = record["status"], "source_or_runtime_changed"
        if stage_name == "electron": archive_fixity(document)
        if record["status"] == "completed_unverified":
            report = ROOT / stage["report"]
            if not report.exists() or sha(report) == before_report or read(report).get("status") != "built_unverified":
                raise ValueError("FRESH_BUILD_REPORT_REQUIRED")
            record["build_report_sha256"] = sha(report)
    except BaseException as error:
        record["original_status"] = record["status"]
        record["status"], record["error_type"] = "stage_failed", type(error).__name__
        if re.fullmatch(r"[A-Z0-9_]+", str(error)):
            record["error_code"] = str(error)
        # Deliberately omit raw exception messages/environment/logs from artifacts.
    finally:
        publish(output / "growth-ledger.json", high_water)
        record["growth_ledger_sha256"] = sha(output / "growth-ledger.json")
        publish(output / "receipt.json", record)
        publish(artifact, record)
    return 0 if record["status"] == "completed_unverified" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["describe", "sidecar", RECOVERY_STAGE, "electron"])
    parser.add_argument("--release", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.stage == "describe":
        if args.output is None: parser.error("describe requires --output")
        publish(args.output, describe())
        return 0
    if args.release is None: parser.error("a stage requires --release")
    return run_stage(args.stage, args.release)


if __name__ == "__main__":
    raise SystemExit(main())
