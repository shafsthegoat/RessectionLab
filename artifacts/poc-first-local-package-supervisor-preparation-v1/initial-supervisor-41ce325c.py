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


def run_stage(stage_name, release_path):
    document = plan()
    stage = next(x for x in document["build_sequence"] if x["stage"] == stage_name)
    release = read(release_path)
    session, artifact = Path(release["session_directory"]), Path(release["receipt_path"])
    if (release.get("schema") != SCHEMA + "-root-release" or release.get("decision") != "release_one_stage"
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
        if stage_name == "electron":
            addition = {"RESECTIONLAB_ELECTRON_ZIP_DIR": env["RESECTIONLAB_ELECTRON_ZIP_DIR"]}
            if release.get("allowed_environment_addition") != addition or release.get("archive") != archive_binding(document):
                raise ValueError("EXACT_LOCAL_ARCHIVE_RELEASE_REQUIRED")
            archive_fixity(document)
        roots = [ROOT / name for name in SCOPES]
        if any(p.is_symlink() for path in roots for p in (path, *path.parents) if p.is_relative_to(ROOT)):
            raise ValueError("MONITORED_ROOT_SYMLINK")
        baseline_path = session / "baseline.json"
        if stage_name == "sidecar":
            publish(baseline_path, {"roots": [str(p) for p in roots], "files": disk_inventory(roots)})
        else:
            prior = read(session / "sidecar/receipt.json")
            if (release.get("sidecar_receipt_sha256") != sha(session / "sidecar/receipt.json")
                    or prior.get("status") != "completed_unverified"
                    or prior.get("source_manifest_sha256") != value_sha(source)
                    or prior.get("baseline_sha256") != sha(baseline_path)
                    or prior.get("growth_ledger_sha256") != sha(session / "sidecar/growth-ledger.json")):
                raise ValueError("SUCCESSFUL_SAME_BASELINE_SIDECAR_REQUIRED")
            high_water = read(session / "sidecar/growth-ledger.json")
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
    parser.add_argument("stage", choices=["describe", "sidecar", "electron"])
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
