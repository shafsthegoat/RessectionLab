#!/usr/bin/env python3
"""Release-gated second one-call FEBio supervisor for a non-patient numerical cube.

This source is preparation only. No release is present in the repository and
importing or inspecting this module cannot start a native process.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import stat
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
PREPARATION = "manifests/experiments/nonpatient-n5-nonuniform-supervisor-preparation-v1.json"
PARENT = "manifests/experiments/nonpatient-tet10-sparse-feasibility-v1.json"
OUTPUT = "outputs/mechanics/nonpatient-n5-nonuniform-supervisor-v1/attempt-01"
CASE = "n5_nonuniform"
CASE_INDEX = 1
PRIOR = "outputs/mechanics/nonpatient-n5-supervisor-v1/attempt-01/receipt.json"
PRIOR_SHA256 = "9d26c0c57eefad06c53fe67b2bd64d9572778f23ed8b4356660654f88746f0ce"
PRIOR_SOURCE_COMMIT = "2f8f53f18111c2e839f677ac9c371e98f2f5175c"
PRIOR_RELEASE_SHA256 = "f2bebf5e1a7b722ff62e22603e63b80698e66fdc28fc3634ec078359f2726796"
PRIOR_DECK_SHA256 = "860dcdc8b29382281dd35ec41b6939bb29f1de5b3d3bd7a2e0adfd6f866a7674"
SOURCE_PATHS = (
    "scripts/mechanics_nonpatient_n5_nonuniform_supervisor.py",
    "scripts/mechanics_nonpatient_sparse_feasibility.py",
    "scripts/mechanics_nonpatient_sparse_deck.py",
    "scripts/mechanics_nonpatient_sparse_output.py",
    "scripts/mechanics_nonpatient_sparse_nonuniform_readout.py",
    "scripts/mechanics_patient_constraints.py",
    "scripts/mechanics_febio_verification.py",
    "scripts/mechanics_hbe_backend.py",
    "scripts/mechanics_hbe_access.py",
    "scripts/mechanics_hbe_evaluation.py",
    "scripts/febio_runtime.py",
)
CAPS = {"native_calls": 1, "attempts": 1, "wall_seconds": 600,
        "sampled_process_group_rss_bytes": 3 * 1024**3,
        "active_output_bytes": 512 * 1024**2, "numerical_threads": 1}
AGGREGATE_CAPS = {"maximum_native_calls": 7, "aggregate_native_wall_seconds": 1800,
                  "prior_calls": 1, "case_index": CASE_INDEX}
THREAD_ENV = {"OMP_NUM_THREADS": "1", "OMP_DYNAMIC": "FALSE",
              "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
              "VECLIB_MAXIMUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"}
HEX = re.compile(r"[0-9a-f]{64}\Z")
COMMIT = re.compile(r"[0-9a-f]{40}\Z")
BACKEND_LINE = re.compile(r"\*\s+Selecting linear solver accelerate\s+\*\Z", re.I)
POLL_SECONDS = .1
MAX_ACTIVE_ENTRIES = 4096
MAX_ACTIVE_DEPTH = 32
MAX_ACTIVE_SCAN_SECONDS = 1.0


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha_file(path: Path, *, maximum: int | None = None) -> str:
    path = Path(path)
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or (maximum is not None and info.st_size > maximum):
        raise ValueError("Bound input is not a regular file within its byte cap")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            digest.update(block)
    if maximum is not None and path.stat().st_size > maximum:
        raise ValueError("Bound input grew while hashing")
    return digest.hexdigest()


def local_path(relative: str, *, root: Path = ROOT) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise ValueError("Repository-relative path required")
    base = root.resolve()
    current = base
    for part in Path(relative).parts:
        current /= part
        if current.is_symlink():
            raise ValueError("Symlink in released path or output ancestor")
    target = (base / relative).resolve()
    if not target.is_relative_to(base):
        raise ValueError("Path escapes repository")
    return target


def audit_loaded_modules(*, root: Path = ROOT) -> None:
    """Constrain the executing Python import closure, including lazy imports.

    The two saved control checkers loaded by the backend use non-``scripts``
    module names and are separately authenticated by verify_profile. The local
    tet10 fixture dynamically loads mechanics_febio_verification; it is pinned
    here and its actual origin is checked below.
    """
    allowed = set(SOURCE_PATHS)
    base = root.resolve()
    for name, module in tuple(sys.modules.items()):
        if not name.startswith("scripts."):
            continue
        file = getattr(module, "__file__", None)
        if file is None:
            continue
        path = Path(file).resolve()
        if not path.is_relative_to(base / "scripts") or str(path.relative_to(base)) not in allowed:
            raise ValueError("Unbound executing scripts import: " + name)
    fixture = sys.modules.get("scripts.mechanics_patient_constraints")
    if fixture is not None:
        patch = getattr(fixture, "patch", None)
        if patch is None or Path(patch.__file__).resolve() != base / "scripts/mechanics_febio_verification.py":
            raise ValueError("Tet10 fixture loaded an unexpected patch helper")


def binding_bytes(binding: dict, expected_path: str, *, root: Path = ROOT, maximum: int = 1024**2) -> bytes:
    if not isinstance(binding, dict) or set(binding) != {"path", "sha256"}:
        raise ValueError("Exact path/SHA256 binding required")
    if binding["path"] != expected_path or not isinstance(binding["sha256"], str) or not HEX.fullmatch(binding["sha256"]):
        raise ValueError("Binding path or digest differs")
    path = local_path(expected_path, root=root)
    if sha_file(path, maximum=maximum) != binding["sha256"]:
        raise ValueError("Bound file changed: " + expected_path)
    data = path.read_bytes()
    if len(data) > maximum or sha_bytes(data) != binding["sha256"]:
        raise ValueError("Bound file changed during read: " + expected_path)
    return data


def git_blob(commit: str, relative: str, *, root: Path = ROOT) -> bytes:
    # Path comes exclusively from SOURCE_PATHS. Size check precedes the read.
    name = f"{commit}:{relative}"
    size = subprocess.run(["/usr/bin/git", "cat-file", "-s", name], cwd=root,
                          env=git_environment(), capture_output=True, check=True, timeout=10)
    if int(size.stdout.strip()) > 1024**2:
        raise ValueError("Committed source exceeds one MiB")
    content = subprocess.run(["/usr/bin/git", "cat-file", "blob", name], cwd=root,
                             env=git_environment(), capture_output=True, check=True, timeout=10).stdout
    if len(content) > 1024**2:
        raise ValueError("Committed source exceeds one MiB")
    return content


def git_environment() -> dict[str, str]:
    result = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    result.update(GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL="/dev/null",
                  GIT_CONFIG_SYSTEM="/dev/null", GIT_OPTIONAL_LOCKS="0")
    return result


def source_hashes(release: dict, *, root: Path = ROOT, require_git: bool = True) -> dict[str, str]:
    commit = release.get("source_commit")
    if not isinstance(commit, str) or not COMMIT.fullmatch(commit):
        raise ValueError("Full source commit required")
    if set(release.get("source_bindings", {})) != set(SOURCE_PATHS):
        raise ValueError("Complete executing source closure required")
    if require_git:
        head = subprocess.run(["/usr/bin/git", "rev-parse", "HEAD"], cwd=root,
                              env=git_environment(), capture_output=True, check=True, timeout=10).stdout.decode().strip()
        if head != commit:
            raise ValueError("Checkout HEAD differs from one-call source release")
    result = {}
    for relative in SOURCE_PATHS:
        binding = release["source_bindings"][relative]
        data = binding_bytes(binding, relative, root=root)
        if require_git and sha_bytes(git_blob(commit, relative, root=root)) != sha_bytes(data):
            raise ValueError("Working source differs from released committed source: " + relative)
        result[relative] = binding["sha256"]
    return result


def validate_preparation(value: dict) -> None:
    if (value.get("schema") != "nonpatient-n5-nonuniform-supervisor-preparation-v1"
            or value.get("status") != "prepared_not_released_or_executed"
            or value.get("release") is not None or value.get("case_id") != CASE
            or value.get("parent_declaration") != PARENT
            or value.get("output_directory") != OUTPUT or value.get("caps") != CAPS
            or value.get("aggregate_caps") != AGGREGATE_CAPS
            or value.get("prior_receipt") != {"path": PRIOR, "sha256": PRIOR_SHA256}
            or value.get("prior_native_wall_seconds") != 0.5189758341293782):
        raise ValueError("One-call preparation or cap drift")


def validate_prior_receipt(binding: dict, *, root: Path = ROOT) -> dict:
    """Require the completed first frozen call, without trusting status alone."""
    receipt = json.loads(binding_bytes(binding, PRIOR, root=root))
    elapsed = receipt.get("elapsed_seconds")
    if (binding["sha256"] != PRIOR_SHA256
            or receipt.get("schema") != "nonpatient-n5-one-call-receipt-v1"
            or receipt.get("case_id") != "n5_affine"
            or receipt.get("status") != "passed_numerical_software_only"
            or receipt.get("native_calls_attempted") != 1
            or receipt.get("no_retry") is not True
            or receipt.get("exit_code") != 0 or receipt.get("kill_reason") is not None
            or receipt.get("source_commit") != PRIOR_SOURCE_COMMIT
            or receipt.get("release_sha256") != PRIOR_RELEASE_SHA256
            or receipt.get("deck_sha256") != PRIOR_DECK_SHA256
            or receipt.get("saved_output_checks", {}).get("passed") is not True
            or receipt.get("affine_oracle", {}).get("passed") is not True
            or type(elapsed) not in (float, int) or not math.isfinite(elapsed)
            or not 0 < elapsed < CAPS["wall_seconds"]):
        raise ValueError("Prior n5 affine receipt is not the frozen passing first call")
    if (AGGREGATE_CAPS["prior_calls"] + CAPS["native_calls"] > AGGREGATE_CAPS["maximum_native_calls"]
            or elapsed + CAPS["wall_seconds"] > AGGREGATE_CAPS["aggregate_native_wall_seconds"]):
        raise ValueError("Remaining aggregate seven-call wall budget is insufficient")
    return {"sha256": binding["sha256"], "elapsed_seconds": elapsed,
            "calls_consumed": AGGREGATE_CAPS["prior_calls"]}


def validate_release(release: dict, *, root: Path = ROOT) -> dict:
    """Validate the externally reviewed release shape, bindings and sources.

    Local hashes cannot authenticate who issued the release. Root review is a
    separate procedural gate before this entry point may be used.
    """
    keys = {"schema", "status", "case_id", "source_commit", "source_bindings",
            "preparation", "parent_declaration", "backend_profile",
            "runtime_identity", "deck_sha256", "output_directory", "prior_receipt"}
    if (not isinstance(release, dict) or set(release) != keys
            or release["schema"] != "nonpatient-n5-nonuniform-one-call-release-v1"
            or release["status"] != "root_released_one_native_call"
            or release["case_id"] != CASE or release["output_directory"] != OUTPUT
            or not isinstance(release["deck_sha256"], str)
            or not HEX.fullmatch(release["deck_sha256"])):
        raise ValueError("Missing or altered root one-call release")
    prep = json.loads(binding_bytes(release["preparation"], PREPARATION, root=root))
    validate_preparation(prep)
    parent = json.loads(binding_bytes(release["parent_declaration"], PARENT, root=root))
    if (parent.get("schema") != "nonpatient-tet10-sparse-feasibility-v1"
            or parent.get("execution_release") is not None
            or parent.get("source_commit_for_execution") is not None
            or parent.get("caps", {}).get("per_call_wall_seconds") != CAPS["wall_seconds"]
            or parent["caps"].get("sampled_process_group_rss_bytes") != CAPS["sampled_process_group_rss_bytes"]
            or parent["caps"].get("active_output_bytes") != CAPS["active_output_bytes"]
            or len(parent["cases"]) != 7 or parent["cases"][CASE_INDEX].get("id") != CASE
            or parent["caps"] != {"maximum_native_calls": 7, "attempts_per_case": 1,
                                  "automatic_retry_or_fallback": False, "numerical_threads": 1,
                                  "sampled_process_group_rss_bytes": CAPS["sampled_process_group_rss_bytes"],
                                  "per_call_wall_seconds": CAPS["wall_seconds"],
                                  "aggregate_native_wall_seconds": AGGREGATE_CAPS["aggregate_native_wall_seconds"],
                                  "active_output_bytes": CAPS["active_output_bytes"]}):
        raise ValueError("Frozen parent declaration differs")
    if (release["backend_profile"] != parent["runtime"]["backend_profile"]
            or release["runtime_identity"] != parent["runtime"]["runtime_identity"]):
        raise ValueError("Release changed repaired runtime/profile")
    if release["prior_receipt"] != prep["prior_receipt"]:
        raise ValueError("Release changed prior n5 affine receipt binding")
    prior = validate_prior_receipt(release["prior_receipt"], root=root)
    if prior["elapsed_seconds"] != prep["prior_native_wall_seconds"]:
        raise ValueError("Prior wall time differs from the prepared aggregate accounting")
    prior_document = json.loads(binding_bytes(release["prior_receipt"], PRIOR, root=root))
    if (prior_document.get("parent_declaration_sha256") != release["parent_declaration"]["sha256"]
            or prior_document.get("runtime_identity_sha256") != release["runtime_identity"]["sha256"]
            or prior_document.get("backend_profile_sha256") != release["backend_profile"]["sha256"]
            or prior_document.get("backend_selection") != [
                "* Selecting linear solver accelerate                                    *"]):
        raise ValueError("Prior call does not share the frozen declaration and repaired backend")
    source_hashes(release, root=root)
    from scripts import mechanics_nonpatient_sparse_feasibility as design
    design.validate_declaration(parent, root=root)
    from scripts import mechanics_hbe_backend as backend
    context = backend.verify_profile(root, release["backend_profile"])
    audit_loaded_modules(root=root)
    if (context["profile_id"] != "accelerate_csc_v1"
            or context["runtime_identity"] != release["runtime_identity"]
            or context["runtime"]["executable_sha256"] !=
               context["runtime"]["libraries"]["install/bin/febio4"]):
        raise ValueError("Runtime identity or backend differs")
    from scripts import mechanics_nonpatient_sparse_deck as deck
    xml, metrics = deck.build_deck(CASE)
    audit_loaded_modules(root=root)
    deck_bytes = xml.encode("utf-8")
    if sha_bytes(deck_bytes) != release["deck_sha256"] or metrics["case_id"] != CASE:
        raise ValueError("Released generated deck differs")
    return {"runtime": context, "deck": deck_bytes, "prior": prior,
            "source_hashes": source_hashes(release, root=root), "parent_sha256": release["parent_declaration"]["sha256"]}


def private_environment() -> dict[str, str]:
    environment = dict(os.environ)
    for key in tuple(environment):
        if (key.startswith(("DYLD_", "PYTHON", "OMP_", "MKL_", "OPENBLAS_", "VECLIB_", "NUMEXPR_"))
                or key in {"LD_PRELOAD", "LD_LIBRARY_PATH", "__PYVENV_LAUNCHER__", "MKLROOT"}):
            environment.pop(key)
    environment.update(THREAD_ENV)
    return environment


def active_bytes(directory: Path) -> int:
    """Bound scan work; reject links/special files and include receipts/deck."""
    total = 0
    entries = 0
    deadline = time.monotonic() + MAX_ACTIVE_SCAN_SECONDS
    stack = [(Path(directory), 0)]
    while stack:
        base, depth = stack.pop()
        if depth > MAX_ACTIVE_DEPTH:
            raise ValueError("Active output tree exceeds depth bound")
        with os.scandir(base) as listing:
            for entry in listing:
                entries += 1
                if entries > MAX_ACTIVE_ENTRIES or time.monotonic() > deadline:
                    raise ValueError("Active output scan exceeded work bound")
                mode = entry.stat(follow_symlinks=False).st_mode
                if stat.S_ISREG(mode):
                    total += entry.stat(follow_symlinks=False).st_size
                    if total > CAPS["active_output_bytes"]:
                        return total
                elif stat.S_ISDIR(mode):
                    stack.append((Path(entry.path), depth + 1))
                else:
                    raise ValueError("Run produced symlink or special file")
    return total


def read_release(path: Path) -> tuple[bytes, tuple[int, int]]:
    """Read one regular unsymlinked release from a stable file identity."""
    if ".." in Path(path).parts:
        raise ValueError("Release path traversal is forbidden")
    path = Path(path).absolute()
    if any(part.is_symlink() for part in (path, *path.parents)):
        raise ValueError("Release path or ancestor is a symlink")
    try:
        before = path.lstat()
    except FileNotFoundError as error:
        raise ValueError("Separate bounded regular release file required") from error
    if not stat.S_ISREG(before.st_mode) or before.st_size > 1024**2:
        raise ValueError("Separate bounded regular release file required")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        opened = os.fstat(descriptor)
        if (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns) != (
                before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns):
            raise ValueError("Release identity changed before read")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            raw = stream.read(1024**2 + 1)
        after = os.fstat(descriptor)
        if (len(raw) > 1024**2 or len(raw) != opened.st_size
                or (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) !=
                   (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns)):
            raise ValueError("Release changed during read")
    finally:
        os.close(descriptor)
    return raw, (opened.st_dev, opened.st_ino)


def durable_json(path: Path, value: dict) -> None:
    payload = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def supervise(executable: str, directory: Path, receipt: dict,
              *, rss_observer=None, popen=None, sleep=time.sleep) -> dict:
    """Start exactly one native child; kill its process group on every unsafe end."""
    if rss_observer is None:
        from scripts.febio_runtime import process_group_rss
        rss_observer = process_group_rss
        audit_loaded_modules()
    if popen is None:
        popen = subprocess.Popen
    command = [executable, "-noconfig", "-no_title", "-i", CASE + ".feb", "-o", CASE + ".log"]
    receipt.update(command=command, native_calls_attempted=1, status="starting",
                   peak_sampled_process_group_rss_bytes=0,
                   peak_sampled_active_output_bytes=0, kill_reason=None, exit_code=None)
    durable_json(directory / "receipt.json", receipt)
    started = time.monotonic()
    process = None
    try:
        with (directory / "console.txt").open("xb") as log:
            process = popen(command, cwd=directory, env=private_environment(), stdout=log,
                            stderr=subprocess.STDOUT, start_new_session=True)
            receipt["pid"] = process.pid
            durable_json(directory / "receipt.json", receipt)
            while True:
                elapsed = time.monotonic() - started
                if elapsed >= CAPS["wall_seconds"]:
                    receipt["kill_reason"] = "wall_cap"
                    break
                code = process.poll()
                rss, members = rss_observer(process.pid, timeout_seconds=min(1., CAPS["wall_seconds"] - elapsed))
                active = active_bytes(directory)
                receipt["peak_sampled_process_group_rss_bytes"] = max(receipt["peak_sampled_process_group_rss_bytes"], rss)
                receipt["peak_sampled_active_output_bytes"] = max(receipt["peak_sampled_active_output_bytes"], active)
                if rss > CAPS["sampled_process_group_rss_bytes"]:
                    receipt["kill_reason"] = "process_group_rss_cap"
                elif active > CAPS["active_output_bytes"]:
                    receipt["kill_reason"] = "active_output_cap"
                elif code is not None:
                    receipt["exit_code"] = code
                    if members:
                        receipt["kill_reason"] = "descendants_outlived_solver"
                    break
                elif not members:
                    if process.poll() is None:
                        raise RuntimeError("Running process group RSS unavailable")
                    continue
                if receipt["kill_reason"]:
                    break
                if time.monotonic() - started >= CAPS["wall_seconds"]:
                    receipt["kill_reason"] = "wall_cap"
                    break
                receipt["elapsed_seconds"] = elapsed
                durable_json(directory / "receipt.json", receipt)
                sleep(POLL_SECONDS)
    except BaseException as error:
        receipt["kill_reason"] = receipt["kill_reason"] or "supervision_exception"
        receipt["supervision_error"] = {"type": type(error).__name__, "message": str(error)}
    finally:
        if process is not None:
            try:
                if receipt["kill_reason"] or process.poll() is None:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                receipt["exit_code"] = process.wait(timeout=2)
            except BaseException as error:
                receipt["kill_reason"] = receipt["kill_reason"] or "cleanup_exception"
                receipt["cleanup_error"] = {"type": type(error).__name__, "message": str(error)}
        receipt["elapsed_seconds"] = time.monotonic() - started
        receipt["status"] = ("native_exit_zero" if receipt["exit_code"] == 0
                              and receipt["kill_reason"] is None
                              and receipt["elapsed_seconds"] < CAPS["wall_seconds"]
                              else "failed_or_incomplete")
        durable_json(directory / "receipt.json", receipt)
    return receipt


def read_output(path: Path, limit: int) -> tuple[str, dict]:
    size = path.stat().st_size
    if size > limit or size < 1:
        raise ValueError("Missing or oversized native output")
    data = path.read_bytes()
    if len(data) != size:
        raise ValueError("Native output changed during read")
    return data.decode("ascii", errors="strict"), {"bytes": size, "sha256": sha_bytes(data)}


def inspect_outputs(directory: Path, receipt: dict) -> dict:
    if receipt["status"] != "native_exit_zero":
        raise ValueError("Native process did not exit cleanly within caps")
    if active_bytes(directory) > CAPS["active_output_bytes"]:
        raise ValueError("Final active output exceeds cap")
    names = {CASE + ".feb", CASE + ".log", CASE + ".nodes.log",
             CASE + ".elements.log", "console.txt", "receipt.json"}
    if {p.name for p in directory.iterdir()} != names:
        raise ValueError("Native output file inventory differs")
    texts, bindings = {}, {}
    for name in sorted(names - {"receipt.json"}):
        limit = 16 * 1024**2 if name in {CASE + ".log", "console.txt"} else CAPS["active_output_bytes"]
        texts[name], bindings[name] = read_output(directory / name, limit)
    console = texts["console.txt"]
    selections = [line.strip() for line in console.splitlines() if "selecting linear solver" in line.lower()]
    forbidden = re.search(r"\b(?:fallback|fall\s+back|switching\s+(?:linear\s+)?solver|selected\s+linear\s+solver)\b", console, re.I)
    if len(selections) != 1 or not BACKEND_LINE.fullmatch(selections[0]) or forbidden:
        raise ValueError("Exactly one actual repaired Accelerate solver selection required")
    from scripts import mechanics_nonpatient_sparse_output as parser
    from scripts import mechanics_nonpatient_sparse_nonuniform_readout as nonuniform
    audit_loaded_modules()
    checked = parser.check_case_outputs(CASE, texts[CASE + ".nodes.log"],
                                        texts[CASE + ".elements.log"], texts[CASE + ".log"])
    readout = nonuniform.check_nonuniform_readout(CASE, texts[CASE + ".nodes.log"],
                                                  texts[CASE + ".elements.log"], texts[CASE + ".log"])
    if checked.get("passed") is not True or readout.get("passed") is not True:
        raise ValueError("Saved-output residual/field or nonuniform local-force/readout failed")
    return {"output_bindings": bindings, "backend_selection": selections,
            "saved_output_checks": checked, "nonuniform_readout": readout}


def execute(release_path: Path, *, root: Path = ROOT) -> dict:
    """One attempt only after root release; preserve all failure evidence."""
    release_raw, release_identity = read_release(release_path)
    release = json.loads(release_raw)
    context = validate_release(release, root=root)
    directory = local_path(OUTPUT, root=root)
    directory.mkdir(parents=True, exist_ok=False)
    receipt = {"schema": "nonpatient-n5-nonuniform-one-call-receipt-v1", "status": "reserved",
               "case_id": CASE, "release_path": str(release_path.resolve()),
               "release_sha256": sha_bytes(release_raw), "source_commit": release["source_commit"],
               "source_hashes_before": context["source_hashes"],
               "parent_declaration_sha256": context["parent_sha256"],
               "runtime_identity_sha256": release["runtime_identity"]["sha256"],
               "backend_profile_sha256": release["backend_profile"]["sha256"],
               "prior_receipt_sha256": context["prior"]["sha256"],
               "prior_native_wall_seconds": context["prior"]["elapsed_seconds"],
               "prior_native_calls_consumed": context["prior"]["calls_consumed"],
               "planned_case_index": CASE_INDEX,
               "deck_sha256": release["deck_sha256"], "no_retry": True,
               "sampling_limit": "Brief between-sample RSS and output peaks may be missed."}
    durable_json(directory / "receipt.json", receipt)
    try:
        deck_path = directory / (CASE + ".feb")
        with deck_path.open("xb") as stream:
            stream.write(context["deck"])
            stream.flush()
            os.fsync(stream.fileno())
        if sha_file(deck_path) != release["deck_sha256"]:
            raise ValueError("Copied deck differs before native call")
        supervise(context["runtime"]["runtime"]["executable"], directory, receipt)
        receipt["aggregate_native_calls_attempted"] = (
            context["prior"]["calls_consumed"] + receipt["native_calls_attempted"])
        receipt["aggregate_native_wall_seconds_through_this_attempt"] = (
            context["prior"]["elapsed_seconds"] + receipt["elapsed_seconds"])
        if receipt["aggregate_native_wall_seconds_through_this_attempt"] > AGGREGATE_CAPS["aggregate_native_wall_seconds"]:
            raise ValueError("Aggregate native wall budget exceeded")
        if receipt["status"] != "native_exit_zero":
            raise ValueError("Native call failed or exceeded resource cap")
        receipt.update(inspect_outputs(directory, receipt))
        if (sha_file(deck_path) != release["deck_sha256"]
                or read_release(release_path) != (release_raw, release_identity)
                or validate_prior_receipt(release["prior_receipt"], root=root) != context["prior"]
                or source_hashes(release, root=root) != context["source_hashes"]):
            raise ValueError("Released input changed after native call")
        from scripts import mechanics_hbe_backend as backend
        after = backend.verify_profile(root, release["backend_profile"])
        audit_loaded_modules(root=root)
        if (after["runtime_identity"] != release["runtime_identity"]
                or after["inputs"] != context["runtime"]["inputs"]):
            raise ValueError("Runtime/profile controls changed after native call")
        receipt["status"] = "passed_numerical_software_only"
    except BaseException as error:
        receipt["status"] = "failed_or_incomplete"
        receipt["failure"] = {"type": type(error).__name__, "message": str(error)}
    finally:
        try:
            receipt["final_active_output_bytes"] = active_bytes(directory)
        except BaseException as error:
            receipt["status"] = "failed_or_incomplete"
            receipt["final_active_output_error"] = {"type": type(error).__name__, "message": str(error)}
        durable_json(directory / "receipt.json", receipt)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, required=True, help="Separate root-issued immutable one-call release")
    parser.add_argument("--execute", action="store_true", help="Explicitly spend the one native call")
    args = parser.parse_args()
    if not args.execute:
        raise ValueError("No native operation without explicit --execute and root release")
    result = execute(args.release)
    print(json.dumps({"status": result["status"], "receipt": str(local_path(OUTPUT) / "receipt.json")}))
    return 0 if result["status"] == "passed_numerical_software_only" else 1


if __name__ == "__main__":
    raise SystemExit(main())
