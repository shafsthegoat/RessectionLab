#!/usr/bin/env python3
"""One-use, release-gated n9 affine numerical-cube supervisor; source only.

The two prior n5 calls are prerequisites. Importing this module does not run
FEBio; native execution requires a separate reviewed release and --execute.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

from scripts import mechanics_nonpatient_n5_supervisor as core


ROOT = Path(__file__).resolve().parents[1]
CASE = "n9_affine"
CASE_INDEX = 2
PREPARATION = "manifests/experiments/nonpatient-n9-affine-supervisor-preparation-v1.json"
PARENT = "manifests/experiments/nonpatient-tet10-sparse-feasibility-v1.json"
OUTPUT = "outputs/mechanics/nonpatient-n9-affine-supervisor-v1/attempt-01"
PRIOR = (
    ("n5_affine", "outputs/mechanics/nonpatient-n5-supervisor-v1/attempt-01/receipt.json",
     "9d26c0c57eefad06c53fe67b2bd64d9572778f23ed8b4356660654f88746f0ce",
     "2f8f53f18111c2e839f677ac9c371e98f2f5175c",
     "f2bebf5e1a7b722ff62e22603e63b80698e66fdc28fc3634ec078359f2726796",
     "860dcdc8b29382281dd35ec41b6939bb29f1de5b3d3bd7a2e0adfd6f866a7674",
     0.5189758341293782),
    ("n5_nonuniform", "outputs/mechanics/nonpatient-n5-nonuniform-supervisor-v1/attempt-01/receipt.json",
     "dae7f8f7646da5b05b048db07b6129388c754824a34396bbe9caa744b8a372e1",
     "65c6d0b8b54514ae078bc72164a4d8dea416951c",
     "5c360159bf32d6eb1d4d65d9e7a745be0fd7f8680609c41e4effbf66352b116f",
     "df19423a89f6af2c72f8fd5ab1566a8fa3c26c253a6b750b31f79bfd9de4e870",
     0.5194279160350561),
)
SOURCE_PATHS = (
    "scripts/mechanics_nonpatient_n9_affine_supervisor.py",
    "scripts/mechanics_nonpatient_n5_supervisor.py",  # bounded pure I/O/runtime helpers
    "scripts/mechanics_nonpatient_sparse_feasibility.py",
    "scripts/mechanics_nonpatient_sparse_deck.py",
    "scripts/mechanics_nonpatient_sparse_output.py",
    "scripts/mechanics_nonpatient_sparse_affine_readout.py",
    "scripts/mechanics_patient_constraints.py",
    "scripts/mechanics_febio_verification.py",
    "scripts/mechanics_hbe_backend.py",
    "scripts/mechanics_hbe_access.py",
    "scripts/mechanics_hbe_evaluation.py",
    "scripts/febio_runtime.py",
)
CAPS = dict(core.CAPS)
AGGREGATE = {"maximum_native_calls": 7, "aggregate_native_wall_seconds": 1800,
             "prior_calls": 2, "case_index": CASE_INDEX}
BACKEND_LINE = core.BACKEND_LINE
EXPECTED_SELECTION = "* Selecting linear solver accelerate                                    *"
HEX = re.compile(r"[0-9a-f]{64}\Z")
COMMIT = re.compile(r"[0-9a-f]{40}\Z")


def audit_loaded_modules(*, root: Path = ROOT) -> None:
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


def source_hashes(release: dict, *, root: Path = ROOT, require_git: bool = True) -> dict[str, str]:
    commit = release.get("source_commit")
    if not isinstance(commit, str) or not COMMIT.fullmatch(commit):
        raise ValueError("Full source commit required")
    if set(release.get("source_bindings", {})) != set(SOURCE_PATHS):
        raise ValueError("Complete executing source closure required")
    if require_git:
        head = subprocess.run(["/usr/bin/git", "rev-parse", "HEAD"], cwd=root,
                              env=core.git_environment(), capture_output=True,
                              check=True, timeout=10).stdout.decode().strip()
        if head != commit:
            raise ValueError("Checkout HEAD differs from one-call source release")
    result = {}
    for relative in SOURCE_PATHS:
        binding = release["source_bindings"][relative]
        data = core.binding_bytes(binding, relative, root=root)
        if require_git and core.sha_bytes(core.git_blob(commit, relative, root=root)) != core.sha_bytes(data):
            raise ValueError("Working source differs from released committed source: " + relative)
        result[relative] = binding["sha256"]
    return result


def validate_preparation(value: dict) -> None:
    expected_prior = [{"path": path, "sha256": digest, "elapsed_seconds": elapsed}
                      for _, path, digest, _, _, _, elapsed in PRIOR]
    if (value.get("schema") != "nonpatient-n9-affine-supervisor-preparation-v1"
            or value.get("status") != "prepared_not_released_or_executed"
            or value.get("release") is not None or value.get("case_id") != CASE
            or value.get("parent_declaration") != PARENT
            or value.get("output_directory") != OUTPUT or value.get("caps") != CAPS
            or value.get("aggregate_caps") != AGGREGATE
            or value.get("prior_receipts") != expected_prior):
        raise ValueError("Third one-call preparation or cap drift")


def validate_prior_receipts(bindings: list[dict], parent: dict,
                            *, root: Path = ROOT) -> dict:
    """Check both receipts and every prior native file before consuming call 3."""
    if not isinstance(bindings, list) or len(bindings) != len(PRIOR):
        raise ValueError("Both ordered prior receipts required")
    elapsed_total = 0.0
    hashes = []
    for index, (binding, expected) in enumerate(zip(bindings, PRIOR, strict=True)):
        case, path, digest, commit, release_digest, deck_digest, elapsed = expected
        if binding != {"path": path, "sha256": digest}:
            raise ValueError("Prior receipt path/SHA/order differs")
        document = json.loads(core.binding_bytes(binding, path, root=root))
        oracle_key = "affine_oracle" if index == 0 else "nonuniform_readout"
        if (document.get("schema") != ("nonpatient-n5-one-call-receipt-v1" if index == 0 else
                                      "nonpatient-n5-nonuniform-one-call-receipt-v1")
                or document.get("case_id") != case
                or document.get("status") != "passed_numerical_software_only"
                or document.get("native_calls_attempted") != 1
                or document.get("no_retry") is not True
                or document.get("exit_code") != 0 or document.get("kill_reason") is not None
                or document.get("source_commit") != commit
                or document.get("release_sha256") != release_digest
                or document.get("deck_sha256") != deck_digest
                or document.get("parent_declaration_sha256") != parent["sha256"]
                or document.get("runtime_identity_sha256") != parent["runtime_identity_sha256"]
                or document.get("backend_profile_sha256") != parent["backend_profile_sha256"]
                or document.get("backend_selection") != [EXPECTED_SELECTION]
                or document.get("saved_output_checks", {}).get("passed") is not True
                or document.get(oracle_key, {}).get("passed") is not True
                or document.get("elapsed_seconds") != elapsed
                or type(elapsed) not in (int, float) or not math.isfinite(elapsed)
                or not 0 < elapsed < CAPS["wall_seconds"]):
            raise ValueError("Prior numerical-only decision or identity differs")
        if index == 1 and (document.get("prior_receipt_sha256") != PRIOR[0][2]
                           or document.get("prior_native_wall_seconds") != PRIOR[0][6]
                           or document.get("aggregate_native_calls_attempted") != 2
                           or document.get("aggregate_native_wall_seconds_through_this_attempt") !=
                              PRIOR[0][6] + elapsed):
            raise ValueError("Second receipt does not account for the first native call")
        names = {case + ".feb", case + ".log", case + ".nodes.log",
                 case + ".elements.log", "console.txt"}
        outputs = document.get("output_bindings")
        if not isinstance(outputs, dict) or set(outputs) != names:
            raise ValueError("Prior native output inventory differs")
        if core.sha_file(core.local_path(path, root=root), maximum=1024**2) != digest:
            raise ValueError("Prior receipt changed during read")
        directory = str(Path(path).parent)
        saved_directory = core.local_path(directory, root=root)
        if ({item.name for item in saved_directory.iterdir()} != names | {"receipt.json"}
                or core.active_bytes(saved_directory) > CAPS["active_output_bytes"]):
            raise ValueError("Prior native directory inventory or output cap differs")
        for name in names:
            item = outputs[name]
            if (not isinstance(item, dict) or set(item) != {"bytes", "sha256"}
                    or type(item["bytes"]) is not int or not 0 < item["bytes"] <= CAPS["active_output_bytes"]
                    or not isinstance(item["sha256"], str) or not HEX.fullmatch(item["sha256"])):
                raise ValueError("Malformed prior native output binding")
            saved = core.local_path(directory + "/" + name, root=root)
            if saved.stat().st_size != item["bytes"] or core.sha_file(saved, maximum=CAPS["active_output_bytes"]) != item["sha256"]:
                raise ValueError("Prior native output changed: " + name)
        elapsed_total += elapsed
        hashes.append(digest)
    if (len(hashes) + CAPS["native_calls"] > AGGREGATE["maximum_native_calls"]
            or elapsed_total + CAPS["wall_seconds"] > AGGREGATE["aggregate_native_wall_seconds"]):
        raise ValueError("Remaining seven-call aggregate wall budget is insufficient")
    return {"sha256": hashes, "elapsed_seconds": elapsed_total, "calls_consumed": len(hashes)}


def validate_release(release: dict, *, root: Path = ROOT) -> dict:
    keys = {"schema", "status", "case_id", "source_commit", "source_bindings",
            "preparation", "parent_declaration", "backend_profile", "runtime_identity",
            "deck_sha256", "output_directory", "prior_receipts"}
    if (not isinstance(release, dict) or set(release) != keys
            or release["schema"] != "nonpatient-n9-affine-one-call-release-v1"
            or release["status"] != "root_released_one_native_call"
            or release["case_id"] != CASE or release["output_directory"] != OUTPUT
            or not isinstance(release["deck_sha256"], str)
            or not HEX.fullmatch(release["deck_sha256"])):
        raise ValueError("Missing or altered n9 affine one-call release")
    prep = json.loads(core.binding_bytes(release["preparation"], PREPARATION, root=root))
    validate_preparation(prep)
    parent = json.loads(core.binding_bytes(release["parent_declaration"], PARENT, root=root))
    from scripts import mechanics_nonpatient_sparse_feasibility as design
    design.validate_declaration(parent, root=root)
    if (len(parent["cases"]) != 7 or parent["cases"][CASE_INDEX] !=
            {"id": CASE, "n": 9, "load": "affine"}
            or parent["caps"] != {"maximum_native_calls": 7, "attempts_per_case": 1,
                                  "automatic_retry_or_fallback": False, "numerical_threads": 1,
                                  "sampled_process_group_rss_bytes": CAPS["sampled_process_group_rss_bytes"],
                                  "per_call_wall_seconds": CAPS["wall_seconds"],
                                  "aggregate_native_wall_seconds": AGGREGATE["aggregate_native_wall_seconds"],
                                  "active_output_bytes": CAPS["active_output_bytes"]}
            or release["backend_profile"] != parent["runtime"]["backend_profile"]
            or release["runtime_identity"] != parent["runtime"]["runtime_identity"]
            or release["prior_receipts"] != [{"path": row["path"], "sha256": row["sha256"]}
                                             for row in prep["prior_receipts"]]):
        raise ValueError("Frozen parent, runtime, or predecessor release differs")
    parent_context = {"sha256": release["parent_declaration"]["sha256"],
                      "runtime_identity_sha256": release["runtime_identity"]["sha256"],
                      "backend_profile_sha256": release["backend_profile"]["sha256"]}
    prior = validate_prior_receipts(release["prior_receipts"], parent_context, root=root)
    if prior["elapsed_seconds"] != sum(row["elapsed_seconds"] for row in prep["prior_receipts"]):
        raise ValueError("Prepared cumulative wall accounting differs")
    bound = source_hashes(release, root=root)
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
    if core.sha_bytes(deck_bytes) != release["deck_sha256"] or metrics["case_id"] != CASE:
        raise ValueError("Released n9 affine deck differs")
    return {"runtime": context, "deck": deck_bytes, "prior": prior,
            "source_hashes": bound, "parent_sha256": parent_context["sha256"]}


def supervise(executable: str, directory: Path, receipt: dict,
              *, rss_observer=None, popen=None, sleep=time.sleep) -> dict:
    """Exactly one native child with the frozen caps and no retry."""
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
    core.durable_json(directory / "receipt.json", receipt)
    started = time.monotonic()
    process = None
    try:
        with (directory / "console.txt").open("xb") as log:
            process = popen(command, cwd=directory, env=core.private_environment(), stdout=log,
                            stderr=subprocess.STDOUT, start_new_session=True)
            receipt["pid"] = process.pid
            core.durable_json(directory / "receipt.json", receipt)
            while True:
                elapsed = time.monotonic() - started
                if elapsed >= CAPS["wall_seconds"]:
                    receipt["kill_reason"] = "wall_cap"
                    break
                code = process.poll()
                rss, members = rss_observer(process.pid, timeout_seconds=min(1., CAPS["wall_seconds"] - elapsed))
                active = core.active_bytes(directory)
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
                receipt["elapsed_seconds"] = elapsed
                core.durable_json(directory / "receipt.json", receipt)
                sleep(core.POLL_SECONDS)
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
        core.durable_json(directory / "receipt.json", receipt)
    return receipt


def inspect_outputs(directory: Path, receipt: dict) -> dict:
    if receipt["status"] != "native_exit_zero":
        raise ValueError("Native process did not exit cleanly within caps")
    if core.active_bytes(directory) > CAPS["active_output_bytes"]:
        raise ValueError("Final active output exceeds cap")
    names = {CASE + ".feb", CASE + ".log", CASE + ".nodes.log",
             CASE + ".elements.log", "console.txt", "receipt.json"}
    if {p.name for p in directory.iterdir()} != names:
        raise ValueError("Native output file inventory differs")
    texts, bindings = {}, {}
    for name in sorted(names - {"receipt.json"}):
        limit = 16 * 1024**2 if name in {CASE + ".log", "console.txt"} else CAPS["active_output_bytes"]
        texts[name], bindings[name] = core.read_output(directory / name, limit)
    selections = [line.strip() for line in texts["console.txt"].splitlines()
                  if "selecting linear solver" in line.lower()]
    forbidden = re.search(r"\b(?:fallback|fall\s+back|switching\s+(?:linear\s+)?solver|selected\s+linear\s+solver)\b",
                          texts["console.txt"], re.I)
    if len(selections) != 1 or not BACKEND_LINE.fullmatch(selections[0]) or forbidden:
        raise ValueError("Exactly one actual repaired Accelerate solver selection required")
    from scripts import mechanics_nonpatient_sparse_output as parser
    from scripts import mechanics_nonpatient_sparse_affine_readout as affine
    audit_loaded_modules()
    checked = parser.check_case_outputs(CASE, texts[CASE + ".nodes.log"],
                                        texts[CASE + ".elements.log"], texts[CASE + ".log"])
    readout = affine.check_affine_readout(texts[CASE + ".nodes.log"],
                                          texts[CASE + ".elements.log"], case_id=CASE)
    if checked.get("passed") is not True or readout.get("passed") is not True:
        raise ValueError("Saved-output residual/field or n9 affine oracle failed")
    return {"output_bindings": bindings, "backend_selection": selections,
            "saved_output_checks": checked, "affine_oracle": readout}


def execute(release_path: Path, *, root: Path = ROOT) -> dict:
    raw, identity = core.read_release(release_path)
    release = json.loads(raw)
    context = validate_release(release, root=root)
    directory = core.local_path(OUTPUT, root=root)
    directory.mkdir(parents=True, exist_ok=False)
    receipt = {"schema": "nonpatient-n9-affine-one-call-receipt-v1", "status": "reserved",
               "case_id": CASE, "release_path": str(release_path.resolve()),
               "release_sha256": core.sha_bytes(raw), "source_commit": release["source_commit"],
               "source_hashes_before": context["source_hashes"],
               "parent_declaration_sha256": context["parent_sha256"],
               "runtime_identity_sha256": release["runtime_identity"]["sha256"],
               "backend_profile_sha256": release["backend_profile"]["sha256"],
               "prior_receipt_sha256": context["prior"]["sha256"],
               "prior_native_wall_seconds": context["prior"]["elapsed_seconds"],
               "prior_native_calls_consumed": context["prior"]["calls_consumed"],
               "planned_case_index": CASE_INDEX, "deck_sha256": release["deck_sha256"],
               "no_retry": True,
               "sampling_limit": "Brief between-sample RSS and output peaks may be missed."}
    core.durable_json(directory / "receipt.json", receipt)
    try:
        deck_path = directory / (CASE + ".feb")
        with deck_path.open("xb") as stream:
            stream.write(context["deck"])
            stream.flush()
            os.fsync(stream.fileno())
        if core.sha_file(deck_path) != release["deck_sha256"]:
            raise ValueError("Copied deck differs before native call")
        supervise(context["runtime"]["runtime"]["executable"], directory, receipt)
        receipt["aggregate_native_calls_attempted"] = (
            context["prior"]["calls_consumed"] + receipt["native_calls_attempted"])
        receipt["aggregate_native_wall_seconds_through_this_attempt"] = (
            context["prior"]["elapsed_seconds"] + receipt["elapsed_seconds"])
        if receipt["aggregate_native_wall_seconds_through_this_attempt"] > AGGREGATE["aggregate_native_wall_seconds"]:
            raise ValueError("Aggregate native wall budget exceeded")
        if receipt["status"] != "native_exit_zero":
            raise ValueError("Native call failed or exceeded resource cap")
        receipt.update(inspect_outputs(directory, receipt))
        if (core.sha_file(deck_path) != release["deck_sha256"]
                or core.read_release(release_path) != (raw, identity)
                or validate_prior_receipts(release["prior_receipts"],
                                           {"sha256": release["parent_declaration"]["sha256"],
                                            "runtime_identity_sha256": release["runtime_identity"]["sha256"],
                                            "backend_profile_sha256": release["backend_profile"]["sha256"]},
                                           root=root) != context["prior"]
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
            receipt["final_active_output_bytes"] = core.active_bytes(directory)
        except BaseException as error:
            receipt["status"] = "failed_or_incomplete"
            receipt["final_active_output_error"] = {"type": type(error).__name__, "message": str(error)}
        core.durable_json(directory / "receipt.json", receipt)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, required=True,
                        help="Separate root-issued immutable one-call release")
    parser.add_argument("--execute", action="store_true", help="Explicitly spend the one native call")
    args = parser.parse_args()
    if not args.execute:
        raise ValueError("No native operation without explicit --execute and root release")
    result = execute(args.release)
    print(json.dumps({"status": result["status"], "receipt": str(core.local_path(OUTPUT) / "receipt.json")}))
    return 0 if result["status"] == "passed_numerical_software_only" else 1


if __name__ == "__main__":
    raise SystemExit(main())
