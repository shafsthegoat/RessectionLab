#!/usr/bin/env python3
"""Release-gated, one-case-at-a-time continuation of the frozen cube benchmark.

This source only prepares cases 3–6. Importing it, or running without a
separate reviewed one-call release, cannot launch FEBio.
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

from scripts import mechanics_nonpatient_n5_supervisor as io


ROOT = Path(__file__).resolve().parents[1]
PREPARATION = "manifests/experiments/nonpatient-sparse-continuation-preparation-v1.json"
PARENT = "manifests/experiments/nonpatient-tet10-sparse-feasibility-v1.json"
OUTPUT_ROOT = "outputs/mechanics/nonpatient-sparse-continuation-v1"
REMAINING = ("n9_nonuniform", "n13_affine", "n13_nonuniform", "n13_nonuniform_half_step")
FIRST_INDEX = 3
FIXED_PRIOR = (
    {"case_id": "n5_affine", "path": "outputs/mechanics/nonpatient-n5-supervisor-v1/attempt-01/receipt.json",
     "sha256": "9d26c0c57eefad06c53fe67b2bd64d9572778f23ed8b4356660654f88746f0ce",
     "source_commit": "2f8f53f18111c2e839f677ac9c371e98f2f5175c",
     "release_sha256": "f2bebf5e1a7b722ff62e22603e63b80698e66fdc28fc3634ec078359f2726796",
     "deck_sha256": "860dcdc8b29382281dd35ec41b6939bb29f1de5b3d3bd7a2e0adfd6f866a7674",
     "elapsed_seconds": 0.5189758341293782},
    {"case_id": "n5_nonuniform", "path": "outputs/mechanics/nonpatient-n5-nonuniform-supervisor-v1/attempt-01/receipt.json",
     "sha256": "dae7f8f7646da5b05b048db07b6129388c754824a34396bbe9caa744b8a372e1",
     "source_commit": "65c6d0b8b54514ae078bc72164a4d8dea416951c",
     "release_sha256": "5c360159bf32d6eb1d4d65d9e7a745be0fd7f8680609c41e4effbf66352b116f",
     "deck_sha256": "df19423a89f6af2c72f8fd5ab1566a8fa3c26c253a6b750b31f79bfd9de4e870",
     "elapsed_seconds": 0.5194279160350561},
    {"case_id": "n9_affine", "path": "outputs/mechanics/nonpatient-n9-affine-supervisor-v1/attempt-01/receipt.json",
     "sha256": "44775df0f7f04b2fdd6ec3f0ba43e595d34a07c6303142a5be1cb20d921f52b1",
     "source_commit": "a61b82c9555ccb52bddba1927a27c9e0ce4eb8b9",
     "release_sha256": "e90840f8ad6afa08ecd9b3d722a7a87a6ba6f8ece7519633d1810f32115eb5bd",
     "deck_sha256": "1dd6d1eeaa4072fed359f8217e1389a748bfafc065d230f643f607d4ff9645c9",
     "elapsed_seconds": 3.649174875114113},
)
SOURCE_PATHS = (
    "scripts/mechanics_nonpatient_sparse_continuation_supervisor.py",
    "scripts/mechanics_nonpatient_n5_supervisor.py",  # pure bounded I/O helpers
    "scripts/mechanics_nonpatient_sparse_feasibility.py",
    "scripts/mechanics_nonpatient_sparse_deck.py",
    "scripts/mechanics_nonpatient_sparse_output.py",
    "scripts/mechanics_nonpatient_sparse_affine_readout.py",
    "scripts/mechanics_nonpatient_sparse_nonuniform_readout.py",
    "scripts/mechanics_patient_constraints.py",
    "scripts/mechanics_febio_verification.py",
    "scripts/mechanics_hbe_backend.py",
    "scripts/mechanics_hbe_access.py",
    "scripts/mechanics_hbe_evaluation.py",
    "scripts/febio_runtime.py",
)
CAPS = dict(io.CAPS)
AGGREGATE = {"maximum_native_calls": 7, "aggregate_native_wall_seconds": 1800,
             "first_continuation_index": FIRST_INDEX}
SELECTION = "* Selecting linear solver accelerate                                    *"
HEX = re.compile(r"[0-9a-f]{64}\Z")
COMMIT = re.compile(r"[0-9a-f]{40}\Z")


def output_directory(case_id: str) -> str:
    if case_id not in REMAINING:
        raise ValueError("Undeclared continuation case")
    return f"{OUTPUT_ROOT}/{case_id}/attempt-01"


def canonical_receipt(index: int, case_id: str) -> str:
    if index < FIRST_INDEX:
        if case_id != FIXED_PRIOR[index]["case_id"]:
            raise ValueError("Frozen predecessor order changed")
        return FIXED_PRIOR[index]["path"]
    return output_directory(case_id) + "/receipt.json"


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
                              env=io.git_environment(), capture_output=True,
                              check=True, timeout=10).stdout.decode().strip()
        if head != commit:
            raise ValueError("Checkout HEAD differs from one-call source release")
    result = {}
    for relative in SOURCE_PATHS:
        binding = release["source_bindings"][relative]
        data = io.binding_bytes(binding, relative, root=root)
        if require_git and io.sha_bytes(io.git_blob(commit, relative, root=root)) != io.sha_bytes(data):
            raise ValueError("Working source differs from released committed source: " + relative)
        result[relative] = binding["sha256"]
    return result


def validate_preparation(value: dict) -> None:
    if (value.get("schema") != "nonpatient-sparse-continuation-preparation-v1"
            or value.get("status") != "prepared_not_released_or_executed"
            or value.get("release") is not None or value.get("parent_declaration") != PARENT
            or value.get("case_ids") != list(REMAINING)
            or value.get("output_root") != OUTPUT_ROOT or value.get("caps") != CAPS
            or value.get("aggregate_caps") != AGGREGATE
            or value.get("fixed_prior_receipts") != list(FIXED_PRIOR)):
        raise ValueError("Continuation preparation or frozen caps drift")


def _finite_seconds(value: object) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value < CAPS["wall_seconds"]:
        raise ValueError("Invalid predecessor native wall duration")
    return float(value)


def validate_prior_chain(bindings: list[dict], case_index: int, parent: dict,
                         *, root: Path = ROOT) -> dict:
    """Require every earlier frozen row, including all saved native file hashes."""
    cases = parent["cases"]
    if (case_index not in range(FIRST_INDEX, len(cases))
            or not isinstance(bindings, list) or len(bindings) != case_index):
        raise ValueError("Complete ordered predecessor chain required")
    prior_hashes: list[str] = []
    elapsed_total = 0.0
    for index, binding in enumerate(bindings):
        case = cases[index]
        case_id = case["id"]
        path = canonical_receipt(index, case_id)
        if (not isinstance(binding, dict) or set(binding) != {"case_id", "path", "sha256"}
                or binding["case_id"] != case_id or binding["path"] != path
                or not isinstance(binding["sha256"], str) or not HEX.fullmatch(binding["sha256"])
                or index < FIRST_INDEX and binding["sha256"] != FIXED_PRIOR[index]["sha256"]):
            raise ValueError("Predecessor case/path/SHA/order differs")
        document = json.loads(io.binding_bytes({"path": path, "sha256": binding["sha256"]}, path, root=root))
        elapsed = _finite_seconds(document.get("elapsed_seconds"))
        expected_schema = ("nonpatient-n5-one-call-receipt-v1" if index == 0 else
                           "nonpatient-n5-nonuniform-one-call-receipt-v1" if index == 1 else
                           "nonpatient-n9-affine-one-call-receipt-v1" if index == 2 else
                           "nonpatient-sparse-continuation-one-call-receipt-v1")
        decision = "affine_oracle" if case["load"] == "affine" else "nonuniform_readout"
        if (document.get("schema") != expected_schema or document.get("case_id") != case_id
                or document.get("status") != "passed_numerical_software_only"
                or document.get("native_calls_attempted") != 1 or document.get("no_retry") is not True
                or document.get("exit_code") != 0 or document.get("kill_reason") is not None
                or document.get("parent_declaration_sha256") != parent["sha256"]
                or document.get("runtime_identity_sha256") != parent["runtime_identity_sha256"]
                or document.get("backend_profile_sha256") != parent["backend_profile_sha256"]
                or document.get("backend_selection") != [SELECTION]
                or document.get("saved_output_checks", {}).get("passed") is not True
                or document.get("saved_output_checks", {}).get("case_id") != case_id
                or document.get(decision, {}).get("passed") is not True
                or document.get(decision, {}).get("case_id") != case_id):
            raise ValueError("Predecessor numerical decision or identity differs")
        if index < FIRST_INDEX:
            fixed = FIXED_PRIOR[index]
            if any(document.get(key) != fixed[key] for key in
                   ("source_commit", "release_sha256", "deck_sha256", "elapsed_seconds")):
                raise ValueError("Pinned predecessor result differs")
        elif (document.get("planned_case_index") != index
              or not isinstance(document.get("source_commit"), str)
              or not COMMIT.fullmatch(document["source_commit"])
              or not isinstance(document.get("release_sha256"), str)
              or not HEX.fullmatch(document["release_sha256"])
              or not isinstance(document.get("deck_sha256"), str)
              or not HEX.fullmatch(document["deck_sha256"])):
            raise ValueError("Continuation predecessor release identity missing")
        if index >= 1:
            expected_hashes = prior_hashes[0] if index == 1 else prior_hashes
            if (document.get("prior_receipt_sha256") != expected_hashes
                    or document.get("prior_native_wall_seconds") != elapsed_total
                    or document.get("prior_native_calls_consumed") != index
                    or document.get("aggregate_native_calls_attempted") != index + 1
                    or document.get("aggregate_native_wall_seconds_through_this_attempt") !=
                       elapsed_total + elapsed):
                raise ValueError("Predecessor cumulative call/wall/hash chain differs")
        names = {case_id + ".feb", case_id + ".log", case_id + ".nodes.log",
                 case_id + ".elements.log", "console.txt"}
        outputs = document.get("output_bindings")
        if not isinstance(outputs, dict) or set(outputs) != names:
            raise ValueError("Predecessor native output binding inventory differs")
        directory = io.local_path(str(Path(path).parent), root=root)
        if ({entry.name for entry in directory.iterdir()} != names | {"receipt.json"}
                or io.active_bytes(directory) > CAPS["active_output_bytes"]):
            raise ValueError("Predecessor output directory inventory/cap differs")
        if io.sha_file(io.local_path(path, root=root), maximum=1024**2) != binding["sha256"]:
            raise ValueError("Predecessor receipt changed during read")
        for name in names:
            item = outputs[name]
            if (not isinstance(item, dict) or set(item) != {"bytes", "sha256"}
                    or type(item["bytes"]) is not int or not 0 < item["bytes"] <= CAPS["active_output_bytes"]
                    or not isinstance(item["sha256"], str) or not HEX.fullmatch(item["sha256"])):
                raise ValueError("Malformed predecessor native output binding")
            saved = io.local_path(str(Path(path).parent / name), root=root)
            if (saved.stat().st_size != item["bytes"]
                    or io.sha_file(saved, maximum=CAPS["active_output_bytes"]) != item["sha256"]):
                raise ValueError("Predecessor native output changed: " + name)
        elapsed_total += elapsed
        prior_hashes.append(binding["sha256"])
    if (len(prior_hashes) + CAPS["native_calls"] > AGGREGATE["maximum_native_calls"]
            or elapsed_total + CAPS["wall_seconds"] > AGGREGATE["aggregate_native_wall_seconds"]):
        raise ValueError("Remaining aggregate seven-call wall budget is insufficient")
    return {"sha256": prior_hashes, "elapsed_seconds": elapsed_total,
            "calls_consumed": len(prior_hashes)}


def validate_release(release: dict, *, root: Path = ROOT) -> dict:
    keys = {"schema", "status", "case_index", "case_id", "source_commit", "source_bindings",
            "preparation", "parent_declaration", "backend_profile", "runtime_identity",
            "deck_sha256", "output_directory", "prior_receipts"}
    if (not isinstance(release, dict) or set(release) != keys
            or release["schema"] != "nonpatient-sparse-continuation-one-call-release-v1"
            or release["status"] != "root_released_one_native_call"
            or type(release["case_index"]) is not int
            or release["case_index"] not in range(FIRST_INDEX, 7)
            or release["case_id"] != REMAINING[release["case_index"] - FIRST_INDEX]
            or release["output_directory"] != output_directory(release["case_id"])
            or not isinstance(release["deck_sha256"], str)
            or not HEX.fullmatch(release["deck_sha256"])):
        raise ValueError("Missing or altered one-case continuation release")
    prep = json.loads(io.binding_bytes(release["preparation"], PREPARATION, root=root))
    validate_preparation(prep)
    parent = json.loads(io.binding_bytes(release["parent_declaration"], PARENT, root=root))
    from scripts import mechanics_nonpatient_sparse_feasibility as design
    design.validate_declaration(parent, root=root)
    case_index = release["case_index"]
    case = parent["cases"][case_index]
    if (case["id"] != release["case_id"]
            or parent["caps"] != {"maximum_native_calls": 7, "attempts_per_case": 1,
                                  "automatic_retry_or_fallback": False, "numerical_threads": 1,
                                  "sampled_process_group_rss_bytes": CAPS["sampled_process_group_rss_bytes"],
                                  "per_call_wall_seconds": CAPS["wall_seconds"],
                                  "aggregate_native_wall_seconds": AGGREGATE["aggregate_native_wall_seconds"],
                                  "active_output_bytes": CAPS["active_output_bytes"]}
            or release["backend_profile"] != parent["runtime"]["backend_profile"]
            or release["runtime_identity"] != parent["runtime"]["runtime_identity"]):
        raise ValueError("Frozen case, parent caps, or repaired runtime differs")
    prior_parent = {"cases": parent["cases"],
                    "sha256": release["parent_declaration"]["sha256"],
                    "runtime_identity_sha256": release["runtime_identity"]["sha256"],
                    "backend_profile_sha256": release["backend_profile"]["sha256"]}
    prior = validate_prior_chain(release["prior_receipts"], case_index, prior_parent, root=root)
    bound = source_hashes(release, root=root)
    from scripts import mechanics_hbe_backend as backend
    runtime = backend.verify_profile(root, release["backend_profile"])
    audit_loaded_modules(root=root)
    if (runtime["profile_id"] != "accelerate_csc_v1"
            or runtime["runtime_identity"] != release["runtime_identity"]
            or runtime["runtime"]["executable_sha256"] !=
               runtime["runtime"]["libraries"]["install/bin/febio4"]):
        raise ValueError("Runtime identity or backend differs")
    from scripts import mechanics_nonpatient_sparse_deck as deck
    xml, metrics = deck.build_deck(case["id"])
    audit_loaded_modules(root=root)
    deck_bytes = xml.encode("utf-8")
    if io.sha_bytes(deck_bytes) != release["deck_sha256"] or metrics["case_id"] != case["id"]:
        raise ValueError("Released continuation deck differs")
    return {"runtime": runtime, "deck": deck_bytes, "prior": prior,
            "source_hashes": bound, "case_id": case["id"], "case_index": case_index,
            "output_directory": release["output_directory"], "prior_parent": prior_parent}


def supervise(case_id: str, executable: str, directory: Path, receipt: dict,
              *, rss_observer=None, popen=None, sleep=time.sleep) -> dict:
    """Start exactly one native process group; no retry or backend fallback."""
    if case_id not in REMAINING:
        raise ValueError("Undeclared continuation case")
    if rss_observer is None:
        from scripts.febio_runtime import process_group_rss
        rss_observer = process_group_rss
        audit_loaded_modules()
    if popen is None:
        popen = subprocess.Popen
    command = [executable, "-noconfig", "-no_title", "-i", case_id + ".feb", "-o", case_id + ".log"]
    receipt.update(command=command, native_calls_attempted=1, status="starting",
                   peak_sampled_process_group_rss_bytes=0,
                   peak_sampled_active_output_bytes=0, kill_reason=None, exit_code=None)
    io.durable_json(directory / "receipt.json", receipt)
    started = time.monotonic()
    process = None
    try:
        with (directory / "console.txt").open("xb") as log:
            process = popen(command, cwd=directory, env=io.private_environment(), stdout=log,
                            stderr=subprocess.STDOUT, start_new_session=True)
            receipt["pid"] = process.pid
            io.durable_json(directory / "receipt.json", receipt)
            while True:
                elapsed = time.monotonic() - started
                if elapsed >= CAPS["wall_seconds"]:
                    receipt["kill_reason"] = "wall_cap"
                    break
                code = process.poll()
                rss, members = rss_observer(process.pid, timeout_seconds=min(1., CAPS["wall_seconds"] - elapsed))
                active = io.active_bytes(directory)
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
                io.durable_json(directory / "receipt.json", receipt)
                sleep(io.POLL_SECONDS)
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
        io.durable_json(directory / "receipt.json", receipt)
    return receipt


def inspect_outputs(case_id: str, directory: Path, receipt: dict) -> dict:
    if case_id not in REMAINING or receipt["status"] != "native_exit_zero":
        raise ValueError("Wrong case or native process did not exit cleanly")
    if io.active_bytes(directory) > CAPS["active_output_bytes"]:
        raise ValueError("Final active output exceeds cap")
    names = {case_id + ".feb", case_id + ".log", case_id + ".nodes.log",
             case_id + ".elements.log", "console.txt", "receipt.json"}
    if {p.name for p in directory.iterdir()} != names:
        raise ValueError("Native output file inventory differs")
    texts, bindings = {}, {}
    for name in sorted(names - {"receipt.json"}):
        limit = 16 * 1024**2 if name in {case_id + ".log", "console.txt"} else CAPS["active_output_bytes"]
        texts[name], bindings[name] = io.read_output(directory / name, limit)
    selections = [line.strip() for line in texts["console.txt"].splitlines()
                  if "selecting linear solver" in line.lower()]
    forbidden = re.search(r"\b(?:fallback|fall\s+back|switching\s+(?:linear\s+)?solver|selected\s+linear\s+solver)\b",
                          texts["console.txt"], re.I)
    if len(selections) != 1 or not io.BACKEND_LINE.fullmatch(selections[0]) or forbidden:
        raise ValueError("Exactly one actual repaired Accelerate solver selection required")
    from scripts import mechanics_nonpatient_sparse_output as parser
    audit_loaded_modules()
    checked = parser.check_case_outputs(case_id, texts[case_id + ".nodes.log"],
                                        texts[case_id + ".elements.log"], texts[case_id + ".log"])
    if case_id == "n13_affine":
        from scripts import mechanics_nonpatient_sparse_affine_readout as affine
        readout_key = "affine_oracle"
        readout = affine.check_affine_readout(texts[case_id + ".nodes.log"],
                                              texts[case_id + ".elements.log"], case_id=case_id)
    else:
        from scripts import mechanics_nonpatient_sparse_nonuniform_readout as nonuniform
        readout_key = "nonuniform_readout"
        readout = nonuniform.check_nonuniform_readout(case_id, texts[case_id + ".nodes.log"],
                                                      texts[case_id + ".elements.log"], texts[case_id + ".log"])
    audit_loaded_modules()
    if (checked.get("passed") is not True or checked.get("case_id") != case_id
            or readout.get("passed") is not True or readout.get("case_id") != case_id):
        raise ValueError("Saved-output residual/field or case-specific readout failed")
    return {"output_bindings": bindings, "backend_selection": selections,
            "saved_output_checks": checked, readout_key: readout}


def execute(release_path: Path, *, root: Path = ROOT) -> dict:
    raw, identity = io.read_release(release_path)
    release = json.loads(raw)
    context = validate_release(release, root=root)
    case_id, case_index = context["case_id"], context["case_index"]
    directory = io.local_path(context["output_directory"], root=root)
    directory.mkdir(parents=True, exist_ok=False)
    receipt = {"schema": "nonpatient-sparse-continuation-one-call-receipt-v1", "status": "reserved",
               "case_id": case_id, "planned_case_index": case_index,
               "release_path": str(release_path.resolve()), "release_sha256": io.sha_bytes(raw),
               "source_commit": release["source_commit"], "source_hashes_before": context["source_hashes"],
               "parent_declaration_sha256": context["prior_parent"]["sha256"],
               "runtime_identity_sha256": release["runtime_identity"]["sha256"],
               "backend_profile_sha256": release["backend_profile"]["sha256"],
               "prior_receipt_sha256": context["prior"]["sha256"],
               "prior_native_wall_seconds": context["prior"]["elapsed_seconds"],
               "prior_native_calls_consumed": context["prior"]["calls_consumed"],
               "deck_sha256": release["deck_sha256"], "no_retry": True,
               "sampling_limit": "Brief between-sample RSS and output peaks may be missed."}
    io.durable_json(directory / "receipt.json", receipt)
    try:
        deck_path = directory / (case_id + ".feb")
        with deck_path.open("xb") as stream:
            stream.write(context["deck"])
            stream.flush()
            os.fsync(stream.fileno())
        if io.sha_file(deck_path) != release["deck_sha256"]:
            raise ValueError("Copied deck differs before native call")
        supervise(case_id, context["runtime"]["runtime"]["executable"], directory, receipt)
        receipt["aggregate_native_calls_attempted"] = (
            context["prior"]["calls_consumed"] + receipt["native_calls_attempted"])
        receipt["aggregate_native_wall_seconds_through_this_attempt"] = (
            context["prior"]["elapsed_seconds"] + receipt["elapsed_seconds"])
        if receipt["aggregate_native_wall_seconds_through_this_attempt"] > AGGREGATE["aggregate_native_wall_seconds"]:
            raise ValueError("Aggregate native wall budget exceeded")
        if receipt["status"] != "native_exit_zero":
            raise ValueError("Native call failed or exceeded resource cap")
        receipt.update(inspect_outputs(case_id, directory, receipt))
        io.binding_bytes(release["preparation"], PREPARATION, root=root)
        io.binding_bytes(release["parent_declaration"], PARENT, root=root)
        if (io.sha_file(deck_path) != release["deck_sha256"]
                or io.read_release(release_path) != (raw, identity)
                or validate_prior_chain(release["prior_receipts"], case_index,
                                        context["prior_parent"], root=root) != context["prior"]
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
            receipt["final_active_output_bytes"] = io.active_bytes(directory)
        except BaseException as error:
            receipt["status"] = "failed_or_incomplete"
            receipt["final_active_output_error"] = {"type": type(error).__name__, "message": str(error)}
        io.durable_json(directory / "receipt.json", receipt)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, required=True,
                        help="Separate immutable root-issued release for exactly one declared case")
    parser.add_argument("--execute", action="store_true", help="Explicitly spend this case's one native call")
    args = parser.parse_args()
    if not args.execute:
        raise ValueError("No native operation without explicit --execute and reviewed release")
    result = execute(args.release)
    print(json.dumps({"status": result["status"],
                      "receipt": str(io.local_path(output_directory(result["case_id"])) / "receipt.json")}))
    return 0 if result["status"] == "passed_numerical_software_only" else 1


if __name__ == "__main__":
    raise SystemExit(main())
