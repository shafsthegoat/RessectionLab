"""One-use, bounded, read-only audit of saved HBE v5 ordinal-11 outputs.

This ignored audit is never an FEBio launcher. It must run only after the root
launcher has terminated and an exact native/sidecar receipt pair is supplied.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
AUDIT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
NATIVE = ROOT / ("outputs/mechanics/hbe-v5-remaining-one-shot-v1/"
                 "11-tension-N24-S120-reference/attempt-01")
SIDECAR = ROOT / ("outputs/mechanics/hbe-v5-later-continuation-v1/"
                  "11-tension-N24-S120-reference/attempt-01")
INNER = ROOT / "build/hbe-v5-ordinal11-pending-release-v1/inner-release.json"
OUTER = ROOT / "build/hbe-v5-ordinal11-pending-release-v1/outer-envelope.json"
COMMIT = "47fa6d225cfe6fa150aa2c76f8b355f60e03a616"
INNER_SHA = "7f1211088385245676ece1c880860afe69d29654c809cf8a4c95c1a3adcb2711"
OUTER_SHA = "2fd42ed10c9e7d300cb6ce1d69e307107e9dfa400f374921dd74dc709169f10a"
RUN_ID = "tension:N24:S120:reference"
MiB = 1024**2


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(MiB), b""):
            digest.update(block)
    return digest.hexdigest()


def exact_regular(path: Path, maximum: int) -> int:
    observed = path.lstat()
    if path.is_symlink() or not stat.S_ISREG(observed.st_mode) or observed.st_size > maximum:
        raise ValueError(f"Saved file is linked, special, or oversized: {path}")
    return observed.st_size


def saved_json(path: Path, maximum: int) -> dict:
    exact_regular(path, maximum)
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise ValueError("Expected saved JSON object")
    return value


def worker(result_path: Path, native_sha: str, sidecar_sha: str, audit_sha: str) -> int:
    if sha(Path(__file__).resolve()) != audit_sha:
        raise ValueError("Independent audit source changed before worker")
    info = preflight(native_sha, sidecar_sha)
    from scripts import mechanics_hbe_v5_stream as stream
    order = saved_json(NATIVE / "readout-work-order.json", MiB)
    saved = saved_json(NATIVE / "readout.json", 16 * MiB)
    result = stream.read_bound_run(ROOT, RUN_ID, order["bindings"],
                                   order["adapter_receipt"])
    if result != saved:
        raise ValueError("Independent complete-stream result differs from saved readout")
    if (result["frame_count"] != 121 or result["steps"] != 120
            or result["representation"] != "reconstructed_full"
            or result["numerical_passed"] is not True):
        raise ValueError("N24 complete-stream grammar or numerical verdict differs")
    compact = {
        "schema": "hbe-v5-ordinal11-independent-saved-replay-v1",
        "exact_readout_equality": True,
        "readout_sha256": sha(NATIVE / "readout.json"),
        "frame_count": result["frame_count"],
        "steps": result["steps"],
        "representation": result["representation"],
        "numerical_passed": result["numerical_passed"],
        "minimum_sampled_J": result["minimum_sampled_J"],
        "largest_criterion_ratio": max(result["criteria_max_ratio"].values()),
        "final_force_N": result["applied_force_N"][-1],
        "final_load_m": result["load_coordinate_full_m"][-1],
        "provenance": {key: result["provenance"].get(key) for key in (
            "output_origin", "generated_fixture_only", "native_output_observed",
            "reconstruction_provenance")},
    }
    postguard(info, native_sha, sidecar_sha, audit_sha)
    compact["preflight"] = info
    compact["audit_source_sha256"] = audit_sha
    compact["postguards_passed"] = True
    with result_path.open("x", encoding="utf-8") as handle:
        json.dump(compact, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    return 0


def preflight(native_sha: str, sidecar_sha: str) -> dict:
    from scripts import mechanics_hbe_v5_remaining_one_shot as remaining
    from launchers import hbe_v5_later_continuation_runtime_v1 as runtime
    from launchers import hbe_v5_ordinal11_continuation_v1 as row11

    if (subprocess.check_output(["/usr/bin/git", "rev-parse", "HEAD"],
                                cwd=ROOT, text=True).strip() != COMMIT):
        raise ValueError("Execution source HEAD changed before independent audit")
    if (sha(INNER) != INNER_SHA or sha(OUTER) != OUTER_SHA):
        raise ValueError("Reviewed release bytes changed")
    runtime._read_release(ROOT, OUTER, row11.SPEC)
    for directory in (NATIVE, SIDECAR):
        if directory.is_symlink() or not stat.S_ISDIR(directory.lstat().st_mode):
            raise ValueError("Saved attempt directory is absent, linked, or special")
    native_path = NATIVE / "receipt.json"
    sidecar_path = SIDECAR / "receipt.json"
    native = saved_json(native_path, MiB)
    sidecar = saved_json(sidecar_path, MiB)
    if sha(native_path) != native_sha or sha(sidecar_path) != sidecar_sha:
        raise ValueError("Exact terminal receipt hash differs")
    if (native.get("status") != "passed_numerical_software_only"
            or native.get("ordinal") != 11 or native.get("run_id") != RUN_ID
            or native.get("source_commit") != COMMIT
            or native.get("observed_head_preflight") != COMMIT
            or native.get("observed_head_after") != COMMIT
            or native.get("release_sha256") != INNER_SHA
            or native.get("native_calls_attempted") != 1
            or native.get("readout_calls_attempted") != 1
            or native.get("no_retry") is not True
            or native.get("caps") != remaining.caps(11)
            or native.get("aggregate_caps") != remaining.AGGREGATE):
        raise ValueError("Native terminal one-call identity differs")
    if (sidecar.get("schema") != "hbe-v5-later-continuation-sidecar-v1"
            or sidecar.get("status") !=
               "passed_numerical_software_only_with_cumulative_ledger_v1"
            or sidecar.get("ordinal") != 11 or sidecar.get("run_id") != RUN_ID
            or sidecar.get("source_commit") != COMMIT
            or sidecar.get("native_receipt_sha256") != native_sha
            or sidecar.get("envelope_sha256") != OUTER_SHA
            or sidecar.get("inner_release_sha256") != INNER_SHA
            or sidecar.get("native_calls_attempted") != 1
            or sidecar.get("hbe_readout_calls_attempted") != 1
            or sidecar.get("policy") != runtime.policy(row11.SPEC)):
        raise ValueError("Continuation sidecar identity differs")
    if {item.name for item in SIDECAR.iterdir()} != {"receipt.json"}:
        raise ValueError("Continuation sidecar has unexpected output")
    bindings = native.get("output_bindings")
    if not isinstance(bindings, dict) or set(NATIVE.iterdir()) != {
        NATIVE / name for name in (*bindings, "receipt.json")
    }:
        raise ValueError("Native closed output inventory differs")
    total_output_bytes = exact_regular(native_path, MiB)
    output_stat_guards = {}
    for name, binding in bindings.items():
        path = NATIVE / name
        size = exact_regular(path, remaining.caps(11)["active_output_bytes"])
        before = path.lstat()
        if (size != binding["bytes"] or sha(path) != binding["sha256"]
                or str(path.relative_to(ROOT)) != binding["path"]):
            raise ValueError(f"Native saved output binding differs: {name}")
        after = path.lstat()
        observed = lambda st: [st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns]
        if observed(before) != observed(after):
            raise ValueError("Saved output changed during checksum")
        output_stat_guards[name] = observed(after)
        total_output_bytes += size
    if (native.get("source_hashes_before") != native.get("source_hashes_after")
            or len(native["source_hashes_after"]) != 20):
        raise ValueError("Frozen source closure differs after native call")
    for path, digest in native["source_hashes_after"].items():
        exact_regular(ROOT / path, MiB)
        if sha(ROOT / path) != digest:
            raise ValueError("Frozen source bytes changed after native call")
    extension = sidecar.get("extension_sources_before")
    if (extension != sidecar.get("extension_sources_after")
            or extension.get("source_hashes") !=
               sidecar.get("extension_sources_terminal", {}).get("source_hashes")
            or len(extension["source_hashes"]) != 9):
        raise ValueError("Extension source closure differs after native call")
    for path, digest in extension["source_hashes"].items():
        exact_regular(ROOT / path, MiB)
        if sha(ROOT / path) != digest:
            raise ValueError("Extension source bytes changed after native call")
    order = saved_json(NATIVE / "readout-work-order.json", MiB)
    readout = saved_json(NATIVE / "readout.json", 16 * MiB)
    if (order.get("source_commit") != COMMIT or order.get("release_sha256") != INNER_SHA
            or order.get("ordinal") != 11 or order.get("run_id") != RUN_ID
            or readout.get("frame_count") != 121 or readout.get("steps") != 120
            or readout.get("representation") != "reconstructed_full"
            or readout.get("numerical_passed") is not True
            or readout.get("provenance", {}).get("output_origin") !=
               "unverified_saved_stream"):
        raise ValueError("Saved readout/work-order identity differs")
    for name in ("nodes", "elements", "solver"):
        bound = native["native_output_bindings"][name + ".log"]
        if any(order["bindings"][name][key] != bound[key]
               for key in ("path", "sha256")):
            raise ValueError("Saved work-order primitive binding differs")
    adapted = native["native_output_bindings"]["specimen.feb"]
    if any(order["bindings"]["adapted_deck"][key] != adapted[key]
           for key in ("path", "sha256")):
        raise ValueError("Saved work-order adapted deck binding differs")
    for stage in ("native", "readout"):
        item = native[stage + "_stage"]
        owned = sidecar["owned_stage_cleanup"][stage]
        if (item["status"] != "completed_within_caps" or item["exit_code"] != 0
                or item["kill_reason"] is not None
                or item["elapsed_seconds"] >= item["wall_cap_seconds"]
                or item["peak_sampled_process_group_rss_bytes"] >
                   item["sampled_process_group_rss_cap_bytes"]
                or owned.get("contained") is not True
                or owned.get("direct_child_reaped") is not True
                or owned.get("fallback_used") is not False
                or owned.get("remaining_members") != []
                or owned.get("errors") != []):
            raise ValueError("Native/readout resource or owned-child cleanup differs")
    for key, count, size in (
        ("preflight_hint_audit", 20, 3_101_186_843),
        ("final_hint_audit", 40, 6_202_373_686),
    ):
        value = sidecar[key]
        if (value["eligible_files"] != count or value["hinted_file_opens"] != count
                or value["eligible_bytes"] != size):
            raise ValueError("No-cache hint coverage differs")
    if any(sidecar[key]["kernel_pressure_mask"] != 1 or
           sidecar[key]["available_percent"] < floor for key, floor in (
               ("host_before_validation", 54),
               ("host_before_native_reservation", 45),
               ("host_immediately_before_native_supervision", 45))):
        raise ValueError("Saved host floor differs")
    charge = sidecar["extension_resource_charge"]
    if (charge["native_closed_output_bytes_at_check"] != total_output_bytes
            or charge["sidecar_reserved_output_bytes"] != MiB
            or charge["charged_current_output_bytes"] != total_output_bytes + MiB):
        raise ValueError("Row-11 sidecar output was not charged exactly once")
    aggregate = remaining._check_aggregate(
        sidecar["charged_previous"], native["native_stage"]["elapsed_seconds"],
        native["readout_stage"]["elapsed_seconds"],
        charge["launcher_inclusive_prep_seconds_with_finalization_reserve"],
        total_output_bytes + MiB, 1)
    if aggregate != charge["aggregate_with_extension"] or aggregate[
            "aggregate_native_calls"] != 12:
        raise ValueError("Row-11 cumulative accounting differs")
    return {"native_receipt_sha256": native_sha, "sidecar_sha256": sidecar_sha,
            "source_commit": COMMIT, "native_output_file_count": len(bindings),
            "native_closed_output_bytes": total_output_bytes,
            "readout_sha256": sha(NATIVE / "readout.json"),
            "native_stage": native["native_stage"],
            "readout_stage": native["readout_stage"],
            "sidecar_status": sidecar["status"],
            "host_available_percent": [sidecar[key]["available_percent"] for key in (
                "host_before_validation", "host_before_native_reservation",
                "host_immediately_before_native_supervision")],
            "charged_row_output_bytes": total_output_bytes + MiB,
            "aggregate": aggregate,
            "output_stat_guards": output_stat_guards,
            "frozen_source_hashes": native["source_hashes_after"],
            "extension_source_hashes": extension["source_hashes"]}



def postguard(info: dict, native_sha: str, sidecar_sha: str, audit_sha: str) -> None:
    if sha(Path(__file__).resolve()) != audit_sha:
        raise ValueError("Independent audit source changed during worker")
    if subprocess.check_output(["/usr/bin/git", "rev-parse", "HEAD"],
                               cwd=ROOT, text=True).strip() != COMMIT:
        raise ValueError("Frozen HEAD changed during saved replay")
    if sha(INNER) != INNER_SHA or sha(OUTER) != OUTER_SHA:
        raise ValueError("Release bytes changed during saved replay")
    if (sha(NATIVE / "receipt.json") != native_sha
            or sha(SIDECAR / "receipt.json") != sidecar_sha
            or sha(NATIVE / "readout.json") != info["readout_sha256"]):
        raise ValueError("Saved receipt/readout changed during replay")
    if {x.name for x in NATIVE.iterdir()} != set(info["output_stat_guards"]) | {"receipt.json"}:
        raise ValueError("Native final inventory changed during replay")
    for name, expected in info["output_stat_guards"].items():
        path = NATIVE / name
        st = path.lstat()
        if (path.is_symlink() or not stat.S_ISREG(st.st_mode)
                or [st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns] != expected):
            raise ValueError("Saved output stat identity changed during replay")
    for bindings in (info["frozen_source_hashes"], info["extension_source_hashes"]):
        for relative, expected in bindings.items():
            exact_regular(ROOT / relative, MiB)
            if sha(ROOT / relative) != expected:
                raise ValueError("Executing source changed during saved replay")


def execute(native_sha: str, sidecar_sha: str, audit_sha: str) -> int:
    from scripts import mechanics_hbe_v5_remaining_one_shot as remaining
    from scripts import mechanics_hbe_v5_n8_one_shot as io
    from scripts import febio_runtime
    from launchers import hbe_v5_ordinal9_continuation_v1 as ownership
    if sha(Path(__file__).resolve()) != audit_sha:
        raise ValueError("Root-selected independent audit source differs")
    for digest in (native_sha, sidecar_sha, audit_sha):
        if len(digest) != 64 or any(x not in "0123456789abcdef" for x in digest):
            raise ValueError("Exact terminal SHA-256 required")
    supervision = AUDIT / "supervision"
    if supervision.exists() or supervision.is_symlink():
        raise ValueError("Independent replay is one-shot; prior supervision exists")
    # Full output checks and unchanged numerical replay share one bounded child.
    supervision.mkdir(parents=True, exist_ok=False)
    result_path = supervision / "CHILD_RESULT.json"
    parent_status = {"schema": "hbe-v5-ordinal11-independent-parent-v1",
                     "status": "started", "native_calls_attempted": 0,
                     "official_hbe_readout_calls_attempted": 0,
                     "independent_replay_calls_attempted": 1,
                     "audit_source_sha256": audit_sha,
                     "original_native_receipt_sha256": native_sha,
                     "original_sidecar_sha256": sidecar_sha}
    io.durable_json(AUDIT / "PARENT_STATUS.json", parent_status)
    stage_receipt = {"schema": "hbe-v5-ordinal11-independent-supervision-v1",
                     "status": "started", "independent_saved_replay_only": True,
                     "official_native_calls_attempted": 0,
                     "original_native_receipt_sha256": native_sha,
                     "original_sidecar_sha256": sidecar_sha,
                     "audit_source_sha256": audit_sha}
    environment = dict(os.environ)
    environment.update(PYTHONDONTWRITEBYTECODE="1",
                       PYTHONPYCACHEPREFIX=str(AUDIT / "fresh-nonexistent-pycache"),
                       OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                       MKL_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1")
    command = [sys.executable, "-B", str(Path(__file__).resolve()), "--worker",
               "--result", str(result_path), "--native-sha", native_sha,
               "--sidecar-sha", sidecar_sha, "--audit-sha", audit_sha]
    owner = ownership.OwnedStage()
    stage = None
    try:
        try:
            stage = remaining.supervise_stage(
                "readout", command, supervision, stage_receipt,
                cwd=ROOT, environment=environment, wall_cap=600,
                rss_cap=3 * 1024**3, output_cap=16 * MiB,
                rss_observer=febio_runtime.process_group_rss, popen=owner)
        finally:
            try:
                cleanup = owner.cleanup(febio_runtime.process_group_rss,
                                        time.monotonic() + 3.)
            except BaseException as error:
                cleanup = {"contained": False, "direct_child_reaped": False,
                           "fallback_used": True, "remaining_members": [],
                           "errors": ["cleanup_exception:" + type(error).__name__]}
            parent_status["owned_child_cleanup"] = cleanup
            stage_receipt["owned_child_cleanup"] = cleanup
            stage_receipt["status"] = "terminal_pending_independent_checks"
            io.durable_json(supervision / "receipt.json", stage_receipt)
        if (not cleanup["contained"] or not cleanup["direct_child_reaped"]
                or cleanup["fallback_used"] or cleanup["errors"]
                or cleanup.get("remaining_members") != []):
            raise ValueError("Independent child cleanup failed")
        if (stage is None or stage["status"] != "completed_within_caps"
                or not result_path.exists()):
            raise ValueError("Bounded independent saved-stream child failed")
        child = saved_json(result_path, MiB)
        info = child["preflight"]
        if (child["exact_readout_equality"] is not True
                or child["readout_sha256"] != info["readout_sha256"]
                or child["audit_source_sha256"] != audit_sha
                or child["postguards_passed"] is not True):
            raise ValueError("Independent child result does not match saved stream")
        postguard(info, native_sha, sidecar_sha, audit_sha)
        if remaining.active_bytes(supervision, 16 * MiB) > 16 * MiB - 65536:
            raise ValueError("Independent closed audit output cap exceeded")
        stage_receipt["status"] = "passed_independent_saved_replay"
        io.durable_json(supervision / "receipt.json", stage_receipt)
        parent_status.update(status="passed_independent_saved_replay",
                             preflight=info,
                             child_result_sha256=sha(result_path),
                             supervision_receipt_sha256=sha(supervision / "receipt.json"),
                             child=child)
        return 0
    except BaseException as error:
        stage_receipt["status"] = "failed_independent_saved_replay"
        io.durable_json(supervision / "receipt.json", stage_receipt)
        parent_status.update(status="failed_independent_saved_replay",
                             failure={"type": type(error).__name__,
                                      "message": str(error)[:500]},
                             supervision_receipt_sha256=sha(supervision / "receipt.json"))
        raise
    finally:
        io.durable_json(AUDIT / "PARENT_STATUS.json", parent_status)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--native-sha")
    parser.add_argument("--sidecar-sha")
    parser.add_argument("--audit-sha")
    parser.add_argument("--result", type=Path)
    args = parser.parse_args()
    if args.worker and not args.execute and args.result is not None and args.audit_sha:
        return worker(args.result, args.native_sha, args.sidecar_sha, args.audit_sha)
    if args.execute and not args.worker and args.native_sha and args.sidecar_sha and args.audit_sha:
        return execute(args.native_sha, args.sidecar_sha, args.audit_sha)
    raise ValueError("Explicit one-use audit mode and exact receipt hashes required")


if __name__ == "__main__":
    raise SystemExit(main())

