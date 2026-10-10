"""One-use, bounded, read-only audit of saved HBE v5 ordinal-10 outputs.

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

ROOT = Path(__file__).resolve().parents[2]
AUDIT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
NATIVE = ROOT / ("outputs/mechanics/hbe-v5-remaining-one-shot-v1/"
                 "10-tension-N24-S60-reference/attempt-01")
SIDECAR = ROOT / ("outputs/mechanics/hbe-v5-later-continuation-v1/"
                  "10-tension-N24-S60-reference/attempt-01")
INNER = ROOT / "build/hbe-v5-ordinal10-pending-release-v1/inner-release.json"
OUTER = ROOT / "build/hbe-v5-ordinal10-pending-release-v1/outer-envelope.json"
COMMIT = "19889bd8ca804f9c1a686a05b4ddc3ae56c0124b"
INNER_SHA = "316b6a866a0feebac0f9ae7342baa0ba4336f330cea39cf774d9a4c6ecc33449"
OUTER_SHA = "459adee627841eefb2bd8362db5171354131fc86e4c5336cff615ca8963d102b"
RUN_ID = "tension:N24:S60:reference"
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


def worker(result_path: Path) -> int:
    from scripts import mechanics_hbe_v5_stream as stream
    order = saved_json(NATIVE / "readout-work-order.json", MiB)
    saved = saved_json(NATIVE / "readout.json", 16 * MiB)
    result = stream.read_bound_run(ROOT, RUN_ID, order["bindings"],
                                   order["adapter_receipt"])
    if result != saved:
        raise ValueError("Independent complete-stream result differs from saved readout")
    if (result["frame_count"] != 61 or result["steps"] != 60
            or result["representation"] != "reconstructed_full"
            or result["numerical_passed"] is not True):
        raise ValueError("N24 complete-stream grammar or numerical verdict differs")
    compact = {
        "schema": "hbe-v5-ordinal10-independent-saved-replay-v1",
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
    with result_path.open("x", encoding="utf-8") as handle:
        json.dump(compact, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    return 0


def preflight(native_sha: str, sidecar_sha: str) -> dict:
    from scripts import mechanics_hbe_v5_remaining_one_shot as remaining
    from launchers import hbe_v5_later_continuation_runtime_v1 as runtime
    from launchers import hbe_v5_ordinal10_continuation_v1 as row10

    if (subprocess.check_output(["/usr/bin/git", "rev-parse", "HEAD"],
                                cwd=ROOT, text=True).strip() != COMMIT):
        raise ValueError("Execution source HEAD changed before independent audit")
    if (sha(INNER) != INNER_SHA or sha(OUTER) != OUTER_SHA):
        raise ValueError("Reviewed release bytes changed")
    runtime._read_release(ROOT, OUTER, row10.SPEC)
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
            or native.get("ordinal") != 10 or native.get("run_id") != RUN_ID
            or native.get("source_commit") != COMMIT
            or native.get("observed_head_preflight") != COMMIT
            or native.get("observed_head_after") != COMMIT
            or native.get("release_sha256") != INNER_SHA
            or native.get("native_calls_attempted") != 1
            or native.get("readout_calls_attempted") != 1
            or native.get("no_retry") is not True
            or native.get("caps") != remaining.caps(10)
            or native.get("aggregate_caps") != remaining.AGGREGATE):
        raise ValueError("Native terminal one-call identity differs")
    if (sidecar.get("schema") != "hbe-v5-later-continuation-sidecar-v1"
            or sidecar.get("status") !=
               "passed_numerical_software_only_with_cumulative_ledger_v1"
            or sidecar.get("ordinal") != 10 or sidecar.get("run_id") != RUN_ID
            or sidecar.get("source_commit") != COMMIT
            or sidecar.get("native_receipt_sha256") != native_sha
            or sidecar.get("envelope_sha256") != OUTER_SHA
            or sidecar.get("inner_release_sha256") != INNER_SHA
            or sidecar.get("native_calls_attempted") != 1
            or sidecar.get("hbe_readout_calls_attempted") != 1
            or sidecar.get("policy") != runtime.policy(row10.SPEC)):
        raise ValueError("Continuation sidecar identity differs")
    if {item.name for item in SIDECAR.iterdir()} != {"receipt.json"}:
        raise ValueError("Continuation sidecar has unexpected output")
    bindings = native.get("output_bindings")
    if not isinstance(bindings, dict) or set(NATIVE.iterdir()) != {
        NATIVE / name for name in (*bindings, "receipt.json")
    }:
        raise ValueError("Native closed output inventory differs")
    total_output_bytes = exact_regular(native_path, MiB)
    for name, binding in bindings.items():
        path = NATIVE / name
        size = exact_regular(path, remaining.caps(10)["active_output_bytes"])
        if (size != binding["bytes"] or sha(path) != binding["sha256"]
                or str(path.relative_to(ROOT)) != binding["path"]):
            raise ValueError(f"Native saved output binding differs: {name}")
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
            or order.get("ordinal") != 10 or order.get("run_id") != RUN_ID
            or readout.get("frame_count") != 61 or readout.get("steps") != 60
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
        ("preflight_hint_audit", 18, 2_908_379_069),
        ("final_hint_audit", 36, 5_816_758_138),
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
        raise ValueError("Row-10 sidecar output was not charged exactly once")
    aggregate = remaining._check_aggregate(
        sidecar["charged_previous"], native["native_stage"]["elapsed_seconds"],
        native["readout_stage"]["elapsed_seconds"],
        charge["launcher_inclusive_prep_seconds_with_finalization_reserve"],
        total_output_bytes + MiB, 1)
    if aggregate != charge["aggregate_with_extension"] or aggregate[
            "aggregate_native_calls"] != 11:
        raise ValueError("Row-10 cumulative accounting differs")
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
            "aggregate": aggregate}


def execute(native_sha: str, sidecar_sha: str) -> int:
    from scripts import mechanics_hbe_v5_remaining_one_shot as remaining
    from scripts import mechanics_hbe_v5_n8_one_shot as io
    if (AUDIT / "supervision").exists():
        raise ValueError("Independent replay is one-shot; prior supervision exists")
    info = preflight(native_sha, sidecar_sha)
    supervision = AUDIT / "supervision"
    supervision.mkdir(parents=True, exist_ok=False)
    result_path = supervision / "CHILD_RESULT.json"
    parent_status = {"schema": "hbe-v5-ordinal10-independent-parent-v1",
                     "status": "started", "native_calls_attempted": 0,
                     "official_hbe_readout_calls_attempted": 0,
                     "independent_replay_calls_attempted": 1,
                     "preflight": info}
    io.durable_json(AUDIT / "PARENT_STATUS.json", parent_status)
    stage_receipt = {"schema": "hbe-v5-ordinal10-independent-supervision-v1",
                     "status": "started", "independent_saved_replay_only": True,
                     "official_native_calls_attempted": 0,
                     "original_native_receipt_sha256": native_sha,
                     "original_readout_sha256": info["readout_sha256"]}
    environment = dict(os.environ)
    environment.update(PYTHONDONTWRITEBYTECODE="1",
                       PYTHONPYCACHEPREFIX=str(AUDIT / "fresh-nonexistent-pycache"),
                       OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                       MKL_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1")
    command = [sys.executable, "-B", str(Path(__file__).resolve()), "--worker",
               "--result", str(result_path)]
    try:
        stage = remaining.supervise_stage(
            "readout", command, supervision, stage_receipt,
            cwd=ROOT, environment=environment, wall_cap=60,
            rss_cap=3 * 1024**3, output_cap=16 * MiB)
        if stage["status"] != "completed_within_caps" or not result_path.exists():
            raise ValueError("Bounded independent saved-stream child failed")
        child = saved_json(result_path, MiB)
        if (child["exact_readout_equality"] is not True
                or child["readout_sha256"] != info["readout_sha256"]):
            raise ValueError("Independent child result does not match saved stream")
        parent_status.update(status="passed_independent_saved_replay",
                             child_result_sha256=sha(result_path),
                             supervision_receipt_sha256=sha(supervision / "receipt.json"),
                             child=child)
        return 0
    except BaseException as error:
        parent_status.update(status="failed_independent_saved_replay",
                             failure={"type": type(error).__name__,
                                      "message": str(error)[:500]})
        raise
    finally:
        io.durable_json(AUDIT / "PARENT_STATUS.json", parent_status)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--native-sha")
    parser.add_argument("--sidecar-sha")
    parser.add_argument("--result", type=Path)
    args = parser.parse_args()
    if args.worker and not args.execute and args.result is not None:
        return worker(args.result)
    if args.execute and not args.worker and args.native_sha and args.sidecar_sha:
        return execute(args.native_sha, args.sidecar_sha)
    raise ValueError("Explicit one-use audit mode and exact receipt hashes required")


if __name__ == "__main__":
    raise SystemExit(main())
