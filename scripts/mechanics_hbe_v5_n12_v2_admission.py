"""Exact ordinal-8 v2 sidecar admission before a future ordinal-9 release.

Ignored proposal only. This never runs a solver or grants a release. The old
remaining-row validator must still rehash all predecessor outputs separately.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import stat
import subprocess

from launchers import hbe_v5_nocache_v2 as launcher
from scripts import mechanics_hbe_v5_remaining_one_shot as remaining


ROOT = Path(__file__).resolve().parents[1]  # Intended scripts/ location after promotion.
RUN_COMMIT = "0ece80d179e28f183b379ba2298961d0a2fd2c18"
INNER = "build/hbe-v5-tension-n12-s60-nocache-v2-final-candidate/inner-release.json"
OUTER = "build/hbe-v5-tension-n12-s60-nocache-v2-final-candidate/outer-envelope.json"
NATIVE = remaining.receipt_path(8)
SIDECAR = launcher.SIDECAR + "/receipt.json"
REVIEW = "artifacts/hbe-v5-tension-n12-nocache-v2-result-v1/RESULT_METADATA.json"
REVIEW_REPORT = "artifacts/hbe-v5-tension-n12-nocache-v2-result-v1/RESULT.md"
REVIEW_REPORT_SHA = "6fb89ad0e7b132796554f285ece490d5a20c45c9f031a067174d7b8ca4abd56e"
SHA = {
    INNER: "7f41865a1bfef4a785b4be262832af6d92d79472abd808a905773dc7ddc181d8",
    OUTER: "2fc24ff3c33923ba33e16ae0842ceb97c1d89b6ede74b2b0e733b0127f74feb7",
    NATIVE: "67f7b82dca721d6d36b16e80e77c5d126b0ea9f9f510f8c2dd54b64716206a8e",
    SIDECAR: "dae900fcedd2dc61f1398a00e40be40b9fa355146c20599c6f01fc5a29b45eb8",
    REVIEW: "6c048fcaa61c19b65a9f3799e69f3bf5b0bc81f2a0bdc890aa90cfb77be81345",
}
SOURCE_HASHES = {
    "launchers/hbe_v5_nocache_v2.py":
        "4271cb49de70a63de5bf8369628208d8b90907434deb6324a16a7b36c314790f",
    "launchers/hbe_v5_nocache_hash_v1.py":
        "96c71d0d9b8d896b713aef7305e3e86d5b12b9d2dca5fcc55515f6b2eb9cf3fc",
    "launchers/hbe_v5_nocache_host_v1.py":
        "2ff8f3e4d30ed0a8976fb01da99403ff5b2b2ffd6e750791fe7a954946d0a00f",
}


def _linked_under(root: Path, path: Path) -> bool:
    current = root
    for part in path.relative_to(root).parts:
        current /= part
        if current.is_symlink():
            return True
    return False


def _small_bytes(root: Path, relative: str, digest: str,
                 maximum: int = 1024**2) -> bytes:
    """Stat-bound exact small read; never opens native meshes/streams."""
    path = root / relative
    if not path.is_relative_to(root) or _linked_under(root, path):
        raise ValueError("Bound admission path is outside root or linked")
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or before.st_size > maximum:
        raise ValueError("Bound admission file absent, special or oversized")
    raw = path.read_bytes()
    after = path.lstat()
    ident = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns)
    if ident(before) != ident(after) or hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError("Bound admission bytes changed")
    return raw


def _small_exact(root: Path, relative: str, digest: str, maximum: int = 1024**2) -> dict:
    value = json.loads(_small_bytes(root, relative, digest, maximum))
    if not isinstance(value, dict):
        raise ValueError("Bound admission JSON must be an object")
    return value


def _historical_sources(root: Path) -> None:
    if set(SOURCE_HASHES) != set(launcher.SOURCES):
        raise ValueError("Exact v2 extension source closure changed")
    for relative, digest in SOURCE_HASHES.items():
        path = root / relative
        if (_linked_under(root, path) or not stat.S_ISREG(path.lstat().st_mode)
                or path.stat().st_size > 128 * 1024):
            raise ValueError("Historical v2 source path is not regular")
        before = path.lstat()
        raw = path.read_bytes()
        after = path.lstat()
        committed = subprocess.run(
            ["/usr/bin/git", "cat-file", "blob", RUN_COMMIT + ":" + relative],
            cwd=root, capture_output=True, check=True, timeout=10).stdout
        ident = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns)
        if (ident(before) != ident(after) or raw != committed
                or hashlib.sha256(raw).hexdigest() != digest):
            raise ValueError("Historical v2 source working/Git bytes differ")


def _normal_host(snapshot: dict, floor: int) -> None:
    if (not isinstance(snapshot, dict) or snapshot.get("kernel_pressure_mask") != 1
            or type(snapshot.get("available_percent")) is not int
            or snapshot["available_percent"] < floor):
        raise ValueError("Historical v2 host gate did not pass")


def _hints(sidecar: dict, root: Path) -> None:
    policy = launcher.POLICY
    before = sidecar.get("preflight_hint_audit", {})
    final = sidecar.get("final_hint_audit", {})
    for value, count, bytes_ in (
        (before, policy["expected_preflight_hint_opens"],
         policy["expected_preflight_hint_bytes"]),
        (final, policy["expected_completed_hint_opens"],
         policy["expected_completed_hint_bytes"]),
    ):
        if (value.get("eligible_files") != count
                or value.get("hinted_file_opens") != count
                or value.get("eligible_bytes") != bytes_
                or not isinstance(value.get("hinted_file_paths"), list)
                or len(value["hinted_file_paths"]) != count):
            raise ValueError("Historical predecessor hint coverage differs")
    paths = before["hinted_file_paths"]
    if final["hinted_file_paths"] != paths + paths:
        raise ValueError("Pre/post predecessor hint paths differ")
    allowed = {root / str(Path(remaining.receipt_path(i)).parent) / name
               for i in range(1, 7) for name in ("nodes.log", "elements.log")}
    if any(not isinstance(path, str) or Path(path) not in allowed for path in paths):
        raise ValueError("Hinted path is not a declared predecessor grid file")


def _resource_charge(sidecar: dict, native: dict) -> None:
    charge = sidecar.get("extension_resource_charge", {})
    previous = sidecar.get("old_validation_identity", {}).get("previous")
    if not isinstance(previous, dict):
        raise ValueError("Original predecessor ledger missing from sidecar")
    for field, cap, strict in (
        ("extension_before_old_execute_wall_seconds",
         launcher.POLICY["extension_before_old_execute_wall_seconds"], True),
        ("extension_after_old_execute_wall_seconds",
         launcher.POLICY["extension_after_old_execute_wall_seconds"], True),
        ("extension_self_peak_rss_bytes_before",
         launcher.POLICY["extension_self_peak_rss_bytes"], False),
        ("extension_self_peak_rss_bytes_after",
         launcher.POLICY["extension_self_peak_rss_bytes"], False),
        ("extension_self_peak_rss_bytes_terminal",
         launcher.POLICY["extension_self_peak_rss_bytes"], False),
    ):
        value = sidecar.get(field)
        if (type(value) not in (int, float) or not math.isfinite(value)
                or value < 0 or (value >= cap if strict else value > cap)):
            raise ValueError("V2 extension wall or RSS cap differs: " + field)
    native_time = native.get("native_stage", {}).get("elapsed_seconds")
    readout_time = native.get("readout_stage", {}).get("elapsed_seconds")
    prep = charge.get("launcher_inclusive_prep_seconds_with_finalization_reserve")
    closed = charge.get("native_closed_output_bytes_at_check")
    total = charge.get("charged_current_output_bytes")
    if (any(type(x) not in (int, float) or not math.isfinite(x) or x < 0
            for x in (native_time, readout_time, prep, closed, total))
            or prep >= remaining.PREP_WALL
            or charge.get("native_stage_seconds") != native_time
            or charge.get("readout_stage_seconds") != readout_time
            or charge.get("per_row_prep_wall_seconds") != remaining.PREP_WALL
            or charge.get("sidecar_reserved_output_bytes") !=
               launcher.POLICY["sidecar_output_cap_bytes"]
            or total != closed + launcher.POLICY["sidecar_output_cap_bytes"]
            or sidecar.get("launcher_elapsed_seconds_terminal", 0) - native_time -
               readout_time > prep):
        raise ValueError("V2 wrapper resource charge differs")
    expected = remaining._check_aggregate(
        previous, native_time, readout_time, prep, total, 1)
    if charge.get("aggregate_with_extension") != expected:
        raise ValueError("V2 aggregate charge differs from frozen cap arithmetic")
    for stage in ("native", "readout"):
        cleanup = sidecar.get("owned_stage_cleanup", {}).get(stage, {})
        if (cleanup.get("contained") is not True
                or cleanup.get("direct_child_reaped") is not True
                or cleanup.get("fallback_used") is not False
                or cleanup.get("errors") != []
                or cleanup.get("remaining_members") != []
                or cleanup.get("exit_code") != 0):
            raise ValueError("V2 owned stage cleanup differs")


def _surcharge(sidecar: dict, native: dict) -> dict:
    charged_prep = sidecar["extension_resource_charge"][
        "launcher_inclusive_prep_seconds_with_finalization_reserve"]
    old_prep = native["prep_elapsed_seconds"]
    if (type(old_prep) not in (int, float) or not math.isfinite(old_prep)
            or old_prep < 0 or charged_prep < old_prep):
        raise ValueError("V2 inclusive preparation is below old native receipt")
    return {"prep_wall_seconds": charged_prep - old_prep,
            "output_bytes": launcher.POLICY["sidecar_output_cap_bytes"],
            "source_ordinal8_native_receipt_sha256": SHA[NATIVE],
            "source_ordinal8_v2_sidecar_sha256": SHA[SIDECAR]}


def _arithmetic_surcharge(previous: dict, exact: dict, index: int) -> dict:
    """Diagnostic arithmetic for one ordinal-8 charge across future baselines.

    Every future old-chain baseline is reconstructed without this sidecar, so
    add the surcharge to that baseline, never to an earlier adjusted ledger.
    No old native receipt is rewritten or promoted by this calculation. Later
    rows additionally need binders for each newer wrapper's own overhead.
    """
    if (type(index) is not int or index not in (9, 10, 11)
            or not isinstance(previous, dict)
            or "v2_surcharge_applied" in previous
            or previous.get("native_calls") != index
            or not isinstance(previous.get("sha256"), list)
            or len(previous["sha256"]) != index
            or previous["sha256"][8] != SHA[NATIVE]):
        raise ValueError("Exact uncharged old predecessor ledger required")
    surcharge = exact.get("surcharge")
    if (not isinstance(surcharge, dict)
            or surcharge.get("source_ordinal8_native_receipt_sha256") != SHA[NATIVE]
            or surcharge.get("source_ordinal8_v2_sidecar_sha256") != SHA[SIDECAR]
            or type(surcharge.get("prep_wall_seconds")) not in (int, float)
            or not math.isfinite(surcharge["prep_wall_seconds"])
            or surcharge["prep_wall_seconds"] < 0
            or surcharge.get("output_bytes") != launcher.POLICY["sidecar_output_cap_bytes"]):
        raise ValueError("Authenticated v2 surcharge required")
    prep_delta = surcharge["prep_wall_seconds"]
    output_delta = surcharge["output_bytes"]
    aggregate = remaining._check_aggregate(previous, 0., 0., prep_delta,
                                           output_delta, 0)
    adjusted = dict(previous)
    adjusted.update({"prep_seconds": aggregate["aggregate_prep_wall_seconds"],
                     "output_bytes": aggregate["aggregate_output_bytes"],
                     "combined_wall_seconds": aggregate["aggregate_combined_wall_seconds"],
                     "combined_output_bytes": aggregate["aggregate_combined_output_bytes"],
                     "v2_surcharge_applied": dict(surcharge)})
    current = remaining.caps(index)
    limits = remaining.AGGREGATE
    if (adjusted["native_seconds"] + current["native_wall_seconds"] >
            limits["native_wall_seconds_all_12"]
            or adjusted["readout_seconds"] + remaining.READOUT_WALL >
               limits["remaining_readout_wall_seconds"]
            or adjusted["prep_seconds"] + remaining.PREP_WALL >
               limits["remaining_prep_wall_seconds"]
            or adjusted["native_seconds"] + adjusted["readout_seconds"] +
               adjusted["prep_seconds"] + current["native_wall_seconds"] +
               remaining.READOUT_WALL + remaining.PREP_WALL >
               limits["known_stage_total_wall_seconds"]
            or adjusted["output_bytes"] + current["active_output_bytes"] >
               limits["active_output_bytes_all_12"]):
        raise ValueError("V2-charged ledger cannot afford the next frozen row")
    return {"old_native_receipt_ledger": dict(previous),
            "authenticated_ordinal8_surcharge": dict(surcharge),
            "charged_cumulative_ledger": adjusted,
            "aggregate_with_surcharge": aggregate,
            "interpretation": "Separate supplemental accounting; old native receipts stay unchanged."}


def continuation_ledger(previous: dict, exact: dict, index: int) -> dict:
    """Admit ordinal-8 surcharge arithmetic only for immediate row 9.

    Rows 10/11 must have their intervening wrapper charges independently
    authenticated before this can become an executable continuation gate.
    """
    if index != 9:
        raise ValueError("Later wrapper charges are not yet authenticated")
    return _arithmetic_surcharge(previous, exact, index)


def _saved_fields(sidecar: dict, native: dict, inner: dict, outer: dict,
                  reviewed: dict, root: Path) -> dict:
    """Check semantic links after the independent exact SHA pins are satisfied."""
    if (outer.get("schema") != "hbe-v5-nocache-launch-envelope-v2"
            or outer.get("status") !=
               "root_released_one_native_call_with_declared_io_policy_v2"
            or outer.get("source_commit") != RUN_COMMIT
            or outer.get("extension_source_bindings") != SOURCE_HASHES
            or outer.get("policy") != launcher.POLICY
            or outer.get("sidecar_directory") != launcher.SIDECAR
            or outer.get("inner_release") != {"path": INNER, "sha256": SHA[INNER]}
            or inner.get("schema") != "hbe-v5-remaining-one-call-release-v1"
            or inner.get("status") != "root_released_one_native_call"
            or inner.get("source_commit") != RUN_COMMIT
            or inner.get("ordinal") != 8 or inner.get("run_id") != remaining.ORDER[8]
            or inner.get("output_directory") != launcher.NATIVE_OUTPUT
            or inner.get("caps") != remaining.caps(8)
            or inner.get("aggregate_caps") != remaining.AGGREGATE):
        raise ValueError("Released v2/old ordinal-8 identity differs")
    if (native.get("schema") != "hbe-v5-remaining-one-call-receipt-v1"
            or native.get("status") != "passed_numerical_software_only"
            or native.get("ordinal") != 8 or native.get("run_id") != remaining.ORDER[8]
            or native.get("release_path") != str(root / INNER)
            or native.get("release_sha256") != SHA[INNER]
            or native.get("source_commit") != RUN_COMMIT
            or native.get("native_calls_attempted") != 1
            or native.get("readout_calls_attempted") != 1
            or native.get("no_retry") is not True
            or native.get("caps") != remaining.caps(8)
            or native.get("aggregate_caps") != remaining.AGGREGATE
            or native.get("source_hashes_before") != native.get("source_hashes_after")
            or native.get("saved_numerical_readout", {}).get("numerical_passed") is not True
            or native.get("saved_numerical_readout", {}).get("physical_validation_pass") is not None):
        raise ValueError("Native ordinal-8 numerical receipt differs")
    expected_sources = {name: binding["sha256"]
                        for name, binding in inner["source_bindings"].items()}
    if native.get("source_hashes_before") != expected_sources:
        raise ValueError("Original HBE source hashes differ from release")
    identity = {"source_commit": RUN_COMMIT, "observed_head": RUN_COMMIT,
                "source_hashes": SOURCE_HASHES}
    if (sidecar.get("schema") != "hbe-v5-nocache-policy-sidecar-v2"
            or sidecar.get("status") !=
               "passed_numerical_software_only_with_declared_io_policy_v2"
            or sidecar.get("source_commit") != RUN_COMMIT
            or sidecar.get("policy") != launcher.POLICY
            or sidecar.get("envelope_sha256") != SHA[OUTER]
            or sidecar.get("inner_release_sha256") != SHA[INNER]
            or sidecar.get("native_output_directory") != launcher.NATIVE_OUTPUT
            or sidecar.get("native_receipt_sha256") != SHA[NATIVE]
            or sidecar.get("native_receipt_status") != native["status"]
            or sidecar.get("native_calls_authorized") != 1
            or sidecar.get("native_calls_attempted") != 1
            or sidecar.get("hbe_readout_calls_attempted") != 1
            or sidecar.get("extension_sources_before") != identity
            or sidecar.get("extension_sources_after") != identity):
        raise ValueError("V2 sidecar does not bind the exact native result")
    _normal_host(sidecar.get("host_before_validation"), 54)
    _normal_host(sidecar.get("host_before_native_reservation"), 45)
    _normal_host(sidecar.get("host_immediately_before_native_supervision"), 45)
    _hints(sidecar, root)
    _resource_charge(sidecar, native)
    if (reviewed.get("schema") != "hbe-v5-n12-v2-independent-saved-result-metadata-v1"
            or reviewed.get("status") != "independent_saved_numerical_software_pass"
            or reviewed.get("ordinal") != 8
            or reviewed.get("run_id") != remaining.ORDER[8]
            or reviewed.get("source_commit") != RUN_COMMIT
            or reviewed.get("native_receipt_sha256") != SHA[NATIVE]
            or reviewed.get("sidecar_receipt_sha256") != SHA[SIDECAR]
            or reviewed.get("released_inner_sha256") != SHA[INNER]
            or reviewed.get("released_outer_sha256") != SHA[OUTER]
            or reviewed.get("readout_sha256") !=
               native.get("saved_numerical_readout", {}).get("readout_sha256")
            or reviewed.get("output_bindings") != native.get("output_bindings")
            or reviewed.get("all_output_bindings_rehashed") is not True
            or reviewed.get("independent_exact_readout_replay") is not True
            or reviewed.get("frame_count") != 61
            or reviewed.get("solver_states_passed") != 60
            or reviewed.get("native_calls") != 1
            or reviewed.get("readout_calls") != 1
            or reviewed.get("physical_validation_pass") is not None
            or reviewed.get("patient_data_accessed") is not False
            or reviewed.get("measured_response_accessed") is not False):
        raise ValueError("Independent numerical-only review binding differs")
    return {"ordinal": 8, "run_id": remaining.ORDER[8],
            "native_receipt_sha256": SHA[NATIVE],
            "v2_sidecar_sha256": SHA[SIDECAR],
            "independent_metadata_sha256": SHA[REVIEW],
            "surcharge": _surcharge(sidecar, native),
            "physical_validation_pass": None,
            "patient_or_measured_response_accessed": False}


def verify_exact(*, root: Path = ROOT) -> dict:
    """Authenticate the exact saved v2 event; no bulk-output verification."""
    root = root.resolve()
    if (root / launcher.PRIOR_POLICY_SIDECAR).exists() or \
            (root / launcher.PRIOR_POLICY_SIDECAR).is_symlink():
        raise ValueError("Unexpected v1 policy sidecar exists")
    values = {name: _small_exact(root, name, digest)
              for name, digest in SHA.items()}
    _small_bytes(root, REVIEW_REPORT, REVIEW_REPORT_SHA)
    sidecar_dir = root / launcher.SIDECAR
    if (sidecar_dir.is_symlink()
            or {path.name for path in sidecar_dir.iterdir()} != {"receipt.json"}
            or (sidecar_dir / "receipt.json").stat().st_size >
               launcher.POLICY["sidecar_output_cap_bytes"]):
        raise ValueError("V2 sidecar output inventory or cap differs")
    _historical_sources(root)
    return _saved_fields(values[SIDECAR], values[NATIVE], values[INNER],
                         values[OUTER], values[REVIEW], root)


def assess_future_release(release: dict, *, root: Path = ROOT) -> dict:
    """Combine exact v2 gate with the unchanged full predecessor validator.

    This is read-only. A separately source-bound execution-sidecar integration
    is still required; returning a charged ledger does not issue a release.
    """
    root = root.resolve()
    exact = verify_exact(root=root)
    index = release.get("ordinal") if isinstance(release, dict) else None
    if (type(index) is not int or index != 9
            or release.get("run_id") != remaining.ORDER[index]
            or not isinstance(release.get("prior_receipts"), list)
            or len(release["prior_receipts"]) != index
            or release["prior_receipts"][8] != {
                "run_id": remaining.ORDER[8], "path": NATIVE, "sha256": SHA[NATIVE]}):
        raise ValueError("Future release does not bind exact v2 predecessor")
    context = remaining.validate_release(release, root=root)
    if context.get("index") != index or verify_exact(root=root) != exact:
        raise ValueError("Old chain or v2 sidecar changed during preflight")
    ledger = continuation_ledger(context["previous"], exact, index)
    return {"v2_ordinal8": exact, "old_context": context,
            "continuation_accounting": ledger,
            "status": "hold_execution_sidecar_integration_required"}
