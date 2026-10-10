"""Candidate shared execution wrapper for HBE v5 tension ordinals 10 and 11.

Proposed tracked path: launchers/hbe_v5_later_continuation_runtime_v1.py.
This source is not released. A separately reviewed, current-HEAD inner and
outer release is necessary before its one-use native path can be exercised.
The immutable row adapter selects only an existing frozen old-run ordinal.
"""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import time
from types import ModuleType

ROOT = Path(__file__).resolve().parents[1]  # Correct after promotion to launchers/.
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
HEX40 = re.compile(r"[0-9a-f]{40}\Z")
COMMON = (
    "launchers/hbe_v5_later_continuation_runtime_v1.py",
    "launchers/hbe_v5_later_continuation_core_v1.py",
    "launchers/hbe_v5_later_ledger_math_v1.py",
    "launchers/hbe_v5_later_ancestry_v1.py",
    "launchers/hbe_v5_ordinal9_continuation_v1.py",
    "scripts/mechanics_hbe_v5_n12_v2_admission.py",
    "launchers/hbe_v5_nocache_hash_v1.py",
    "launchers/hbe_v5_nocache_host_v1.py",
)
POLICY_BASE = {
    "io_policy": "darwin_F_NOCACHE_read_only_later_v2_continuation_v1",
    "minimum_hint_bytes": 16 * 1024**2,
    "initial_available_percent_floor": 54,
    "pre_native_available_percent_floor": 45,
    "normal_kernel_pressure_mask": 1,
    "sidecar_output_cap_bytes": 1024**2,
    "extension_before_old_execute_wall_seconds": 45,
    "extension_after_old_execute_wall_seconds": 45,
    "extension_self_peak_rss_bytes": 3 * 1024**3,
    "owned_stage_cleanup_seconds": 3,
    "finalization_prep_reserve_seconds": 1,
}


def sources(spec: dict) -> tuple[str, ...]:
    return (spec["adapter_source"], *COMMON)


def policy(spec: dict) -> dict:
    count = spec["expected_preflight_hint_opens"]
    size = spec["expected_preflight_hint_bytes"]
    if type(count) is not int or count != 2 * (spec["index"] - 1):
        raise ValueError("Two hinted node/element hashes per later predecessor required")
    if type(size) is not int or size < 16 * 1024**2:
        raise ValueError("Exact reviewed predecessor hint byte count required")
    return {**POLICY_BASE, "predecessor_ordinals": list(range(spec["index"])),
            "expected_preflight_hint_opens": count,
            "expected_preflight_hint_bytes": size,
            "expected_completed_hint_opens": count * 2,
            "expected_completed_hint_bytes": size * 2}


def validate_spec(spec: dict) -> None:
    from scripts import mechanics_hbe_v5_remaining_one_shot as remaining
    required = {"index", "run_id", "adapter_source", "sidecar_directory",
                "native_output_directory", "expected_preflight_hint_opens",
                "expected_preflight_hint_bytes"}
    if (not isinstance(spec, dict) or set(spec) != required
            or type(spec["index"]) is not int or spec["index"] not in (10, 11)
            or spec["run_id"] != remaining.ORDER[spec["index"]]
            or spec["native_output_directory"] !=
               remaining.output_directory(spec["index"])
            or spec["adapter_source"] !=
               f"launchers/hbe_v5_ordinal{spec['index']}_continuation_v1.py"
            or spec["sidecar_directory"] !=
               ("outputs/mechanics/hbe-v5-later-continuation-v1/"
                f"{spec['index']:02d}-{spec['run_id'].replace(':', '-')}/attempt-01")):
        raise ValueError("Only exact frozen later tension rows are selectable")
    if (spec["index"] == 11
            and spec["expected_preflight_hint_bytes"] != 3_101_186_843):
        raise ValueError("Ordinal 11 requires exact observed row-10 geometry bytes")
    policy(spec)


def _local(root: Path, relative: str) -> Path:
    path = Path(relative)
    if not relative or path.is_absolute() or ".." in path.parts:
        raise ValueError("Repository-relative extension path required")
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


def verify_sources(root: Path, commit: str, bindings: dict, spec: dict,
                   *, require_head: bool) -> dict:
    expected = sources(spec)
    if (not isinstance(commit, str) or not HEX40.fullmatch(commit)
            or not isinstance(bindings, dict) or set(bindings) != set(expected)):
        raise ValueError("Exact later-row source closure required")
    head = _git(root, "rev-parse", "HEAD").decode().strip()
    if require_head and head != commit:
        raise ValueError("Checkout differs from reviewed source commit")
    for relative in expected:
        digest = bindings[relative]
        if not isinstance(digest, str) or not HEX64.fullmatch(digest):
            raise ValueError("Full source hash required")
        path = _local(root, relative)
        before = path.lstat()
        if not stat.S_ISREG(before.st_mode) or before.st_size > 128 * 1024:
            raise ValueError("Extension source absent, special or oversized")
        raw = path.read_bytes()
        after = path.lstat()
        ident = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns)
        if ident(before) != ident(after) or hashlib.sha256(raw).hexdigest() != digest:
            raise ValueError("Extension source changed")
        name = f"{commit}:{relative}"
        if (int(_git(root, "cat-file", "-s", name)) != len(raw)
                or _git(root, "cat-file", "blob", name) != raw):
            raise ValueError("Committed extension source bytes differ")
    return {"source_commit": commit, "observed_head": head,
            "source_hashes": dict(bindings)}


def _bound_module(root: Path, relative: str, digest: str, name: str):
    path = _local(root, relative)
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or before.st_size > 128 * 1024:
        raise ValueError("Bound helper absent or oversized")
    raw = path.read_bytes()
    after = path.lstat()
    ident = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns)
    if ident(before) != ident(after) or hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError("Bound helper source changed")
    module = ModuleType(name)
    module.__file__ = str(path)
    exec(compile(raw, str(path), "exec"), module.__dict__)
    return module


def _literal_adapter_spec(root: Path, relative: str) -> dict:
    """Read a row declaration without executing untrusted adapter bytes.

    The full source/commit binding is checked by ``_read_release`` before
    any extension code is run. Row adapters are declarative, not launchers.
    """
    path = _local(root, relative)
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or before.st_size > 128 * 1024:
        raise ValueError("Declarative row adapter absent or oversized")
    raw = path.read_bytes()
    after = path.lstat()
    ident = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns)
    if ident(before) != ident(after):
        raise ValueError("Declarative row adapter changed")
    tree = ast.parse(raw, filename=str(path))
    declarations = [node for node in tree.body if isinstance(node, ast.Assign)
                    and len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Name)
                    and node.targets[0].id == "SPEC"]
    if len(declarations) != 1:
        raise ValueError("Exactly one literal row declaration required")
    try:
        spec = ast.literal_eval(declarations[0].value)
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError) as error:
        raise ValueError("Literal row declaration required") from error
    if not isinstance(spec, dict):
        raise ValueError("Literal row declaration must be an object")
    return spec


def _read_release(root: Path, envelope_path: Path, spec: dict):
    from scripts import mechanics_hbe_v5_n8_one_shot as io
    from scripts import mechanics_hbe_v5_remaining_one_shot as remaining
    validate_spec(spec)
    if Path(__file__).resolve() != _local(root, COMMON[0]):
        raise ValueError("Runtime must execute from bound tracked path")
    raw, identity = io.read_release(envelope_path)
    outer = json.loads(raw)
    keys = {"schema", "status", "source_commit", "extension_source_bindings",
            "inner_release", "sidecar_directory", "policy",
            "prior_extension_descriptor"}
    if (not isinstance(outer, dict) or set(outer) != keys
            or outer["schema"] != "hbe-v5-later-continuation-envelope-v1"
            or outer["status"] != "root_released_one_native_call_with_extension_ledger"
            or outer["sidecar_directory"] != spec["sidecar_directory"]
            or outer["policy"] != policy(spec)
            or (spec["index"] == 10 and outer["prior_extension_descriptor"] is not None)
            or (spec["index"] == 11 and not isinstance(
                outer["prior_extension_descriptor"], dict))):
        raise ValueError("Exact separately reviewed later-row envelope required")
    sources_before = verify_sources(root, outer["source_commit"],
                                    outer["extension_source_bindings"], spec,
                                    require_head=True)
    binding = outer["inner_release"]
    if (not isinstance(binding, dict) or set(binding) != {"path", "sha256"}
            or not isinstance(binding["sha256"], str)
            or not HEX64.fullmatch(binding["sha256"])):
        raise ValueError("Exact old one-row release binding required")
    inner_path = _local(root, binding["path"])
    inner_raw, inner_identity = io.read_release(inner_path)
    if hashlib.sha256(inner_raw).hexdigest() != binding["sha256"]:
        raise ValueError("Old one-row release bytes changed")
    inner = json.loads(inner_raw)
    if (inner.get("ordinal") != spec["index"]
            or inner.get("run_id") != remaining.ORDER[spec["index"]]
            or inner.get("status") != "root_released_one_native_call"
            or inner.get("source_commit") != outer["source_commit"]):
        raise ValueError("Old frozen row release identity differs")
    return (outer, raw, identity, sources_before,
            inner_path, inner_raw, inner_identity)


def _write_sidecar(sidecar: Path, record: dict) -> None:
    from scripts import mechanics_hbe_v5_n8_one_shot as io
    raw = (json.dumps(record, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    if len(raw) > 1024**2 - 4096:
        raise ValueError("Extension sidecar exceeds reserved one MiB")
    io.durable_json(sidecar / "receipt.json", record)


def _small_saved(root: Path, relative: str, expected_sha: str) -> tuple[dict, bytes]:
    from scripts import mechanics_hbe_v5_n8_one_shot as io
    path = _local(root, relative)
    raw, _ = io.read_release(path)
    if hashlib.sha256(raw).hexdigest() != expected_sha:
        raise ValueError("Exact saved metadata SHA differs")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("Saved metadata must be an object")
    return value, raw


def _historical_source_blobs(root: Path, commit: str, bindings: dict,
                             expected_paths: tuple[str, ...]) -> None:
    """Recover the exact source bytes declared by a pinned past sidecar."""
    if (not isinstance(commit, str) or not HEX40.fullmatch(commit)
            or not isinstance(bindings, dict)
            or set(bindings) != set(expected_paths)):
        raise ValueError("Historical extension source closure differs")
    for relative in expected_paths:
        digest = bindings[relative]
        if not isinstance(digest, str) or not HEX64.fullmatch(digest):
            raise ValueError("Historical extension source SHA differs")
        name = f"{commit}:{relative}"
        try:
            size = int(_git(root, "cat-file", "-s", name))
        except (subprocess.CalledProcessError, KeyError, ValueError) as error:
            raise ValueError("Historical extension source blob unavailable") from error
        if size <= 0 or size > 128 * 1024:
            raise ValueError("Historical extension source size differs")
        try:
            raw = _git(root, "cat-file", "blob", name)
        except (subprocess.CalledProcessError, KeyError) as error:
            raise ValueError("Historical extension source blob unavailable") from error
        if len(raw) != size or hashlib.sha256(raw).hexdigest() != digest:
            raise ValueError("Historical extension source blob differs")


def _bind_native_receipt(root: Path, spec: dict, record: dict) -> dict | None:
    """Retain the old durable receipt even on a later extension failure."""
    from scripts import mechanics_hbe_v5_n8_one_shot as io
    path = _local(root, spec["native_output_directory"] + "/receipt.json")
    if not path.exists() and not path.is_symlink():
        if record.get("native_receipt_sha256") is not None:
            raise ValueError("Previously bound native receipt disappeared")
        return None
    raw, _ = io.read_release(path)
    digest = hashlib.sha256(raw).hexdigest()
    previous = record.get("native_receipt_sha256")
    if previous is not None and previous != digest:
        raise ValueError("Native receipt changed after first binding")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("Native receipt must be a JSON object")
    record["native_receipt_sha256"] = digest
    record["native_receipt_status"] = value.get("status")
    record["native_calls_attempted"] = value.get("native_calls_attempted")
    record["hbe_readout_calls_attempted"] = value.get("readout_calls_attempted")
    return value


def _ancestors(root: Path, spec: dict, context: dict, modules: dict,
               descriptor: dict | None) -> tuple[list[dict], dict]:
    """Bind all previous extension charges to the already validated old chain."""
    if spec["index"] == 11:
        return _ancestors_through10(root, spec, context, modules, descriptor)
    if spec["index"] != 10 or descriptor is not None:
        raise ValueError("Exact ordered extension ancestry required")
    row9 = modules["row9"]
    v2 = row9._load_bound_admission(
        root, modules["bindings"]["scripts/mechanics_hbe_v5_n12_v2_admission.py"])
    exact8 = v2.verify_exact(root=root)
    ancestry = modules["ancestry"]
    sidecar, _ = _small_saved(
        root, "outputs/mechanics/hbe-v5-ordinal9-continuation-v1/"
        "09-tension-N16-S60-reference/attempt-01/receipt.json",
        ancestry.ROW9_SIDECAR_SHA)
    _historical_source_blobs(
        root, ancestry.ROW9_COMMIT,
        sidecar.get("extension_sources_before", {}).get("source_hashes"),
        row9.SOURCES)
    outer9, _ = _small_saved(
        root, "build/hbe-v5-tension-n16-s60-continuation-final-release-v1/"
        "outer-envelope.json", ancestry.ROW9_OUTER_SHA)
    expected_inner9 = (
        "build/hbe-v5-tension-n16-s60-continuation-final-release-v1/"
        "inner-release.json")
    if (outer9.get("schema") != "hbe-v5-ordinal9-continuation-envelope-v1"
            or outer9.get("status") != "root_released_one_native_call_with_v2_ledger"
            or outer9.get("source_commit") != ancestry.ROW9_COMMIT
            or outer9.get("extension_source_bindings") !=
               sidecar["extension_sources_before"]["source_hashes"]
            or outer9.get("inner_release") != {
                "path": expected_inner9, "sha256": ancestry.ROW9_INNER_SHA}):
        raise ValueError("Ordinal-9 outer release lineage differs")
    inner9, _ = _small_saved(root, expected_inner9, ancestry.ROW9_INNER_SHA)
    if (inner9.get("source_commit") != ancestry.ROW9_COMMIT
            or inner9.get("ordinal") != 9
            or inner9.get("run_id") != ancestry.ROW9_RUN):
        raise ValueError("Ordinal-9 inner release lineage differs")
    native, _ = _small_saved(
        root, "outputs/mechanics/hbe-v5-remaining-one-shot-v1/"
        "09-tension-N16-S60-reference/attempt-01/receipt.json",
        ancestry.ROW9_NATIVE_SHA)
    side_dir = _local(root, "outputs/mechanics/hbe-v5-ordinal9-continuation-v1/"
                      "09-tension-N16-S60-reference/attempt-01")
    if (side_dir.is_symlink() or {x.name for x in side_dir.iterdir()} != {"receipt.json"}
            or (side_dir / "receipt.json").stat().st_size > 1024**2):
        raise ValueError("Ordinal-9 sidecar inventory or cap differs")
    claim9 = ancestry.row9_charge(
        old_after9=context["previous"], row8_exact=exact8,
        sidecar=sidecar, sidecar_sha256=ancestry.ROW9_SIDECAR_SHA,
        native=native, native_sha256=ancestry.ROW9_NATIVE_SHA,
        remaining=modules["remaining"], row9_policy=row9.POLICY)
    native8, _ = _small_saved(root, v2.NATIVE, exact8["native_receipt_sha256"])
    delta8 = exact8["surcharge"]
    prep8 = native8.get("prep_elapsed_seconds")
    if type(prep8) not in (int, float) or not math.isfinite(prep8) or prep8 < 0:
        raise ValueError("Ordinal-8 saved native prep differs")
    claim8 = {"ordinal": 8,
              "native_receipt_sha256": exact8["native_receipt_sha256"],
              "sidecar_sha256": exact8["v2_sidecar_sha256"],
              "native_prep_seconds": prep8,
              "launcher_inclusive_prep_seconds": prep8 + delta8["prep_wall_seconds"],
              "reserved_sidecar_output_bytes": delta8["output_bytes"]}
    claims = [claim8, claim9]
    charged = modules["ledger"].charge_old_baseline(
        context["previous"], copy.deepcopy(context["previous"]), claims, 10)
    return claims, charged, {"ordinal8_exact": exact8,
                             "ordinal9_native_receipt_sha256":
                                 ancestry.ROW9_NATIVE_SHA,
                             "ordinal9_sidecar_sha256":
                                 ancestry.ROW9_SIDECAR_SHA}


def _ancestors_through10(root: Path, spec: dict, context: dict, modules: dict,
                         descriptor: dict | None) -> tuple[list[dict], dict, dict]:
    """Bind the actual row-10 policy event after the frozen chain reached it."""
    ancestry = modules["ancestry"]
    if descriptor != ancestry.ROW10_DESCRIPTOR:
        raise ValueError("Exact independently reviewed row-10 descriptor required")
    review, _ = _small_saved(root, ancestry.ROW10_INDEPENDENT_PATH,
                             ancestry.ROW10_INDEPENDENT_SHA)
    replay = review.get("saved_replay", {})
    if (review.get("verdict") != "GO_saved_numerical_software_result_only"
            or review.get("source_commit") != ancestry.ROW10_COMMIT
            or review.get("native_receipt_sha256") != ancestry.ROW10_NATIVE_SHA
            or review.get("sidecar_receipt_sha256") != ancestry.ROW10_SIDECAR_SHA
            or review.get("inner_release_sha256") != ancestry.ROW10_INNER_SHA
            or review.get("outer_envelope_sha256") != ancestry.ROW10_OUTER_SHA
            or review.get("physical_validation_pass") is not None
            or replay.get("numerical_passed") is not True
            or replay.get("exact_readout_equality") is not True
            or replay.get("frame_count") != 61
            or replay.get("representation") != "reconstructed_full"):
        raise ValueError("Committed independent row-10 numerical audit differs")
    original = "build/hbe-v5-ordinal10-pending-release-v1/"
    archived = "artifacts/hbe-v5-tension-n24-continuation-result-v1/release/"
    outer10, _ = _small_saved(root, original + "outer-envelope.json",
                              ancestry.ROW10_OUTER_SHA)
    inner10, _ = _small_saved(root, original + "inner-release.json",
                              ancestry.ROW10_INNER_SHA)
    _small_saved(root, archived + "outer-envelope.json", ancestry.ROW10_OUTER_SHA)
    _small_saved(root, archived + "inner-release.json", ancestry.ROW10_INNER_SHA)
    sidecar10, _ = _small_saved(
        root, "outputs/mechanics/hbe-v5-later-continuation-v1/"
        "10-tension-N24-S60-reference/attempt-01/receipt.json",
        ancestry.ROW10_SIDECAR_SHA)
    native10, native_raw = _small_saved(
        root, "outputs/mechanics/hbe-v5-remaining-one-shot-v1/"
        "10-tension-N24-S60-reference/attempt-01/receipt.json",
        ancestry.ROW10_NATIVE_SHA)
    if (replay.get("readout_sha256") !=
            native10.get("saved_numerical_readout", {}).get("readout_sha256")
            or replay.get("readout_sha256") !=
               native10.get("output_bindings", {}).get("readout.json", {}).get("sha256")):
        raise ValueError("Independent row-10 replay/readout binding differs")
    source_hashes10 = sidecar10.get("extension_sources_before", {}).get("source_hashes")
    _historical_source_blobs(root, ancestry.ROW10_COMMIT, source_hashes10,
                             sources({"adapter_source":
                                      "launchers/hbe_v5_ordinal10_continuation_v1.py"}))
    if (outer10.get("source_commit") != ancestry.ROW10_COMMIT
            or outer10.get("extension_source_bindings") != source_hashes10
            or outer10.get("inner_release") != {
                "path": original + "inner-release.json",
                "sha256": ancestry.ROW10_INNER_SHA}
            or inner10.get("source_commit") != ancestry.ROW10_COMMIT
            or inner10.get("ordinal") != 10
            or inner10.get("run_id") != ancestry.ROW10_RUN
            or native10.get("release_sha256") != ancestry.ROW10_INNER_SHA):
        raise ValueError("Ordinal-10 release/source lineage differs")
    side_dir = _local(root, "outputs/mechanics/hbe-v5-later-continuation-v1/"
                      "10-tension-N24-S60-reference/attempt-01")
    if (side_dir.is_symlink() or {item.name for item in side_dir.iterdir()}
            != {"receipt.json"} or (side_dir / "receipt.json").stat().st_size > 1024**2):
        raise ValueError("Ordinal-10 sidecar inventory or cap differs")
    old_after9 = sidecar10.get("old_validation_identity", {}).get("previous")
    if not isinstance(old_after9, dict):
        raise ValueError("Ordinal-10 old-chain checkpoint missing")
    claims, charged9, evidence9 = _ancestors(
        root, {"index": 10}, {"previous": old_after9}, modules, None)
    row10_policy = policy({"index": 10,
                           "expected_preflight_hint_opens": 18,
                           "expected_preflight_hint_bytes": 2_908_379_069})
    claim10 = ancestry.row10_charge(
        old_after10=context["previous"], old_after9=old_after9,
        charged_after9=charged9["charged_cumulative_ledger"],
        prior_claims=claims, sidecar=sidecar10,
        sidecar_sha256=ancestry.ROW10_SIDECAR_SHA,
        native=native10, native_sha256=ancestry.ROW10_NATIVE_SHA,
        native_receipt_bytes=len(native_raw), remaining=modules["remaining"],
        row10_policy=row10_policy)
    claims = [*claims, claim10]
    charged = modules["ledger"].charge_old_baseline(
        context["previous"], copy.deepcopy(context["previous"]), claims, 11)
    if (charged["charged_cumulative_ledger"] !=
            sidecar10["supplemental_ledger_after_row"]["charged_cumulative_ledger"]):
        raise ValueError("Ordinal-10 cumulative ledger cannot continue")
    return claims, charged, {**evidence9,
                             "ordinal10_independent_metadata_sha256":
                                 ancestry.ROW10_INDEPENDENT_SHA,
                             "ordinal10_native_receipt_sha256":
                                 ancestry.ROW10_NATIVE_SHA,
                             "ordinal10_sidecar_sha256":
                                 ancestry.ROW10_SIDECAR_SHA}


def _charge(remaining, spec: dict, record: dict, started: float,
            native_receipt: dict | None, root: Path, phase: str) -> None:
    """Reserve finalization/sidecar and charge every wrapper second once."""
    native = (native_receipt or {}).get("native_stage", {}).get("elapsed_seconds", 0.)
    readout = (native_receipt or {}).get("readout_stage", {}).get("elapsed_seconds", 0.)
    if any(type(x) not in (int, float) or not math.isfinite(x) or x < 0
           for x in (native, readout)):
        raise ValueError("Saved stage timing differs")
    elapsed = time.monotonic() - started
    prep = max(0., elapsed - native - readout) + 1
    if prep >= remaining.PREP_WALL:
        raise ValueError("Launcher-inclusive frozen prep cap exhausted")
    directory = _local(root, spec["native_output_directory"])
    output = (remaining.active_bytes(directory,
              remaining.caps(spec["index"])["active_output_bytes"])
              if directory.exists() else 0)
    charged_output = output + 1024**2
    ledger = remaining._check_aggregate(
        record["charged_previous"], native, readout, prep,
        charged_output, 1)
    value = {"phase": phase, "launcher_elapsed_seconds_at_check": elapsed,
             "native_stage_seconds": native, "readout_stage_seconds": readout,
             "launcher_inclusive_prep_seconds_with_finalization_reserve": prep,
             "per_row_prep_wall_seconds": remaining.PREP_WALL,
             "native_closed_output_bytes_at_check": output,
             "sidecar_reserved_output_bytes": 1024**2,
             "charged_current_output_bytes": charged_output,
             "aggregate_with_extension": ledger}
    if phase == "post_execution":
        record["extension_resource_charge"] = value
        if native_receipt is not None:
            digest = record.get("native_receipt_sha256")
            if (not isinstance(digest, str) or not HEX64.fullmatch(digest)
                    or native_receipt.get("ordinal") != spec["index"]
                    or native_receipt.get("run_id") != spec["run_id"]
                    or native_receipt.get("status") != "passed_numerical_software_only"
                    or native_receipt.get("native_calls_attempted") != 1
                    or native_receipt.get("readout_calls_attempted") != 1
                    or native_receipt.get("no_retry") is not True
                    or native_receipt.get("prior_receipt_sha256") !=
                       record["old_validation_identity"]["previous"]["sha256"]):
                raise ValueError("Saved one-call native lineage differs")
            after = dict(record["charged_previous"])
            after.update({
                "sha256": after["sha256"] + [digest],
                "native_seconds": ledger["aggregate_native_wall_seconds"],
                "readout_seconds": ledger["aggregate_readout_wall_seconds"],
                "prep_seconds": ledger["aggregate_prep_wall_seconds"],
                "output_bytes": ledger["aggregate_output_bytes"],
                "native_calls": ledger["aggregate_native_calls"],
                "combined_wall_seconds": ledger["aggregate_combined_wall_seconds"],
                "combined_output_bytes": ledger["aggregate_combined_output_bytes"],
            })
            record["supplemental_ledger_after_row"] = {
                "schema": "hbe-v5-later-continuation-ledger-v1",
                "through_ordinal": spec["index"],
                "native_receipt_sha256": digest,
                "predecessor_extension_claims": record["predecessor_extension_claims"],
                "inclusive_prep_seconds": prep,
                "reserved_sidecar_output_bytes": 1024**2,
                "charged_cumulative_ledger": after,
            }
    else:
        record.setdefault("pre_native_resource_checks", []).append(value)


def execute_envelope(spec: dict, envelope_path: Path, *, root: Path = ROOT) -> dict:
    """Candidate one-use row execution; root has issued no such envelope."""
    from scripts import mechanics_hbe_v5_n8_one_shot as io
    from scripts import mechanics_hbe_v5_remaining_one_shot as remaining
    from scripts import febio_runtime
    validate_spec(spec)
    started = time.monotonic()
    (outer, outer_raw, outer_identity, before_sources,
     inner_path, inner_raw, inner_identity) = _read_release(root, envelope_path, spec)
    bindings = outer["extension_source_bindings"]
    modules = {"bindings": bindings, "remaining": remaining}
    for name, path in (("core", COMMON[1]), ("ledger", COMMON[2]),
                       ("ancestry", COMMON[3]), ("row9", COMMON[4]),
                       ("hint", COMMON[6]), ("host", COMMON[7])):
        modules[name] = _bound_module(root, path, bindings[path], "bound_later_" + name)
    host = modules["host"].FastDarwinSampler()
    sidecar = _local(root, spec["sidecar_directory"])
    native_output = _local(root, spec["native_output_directory"])
    if sidecar.exists() or sidecar.is_symlink() or native_output.exists() or native_output.is_symlink():
        raise ValueError("Later-row native or sidecar attempt already consumed")
    sidecar.mkdir(parents=True, exist_ok=False)
    record = {"schema": "hbe-v5-later-continuation-sidecar-v1",
              "status": "preflight_pending", "ordinal": spec["index"],
              "run_id": spec["run_id"], "source_commit": outer["source_commit"],
              "envelope_sha256": hashlib.sha256(outer_raw).hexdigest(),
              "inner_release_sha256": hashlib.sha256(inner_raw).hexdigest(),
              "extension_sources_before": before_sources,
              "policy": policy(spec), "native_calls_authorized": 1,
              "native_calls_attempted": 0, "hbe_readout_calls_attempted": 0,
              "native_output_directory": spec["native_output_directory"],
              "native_receipt_sha256": None,
              "physical_validation_pass": None}
    _write_sidecar(sidecar, record)
    audit = {"eligible_bytes": 0, "eligible_files": 0,
             "hinted_file_opens": 0, "hinted_file_paths": []}
    old_finished = None
    try:
        before = host.host()
        record["host_before_validation"] = before
        if (before["kernel_pressure_mask"] != 1
                or before["available_percent"] < 54):
            raise ValueError("Initial 54%/normal host floor not met")
        allowed = [_local(root, str(Path(remaining.receipt_path(i)).parent))
                   for i in range(spec["index"])]

        def exact_source_release(*, require_head: bool):
            result = verify_sources(root, outer["source_commit"], bindings,
                                    spec, require_head=require_head)
            if (io.read_release(envelope_path) != (outer_raw, outer_identity)
                    or io.read_release(inner_path) != (inner_raw, inner_identity)):
                raise ValueError("Extension or frozen release changed")
            return result

        def preflight(context, snapshot):
            if (context["index"] != spec["index"]
                    or context["run_id"] != spec["run_id"]):
                raise ValueError("Wrong old frozen row selected")
            claims, charged, ancestry_evidence = _ancestors(
                root, spec, context, modules, outer["prior_extension_descriptor"])
            record["predecessor_extension_claims"] = claims
            record["predecessor_extension_evidence"] = ancestry_evidence
            record["charged_previous"] = charged["charged_cumulative_ledger"]
            record["continuation_ledger"] = charged
            record["old_validation_identity"] = {
                "index": context["index"], "run_id": context["run_id"],
                "adapted_deck_sha256": hashlib.sha256(context["deck"]).hexdigest(),
                "adapter_receipt": context["adapter_receipt"],
                "previous": context["previous"],
                "source_hashes": context["source_hashes"],
                "observed_head_preflight": context["observed_head_preflight"]}
            record["preflight_hint_audit"] = snapshot
            current = host.host()
            record["host_before_native_reservation"] = current
            if current["kernel_pressure_mask"] != 1 or current["available_percent"] < 45:
                raise ValueError("Unchanged 45%/normal native floor not met")
            exact_source_release(require_head=True)
            _charge(remaining, spec, record, started, None, root, "reservation")
            record["status"] = "preflight_passed_before_native_reservation"
            _write_sidecar(sidecar, record)

        def pre_native():
            exact_source_release(require_head=True)
            current = host.host()
            record["host_immediately_before_native_supervision"] = current
            if current["kernel_pressure_mask"] != 1 or current["available_percent"] < 45:
                raise ValueError("Unchanged 45%/normal native supervision floor not met")
            _charge(remaining, spec, record, started, None, root, "supervision")
            _write_sidecar(sidecar, record)

        def cleanup(stage, result):
            record.setdefault("owned_stage_cleanup", {})[stage] = result
            _write_sidecar(sidecar, record)

        record["extension_before_old_execute_wall_seconds"] = time.monotonic() - started
        record["extension_self_peak_rss_bytes_before"] = modules["row9"]._self_peak_rss_bytes()
        if (record["extension_before_old_execute_wall_seconds"] >= 45
                or record["extension_self_peak_rss_bytes_before"] > 3 * 1024**3):
            raise ValueError("Extension pre-execute resource cap exhausted")
        record["native_calls_attempted"] = None
        record["hbe_readout_calls_attempted"] = None
        native_result, audit = modules["core"].scoped_old_execute(
            remaining, modules["hint"], allowed,
            lambda: remaining.execute(inner_path, root=root),
            expected_hint_opens=policy(spec)["expected_preflight_hint_opens"],
            expected_hint_bytes=policy(spec)["expected_preflight_hint_bytes"],
            preflight_check=preflight, native_stage_check=pre_native,
            stage_cleanup_callback=cleanup,
            owner_factory=modules["row9"].OwnedStage,
            observer=febio_runtime.process_group_rss)
        old_finished = time.monotonic()
        record["final_hint_audit"] = copy.deepcopy(audit)
        native_receipt = _bind_native_receipt(root, spec, record)
        if native_receipt != native_result:
            raise ValueError("Saved native receipt differs from returned old result")
        exact_source_release(require_head=False)
        record["extension_sources_after"] = verify_sources(
            root, outer["source_commit"], bindings, spec, require_head=False)
        record["extension_after_old_execute_wall_seconds"] = time.monotonic() - old_finished
        record["extension_self_peak_rss_bytes_after"] = modules["row9"]._self_peak_rss_bytes()
        if (record["extension_after_old_execute_wall_seconds"] >= 45
                or record["extension_self_peak_rss_bytes_after"] > 3 * 1024**3):
            raise ValueError("Extension post-execute resource cap exhausted")
        _charge(remaining, spec, record, started, native_receipt, root, "post_execution")
        if (native_result.get("status") != "passed_numerical_software_only"
                or audit["eligible_files"] != policy(spec)["expected_completed_hint_opens"]
                or audit["hinted_file_opens"] != audit["eligible_files"]
                or audit["eligible_bytes"] != policy(spec)["expected_completed_hint_bytes"]):
            raise ValueError("Native result or post-native hints differ")
        record["status"] = "passed_numerical_software_only_with_cumulative_ledger_v1"
    except BaseException as error:
        record["status"] = "failed_or_incomplete"
        record["failure"] = {"type": type(error).__name__, "message": str(error)[:500]}
    finally:
        if "final_hint_audit" not in record:
            record["final_hint_audit"] = audit
        try:
            saved_receipt = _bind_native_receipt(root, spec, record)
            if saved_receipt is not None and "charged_previous" in record:
                _charge(remaining, spec, record, started, saved_receipt,
                        root, "post_execution")
            record["extension_self_peak_rss_bytes_terminal"] = modules["row9"]._self_peak_rss_bytes()
            if record["extension_self_peak_rss_bytes_terminal"] > 3 * 1024**3:
                raise ValueError("Extension self RSS cap exceeded")
            record["extension_sources_terminal"] = verify_sources(
                root, outer["source_commit"], bindings, spec, require_head=False)
            if record["extension_sources_terminal"]["source_hashes"] != before_sources["source_hashes"]:
                raise ValueError("Extension source changed before final sidecar")
            if (io.read_release(envelope_path) != (outer_raw, outer_identity)
                    or io.read_release(inner_path) != (inner_raw, inner_identity)):
                raise ValueError("Extension releases changed before final sidecar")
        except BaseException as error:
            record["status"] = "failed_or_incomplete"
            record["terminal_binding_failure"] = type(error).__name__
        if record["native_calls_attempted"] is None and not native_output.exists():
            record["native_calls_attempted"] = 0
            record["hbe_readout_calls_attempted"] = 0
        record["launcher_elapsed_seconds_terminal"] = time.monotonic() - started
        _write_sidecar(sidecar, record)
        closed = sidecar / "receipt.json"
        if (sidecar.is_symlink() or {x.name for x in sidecar.iterdir()} != {"receipt.json"}
                or not stat.S_ISREG(closed.lstat().st_mode)
                or closed.is_symlink() or closed.stat().st_size > 1024**2):
            record["status"] = "failed_or_incomplete"
            record["terminal_sidecar_inventory_failure"] = True
            _write_sidecar(sidecar, record)
        charge = record.get("extension_resource_charge")
        if charge is not None:
            spent = (time.monotonic() - started - charge["native_stage_seconds"]
                     - charge["readout_stage_seconds"])
            if (spent > charge["launcher_inclusive_prep_seconds_with_finalization_reserve"]
                    or spent >= remaining.PREP_WALL):
                record["status"] = "failed_or_incomplete"
                record["terminal_finalization_failure"] = True
                record["launcher_elapsed_seconds_terminal"] = time.monotonic() - started
                _write_sidecar(sidecar, record)
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--ordinal", type=int, choices=(10, 11))
    parser.add_argument("--envelope", type=Path)
    args = parser.parse_args()
    if not args.execute or args.envelope is None or args.ordinal is None:
        raise ValueError("Explicit --execute, ordinal and reviewed envelope required")
    from scripts import mechanics_hbe_v5_n8_one_shot as io
    raw, _ = io.read_release(args.envelope)
    candidate = json.loads(raw)
    bindings = candidate.get("extension_source_bindings", {})
    relative = f"launchers/hbe_v5_ordinal{args.ordinal}_continuation_v1.py"
    digest = bindings.get(relative)
    if not isinstance(digest, str) or not HEX64.fullmatch(digest):
        raise ValueError("Exact reviewed row adapter source binding required")
    spec = _literal_adapter_spec(ROOT, relative)
    if spec["index"] != args.ordinal:
        raise ValueError("Bound adapter selected a different ordinal")
    # Authenticate the whole current-HEAD source closure and both release
    # byte strings before executing any row-dependent extension code.
    _read_release(ROOT, args.envelope, spec)
    result = execute_envelope(spec, args.envelope)
    print(json.dumps({"status": result["status"],
                      "ordinal": spec["index"],
                      "sidecar_directory": spec["sidecar_directory"]}))
    return 0 if result["status"] == "passed_numerical_software_only_with_cumulative_ledger_v1" else 1


if __name__ == "__main__":
    raise SystemExit(main())
