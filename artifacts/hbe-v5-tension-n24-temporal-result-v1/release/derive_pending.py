"""Derive one exact pending row-11 release after source promotion.

No native call, attempt reservation, predecessor solver-log read or measured
response access occurs here. Root must select a final committed HEAD, inspect
both output byte strings and separately authorize any one-use execution.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import stat
import sys
from types import ModuleType

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from launchers import hbe_v5_later_continuation_runtime_v1 as runtime
from launchers import hbe_v5_later_ancestry_v1 as ancestry
from launchers import hbe_v5_later_ledger_math_v1 as ledger
from launchers import hbe_v5_ordinal11_continuation_v1 as row11
from launchers import hbe_v5_ordinal9_continuation_v1 as row9
from scripts import mechanics_hbe_branch_calibration_v5 as v5
from scripts import mechanics_hbe_v5_remaining_one_shot as remaining
from scripts import mechanics_hbe_v5_source_bindings as sources

BASE = ROOT / "build/hbe-v5-ordinal10-pending-release-v1/derive_pending.py"
BASE_SHA = "985c300269ad03d3748735291ebe00b78a0b2dc47e52641a27c44a4d8c97d2a0"


def bound_helpers(path: Path = BASE):
    """Execute only the exact reviewed, bounded row-10 helper bytes."""
    if any(item.is_symlink() for item in (path, *path.parents)):
        raise ValueError("Symlinked row-10 helper refused")
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or before.st_size > 128 * 1024:
        raise ValueError("Regular bounded row-10 helper required")
    raw = path.read_bytes()
    after = path.lstat()
    ident = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns)
    if ident(before) != ident(after) or hashlib.sha256(raw).hexdigest() != BASE_SHA:
        raise ValueError("Reviewed row-10 helper bytes differ")
    module = ModuleType("bound_row10_derivation_helpers")
    module.__file__ = str(path)
    exec(compile(raw, str(path), "exec"), module.__dict__)
    return module


helpers = bound_helpers()

OUT = Path(__file__).resolve().parent
INNER = OUT / "inner-release.json"
OUTER = OUT / "outer-envelope.json"
METADATA = OUT / "DERIVATION.json"
PRIOR = "build/hbe-v5-ordinal10-pending-release-v1/"


def old_after10(sidecar: dict, native: dict) -> dict:
    """Small-metadata copy of the frozen old ledger, for derivation only.

    The launcher will independently run the complete frozen predecessor chain
    before any native supervision; this construction is not its replacement.
    """
    previous = sidecar["old_validation_identity"]["previous"]
    result = copy.deepcopy(previous)
    result["sha256"].append(ancestry.ROW10_NATIVE_SHA)
    native_seconds = native["native_stage"]["elapsed_seconds"]
    readout_seconds = native["readout_stage"]["elapsed_seconds"]
    prep = native["prep_elapsed_seconds"]
    closed = sidecar["extension_resource_charge"]["native_closed_output_bytes_at_check"]
    for key, value in (("native_seconds", native_seconds),
                       ("readout_seconds", readout_seconds),
                       ("prep_seconds", prep), ("output_bytes", closed),
                       ("combined_wall_seconds", native_seconds + readout_seconds + prep),
                       ("combined_output_bytes", closed), ("native_calls", 1)):
        result[key] += value
    return result


def derive(head: str) -> dict:
    if helpers.git("rev-parse", "HEAD").decode().strip() != head:
        raise ValueError("Root-selected committed HEAD changed")
    spec = row11.SPEC
    runtime.validate_spec(spec)
    if any(p.exists() or p.is_symlink() for p in (INNER, OUTER, METADATA)):
        raise ValueError("Pending row-11 release files are one-use")
    for relative in (spec["native_output_directory"], spec["sidecar_directory"]):
        path = runtime._local(ROOT, relative)
        if path.exists() or path.is_symlink():
            raise ValueError("Row-11 native or sidecar attempt already exists")
    prior = helpers.small(ROOT / (PRIOR + "inner-release.json"),
                          ancestry.ROW10_INNER_SHA)
    outer10 = helpers.small(ROOT / (PRIOR + "outer-envelope.json"),
                            ancestry.ROW10_OUTER_SHA)
    sidecar10 = helpers.small(ROOT / (
        "outputs/mechanics/hbe-v5-later-continuation-v1/"
        "10-tension-N24-S60-reference/attempt-01/receipt.json"),
        ancestry.ROW10_SIDECAR_SHA)
    native10 = helpers.small(ROOT / remaining.receipt_path(10),
                             ancestry.ROW10_NATIVE_SHA)
    if (prior.get("status") != "root_released_one_native_call"
            or prior.get("ordinal") != 10
            or prior.get("run_id") != remaining.ORDER[10]
            or len(prior.get("prior_receipts", [])) != 10
            or prior["prior_receipts"][1]["sha256"] != remaining.N12_FAILED_RECEIPT_SHA
            or outer10.get("inner_release", {}).get("sha256") !=
               ancestry.ROW10_INNER_SHA):
        raise ValueError("Exact row-10 predecessor release differs")
    modules = {"row9": row9, "ledger": ledger, "remaining": remaining,
               "ancestry": ancestry, "bindings": {
                   "scripts/mechanics_hbe_v5_n12_v2_admission.py":
                       "20d2aab7cfe22001c255c88f4b3ffd9023eb6ba134bdd08a0aab9c05b3d52295"}}
    claims, charged, evidence = runtime._ancestors(
        ROOT, spec, {"previous": old_after10(sidecar10, native10)}, modules,
        ancestry.ROW10_DESCRIPTOR)
    if [item["ordinal"] for item in claims] != [8, 9, 10]:
        raise ValueError("Three exact prior sidecar charges required")
    remaining.validate_preparation(json.loads((ROOT / remaining.PREPARATION).read_text()))
    sources.validate_binding_manifest(ROOT, inspect_sources=False)
    study, previous_study = v5.validate_preparation(ROOT)
    source_map = json.loads((ROOT / remaining.SOURCE_MAP).read_text())
    run_id = remaining.ORDER[11]
    item = source_map["source_decks"][source_map["run_source_keys"][run_id]]
    old_binding = {key: item[key] for key in ("path", "sha256")}
    mesh_binding = source_map["meshes"][item["mesh"]]
    old = sources._read_bound(ROOT, old_binding, maximum=sources.MAX_SOURCE_BYTES)
    mesh = sources._read_bound(ROOT, mesh_binding, maximum=sources.MAX_MESH_BYTES)
    native_domain = v5.run_spec(study, previous_study, run_id)["native_domain"]
    topology = sources.validate_topology_bc(json.loads(mesh), old,
                                            native_domain=native_domain)
    _, adapted, adapter = v5.adapt_deck(study, previous_study, run_id, old)
    old_bindings = {relative: {"path": relative,
                               "sha256": helpers.source(head, relative)}
                    for relative in remaining.SOURCE_PATHS}
    extension_bindings = {relative: helpers.source(head, relative)
                          for relative in runtime.sources(spec)}
    if len(old_bindings) != 20 or len(extension_bindings) != 9:
        raise ValueError("Frozen and extension source closure differs")
    runtime.verify_sources(ROOT, head, extension_bindings, spec, require_head=True)
    inner = copy.deepcopy(prior)
    inner.update({
        "ordinal": 11, "run_id": run_id, "source_commit": head,
        "source_bindings": old_bindings,
        "old_source_deck": old_binding, "native_mesh": mesh_binding,
        "adapted_deck_sha256": helpers.sha(adapted.encode()),
        "adapter_receipt": adapter,
        "output_directory": remaining.output_directory(11),
        "caps": remaining.caps(11),
        "prior_receipts": [*prior["prior_receipts"], {
            "path": remaining.receipt_path(10), "run_id": remaining.ORDER[10],
            "sha256": ancestry.ROW10_NATIVE_SHA}],
    })
    if (inner["status"] != "root_released_one_native_call"
            or len(inner["prior_receipts"]) != 11
            or helpers.git("rev-parse", "HEAD").decode().strip() != head):
        raise ValueError("Frozen row-11 release or HEAD differs")
    inner_raw = helpers.write_once(INNER, inner)
    outer = {
        "schema": "hbe-v5-later-continuation-envelope-v1",
        "status": "root_released_one_native_call_with_extension_ledger",
        "source_commit": head,
        "extension_source_bindings": extension_bindings,
        "inner_release": {"path": str(INNER.relative_to(ROOT)),
                          "sha256": helpers.sha(inner_raw)},
        "sidecar_directory": spec["sidecar_directory"],
        "policy": runtime.policy(spec),
        "prior_extension_descriptor": ancestry.ROW10_DESCRIPTOR,
    }
    outer_raw = helpers.write_once(OUTER, outer)
    result = {
        "status": "PENDING_ROOT_AUTHORIZATION_NO_NATIVE_ATTEMPT",
        "source_commit": head, "ordinal": 11, "run_id": run_id,
        "inner_sha256": helpers.sha(inner_raw),
        "outer_sha256": helpers.sha(outer_raw),
        "prior_row10_native_sha256": ancestry.ROW10_NATIVE_SHA,
        "prior_row10_sidecar_sha256": ancestry.ROW10_SIDECAR_SHA,
        "prior_independent_metadata_sha256": ancestry.ROW10_INDEPENDENT_SHA,
        "prior_sidecar_charge_order": [item["ordinal"] for item in claims],
        "prior_sidecar_prep_surcharge_seconds": charged["incremental_prep_seconds"],
        "prior_sidecar_reserved_output_bytes": charged["incremental_output_bytes"],
        "expected_preflight_hint_opens": spec["expected_preflight_hint_opens"],
        "expected_preflight_hint_bytes": spec["expected_preflight_hint_bytes"],
        "old_source_count": len(old_bindings),
        "extension_source_count": len(extension_bindings),
        "old_source_deck_sha256": helpers.sha(old),
        "native_mesh_sha256": helpers.sha(mesh),
        "adapted_deck_sha256": helpers.sha(adapted.encode()),
        "adapter_receipt": adapter,
        "native_domain": native_domain,
        "native_nodes": topology["node_count"],
        "native_hex8": topology["hex8_count"],
        "old_predecessor_bulk_rehash": False,
        "native_calls_by_derivation": 0,
        "physical_validation_pass": None,
        "prior_extension_evidence": evidence,
    }
    helpers.write_once(METADATA, result)
    return result


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Provide final root-selected source commit")
    result = derive(sys.argv[1])
    print(json.dumps({key: result[key] for key in (
        "source_commit", "ordinal", "run_id", "inner_sha256",
        "outer_sha256", "expected_preflight_hint_bytes",
        "native_calls_by_derivation")}, sort_keys=True))
