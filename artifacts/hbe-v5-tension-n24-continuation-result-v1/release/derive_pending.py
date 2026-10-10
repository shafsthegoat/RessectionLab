"""Derive exact pending ordinal-10 release bytes without executing a row.

This reads the 2.3 MiB frozen source deck, 3.0 MiB native mesh, committed
source files and small saved receipts/releases. It never traverses predecessor
solver outputs, reserves an attempt, launches FEBio or runs a readout.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import stat
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from launchers import hbe_v5_later_continuation_runtime_v1 as runtime
from launchers import hbe_v5_ordinal10_continuation_v1 as row10
from launchers import hbe_v5_later_ancestry_v1 as ancestry
from scripts import mechanics_hbe_branch_calibration_v5 as v5
from scripts import mechanics_hbe_v5_remaining_one_shot as remaining
from scripts import mechanics_hbe_v5_source_bindings as sources

OUT = Path(__file__).resolve().parent
INNER = OUT / "inner-release.json"
OUTER = OUT / "outer-envelope.json"
METADATA = OUT / "DERIVATION.json"
PREVIOUS_INNER = ROOT / (
    "build/hbe-v5-tension-n16-s60-continuation-final-release-v1/inner-release.json")
PREVIOUS_OUTER = ROOT / (
    "build/hbe-v5-tension-n16-s60-continuation-final-release-v1/outer-envelope.json")
ROW9_SIDECAR = ROOT / (
    "outputs/mechanics/hbe-v5-ordinal9-continuation-v1/"
    "09-tension-N16-S60-reference/attempt-01/receipt.json")
ROW9_NATIVE = ROOT / remaining.receipt_path(9)


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def git(*args: str) -> bytes:
    return runtime._git(ROOT, *args)


def small(path: Path, digest: str) -> dict:
    before = path.lstat()
    if path.is_symlink() or not stat.S_ISREG(before.st_mode) or before.st_size > 1024**2:
        raise ValueError("Pinned small ancestor is linked, absent or oversized")
    raw = path.read_bytes()
    after = path.lstat()
    ident = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns)
    if ident(before) != ident(after) or sha(raw) != digest:
        raise ValueError("Pinned small ancestor bytes changed")
    result = json.loads(raw)
    if not isinstance(result, dict):
        raise ValueError("Pinned small ancestor must be an object")
    return result


def source(head: str, relative: str) -> str:
    path = runtime._local(ROOT, relative)
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or before.st_size > 1024**2:
        raise ValueError("Committed source absent or oversized")
    raw = path.read_bytes()
    after = path.lstat()
    ident = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns)
    if ident(before) != ident(after):
        raise ValueError("Committed source changed during read")
    name = head + ":" + relative
    size = int(git("cat-file", "-s", name))
    if size != len(raw) or size > 1024**2 or git("cat-file", "blob", name) != raw:
        raise ValueError("Working and committed source differ")
    return sha(raw)


def write_once(path: Path, value: dict) -> bytes:
    raw = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    with path.open("xb") as stream:
        stream.write(raw)
    return raw


def derive(head: str) -> dict:
    if git("rev-parse", "HEAD").decode().strip() != head:
        raise ValueError("Root-selected committed HEAD changed")
    spec = row10.SPEC
    runtime.validate_spec(spec)
    if any(path.exists() or path.is_symlink() for path in (INNER, OUTER, METADATA)):
        raise ValueError("Pending release files are one-use and must be absent")
    if (ROOT / spec["native_output_directory"]).exists() or (
            ROOT / spec["sidecar_directory"]).exists():
        raise ValueError("Ordinal-10 attempt already exists")
    prior = small(PREVIOUS_INNER, ancestry.ROW9_INNER_SHA)
    prior_outer = small(PREVIOUS_OUTER, ancestry.ROW9_OUTER_SHA)
    sidecar9 = small(ROW9_SIDECAR, ancestry.ROW9_SIDECAR_SHA)
    native9 = small(ROW9_NATIVE, ancestry.ROW9_NATIVE_SHA)
    if (prior.get("status") != "root_released_one_native_call"
            or prior.get("ordinal") != 9
            or prior.get("run_id") != remaining.ORDER[9]
            or len(prior.get("prior_receipts", [])) != 9
            or prior["prior_receipts"][1]["sha256"] != remaining.N12_FAILED_RECEIPT_SHA
            or prior_outer.get("inner_release", {}).get("sha256") !=
               ancestry.ROW9_INNER_SHA
            or prior_outer.get("source_commit") != ancestry.ROW9_COMMIT
            or sidecar9.get("status") != ancestry.ROW9_STATUS
            or sidecar9.get("native_receipt_sha256") != ancestry.ROW9_NATIVE_SHA
            or native9.get("status") != "passed_numerical_software_only"
            or native9.get("source_commit") != ancestry.ROW9_COMMIT):
        raise ValueError("Ordinal-9 numerical-only ancestry differs")
    remaining.validate_preparation(json.loads((ROOT / remaining.PREPARATION).read_text()))
    sources.validate_binding_manifest(ROOT, inspect_sources=False)
    study, previous_study = v5.validate_preparation(ROOT)
    source_map = json.loads((ROOT / remaining.SOURCE_MAP).read_text())
    run_id = remaining.ORDER[10]
    item = source_map["source_decks"][source_map["run_source_keys"][run_id]]
    old_binding = {key: item[key] for key in ("path", "sha256")}
    mesh_binding = source_map["meshes"][item["mesh"]]
    old = sources._read_bound(ROOT, old_binding, maximum=sources.MAX_SOURCE_BYTES)
    mesh = sources._read_bound(ROOT, mesh_binding, maximum=sources.MAX_MESH_BYTES)
    native_domain = v5.run_spec(study, previous_study, run_id)["native_domain"]
    topology = sources.validate_topology_bc(json.loads(mesh), old,
                                            native_domain=native_domain)
    _, adapted, adapter = v5.adapt_deck(study, previous_study, run_id, old)
    old_bindings = {relative: {"path": relative, "sha256": source(head, relative)}
                    for relative in remaining.SOURCE_PATHS}
    extension_bindings = {relative: source(head, relative)
                          for relative in runtime.sources(spec)}
    if len(old_bindings) != 20 or len(extension_bindings) != 9:
        raise ValueError("Old and extension source closures differ")
    runtime.verify_sources(ROOT, head, extension_bindings, spec, require_head=True)
    inner = copy.deepcopy(prior)
    inner.update({
        "ordinal": 10, "run_id": run_id, "source_commit": head,
        "source_bindings": old_bindings,
        "old_source_deck": old_binding, "native_mesh": mesh_binding,
        "adapted_deck_sha256": sha(adapted.encode()),
        "adapter_receipt": adapter,
        "output_directory": remaining.output_directory(10),
        "caps": remaining.caps(10),
        "prior_receipts": [*prior["prior_receipts"], {
            "path": remaining.receipt_path(9), "run_id": remaining.ORDER[9],
            "sha256": ancestry.ROW9_NATIVE_SHA}],
    })
    if (inner["status"] != "root_released_one_native_call"
            or len(inner["prior_receipts"]) != 10
            or git("rev-parse", "HEAD").decode().strip() != head):
        raise ValueError("Frozen one-row release or HEAD differs")
    inner_raw = write_once(INNER, inner)
    outer = {
        "schema": "hbe-v5-later-continuation-envelope-v1",
        "status": "root_released_one_native_call_with_extension_ledger",
        "source_commit": head,
        "extension_source_bindings": extension_bindings,
        "inner_release": {"path": str(INNER.relative_to(ROOT)),
                          "sha256": sha(inner_raw)},
        "sidecar_directory": spec["sidecar_directory"],
        "policy": runtime.policy(spec),
        "prior_extension_descriptor": None,
    }
    outer_raw = write_once(OUTER, outer)
    metadata = {
        "status": "PENDING_ROOT_AUTHORIZATION_NO_NATIVE_ATTEMPT",
        "source_commit": head, "ordinal": 10, "run_id": run_id,
        "inner_sha256": sha(inner_raw), "outer_sha256": sha(outer_raw),
        "prior_inner_sha256": ancestry.ROW9_INNER_SHA,
        "prior_outer_sha256": ancestry.ROW9_OUTER_SHA,
        "prior_sidecar_sha256": ancestry.ROW9_SIDECAR_SHA,
        "prior_native_sha256": ancestry.ROW9_NATIVE_SHA,
        "old_source_count": len(old_bindings),
        "extension_source_count": len(extension_bindings),
        "old_source_deck_sha256": sha(old),
        "native_mesh_sha256": sha(mesh),
        "adapted_deck_sha256": sha(adapted.encode()),
        "adapter_receipt": adapter,
        "native_domain": native_domain,
        "native_nodes": topology["node_count"],
        "native_hex8": topology["hex8_count"],
        "old_predecessor_bulk_rehash": False,
        "native_calls_by_derivation": 0,
        "physical_validation_pass": None,
    }
    write_once(METADATA, metadata)
    return metadata


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Provide root-selected full source commit")
    print(json.dumps(derive(sys.argv[1]), sort_keys=True))
