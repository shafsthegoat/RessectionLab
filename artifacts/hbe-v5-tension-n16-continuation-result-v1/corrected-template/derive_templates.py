"""Derive non-executable ordinal-9 HBE templates at one committed HEAD.

This reads only source/deck/mesh and small receipts. It never validates the
2.85 GB predecessor chain, writes a release, or launches FEBio. Root must
review the resulting bytes and separately issue any one-use release.
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
from launchers import hbe_v5_ordinal9_continuation_v1 as wrapper
from scripts import mechanics_hbe_branch_calibration_v5 as v5
from scripts import mechanics_hbe_v5_remaining_one_shot as remaining
from scripts import mechanics_hbe_v5_n12_v2_admission as admission
from scripts import mechanics_hbe_v5_source_bindings as sources

OUT = Path(__file__).resolve().parent
PREVIOUS_INNER = ROOT / "build/hbe-v5-tension-n12-s60-nocache-v2-final-candidate/inner-release.json"
PREVIOUS_SHA = "7f41865a1bfef4a785b4be262832af6d92d79472abd808a905773dc7ddc181d8"
ORDINAL8_RECEIPT_SHA = "67f7b82dca721d6d36b16e80e77c5d126b0ea9f9f510f8c2dd54b64716206a8e"
INNER = OUT / "INNER_TEMPLATE_NOT_RELEASED.json"
OUTER = OUT / "OUTER_TEMPLATE_NOT_RELEASED.json"
INDEX = 9


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def git(*args: str) -> bytes:
    return subprocess.run(["/usr/bin/git", *args], cwd=ROOT, check=True,
                          capture_output=True, timeout=10).stdout


def exact_small(path: Path, expected: str) -> bytes:
    before = path.lstat()
    if path.is_symlink() or not stat.S_ISREG(before.st_mode) or before.st_size > 1024**2:
        raise ValueError("Bound compact input is linked, absent or oversized")
    raw = path.read_bytes()
    after = path.lstat()
    ident = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns)
    if ident(before) != ident(after) or sha(raw) != expected:
        raise ValueError("Bound compact input bytes changed")
    return raw


def exact_source(head: str, relative: str) -> str:
    path = ROOT / relative
    before = path.lstat()
    if path.is_symlink() or not stat.S_ISREG(before.st_mode) or before.st_size > 1024**2:
        raise ValueError("Bound source is linked, absent or oversized: " + relative)
    raw = path.read_bytes()
    after = path.lstat()
    ident = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns)
    if ident(before) != ident(after) or git("show", head + ":" + relative) != raw:
        raise ValueError("Bound source differs from HEAD: " + relative)
    return sha(raw)


def write_json(path: Path, value: dict) -> bytes:
    raw = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    path.write_bytes(raw)
    return raw


def derive(head: str) -> dict:
    if git("rev-parse", "HEAD").decode().strip() != head:
        raise ValueError("Root-selected source HEAD changed")
    exact_v2 = admission.verify_exact(root=ROOT)  # small receipts and source only
    if (exact_v2["native_receipt_sha256"] != ORDINAL8_RECEIPT_SHA
            or exact_v2["physical_validation_pass"] is not None):
        raise ValueError("Ordinal-8 numerical-only ancestry differs")
    prior = json.loads(exact_small(PREVIOUS_INNER, PREVIOUS_SHA))
    if (prior.get("status") != "root_released_one_native_call"
            or prior.get("ordinal") != 8
            or prior.get("run_id") != remaining.ORDER[8]
            or len(prior.get("prior_receipts", [])) != 8):
        raise ValueError("Original ordinal-8 release identity differs")
    receipt8 = ROOT / remaining.receipt_path(8)
    exact_small(receipt8, ORDINAL8_RECEIPT_SHA)
    remaining.validate_preparation(json.loads((ROOT / remaining.PREPARATION).read_text()))
    sources.validate_binding_manifest(ROOT, inspect_sources=False)
    study, prior_study = v5.validate_preparation(ROOT)
    source_map = json.loads((ROOT / remaining.SOURCE_MAP).read_text())
    run_id = remaining.ORDER[INDEX]
    source_item = source_map["source_decks"][source_map["run_source_keys"][run_id]]
    old_binding = {key: source_item[key] for key in ("path", "sha256")}
    mesh_binding = source_map["meshes"][source_item["mesh"]]
    old = sources._read_bound(ROOT, old_binding, maximum=sources.MAX_SOURCE_BYTES)
    mesh = sources._read_bound(ROOT, mesh_binding, maximum=sources.MAX_MESH_BYTES)
    native_domain = v5.run_spec(study, prior_study, run_id)["native_domain"]
    topology = sources.validate_topology_bc(json.loads(mesh), old,
                                             native_domain=native_domain)
    _, adapted, adapter = v5.adapt_deck(study, prior_study, run_id, old)
    adapted_bytes = adapted.encode()
    old_bindings = {}
    for relative in remaining.SOURCE_PATHS:
        old_bindings[relative] = {"path": relative,
                                  "sha256": exact_source(head, relative)}
    extension_hashes = {relative: exact_source(head, relative)
                        for relative in wrapper.SOURCES}
    if len(old_bindings) != 20 or len(extension_hashes) != 4:
        raise ValueError("Expected frozen twenty plus four source closure")
    inner = copy.deepcopy(prior)
    inner.update({
        "status": "TEMPLATE_NOT_RELEASED",
        "ordinal": INDEX, "run_id": run_id, "source_commit": head,
        "source_bindings": old_bindings,
        "old_source_deck": old_binding, "native_mesh": mesh_binding,
        "adapted_deck_sha256": sha(adapted_bytes),
        "adapter_receipt": adapter,
        "output_directory": remaining.output_directory(INDEX),
        "caps": remaining.caps(INDEX),
        "prior_receipts": [*prior["prior_receipts"], {
            "path": remaining.receipt_path(8), "run_id": remaining.ORDER[8],
            "sha256": ORDINAL8_RECEIPT_SHA}],
    })
    if (inner["prior_receipts"][1]["sha256"] != remaining.N12_FAILED_RECEIPT_SHA
            or len(inner["prior_receipts"]) != 9
            or (ROOT / remaining.output_directory(INDEX)).exists()
            or (ROOT / remaining.output_directory(INDEX)).is_symlink()
            or (ROOT / wrapper.SIDECAR).exists()
            or (ROOT / wrapper.SIDECAR).is_symlink()):
        raise ValueError("Original N12 failure or one-use target boundary differs")
    if git("rev-parse", "HEAD").decode().strip() != head:
        raise ValueError("HEAD changed before template write")
    OUT.mkdir(parents=True, exist_ok=True)
    if INNER.exists() or OUTER.exists() or INNER.is_symlink() or OUTER.is_symlink():
        raise ValueError("Template output is one-use and must be absent")
    inner_raw = write_json(INNER, inner)
    outer = {
        "schema": "hbe-v5-ordinal9-continuation-envelope-v1",
        "status": "TEMPLATE_NOT_RELEASED", "source_commit": head,
        "extension_source_bindings": extension_hashes,
        "inner_release": {"path": str(INNER.relative_to(ROOT)),
                          "sha256": sha(inner_raw)},
        "sidecar_directory": wrapper.SIDECAR,
        "policy": wrapper.POLICY,
    }
    outer_raw = write_json(OUTER, outer)
    metadata = {
        "status": "TEMPLATE_NOT_RELEASED", "source_commit": head,
        "ordinal": INDEX, "run_id": run_id,
        "inner_sha256": sha(inner_raw), "outer_sha256": sha(outer_raw),
        "adapted_deck_sha256": sha(adapted_bytes),
        "adapter_receipt": adapter,
        "old_source_count": len(old_bindings),
        "extension_source_count": len(extension_hashes),
        "ordinal8_receipt_sha256": ORDINAL8_RECEIPT_SHA,
        "ordinal8_v2_sidecar_sha256": exact_v2["v2_sidecar_sha256"],
        "source_deck_sha256": sha(old), "mesh_sha256": sha(mesh),
        "native_domain": native_domain,
        "native_nodes": topology["node_count"],
        "native_hex8": topology["hex8_count"],
        "no_native_or_full_predecessor_read": True,
    }
    write_json(OUT / "TEMPLATE_METADATA.json", metadata)
    return metadata


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Expected one root-selected full source commit")
    print(json.dumps(derive(sys.argv[1]), sort_keys=True))
