"""Derive non-executable ordinal-8 v2 old/outer templates at one HEAD.

This performs only small source/release reads. It never validates predecessor
outputs, creates attempt directories, reads measured responses, or runs FEBio.
It requires the reviewed v2 launcher to have been committed by root first.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from launchers import hbe_v5_nocache_v2 as launcher

ORIGINAL = ROOT / "build/hbe-v5-tension-n12-s60-release-prep/RELEASE_TEMPLATE.json"
ORIGINAL_SHA256 = "32d7566d10d1a671fe1d7fcd2cb3bfc73dc70c3632d286480fd860e2e15d2feb"
OUT = Path(__file__).resolve().parent
INNER = OUT / "INNER_TEMPLATE_NOT_RELEASED.json"
OUTER = OUT / "OUTER_TEMPLATE_NOT_RELEASED.json"


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def git(*args: str) -> bytes:
    return subprocess.run(["/usr/bin/git", *args], cwd=ROOT, check=True,
                          capture_output=True, timeout=10).stdout


def write_json(path: Path, value: dict) -> bytes:
    raw = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    path.write_bytes(raw)
    return raw


def main() -> None:
    head = git("rev-parse", "HEAD").decode().strip()
    original_raw = ORIGINAL.read_bytes()
    if sha(original_raw) != ORIGINAL_SHA256:
        raise ValueError("Original audited ordinal-8 template changed")
    inner = json.loads(original_raw)
    if (inner["status"] != "TEMPLATE_NOT_RELEASED"
            or inner["ordinal"] != 8
            or inner["run_id"] != "tension:N12:S60:reference"
            or set(inner["source_bindings"]) != {
                "scripts/febio_runtime.py", "scripts/mechanics_febio_verification.py",
                "scripts/mechanics_hbe_access.py", "scripts/mechanics_hbe_backend.py",
                "scripts/mechanics_hbe_branch_calibration_v4.py",
                "scripts/mechanics_hbe_branch_calibration_v5.py",
                "scripts/mechanics_hbe_evaluation.py",
                "scripts/mechanics_hbe_halfheight_readout.py",
                "scripts/mechanics_hbe_outputs.py", "scripts/mechanics_hbe_physics.py",
                "scripts/mechanics_hbe_readout.py", "scripts/mechanics_hbe_v5_frame.py",
                "scripts/mechanics_hbe_v5_n12_admission.py",
                "scripts/mechanics_hbe_v5_n12_saved_replay.py",
                "scripts/mechanics_hbe_v5_n8_canary.py",
                "scripts/mechanics_hbe_v5_n8_one_shot.py",
                "scripts/mechanics_hbe_v5_remaining_one_shot.py",
                "scripts/mechanics_hbe_v5_source_bindings.py",
                "scripts/mechanics_hbe_v5_stream.py",
                "scripts/mechanics_patient_constraints.py"}):
        raise ValueError("Original frozen row identity or source closure changed")
    for relative, binding in inner["source_bindings"].items():
        if relative != binding["path"]:
            raise ValueError("Frozen source binding path differs")
        current = (ROOT / relative).read_bytes()
        committed = git("show", head + ":" + relative)
        if current != committed or sha(current) != binding["sha256"]:
            raise ValueError("Frozen HBE source changed: " + relative)
    extensions = {}
    for relative in launcher.SOURCES:
        current = (ROOT / relative).read_bytes()
        committed = git("show", head + ":" + relative)
        if current != committed:
            raise ValueError("Extension source differs from HEAD: " + relative)
        extensions[relative] = sha(current)
    if git("rev-parse", "HEAD").decode().strip() != head:
        raise ValueError("HEAD changed during template derivation")
    inner["source_commit"] = head
    inner_raw = write_json(INNER, inner)
    outer = {
        "schema": "hbe-v5-nocache-launch-envelope-v2",
        "status": "TEMPLATE_NOT_RELEASED",
        "source_commit": head,
        "extension_source_bindings": extensions,
        "inner_release": {"path": str(INNER.relative_to(ROOT)),
                          "sha256": sha(inner_raw)},
        "sidecar_directory": launcher.SIDECAR,
        "policy": launcher.POLICY,
    }
    outer_raw = write_json(OUTER, outer)
    print(json.dumps({"source_commit": head, "inner_template_sha256": sha(inner_raw),
                      "outer_template_sha256": sha(outer_raw),
                      "old_source_count": len(inner["source_bindings"]),
                      "extension_source_hashes": extensions,
                      "status": "TEMPLATE_NOT_RELEASED"}, sort_keys=True))


if __name__ == "__main__":
    main()
