"""Exact previously reviewed RL256 loader, restricted to the local release.

This adapter never accepts a checkpoint or source path from IPC. The original
V3 safe loader remains the only deserializer, with weights_only=True and its
scoped NumPy allowlist. It binds the original two-step training environment;
the transfer projection is constructed only after that validation succeeds.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import stat
from types import ModuleType

from .legacy_transfer_episode import (CHECKPOINT_SHA, ORIGINAL_ARCHITECTURE_SHA,
    ORIGINAL_PARAMETER_SHA, ORIGINAL_VALIDATION_SHA)

LOADER_NAME = "legacy_checkpoint_rollout_v3.py"
LOADER_SHA256 = "b48831841b27742d123d1338ef2ff05005f39f1d6b755617c2c7bc959ea749e4"


def _root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    raise FileNotFoundError("The fixed local RL256 checkout and model asset are unavailable")


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _bounded_checkpoint_bytes(path: Path, expected_size: int, expected_sha256: str) -> bytes:
    if type(expected_size) is not int or not 0 < expected_size <= 394009:
        raise ValueError("Invalid fixed RL256 checkpoint byte bound")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size != expected_size:
            raise ValueError("Fixed RL256 checkpoint is not the exact regular file")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            payload = stream.read(expected_size + 1)
            if len(payload) != expected_size or stream.read(1):
                raise ValueError("Fixed RL256 checkpoint length changed during bounded read")
    finally:
        os.close(descriptor)
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ValueError("Fixed RL256 checkpoint bytes changed after preflight")
    return payload


def load_fixed_rl256_policy():
    """Load once in the bounded child; never invoked by workspace import."""
    root = _root()
    path = Path(__file__).with_name(LOADER_NAME)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 128 * 1024:
        raise ValueError("Reviewed V3 checkpoint loader source is not a bounded regular file")
    with path.open("rb") as source:
        payload = source.read(128 * 1024 + 1)
    if hashlib.sha256(payload).hexdigest() != LOADER_SHA256:
        raise ValueError("Reviewed V3 checkpoint loader source changed")
    # Compile precisely the bytes just hashed. importlib may reuse a valid
    # sibling .pyc even when a .py file has been checked separately.
    module = ModuleType("fixed_reviewed_rl256_loader_v3")
    module.__file__ = str(path)
    exec(compile(payload, str(path), "exec"), module.__dict__)
    # The reviewed one-shot file originally lived under build/. Its global
    # checkout path is relocated, without changing the byte-pinned loader
    # implementation, when this sibling source is installed in src/.
    module.ROOT = root
    module.ART = root / "artifacts/native-opening-rl-capacity-v1"
    module.CHECKPOINT = module.ART / "RL-256.pt"
    if (module.ROOT != root or module.EXPECTED["checkpoint_file_sha256"] != CHECKPOINT_SHA.removeprefix("sha256:")
            or module.EXPECTED["checkpoint_bytes"] != 394009 or
            module.EXPECTED["architecture_hash"] != ORIGINAL_ARCHITECTURE_SHA or
            module.EXPECTED["parameter_hash"] != ORIGINAL_PARAMETER_SHA):
        raise ValueError("Reviewed V3 release constants differ")
    old_task, release = module.preflight()  # Original source and bytes, no deserialization yet.
    # The reviewed V3 function calls CHECKPOINT.read_bytes() after preflight.
    # Feed it exactly one bounded, independently rechecked byte string instead
    # of reopening an unbounded path after the streaming preflight hash.
    checkpoint_path = module.CHECKPOINT
    expected_size = module.EXPECTED["checkpoint_bytes"]
    checkpoint_bytes = _bounded_checkpoint_bytes(checkpoint_path, expected_size,
        module.EXPECTED["checkpoint_file_sha256"])

    class _VerifiedCheckpointBytes:
        def read_bytes(self):
            return checkpoint_bytes

    module.CHECKPOINT = _VerifiedCheckpointBytes()
    try:
        policy, identity, receipt = module.load_bound_policy(old_task, release)
    finally:
        module.CHECKPOINT = checkpoint_path
    if (identity.checkpoint_sha256 != CHECKPOINT_SHA or
            identity.checkpoint_validation_receipt_sha256 != ORIGINAL_VALIDATION_SHA or
            policy.architecture_hash != ORIGINAL_ARCHITECTURE_SHA or
            identity.parameter_hash != ORIGINAL_PARAMETER_SHA):
        raise ValueError("Safely loaded RL256 differs from the fixed original identity")
    return policy, identity, receipt
