"""Small integrity and process controls for real-data intake.

No patient generation, scientific transformation or source eligibility decision.
"""
from __future__ import annotations

import hashlib
from contextlib import contextmanager
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import time


class IntakeDeadline(TimeoutError):
    pass


def check_deadline(deadline: float | None) -> None:
    if deadline is not None and time.monotonic() >= deadline:
        raise IntakeDeadline("Intake work budget expired")


@contextmanager
def termination_cleanup():
    """Route ordinary SIGTERM through worker cleanup and receipt finalizers.

    Repeated SIGTERM is ignored during cleanup. SIGKILL and filesystem failure
    cannot promise a final receipt; the worker has its own separate watchdog.
    Call from the main thread and scope around the entire batch.
    """
    previous = signal.getsignal(signal.SIGTERM)

    def terminate(signum, frame):
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        raise KeyboardInterrupt("Intake terminated by SIGTERM")

    signal.signal(signal.SIGTERM, terminate)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous)


def atomic_preserve(path: Path, payload: bytes) -> None:
    """Publish complete bytes without replacing an existing record.

    A hard interruption may leave a temporary sibling, never a partially
    published destination. The fsynced temporary is linked atomically.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix="." + path.name + ".", suffix=".partial", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.is_symlink() or path.read_bytes() != payload:
                raise ValueError(f"Preserve existing different record: {path}")
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        Path(temporary).unlink(missing_ok=True)


def verify_source_file(path: Path, entry: dict, *, receipt_sha: str | None = None,
                       deadline: float | None = None) -> str:
    """Stream size + published fixity + receipt identity in one bounded pass."""
    check_deadline(deadline)
    if path.is_symlink() or path.stat().st_size != entry["bytes"]:
        raise ValueError(f"Source size/type mismatch: {path}")
    sha = hashlib.sha256()
    md5 = hashlib.md5()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            check_deadline(deadline)
            sha.update(block)
            md5.update(block)
    check_deadline(deadline)
    digest = sha.hexdigest()
    if entry.get("expected_md5") and md5.hexdigest() != entry["expected_md5"]:
        raise ValueError(f"Published source MD5 mismatch: {path}")
    if entry.get("sha256") and digest != entry["sha256"]:
        raise ValueError(f"Published source SHA256 mismatch: {path}")
    if receipt_sha is not None and digest != receipt_sha:
        raise ValueError(f"Acquisition receipt SHA256 mismatch: {path}")
    return digest


def supervise(command: list[str], log: Path, *, deadline: float, on_start) -> tuple[str, int | None]:
    """Wait for one worker, killing its process group on timeout/interruption."""
    check_deadline(deadline)
    with log.open("xb") as output:
        process = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            on_start(process.pid)
            try:
                code = process.wait(timeout=max(0.001, deadline - time.monotonic()))
                return ("completed" if code == 0 else "worker_failed"), code
            except subprocess.TimeoutExpired:
                return "timeout", None
        finally:
            # Each worker is a dedicated group; do not leave descendants behind.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
