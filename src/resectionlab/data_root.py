"""Guarded local data storage; never create a missing volume mount point."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import plistlib
import shutil
import subprocess
import sys
import uuid

ROOT_ENV = "RESSECTIONLAB_DATA_ROOT"
MARKER = ".ressectionlab-data-root.json"
LAYOUT = ("sources", "prepared", "cases/real", "cases/synthetic", "cases/controls",
          "catalog", "qc", "quarantine", "archive", "staging", "cache")


class DataRootError(ValueError):
    pass


def configured_root() -> Path:
    value = os.environ.get(ROOT_ENV)
    if not value or not Path(value).is_absolute():
        raise DataRootError(f"Set {ROOT_ENV} to the mounted data directory.")
    return Path(value)


def volume_info(root: Path) -> dict:
    """Inspect the existing mount without mkdir or an internal-disk fallback."""
    root = Path(root)
    if not root.is_absolute() or not root.is_dir() or root.is_symlink():
        raise DataRootError("External data directory is unavailable. Connect its volume.")
    if sys.platform != "darwin":
        raise DataRootError("External-volume verification currently requires macOS diskutil.")
    mount_candidate = root.resolve()
    while mount_candidate.parent != mount_candidate and not os.path.ismount(mount_candidate):
        mount_candidate = mount_candidate.parent
    try:
        info = plistlib.loads(subprocess.check_output(["diskutil", "info", "-plist", str(mount_candidate)], timeout=15))
    except (subprocess.SubprocessError, ValueError) as exc:
        raise DataRootError("Cannot verify the external data volume.") from exc
    mount = Path(info.get("MountPoint", "/"))
    if (not info.get("MountPoint") or info.get("Internal", True) or mount == Path("/")
            or not os.path.ismount(mount) or not root.resolve().is_relative_to(mount)
            or root.stat().st_dev != mount.stat().st_dev
            or not info.get("WritableVolume") or not info.get("VolumeUUID")):
        raise DataRootError("The data root must be on a mounted, writable external volume.")
    usage = shutil.disk_usage(root)
    return {"volume_uuid": info["VolumeUUID"], "mount_point": str(mount),
            "device": info.get("DeviceIdentifier"), "filesystem": info.get("FilesystemType"),
            "total_bytes": usage.total, "free_bytes": usage.free}


def checked_root(root=None, *, required_bytes=0, expected_uuid=None, reserve_bytes=None) -> Path:
    root = configured_root() if root is None else Path(root)
    info = volume_info(root)
    if expected_uuid and info["volume_uuid"] != expected_uuid:
        raise DataRootError("A different external volume is mounted at the configured path.")
    marker = root / MARKER
    if not marker.is_file() or marker.is_symlink() or marker.stat().st_size > 16384:
        raise DataRootError("The external data-root identity marker is missing or invalid.")
    document = json.loads(marker.read_text())
    if document.get("schema") != "ressectionlab-data-root-v1" or document.get("volume_uuid") != info["volume_uuid"]:
        raise DataRootError("External volume identity differs from the data-root record.")
    reserve = max(50 * 1024**3, info["total_bytes"] // 10) if reserve_bytes is None else reserve_bytes
    if required_bytes < 0 or reserve < 0 or info["free_bytes"] - required_bytes < reserve:
        raise DataRootError("Insufficient external free space after the configured reserve.")
    if not os.access(root, os.W_OK):
        raise DataRootError("External data root is not writable.")
    return root.resolve()


def initialize(root, *, expected_uuid) -> dict:
    root = Path(root)
    info = volume_info(root)
    if info["volume_uuid"] != expected_uuid:
        raise DataRootError("External volume UUID does not match the requested destination.")
    marker = root / MARKER
    if not marker.exists():
        atomic_json(marker, {"schema": "ressectionlab-data-root-v1", "volume_uuid": expected_uuid,
                             "root_id": str(uuid.uuid4()), "mount_point": info["mount_point"]})
    checked_root(root, expected_uuid=expected_uuid)
    for relative in LAYOUT:
        destination = safe_path(root, relative)
        destination.mkdir(parents=True, exist_ok=True)
    return info


def safe_path(root, relative) -> Path:
    root = Path(root).resolve()
    relative = Path(relative)
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise DataRootError("Data path must be relative to the configured root.")
    candidate = root / relative
    if not candidate.resolve().is_relative_to(root):
        raise DataRootError("Data path escapes the configured root.")
    return candidate


def sha256_file(path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def atomic_json(path, document):
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x") as stream:
            json.dump(document, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
