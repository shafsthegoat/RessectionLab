#!/usr/bin/env python3
"""Prepare only the frozen VitalDB DEVELOPMENT case3 and MIT parser caches.

This is a local reproduction invocation, separate from the historical study's
one-shot acquisition receipts. No source/role/clock declaration is rewritten.
Existing caches must verify; corruption and earlier failed attempts refuse.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import datetime as dt
import hashlib
import io
import json
import os
from pathlib import Path
import stat
import subprocess
import tarfile
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
STATE = Path("build/vitaldb-cache-preparation-v1")
REFERENCES = {
    "manifests/vitaldb-development-v1.json": "8f2053f5b3a435cc3b84c27e868c22be93dd1e6995bd44ebc414cc9c5e5c98a0",
    "manifests/vitaldb-official-mirror-attempt-v1.json": "3cf0fb91715286e3cad2425aa58eb12acf4811becbd96bc40d2f4e96640ae2a5",
    "manifests/vitaldb-native-component-v1.json": "7170c6d5c521b845e937eb69b16d0083650823ae8da4a6da201eeff71bfe9679",
    "manifests/vitaldb-native-clock-v1.json": "11bf6b1b97cdf894a64f4c0110a2d52f053bcb6ca52237bf2515c417df253dc8",
    "artifacts/vitaldb-recorded-component-v1/PhysioNet-CC-BY-4.0.txt": "9a78e7f22742dde9f66ae235ec793ba2212019dc4d0ced75c4da09ced0b35fb2",
    "artifacts/vitaldb-recorded-component-v1/upstream-MIT-LICENSE.txt": "87fa7cab278f36431b02007807f76c1b0b4d8534777d2337f1f1902294eea1d2",
    "artifacts/vitaldb-recorded-component-v1/rights-and-parser.json": "eb43776083dced77505dee6aa57c3f9bce4662cf34c140056f3eff67568913a5",
}


class CacheError(ValueError):
    pass


@dataclass(frozen=True)
class Asset:
    name: str
    relative_path: str
    size: int
    sha256: str
    url: str = ""
    seconds: int = 0


PARSER = Asset("parser", "build/vitaldb-source-v1/vitaldb-1.7.2.tar.gz", 67723,
    "70e0ce784b13d52bbf6a315b742ab06d5ed0599b8b01f76182971a391dd64130",
    "https://files.pythonhosted.org/packages/f4/1c/f48900276672017f7f72f3d2c0b94ac435c80fc0907ec663cedb6be76cf3/vitaldb-1.7.2.tar.gz", 120)
RECORDING = Asset("recording", "data/vitaldb-1.0.0/mirror-01/0003.vital", 6537712,
    "573db0941d580167833f84497f5d1c4a1391443eeec3f95852287ea09db024e4",
    "https://physionet-open.s3.amazonaws.com/vitaldb/1.0.0/vital_files/0003.vital", 300)
UTILS = Asset("utils", "build/vitaldb-source-v1/upstream-utils.py", 118991,
    "b0e88b1365c8d9a88814123c9a5b5a5d3c696a5c4849c7db3ebda476cc7134e3")


def _read_verified(path: Path, sha256: str, limit: int, exact_size: int | None = None) -> bytes:
    if path.is_symlink():
        raise CacheError(f"Symlink cache/reference refused: {path}")
    # Nonblocking/no-follow open also refuses FIFOs and a last-component symlink
    # installed between the initial path check and opening the file.
    descriptor = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW)
    with os.fdopen(descriptor, "rb") as handle:
        info = os.fstat(handle.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size > limit:
            raise CacheError(f"Nonregular or oversized file: {path}")
        if exact_size is not None and info.st_size != exact_size:
            raise CacheError(f"Frozen size mismatch: {path}")
        data = handle.read(limit + 1)
    if len(data) > limit or (exact_size is not None and len(data) != exact_size) or hashlib.sha256(data).hexdigest() != sha256:
        raise CacheError(f"Frozen SHA256/size mismatch: {path}")
    return data


def _json_new(path: Path, content: dict) -> None:
    with path.open("x") as handle:
        json.dump(content, handle, indent=2, allow_nan=False)
        handle.write("\n")


def _contained(root: Path, relative: str | Path) -> Path:
    path = root / relative
    if not path.resolve().is_relative_to(root.resolve()):
        raise CacheError(f"Cache path escapes checkout: {relative}")
    return path


def _curl(asset: Asset, partial: Path, headers: Path) -> dict:
    process = subprocess.run([
        "curl", "--disable", "--no-location", "--silent", "--show-error", "--fail",
        "--proto", "=https", "--tlsv1.2", "--retry", "0", "--connect-timeout", "20",
        "--max-time", str(asset.seconds), "--max-filesize", str(asset.size),
        "--dump-header", str(headers), "--output", str(partial),
        "--write-out", "%{http_code}", asset.url,
    ], capture_output=True, text=True, timeout=asset.seconds + 5, check=False)
    return {"exit_code": process.returncode, "http_status": process.stdout,
            "stderr": process.stderr[-2000:]}


def _download(root: Path, state: Path, asset: Asset, transport) -> dict:
    destination = _contained(root, asset.relative_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    marker = state / f"{asset.name}-attempt.json"
    partial = state / f"{asset.name}.partial"
    headers = state / f"{asset.name}.headers"
    receipt_path = state / f"{asset.name}-receipt.json"
    if any(path.exists() or path.is_symlink() for path in (marker, partial, headers, receipt_path)):
        raise CacheError(f"Prior local {asset.name} attempt exists; automatic retry refused")
    receipt = {"asset": asset.name, "url": asset.url, "destination": asset.relative_path,
        "expected_bytes": asset.size, "expected_sha256": asset.sha256,
        "max_seconds": asset.seconds, "attempts": 1, "redirects": False,
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "started_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "status": "failed"}
    # Reserve before contacting the network. Markers/partials survive failure and
    # process interruption; historical committed acquisition receipts are untouched.
    _json_new(marker, receipt)
    started = time.monotonic()
    try:
        with partial.open("xb"):
            pass
        receipt.update(transport(asset, partial, headers))
        if receipt["exit_code"] or receipt["http_status"] != "200":
            raise CacheError("Transfer failed/non-200; no redirects or retries")
        _read_verified(partial, asset.sha256, asset.size, asset.size)
        # Atomic, no-replace publication; a concurrent destination is never erased.
        os.link(partial, destination)
        partial.unlink()
        receipt.update(status="downloaded", bytes=asset.size, sha256=asset.sha256)
        return receipt
    except Exception as exc:
        receipt["error"] = str(exc)
        raise
    finally:
        receipt["elapsed_seconds"] = round(time.monotonic() - started, 6)
        if partial.exists():
            receipt["partial_bytes"] = partial.stat().st_size
        _json_new(receipt_path, receipt)


def _extract_utils(root: Path, state: Path) -> None:
    archive = _read_verified(root / PARSER.relative_path, PARSER.sha256, PARSER.size, PARSER.size)
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as tf:
        selected = [m for m in tf.getmembers() if m.name == "vitaldb-1.7.2/vitaldb/utils.py"]
        if len(selected) != 1 or not selected[0].isfile() or selected[0].size > 1048576:
            raise CacheError("Unsupported source archive utils.py member")
        data = tf.extractfile(selected[0]).read(1048577)
    if len(data) != UTILS.size or hashlib.sha256(data).hexdigest() != UTILS.sha256:
        raise CacheError("Authenticated utils.py member mismatch")
    temporary = state / ("utils-" + uuid.uuid4().hex + ".verified")
    with temporary.open("xb") as handle:
        handle.write(data)
    os.link(temporary, _contained(root, UTILS.relative_path))
    temporary.unlink()


def prepare(root: Path = ROOT, *, offline: bool = False, transport=None) -> dict:
    """Local cache preparation; transport injection is only for isolated tests."""
    state = _contained(root, STATE)
    state.mkdir(parents=True, exist_ok=True)
    result = {"schema": "resectionlab.vitaldb-cache-preparation.v1", "status": "refused",
        "scope": "reproduce existing DEVELOPMENT case3/subject2861; no new study or source selection",
        "offline": offline, "network_attempts": 0, "downloaded_bytes": 0,
        "new_people": 0, "training_examples": 0, "RL_transitions": 0,
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "started_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "assets": {}}
    try:
        for relative, sha in REFERENCES.items():
            _read_verified(root / relative, sha, 1048576)
        component = json.loads((root / "manifests/vitaldb-native-component-v1.json").read_text())
        evidence = component["native_header_evidence"]
        _read_verified(root / evidence["path"], evidence["sha256"], 1048576)
        result["frozen_reference_hashes"] = dict(REFERENCES)
        # Validate every existing asset before any transfer. Never repair corrupt
        # caches or erase an existing different utils.py during extraction.
        for asset in (PARSER, UTILS, RECORDING):
            path = _contained(root, asset.relative_path)
            if path.exists() or path.is_symlink():
                _read_verified(path, asset.sha256, asset.size, asset.size)
                result["assets"][asset.name] = "verified_cached"
        missing_downloads = [a for a in (PARSER, RECORDING) if a.name not in result["assets"]]
        if offline and missing_downloads:
            raise CacheError("Offline cache is incomplete; no network attempt made")
        def counted_transport(asset, partial, headers):
            result["network_attempts"] += 1
            return (transport or _curl)(asset, partial, headers)
        for asset in (PARSER, RECORDING):
            if asset.name not in result["assets"]:
                _download(root, state, asset, counted_transport)
                result["downloaded_bytes"] += asset.size
                result["assets"][asset.name] = "downloaded_verified"
            if asset is PARSER and "utils" not in result["assets"]:
                _extract_utils(root, state)
                result["assets"]["utils"] = "extracted_verified"
        result["status"] = "ready"
    except Exception as exc:
        result["error"] = str(exc)
    finally:
        result["receipt_path"] = str(STATE / ("run-" + uuid.uuid4().hex + ".json"))
        _json_new(root / result["receipt_path"], result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true", help="Verify/extract local caches only; refuse if either download is missing")
    args = parser.parse_args()
    result = prepare(offline=args.offline)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0 if result["status"] == "ready" else 2


if __name__ == "__main__":
    raise SystemExit(main())
