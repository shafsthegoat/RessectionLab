#!/usr/bin/env python3
"""Single-attempt, bounded acquisition for the prospective VitalDB pilot only.

No library installation or upstream code execution. All transfers are HTTPS,
with no redirects or retries. Partial bytes and failure receipts are retained.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]
DECLARATION = ROOT / "manifests/vitaldb-development-v1.json"
DECLARATION_SHA256 = "8f2053f5b3a435cc3b84c27e868c22be93dd1e6995bd44ebc414cc9c5e5c98a0"
OUT = ROOT / "artifacts/vitaldb-recorded-component-v1"
BUILD = ROOT / "build/vitaldb-source-v1"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fetch(name: str, url: str, destination: Path, limit: int,
          expected_size: int | None = None, expected_hash: str | None = None) -> dict:
    if destination.exists():
        raise ValueError(f"Refusing overwrite or implicit retry: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    partial = destination.with_name(destination.name + "." + stamp + ".partial")
    header = BUILD / f"{name}-{stamp}.headers"
    started = time.monotonic()
    receipt = {"name": name, "url": url, "started_at_utc": stamp,
               "declaration_sha256": digest(DECLARATION),
               "acquisition_code_sha256": digest(Path(__file__)),
               "attempts": 1, "redirects_allowed": False, "max_seconds": 120,
               "max_bytes": limit, "status": "failed"}
    try:
        # curl bounds both total and connect time, and declines oversized Content-Length
        # and streamed bodies. Outer timeout is a second, independent process bound.
        proc = subprocess.run([
            "curl", "--silent", "--show-error", "--fail", "--proto", "=https",
            "--tlsv1.2", "--retry", "0", "--connect-timeout", "20",
            "--max-time", "120", "--max-filesize", str(limit),
            "--dump-header", str(header), "--output", str(partial),
            "--write-out", "%{http_code}", url,
        ], capture_output=True, text=True, timeout=125, check=False)
        receipt.update(exit_code=proc.returncode, http_status=proc.stdout,
                       stderr=proc.stderr[-2000:])
        if proc.returncode or proc.stdout != "200":
            raise ValueError("Transfer failed or non-200 response (redirects refused)")
        size, sha = partial.stat().st_size, digest(partial)
        if size > limit or (expected_size is not None and size != expected_size):
            raise ValueError("Download size mismatch")
        if expected_hash is not None and sha != expected_hash:
            raise ValueError("Download SHA256 mismatch")
        partial.rename(destination)
        receipt.update(status="acquired", bytes=size, sha256=sha,
                       local_path=str(destination.relative_to(ROOT)))
    except Exception as exc:
        receipt["error"] = str(exc)
        if partial.exists():
            receipt.update(partial_bytes=partial.stat().st_size,
                           partial_path=str(partial.relative_to(ROOT)))
        raise
    finally:
        receipt["elapsed_seconds"] = round(time.monotonic() - started, 6)
        if header.exists():
            receipt["headers_sha256"] = digest(header)
            receipt["headers_path"] = str(header.relative_to(ROOT))
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / f"acquisition-{name}-{stamp}.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=("rights-parser", "recording"))
    args = ap.parse_args()
    if digest(DECLARATION) != DECLARATION_SHA256:
        raise ValueError("Prospective declaration changed")
    m = json.loads(DECLARATION.read_text())
    BUILD.mkdir(parents=True, exist_ok=True)
    rights = BUILD / "PhysioNet-1.0.0-LICENSE.txt"
    archive = BUILD / "vitaldb-1.7.2.tar.gz"
    if args.stage == "rights-parser":
        license_receipt = fetch("rights", "https://physionet.org/files/vitaldb/1.0.0/LICENSE.txt",
                                rights, 1048576)
        license_text = rights.read_text()
        if "Attribution 4.0 International" not in license_text:
            raise ValueError("Expected CC BY 4.0 rights text missing; no payload admission")
        p = m["parser_reference"]
        fetch("parser", p["url"], archive, p["bytes"], p["bytes"], p["sha256"])
        with tarfile.open(archive, "r:gz") as tf:
            for suffix, output in (("vitaldb/utils.py", "upstream-utils.py"),
                                   ("LICENSE", "upstream-LICENSE.txt")):
                member = tf.getmember("vitaldb-1.7.2/" + suffix)
                if not member.isfile() or member.size > 1048576:
                    raise ValueError("Unsupported source archive member")
                data = tf.extractfile(member).read(1048577)
                (BUILD / output).write_bytes(data)
        if digest(BUILD / "upstream-utils.py") != p["utils_sha256"]:
            raise ValueError("Pinned parser source mismatch")
        mit = (BUILD / "upstream-LICENSE.txt").read_text()
        if "MIT License" not in mit or "Copyright" not in mit:
            raise ValueError("Expected MIT notice missing")
        # Preserve complete legal notices in Git; other downloaded bytes stay ignored.
        (OUT / "PhysioNet-CC-BY-4.0.txt").write_text(license_text)
        (OUT / "upstream-MIT-LICENSE.txt").write_text(mit)
        (OUT / "rights-and-parser.json").write_text(json.dumps({
            "schema": "resectionlab.vitaldb-rights.v1", "rights": license_receipt,
            "parser_reference": p, "retained_utils_path": str((BUILD / "upstream-utils.py").relative_to(ROOT)),
            "mit_sha256": digest(BUILD / "upstream-LICENSE.txt"),
            "source_citation": "Lee H, Jung C (2022), VitalDB v1.0.0, PhysioNet, doi:10.13026/czw8-9p62",
            "original_publication": "Lee HC et al., Scientific Data 9, 279 (2022), doi:10.1038/s41597-022-01411-5",
            "changes": "Narrow reader derived from MIT packet layout; no upstream execution, installation, sample helpers or API download.",
        }, indent=2) + "\n")
    else:
        evidence = json.loads((OUT / "rights-and-parser.json").read_text())
        if digest(rights) != evidence["rights"]["sha256"]:
            raise ValueError("Retained rights changed")
        if digest(archive) != m["parser_reference"]["sha256"]:
            raise ValueError("Retained pinned parser archive changed")
        s = m["source"]
        fetch("recording", s["file_url"], ROOT / s["local_path"], s["bytes"], s["bytes"], s["sha256"])


if __name__ == "__main__":
    main()
