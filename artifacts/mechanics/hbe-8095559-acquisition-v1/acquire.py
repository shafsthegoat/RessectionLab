"""Acquire the three prospectively pinned HBE files; never execute or extract them."""
from __future__ import annotations

import hashlib
import json
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
MANIFEST = ROOT / "manifests/hbe_8095559_acquisition.json"
MANIFEST_SHA256 = "dd3ebc07046834d6ae002524bbdfa0073ab8c7fa7a1cea8239b497266078b6b4"
OUT = Path(__file__).resolve().parent


class RejectRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise RuntimeError(f"Unexpected redirect {code}: {newurl}")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    assert sha(MANIFEST) == MANIFEST_SHA256
    declaration = json.loads(MANIFEST.read_text())
    destination = ROOT / declaration["raw_directory"]
    record = {
        "schema_version": 1,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_sha256": MANIFEST_SHA256,
        "driver_sha256": sha(Path(__file__)),
        "file_outcomes": [],
        "curve_values_inspected": False,
        "archive_extracted": False,
        "downloaded_code_executed": False,
    }
    receipt = OUT / "acquisition-receipt.json"
    assert not receipt.exists()
    opener = urllib.request.build_opener(RejectRedirects())
    started = time.monotonic()
    try:
        for source in declaration["files"]:
            target = destination / source["filename"]
            temporary = target.with_name(target.name + ".partial")
            if target.exists() or temporary.exists():
                raise FileExistsError(f"Refuse overwrite or implicit retry: {target}")
            md5, sha256 = hashlib.md5(), hashlib.sha256()
            outcome = {"filename": source["filename"], "request_url": source["url"]}
            record["file_outcomes"].append(outcome)
            with opener.open(source["url"], timeout=45) as response:
                if response.status != 200 or response.geturl() != source["url"]:
                    raise ValueError("Unexpected source response")
                length = response.headers.get("Content-Length")
                if length is not None and int(length) != source["bytes"]:
                    raise ValueError("Provider response length differs from declaration")
                outcome.update(status=response.status, final_url=response.geturl(),
                               response_headers=dict(response.headers))
                total = 0
                with temporary.open("xb") as handle:
                    while block := response.read(1024 * 1024):
                        total += len(block)
                        if total > source["bytes"]:
                            raise ValueError("Transfer exceeds declared byte count")
                        handle.write(block)
                        md5.update(block)
                        sha256.update(block)
            outcome.update(bytes=total, md5=md5.hexdigest(), sha256=sha256.hexdigest())
            if total != source["bytes"] or "md5:" + md5.hexdigest() != source["provider_checksum"]:
                raise ValueError("Provider length/checksum verification failed")
            temporary.rename(target)
            outcome.update(verified=True, local_path=str(target.relative_to(ROOT)))
            receipt.write_text(json.dumps(record, indent=2) + "\n")
            print(json.dumps(outcome), flush=True)
        record["status"] = "completed"
    except BaseException as exc:
        record.update(status="failed", failure_type=type(exc).__name__, failure=str(exc))
        raise
    finally:
        record.update(elapsed_seconds=time.monotonic() - started,
                      completed_at_utc=datetime.now(timezone.utc).isoformat(),
                      manifest_unchanged=sha(MANIFEST) == MANIFEST_SHA256)
        receipt.write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
