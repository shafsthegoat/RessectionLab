"""Acquire only the prospectively pinned ReMIND-001 preoperative development slice.

No collection search, diagnosis-based reranking, or automatic case preparation occurs
here. Run with the isolated, version-recorded IDC acquisition environment. Image
objects are source-pinned by IDC series UUID, S3 object identity, length and ETag.
The ordinary single-part S3 ETag is checked as MD5; SHA256 is recorded on receipt.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import urllib.request

MANIFEST_SHA256 = "3dad212f49655937d8ef3d0dff998eb796de9f885c4328e9f4202503157ac08f"


class RejectRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError("Redirect from pinned IDC object is not permitted")


def validate_manifest(manifest: dict) -> list[dict]:
    if manifest["patient_id"] != "ReMIND-001":
        raise ValueError("This pilot is restricted to the declared development patient")
    roles = {"structural_t1ce", "structural_t2", "cerebrum_annotation", "tumor_annotation"}
    if len(manifest["series"]) != 4 or {s["role"] for s in manifest["series"]} != roles:
        raise ValueError("Unexpected series selection")
    objects = []
    for series in manifest["series"]:
        if (series["PatientID"] != "ReMIND-001" or series["StudyDescription"] != "Preop"
                or series["license_short_name"] != "CC BY 4.0"):
            raise ValueError("Unapproved patient, visit, or license")
        if len(series["objects"]) != series["instanceCount"]:
            raise ValueError("Instance count mismatch")
        for obj in series["objects"]:
            expected_prefix = series["crdc_series_uuid"] + "/"
            if (not obj["key"].startswith(expected_prefix) or not obj["key"].endswith(".dcm")
                    or ".." in obj["key"] or obj["key"].count("/") != 1
                    or obj["url"] != "https://idc-open-data.s3.amazonaws.com/" + obj["key"]
                    or not 0 < obj["bytes"] <= 16 * 1024**2
                    or len(obj["etag_md5"]) != 32
                    or any(c not in "0123456789abcdef" for c in obj["etag_md5"])):
                raise ValueError("Invalid pinned IDC object")
            objects.append({**obj, "role": series["role"]})
    if (len(objects) != 386 or len({o["key"] for o in objects}) != 386
            or sum(o["bytes"] for o in objects) != 59_163_552
            or manifest["expected_total_bytes"] != 59_163_552):
        raise ValueError("Pilot count or byte budget differs from declared selection")
    return objects


def acquire_one(obj: dict, output: Path) -> dict:
    destination = output / obj["key"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    cached = destination.exists()
    if cached:
        data = destination.read_bytes()
    else:
        request = urllib.request.Request(obj["url"], headers={"If-Match": '"' + obj["etag_md5"] + '"'})
        with urllib.request.build_opener(RejectRedirects()).open(request, timeout=45) as response:
            if (response.status != 200 or response.url != obj["url"]
                    or int(response.headers.get("Content-Length", "-1")) != obj["bytes"]
                    or response.headers.get("ETag", "").strip('"') != obj["etag_md5"]):
                raise ValueError("IDC response identity or size mismatch")
            data = response.read(obj["bytes"] + 1)
    if len(data) != obj["bytes"] or hashlib.md5(data).hexdigest() != obj["etag_md5"]:
        raise ValueError("IDC object length or MD5 mismatch: " + obj["key"])
    if not cached:
        pending = destination.with_suffix(".pending")
        pending.write_bytes(data)
        pending.replace(destination)
    return {"key": obj["key"], "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
            "md5_verified": True, "cached": cached, "seconds": time.monotonic() - started}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    raw = args.manifest.read_bytes()
    if hashlib.sha256(raw).hexdigest() != MANIFEST_SHA256:
        raise ValueError("Manifest differs from the prospectively declared four-series selection")
    manifest = json.loads(raw)
    objects = validate_manifest(manifest)
    if args.receipt.exists():
        raise ValueError("Refusing to overwrite a prior attempt receipt")
    record = {"schema": "resectionlab.remind-acquisition-receipt.v1",
              "started_utc": datetime.now(timezone.utc).isoformat(),
              "manifest_sha256": hashlib.sha256(raw).hexdigest(),
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "status": "running", "objects": []}
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(record, indent=2) + "\n")
    started = time.monotonic()
    try:
        with ThreadPoolExecutor(max_workers=4) as executor:
            for result in executor.map(lambda obj: acquire_one(obj, args.output), objects):
                record["objects"].append(result)
        record["status"] = "complete"
    except Exception as exc:
        record["status"] = "failed"
        record["error"] = repr(exc)
        raise
    finally:
        record["elapsed_seconds"] = time.monotonic() - started
        record["finished_utc"] = datetime.now(timezone.utc).isoformat()
        record["actual_bytes_verified"] = sum(o["bytes"] for o in record["objects"])
        args.receipt.write_text(json.dumps(record, indent=2) + "\n")
        print(json.dumps({k: v for k, v in record.items() if k != "objects"}, indent=2))


if __name__ == "__main__":
    main()
