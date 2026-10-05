#!/usr/bin/env python3
"""Lossless copies of three pilot JSON payloads; never change/delete raw evidence."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

TARGETS = ("pilot/accounting.json", "pilot/history-freeze.json", "pilot/candidate-record.json")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def pack(root):
    authority = json.loads((root / "launcher-status.json").read_text())
    if authority.get("status") not in ("completed", "failed"):
        raise ValueError("Only a final attempt can be packaged")
    rows = []
    for name in TARGETS:
        path = root / name
        raw = path.read_bytes()
        compressed = gzip.compress(raw, compresslevel=9, mtime=0)
        destination = path.with_suffix(path.suffix + ".gz")
        if destination.exists():
            if destination.read_bytes() != compressed:
                raise FileExistsError("Existing gzip differs; evidence will not be overwritten")
        else:
            with destination.open("xb") as stream:
                stream.write(compressed)
        recovered = gzip.decompress(destination.read_bytes())
        if recovered != raw or json.loads(recovered) != json.loads(raw):
            raise ValueError("Compression changed evidence")
        if path.read_bytes() != raw:
            raise ValueError("Original evidence changed during packaging")
        rows.append({"raw_path": name, "gzip_path": str(destination.relative_to(root)),
            "raw_bytes": len(raw), "gzip_bytes": len(compressed), "raw_sha256": digest(raw),
            "gzip_sha256": digest(compressed), "decompressed_sha256": digest(recovered),
            "byte_roundtrip_equal": True, "json_roundtrip_equal": True,
            "raw_original_retained_unchanged": True})
    return {"scope": "Lossless storage copies only; no experiment receipt or runtime source changed",
        "files": rows, "total_raw_bytes": sum(row["raw_bytes"] for row in rows),
        "total_gzip_bytes": sum(row["gzip_bytes"] for row in rows),
        "pack_script_sha256": digest(Path(__file__).read_bytes()),
        "report_consumer": "scripts/report_native_axis_pilot.py accepts .json.gz when raw JSON is absent and refuses raw/gzip disagreement"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    result = pack(args.run)
    with args.receipt.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")


if __name__ == "__main__":
    main()
