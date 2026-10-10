#!/usr/bin/env python3
"""Fetch one pinned, hash-verified structural demonstration case.

The default manifest deliberately identifies a third-party public mirror.
Verification proves agreement with that pinned mirror, not with unavailable
TCIA source bytes. No credentials, full cohorts, or clinical labels are fetched.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Callable
from urllib.parse import urlparse
from urllib.request import Request, urlopen


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "manifests/data_acquisition_ucsf.json"
CHUNK_BYTES = 1024 * 1024


class AcquisitionError(RuntimeError):
    """A source file or transfer does not satisfy the pinned contract."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checked_path(output_root: Path, relative: str) -> Path:
    """Prevent even a locally edited manifest from escaping the data folder."""
    root = output_root.resolve()
    if Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise AcquisitionError(f"Unsafe destination path: {relative}")
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or path == root:
        raise AcquisitionError(f"Destination escapes output folder: {relative}")
    return path


def validate_manifest(manifest: dict, output_root: Path) -> list[dict]:
    files = manifest.get("files", [])
    if not files:
        raise AcquisitionError("Manifest contains no files")
    total = 0
    destinations = set()
    for entry in files:
        destination = checked_path(output_root, entry["path"])
        if destination in destinations:
            raise AcquisitionError(f"Duplicate destination: {entry['path']}")
        destinations.add(destination)
        if not isinstance(entry["size_bytes"], int) or entry["size_bytes"] <= 0:
            raise AcquisitionError("Every file requires a positive pinned size")
        if not re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]):
            raise AcquisitionError("Every file requires a pinned SHA256")
        url = urlparse(entry["source_url"])
        if url.scheme != "https" or not url.netloc or url.username or url.password:
            raise AcquisitionError("Source URLs must use HTTPS without credentials")
        total += entry["size_bytes"]
    ceiling = manifest["storage_forecast"]["max_download_bytes"]
    if total > ceiling:
        raise AcquisitionError(f"Manifest exceeds its {ceiling:,}-byte download limit")
    return files


def verify_file(path: Path, entry: dict) -> None:
    if path.stat().st_size != entry["size_bytes"]:
        raise AcquisitionError(f"Size mismatch: {path}")
    if sha256_file(path) != entry["sha256"]:
        raise AcquisitionError(f"SHA256 mismatch: {path}")


def acquire_file(entry: dict, output_root: Path, *, opener: Callable = urlopen) -> str:
    """Resume to a partial file, verify it, then publish it atomically.

    Existing files are immutable. An interrupted download remains resumable.
    A wrong hash remains visible as a failure and is never renamed as complete.
    """
    path = checked_path(output_root, entry["path"])
    if path.exists():
        verify_file(path, entry)
        return "already_verified"
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = checked_path(output_root, entry["path"] + ".partial")
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > entry["size_bytes"]:
        raise AcquisitionError(f"Partial file exceeds expected size: {partial}")
    if offset == entry["size_bytes"]:
        verify_file(partial, entry)
        partial.replace(path)
        return "completed_from_partial"

    headers = {"Accept-Encoding": "identity", "User-Agent": "RessectionLab-public-case/1"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    request = Request(entry["source_url"], headers=headers)
    with opener(request, timeout=45) as response:
        if urlparse(response.geturl()).scheme != "https":
            raise AcquisitionError("Download redirected away from HTTPS")
        if offset and response.status == 206:
            expected = f"bytes {offset}-{entry['size_bytes'] - 1}/{entry['size_bytes']}"
            if response.headers.get("Content-Range") != expected:
                raise AcquisitionError("Resume response has an unexpected Content-Range")
            mode = "ab"
        elif response.status == 200:
            # A server may ignore Range. Restart only this reproducible partial.
            offset = 0
            mode = "wb"
        else:
            raise AcquisitionError(f"Unexpected download HTTP status: {response.status}")
        if response.headers.get("Content-Encoding", "identity") != "identity":
            raise AcquisitionError("Source unexpectedly applied transport compression")
        content_length = response.headers.get("Content-Length")
        if content_length and int(content_length) != entry["size_bytes"] - offset:
            raise AcquisitionError("Response size differs from pinned source size")
        with partial.open(mode) as handle:
            size = offset
            while chunk := response.read(CHUNK_BYTES):
                size += len(chunk)
                if size > entry["size_bytes"]:
                    raise AcquisitionError("Download exceeds its pinned file size")
                handle.write(chunk)
            handle.flush()
            os.fsync(handle.fileno())
    verify_file(partial, entry)
    partial.replace(path)
    return "downloaded_and_verified"


def inspect_nifti(manifest: dict, output_root: Path) -> dict:
    """Perform structural integrity checks without claiming clinical anatomy QC."""
    try:
        import nibabel as nib
        import numpy as np
    except ImportError as error:
        raise AcquisitionError("NIfTI QC requires the project's nibabel and numpy dependencies") from error

    reports = []
    reference_shape = None
    reference_affine = None
    for entry in manifest["files"]:
        path = checked_path(output_root, entry["path"])
        image = nib.load(path)
        array = np.asanyarray(image.dataobj)
        qform, qcode = image.get_qform(coded=True)
        sform, scode = image.get_sform(coded=True)
        if array.ndim != 3 or not np.isfinite(array).all():
            raise AcquisitionError(f"Expected a finite structural 3-D volume: {path}")
        if not qcode and not scode:
            raise AcquisitionError(f"Volume lacks an authoritative NIfTI form: {path}")
        if qcode and scode and not np.allclose(qform, sform, atol=1e-4):
            raise AcquisitionError(f"qform/sform disagree: {path}")
        if not np.isfinite(image.affine).all() or abs(np.linalg.det(image.affine[:3, :3])) < 1e-9:
            raise AcquisitionError(f"Invalid voxel-to-world affine: {path}")
        if reference_shape is None:
            reference_shape, reference_affine = image.shape, image.affine
        elif image.shape != reference_shape or not np.allclose(image.affine, reference_affine, atol=1e-4):
            raise AcquisitionError(f"Structural volumes do not share geometry: {path}")
        item = {
            "path": entry["path"], "shape": list(image.shape),
            "spacing_mm": [float(x) for x in image.header.get_zooms()],
            "axis_codes": list(nib.aff2axcodes(image.affine)),
            "voxel_to_world": image.affine.tolist(),
            "qform_code": int(qcode), "sform_code": int(scode),
            "all_finite": True,
        }
        if entry["role"] == "supplied_tumor_annotation":
            values, counts = np.unique(array, return_counts=True)
            expected_labels = {int(x) for x in manifest["label_dictionary"]}
            if not set(values.tolist()).issubset(expected_labels) or not np.any(array):
                raise AcquisitionError("Tumor annotation contains unexpected labels or is empty")
            item["label_voxels"] = {str(int(v)): int(c) for v, c in zip(values, counts)}
        reports.append(item)
    return {"status": "structural_integrity_checks_passed", "files": reports,
            "limits": ["No source-byte equivalence to TCIA established", "No diffusion or functional localization verified", "No clinician anatomical review"]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "data")
    parser.add_argument("--verify-only", action="store_true", help="Hash existing files without downloading")
    parser.add_argument("--inspect-nifti", action="store_true", help="Also check NIfTI integrity and alignment")
    parser.add_argument("--dry-run", action="store_true", help="Print the fixed download scope without network access")
    args = parser.parse_args(argv)
    try:
        manifest = json.loads(args.manifest.read_text())
        entries = validate_manifest(manifest, args.output_root)
        total = sum(e["size_bytes"] for e in entries)
        if args.dry_run:
            print(json.dumps({"case_id": manifest["case_id"], "files": len(entries), "bytes": total,
                              "source": manifest["mirror"], "primary_source_byte_match": None}, indent=2))
            return 0
        args.output_root.mkdir(parents=True, exist_ok=True)
        if not args.verify_only and shutil.disk_usage(args.output_root).free < total:
            raise AcquisitionError("Insufficient local free space for the selected case")
        results = []
        for entry in entries:
            if args.verify_only:
                verify_file(checked_path(args.output_root, entry["path"]), entry)
                state = "verified"
            else:
                state = acquire_file(entry, args.output_root)
            results.append({"path": entry["path"], "status": state, "sha256": entry["sha256"]})
            print(f"{state}: {Path(entry['path']).name}", flush=True)
        report = {"case_id": manifest["case_id"], "source_manifest_sha256": sha256_file(args.manifest),
                  "files": results, "primary_source_byte_equivalence": "unverified"}
        if args.inspect_nifti:
            report["nifti_qc"] = inspect_nifti(manifest, args.output_root)
        report_path = args.output_root / f"{manifest['case_id']}_acquisition_report.json"
        # Avoid accepting a case identifier as a path in a custom manifest.
        report_path = checked_path(args.output_root, report_path.name)
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        print(f"Report: {report_path}")
        return 0
    except (AcquisitionError, OSError, ValueError, KeyError) as error:
        print(f"Acquisition failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
