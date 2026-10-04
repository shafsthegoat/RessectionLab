#!/usr/bin/env python3
"""Acquire one of the pinned, source-verified BTC_preop development cases.

PAT28 defaults to 15 files with diffusion; PAT05 contains 7 structural files.
The frozen metadata queue additionally permits structural-only PAT16/PAT20.
Original images, orientations and annotations remain unchanged. Every case
is permanently assigned to development, never outer-final evaluation.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
from typing import Callable
from urllib.parse import parse_qs, quote, urlparse
from urllib.request import Request, urlopen

# Keep transport, resumable partials and corruption handling identical to the
# other public-data fetcher; BTC adds its own source and annex provenance gates.
from acquire_public_case import (
    AcquisitionError,
    CHUNK_BYTES,
    acquire_file as _acquire_file,
    checked_path,
    sha256_file,
    validate_manifest as _validate_transfer_manifest,
    verify_file as _verify_transfer_file,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "manifests/btc_acquisition.json"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "data/diffusion_source/ds001226-v5.0.1"
COMMIT = "359d372c5e972a161966312128adb365870df949"
SUBJECT = "sub-PAT28"
RAW_BASE = f"https://raw.githubusercontent.com/OpenNeuroDatasets/ds001226/{COMMIT}/"
S3_BASE = "https://s3.amazonaws.com/openneuro.org/ds001226/"
MAX_DOWNLOAD_BYTES = 64_808_763
QUEUED_SUBJECTS = ("sub-PAT16", "sub-PAT20")
QUEUE_MANIFEST = REPO_ROOT / "manifests/development_acquisition_queue.json"
QUEUE_SHA256 = "a768aaefd78146540cf20c13193e22268344206a4a396ffb654e7d40af7b8898"
REVIEWED_MODES = {"sub-PAT28": "diffusion", "sub-PAT05": "structural",
                  **dict.fromkeys(QUEUED_SUBJECTS, "structural")}
DOWNLOAD_LIMITS = {"sub-PAT28": MAX_DOWNLOAD_BYTES, "sub-PAT05": 100_000_000,
                   "sub-PAT16": 18_210_576, "sub-PAT20": 17_546_223}
STRUCTURAL_SELECTION = REPO_ROOT / "manifests/btc_pat05_selection.json"


def select_additional_structural_subject(metadata: str) -> str:
    """Reproduce the locked metadata-only selection, without viewing images."""
    rows = csv.DictReader(metadata.splitlines(), delimiter="\t")
    candidates = sorted(row["participant_id"] for row in rows
                        if row["participant_id"] != SUBJECT
                        and any(term in row.get("tumor type & grade", "").lower()
                                for term in ("glioma", "glioblastoma", "astrocytoma")))
    if not candidates:
        raise AcquisitionError("No eligible additional glioma subject in the source metadata")
    return candidates[0]


def selected_queued_subjects(metadata: str) -> list[str]:
    """Extend the original selection rule without using images or outcomes."""
    rows = csv.DictReader(metadata.splitlines(), delimiter="\t")
    return sorted(row["participant_id"] for row in rows
                  if row["participant_id"] not in {"sub-PAT05", "sub-PAT28"}
                  and any(term in row.get("tumor type & grade", "").lower()
                          for term in ("glioma", "glioblastoma", "astrocytoma")))[:2]


def queued_manifest(subject: str) -> dict:
    """Build a structural acquisition contract from the exact frozen queue.

    Image SHA256 values remain absent until published annex MD5 and size have
    been checked against downloaded bytes. This never permits arbitrary IDs.
    """
    if subject not in QUEUED_SUBJECTS:
        raise AcquisitionError("Subject is outside the frozen development queue")
    if sha256_file(QUEUE_MANIFEST) != QUEUE_SHA256:
        raise AcquisitionError("Development queue differs from its frozen identity")
    queue = json.loads(QUEUE_MANIFEST.read_text())
    case = next(item for item in queue["candidates"] if item["subject"] == subject)
    if (queue["selection"]["selected_subjects"] != list(QUEUED_SUBJECTS)
            or case["outer_role"] != "development" or case["outer_role_locked"] is not True
            or case["overlap"]["eligible_for_outer_final"] is not False):
        raise AcquisitionError("Queued case must retain its locked development role")
    entries = [dict(item) for item in queue["shared_release_files"]]
    for item in case["files"]:
        if item["scope"] != "structural_minimum":
            continue
        entry = {"path": item["path"], "git_url": RAW_BASE + quote(item["path"]),
                 "source_url": item["source_url"], "bytes": item["expected_bytes"],
                 "sha256": item.get("sha256")}
        if "expected_annex_md5" in item:
            entry.update(expected_md5=item["expected_annex_md5"],
                         expected_bytes=item["expected_bytes"])
        entries.append(entry)
    return {"schema_version": 1, "dataset": "BTC_preop", "accession": "ds001226",
            "release": "5.0.1", "git_commit": COMMIT, "license": "CC0",
            "doi": queue["source"]["doi"], "primary_source": queue["source"]["primary_source"],
            "subject": subject, "role": "development", "outer_role_locked": True,
            "eligible_for_outer_final": False, "acquisition_mode": "structural",
            "selection_queue": "manifests/development_acquisition_queue.json",
            "selection_queue_sha256": QUEUE_SHA256, "files": entries,
            "storage_forecast": {"max_download_bytes": DOWNLOAD_LIMITS[subject]},
            "annotation_semantics": {"kind": "supplied_source_annotation_pending_geometry_QC",
                                     "clinical_probability": False, "source_values_preserved": True},
            "clinical_context": {"preoperative_availability": "unknown",
                                 "eligible_as_preoperative_policy_input": False}}


def _validate_queue_binding(manifest: dict) -> None:
    expected = queued_manifest(manifest["subject"])
    for key in ("role", "outer_role_locked", "eligible_for_outer_final", "acquisition_mode",
                "selection_queue", "selection_queue_sha256", "annotation_semantics", "clinical_context"):
        if manifest.get(key) != expected[key]:
            raise AcquisitionError(f"Manifest changed frozen development selection field: {key}")
    pinned = {item["path"]: item for item in expected["files"]}
    for entry in manifest["files"]:
        source = pinned.get(entry["path"])
        if source is None:
            raise AcquisitionError("Manifest contains a file outside the frozen structural queue")
        for key, value in source.items():
            # SHA256 is established only after a new image matches its annex
            # checksum. Metadata SHA256 and all source identity fields are fixed.
            if key == "sha256" and value is None:
                continue
            if entry.get(key) != value:
                raise AcquisitionError(f"Manifest differs from frozen source metadata: {entry['path']} ({key})")


def expected_paths(subject: str = SUBJECT) -> set[str]:
    """Closed acquisition scope: one case, without fMRI or postoperative data."""
    paths = {"README", "CHANGES", "dataset_description.json", "participants.tsv"}
    if subject not in REVIEWED_MODES:
        raise AcquisitionError("BTC subject is outside the reviewed source scope")
    prefix = f"{subject}/ses-preop"
    paths |= {f"{prefix}/anat/{subject}_ses-preop_T1w.{ext}" for ext in ("json", "nii.gz")}
    if REVIEWED_MODES[subject] == "diffusion":
        paths |= {
            f"{prefix}/dwi/{subject}_ses-preop_acq-{phase}_dwi.{ext}"
            for phase in ("AP", "PA") for ext in ("bval", "bvec", "json", "nii.gz")
        }
    paths.add(f"derivatives/tumor_masks/{subject}/anat/{subject}_space_T1_label-tumor.nii")
    return paths


def transfer_entry(entry: dict) -> dict:
    return {**entry, "size_bytes": entry["bytes"]}


def validate_manifest(manifest: dict, output_root: Path, *, allow_pending_annex: bool = False) -> list[dict]:
    required = {"dataset": "BTC_preop", "accession": "ds001226", "release": "5.0.1",
                "git_commit": COMMIT, "license": "CC0"}
    subject = manifest.get("subject")
    if any(manifest.get(key) != value for key, value in required.items()) or subject not in REVIEWED_MODES:
        raise AcquisitionError("BTC manifest does not identify the reviewed source snapshot")
    files = manifest.get("files", [])
    paths = expected_paths(subject)
    if len(files) != len(paths) or {entry["path"] for entry in files} != paths:
        raise AcquisitionError(f"BTC manifest must contain exactly the {len(paths)} reviewed case files")
    if subject in QUEUED_SUBJECTS:
        _validate_queue_binding(manifest)
    if subject == "sub-PAT05":
        selection = json.loads(STRUCTURAL_SELECTION.read_text())
        if (manifest.get("selection_manifest_sha256") != sha256_file(STRUCTURAL_SELECTION)
                or selection.get("selected_subject") != subject
                or selection.get("role") != "development"
                or selection.get("selection_used_imaging_or_planning_results") is not False):
            raise AcquisitionError("Structural acquisition requires the unchanged predeclared selection")
        participants = next(entry for entry in files if entry["path"] == "participants.tsv")
        if participants["sha256"] != selection["source_metadata_sha256"]:
            raise AcquisitionError("Structural selection metadata differs from its predeclared source")
    for entry in files:
        if type(entry["bytes"]) is not int:
            raise AcquisitionError("Every file requires an integer pinned size")
        if entry["git_url"] != RAW_BASE + quote(entry["path"]):
            raise AcquisitionError("Source Git URL does not match the pinned snapshot")
        source = urlparse(entry["source_url"])
        if entry["path"].endswith((".nii", ".nii.gz")):
            base = source._replace(query="", fragment="").geturl()
            version = parse_qs(source.query, strict_parsing=True)
            if base != S3_BASE + quote(entry["path"]) or set(version) != {"versionId"}:
                raise AcquisitionError("Imaging requires its exact official versioned S3 object")
            if len(version["versionId"]) != 1 or not version["versionId"][0] or source.fragment:
                raise AcquisitionError("Imaging requires one pinned S3 version")
            if not re.fullmatch(r"[0-9a-f]{32}", entry.get("expected_md5", "")):
                raise AcquisitionError("Imaging requires its pinned annex MD5")
            if entry.get("expected_bytes") != entry["bytes"]:
                raise AcquisitionError("Imaging length differs from its annex pointer")
        elif entry["source_url"] != entry["git_url"]:
            raise AcquisitionError("Metadata must come from the pinned official Git snapshot")
    normalized = {"files": [transfer_entry(entry) for entry in files],
                  "storage_forecast": {"max_download_bytes": DOWNLOAD_LIMITS[subject]}}
    if allow_pending_annex and subject in QUEUED_SUBJECTS:
        # Exact queue binding above replaces the unavailable SHA256 for these
        # two images only. No unbound source or optional diffusion is accepted.
        if sum(entry["bytes"] for entry in files) > DOWNLOAD_LIMITS[subject]:
            raise AcquisitionError("Queued manifest exceeds its pinned download limit")
        for entry in files:
            checked_path(output_root, entry["path"])
            if entry.get("sha256") is None and "expected_md5" not in entry:
                raise AcquisitionError("Pending source must have a published annex checksum")
            if entry.get("sha256") is not None and not re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]):
                raise AcquisitionError("Invalid SHA256 for queued source")
    else:
        if any(entry.get("sha256") is None for entry in files):
            raise AcquisitionError("Source image SHA256 is pending annex-verified acquisition")
        _validate_transfer_manifest(normalized, output_root)
    return files


def verify_file(path: Path, entry: dict) -> None:
    """Check local SHA-256 and the separately recorded source annex checksum."""
    if entry.get("sha256") is None:
        if path.stat().st_size != entry["bytes"] or "expected_md5" not in entry:
            raise AcquisitionError(f"Pending source size/checksum contract mismatch: {path}")
    else:
        _verify_transfer_file(path, transfer_entry(entry))
    if "expected_md5" in entry:
        with path.open("rb") as handle:
            actual = hashlib.file_digest(handle, "md5").hexdigest()
        if actual != entry["expected_md5"]:
            raise AcquisitionError(f"Source annex MD5 mismatch: {path}")


def acquire_file(entry: dict, output_root: Path, *, opener: Callable = urlopen) -> str:
    if entry.get("sha256") is None:
        return _acquire_annex_file(entry, output_root, opener=opener)
    state = _acquire_file(transfer_entry(entry), output_root, opener=opener)
    verify_file(checked_path(output_root, entry["path"]), entry)
    return state


def _acquire_annex_file(entry: dict, output_root: Path, *, opener: Callable) -> str:
    """Stage one bounded image; verify published MD5 before atomic publication."""
    path = checked_path(output_root, entry["path"])
    if path.exists():
        verify_file(path, entry)
        return "already_verified"
    partial = checked_path(output_root, entry["path"] + ".partial")
    partial.parent.mkdir(parents=True, exist_ok=True)
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > entry["bytes"]:
        raise AcquisitionError("Pending annex partial exceeds pinned size")
    if offset == entry["bytes"]:
        verify_file(partial, entry)
        partial.replace(path)
        return "completed_from_partial"
    headers = {"Accept-Encoding": "identity", "User-Agent": "RessectionLab-BTC-queue/1"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    version = parse_qs(urlparse(entry["source_url"]).query)["versionId"][0]
    with opener(Request(entry["source_url"], headers=headers), timeout=45) as response:
        if response.geturl() != entry["source_url"] or response.headers.get("x-amz-version-id") != version:
            raise AcquisitionError("Annex response differs from the pinned official object version")
        if response.headers.get("Content-Encoding", "identity") != "identity":
            raise AcquisitionError("Annex response unexpectedly applied transport compression")
        if offset and response.status == 206:
            expected_range = f"bytes {offset}-{entry['bytes'] - 1}/{entry['bytes']}"
            if response.headers.get("Content-Range") != expected_range:
                raise AcquisitionError("Annex resume response has an unexpected Content-Range")
            mode = "ab"
        elif response.status == 200:
            offset, mode = 0, "wb"
        else:
            raise AcquisitionError(f"Unexpected annex download HTTP status: {response.status}")
        if int(response.headers.get("Content-Length", -1)) != entry["bytes"] - offset:
            raise AcquisitionError("Annex response length differs from the pinned source size")
        with partial.open(mode) as handle:
            size = offset
            while chunk := response.read(CHUNK_BYTES):
                size += len(chunk)
                if size > entry["bytes"]:
                    raise AcquisitionError("Annex download exceeds its pinned file size")
                handle.write(chunk)
            handle.flush()
            os.fsync(handle.fileno())
    verify_file(partial, entry)
    partial.replace(path)
    return "downloaded_and_verified"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--manifest", type=Path, help="Previously verified acquisition manifest; defaults to PAT28")
    source.add_argument("--queued-subject", choices=QUEUED_SUBJECTS,
                        help="Acquire only the structural files of this frozen development candidate")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--dry-run", action="store_true", help="Show the fixed scope without fetching files")
    parser.add_argument("--verify-only", action="store_true", help="Verify existing files without network access")
    args = parser.parse_args(argv)
    try:
        input_manifest = args.manifest or DEFAULT_MANIFEST
        manifest = (queued_manifest(args.queued_subject) if args.queued_subject
                    else json.loads(input_manifest.read_text()))
        entries = validate_manifest(manifest, args.output_root, allow_pending_annex=bool(args.queued_subject))
        total = sum(entry["bytes"] for entry in entries)
        if args.dry_run:
            print(json.dumps({"accession": manifest["accession"], "release": manifest["release"],
                              "subject": manifest["subject"], "files": len(entries), "bytes": total,
                              "license": manifest["license"], "output_root": str(args.output_root),
                              "role": manifest["role"]}, indent=2))
            return 0
        args.output_root.mkdir(parents=True, exist_ok=True)
        remaining = sum(entry["bytes"] for entry in entries
                        if not checked_path(args.output_root, entry["path"]).exists())
        if not args.verify_only and shutil.disk_usage(args.output_root).free < remaining:
            raise AcquisitionError("Insufficient local space for this case")
        results = []
        for entry in entries:
            if args.verify_only:
                verify_file(checked_path(args.output_root, entry["path"]), entry)
                state = "verified"
            else:
                state = acquire_file(entry, args.output_root)
            if entry["path"] == "participants.tsv" and manifest["subject"] == "sub-PAT05":
                selected = select_additional_structural_subject(checked_path(args.output_root, entry["path"]).read_text())
                if selected != manifest["subject"]:
                    raise AcquisitionError("Downloaded metadata does not reproduce the predeclared selection")
            if entry["path"] == "participants.tsv" and manifest["subject"] in QUEUED_SUBJECTS:
                selected = selected_queued_subjects(checked_path(args.output_root, entry["path"]).read_text())
                if selected != list(QUEUED_SUBJECTS):
                    raise AcquisitionError("Downloaded metadata does not reproduce the frozen queue selection")
            if entry.get("sha256") is None:
                entry["sha256"] = sha256_file(checked_path(args.output_root, entry["path"]))
            results.append({"path": entry["path"], "status": state, "sha256": entry["sha256"],
                            "source_annex_md5": entry.get("expected_md5")})
            print(f"{state}: {Path(entry['path']).name}", flush=True)
        if args.queued_subject:
            # Keep the pre-image queue immutable. Publish a separate verified
            # acquisition manifest for preparation only after every file passes.
            validate_manifest(manifest, args.output_root)
            manifest["verified_at"] = datetime.now(timezone.utc).isoformat()
            manifest["source_content_status"] = "matches_pinned_official_release_checksums"
            manifest["anatomical_qc_status"] = "not_assessed_by_download_verification"
            verified_name = f"btc_{manifest['subject']}_verified_manifest.json"
            verified_path = checked_path(args.output_root, verified_name)
            staged_path = checked_path(args.output_root, verified_name + ".partial")
            staged_path.write_text(json.dumps(manifest, indent=2) + "\n")
            staged_path.replace(verified_path)
        report = {"accession": manifest["accession"], "release": manifest["release"],
                  "subject": manifest["subject"],
                  "source_manifest_sha256": sha256_file(verified_path if args.queued_subject else input_manifest),
                  "checked_at": datetime.now(timezone.utc).isoformat(), "files": results,
                  "source_content_status": "matches_pinned_official_release_checksums",
                  "anatomical_qc_status": "not_assessed_by_download_verification"}
        if args.queued_subject:
            report["selection_queue_sha256"] = QUEUE_SHA256
            report["verified_manifest"] = str(verified_path)
        report_name = "btc_acquisition_report.json" if manifest["subject"] == SUBJECT else f"btc_{manifest['subject']}_acquisition_report.json"
        report_path = checked_path(args.output_root, report_name)
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        print(f"Report: {report_path}")
        return 0
    except (AcquisitionError, OSError, ValueError, KeyError, TypeError) as error:
        print(f"BTC acquisition failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
