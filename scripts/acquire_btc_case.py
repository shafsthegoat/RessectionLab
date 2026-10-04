#!/usr/bin/env python3
"""Acquire one of the pinned, source-verified BTC_preop development cases.

PAT28 defaults to 15 files with diffusion; the separately predeclared PAT05
manifest contains 7 structural files. Original images, orientations and source
annotations remain unchanged. Neither case is held-out final evaluation.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
from typing import Callable
from urllib.parse import parse_qs, quote, urlparse
from urllib.request import urlopen

# Keep transport, resumable partials and corruption handling identical to the
# other public-data fetcher; BTC adds its own source and annex provenance gates.
from acquire_public_case import (
    AcquisitionError,
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
REVIEWED_MODES = {"sub-PAT28": "diffusion", "sub-PAT05": "structural"}
DOWNLOAD_LIMITS = {"sub-PAT28": MAX_DOWNLOAD_BYTES, "sub-PAT05": 100_000_000}
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


def validate_manifest(manifest: dict, output_root: Path) -> list[dict]:
    required = {"dataset": "BTC_preop", "accession": "ds001226", "release": "5.0.1",
                "git_commit": COMMIT, "license": "CC0"}
    subject = manifest.get("subject")
    if any(manifest.get(key) != value for key, value in required.items()) or subject not in REVIEWED_MODES:
        raise AcquisitionError("BTC manifest does not identify the reviewed source snapshot")
    files = manifest.get("files", [])
    paths = expected_paths(subject)
    if len(files) != len(paths) or {entry["path"] for entry in files} != paths:
        raise AcquisitionError(f"BTC manifest must contain exactly the {len(paths)} reviewed case files")
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
    _validate_transfer_manifest(normalized, output_root)
    return files


def verify_file(path: Path, entry: dict) -> None:
    """Check local SHA-256 and the separately recorded source annex checksum."""
    _verify_transfer_file(path, transfer_entry(entry))
    if "expected_md5" in entry:
        with path.open("rb") as handle:
            actual = hashlib.file_digest(handle, "md5").hexdigest()
        if actual != entry["expected_md5"]:
            raise AcquisitionError(f"Source annex MD5 mismatch: {path}")


def acquire_file(entry: dict, output_root: Path, *, opener: Callable = urlopen) -> str:
    state = _acquire_file(transfer_entry(entry), output_root, opener=opener)
    verify_file(checked_path(output_root, entry["path"]), entry)
    return state


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--dry-run", action="store_true", help="Show the fixed scope without fetching files")
    parser.add_argument("--verify-only", action="store_true", help="Verify existing files without network access")
    args = parser.parse_args(argv)
    try:
        manifest = json.loads(args.manifest.read_text())
        entries = validate_manifest(manifest, args.output_root)
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
            results.append({"path": entry["path"], "status": state, "sha256": entry["sha256"],
                            "source_annex_md5": entry.get("expected_md5")})
            print(f"{state}: {Path(entry['path']).name}", flush=True)
        report = {"accession": manifest["accession"], "release": manifest["release"],
                  "subject": manifest["subject"], "source_manifest_sha256": sha256_file(args.manifest),
                  "checked_at": datetime.now(timezone.utc).isoformat(), "files": results,
                  "source_content_status": "matches_pinned_official_release_checksums",
                  "anatomical_qc_status": "not_assessed_by_download_verification"}
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
