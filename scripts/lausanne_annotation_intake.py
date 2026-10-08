#!/usr/bin/env python3
"""Bounded, original-grid QC of the frozen TRAIN manual annotation inventory.

Default/preflight reads metadata only. `batch --execute` is an explicit payload
operation. Each invocation makes at most one transfer attempt per chosen mask;
subsequent invocations can resume retained partials. All attempts are immutable.
No labels are repaired, resampled, admitted for training, or passed to planning.
"""
from __future__ import annotations

import argparse
import base64
from collections import Counter
from datetime import datetime, timezone
import fcntl
import gzip
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import re
import shutil
import signal
import sys
import time
from urllib.parse import parse_qs, urlsplit
import xml.etree.ElementTree as ET

import lausanne_train_intake as originals
from acquire_btc_case import acquire_file, open_without_redirect
from acquire_public_case import AcquisitionError
from real_intake_io import (atomic_preserve, check_deadline, supervise,
                            termination_cleanup, verify_source_file)

ROOT = originals.ROOT
DATA = originals.DATA
CACHE = DATA / "train-annotation-intake-v1"
MANIFEST = ROOT / "manifests/lausanne-train-annotations-v1.json"
MANIFEST_SHA = "f779066f5cb784f623446f12565eb1405994ac9e3b41c1fef8210546598de624"
SOURCE_NAMES = (*originals.SOURCE_NAMES, "scripts/lausanne_annotation_intake.py",
                "src/resectionlab/critical_evidence.py", str(MANIFEST.relative_to(ROOT)),
                "manifests/lausanne-component-cohort-v1.json", "manifests/lausanne-train-originals-v1.json")
HEX = re.compile(r"[0-9a-f]{64}")
MASK = re.compile(r"derivatives/manual_masks/(sub-[0-9]{3})/(ses-[0-9]{8})/anat/"
                  r"\1_\2_desc-Lesion_[1-9][0-9]*_mask\.nii\.gz")


def encoded(value):
    return (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def safe_path(path: Path) -> Path:
    """Reject symlinks in every workspace component, including partial files."""
    path = Path(path)
    try:
        parts = path.relative_to(ROOT).parts
    except ValueError as error:
        raise AcquisitionError("Path outside this checkout") from error
    if not parts or ".." in parts or ROOT.is_symlink():
        raise AcquisitionError("Unsafe workspace path")
    current = ROOT
    for part in parts:
        current = current / part
        if current.is_symlink():
            raise AcquisitionError("Symlink in intake path")
    return path


def read_metadata(path: Path, *, maximum=2 * 1024**2) -> bytes:
    path = safe_path(path)
    if not path.is_file() or path.stat().st_size > maximum:
        raise AcquisitionError("Missing or oversized metadata: " + str(path))
    return path.read_bytes()


def source_metadata(entry: dict, *, current_tracked=False) -> bytes:
    cached = safe_path(CACHE / "metadata" / (entry["sha256"] + ".bin"))
    path = cached if cached.exists() and not current_tracked else safe_path(ROOT / entry["path"])
    payload = read_metadata(path)
    if len(payload) != entry["bytes"] or digest(payload) != entry["sha256"]:
        raise AcquisitionError("Frozen source metadata differs: " + entry["path"])
    return payload


def validate_record(row: dict, source: dict, sessions: dict, people: dict,
                    versions: dict) -> None:
    match = MASK.fullmatch(row["path"])
    identity = row["subject"], row["session"]
    if (not match or match.groups() != identity or row["role"] != "TRAIN"
            or identity not in sessions or people[row["subject"]]["group"] != "patient"):
        raise AcquisitionError("Annotation outside the frozen TRAIN patient scope")
    for key in ("path", "subject", "session", "role", "status",
                "annotation_subtype_status", "source_crosswalk_entry",
                "annotation_available_at", "review_available_at"):
        if row[key] != source[key]:
            raise AcquisitionError("Annotation differs from qualified source record")
    if row["status"] == "metadata_failed":
        if (row["metadata_failure"] != source["error"] or "file" in row
                or row["metadata_request_numbers"] != [r["number"] for r in source["requests"]]):
            raise AcquisitionError("Failed source record was promoted or changed")
        if len(row["failed_metadata_responses"]) != len(source["requests"]):
            raise AcquisitionError("Failed metadata response proof missing")
        for saved, request in zip(row["failed_metadata_responses"], source["requests"], strict=True):
            body = base64.b64decode(saved["body_base64"], validate=True)
            if (len(body) > 2048 or len(body) != saved["bytes"] or digest(body) != saved["sha256"]
                    or saved["sha256"] != request["body"]["sha256"]
                    or saved["request_number"] != request["number"]):
                raise AcquisitionError("Failed source response bytes changed")
        return
    if row["status"] != "metadata_qualified":
        raise AcquisitionError("Unknown metadata outcome")
    for key in ("file", "original_reference", "pointer_git_blob_sha1", "pointer_sha256",
                "sidecar_sha256", "sidecar", "version_proof"):
        if row[key] != source[key]:
            raise AcquisitionError("Qualified pointer/sidecar/version source changed")
    file = row["file"]
    pointer = base64.b64decode(row["pointer_base64"], validate=True)
    sidecar = base64.b64decode(row["sidecar_base64"], validate=True)
    annex = re.search(rb"MD5E-s([0-9]+)--([0-9a-f]{32})\.nii\.gz\s*$", pointer)
    git_blob = hashlib.sha1(b"blob " + str(len(pointer)).encode() + b"\0" + pointer).hexdigest()
    if (len(pointer) > 2048 or not annex or int(annex[1]) != file["bytes"]
            or annex[2].decode() != file["expected_md5"] or file["sha256"] is not None
            or file["path"] != row["path"] or digest(pointer) != row["pointer_sha256"]
            or git_blob != row["pointer_git_blob_sha1"]
            or len(sidecar) > 2048 or digest(sidecar) != row["sidecar_sha256"]
            or json.loads(sidecar) != row["sidecar"]):
        raise AcquisitionError("Source pointer or sidecar bytes differ")
    for body, request in zip((pointer, sidecar), source["requests"], strict=True):
        if not request["ok"] or digest(body) != request["body"]["sha256"]:
            raise AcquisitionError("Metadata response proof differs")
    expected = next(f for f in sessions[identity]["files"] if f["path"].endswith("_angio.nii.gz"))
    if (row["original_reference"] != expected or row["sidecar"] != {
            "Type": "Lesion", "RawSources": Path(expected["path"]).name, "Space": "orig"}):
        raise AcquisitionError("Mask must reference the exact same-session original TOF")
    parsed = urlsplit(file["source_url"])
    query = parse_qs(parsed.query, strict_parsing=True)
    if (parsed._replace(query="").geturl() != originals.S3 + row["path"]
            or set(query) != {"versionId"} or len(query["versionId"]) != 1
            or not query["versionId"][0] or query["versionId"][0] == "null"):
        raise AcquisitionError("Unpinned annotation object URL")
    version = versions.get(("ds003949/" + row["path"], query["versionId"][0]))
    selected = row["version_proof"]["selected"]
    if (not version or int(version["Size"]) != file["bytes"]
            or version["ETag"].strip('"') != file["expected_md5"]
            or any(selected[k] != version[k] for k in ("Key", "VersionId", "Size", "ETag", "LastModified"))):
        raise AcquisitionError("Official version proof differs from annex-published fixity")


def preflight() -> tuple[dict, dict]:
    """No network, source images, mask contents, or original receipt access."""
    payload = read_metadata(MANIFEST)
    if digest(payload) != MANIFEST_SHA:
        raise AcquisitionError("Frozen annotation manifest changed")
    manifest = json.loads(payload)
    cohort = json.loads(source_metadata(manifest["cohort"], current_tracked=True))
    index = json.loads(source_metadata(manifest["original_index"], current_tracked=True))
    metadata = {k: source_metadata(v) for k, v in manifest["metadata_sources"].items()}
    if (cohort["members"] != originals.members_from_source(metadata["participants"])
            or index["cohort_sha256"] != manifest["cohort"]["sha256"]
            or json.loads(metadata["rights"])["License"] != "CC0"):
        raise AcquisitionError("Cohort, original index or source rights differ")
    people = {m["subject"]: m for m in cohort["members"] if m["role"] == "TRAIN"}
    sessions = {(r["subject"], r["session"]): r for r in index["sessions"]}
    if set(sessions) != {(m["subject"], "ses-" + s) for m in people.values() for s in m["sessions"]}:
        raise AcquisitionError("Original index does not cover exact frozen TRAIN sessions")
    qualified = json.loads(metadata["qualification"])
    if (qualified["cohort_sha256"] != manifest["cohort"]["sha256"]
            or qualified["original_index_sha256"] != manifest["original_index"]["sha256"]
            or qualified["git_commit"] != manifest["git_commit"]):
        raise AcquisitionError("Annotation qualification uses different source pins")
    xml = ET.fromstring(metadata["versions"])
    ns = {"s": "http://s3.amazonaws.com/doc/2006-03-01/"}
    if xml.findtext("s:IsTruncated", namespaces=ns) != "false":
        raise AcquisitionError("Version inventory was truncated")
    versions = {}
    for item in xml.findall("s:Version", ns):
        v = {k: item.findtext("s:" + k, namespaces=ns) for k in
             ("Key", "VersionId", "Size", "ETag", "LastModified")}
        versions[(v["Key"], v["VersionId"])] = v
    records = manifest["records"]
    if len(records) != 148 or [r["path"] for r in records] != [r["path"] for r in qualified["records"]]:
        raise AcquisitionError("Candidate inventory differs or omits failed records")
    for row, source in zip(records, qualified["records"], strict=True):
        validate_record(row, source, sessions, people, versions)
    passed = [r for r in records if r["status"] == "metadata_qualified"]
    counts = manifest["counts"]
    if (len(passed) != 144 or len({r["subject"] for r in passed}) != 106
            or len({(r["subject"], r["session"]) for r in passed}) != 117
            or sum(r["file"]["bytes"] for r in passed) != 13977015
            or Counter(r["annotation_subtype_status"] for r in passed) != counts["annotation_subtypes"]):
        raise AcquisitionError("Frozen annotation counts changed")
    return manifest, sessions


def execution_source() -> dict:
    return {"files": {name: digest(read_metadata(ROOT / name)) for name in SOURCE_NAMES},
            "python": sys.version, "numpy": importlib.metadata.version("numpy"),
            "nibabel": importlib.metadata.version("nibabel")}


def retain_source(run: Path, source: dict) -> None:
    for name, sha in source["files"].items():
        payload = read_metadata(ROOT / name)
        if digest(payload) != sha:
            raise AcquisitionError("Executing source changed during snapshot")
        atomic_preserve(safe_path(run / "source-snapshot" / name), payload)
    atomic_preserve(safe_path(run / "source.json"), encoded(source))


def validate_execution(source: dict, run: Path) -> None:
    if source != execution_source() or read_metadata(run / "source.json") != encoded(source):
        raise AcquisitionError("Executing source/runtime changed")
    for name, sha in source["files"].items():
        if digest(read_metadata(run / "source-snapshot" / name)) != sha:
            raise AcquisitionError("Execution snapshot changed")


def original_receipt_contract(row: dict, saved: dict, index_sha: str, reference: dict) -> dict:
    """Inspect receipt metadata only; distinguish TOF from the other modality."""
    if (saved.get("subject") != row["subject"] or saved.get("session") != row["session"]
            or saved.get("role") != "TRAIN" or saved.get("status") != "acquired"
            or saved.get("index_sha256") != index_sha
            or any(saved.get(k) is not False for k in ("scanner_frame_admitted", "spatial_planning_admitted"))
            or saved.get("registration") != "unverified" or saved.get("anatomical_coverage") != "unreviewed"
            or saved.get("component_optimizer_updates") != 0 or saved.get("recorded_rl_transitions") != 0
            or len(saved.get("files", [])) != len(row["files"])):
        raise AcquisitionError("Original receipt identity/role/claims differ")
    for entry, receipt in zip(row["files"], saved["files"], strict=True):
        if (receipt["path"] != entry["path"] or receipt["bytes"] != entry["bytes"]
                or not HEX.fullmatch(receipt.get("sha256", ""))):
            raise AcquisitionError("Original receipt file binding differs")
    expected = [e["path"] for e in row["files"] if e["path"].endswith(".nii.gz")]
    qc = saved.get("integrity_qc", [])
    if [q.get("path") for q in qc] != expected or reference not in row["files"]:
        raise AcquisitionError("Original receipt lacks exact two-image QC inventory")
    for item in qc:
        if item["status"] == "qc_failed":
            if not isinstance(item.get("reason"), str) or not item["reason"]:
                raise AcquisitionError("Original QC failure lacks reason")
        elif item["status"] == "header_and_scalar_checks_passed":
            h = item["header"]
            if (h["file"] != Path(item["path"]).name or h["status"] != "passed_header_checks"
                    or h["physical_units"] != "mm" or len(h["shape"]) != 3
                    or any(type(n) is not int or n <= 0 for n in h["shape"])
                    or math.prod(h["shape"]) != item["voxels"]
                    or not all(math.isfinite(item[k]) for k in ("minimum", "maximum"))
                    or item["minimum"] > item["maximum"]):
                raise AcquisitionError("Inconsistent original scalar/header QC metadata")
        else:
            raise AcquisitionError("Unknown original QC status")
    tof = next(q for q in qc if q["path"] == reference["path"])
    other = [q for q in qc if q["path"] != reference["path"]]
    return {"referenced_tof_qc": tof, "other_modality_qc": other,
            "referenced_tof_passed": tof["status"] == "header_and_scalar_checks_passed",
            "full_pair_current_fixity_checked": False,
            "source_receipt_pair_qc": "failed" if any(q["status"] == "qc_failed" for q in qc) else "passed"}


def decoding_budget(shape, itemsize: int, bounds: dict) -> dict:
    if (len(shape) != 3 or any(type(n) is not int or n <= 0 for n in shape)
            or itemsize not in (1, 2, 4, 8)):
        raise AcquisitionError("Unsupported annotation dimensions/dtype")
    voxels = math.prod(shape)
    if (voxels > bounds["max_mask_voxels"] or voxels * itemsize > bounds["max_uncompressed_mask_bytes"]
            or voxels * 8 > bounds["max_logical_float64_bytes"]):
        raise AcquisitionError("Annotation exceeds declared decoded size bounds")
    # Native byte buffer + float64 scaling + at most three boolean temporaries.
    chunk = min(bounds["max_chunk_voxels"], bounds["max_decoded_chunk_working_bytes"] // (itemsize + 11))
    if chunk < 1:
        raise AcquisitionError("Invalid decoding chunk budget")
    return {"voxels": voxels, "native_payload_bytes": voxels * itemsize,
            "logical_float64_bytes": voxels * 8, "chunk_voxels": chunk,
            "chunk_working_bytes_bound": chunk * (itemsize + 11)}


def inspect_mask(path: Path, sha: str, bounds: dict, deadline: float) -> dict:
    """Stream exact source values in bounded chunks; no image-sized allocation."""
    import nibabel as nib
    import numpy as np
    from resectionlab.critical_evidence import nifti1_header_record

    with gzip.open(safe_path(path), "rb") as stream:
        raw = stream.read(348)
        header = nib.Nifti1Header(binaryblock=raw, check=False)
        if int(header["sizeof_hdr"]) != 348 or bytes(header["magic"]) != b"n+1\0":
            raise AcquisitionError("Only original single-file NIfTI-1 masks are supported")
        dtype = header.get_data_dtype()
        if dtype.kind not in "iuf":
            raise AcquisitionError("Mask datatype must be a real scalar")
        budget = decoding_budget([int(n) for n in header.get_data_shape()], dtype.itemsize, bounds)
        offset = float(header["vox_offset"])
        if not math.isfinite(offset) or not offset.is_integer() or not 352 <= offset <= bounds["max_nifti_data_offset"]:
            raise AcquisitionError("Mask data offset exceeds bounded NIfTI contract")
        if int(offset) + budget["native_payload_bytes"] > bounds["max_uncompressed_mask_bytes"]:
            raise AcquisitionError("Full uncompressed mask exceeds declared byte allowance")
        padding = stream.read(int(offset) - 348)
        if len(padding) != int(offset) - 348 or any(padding):
            raise AcquisitionError("Mask extensions/nonzero padding need separate qualification")
        slope, intercept = header.get_slope_inter()
        slope, intercept = (1., 0.) if slope is None else (float(slope), float(intercept))
        if not math.isfinite(slope) or not math.isfinite(intercept):
            raise AcquisitionError("Nonfinite mask scaling")
        remaining = budget["voxels"]
        positives = 0
        while remaining:
            check_deadline(deadline)
            n = min(remaining, budget["chunk_voxels"])
            payload = stream.read(n * dtype.itemsize)
            if len(payload) != n * dtype.itemsize:
                raise AcquisitionError("Truncated mask scalar payload")
            values = np.frombuffer(payload, dtype=dtype).astype(np.float64)
            values *= slope
            values += intercept
            if not np.isfinite(values).all() or not ((values == 0) | (values == 1)).all():
                raise AcquisitionError("Mask source values are not finite exact binary labels")
            positives += int(np.count_nonzero(values))
            remaining -= n
            del values, payload
        if stream.read(1):
            raise AcquisitionError("Unexpected scalar payload after declared mask array")
    if not 0 < positives < budget["voxels"]:
        raise AcquisitionError("Mask must contain foreground and background")
    return {"status": "passed", "raw_grid": nifti1_header_record(raw, sha),
            "dtype": dtype.str, "source_scaling": [slope, intercept], "decoding_budget": budget,
            "positive_voxels": positives, "background_voxels": budget["voxels"] - positives,
            "background_semantics": "unknown", "array_operation": "none; source values counted only"}


def grid_proof(mask: dict, reference: dict) -> dict:
    """Reuse the fixed precision envelope; also accept truly equal coded grids."""
    import nibabel as nib
    import numpy as np
    from resectionlab.critical_evidence import nifti1_header_record, source_reference_grid_metrics

    try:
        metrics = source_reference_grid_metrics(mask, reference)
        return {"rule": "source_reference_serialization_equivalence", "metrics": metrics,
                "array_operation": "unchanged source voxel indices"}
    except ValueError as precision_error:
        # The existing envelope requires coded sforms. A qform-only source can
        # instead pass exact equality; no additional numerical tolerance exists.
        shapes, affines, units = [], [], []
        for record in (mask, reference):
            raw = base64.b64decode(record["header_base64"], validate=True)
            if nifti1_header_record(raw, record["source_file_sha256"]) != record:
                raise AcquisitionError("Raw grid proof was modified")
            h = nib.Nifti1Header(binaryblock=raw, check=False)
            if int(h["sizeof_hdr"]) != 348 or bytes(h["magic"]) != b"n+1\0":
                raise AcquisitionError("Exact grid requires original NIfTI-1 headers")
            q, qc = h.get_qform(coded=True)
            s, sc = h.get_sform(coded=True)
            if ((not qc and not sc) or (qc and qc not in (1, 2)) or (sc and sc not in (1, 2))
                    or (qc and sc and not np.array_equal(q, s))):
                raise AcquisitionError("Unresolved coded grid: " + str(precision_error))
            a = s if sc else q
            if (not np.isfinite(a).all() or not np.array_equal(a[3], [0, 0, 0, 1])
                    or abs(np.linalg.det(a[:3, :3])) < 1e-12):
                raise AcquisitionError("Invalid exact coded transform")
            shapes.append(h.get_data_shape()); affines.append(a); units.append(h.get_xyzt_units()[0])
        if (len(shapes[0]) != 3 or shapes[0] != shapes[1] or any(n <= 0 for n in shapes[0])
                or units[1] != "mm" or units[0] not in ("mm", "unknown")
                or not np.array_equal(affines[0], affines[1])):
            raise AcquisitionError("Original grid unresolved: " + str(precision_error))
        return {"rule": "exact_coded_native_voxel_grid_v1", "all_voxel_centres_keep_reference_index": True,
                "reference_mm_inherited": units[0] == "unknown", "array_operation": "unchanged source voxel indices"}


def verify_reference(row: dict, original: dict, manifest: dict, trial: Path, deadline: float) -> dict:
    from resectionlab.critical_evidence import nifti1_header_record

    receipt_path = safe_path(originals.CACHE / "acquired" / (row["subject"] + "_" + row["session"] + ".json"))
    if not receipt_path.exists():
        return {"status": "deferred_missing_original_receipt", "receipt_path": str(receipt_path.relative_to(ROOT))}
    payload = read_metadata(receipt_path, maximum=128 * 1024)
    saved = json.loads(payload)
    result = original_receipt_contract(original, saved, manifest["original_index"]["sha256"], row["original_reference"])
    atomic_preserve(safe_path(trial / "original-receipt.json"), payload)
    result.update(original_receipt_sha256=digest(payload),
                  original_receipt_snapshot=str((trial / "original-receipt.json").relative_to(ROOT)))
    try:
        originals.validate_retained_source(saved, deadline=deadline)
        if not result["referenced_tof_passed"]:
            return {**result, "status": "failed_referenced_tof_qc"}
        file = row["original_reference"]
        measured = next(f for f in saved["files"] if f["path"] == file["path"])
        path = safe_path(DATA / file["path"])
        sha = verify_source_file(path, file, receipt_sha=measured["sha256"], deadline=deadline)
        with gzip.open(path, "rb") as stream:
            raw = stream.read(348)
        return {**result, "status": "passed", "original_tof_sha256": sha,
                "raw_grid": nifti1_header_record(raw, sha), "current_tof_fixity_checked": True}
    except BaseException as error:
        error.annotation_reference_qc = {**result, "status": "failed_or_incomplete"}
        raise


def transfer(row: dict, deadline: float) -> tuple[str, list[dict]]:
    """Whitelist one exact immutable mask object and retain its response header."""
    entry = row["file"]
    safe_path(DATA / entry["path"])
    safe_path(DATA / (entry["path"] + ".partial"))
    requests = []

    def opener(request, *, timeout):
        check_deadline(deadline)
        if requests or request.full_url != entry["source_url"] or request.get_method() != "GET":
            raise AcquisitionError("Transfer outside exact one-request mask whitelist")
        record = {"url": request.full_url, "method": "GET", "headers": dict(request.header_items()),
                  "started_at": datetime.now(timezone.utc).isoformat()}
        requests.append(record)
        response = open_without_redirect(request, timeout=min(timeout, max(.001, deadline-time.monotonic())))
        record.update(status=response.status, final_url=response.geturl(), response_headers=dict(response.headers))
        return response

    try:
        return acquire_file(entry, DATA, opener=opener), requests
    except BaseException as error:
        # Carry attempted request evidence through the worker's failure receipt.
        error.annotation_requests = requests
        raise


def worker(mask_path: str, run: Path, trial: Path, max_seconds: float, *, existing_only=False) -> dict:
    if not math.isfinite(max_seconds) or not 0 < max_seconds <= 120:
        raise AcquisitionError("Invalid worker time allowance")
    started = time.monotonic()
    deadline = started + max_seconds
    result = {"schema": "lausanne-annotation-attempt-v1", "manifest_sha256": MANIFEST_SHA,
              "path": mask_path, "status": "failed_or_incomplete", "acquisition": {"status": "not_attempted"},
              "content_qc": {"status": "not_assessed"}, "reference_qc": {"status": "not_assessed"},
              "grid_qc": {"status": "not_assessed"}, "requests": [], "training_admitted": False,
              "scanner_frame_admitted": False, "spatial_planning_admitted": False,
              "annotation_available_at": None, "review_available_at": None,
              "optimizer_updates": 0, "recorded_rl_transitions": 0}
    stage = "preflight"
    def expire(*_):
        raise TimeoutError("Declared annotation worker deadline")
    previous_alarm = signal.signal(signal.SIGALRM, expire)
    signal.setitimer(signal.ITIMER_REAL, max_seconds)
    try:
        manifest, sessions = preflight()
        row = next((r for r in manifest["records"] if r["path"] == mask_path), None)
        if not row or row["status"] != "metadata_qualified":
            raise AcquisitionError("Worker mask is not in qualified TRAIN whitelist")
        if not 0 < max_seconds <= manifest["bounds"]["max_worker_seconds"]:
            raise AcquisitionError("Invalid worker time allowance")
        source = json.loads(read_metadata(run / "source.json"))
        validate_execution(source, run)
        result.update(subject=row["subject"], session=row["session"], role="TRAIN",
                      annotation_subtype_status=row["annotation_subtype_status"], semantics=manifest["semantics"],
                      execution_source_record=str((run / "source.json").relative_to(ROOT)),
                      execution_source_sha256=digest(encoded(source)), existing_files_only=existing_only)
        path = safe_path(DATA / row["path"])
        check_deadline(deadline)
        stage = "acquisition"
        result[stage] = {"status": "failed_or_incomplete"}
        state, requests = ("existing_source_only", []) if existing_only else transfer(row, deadline)
        result["requests"] = requests
        sha = verify_source_file(path, row["file"], deadline=deadline)
        result["acquisition"] = {"status": "source_verified", "transfer_status": state,
                                 "bytes": row["file"]["bytes"], "sha256": sha}
        stage = "content_qc"
        result[stage] = {"status": "failed_or_incomplete"}
        result["content_qc"] = inspect_mask(path, sha, manifest["bounds"], deadline)
        stage = "reference_qc"
        result[stage] = {"status": "failed_or_incomplete"}
        result["reference_qc"] = verify_reference(row, sessions[(row["subject"], row["session"])], manifest, trial, deadline)
        if result["reference_qc"]["status"] == "passed":
            result["grid_qc"] = {"status": "unresolved"}
            try:
                proof = grid_proof(result["content_qc"]["raw_grid"], result["reference_qc"]["raw_grid"])
                result["grid_qc"] = {"status": "passed", "proof": proof}
            except (ValueError, AcquisitionError) as error:
                result["grid_qc"]["reason"] = str(error)
        else:
            result["grid_qc"] = {"status": "deferred_reference_not_passed"}
        stage = "final_source_verification"
        check_deadline(deadline)
        verify_source_file(path, row["file"], receipt_sha=sha, deadline=deadline)
        validate_execution(source, run)
        result["status"] = "qc_complete" if result["grid_qc"]["status"] == "passed" else "content_passed_reference_or_grid_pending"
    except BaseException as error:
        result.update(status="failed_or_incomplete", failed_stage=stage,
                      error={"type": type(error).__name__, "message": str(error)})
        result["requests"] = getattr(error, "annotation_requests", result["requests"])
        result["reference_qc"] = getattr(error, "annotation_reference_qc", result["reference_qc"])
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_alarm)
        result["elapsed_seconds"] = time.monotonic() - started
        if result["elapsed_seconds"] >= max_seconds:
            result["status"] = "failed_or_incomplete"
            result["deadline_exceeded"] = True
        atomic_preserve(safe_path(trial / "receipt.json"), encoded(result))
    return result


def initial_outcomes(manifest: dict, only_mask: str | None) -> list[dict]:
    if only_mask is not None and only_mask not in {r["path"] for r in manifest["records"] if r["status"] == "metadata_qualified"}:
        raise AcquisitionError("Requested mask is outside qualified TRAIN whitelist")
    return [{"path": r["path"], "subject": r["subject"], "session": r["session"], "role": "TRAIN",
             "annotation_subtype_status": r["annotation_subtype_status"],
             "status": "metadata_failed_excluded" if r["status"] == "metadata_failed" else
             ("outside_bounded_mask_scope" if only_mask and r["path"] != only_mask else "deferred_not_attempted"),
             **({"metadata_failure": r["metadata_failure"]} if r["status"] == "metadata_failed" else {})}
            for r in manifest["records"]]


def retain_worker_receipt(outcome: dict, trial: Path, source_sha: str) -> None:
    """Link a bounded receipt even when supervision was interrupted."""
    receipt_path = safe_path(trial / "receipt.json")
    if not receipt_path.exists():
        return
    raw = read_metadata(receipt_path)
    outcome["receipt_sha256"] = digest(raw)
    receipt = json.loads(raw)
    if (receipt.get("manifest_sha256") != MANIFEST_SHA or receipt.get("path") != outcome["path"]
            or receipt.get("execution_source_sha256") != source_sha
            or any(receipt.get(k) is not False for k in
                   ("training_admitted", "scanner_frame_admitted", "spatial_planning_admitted"))):
        raise AcquisitionError("Worker receipt source/claim binding differs")
    outcome.update(receipt_status=receipt["status"],
                   acquisition_status=receipt["acquisition"]["status"],
                   content_qc_status=receipt["content_qc"]["status"],
                   reference_qc_status=receipt["reference_qc"]["status"],
                   grid_qc_status=receipt["grid_qc"]["status"])


def batch(run_id: str, max_seconds: int, max_bytes: int, *, only_mask=None, existing_only=False) -> dict:
    started = time.monotonic()
    deadline = started + max_seconds
    manifest, _ = preflight()
    bounds = manifest["bounds"]
    if (not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,79}", run_id)
            or not 1 <= max_seconds <= bounds["max_batch_seconds"]
            or not 1 <= max_bytes <= bounds["max_batch_source_bytes"]):
        raise AcquisitionError("Batch identity/time/byte bounds invalid")
    outcomes = initial_outcomes(manifest, only_mask)
    safe_path(CACHE).mkdir(parents=True, exist_ok=True)
    with safe_path(CACHE / "intake.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run = safe_path(CACHE / "attempts" / run_id)
        run.mkdir(parents=True, exist_ok=False)
        attempted_bytes = 0
        report = {"schema": "lausanne-annotation-batch-v1", "manifest_sha256": MANIFEST_SHA,
                  "run_id": run_id, "max_seconds": max_seconds, "max_source_bytes": max_bytes,
                  "existing_files_only": existing_only, "automatic_retries": 0, "outcomes": outcomes,
                  "training_admitted": False, "scanner_frame_admitted": False, "spatial_planning_admitted": False,
                  "optimizer_updates": 0, "recorded_rl_transitions": 0}
        try:
            for entry in (manifest["cohort"], manifest["original_index"], *manifest["metadata_sources"].values()):
                check_deadline(deadline)
                atomic_preserve(safe_path(CACHE / "metadata" / (entry["sha256"] + ".bin")), source_metadata(entry))
            source = execution_source()
            retain_source(run, source)
            report["execution_source_sha256"] = digest(encoded(source))
            atomic_preserve(safe_path(run / "declaration.json"), encoded(report))
            if shutil.disk_usage(DATA).free < max_bytes + 128 * 1024**2:
                raise AcquisitionError("Insufficient declared transfer and metadata reserve")
            for row, outcome in zip(manifest["records"], outcomes, strict=True):
                if outcome["status"] != "deferred_not_attempted":
                    continue
                check_deadline(deadline)
                if attempted_bytes + row["file"]["bytes"] > max_bytes:
                    outcome["status"] = "deferred_byte_budget"
                    continue
                if deadline - time.monotonic() < 5:
                    raise TimeoutError("Insufficient time to start another mask worker")
                validate_execution(source, run)
                attempted_bytes += row["file"]["bytes"]
                trial = safe_path(run / Path(row["path"]).name)
                outcome.update(status="worker_launch_pending", attempt=str(trial.relative_to(ROOT)))
                try:
                    trial.mkdir()
                    seconds = min(bounds["max_worker_seconds"], deadline-time.monotonic()-1)
                    command = [sys.executable, str(Path(__file__).resolve()), "worker", "--execute",
                               "--manifest-sha", MANIFEST_SHA, "--mask", row["path"], "--run-id", run_id,
                               "--worker-seconds", str(seconds)]
                    if existing_only:
                        command.append("--existing-only")

                    def worker_started(pid):
                        outcome.update(status="worker_started", worker_pid=pid)
                        atomic_preserve(trial / "started.json", encoded({"worker_pid": pid, "max_seconds": seconds}))

                    status, code = supervise(command, trial / "worker.log",
                                             deadline=min(deadline, time.monotonic()+seconds+.5),
                                             on_start=worker_started)
                    outcome.update(status=status, exit_code=code)
                    retain_worker_receipt(outcome, trial, report["execution_source_sha256"])
                except BaseException as error:
                    outcome.update(status="failed_or_interrupted_after_start" if "worker_pid" in outcome else "worker_start_failed",
                                   error={"type": type(error).__name__, "message": str(error)})
                    try:
                        retain_worker_receipt(outcome, trial, report["execution_source_sha256"])
                    except BaseException as receipt_error:
                        outcome["receipt_verification_error"] = {"type": type(receipt_error).__name__,
                                                                 "message": str(receipt_error)}
                    raise
            validate_execution(source, run)
            report["status"] = "bounded_attempts_finished"
        except BaseException as error:
            report.update(status="interrupted_or_failed", error={"type": type(error).__name__, "message": str(error)})
        finally:
            report.update(attempted_source_bytes=attempted_bytes, elapsed_seconds=time.monotonic()-started,
                          outcome_counts=dict(Counter(r["status"] for r in outcomes)))
            atomic_preserve(safe_path(run / "batch.json"), encoded(report))
        return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs="?", choices=("preflight", "batch", "worker"), default="preflight")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--manifest-sha")
    parser.add_argument("--run-id")
    parser.add_argument("--mask")
    parser.add_argument("--max-seconds", type=int, default=600)
    parser.add_argument("--max-bytes", type=int, default=13977015)
    parser.add_argument("--worker-seconds", type=float)
    parser.add_argument("--existing-only", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "preflight":
        manifest, _ = preflight()
        print(json.dumps({"status": "metadata_preflight_passed", "manifest_sha256": MANIFEST_SHA, **manifest["counts"]}))
        return 0
    if not args.execute or args.manifest_sha != MANIFEST_SHA or not args.run_id:
        parser.error("Payload operation requires --execute, exact --manifest-sha, and a new --run-id")
    with termination_cleanup():
        if args.command == "batch":
            result = batch(args.run_id, args.max_seconds, args.max_bytes, only_mask=args.mask, existing_only=args.existing_only)
        else:
            if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,79}", args.run_id) or not args.mask or args.worker_seconds is None:
                parser.error("Worker requires valid run identity, exact mask path and seconds")
            run = safe_path(CACHE / "attempts" / args.run_id)
            trial = safe_path(run / Path(args.mask).name)
            if not trial.is_dir() or (trial / "receipt.json").exists():
                parser.error("Worker requires a fresh owner-prepared attempt directory")
            with safe_path(trial / "worker-claim.json").open("xb") as handle:
                handle.write(encoded({"mask": args.mask, "manifest_sha256": MANIFEST_SHA,
                                      "max_seconds": args.worker_seconds}))
            result = worker(args.mask, run, trial, args.worker_seconds, existing_only=args.existing_only)
    print(json.dumps({k: result.get(k) for k in ("status", "elapsed_seconds", "outcome_counts", "error")}))
    return int(result["status"] in ("failed_or_incomplete", "interrupted_or_failed"))


if __name__ == "__main__":
    raise SystemExit(main())
