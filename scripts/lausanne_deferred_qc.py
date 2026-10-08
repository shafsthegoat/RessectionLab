#!/usr/bin/env python3
"""Offline, separately recorded QC; preflight reads only retained metadata.

No downloader or legacy acquired-receipt writer is called. Explicit execution
reviews only the frozen new originals or the frozen reference-only followups.
"""
from __future__ import annotations

import argparse
import base64
from collections import Counter
from datetime import datetime, timezone
import fcntl
import gzip
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import sys
import time

import lausanne_annotation_intake as annotations
from real_intake_io import atomic_preserve, check_deadline, supervise, termination_cleanup, verify_source_file

ROOT = annotations.ROOT
DATA = annotations.DATA
CACHE = DATA / "deferred-qc-v1"
MANIFEST = ROOT / "manifests/lausanne-deferred-qc-v1.json"
MANIFEST_SHA = "65224289e26db6656d99ee9bdfb089bf80ca309fcbdee63fb469989e9e3d6e19"
INDEX_SHA = "6f1fc7812af0d66550076aa08d37d7f36f08d764fdab701bbbc9ca0608629e66"
GOOD_TRANSFER = {"byte_verified", "existing_byte_verified"}
CLAIMS = {"anatomy_qc": "not_run", "annotation_semantics_review": "not_run",
          "training_admitted": False, "scanner_frame_admitted": False,
          "spatial_planning_admitted": False, "annotation_available_at": None,
          "review_available_at": None, "optimizer_updates": 0, "recorded_rl_transitions": 0,
          "network_requests": 0, "array_operation": "none"}
TRANSFER_CLAIMS = {"header_qc": "not_run", "geometry_qc": "not_run", "anatomy_qc": "not_run",
                   "training_admitted": False, "spatial_planning_admitted": False,
                   "decoded_array_bytes": 0, "optimizer_updates": 0, "recorded_rl_transitions": 0}
Refusal = annotations.AcquisitionError
encode, digest, safe_path, read_metadata = (annotations.encoded, annotations.digest,
                                          annotations.safe_path, annotations.read_metadata)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    atomic_preserve(safe_path(path), encode(value))


def bound_json(path):
    return json.loads(read_metadata(path))


def claims_match(record, expected):
    return all(type(record.get(k)) is type(v) and record.get(k) == v for k, v in expected.items())


def proof_bytes(proof):
    path = safe_path(ROOT / proof["path"])
    raw = read_metadata(path)
    if len(raw) != proof["bytes"] or digest(raw) != proof["sha256"]:
        raise Refusal("Retained metadata proof changed: " + proof["path"])
    return raw


def snapshot_files(files, directory):
    if not files or len(files) > 64:
        raise Refusal("Invalid retained source inventory")
    for name, sha in files.items():
        if Path(name).is_absolute() or ".." in Path(name).parts or not annotations.HEX.fullmatch(sha):
            raise Refusal("Invalid retained source path/hash")
        if digest(read_metadata(safe_path(directory / name))) != sha:
            raise Refusal("Retained source snapshot changed: " + name)


def expected_source(row, index):
    session = next(r for r in index["sessions"] if
                   (r["subject"], r["session"]) == (row["subject"], row["session"]))
    entry = next(e for e in session["files"] if e["path"] == row["path"])
    if not entry["path"].endswith(("_T1w.nii.gz", "_angio.nii.gz")) or row["role"] != "TRAIN":
        raise Refusal("Review outside original TRAIN images")
    return {"key": "lausanne-" + digest(entry["path"].encode())[:24], "dataset": "lausanne",
            "role": "TRAIN", "person": row["subject"], "session": row["session"],
            "provider": "openneuro_s3", "release": {"git_commit": index["git_commit"], "license": index["license"]},
            "manifest_sha256": INDEX_SHA, "entry": entry}


def transfer_contract(row, index, manifest):
    """Metadata only. A good child under a failed parent is never accepted."""
    source = expected_source(row, index)
    if not 0 < source["entry"]["bytes"] <= manifest["bounds"]["max_source_file_bytes"]:
        raise Refusal("Original compressed source exceeds review byte bound")
    records = {k: json.loads(proof_bytes(v)) for k, v in row["transfer"].items()}
    intent, result, outcome = (records[k] for k in ("intent", "result", "outcome"))
    declaration = json.loads(proof_bytes(row["declaration"]))
    trial = Path(row["transfer"]["intent"]["path"]).parent
    run = Path(row["declaration"]["path"]).parent
    binding = digest(encode(source))
    if (source != row["source"] or declaration["schema"] != "continuous-source-acquisition-v1"
            or declaration["execution"]["files"] != manifest["protected_live_sources"]
            or declaration["execution"]["manifests"].get(manifest["original_index"]["path"]) != manifest["original_index"]["sha256"]
            or not str(trial).startswith("data/acquisition/continuous-source-v1/objects/" + source["key"] + "/attempts/")
            or not str(run).startswith("data/acquisition/continuous-source-v1/runs/")
            or intent.get("run") != str(run) or intent.get("source_key") != source["key"]
            or intent.get("source_binding") != binding or intent.get("declaration_sha256") != row["declaration"]["sha256"]
            or any(Path(v["path"]).parent != trial for v in row["transfer"].values())):
        raise Refusal("Transfer declaration/source/intent binding differs")
    for r in (result, outcome):
        if (r.get("status") not in GOOD_TRANSFER or r.get("source_key") != source["key"]
                or r.get("source_binding") != binding or r.get("intent_sha256") != row["transfer"]["intent"]["sha256"]
                or not claims_match(r, TRANSFER_CLAIMS) or r.get("verified_bytes") != source["entry"]["bytes"]
                or not annotations.HEX.fullmatch(r.get("sha256", ""))):
            raise Refusal("Transfer result and parent must both be byte-verified, without QC claims")
    if (result.get("source") != source or result.get("declaration_sha256") != row["declaration"]["sha256"]
            or outcome.get("attempt") != str(trial) or outcome.get("provider") != source["provider"]
            or outcome.get("result_sha256") != row["transfer"]["result"]["sha256"]
            or outcome.get("sha256") != result["sha256"] or outcome.get("supervision") != "completed"
            or outcome.get("returncode") != 0 or not claims_match(declaration, TRANSFER_CLAIMS)):
        raise Refusal("Transfer parent/result chain differs")
    snapshot_files({**declaration["execution"]["files"], **declaration["execution"]["manifests"]},
                   ROOT / run / "snapshot")
    return result


def prior_mask_contract(row, annotation):
    """Authenticate the existing scalar review; do not repeat that review."""
    receipt = json.loads(proof_bytes(row["receipt"]))
    batch = json.loads(proof_bytes(row["batch"]))
    source = json.loads(proof_bytes(row["source"]))
    outcomes = [r for r in batch["outcomes"] if r["path"] == row["path"]]
    acquisition = receipt.get("acquisition", {})
    if (len(outcomes) != 1 or outcomes[0].get("status") != "completed" or outcomes[0].get("exit_code") != 0
            or outcomes[0].get("receipt_sha256") != row["receipt"]["sha256"]
            or outcomes[0].get("attempt") != str(Path(row["receipt"]["path"]).parent)
            or receipt.get("schema") != "lausanne-annotation-attempt-v1"
            or any(receipt.get(k) != row[k] for k in ("path", "subject", "session", "role", "annotation_subtype_status"))
            or receipt.get("manifest_sha256") != annotations.MANIFEST_SHA
            or receipt.get("status") != "content_passed_reference_or_grid_pending"
            or acquisition.get("status") != "source_verified" or acquisition.get("bytes") != annotation["file"]["bytes"]
            or not annotations.HEX.fullmatch(acquisition.get("sha256", ""))
            or receipt.get("content_qc", {}).get("status") != "passed"
            or receipt.get("content_qc", {}).get("background_semantics") != "unknown"
            or receipt.get("reference_qc", {}).get("status") != "deferred_missing_original_receipt"
            or receipt.get("grid_qc", {}).get("status") != "deferred_reference_not_passed"
            or receipt.get("execution_source_record") != row["source"]["path"]
            or receipt.get("execution_source_sha256") != row["source"]["sha256"]
            or batch.get("execution_source_sha256") != row["source"]["sha256"]
            or any(receipt.get(k) is not False for k in ("training_admitted", "scanner_frame_admitted", "spatial_planning_admitted"))
            or receipt.get("annotation_available_at") is not None or receipt.get("review_available_at") is not None):
        raise Refusal("Prior mask scalar-review/source/parent contract differs")
    if (row["annotation_available_at"] is not None or row["review_available_at"] is not None
            or row["background_semantics"] != "unknown"
            or row["annotation_subtype_status"] != annotation["annotation_subtype_status"]):
        raise Refusal("Preserved annotation timing/subtype/background semantics differ")
    validate_raw_grid(receipt["content_qc"]["raw_grid"], acquisition["sha256"])
    snapshot_files(source["files"], ROOT / Path(row["source"]["path"]).parent / "source-snapshot")
    return receipt


def validate_raw_grid(record, sha):
    raw = base64.b64decode(record["header_base64"], validate=True)
    if (len(raw) != 348 or record.get("header_sha256") != digest(raw)
            or record.get("source_file_sha256") != "sha256:" + sha):
        raise Refusal("Stored original header/source binding differs")
    return raw


def preflight():
    raw = read_metadata(MANIFEST)
    if digest(raw) != MANIFEST_SHA:
        raise Refusal("Frozen deferred-QC worklist changed")
    m = json.loads(raw)
    annotation_manifest, sessions = annotations.preflight()
    index = json.loads(proof_bytes(m["original_index"]))
    cohort = json.loads(proof_bytes(m["cohort"]))
    proof_bytes(m["annotation_manifest"])
    queue = json.loads(proof_bytes(m["basis_queue"]))
    people = {p["subject"]: p for p in cohort["members"] if p["role"] == "TRAIN"}
    expected_images = [r["path"] for r in queue["records"] if r["next_qc_stage"] == "file_qc_pending_transfer_receipt"]
    expected_masks = [r["path"] for r in queue["records"] if r["next_qc_stage"] == "join_reviewed_same_source_TOF_then_grid_proof"]
    preserved = [r for r in queue["records"] if r["next_qc_stage"] in (
        "known_frame_conflict_review", "known_mask_grid_conflict_review", "metadata_excluded_not_in_qualified_download_scope")]
    if (len(expected_images) != 26 or len(expected_masks) != 27 or len(set(expected_images)) != 26
            or len(set(expected_masks)) != 27 or [r["path"] for r in m["originals"]] != expected_images
            or [r["path"] for r in m["reference_followups"]] != expected_masks or m["preserved_failures"] != preserved):
        raise Refusal("Deferred worklist changed or historical failures dropped")
    for name, sha in m["protected_live_sources"].items():
        if digest(read_metadata(ROOT / name)) != sha:
            raise Refusal("Protected acquisition source changed")
    for row in m["originals"]:
        if people[row["subject"]]["group"] != row["source_group"]:
            raise Refusal("Original source group changed")
        transfer_contract(row, index, m)
    annotation_rows = {r["path"]: r for r in annotation_manifest["records"]}
    for row in m["reference_followups"]:
        a = annotation_rows[row["path"]]
        if (a["status"] != "metadata_qualified" or a["original_reference"]["path"] != row["reference_path"]
                or any(a[k] != row[k] for k in ("subject", "session", "role", "annotation_subtype_status",
                                               "annotation_available_at", "review_available_at"))
                or row["role"] != "TRAIN" or row["background_semantics"] != "unknown"):
            raise Refusal("Mask semantics or exact source reference changed")
        prior_mask_contract(row, a)
        if row["reference_mode"] == "legacy_session_receipt":
            saved = json.loads(proof_bytes(row["legacy_receipt"]))
            contract = annotations.original_receipt_contract(sessions[(row["subject"], row["session"])], saved,
                m["original_index"]["sha256"], a["original_reference"])
            if not contract["referenced_tof_passed"]:
                raise Refusal("Frozen legacy TOF review did not pass")
        elif row["reference_mode"] != "separate_original_review" or row["reference_path"] not in expected_images:
            raise Refusal("Unknown reference-review route")
    return m, annotation_manifest, sessions


def execution_source(manifest):
    for module in (annotations, annotations.originals, sys.modules["real_intake_io"]):
        if Path(module.__file__).resolve() != ROOT / "scripts" / (module.__name__ + ".py"):
            raise Refusal("Imported review helper is outside this checkout")
    names = set(manifest["protected_live_sources"]) | set(annotations.SOURCE_NAMES) | {
        "scripts/lausanne_deferred_qc.py", "manifests/lausanne-deferred-qc-v1.json"}
    return {"files": {name: digest(read_metadata(ROOT / name)) for name in sorted(names)},
            "python": sys.version, "numpy": importlib.metadata.version("numpy"),
            "nibabel": importlib.metadata.version("nibabel")}


def validate_execution(run, declaration, manifest, *, current):
    source = declaration["execution"]
    expected = execution_source(manifest)
    if (set(source["files"]) != set(expected["files"])
            or source["files"].get("manifests/lausanne-deferred-qc-v1.json") != MANIFEST_SHA
            or any(source["files"].get(name) != sha for name, sha in manifest["protected_live_sources"].items())):
        raise Refusal("Review source inventory or protected dependency pins differ")
    if current and source != execution_source(manifest):
        raise Refusal("Executing review source/runtime changed")
    snapshot_files(source["files"], run / "source-snapshot")


def scalar_budget(shape, itemsize, bounds):
    if len(shape) != 3 or any(type(n) is not int or n < 1 for n in shape) or itemsize not in (1, 2, 4, 8):
        raise Refusal("Unsupported original scalar dimensions/dtype")
    voxels = math.prod(shape)
    chunk = min(bounds["scalar_chunk_voxels"], bounds["max_scalar_workspace_bytes"] // (itemsize + 9))
    if voxels > bounds["max_voxels"] or voxels * itemsize > bounds["max_native_payload_bytes"] or chunk < 1:
        raise Refusal("Original decoded size exceeds declared bounds")
    return {"voxels": voxels, "native_payload_bytes": voxels * itemsize, "chunk_voxels": chunk,
            "scalar_workspace_bytes_bound": chunk * (itemsize + 9)}


def inspect_original(path, sha, bounds, deadline):
    """Future explicit payload operation: bounded streaming, no image allocation."""
    import nibabel as nib
    import numpy as np
    from resectionlab.critical_evidence import nifti1_header_record
    from resectionlab.imaging import inspect_nifti, ImagingError

    result = {"header_qc": {"status": "not_run"}, "scalar_qc": {"status": "not_run"},
              "geometry_qc": {"status": "not_run"}}
    try:
        with gzip.open(safe_path(path), "rb") as stream:
            raw = stream.read(348)
            h = nib.Nifti1Header(binaryblock=raw, check=False)
            if int(h["sizeof_hdr"]) != 348 or bytes(h["magic"]) != b"n+1\0":
                raise Refusal("Only exact single-file NIfTI-1 originals are supported")
            dtype = h.get_data_dtype()
            if dtype.kind not in "iuf" or int(h["bitpix"]) != dtype.itemsize * 8:
                raise Refusal("Original requires real scalar dtype")
            budget = scalar_budget([int(n) for n in h.get_data_shape()], dtype.itemsize, bounds)
            offset = float(h["vox_offset"])
            if not math.isfinite(offset) or not offset.is_integer() or not 352 <= offset <= bounds["max_nifti_data_offset"]:
                raise Refusal("Original data offset outside declared bound")
            extensions = annotations.inspect_extensions(stream.read(int(offset) - 348), data_offset=int(offset),
                endian=h.endianness, maximum_offset=bounds["max_nifti_data_offset"])
            slope, intercept = h.get_slope_inter()
            slope, intercept = (1., 0.) if slope is None else (float(slope), float(intercept))
            if not math.isfinite(slope) or not math.isfinite(intercept):
                raise Refusal("Nonfinite original scaling")
            for module_name in ("resectionlab.imaging", "resectionlab.critical_evidence"):
                if Path(sys.modules[module_name].__file__).resolve() != ROOT / "src" / (module_name.replace(".", "/") + ".py"):
                    raise Refusal("Imported scientific QC code is outside this checkout")
            result.update(raw_grid=nifti1_header_record(raw, sha), decoding_budget=budget,
                          header_qc={"status": "passed", "extensions": extensions, "dtype": dtype.str,
                                     "source_scaling": [slope, intercept]})
            check_deadline(deadline)
            try:
                result["geometry_qc"] = {"status": "passed", "header": inspect_nifti(path)}
            except ImagingError as error:
                result["geometry_qc"] = {"status": "failed", "reason": str(error), "code": error.code}
            remaining, minimum, maximum = budget["voxels"], math.inf, -math.inf
            result["scalar_qc"] = {"status": "failed_or_incomplete"}
            while remaining:
                check_deadline(deadline)
                count = min(remaining, budget["chunk_voxels"])
                payload = stream.read(count * dtype.itemsize)
                if len(payload) != count * dtype.itemsize:
                    raise Refusal("Truncated original scalar payload")
                values = np.frombuffer(payload, dtype=dtype).astype(np.float64)
                values *= slope
                values += intercept
                if not np.isfinite(values).all():
                    raise Refusal("Nonfinite source intensities")
                minimum, maximum = min(minimum, float(values.min())), max(maximum, float(values.max()))
                remaining -= count
                del values, payload
            if stream.read(1):
                raise Refusal("Unexpected bytes after declared original scalar array")
            result["scalar_qc"] = {"status": "passed", "voxels": budget["voxels"], "minimum": minimum,
                                   "maximum": maximum, "dtype_after_scaling": "float64", "array_retained": False}
    except (ValueError, OSError, Refusal) as error:
        result["error"] = {"type": type(error).__name__, "message": str(error)}
    return result


def review_key(path):
    return digest(path.encode())[:24]


def review_receipt_contract(receipt, row, intent_sha, declaration_sha):
    if (receipt.get("schema") != "lausanne-file-review-v1" or receipt.get("manifest_sha256") != MANIFEST_SHA
            or receipt.get("path") != row["path"] or receipt.get("subject") != row["subject"]
            or receipt.get("session") != row["session"] or receipt.get("role") != "TRAIN"
            or receipt.get("intent_sha256") != intent_sha or receipt.get("declaration_sha256") != declaration_sha
            or not claims_match(receipt, CLAIMS)):
        raise Refusal("Separate review receipt identity/source/claims differ")
    if receipt.get("status") not in ("review_passed", "review_failed", "failed_or_incomplete"):
        raise Refusal("Unknown separate review status")


def completed_original_review(receipt_path, row, manifest):
    """Validate a separately reviewed original without opening that image."""
    receipt_path = safe_path(receipt_path)
    trial, run = receipt_path.parent, receipt_path.parent.parent.parent
    if receipt_path.name != "receipt.json" or run.parent != safe_path(CACHE / "runs") or trial.name != review_key(row["path"]):
        raise Refusal("Review outside the separate immutable namespace")
    declaration_raw = read_metadata(run / "declaration.json")
    declaration = json.loads(declaration_raw)
    intent_raw = read_metadata(trial / "intent.json")
    intent = json.loads(intent_raw)
    receipt_raw = read_metadata(receipt_path)
    receipt = json.loads(receipt_raw)
    outcome = bound_json(trial / "outcome.json")
    if (declaration.get("schema") != "lausanne-deferred-qc-run-v1"
            or declaration.get("manifest_sha256") != MANIFEST_SHA or declaration.get("stage") != "originals"
            or not claims_match(declaration, CLAIMS)
            or declaration.get("run_id") != run.name or row["path"] not in declaration.get("selected_paths", [])
            or intent.get("path") != row["path"] or intent.get("stage") != "originals"
            or intent.get("declaration_sha256") != digest(declaration_raw)
            or outcome.get("status") != "completed" or outcome.get("exit_code") != 0
            or outcome.get("path") != row["path"] or outcome.get("intent_sha256") != digest(intent_raw)
            or outcome.get("receipt_sha256") != digest(receipt_raw)):
        raise Refusal("Separate review parent/declaration/intent chain differs")
    review_receipt_contract(receipt, row, digest(intent_raw), digest(declaration_raw))
    # The initial bridge recognizes only this reviewed implementation/runtime.
    # Retaining arbitrary code is not authority to accept its scientific claims.
    validate_execution(run, declaration, manifest, current=True)
    for name, proof in {**row["transfer"], "declaration": row["declaration"]}.items():
        if read_metadata(trial / "input-snapshot" / (name + ".json")) != proof_bytes(proof):
            raise Refusal("Review transfer receipt snapshot changed")
    if (receipt["status"] != "review_passed" or receipt.get("stage") != "originals"
            or receipt.get("source") != row["source"] or receipt.get("transfer_proofs") != row["transfer"]
            or receipt.get("bytes") != row["source"]["entry"]["bytes"]
            or not annotations.HEX.fullmatch(receipt.get("sha256", ""))
            or any(receipt.get(k, {}).get("status") != "passed" for k in ("header_qc", "scalar_qc", "geometry_qc"))
            or receipt.get("current_fixity_checked_before_and_after") is not True):
        raise Refusal("Referenced TOF lacks a successful separate integrity/geometry review")
    transfer = json.loads(proof_bytes(row["transfer"]["result"]))
    if receipt["sha256"] != transfer["sha256"]:
        raise Refusal("Separate review differs from downloaded source hash")
    validate_raw_grid(receipt["raw_grid"], receipt["sha256"])
    h, scalar = receipt["geometry_qc"]["header"], receipt["scalar_qc"]
    if (h.get("status") != "passed_header_checks" or h.get("file") != Path(row["path"]).name
            or h.get("physical_units") != "mm" or len(h.get("shape", [])) != 3
            or any(type(n) is not int or n <= 0 for n in h["shape"])
            or math.prod(h["shape"]) != scalar["voxels"]
            or not all(math.isfinite(scalar[k]) for k in ("minimum", "maximum")) or scalar["minimum"] > scalar["maximum"]):
        raise Refusal("Inconsistent separate scalar/header review metadata")
    return receipt, receipt_raw


def separately_reviewed_reference(row, original, annotation_manifest, trial, deadline, receipt_path):
    """Typed resolver used only when an explicit separate TOF receipt is supplied."""
    m, frozen_annotations, sessions = preflight()
    reference = row["original_reference"]
    matches = [r for r in m["originals"] if r["path"] == reference["path"]]
    source_row = next((r for r in frozen_annotations["records"] if r["path"] == row["path"]), None)
    if (row != source_row or original != sessions.get((row["subject"], row["session"]))
            or row.get("role") != "TRAIN" or original.get("role") != "TRAIN"
            or len(matches) != 1 or not reference["path"].endswith("_angio.nii.gz")
            or matches[0]["subject"] != row["subject"] or matches[0]["session"] != row["session"]
            or matches[0]["source"]["entry"] != reference
            or annotation_manifest["original_index"]["sha256"] != m["original_index"]["sha256"]
            or (original["subject"], original["session"]) != (row["subject"], row["session"])):
        raise Refusal("Separate reference must be the exact same-session original TOF")
    saved, raw = completed_original_review(receipt_path, matches[0], m)
    atomic_preserve(safe_path(trial / "separate-tof-review.json"), raw)
    sha = verify_source_file(safe_path(DATA / reference["path"]), reference,
                             receipt_sha=saved["sha256"], deadline=deadline)
    return {"status": "passed", "reference_review_kind": "separate_original_file_review",
            "review_receipt_path": str(receipt_path.relative_to(ROOT)), "review_receipt_sha256": digest(raw),
            "review_receipt_snapshot": str((trial / "separate-tof-review.json").relative_to(ROOT)),
            "referenced_tof_passed": True, "referenced_tof_qc": {k: saved[k] for k in ("header_qc", "scalar_qc", "geometry_qc")},
            "other_modality_qc": [], "other_modality_review": "not_assessed_by_this_reference_resolution",
            "source_receipt_pair_qc": "not_assessed", "full_pair_current_fixity_checked": False,
            "current_tof_fixity_checked": True, "original_tof_sha256": sha, "raw_grid": saved["raw_grid"]}


def followup_reference(row, manifest, annotation_manifest, sessions, trial, deadline, review_paths):
    a = next(a for a in annotation_manifest["records"] if a["path"] == row["path"])
    prior = prior_mask_contract(row, a)
    atomic_preserve(trial / "prior-mask-receipt.json", proof_bytes(row["receipt"]))
    sha = verify_source_file(safe_path(DATA / row["path"]), a["file"], receipt_sha=prior["acquisition"]["sha256"], deadline=deadline)
    separate = None
    if row["reference_mode"] == "legacy_session_receipt":
        proof_bytes(row["legacy_receipt"])
    else:
        if row["reference_path"] not in review_paths:
            raise Refusal("Separate TOF review was not explicitly supplied")
        separate = safe_path(ROOT / review_paths[row["reference_path"]])
    reference = annotations.verify_reference(a, sessions[(row["subject"], row["session"])],
        annotation_manifest, trial, deadline, separate_review_receipt=separate)
    grid = {"status": "deferred_reference_not_passed"}
    if reference["status"] == "passed":
        try:
            grid = {"status": "passed", "proof": annotations.grid_proof(prior["content_qc"]["raw_grid"], reference["raw_grid"])}
        except (ValueError, Refusal) as error:
            grid = {"status": "unresolved", "reason": str(error)}
    verify_source_file(safe_path(DATA / row["path"]), a["file"], receipt_sha=sha, deadline=deadline)
    return {"status": "review_passed" if grid["status"] == "passed" else "review_failed",
            "prior_mask_receipt": row["receipt"], "mask_sha256": sha, "mask_scalars_decoded": False,
            "content_qc": {"status": "reused_source_bound_prior_pass", "receipt_sha256": row["receipt"]["sha256"]},
            "reference_qc": reference, "grid_qc": grid, "annotation_subtype_status": row["annotation_subtype_status"],
            "background_semantics": "unknown"}


def worker(run, trial, seconds):
    started = time.monotonic()
    deadline = started + seconds
    m, annotation_manifest, sessions = preflight()
    declaration_raw = read_metadata(run / "declaration.json")
    declaration = json.loads(declaration_raw)
    intent_raw = read_metadata(trial / "intent.json")
    intent = json.loads(intent_raw)
    stage = declaration["stage"]
    if stage not in ("originals", "references"):
        raise Refusal("Worker has unknown review stage")
    rows = m["originals" if stage == "originals" else "reference_followups"]
    row = next((r for r in rows if r["path"] == intent["path"]), None)
    if (not row or trial != run / "items" / review_key(row["path"]) or intent.get("stage") != stage
            or intent.get("supervisor_pid") != os.getppid() or intent.get("declaration_sha256") != digest(declaration_raw)
            or declaration.get("manifest_sha256") != MANIFEST_SHA or declaration.get("run_id") != run.name
            or intent.get("max_seconds") != seconds or not claims_match(declaration, CLAIMS)
            or row["path"] not in declaration["selected_paths"] or not 0 < seconds <= m["bounds"]["max_worker_seconds"]):
        raise Refusal("Worker identity/scope/time contract differs")
    result = {"schema": "lausanne-file-review-v1", "manifest_sha256": MANIFEST_SHA, "stage": stage,
              "path": row["path"], "subject": row["subject"], "session": row["session"], "role": "TRAIN",
              "intent_sha256": digest(intent_raw), "declaration_sha256": digest(declaration_raw),
              "status": "failed_or_incomplete", "started_at": utc_now(), **CLAIMS}
    try:
        validate_execution(run, declaration, m, current=True)
        if stage == "originals":
            transfer = transfer_contract(row, json.loads(proof_bytes(m["original_index"])), m)
            for name, proof in {**row["transfer"], "declaration": row["declaration"]}.items():
                atomic_preserve(safe_path(trial / "input-snapshot" / (name + ".json")), proof_bytes(proof))
            entry = row["source"]["entry"]
            sha = verify_source_file(safe_path(DATA / row["path"]), entry, receipt_sha=transfer["sha256"], deadline=deadline)
            result.update(source=row["source"], transfer_proofs=row["transfer"], sha256=sha, bytes=entry["bytes"])
            result.update(inspect_original(DATA / row["path"], sha, m["bounds"], deadline))
            verify_source_file(safe_path(DATA / row["path"]), entry, receipt_sha=sha, deadline=deadline)
            result["current_fixity_checked_before_and_after"] = True
            result["status"] = "review_passed" if all(result[k]["status"] == "passed" for k in
                ("header_qc", "scalar_qc", "geometry_qc")) else "review_failed"
        else:
            result.update(followup_reference(row, m, annotation_manifest, sessions, trial, deadline, declaration["review_paths"]))
        check_deadline(deadline)
        validate_execution(run, declaration, m, current=True)
    except BaseException as error:
        result.update(status="failed_or_incomplete", error={"type": type(error).__name__, "message": str(error)})
    finally:
        result.update(finished_at=utc_now(), elapsed_seconds=time.monotonic() - started)
        save(trial / "receipt.json", result)
    return result


def selected_rows(manifest, stage, only_path):
    if stage not in ("originals", "references"):
        raise Refusal("Unknown review stage")
    rows = manifest["originals" if stage == "originals" else "reference_followups"]
    if only_path is not None and only_path not in {r["path"] for r in rows}:
        raise Refusal("Requested path outside frozen stage whitelist")
    return [r for r in rows if only_path is None or r["path"] == only_path]


def review_paths_for_runs(manifest, run_ids, needed_paths):
    paths = {}
    for run_id in run_ids:
        if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,79}", run_id):
            raise Refusal("Invalid explicit original review run")
        for row in manifest["originals"]:
            if row["path"] not in needed_paths or not row["path"].endswith("_angio.nii.gz"):
                continue
            p = safe_path(CACHE / "runs" / run_id / "items" / review_key(row["path"]) / "receipt.json")
            if p.exists():
                # Qualification happens per target in the worker. A failed or
                # missing review is retained there, without blocking other TOFs.
                if row["path"] in paths:
                    raise Refusal("Ambiguous duplicate supplied TOF reviews")
                paths[row["path"]] = str(p.relative_to(ROOT))
    return paths


def retain_attempt_receipt(outcome, trial, row, intent, declaration):
    """Retain readable bytes even when their claimed review cannot be validated."""
    raw = read_metadata(trial / "receipt.json")
    outcome["receipt_sha256"] = digest(raw)
    receipt = json.loads(raw)
    review_receipt_contract(receipt, row, digest(encode(intent)), digest(encode(declaration)))
    outcome["review_status"] = receipt["status"]


def batch(run_id, stage, seconds, *, only_path=None, review_runs=()):
    started = time.monotonic()
    m, _, _ = preflight()
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,79}", run_id) or not 1 <= seconds <= m["bounds"]["max_batch_seconds"]:
        raise Refusal("Invalid review run/time bound")
    selected = selected_rows(m, stage, only_path)
    needed = {r["reference_path"] for r in selected if stage == "references"
              and r["reference_mode"] == "separate_original_review"}
    if needed and not review_runs:
        raise Refusal("Reference followups require explicit original review runs")
    review_paths = review_paths_for_runs(m, review_runs, needed)
    deadline = started + seconds
    safe_path(CACHE).mkdir(parents=True, exist_ok=True)
    with safe_path(CACHE / "review.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run = safe_path(CACHE / "runs" / run_id)
        run.mkdir(parents=True, exist_ok=False)
        source = execution_source(m)
        for name in source["files"]:
            atomic_preserve(safe_path(run / "source-snapshot" / name), read_metadata(ROOT / name))
        declaration = {"schema": "lausanne-deferred-qc-run-v1", "manifest_sha256": MANIFEST_SHA,
                       "run_id": run_id, "stage": stage, "max_seconds": seconds, "workers": 1,
                       "automatic_retries": 0, "selected_paths": [r["path"] for r in selected],
                       "review_paths": review_paths, "execution": source, "created_at": utc_now(), **CLAIMS}
        save(run / "declaration.json", declaration)
        outcomes = [{"path": r["path"], "status": "deferred_not_selected" if r not in selected else "deferred_not_attempted"}
                    for r in m["originals" if stage == "originals" else "reference_followups"]]
        report = {"schema": "lausanne-deferred-qc-batch-v1", "manifest_sha256": MANIFEST_SHA,
                  "run_id": run_id, "stage": stage, "outcomes": outcomes, "preserved_failures": m["preserved_failures"], **CLAIMS}
        try:
            for row in selected:
                check_deadline(deadline)
                if deadline - time.monotonic() < 5:
                    raise TimeoutError("Insufficient time for another supervised review")
                validate_execution(run, declaration, m, current=True)
                check_deadline(deadline)
                trial = safe_path(run / "items" / review_key(row["path"]))
                trial.mkdir(parents=True, exist_ok=False)
                allowance = min(m["bounds"]["max_worker_seconds"], deadline - time.monotonic() - 1)
                intent = {"path": row["path"], "stage": stage, "supervisor_pid": os.getpid(),
                          "declaration_sha256": digest(encode(declaration)), "max_seconds": allowance}
                save(trial / "intent.json", intent)
                outcome = next(o for o in outcomes if o["path"] == row["path"])
                outcome.update(status="failed_or_incomplete", intent_sha256=digest(encode(intent)),
                               attempt=str(trial.relative_to(ROOT)))
                problem = None
                try:
                    command = [sys.executable, str(Path(__file__).resolve()), "worker", "--execute", "--manifest-sha", MANIFEST_SHA,
                               "--run-id", run_id, "--path", row["path"], "--worker-seconds", str(allowance)]
                    status, code = supervise(command, trial / "worker.log", deadline=min(deadline, time.monotonic() + allowance + .5),
                                             on_start=lambda pid: outcome.update(worker_pid=pid))
                    outcome.update(supervision_status=status, exit_code=code)
                    check_deadline(deadline)
                except BaseException as error:
                    problem = error
                    outcome["error"] = {"type": type(error).__name__, "message": str(error)}
                finally:
                    # Supervision may raise after the child wrote its receipt.
                    # Evidence binding must therefore happen on every exit path.
                    try:
                        retain_attempt_receipt(outcome, trial, row, intent, declaration)
                    except BaseException as error:
                        outcome["receipt_binding_error"] = {"type": type(error).__name__, "message": str(error)}
                        if problem is None:
                            problem = error
                    try:
                        check_deadline(deadline)
                        if problem is None:
                            validate_execution(run, declaration, m, current=True)
                            check_deadline(deadline)
                    except BaseException as error:
                        outcome["completion_validation_error"] = {"type": type(error).__name__, "message": str(error)}
                        if problem is None:
                            problem = error
                    if problem is None:
                        outcome["status"] = outcome["supervision_status"]
                    else:
                        outcome.setdefault("error", {"type": type(problem).__name__, "message": str(problem)})
                    save(trial / "outcome.json", outcome)
                if problem is not None:
                    raise problem
            validate_execution(run, declaration, m, current=True)
            check_deadline(deadline)
            report["status"] = "bounded_attempts_finished"
        except BaseException as error:
            report.update(status="failed_or_incomplete", error={"type": type(error).__name__, "message": str(error)})
        finally:
            report.update(elapsed_seconds=time.monotonic() - started, outcome_counts=dict(Counter(o["status"] for o in outcomes)))
            save(run / "batch.json", report)
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("preflight", "batch", "worker"), nargs="?", default="preflight")
    p.add_argument("--stage", choices=("originals", "references"))
    p.add_argument("--execute", action="store_true")
    p.add_argument("--manifest-sha")
    p.add_argument("--run-id")
    p.add_argument("--path")
    p.add_argument("--review-run", action="append", default=[])
    p.add_argument("--max-seconds", type=int, default=1800)
    p.add_argument("--worker-seconds", type=float)
    a = p.parse_args(argv)
    if a.command == "preflight":
        m, _, _ = preflight()
        print(json.dumps({"status": "metadata_preflight_passed", "manifest_sha256": MANIFEST_SHA, **m["counts"]}))
        return 0
    if not a.execute or a.manifest_sha != MANIFEST_SHA or not a.run_id:
        p.error("Scientific review requires --execute, exact --manifest-sha and a fresh --run-id")
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,79}", a.run_id):
        p.error("Invalid review run identity")
    with termination_cleanup():
        if a.command == "batch":
            if not a.stage:
                p.error("Batch requires an explicit stage")
            result = batch(a.run_id, a.stage, a.max_seconds, only_path=a.path, review_runs=a.review_run)
        else:
            if not a.path or a.worker_seconds is None or not math.isfinite(a.worker_seconds) or not 0 < a.worker_seconds <= 120:
                p.error("Worker requires exact path and bounded worker seconds")
            run = safe_path(CACHE / "runs" / a.run_id)
            trial = safe_path(run / "items" / review_key(a.path))
            if not trial.is_dir() or (trial / "receipt.json").exists():
                p.error("Worker requires a fresh supervisor-prepared attempt")
            with safe_path(trial / "worker-claim.json").open("xb") as handle:
                handle.write(encode({"path": a.path, "pid": os.getpid()}))
            result = worker(run, trial, a.worker_seconds)
    print(json.dumps({k: result.get(k) for k in ("status", "elapsed_seconds", "outcome_counts", "error")}))
    return int(result["status"] == "failed_or_incomplete")


if __name__ == "__main__":
    raise SystemExit(main())
