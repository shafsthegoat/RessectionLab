#!/usr/bin/env python3
"""Resumable original-only intake for the frozen Lausanne TRAIN partition.

Index source metadata first. Download sessions serially in supervised workers;
each worker verifies byte/version identity and actual scalar/header integrity.
All 210 TRAIN sessions remain in the denominator, including failures. SELECT
and MEASUREMENT_EVAL payloads are never selected. No model or learning runs.
"""
from __future__ import annotations

import argparse
import base64
import fcntl
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import threading
import time
from urllib.parse import parse_qs, quote, urlparse
from urllib.request import Request

from acquire_lausanne_pilot import (COHORT, COMMIT, DATA, RAW, ROOT,
    S3, members_from_source, json_bytes)
from acquire_btc_case import acquire_file, open_without_redirect, verify_file
from acquire_public_case import AcquisitionError, sha256_file
from real_intake_io import (IntakeDeadline, atomic_preserve as preserve,
    check_deadline, supervise, termination_cleanup, verify_source_file)

COHORT_SHA = "891a53680edfce604eed9a1791bc790efa615db8ce54da0c92de834a6b87d49a"
INDEX = ROOT / "manifests/lausanne-train-originals-v1.json"
CACHE = DATA / "train-intake-v1"
MAX_IMAGE_BYTES = 256 * 1024**2
MAX_TOTAL_BYTES = 20 * 1024**3
SOURCE_NAMES = ("scripts/lausanne_train_intake.py", "scripts/real_intake_io.py",
    "scripts/acquire_lausanne_pilot.py", "scripts/acquire_btc_case.py", "scripts/acquire_public_case.py",
    "src/resectionlab/imaging.py", "src/resectionlab/core.py", "src/resectionlab/data_policy.py",
    "src/resectionlab/structural_evidence.py")


def train_sessions() -> list[tuple[str, str]]:
    if sha256_file(COHORT) != COHORT_SHA:
        raise AcquisitionError("Frozen cohort changed")
    cohort = json.loads(COHORT.read_bytes())
    source = (DATA / "source-metadata/participants.tsv").read_bytes()
    if cohort["members"] != members_from_source(source):
        raise AcquisitionError("Cohort differs from published source and fixed roles")
    sessions = [(m["subject"], "ses-" + visit) for m in cohort["members"]
                if m["role"] == "TRAIN" for visit in m["sessions"]]
    if len(sessions) != 210 or len({person for person, _ in sessions}) != 199:
        raise AcquisitionError("TRAIN denominator changed")
    return sorted(sessions)


def source_metadata(relative: str) -> bytes:
    with open_without_redirect(Request(RAW + relative, headers={"Accept-Encoding": "identity"}), timeout=30) as response:
        if response.status != 200 or response.geturl() != RAW + relative:
            raise AcquisitionError("Unexpected pinned Git metadata response")
        payload = response.read(65537)
    if len(payload) > 65536:
        raise AcquisitionError("Source metadata exceeds fixed 64-KiB limit")
    return payload


def resolve_session(subject: str, session: str) -> dict:
    prefix = f"{subject}/{session}/anat/{subject}_{session}_"
    files = []
    for modality in ("T1w", "angio"):
        path = prefix + modality + ".nii.gz"
        pointer = source_metadata(path)
        match = re.search(rb"MD5E-s([0-9]+)--([0-9a-f]{32})\.nii\.gz\s*$", pointer)
        if not match:
            raise AcquisitionError("Unrecognized source Git-annex pointer")
        size, digest = int(match[1]), match[2].decode()
        if not 0 < size <= MAX_IMAGE_BYTES:
            raise AcquisitionError("Original compressed image exceeds declared file limit")
        with open_without_redirect(Request(S3 + path, method="HEAD"), timeout=30) as response:
            version = response.headers.get("x-amz-version-id")
            if (response.status != 200 or response.geturl() != S3 + path or not version
                    or version == "null" or int(response.headers.get("Content-Length", -1)) != size
                    or response.headers.get("ETag", "").strip('"') != digest):
                raise AcquisitionError("Current official object differs from pinned Git-annex size/MD5/version")
        files.append({"path": path, "bytes": size, "sha256": None, "expected_md5": digest,
                      "source_url": S3 + path + "?versionId=" + quote(version, safe=""),
                      "git_url": RAW + path, "pointer_sha256": hashlib.sha256(pointer).hexdigest(),
                      "pointer_base64": base64.b64encode(pointer).decode()})
        path = prefix + modality + ".json"
        payload = source_metadata(path)
        sidecar = json.loads(payload)
        if not {"ORIGINAL", "PRIMARY"}.issubset(sidecar.get("ImageType", [])):
            raise AcquisitionError("Source sidecar is not declared ORIGINAL/PRIMARY")
        files.append({"path": path, "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest(),
                      "source_url": RAW + path, "git_url": RAW + path,
                      "metadata_base64": base64.b64encode(payload).decode()})
    return {"subject": subject, "session": session, "role": "TRAIN", "cohort_sha256": COHORT_SHA,
            "git_commit": COMMIT, "files": files, "status": "source_indexed"}


def validate_session(record: dict, identity: tuple[str, str]) -> None:
    subject, session = identity
    prefix = f"{subject}/{session}/anat/{subject}_{session}_"
    expected = {prefix + modality + suffix for modality in ("T1w", "angio") for suffix in (".nii.gz", ".json")}
    if (record.get("subject") != subject or record.get("session") != session or record.get("role") != "TRAIN"
            or record.get("cohort_sha256") != COHORT_SHA or record.get("git_commit") != COMMIT
            or record.get("status") != "source_indexed" or len(record.get("files", [])) != 4
            or {entry["path"] for entry in record["files"]} != expected):
        raise AcquisitionError("Session source/role contract differs")
    for entry in record["files"]:
        path = entry["path"]
        if entry["git_url"] != RAW + path or type(entry["bytes"]) is not int or entry["bytes"] <= 0:
            raise AcquisitionError("Invalid source path or size")
        if path.endswith(".nii.gz"):
            parsed = urlparse(entry["source_url"])
            query = parse_qs(parsed.query, strict_parsing=True)
            pointer = base64.b64decode(entry["pointer_base64"], validate=True)
            expected_key = f"MD5E-s{entry['bytes']}--{entry['expected_md5']}.nii.gz"
            if (parsed._replace(query="").geturl() != S3 + path or set(query) != {"versionId"}
                    or len(query["versionId"]) != 1 or not query["versionId"][0]
                    or not re.fullmatch(r"[0-9a-f]{32}", entry["expected_md5"])
                    or entry["bytes"] > MAX_IMAGE_BYTES or entry["sha256"] is not None
                    or hashlib.sha256(pointer).hexdigest() != entry["pointer_sha256"]
                    or not pointer.decode().strip().endswith(expected_key)):
                raise AcquisitionError("Invalid pinned image source/version/pointer")
        else:
            payload = base64.b64decode(entry["metadata_base64"], validate=True)
            if (entry["source_url"] != RAW + path or len(payload) != entry["bytes"] or len(payload) > 65536
                    or hashlib.sha256(payload).hexdigest() != entry["sha256"]
                    or not {"ORIGINAL", "PRIMARY"}.issubset(json.loads(payload).get("ImageType", []))):
                raise AcquisitionError("Invalid original source sidecar")


def build_index() -> None:
    sessions = train_sessions()
    if INDEX.exists():
        validated_index()
        print(json.dumps({"status": "existing_index_verified", "sha256": sha256_file(INDEX)}))
        return
    cached = {}
    pending = []
    for identity in sessions:
        path = CACHE / "source-index" / ("_".join(identity) + ".json")
        if path.exists():
            record = json.loads(path.read_bytes())
            validate_session(record, identity)
            cached[identity] = record
        else:
            pending.append(identity)
    failures = []
    # Network reads only in workers; the owner serializes local evidence writes.
    with ThreadPoolExecutor(max_workers=4) as executor:
        jobs = {executor.submit(resolve_session, *identity): identity for identity in pending}
        for future in as_completed(jobs):
            identity = jobs[future]
            try:
                record = future.result()
                validate_session(record, identity)
                preserve(CACHE / "source-index" / ("_".join(identity) + ".json"), json_bytes(record))
                cached[identity] = record
            except Exception as error:
                failure = {"subject": identity[0], "session": identity[1],
                           "type": type(error).__name__, "reason": str(error)}
                failures.append(failure)
                stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
                preserve(CACHE / "attempts" / ("index-error-" + stamp + ".json"), json_bytes(failure))
            if (len(cached) + len(failures)) % 10 == 0:
                print(json.dumps({"source_indexed_sessions": len(cached), "failures_this_attempt": len(failures),
                                  "expected_sessions": len(sessions)}), flush=True)
    if failures:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        preserve(CACHE / "attempts" / ("index-failures-" + stamp + ".json"), json_bytes({"failures": failures}))
        raise AcquisitionError(f"{len(failures)} sessions unresolved; verified source records retained for resume")
    total = sum(entry["bytes"] for record in cached.values() for entry in record["files"])
    if total > MAX_TOTAL_BYTES:
        raise AcquisitionError("TRAIN source volume exceeds predeclared 20-GiB allowance")
    preserve(INDEX, json_bytes({"schema": "lausanne-train-originals-v1", "cohort_sha256": COHORT_SHA,
        "git_commit": COMMIT, "license": "CC0", "declared_at": datetime.now(timezone.utc).isoformat(),
        "subjects": 199, "session_count": 210, "bytes": total,
        "sessions": [cached[identity] for identity in sessions],
        "purpose": "Original anatomy intake for component work; no training or spatial-planning admission",
        "recorded_rl_transitions": 0}))
    print(json.dumps({"status": "full_train_source_index_frozen", "sessions": 210, "bytes": total,
                      "index_sha256": sha256_file(INDEX)}), flush=True)


def validated_index() -> dict:
    sessions = train_sessions()
    record = json.loads(INDEX.read_bytes())
    if (record.get("schema") != "lausanne-train-originals-v1" or record.get("subjects") != 199
            or record.get("session_count") != 210 or record["cohort_sha256"] != COHORT_SHA
            or record["git_commit"] != COMMIT or record["license"] != "CC0"
            or [(r["subject"], r["session"]) for r in record["sessions"]] != sessions):
        raise AcquisitionError("Index is not the full frozen TRAIN partition")
    for row, identity in zip(record["sessions"], sessions, strict=True):
        validate_session(row, identity)
    total = sum(entry["bytes"] for row in record["sessions"] for entry in row["files"])
    if total != record["bytes"] or total > MAX_TOTAL_BYTES:
        raise AcquisitionError("Index total is inconsistent or exceeds limit")
    return record


def acquire_session(identity: str, index_sha: str, output: Path, source_record: Path,
                    *, existing_only: bool = False) -> None:
    from resectionlab.imaging import inspect_nifti
    import nibabel as nib
    import numpy as np

    source = json.loads(source_record.read_bytes())
    verify_execution_source(source)
    if sha256_file(INDEX) != index_sha:
        raise AcquisitionError("Worker source index changed")
    index = validated_index()
    matches = [row for row in index["sessions"] if row["subject"] + "/" + row["session"] == identity]
    if len(matches) != 1:
        raise AcquisitionError("Worker session is outside frozen TRAIN")
    row = matches[0]
    files = []
    qc = []
    started = time.monotonic()
    for entry in row["files"]:
        path = DATA / entry["path"]
        if existing_only:
            # Do not publish missing source sidecars or invoke the downloader.
            verify_source_file(path, entry)
            status = "existing_source_verified_no_acquisition"
        elif path.name.endswith(".json"):
            preserve(path, base64.b64decode(entry["metadata_base64"], validate=True))
            status = "pinned_metadata_published"
        else:
            status = acquire_file(entry, DATA)
        verify_file(path, entry)
        files.append({"path": entry["path"], "bytes": path.stat().st_size, "sha256": sha256_file(path), "status": status})
        print(json.dumps({"session": identity, "file": entry["path"], "status": status}), flush=True)
    for entry in row["files"]:
        if not entry["path"].endswith(".nii.gz"):
            continue
        item = {"path": entry["path"], "status": "unassessed"}
        try:
            path = DATA / entry["path"]
            header = inspect_nifti(path)
            if int(np.prod(header["shape"])) * 4 > 512 * 1024**2:
                raise AcquisitionError("Decoded image exceeds 512-MiB worker image allowance")
            values = nib.load(path).get_fdata(dtype=np.float32)
            if not np.isfinite(values).all():
                raise AcquisitionError("Nonfinite source intensities")
            item.update(status="header_and_scalar_checks_passed", header=header,
                        minimum=float(values.min()), maximum=float(values.max()), voxels=int(values.size))
            del values
        except TimeoutError:
            raise
        except (ValueError, OSError, AcquisitionError) as error:
            item.update(status="qc_failed", reason=str(error))
        qc.append(item)
    verify_execution_source(source)
    if sha256_file(INDEX) != index_sha:
        raise AcquisitionError("Worker source index changed during execution")
    preserve(output, json_bytes({"subject": row["subject"], "session": row["session"], "role": "TRAIN",
        "status": "acquired", "index_sha256": index_sha, "files": files, "integrity_qc": qc,
        "execution_source_sha256": hashlib.sha256(json_bytes(source)).hexdigest(),
        "execution_source_record": str(source_record.relative_to(ROOT)),
        "existing_files_only": existing_only,
        "elapsed_seconds": time.monotonic() - started, "acquired_at": datetime.now(timezone.utc).isoformat(),
        "scanner_frame_admitted": False, "spatial_planning_admitted": False,
        "registration": "unverified", "anatomical_coverage": "unreviewed",
        "component_optimizer_updates": 0, "recorded_rl_transitions": 0}))


def execution_source() -> dict:
    import importlib.metadata
    return {"files": {name: sha256_file(ROOT / name) for name in SOURCE_NAMES},
            "python": sys.version, "numpy": importlib.metadata.version("numpy"),
            "nibabel": importlib.metadata.version("nibabel")}


def verify_execution_source(record: dict) -> None:
    if execution_source() != record:
        raise AcquisitionError("Intake execution source/runtime changed")


def retain_execution_source(run: Path, source: dict, *, deadline: float) -> str:
    for name, digest in source["files"].items():
        check_deadline(deadline)
        payload = (ROOT / name).read_bytes()
        if hashlib.sha256(payload).hexdigest() != digest:
            raise AcquisitionError("Source changed while retaining execution snapshot")
        preserve(run / "source-snapshot" / name, payload)
    payload = json_bytes(source)
    preserve(run / "source.json", payload)
    return hashlib.sha256(payload).hexdigest()


def validate_retained_source(saved: dict, *, deadline: float | None = None) -> None:
    """Resolve historical provenance without equating it to today's code."""
    reference = Path(saved.get("execution_source_record", ""))
    path = ROOT / reference
    if (reference.is_absolute() or ".." in reference.parts or path.name != "source.json"
            or path.is_symlink() or not path.resolve().is_relative_to((CACHE / "attempts").resolve())
            or path.stat().st_size > 128 * 1024):
        raise AcquisitionError("Invalid retained execution source reference")
    check_deadline(deadline)
    payload = path.read_bytes()
    source = json.loads(payload)
    if (payload != json_bytes(source) or hashlib.sha256(payload).hexdigest() != saved["execution_source_sha256"]
            or set(source.get("files", {})) != set(SOURCE_NAMES)
            or any(not isinstance(source.get(key), str) or not source[key] for key in ("python", "numpy", "nibabel"))):
        raise AcquisitionError("Retained execution source digest/schema mismatch")
    for name, digest in source["files"].items():
        snapshot = path.parent / "source-snapshot" / name
        if (not re.fullmatch(r"[0-9a-f]{64}", digest)
                or not snapshot.resolve().is_relative_to((path.parent / "source-snapshot").resolve())):
            raise AcquisitionError("Invalid retained source snapshot")
        verify_source_file(snapshot, {"bytes": snapshot.stat().st_size, "sha256": digest}, deadline=deadline)


def validate_receipt(row: dict, saved: dict, index_sha: str, *, deadline=None) -> str:
    if (saved.get("subject") != row["subject"] or saved.get("session") != row["session"]
            or saved.get("role") != "TRAIN" or saved.get("status") != "acquired"
            or saved.get("index_sha256") != index_sha or saved.get("scanner_frame_admitted") is not False
            or saved.get("spatial_planning_admitted") is not False
            or saved.get("registration") != "unverified" or saved.get("anatomical_coverage") != "unreviewed"
            or saved.get("component_optimizer_updates") != 0 or saved.get("recorded_rl_transitions") != 0
            or len(saved.get("files", [])) != len(row["files"])
            or not re.fullmatch(r"[0-9a-f]{64}", saved.get("execution_source_sha256", ""))):
        raise AcquisitionError("Acquisition receipt identity/status/claim mismatch")
    validate_retained_source(saved, deadline=deadline)
    for entry, receipt in zip(row["files"], saved["files"], strict=True):
        if receipt["path"] != entry["path"] or receipt["bytes"] != entry["bytes"]:
            raise AcquisitionError("Acquired source inventory differs")
        verify_source_file(DATA / entry["path"], entry, receipt_sha=receipt["sha256"], deadline=deadline)
    images = [entry["path"] for entry in row["files"] if entry["path"].endswith(".nii.gz")]
    if [item.get("path") for item in saved.get("integrity_qc", [])] != images:
        raise AcquisitionError("Receipt lacks exact two-image QC inventory")
    for item in saved["integrity_qc"]:
        if item["status"] == "qc_failed":
            if not isinstance(item.get("reason"), str) or not item["reason"]:
                raise AcquisitionError("QC failure has no recorded reason")
        elif item["status"] == "header_and_scalar_checks_passed":
            header = item["header"]
            if (header["file"] != Path(item["path"]).name or header["status"] != "passed_header_checks"
                    or header["physical_units"] != "mm" or len(header["shape"]) != 3
                    or any(type(size) is not int or size <= 0 for size in header["shape"])
                    or math.prod(header["shape"]) != item["voxels"]
                    or not all(math.isfinite(item[key]) for key in ("minimum", "maximum"))
                    or item["minimum"] > item["maximum"]):
                raise AcquisitionError("Inconsistent saved scalar/header QC")
        else:
            raise AcquisitionError("Unknown acquired QC status")
    return ("acquired_qc_failed" if any(item["status"] == "qc_failed" for item in saved["integrity_qc"])
            else "acquired_qc_passed")


def missing_payload_bytes(index: dict, *, deadline: float) -> int:
    missing = 0
    for row in index["sessions"]:
        check_deadline(deadline)
        for entry in row["files"]:
            path = DATA / entry["path"]
            if path.exists():
                continue  # Existing bytes must pass source checks; never overwritten.
            partial = path.with_name(path.name + ".partial")
            retained = min(partial.stat().st_size, entry["bytes"]) if partial.exists() else 0
            missing += entry["bytes"] - retained
    return missing


def batch(max_seconds: int, max_bytes: int, index_sha: str, *,
          only_session: str | None = None, existing_only: bool = False) -> None:
    started = time.monotonic()
    deadline = started + max_seconds
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run = CACHE / "attempts" / ("batch-" + stamp)
    run.mkdir(parents=True)
    source = None
    source_sha = None
    outcomes = []
    deferred = []
    verified_cached = []
    attempted_bytes = 0
    status = "failed"
    problem = None
    expected_sessions = []
    try:
        source = execution_source()
        source_sha = retain_execution_source(run, source, deadline=deadline)
        expected_sessions = [subject + "/" + session for subject, session in train_sessions()]
        preserve(run / "declaration.json", json_bytes({"index_sha256": index_sha,
            "cohort_sha256": COHORT_SHA, "execution_source_sha256": source_sha,
            "max_seconds": max_seconds, "max_attempted_source_bytes": max_bytes,
            "only_session": only_session, "existing_files_only": existing_only,
            "expected_sessions": expected_sessions,
            "timing": "includes source/cache verification; final receipt writing may overrun slightly"}))
        if only_session is not None and only_session not in expected_sessions:
            raise AcquisitionError("Explicit session restriction is outside frozen TRAIN")
        if existing_only and only_session is None:
            raise AcquisitionError("Existing-files-only replay requires an explicit TRAIN session")
        if not index_sha or sha256_file(INDEX) != index_sha:
            raise AcquisitionError("Explicit predeclared index SHA256 required")
        index = validated_index()
        needed = 0 if existing_only else missing_payload_bytes(index, deadline=deadline)
        if shutil.disk_usage(DATA).free < needed + 1024**3:
            raise AcquisitionError("Insufficient free space for remaining payload bytes plus 1 GiB")
        for row in index["sessions"]:
            check_deadline(deadline)
            identity = row["subject"] + "/" + row["session"]
            if only_session is not None and identity != only_session:
                deferred.append({"session": identity, "status": "outside_bounded_session_scope"})
                continue
            destination = CACHE / "acquired" / (row["subject"] + "_" + row["session"] + ".json")
            if destination.exists():
                saved = json.loads(destination.read_bytes())
                cached_status = validate_receipt(row, saved, index_sha, deadline=deadline)
                verified_cached.append({"session": identity, "status": cached_status})
                continue
            size = sum(entry["bytes"] for entry in row["files"])
            if attempted_bytes + size > max_bytes:
                deferred.append({"session": identity, "status": "deferred_byte_budget", "required_source_bytes": size})
                continue
            if deadline - time.monotonic() < 5:
                raise IntakeDeadline("Insufficient batch time to start another worker")
            verify_execution_source(source)
            attempted_bytes += size
            trial = run / (row["subject"] + "_" + row["session"])
            trial.mkdir()
            output, log = trial / "acquisition.json", trial / "worker.log"
            worker_seconds = min(600., max(1., deadline - time.monotonic() - 1.))
            command = [sys.executable, str(Path(__file__).resolve()), "worker", "--session", identity,
                       "--index-sha", index_sha, "--output", str(output), "--source-record", str(run / "source.json"),
                       "--worker-max-seconds", str(worker_seconds)]
            if existing_only:
                command.append("--existing-only")
            outcome = {"session": identity, "status": "interrupted", "exit_code": None,
                       "output": str(output.relative_to(ROOT)), "log": str(log.relative_to(ROOT))}
            print(json.dumps({"starting_session": identity, "source_bytes_budgeted": size}), flush=True)
            def on_start(pid):
                preserve(trial / "started.json", json_bytes({"session": identity, "worker_pid": pid,
                    "index_sha256": index_sha, "execution_source_sha256": source_sha,
                    "worker_max_seconds": worker_seconds, "started_at": datetime.now(timezone.utc).isoformat()}))
            try:
                worker_status, code = supervise(command, log, deadline=min(deadline, time.monotonic() + worker_seconds + .5), on_start=on_start)
                outcome.update(status=worker_status, exit_code=code)
                if worker_status == "completed":
                    saved = json.loads(output.read_bytes())
                    if saved.get("execution_source_sha256") != source_sha:
                        raise AcquisitionError("Worker receipt source binding differs")
                    outcome["status"] = validate_receipt(row, saved, index_sha, deadline=deadline)
                    preserve(destination, output.read_bytes())
                elif code == 124:
                    outcome["status"] = "timeout"
            except BaseException as error:
                outcome.update(status="interrupted" if isinstance(error, (KeyboardInterrupt, IntakeDeadline)) else "receipt_or_worker_error",
                               reason=f"{type(error).__name__}: {error}")
                raise
            finally:
                preserve(trial / "outcome.json", json_bytes(outcome))
                outcomes.append(outcome)
                print(json.dumps(outcome), flush=True)
        status = "bounded_batch_finished"
    except IntakeDeadline as error:
        status, problem = "time_budget_exhausted", str(error)
    except BaseException as error:
        status, problem = "failed_or_interrupted", f"{type(error).__name__}: {error}"
        raise
    finally:
        # Closure probes must not suppress the failure receipt they explain.
        closure_errors = []
        source_unchanged = index_unchanged = False
        try:
            source_unchanged = source is not None and execution_source() == source
        except Exception as error:
            closure_errors.append({"check": "execution_source", "reason": f"{type(error).__name__}: {error}"})
        try:
            index_unchanged = sha256_file(INDEX) == index_sha
        except Exception as error:
            closure_errors.append({"check": "index", "reason": f"{type(error).__name__}: {error}"})
        if not source_unchanged or not index_unchanged:
            status = "execution_binding_changed"
        complete = {entry["session"] for entry in verified_cached + outcomes if entry["status"].startswith("acquired_qc_")}
        preserve(run / "batch.json", json_bytes({"status": status, "problem": problem,
            "index_sha256": index_sha, "cohort_sha256": COHORT_SHA, "execution_source_sha256": source_sha,
            "source_unchanged": source_unchanged, "index_unchanged": index_unchanged,
            "closure_check_errors": closure_errors, "expected_inventory_resolved": len(expected_sessions) == 210,
            "max_seconds": max_seconds, "max_attempted_source_bytes": max_bytes,
            "only_session": only_session, "existing_files_only": existing_only,
            "elapsed_seconds": time.monotonic() - started, "attempted_source_bytes": attempted_bytes,
            "byte_accounting": "source-size budget, not measured network transfer", "outcomes": outcomes,
            "verified_cached": verified_cached, "deferred": deferred, "expected_session_count": 210,
            "not_verified_acquired_in_this_batch": [identity for identity in expected_sessions if identity not in complete]}))
        print(json.dumps({"batch_receipt": str((run / "batch.json").relative_to(ROOT)), "status": status,
                          "attempts": len(outcomes)}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["index", "batch", "worker"])
    parser.add_argument("--max-seconds", type=int, default=600)
    parser.add_argument("--max-bytes", type=int, default=256 * 1024**2)
    parser.add_argument("--session")
    parser.add_argument("--existing-only", action="store_true",
                        help="Verify existing files without acquisition; batch requires --session")
    parser.add_argument("--index-sha")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--source-record", type=Path)
    parser.add_argument("--worker-max-seconds", type=float, default=600.)
    args = parser.parse_args()
    if args.action == "index":
        build_index()
    elif args.action == "worker":
        if (not args.session or not args.index_sha or args.output is None or args.source_record is None
                or not math.isfinite(args.worker_max_seconds) or not 1 <= args.worker_max_seconds <= 600):
            parser.error("worker requires exact session, index hash, output, source binding and 1–600-second limit")
        watchdog = threading.Timer(args.worker_max_seconds, lambda: os._exit(124))
        watchdog.daemon = True
        watchdog.start()
        try:
            acquire_session(args.session, args.index_sha, args.output, args.source_record, existing_only=args.existing_only)
        finally:
            watchdog.cancel()
    else:
        if not 5 <= args.max_seconds <= 1800 or not 1 <= args.max_bytes <= 2 * 1024**3:
            parser.error("batch must fit 5–1800 seconds and at most 2 GiB attempted source bytes")
        if not args.index_sha:
            parser.error("batch requires the frozen index SHA256")
        if args.existing_only and not args.session:
            parser.error("existing-files-only replay requires an explicit --session")
        CACHE.mkdir(parents=True, exist_ok=True)
        with (CACHE / "batch.lock").open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with termination_cleanup():
                batch(args.max_seconds, args.max_bytes, args.index_sha,
                      only_session=args.session, existing_only=args.existing_only)


if __name__ == "__main__":
    main()
