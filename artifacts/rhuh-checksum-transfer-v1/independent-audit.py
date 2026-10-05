#!/usr/bin/env python3
"""Saved-byte audit only: no network, transfer, image decoder or outcome input."""
from collections import Counter
from datetime import datetime
import ast
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
REL = OUT.relative_to(ROOT).as_posix()
EXPECTED_PAYLOAD_SHA = "7c6ba3fa767bf169679db30caa854e671796416ca365fac6e57662796177c130"
EXPECTED_SOURCE_SHA = "f9ad9f4d64cab2314be865aad47e4300e23aa8b83457ee83656a6921e12edaf0"
BOUND_INPUTS = {}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def read(relative):
    path = ROOT / relative
    assert not path.is_symlink() and stat.S_ISREG(path.stat().st_mode)
    raw = path.read_bytes()
    BOUND_INPUTS[relative] = {"bytes": len(raw), "sha256": sha(raw)}
    return raw


def load(relative):
    return json.loads(read(relative))


def git_bytes(commit, relative):
    return subprocess.run(
        ["git", "show", f"{commit}:{relative}"], cwd=ROOT,
        check=True, capture_output=True, timeout=10,
    ).stdout


def safe_relative(name):
    path = PurePosixPath(name)
    assert not path.is_absolute() and path.as_posix() == name
    assert all(part not in ("", ".", "..") for part in name.split("/"))
    assert "\\" not in name and ":" not in name and "%" not in name
    assert not any(ord(character) < 32 for character in name)
    return path


def audit():
    parent = load(REL + "/parent-execution.json")
    release_raw = read(REL + "/release.json")
    release = json.loads(release_raw)
    started = load(REL + "/execution-started.json")
    preflight = load(REL + "/preflight.json")
    assert parent["release_sha256"] == sha(release_raw)
    assert release["execution_authorized"] is True
    assert release["patient_image_requests_allowed"] is False
    assert release["image_acquisition_released"] is False
    assert release["publisher_digest_for_checksum_file"] is None
    assert release["maximum_client_attempts"] == started["maximum_attempts"] == 1
    assert release["maximum_wall_seconds"] == 60 and release["outer_watchdog_seconds"] == 59
    assert release["maximum_file_bytes"] == 65536 and release["expected_payload_bytes"] == 61787
    assert release["source_sha256"] == started["source_sha256"] == parent["source_sha256_after"] == EXPECTED_SOURCE_SHA
    assert preflight["exit_code"] == 0 and preflight["release_commit"] == started["release_commit"]
    assert preflight["summary"]["status"] == "local_checks_passed_no_network_or_transfer"
    assert release_raw == git_bytes(started["release_commit"], REL + "/release.json")
    assert datetime.fromisoformat(release["released_utc"]) < datetime.fromisoformat(started["started_utc"])

    archive_raw = read(REL + "/source.tar.gz")
    assert sha(archive_raw) == release["source_archive_sha256"]
    archive_bindings = {}
    with tarfile.open(fileobj=io.BytesIO(archive_raw), mode="r:gz") as archive:
        names = [member.name for member in archive.getmembers()]
        assert len(names) == len(set(names))
        for member in archive.getmembers():
            safe_relative(member.name.rstrip("/"))
            assert member.isfile() or member.isdir()
            if member.isfile():
                raw = archive.extractfile(member).read()
                assert raw == git_bytes(release["source_checkpoint"], member.name)
                assert raw == read(member.name)
                archive_bindings[member.name] = sha(raw)
    assert len(archive_bindings) == release["archived_file_count"] == 22
    assert archive_bindings[release["source_file"]] == EXPECTED_SOURCE_SHA
    assert archive_bindings["artifacts/rhuh-checksum-transfer-preparation-v1/preparation-manifest.json"] == release["preparation_manifest_sha256"]
    assert archive_bindings["artifacts/rhuh-checksum-transfer-review-v1/verification.json"] == release["independent_review_sha256"]
    constants = {}
    for node in ast.parse(read(release["source_file"]).decode()).body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    constants[target.id] = node.value.value

    summary = parent["parent_summary"]
    assert parent["exit_code"] == 0 and summary["accepted"] is True
    assert summary["status"] == "received_length_checked_local_hash_recorded"
    assert summary["images_requested"] is False and parent["stderr_bytes"] == 0
    worker_raw = read(summary["receipt"])
    assert worker_raw == read(REL + "/worker-receipt.json")
    worker = json.loads(worker_raw)
    assert worker["supervisor_acceptance_required"] is True
    assert worker["status"] == summary["status"] and worker["client_exit_code"] == 0
    assert worker["transfer_started"] is True and worker["images_requested"] is False
    assert worker["attempt_limit"] == 1 and worker["max_bytes"] == 65536 and worker["max_wall_seconds"] == 60
    assert worker["client_sha256"] == constants["EXECUTABLE_SHA256"]
    assert worker["resolved_source_sha256"] == constants["RESOLVED_SOURCE_SHA256"]
    assert worker["source"] == release["public_source"] == constants["SOURCE"]
    assert worker["package_id"] == "684" and worker["publisher_checksum_for_checksum_file_known"] is False
    assert datetime.fromisoformat(started["started_utc"]) <= datetime.fromisoformat(worker["started_utc"])
    assert 0 < worker["elapsed_seconds"] <= summary["elapsed_seconds"] <= parent["elapsed_seconds_observed_by_launcher"] < 60
    payload_record = worker["payload"]
    assert payload_record == summary["final_payload"]
    payload = read(payload_record["path"])
    assert len(payload) == payload_record["bytes"] == 61787
    assert sha(payload) == payload_record["sha256"] == EXPECTED_PAYLOAD_SHA
    assert payload == read(REL + "/received-checksums.sums")
    directory = (ROOT / payload_record["path"]).parent
    assert sorted(item.name for item in directory.iterdir()) == [constants["SOURCE"].removeprefix("/")]

    lines = payload.decode("ascii").splitlines()
    parsed, separators, naming_exceptions = [], Counter(), []
    for number, line in enumerate(lines, 1):
        match = re.fullmatch(r"([0-9a-f]+)([ \t]+)([^ \t\r\n]+)", line)
        assert match is not None, f"Unexpected row format: {number}"
        token, separator, name = match.groups()
        path = safe_relative(name)
        assert len(path.parts) == 4 and path.parts[0] == "RHUH-GBM_nii_v1"
        assert re.fullmatch(r"RHUH-\d{4}", path.parts[1]) and path.parts[2] in ("0", "1", "2")
        assert name.endswith(".nii.gz")
        if not re.fullmatch(r"RHUH-GBM_nii_v1/(RHUH-\d{4})/([012])/\1_\2_(adc|flair|t1|t1ce|t2|segmentations)\.nii\.gz", name):
            naming_exceptions.append({"line": number, "path": name})
        parsed.append({"line": number, "path": path.as_posix(), "raw_digest_token": token, "digest_width": len(token)})
        separators[repr(separator)] += 1
    paths = Counter(row["path"] for row in parsed)
    widths = Counter(row["digest_width"] for row in parsed)
    assert len(parsed) == 720 and widths == {32: 720}
    assert max(paths.values()) == 1
    by_path = {row["path"]: row for row in parsed}
    pair_path = "artifacts/rhuh-one-pair-acquisition-preparation-v1/bounded-pair-manifest.json"
    pair_raw = read(pair_path)
    assert pair_raw == git_bytes(release["source_checkpoint"], pair_path)
    pair = json.loads(pair_raw)
    assert pair["case_id"] == "RHUH-0001" and pair["image_acquisition_released"] is False
    assert len(pair["files"]) == 12
    selected = []
    for item in pair["files"]:
        public_path = item["public_package_path"]
        assert public_path.startswith("/") and not public_path.startswith("//")
        normalized = safe_relative(public_path[1:]).as_posix()
        assert normalized in by_path
        assert item["patient_id"] == "RHUH-0001" and item["visit"] in (0, 1)
        assert item["expected_compressed_bytes"] is None
        assert item["source_checksum_algorithm"] is None and item["source_checksum"] is None
        selected.append({**by_path[normalized], "public_package_path": public_path,
                         "visit": item["visit"], "modality": item["modality"],
                         "source_checksum_algorithm": None, "compressed_bytes": None})
    assert len({row["path"] for row in selected}) == 12
    assert {row["modality"] for row in selected} == {"adc", "flair", "t1", "t1ce", "t2", "segmentations"}
    assert Counter(row["visit"] for row in selected) == {0: 6, 1: 6}

    for name, binding in BOUND_INPUTS.items():
        assert sha((ROOT / name).read_bytes()) == binding["sha256"], f"Input changed during audit: {name}"
    report = {
        "schema": "rhuh-checksum-saved-byte-audit-v1", "status": "passed_saved_record_consistency_checks",
        "audit_source_sha256": sha(Path(__file__).read_bytes()), "consulted_input_bindings": BOUND_INPUTS,
        "source_closure": {"archived_regular_files": 22, "all_match_committed_source_and_current_bytes": True,
                           "source_checkpoint": release["source_checkpoint"], "release_commit": started["release_commit"]},
        "transfer": {"recorded_attempts": 1, "parent_accepted": True, "exit_code": 0,
                     "launcher_seconds": parent["elapsed_seconds_observed_by_launcher"],
                     "parent_seconds": summary["elapsed_seconds"], "worker_seconds": worker["elapsed_seconds"],
                     "payload": payload_record, "publisher_digest_for_index_known": False},
        "index": {"rows": len(parsed), "fields_per_row": 2, "digest_width_histogram": dict(widths),
                  "all_tokens_lowercase_hex": True, "duplicate_paths": [], "unsafe_paths": [],
                  "duplicate_digest_tokens": sum(value - 1 for value in Counter(row["raw_digest_token"] for row in parsed).values()),
                  "separator_histogram": dict(separators), "algorithm_header_present": False,
                  "size_fields_present": False, "source_checksum_algorithm": None,
                  "checksum_byte_domain": None, "noncanonical_basename_rows": naming_exceptions,
                  "checksum_tokens_padded_or_modified": False},
        "selection": {"expected": 12, "present": 12, "missing": [], "rows": selected,
                      "compressed_sizes_known": 0, "total_compressed_image_bytes": None,
                      "image_acquisition_ready": False},
        "scope": ["Read only saved metadata, checksum bytes, source archive and local Git objects; no network, transfers, image decoding or clinical outcome lookup.",
                  "The 32-character hex format is consistent with several possible algorithms; width alone does not establish MD5 or any other algorithm or establish compressed versus decompressed byte domain. No digest padding or reinterpretation was performed.",
                  "Release, source and receipts are internally consistent with declared bounds. Saved records cannot independently reconstruct process-group cleanup, OS resource enforcement, network requests or absence of an unrecorded attempt.",
                  "The 12-image manifest remains unchanged and unreleased; actual compressed sizes and authoritative digest-algorithm evidence remain missing."],
        "checker_development_note": "The initial checker incorrectly required uniform patient_visit_modality basenames across all 720 rows. It stopped at row 45 RHUH-0035/2/segmentation.nii.gz. The corrected checker retains that safe, nonselected naming exception; all 12 selected exact names remain required. This was a checker assumption, not a transfer failure; no input bytes were changed."}
    output = OUT / "independent-audit.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "rows": 720, "selected_present": 12,
                      "digest_algorithm": None, "known_image_sizes": 0,
                      "receipt_sha256": sha(output.read_bytes())}))


if __name__ == "__main__":
    audit()
