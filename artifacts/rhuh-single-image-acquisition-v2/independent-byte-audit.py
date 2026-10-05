#!/usr/bin/env python3
"""Independently reconcile saved compressed bytes; never decode the image."""
import ast
from datetime import datetime, timedelta
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import stat
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
REL = OUT.relative_to(ROOT).as_posix()
SOURCE = "/RHUH-GBM_nii_v1/RHUH-0001/0/RHUH-0001_0_t1.nii.gz"
PAYLOAD = "outputs/rhuh-single-image-v2/quarantine/run-kxn_jecp/RHUH-0001_0_t1.nii.gz"
PAYLOAD_SHA = "b3b9fa69b87221062261b8b16fbddfdac2c678104b354ba371b069fd784e3625"
TOKEN = "a5c579950d0431d04928e5b59e91ce9d"
MAX_BYTES = 64 * 1024 * 1024
INPUTS = {}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def safe_relative(name):
    path = PurePosixPath(name)
    assert not path.is_absolute() and path.as_posix() == name
    assert all(part not in ("", ".", "..") for part in name.split("/"))
    assert "\\" not in name and not any(ord(character) < 32 for character in name)
    return path


def read(relative):
    safe_relative(relative)
    path = ROOT / relative
    assert not path.is_symlink() and path.is_file()
    raw = path.read_bytes()
    INPUTS[relative] = {"bytes": len(raw), "sha256": sha(raw)}
    return raw


def load(relative):
    return json.loads(read(relative))


def binding(relative):
    return {"path": relative, "sha256": INPUTS[relative]["sha256"]}


def git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, check=True,
                          capture_output=True, timeout=10).stdout


def file_identity(info):
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns, info.st_mode)


def hash_original():
    """Only sequential original-byte reads; no gzip/NIfTI library is imported."""
    path = ROOT / PAYLOAD
    assert all(not item.is_symlink() for item in [path, *path.parents])
    before = path.stat()
    assert stat.S_ISREG(before.st_mode) and stat.S_IMODE(before.st_mode) == 0o444
    assert 0 < before.st_size <= MAX_BYTES
    sha256, md5, total = hashlib.sha256(), hashlib.md5(usedforsecurity=False), 0
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW), "rb") as stream:
        assert file_identity(os.fstat(stream.fileno())) == file_identity(before)
        while chunk := stream.read(min(1024 * 1024, MAX_BYTES - total + 1)):
            total += len(chunk)
            assert total <= MAX_BYTES
            sha256.update(chunk)
            md5.update(chunk)
    assert total == before.st_size
    assert file_identity(path.stat()) == file_identity(before)
    return {"measured_compressed_bytes": total, "sha256": sha256.hexdigest(),
            "candidate_md5_of_original_compressed_bytes": md5.hexdigest()}, before


def audit():
    release_raw = read(REL + "/release.json")
    release = json.loads(release_raw)
    started = load(REL + "/execution-started.json")
    preflight = load(REL + "/preflight.json")
    parent = load(REL + "/parent-execution.json")
    summary_raw = read(REL + "/parent-summary.json")
    summary = json.loads(summary_raw)
    preservation = load(REL + "/original-preservation.json")
    assert parent["release_sha256"] == started["release_sha256"] == sha(release_raw)
    assert sha(summary_raw) == parent["parent_summary_sha256"] == "5853796ca7eba872e798bba56eadf02eb38e8fa7cd19c1cf6a7cd2c284ccabac"
    assert len(summary_raw) == parent["stdout_bytes"] and summary == parent["parent_summary"]
    assert release_raw == git("show", f'{started["release_commit"]}:{REL}/release.json')
    git("merge-base", "--is-ancestor", release["source_commit"], started["release_commit"])
    release_commit_time = datetime.fromisoformat(git("show", "-s", "--format=%cI", started["release_commit"]).decode().strip())
    execution_start = datetime.fromisoformat(started["started_utc"])
    declared_release = datetime.fromisoformat(release["released_utc"])
    # Git stores seconds, while the release JSON preserves subsecond time.
    # The exact JSON is already verified inside this commit above.
    assert declared_release < release_commit_time + timedelta(seconds=1)
    assert declared_release < execution_start and release_commit_time < execution_start
    assert release["execution_authorized"] is True
    assert release["image_decoding_released"] is False and release["scientific_use_released"] is False
    assert release["source"] == SOURCE and release["published_token"] == TOKEN
    assert release["expected_compressed_bytes"] is None and release["publisher_algorithm_confirmed"] is False
    assert release["max_retained_file_bytes"] == MAX_BYTES
    assert release["max_client_invocations"] == started["maximum_attempts"] == 1
    assert release["max_whole_lifecycle_seconds"] == 60 and release["parent_watchdog_seconds"] == 59
    assert release["role"] == "acquisition_compatibility_development"
    assert release["prior_cohort_outcomes_inspected"] is True and release["no_untouched_test_claim"] is True
    assert preflight["release_commit"] == started["release_commit"] and preflight["exit_code"] == 0
    assert preflight["summary"]["status"] == "preflight_passed_no_network_no_transfer_no_decode"

    archive_raw = read(REL + "/source.tar.gz")
    assert sha(archive_raw) == release["source_archive_sha256"]
    archived = {}
    # This opens the small SOURCE archive, never the compressed image as an archive.
    with tarfile.open(fileobj=io.BytesIO(archive_raw), mode="r:gz") as archive:
        names = [member.name for member in archive.getmembers()]
        assert len(names) == len(set(names))
        for member in archive.getmembers():
            safe_relative(member.name.rstrip("/"))
            assert member.isfile() or member.isdir()
            if member.isfile():
                raw = archive.extractfile(member).read()
                assert raw == git("show", f'{release["source_commit"]}:{member.name}')
                assert raw == read(member.name)
                archived[member.name] = sha(raw)
    assert len(archived) == release["source_archive_file_count"] == 23
    assert len(release["source_closure_sha256"]) == 7
    assert all(archived[name] == digest for name, digest in release["source_closure_sha256"].items())
    constants = {}
    image_source = "scripts/acquire_rhuh_single_image.py"
    for node in ast.parse(read(image_source).decode()).body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
            constants.update({target.id: node.value.value for target in node.targets if isinstance(target, ast.Name)})
    assert constants["SOURCE"] == SOURCE and constants["PUBLISHED_TOKEN"] == TOKEN

    prep_path = "artifacts/rhuh-single-image-preparation-v2/preparation-manifest.json"
    proposal_path = "artifacts/rhuh-single-image-acquisition-proposal-v2/proposal.json"
    review_path = "artifacts/rhuh-single-image-review-v2/verification.json"
    prep = load(prep_path)
    proposal = load(proposal_path)
    load(review_path)
    assert INPUTS[prep_path]["sha256"] == release["preparation_manifest_sha256"]
    assert INPUTS[review_path]["sha256"] == release["independent_review_sha256"]
    assert INPUTS[proposal_path]["sha256"] == constants["PROPOSAL_SHA256"] == release["proposal_sha256"]
    assert proposal["selection"]["public_source_path"] == SOURCE
    assert proposal["selection"]["published_checksum_token"] == TOKEN
    assert proposal["prospective_transfer"]["known_expected_compressed_bytes"] is None
    v1_binding = prep["bindings"]["unchanged_original_v1_manifest"]
    v1 = load(v1_binding["path"])
    assert INPUTS[v1_binding["path"]]["sha256"] == v1_binding["sha256"] == proposal["original_v1_manifest_sha256"]
    assert v1["image_acquisition_released"] is False and v1["acquisition_ready"] is False
    assert all(row["expected_compressed_bytes"] is None for row in v1["files"])
    index_binding = prep["bindings"]["checksum_index"]
    index = read(index_binding["path"])
    assert sha(index) == constants["INDEX_SHA256"] == release["index_sha256"] == index_binding["sha256"]
    rows = [line.split() for line in index.decode("ascii").splitlines()]
    assert len(rows) == 720
    assert [row for row in rows if len(row) == 2 and row[1] == SOURCE[1:]] == [[TOKEN, SOURCE[1:]]]
    mapping_binding = prep["bindings"]["metadata_spec_binding_receipt"]
    mapping = load(mapping_binding["path"])
    assert INPUTS[mapping_binding["path"]]["sha256"] == mapping_binding["sha256"]
    assert mapping["public_source_path"] == SOURCE and mapping["resolved_source_sha256"] == constants["RESOLVED_SOURCE_SHA256"]

    assert parent["exit_code"] == 0 and summary["accepted"] is True and parent["stderr_bytes"] == 0
    assert parent["source_closure_unchanged"] is True
    assert summary["status"] == "compressed_digest_compatible_quarantined" and summary["images_requested"] is True
    worker_raw = read(summary["receipt"])
    assert worker_raw == read(REL + "/worker-receipt.json")
    worker = json.loads(worker_raw)
    assert worker["status"] == summary["status"] and worker["client_exit_code"] == 0
    assert worker["transfer_started"] is True and worker["supervisor_acceptance_required"] is True
    assert worker["source"] == SOURCE and worker["resolved_source_sha256"] == constants["RESOLVED_SOURCE_SHA256"]
    assert worker["proposal_sha256"] == release["proposal_sha256"] and worker["checksum_index_sha256"] == release["index_sha256"]
    assert worker["max_retained_compressed_bytes"] == MAX_BYTES and worker["max_client_invocations"] == 1
    assert worker["expected_compressed_bytes"] is None and worker["publisher_algorithm_confirmed"] is False
    assert worker["decoded"] is False and worker["scientific_use_released"] is False
    assert execution_start <= datetime.fromisoformat(worker["started_utc"])
    assert 0 < worker["elapsed_seconds"] <= summary["elapsed_seconds"] <= parent["elapsed_seconds_observed_by_launcher"] < 60

    reported = summary["final_payload"]
    assert reported == worker["payload"] and reported["path"] == PAYLOAD
    assert reported["expected_compressed_bytes"] is None and reported["prior_length_match_claim"] is False
    assert reported["publisher_algorithm_confirmed"] is False and reported["decoded"] is False
    assert reported["scientific_use_released"] is False and reported["candidate_matches_published_token"] is True
    assert reported["acceptance_scope"] == "format_byte_domain_compatibility_only"
    actual, image_before = hash_original()
    assert actual == {key: reported[key] for key in actual}
    assert actual["measured_compressed_bytes"] == 6337221 and actual["sha256"] == PAYLOAD_SHA
    assert actual["candidate_md5_of_original_compressed_bytes"] == reported["published_checksum_token"] == TOKEN
    assert sorted(item.name for item in (ROOT / PAYLOAD).parent.iterdir()) == [Path(PAYLOAD).name]
    assert preservation["path"] == PAYLOAD and preservation["compressed_bytes"] == actual["measured_compressed_bytes"]
    assert preservation["sha256"] == preservation["bytes_after_sha256"] == actual["sha256"]
    assert preservation["mode_after"] == "0o444" and preservation["decoded"] is False

    for name, expected in INPUTS.items():
        assert sha((ROOT / name).read_bytes()) == expected["sha256"], f"Input drift: {name}"
    actual_after, image_after = hash_original()
    assert actual_after == actual and file_identity(image_before) == file_identity(image_after)
    reviewer = {"path": Path(__file__).relative_to(ROOT).as_posix(), "sha256": sha(Path(__file__).read_bytes())}
    acquisition_sources = {role: binding(path) for role, path in {
        "image_helper": image_source, "public_transfer": "scripts/rhuh_public_transfer.py",
        "checksum_helper": "scripts/acquire_rhuh_checksum.py"}.items()}
    report = {
        "schema": "resectionlab.rhuh-compressed-byte-reconciliation.v1", "accepted": True,
        "source": SOURCE, "proposal_sha256": release["proposal_sha256"],
        "checksum_index_sha256": release["index_sha256"], "resolved_source_sha256": constants["RESOLVED_SOURCE_SHA256"],
        "parent_summary": binding(REL + "/parent-summary.json"),
        "parent_execution": binding(REL + "/parent-execution.json"),
        "worker_receipt": binding(REL + "/worker-receipt.json"),
        "transfer_release": binding(REL + "/release.json"), "source_archive": binding(REL + "/source.tar.gz"),
        "reviewer_source": reviewer, "acquisition_source": acquisition_sources["image_helper"],
        "acquisition_sources": acquisition_sources, "payload": reported,
        "immutable_original": True, "original_mode": "0o444",
        "immutable_original_meaning": "Preserved original compressed bytes, read-only permissions and identical independently recomputed hashes before/after audit; not an OS immutable-file guarantee.",
        "publisher_algorithm_confirmed": False, "prior_length_match_claim": False,
        "decoded": False, "scientific_use_released": False,
        "independent_computation": actual,
        "history_checks": {"release_committed_before_execution": True, "source_commit": release["source_commit"],
            "release_commit": started["release_commit"], "archived_files_matching_git_and_current_bytes": 23,
            "code_and_test_closure_count": 7, "v1_manifest_unchanged_and_unreleased": True,
            "recorded_client_attempts": 1, "worker_seconds": worker["elapsed_seconds"],
            "parent_seconds": summary["elapsed_seconds"], "launcher_seconds": parent["elapsed_seconds_observed_by_launcher"]},
        "consulted_inputs": INPUTS,
        "checker_development_note": "The initial metadata assertion compared a subsecond release time with a whole-second Git timestamp and stopped before image hashing. The corrected check uses Git's one-second timestamp interval, the exact committed release bytes and both recorded times preceding execution. No source records or image bytes were changed.",
        "limits": ["Only saved public metadata, source/Git records and original compressed bytes were read. No network, extra transfer, image decompression, header, voxel or clinical-outcome inspection.",
                   "The candidate MD5 matches the published token for these compressed bytes. The publisher's algorithm remains undeclared; no prior-length, cryptographic-authenticity, patient identity, modality or anatomy claim follows.",
                   "Saved receipts and frozen code support the recorded lifecycle and bounds, but cannot independently reconstruct OS enforcement, process-group state or network traffic.",
                   "The original V1 12-file prerequisites remain unmet. Any image decode or scientific use requires a separate root release."]}
    target = OUT / "independent-byte-audit.json"
    target.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"accepted": True, **actual, "audit_source_sha256": reviewer["sha256"],
                      "audit_receipt_sha256": sha(target.read_bytes())}))


if __name__ == "__main__":
    audit()
