#!/usr/bin/env python3
"""Preparation or separately released one-image compressed-byte experiment.

Default invocation is local preflight only. --execute is reserved for a later
root release. No decoding, alternate digest sweep, retry, or scientific-use
promotion exists. Compatible compressed bytes remain quarantined.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import stat
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts import rhuh_public_transfer as public_transfer

Rejected = public_transfer.Rejected
SOURCE = "/RHUH-GBM_nii_v1/RHUH-0001/0/RHUH-0001_0_t1.nii.gz"
PUBLISHED_TOKEN = "a5c579950d0431d04928e5b59e91ce9d"
RESOLVED_SOURCE_SHA256 = "d88bfbe72422b010c503aeafe777daf5d74847d46f98e6c86662ce201888686d"
PROPOSAL = ROOT / "artifacts/rhuh-single-image-acquisition-proposal-v2/proposal.json"
PROPOSAL_SHA256 = "d1d8fea9771865bf33f282dfa5edfbd217021e01c7cd3212a513e5db9e8da9b2"
INDEX = ROOT / "outputs/rhuh-one-pair/metadata/checksum-quarantine/run-1jjo0q5c/RHUH-GBM-nii-v1.sums"
INDEX_SHA256 = "7c6ba3fa767bf169679db30caa854e671796416ca365fac6e57662796177c130"
MAX_BYTES = 64 * 1024 * 1024
EXPECTED_BYTES = None  # Unknown before transfer; never a prior-length-match claim.
WALL_SECONDS = 60
SUCCESS = "compressed_digest_compatible_quarantined"
QUARANTINE_ROOT = ROOT / "outputs/rhuh-single-image-v2/quarantine"
request = public_transfer.request


class CompatibilityMismatch(Rejected):
    def __init__(self, observation):
        super().__init__("candidate_compressed_md5_mismatch")
        self.observation = observation


def provenance_checks() -> None:
    for path in [PROPOSAL, INDEX]:
        if any(parent.is_symlink() for parent in [path, *path.parents]):
            raise Rejected("provenance_path_is_symlink")
    raw = public_transfer.bounded_regular_bytes(PROPOSAL, 16_384)
    if hashlib.sha256(raw).hexdigest() != PROPOSAL_SHA256:
        raise Rejected("proposal_hash_mismatch")
    proposal = json.loads(raw)
    selection = proposal.get("selection", {})
    if (selection.get("public_source_path") != SOURCE
            or selection.get("published_checksum_token") != PUBLISHED_TOKEN
            or selection.get("patient") != "RHUH-0001" or selection.get("visit") != 0
            or selection.get("modality_source_label") != "T1"
            or selection.get("checksum_index_sha256") != INDEX_SHA256):
        raise Rejected("proposal_selection_mismatch")
    raw = public_transfer.bounded_regular_bytes(INDEX, 65_536)
    if hashlib.sha256(raw).hexdigest() != INDEX_SHA256:
        raise Rejected("checksum_index_hash_mismatch")
    rows = [line.split() for line in raw.decode("ascii").splitlines()]
    if len(rows) != 720 or any(len(row) != 2 or not re.fullmatch("[0-9a-f]{32}", row[0]) for row in rows):
        raise Rejected("checksum_index_format_mismatch")
    matched = [row for row in rows if row[1] == SOURCE.lstrip("/")]
    if matched != [[PUBLISHED_TOKEN, SOURCE.lstrip("/")]]:
        raise Rejected("exact_source_checksum_binding_mismatch")


def local_checks() -> None:
    provenance_checks()
    public_transfer.local_client_checks(root=ROOT, quarantine_root=QUARANTINE_ROOT,
                                        **public_transfer.reviewed_client(ROOT))


def fresh_spec(deadline: float) -> dict:
    # Binding is also checked here so direct metadata preparation cannot bypass it.
    provenance_checks()
    return public_transfer.fresh_spec(SOURCE, deadline, request_fn=request, failure_prefix="image")


def validate_spec(spec: dict) -> None:
    public_transfer.validate_spec(spec, public_source=SOURCE,
        resolved_sha256=RESOLVED_SOURCE_SHA256, failure_prefix="image")


def transfer_invocation(spec: dict, destination: Path):
    validate_spec(spec)
    client = public_transfer.reviewed_client(ROOT)
    return public_transfer.transfer_invocation(spec, destination,
        executable=client["executable"], client_key=client["client_key"])


def child_limits():
    resource.setrlimit(resource.RLIMIT_FSIZE, (MAX_BYTES, MAX_BYTES))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def verify_payload(directory: Path) -> dict:
    """Bound and hash original bytes only; no gzip/NIfTI parsing or allocation."""
    if directory.is_symlink():
        raise Rejected("quarantine_directory_is_symlink")
    payload = directory / Path(SOURCE).name
    if list(directory.iterdir()) != [payload]:
        raise Rejected("unexpected_quarantine_contents")
    sha = hashlib.sha256()
    candidate = hashlib.md5(usedforsecurity=False)
    total = 0
    with os.fdopen(os.open(payload, os.O_RDONLY | os.O_NOFOLLOW), "rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or not 0 < info.st_size <= MAX_BYTES:
            raise Rejected("compressed_payload_size_or_type_rejected")
        while chunk := stream.read(min(1_048_576, MAX_BYTES - total + 1)):
            total += len(chunk)
            if total > MAX_BYTES:
                raise Rejected("compressed_payload_cap_exceeded")
            sha.update(chunk)
            candidate.update(chunk)
        if total != info.st_size:
            raise Rejected("compressed_payload_changed_during_read")
    record = {
        "path": str(payload.relative_to(ROOT)), "measured_compressed_bytes": total,
        "expected_compressed_bytes": None, "prior_length_match_claim": False,
        "sha256": sha.hexdigest(), "candidate_md5_of_original_compressed_bytes": candidate.hexdigest(),
        "published_checksum_token": PUBLISHED_TOKEN,
        "candidate_matches_published_token": candidate.hexdigest() == PUBLISHED_TOKEN,
        "publisher_algorithm_confirmed": False, "acceptance_scope": "format_byte_domain_compatibility_only",
        "decoded": False, "scientific_use_released": False,
    }
    if not record["candidate_matches_published_token"]:
        raise CompatibilityMismatch(record)
    return record


def reconcile_worker_receipt(receipt_path: Path) -> dict:
    receipt = json.loads(public_transfer.bounded_regular_bytes(receipt_path, 16_384))
    directory = receipt_path.parent / receipt_path.name.removesuffix("-receipt.json")
    final = verify_payload(directory)
    if (receipt.get("payload") != final or receipt.get("source") != SOURCE
            or receipt.get("status") != SUCCESS or receipt.get("proposal_sha256") != PROPOSAL_SHA256
            or receipt.get("checksum_index_sha256") != INDEX_SHA256
            or receipt.get("supervisor_acceptance_required") is not True):
        raise Rejected("final_compressed_payload_reconciliation_failed")
    return final


def execute() -> tuple[dict, Path]:
    """Worker-only action; the outer supervisor owns its entire lifecycle."""
    started = time.monotonic()
    QUARANTINE_ROOT.mkdir(parents=True, exist_ok=True, mode=0o700)
    directory = Path(tempfile.mkdtemp(prefix="run-", dir=QUARANTINE_ROOT))
    receipt = {
        "source": SOURCE, "proposal_sha256": PROPOSAL_SHA256, "checksum_index_sha256": INDEX_SHA256,
        "resolved_source_sha256": RESOLVED_SOURCE_SHA256, "started_utc": datetime.now(timezone.utc).isoformat(),
        "supervisor_acceptance_required": True, "status": "failed", "transfer_started": False,
        "max_retained_compressed_bytes": MAX_BYTES, "max_client_invocations": 1,
        "expected_compressed_bytes": None, "publisher_algorithm_confirmed": False,
        "decoded": False, "scientific_use_released": False,
    }
    process = None
    try:
        spec = fresh_spec(started + WALL_SECONDS - 3)
        argv, env = transfer_invocation(spec, directory / Path(SOURCE).name)
        public_transfer.remaining(started + WALL_SECONDS - 3)
        process = subprocess.Popen(argv, env=env, cwd=directory, stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            start_new_session=False, preexec_fn=child_limits)
        receipt["transfer_started"] = True
        code = process.wait(timeout=max(0.01, started + WALL_SECONDS - 3 - time.monotonic()))
        receipt["client_exit_code"] = code
        if code:
            raise Rejected("client_failed_no_retry")
        receipt["payload"] = verify_payload(directory)
        receipt["status"] = SUCCESS
    except CompatibilityMismatch as exc:
        receipt["failure"] = str(exc)
        receipt["observed_unaccepted_payload"] = exc.observation
    except Rejected as exc:
        receipt["failure"] = str(exc)
    except Exception as exc:
        receipt["failure"] = "redacted_exception_" + type(exc).__name__
    finally:
        if process is not None and process.poll() is None:
            process.kill()
            process.wait(timeout=1)
        receipt["elapsed_seconds"] = round(time.monotonic() - started, 6)
    path = QUARANTINE_ROOT / (directory.name + "-receipt.json")
    path.write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt, path


def run_worker() -> int:
    try:
        local_checks()
    except Exception as exc:
        print(json.dumps({"status": "local_check_rejected", "failure_type": type(exc).__name__}))
        return 2
    receipt, path = execute()
    print(json.dumps({"status": receipt["status"], "receipt": str(path.relative_to(ROOT))}))
    return 0 if receipt["status"] == SUCCESS else 1


def _supervise(started: float, worker_group: list[int]) -> int:
    return public_transfer.supervise_worker(started, worker_group, root=ROOT,
        quarantine_root=QUARANTINE_ROOT,
        worker_argv=[sys.executable, str(Path(__file__).resolve()), "--execute", "--_worker"],
        reconcile=reconcile_worker_receipt, success_status=SUCCESS,
        wall_seconds=WALL_SECONDS, images_requested=True)


def supervise(started: float) -> int:
    return public_transfer.hard_watchdog(started, _supervise, wall_seconds=WALL_SECONDS)


def main() -> int:
    started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="Use only after separate root execution release.")
    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args._worker:
        if not args.execute:
            parser.error("internal worker requires the execution action")
        return run_worker()
    if args.execute:
        return supervise(started)
    try:
        local_checks()
    except Exception as exc:
        print(json.dumps({"status": "local_check_rejected", "failure_type": type(exc).__name__}))
        return 2
    print(json.dumps({"status": "preflight_passed_no_network_no_transfer_no_decode", "source": SOURCE}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
