#!/usr/bin/env python3
"""One explicitly released, bounded RHUH public checksum transfer.

No images, retries, account credentials, shell interpolation, or saved transfer
specifications. Running without --execute performs local preparation checks only.
IBM's published token-client key is a common bootstrap, not TCIA authorization;
the provider must issue a fresh token for the sole allowed checksum source.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import html
import json
import os
from pathlib import Path
import re
import resource
import signal
import stat
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts import rhuh_public_transfer as public_transfer
BASE = "https://faspex.cancerimagingarchive.net/aspera/faspex"
COLLECTION = "https://www.cancerimagingarchive.net/collection/rhuh-gbm/"
SOURCE = "/RHUH-GBM-nii-v1.sums"
# Faspex resolves the public package-relative request to a server-side path.
# Pin its reviewed digest; a changed mapping requires another metadata review.
RESOLVED_SOURCE_SHA256 = "cca6151d701646f481c1eb723623bcd5f2abd1fd55859a24f1a12dae2de5cc12"
EXPECTED_BYTES = 61_787
MAX_BYTES = 65_536
WALL_SECONDS = 60
HOST = "144.30.235.113"
EXECUTABLE = ROOT / (
    "build/rhuh-transfer-client-inspection-v1/inspected/"
    "ibm-aspera-transfer-sdk-macos-arm64-1.1.9/bin/ascp"
)
EXECUTABLE_SHA256 = "0d73689f6a4d5b6f7ddd90a86459be0aa211b53d40c9e01b80bed5f44f793c4e"
LICENSE = EXECUTABLE.parent.parent / "etc/aspera-license"
LICENSE_SHA256 = "ad474b3ff9697cc675b38d49e0b3880674c214a403f1ee5572bb424eff66fa3e"
CLIENT_KEY = ROOT / "build/rhuh-checksum-transfer-preparation-v1/ibm-public-token-client-rsa.pem"
CLIENT_KEY_SHA256 = "4481cc956592f2caae6fe66be06a11f3ed69b88f2c9da4ae99e5c1297d48ed43"
QUARANTINE_ROOT = ROOT / "outputs/rhuh-one-pair/metadata/checksum-quarantine"


Rejected = public_transfer.Rejected


NoRedirect = public_transfer.NoRedirect


digest = public_transfer.digest


require_regular = public_transfer.require_regular


def local_checks() -> None:
    public_transfer.local_client_checks(root=ROOT, quarantine_root=QUARANTINE_ROOT,
        executable=EXECUTABLE, executable_sha256=EXECUTABLE_SHA256,
        client_key=CLIENT_KEY, client_key_sha256=CLIENT_KEY_SHA256,
        license_file=LICENSE, license_sha256=LICENSE_SHA256)


remaining = public_transfer.remaining


request = public_transfer.request


def fresh_spec(deadline: float) -> dict:
    return public_transfer.fresh_spec(SOURCE, deadline, request_fn=request)


def validate_spec(spec: dict) -> None:
    public_transfer.validate_spec(spec, public_source=SOURCE,
        resolved_sha256=RESOLVED_SOURCE_SHA256, remote_host=HOST)


def transfer_invocation(spec: dict, destination: Path):
    validate_spec(spec)
    return public_transfer.transfer_invocation(spec, destination,
        executable=EXECUTABLE, client_key=CLIENT_KEY, remote_host=HOST)


def child_limits():
    resource.setrlimit(resource.RLIMIT_FSIZE, (MAX_BYTES, MAX_BYTES))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def wall_alarm(signum, frame):
    raise Rejected("wall_time_limit")


def verify_payload(directory: Path) -> dict:
    if directory.is_symlink():
        raise Rejected("quarantine_directory_is_symlink")
    entries = list(directory.iterdir())
    payload = directory / SOURCE.lstrip("/")
    if entries != [payload] or not stat.S_ISREG(payload.lstat().st_mode):
        raise Rejected("unexpected_quarantine_contents")
    with os.fdopen(os.open(payload, os.O_RDONLY | os.O_NOFOLLOW), "rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size != EXPECTED_BYTES:
            raise Rejected("checksum_length_mismatch")
        content = stream.read(EXPECTED_BYTES + 1)
        if len(content) != EXPECTED_BYTES:
            raise Rejected("checksum_length_mismatch")
    return {"path": str(payload.relative_to(ROOT)), "bytes": EXPECTED_BYTES,
            "sha256": hashlib.sha256(content).hexdigest()}


def reconcile_worker_receipt(receipt_path: Path) -> dict:
    """Accept bytes only after no worker process-group members remain."""
    with os.fdopen(os.open(receipt_path, os.O_RDONLY | os.O_NOFOLLOW), "rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size > 8192:
            raise Rejected("worker_receipt_not_bounded_regular_file")
        raw = stream.read(8193)
        if len(raw) > 8192:
            raise Rejected("worker_receipt_not_bounded_regular_file")
    receipt = json.loads(raw)
    directory = receipt_path.parent / receipt_path.name.removesuffix("-receipt.json")
    final_payload = verify_payload(directory)
    if (receipt.get("payload") != final_payload or receipt.get("source") != SOURCE
            or receipt.get("status") != "received_length_checked_local_hash_recorded"):
        raise Rejected("worker_receipt_payload_reconciliation_failed")
    return final_payload


def execute() -> tuple[dict, Path]:
    started = time.monotonic()
    # Two seconds remain for kill/reap and the credential-free local receipt.
    deadline = started + WALL_SECONDS - 2
    QUARANTINE_ROOT.mkdir(parents=True, exist_ok=True, mode=0o700)
    directory = Path(tempfile.mkdtemp(prefix="run-", dir=QUARANTINE_ROOT))
    receipt = {"source": SOURCE, "package_id": "684", "images_requested": False,
               "supervisor_acceptance_required": True,
               "started_utc": datetime.now(timezone.utc).isoformat(),
               "client_sha256": EXECUTABLE_SHA256,
               "resolved_source_sha256": RESOLVED_SOURCE_SHA256,
               "attempt_limit": 1, "max_bytes": MAX_BYTES, "max_wall_seconds": WALL_SECONDS,
               "status": "failed", "transfer_started": False}
    process = None
    previous_alarm_handler = signal.signal(signal.SIGALRM, wall_alarm)
    signal.setitimer(signal.ITIMER_REAL, WALL_SECONDS - 2)
    try:
        spec = fresh_spec(deadline)
        argv, env = transfer_invocation(spec, directory / SOURCE.lstrip("/"))
        remaining(deadline)
        process = subprocess.Popen(
            argv, env=env, cwd=directory, stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            # Inherit the supervised worker's group so the parent can stop
            # every descendant even after this client leader has exited.
            start_new_session=False, preexec_fn=child_limits,
        )
        receipt["transfer_started"] = True
        try:
            code = process.wait(timeout=max(0.01, deadline - time.monotonic()))
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=1)
        receipt["client_exit_code"] = code
        if code != 0:
            raise Rejected("client_failed_no_retry")
        receipt["payload"] = verify_payload(directory)
        receipt["status"] = "received_length_checked_local_hash_recorded"
        receipt["publisher_checksum_for_checksum_file_known"] = False
    except Rejected as exc:
        receipt["failure"] = str(exc)
    except Exception as exc:
        # Exception messages can contain signed URLs or environment values.
        receipt["failure"] = "redacted_exception_" + type(exc).__name__
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_alarm_handler)
        if process is not None and process.poll() is None:
            process.kill()
            process.wait(timeout=1)
        receipt["elapsed_seconds"] = round(time.monotonic() - started, 6)
    receipt_path = QUARANTINE_ROOT / (directory.name + "-receipt.json")
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt, receipt_path


def run_worker() -> int:
    """All local checks, metadata, transfer, verification, and receipt writes."""
    try:
        local_checks()
    except Rejected as exc:
        print(json.dumps({"status": "local_check_rejected", "reason": str(exc)}))
        return 2
    receipt, receipt_path = execute()
    print(json.dumps({"status": receipt["status"], "receipt": str(receipt_path.relative_to(ROOT))}))
    return 0 if receipt["status"] != "failed" else 1


def _supervise(started: float, worker_group: list[int]) -> int:
    return public_transfer.supervise_worker(started, worker_group, root=ROOT,
        quarantine_root=QUARANTINE_ROOT,
        worker_argv=[sys.executable, str(Path(__file__).resolve()), "--execute", "--_worker"],
        reconcile=reconcile_worker_receipt,
        success_status="received_length_checked_local_hash_recorded", wall_seconds=WALL_SECONDS)


def supervise(started: float) -> int:
    return public_transfer.hard_watchdog(started, _supervise, wall_seconds=WALL_SECONDS)


def main() -> int:
    started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="Run the separately reviewed checksum-only transfer once.")
    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args._worker:
        if not args.execute:
            parser.error("internal worker requires the released execution action")
        return run_worker()
    if args.execute:
        return supervise(started)
    try:
        local_checks()
    except Rejected as exc:
        print(json.dumps({"status": "local_check_rejected", "reason": str(exc)}))
        return 2
    print(json.dumps({"status": "local_checks_passed_no_network_or_transfer", "source": SOURCE}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
