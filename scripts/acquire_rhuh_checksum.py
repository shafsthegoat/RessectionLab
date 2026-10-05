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


class Rejected(Exception):
    """A fixed, credential-free failure reason suitable for a local receipt."""


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require_regular(path: Path) -> None:
    if path.is_symlink() or not path.is_file():
        raise Rejected("non_regular_local_dependency")


def local_checks() -> None:
    for path, expected in [
        (EXECUTABLE, EXECUTABLE_SHA256), (CLIENT_KEY, CLIENT_KEY_SHA256),
        (LICENSE, LICENSE_SHA256),
    ]:
        require_regular(path)
        if digest(path) != expected:
            raise Rejected("local_dependency_hash_mismatch")
    if not os.access(EXECUTABLE, os.X_OK):
        raise Rejected("reviewed_client_is_not_executable")
    if CLIENT_KEY.stat().st_mode & 0o077:
        raise Rejected("client_key_permissions_too_broad")
    if subprocess.run(
        ["codesign", "--verify", "--strict", str(EXECUTABLE)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10,
    ).returncode:
        raise Rejected("client_signature_invalid")
    # Every planned write must remain beneath a normal, ignored workspace path.
    for path in [QUARANTINE_ROOT, *QUARANTINE_ROOT.parents]:
        if path.is_symlink():
            raise Rejected("quarantine_parent_is_symlink")
        if path == ROOT:
            break
    checked = subprocess.run(
        ["git", "check-ignore", "--quiet", str(QUARANTINE_ROOT / "probe")],
        cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5,
    )
    if checked.returncode:
        raise Rejected("quarantine_not_git_ignored")


def remaining(deadline: float) -> float:
    seconds = deadline - time.monotonic()
    if seconds <= 0:
        raise Rejected("wall_time_limit")
    return min(seconds, 15.0)


def request(url: str, deadline: float, *, data=None, headers=None, limit=1_048_576):
    """No redirect following or retries; response and credentials stay in memory."""
    opener = urllib.request.build_opener(NoRedirect())
    req = urllib.request.Request(url, data=data, headers=headers or {})
    try:
        response = opener.open(req, timeout=remaining(deadline))
    except urllib.error.HTTPError as exc:
        response = exc
    with response:
        raw = response.read(limit + 1)
        if len(raw) > limit:
            raise Rejected("metadata_response_too_large")
        remaining(deadline)
        return response.status, response.headers, raw


def fresh_spec(deadline: float) -> dict:
    status, _, raw = request(COLLECTION, deadline)
    if status != 200:
        raise Rejected("collection_metadata_http_failure")
    links = {
        html.unescape(link)
        for link in re.findall(r"https://faspex[^\"\s<]+", raw.decode("utf-8"))
        if "context=" in link
    }
    if len(links) != 1:
        raise Rejected("public_package_link_not_unique")
    public = urllib.parse.urlsplit(links.pop())
    if public.scheme != "https" or public.netloc != "faspex.cancerimagingarchive.net":
        raise Rejected("unexpected_public_package_origin")
    contexts = urllib.parse.parse_qs(public.query).get("context", [])
    if len(contexts) != 1:
        raise Rejected("invalid_public_package_context")
    status, _, raw = request(BASE + "/config.js", deadline, limit=262_144)
    if status != 200:
        raise Rejected("public_configuration_http_failure")
    config = raw.decode("utf-8")
    client_match = re.search(r"client_id: '([^']+)'", config)
    redirect_match = re.search(r"redirect_uri: '([^']+)'", config)
    if client_match is None or redirect_match is None:
        raise Rejected("public_configuration_shape_changed")
    client, redirect = client_match.group(1), redirect_match.group(1)
    if redirect != "/aspera/faspex/token":
        raise Rejected("public_oauth_redirect_changed")
    query = urllib.parse.urlencode({
        "response_type": "code", "state": contexts[0],
        "client_id": client, "redirect_uri": redirect,
    })
    status, headers, _ = request(BASE + "/auth/authorize_public_link?" + query, deadline)
    if status != 302:
        raise Rejected("public_authorization_http_failure")
    location = urllib.parse.urlsplit(urllib.parse.urljoin(BASE, headers.get("Location", "")))
    if location.scheme != "https" or location.netloc != public.netloc:
        raise Rejected("authorization_redirect_origin_changed")
    codes = urllib.parse.parse_qs(location.query).get("code", [])
    if len(codes) != 1:
        raise Rejected("public_authorization_code_missing")
    data = urllib.parse.urlencode({
        "grant_type": "authorization_code", "code": codes[0],
        "client_id": client, "redirect_uri": redirect,
    }).encode()
    status, _, raw = request(BASE + "/auth/token", deadline, data=data, headers={
        "Content-Type": "application/x-www-form-urlencoded",
    })
    if status != 201:
        raise Rejected("public_token_http_failure")
    access_token = json.loads(raw).get("access_token")
    if not isinstance(access_token, str) or not access_token:
        raise Rejected("public_token_missing")
    status, _, raw = request(
        BASE + "/api/v5/packages/684/transfer_spec/download?transfer_type=connect&type=received",
        deadline, data=json.dumps({"paths": [{"path": SOURCE}]}).encode(),
        headers={"Authorization": "Bearer " + access_token, "Content-Type": "application/json"},
    )
    if status != 201:
        raise Rejected("checksum_spec_http_failure")
    return json.loads(raw)


def validate_spec(spec: dict) -> None:
    paths = spec.get("paths")
    if not isinstance(paths, list) or len(paths) != 1 or not isinstance(paths[0], dict):
        raise Rejected("checksum_spec_source_mismatch")
    source = paths[0].get("source")
    if (set(paths[0]) != {"source"} or not isinstance(source, str)
            or not source.startswith("/") or not source.endswith(SOURCE)
            or any(character in source for character in ["?", "\n", "\r"])
            or hashlib.sha256(source.encode()).hexdigest() != RESOLVED_SOURCE_SHA256):
        raise Rejected("checksum_spec_source_mismatch")
    if spec.get("direction") != "receive" or spec.get("authentication") != "token":
        raise Rejected("checksum_spec_authentication_or_direction_mismatch")
    if spec.get("remote_host") != HOST:
        raise Rejected("checksum_spec_host_changed")
    if spec.get("ssh_port") != 33001 or spec.get("fasp_port") != 33001:
        raise Rejected("checksum_spec_port_changed")
    if spec.get("cipher") != "aes-128":
        raise Rejected("checksum_spec_cipher_changed")
    if not re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", spec.get("remote_user", "")):
        raise Rejected("checksum_spec_remote_user_invalid")
    if not isinstance(spec.get("token"), str) or not spec["token"]:
        raise Rejected("checksum_transfer_token_missing")
    if not isinstance(spec.get("cookie", ""), str):
        raise Rejected("checksum_spec_cookie_invalid")
    if spec.get("source_root") or spec.get("multi_session", 0) not in (0, 1, None):
        raise Rejected("checksum_spec_requires_unreviewed_transfer_mode")


def transfer_invocation(spec: dict, destination: Path):
    validate_spec(spec)
    argv = [
        str(EXECUTABLE), "--mode=recv", "--host=" + HOST,
        "--user=" + spec["remote_user"], "-P", "33001", "-O", "33001",
        "-i", str(CLIENT_KEY), "--policy=fair", "-l", "10000", "-m", "0",
        "-c", "aes-128", "-k", "0", "--overwrite=never", "-L-", "-q",
        spec["paths"][0]["source"], str(destination),
    ]
    # Do not inherit other Aspera credentials, proxies, SSH agents, or debug flags.
    env = {key: os.environ[key] for key in ["PATH", "LANG", "LC_ALL", "TMPDIR"] if key in os.environ}
    env.update(ASPERA_SCP_TOKEN=spec["token"], ASPERA_SCP_COOKIE=spec.get("cookie", ""))
    return argv, env


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
    """Bound the complete worker lifecycle, including checks and file writes.

    The worker and client share a process group. The supervisor always kills
    that group, including after an ordinary worker exit, then spends at most
    one second reaping its direct child. No worker/client stderr is forwarded.
    """
    deadline = started + WALL_SECONDS - 3
    worker = None
    result = {"status": "supervisor_failed", "images_requested": False}
    exit_code = 1
    try:
        worker = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "--execute", "--_worker"],
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        worker_group.append(worker.pid)
        allowance = deadline - time.monotonic()
        if allowance <= 0:
            raise subprocess.TimeoutExpired("checksum_worker", 0)
        output, _ = worker.communicate(timeout=allowance)
        if len(output) > 8192:
            raise Rejected("worker_summary_too_large")
        summary = json.loads(output)
        status = summary.get("status")
        if status not in {"failed", "local_check_rejected", "received_length_checked_local_hash_recorded"}:
            raise Rejected("worker_summary_invalid")
        result["status"] = status
        if "receipt" in summary:
            relative = Path(summary["receipt"])
            expected_parent = QUARANTINE_ROOT.relative_to(ROOT)
            if relative.parent != expected_parent or not relative.name.endswith("-receipt.json"):
                raise Rejected("worker_receipt_path_invalid")
            result["receipt"] = str(relative)
        if worker.returncode == 0 and status == "received_length_checked_local_hash_recorded":
            exit_code = 0
    except subprocess.TimeoutExpired:
        result["status"] = "supervisor_timeout_partial_files_unverified"
    except Exception as exc:
        result["status"] = "supervisor_failed_" + type(exc).__name__
    finally:
        if worker is not None:
            # Do not gate group cleanup on poll(): descendants can survive
            # an already exited leader and must still be terminated.
            try:
                os.killpg(worker.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            else:
                if exit_code == 0:
                    # communicate() already reaped a successful worker. A
                    # remaining group means possible descendant writers;
                    # SIGKILL delivery is asynchronous, so reject success.
                    result["status"] = "surviving_descendants_payload_unaccepted"
                    exit_code = 1
            try:
                worker.wait(timeout=min(1.0, max(0.01, started + WALL_SECONDS - time.monotonic())))
            except subprocess.TimeoutExpired:
                result["status"] = "worker_kill_requested_reap_unconfirmed"
                exit_code = 1
            if worker.stdout is not None:
                worker.stdout.close()
    if exit_code == 0:
        try:
            if time.monotonic() >= started + WALL_SECONDS - 1:
                raise Rejected("insufficient_time_for_final_reconciliation")
            result["final_payload"] = reconcile_worker_receipt(ROOT / result["receipt"])
        except Exception as exc:
            result["status"] = "final_reconciliation_failed_" + type(exc).__name__
            exit_code = 1
    result["elapsed_seconds"] = round(time.monotonic() - started, 6)
    if result["elapsed_seconds"] >= WALL_SECONDS:
        result["status"] = "wall_limit_exceeded_not_accepted"
        exit_code = 1
    result["accepted"] = exit_code == 0
    print(json.dumps(result))
    return exit_code


def supervise(started: float) -> int:
    """Hard outer watchdog includes final bounded receipt/payload readback."""
    worker_group = []

    def hard_stop(signum, frame):
        if worker_group:
            try:
                os.killpg(worker_group[0], signal.SIGKILL)
            except ProcessLookupError:
                pass
        # Exit 124 is an unaccepted timeout, never successful acquisition.
        # Do not format exceptions or flush buffers that could delay exit.
        os._exit(124)

    previous = signal.signal(signal.SIGALRM, hard_stop)
    signal.setitimer(signal.ITIMER_REAL, max(0.001, started + WALL_SECONDS - 1 - time.monotonic()))
    try:
        return _supervise(started, worker_group)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


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
