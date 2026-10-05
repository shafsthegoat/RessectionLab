"""Shared public RHUH protocol and bounded process supervision.

Callers supply immutable source bindings, worker entrypoints and final verifiers.
This module grants no acquisition release and performs no work on import.
"""
from __future__ import annotations
import hashlib
import html
import json
import os
from pathlib import Path
import re
import signal
import stat
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
BASE = "https://faspex.cancerimagingarchive.net/aspera/faspex"
COLLECTION = "https://www.cancerimagingarchive.net/collection/rhuh-gbm/"
HOST = "144.30.235.113"


def reviewed_client(root: Path) -> dict:
    """Pinned, previously inspected local runtime; no installation or execution."""
    base = root / "build/rhuh-transfer-client-inspection-v1/inspected/ibm-aspera-transfer-sdk-macos-arm64-1.1.9"
    return {
        "executable": base / "bin/ascp",
        "executable_sha256": "0d73689f6a4d5b6f7ddd90a86459be0aa211b53d40c9e01b80bed5f44f793c4e",
        "license_file": base / "etc/aspera-license",
        "license_sha256": "ad474b3ff9697cc675b38d49e0b3880674c214a403f1ee5572bb424eff66fa3e",
        "client_key": root / "build/rhuh-checksum-transfer-preparation-v1/ibm-public-token-client-rsa.pem",
        "client_key_sha256": "4481cc956592f2caae6fe66be06a11f3ed69b88f2c9da4ae99e5c1297d48ed43",
    }


def bounded_regular_bytes(path: Path, maximum: int) -> bytes:
    """Read small provenance/receipt files without symlinks or unbounded reads."""
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW), "rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size > maximum:
            raise Rejected("not_bounded_regular_file")
        raw = stream.read(maximum + 1)
        if len(raw) > maximum:
            raise Rejected("not_bounded_regular_file")
    return raw


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

def local_client_checks(*, root, quarantine_root, executable, executable_sha256, client_key, client_key_sha256, license_file, license_sha256) -> None:
    for path, expected in [
        (executable, executable_sha256), (client_key, client_key_sha256),
        (license_file, license_sha256),
    ]:
        require_regular(path)
        if digest(path) != expected:
            raise Rejected("local_dependency_hash_mismatch")
    if not os.access(executable, os.X_OK):
        raise Rejected("reviewed_client_is_not_executable")
    if client_key.stat().st_mode & 0o077:
        raise Rejected("client_key_permissions_too_broad")
    if subprocess.run(
        ["codesign", "--verify", "--strict", str(executable)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10,
    ).returncode:
        raise Rejected("client_signature_invalid")
    # Every planned write must remain beneath a normal, ignored workspace path.
    for path in [quarantine_root, *quarantine_root.parents]:
        if path.is_symlink():
            raise Rejected("quarantine_parent_is_symlink")
        if path == root:
            break
    checked = subprocess.run(
        ["git", "check-ignore", "--quiet", str(quarantine_root / "probe")],
        cwd=root, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5,
    )
    if checked.returncode:
        raise Rejected("quarantine_not_git_ignored")

def fresh_spec(public_source: str, deadline: float, *, request_fn=request, failure_prefix="checksum") -> dict:
    status, _, raw = request_fn(COLLECTION, deadline)
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
    status, _, raw = request_fn(BASE + "/config.js", deadline, limit=262_144)
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
    status, headers, _ = request_fn(BASE + "/auth/authorize_public_link?" + query, deadline)
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
    status, _, raw = request_fn(BASE + "/auth/token", deadline, data=data, headers={
        "Content-Type": "application/x-www-form-urlencoded",
    })
    if status != 201:
        raise Rejected("public_token_http_failure")
    access_token = json.loads(raw).get("access_token")
    if not isinstance(access_token, str) or not access_token:
        raise Rejected("public_token_missing")
    status, _, raw = request_fn(
        BASE + "/api/v5/packages/684/transfer_spec/download?transfer_type=connect&type=received",
        deadline, data=json.dumps({"paths": [{"path": public_source}]}).encode(),
        headers={"Authorization": "Bearer " + access_token, "Content-Type": "application/json"},
    )
    if status != 201:
        raise Rejected(f"{failure_prefix}_spec_http_failure")
    return json.loads(raw)

def validate_spec(spec: dict, *, public_source: str, resolved_sha256: str, remote_host=HOST, failure_prefix="checksum") -> None:
    paths = spec.get("paths")
    if not isinstance(paths, list) or len(paths) != 1 or not isinstance(paths[0], dict):
        raise Rejected(f"{failure_prefix}_spec_source_mismatch")
    source = paths[0].get("source")
    if (set(paths[0]) != {"source"} or not isinstance(source, str)
            or not source.startswith("/") or not source.endswith(public_source)
            or any(character in source for character in ["?", "\n", "\r"])
            or hashlib.sha256(source.encode()).hexdigest() != resolved_sha256):
        raise Rejected(f"{failure_prefix}_spec_source_mismatch")
    if spec.get("direction") != "receive" or spec.get("authentication") != "token":
        raise Rejected(f"{failure_prefix}_spec_authentication_or_direction_mismatch")
    if spec.get("remote_host") != remote_host:
        raise Rejected(f"{failure_prefix}_spec_host_changed")
    if spec.get("ssh_port") != 33001 or spec.get("fasp_port") != 33001:
        raise Rejected(f"{failure_prefix}_spec_port_changed")
    if spec.get("cipher") != "aes-128":
        raise Rejected(f"{failure_prefix}_spec_cipher_changed")
    if not re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", spec.get("remote_user", "")):
        raise Rejected(f"{failure_prefix}_spec_remote_user_invalid")
    if not isinstance(spec.get("token"), str) or not spec["token"]:
        raise Rejected(f"{failure_prefix}_transfer_token_missing")
    if not isinstance(spec.get("cookie", ""), str):
        raise Rejected(f"{failure_prefix}_spec_cookie_invalid")
    if spec.get("source_root") or spec.get("multi_session", 0) not in (0, 1, None):
        raise Rejected(f"{failure_prefix}_spec_requires_unreviewed_transfer_mode")

def transfer_invocation(spec: dict, destination: Path, *, executable: Path, client_key: Path, remote_host=HOST):
    argv = [
        str(executable), "--mode=recv", "--host=" + remote_host,
        "--user=" + spec["remote_user"], "-P", "33001", "-O", "33001",
        "-i", str(client_key), "--policy=fair", "-l", "10000", "-m", "0",
        "-c", "aes-128", "-k", "0", "--overwrite=never", "-L-", "-q",
        spec["paths"][0]["source"], str(destination),
    ]
    # Do not inherit other Aspera credentials, proxies, SSH agents, or debug flags.
    env = {key: os.environ[key] for key in ["PATH", "LANG", "LC_ALL", "TMPDIR"] if key in os.environ}
    env.update(ASPERA_SCP_TOKEN=spec["token"], ASPERA_SCP_COOKIE=spec.get("cookie", ""))
    return argv, env

def supervise_worker(started: float, worker_group: list[int], *, root: Path, quarantine_root: Path, worker_argv: list[str], reconcile, success_status: str, wall_seconds=60, images_requested=False) -> int:
    """Bound the complete worker lifecycle, including checks and file writes.

    The worker and client share a process group. The supervisor always kills
    that group, including after an ordinary worker exit, then spends at most
    one second reaping its direct child. No worker/client stderr is forwarded.
    """
    deadline = started + wall_seconds - 3
    worker = None
    result = {"status": "supervisor_failed", "images_requested": images_requested}
    exit_code = 1
    try:
        worker = subprocess.Popen(
            worker_argv,
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
        if status not in {"failed", "local_check_rejected", success_status}:
            raise Rejected("worker_summary_invalid")
        result["status"] = status
        if "receipt" in summary:
            relative = Path(summary["receipt"])
            expected_parent = quarantine_root.relative_to(root)
            if relative.parent != expected_parent or not relative.name.endswith("-receipt.json"):
                raise Rejected("worker_receipt_path_invalid")
            result["receipt"] = str(relative)
        if worker.returncode == 0 and status == success_status:
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
                worker.wait(timeout=min(1.0, max(0.01, started + wall_seconds - time.monotonic())))
            except subprocess.TimeoutExpired:
                result["status"] = "worker_kill_requested_reap_unconfirmed"
                exit_code = 1
            if worker.stdout is not None:
                worker.stdout.close()
    if exit_code == 0:
        try:
            if time.monotonic() >= started + wall_seconds - 1:
                raise Rejected("insufficient_time_for_final_reconciliation")
            result["final_payload"] = reconcile(root / result["receipt"])
        except Exception as exc:
            result["status"] = "final_reconciliation_failed_" + type(exc).__name__
            exit_code = 1
    result["elapsed_seconds"] = round(time.monotonic() - started, 6)
    if result["elapsed_seconds"] >= wall_seconds:
        result["status"] = "wall_limit_exceeded_not_accepted"
        exit_code = 1
    result["accepted"] = exit_code == 0
    print(json.dumps(result))
    return exit_code

def hard_watchdog(started: float, lifecycle, *, wall_seconds=60) -> int:
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
    signal.setitimer(signal.ITIMER_REAL, max(0.001, started + wall_seconds - 1 - time.monotonic()))
    try:
        return lifecycle(started, worker_group)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
