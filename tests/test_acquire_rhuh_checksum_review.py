"""Independent transport controls; no provider, client, or real credential access.

The lifecycle controls invoke Python functions with simulated processes, clocks,
and public metadata. They never invoke the command-line --execute path.
"""
import argparse
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import signal
from types import SimpleNamespace

import pytest


SOURCE = Path(__file__).resolve().parents[1] / "scripts/acquire_rhuh_checksum.py"
SPEC = importlib.util.spec_from_file_location("checksum_independent_review", SOURCE)
acquire = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(acquire)


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(acquire, "ROOT", tmp_path)
    monkeypatch.setattr(acquire, "QUARANTINE_ROOT", tmp_path / "quarantine")
    monkeypatch.setattr(acquire.signal, "signal", lambda *args: None)
    monkeypatch.setattr(acquire.signal, "setitimer", lambda *args: None)
    def forbidden(*args, **kwargs):
        pytest.fail("Network or unsimulated process access in analytical review")
    monkeypatch.setattr(acquire, "request", forbidden)
    monkeypatch.setattr(acquire.subprocess, "Popen", forbidden)
    return tmp_path


def test_whole_lifecycle_budget_includes_local_checks(isolated, monkeypatch):
    """The repaired supervisor charges simulated preparation to its deadline.

    Original source 707e728f instead spent 15 + 58 = 73 simulated seconds;
    that exact original test and source are retained in the review artifacts.
    """
    clock = [0.0]
    monkeypatch.setattr(acquire.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", lambda self: argparse.Namespace(execute=True, _worker=False))
    def forbidden_precheck():
        pytest.fail("Local checks must occur within the supervised worker")
    monkeypatch.setattr(acquire, "local_checks", forbidden_precheck)
    calls = []
    killed = []
    class TimedWorker:
        pid = 918274
        stdout = io.BytesIO()
        def communicate(self, timeout):
            assert timeout == 42
            clock[0] += timeout
            raise acquire.subprocess.TimeoutExpired("fixture", timeout)
        def wait(self, timeout):
            clock[0] += timeout
            return -9
    def fake_worker(*args, **kwargs):
        calls.append((args, kwargs))
        clock[0] += 15.0
        return TimedWorker()
    monkeypatch.setattr(acquire.subprocess, "Popen", fake_worker)
    monkeypatch.setattr(acquire.os, "killpg", lambda *args: killed.append(args))
    assert acquire.main() == 1
    assert len(calls) == 1 and killed == [(TimedWorker.pid, signal.SIGKILL)]
    assert clock[0] <= acquire.WALL_SECONDS


def test_exited_leader_still_cleans_process_group(isolated, monkeypatch):
    """An exited process leader does not prove all its descendants exited."""
    killed = []
    class ExitedLeader:
        pid = 918273
        returncode = 2
        stdout = io.BytesIO()
        def communicate(self, **kwargs):
            return b'{"status":"local_check_rejected"}', None
        def wait(self, **kwargs):
            return self.returncode
        def poll(self):
            return self.returncode
    monkeypatch.setattr(acquire.subprocess, "Popen", lambda *args, **kwargs: ExitedLeader())
    monkeypatch.setattr(acquire.os, "killpg", lambda pid, sig: killed.append((pid, sig)))
    assert acquire.supervise(acquire.time.monotonic()) != 0
    assert (ExitedLeader.pid, signal.SIGKILL) in killed
    assert ExitedLeader.stdout.closed


def test_final_writer_change_cannot_keep_precleanup_payload_hash(isolated, monkeypatch):
    """Model a writer still alive after the worker's earlier payload hash."""
    directory = acquire.QUARANTINE_ROOT / "run-fixture"
    directory.mkdir(parents=True)
    payload = directory / acquire.SOURCE.lstrip("/")
    payload.write_bytes(b"a" * acquire.EXPECTED_BYTES)
    receipt = {"status": "received_length_checked_local_hash_recorded", "source": acquire.SOURCE,
               "payload": acquire.verify_payload(directory),
               "publisher_checksum_for_checksum_file_known": False}
    receipt_path = acquire.QUARANTINE_ROOT / "run-fixture-receipt.json"
    receipt_path.write_text(json.dumps(receipt))
    class ExitedWorker:
        pid = 918275
        returncode = 0
        stdout = io.BytesIO()
        def communicate(self, **kwargs):
            return json.dumps({"status": receipt["status"],
                "receipt": str(receipt_path.relative_to(isolated))}).encode(), None
        def wait(self, **kwargs):
            return 0
    monkeypatch.setattr(acquire.subprocess, "Popen", lambda *args, **kwargs: ExitedWorker())
    def final_write_before_termination(*args):
        payload.write_bytes(b"b" * acquire.EXPECTED_BYTES)
    monkeypatch.setattr(acquire.os, "killpg", final_write_before_termination)
    assert acquire.supervise(acquire.time.monotonic()) != 0


@pytest.mark.parametrize("late_mutation", [False, True])
def test_final_reconciliation_accepts_only_matching_bytes(isolated, monkeypatch, capsys, late_mutation):
    directory = acquire.QUARANTINE_ROOT / "run-finalized"
    directory.mkdir(parents=True)
    payload = directory / acquire.SOURCE.lstrip("/")
    payload.write_bytes(b"a" * acquire.EXPECTED_BYTES)
    record = {"status": "received_length_checked_local_hash_recorded", "source": acquire.SOURCE,
              "payload": acquire.verify_payload(directory),
              "supervisor_acceptance_required": True,
              "publisher_checksum_for_checksum_file_known": False}
    receipt_path = acquire.QUARANTINE_ROOT / "run-finalized-receipt.json"
    receipt_path.write_text(json.dumps(record))
    if late_mutation:
        payload.write_bytes(b"b" * acquire.EXPECTED_BYTES)
    launches = []
    class Worker:
        pid = 919000
        returncode = 0
        stdout = io.BytesIO()
        def communicate(self, **kwargs):
            return json.dumps({"status": record["status"],
                "receipt": str(receipt_path.relative_to(isolated))}).encode(), None
        def wait(self, **kwargs):
            return 0
    def launch(*args, **kwargs):
        launches.append((args, kwargs))
        return Worker()
    def absent_group(*args):
        raise ProcessLookupError
    monkeypatch.setattr(acquire.subprocess, "Popen", launch)
    monkeypatch.setattr(acquire.os, "killpg", absent_group)
    code = acquire.supervise(acquire.time.monotonic())
    result = json.loads(capsys.readouterr().out)
    assert len(launches) == 1
    assert result["accepted"] is (not late_mutation)
    assert (code == 0) is (not late_mutation)
    if not late_mutation:
        assert result["final_payload"] == record["payload"]
    else:
        assert "final_payload" not in result


def test_outer_watchdog_covers_final_readback(isolated, monkeypatch):
    """Invoke the installed alarm handler as a function; no real alarm fires."""
    handlers = []
    timers = []
    killed = []
    class HardExit(BaseException):
        pass
    def exit_immediately(code):
        assert code == 124
        raise HardExit
    def stuck_lifecycle(started, worker_group):
        worker_group.append(919001)
        handlers[0](signal.SIGALRM, None)
    monkeypatch.setattr(acquire.signal, "signal", lambda sig, fn: handlers.append(fn))
    monkeypatch.setattr(acquire.signal, "setitimer", lambda *args: timers.append(args))
    monkeypatch.setattr(acquire.time, "monotonic", lambda: 10.0)
    monkeypatch.setattr(acquire.os, "killpg", lambda *args: killed.append(args))
    monkeypatch.setattr(acquire.os, "_exit", exit_immediately)
    monkeypatch.setattr(acquire, "_supervise", stuck_lifecycle)
    with pytest.raises(HardExit):
        acquire.supervise(10.0)
    assert timers[0] == (signal.ITIMER_REAL, 59.0)
    assert killed == [(919001, signal.SIGKILL)]


def test_failed_client_never_promotes_complete_or_partial_payload(isolated, monkeypatch):
    class Failed:
        pid = 818273
        def wait(self, **kwargs):
            return 7
        def poll(self):
            return 7
    called = []
    monkeypatch.setattr(acquire, "fresh_spec", lambda deadline: {})
    monkeypatch.setattr(acquire, "transfer_invocation", lambda *args: (["FAKE_CLIENT"], {}))
    monkeypatch.setattr(acquire.subprocess, "Popen", lambda *args, **kwargs: Failed())
    monkeypatch.setattr(acquire.os, "killpg", lambda *args: None)
    monkeypatch.setattr(acquire, "verify_payload", lambda directory: called.append(directory))
    receipt, path = acquire.execute()
    assert receipt["status"] == "failed" and not called
    assert "payload" not in receipt
    assert json.loads(path.read_text())["status"] == "failed"


def test_exact_length_yields_local_hash_not_publisher_verification(isolated):
    directory = isolated / "payload"
    directory.mkdir()
    raw = bytes(range(256)) * 241 + b"a" * 91
    assert len(raw) == acquire.EXPECTED_BYTES
    target = directory / acquire.SOURCE.lstrip("/")
    target.write_bytes(raw)
    record = acquire.verify_payload(directory)
    assert record == {"path": str(target.relative_to(isolated)),
                      "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def test_metadata_byte_limit_refuses_before_decode(monkeypatch):
    class Response:
        status = 200
        headers = {}
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
        def read(self, length):
            assert length == 17
            return b"\xff" * length
    class Opener:
        def open(self, *args, **kwargs):
            return Response()
    monkeypatch.setattr(acquire.urllib.request, "build_opener", lambda *args: Opener())
    with pytest.raises(acquire.Rejected, match="metadata_response_too_large"):
        acquire.request("https://invalid.example/fixture", acquire.time.monotonic() + 5, limit=16)


def test_public_authorization_cannot_redirect_to_another_origin(monkeypatch):
    replies = iter([
        (200, {}, b'<a href="https://faspex.cancerimagingarchive.net/public?context=fixture">x</a>'),
        (200, {}, b"client_id: 'public', redirect_uri: '/aspera/faspex/token'"),
        (302, {"Location": "https://other.invalid/token?code=never_send"}, b""),
    ])
    calls = []
    def request(*args, **kwargs):
        calls.append(args[0])
        return next(replies)
    monkeypatch.setattr(acquire, "request", request)
    with pytest.raises(acquire.Rejected, match="authorization_redirect_origin_changed"):
        acquire.fresh_spec(100)
    assert len(calls) == 3


@pytest.mark.parametrize("corruption", ["key_bytes", "key_permissions", "signature", "quarantine_symlink"])
def test_local_provenance_failures_stop_before_transfer(isolated, monkeypatch, corruption):
    """All bytes here are inert fixture text, not real keys or executables."""
    for name in ["EXECUTABLE", "CLIENT_KEY", "LICENSE"]:
        path = isolated / name.lower()
        raw = ("analytical fixture " + name).encode()
        path.write_bytes(raw)
        path.chmod(0o700 if name == "EXECUTABLE" else 0o600)
        monkeypatch.setattr(acquire, name, path)
        monkeypatch.setattr(acquire, name + "_SHA256", hashlib.sha256(raw).hexdigest())
    calls = []
    def local_command(argv, **kwargs):
        calls.append(argv)
        assert argv[0] in ("codesign", "git")
        return SimpleNamespace(returncode=int(corruption == "signature" and argv[0] == "codesign"))
    monkeypatch.setattr(acquire.subprocess, "run", local_command)
    expected = {
        "key_bytes": "local_dependency_hash_mismatch",
        "key_permissions": "client_key_permissions_too_broad",
        "signature": "client_signature_invalid",
        "quarantine_symlink": "quarantine_parent_is_symlink",
    }[corruption]
    if corruption == "key_bytes":
        acquire.CLIENT_KEY.write_bytes(b"modified analytical fixture")
    elif corruption == "key_permissions":
        acquire.CLIENT_KEY.chmod(0o644)
    elif corruption == "quarantine_symlink":
        outside = isolated / "other"
        outside.mkdir()
        acquire.QUARANTINE_ROOT.symlink_to(outside, target_is_directory=True)
    with pytest.raises(acquire.Rejected, match=expected):
        acquire.local_checks()
    assert not any(argv[0] == "git" for argv in calls)
