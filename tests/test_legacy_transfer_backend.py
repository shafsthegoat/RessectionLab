"""No-checkpoint controls for the fixed backend child and durable failure path."""
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import os
import subprocess
import sys
import pytest

from resectionlab import fixed_rl256_loader, legacy_transfer_supervisor, legacy_transfer_worker


def test_existing_attempt_refuses_before_source_or_spawn(tmp_path, monkeypatch):
    output = tmp_path / "existing"
    output.mkdir()
    monkeypatch.setattr(legacy_transfer_supervisor, "_check_sources",
                        lambda: (_ for _ in ()).throw(AssertionError("source opened")))
    with pytest.raises(FileExistsError):
        legacy_transfer_supervisor.run_attempt(output)
    assert not (output / "attempt.json").exists()


def test_cancel_before_source_or_spawn_writes_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(legacy_transfer_supervisor, "_check_sources",
                        lambda: (_ for _ in ()).throw(AssertionError("source opened")))
    output = tmp_path / "cancelled"
    with pytest.raises(InterruptedError):
        legacy_transfer_supervisor.run_attempt(output, cancelled=lambda: True)
    result = json.loads((output / "failure.json").read_text())
    assert result["status"] == "failed" and result["errorType"] == "InterruptedError"
    assert not (output / "worker.log").exists()


def test_cancel_during_owned_child_reaps_and_keeps_receipt(tmp_path, monkeypatch):
    class Child:
        pid = 712345
        returncode = None
        def poll(self): return self.returncode
        def terminate(self): self.returncode = -15
        def kill(self): self.returncode = -9
        def wait(self, timeout): return self.returncode
    child = Child()
    monkeypatch.setattr(legacy_transfer_supervisor, "_check_sources", lambda: tmp_path)
    launches = []
    monkeypatch.setattr(legacy_transfer_supervisor.subprocess, "Popen",
                        lambda *a, **k: (launches.append((a, k)) or child))
    calls = iter((False, True))
    with pytest.raises(RuntimeError, match="cancelled"):
        legacy_transfer_supervisor.run_attempt(tmp_path / "owned", cancelled=lambda: next(calls, True))
    saved = json.loads((tmp_path / "owned/supervision.json").read_text())
    assert saved["status"] == "failed" and saved["stopReason"] == "cancelled"
    assert saved["workerTerminationConfirmed"] and saved["unresolvedWorkerPid"] is None
    assert child.poll() == -15
    assert launches and launches[0][0][0][1:4] == ["-I", "-B", "-X"]
    assert "fresh-pycache" in launches[0][0][0][4]
    assert launches[0][1]["pass_fds"]


def test_parent_lease_eof_terminates_generated_idle_child(tmp_path):
    read_fd, write_fd = os.pipe()
    repo = next(parent for parent in Path(__file__).resolve().parents
                if (parent / "pyproject.toml").is_file())
    stage = repo / "src/resectionlab"
    source = repo / "src"
    code = ("import sys,time;sys.path.insert(0," + repr(str(source)) + ");"
            "import resectionlab;resectionlab.__path__.insert(0," + repr(str(stage)) + ");"
            "from resectionlab.legacy_transfer_worker import _require_parent_lease;"
            "_require_parent_lease();print('ready',flush=True);time.sleep(10)")
    env = {**os.environ, "RESECTIONLAB_PARENT_LEASE_FD": str(read_fd)}
    child = subprocess.Popen([sys.executable, "-I", "-B", "-c", code],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        pass_fds=(read_fd,), env=env)
    os.close(read_fd)
    try:
        assert child.stdout.readline().strip() == "ready"
        os.close(write_fd)
        assert child.wait(timeout=3) == 143
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=3)


def test_existing_worker_result_refuses_before_fixed_loader(tmp_path, monkeypatch):
    target = tmp_path / "result.json"
    target.write_text("earlier attempt")
    monkeypatch.setattr(fixed_rl256_loader, "load_fixed_rl256_policy",
                        lambda: (_ for _ in ()).throw(AssertionError("checkpoint loader called")))
    with pytest.raises(FileExistsError):
        legacy_transfer_worker.execute(target)
    assert target.read_text() == "earlier attempt"


def test_wrong_fake_loader_identity_fails_before_actor_forward(tmp_path, monkeypatch):
    monkeypatch.setattr(fixed_rl256_loader, "load_fixed_rl256_policy",
        lambda: (None, SimpleNamespace(checkpoint_sha256="sha256:" + "0" * 64), {}))
    target = tmp_path / "result.json"
    with pytest.raises(RuntimeError, match="another policy identity"):
        legacy_transfer_worker.execute(target)
    saved = json.loads(target.read_text())
    assert saved["status"] == "failed" and saved["stage"] == "fixed_loader"


def test_fixed_checkpoint_read_is_bounded_and_regular(tmp_path):
    source = tmp_path / "tiny.pt"
    source.write_bytes(b"fixed-test-checkpoint")
    expected = hashlib.sha256(source.read_bytes()).hexdigest()
    assert fixed_rl256_loader._bounded_checkpoint_bytes(
        source, source.stat().st_size, expected) == b"fixed-test-checkpoint"
    with pytest.raises(ValueError, match="exact regular file"):
        fixed_rl256_loader._bounded_checkpoint_bytes(source, source.stat().st_size - 1, expected)
    with pytest.raises(ValueError, match="changed after preflight"):
        fixed_rl256_loader._bounded_checkpoint_bytes(source, source.stat().st_size, "0" * 64)
    link = tmp_path / "link.pt"
    link.symlink_to(source)
    with pytest.raises(OSError):
        fixed_rl256_loader._bounded_checkpoint_bytes(link, source.stat().st_size, expected)
