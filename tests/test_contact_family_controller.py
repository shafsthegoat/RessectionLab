"""Owned-child failure controls without a family task, checkpoint, or actor."""
from __future__ import annotations

import json

import pytest


from resectionlab import contact_family_supervisor as supervisor


def test_existing_attempt_refuses_before_source_or_spawn(tmp_path, monkeypatch):
    output = tmp_path / "existing"
    output.mkdir()
    monkeypatch.setattr(supervisor, "_check_sources",
                        lambda: (_ for _ in ()).throw(AssertionError("source read")))
    with pytest.raises(FileExistsError):
        supervisor.run_attempt(output, layout_id="pcf-00", goal_id="surface", selector="STOP")
    assert not (output / "attempt.json").exists()


def test_cancel_before_source_or_spawn_writes_failure(tmp_path, monkeypatch):
    output = tmp_path / "cancelled"
    monkeypatch.setattr(supervisor, "_check_sources",
                        lambda: (_ for _ in ()).throw(AssertionError("source read")))
    with pytest.raises(InterruptedError):
        supervisor.run_attempt(output, layout_id="pcf-00", goal_id="surface",
                               selector="STOP", cancelled=lambda: True)
    saved = json.loads((output / "failure.json").read_text())
    assert saved["status"] == "failed" and saved["errorType"] == "InterruptedError"
    assert not (output / "worker.log").exists()


def test_cancel_during_owned_fake_child_reaps_and_retains_receipt(tmp_path, monkeypatch):
    class Child:
        pid = 712345
        returncode = None
        def poll(self): return self.returncode
        def terminate(self): self.returncode = -15
        def kill(self): self.returncode = -9
        def wait(self, timeout): return self.returncode
    child = Child()
    monkeypatch.setattr(supervisor, "_check_sources", lambda: tmp_path)
    launches = []
    monkeypatch.setattr(supervisor.subprocess, "Popen",
                        lambda *args, **kwargs: (launches.append((args, kwargs)) or child))
    calls = iter((False, True))
    with pytest.raises(RuntimeError, match="cancelled"):
        supervisor.run_attempt(tmp_path / "owned", layout_id="pcf-01", goal_id="deep",
                               selector="SEARCH", cancelled=lambda: next(calls, True))
    saved = json.loads((tmp_path / "owned/supervision.json").read_text())
    assert saved["status"] == "failed" and saved["stopReason"] == "cancelled"
    assert saved["workerTerminationConfirmed"] and saved["unresolvedWorkerPid"] is None
    assert child.poll() == -15
    assert launches[0][0][0][1:4] == ["-I", "-B", "-X"]
    assert launches[0][0][0][-6:] == ["--layout-id", "pcf-01", "--goal-id", "deep",
                                      "--selector", "SEARCH"]


def test_unpinned_source_closure_refuses_before_worker(tmp_path, monkeypatch):
    monkeypatch.setattr(supervisor, "SOURCE_SHA256", {})
    monkeypatch.setattr(supervisor.subprocess, "Popen",
                        lambda *args, **kwargs: (_ for _ in ()).throw(
                            AssertionError("worker spawn attempted")))
    with pytest.raises(RuntimeError, match="not been reviewed"):
        supervisor.run_attempt(tmp_path / "unreleased", layout_id="pcf-00",
                               goal_id="surface", selector="STOP")
    saved = json.loads((tmp_path / "unreleased/failure.json").read_text())
    assert saved["status"] == "failed"
    assert not (tmp_path / "unreleased/worker.log").exists()
