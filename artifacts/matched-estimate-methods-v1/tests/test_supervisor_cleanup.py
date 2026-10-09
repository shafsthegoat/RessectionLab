"""Signal-error controls use fake processes; no worker is launched."""
import json
import subprocess

import pytest

import frozen_preflight_supervisor as supervisor


class Process:
    pid = 54321
    def __init__(self, term_denied=False, kill_denied=False, exited=False):
        self.code = 0 if exited else None
        self.term_denied, self.kill_denied = term_denied, kill_denied
    def poll(self): return self.code
    def terminate(self):
        if self.term_denied: raise PermissionError("PID TERM denied")
        self.code = -15
    def kill(self):
        if self.kill_denied: raise PermissionError("PID KILL denied")
        self.code = -9
    def wait(self, timeout):
        assert timeout == 2
        if self.code is None: raise subprocess.TimeoutExpired("fake", timeout)
        return self.code


@pytest.mark.parametrize("term_denied,kill_denied,expected", [(False, False, -15), (True, False, -9), (True, True, None)])
def test_denied_group_signal_preserves_bounded_terminal_receipt(monkeypatch, tmp_path, term_denied, kill_denied, expected):
    process = Process(term_denied, kill_denied)
    monkeypatch.setattr(supervisor.subprocess, "Popen", lambda *args, **kwargs: process)
    clock = iter([0., 2., 3.])
    monkeypatch.setattr(supervisor.time, "perf_counter", lambda: next(clock))
    def denied(*args): raise PermissionError("group signal denied")
    monkeypatch.setattr(supervisor.os, "killpg", denied)
    result = supervisor.supervise_worker(["fake"], tmp_path,
        {"max_wall_seconds": 1., "max_rss_bytes": 1024}, "declaration")
    assert result["status"] == "failed" and result["returncode"] == expected
    assert result["worker_termination_confirmed"] is (expected is not None)
    assert result["unresolved_worker_pid"] == (process.pid if expected is None else None)
    assert result["cleanup_errors"]
    assert json.loads((tmp_path / "supervisor.json").read_text()) == result
    assert json.loads((tmp_path / "supervisor-failure.json").read_text()) == result


def test_exited_worker_needs_no_signal_or_unbounded_wait(monkeypatch, tmp_path):
    monkeypatch.setattr(supervisor.subprocess, "Popen", lambda *args, **kwargs: Process(exited=True))
    def forbidden(*args): raise AssertionError("Exited worker must not be signaled")
    monkeypatch.setattr(supervisor.os, "killpg", forbidden)
    result = supervisor.supervise_worker(["fake"], tmp_path,
        {"max_wall_seconds": 1., "max_rss_bytes": 1024}, "declaration")
    assert result["status"] == "complete" and result["worker_termination_confirmed"]
    assert result["cleanup_errors"] == []


def test_exit_zero_after_deadline_is_failed_with_exact_final_elapsed(monkeypatch, tmp_path):
    monkeypatch.setattr(supervisor.subprocess, "Popen", lambda *args, **kwargs: Process(exited=True))
    clock = iter([0., 1.1])
    monkeypatch.setattr(supervisor.time, "perf_counter", lambda: next(clock))
    result = supervisor.supervise_worker(["fake"], tmp_path,
        {"max_wall_seconds": 1., "max_rss_bytes": 1024}, "declaration")
    assert result["returncode"] == 0 and result["worker_termination_confirmed"]
    assert result["status"] == "failed" and result["timed_out"]
    assert result["termination_reason"] == "parent_wall_budget_exceeded"
    assert result["seconds"] == 1.1
    assert json.loads((tmp_path / "supervisor-failure.json").read_text()) == result
