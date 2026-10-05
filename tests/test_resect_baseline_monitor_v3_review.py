"""Independent bounded-grace tests; one tiny nonpatient OS-process control."""
import ast
import importlib.util
import json
import os
from pathlib import Path
import select
import subprocess
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).parents[1]
V3 = ROOT / "artifacts/mechanics/resect-case4-baseline-alignment-v3"


@pytest.fixture
def monitor():
    spec = importlib.util.spec_from_file_location("baseline_v3_independent", V3 / "align.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Worker:
    pid = 222

    def __init__(self, clock, wait_elapsed=0, code=0):
        self.clock = clock
        self.wait_elapsed = wait_elapsed
        self.code = code
        self.exit_code = None
        self.wait_requests = []
        self.kills = 0

    def poll(self):
        return self.exit_code

    def wait(self, timeout):
        self.wait_requests.append(timeout)
        self.clock[0] += self.wait_elapsed
        if self.exit_code is not None:
            return self.exit_code
        if self.code is None:
            raise subprocess.TimeoutExpired("constructed child", timeout)
        self.exit_code = self.code
        return self.exit_code

    def kill(self):
        self.kills += 1
        self.exit_code = -9


def mocked_query(module, monkeypatch, clock, text="111 80 S\n222 0 ?E\n", code=0, elapsed=0):
    calls = []
    monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])

    def query(*args, **kwargs):
        calls.append(kwargs["timeout"])
        clock[0] += elapsed
        return SimpleNamespace(returncode=code, stdout=text, stderr="constructed monitor record")

    monkeypatch.setattr(module.subprocess, "run", query)
    return calls


def test_query_consumption_reduces_single_grace_to_remaining_deadline(monitor, monkeypatch):
    clock = [20.0]
    calls = mocked_query(monitor, monkeypatch, clock, elapsed=.02)
    child = Worker(clock, wait_elapsed=.005)
    result = monitor.resource_probe(child, 111, deadline=20.03)
    assert result["failure"] is None
    assert calls == pytest.approx([.03])
    assert child.wait_requests == pytest.approx([.01])
    assert result["termination_grace"]["elapsed_seconds"] == pytest.approx(.005)
    assert result["exit_code_after_query"] is None and result["exit_code_after_grace"] == 0


def test_requested50ms_and_observed_scheduler_overshoot_are_distinct(monitor, monkeypatch):
    clock = [10.0]
    mocked_query(monitor, monkeypatch, clock)
    child = Worker(clock, wait_elapsed=.056)
    result = monitor.resource_probe(child, 111, deadline=11.)
    assert result["failure"] is None
    assert child.wait_requests == [.05]
    assert result["termination_grace"]["elapsed_seconds"] == pytest.approx(.056)
    assert result["termination_grace"]["confirmed_exit_code"] == 0


def test_successful_exit_cannot_hide_overall_work_deadline_violation(monitor, monkeypatch):
    clock = [10.0]
    mocked_query(monitor, monkeypatch, clock)
    child = Worker(clock, wait_elapsed=.011)
    result = monitor.resource_probe(child, 111, deadline=10.01)
    assert child.wait_requests == pytest.approx([.01])
    assert result["exit_code_after_grace"] == 0
    assert "WORK_DEADLINE_EXHAUSTED" in result["failure"]


@pytest.mark.parametrize("state", ["?E", "Z", "R"])
def test_process_state_text_never_substitutes_for_direct_exit_confirmation(monitor, monkeypatch, state):
    clock = [10.0]
    mocked_query(monitor, monkeypatch, clock, text=f"111 80 S\n222 0 {state}\n")
    child = Worker(clock, wait_elapsed=.05, code=None)
    result = monitor.resource_probe(child, 111, deadline=11.)
    assert len(child.wait_requests) == 1
    assert result["termination_grace"]["timed_out"]
    assert "LIVE_CHILD_EXIT_GRACE_TIMEOUT" in result["failure"]
    assert result["exit_code_after_grace"] is None


@pytest.mark.parametrize("text,code", [("", 0), ("111 80 S\n222 0 ?E\n", 1),
    ("111 80 S\n222 bad ?E\n", 0), ("111 80 S\n111 80 S\n", 0)])
def test_invalid_query_never_waits_even_if_exit_could_be_confirmed(monitor, monkeypatch, text, code):
    clock = [10.0]
    mocked_query(monitor, monkeypatch, clock, text=text, code=code)
    child = Worker(clock)
    result = monitor.resource_probe(child, 111, deadline=11.)
    assert result["failure"] and not child.wait_requests
    assert "termination_grace" not in result


@pytest.mark.parametrize("failure", ["cleanup_overrun", "RSS_overrun"])
def test_final_supervisor_caps_remain_authoritative_after_child_exit(monitor, monkeypatch, tmp_path, failure):
    clock = [0.0]
    child = Worker(clock, wait_elapsed=1.1 if failure == "cleanup_overrun" else 0)
    (tmp_path / "root-release.json").write_text("{}")
    monkeypatch.setattr(monitor, "HERE", tmp_path)
    monkeypatch.setattr(monitor.sys, "argv", ["align.py", "--execute"])
    monkeypatch.setattr(monitor.sys, "addaudithook", lambda _: None)
    monkeypatch.setattr(monitor, "authenticated_declaration", lambda: ({"files": [], "budget": {"wall_seconds": 1, "RSS_bytes": 1000}}, {}))
    monkeypatch.setattr(monitor, "verify", lambda _: {})
    monkeypatch.setattr(monitor.subprocess, "Popen", lambda *a, **k: child)
    monkeypatch.setattr(monitor.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(monitor.time, "sleep", lambda _: None)

    def probe(*args, **kwargs):
        assert kwargs["deadline"] == .75
        child.exit_code = 0
        return {"failure": None, "combined_RSS_bytes": 1001 if failure == "RSS_overrun" else 999}

    monkeypatch.setattr(monitor, "resource_probe", probe)
    with pytest.raises(SystemExit) as stopped:
        monitor.main()
    receipt = json.loads((tmp_path / "resource-receipt.json").read_text())
    assert stopped.value.code != 0 and receipt["status"] == "failed"
    assert receipt["exit_code"] == 0
    assert receipt["failure"] == ("OVERALL_WALL_CAP_EXCEEDED" if failure == "cleanup_overrun" else "WALL_OR_RSS_LIMIT")
    assert child.wait_requests[0] <= 1


def test_natural_os_process_lifecycle_with_actual_ps(monitor):
    """No mocked ps: one declared tiny stdlib child, killed/reaped on failure."""
    child = subprocess.Popen([monitor.sys.executable, "-I", "-c",
        "import sys,time; print('ready',flush=True); sys.stdin.readline(); time.sleep(.02)"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    samples = []
    start = monitor.time.monotonic()
    try:
        assert select.select([child.stdout], [], [], 1)[0]
        assert child.stdout.readline().strip() == "ready"
        child.stdin.write("exit\n")
        child.stdin.flush()
        deadline = monitor.time.monotonic() + 1
        while child.poll() is None:
            record = monitor.resource_probe(child, os.getpid(), deadline=deadline)
            samples.append(record)
            assert record["failure"] is None, json.dumps(record)
            assert monitor.time.monotonic() < deadline
        assert child.wait(timeout=.2) == 0
        assert samples
        print("REAL_OS_CONTROL=" + json.dumps({"status": "passed", "child_pid": child.pid,
            "elapsed_seconds": monitor.time.monotonic() - start, "actual_ps": True,
            "patient_access": False, "samples": samples}))
    finally:
        if child.poll() is None:
            child.kill()
        child.wait(timeout=1)
        child.stdin.close()
        child.stdout.close()
        child.stderr.close()


def test_scientific_access_and_render_contracts_match_committed_first_attempt():
    original = subprocess.check_output(["git", "show", "3247411:artifacts/mechanics/resect-case4-baseline-alignment-v1/align.py"], cwd=ROOT).decode()
    functions = lambda text: {node.name: ast.dump(node, include_attributes=False) for node in ast.parse(text).body if isinstance(node, ast.FunctionDef)}
    first, third = functions(original), functions((V3 / "align.py").read_text())
    assert set(third) - set(first) == {"resource_probe"}
    assert all(first[name] == third[name] for name in first if name != "main")
    first_decl = json.loads(subprocess.check_output(["git", "show", "3247411:artifacts/mechanics/resect-case4-baseline-alignment-v1/declaration.json"], cwd=ROOT))
    third_decl = json.loads((V3 / "declaration.json").read_text())
    historical_metadata = {"id", "status", "preparation_note", "prior_attempt", "monitor_v2", "monitor_v3"}
    assert {k: v for k, v in first_decl.items() if k not in historical_metadata} == {
        k: v for k, v in third_decl.items() if k not in historical_metadata}
