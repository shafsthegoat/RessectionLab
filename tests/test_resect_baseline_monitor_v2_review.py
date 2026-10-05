"""Independent V2 supervision controls: mocked workers, no patient access."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).parents[1]
V1 = ROOT / "artifacts/mechanics/resect-case4-baseline-alignment-v1"
V2 = ROOT / "artifacts/mechanics/resect-case4-baseline-alignment-v2"


@pytest.fixture
def monitor():
    spec = importlib.util.spec_from_file_location("baseline_v2_independent", V2 / "align.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Worker:
    pid = 222

    def __init__(self):
        self.exit_code = None
        self.kill_count = 0

    def poll(self):
        return self.exit_code

    def kill(self):
        self.kill_count += 1
        self.exit_code = -9

    def wait(self, timeout):
        assert self.exit_code is not None
        return self.exit_code


def execute_mock(module, monkeypatch, tmp_path, *, stdout, query_code=0, worker_exit=None):
    worker = Worker()
    (tmp_path / "root-release.json").write_text("{}")
    monkeypatch.setattr(module, "HERE", tmp_path)
    monkeypatch.setattr(module.sys, "argv", ["align.py", "--execute"])
    monkeypatch.setattr(module.sys, "addaudithook", lambda _: None)
    monkeypatch.setattr(module, "authenticated_declaration", lambda: ({"files": [], "budget": {"wall_seconds": 1, "RSS_bytes": 2**30}}, {}))
    monkeypatch.setattr(module, "verify", lambda _: {})
    monkeypatch.setattr(module.subprocess, "Popen", lambda *a, **k: worker)
    monkeypatch.setattr(module.os, "getpid", lambda: 111)
    monkeypatch.setattr(module.time, "sleep", lambda _: None)
    queries = []

    def query(command, **kwargs):
        queries.append(command)
        assert command == ["ps", "-o", "pid=,rss=,stat=", "-p", "111,222"]
        worker.exit_code = worker_exit
        return SimpleNamespace(returncode=query_code, stdout=stdout, stderr="mock diagnostic")

    monkeypatch.setattr(module.subprocess, "run", query)
    with pytest.raises(SystemExit) as exit_result:
        module.main()
    receipt = json.loads((tmp_path / "resource-receipt.json").read_text())
    raw_probe_bytes = (tmp_path / "resource-probes.jsonl").read_bytes()
    probes = [json.loads(line) for line in raw_probe_bytes.splitlines()]
    assert len(probes) == len(queries) == 1
    assert receipt["resource_probes_sha256"] == hashlib.sha256(raw_probe_bytes).hexdigest()
    assert probes[0]["stdout"] == stdout
    assert probes[0]["stderr"] == "mock diagnostic"
    assert probes[0]["returncode"] == query_code
    assert receipt["result_sha256"] is None  # The mocked worker never produced patient output.
    return worker, receipt, probes[0], exit_result.value.code


@pytest.mark.parametrize("stdout,exit_code", [
    ("111 80 S\n", 0), ("111 80 S\n222 0 Z\n", 0), ("111 80 S\n", 3),
])
def test_confirmed_exit_race_preserves_worker_result_and_probe_evidence(monitor, monkeypatch, tmp_path, stdout, exit_code):
    worker, receipt, probe, code = execute_mock(monitor, monkeypatch, tmp_path,
        stdout=stdout, worker_exit=exit_code)
    assert worker.kill_count == 0
    assert code == exit_code
    assert receipt["status"] == ("passed" if exit_code == 0 else "failed")
    assert receipt["failure"] is None
    assert probe["child_observed_exited_during_query"] is True
    assert probe["exit_code_after_query"] == exit_code
    assert probe["combined_RSS_bytes"] == 80 * 1024


@pytest.mark.parametrize("stdout,query_code,worker_exit", [
    ("111 80 S\n", 0, None), ("111 80 S\n222 0 Z\n", 0, None),
    ("", 0, None), ("111 80 S\n222 malformed Z\n", 0, 0),
    ("111 80 S\n", 1, 0), ("", 0, 0),
])
def test_alive_unobservable_or_failed_query_cannot_become_success(monitor, monkeypatch, tmp_path, stdout, query_code, worker_exit):
    worker, receipt, probe, code = execute_mock(monitor, monkeypatch, tmp_path,
        stdout=stdout, query_code=query_code, worker_exit=worker_exit)
    assert receipt["status"] == "failed" and receipt["failure"] == "RSS_MONITOR_FAILED"
    assert probe["failure"] and code != 0
    assert worker.kill_count == (1 if worker_exit is None else 0)


def test_decisive_live_poll_is_retained_if_worker_exits_during_error_handling(monitor, monkeypatch):
    worker = Worker()
    polls = iter((None, 0))
    worker.poll = lambda: next(polls)
    monkeypatch.setattr(monitor.subprocess, "run", lambda *a, **k:
        SimpleNamespace(returncode=0, stdout="111 80 S\n", stderr=""))
    sample = monitor.resource_probe(worker, 111)
    assert "LIVE_CHILD_RSS_MISSING_OR_ZERO" in sample["failure"]
    assert sample["exit_code_after_query"] is None, "The decisive live observation was overwritten by a later exit"
    assert sample["exit_code_at_error"] == 0


def test_numerical_access_render_and_all_scientific_settings_match_committed_v1():
    old = subprocess.check_output(["git", "show", "3247411:artifacts/mechanics/resect-case4-baseline-alignment-v1/align.py"], cwd=ROOT).decode()
    current = (V2 / "align.py").read_text()
    old_tree, new_tree = ast.parse(old), ast.parse(current)
    function_nodes = lambda tree: {node.name: ast.dump(node, include_attributes=False)
                                   for node in tree.body if isinstance(node, ast.FunctionDef)}
    before, after = function_nodes(old_tree), function_nodes(new_tree)
    assert set(after) - set(before) == {"resource_probe"}
    assert all(before[name] == after[name] for name in before if name != "main")
    old_nonfunctions = [ast.dump(node, include_attributes=False) for node in old_tree.body if not isinstance(node, ast.FunctionDef)]
    new_nonfunctions = [ast.dump(node, include_attributes=False) for node in new_tree.body if not isinstance(node, ast.FunctionDef)]
    assert old_nonfunctions == new_nonfunctions
    first = json.loads((V1 / "declaration.json").read_text())
    second = json.loads((V2 / "declaration.json").read_text())
    allowed_differences = {"id", "status", "preparation_note", "prior_attempt", "monitor_v2"}
    assert {k: v for k, v in first.items() if k not in allowed_differences} == {
        k: v for k, v in second.items() if k not in allowed_differences}
    assert (V1 / "coordinate-justification.json").read_bytes() == (V2 / "coordinate-justification.json").read_bytes()
    assert "unknown" in second["prior_attempt"]["exact_monitor_cause"]
    frozen = json.loads((V1 / "failed-attempt-freeze.json").read_text())
    for name, identity in frozen["files"].items():
        assert hashlib.sha256((V1 / name).read_bytes()).hexdigest() == identity
