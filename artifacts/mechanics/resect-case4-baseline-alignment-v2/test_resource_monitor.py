"""Mocked PID/exit races only; no patient inputs or real child processes."""
import ast
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


HERE = Path(__file__).parent
spec = importlib.util.spec_from_file_location('baseline_monitor_v2', HERE/'align.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class Child:
    pid = 222

    def __init__(self, exit_code=None):
        self.exit_code = exit_code
        self.killed = False

    def poll(self):
        return self.exit_code

    def kill(self):
        self.killed = True
        self.exit_code = -9

    def wait(self, timeout=None):
        assert self.exit_code is not None
        return self.exit_code


@pytest.mark.parametrize('output,exit_code,success', [
    ('111 80 S\n222 120 R\n', None, True),
    ('111 80 S\n', 0, True),
    ('111 80 S\n222 0 Z\n', 0, True),
    ('111 80 S\n', 2, True),  # Monitoring works; main still rejects worker exit2.
    ('111 80 S\n', None, False),
    ('111 80 S\n222 0 Z\n', None, False),
    ('222 120 R\n', 0, False),
    ('111 0 S\n222 120 R\n', None, False),
    ('111 80 S\n333 120 R\n', 0, False),
    ('111 80 S\n111 90 S\n', 0, False),
    ('111 80 S\n222 -1 Z\n', 0, False),
    ('111 bad S\n222 120 R\n', None, False),
    ('111 80\n222 120 R\n', None, False),
    ('', 0, False),
])
def test_pid_exit_semantics(monkeypatch, output, exit_code, success):
    monkeypatch.setattr(module.subprocess, 'run', lambda *a, **k:
                        SimpleNamespace(returncode=0, stdout=output, stderr=''))
    result = module.resource_probe(Child(exit_code), 111)
    assert (result['failure'] is None) == success
    assert result['stdout'] == output
    assert result['exit_code_after_query'] == exit_code
    if success:
        assert result['combined_RSS_bytes'] >= 80*1024
        assert result['child_observed_exited_during_query'] == (exit_code is not None)


@pytest.mark.parametrize('problem', ['timeout', 'error', 'oversized'])
def test_query_errors_are_retained_even_after_worker_exit(monkeypatch, problem):
    def query(*args, **kwargs):
        if problem == 'timeout':
            raise module.subprocess.TimeoutExpired('ps', 2, output=b'partial', stderr=b'slow probe')
        return SimpleNamespace(returncode=1 if problem == 'error' else 0,
                               stdout='x'*16385 if problem == 'oversized' else '111 80 S\n',
                               stderr='diagnostic')
    monkeypatch.setattr(module.subprocess, 'run', query)
    result = module.resource_probe(Child(0), 111)
    assert result['failure']
    assert result['exit_code_after_query'] == 0
    assert result['stderr']
    assert len(result['stdout']) <= 16384


@pytest.mark.parametrize('exits_during_probe,worker_code', [(True,0), (True,2), (False,None)])
def test_supervisor_retains_exit_or_stops_unmonitored_worker(monkeypatch, tmp_path, exits_during_probe, worker_code):
    child = Child()
    (tmp_path/'root-release.json').write_text('{}')
    monkeypatch.setattr(module, 'HERE', tmp_path)
    monkeypatch.setattr(module.sys, 'argv', ['align.py', '--execute'])
    monkeypatch.setattr(module.sys, 'addaudithook', lambda callback: None)
    monkeypatch.setattr(module, 'make_patient_guard', lambda paths: lambda *a: None)
    monkeypatch.setattr(module, 'authenticated_declaration', lambda: ({'files':[], 'budget':{'wall_seconds':1,'RSS_bytes':2**30}},{}))
    monkeypatch.setattr(module, 'verify', lambda _: {})
    monkeypatch.setattr(module.subprocess, 'Popen', lambda *a, **k: child)
    monkeypatch.setattr(module.os, 'getpid', lambda: 111)
    monkeypatch.setattr(module.time, 'sleep', lambda _: None)
    def query(*args, **kwargs):
        if exits_during_probe:
            child.exit_code = worker_code
        return SimpleNamespace(returncode=0, stdout='111 80 S\n222 0 Z\n', stderr='')
    monkeypatch.setattr(module.subprocess, 'run', query)
    with pytest.raises(SystemExit) as done:
        module.main()
    receipt = json.loads((tmp_path/'resource-receipt.json').read_text())
    probes = [json.loads(line) for line in (tmp_path/'resource-probes.jsonl').read_text().splitlines()]
    assert len(probes) == 1
    if exits_during_probe:
        assert not child.killed
        assert done.value.code == worker_code
        assert receipt['failure'] is None
        assert receipt['status'] == ('passed' if worker_code == 0 else 'failed')
    else:
        assert child.killed
        assert receipt['failure'] == 'RSS_MONITOR_FAILED'
        assert receipt['status'] == 'failed'


def test_all_numerical_access_and_render_functions_are_unchanged_from_v1():
    old = ast.parse((HERE.parent/'resect-case4-baseline-alignment-v1/align.py').read_text())
    new = ast.parse((HERE/'align.py').read_text())
    functions = lambda tree: {n.name:ast.dump(n,include_attributes=False) for n in tree.body if isinstance(n,ast.FunctionDef)}
    before, after = functions(old), functions(new)
    assert set(after)-set(before) == {'resource_probe'}
    assert all(before[name] == after[name] for name in before if name != 'main')
    a = json.loads((HERE.parent/'resect-case4-baseline-alignment-v1/declaration.json').read_text())
    b = json.loads((HERE/'declaration.json').read_text())
    for key in ('files','fit','views','acceptance','budget','forbidden','allowed','helper_snapshot_manifest','helper_snapshot_manifest_sha256','baseline_header_receipt_sha256'):
        assert a[key] == b[key]
