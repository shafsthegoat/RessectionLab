"""Constructed monitor records and two tiny real processes; never patient data."""
import ast
import importlib.util
import json
from pathlib import Path
import select
from types import SimpleNamespace

import pytest

HERE = Path(__file__).parent
spec = importlib.util.spec_from_file_location('baseline_monitor_v3', HERE/'align.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class Child:
    pid = 222

    def __init__(self, wait_result=0):
        self.exit_code = None
        self.wait_result = wait_result
        self.waits = []
        self.killed = False

    def poll(self):
        return self.exit_code

    def wait(self, timeout):
        self.waits.append(timeout)
        if self.exit_code is not None:
            return self.exit_code
        if self.wait_result is None:
            raise module.subprocess.TimeoutExpired('constructed child', timeout)
        self.exit_code = self.wait_result
        return self.exit_code

    def kill(self):
        self.killed = True
        self.exit_code = -9


def query(monkeypatch, stdout='111 80 S\n222 0 ?E\n', returncode=0):
    monkeypatch.setattr(module.subprocess, 'run', lambda *a, **k:
                        SimpleNamespace(returncode=returncode, stdout=stdout, stderr=''))


@pytest.mark.parametrize('code', [0, 2, -9, None])
def test_one_bounded_wait_confirms_exit_or_retains_live_failure(monkeypatch, code):
    query(monkeypatch)
    child = Child(code)
    result = module.resource_probe(child, 111, deadline=module.time.monotonic()+1)
    assert len(child.waits) == 1 and 0 < child.waits[0] <= .05
    assert result['exit_code_after_query'] is None
    assert result['termination_grace']['confirmed_exit_code'] == code
    assert (result['failure'] is None) == (code is not None)
    if code is None:
        assert result['termination_grace']['timed_out']
        assert result['exit_code_at_error'] is None
    else:
        assert result['exit_code_after_grace'] == code


@pytest.mark.parametrize('stdout,returncode', [('',0), ('111 80 S\n',1),
    ('111 bad S\n222 0 ?E\n',0), ('111 80 S\n333 0 Z\n',0),
    ('111 80 S\n111 80 S\n',0), ('111 0 S\n222 0 Z\n',0),
    ('111 80 S\n222 -1 Z\n',0)])
def test_invalid_queries_never_receive_exit_grace(monkeypatch, stdout, returncode):
    query(monkeypatch,stdout,returncode)
    child=Child()
    result=module.resource_probe(child,111)
    assert result['failure'] and not child.waits
    assert 'termination_grace' not in result


def test_query_timeout_never_receives_grace(monkeypatch):
    def timeout(*a, **k):
        raise module.subprocess.TimeoutExpired('ps', k['timeout'], stderr='probe timeout')
    monkeypatch.setattr(module.subprocess,'run',timeout)
    child=Child(); result=module.resource_probe(child,111)
    assert result['failure'] and not child.waits and result['stderr']=='probe timeout'


def test_query_and_grace_shrink_to_remaining_work_budget(monkeypatch):
    now=[10.0];timeouts=[]
    monkeypatch.setattr(module.time,'monotonic',lambda:now[0])
    def response(*a,**k):
        timeouts.append(k['timeout']);now[0]=10.005
        return SimpleNamespace(returncode=0,stdout='111 80 S\n',stderr='')
    monkeypatch.setattr(module.subprocess,'run',response)
    child=Child();result=module.resource_probe(child,111,deadline=10.01)
    assert result['failure'] is None
    assert timeouts[0]==pytest.approx(.01)
    assert child.waits[0]==pytest.approx(.005)


def test_expired_work_budget_does_not_query_or_wait(monkeypatch):
    monkeypatch.setattr(module.subprocess,'run',lambda *a,**k:pytest.fail('query after budget'))
    child=Child();result=module.resource_probe(child,111,deadline=module.time.monotonic()-1)
    assert result['failure'] and not child.waits


@pytest.mark.parametrize('code', [0, 2, None])
def test_supervisor_exit_grace_preserves_failure_and_cleanup(monkeypatch,tmp_path,code):
    child=Child(code);query(monkeypatch)
    (tmp_path/'root-release.json').write_text('{}')
    monkeypatch.setattr(module,'HERE',tmp_path)
    monkeypatch.setattr(module.sys,'argv',['align.py','--execute'])
    monkeypatch.setattr(module.sys,'addaudithook',lambda _:None)
    monkeypatch.setattr(module,'make_patient_guard',lambda _:lambda *a:None)
    monkeypatch.setattr(module,'authenticated_declaration',lambda:({'files':[],'budget':{'wall_seconds':1,'RSS_bytes':2**30}},{}))
    monkeypatch.setattr(module,'verify',lambda _: {})
    monkeypatch.setattr(module.os,'getpid',lambda:111)
    monkeypatch.setattr(module.subprocess,'Popen',lambda *a,**k:child)
    with pytest.raises(SystemExit) as done:
        module.main()
    receipt=json.loads((tmp_path/'resource-receipt.json').read_text())
    assert receipt['status']==('passed' if code==0 else 'failed')
    assert child.killed==(code is None)
    assert done.value.code==(code if code is not None else -9)
    assert receipt['cleanup_reserve_seconds']==.25


@pytest.mark.parametrize('sleep_seconds,expect_exit', [(.02,True),(1.0,False)])
def test_tiny_real_process_confirmation_and_timeout(monkeypatch,sleep_seconds,expect_exit):
    """The process is real; only its deficient ps observation is constructed."""
    process=module.subprocess.Popen([module.sys.executable,'-I','-c',
        f"import time; print('ready',flush=True); time.sleep({sleep_seconds!r})"],
        stdout=module.subprocess.PIPE,text=True)
    try:
        assert select.select([process.stdout],[],[],2)[0]
        assert process.stdout.readline().strip()=='ready'
        monkeypatch.setattr(module.subprocess,'run',lambda *a,**k:SimpleNamespace(
            returncode=0,stdout=f'111 80 S\n{process.pid} 0 ?E\n',stderr=''))
        result=module.resource_probe(process,111,deadline=module.time.monotonic()+.5)
        assert result['exit_code_after_query'] is None
        assert 'termination_grace' in result
        assert (result['failure'] is None)==expect_exit
        assert result['termination_grace']['timed_out']==(not expect_exit)
        print(json.dumps({'real_child_control':sleep_seconds,'result':result}))
    finally:
        if process.poll() is None:
            process.kill()
        process.wait(timeout=1)
        process.stdout.close()


def test_only_monitoring_changes_from_frozen_v2():
    v2=HERE.parent/'resect-case4-baseline-alignment-v2'
    functions=lambda path:{n.name:ast.dump(n,include_attributes=False) for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef)}
    a,b=functions(v2/'align.py'),functions(HERE/'align.py')
    assert set(a)==set(b)
    assert all(a[k]==b[k] for k in a if k not in ('resource_probe','main'))
    da,db=json.loads((v2/'declaration.json').read_text()),json.loads((HERE/'declaration.json').read_text())
    for k in ('files','fit','views','acceptance','budget','allowed','forbidden','helper_snapshot_manifest_sha256','baseline_header_receipt_sha256'):
        assert da[k]==db[k]
