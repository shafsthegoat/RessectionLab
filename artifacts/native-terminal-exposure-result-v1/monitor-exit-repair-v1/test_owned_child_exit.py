"""Deterministic fake child/kernel controls; no Popen, native calls or sleeps."""
import errno
import importlib.util
from pathlib import Path
import signal
import subprocess

import pytest

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('owned_child_exit_controls',HERE/'owned_child_exit.py')
M=importlib.util.module_from_spec(spec);spec.loader.exec_module(M)


class FakeKernel:
    def __init__(self):self.live={1234:(10,20)};self.signals=[];self.group_error=None
    def getpgid(self,pid):
        if pid not in self.live:raise ProcessLookupError(errno.ESRCH,'gone')
        return 1234
    def killpg(self,pid,sig):
        self.signals.append(('group',pid,int(sig)))
        if self.group_error:raise self.group_error
        self.process.exit(-int(sig))
    def kill(self,pid,sig):self.signals.append(('pid',pid,int(sig)));self.live.pop(pid,None)


class FakeProcess:
    pid=1234
    def __init__(self,kernel):self.kernel=kernel;kernel.process=self;self.returncode=None;self.wait_exit=None;self.stubborn=False;self.waits=[]
    def poll(self):return self.returncode
    def exit(self,code):
        if not self.stubborn:self.returncode=code;self.kernel.live.pop(self.pid,None)
    def wait(self,timeout):
        self.waits.append(timeout)
        if self.returncode is None and self.wait_exit is not None:self.exit(self.wait_exit)
        if self.returncode is None:raise subprocess.TimeoutExpired('fake-child',timeout)
        return self.returncode
    def send_signal(self,sig):self.kernel.signals.append(('direct',self.pid,int(sig)));self.exit(-int(sig))
    def kill(self):self.send_signal(signal.SIGKILL)


class FakeSampler:
    def __init__(self,kernel):self.kernel=kernel;self.pids=[1234];self.error=None;self.group_calls=0;self.hidden_identity=False
    def process_start_identity(self,pid):return None if self.hidden_identity else self.kernel.live.get(pid)
    def group(self,pgid,pid):
        self.group_calls+=1
        if self.error:raise self.error
        return {'pids':list(self.pids),'process_group_resident_bytes':100 if self.pids else 0,
                'detached_descendant_start_identities':{}}


def fixture():
    kernel=FakeKernel();process=FakeProcess(kernel);sampler=FakeSampler(kernel)
    lifecycle=M.OwnedChildExit(process,sampler,kernel=kernel,members=lambda pgid:list(kernel.live),clock=lambda:1.)
    return lifecycle,process,sampler,kernel


def test_missing_sample_resolves_only_by_exact_direct_child_wait_and_no_postexit_signal():
    life,process,sampler,kernel=fixture();sampler.pids=[];process.wait_exit=0
    result=life.sample()
    assert result=={'state':'exited','measurement':None,'rss_observed':False}
    terminal=life.cleanup(deadline=4.)
    assert terminal['verified'] and terminal['child_reaped'] and terminal['exit_code']==0
    assert sampler.group_calls==1 and kernel.signals==[]
    event=next(e for e in terminal['events'] if e['phase']=='sample.exit_race_confirmed')
    assert event['pid']==1234 and event['start_identity']==(10,20) and event['wait_returncode']==0
    assert event['grace_seconds']==.05


def test_live_unknown_memory_stays_failed_even_after_successful_cleanup():
    life,process,sampler,kernel=fixture();sampler.pids=[]
    with pytest.raises(M.LifecycleFailure,match='remained live'):life.sample()
    terminal=life.cleanup(deadline=4.)
    assert not terminal['verified'] and terminal['child_reaped']
    assert any(e['phase']=='sample.live_unknown_memory' for e in terminal['errors'])


def test_nonzero_natural_exit_is_recorded_not_changed_to_success():
    life,process,sampler,kernel=fixture();sampler.pids=[];process.wait_exit=7
    assert life.sample()['state']=='exited'
    terminal=life.cleanup(deadline=4.)
    assert terminal['verified'] and terminal['exit_code']==7 and not kernel.signals
    # Caller must additionally require exit_code==0 for successful execution.


def test_group_observer_eperm_is_not_forgiven_by_later_exit():
    life,process,sampler,kernel=fixture();sampler.error=PermissionError(errno.EPERM,'observer denied')
    with pytest.raises(M.LifecycleFailure):life.sample()
    process.exit(0);terminal=life.cleanup(deadline=4.)
    assert not terminal['verified'] and not kernel.signals
    error=terminal['errors'][0]
    assert error['phase']=='sample.group' and error['root_pid']==1234 and error['affected_pid'] is None and error['errno']==errno.EPERM


def test_cleanup_signal_eperm_uses_direct_child_fallback_but_remains_failure():
    life,process,sampler,kernel=fixture();kernel.group_error=PermissionError(errno.EPERM,'signal denied')
    terminal=life.cleanup(deadline=4.)
    assert not terminal['verified'] and terminal['child_reaped']
    assert ('direct',1234,int(signal.SIGTERM)) in kernel.signals
    error=next(e for e in terminal['errors'] if e['phase']=='cleanup.group_signal')
    assert error['affected_pid']==1234 and error['pgid']==1234 and error['errno']==errno.EPERM


def test_already_reaped_root_never_enters_legacy_group_rss_or_group_signal():
    life,process,sampler,kernel=fixture();process.exit(0)
    sampler.error=AssertionError('dead-root RSS must never be called')
    assert life.sample()['state']=='exited'
    assert life.cleanup(deadline=4.)['verified']
    assert sampler.group_calls==0 and kernel.signals==[]


def test_cleanup_membership_permission_failure_is_unknown_not_empty():
    life,process,sampler,kernel=fixture();process.exit(0)
    def denied(pgid):raise PermissionError(errno.EPERM,'membership denied')
    life.members=denied
    terminal=life.cleanup(deadline=4.)
    assert not terminal['verified'] and terminal['remaining_state_unknown']
    assert terminal['errors'][-1]['phase']=='cleanup.verify' and terminal['errors'][-1]['errno']==errno.EPERM
    assert not kernel.signals


def test_unseen_or_reused_pid_never_receives_signal():
    life,process,sampler,kernel=fixture();process.exit(0);kernel.live[9999]=(30,40)
    terminal=life.cleanup(deadline=4.)
    assert not terminal['verified'] and not kernel.signals
    assert 'Unowned or reused' in terminal['errors'][-1]['message']


def test_missing_kernel_identity_does_not_pass_for_existing_pid():
    kernel=FakeKernel();process=FakeProcess(kernel);sampler=FakeSampler(kernel);sampler.hidden_identity=True
    life=M.OwnedChildExit(process,sampler,kernel=kernel,members=lambda pgid:list(kernel.live),clock=lambda:1.)
    with pytest.raises(M.LifecycleFailure):life.sample()
    terminal=life.cleanup(deadline=4.)
    assert not terminal['verified'] and terminal['child_reaped']
    assert all(row[0]!='group' for row in kernel.signals)


def test_unreaped_stubborn_child_fails_closed_and_records_fallback():
    life,process,sampler,kernel=fixture();process.stubborn=True
    terminal=life.cleanup(deadline=4.)
    assert not terminal['verified'] and not terminal['child_reaped']
    assert terminal['remaining_owned_pids']==[1234]
    assert any(e['phase']=='cleanup.last_direct_child_kill' for e in terminal['events'])


def test_normal_live_rss_still_counted_and_deadline_failure_stays_failure():
    life,process,sampler,kernel=fixture()
    observed=life.sample();assert observed['state']=='running' and observed['rss_observed']
    assert observed['measurement']['process_group_resident_bytes']==100
    terminal=life.cleanup(deadline=.5)
    assert not terminal['verified'] and terminal['child_reaped']
    assert any('deadline' in e['message'] for e in terminal['errors'])
