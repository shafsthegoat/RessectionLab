"""Prospective owned-child exit handling; no process launch or model imports.

Retains the existing sampler for live RSS. A missing live sample is never zero
RSS: only an exact Popen.wait result within a short grace resolves an exit race.
Cleanup uses membership-only enumeration after reap, avoiding dead-root RSS and
never signaling an already-reaped process group. Any observer/signal error stays
a failure even if direct-child fallback subsequently removes the worker.
"""
import ctypes
import errno
import os
import signal
import subprocess
import time


class LifecycleFailure(RuntimeError):
    pass


def darwin_group_members(sampler, pgid):
    """Reuse bounded libproc enumeration without probing a reaped root's RSS."""
    ctypes.set_errno(0)
    pids=sampler._pid_list(sampler.proc.proc_listpgrppids,pgid)
    code=ctypes.get_errno()
    if code:raise OSError(code,'group membership enumeration errno for PGID '+str(pgid))
    return pids


class OwnedChildExit:
    def __init__(self,process,sampler,*,kernel=os,members=None,clock=time.monotonic,exit_grace_seconds=.05):
        if not 0<exit_grace_seconds<=.1:raise ValueError('Exit grace must be at most100ms')
        self.process=process;self.sampler=sampler;self.kernel=kernel;self.clock=clock
        self.members=members or (lambda pgid:darwin_group_members(sampler,pgid))
        self.grace=exit_grace_seconds;self.events=[];self.errors=[];self.seen={};self.root_identity=None
        self.exit_code=None;self.reaped=False;self.cleanup_verified=False
        if process.poll() is not None:
            self._reap(0.,'bind.already_exited')
        else:
            try:
                identity=self._identity(process.pid)
                if identity is None or kernel.getpgid(process.pid)!=process.pid:
                    raise LifecycleFailure('Live new-session child ownership could not be bound')
                self.root_identity=identity;self.seen[process.pid]=identity
                self.events.append({'phase':'bind','pid':process.pid,'pgid':process.pid,'start_identity':list(identity)})
            except Exception as error:
                # Retain the handle so cleanup can still contain the direct
                # child. Unknown binding prevents every successful sample.
                self._error('bind',error,process.pid)

    def _error(self,phase,error,pid=None):
        record={'phase':phase,'root_pid':self.process.pid,'pgid':self.process.pid,'affected_pid':pid,
                'exception':type(error).__name__,'errno':getattr(error,'errno',None),'message':str(error),
                'child_returncode':self.process.returncode}
        self.errors.append(record);self.events.append(dict(record))

    def _identity(self,pid):
        identity=self.sampler.process_start_identity(pid)
        if identity is None:
            # Existing sampler's None alone does not distinguish inaccessible
            # metadata from absence. Require an independent ESRCH observation.
            try:self.kernel.getpgid(pid)
            except ProcessLookupError:return None
            raise LifecycleFailure('Existing PID has no verifiable start identity: '+str(pid))
        return tuple(identity)

    def _reap(self,timeout,phase):
        try:code=self.process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:return False
        self.exit_code=code;self.reaped=True
        self.events.append({'phase':phase,'pid':self.process.pid,'start_identity':self.root_identity,
                            'wait_returncode':code,'direct_child_reaped':True,'grace_seconds':timeout})
        return True

    def sample(self):
        if self.errors:raise LifecycleFailure('Prior lifecycle error; successful monitoring is closed')
        if self.process.poll() is not None:
            self._reap(0.,'sample.observed_exit')
            return {'state':'exited','measurement':None,'rss_observed':False}
        try:measured=self.sampler.group(self.process.pid,self.process.pid)
        except Exception as error:
            # Legacy sampler.group may not identify which member failed. Do
            # not relabel that unknown PID as the root or forgive its error.
            self._error('sample.group',error,None)
            raise LifecycleFailure('Live group RSS unverified') from error
        for pid in measured['pids']:
            try:
                identity=self._identity(pid)
                if identity is not None:
                    if pid in self.seen and self.seen[pid]!=identity:raise LifecycleFailure('Observed PID identity changed')
                    self.seen[pid]=identity
            except Exception as error:
                self._error('sample.member_identity',error,pid);raise LifecycleFailure('Owned identity unverified') from error
        if self.process.pid not in measured['pids']:
            self.events.append({'phase':'sample.missing_root','pid':self.process.pid,
                                'reported_pids':list(measured['pids']),'rss_observed':False})
            if self._reap(self.grace,'sample.exit_race_confirmed'):
                return {'state':'exited','measurement':None,'rss_observed':False}
            error=LifecycleFailure('Owned child remained live after missing RSS sample')
            self._error('sample.live_unknown_memory',error,self.process.pid);raise error
        return {'state':'running','measurement':measured,'rss_observed':True}

    def _survivors(self):
        members=set(self.members(self.process.pid));survivors=set()
        for pid in sorted(members | set(self.seen)):
            if pid==self.process.pid and self.reaped:
                if pid in members:raise LifecycleFailure('Reaped root PID/PGID is present again; ownership unknown')
                continue
            try:identity=self._identity(pid)
            except Exception as error:
                self._error('cleanup.member_identity',error,pid);raise
            if identity is None:continue
            if pid not in self.seen or self.seen[pid]!=identity:
                raise LifecycleFailure('Unowned or reused group/descendant PID: '+str(pid))
            survivors.add(pid)
        return sorted(survivors)

    def _signal_child(self,sig):
        if self.process.poll() is not None:
            self._reap(0.,'cleanup.pre_signal_exit');return
        try:
            if self.root_identity is None or self._identity(self.process.pid)!=self.root_identity:
                raise LifecycleFailure('Child identity changed before group signal')
            if self.kernel.getpgid(self.process.pid)!=self.process.pid:raise LifecycleFailure('Child PGID changed')
            # Re-poll after kernel identity checks; a late ESRCH is still benign.
            if self.process.poll() is not None:
                self._reap(0.,'cleanup.identity_check_exit');return
            self.kernel.killpg(self.process.pid,sig)
            self.events.append({'phase':'cleanup.group_signal','pid':self.process.pid,'pgid':self.process.pid,
                                'signal':int(sig),'start_identity':self.root_identity})
        except ProcessLookupError:
            if not self._reap(self.grace,'cleanup.esrch_exit_confirmed'):
                self._error('cleanup.group_signal',LifecycleFailure('ESRCH but child exit unconfirmed'),self.process.pid)
        except Exception as error:
            self._error('cleanup.group_signal',error,self.process.pid)
            # Keep the error fatal, but contain the exact Popen child where
            # possible. Popen's signal method checks/reaps its own child first.
            try:
                if self.process.poll() is None:
                    self.process.send_signal(sig)
                    self.events.append({'phase':'cleanup.direct_child_fallback','pid':self.process.pid,'signal':int(sig)})
            except Exception as final:self._error('cleanup.direct_child_fallback',final,self.process.pid)

    def cleanup(self,*,deadline):
        remaining=[]
        try:
            for sig,wait_limit in ((signal.SIGTERM,.5),(signal.SIGKILL,1.5)):
                if self.clock()>=deadline:raise LifecycleFailure('Cleanup deadline exhausted')
                self._signal_child(sig)
                self._reap(min(wait_limit,max(0.,deadline-self.clock())),'cleanup.wait')
                remaining=self._survivors()
                for pid in remaining:
                    if pid==self.process.pid:continue
                    try:
                        if self._identity(pid)!=self.seen[pid]:raise LifecycleFailure('Descendant identity changed before signal')
                        self.kernel.kill(pid,sig)
                        self.events.append({'phase':'cleanup.identified_descendant_signal','pid':pid,
                                            'signal':int(sig),'start_identity':self.seen[pid]})
                    except ProcessLookupError:pass
                    except Exception as error:self._error('cleanup.identified_descendant_signal',error,pid)
                if self.reaped and not remaining:break
            remaining=self._survivors()
            self.cleanup_verified=self.reaped and not remaining and not self.errors and self.clock()<=deadline
            if not self.reaped or remaining:raise LifecycleFailure('Child/known descendants remain or are unreaped')
        except Exception as error:
            self._error('cleanup.verify',error,None);self.cleanup_verified=False
            try:
                if self.process.poll() is None:
                    self.process.kill()
                    self.events.append({'phase':'cleanup.last_direct_child_kill','pid':self.process.pid})
                self._reap(min(.25,max(0.,deadline-self.clock())),'cleanup.last_direct_child_wait')
            except Exception as final:self._error('cleanup.last_direct_child_fallback',final,self.process.pid)
        return {'verified':self.cleanup_verified,'child_reaped':self.reaped,'exit_code':self.exit_code,
                'remaining_owned_pids':remaining,'errors':list(self.errors),'events':list(self.events),
                'remaining_state_unknown':not self.cleanup_verified,
                'owned_root_pid':self.process.pid,'owned_root_start_identity':self.root_identity}
