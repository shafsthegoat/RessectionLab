Prospective child-exit supervision correction; no execution release

The saved terminal-run-01 remains supervision FAILED. It records science
criterion true/worker exit0 but live_group_rss_unknown and cleanup EPERM, with
no surviving owned PIDs. The receipt does not identify the failed cleanup
syscall/PID sufficiently to prove its cause; this proposal cannot turn that
historical execution into a clean run.

Source inspection identifies a reproducible race: Popen.poll sees live; the
group sampler sees no process during child exit; another immediate poll can
still see live before wait status is available. The old parent refuses that
sample and cleanup can then signal/probe a disappearing process. The repair
distinguishes an exact observed child exit from unknown live memory:

owned_child_exit.py binds the direct Popen PID to kernel start identity/PGID.
For a missing root RSS sample, it allows one <=50ms Popen.wait on that exact
child. Confirmed reap records returncode/start identity/grace and does NOT
record zero RSS or a successful memory sample. Timeout remains fatal. All
sampler exceptions remain fatal; an EPERM is never excused by a later exit.

After reap, cleanup enumerates only bounded group membership and checks known
descendant identities; it does not call the legacy dead-root RSS sampler or
signal its process group. Unknown/reused members and unverifiable enumeration
fail closed. Every signal requires live owned identity. Signal EPERM uses the
direct Popen fallback for containment, retains phase/PID/PGID/errno/error, and
still fails. Unknown identity None requires independent ESRCH before absence.
Cleanup respects the existing caller deadline; a final direct-child kill/reap
is bounded and never converts an error into successful verification.

future-runner.patch is UNAPPLIED, against the frozen terminal run_owned.py.
It introduces the small source-bound helper, uses its observed-exit result and
records its structured lifecycle receipt. It does not alter worker/model/data,
RSS/wall/output caps or scientific acceptance. It requires helper verification,
child exit0, and all previous report/resource/source gates for future success.
No complete copied runner or new execution directory is created here.

Eleven deterministic fake child/kernel tests are prepared; no subprocess,
sleep, native model, real sampler or training runs. They cover confirmed exit,
live unknown RSS, nonzero exit, observer EPERM, cleanup EPERM with fallback,
no post-exit group signal/RSS, unavailable group metadata, unknown/reused PID,
missing identity, unreaped child and deadline failure. Root controls test slot:
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -I -B -m pytest -q -p no:cacheprovider build/native-cut-residual-diagnostic-v1/monitor-exit-repair-v1/test_owned_child_exit.py
Only AST/source preparation has run so far.

Root also runs one real Darwin benign-child smoke after those controls:
  .venv/bin/python -I -B build/native-cut-residual-diagnostic-v1/monitor-exit-repair-v1/smoke_darwin_exit.py
One Python child waits on an inherited pipe, allowing a real live RSS sample.
Root releases the pipe and reaps exit0 between two samples. The second sample
records exit rather than zero RSS; cleanup calls the exact existing libproc
group enumeration after reap. Deadline5s; output darwin-smoke-01 is exclusive;
any enumeration errno/unknown cleanup remains failure, without automatic retry.
This validates the real Darwin post-exit seam, not the model. The deterministic
fake control separately covers exit happening inside a missing-RSS sample.

Limits: live RSS is sampled, so transient peaks/descendants between samples
are not proven absent. The fixed worker still needs its existing no-spawn
boundary. Existing sampler.group exceptions can identify only its call phase
and root/PGID when the legacy exception omits the actual member PID; affected
PID remains null rather than guessed. A process exiting later than the50ms
grace remains a conservative failure. Kernel calls cannot be preempted by a
Python deadline. Immediate exit before kernel identity capture uses direct
Popen reap plus verified empty group, and records absent start identity.
No retroactive clean-run claim, model rerun, or changed performance result.
