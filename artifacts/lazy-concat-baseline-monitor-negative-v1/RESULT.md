# Matched 64³ baseline: monitor failure, no numerical result

The first generated baseline attempt under active pair contract SHA-256
`4c6d17d8287f7ad791ef1002cf931beed7ee561dafb3c99547020993e0f93374`
stopped at 2026-10-09 10:46:34 UTC. It produced **no logits or result file**;
the conditional lazy arm was not run. The accepted 30-second preflight and 29
saved fast samples stayed at kernel pressure normal (`1`), 53% available
memory and flat swap. Peak sampled process-group RSS was 582,926,336 bytes,
well below the unchanged 3 GiB cap. This is a monitor exception, not a model
memory failure or a network-parity result.

The fast sampler raised `PermissionError: [Errno 1] proc_pidinfo failed for
PID 98969` at 1.638 seconds. The supervisor's last sampled phase was
`after_strict_copy_weights_cleanup`; the worker log separately reached
`after_tiled_conv3d_attachment`. No generated forward started. The worker
does launch brief `ps` children for phase RSS measurements, but the identity
of PID 98969 was not captured, so attributing it to one of those children is
only a hypothesis. A robust successor must either eliminate such subprocesses
or verify disappearance/reuse before treating a failed PID read as transient;
it must still fail closed for a live inaccessible group member.

The identity-bound finalizer signaled root PID 98947 and retained a transient
same-group PID 98970 as `residual_possible=true` with manual attention. Later
independent read-only process inspection found PIDs 98947, 98969 and 98970
absent; that does not rewrite the immediate cleanup uncertainty. The exact
TractoInferno downloader PID 92434 remained identity-matched, below its
512 MiB allowance and running at later inspection. The controller did not
signal the downloader.

This package contains direct frozen runner-source snapshots, active and
preserved original contracts, the exact downloader manifest, generated-input
receipt, preflight/launch/finalization/monitor receipts, worker log and
independent audit. It excludes weights, NPY arrays, patient images and
credentials. The source and numerical model control remain unaccepted; no
retry is authorized by this artifact.
