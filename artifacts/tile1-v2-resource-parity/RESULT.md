# Generated tile1 resource and parity control, 2026-10-09

The parent ran one exact-source reviewed generated 64³ tile1 CPU control.
`tiled/supervision.json` records exit 0, accepted resource guard, 39 fast
samples all at kernel pressure mask 1, stable swap usage, no watchdog or post
guard reason, and sampled peak process-group RSS **795,983,872 bytes**. The
preflight has seven accepted samples, the launch receipt binds PID/PGID and
kernel start identity before the model worker, and finalization reports no
cleanup error or residual. Worker output is finite contiguous FP32
`[1,3,64,64,64]` with all expected module/alias checks.

The exact source-pinned saved-only comparator ran once after the model exited.
Against the earlier accepted tile4 reference on the same generated input and
checkpoint, all **786,432 logits are byte-identical** (raw SHA
`52252550d12d7fc0ddd930c044f58cd6e1bca00606646bbd9780687bce3f2853`),
with zero decoded disagreement across 262,144 cells. The saved tile4 sampled
peak was **956,301,312 bytes**, making this single tile1 run's sampled peak
160,317,440 bytes (16.8%) lower. Instrumented forward times were 0.575 s
and 0.553 s; one run per tile and hooks do not support a speed claim. Four
whole-tumor and two tumor-core logits lie within ±0.001 of zero in each arm,
but all values match exactly.

The original `artifacts/tile1-cleanup-negative-v1/` attempt remains negative
for resource acceptance: its supervisor failed with `PermissionError` in
cleanup and wrote no `supervision.json`, although it saved matching logits.
The replacement did not reconstruct that missing trace; it made a new,
prospectively guarded measurement. The reviewed replacement supervisor is
SHA `323b628663fca238b764d05d74f386974321ebcdf7707e83f82dc06ecd4b07fe`
under contract SHA
`27bb2d589ed8dfa4d5a848db217f9eea02388c710131396062c604daf6e4fab5`.
Eight generated launch/failure tests passed both locally and independently
before execution. Earlier approved bytes remain in the ignored source
directory's `withdrawn-reviewed-hashes/` with a withdrawal note.

This is software parity and single-run resource feasibility at 64³ with a
generated input. It does not establish full128³ feasibility, performance on
real patient scans, target localization, or clinical planning accuracy.
Independent post-run receipt audit passed: `INDEPENDENT_REVIEW.md`
(SHA-256 `d75bf8db50ae5602819fe4f454a1670c04faa90b7444bf278f1574ae5758b369`).
It checked the exact launch/finalization identity, 39 fast and five slow guard
samples, logits bytes, comparator result, and absence of a remaining process
group.
