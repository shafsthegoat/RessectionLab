# Generated CPU forward and memory limit

A pinned 31,197,263-parameter network completed one **generated zero-input
64³ forward** with finite output shape [1,3,64,64,64]. This is smaller than its
128³ declared inference patch and establishes no patient accuracy.

| Attempt | Sampled group peak | Cap | Outcome |
| --- | ---: | ---: | --- |
| First 64³ smoke | 3,161,360 KiB | 3,145,728 KiB | Stopped; no completed output |
| Separate phase-instrumented diagnostic | 3,017,936 KiB | 3,145,728 KiB | Completed; finite output |

The second run took 3.529 s overall, with 1.660 s in the forward. Its pre-forward
self RSS was 566,016 KiB (552.75 MiB); its peak left only 127,792 KiB of sampled
headroom. It localizes its own memory jump to the forward. The first run lacked
phase markers, so its failure stage remains unproven. One pass after one
resource failure does not establish reliable 64³ capacity. The full 128³ patch
and patient inference remain untested.

Strict load and state equality checks passed; 88 alias groups/184 compared
pairs shared storage. All 292 checkpoint state entries match architecture
shapes, but alias-counted entries are not unique parameters. Scoped safe-load
globals were restored, and the checkpoint hash stayed unchanged. Both process
groups ended. No model training, patient scan, annotation or clinical outcome
was used. No full output tensor was saved for numerical parity.

Both original outcomes and independent reviews travel with this record:
[attempt 1](CPU_ATTEMPT_1.md), [review 1](ATTEMPT_1_REVIEW.md),
[diagnostic 2](CPU_DIAGNOSTIC_2.md), [review 2](DIAGNOSTIC_2_REVIEW.md).
The second launcher’s pre-child Python-version failure is also retained.
Next is memory-conscious exact-patch inference investigation; MPS availability
alone does not prove this network can run within an acceptable memory budget.
