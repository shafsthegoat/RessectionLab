# GlioMODA generated 64³ CPU memory diagnostic, v2

This was one separately reviewed, generated-only software diagnostic. The v1 failure and its receipts in `gliomoda-nnunet-zero-smoke-v1/` were left intact. No patient inputs, training, label inspection, or full 128³ model patch were used.

The first v2 launcher invocation failed before `Popen` because host Python 3.9 cannot evaluate a `str | None` annotation; it created no child or output files. The annotation was removed, the supervisor was re-hashed and independently re-reviewed, and then **one** supervised diagnostic call ran. The worker SHA-256 was `6a83d74c12575682de4d9e678a2e1d4900babddcb70c711a15eb282a68646bf1`; the corrected supervisor SHA-256 was `2992df779bde4e927b3438935d7e80a984483e8bb2ed04eca6af9159df61f154`.

The one actual diagnostic exited 0 with no watchdog trigger in 3.529 s. Eleven flushed phase markers show self RSS of 564,800 KiB immediately after scoped `weights_only=True` loading, 565,888 KiB after strict state copy, and **566,016 KiB before forward**. The sampled process-group RSS rose during the generated forward to **3,017,936 KiB**, only **127,792 KiB below** the unchanged 3 GiB cap; after-forward self RSS was 2,956,112 KiB. The forward itself took 1.660 s. Its output shape was `[1,3,64,64,64]` with zero nonfinite values.

All 292 state key/shapes matched. The strict CPU copy matched checkpoint values, all 88 alias groups (184 compared pairs) shared storage, and safe serialization globals were restored. The checkpoint SHA-256 remained `0e29f882310fe8cb076d6cadb982067ef53c6a32231f40ae17d9c173aa4307b3`. The process group ended. An independent read-only audit confirmed the saved result and phase order.

This localizes **v2's** memory jump to the forward stage, not the checkpoint loader or model construction. It does not prove exactly where v1 crossed the cap because v1 lacked stage markers; v1 exceeded the same cap at 3,161,360 KiB and remains a genuine negative result. The v2 margin is narrow and execution-dependent. Full 128³ CPU inference remains on hold under this cap, and neither run establishes patient inference, segmentation quality, surgical utility, or clinical validity. No full output tensor or output hash was saved for numerical parity.

Evidence: `generated-smoke.log`, `generated-smoke-supervision.json`, `generated-smoke-result.json`, `preflight.json`; all ignored local build records. No tracked source or patient split changed.
