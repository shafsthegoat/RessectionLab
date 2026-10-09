# Independent audit: generated GlioMODA 64³ memory diagnostic v2

Date: 2026-10-09. Read-only saved-result review. No model load, forward, patient array, training or tracked edit by this reviewer. The v1 failed run and its outputs were preserved; this was a separately declared v2 diagnostic with stage instrumentation and the same 3-GiB/120-second cap.

## Integrity and one-call result

- Reviewed worker SHA-256 `6a83d74c12575682de4d9e678a2e1d4900babddcb70c711a15eb282a68646bf1` and corrected supervisor SHA-256 `2992df779bde4e927b3438935d7e80a984483e8bb2ed04eca6af9159df61f154`. The first host-Python-3.9 invocation of the earlier supervisor failed on a type annotation before `Popen`; no v2 log, child or result existed then. The corrected invocation produced the one actual v2 child attempt.
- `generated-smoke-supervision.json` reports exit 0, no watchdog/monitor error, elapsed 3.529384708 seconds, sampled group-RSS peak **3,017,936 KiB** against **3,145,728 KiB** cap, leaving **127,792 KiB** sampled margin. The 28 logged RSS samples have monotonic times and reproduce that peak. Process group 56970 had no remaining members on independent `ps` inspection.
- The child produced `generated-smoke-result.json` with input `[1,2,64,64,64]`, output `[1,3,64,64,64]`, zero nonfinite output values, and a 1.659644667-second forward. All 292 state keys/shapes matched; strict CPU-loaded values matched checkpoint. The 88 constructed alias groups comprised 184 compared pairs, all sharing the same checkpoint storage, and model alias-name grouping persisted. Scoped four-entry `weights_only=True` safe globals were restored. The checkpoint remained 250,184,062 bytes; independent SHA-256 after the attempt was `0e29f882310fe8cb076d6cadb982067ef53c6a32231f40ae17d9c173aa4307b3`.
- The worker source reads only the pinned nonpatient model checkpoint, plans/dataset JSON, static preflight and prior inspection receipt, then creates a zero-valued tensor. No patient image or annotation appears in the input path.

## Localized resource finding

The worker emitted ordered, flushed phase markers from import through post-forward. Its self RSS was **566,016 KiB immediately before forward**, then **2,956,112 KiB immediately after forward**. The supervisor's peak **3,017,936 KiB** occurred while the last flushed phase was `before_generated_forward`. Thus v2's major memory increase occurred during the model forward, consistent with a large CPU Slow3d convolution workspace. The sampled peak is not an exact instantaneous maximum and cannot prove the size of any one backend allocation.

The original v1 64³ run **failed** the same 3-GiB sampled RSS limit at 3,161,360 KiB. V2's peak was only 127,792 KiB below the limit, and phase instrumentation may perturb timing/allocation. One v2 pass cannot establish repeatable capacity even at 64³. V1 lacked intermediate markers, so v2 cannot retrospectively prove which v1 stage crossed its cap. The declared model patch is 128³; static preflight held that size because a plausible Slow3d columns workspace alone exceeds 3 GiB. No full-patch runtime, preoperative image preprocessing, native-frame segmentation, transfer accuracy, clinical relevance or RL result was measured here.

**Decision:** GO to summarize as a *single generated-only architecture/load/forward smoke pass with narrow 64³ memory margin*, while retaining v1 as a negative resource result. Continue to HOLD 128³ and all patient inference claims under this runtime/cap. Future performance work needs a separately frozen memory strategy and independent patient-specific input/label gates.
