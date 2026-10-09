# Independent audit: one generated GlioMODA CPU smoke, negative resource result

Date: 2026-10-09. Reviewed exact saved supervisor receipt and log for the single authorized generated-zero 64³ CPU attempt. No checkpoint load, model forward, patient data or tracked edit was performed by this reviewer.

## Verified outcome

- Saved receipt: `build/limited-input-guard-design/gliomoda-nnunet-zero-smoke-v1/generated-smoke-supervision.json`. Exit code `-9`, `watchdog_reason=process_group_rss_cap_exceeded`, `monitor_error=null`, elapsed 2.745656334 seconds. The 120-second wall cap was not reached.
- Sampled peak process-group RSS was **3,161,360 KiB** versus **3,145,728 KiB (3 GiB)** cap, an observed exceedance of 15,632 KiB. This is a sampled peak; exact instantaneous maximum is unknown. The watchdog killed the child; a fresh `ps` query found no remaining member of process group 56430.
- The log contains only `one scoped CPU weights-only checkpoint load starting`. There is no `generated-smoke-result.json` or `generated-smoke-failure.json`; the SIGKILL bypassed the worker's exception handler. The checkpoint remains 250,184,062 bytes and independently streamed SHA-256 `0e29f882310fe8cb076d6cadb982067ef53c6a32231f40ae17d9c173aa4307b3`. No source weight was changed.
- The reviewed worker/supervisor source hashes were `ff6ca82183dcc9523836cf52c470de023a1efc5deaf23c4bdad9f0dacd8cb79d` and `231c70080ca6b49b7f5384ed698cc4b1e2232e9b9098e415c4a0189161fde6b7`, respectively. The one-shot output sentinel and `no_retry=true` remain intact.

## Interpretation

**NO-GO for claiming a completed generated forward, a 64³ runtime fit, a patient prediction or any accuracy result.** The attempt demonstrates that the combined code path exceeded the prescribed sampled 3-GiB memory limit and was stopped. The log has no stage markers after the pre-load print, so the resource spike cannot be localized to checkpoint load, architecture construction, alias checking, strict copy, or a partially started forward. It is incorrect to state that no forward was started; only that none completed or produced an accepted output. A prior isolated safe metadata load of the same checkpoint reported 491,008 KiB sampled peak, which makes a load-only explanation unproven.

There was no unrestricted pickle fallback and no second attempt. The original preflight already held the 128³ model patch because a plausible Slow3d columns workspace exceeded the cap; this negative 64³ attempt strengthens that resource concern. It does not invalidate the earlier static key/shape or scoped metadata checks.

## Bounded future direction, not a retry authorization

Before any new experiment, freeze a new script/release that emits flushed stage markers with process RSS after each major step (safe load, trim optimizer state, alias check, network build, strict copy, pre-forward, post-forward). Recalculate a conservative cap from measured stage peaks. Consider exporting **only** hash-bound network weights to a safe tensor-only local format in a separate bounded process, eliminating optimizer-state materialization during model forward; retain original archive/checkpoint hashes and alias map. A smaller stride-compatible anisotropic generated input (for example 64×32×32, with bottleneck spatial volume >1 for InstanceNorm) could test the API under less workspace, but it would not prove full 128³ inference. A subsequent run requires its own declared purpose, independent preflight and unchanged patient/privacy boundary. No immediate rerun is supported by this review.
