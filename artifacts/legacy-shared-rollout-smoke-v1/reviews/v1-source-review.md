# Frozen legacy policy rollout source review

**GO for one root-scheduled generated-only attempt using the exact wrapper below.** This is the legacy aspiration-only two-step task; it does not run or validate a learned mixed-instrument policy. No checkpoint payload, model forward, patient image or real worker was accessed by this reviewer.

Pinned sources:

- `legacy_checkpoint_rollout.py`: `0156ada24f0b1443f30db1a3a00415e11dbda674eb9274bf9acb718e503aa9a3`
- `supervise_legacy_rollout.py`: `5e2565af3e36d906dbbf20cc0b1361050c52be31e1c09976af5c2d9bb9a52254`
- Previously reviewed `frozen_preflight_supervisor.py`: `8d3db9c04e4b2e040f6b6f5bc153618f277233fb93c12c2cabdf418493e58048`
- Tracked `shared_episode.py`: `81d6b1faac3433f1a2eb09ff4ef70e773b69922c056495caf1a879fbbc381a4d`

The loader authenticates the exact 394,009-byte RL256 checkpoint against frozen release/index/review and current policy/observation-source identities. It rechecks the bytes immediately before an in-memory CPU `weights_only=True` load, constructs the declared architecture, requires a strict state dict, and rechecks architecture, parameters, context and generated task identities. No optimizer state is restored for training. The two methods use the same source, environment and initial observation; completed output now requires all three agreement flags and zero new updates.

Both the attempt directory and result file are reserved before load/execution, preventing an existing path from rerunning inference before failing. Started and failure-stage receipts survive ordinary errors, with no automatic retry. The wrapper source-checks before and after and reuses the corrected supervisor: 90-second wall bound, 1-GiB sampled worker RSS bound, one native thread, bounded TERM/KILL and owned-PID fallback. Unconfirmed cleanup preserves the unresolved PID and a failed terminal receipt. RSS is sampled worker-process memory, not a guarantee about transient peaks or total host usage.

The initial wrapper reused the known defective supervisor with unhandled group-signal permission errors and an unbounded final wait. That version is excluded. The reviewed wrapper binds the corrected source explicitly.

**12 independent-run controls passed in 0.59 seconds:** seven no-load reservation/source/command/completion controls plus five fake signal, reap and final-elapsed controls. No blocked payload/network/real-spawn attempt occurred, and bound sources were unchanged. Receipt `test-review.json` SHA-256: `070b06944f4b8eef0d1b7c989c40232413bb02d6e6b6305828fb898ddc21f273`.

The root must use a fresh output directory and this wrapper, rather than directly invoking the unbounded worker. Actual runtime/results still require their saved receipt audit. This source GO does not establish patient transfer, clinical validity, broad policy efficacy or RL superiority.
