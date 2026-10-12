# Replay-only recovery

The original run remains failed. Both complete plans had sealed before the retained replay failed its final Python equality check. Native `to_history_record` and `compact_history` preserve tuple coordinates; historical JSON contains lists. The generated regression executes the exact old compactor and demonstrates that mismatch. The original failing replay projection was not published, so this diagnosis does not establish the absence of another mismatch. Both new full replays must pass.

The correction compares canonical JSON bytes for every compact-history and decision field, with no omitted fields or numeric tolerance. Changed numeric coordinates, reward, state-bound history digest, or missing fields still fail. Each fresh replay certificate/projection is now written before assertions, preserving evidence if parity fails.

Inputs are exactly the failed report, clean containment receipt, old source index, both sealed plans, original TAIL report, and independent recovery-point audit. `recovery_inputs.py` pins them and validates completed source/head/common-state/choice/return seals before any source visit. The only payload allowlist is the original five public ReMIND-045 arrays. No checkpoint, feature cache, fitted model, fitting, scoring, or replanning is loaded or called. The source/geometry/world, actions, rewards and absolute horizon remain unchanged. Both authoritative fresh replays receive independent geometry checks, and the retained full route must reproduce exactly.

Original costs remain paid: 3458 native previews, 24 policy forwards, 64 head updates, one direct and one tail solve, 66.722809208 seconds worker / 67.316874916 seconds parent. Recovery reports its source admission and both replays separately and adds them to the original pair; earlier cache/direct/tail costs remain referenced without amortization.

Forecast 35–65 seconds from the original approximately 22-second retained replay. Fixed ceilings are 300 seconds worker / 330 parent including cleanup, one thread, 3 GiB RSS, 12000 previews, zero policy forwards, 2 MiB output, 8 MiB whole shared stage. No automatic retry. Original source/index/run files remain unchanged.

Root should run the eight compact tests using the existing owned 60-second control supervisor, rebound to this directory:

```
.venv/bin/python -I -B -m pytest -q build/continuation-endgame-pair-v1/replay-recovery/test_recovery.py
```

The batch includes exact old compactor regression, saved-plan admission/substitutions, evidence-before-assert ordering, false release, and isolated actual imports with patient/checkpoint/cache opens denied. No actual native replay occurs in controls. No test has been executed by the source author.

After root review/controls, and recording/removing only completed pytest convenience links, root freezes then launches once:

```
.venv/bin/python -I -B build/continuation-endgame-pair-v1/replay-recovery/freeze_sources.py --release
.venv/bin/python -I -B build/continuation-endgame-pair-v1/replay-recovery/run_owned.py --source-index-sha256 <PRINTED_SHA256>
.venv/bin/python -I -B -X pycache_prefix=build/continuation-endgame-pair-v1/absent-recovery-parent-cache build/continuation-endgame-pair-v1/replay-recovery/run_owned.py --source-index-sha256 <PRINTED_SHA256> --execute
```

The parent cache prefix and `recovery-01` must be absent. `source-index.json` is written exclusively. The existing exact owned-child helper and Darwin sampler are reused unchanged. A successful result establishes only this fixed paired simulator intervention, not a new deployable model or clinical benefit.
