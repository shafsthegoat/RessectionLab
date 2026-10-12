# One common-state continuation pair

This source-only runner tests one deterministic intervention on the annotation-assisted ReMIND-045 TRAIN simulator. It is not a policy update, measured surgical result, or deployable improvement claim. No execution has been performed by its author.

Both planning arms start fresh at the original absolute horizon 24 and reproduce the saved eight TAIL decisions, including public observations, native states, current tool, and realized reward. At index 8 they must produce the same complete ordered certified inventory, immediate rewards, and frozen-tail scores. RETAINED_TAIL takes `nominal-cavity-796668398d38557bae703113`; IMMEDIATE_BEST takes the stable original-inventory immediate maximum `nominal-cavity-22839a32a529e5e57ad1d0b7`. Every later choice uses the unchanged tail rule. The adapter receives only the eight prefix checks and current-state declaration; it receives no saved future actions or outcomes.

Both complete plans seal before either independent fresh authoritative replay. The retained arm must reproduce the original full route, native decision/history projections, outcomes, and nominal history hash exactly. A mismatch or incomplete branch is a failed contrast. The report separates the prefix from each remaining return, including the forced action once, and compares full target/outside/contact/path/tool/reward outcomes. It does not require either treatment to win.

`endgame_planner.py` derives from canonical planner SHA256 `fe43285ae70d3ff9944e2f4b2e532f7205d0148e5fd30aae99216a8d1243ff37`; `planner-adapter.patch` makes the delta explicit. The final catalog diagnostic aggregates only the already populated ledger after the terminal choice is fixed and before the terminal transition. It is published only for a sealed complete plan, labelled as a preterminal catalog, with zero additional queries. It does not establish physical exhaustion after STOP.

The exact existing 497-row feature cache is read without copying or reconstructing patients. The fixed 64-update head, direct calibration, and tail solve reproduce their accepted head/math identities. Only ReMIND-045's five declared public arrays may be opened. The other three TRAIN identities remain cache provenance; SELECT/EVAL remain closed. Prior cache-fill failure, direct, tail, and original64 costs stay separately referenced. Current setup, repeated prefix/scoring, both plans and both fresh replays are fully charged.

Expected wall time is roughly 100–250 seconds based on the previous 269.9-second twelve-route run; this is a forecast, not a speed comparison. Ceilings: one thread, worker 600 seconds, parent 630 including cleanup, 3 GiB sampled RSS, 30,000 native previews, 128 frozen-policy forwards, 4 MiB output and 8 MiB whole stage. No automatic retry. Expected output is under 1 MiB: two small route histories/certificates, two plan records, compact fits, metadata and costs. No checkpoint, cache, decoded image, or feature copy is written.

## Root controls and launch

Use the existing owned control supervisor from `build/continuation-tail-v1/model/run_controls.py`, rebinding its source/test/bootstrap paths to this directory. Its existing 60-second/1-GiB/one-thread envelope runs:

```
.venv/bin/python -I -B -m pytest -q build/continuation-endgame-pair-v1/runner/test_runner.py
```

There are 11 focused cases: actual adapted-loop fake-interface controls (including absolute horizon), exact saved metadata declaration, negative accounting and catalog counts, source ordering, false-release entrypoints, and a bounded real isolated import smoke with patient/checkpoint opens denied. No native task, fitting or patient payload is part of these controls. Generated fixture files should remain well below 1 MiB. Record and remove only completed pytest convenience links before the strict whole-tree source freeze.

After successful controls and root review, root freezes the exact current source/metadata closure once:

```
.venv/bin/python -I -B build/continuation-endgame-pair-v1/runner/freeze_sources.py --release
```

Use the printed SHA256 (or hash the newly created source-index.json) in both commands. The source index is exclusive and is not replaced automatically.

```
.venv/bin/python -I -B build/continuation-endgame-pair-v1/runner/run_owned.py --source-index-sha256 <EXACT_SHA256>
.venv/bin/python -I -B -X pycache_prefix=build/continuation-endgame-pair-v1/absent-parent-cache build/continuation-endgame-pair-v1/runner/run_owned.py --source-index-sha256 <EXACT_SHA256> --execute
```

The parent cache path must remain absent. Root alone owns the actual release and run. The original tail source, source index, report and audit are unchanged.
