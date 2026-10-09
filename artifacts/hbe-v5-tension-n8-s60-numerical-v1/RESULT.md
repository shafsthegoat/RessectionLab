# HBE v5 tension N8 S60 numerical result

**Ordinal 7 passed its numerical software checks and independent saved replay.** All 61 frames / 60 solver states and 75 displacement probes were reproduced byte for byte. This is a direct full-native 1,045-node / 768-Hex8 specimen fixture.

- Simulated endpoint tension force: **+0.0271175517171575 N** at **+0.0007360099999999 m** displacement.
- Every loaded force is positive and increasing; all probe vectors are finite. The complete 61-state signed force, energy, work and maximum probe-norm paths are retained in `signed-path.json`.
- One native call: **2.180279 s**; one original readout: **1.561998 s**. Independent replay: **1.746642 s**, zero native calls. All exited 0 within frozen sampled caps, with no retry.
- Closed original output: **15,857,006 B**, below 64 MiB. Eight cumulative native calls; all frozen aggregate limits remain satisfied.

The first tension row cannot establish mesh convergence, full twelve-row qualification, measured-force accuracy, tissue calibration, or patient validity. The original compression N12 failure, its exact separate supplement, and its missing original post-run source/runtime guard record remain preserved; historical N8 preparation/readout timings remain unknown.

`INDEPENDENT_REVIEW.md` records the decision and limitations. Exact release/terminal/replay receipts and `source-bindings.json` preserve source, input, output, runtime and prior-chain accounting. Bulk raw logs, full readouts and all-probe response arrays are omitted. `audit.py` and `finish.py` preserve the exact audit source; `expected-terminal.json` binds their hashes and the replay receipt records their original invocation. These source guards bind the completed audit commit, not later archive commits.
