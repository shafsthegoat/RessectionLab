# HBE v5 N36 S120 numerical result

**The one-shot ordinal-6 row passed its numerical software checks and independent saved replay.** All 121 frames / 120 solver states were reproduced byte for byte. The frozen N36 S60/S120 diagnostic passed at every one of 61 shared load states and all 75 displacement probes.

- Maximum simulated-force change: **1.8068706253426825e-12 N**, below **7.271855906202017e-5 N**; worst shared state 38.
- Maximum probe-vector change: **2.313705266673336e-14 m**, below **8e-7 m**; worst shared state 26, probe 43 (zero-based).
- Simulated endpoint force: **−0.03635127953101008 N** at full displacement **−0.00073726 m**.
- One native call: **919.116218 s**; one original readout: **78.893169 s**. Independent replay: **78.576757 s**, zero native calls. All exited 0 within frozen sampled resource limits; no retry.

This is the same 39,610-node / 34,992-Hex8 native lower-half fixture and material used by S60, with a reflected full response. It establishes a passing load-step sensitivity diagnostic at the declared shared states. It does not release the full twelve-row comparison, establish spatial or continuum accuracy, validate measured force or patient mechanics, or fit tissue properties. The exact N12 supplemental exception and missing original post-run source/runtime guard record remain preserved; historical N8 preparation/readout timing remains unknown.

`INDEPENDENT_REVIEW.md` gives the audit decision and limits. `release.json` and `receipt.json` are exact copies; `source-bindings.json` records source/input/output/runtime ancestry and accounting. `temporal-comparison.json` retains all 61 signed force differences and per-state maximum probe-vector differences, with a hash binding to the full ignored comparison. Bulk raw logs, complete readouts, and full per-probe arrays are omitted. `audit.py` and `finish.py` preserve the exact independent audit source; their hashes are bound in `expected-terminal.json`, and their original invocation directory is recorded in the independent replay receipt. Their source guards refer to the completed audit commit, not a later archive commit.
