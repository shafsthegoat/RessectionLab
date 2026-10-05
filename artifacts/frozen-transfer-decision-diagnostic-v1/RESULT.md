# Frozen transfer: where the saved decisions diverge

This post-hoc diagnostic reads the completed precision-repeat JSON archive only.
No patient arrays, checkpoint, simulator or policy were executed. All five
prescribed TRAIN cases remain represented: PAT16/PAT20 blocked, PAT25 STOP-only,
and PAT22/PAT28 with non-STOP choices. The historical results stay unchanged.

| Initial shared state | PAT22 | PAT28 |
|---|---:|---:|
| Legal actions, including STOP | 77 | 71 |
| Greedy first action's rank under the frozen policy | 8 | 4 |
| Policy's selected-action probability | 0.014097 | 0.015256 |
| Greedy first action's policy probability | 0.014015 | 0.015193 |
| Entropy / maximum uniform entropy | 0.997152 | 0.996870 |
| First-step search-minus-policy reward | 9.798 | 19.998 |
| Total search-minus-policy reward | 116.800 | 119.004 |
| Later-step arithmetic remainder | 107.002 | 99.006 |

The first observations, action identifiers and masks match exactly within each
pair. The policy puts the greedy first choice near the top of a weakly separated
ranking. Its softmax remains close to uniform in entropy, but the executed policy
uses deterministic argmax: this does **not** establish random behavior, clinical
uncertainty or a specific optimization failure.

The later-step remainder is 91.61% and 83.20% of the total reward gap. Those states
follow different first actions, so this is descriptive arithmetic, not matched-state
regret or proof that later decisions caused that share of the gap. The findings
support inspecting sequential ranking and visited-state coverage after the fixed
access comparison. They do not justify a broad training sweep or identify an
architecture change by themselves.

`analyze.py` verifies archive/member identities, accepted historical episodes,
matching initial observations, saved softmax arithmetic and return sums.
`summary.json` retains full precision. Independent saved-record verification is
reported separately. These are geometric TRAIN diagnostics, not new-patient or
neurological-outcome validation.
