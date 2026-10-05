# Saved native-axis decisions after one RAW update

All 6 prescribed initial/latest pairs matched exactly on ordered action IDs, RAW feature rows, state and mask. They represent 3 distinct states repeated across deterministic selection seeds. Chosen actions were unchanged in every pair.

| State / step | Non-STOP rows / unique rows | Aliased rows | Initial → latest top-two margin | Actions changing rank | Initial → latest value |
|---|---:|---:|---:|---:|---:|
| 1 / 0 | 26 / 26 | 0 | 0.000251755 → 0.00409512 | 8 | -0.101595 → -0.071742 |
| 2 / 1 | 24 / 24 | 0 | 0.00666936 → 0.0101746 | 17 | -0.0994987 → -0.0677098 |
| 3 / 2 | 22 / 22 | 0 | 0.0181364 → 0.0300396 | 12 | -0.0964879 → -0.0629593 |

The JSON retains every action's actual saved logit, rank and paired change, including logit differences after subtracting each state's mean. These describe the recorded update; no policy forward or softmax was computed. Selection uses the first maximum row, and the latest weights remain distinct from the selected checkpoint.

Exact repeated 15-feature rows expose actions that this representation presents identically to the actor in a shared state. This is a representational observation; it does not establish equal future outcomes, prove that every optimal route is unrepresentable, or attribute the unchanged selection score to aliasing. Distinct feature rows can also produce equal logits. No near-equality threshold or favorable subset was chosen after seeing results.

These are three steps in one previously studied development patient, with deterministic repeats and the same unreviewed support/hypothetical access as the completed pilot. This post hoc diagnostic establishes neither clinical efficacy nor robustness and chooses no new settings. It reads immutable saved outputs and checkpoint bytes only: zero new model forwards, simulation episodes, random draws, gradients, final worlds or stress worlds.
