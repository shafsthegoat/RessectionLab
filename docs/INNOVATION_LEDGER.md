# Innovation ledger

Decisions follow measured failures and comparisons. No novelty claim is established
by this ledger. Updated October 5, 2026 UTC; existing patient splits remain fixed.

| Hypothesis / type | Discriminating experiment and evidence | Decision | Next evidence |
|---|---|---|---|
| Two small reward updates improve choices / baseline | PAT05 two REINFORCE updates changed weights, not argmax; score12.70 before/after vs greedy410.31 | HOLD training expansion | A justified mechanism before more updates; retain negative |
| Search labels teach useful choices / baseline adaptation | BC8 raises PAT05 score to139.90; later choices still remove other tissue without target | PROMOTE diagnostic, not efficacy | Match extra optimization against new state coverage |
| Learner-visited states repair later choices / DAgger adaptation | Same starting weights/Adam;8 extra updates×6 examples each:290.51 augmented vs239.11 control; search410.31 | PROMOTE development transfer | Separate real TRAIN anatomy; no repeated PAT05-only tuning |
| Frozen ranking helps on different anatomy / adaptation | Five-case attempt: PAT22 positive but below search; PAT25 no legal motions; PAT16/20 support conflicts; PAT28 audit rejection | HOLD population training | Fix arithmetic without loosening gates; preserve candidate failures and all denominators |
| The audit rejections reflect reduction precision / correctness hypothesis | PAT22 target union vs per-action totals differ2.75e−5mm³; PAT28 normal totals also disagree; float32 products confirmed in source | ITERATE narrow repair | Float64 production arithmetic, separate independent accumulation, unchanged tolerance; failed first attempt remains failed |
| Anatomical/network injury predicts measured outcomes beyond volume / new hypothesis | RHUH clinical/imaging, BTC longitudinal and63-patient connectivity audits in progress | HOLD integration | Actual accessible labels/features/timing, simple patient-level baseline, leakage and counterfactual review |
| Conditional mechanics adds useful geometric updating / adaptation | Analytical controls pass; specimen convergence and patient surface fidelity remain unresolved | HOLD reward use | Relevant numerical and measured validation before a planning claim |

Current experiment evidence:
[RL](../artifacts/pat05-real-geometric-learning-v1/RESULT.md),
[BC8](../artifacts/pat05-real-geometric-imitation-v1/RESULT.md),
[matched state aggregation](../artifacts/pat05-real-visited-imitation-v1/RESULT.md),
[five-case transfer attempt](../artifacts/remaining-training-frozen-spatial-v1/RESULT.md).
The state-aggregation mechanism draws on
[Ross, Gordon and Bagnell (2011)](https://proceedings.mlr.press/v15/ross11a.html);
one bounded geometric experiment does not inherit clinical or transfer guarantees.

Cheapest next discriminating work: correct the demonstrated accounting defect and
repeat only the declared fixed-model comparison; in parallel establish whether
real outcome data support one useful observational baseline. No extra architecture,
RL sweep, neurological reward or UI expansion is justified by current evidence.
