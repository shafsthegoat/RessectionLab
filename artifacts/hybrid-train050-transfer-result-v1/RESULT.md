# Frozen proposal ranking fails the new-anatomy quality test

The frozen balanced ridge scorer does not preserve search quality on the next
fixed, previously untrained TRAIN anatomy, ReMIND-050. Both complete plans pass
fresh uncached geometry replay. The model and original four-patient fitting
cohort are unchanged; there is no refit, policy update or held-out evaluation.

| Measure | Frozen top-eight proposals | Lean exhaustive search |
| --- | ---: | ---: |
| Supplied target cells removed | 6 | 37 |
| Simulated target removal, mm³ | 5.722050 | 35.285977 |
| Outside-target removal, mm³ | 0 | 0 |
| Frozen nominal return | 5.618838 | 34.830407 |
| Movements before STOP | 2 | 11 |
| Online geometry previews | 24 | 1,092 |
| Source preparation through plan, seconds | 0.940494 | 10.518894 |

The quality gate is **false**. Lower cost does not compensate for lost task
performance. Search remains the stronger baseline on this anatomy. Even search
removes only 0.1892% of the supplied target; neither result is useful complete
surgical resection, a measured intervention or clinical accuracy evidence.
Timing is one fixed-order pair, with the model first, not a stable speed estimate.

The failure is directly visible at a shared state. Both methods choose the same
first five-cell action. At the next identical state, search finds 32 positive
legal actions. The learned top eight include only one. Search's best seven-cell
action has learned rank 46 and reward 6.637443; the shortlist instead chooses one
cell with reward 0.886792. This establishes inadequate shortlist recall after
transfer. It does not establish a particular distribution-shift mechanism.

At the later divergent model state, all eight queried choices are shaft-blocked
and 83 emitted proposals remain unchecked. No saved exhaustive state matches it,
so the current evidence does not tell whether those omitted choices can help.
The next single contrast will check all remaining emitted choices when the
shortlist has no positive feasible action, before accepting STOP. This preserves
the model, reward and physical rules; it may prevent premature STOP but cannot
guarantee repair of the earlier suboptimal choice. All fallback checks must count.

Both plans were sealed before either authoritative replay. Independent saved
validation recomputes the negative comparison, confirms the same public inputs,
model identity, source bindings, closed SELECT/EVAL roles and complete denominator.
The parent finishes in 32.400835 s with 539,295,744 B sampled peak memory and
2,842,222 B output. Total planning plus replay counts are 2,232 previews,
30 transitions, six ridge scores and one model load. Source construction and
checking are charged; the original training cost remains separately recorded
in the balanced-v2 result. Cleanup is clean. No new data acquisition is needed.

These are simulator actions on public annotation-assisted anatomy. The supplied
whole-tumor label is not a prescribed surgical target, automatic support is not
measured tissue occupancy, and functional/vascular/material/clinical outcomes
remain unvalidated. No raw-scan-only or RL improvement is claimed.

`preservation-index.json` maps exact original paths to this compact archive.
Canonical code is at `f626b50`; ignored execution sources are already preserved
in the adjacent transfer-preparation artifact. The five diagnosis inventories
support shared-state comparisons. Other original replay inventories remain
local with hashes in the saved audit; rerunning its full checks requires those
JSON files at their original paths. Patient arrays and weights stay outside Git.
