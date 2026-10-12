# Calibrated continuation: target recovery, still worse than greedy

One fixed final-readout solve reduces logged weighted training MSE from
131.800984 to 87.912479 (33.30%). All twelve fresh routes complete and pass
independent full-tool geometric replay. The original Q64 fit and all eight
GREEDY/Q64 routes reproduce their retained counterparts. The calibrated model
recovers the target volume previously missed by Q64 on both productive cases,
but the unchanged useful-improvement gate fails. Greedy remains the baseline.
The independent saved-result audit passes 3,907 checks on its first execution
in 0.350 seconds, including scalar recomputation from all 497 saved predictions.
That checks the saved numerical claims, not an independent neural forward pass.

| TRAIN case | Greedy / calibrated target mm³ | Greedy / calibrated movements | Greedy / calibrated tool changes | Greedy / calibrated return |
|---|---:|---:|---:|---:|
| ReMIND-002 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| ReMIND-015 | 55.313110 / 55.313110 | 13 / 22 | 0 / 17 | 54.866215 / 54.059426 |
| ReMIND-018 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| ReMIND-045 | 48.637450 / 48.637450 | 12 / 20 | 0 / 13 | 48.181963 / 47.487688 |

Target removal, outside-target removal and contact bounds match greedy within
the frozen tolerance. Planning previews rise from 2,231 to 3,730 (+67.19%),
transitions from 29 to 46, and complete-tool travel from 152.383 to 243.447 mm.
The calibrated model uses 42 movements versus Q64's 48 and reaches 58/51 target
cells versus Q64's 53/44. That is progress against the previous learned model,
not superiority over search. Both guard-enforced immediate stops remain in the
four-case denominator; they do not demonstrate learned negative-action avoidance.

## What changed and what failed

The original IL64 encoder, its actor/STOP/critic and the 64-update value head's
hidden layer stay frozen. One weighted minimum-change SVD solve adjusts only
the value head's final linear layer. The same 497 logged rows, subject/trace/state
weights, signed mandatory-action targets, reward, world, horizon and full
candidate inventory remain fixed. The hidden design has rank 9 of 33; this
describes its logged-data fit and does not prove that action interaction is
absent everywhere. Lower scalar error now recovers endpoint coverage but still
misranks efficient sequences.

The next causal test separates exact immediate reward from learned future
return. The current planner already computes immediate reward for every
candidate. Fit the remaining recorded continuation instead of asking the head
to approximate that known immediate term, then combine them exactly once.
This changes the prediction target, not the task reward, tool penalty or
geometric checker. Freeze that separate experiment before execution and compare
complete routes against greedy and this calibrated baseline. More fitting alone
and improvement over the weaker Q64 baseline are insufficient acceptance.

## Cost, recovery and evidence scope

The recovery loads the one authenticated 670,261-byte cache without copying it
or reconstructing features. It repeats the fixed 64 head updates for exact
baseline reproduction and performs one 0.016660-second calibration. Two extra
head batches save all 497 predictions for independent scalar error calculation;
they require no encoder or native preview. Recovery uses 21,574 previews,
90 frozen-policy forwards, 385.569 seconds parent time and 1,238,384,640 bytes
sampled peak process-group RSS. These include the fresh replay work; online
planning costs remain separate in the report. The process exits zero, is reaped,
and has no remaining owned processes. No budget or source pin changes.

The original cache-publication failure is preserved separately. It paid 45,560
previews, 497 frozen forwards and 1,285.243 seconds parent time before a duplicate
exclusive metadata write failed. Its saved cache passed the independent recovery
point audit. Combined collection failure plus recovery cost is 67,134 previews,
587 frozen forwards and 1,670.812 seconds parent time. The older negative Q64
comparison remains an additional historical cost, not erased or amortized away.

These are simulator-generated continuations in four already exposed,
annotation-assisted TRAIN anatomies. They are not recorded surgical decisions,
held-out patient transfer or clinical evidence. The productive routes remove
less than 1% of either full target. No SELECT/EVAL data are opened. Geometric
certification checks the declared idealized interaction; no new independent
physical measurement validates the policy, cutting forces or injury response.

No dataset addition is justified by this fitting/ranking failure. Before this
compact archive, measured extension growth is 1.664 GiB and free space is
328.805 GiB; the shared experiment uses 8,998,912 allocated bytes of 32 MiB.
Patient arrays, cached features and weights remain outside Git. Source, fixed
configuration and controls are in `../train-continuation-readout-recovery-v1/`.
The final independent audit and evidence hashes accompany this result.
