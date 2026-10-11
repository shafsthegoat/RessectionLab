# Recorded-action value fit: completed, rejected for planning

The fixed comparison completed all 36 reconstructed TRAIN episodes, 497 action
features, 64 head updates and 16 fresh method/case routes. The independent saved
result audit passed 4,072 checks on its first attempt; every complete route has
an accepted independent full-tool geometry certificate. All original inputs,
rewards, world assumptions, splits and horizons remained fixed.

| TRAIN case | Greedy target mm³ | Learned Q target mm³ | Greedy/Q moves | Greedy/Q path mm |
|---|---:|---:|---:|---:|
| ReMIND-002 | 0 | 0 | 0 / 0 | 0 / 0 |
| ReMIND-015 | 55.313110 | 50.544739 | 13 / 24 | 56.895 / 112.228 |
| ReMIND-018 | 0 | 0 | 0 / 0 | 0 / 0 |
| ReMIND-045 | 48.637450 | 41.961722 | 12 / 24 | 95.488 / 196.766 |

Greedy remains the baseline. The learned candidate loses target volume and
return on both productive cases and hits the 24-action horizon. Planning
previews increase from 2,231 to 4,104 (+83.95%). Both methods have zero
outside-target removal in these four endpoints; that alone is insufficient.
The two STOP outcomes are enforced by the common actual-positive-action guard,
not learned negative-action avoidance. Frozen IL64 matches greedy on 015 and
matches 045 target volume using twice the moves; frozen RL8 is worse. Exact
four-method outcomes and costs are retained in the terminal diagnosis.

## Measured failure and next change

Weighted training MSE falls from 325.660767 to 131.800980 (59.53%), but planning
worsens. In each productive initial state the head picks narrow suction with
immediate reward about 1.87 over an available aspirator action worth about
5.69. On 015 its greedy alternative ranks 22nd by predicted Q; on 045, 13th.
All 48 productive decisions use positive Q ranking, with no greedy fallback.
Local immediate regret is not an executed counterfactual continuation.

A separate direct-prefix diagnostic found only two conflicting exact public
state/action label groups. Their optimistic weighted variance floor is
0.231623 MSE, only 0.176% of final fitting error. This rejects label conflict
as a sufficient explanation. It does not distinguish lossy frozen features,
head underfitting or extrapolation from behavior-supported targets. These
labels describe recorded continuations, not optimal action values. There are
147 negative mandatory-action labels and no successful negative-cost
preparation continuation in this corpus.

The next bounded iteration will test the actual value-fitting/action-ranking
bottleneck and retain reusable exact frozen features. Lower fit loss alone
will not promote a policy: complete-route quality and costs must improve under
the same comparison rules. No new dataset is justified by this failure.

## Full costs and limits

Owned parent time: 1,831.976 seconds (30.53 min), sampled peak process-group RSS:
1,510,948,864 bytes. Work: 80,659 native previews, 666 frozen policy forwards,
64 head updates. Corpus reconstruction takes 1,127.214 seconds, including
59.466 seconds of feature extraction; the fit takes 0.104294 seconds. These
inclusive times must not be summed. All declared limits hold; the child exited
zero, was reaped and has no remaining owned processes. Upfront costs are not
amortized away. The 520,856-byte feature array was not saved in this run, so
its cache cannot be manufactured retrospectively.

This is supervised continuation fitting on simulator-generated experience in
four existing annotation-assisted TRAIN anatomies. It is not observed surgical
action data, independent patient evaluation, force/cutting validation, or
clinical evidence. No SELECT/EVAL data were opened. The independent geometry
checker verifies the declared idealized model; no new independent physical
measurement validates these decisions.

Storage after the run and diagnostics, before this compact archive: 1.652 GiB
additional allocated growth and 330.035 GiB free. The experiment stage closes
at 3,149,824 bytes of its 8-MiB allowance. No data, model weights or feature arrays
are committed here. The original head-only checkpoint remains ignored locally.

`inventory.json` binds the copied compact evidence. Existing source controls
and exact runner are in `../train-continuation-value-source-v1/`; the source
index references existing historical inputs without duplicating them.
