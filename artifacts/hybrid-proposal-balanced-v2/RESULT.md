# Balanced proposal learning recovers search quality on the fixed TRAIN task

The one prespecified loss-weighting contrast passes its prospective TRAIN gate.
The learned top-eight shortlist now matches exhaustive search on all four
patients' target removal, return, outside-target removal and contact measure.
It fixes the original final-action regression on 045. The original experiment
and its failed gates remain unchanged; this is not a held-out or RL result.

| All four TRAIN cases | Original ridge, fresh control | Balanced ridge | Lean exhaustive |
|---|---:|---:|---:|
| Target cells, 002/015/018/045 | 0 / 58 / 0 / 51 | 0 / 58 / 0 / 51 | 0 / 58 / 0 / 51 |
| 045 geometric return | 48.151409795 | 48.181962921 | 48.181962921 |
| 045 tool changes | 1 | 0 | 0 |
| Online complete-tool previews | 232 | 232 | 2,592 |
| Source preparation through complete plans | 10.644792 s | 12.351542 s | 45.251440 s |
| Separate source/replay/audit time | 26.577163 s | 30.349176 s | 75.051737 s |

The balanced model uses 91.05% fewer previews and 72.70% less measured planning
wall time than the fresh exhaustive control. That control no longer computes
unused features or priorities. All methods retain inventory logging, whose cost
is separately measured. This is one fixed-order run, not sustained latency or
minimal uninstrumented search. Balanced planning was slower than the fresh
original ridge despite equal query counts; it is a quality repair, not a measured
speed improvement over that model. Training and independent replay costs are
separate from the online figures above.

Only loss weights changed. The same 2,592 rows from all 29 teacher states, 681
public features, feature normalization, ridge penalty 0.001 and top-eight limit
remain. Each state retains equal total weight. Mixed states split that weight
between positive and zero labels; single-class states, including all-zero
states, retain uniform weights. A correctly centered free intercept avoids
changing the regularizer. Public source/target annotations, world, tools,
rewards and complete-tool checking remain fixed. Labels are clipped immediate
simulated rewards, not surgeon demonstrations, safety estimates or long-horizon
values. No neural forward, optimizer update or checkpoint load occurs.

The [independent saved audit](../hybrid-proposal-balanced-independent-v2/RESULT-REPORT.txt) passes 4,662 checks. On the decisive 045 state, the
best candidate moves from rank nine to rank three; including it removes exactly
the 0.030553126 switch/travel loss. Every saved teacher state's top-eight
best-positive regret is now zero, although individual candidate ranks do not
uniformly improve. The new weighted fitting error improves while the original
weighted error worsens, as expected for a changed objective. Complete simulated
behavior, rather than training loss alone, supplies the acceptance evidence.
Fresh original parameters/actions/outcomes and exhaustive histories reproduce
their predecessors. All twelve plans precede independent uncached replay.

The whole owned run completes in 270.445612 s at 720,797,696 B sampled peak,
with 28 source visits, 232 transitions, 11,296 native previews and 174 linear
score calls. The two fixed solves take 0.013016 and 0.012607 s. The parent records
no surviving owned processes or cleanup errors. Thirty-seven generated controls
passed before execution, including coefficient/intercept equations, unchanged
normalization, complete execution and later-source failure handling.

These annotation-assisted, post-exposure simulations remove less than 1% of
each positive patient's full target. Outside-target removal is zero; contact
is not an injury measure. This establishes a useful computational component on
its four training patients, not scan-only planning, full resection, tissue-force
fidelity, clinical safety or generalization. SELECT037 and EVAL067 remain closed.
SELECT013 is development-exposed and its negative-immediate-reward root requires
useful lookahead; rerunning the same greedy STOP cannot demonstrate progress.
The [transfer readiness review](../hybrid-transfer-readiness-v1/REPORT.txt) rules out another unchanged greedy STOP benchmark. The next measured bottleneck is discarded successor-inventory work in the
retained beam search, before another model or training sweep.

Reproduction: source-index.json freezes canonical source at 3da7d96; the runtime
source map retains the nine noncanonical dependencies at their original paths.
Generated controls are archived separately and retain their original build-path
imports. Patient arrays, coefficients and bulk per-state ledgers stay local.
Their original hashes are retained for the independent saved audit. No acquired
rerun or changed patient role is implicit in this record.
