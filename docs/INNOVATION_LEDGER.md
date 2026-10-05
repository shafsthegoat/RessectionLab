# Innovation ledger

Decisions follow measured failures and comparisons. No novelty claim is established
by this ledger. Updated October 5, 2026 UTC; existing patient splits remain fixed.

| Hypothesis / type | Discriminating experiment and evidence | Decision | Next evidence |
|---|---|---|---|
| Two small reward updates improve choices / baseline | PAT05 two REINFORCE updates changed weights, not argmax; score12.70 before/after vs greedy410.31 | HOLD training expansion | A justified mechanism before more updates; retain negative |
| Search labels teach useful choices / baseline adaptation | BC8 raises PAT05 score to139.90; later choices still remove other tissue without target | PROMOTE diagnostic, not efficacy | Match extra optimization against new state coverage |
| Learner-visited states repair later choices / DAgger adaptation | Same starting weights/Adam;8 extra updates×6 examples each:290.51 augmented vs239.11 control; search410.31 | PROMOTE development transfer | Separate real TRAIN anatomy; no repeated PAT05-only tuning |
| Frozen ranking helps on different anatomy / adaptation | Precision repeat: PAT22 672.30 vs greedy789.10; PAT28 493.53 vs612.54; PAT25 no legal motions; PAT16/20 support conflicts; all nine episodes audited | HOLD population training | Diagnose access/proposal coverage from saved records; preserve failures and all five patients in the denominator |
| Fixed preparation allows an informative speed–quality comparison / baseline comparison | V2: twelve accepted STOP/IL/greedy episodes on four of six TRAIN patients; two historical blocks retained. Search wins all four returns; IL uses140 vs264–280 previews and16.55–22.35 vs30.77–42.62 online seconds | RETAIN search and frozen IL tradeoff; no RL superiority | Single fixed-order timing and crop/full-field representation differ; separate shared preparation/audit. V1 complete metrics/history match exactly, as do decisions apart from timing/status fields; failed outcomes remain unchanged |
| Weakly separated sequential rankings explain part of transfer failure / diagnostic hypothesis | Saved first-state greedy choices rank8/77 and4/71; entropy fractions0.997152/0.996870; independent arithmetic audit passes | INVESTIGATE after fixed preparation comparison | Later states diverge; large later reward remainders are descriptive, not causal. No new training or architecture conclusion |
| Shortest local exit supplies a useful full-tool access / geometric assumption | PAT25 original78shaft failures all occur before insertion; other five fixed exits accept58–78 previews, but two hide the whole target from the actor crop | REVISE prospective access/input contract | Whole-shaft ingress plus explicit actor coverage; same allowed alternatives for every method; original STOP-only result stays unchanged |
| Fixed initial whole-tool ingress screening restores action availability / preparation repair | PAT25 committed comparison: all6 exits/468 static checks; shortest of5 eligible exits yields78/78 native previews vs original0/78; target remains100% visible; independent saved audit passes | PROMOTE identical TRAIN preparation comparison | Apply same rule to other permitted TRAIN cases before zero-learning methods comparison; retain outside-image/exposure unknowns and decreased deepest-shaft visibility; no policy gain yet |
| Target-centered and whole-source views repair missing context / representation hypothesis | Partial fixed TRAIN run: three completed, one failed, two blocked. Legacy/local target mass already100%; local-only shaft visibility decreases; coarse views increase centerline coverage without changing acceptance | RETAIN coarse-context candidate; no learning gain | Correct PAT05 exact-record binding separately; visibility is not full-tool clearance or an access-selection gate |
| The audit rejections reflect reduction precision / correctness hypothesis | Analytical repair controls and nine repeat episodes pass at unchanged tolerance; action/observation/geometry histories exactly match the original attempt; maximum return arithmetic change0.000026317 | PROMOTE numerical correction only | Original three failed outcomes stay failed; no policy improvement or physical-fidelity claim |
| Enhancing volume improves observed-outcome prediction beyond baseline function / baseline | RHUH40 fixed leave-one-out: full-minus-KPS Brier+0.00120647 (worse); full model detects3/14 recorded deficits; all9,092 fits and saved predictions audited | RETAIN negative comparator; no planning reward | Previously inspected observational cohort; route-invariant features cannot rank strategies |
| Anatomical/network injury predicts measured outcomes beyond volume / new hypothesis | RHUH simple baseline complete, timing/domain absent; injury images unverified; BTC linkage unresolved;63-patient labels/preprocessing discrepant | HOLD integration | Validate observed imaging measurements and endpoint meaning before feature expansion; no route-risk interpretation |
| Conditional mechanics adds useful geometric updating / adaptation | Analytical controls pass; specimen convergence and patient surface fidelity remain unresolved | HOLD reward use | Relevant numerical and measured validation before a planning claim |

Current experiment evidence:
[RL](../artifacts/pat05-real-geometric-learning-v1/RESULT.md),
[BC8](../artifacts/pat05-real-geometric-imitation-v1/RESULT.md),
[matched state aggregation](../artifacts/pat05-real-visited-imitation-v1/RESULT.md),
[five-case transfer attempt](../artifacts/remaining-training-frozen-spatial-v1/RESULT.md),
[precision-only repeat](../artifacts/remaining-training-frozen-spatial-float64-v1/RESULT.md),
[six-exit diagnostic](../artifacts/pat25-six-exit-access-diagnostic-v1/RESULT.md),
[partial observation diagnostic](../artifacts/training-observation-coverage-v1/RESULT.md),
[observational baseline](../artifacts/rhuh-preoperative-baseline-v1/RESULT.md).
The state-aggregation mechanism draws on
[Ross, Gordon and Bagnell (2011)](https://proceedings.mlr.press/v15/ross11a.html);
one bounded geometric experiment does not inherit clinical or transfer guarantees.

The fixed preparation comparison now completed on PAT05/PAT22/PAT25/PAT28;
only PAT25 changed access, and all four crops retain the nominal target. The
separate saved-record audit accepts twelve episodes and preserves both historical
support blocks. See [V2 results](../artifacts/prepared-training-planner-comparison-v2/RESULT.md).
No further architecture sweep follows from this result. The next bounded RL
diagnostic should separate actor and critic contributions on the already retained
PAT05 development episodes and measure actual parameter/logit changes; a large
critic gradient plus global clipping alone does not establish suppressed Adam
updates. This diagnostic has not run and must preserve the baseline checkpoint.
Improved context remains a candidate, not a demonstrated explanation: the
completed legacy crops already retained the target. One checksum transfer and the separately
released single-image compatibility experiment succeeded. The latter received
6,337,221 compressed bytes under its 64-MiB retained-file cap and matched the
predeclared compressed-byte MD5 candidate; an independent byte/provenance audit
passed. The separately released format inspection now accepts scalar 240 x 240 x
155 float32 storage, a numeric 1-mm qform and finite values; its saved-result audit
also passed. A fixed three-plane display and saved-result audit now pass; visual
review leaves inferior anatomical extent unresolved. Registration and full-volume
coverage remain unaccepted. This does not confirm the publisher's algorithm, supply target/injury labels
or satisfy the unchanged 12-file declaration. See
[single-image format evidence](../artifacts/rhuh-single-image-inspection-v1/RESULT.md).
Observation coverage, full-tool ingress and action coverage remain separate gaps.
No RL sweep, neurological reward or UI expansion is justified by current evidence.
