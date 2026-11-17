# Fixed preparation diagnostic

A fixed 0.4 mm preparation explains the high-budget target-treatment gain. The costed control treats 75.63% on average versus seed 17’s 73.55%:15 layouts tie, one improves. Delayed-preparation layouts reach 100% with either. Low-budget control 65.49% is below seed 17’s 73.55% because it pays unchanged greedy preview costs. These previously inspected layouts provide diagnostic evidence, not a new untouched test.

The control does not dominate seed 17. Parent 202 removes 54 mm³ of permitted CST membership versus 0 for seed 17, with the same 54 mm³ neural/motor removal. The learned selector uses fewer online requests. Benchmark redesign and simple-control testing should precede another learning campaign.

One actual narrow-tip exchange consumes a selection, request, modeled time and reward cost. Every later decision calls the original greedy function. The `6e04` package and exact original compiled binary are archived separately from current planner edits; every original benchmark file remains byte-identical. Native/adapter proofs and an independent accepted-history audit check the selected outcomes.

|Tier|Controller|Qualified target|Neural mm³|CST mm³|Requests/case|Actual refusals|Preview refusals|Seconds/case¹|
|---|---|---:|---:|---:|---:|---:|---:|---:|
|low|greedy|41.03%|13.50|8.44|26.19|0|69|11.95|
|low|search|56.21%|13.50|6.75|34.75|0|97|20.68|
|low|learned-s17|73.55%|13.50|5.06|12.94|17|0|5.81|
|low|learned-s43|41.03%|13.50|8.44|10.69|37|0|4.81|
|low|learned-s71|41.03%|13.50|5.06|10.75|36|0|4.77|
|low|fixed-narrow-greedy|65.49%|13.50|8.44|32.62|0|27|15.26|
|high|greedy|41.03%|13.50|8.44|27.38|0|76|12.20|
|high|search|75.63%|13.50|8.44|60.69|0|147|33.45|
|high|learned-s17|73.55%|13.50|5.06|12.94|17|0|5.89|
|high|learned-s43|41.03%|13.50|8.44|10.69|37|0|4.88|
|high|learned-s71|41.03%|13.50|5.06|10.75|36|0|4.86|
|high|fixed-narrow-greedy|75.63%|13.50|8.44|39.88|0|44|17.44|

¹Prior controllers retain historical measured times, not newly matched speed trials.

|Tier/family|Fixed narrow|Seed17|Seed43|Seed71|Greedy|Search|
|---|---:|---:|---:|---:|---:|---:|
|low / narrow-preparation-gated|65.33%|100.00%|12.60%|12.60%|12.60%|46.83%|
|low / gated-critical-residual|57.34%|54.91%|12.25%|12.25%|12.25%|42.31%|
|low / permitted-overlap-budget|54.91%|54.91%|54.91%|54.91%|54.91%|51.34%|
|low / critical-residual|84.38%|84.38%|84.38%|84.38%|84.38%|84.38%|
|high / narrow-preparation-gated|100.00%|100.00%|12.60%|12.60%|12.60%|100.00%|
|high / gated-critical-residual|63.24%|54.91%|12.25%|12.25%|12.25%|63.24%|
|high / permitted-overlap-budget|54.91%|54.91%|54.91%|54.91%|54.91%|54.91%|
|high / critical-residual|84.38%|84.38%|84.38%|84.38%|84.38%|84.38%|

All 32 primary endpoints have checked STOP and whole-tool exit, zero committed hard violations and zero final debris/fluid. This discrete task generates no debris/fluid, so it does not demonstrate a cleanup controller. Direct neural/motor/CST counts remain separate. Strain/stress, thermal, blood and functional encounter outcomes remain unknown; earlier access costs and whole-procedure success remain unavailable.

The primary control uses 1,160 online requests (876 previews,284 execution),316 actual attempts including 32 logical STOPs,0 actual refusals and 71 preview refusals. All 45 attempts, including 13 predeclared clean timing retries, remain saved:1,604 online requests plus 786 native/adapter verification requests. The separate accepted-history audit uses 199 requests. No optimizer updates occurred.

## Development-only variation

Parents 4000/4004 were declared before outcomes. One ungated source route is offset 2.2 mm; other routes retain their neural gates. Labels, memberships, bounds and physics are unchanged. Wide-only greedy treats 14.29%/12.50%; narrow-only greedy 85.71%/87.50%; an untrained rule using current tip and source/remaining-route features reaches 100%/100%. Its actual sequence is wide work→narrow exchange→remaining work→checked STOP. Each state rule costs 13 requests; all six methods/cases replay exactly, with zero direct neural/structure removal.

This challenges a constant tip within the declared inventory. A fixed two-stage rule still solves both cases; these are not generalization evidence or learning admission. Proposed future groups are 8000–8031 TRAIN,8100–8107 VALIDATION and 8200–8215 ASSESSMENT, currently non-executable. A later goal must prospectively define richer variation/simple controls, deduplicate geometry, group related variants, fit only TRAIN, select on complete VALIDATION and assess once after freezing.

## Inspect and reproduce

Local packet: `testcases/canonical-endpoint-control-v1/diagnostic/index.html`. Adjacent CSVs contain all 192 original/control rows,160 paired differences and every family’s treatment, completion, STOP, neural/structure harm, refusals, cleanup, calls and time. Compact bindings: `docs/fixed-preparation-diagnostic.json`. Actual learned/control desktop replay is delivered separately.

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/run_fixed_preparation_diagnostic.py --output testcases/canonical-endpoint-control-v1/reproduction
```

The reproduction runner refuses existing result directories. Exact executed runner bytes remain under `diagnostic/implementation/`; only the output-preservation preflight was added afterward. Engine/controller conditions are unchanged. Existing acquired-case holds remain intact.

Two further predeclared runs apply the same actual, charged initial 0.4 mm prefix on parent 4000. The adaptive rule keeps three gated narrow-tip works first, then widens for the off-axis remainder: 100% treatment, 14 online requests and 28 replay requests. The fixed wide→narrow schedule widens immediately, treats the off-axis column, then stops with 14.29% treatment because the common available mask forbids reusing the consumed narrow preparation: 5 online and 10 replay requests. Both have clean checked STOP, zero direct neural/structure removal and unchanged source geometry/bounds. This shows current-tip/remaining-geometry dependent ordering under the finite one-use inventory, not physical impossibility of another exchange. A conditional hand-coded schedule still solves it; no learning or generalization claim follows. Exactly two outcomes were run and retained.

The two-run delta independently passed review: [state-intervention review receipt](/Users/shaf/coding_projects/RessectionLab/testcases/canonical-endpoint-control-v1/independent-review/development-state-intervention-review-v1.json) (`testcases/canonical-endpoint-control-v1/independent-review/development-state-intervention-review-v1.json`, SHA-256 `380a5ede11404e9d21ed28aa4aa1250bdd0083e441db026eaf5699f18a281803`). Its separate verification replay adds 38 physical requests and two logical STOP attempts; these are review costs, not additional development rollouts.
