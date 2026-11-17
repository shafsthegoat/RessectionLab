# Generated sequential decision results

Retained the exact 160-row job grid across 16 held-out layouts. Validation-selected learned seeds: [17, 43, 71]. Treatment uses the full original target; declared-task completion and 100% removal are separate. Earlier access costs are unknown.

| Tier | Method | Raw target · known | Qualified target · all runs | Verified task complete | Raw100% removed | Verified STOP/exit | Neural mm³ | Calls | Online seconds |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
|high|greedy|41.0%|41.0%|43.8%|0.0%|100.0%|13.50|27.4|12.20|
|high|learned|51.9%|51.9%|58.3%|8.3%|100.0%|13.50|11.5|5.21|
|high|search|75.6%|75.6%|93.8%|25.0%|100.0%|13.50|60.7|33.45|
|low|greedy|41.0%|41.0%|43.8%|0.0%|100.0%|13.50|26.2|11.95|
|low|learned|51.9%|51.9%|58.3%|8.3%|100.0%|13.50|11.5|5.13|
|low|search|56.2%|56.2%|50.0%|0.0%|100.0%|13.50|34.8|20.68|

Campaign wall time including the preserved diagnostic and repair interval:94.9min. Charged training-call upper bound:8470; current formal requests:7085. Frozen evaluation:39.7min. Replay requests:1360.

The case-level CSV retains actual and preview refusals, interrupted requests, cleanup, direct motor/CST memberships, every endpoint and all artifact hashes. Paired differences use zero qualified treatment for failed/truncated endpoints while retaining their actual physical treatment separately. Seed ranges are descriptive; no clinical or statistical-superiority claim follows from this small generated cohort.

Paired learned minus baseline (qualified treatment fraction):

- high versus greedy: mean +10.8 points; delayed-benefit +29.1; wins/ties/losses 8/40/0.
- high versus search: mean -23.8 points; delayed-benefit -58.3; wins/ties/losses 0/31/17.
- low versus greedy: mean +10.8 points; delayed-benefit +29.1; wins/ties/losses 8/40/0.
- low versus search: mean -4.3 points; delayed-benefit -5.1; wins/ties/losses 10/21/17.

low: Mean delayed-preparation difference versus greedy: +29.1 percentage points. Improvement occurred in 1/3 seeds (seed 17: +87.4 points; seed 43: +0.0 points; seed 71: +0.0 points). Learned mean treatment is below search (-4.3 percentage points). These results do not establish robust learned superiority.

high: Mean delayed-preparation difference versus greedy: +29.1 percentage points. Improvement occurred in 1/3 seeds (seed 17: +87.4 points; seed 43: +0.0 points; seed 71: +0.0 points). Learned mean treatment is below search (-23.8 percentage points). These results do not establish robust learned superiority.

| Tier | Seed | Qualified target | Qualified delayed preparation | Verified STOP/exit |
|---|---:|---:|---:|---:|
|high|17|73.5%|100.0%|16/16|
|high|43|41.0%|12.6%|16/16|
|high|71|41.0%|12.6%|16/16|
|low|17|73.5%|100.0%|16/16|
|low|43|41.0%|12.6%|16/16|
|low|71|41.0%|12.6%|16/16|

Unmet criteria:none in the declared generated experiment. Functional consequences, patient execution, earlier access costs and general physics/RL admission remain unavailable.
