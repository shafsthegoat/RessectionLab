# Native patient follow-on after initial-geometry caching

This development run uses the preserved `native-ucsf0004-v1` runtime and runner,
with only the verified cache change in `native_simulation.py`. The design records
commit `3a5581cf63250f55fc08a4b5fe6c57df97b5de8a`, all snapshot file hashes, the
same seeds and budgets, and the unchanged model assumptions. The original study
is untouched. Final evaluation and stress worlds were not opened.

Each seed had the original 30-second cooperative optimization-and-selection
budget. Simulator, training configuration, and initial policy checkpoint hashes
match the original run exactly.

| Seed | Updates, original → cached | Optimization transitions | Selected score, original → cached |
| --- | ---: | ---: | ---: |
| 11 | 7 → 14 | 37 → 68 | 245.24 → 245.24 |
| 23 | 6 → 11 | 35 → 64 | 171.42 → 171.42 |
| 47 | 5 → 11 | 30 → 63 | 139.62 → 245.16 |

All actors changed. Seed 23 retained its initial checkpoint because no tested
checkpoint improved its selection score. Seed 47 improved after more updates;
its selected score still trails SEARCH and GREEDY, which both score 245.24.
Their measured computation times were 6.22 and 0.75 seconds respectively.
This run supplies no evidence of an RL advantage over search.

Per-arm cold preparation cost 1.92–1.95 seconds. Bounded training took
30.06–30.36 seconds, including 11.97–13.06 seconds of selection-panel work.
Selected and initial candidate extraction added 1.02–2.06 seconds per seed.
Independent source-grid validation added 19.50 seconds for the frozen candidate
set; all native audits passed. Costs are separated in `summary.json`.

The implementation change improved useful throughput in this workload, but the
comparison uses historical timings under uncontrolled system load. It is not a
randomized runtime benchmark. This remains one structural development patient
with unavailable motor/language evidence, hypothetical access, and unvalidated
tissue mechanics; the scores are declared simulation objectives.

`frozen-workspace/` and `comparison/source-snapshot/` preserve the executed source.
`comparison/` contains the source and world manifests, selected checkpoint
identities, optimization histories, candidate freeze, and independent audits.
The ignored local `.pt` files remain available in the scratch directories;
checkpoint hashes are recorded in the committed JSON results.
