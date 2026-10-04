# Native axis public preflight V2

The completed preflight took **149.376 seconds** end to end and peaked at **1.314 GiB**. All five episodes and both native geometry audits completed; there were zero gradients, final-world uses or stress-world uses.

The unchanged seed-11 RAW policy needed **48.194 seconds** for its two-world panel using one reused simulator, including 0.214 seconds of episode JSON export. The earlier 30-second online limit would not cover this observed initial panel. No future training budget is declared.

| Episode | Reset (s) | Episode, including export (s) | Decisions (s) | Transitions + next inventory (s) | Previews, including reset |
|---|---:|---:|---:|---:|---:|
| greedy | 7.111 | 14.947 | 2.188 | 11.992 | 66 |
| selection-1 | 7.222 | 14.220 | 1.312 | 12.154 | 66 |
| selection-2 | 7.298 | 14.337 | 1.296 | 12.273 | 66 |
| untrained-selection-1 | 7.236 | 16.737 | 2.704 | 13.458 | 72 |
| untrained-selection-2 | 7.350 | 16.871 | 2.750 | 13.553 | 72 |

The frozen greedy-sequence panel cost 43.077 seconds. Both panel totals sum measured resets and episodes; they exclude caller bookkeeping and repeated clone/factory setup. One initial integrity-checked clone cost 0.293 seconds. The preflight also re-reads the inventory and observation before each policy action, while the generic learner can reuse a returned observation. Cloning adds work and different read patterns may remove work; this panel is neither an exact generic-learner cost nor a rigorous lower bound.

Case/native preparation cost 0.560 seconds; cold adapter construction, including the full initial inventory, cost 8.047 seconds. Initial inventory inspection added 1.404 seconds. RAW policy import, seeded initialization and checkpoint export cost 0.741 seconds. Independent native checks cost 14.967 and 8.888 seconds (23.855 total), shared across exactly matching histories from the five completed episodes.

The initial inventory certified all 26 primary rays plus STOP. Across 21 complete inventories, there were 368 successful primary previews and no fallback attempts, consuming 84.027 seconds. The full model/cavity/proposal/phase keys identify 112 unique geometries and 256 repeated previews (58.629 seconds of observed repeated work). The initial inventory was constructed 6 times. This is a candidate for a separately checked optimization, not a predicted caching speedup.

Retained reset-local counters record 71 integrity checks and 21.198 seconds. They omit checks erased when counters reset and therefore are not a global total. Preview and integrity times overlap the phase totals above. The recorded profile duration is 148.183 seconds; the named, nonoverlapping phases leave 0.515 seconds of unseparated checks, serialization and orchestration. The worker receipt covers 148.814 seconds before its final export; launcher time additionally includes process startup, source-copy preparation and teardown.

| Complete development path | Modeled return | Target removed (mm³) | Normal removed (mm³) | Cumulative partial normal contact (mm³) | Residual target (mm³) |
|---|---:|---:|---:|---:|---:|
| greedy | 1098.77 | 1113 | 62 | 177 | 40806 |
| untrained | 441.60 | 445 | 13 | 72 | 41474 |

The two seed replays are deterministic checks, not independent patients or uncertainty samples. These are a greedy path and an unchanged random initialization, not a trained-policy benchmark. Greedy removed 2.655% of the declared radiological target. All paths stopped at the three-cut horizon. Their terminal provider ledgers still list 14 greedy-path or 20 untrained-path primary proposals; these terminal proposals were not previewed or certified, so STOP-only terminal inventories do not show proposal exhaustion. Partial contact remains separate from removed tissue.

The cancellation callback ran 1983 times for 0.009445 seconds. The deliberate request before cached inventory access returned in 9.75 microseconds without changing committed state. This does not measure worst-case cancellation inside a preview. No wall/RSS limit was reached.

A fair RAW comparison still needs a bounded measurement of a complete update batch and its accounting/factory overhead before caps are chosen. The current observations do not predict gradient throughput or the costs of other sampled paths. Any optimization must preserve source/cavity integrity, complete inventories, geometry certificates and the committed-transition cancellation contract.

This is one previously studied structural mirror-derived patient with unreviewed support and hypothetical access; motor/language outcomes, clinical probabilities and tissue mechanics remain unassessed. Other agent computation was held; normal macOS/desktop activity and OS file caches were uncontrolled. The original V1 preparation failure remains preserved separately.

Source commit: `69304172a74c50a8b83025c863edd979ff26e4fd`. Runtime: `sha256:ee931752f4af906c8561a8ea334bf4233fad0dd78b68fdc8f22846c17d4151f3`. Decision model: `sha256:1d06e7792ce78ecf7ef461ba5ff2b6f78e2367d3b3bc030a80d70b49e6e747f5`. Exact values and input hashes are in `cost-summary.json` and `report-source.json`; `report.py` accepts raw JSON or lossless gzip without changing the original receipts.
