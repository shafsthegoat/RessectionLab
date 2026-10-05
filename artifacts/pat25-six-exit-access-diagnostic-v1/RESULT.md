# PAT25: six existing hypothetical exits

**The original STOP-only inventory was access-dependent within the declared estimated-support geometry.** Five alternative existing axis exits produced accepted native previews under the same source support, annotation, two generic tools and checks. No alternative was selected as a replacement. No cut, transition, policy evaluation, search or training was performed.

The single completed development diagnostic inspected all six exits in the frozen order below. Each supplied all 78 unique previews: 13 source columns × two tools × three endpoint families. There were no missing, duplicate, capped or unavailable slots; the total was 468. The original selected access reproduced its saved inventory and model identity exactly.

| Outward source-axis exit | Accepted previews | Shaft rejected | Reach rejected | Target cells in actor crop |
|---|---:|---:|---:|---:|
| x−, original | 0 | 78 | 0 | 16,526 / 16,526 |
| x+ | 78 | 0 | 0 | 16,526 / 16,526 |
| y− | 62 | 10 | 6 | 0 / 16,526 |
| y+ | 58 | 18 | 2 | 16,504 / 16,526 |
| z− | 78 | 0 | 0 | 16,526 / 16,526 |
| z+ | 72 | 6 | 0 | 0 / 16,526 |

These signs identify source-grid axes, not anatomical direction labels. [Comparison plot](comparison.png); [machine-readable summary](compact-summary.json).

## Why the original access failed

All 78 original shaft failures occurred at **microstep 0, insertion fraction 0**, before any preview removal. Its local outward neighbor `[91,139,118]` is outside estimated support and connected to exterior free space. Thus an enclosed void does not explain this result.

The 120 mm backward shaft nevertheless re-enters estimated tissue behind that locally empty access. For the central fine-tool opening, the trace records 61 blocked support cells spanning source indices `[21,139,118]` to `[86,139,118]`. The wide tool records 567 blocked cells within `[20,138,117]`–`[89,140,119]`. These are cell bounds, not a claim that every cell inside the bounds is occupied. Changing the distal endpoint cannot clear a collision already present at entry.

All six local outward neighbors were externally connected. That one-cell connectivity condition therefore does not establish whole-shaft clearance. Every shaft rejection at the other exits also occurred at entry. The original shortest-local-exit rule did not test the backward shaft before choosing an access; this diagnostic exposes that limitation without changing the rule or the original result.

## Coverage and interpretation

The source nominal target remains available and unchanged at every exit. **Availability is different from actor visibility:** the fixed access-centered 64³ crop contains no target at y− or z+, although native geometry and proposal generation retain the full source grid. The y+ crop omits 22 cells. These accesses cannot be treated as equivalent policy inputs without addressing that explicit information gap.

Even the alternative accepted-sweep bounding boxes exclude 11,575–13,362 of the 16,526 target centers. They are optimistic initial envelopes, not unions of removable cells or proof of eventual resection. Passing a preview does not mean it removes target. All previews remain uncommitted; no executed trajectory was independently replayed. Support is estimated and unreviewed, access hypothetical, and function, vessels, mechanics and clinical utility remain unvalidated.

## Resources and provenance

Worker time was **105.60045 s**; supervisor wall time **106.11235 s**. Worker peak and supervisor-sampled peak RSS were both **960,036,864 bytes** (about 0.894 GiB; 507 samples). CPU thread settings were one; the bound was 6 GiB and 180 s including reserved supervision/termination time. The run exited 0 without timeout or retry; sampled RSS can miss transient peaks.

Manifest SHA256: `a9c7f36f7dc71a83723f169125b80e8f0be058b61f527efc08d067914359d123`. Diagnostic source SHA256: `a88f8176b8c79d528b15207811d99b9d061a2fcda19029dfbf058d61acf97f6a`; all 61 bound source files matched.

The intended pre-run commit/archive/release block failed its whitespace check, but a subsequent shell line launched the reviewed, prospectively hash-bound worker. This is **not a precommitted immutable-archive execution**. The [launch deviation](launch-deviation.json) has SHA256 `d6a560ea280170ab23e7a53fc915aae7417f959e4ef44103fe7b4b0f3dc0499f`. Source archive `3444b6ab5c28ec6ce2c9448bc415d6d9588df0d1fab755f15c804f910d28a34a` was created after terminal completion at 04:41:01 UTC; source/deviation commit `9618f09` also followed the run. [Preservation receipt](post-run-source-receipt.json) and summary retain exact identities. `report.py` reproduces the summary and plot from saved JSON only.
