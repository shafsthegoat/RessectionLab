# Fixed-frontier clearing result — October 12, 2026

Adding the existing focused clearing actions after the retained twelve-move TRAIN045 prefix recovers **29 additional target cells (27.656589 mm³)**. The complete independently checked route earns **26.985133 more nominal return**, including all added costs. This passes the declared useful-gain criterion. It is an action-family capability result in one annotation-assisted simulation, not an RL improvement or physical/clinical validation.

| Complete endpoint | Original catalog + STOP | Focused clearing |
|---|---:|---:|
| Target cells removed | 51 | 80 |
| Target volume removed, mm³ | 48.637450 | 76.294040 |
| Fraction of full supplied target | 0.201573% | 0.316193% |
| Outside-target removal, mm³ | 0 | 0 |
| Nominal return | 48.090857 | 75.075990 |
| Tool movements | 12 | 24 |
| Tool switches | 3 | 9 |
| Complete tool travel, mm | 96.593776 | 228.050068 |
| Cumulative contacted-tissue upper bound, mm³ | 202.179206 | 237.465199 |

The larger contact bound and longer movement remain disadvantages; zero outside-target removal does not establish absence of injury. The full target denominator is unchanged at 24,128.943798 mm³. The new route ends at the absolute 24-action horizon, not a completed resection.

## What changed and what the result explains

All 91 distinct proposals in the old terminal catalog were shaft-blocked. The added family uses the existing corrected obstruction-centered, footprint and exposed-opening proposals, bounded focus allocation and certified pages. It retains the same physical anatomy, instruments, aperture, reward and full-shaft checks. The old and new catalogs have explicitly different source/task bindings; all twelve physical prefix actions, rewards and committed masks are joined, rather than pretending the historical prefix used the new catalog.

Every added action has **positive immediate reward** (minimum 0.880933). Ten blocker-removal cuts already remove 25.749238 mm³ of target; two subsequent goal cuts add 1.907351 mm³. There are eight obstruction-opening, two footprint-opening and two exposed-opening actions, with no recursive prerequisite moves or negative-cost preparation. The observed bottleneck was missing useful actions, not demonstrated failure to learn delayed credit.

Next compare frozen learned TAIL ranking with immediate GREEDY under the **same expanded certified catalog**, public focus controller, fixed prefix and horizon. The existing positive-immediate guard remains appropriate for this measured contrast. New focused-context admission and packet-based page aggregation must be explicit; the original task guards must not be bypassed. No fitting or new acquisition is justified before this matched comparison.

## Validation, cost and preserved failures

The original complete route reproduces exactly. Fresh full-tool replay validates the expanded route; selected and explored endpoints coincide. The independent saved audit passes **5,031 checks**, including original-world, prefix, packet, geometry, complete-cost and useful-gain joins. The audit's initial exact floating-point check rejected a one-ULP difference between sequential accumulation and `math.fsum` (7.1e-15). A narrow scalar correction uses the reader's existing 1e-9 tolerance, with exact identity checks unchanged and material-drift rejection tested. Original reader and failed receipt remain archived.

The successful run takes **203.790 parent seconds**, 8,357 native previews, 73 transitions and 810,942,464 bytes sampled peak RSS. Suffix selection itself takes 62.081 seconds, with 2,469 previews including its scoped work. Policy forwards, optimizer steps, checkpoint loads and feature-cache loads are all zero.

Two earlier attempts remain paid: a 4.028-second instrumentation-wrapper failure before source admission, and a 101.726-second output-cap failure after suffix exploration. The first counter remains **null/unreliable**, with zero observed admissions; it is not rewritten as a certified zero. Known previews across the latter two attempts total 13,283; all attempts total 109 transitions and 309.544 parent seconds. These totals exclude earlier tail/pair experiments and generated software controls, which retain their own receipts.

The transport repair preserves the full JSON schema through deterministic gzip, bounded at 32 MiB decompressed. It reduces the new output allowance from 6 to 3 MiB within the unchanged shared 8 MiB stage. Protocol, actual-worker instrumentation, lossless 24-action serialization, refusal and isolated-import controls pass; original fixture and output failures remain available. No numerical limit, world assumption, reward or patient split changed.

## Evidence and reproduction

[Independent interpretation](snapshot/independent/RESULT.txt) and [final audit](snapshot/independent/compact-audit-02.json) describe the scope and limitations. [Final report](snapshot/recovery-compact-01/report.json), [supervisor](snapshot/recovery-compact-01/launcher.json), [sealed plans](snapshot/recovery-compact-01/plans.json) and [source handoff](snapshot/recovery-compact/HANDOFF.txt) retain exact pins and commands.

[INVENTORY.json](INVENTORY.json) maps every archived file to its original repository path and SHA-256. Canonical code is pinned to `f64786ae34e386083dd5914bab091f355c8f89ad`. The detailed compressed selection and first-failure traces remain local at the inventoried paths; the saved audit requires those exact files. Restoring metadata alone cannot reproduce an audit or scientific run: retain the admitted local source arrays and pinned dependencies, and use a separately declared fresh output directory for any new attempt. No patient arrays, weights or bulk generated traces are committed.

The result uses exposed TRAIN045, supplied target annotations, estimated support and idealized whole-cell removal. Held SELECT037 and EVAL067 remain untouched. It supplies no measured cutting-force response, neurological outcome, scan-only generalization or held-out patient benefit.
