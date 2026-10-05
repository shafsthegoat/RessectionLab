# Terminal fine-mesh timeout; measured comparison not reached

The single continuation from commit `a9f2783adb9ad4dbc41d9eafc29e4c11f0313388` stopped at the unchanged 90-second per-solver limit. No retry, solver change, cap increase or measured-curve access occurred.

The original `compression:N4:S60:reference` output was rebuilt with the reviewed nine-digit header parser and matched the independently accepted complete readout exactly. Its original execution remains charged. The next case, `compression:N8:S60:reference` (768 hex elements), completed in **35.9023 s** and passed the declared individual numerical checks across 61 states.

`compression:N12:S60:reference` (2,592 hex elements) timed out. Its receipt records **90.0140 s including cleanup**, exit `-9`, kill sent and child reaped. Seven of 60 nonzero steps completed; the eighth had begun. Partial primitives and logs remain retained. This is an incomplete solve, not a numerical or physical pass. Seventeen later cases were not executed. Total solver calls across both attempts are three: one original reused call, one new complete call and one new interrupted call.

The continuation's supervised worker took **130.1044 s**; the outer launcher took **132.1285 s**. The original full parent interval of 4.265553792 s was conservatively debited once, giving a continuation worker cap of 895.734446208 s. Observed peak process-group RSS was **283,656,192 bytes**; the output watcher observed at most **56,009,458 bytes** in the shared raw root and **15,988,417 bytes** in an active case. The per-case timeout ended the work; aggregate time, memory and output limits were not triggered. Sampling can miss brief peaks.

The retained completed N8 solver log reports 35.8333 s solve time, including 32.4757 s in its linear solver (about 90.63%). The fine case has no completed timing summary. These logs support investigating the existing linear-solver bottleneck; they do not establish a scaling law or performance of another backend.

No reference refinement/step/scale group passed because the required cases were incomplete. Calibration and held-out access flags remain false; no access ledger, fit, prediction freeze or descriptive measured comparison exists. No tissue-material or clinical validation claim follows from this attempt.

`post-execution-integrity.json` confirms unchanged hashes for 109 baseline inputs, 4,426 archived source files, 52 prepared files, all 14 files from the original failed attempt, and the unopened measured-data archive. The 23 new raw files (24,529,982 bytes) are indexed and retained read-only outside Git. `outcome.json`, both solver execution receipts, `medium-readout-summary.json`, `saved-performance-diagnostic.json`, and the saved state/supervision records provide compact evidence. Hashes cannot replace missing raw files. The original parser failure remains preserved separately.
