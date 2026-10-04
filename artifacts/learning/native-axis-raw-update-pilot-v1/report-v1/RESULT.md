# Native axis RAW update pilot

The pilot completed one actual actor update from two complete optimization episodes, with complete two-world selection before and after, and six accepted native history certificates. Initial mean return was 441.60; updated mean return was 441.60. Selection retained the initial checkpoint; the completed actor update did not improve the selected result.

| Episode | Phase | Return | Transitions / cuts | Target removed (mm³) | Normal removed (mm³) | Newly charged partial normal contact (mm³) |
|---|---|---:|---:|---:|---:|---:|
| 1 | initial_selection | 441.60 | 3 / 3 | 445 | 13 | 72 |
| 2 | initial_selection | 441.60 | 3 / 3 | 445 | 13 | 72 |
| 3 | optimization | 122.65 | 3 / 3 | 124 | 4 | 49 |
| 4 | optimization | 771.55 | 3 / 3 | 782 | 45 | 137 |
| 5 | updated_selection | 441.60 | 3 / 3 | 445 | 13 | 72 |
| 6 | updated_selection | 441.60 | 3 / 3 | 445 | 13 | 72 |

A separate setup reset supplied dimensions and had zero decisions, transitions or score; it is excluded from the six complete episodes above. No optimization batch or selection panel was discarded. Partial contact is separately charged exposure and does not count as removed tissue. The two selection seeds are deterministic replays, not independent patients or uncertainty samples.

| Cost | Seconds | Scope |
|---|---:|---|
| Full launcher | 221.139 | Source copy, subprocess, worker and parent lifecycle |
| Worker | 220.335 | Preparation, training, validation and publication through its measured receipt |
| Preparation | 9.504 | Includes 8.084s cold native setup |
| Full trainer call | 179.115 | Initialization, online work and final learner/accounting exports |
| Learner initialization | 8.292 | Includes zero-transition shape probe and 0.298s clone |
| Online optimization + selection | 169.590 | Declared 300s cooperative cap |
| Initial selection panel | 51.889 | Both worlds, actual clone/reset/logging costs included |
| Updated selection panel | 63.424 | Both worlds, actual clone/reset/logging costs included |
| Optimization and other online work | 54.277 | Residual includes batch, Adam, intermediate exports and bookkeeping |
| Final checkpoint export | 0.001 | Last atomic tensor write |
| Full native validation | 29.389 | 3 unique checks covering all six frozen histories, plus guards/receipt exports |

These phases are nested and must not be added. Peak worker RSS was 2.219GiB. The online clock overshoot was 0.000s. Seven actual factory clones were recorded; their individual times are retained in the machine-readable report. Separate reset, decision serialization, journal fsync, and actor/critic compute timings were not instrumented, so their individual bottleneck contributions cannot be inferred.

The six final-transition counter snapshots record 432 native previews taking 98.248s and 66 integrity checks taking 19.607s. These are observed reset-local prefixes, not whole-process totals: setup, clone and subsequent model/metrics checks can lie outside them. The updated panel was 11.535s slower despite the same selected actions; the available timers do not isolate why. No cache speedup or logging bottleneck is inferred from repetition alone.

The recorded pre-clipping total gradient norm was 5117.17; the post-clipping actor norm was 4.57679. Initial and latest actor hashes differ. The selected initial checkpoint remains distinguished from the latest trained checkpoint; post-update decisions are bound to latest weights regardless of selection.

The report joins 18 actual decision records to their authenticated outcomes. It verifies 49 frozen executable input files, original checkpoint byte hashes, final publication authorities, and all six frozen history/certificate identities. It performs no new policy forward, simulator episode, random draw or gradient update. Independent source-cell/forward reconstruction remains documented in its separate audit receipt.

This is one previously studied structural mirror-derived development patient. Brain support and hypothetical access remain unreviewed; motor/language injury probabilities and tissue mechanics are not validated. No final or stress world was opened. Other agent computation was paused during the timed run, while ordinary desktop/OS background load and file-cache state were uncontrolled. This pilot supports integration and cost accounting, not an efficacy claim or a new training-budget choice.

Reproduce with `report_native_axis_pilot.py --run <attempt> --baseline <execution-baseline.json> --output <fresh-report-directory>`. The reader accepts exact `.json.gz` replacements and refuses raw/compressed disagreement. `report-source.json` records uncompressed source byte hashes and the reporting-script hash.
