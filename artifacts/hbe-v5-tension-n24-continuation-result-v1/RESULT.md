# N24 tension: numerical result and independent saved replay

Ordinal 10 completed one native solve at source `19889bd8`. All 61 frames and
60 steps pass the frozen numerical criteria, and an independent saved-stream
replay exactly reproduces the retained readout. Eleven of twelve numerical rows
are now complete. Physical validation remains unestablished.

| Stage | Seconds | Sampled peak RSS, bytes | Result |
| --- | ---: | ---: | --- |
| Native solve | 65.7557 | 496,304,128 | Exit 0, reaped |
| Official readout | 12.7163 | 299,008,000 | Exit 0, reaped |
| Independent saved replay | 12.9193 | 311,590,912 | Exit 0, exact equality |

The modeled N24 lower-half specimen is reflected into the full representation.
Final modeled reaction is 0.0266433 N at 0.00073601 m. The largest normalized
numerical criterion ratio is 0.000746802 and minimum sampled Jacobian is
0.878487; sampled positivity does not prove positivity everywhere. These are
simulation results, not measured tissue forces or patient outcomes.

The [independent audit](independent/REPORT.md) authenticates all eight output
files, 20 original and nine extension sources, release bindings, sidecar
accounting and clean child termination. Cumulative native-chain accounting is
11 calls, 2,229.209 seconds and 3,133,419,421 bytes; the independent saved replay
is reported separately above. The [compact index](source-index.json) preserves
exact receipts and releases. Large numerical arrays stay local and outside Git.
The final row needs its own actual-predecessor admission before execution.
