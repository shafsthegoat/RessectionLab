# N16 tension: numerical result and independent saved replay

Ordinal 9 completed one native FEBio solve from source commit
`e804aa88446d609b7e135973a1159f463b767cf2`. All 61 frames and 60 steps pass the
frozen numerical checks. The separate saved-stream replay reproduced the whole
readout exactly. This completes ten of the twelve planned native rows; two
remain. No measured-force accuracy, patient interaction or clinical validity is
established.

| Stage | Elapsed seconds | Sampled peak RSS, bytes | Result |
| --- | ---: | ---: | --- |
| Native solve | 10.5123 | 129,007,616 | Exit 0, reaped |
| Official readout | 4.3795 | 196,820,992 | Exit 0, reaped |
| Independent saved replay | 4.3729 | 201,703,424 | Exit 0, exact equality |

The N16 lower-half native specimen is reflected to the full representation.
The final **modeled** reaction is 0.0267640093 N at 0.00073601 m; minimum sampled
Jacobian is 0.8855858292. Neither is an observed physical measurement. The largest
normalized numerical criterion is 0.0011041911. Host availability was 65/65/64%
with normal pressure at the three launcher gates. The separate cumulative
ledger charges ten native calls, 2,139.1564 seconds and 2,935,674,436 bytes,
including earlier supplements and one full sidecar reserve for this row.

The [independent report](independent/REPORT.md) verifies all 20 native and four
extension source hashes, both releases, all eight native output hashes,
resource accounting and child cleanup. [Exact compact evidence](source-index.json)
preserves receipts and release bytes; native streams, mesh and material-response
data remain outside this package.

Three negatives remain visible. Initial template derivation incorrectly assumed
a full native domain and refused before execution; the corrected helper uses the
frozen run specification. The first read-only feasibility attempt then refused a
direct import after its predecessor checks; the versioned repair uses the
reviewed private loader without widening the old import allowlist. That second
feasibility attempt passed in 6.177 seconds. Finally, independent replay succeeded
and wrote its durable result, but the enclosing audit command exited 1 on an
unrelated final bookkeeping import. Its supervisor receipt still says `started`;
the report and metadata preserve this discrepancy. No native or replay retry
was used to hide these failures.

The next row needs a separately reviewed continuation adapter that charges this
row's sidecar overhead once. This result is not an execution release for it.
