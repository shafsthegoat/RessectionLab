# Independent prospective public cache runner review

The repaired runner passed **25 independent synthetic tests in 1.47 seconds**.
The owner separately ran seven tests. The numerical engine, adapter, proposal
provider, geometry, evaluator and cache prototype remain byte-identical to
`39fc208`. The reviewed runner SHA-256 is
`5ba33718a1f58cf1f29197887753e9ad6d043d79c8d3f7ad5544d4189e6c79c1`.
The complete hashes and reviewed scope are in [review.json](review.json).

Five regressions were actually reproduced against the preserved initial
runner, with **five expected failures in 0.33 seconds**:

- The shared uncached setup did not enforce its declared preview count.
- A measurement-hook cleanup error omitted the phase failure receipt.
- The launcher could accept a result payload marked failed.
- The launcher did not bind the worker status's runtime identity.
- Malformed result JSON escaped without publishing an explicit failed launcher receipt.

The old source, reproduction script and raw negative output are retained.
All five cases now pass. The first repaired test run also exposed an error in
the review harness: comparing serialized lists directly with an in-memory
tuple history. Its failed output is retained; comparing complete canonical
histories corrected the test without changing production code.

The positive fixture executes real native geometry on a 7×7×7 synthetic grid,
including a paid removal. Reference, cold, warm and reversal traces must agree
exactly on inventories, certificates, features, rewards, source masks and cavity
identity. Deliberately changing these outputs, the declared work count, source
identity or bundle identity prevents result publication. Cancellation after an
actual commit retains the committed history. Mocked launcher cases cover
timeouts, hard termination, malformed/non-object receipts, status/hash/runtime
inconsistency and existing-output protection.

Audit callbacks are stubs in these orchestration tests. They check that both
hooks are restored before auditing and that a rejected audit withholds success;
they do not independently certify the geometry checker. The actual runner calls
the unchanged independent native checker four times after restoration and
requires that cache-call counters remain unchanged across those audits.

No public patient bundle was opened, no public cache benchmark ran, and no
gradient or final/stress world was used. Public execution remains a separate
parent-authorized step using committed frozen source. Any later timing claim
must retain shared setup, cold-cache construction, reset, transition,
verification, export and audit costs. The 32 MiB cap covers retained array
payload; the separately measured 6 GiB process limit is cooperative and includes
other memory. This review establishes no patient speedup or clinical claim.

Reproduction uses
`artifacts/public-capsule-probe-independent-v1/reproduce_initial_failures.py`
for the deliberately failing old source, and
`tests/test_public_capsule_probe_review.py` for the repaired source. The
`write_review_receipt.py` helper verifies the saved outcomes and protected-file
identity before regenerating the receipt.
