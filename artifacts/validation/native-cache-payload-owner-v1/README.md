# Prospective payload-sizing review

Only synthetic computation is authorized in this slice. No saved public-query
payload sizing has run. The script and declaration require a separate parent
release after review and commit.

`initial-owner-tests.txt` records 31 tiny synthetic tests passing in 0.36 seconds.
They compare the integer LRU model against the real `ExactCapsuleCoverCache`,
including both phases, complete byte counters, empty covers, zero caps,
oversized bypass, entry pressure, signed zero and five affine frames.

Independent review then reproduced three contradictory launcher-authority
failures and a fourth failure in which changing the baseline-comparison file
during later capacity analysis could still yield completed authority. The
original script and failing logs are retained under
`../native-cache-payload-independent-v1/`. These are actual observed failures,
not hypothetical concerns. Repairs reject those states and recheck the hashes
captured immediately after output files are written.

A fresh-process full synthetic CLI test was added after the initial owner run.
It uses a 7×7×7 synthetic frame and real frozen geometry, with the actual 512 MiB
RSS guard isolated from the parent test process's memory history. Its initial
fixture incorrectly combined two cache namespaces with different capacity
settings; production rejected it before geometry import. The test-only repair
and original failure are retained separately from the production failures.

The final combined run passed all 85 tests in 1.03 seconds. Its exact output is
`final-combined-tests.txt`; `owner-review.json` binds the source, declaration,
tests, limits and preserved failure receipts. This includes the fresh-process
test and all 53 independent cases. No public-query sizing result follows from
these synthetic checks.
