# Saved cache query reconstruction

Status: synthetic validation completed; public artifact reconstruction has not
run. The prospective declaration is
`manifests/experiments/native-cache-query-keys-v1.json`. It requires a committed
source/test receipt and a separate execution release.

The first public cache probe recorded zero hits at 32 MiB and 16,384 entries.
That result does not establish that every query was unique or that a particular
larger capacity would help. Its saved certificates offer a smaller next step:
recover the ordered query arguments and count their actual reuse without a
patient simulation or capsule-cover computation.

The four saved inventories contain 26, 22, 18 and zero accepted certificates,
with no rejected preview attempts. Their 4,406, 3,722 and 3,054 microsteps imply
22,364 calls under the frozen native code's two-call sequence. Each microstep
queries the active capsule first, then the shaft capsule. The script uses stored
active endpoints and reconstructs the shaft endpoints from stored tip positions,
axis and declared tool lengths using the same separate float64 multiplication
and subtraction. It follows the saved action list, not dictionary key order,
and requires complete ordered evidence throughout. Rejected/partial previews
would make this reconstruction incomplete and are rejected.

The output preserves exact little-endian float64 argument bytes, including
signed zero. It binds the immutable cache namespace, source/configuration,
affine, shape and frozen callsite hashes. The original cache tuple also contains
an inverse affine and derived frame values that were not serialized. Those
bytes are explicitly omitted. This is an exact variable-argument projection
within one bound invariant frame, not an invented full cache-key serialization.

The diagnostic reports distinct argument keys, occurrence frequencies and
distinct-key reuse distances over the saved sequence and its exact repeat.
Small entry-only LRU models show hypothetical counts with unlimited payload.
Their upper-bound interpretation requires the same no-bypass admission policy:
skipping an oversized query can reduce cache pollution. The original 32 MiB
probe had zero bypasses, but arbitrary other byte limits could change that.
These are not predictions for a byte-limited cache. The script counts exact
argument bytes rather than assuming hashes cannot collide.

Full payload sizes remain unknown. Stored active contact cells were filtered
by source tissue and the complete shaft covers were not retained. Neither can
be used as the full array payload. A later, separately declared pure-geometry
diagnostic could evaluate one cover per distinct reconstructed query on the
bound empty source frame, retaining only cell counts. It would need to reproduce
the observed 32 MiB eviction statistics before informing a larger-capacity
experiment. No such geometry work is authorized or performed by this script.

Synthetic validation independently instruments a real two-cut native fixture,
saves both its certificates and the observed call arguments, and compares every
byte and call position after loading the saved files. Separate tests check
floating-point edge cases, malformed/incomplete evidence, query caps, brute-force
reuse-distance agreement, and entry-only LRU counts. The executed synthetic CLI
is tested with geometry calls explicitly forbidden. Its evidence is retained in
`artifacts/native-cache-query-review-v1/`.

The script uses the standard library only. Default invocation writes a
declaration without opening the bound public inputs. Explicit execution has a
22,364-query cap, a 32 MiB decompressed-trace cap, and cooperative 60-second/
512 MiB process limits. Existing output is rejected, source inputs are hashed
before and after analysis, and partial work receives a failed status. There is
no patient loading, simulator stepping, feasibility caching, policy update,
selection score or final/stress world access.
