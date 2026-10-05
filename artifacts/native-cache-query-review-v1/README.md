# Prospective query reconstruction: synthetic evidence

The final suite passed **53 tests in 1.30 seconds**: 15 owner cases and 38
independent cases. Earlier incremental passing outputs are retained.
Source, declaration, tests, fixture and
unchanged numerical-module hashes are bound in `owner-review.json`.

Independent review reproduced four failures: duplicate accepted action IDs,
and runtime mismatches in the launch-source, worker and result authorities.
The initial source/tests and raw negative evidence remain in
`artifacts/validation/native-cache-query-reconstruction-independent-v1/`.
All four are repaired. A later reporting-string assertion failed because it
expected “upper bounds”; that output is also retained separately. The final
wording explicitly restricts those bounds to the same no-bypass admission
policy, since skipping oversized queries can change cache pollution.

`synthetic-instrumented-fixture.json.gz` contains a saved 9×9×9 native phantom
with two committed cuts and 114 independently observed capsule-query argument
records. The test instrumentation records the actual start/end/radius bytes
before forwarding every call to the original geometry function. It also checks
that the complete scene descriptor remains identical. Reconstruction from the
saved certificates matches all 114 records in their original order.

Other tests preserve signed zeros and adjacent float values; check the frozen
two-operation shaft arithmetic against NumPy; reject partial, missing, reordered
or inconsistent evidence; compare reuse distances with a separate brute-force
definition; and check entry-only LRU against those distances. The synthetic
executed CLI succeeds with native and geometry cover calls made forbidden.
The default CLI does not read the analysis inputs and existing output is refused.

No public query trace has been reconstructed. No new public geometry,
simulation, learning, final world or stress world was executed. The diagnostic
remains prospective until its source/declaration/review are committed and the
parent explicitly releases execution. An independent review is recorded
separately. Full payload sizes remain unknown; this slice cannot recommend a
byte capacity or claim a speedup.
