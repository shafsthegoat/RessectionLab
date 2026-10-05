# Prospective single-history comparison

**V2 completed once; V1 remains a preserved harness failure.** This numerical control compares the opt-in batch
backend with the unchanged scalar default on one previously certified analytic
history. It loads no patients and performs no learning or full matrix rerun.

The declaration is
`manifests/experiments/independent-native-batch-history-v2.json`; the runner is
`scripts/compare_independent_native_batch.py`. Root must commit/review these
files and release an immutable source archive before execution. V1 remains
failed and will not be rerun. Its [preserved result](../artifacts/independent-native-batch-history-v1/RESULT.md)
records a successful first scalar audit followed by a tuple/list certificate
comparison failure; batch and the final scalar phase did not start.

V2 changes certificate equality to exact canonical JSON bytes so the real
audit dataclass's tuple fields agree with their saved JSON list representation.
The actual `NativeRemovalAudit.to_dict()` serialization roundtrip is now tested;
one-ULP numerical, source-identity, rejection, and integer/float representation
changes remain rejected. No numerical tolerance, model code, scene, trace logic,
workload or resource cap changes. The V2 declaration pins V1's failed receipts.

The released V2 run from immutable commit `ba43763` completed with exact
certificate and full prefix/contact/witness trace parity. Complete instrumented
audits took 43.007 s scalar, 4.016 s batch, and 43.176 s scalar, within the
original caps. This is one fixed numerical history and runtime, with no patient
or learning timing claim. The scalar production default remains unchanged.
See the [V2 result and original receipts](../artifacts/independent-native-batch-history-v2/RESULT.md).

The only scene is the saved `tilted_20_degrees-0.125mm` row: the same physical
half-space, tool, entry, target, and 0.0625 mm microsteps. One native stroke is
reconstructed using the original generator. Source, configuration, and complete
97-microstep history hashes must match the prior report **before timing**. The
saved prior independent certificate must also match every completed phase.

Phase order is fixed: scalar → batch → scalar, with batch size 256. Identical
observational hooks record each active-contact set, all cell-query endpoints and
first physical witnesses, and the remaining-tissue/connected-cavity digest after
every microstep. The hooks delegate to the unchanged checks and restore original
callables on success or failure. Exact trace and certificate equality is required;
backend query counters must also demonstrate that the middle phase used batching.

The reported audit times include the common tracing overhead. Hash/trace capture
time is measured separately, without using subtraction to assert speed. Common
setup through history export and later per-phase gzip export are separate from
audit timing. The parent checks the actual saved history and trace payloads,
not just a successful process exit or claimed digest.

Limits are one worker, numerical-library thread controls set to one, 140 seconds
cooperative time, 150 seconds parent wall time, and 3 GiB worker memory. No macOS
CPU-affinity guarantee is claimed. The original native reconstruction API has no
cooperative callback, so the parent watchdog bounds that single setup call;
checks run before reconstruction and before audit phases. During auditing,
existing microstep and new bounded-batch cancellation points remain active.
The parent samples memory with a bounded subprocess timeout and kills/reaps a
worker on supervision failure. Peak RSS and sample limitations remain explicit.

The process retains partial phase reports and failure records. There is one
attempt, with no retry or cap increase. A cancelled, mismatched, unrestored, or
unaudited phase cannot count as complete. Gzip payloads are deterministic and
bounded to 64 MiB uncompressed when verified. Fresh output is mandatory.

Initial static review found supervision and backend-evidence gaps before any
real execution. The original runner and mocked failures are retained under
`artifacts/independent-native-batch-comparison-v1/` and
`artifacts/independent-native-batch-comparison-review-v1/`. These are harness
validation records, not benchmark results. A later successful comparison would
support only this one fixed history/runtime; it would not establish patient,
learning, universal-speed, or clinical-safety claims.
