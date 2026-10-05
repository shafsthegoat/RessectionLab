# Opt-in native-history distance batches

`independent_check_native_history` accepts `distance_backend="batch"` and
`distance_batch_size=256`. **The default remains `"scalar"`.** No existing
patient, policy, route, UI, or audit caller opts in automatically. Record the
backend, batch size, and source version in experiment metadata; scientific
certificate fields and the contained-cell model are unchanged.

Only two native-history scans use the option: completeness of recorded active
contacts, and swept-shaft collision against tissue remaining **before** a
microstep removes any cells. The complete tool's hard-exclusion checks, access
checks, corner containment, six-face frontier, and removal accounting remain in
the same order. General pose, motion, and route validators retain scalar distance
checks. No source occupancy, contact, distance, or feasibility is cached.

The kernel preserves the independent scalar piecewise-quadratic mathematics;
the original `segment_box_distance_sq` remains unchanged and decides comparisons
near a threshold. Active contacts keep their original 1e-9 squared-distance
tolerance; shaft contacts keep 1e-10. Physical transforms, source-grid bounds,
voxel enumeration order, and first physical witnesses are unchanged. The kernel
does not import planning collision helpers.

Box construction and kernel work are bounded by the selected batch size, limited
to 1–4096. Existing local candidate-index enumeration and the evaluator's full
source/cavity masks retain their original memory costs. No total-process memory
bound is implied. Cancellation is checked between bounded batches and returns
the existing incomplete `independent_validation_cancelled` result, preserving
previously checked prefix accounting. It cannot produce a complete certificate.
Invalid inputs and arithmetic failures remain errors, never successful STOP.

The initial owner run passed 59 focused checks, including existing evaluation
tests. New complete two-cut numerical histories compare exact certificates,
active-contact sets, shaft queries, and every remaining-tissue/exterior-cavity
prefix under identity, anisotropic-oblique, and mirrored LPS source frames.
Negatives retain missing-contact, uncontained-removal, hard-exclusion,
short-tip future-removal borrowing, and accounting rejection. These fixtures
are numerical unit controls, not synthetic patient training.

An independent reviewer passed another 14 controls: hand-calculated contact and
cavity prefixes with retained partial cells, disconnected frontier rejection,
first physical witness parity, scalar hard checks, and cancellation in contact
or shaft batches after an accepted cut. The separate owner/reviewer runs total
73 focused passing checks; their receipts are retained under
`artifacts/independent-native-batch-v1/` and
`artifacts/independent-native-batch-review-v1/`.

The reviewed standalone kernel control showed small non-bitwise oblique distance
differences and negative scalar-fallback timings. Those findings remain in
[the kernel report](../artifacts/independent-geometry-batch-prototype-v1/RESULT.md).
They do not establish full native-audit speed or universal numerical equivalence.

No full-size history timing has run for this integration. A future comparison
requires a prospective declaration and root release. The proposed single
analytic history is the previously certified `tilted_20_degrees-0.125mm` row;
its source, native configuration, and committed-history hashes must match the
saved control before scalar → batch → scalar timing. Do not repeat the full
16-row matrix or infer end-to-end performance from the small kernel timings.
