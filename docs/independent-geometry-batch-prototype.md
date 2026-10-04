# Independent segment/box batch prototype

This is an unwired numerical control. `evaluation.py` and its scalar
`segment_box_distance_sq` remain unchanged. The prototype lives under `scripts/`
so it does not change the source inventory of the frozen PAT05 profile.

The fixed-physical-scene discretization control spent 151.95 seconds in
independent audits and 17.16 seconds in native strokes. That identifies a phase,
not a measured function-level bottleneck. Static inspection found scalar
per-cell distance loops in both contact completeness and shaft collision checks.
No full audit speedup has been measured for this prototype.

`segment_box_distances_sq` evaluates one finite segment against physical
axis-aligned boxes. It follows the independent scalar routine's face-crossing
intervals and quadratic stationary points. Inner products use `numpy.vecdot`;
active coordinates retain their original order. Neither the prototype nor its
probe imports planning collision helpers. Grid rotation, mirrored orientation,
spacing, source identity, and causal occupancy remain the caller's responsibility.

The contact helpers preserve input order and accept the caller's original squared
distance tolerance. The existing checker uses different tolerances for shaft
collision (1e-10) and active contact (1e-9); these are not merged. Near a threshold,
the original scalar routine makes the decision. The proximity guard is not a
formal floating-point error proof. Unit comparisons permit 2e-14 relative and
absolute squared-distance error across runtimes; the probe additionally reports
bitwise agreement and requires identical contact indices and first witnesses.

Work arrays are limited to 256 boxes by default, with a hard limit of 4096 per
batch. Input boxes must already be float64 arrays, avoiding an implicit full
volume conversion. The returned distance array is O(N); returned contact indices
are O(number of contacts). No feasibility, scene, distance, or occupancy cache is
used. Inputs are borrowed synchronously and must not be changed during a call.
Cancellation raises before a result is returned, with checks around bounded
chunks. Nonfinite inputs, reversed boxes, invalid limits, and overflowing
arithmetic are refused. Early collision still validates every input box.

The owner controls cover analytic face/edge/corner distances, random numerical
coordinates, degenerate segments and cells, one-ULP tangency, both original
tolerances, anisotropic and rotated/mirrored grid reexpression, large translations,
ordering, bounded chunk size, and cancellation. Grid reexpression here is not a
replacement for the separately completed fixed-scene revoxelization control.
These are geometry unit controls, never patient training data.

The prospective probe `scripts/probe_independent_geometry_batch.py` fixes five
1024-box workloads, two repetitions in scalar → batch → scalar order, contact
checks, and separate 1024/16384-box allocation observations. It writes source
hashes and arguments before measurement, refuses reused output directories,
preserves failed/partial rows, and uses a 45-second process alarm. Tangency and
large coordinates deliberately exercise potentially costly scalar fallback.
Its timing scope excludes transformations, voxel enumeration, connectivity,
full-tool checks, and native histories. The single released probe completed;
[results and limitations](../artifacts/independent-geometry-batch-prototype-v1/RESULT.md)
include the floating-point differences and fallback costs. No probe was retried.

Integration requires separate review and authorization after the PAT05 profile.
Any later evaluator change must retain the original scalar oracle and compare
complete histories: acceptance/reason/witness, contacts, every prefix of remaining
tissue, all-corner containment, aperture/hard exclusions, frontier connectivity,
and prior-tissue swept-shaft checks. This prototype alone establishes none of
those integration equivalences.

## Narrow integration proposal — not implemented

After the PAT05 v2 source archive is frozen, move the reviewed numerical kernel
into `src/resectionlab/independent_geometry_batch.py`. Add an explicit opt-in
backend to `independent_check_native_history`, initially leaving the scalar
backend as default. Batch only its active-contact completeness scan and its
prior-tissue shaft scan. General pose/motion and route validators retain their
current scalar implementation; full-tool hard-exclusion checks still run.

Keep the current local-frame transforms, broad-phase index bounds, `np.argwhere`
order, distinct tolerances, and source-cell witnesses. Construct box arrays in
bounded chunks from the current remaining mask. Do not cache scene occupancy or
move any clearance check after removal. Translate the batch cancellation exception
into the existing incomplete `independent_validation_cancelled` result. The
original scalar distance routine remains the threshold oracle and reference
backend. A kernel backend label is diagnostic metadata, not a change to source,
tool, reward, tissue, or clinical assumptions.

Before changing the default, compare both backends on complete accepted and
rejected native histories, including missing contacts, uncontained removal,
disconnected cells, hard obstacles, and the short-tip future-removal borrowing
counterexample. Compare certificates and first witnesses, every contact set,
and each prefix of remaining tissue; test cancellation without partial success.

One subsequent timed analytic control can reconstruct the already completed
`tilted_20_degrees-0.125mm` history from the frozen discretization declaration.
Require its saved source, config, and committed-history hashes to match before
timing. Then compare complete scalar → batch → scalar audits on that identical
history, counting preparation separately and preserving a bounded failure rather
than increasing the cap. The historical scalar audit was 43.44 seconds; that is
context, not a new speed comparison. No full matrix or patient run is proposed.
