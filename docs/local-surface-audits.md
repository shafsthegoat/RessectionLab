# Reusing completed surface-intersection checks

The smoothed-surface worker can now reuse a completed intersection audit of
unchanged source triangles and unchanged physical coordinates. It tests dirty
triangles against a small current tree and a compact tree from the certified
base surface. Sinc filtering, face repair, displacement limits, source masks,
MRI and physical simulation retain their existing behavior.

This addresses part of the remaining viewer cost after the
[earlier allocation and duplicate-audit improvements](continuum-performance.md).
Polynomial filtering and much of its preparation still operate on the whole
layer. This change does not establish responsive replay or complete Stage 5.
First imported-branch physics admission and the independent physiological
validation gates remain open. No training or new patient acquisition is involved.

## Measured result

Three alternating paired repeats, after one warmup per implementation, compare
the original episode against `66b85c87` on the same Apple M2 Pro/16 GiB machine.
All 528 preparations matched all five buffers and non-timing diagnostics exactly.
Percentiles describe this repeated episode on this machine, rather than unseen
patients or hardware.

| CPU worker preparation | Baseline | Incremental audit |
|---|---:|---:|
| Changed brain p50 / p95, 69 samples | 3.173 / 3.417 s | 2.117 / 2.319 s |
| Changed tumor p50 / p95, 123 samples | 0.440 / 0.649 s | 0.439 / 0.643 s |
| All 66 jobs per episode, median total | 98.558 s | 75.601 s |
| Initial brain p50 | 4.574 s | 4.537 s |
| Maximum reported worker residency | 43.4 MiB | 91.2 MiB |

Changed brain preparation takes one-third less time; total preparation takes
23.3% less time. The current runs used 69 partial and three full brain audits.
Small tumor surfaces retain their full path. Whole-process peak RSS across the
paired process was 650.1 MiB; it cannot establish separate implementation peak
memory. CPU timing includes extraction, assembly, conditioning, certificate
maintenance and normals, while excluding hashes, disk writes and profiler work.
It does not measure GPU or screen presentation. Two-second changed brain work
still leaves a substantial responsiveness requirement open.

## Conditions for reuse

A worker-owned certificate is published only after the final candidate has an
explicitly completed empty new-contact result. It stores detached Float64
source/physical coordinates, an active-vertex mask, ordered source vertex IDs
for every triangle, and a compact BVH of that physical surface. Weak bindings
map immutable source-chunk buffers to their prior canonical face positions;
discarded chunk buffers are not kept alive by the certificate.

Before skipping any pair, the worker checks:

- The current stable vertex-ID mapping is injective and in range.
- Source-row ranges and triangle offsets are complete and valid.
- A known source face has the same ordered stable corner IDs.
- Known faces retain their relative canonical order. This matters because SAT
  subtracts the first triangle's origin, and floating-point symmetry is not
  assumed.
- Each retained corner has finite, bit-exact source and physical coordinates,
  including signed zero, compared with the certified base.

A face is dirty if any condition fails locally or any corner changes. Invalid
ordering/provenance, unavailable certification or insufficient memory uses the
full reference audit. New faces, removed faces, new chunks and restored source
cells cannot silently acquire certification from their position in an array.

## Why the two trees cover all required pairs

Pairs with two clean faces have the same source geometry, physical geometry,
vertex sharing and canonical argument order as a previously completed empty
audit. Every other pair contains a dirty face. Dirty/dirty pairs are checked
in a current dirty-face BVH; dirty/clean pairs use the base BVH and an explicit
mapping from certified old faces to current clean faces. Removed or changed old
faces have no clean mapping. Base node bounds are conservative for the retained
clean geometry. Each candidate pair still receives the reference per-face AABB
test, the same epsilon, shared-index exclusion, SAT and source-contact exemption.
SAT arguments are normalized to current canonical face order.

The private incremental path may return contact vertices in a different Set
insertion order. That order has no observable effect on the current fairing
consumer: all contact vertices receive their immutable original coordinates
before quantization; unique backoff counts depend on membership; component
roots remain fixed; rollback scans vertices numerically. Independent controls
compare contact membership and require exact final buffers and diagnostics.
The public full-audit function preserves its reference traversal and Set order.
An order-dependent future consumer would require a fresh equivalence review.

The fixed base can span several edits. More than 35% dirty faces takes the full
audit and may publish a new base after successful final checking. Geometry
below 100,000 faces uses the full path: measured certificate overhead exceeded
the savings for the small lesion in the development workload. These cutoffs
select exact implementations; neither changes conditioning or safety criteria.

## Memory and failure behavior

Certificate coordinates, triangle IDs, rank arrays and compact-tree arrays are
charged to the existing 128 MiB worker residency budget together with masks,
source chunks and registry data. Face AABB arrays are temporary and are not
retained in the base tree. Process RSS is measured separately; the cache budget
is not a promise about total process memory.

An exception or nonempty final result cannot publish a replacement certificate.
Every new quantized geometry is compared against the fixed certificate again,
including geometry after face/contact repair and component restoration. Small,
uncertain, over-budget and high-complexity cases retain their full reference or
existing explicit source fallback. Display outputs never expose writable
certificate buffers, and transfer does not detach worker residency.

## Verification workload

The frozen baseline is `66b85c87`, with exact worker modules retained under
ignored `testcases/outputs/simulator-qualification-20261029/local-surface-v1/`.
The source is the original 120-action UCSF-PDGM0007 operative recording, archive
SHA256 `564ef8d81d806bcadd5c825e7ed98704520cbb19d8a90049462073d8bae2ddc3`.
All actions are executed to the exact checkpoint before preparing fixtures.
Existing identical display masks are reused after matching MRI/label hashes,
shape, affine and mask bytes. No MRI volume is copied.

The full sequence has 23 brain-changing and 41 tumor-changing states. Paired
worker measurement includes all of them, separate initial preparations, one
warmup per implementation and three alternating paired repetitions. All raw
and smoothed positions/normals, triangle IDs and non-timing diagnostics must
match the baseline. Removal includes the original exposure: 779 mm³ target and
5,070 mm³ non-target at the final endpoint. CPU worker timings are separate
from native frame/HUD/GPU completion and from physical screen presentation.

Hidden native verification exercised five partial audit calls and one full brain
audit, with fixed microscope optics, 219 mm³ additional target removal, matched
GPU/HUD states, backward scrubbing and save/reopen. Nine seeks measured p50
2.231 s, p95 7.562 s and maximum 8.794 s; they include first physical prefix
admission, cached/no-op work and polling. These remain slow native updates.

Bounded independent adversarial controls cover fixed-base changes, both sides
of current face order, AABB/SAT epsilon differences, source contact exemptions,
signed zero, new/removed/reordered faces, ID reuse/duplication, budget fallback,
contact backoff and component restoration. Executable checkers, seeds, exact
commands, hashes and their scope remain in the ignored evidence directory.
These implementation checks do not validate patient mechanics or injury.

Reproduce the full acquired fixture while reusing prepared masks:

```sh
.venv/bin/python scripts/prepare_surface_replay_benchmark.py \
  --case testcases/outputs/operative-visual-20261028/operative-v4/operative-rehearsal.ressectionlab \
  --branch operative-cusa --all-actions \
  --reuse-fixture testcases/outputs/simulator-qualification-20261029/performance-v1/surface-fixture-v1/fixture.json \
  --output testcases/outputs/simulator-qualification-20261029/local-surface-v1/new-fixture

node --experimental-strip-types scripts/benchmark_surface_replay.mjs \
  testcases/outputs/simulator-qualification-20261029/local-surface-v1/new-fixture/fixture.json \
  testcases/outputs/simulator-qualification-20261029/local-surface-v1/baseline-source \
  testcases/outputs/simulator-qualification-20261029/local-surface-v1/new-paired 3
```

Choose new output directories to preserve prior evidence. Local polynomial
filtering with a proven dependency halo and faster first physical branch
admission remain subsequent work; the current full paths remain references.
