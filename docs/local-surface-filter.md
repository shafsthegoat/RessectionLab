# Exact polynomial filtering in a source dependency halo

The smoothed-surface worker now recomputes the windowed-sinc polynomial in a
graph dependency halo after a local source edit. It reuses unchanged **raw,
uncapped** filter results, then runs the existing displacement limits, face
repair and [certified intersection checks](local-surface-audits.md). Full
filtering remains the reference and the fallback.

This implements local polynomial work while preserving physical cavity masks,
source MRI and all physical engine semantics. Global graph/face preparation,
normals, first physical branch admission and the independent physiological
validation gates remain separate costs or open requirements. No training,
patient acquisition or clinical outcome validation follows from this change.

## Measured result

On the same Apple M2 Pro / 16 GiB machine, three alternating paired repetitions
after warmups produced 528 exact preparations against `84c94507`:

| CPU worker preparation | Baseline | Local filter + scalar quantization |
|---|---:|---:|
| Changed brain p50 / p95, 69 samples | 1.969 / 2.092 s | 1.163 / 1.291 s |
| Changed tumor p50 / p95, 123 samples | 0.421 / 0.598 s | 0.362 / 0.505 s |
| All 66 jobs per episode, median total | 70.858 s | 49.140 s |
| Initial brain p50 | 4.627 s | 4.401 s |
| Maximum committed worker residency | 91.2 MiB | 111.3 MiB |

Changed brain preparation takes 40.9% less time; total preparation takes 30.6%
less time. All 69 changed-brain preparations used local polynomial work. The
median affected set contains 32,684 vertices; the largest dependency halo has
80,144 vertices, versus about 423,000 active vertices in the brain surface.
Small tumor surfaces use the full polynomial and benefit from scalar
quantization. Initial surfaces still require full preparation.

The raw field cache peaks at 19.9 MiB. Shared worker residency remains below
128 MiB; the paired process's peak RSS was 818.6 MiB, including temporary
computation and staging allocations. That whole-process high-water mark cannot
establish separate per-implementation peak-memory reductions. All cache,
staging and eviction costs are included in worker timing; hashes, report writes
and profiling are excluded. Percentiles describe this frozen repeated workload
on this machine, not unseen cases. One-second updates and first branch admission
still leave substantial native replay latency open.

## Dependency and exactness argument

Each source edit marks every old and new triangle vertex in every replaced
source chunk. New active IDs and changed original coordinates also enter the
dirty set D. A vertex outside D retains its original coordinates, degree,
pin status and **ordered** stable-ID neighbor row. The source row stitch keeps
unchanged triangles in canonical order. Duplicate neighbors and their floating
point addition order are preserved.

For a polynomial of degree K, changed recurrence terms travel at most one graph
edge per order. Outputs outside K hops of D retain their exact prior raw result.
The changed outputs depend on at most a 2K-hop neighborhood of D. The worker
initializes T0 from original coordinates throughout that halo and computes Tj
only where `distance(D) <= 2K-j`. Every needed preceding neighbor term and
T(j-2) value is then available. Coefficients accumulate in the reference order.
This is a graph dependency boundary; a physical-distance cutoff is not used.

The original full recurrence is retained. Uncertain provenance, dense halos,
invalid IDs, insufficient budget, changed coefficients/registry, nonfinite data
or missing dependencies take that full path. Signed-zero coefficient identity
is preserved. The local path is currently admitted with the same large-surface
eligibility gate as the contact cache; small lesions retain full filtering.

## Source lineage and transactional state

Each worker source registry has a monotonically increasing update generation.
The old/new incident-vertex snapshot belongs to that generation transition.
A raw cache can be reused only for the same generation or its immediate
successor. An update becomes invalid before extraction starts and remains
invalid after a failure; a fresh complete source admission restores eligibility.
Retries cannot consume or lose the transition's seed snapshot.

This handles an otherwise subtle failure: a high-complexity display fallback
can return before capturing a raw result. The following edit's dirty vertices
cover only its own transition. A generation gap therefore forces full filtering
instead of reusing an older field with incomplete change coverage.

The cache stores detached original and raw Float64 coordinates plus active flags
by historical source vertex capacity. Capture happens before clipping/repair;
publication happens only after successful fairing. A failed fairing cannot
replace the committed raw field, and transferred display buffers cannot detach
it. Previously repaired or capped positions never serve as raw polynomial data.

Shared admission reserves both old and staged raw fields. Disposable other
layers yield residency when required. Source chunks, registries, raw caches and
contact certificates remain under the existing 128 MiB committed worker budget.
Temporary computation/frame-staging allocations and total process RSS are
measured separately; the residency budget is not a total-process memory limit.

## Remaining global arithmetic

Quantization also avoids per-vertex arrays, slices and reducers. Scalar
expressions retain each reducer's initial zero, left-to-right addition order,
and Float32 assignment before physical-coordinate conversion. This change has
its own exact prepared-buffer comparison against the retained implementation.
Sinc coefficients, iterations, caps, normals, contact/face repair and restoration
criteria do not change. Graph and face preparation are still global.

## Qualification workload

The retained baseline is `84c94507`. Source modules and executable evidence are
under ignored `testcases/outputs/simulator-qualification-20261029/local-filter-v1/`.
The existing original 120-action UCSF-PDGM0007 fixture and its shared compressed
display masks are reused; no source dataset or MRI is copied.

One warmup per implementation and three alternating paired repetitions cover
all 23 brain-changing and 41 tumor-changing states, plus initial preparations.
Every raw/smoothed position/normal buffer, triangle ID and non-timing diagnostic
must match. Exact removal includes the historical access cost: 779 mm³ target
and 5,070 mm³ non-target. CPU worker time is separate from native GPU/HUD
completion and physical screen presentation.

Independent controls cover raw byte parity, the K/2K border, cancellation-sensitive
neighbor order, cuts/pins, signed zero, coefficient/registry changes, sparse/dense
and budget fallback, capture ownership, actual source-chunk seed coverage,
successive cuts/undo/reappearance, skipped generations and extraction failure.
They also compare complete prepared buffers and diagnostics with the retained
baseline under reflected and sheared affines. These are implementation controls,
not measured patient mechanics or injury validation.

Run the paired workload from the primary checkout with a fresh output path:

```sh
node --experimental-strip-types scripts/benchmark_surface_replay.mjs \
  testcases/outputs/simulator-qualification-20261029/local-surface-v1/operative-fixture-v1/fixture.json \
  testcases/outputs/simulator-qualification-20261029/local-filter-v1/baseline-source \
  testcases/outputs/simulator-qualification-20261029/local-filter-v1/new-paired 3
```

The hidden operative verifier separately checks microscope state, source counts,
optics, GPU/HUD synchronization, scrubbing and save/cold reopen. The complete
research goal remains open; implementation parity does not close specimen,
device, patient anatomical prediction or physiological RL gates.

The native run passed five local polynomial preparations, fixed optics and
219 mm³ additional target removal, with unchanged source and synchronized
GPU/HUD state. Nine seeks measured p50 1.951 s, p95 5.501 s and maximum 5.649 s;
these include physical prefix admission, cache/no-op work and polling. Cold
reopen preserved the exact cursor, anatomy counts, tools and optical settings.
All 1,119 desktop tests and the production build passed. Native responsiveness
therefore remains distinct from the lower CPU worker latency.
