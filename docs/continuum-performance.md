# Acquired-case simulator performance

The acquired UCSF-PDGM0007 continuum episode now executes with **8.9% higher
headless throughput**, while changed smoothed brain preparation is **2.49×
faster**. Every recorded physical state/receipt and all five surface buffers
remain exact. Changed brain preparation still takes about three seconds;
responsive local conditioning remains open.

This is implementation qualification. It supplies no new tissue measurement,
device calibration, intraoperative prediction or clinical outcome validation.
The [simulator evidence gates](simulator-qualification.md) still hold substantial
RL training. No optimizer or policy training ran in this work.

## Frozen workload and measurements

[Protocol](../manifests/experiments/continuum-performance-v1.json) binds the
existing `continuum-v1/rehearsal-v6/continuum-rehearsal.ressectionlab` archive
(`0098232cceb946cbc5cccf00ca81b3b8b5c768c726d379fc8e8dbb503cd13127`),
its acquired source identity and the `sustained-hold` branch. Baseline revision
is `02bca78a`. Retained baseline modules run in the same process, with no branch
switch or separate checkout. Inputs and local evidence live under ignored
`testcases/outputs/simulator-qualification-20261029/performance-v1/`.

Hardware: Apple M2 Pro, 16 GiB RAM, macOS 26.5.2; Python 3.12.12, Node 22.17.1,
Electron 44.5.1. Each primary comparison has one warmup per variant and three
paired repetitions with alternating order. Timings exclude independent equality
hashes, report writes and profiler overhead.

| Measurement | Baseline | Revised |
|---|---:|---:|
| Headless complete episode, median | 30.713 s | 28.197 s |
| Accepted/partial actions per second | 0.944 | 1.028 |
| Episodes per second | 0.03256 | 0.03546 |
| Changed brain CPU preparation, p50 / p95 | 7.632 / 7.799 s | 3.069 / 3.155 s |
| Initial brain CPU preparation, p50 / p95 | 8.771 / 9.093 s | 4.381 / 4.403 s |
| Changed tumor CPU preparation, p50 / p95 | 1.166 / 1.188 s | 0.596 / 0.610 s |

The 29-action headless episode contains 26 complete commits and three checked
partial collision stops. Every intermediate full state and receipt matches;
the final saved snapshot matches exactly. Removal remains **568 mm³ target and
4,940 mm³ non-target**, including historical exposure; residual target is
18,800 mm³. The improvement does not make that access cost disappear.

Renderer numbers include worker extraction, assembly, conditioning and normals.
The brain surface grows from 843,012 to 845,776 triangles at the actual exposure
cut. Unchanged masks reuse the prior display and are omitted. There are three
changed-brain and six changed-tumor timing samples, so p95 describes this small
frozen workload. These CPU measurements exclude Electron, GPU completion and
physical screen presentation. Separate hidden native receipts verify GPU/frame/
HUD agreement, source counts, optics, STOP cleanup, saving and cold reopen.
The native sequence's 11 branch/seek operations measured p50 0.316 s and p95
17.571 s, with a 29.379 s first sustained-branch admission. That sequence
includes physics verification, cached updates and no-op seeks; it is separate
from worker latency. First branch admission is another unresolved responsiveness
cost, even when subsequent cached seeks are fast.

Whole-process peak RSS was 337.4 MiB for the headless paired process and 614.5
MiB for the renderer paired process. Worker residency peaked at 43.4 MiB,
below its existing 128 MiB budget. The reports include per-job post-return
memory samples. Shared-process high-water marks cannot establish separate
baseline/revised peak-memory reductions; none is claimed.

## What changed

Headless rollback previously recursively deep-copied every immutable scalar in
continuum corner/history fields. It now uses the existing detached JSON-graph
copier with a shared memo. Aliases, cycles, negative zero, ordinary subclass
fallback and the existing read-only flap-component exemption survive. Exact
type-identity guards avoid invoking subclass metaclass equality. Reference
neighbor pairs and their physical lengths now share the existing continuum
operator's source/topology/support signature; scalar norm arithmetic and pair
iteration order remain unchanged.

A separate cProfile pass on the slow CUSA action attributes rollback time
from 3.213 to 1.741 seconds and continuum tick time from 4.905 to 3.932 seconds.
Inclusive profiler costs overlap and include substantial profiling overhead;
the paired unprofiled episode is the throughput result.

The viewer CPU profile attributed about 70% of samples to intersection auditing.
The BVH now reuses its traversal stack and uses scalar overlap/shared-index
checks, avoiding arrays, callbacks and typed-array views at each query. Axis
order, epsilon, traversal, triangle predicate, adjacency exclusion, source
contacts and returned Set order are preserved. Fairing also memoizes audit
results within a single quantized physical-geometry revision. Every geometry
update invalidates that memo, including face/contact repair and component
restoration. The same geometry no longer receives a duplicate audit.

Sinc coefficients, iterations, physical displacement/radius limits, face-area
and orientation checks, contact repair, component rollback, source triangles,
MRI and occupancy are unchanged. This is still whole-layer conditioning.

## Verification and reproduction

312 selected Python tests passed, including rollback, cancellation, source
invalidation, mechanics, CUSA, vascular conservation, position and replay cache.
All three desktop suites and the production build passed. A new exhaustive
triangle-pair checker independently tests BVH candidate rejection.
The original 120-action acquired recording and three saved vascular plans also
replay to their exact full checkpoints with unchanged source archives.

A bounded independent reviewer preserved executable controls under
`independent-review-v1/`: 110 adversarial BVH fixtures plus six epsilon controls
matched the retained baseline; 150 smoothing fixtures matched exact output and
diagnostics, including 106 repairs and 15 component restorations. All 155
changed-geometry audit transitions were checked. Omitted scans had identical
physical geometry and ordered audit results.

Run from the primary repository checkout, choosing fresh output directories:

```sh
.venv/bin/python scripts/benchmark_continuum_replay.py \
  --case testcases/outputs/simulator-qualification-20261029/continuum-v1/rehearsal-v6/continuum-rehearsal.ressectionlab \
  --baseline testcases/outputs/simulator-qualification-20261029/performance-v1/baseline-source \
  --output testcases/outputs/simulator-qualification-20261029/performance-v1/new-headless

.venv/bin/python scripts/prepare_surface_replay_benchmark.py \
  --case testcases/outputs/simulator-qualification-20261029/continuum-v1/rehearsal-v6/continuum-rehearsal.ressectionlab \
  --output testcases/outputs/simulator-qualification-20261029/performance-v1/new-fixture

node --experimental-strip-types scripts/benchmark_surface_replay.mjs \
  testcases/outputs/simulator-qualification-20261029/performance-v1/new-fixture/fixture.json \
  testcases/outputs/simulator-qualification-20261029/performance-v1/baseline-source \
  testcases/outputs/simulator-qualification-20261029/performance-v1/new-surface 3
```

`scripts/profile_surface_replay.mjs` additionally captures an inspector profile.
It deliberately prepares all frozen states, including unchanged masks, for
attribution; its totals are not an application latency benchmark. The primary
paired worker benchmark omits unchanged jobs. The hidden native harness is
`desktop/electron/verify-continuum-rehearsal.cjs`; it records complete branch
selection and seek latency through GPU completion, including cache hits/no-ops,
and never shows its window. Its polling latency is explicit.

## Remaining work

Whole-layer filtering and spatial audits still dominate the revised viewer.
Dirty regions need a proven dependency halo and reusable unchanged geometry/
spatial checks, with the current whole-layer path retained as an exact reference.
Headless occupancy hashing and the local solver also remain material costs.
These timings do not establish independent-episode scaling or a sufficiently
qualified environment for meaningful patient-specific RL comparisons.
