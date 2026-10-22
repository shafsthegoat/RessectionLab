# Rehearsal performance and remaining gates

Measured October 22, 2026 on an Apple Silicon Mac with 16 GiB RAM,
macOS 26.5.2 and Electron 44.5.1. These are bounded engineering measurements
on the saved sub-06 research case. These performance checks ran no patient RL
training or optimizer updates. Broad Python regression coverage is recorded
separately from this performance scope.

Correction for the overall task: two broad Python regression selections each
ran an existing generated pilot test containing one actual in-memory REINFORCE/
Adam update. That violated the no-RL instruction. No patient fitting or learned
checkpoint resulted. Pilot policy/optimizer tests now require an explicit pytest
opt-in and are skipped by default; the dedicated readiness and performance
receipts retain their stated zero-update scope.

## What changed

* Closed action timelines and provenance disclosures defer their contents.
  The primary timeline mounts zero hidden action buttons; opening it exposes
  all 2,148 recorded actions.
* Physical surface normals use fixed three-dimensional arithmetic without
  allocating arrays and closures for every triangle. Arithmetic order is
  preserved and compared byte for byte with the retained implementation.
* Source indexing uses a rolling window of three half-cell X planes for raster
  triangle soups. Vertex IDs and triangle order remain exact. Arbitrary order
  falls back to the retained indexing implementation. The worker-local lookup
  has a 32 MiB cap and retained 1,920,996 bytes for the measured source.
* Incremental source chunks retain stable vertex IDs rather than triangle soups.
  Only changed chunks perform coordinate lookup. Raster row assembly assigns
  canonical first-appearance IDs, preserving every vertex byte and face index.
  The 128 MiB worker budget includes masks, index rows and the vertex registry;
  exhaustion discards the optimization and rebuilds the complete exact surface.
  Unsupported Float32 half-grid ranges also retain the original path.
* Newly prepared read-only surfaces defer visual-picking BVH construction until
  an actual pick. The first pick uses the exact ordinary raycast while requesting
  acceleration. Source and editable surfaces retain eager admission. Disposing
  a never-picked frame releases its binding.
* Playback uses a wall-clock deadline that includes time already spent fetching
  and preparing preceding observations. Late playback removes idle delay;
  every recorded observation is still requested and displayed in order.
* A source/branch-bound metadata token avoids sending the complete timeline,
  full-case metadata and stable scene context on every replay frame. The client
  restores resident context before display admission. Unknown or stale tokens
  fail closed.
* Heavy flap-component records can also reuse an explicitly authenticated field
  hash from the observed-frame descriptor. Reuse requires the same source,
  branch, timeline generation and exact field hash. A changed action-boundary
  field is transmitted in full; missing or stale metadata clears the client
  acknowledgment and fails closed. This field optimization was added after the
  rolling-index pass in the table below; the final indexed-chunk pass includes
  it. Its separate field-size diagnostic receipt is also reported.

These changes do not simplify source meshes, interpolate tool/tissue effects,
change physics or skip recorded frames.

## Measured results

The source-only CPU benchmarks use the exact original bone isosurface:
1,384,642 indexed vertices and 2,754,228 triangles. Five balanced samples after
warmup produced:

| CPU component | Retained median | Updated median | Ratio |
| --- | ---: | ---: | ---: |
| Physical normals | 518.23 ms | 68.63 ms | 7.55× |
| Exact source indexing | 499.04 ms | 350.45 ms | 1.42× |

Both benchmarks require byte-identical outputs and unchanged source masks.
They exclude extraction, IPC, browser paint, GPU work and physics.

A later exact-source incremental assembly comparison toggled one real occupied
source cell through six delete/undo rounds, with five balanced samples after
warmup. Canonical positions and triangle indices matched byte for byte in every
state. Stitching triangle soups and re-indexing took **349.56 ms median**;
indexed chunk rows and canonical ID remapping took **65.24 ms** (5.36×).
Changed-chunk extraction and vertex lookup took 10.89 ms median, reported
separately. Resident worker storage fell from 108,607,180 to 84,448,748 bytes.
Initial admission increased from 2.51 to 2.89 seconds. This CPU comparison does
not establish an application playback rate.

Actual Electron playback measured action 2, observed frames 0–11, on
`full-primary`. All measured frame state IDs matched the baseline. The global
Python monitoring guard recorded zero public physics-step attempts in the
successful GUI runs. Each run checked pause, the submitted/displayed state,
complete timeline expansion and unchanged native source bytes.

| Actual Electron workload | Baseline | Renderer changes | Metadata/pacing | Rolling index | Indexed chunks/fields |
| --- | ---: | ---: | ---: | ---: | ---: |
| Play to paused frame 11, including first-frame admission | 46.14 s | 38.12 s | 30.13 s | 27.90 s | 20.85 s |
| Warm replay IPC median | 962 ms | 799 ms | 357 ms | 434 ms | 185 ms |
| Median of last 11 surface preparations | 1,075 ms | 597 ms | 548 ms | 417 ms | 165 ms |
| Warm submitted-state throughput | 0.077× | 0.106× | 0.173× | 0.188× | 0.398× |

Warm throughput is simulated elapsed time divided by wall time between the
first CPU submissions of frames 1 and 11. CPU submission is not compositor
presentation. The 12-frame sequence spans only 2.2 simulated seconds; it is not
a full-case playback benchmark.

The numerical implementation was frozen at the original authenticated cache
version. The metadata/pacing and final passes loaded the current transport
dispatch over that frozen numerical implementation. This explicitly separated
the transport experiment from numerical changes. It does **not** establish
production cache admission for the changed numerical implementation. No cache
fingerprint or verification guard was relaxed.

The final indexed-chunk pass used no field-size or delivery observer. It
submitted all 12 baseline state IDs exactly, paused, expanded all 2,148 timeline
actions, preserved the saved source and recorded zero public physics attempts.
Its final mesh job spent 58.9 ms in canonical assembly and 69.8 ms in physical
normals, with 95.54 MB retained across worker layers. The first observed-frame
handler still took 15.04 seconds. No additional full-case playback was run.

A four-frame field-size diagnostic passed with the authenticated
flap-component protocol. The first complete response was 4,720,182 bytes;
subsequent compact responses were 60,376–61,506 bytes. The 1,064,014-byte
flap-component field was present in the first response and absent from each
compact wire view. The client hydrated it before display admission, and all
four frame state IDs matched the original baseline. The run checked exact
sequence, pause, displayed/submitted identity, complete timeline expansion,
unchanged source bytes and zero physics calls. Extra per-field serialization
makes this a functional/byte diagnostic, excluded from the timing table.

These short GUI passes are individual runs. A separate canonical Python audit
occupied another core during earlier work. It had ended before the final
indexed-chunk microbenchmark and GUI pass. Timings are scoped observations, not
a controlled hardware qualification or a guaranteed speedup.

## Imported endpoint inspection

The current production application also passed a separate import-only GUI
check, without any frozen numerical substitution. The primary saved endpoint
became accepted and visible about **14.11 seconds from launch**. Its backend
`inspectArchive` request took 2.45 seconds. Switching to the vascular endpoint
took 5.08 seconds including accepted display preparation.
The process and application user-data directory were fresh; the OS file cache
was uncontrolled, so this is not a cold-disk measurement.

Both endpoint state IDs matched their saved checkpoints. The UI visibly labeled
them **Imported endpoint · unverified · read only**. Replay, Apply, Preview,
Branch, Undo and Redo were disabled. The check used only `inspectArchive`,
framed the operative field, saved screenshots and recorded zero physics calls.
This is source/checkpoint self-consistency inspection; verification is still
required before planning, replay or editing. It does not create RL state or
declare historical outcomes verified.

## Headless diagnostics and canonical audit

The latest ten warm display intervals totaled 5,019.4 ms. Aligned trace clocks
attribute mean per-interval time to 257.9 ms IPC, 174.7 ms worker preparation,
49.5 ms render CPU and 19.9 ms combined remaining gaps. The 185 ms IPC median
hides two handler tails of 634 and 457 ms. The trace does not isolate GPU fence
completion or split the small remaining preparation gap into hydration, cloning
and modeled-volume work. Playback's requested wait was already clamped to zero.
A bounded single-receipt prefetch could overlap transport with preparation, but
must use FIFO display admission rather than the existing coalescing seek queue,
which can discard superseded requests. That change is not implemented or
qualified here; IPC alone still averages longer than the 200 ms source interval.

Controlled backend measurements started from authenticated frozen legacy
endpoint states and executed real fresh `inspect` transitions. At the primary
endpoint with 2,148 prior actions, the median fell from **2,677.35 to 244.33 ms**
(10.96×); at the vascular endpoint, 304.22 to 130.11 ms (2.34×). Complete
transition signatures matched in each comparison. These are diagnostic inspect
workloads, not matched resection/field-management policy throughput. Public
`clone()` intentionally retains detached history; engine-internal transactional
steps avoid recopying committed ancestry. Owned-state hashing retains the same
canonical JSON identity and detects public edits.

The frozen optimized implementation also replayed all **73 vascular actions**
canonically in 92.624 seconds. Every complete action trace record equaled the
original canonical trace, including events, partial reasons and conservation
ledgers. The full primary audit reached its 45-minute wall cap at **885 of 2,148
actions**. Every completed trace row matched exactly, but its endpoint was never
verified and no complete cache admission was published. These frozen audit
receipts predate the later current optional-CUSA/RL/display changes. They cannot
qualify the final current implementation's complete patient replay.

Evidence: `backend-baseline.json`, `backend-digest-optimized.json`,
`vascular-canonical-after.json`, `vascular-exact-trace-comparison.json`, and
`primary-canonical-after-incomplete.json` in the ignored evidence directory.

## Gates that remain open

* **At least 1× observed-state playback is not achieved.** The final measured
  warm state throughput was 0.398×, even with 4× selected. Exact source surface
  preparation and replay reconstruction remain material costs.
* **A 50 ms p95 delivered-frame interval is not established.** A separate
  trusted camera-orbit control delivered 42 Electron frames with a 50.99 ms
  nearest-rank p95 interval. This control includes frame-subscription observer
  overhead and is separate from primary playback timing.
* Full final primary/vascular GUI playthroughs were not run. Twelve sequential
  early drilling frames do not qualify all actions or tool families.
* New numerical code requires its own canonical verification and observed-frame
  preparation. Old private caches cannot establish new numerical identity.

## Reproducing the checks

Build and run desktop regressions:

```sh
pnpm --dir desktop test
pnpm --dir desktop build
```

The final October 22 run passed 818 desktop tests: 171 main-process, 432 renderer and
215 viewer tests. Build passed with the existing bundle-size advisory.

For a bounded current-production imported-endpoint check:

```sh
PLAYBACK_IMPORT_ONLY=1 \
PLAYBACK_OUTPUT="$PWD/testcases/outputs/rehearsal-import-check" \
RESSECTIONLAB_VERIFIED_CACHE_MAX_DISK_BYTES=8589934592 \
RESSECTIONLAB_OBSERVED_CACHE_MAX_DISK_BYTES=8589934592 \
desktop/node_modules/.bin/electron desktop/electron/benchmark-surgical-playback.cjs \
  --case "$PWD/testcases/outputs/full-case-rehearsal-20261020/final/rehearsal.ressectionlab"
```

The harness requires CPython 3.12 and the published no-step monitoring helper.
It refuses public physics steps before execution. Default playback mode needs
genuinely authenticated observed frames already available for the chosen
numerical implementation; an unavailable frame fails the check. Do not treat a
cache miss as permission to run a full patient replay. `PLAYBACK_FIELD_SIZES=1`
adds per-field JSON size diagnostics, and `PLAYBACK_ORBIT=1` adds the separate
delivery observer control. Their overhead must be identified in reports.

Local evidence remains under the ignored directory
`testcases/outputs/rl-readiness-20261022/`:

* `desktop-performance-summary.json`: exact per-frame identities, timings,
  throughput definitions and open targets.
* `desktop-baseline/playback-authenticated/report.json`: original successful
  GUI baseline.
* `desktop-final-playback/report.json`: earlier rolling-index split-layer pass.
* `desktop-indexed-chunks-final/report.json`: final indexed-chunk/field-reuse
  split-layer playback pass; `settled.png` is its observed final app state.
* `desktop-heavy-field-diagnostic/report.json`: later authenticated field-reuse
  functional/byte diagnostic.
* `desktop-import-current/report.json` and branch PNGs: current-production
  import-only observation.
* `source-normals-benchmark.json`, `source-index-final-benchmark.json`,
  `source-chunk-assembly-benchmark.json`: exact-source CPU comparisons.
* `indexed-chunks-full-tests.log`, `indexed-chunks-build.log`: final indexed
  chunk regression/build receipts, including memory and transfer behavior.
* `desktop-protocol-full-tests.log`, `desktop-protocol-build.log`: final desktop
  validation, including field-token hydration.

Two failed diagnostic passes are retained separately. Their monitoring guards
prevented physics after a cache-path error and after disposable observed cache
eviction by tests. The tests now isolate cache directories. One missing legacy
action cache was regenerated through its original public engine action from an
authenticated canonical boundary; no full patient replay was used for the GUI
measurements above.
