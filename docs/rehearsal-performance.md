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
* Raw bone surfaces retain exact normals for stable source vertices. Changed
  chunks invalidate both old and new incident vertices; their sums are replayed
  in complete raster face order. Unaffected normals retain their original bits.
  Normal arrays and dirty flags share the existing 128 MiB retained-cache cap.
  If they do not fit, preparation uses the complete exact normal calculation.
  Disposable scratch and transferable output buffers are additional.
* Newly prepared read-only surfaces defer visual-picking BVH construction until
  an actual pick. The first pick uses the exact ordinary raycast while requesting
  acceleration. Source and editable surfaces retain eager admission. Disposing
  a never-picked frame releases its binding.
* Playback uses a wall-clock deadline that includes time already spent fetching
  and preparing preceding observations. Late playback removes idle delay;
  every recorded observation is still requested and displayed in order.
* Automatic replay has a separate FIFO lane with one prefetched compact wire
  receipt, limited to 8 MiB serialized UTF-8 data. JavaScript object overhead is
  additional. Transport overlaps the active frame's preparation; future geometry
  is not prepared until that frame is submitted. Manual commands wait behind
  the in-flight read, while pause discards unadmitted data. Source generations,
  failures and pause/restart leases invalidate stale receipts. Automatic replay
  never enters the coalescing manual-seek queue.
  Full reconstructed-prefix read-only receipts may use the same single slot and
  byte ceiling; their source/branch/authority is checked before buffering.
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

A further comparison on the same source reused unaffected raw bone normals.
Canonical positions, indices and normals matched byte for byte on every
delete/undo state. Assembly plus physical normals fell from **157.09 to 92.44 ms
median** (1.70×). Normal-cache residency increased retained storage from
84,448,748 to 111,711,724 bytes, within the same 134,217,728-byte cap. The first
normal-cache fill cost 248.18 ms versus 172.65 ms for a complete calculation;
the benefit applies to subsequent local edits. Conditioned surface smoothing
uses its existing calculation.

Actual Electron playback measured action 2, observed frames 0–11, on
`full-primary`. All measured frame state IDs matched the baseline. The global
Python monitoring guard recorded zero public physics-step attempts in the
successful GUI runs. Each run checked pause, the submitted/displayed state,
complete timeline expansion and unchanged native source bytes.

| Actual Electron workload | Baseline | Renderer changes | Metadata/pacing | Rolling index | Indexed chunks/fields | FIFO prefetch | Cached bone normals |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Play to paused frame 11, including first-frame admission | 46.14 s | 38.12 s | 30.13 s | 27.90 s | 20.85 s | 18.72 s | 18.61 s |
| Warm replay IPC median | 962 ms | 799 ms | 357 ms | 434 ms | 185 ms | 202 ms | 207 ms |
| Median of last 11 surface preparations | 1,075 ms | 597 ms | 548 ms | 417 ms | 165 ms | 160 ms | 127 ms |
| Warm submitted-state throughput | 0.077× | 0.106× | 0.173× | 0.188× | 0.398× | 0.687× | 0.720× |

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

The later prefetch pass CPU-submitted all twelve exact
`(branchId, position, frameIndex, stateId)` identities in order. Explicit tuples
distinguish observations that legitimately share a state hash. The harness
counts submitted frames rather than returned lookahead and clicks the ordinary
pause control synchronously at target frame 11. Warm frames 1→11 covered two
simulated seconds in 2.911 wall seconds (0.687×). There were twelve successful
frame responses plus one later cancelled, unadmitted request during sidecar
shutdown. No patient step was attempted; the saved source stayed unchanged.
The first prefetch GUI attempt failed on an IPC whitelist rejection of an
accidentally forwarded scheduler field; that failure is preserved separately.
The normalized cursor contract and actual-hook regression fixed it before the
successful replacement pass.

The later cached-normal GUI pass used the same twelve tuples and all baseline
state IDs, with zero physics calls and all eight harness checks passing. Warm
frames 1→11 took 2.780 wall seconds (0.720×). The last mesh job took 105.4 ms
and retained 122,806,540 bytes across layers, including normals. Warm replay IPC
still dominated. The nearest-rank p95 of ten warm **CPU-submission state gaps**
was 587.5 ms; this is neither compositor delivery nor a 50 ms frame-interval
qualification. There was one cancelled unadmitted read at sidecar shutdown.

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
This is source/checkpoint self-consistency inspection. Full verification is
still required before planning or editing. The separate read-only prefix mode
below can reconstruct saved actions without admitting the complete archive.
Neither inspection nor prefix reconstruction creates RL state.

## Current-production read-only prefix replay

The imported endpoint now offers **Replay from start · read only**. It starts
from the original source and protection contract, executes saved public actions
as requested, and captures their actual substeps. Imported checkpoint state
never seeds this reconstruction. The banner reports the reconstructed prefix
and original action count; the remainder stays unverified. Editing, planning,
branching and RL export remain unavailable in this mode. Explicit full-archive
verification is still a separate operation.

Each request computes at most four new actions. Far seeks stop at the actual
reached boundary and report the budget stop. Actions execute once per owner;
frame reads and retained boundary reads execute no physics. Evicted data becomes
unavailable rather than silently recomputing old actions. Rewinding to zero
creates a new bounded owner. Cancellation or a failed action preserves earlier
committed prefix states and reports the failed index.

The desktop owner retains at most 64 MiB of compressed observations, 16 MiB of
compressed checkpoints and 4 MiB of computed receipts. The original immutable
snapshot is borrowed from the existing loaded bundle, with a separate 128 MiB
input limit. Source anatomy, pure geometry/protection runtime, decoded views
and other transient allocations are additional. Source object, action log,
protection and implementation identities bind the owner; no verified/observed
disk cache or editable surgical session is populated. Even a complete matching
endpoint would retain read-only authority.

The first production GUI qualification used the complete unchanged original
archive and current Python code, without frozen numerical substitution. It
reconstructed exactly three original actions and CPU-submitted frames 0–11 of
action 2, with every state ID matching the preserved baseline. Warm reads made
zero physics calls. All fourteen harness checks passed, including read-only
authority, original 2,148-action count, no canonical admission and unchanged
source/Python bytes. Only the three reconstructed timeline entries were shown.

Prefix initialization took 7.00 seconds, advancing to boundary 2 took 3.70
seconds, and preparing action 2 with its observations took 41.77 seconds. Warm
frames 1→11 covered two simulated seconds in 7.361 wall seconds (**0.272×**),
with a 787.7 ms p95 CPU-submission gap. This run exposed eleven prefetched full
responses being discarded and refetched: 24 completed frame responses for
twelve submissions, including one unadmitted lookahead completed at shutdown.
This measurement is a separate production workload from the frozen-numerics
0.720× experiment above.

A matched GUI run then retained those full read-only prefix responses within
the existing single-slot 8 MiB prefetch limit. Duplicate fetches fell **11→0**;
twelve submitted frames required thirteen completed responses, including one
unadmitted shutdown lookahead. Warm throughput improved **0.272→0.524×**
(1.93×), with warm elapsed time falling from 7.361 to 3.814 seconds. All fourteen
checks passed again, with the same original archive, Python source manifest,
harness, exact twelve state IDs and three preparation steps. Warm reads again
made zero physics calls. The warm CPU-submission gap p95 fell to 413.4 ms;
worker preparation median was 115.4 ms and demanded IPC median 301.2 ms.
Cold action-2 preparation remained 42.17 seconds. This fix closes duplicate
fetches, not the 1× or compositor targets.

The owner retained 33,481,180 bytes of frames, 432,316 bytes of checkpoints and
268,522 bytes of receipts. It shared 111,442,521 bytes of existing immutable
input; peak resident memory for the complete Python process was 854,032,384
bytes. These are one-run observations, not a whole-application memory bound.
The matched prefetch run retained the same private bytes and recorded a
792,887,296-byte whole-Python peak RSS.
The earlier 16 MiB-window attempt failed honestly during action 2 and retained
its earlier committed prefix. The bridge's explicit 64 MiB window is within
the replay class's unchanged hard ceiling.

This qualifies a bounded current-production replay path, not the full primary
or vascular operation, a microscope demonstration, compositor timing, or the
remaining playback targets.

## Headless diagnostics and canonical audit

The latest ten warm display intervals totaled 5,019.4 ms. Aligned trace clocks
attribute mean per-interval time to 257.9 ms IPC, 174.7 ms worker preparation,
49.5 ms render CPU and 19.9 ms combined remaining gaps. The 185 ms IPC median
hides two handler tails of 634 and 457 ms. The trace does not isolate GPU fence
completion or split the small remaining preparation gap into hydration, cloning
and modeled-volume work. Playback's requested wait was already clamped to zero.
The implemented bounded prefetch overlaps this serial chain while preserving
FIFO display admission. The older numerical implementation's IPC alone still
averages longer than the 200 ms source interval; overlap cannot remove that cost.

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

A separate **current-source** prefix probe canonically replayed only the
original first three legacy actions in memory, then prepared action 2 in
isolated observed caches. It executed six preparation step calls (three
canonical, two boundary replay, one observed action). Twelve timed display reads
ran under the published no-step monitor, matched all original frame IDs, and
took **98.7 ms median / 99.2 ms mean / 102.8 ms p95**. Total evaluation was
33.02 seconds, including 10.22 seconds canonical preparation and 18.02 seconds
observed-action preparation. It wrote no duplicate patient workspace or source
file. Its shortened archive is not a full-case admission or GUI qualification.

One further genuine original drilling action from an authenticated current
boundary took **6.17 seconds** without a profiler. A separate cProfile execution
produced the exact canonical receipt and endpoint too. In that instrumented
sample, the action called `touched` 8,170 times, including 7,270 bone predicates
inside contact localization, with 7,555 capsule-cover misses. This identifies repeated contact
queries as an engineering target; profiled inclusive times are not uninstrumented
throughput. The failed first diagnostic compared unlike receipt/history shapes;
its failure remains preserved, followed by the corrected exact-history check.

Two subsequent exact optimizations reuse immutable candidate-grid slabs and a
sorted representation of each occupancy object's immutable cleared-cell set.
The spatial cache retains at most 8 MiB of candidate grid/world-center payload
plus its existing 8 MiB of final covers; it caches no occupancy or feasibility.
Every query still computes its own distances and current material membership.
The cleared-set representation has a separate 8 MiB cap per occupancy object;
small queries and unusual or publicly replaced sets retain their prior lookup.

Three matched fresh original action-2 trials measured **5.899 → 4.511 seconds
median** (1.31×). Complete result, history, snapshot and metrics were identical,
including the canonical endpoint hash. Candidate-grid reuse alone measured
5.173 seconds. This single early drilling action does not qualify the requested
10× matched full workload or sustained candidate-policy throughput. Independent
geometry and ownership probes found no mismatches. A subsequent current-source
prefix check again matched all twelve original observed IDs with zero read-time
steps; its wall times overlap the broad regression run and are not a matched
performance comparison.

Evidence: `backend-baseline.json`, `backend-digest-optimized.json`,
`vascular-canonical-after.json`, `vascular-exact-trace-comparison.json`, and
`primary-canonical-after-incomplete.json` in the ignored evidence directory.

## Gates that remain open

* **At least 1× observed-state playback is not achieved.** The final measured
  frozen-numerics transport experiment reached 0.720×; the separate current
  production read-only prefix path reached 0.524×, even with 4× selected.
  Exact source surface preparation and replay reconstruction remain material
  costs.
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

The latest October 22 validation passed 864 desktop tests: 181 main-process,
464 renderer and 219 viewer tests. Build passed with the existing bundle-size
advisory.

The latest Python regression run passed **733 tests, with seven pilot tests
skipped**, in 115.10 seconds, with two existing Gymnasium advisories. It includes
surgical/full-case tests, independent geometry checks and the read-only prefix
contracts. The pilot opt-in was not enabled.

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
Import-only and default playback refuse public physics steps before execution.
Default playback mode needs
genuinely authenticated observed frames already available for the chosen
numerical implementation; an unavailable frame fails the check. Do not treat a
cache miss as permission to run a full patient replay. `PLAYBACK_FIELD_SIZES=1`
adds per-field JSON size diagnostics, and `PLAYBACK_ORBIT=1` adds the separate
delivery observer control. Their overhead must be identified in reports.

For the separate bounded current-production prefix qualification, choose a
fresh output directory and allow time for the three cold action preparations:

```sh
PLAYBACK_PREFIX=1 PLAYBACK_MAX_MS=90000 \
PLAYBACK_OUTPUT="$PWD/testcases/outputs/rehearsal-prefix-check" \
desktop/node_modules/.bin/electron desktop/electron/benchmark-surgical-playback.cjs \
  --case "$PWD/testcases/outputs/full-case-rehearsal-20261020/final/rehearsal.ressectionlab"
```

This mode permits exactly the original actions 0 and 1 while seeking boundary 2,
then action 2 while preparing its first observed frame. It refuses unexpected
or additional step attempts before execution. Subsequent frame reads must be
step-free. It rejects frozen Python/bridge substitution and additional timing
observers. The benchmark pauses at CPU-submitted frame 11 and checks private
retention and absence of canonical admission. It does not run the complete
saved operation or any learning.

Local evidence remains under the ignored directory
`testcases/outputs/rl-readiness-20261022/`:

* `desktop-performance-summary.json`: exact per-frame identities, timings,
  throughput definitions and open targets.
* `desktop-baseline/playback-authenticated/report.json`: original successful
  GUI baseline.
* `desktop-final-playback/report.json`: earlier rolling-index split-layer pass.
* `desktop-indexed-chunks-final/report.json`: final indexed-chunk/field-reuse
  split-layer playback pass; `settled.png` is its observed final app state.
* `desktop-prefetch-qualified/report.json`: later FIFO-prefetch pass, with exact
  submitted tuples distinct from fetched/cancelled lookahead.
* `desktop-incremental-normals-qualified/report.json`: latest cached-bone-normal
  GUI pass; `settled.png` shows its accepted final app state.
* `incremental-normals-benchmark.json`, `incremental-normals-full-tests.log`,
  `incremental-normals-build.log`: exact-source CPU and final desktop checks.
* `prefetch-qualified-full-tests.log`, `prefetch-owner-build.log`: final prefetch
  regression/build receipts, including tuple/IPC and source-owner guards.
* `current-prefix-profile/report.json`: bounded current-source transport probe;
  `current-drilling-profile-v2/report.json`: genuine-action profile.
* `candidate-grid-comparison.json`, `candidate-occupancy-comparison.json`:
  matched fresh drilling comparisons with exact complete transition payloads.
* `candidate-occupancy-review.json`: independent old/new ownership comparisons.
* `current-prefix-geometry-optimized/report.json`: final optimized-source
  functional prefix check, with concurrent-regression timing caveat.
* `candidate-final-no-learning-tests.log`: final numerical regression receipt.
* `desktop-heavy-field-diagnostic/report.json`: later authenticated field-reuse
  functional/byte diagnostic.
* `desktop-import-current/report.json` and branch PNGs: current-production
  import-only observation.
* `desktop-current-prefix-qualified/{report,analysis}.json`: current-production
  read-only prefix GUI pass, including completed shutdown lookahead accounting.
* `desktop-current-prefix-prefetch-qualified/{report,analysis}.json`: matched
  full-prefix prefetch improvement, with no duplicate fetches and exact state IDs.
* `prefix-full-receipt-{renderer,main}-tests.log`, `prefix-viewer-tests.log` and
  `prefix-full-receipt-build.log`: final desktop regression/build checks.
* `current-primary-prefix-replay/report.json`: retained failed 16 MiB-window
  diagnostic; `current-primary-prefix-replay-64m/report.json`: bounded successful
  backend diagnostic of the explicit bridge window.
* `prefix-full-no-learning-tests.log`: 733 passing Python tests and seven
  skipped pilot tests, including prefix ownership and bridge regressions.
* `source-normals-benchmark.json`, `source-index-final-benchmark.json`,
  `source-chunk-assembly-benchmark.json`: exact-source CPU comparisons.
* `indexed-chunks-full-tests.log`, `indexed-chunks-build.log`: final indexed
  chunk regression/build receipts, including memory and transfer behavior.
* `desktop-protocol-full-tests.log`, `desktop-protocol-build.log`: final desktop
  validation, including field-token hydration.

Two earlier failed diagnostic passes are retained separately. Their monitoring guards
prevented physics after a cache-path error and after disposable observed cache
eviction by tests. The tests now isolate cache directories. One missing legacy
action cache was regenerated through its original public engine action from an
authenticated canonical boundary; no full patient replay was used for the GUI
measurements above.
