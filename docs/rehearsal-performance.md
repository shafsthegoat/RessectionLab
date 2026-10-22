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

A later normal pass avoids scanning faces outside replaced chunks and their
immediate source-grid neighbors. Old and new chunk vertices mark the union;
remaining contributions retain complete raster row/triangle order. Initial
admission, affine changes, lost ownership bookkeeping and uncertain inputs use
the complete scan. The selection bitset adds 37 retained buffer bytes within
the same 128 MiB budget; it does not alter output indexing or picking.

After five complete delete/undo warmup cycles per implementation, eight balanced
paired source states measured extraction plus assembly/normals at
**100.77→92.83 ms median** (7.9% less CPU time). Normal preparation fell
29.63→20.45 ms, with scanned faces reduced from 2,754,216 to 349,396 and identical
incident-face sums. Positions, indices and normals matched byte for byte for
all 19 first-fill/warmup/measured pairs. First fills still use the full scan;
retained buffers measured 111,711,761 bytes. The initial shorter-warmup receipt
is preserved separately with its JIT qualification. This CPU result does not
establish compositor cadence or the full application playback rate.

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

## Optional close instrument view

With an accepted inserted instrument, choose **Close tool view** in Rehearse.
The camera pins to that exact accepted tip in RAS millimeters, using recorded
effective activation to select the working hand. Its vertical field spans
18–42 mm according to the declared working radius. **Refocus accepted tool**
frames the latest accepted tip; ordinary orbit remains available. The camera
does not follow future frames automatically.

The optional context-transparency control adjusts display alpha by declared
material. It changes no ownership, masks, state, collisions or rewards. Source
slices stay linked to the tip. **Restore previous view** restores the saved
camera, appearance and cursor; an absent camera uses the explicit anatomy-fit
fallback. Switching source object, branch, scene or authority invalidates the
old restore snapshot. Whole-head navigation and the original default playback
camera remain available.

Actual-app qualification used the preserved primary archive at action 2,
observed frame 11. The accepted legacy cranial drill was recognizable at a
42 mm field, with its shaft/fluted end and nearby voxel bone/target visible.
Linked source slices moved to the accepted tip and restored afterwards. Twenty
harness checks passed, including exact state identity, absence of replay
commands during the display controls, and completed settled GPU frames after
close and restore. Before/close/restored PNGs were inspected independently.
The settled before/restore PNG files are byte-identical.
This is an optional close-up check on early drilling; it does not qualify a
complete microscope workflow, all tools, or the complication/recovery sequence.

The first check's 100 ms screenshot delay was shorter than the renderer's
150 ms adaptive interaction period. Its eighteen functional checks passed,
but the coarse restore image did not establish settled appearance. That
receipt remains preserved. The corrected check waits for a noninteractive,
nonadaptive CPU frame and its matching completed GPU fence after each
appearance change, with those listeners added only after primary playback
timing. This is settled GPU completion and captured-pixel evidence, not a
compositor timing measurement.

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

Prefix frames have their own display acknowledgment names, `prefixToken` and
`prefixFieldTokens`. Canonical commands and boundary requests reject these
arguments. A private owner nonce binds the generation to the original source,
action log, protection and implementation; the generation changes when the
computed prefix grows. Only an exact generation permits omission of resident
timeline, scene, catalog and protection descriptors. An exact observed
`flapComponents` content hash additionally permits omission before view copying.
Missing, stale or changed acknowledgments receive complete data. Authority and
progress always travel in full. The client checks the raw receipt before
buffering, then restores resident data and rechecks the complete computed
timeline only when that frame is demanded. These tokens grant no engine or RL
authority. Restored private frame/checkpoint state is still hashed and compared
before its checked identity is reused for display, avoiding a second hash.

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

A subsequent current-source bridge comparison measured the prefix display
stream itself. It reconstructed the same three original actions once, then
made 48 step-free reads: full and compact sequences in both orders, covering
all twelve original frames twice per mode. Restoring compact data reproduced
the complete full response's view, computed timeline, verification, playback,
branch, position and archive metadata exactly. Median handler time fell from
**191.6 to 67.9 ms** (2.82×); median wire size fell from **1,305,570 to 77,113
bytes**. Source files and loaded archive bytes stayed unchanged, with no
canonical admission or learning. This is an IPC diagnostic, not GUI throughput.
It took 70.38 seconds overall, including 49.80 seconds of cold observation
preparation. Field digest creation adds preparation work; warm reuse must not
be presented as eliminating that cold cost.

The subsequent actual Electron prefix-stream pass CPU-submitted all twelve
original frame identities and passed all fourteen checks. Warm frames 1→11
covered two simulated seconds in **1.892 wall seconds (1.057×)**, up from
0.524× on the previous full-response production path. The wider submitted
frame 0→11 window covered 2.2 simulated seconds in 2.1432 wall seconds (1.027×).
Both windows exclude cold action preparation. Median warm IPC fell to **78.5
ms**, with a 101.4 ms p95, and median wire size fell 94.1% to 76,339 bytes.
Worker preparation median remained 116.4 ms. CPU-submission gaps had a 172.2 ms
median and **258.9 ms p95**, not a compositor/delivery measurement. There were
thirteen completed responses for twelve submissions, with no duplicate fetches
and one unadmitted lookahead.

Cold action-2 preparation took **50.22 seconds**, compared with 42.17 seconds
in the previous individual GUI pass. Prefix initialization took 7.11 seconds
and seeking boundary 2 took 3.82 seconds. These cold costs remain an explicit
blocker for sustained operation. The archive stayed read only at 3/2,148
actions, with exactly three preparation steps and zero warm steps. Retained
frames, including stream metadata, totaled 33,609,379 bytes within 64 MiB;
checkpoints and receipts retained their earlier sizes. The whole Python process
peaked at 756,056,064 resident bytes in this run.

The changed Python display transport has a new source manifest, preserved
throughout the run. The harness also changed its post-playback check to count
actual hydrated UI timeline entries against complete prefix progress, because
compact wire responses omit the timeline. Timers, submitted-frame matching,
pause and the strict three-step monitor were unchanged. The short warm-rate
result does not qualify full-operation playback, cold preparation, microscope
readability or the 50 ms compositor target.

Cold capture profiling subsequently isolated Python JSON iteration as a material
cost. A bounded native-fragment encoder preserves insertion order, escaping,
numeric spelling and raw/compressed caps without allocating a complete large
JSON string first. Fragments have a conservative 16 KiB bound and at most 512
entries; large strings use bounded pieces. Unsupported/custom and circular
values retain the standard encoder path. Nine balanced pairs on a saved
checkpoint measured **122.1→54.8 ms** (2.23×), with identical compressed bytes,
JSON bytes and decoded values. The first scalar-only candidate regressed and
remains preserved as a failed performance experiment.

A frozen `1d032b46` versus current-source three-action probe then matched all
**152** observed action-2 frame records, its canonical prefix endpoint and all
outcome statuses exactly. Cold preparation fell **49.43→40.70 seconds**
(1.21×); total evaluation fell 69.95→61.11 seconds. Both runs prepared only
three saved actions and performed zero physics during reads. These receipts
preserve the smaller end-to-end gain separately from pure serialization speed.

Compression callbacks also repeatedly traversed the retained frame graph to
check its memory cap. The owner now retains an exact graph count, refreshed on
publication and every eviction/rebuild. Proposed publication is still measured
as a complete map, including table capacity and shared references; staged data
uses its existing independent admission. A representative generated 152-frame
ownership graph reduced encoding with these callbacks **201.9→54.2 ms** (3.72×),
with identical compressed/JSON bytes and exact fresh ownership checks. This
microbenchmark used copies of one preserved checkpoint, with zero engine
actions; it is separate from actual capture timing.

The final combined encoder/accounting probe reduced the same action-2 cold
preparation **49.43→19.88 seconds** (2.49× over the frozen source, 2.05× over
the encoder-only version). All 152 observed frame records, the canonical prefix
endpoint and outcome statuses remained exact across all three versions.
Overall evaluation, including 48 balanced full/compact reads, fell
69.95→40.34 seconds. Each source version was measured once end to end; these
are not repeated timing trials. The original archive stayed unchanged, only
three saved preparation actions executed, and display reads performed zero
physics steps. Whole-operation and matched headless targets remain separate.

The final current-production Electron check independently measured the same
action's cold handler **50.22→20.22 seconds** (2.48×). Warm frames 1→11 covered
two simulated seconds in 1.891 wall seconds (**1.058×**), essentially unchanged
from the stream-only pass. All twelve original frame identities were submitted
in order; one prefetched frame 12 remained unadmitted. Warm IPC median/p95 was
81.1/112.9 ms, worker preparation median 115.3 ms, and warm CPU-submission gap
p95 252.9 ms. These exclude cold preparation and are not compositor intervals.
The strict guard allowed three original preparation actions and zero read-time
steps; the twenty checks included the separate post-timing close-view QA.
Whole-Python peak RSS was 788,660,224 bytes; private frame/checkpoint/history
retention remained 33,609,379/432,316/268,522 bytes, with 111,442,521 bytes of
immutable input shared separately. Source archive and loaded Python files
remained unchanged. No numerical, cache-admission or RL authority was relaxed.

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

A later stock-engine drill contact search reuses one validated occupancy owner
within that pure search. Every new search validates again after its first
geometric query; effects and microsteps never share this lease. Definition-time
method/kernel/helper identities, code and defaults admit only the finite stock
Python query path. Instance/class/body replacements, custom cache/grid/owners
and engine subclasses retain dynamic `touched` dispatch. The first geometric
query still precedes occupancy validation. An independent review caught a
replaceable-kernel gap in the first draft; the corrected guard and stateful
override regressions passed before the patient comparison.

Three balanced original action-2 pairs measured **4.251→3.985 seconds median**
(6.26% less fresh step time). Two separate profiling trials reduced occupancy
calls from 8,770 to 1,800; all 8,170 geometry/material queries and 7,270 bone
queries remained unchanged. All eight complete outcomes, histories, snapshots
and metrics matched the preserved canonical transition exactly, including
endpoint `sha256:3c60100d51b55b4f1a16534e24c338d7fee685f12253eca972a76dca01585788`.
The batch took 55.50 seconds with a 120-second cap. Each trial used a fresh
stock engine/cache from the same authenticated prior-implementation boundary-2
benchmark seed; this granted no current cache authority. Loading (0.859–0.918 s)
and construction/seed restoration (0.049–0.051 s) were reported separately.
This modest single-action gain does not qualify 10× matched full-workload
throughput. Read-only prefix subclasses use the original path, so this result
does not establish another GUI cold-preparation gain. No policy or optimizer ran.

One later diagnostic used a preserved boundary-1,954 primary checkpoint for
exactly its next original CUSA action. The checkpoint omitted prior receipts,
so all 1,954 were reconstructed from the complete canonical publication with
the stock receipt fields; this projection matched all 42 shared-prefix receipts
from the fresh vascular run and three preserved native receipts. This is an
explicit historical diagnostic seed, with no current cache or planning authority.
The one public step accepted 19 removed cells and exactly matched its complete
canonical record and physical endpoint. The first missing-receipt attempt
stopped before any step and remains preserved.

The corrected diagnostic took 43.720 seconds including setup under its
60-second/1.5 GiB bounds. Its **34.364-second profiled step is not normal
transition throughput**. With 414 microsteps, cumulative occupancy cost was
13.058 seconds; the strict integer-type generator visited 103,809,893 ledger
entries and accounted for 7.583 seconds of self time. Thermal updates took
6.786 seconds and rollback checkpoint validation/copying 6.478 seconds. Of the
latter, 6.325 seconds was flat-type validation; actual deepcopy plus native
list/dict copying totaled 0.146 seconds. Thermal cost was mainly repeated cooling
of the retained bone-cell ledger, rather than dose integration. Geometry took only
0.238 seconds. Thus the early drilling profile does not describe late CUSA
costs; large retained ledgers require separate measurement. Source, protection,
action and start/end identities remained unchanged, with zero policy executions
or optimizer updates.

The subsequent optimization moves only strict integer-list validation into
import-bound C iterators. It preserves `type(cell) is int`, order and
short-circuiting; no validation or fresh tuple-key snapshot is skipped. Exact
built-in removed lists use it in occupancy and rollback checkpoint admission.
Occupancy list subclasses keep the original generator; checkpoint subclasses
keep ordinary deepcopy. All flat-map guards, collision/protection checks and
physical laws remain unchanged. The 50,000-ID pure-predicate fixture measured
1.5364→0.7605 ms median, with strict-type, mutation, error and callback checks;
this isolated predicate result is separate from the physical comparison.

Three alternating fresh before/after pairs of the same late CUSA action measured
**15.4605→13.3691 seconds median** (13.53% less step time; 1.1564×). Every complete
outcome/history/snapshot/metrics payload was byte-identical across all six
independent engines. Each also matched the original full canonical row and
physical endpoint. An original standalone full action-1,955 payload was
unavailable; the original comparison does not claim unavailable metrics JSON.
Setup and export timings are recorded separately per trial. The entire batch
took 176.729 seconds within 180; every child stayed below 60 seconds and peak
RSS was 1,124,581,376 bytes, below 1.5 GiB. The 211-file frozen/current manifests
differ only in `surgical_engine.py` and remained unchanged during timing.
This is one late action, not a 10× full-workload or online-decision qualification.
No policy, optimizer, cache publication or editable admission ran.

## Long-operation history and optional idle cooling

A bounded census exposed a separate full-primary storage blocker. Historical
canonical receipt containers alone, excluding actions and all scalar ownership,
require at least 30,805,752 bytes and exceed the prefix owner's 4 MiB history cap
at projected position 174. This is a static layout calculation, not an observed
current replay stopping position. The first census overcounted shared empty
tuples; its corrected version excludes every empty-list cost conservatively.

The prefix owner now compresses complete receipt bodies and row-local display
projections independently. Full actions still borrow the bounded immutable
input. Checksums bind origin, action log, protection, implementation, ordinal
and canonical/display domain. Only the private execution shell uses interned
status/reason rows for the unchanged collision metric; full receipts remain
available immutably. Public engine history is unchanged. Acknowledged warm
frames decode no receipt/display history. Existing whole-history callbacks
still receive complete immutable receipts on explicit demand, under a declared
per-graph materialization ceiling. Decoded, frozen and thawed graphs may coexist;
that ceiling is not a combined process-memory bound.

Checkpoint, frame, history and endpoint proposals now finish before publication.
A generated allocation-failure regression verifies that a failed frame proposal
cannot evict the previously committed checkpoint. The retained history quota
remains **4 MiB**, including both compressed stores, indexes, digests, metric
interning metadata and containers.

The current storage helpers accepted all **2,148 historical projected receipts**
in a guarded, unadmitted scratch calculation: **3,204,279 bytes**, with **990,025
bytes (23.60%) headroom**. Complete bodies and display rows round-tripped exactly,
including float representations. The calculation took 14.663 seconds, peaked at
69,517,312 bytes RSS, and entered no owner constructor, seek, engine constructor
or physics step. It does not qualify current-generated full-primary receipts,
their transient views, or the actual primary playthrough.

An optional native loop implements the unchanged scalar idle-cooling law on
strictly admitted finite, subthreshold, unpowered cell ledgers. Other inputs use
the existing Python loop. It preserves key order, dictionary identity, signed
zero, separate floating operations and sorted sequential energy accumulation.
Watched dictionaries decline before mutation; all replacement floats are staged
before commit. The local build is restricted to **CPython 3.12/macOS arm64** and
binds its C source, recipe, compiler flags, headers, runtime and binary. Selection
is fixed at import. A missing configuration keeps Python mode; invalid configured
builds or later source/binary/configuration drift raise an error. Verified and
observed/prefix identities include the selected implementation. No runtime
compilation, package installation or new dependency is required.

The ignored pure-helper probe measured 9.194 to 1.753 ms median (5.244×), with
exact full-state bits and checked fallbacks. That is not the integrated gain.
Two alternating matched pairs of the actual original late CUSA action measured
**15.2034 to 13.4290 seconds median** (**11.67% lower latency; 1.132×**), including
live identity guards and symmetric entry/return counters. Each native trial
admitted all 414 cooling intervals. All four complete outcome/history/snapshot/
metrics payloads were byte-identical, and the original full canonical row and
physical endpoint matched. The diagnostic historical seed grants no admission;
an original standalone full action payload remains unavailable. Parent wall was
128.625 seconds (<150), maximum child wall 31.231 seconds (<60), and peak child
RSS 906,493,952 bytes (<1.5 GiB). Setup/export are separate. Two samples per mode
do not establish a full-workload or GUI speedup.

The final guarded surgical/full-case/independent-geometry selection passed
**882 tests, with seven pilot tests skipped, in 149.08 seconds**, including the
explicit optional-native generated checks. Two existing Gymnasium advisories
remain. The focused native selection passed 33 tests; the corrected fallback/
history/cache selection passed 140, with 28 optional-native checks skipped.
Its earlier compact/full-wire assertion failure is preserved; the fix restored
acknowledged metadata before comparison. These qualifications ran no learning;
the two earlier fixture updates remain disclosed in the readiness document.

To build an optional local kernel, choose a fresh output directory:

```sh
PYTHONPATH=src .venv/bin/python -m resectionlab.build_surgical_idle_native --output-dir testcases/outputs/native-idle-local
RESSECTIONLAB_IDLE_THERMAL_BUILD="$PWD/testcases/outputs/native-idle-local/build-receipt.json" ./scripts/open_shared_rehearsal.sh --allow-primary-display
```

This selects the development interpreter's explicit build. Packaged sidecars
and other runtimes need separate qualification. Leave the variable unset for
Python mode. Existing caches and the earlier full vascular GUI receipt do not
qualify this later implementation. The separately repeated qualification below
binds the later source and selected native build. Full-primary playback and the
remaining engineering gates stay open.

## Optional native prefix JSON snapshots

A separate optional C writer preserves the exact compact JSON bytes used by
prefix frame, checkpoint and receipt capture. It admits only exact built-in
trees within depth 64, one million expanded nodes and a 16 MiB ASCII ceiling.
Unsupported inputs retain the legacy encoder. Generic encoding callbacks also
retain their previous behavior; the native snapshot is selected explicitly only
for capture-owned trees. State-ID rehashing is unchanged.

The writer takes its complete immutable snapshot before returning to Python
chunking, compression or growth callbacks. It uses captured stock C string and
finite-float formatting helpers and its own signed-int64 decimal conversion.
This avoids the integer signal checks that blocked the earlier whole-tree stock
encoder prototype. The writer checks pending signals before final admission,
then completes source reads without an interpreter callback. The critical write
neither disables signals nor calls Python. Conservative recursion
headroom and strict runtime/helper identity checks bound this private-ABI path.

The revised ignored writer prototype used the same two independently audited
v3 states: **55.884→11.090 ms** early and **63.086→14.003 ms** late, with exact
raw JSON, float bits, order and compressed bytes. Its twelve actual-state
pipelines, 36 small generated pipelines and pending-signal fixture finished in
5.686 seconds with 238,452,736-byte sampled process-tree RSS (<30 s/1 GiB).
The signal fixture kept 200,000 pre-handler floats intact during a 51.282 ms
write; the handler mutated the source only after the snapshot. This is component
evidence, not a fresh-engine or application throughput measurement.

The production build is restricted to **CPython 3.12.12/macOS arm64**. Selection
is fixed at import through `RESSECTIONLAB_PREFIX_JSON_BUILD`; unset means Python
mode. Invalid configuration or subsequent source/build/binary/runtime drift
fails closed. The prefix seal binds the adapter, recipe and selected identity.
Full checks bind runtime image bytes at owner admission and action publication;
warm reads check loaded helpers and file identity metadata. No planning, RL or
canonical authority is granted. The 48 MiB logical native scratch allowance plus
bounded C stack/metadata is separate from input graphs, compression, retained
storage and process RSS. Allocation-error cleanup was source-reviewed, without
allocator fault injection.

The single explicit build took **0.441 seconds** (0.301 s compiler) and produced
a 70,464-byte binary. The final guarded qualification passed **929 tests in
161.32 seconds**, with two existing Gymnasium advisories. The pilot module was
excluded entirely. The parent selected the native JSON codec; generated child
checks covered both optional builds and Python fallbacks. External wall time was
161.721 seconds (<240), sampled process-tree RSS 1,606,320,128 bytes (<3 GiB), and
all 259 bound source/test files stayed unchanged. An earlier focused run passed
139 checks but hit one test-fixture `E2BIG` before launching a child, caused by
an oversized pytest parameter ID; concise IDs fixed the fixture. Its failure is
preserved. These checks selected no learning path; the earlier two fixture
updates remain disclosed above.

Build into a fresh ignored directory, then select its receipt explicitly:

```sh
PYTHONPATH=src .venv/bin/python -m resectionlab.build_surgical_prefix_json_native --output-dir testcases/outputs/prefix-json-local
RESSECTIONLAB_PREFIX_JSON_BUILD="$PWD/testcases/outputs/prefix-json-local/build-receipt.json" ./scripts/open_shared_rehearsal.sh --allow-primary-display
```

The idle-cooling build variable is independent; set both to select both kernels.
Other runtimes and packaged sidecars need separate qualification. The v4 GUI
receipt below predates this codec; the separate v5 receipt qualifies its selected
saved-replay implementation and measures its application result.

## Complete vascular GUI replay

The production application at `cc868257` completed the original **73-action vascular
branch**, from source through its final saved endpoint, with its existing tool
and effect layers active.
All **1,149 fresh observed tuples** were CPU-submitted in order, including 113
legitimate repeated state hashes at distinct action/frame positions. No fetched
observation was duplicated, omitted or left unadmitted. The application stopped
automatically at action 72/frame 2; the complete reconstructed timeline contained
73 entries. This qualifies the full vascular saved-replay path, not the primary
2,148-action operation or an editable planning session.

A separate post-GUI, zero-step process decoded and independently hashed every
fresh state, checked adjacent ancestry, timestamps, effective activation and
heavy-field tokens, and reproduced every complete historical canonical action
record, including fluid/wound/debris accounting. The endpoint matched
`sha256:af496ffbde3ca2ab71477c9226fe9f846f08171efd308362431414584cd3c87e`.
Historical microframe hashes were unavailable: their original count/order and
all 73 canonical boundaries are checked, while fresh microframe identities are
independently checked against the actual CPU submissions. These are CPU
submissions, not compositor presentation measurements.
The receipt binds the complete Python/build manifest. Later source changes
require fresh cache admission; this result does not authorize their cached replay.

| Full vascular measurement | Result |
| --- | ---: |
| Saved simulated duration | 202.569 s |
| Replay command through final CPU submission, including cold reconstruction | 395.930 s / **0.512×** |
| Same-action CPU intervals, excluding gaps between actions | 153.827 s / **1.317×** |
| Guarded Python sidecar wall, including source opening/setup | 413.564 s |
| Separate post-GUI independent audit | 69.123 s |
| 73 cold bridge executions, cumulative / p95 | 230.108 s / 8.210 s |
| 1,076 warm bridge executions, cumulative / p95 | 99.211 s / 116.065 ms |
| Warm CPU-submission interval, median / p95 | 136.3 / 205.5 ms |
| Peak whole Python resident memory during GUI | 805,994,496 bytes |
| Peak retained checkpoints / observations / receipts | 16,757,408 / 67,038,087 / 926,278 bytes |

The retained limits remained 16 MiB for checkpoints, 64 MiB for observations and
4 MiB for receipts. The 10,235,381-byte immutable source record was borrowed from
its private owner within the separate 128 MiB input ceiling. The independent
audit spool held 215,884,624 compressed bytes and an 845,738-byte manifest, below
their 1 GiB/16 MiB caps. Copy/hash/write time was 0.195 seconds and remained
included in GUI timing. Source anatomy, JavaScript/GPU allocations and transient
allocations are additional to the retained graph counts; the whole-process
Python limit was 3 GiB. No canonical engine session, disk replay cache or frame
admission was created, even after the endpoint matched.

The cold path is the largest measured remaining cost. Its bridge timing includes
physical reconstruction, observed-state capture and view preparation; this run
does not separate those internals. Warm IPC totaled 106.334 seconds (p95 125.805
ms), while 464 mesh preparations totaled 52.418 seconds and can overlap backend
work. These totals must not be added as independent wall intervals. The 448 bone
jobs had median/p95 117.6/139.8 ms. Late vascular actions needed no new surface
meshes, leaving backend/IPC/handoff costs visible without remeshing.

A separate no-step capture diagnostic reused two independently audited states
(early drill action 2/frame 76 and final inspect action 72/frame 0). Ten captures
and 14 compressed-state encodes reproduced the retained blobs byte for byte;
source/build identities and the no-step guard remained intact. Total diagnostic
time was 2.285 seconds with 131,923,968-byte peak RSS. Capture medians were
97.19/109.01 ms, including 56.64/64.26 ms encoding and 30.31/34.46 ms state hashing.
Separate component passes measured JSON formatting plus UTF-8 at 51.22/58.11 ms
versus 4.84/5.93 ms compression. These passes repeat unchanged states; they are
not fresh transitions or independent partitions of a GUI interval. Local
monitoring overhead is included, and the small empty-owner stage does not
qualify a 152-frame eviction tail. No production function changed or replay
speedup resulted. That first probe did not split bound validation, traversal and
native encoding.

A subsequent component diagnostic reused the exact 211-file source version of
those audited states. Its first attempt correctly refused the later engine
identity before encoding; both receipts are preserved. The corrected no-step
probe found 146,850/164,960 scalar admission calls per sampled state. Its local
profiler heavily inflated elapsed time, so instrumented fractions are not
ordinary encoder throughput. Repeated failed container-admission scans accounted
for only 0.185/1.430 instrumented ms and failed the declared materiality gate;
that candidate was stopped.

An ignored scalar-admission prototype then measured three balanced unprofiled
pairs per state. Median encoding fell from 56.652 to 53.368 ms early and from
63.994 to 60.908 ms late. All compressed and raw bytes matched, with identical
fragment counts and maxima; direct fragment-list equivalence was checked on
generated cases. Nineteen generated cases covered Unicode, signed zero,
nonfinite/large numbers, subclasses, cycles and small budgets. The probe took
2.271 seconds, peaked at 121,995,264 bytes RSS and executed zero steps. One early
pair regressed. The roughly 3 ms component gain remains unpromoted, with no
full-capture, current-engine admission or replay speedup claimed.

A separate pure idle-cooling NumPy prototype reused the bound 15,238-cell late
ledger. All 30 paired calls preserved complete floating-point bits, key order
and dictionary identities, including fallback fixtures. It was slower after
validation, conversion and writes: 8.024 to 12.280 ms median. The 2.613-second,
128,286,720-byte-RSS experiment executed no engine constructors or steps and
changed no production source. That candidate was rejected.

A separate ignored C structural-admission prototype then admitted complete
strict built-in JSON trees under a conservative 16 MiB ASCII bound, followed by
stock `json.dumps` and the existing compression format. Three balanced pairs
per unchanged audited state measured **54.614 to 14.594 ms** early and **61.974
to 17.749 ms** late, saving 40.020/44.224 ms. Complete raw compact JSON, decoded
floating bits/key order and compressed bytes matched. The single 5.421-second
run peaked at 235,552,768 bytes RSS, compiled once and performed 12 actual-state
encodes plus 36 separate generated-fixture encodes, with no owner/engine
construction or physics. Both declared component gates (10 ms and 20% lower
median latency) passed. This is a material candidate, not a production change:
mutation between admission and encoding, atomic callback behavior and transient
memory need their own proof. The existing idle kernel/build and all production
sources stayed unchanged. Evidence is `native-json-admission-prototype/`.

Production design found a concrete atomicity blocker: the pinned CPython 3.12
integer-to-decimal path can [check Python signals](https://github.com/python/cpython/blob/v3.12.12/Objects/longobject.c)
during the stock whole-tree
encoder. A signal handler could mutate the admitted tree before encoding ends.
The prototype remains valid for its unchanged private inputs, but that design
cannot yet be used as a general bounded production fast path. Signal checks
have not been disabled or bypassed.

A separate exact-source row-enumeration prototype replaced a repeated global
mesh-row sort with bounded raster enumeration. Complete output-buffer hashes
matched across 19 paired states and a forced fallback. Warm assembly improved
79.405 to 74.202 ms (5.204 ms, 6.55%), below its declared 10 ms and 15% gate.
It was stopped without a production change or retry. The single job took
11.239 seconds with 740,425,728-byte sampled Node RSS, 3,056-byte transient
scratch and unchanged 111,711,761-byte retained worker storage. These are CPU
component measurements, not a GUI speedup. Evidence is
`row-enumeration-prototype/{report,analysis}.json`.

The actual endpoint screenshot shows the read-only 73/73 label and enlarged
target/context. Both tools are withdrawn. The optional operative-field camera
stayed pinned to the early accepted tool, leaving its axial slice outside image
support. This is a demonstrated view limitation; full-sequence microscope
readability and an actual-app demonstration remain open.
The existing equivalent-volume fluid/debris glyphs do not qualify the requested
droplet, pooling, irrigation-stream and aspiration visualization.

Evidence is under `desktop-full-vascular-current-v3/`. Two failed attempts remain
preserved: an inert Electron entry-point launch with zero actions, then a
one-action/three-observation run stopped before action two by immutable-record
serialization and lazy subclass-inventory checks. The corrected harness keeps
strict ancestry and inventory guards. A separate no-step audit of that partial
spool verified its first row/states and correctly refused full-run acceptance.
The harness fixes passed 15 generated Python and 11 Node checks and independent
review before the successful retry; no training command or policy ran.

### Repeated full vascular qualification with compressed history/native cooling

The later implementation completed the same **73 actions and 1,149 observations**
with all 15 harness checks passing. Independent review compared the submitted
tuples to the audited tuples in exact order, retaining all 113 repeated hashes.
All 73 complete historical canonical records and the final endpoint matched.
The separate audit executed zero physics steps. All fresh observation IDs also
match the available prior v3 run; original historical microframe hashes remain
unavailable. The native source, adapter, recipe, build receipt and binary bindings
were rehashed after the run and matched. No planning/cache authority was granted.

| Later full vascular measurement | Result |
| --- | ---: |
| Replay command through automatic stop, including cold reconstruction | 410.825 s / **0.493×** |
| Same-action CPU-submission ratio, excluding inter-action gaps | **1.272×** |
| Guarded Python sidecar wall, including source opening/setup | 430.989 s |
| Separate independent audit | 69.725 s |
| 73 cold bridge executions, cumulative / p95 | 239.140 s / 8.518 s |
| 1,076 warm bridge executions, cumulative / p95 | 100.412 s / 124.305 ms |
| Warm CPU-submission interval, median / p95 | 125.8 / 219.9 ms |
| Peak whole Python resident memory | 789,381,120 bytes |
| Peak retained checkpoints / observations / receipts | 16,757,408 / 67,038,087 / 137,061 bytes |

The observed total wall time was **3.76% longer** than the prior 395.930-second
run. Same-action intervals were also slower. Multiple backend changes and
different host runs prevent attributing that difference to a single component;
this is **not a GUI speedup**. Receipt retention fell 85.20% in this workload,
under the unchanged 4 MiB quota. Other retained quotas and the 3 GiB Python limit
passed. The independent spool again held 215,884,624 compressed bytes, below
1 GiB. Copy/hash/write overhead remains included in playback timing.

The endpoint screenshot again shows 73/73 read only and withdrawn tools. Dense
translucent anatomy and a pinned early-tool axial plane outside image support
remain visible limitations. Static effect glyphs do not establish flowing fluid
effects. Full-sequence operative readability, actual compositor timing and the
full primary replay remain open. This run performed no policy or optimizer work;
the two earlier fixture updates remain separately disclosed.

Evidence is under `desktop-full-vascular-native-history-v4/`, with the explicit
native/prelaunch binding in the sibling
`desktop-full-vascular-native-history-v4-binding.json`. This receipt qualifies
that read-only saved replay implementation, not subsequent source versions.

### Full vascular qualification with native JSON snapshots

The next full run selected both explicit native builds and passed all 15 harness
checks. It executed the 73 original actions once, CPU-submitted all 1,149 audited
tuples in order, and preserved all 113 adjacent repeated state IDs. The separate
zero-step audit matched every complete historical canonical record and the saved
endpoint. Available fresh trajectories also matched both v3 and v4 exactly;
original historical all-microframe IDs remain unavailable. All source, renderer,
C/build/binary/helper-image bindings and the original archive stayed unchanged.
No canonical, planning or RL authority was granted.

A separate local `PY_RETURN` observer recorded **1,369 actual native JSON
admissions, zero fallbacks and zero outside-command returns**. Warm reads encoded
zero snapshots. These counts are encoder calls, not frames, and the observer
retains only numeric counters. Its cost is included. The original global
physics-step guard remains separate. Focused qualification passed **28 tests in
1.023 seconds** (1.359 s external wall, 87,244,800-byte process-tree RSS). The first
run's four failures exposed exit cleanup masking the original drift diagnostic;
both context-manager exits now preserve that error while retaining sticky
failure and unconditional cleanup. Original failure evidence is preserved.

| Native JSON full vascular measurement | Result |
| --- | ---: |
| Play command through automatic stop, including cold reconstruction | 388.511 s / **0.5214×** |
| Same-action CPU-submission intervals | 171.944 s / **1.1781×** |
| Guarded Python sidecar / separate zero-step audit | 409.236 / 78.190 s |
| 73 cold bridge executions, cumulative / p95 | 202.633 s / 6.729 s |
| 1,076 warm bridge executions, cumulative / median / p95 | 113.541 s / 108.405 / 151.681 ms |
| Warm CPU-submission gaps, median / p95 | 145.35 / 238.4 ms |
| Peak whole Python resident memory | 751,730,688 bytes |

Play-command wall time was **5.43% lower** than v4, while same-action intervals
were **7.98% longer**. The renderer and JS benchmark were unchanged, but codec,
seal and monitoring changes plus separate host runs prevent isolating causality.
Cold reconstruction improved in this observation; warm reads did not. Neither
1× cold-inclusive playback nor the delivered-frame target passed. Component
measurements must not be substituted for the full-run result.

Retained checkpoint/frame/history/input peaks remained 16,757,408 / 67,038,087 /
137,061 / 10,235,381 bytes, within the unchanged 16/64/4/128 MiB limits. The
215,884,624-byte spool and 845,738-byte manifest stayed within 1 GiB/16 MiB;
0.201 seconds of spool copy/hash/write is included. Python and audit memory and
the separate 600/180-second limits passed. These counters are separate from
whole Electron/GPU memory.

Actual endpoint pixels show 73/73 read only, disabled close-tool view, a large
purple target with static yellow glyphs and dense translucent context. The
pinned axial plane at −369.9 mm is outside image support and blank. This endpoint
does not establish full-sequence readability, flowing effects or compositor
cadence. Full-primary playback remains open. No policy or optimizer ran in this
replay/audit; the earlier two fixture updates remain separately disclosed.

Evidence is `desktop-full-vascular-native-json-v5/{report,analysis,physics-guard,spool-audit}.json`,
with sibling prelaunch/post-run binding receipts. The codec milestone is
`66fe1b18`, and the reviewed qualification monitor is `2e379cc3`.

A subsequent no-step probe reused the same two audited v3 states under both
current native build selections. Exactly 12 decode/full-digest rounds preserved
every expected state ID; no owner, engine or geometry constructor ran.

| Separate warm-read component median | Early state | Late state |
| --- | ---: | ---: |
| Bounded zlib/JSON decode | 9.726 ms | 11.423 ms |
| Complete production state digest | 30.069 ms | 34.863 ms |
| Separate exact admission probe | 17.958 ms | 18.707 ms |
| Separate sorted JSON/UTF-8/SHA probe | 11.778 ms | 15.420 ms |

The two implementation-check seal pairs together measured 18.871 ms median.
These separate probes are not additive timing partitions or complete warm-bridge
attribution; view/metrics, bridge JSON-budget checks and IPC remain unmeasured.
The single probe took 3.790 seconds with 175,947,776-byte sampled process-tree
RSS (<30 s/1 GiB), and 53,869,398-byte logical payload scratch (<64 MiB).
Source, inputs, runtime, both builds and exact state identities stayed unchanged.
Evidence is `warm-read-decomposition-probe/`. It identifies a digest target for
another bounded prototype; it qualifies no further production optimization.

### Exact canonical snapshot hashing

A separate ignored prototype replaced the complete strict state-digest pipeline
with a bounded sorted ASCII snapshot followed by SHA-256 over every byte. Three
balanced pairs per audited state measured **30.185→7.971 ms** early and
**35.204→11.625 ms** late, saving 22.213/23.579 ms (73.59%/66.98%). Complete sorted
JSON bytes, decoded float bits/key order and the original complete state IDs
matched. The run used twelve actual pipelines, 64 fixture pipelines, 36 direct
admission calls and one pending-signal encode: 75 native entries total. All 19
eligible fixtures were required to admit. During a 19.655 ms sorted write, a
1 ms pending signal mutated the input only after the immutable snapshot existed.

The single compile/run took 5.013 seconds with 331,563,008-byte sampled
process-tree RSS (<30 s/1 GiB). No owner/engine constructor, action, policy or
optimizer ran. These component gains do not establish application throughput.
Evidence is `native-canonical-json-prototype/`.

The subsequent implementation adds a separate `try_encode_canonical` entrypoint
to the optional prefix kernel; insertion-order capture remains separate. A
private prefix helper recomputes the full scene/state/protection digest from
decoded state. It retains the original property for custom engine types,
property or attribute-lookup overrides, altered digest/JSON dependencies and
unsupported native inputs. Static descriptor/module guards preserve fallback
side effects. The public engine property and numerical laws remain unchanged;
implementation-bound caches can still invalidate conservatively.

Canonical ordering uses exact Unicode codepoints and an iterative heapsort of
borrowed key/value pairs. A preallocated arena is at most 16 MiB; the canonical
logical scratch allowance is 64 MiB plus bounded C stack/scalar metadata.
Insertion capture retains its 48 MiB allowance and allocates no pair arena.
Both paths retain depth 64, one million expanded nodes, 16 MiB ASCII, strict
runtime/helper identity and immutable completion before Python callbacks.
Allocation cleanup is source-reviewed, without allocator fault injection.
The revised build schema is `surgical-prefix-json-native.v2`; old build receipts
must not be substituted for its fresh explicit build. Production regression and
the subsequent GUI result have separate receipts.

The single revised build took **0.636 seconds** (0.448 s compiler), produced a
71,072-byte binary and emitted no warnings. The guarded regression selection
passed **952 tests in 182.99 seconds**, with two existing Gymnasium advisories.
The pilot module was excluded entirely. Parent mode selected native canonical
JSON; generated subprocess checks covered both explicit native builds and
fallbacks. External wall was 183.462 seconds (<240), sampled process-tree RSS
1,212,989,440 bytes (<3 GiB), and all 259 bound source/test files stayed unchanged.
No learning path was selected. A separate dual-counter qualification passed
34 generated tests in 1.034 seconds (1.423 s external, 89,358,336-byte tree RSS),
retaining the original physics-step guard and distinguishing canonical hashing
from insertion-order capture.

### Full vascular qualification with canonical hashing

The separate v6 application run at `3febf0da` passed all 15 checks. It prepared
the 73 original saved actions once, submitted all 1,149 observation tuples in
order, and retained the 113 adjacent repeated state IDs. Its independent
zero-step audit matched every complete historical canonical record and the
saved endpoint. All fresh tuples also matched v3, v4 and v5 exactly. Historical
all-microframe IDs remain unavailable. All bound source, renderer, build,
binary, helper-image and original archive bytes stayed unchanged.

The two independent local counters observed **1,369 insertion-order native
admissions and 2,741 canonical native admissions**, with zero fallbacks or
outside-command returns. Each of the 1,076 warm reads performed exactly one
complete canonical native hash and zero insertion-order captures. Both monitors
closed without failure. Counts describe encoder calls, not frames; their cost
is included. No canonical/session/cache, planning or RL authority was granted.

| Canonical native JSON full vascular measurement | Result |
| --- | ---: |
| Play command through automatic stop, including cold reconstruction | 318.613 s / **0.6358×** |
| Same-action CPU-submission intervals | 147.550 s / **1.3729×** |
| Guarded Python sidecar / separate zero-step audit | 337.836 / 69.405 s |
| 73 cold bridge executions, cumulative / p95 | 157.918 s / 5.247 s |
| 1,076 warm bridge executions, cumulative / median / p95 | 79.786 s / 74.592 / 108.108 ms |
| Warm IPC, median / p95 | 81.566 / 117.492 ms |
| Warm CPU-submission gaps, median / p95 | 109.0 / 224.6 ms |
| 464 surface preparations, cumulative / median / p95 | 56.266 s / 122.7 / 156.3 ms |
| Peak whole Python resident memory | 707,166,208 bytes |

Cold-inclusive wall time was **17.99% lower** than v5; warm intervals were
14.19% shorter. The renderer and JS harness stayed unchanged, while the codec,
seals and counter guard changed. Separate uncontrolled host runs and slightly
different initial framing prevent isolated causal attribution. This remains
CPU-submission timing, not compositor delivery. Full cold-inclusive playback
still misses 1×, and the delivered-frame target remains unqualified.

The unchanged checkpoint/frame/history/input quotas passed with peaks of
16,757,408 / 67,038,087 / 137,061 / 10,235,381 bytes. The 215,884,624-byte spool,
845,738-byte manifest and 1,036 distinct compressed blobs stayed within their
separate caps; 0.234 seconds of spool copy/hash/write is included. GUI/audit wall
and Python RSS limits passed. These counters exclude whole Electron/GPU memory.

Actual endpoint pixels show 73/73 read only, withdrawn tools, disabled close-tool
view, a purple target/static yellow glyphs, dense translucent context and a blank
axial plane at −370.3 mm. Coronal and sagittal slices are visible. This endpoint
does not qualify full-sequence operative readability, flowing effects or
compositor cadence. No policy or optimizer ran in this replay/audit; the two
earlier fixture updates remain separately disclosed.

Evidence is `desktop-full-vascular-native-json-canonical-v6/{report,analysis,physics-guard,spool-audit}.json`,
with sibling prelaunch/post-run binding receipts. The analysis SHA-256 is
`28cd4092a360d817d3484fa2e329c8ac0f3fd6f78b2947997ecbfb7a29253178`.

## Gates that remain open

* **Fresh cold-inclusive playback still misses 1×.** The latest vascular
  preparation at `df779989` reached 0.708×, with 1.494× for same-action CPU
  intervals. Its separate zero-step prepared-only reopen reached 1.137× over
  all 1,149 observations, with 73 cache loads and unchanged cache bytes. This is
  filesystem-warm repeat playback, not fresh simulation or training throughput;
  see [prepared-prefix-replay.md](prepared-prefix-replay.md). Prior fresh runs
  reached 0.636×/1.373×, 0.521×/1.178×, 0.493×/1.272× and
  0.512×/1.317×. The short
  current-production prefix sample reached 1.058× warm, even with 4× selected,
  while cold preparation of one 30-second action took 20.22 seconds. The
  separate frozen-numerics transport experiment reached 0.720×. Exact source
  surface preparation and replay reconstruction remain material costs.
* **A 50 ms p95 delivered-frame interval is not established.** A separate
  trusted camera-orbit control delivered 42 Electron frames with a 50.99 ms
  nearest-rank p95 interval. This control includes frame-subscription observer
  overhead and is separate from primary playback timing.
* The full final primary GUI playthrough remains unrun. The complete vascular
  saved replay passed; this does not qualify all tool families, modern CUSA
  laws, full-sequence operative readability or primary-case performance.
* New numerical code requires its own canonical verification and observed-frame
  preparation. Old private caches cannot establish new numerical identity.

## Reproducing the checks

The separate [prepared-observation replay mode](prepared-prefix-replay.md)
persists committed read-only observations for later launches. Its preparation
and repeat-play measurements are separate from the fresh v6 result above.
Cached observations are never new transitions or training throughput.

Build and run desktop regressions:

```sh
pnpm --dir desktop test
pnpm --dir desktop build
```

The earlier October 22 normal-scan validation passed 880 desktop tests. The
later prepared-observation implementation passed **899**: 196 main-process,
479 renderer and 224 viewer tests. Build passed with the existing bundle-size
advisory. The separate prepared-mode qualification and resource limits are in
[prepared-prefix-replay.md](prepared-prefix-replay.md).

The earlier exact-list Python qualification passed **832 tests, with seven pilot tests
skipped**, in 119.96 seconds, with two existing Gymnasium advisories. It includes
surgical/full-case tests, independent geometry checks and the read-only prefix
contracts, bounded encoder equivalence and exact retained-frame accounting.
It also checks stock drill query reuse and dynamic fallback for replaced query
dependencies, plus exact-list C-iterator validation and preserved subclass/error
paths. The pilot opt-in was not enabled.

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

Add `PLAYBACK_CLOSE_TOOL=1` to that prefix command for the separate post-timing
close/refocus/restore display check. It waits for settled CPU submission and
completed GPU fences, then writes before/close/restored screenshots. This flag
does not change primary playback timers or permit additional physics actions.

For the separate full vascular qualification, use a fresh ignored directory:

```sh
VASCULAR_FRAME_FIELD=1 \
PLAYBACK_OUTPUT="$PWD/testcases/outputs/rl-readiness-20261022/rehearsal-vascular-check" \
desktop/node_modules/.bin/electron desktop/electron/benchmark-surgical-vascular.cjs \
  --case "$PWD/testcases/outputs/full-case-rehearsal-20261020/final/rehearsal.ressectionlab"
```

This permits exactly 73 original saved steps, once and in order; warm frame
reads must execute none. It has a 600-second GUI watchdog and 3 GiB Python RSS
ceiling, followed by a separate 180-second no-step audit. Do not run other CPU
benchmarks or broad tests concurrently. It refuses frozen-code substitution,
unexpected commands, resumed steps, altered sources and additional timing
observers. `VASCULAR_FRAME_FIELD=1` performs one ordinary display-only operative
field framing after an inserted tool is accepted; the camera then stays pinned.

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
* `pure-contact-original-action-comparison/report.json`: bounded eight-step
  matched fresh original-action comparison with complete payload equality.
* `pure-contact-generated-v2/{report.json,qualification.log}` and
  `pure-contact-final-no-learning-tests.log`: focused dynamic-fallback and full
  guarded numerical qualification, with 816 passed and seven pilots skipped.
* `late-cusa-1954-current-profile-v2/`: one exact late original CUSA step,
  historical diagnostic ancestry binding, complete canonical comparison and
  cProfile costs; the first zero-step missing-receipt failure is preserved in
  `late-cusa-1954-current-profile/`.
* `removed-integer-guard-experiment/report.json`,
  `integer-guard-generated-v2/{report.json,qualification.log}`,
  `integer-guard-final-no-learning-tests.log`, and
  `integer-guard-late-cusa-comparison/report.json`: isolated predicate,
  focused/full guarded regressions and six exact fresh late-action comparisons.
  The first focused-test expectation failure is preserved separately; the
  production patch did not change between focused runs.
* `capture-cost-profile/{report,analysis}.json`: two audited states reused for
  bounded, byte-exact no-step capture component diagnostics.
* `capture-json-finer-profile/` and `capture-json-finer-profile-v2/`: preserved
  source-identity refusal and corrected frozen-source component profiling.
* `capture-json-scalar-prototype/{report,analysis}.json`: ignored exact-byte
  scalar-admission prototype with small, noisy component savings; not promoted.
* `idle-thermal-prototype-experiment/report.json`: exact but slower pure NumPy
  cooling prototype; not promoted.
* `desktop-heavy-field-diagnostic/report.json`: later authenticated field-reuse
  functional/byte diagnostic.
* `desktop-import-current/report.json` and branch PNGs: current-production
  import-only observation.
* `desktop-current-prefix-qualified/{report,analysis}.json`: current-production
  read-only prefix GUI pass, including completed shutdown lookahead accounting.
* `desktop-current-prefix-prefetch-qualified/{report,analysis}.json`: matched
  full-prefix prefetch improvement, with no duplicate fetches and exact state IDs.
* `current-prefix-stream-comparison/report.json`: balanced full/compact bridge
  comparison with exact restored payloads and twelve original frame IDs.
* `desktop-current-prefix-stream-qualified/{report,analysis}.json`: current
  production stream GUI pass, including scoped warm rate, wire sizes and cold
  preparation; `settled.png` is the accepted final app view.
* `prefix-encoder-benchmark-v2/report.json`: bounded native JSON fragments with
  exact compressed bytes; the earlier scalar-only regression remains separate.
* `prefix-frame-accounting-benchmark/report.json`: generated private ownership
  microbenchmark, exact retained counts and unchanged encoding/quota caps.
* `current-prefix-capture-{baseline,optimized,accounting}/report.json` and
  `prefix-capture-accounting-comparison.json`: frozen, encoder-only and final
  three-action capture measurements with all 152 frame records compared.
* `desktop-current-prefix-capture-close-qualified/{report,analysis}.json`:
  functional pass with the initial unsettled screenshot limitation retained.
* `desktop-current-prefix-capture-close-settled/{report,analysis}.json` and
  `before-close-tool.png`, `close-tool.png`, `restored-close-tool.png`: corrected
  actual-app close-view qualification after settled GPU completion.
* `desktop-full-vascular-current-v3/{report,analysis,physics-guard,spool-audit}.json`,
  `audited-canonical.actions.jsonl`, `audited-observations.json` and `settled.png`:
  complete current-production vascular replay and independent post-GUI audit.
* `desktop-full-vascular-current/launch-failure.json`,
  `desktop-full-vascular-current-v2/{report,analysis}.json`, and
  `desktop-full-vascular-current-v2-first-row-audit/`: preserved failed launches
  and the separately qualified partial no-step diagnostic.
* `capture-accounting-final-no-learning-tests.log`, `close-tool-{main,renderer,viewer}-tests.log`
  and `close-tool-build.log`: latest guarded regressions and desktop build.
* `normal-face-scan-benchmark-v2.json`, `normal-face-scan-analysis.json`:
  balanced warmed source CPU comparison, exact complete geometry and scoped
  extraction/assembly/normal timings; first JIT diagnostic retained separately.
* `normal-halo-desktop-full-tests.log`, `normal-halo-build.log`: all 880 desktop
  regressions and the production build after the normal scan change.
* `prefix-stream-full-no-learning-tests.log`, `prefix-stream-{main,renderer,viewer}-tests.log`
  and `prefix-stream-build.log`: latest prefix stream regression/build checks.
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

## Fresh declared-skill execution after replay persistence

One generated 12-skill profile identified 944 optional native cooling identity
checks even though the script never drills and its temperature/power maps stay
empty. The checks consumed 432 ms cumulative within 2.308 seconds of profiled
execution. This includes profiler overhead; overlapping costs are not additive.
The Python path now recognizes only an exact empty thermal loop after preserving
its scalar calculations. It reads no dose/cooling fields and performs no kernel
call. Actual native calls and explicit identity reads retain all live guards.
Consequently later native drift is detected on actual use rather than on an
empty no-op; invalid configured builds still fail at import.

A separate bounded three-seed readiness run constructs all 15 declared nominal
skill options from each fresh observation before every prescribed decision.
The same menu and IDs persist through the six-skill snapshot-resumed tail.
Whole 12-request episodes took **0.955–1.102 seconds (10.89–12.57 requests/s)**,
including setup, observations, all options, primitive execution and snapshot
bookkeeping. Whole-decision p95 was **142–165 ms**. All final native/RL snapshots,
complete public skill receipts, cleanup and conservation match the preserved
native baseline exactly. This is a short generated task, with no actor or policy,
and establishes neither sustained patient 10 Hz nor a matched 10× improvement.
See [the readiness contract and per-seed measurements](rl-readiness.md).

Evidence is `readiness-script-hotpath-profile-v1/` and `skill-options-cli-v2/`
under the existing ignored output root. The former preserves an explicit
correction of its copied launcher's zero-action label to 15 generated primitives.
The first CLI attempt preserves a reset-cap failure: restore performs extra
resets, so its diagnostic cap was corrected from eight to eleven without source
changes. The completed job took 9.825 seconds externally with 90.6 MB sampled
tree RSS, under 30 seconds/1 GiB; it recorded zero policy/optimizer/patient work.
Earlier two fixture optimizer updates remain disclosed at the top of this file.

The new Python implementation invalidates previous prepared-prefix packages by
their existing full-source fingerprint. The v7/v8 GUI pair remains evidence for
its recorded `df779989` source; it is not a GUI qualification of this later code.
Final surgical/full-case/geometry regressions passed 1,081 tests in 181.42 seconds,
with the pilot module entirely excluded and two existing Gymnasium advisories.
External wall was 181.853 seconds and peak sampled tree RSS 1,085,046,784 bytes,
within 240 seconds/3 GiB. No desktop source changed in this milestone.

## Operative framing within source imaging

The blank axial pane in v8 was pinned by its explicit framing command during
approach, at vascular action 1/frame 2, state `3ab9aea7…`. The recorded drill tip
was at Z = −369.9433 mm; the original acquired MRI ends at −374.25 mm. Its
220 × 220 × 181 grid and affine therefore put this valid recorded approach tip
**4.3067 mm outside the scan**. This is not a missing image slice or a modified
tool pose. Existing audited compressed bytes and original array headers confirm
the diagnosis; both original archives were rehashed unchanged.

“Frame operative field” now considers inserted tools inside the acquired voxel
grid, then the existing annotation/context center when it belongs to the same
primary source. It uses the inverse RAS affine, including LPS and oblique grids,
rather than an enclosing world-axis box. It does not clamp a recorded pose,
invent an anatomical point or follow automatically. The App updates the active
image cursor; a selected alternate source cannot inherit the primary annotation.
Feedback says annotation rather than claiming a tumor target where only broad
support anatomy exists.

All 481 renderer tests passed in 6.364 seconds; external wall was 6.558 seconds
and sampled tree RSS 805,830,656 bytes, below 90 seconds/1.5 GiB. Type checking
and production build passed in 1.781 seconds externally, with the existing bundle
size advisory. Evidence is `operative-field-source-containment-v1.json` and
`renderer-normals-{renderer,build}-final-framing-v1.{json,log}`. Updated actual-app
pixels and the complete visible-interaction sequence remain unverified.

### Rejected narrower normal invalidation

A separate renderer candidate compared changed triangle ranges within replaced
chunks. It preserved complete canonical position/index/normal buffers and reduced
dirty vertices from 5,754 to 14 in the source-bone delete/undo fixture. Its complete
update-plus-assembly median nevertheless worsened from **103.452 to 108.846 ms**
in six balanced warm pairs against frozen `0b7d38d9`; comparison work outweighed
the saved reduction. The candidate and its dependent test were removed from
production, with exact source and receipts preserved under
`changed-triangle-normals-v1/`. The job stayed within 60 seconds/1 GiB and made
zero simulator, GUI, policy or optimizer calls. A smaller dirty count alone is
not a performance improvement. Complete index/vertex export remains the next
demonstrated mesh bottleneck.

## Stable source IDs for raw bone updates

Incremental, unsmoothed bone snapshots can now retain their worker-owned source
vertex IDs. Current triangles keep the exact raster order and reference the same
position and normal bits, while export copies contiguous rows instead of
compacting and remapping every corner. The initial normal admission, smoothed
meshes and non-admitted caches retain the canonical path. Complete transferable
snapshots are capped at 64 MiB; the existing picking-input and 128 MiB resident
budgets still apply. Historical vertex slots are never recycled. If their
high-water storage reaches the cap, export falls back to canonical compaction.

Each chunk retains its actual active bounds, charged to the resident budget and
discarded first under pressure. Active bounds govern culling, transparent-object
sorting and picking rejection. Explicit camera fitting visits only referenced
vertices, once each, so old slots cannot expand the view. Full GPU uploads remain.

Six balanced source-bone delete/undo pairs against frozen `9355bd15` reduced
median update plus export from **103.135 to 43.709 ms (2.360×)**. Export medians
were 88.701 versus 29.032 ms; every pair improved. Complete indexed corner
position/normal bits, triangle order and active bounds matched. These checks ran
outside the timed scope. Initial construction was slightly slower, 3.264 versus
3.366 seconds. Peak output was 66,282,144 bytes and retained cache storage
111,755,537 bytes. External wall was 11.882 seconds and sampled tree RSS
635,797,504 bytes, below 60 seconds/1 GiB. This component result does not establish
application FPS, compositor timing, fresh replay speed or the matched 10× gate.

Final desktop checks passed **909 tests**: 481 renderer, 232 viewer and 196 main
process tests. Type checking and production build passed with the existing bundle
size advisory. All jobs were serialized and within their external limits. Evidence
is `stable-index-export-component-v1/` and
`stable-index-export-{renderer,viewer,main,build}-final-v1.{json,log}` under the
ignored output root. Independent source and component-evidence review found no
material defect. A new actual-app full vascular measurement is still required;
the previous v7/v8 results describe earlier source and builds.
