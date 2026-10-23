# Prepared observations: read-only replay

Fresh read-only reconstruction can save its committed observations for later
replay. In Rehearse, **Replay from start · read only** executes the original
saved actions and prepares these observations. After reopening the same archive,
**Replay prepared observations · read only** uses only matching prepared data.
A missing, evicted, corrupt or stale action stops replay. It never triggers
physics as a fallback. The explicit fresh-replay button prepares missing data.

The mode is fixed when replay starts. Losing the backend owner, changing source
or selecting another branch requires another explicit start. Cached states
remain read only; they cannot authorize planning, edits, RL reset/resume or
training. Full archive verification remains a separate operation.

New observations can also retain bounded, authenticated
[recorded field transfers](recorded-field-transfers.md). Static arrows describe
the preceding interval while anatomy stays at its exact endpoint. Display
sampling does not change physical fields, action receipts or RL observations.
The recorder changes implementation seals, so caches prepared before this
change require fresh reconstruction with the current source/native selections.

For an isolated local cache, launch from the primary checkout:

```sh
pnpm --dir desktop build
RESSECTIONLAB_PREPARED_PREFIX_CACHE_DIRECTORY="$PWD/testcases/outputs/prepared-prefix-local" \
desktop/node_modules/.bin/electron desktop \
  --case "$PWD/testcases/outputs/full-case-rehearsal-20261020/final/rehearsal.ressectionlab"
```

For the bounded vascular qualification, select the saved **full-vascular** branch
before starting replay. The 2,148-action primary branch is a separate, unfinished
performance gate.

Use the same cache directory and native build selections on later launches.
Optional native kernels retain the explicit build/selection instructions in
[rehearsal-performance.md](rehearsal-performance.md). Changing code, native
builds, the scene, full archive, action log or protection contract invalidates
the prepared data. The default cache is under the ignored
`testcases/outputs/.prepared-prefix-cache/prepared-prefix-v1/` directory.

Each action stores a complete compressed receipt, compressed state observations,
activation, timestamps, field tokens and its endpoint checkpoint. A separate
local key authenticates the exact source, implementation, action ordinal,
ancestry and blob contents. Only the committed prefix publisher writes this
namespace. Benchmark spools and imported checkpoints are never cache authority.
Every displayed state still undergoes bounded decompression and a complete
scene/state/protection hash check. Display rows are rebuilt from the complete
receipt using the current projection.

The disk limit is 512 MiB across actions, with at most 64 MiB of binary data per
action and a 1 MiB manifest. Old actions may be evicted; interrupted remnants
remain charged to the quota. Existing 16 MiB checkpoint, 64 MiB frame and 4 MiB
history ownership limits still apply. Disk packages can temporarily coexist
with detached binary slices and parsed metadata; these allocations are separate
from retained counters and whole-process RSS. A busy cache lock is an immediate
skip or miss. Cache-write failure leaves the completed in-memory action intact.

Preparation cost, prepared playback and fresh transition throughput are separate
measurements. Cache hits are never counted as new actions or RL decisions. The
full primary replay, compositor timing, operative readability and genuine online
decision throughput remain separate gates. This feature performs no learning.
The development checkout is the current validation scope; packaged persistence
with its default installation paths has not been qualified.

The final generated qualification passed 1,042 Python surgical/full-case and
geometry tests in 171.68 seconds, excluding the pilot module entirely. The
focused cache/prefix/bridge selection passed 172 tests; the playback monitor
passed 41. All 899 desktop tests passed (196 main, 479 renderer, 224 viewer),
and the production build passed with the existing bundle-size advisory.
All 592 bound source/test files stayed unchanged during each job.

Python external wall/RSS were 172.088 seconds / 1,329,446,912 bytes, under
240 seconds / 3 GiB. The initial default-parallel renderer launcher hit its
1.5 GiB tree limit and was stopped; the complete two-worker rerun passed with
966,311,936-byte sampled tree RSS. No application code changed between those
renderer attempts. Logs and external receipts are the ignored
`prepared-prefix-{focused,python-full,monitor,main,renderer,viewer,build}-*`
files under `testcases/outputs/rl-readiness-20261022/`.

## Actual application qualification

One fresh preparation and one prepared-only reopen used the same source at
`df779989`, renderer, harness and two explicit native builds. Both passed all
16 checks and separate zero-step audits. All 73 full historical canonical rows,
the saved endpoint and 1,149 ordered observation tuples matched exactly;
113 adjacent repeated state IDs remained distinct observations. Fresh v3–v6
trajectories also matched. Historical all-microframe IDs remain unavailable.

| Full vascular measurement | Fresh preparation v7 | Prepared-only reopen v8 |
| --- | ---: | ---: |
| Play through automatic stop | 286.184 s / 0.7078× | **178.119 s / 1.1373×** |
| Same-action CPU-submission intervals | 135.569 s / 1.4942× | 134.444 s / 1.5067× |
| Public physics steps / cached actions loaded | 73 / 0 | **0 / 73** |
| New cache publications / misses | 73 / 0 | 0 / 0 |
| 73 action-start bridge requests, sum / p95 | 138.648 s / 4.604 s | 31.635 s / 0.507 s |
| Warm bridge p95 / warm IPC p95 | 86.614 / 95.493 ms | 82.909 / 91.335 ms |
| Warm CPU-submission gap p95 | 198.7 ms | 199.6 ms |
| Separate zero-step audit | 67.545 s | 67.917 s |
| Peak Python resident memory | 748,601,344 bytes | 805,355,520 bytes |

Prepared replay used **37.76% less play-command wall time** in this pair. Its
preflight hashes all prepared files, making this a **filesystem-warm app reopen**,
not cold-disk timing. Startup through automatic stop was 196.972 seconds in v8;
the separate audit is excluded from playback timing. Source, native builds,
cache hashing/decoding, IPC, native-return counters, spool writes and mesh work
are accounted for within their declared scopes. Individual component timings
overlap. Host conditions and slightly different accepted-tool framing limit
causal attribution. Cache hits do not establish fresh transition throughput.

The cache held 234,497,551 bytes in 221 files. Every file path, byte and key hash
stayed unchanged through v8; the completed v7 report/guard/audit/tuple/canonical
evidence stayed bound too. V8 had zero permitted, forbidden or public step
attempts. Actual native admissions were 147 insertion and 1,227 canonical, with
zero fallbacks; each warm read performed one complete canonical hash and no
snapshot capture. Fresh v7 observed 1,369/2,741 admissions, also without fallback.

Both runs stayed within 600-second GUI / 180-second audit / 3 GiB Python limits.
V8 retained 16,766,703 checkpoint / 67,097,583 frame / 137,109 history /
10,235,381 immutable-input bytes, within the existing 16/64/4/128 MiB ceilings.
Its 215,884,624-byte audit spool and 869,682-byte manifest stayed within
1 GiB/16 MiB; 0.312 seconds of spool overhead is included in playback time.
Whole Electron/GPU memory is separate from these counters.

Actual endpoint pixels show **Prepared observations · read only**, 73/73,
withdrawn tools and disabled close-tool view. Dense translucent context, static
yellow glyphs and a blank axial plane persist. This endpoint check does not
qualify full-sequence operative readability, flowing effects or compositor
cadence. The 50 ms delivered-frame target remains open. Full-primary playback,
fresh 1× and candidate-inclusive 10 Hz also remain open.

Evidence is under the ignored `desktop-full-vascular-prepared-prepare-v7/` and
`desktop-full-vascular-prepared-only-v8/` output directories, with sibling binding
receipts. Analysis SHA-256 values are
`d701c0719a89b49db626cff67ce07c75315fb4312eada7077a2ee1c904293476` and
`e4420621361a51145e50e44bb0c3ac3058c6bfda3e8a798ba9056ed9772fc8eb`.
No policy or optimizer ran in this pair. The two earlier fixture updates remain
disclosed in [rl-readiness.md](rl-readiness.md).

The following command is historical for the exact v7/v8 implementation. Later
Python/native changes invalidate that cache and its old idle build; it is not a
current-checkout launch command. New builds/preparation must use fresh directories.

To reproduce that earlier Mac configuration on its matching source version:

```sh
RESSECTIONLAB_IDLE_THERMAL_BUILD="$PWD/testcases/outputs/rl-readiness-20261022/idle-native-production-build/build-receipt.json" \
RESSECTIONLAB_PREFIX_JSON_BUILD="$PWD/testcases/outputs/rl-readiness-20261022/prefix-json-canonical-native-production-build/build-receipt.json" \
RESSECTIONLAB_PREPARED_PREFIX_CACHE_DIRECTORY="$PWD/testcases/outputs/rl-readiness-20261022/prepared-vascular-v7-cache" \
desktop/node_modules/.bin/electron desktop \
  --case "$PWD/testcases/outputs/full-case-rehearsal-20261020/final/rehearsal.ressectionlab"
```

Choose **full-vascular**, then **Replay prepared observations · read only**.
These local cache/native artifacts are disposable, platform-specific and not
committed to Git. Changed source or build identities require fresh preparation.

## Later native implementation qualification

Fresh v11 on `3d4d73d7` subsequently published 73 packages and passed all 17 GUI
and audit gates, with 1,149 exact ordered observations and explicit same-clock
receipt-before-submission proof. It used the newer integer-capable idle build.
Its cache is `prepared-vascular-v11-cache`; full fresh playback was 0.707× and
warm same-action intervals 1.738×. This was fresh preparation, not a separate
prepared-only throughput measurement. V10's earlier clock-association failure
remains preserved and is not retroactively promoted.

V11 packages are historical and require their matching `3d4d73d7` source.
The recorded-transfer implementation invalidates them. Fresh v13 on `282525c3`
published 73 new packages and passed all 18 transport/authority/audit checks,
including exact ordered observations and a stable intermediate capture bracket.
Its full/warm rates were 0.682×/1.668×. This is fresh preparation; a separate
prepared-only reopen was not timed for v13. The failed partial v12 run/cache
remain preserved and unqualified.

The subsequent prefix JSON budget optimization changes full/light Python seals,
so v13 packages also require their matching historical source. Fresh v14 on
`8b4964b7` published 73 packages and passed all 18 application/audit gates;
all observed states, full canonical rows and v13 transfer metadata match.
Full/warm CPU rates were 0.690×/1.701×. A separate prepared-only diagnostic
timed 12 identical retained reads, not a full prepared playback or fresh actions.

On this Mac with the matching current Python/native source, open the v14
packages using the ordinary application (receipt instrumentation is off):

```sh
RESSECTIONLAB_IDLE_THERMAL_BUILD="$PWD/testcases/outputs/rl-readiness-20261022/idle-integer-native-production-build/build-receipt.json" \
RESSECTIONLAB_PREFIX_JSON_BUILD="$PWD/testcases/outputs/rl-readiness-20261022/prefix-json-canonical-native-production-build/build-receipt.json" \
RESSECTIONLAB_PREPARED_PREFIX_CACHE_DIRECTORY="$PWD/testcases/outputs/rl-readiness-20261022/prepared-vascular-v14-cache" \
desktop/node_modules/.bin/electron desktop \
  --case "$PWD/testcases/outputs/full-case-rehearsal-20261020/final/rehearsal.ressectionlab"
```

Choose **full-vascular**, then **Replay prepared observations · read only**.
These source-bound observations grant no editable, planning or RL authority.
Further Python/native changes invalidate them. Full-primary playback, flowing
effects, operative readability and delivered-frame timing remain open.

While paused at an inserted instrument, **Close tool view** can pin a local
anatomy cutaway with context transparency. Refocus explicitly follows the latest
accepted tip; orbit and replay do not move the pinned window. Restore previous
view recovers the saved context, clipping and linked cursor. This affects display
only and grants no planning or training authority.
