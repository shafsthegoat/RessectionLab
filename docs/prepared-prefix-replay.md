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

Actual-app prepare/reopen measurements remain pending. The reopen preflight
hashes the prepared files, so its later timing will describe a filesystem-warm
reopen rather than cold-disk latency. No learning path was selected in these
qualifications; the two earlier fixture updates remain disclosed in
[rl-readiness.md](rl-readiness.md).
