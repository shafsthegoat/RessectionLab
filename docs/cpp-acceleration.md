# C++ acceleration acceptance record

The C++17 backend is integrated into the Python simulator and actual signed Mac
application. The original **2× complete fresh planning target remains unmet**.
Useful correct improvements are retained; kernel speedups do not complete that gate.
All controls are generated engineering cases. Patient validation is unavailable;
the project still has zero completed patient tumor tasks and makes no clinical claim.

## Acceptance matrix

| Gate | Status | Evidence / remaining dependency |
|---|---|---|
| M1 fixed workloads, contracts, actual profiles | passed | Original frozen manifest and portable fixture; supplementary simulation/RL manifest qualified and frozen |
| M2 production C++ integration | passed | Exact native cover, scalar collision and JSON results; source, installed and frozen execution |
| M3 production multi-item batches | passed | Fresh planner uses batches of up to three, exact 370-call trace; bounded cancellation below 100 ms |
| M4 source, external clean wheel, frozen app | passed | Current 23-stage delivery passes all four runtimes; independent source/new wheel/signed variants reproduce checkpoint and trajectory dispatch |
| M5 actual lifecycle and independent acceptance | passed | Four actual signed C++/reference warm/cold flows, refusal, cancellation, exact restore and codesign; independent C++ warm/cold, three saved-case full replays and all 80 measured / 14 warmup samples pass |
| Original representative ≥2× throughput | pending; measured below target | Final five-pair result 17.421058 → 12.740213 s, **1.3674×**, unchanged 2× fails |
| Original other latency and memory limits | passed | All four current exact workloads, ≤10% latency/memory gates and 1 GiB hard cap pass |
| Supplementary second-pass improvement | passed | Current code vs frozen first C++: simulation 1.0647× and complete RL episodes 1.0760×; five paired fresh processes, exactness and memory pass |

Host: Apple M2 Pro arm64, 16 GiB, Apple Clang 21.0.0, CPython 3.12.14 arm64.
Use this checkout's `src` explicitly: the available development environment is
shared with a neighboring editable checkout. System Python 3.9 is unsupported.
All numerical pools are fixed to one thread; only one heavy job runs at a time.

## Fixed comparison and results

The original comparison baseline is `27f59e85c5b0d6f5b918989bec32976789bf391a`.
Delivery targets `origin/shaf/decision-system-v1`; it is fetched and reconciled
before each normal push. The original manifest SHA256 remains
`f507080eb60f214fd4f050e1595d551bff6222b6fef6cad6d6f7b7a98abb0f67`.
Five alternating fresh-process pairs follow one warmup per backend/workload.
Complete fixture creation plus public computation is timed; imports, audits and
output encoding are separate. Original stress also includes five state digests;
its result is not a mechanics-only speedup. All current samples, variability, exact
hashes, counters, setup/compute splits and memory are retained in the compact
[current trajectory receipt](../artifacts/cpp-acceleration-v1/trajectory-followup-acceptance-v5.json).
The [previous occupancy receipt](../artifacts/cpp-acceleration-v1/occupancy-fastpath-acceptance-v3.json),
[previous motion receipt](../artifacts/cpp-acceleration-v1/motion-followup-acceptance-v2.json) and
[previous final receipt](../artifacts/cpp-acceleration-v1/final-performance-acceptance.json)
retain the earlier five-pair runs separately; versions are never pooled.
The [first-increment receipt](../artifacts/cpp-acceleration-v1/first-increment-acceptance.json)
retains the earlier 1.2313× working increment before the bounded second port.

| Original workload | Pre-port reference median | C++ median | Speedup | Exact / memory / latency |
|---|---:|---:|---:|---|
| Small geometry | 1.881 ms | 1.446 ms | 1.301× | pass |
| Fresh Generate | 17.421058 s | 12.740213 s | 1.367× | exact/memory pass; 2× fails |
| Standard-physics refusal | 1.393655 s | 1.057359 s | 1.318× | pass |
| Geometry + active continuum + state recording | 0.974171 s | 0.517436 s | 1.883× | pass |

The representative performs 370 physical calls (129 previews, 241 steps), exports
112 actions, treats 18/27 target cells and records 334 mm³ other removal in both
backends. Complete snapshots, ordered receipts/contact sets, normalized reports,
and physical-call traces match the captured pre-port source. The UI's different
longer Generate request treats 27/27; its timings are separate observations.
Only enumerated measured wall fields and process-local goal identities are
normalized. Raw report seals and registered goals are independently validated;
there is no generic metadata exclusion or new numerical tolerance.

Supporting five-pair complete-call results: capsule batches 4.59×, scalar segment
predicates 3.01–3.80×, 49 KB source-record JSON detachment 4.39×. Tiny metadata can
be slower because live integrity checks dominate. Angular reuse keeps SciPy's
original math and 64 immutable entries: repeated orientations improve 2.266×,
while distinct cold orientations cost 24.1% more. Original integrated regression
controls decide retention, not the favorable microbenchmark alone.

## Implementation and boundaries

One setuptools/pybind11 C++17 extension provides fused bounded capsule traversal,
exact-refined scalar collision predicates and builtin JSON detachment. No new
build framework, mandatory OpenMP/CUDA, import-time compilation or changed physics.
NumPy refinement retains ambiguous-boundary/sheared-affine behavior; the original
float-valued distance remains authoritative where required. Owned results and
snapshots protect input lifetime. Floating contraction and fast-math are disabled.

`reference|cpp|auto` diagnostics identify the actual module, ABI, compiler, source,
header, recipe and binary. Live source/binary/entrypoint integrity failures
propagate. Auto fallback is limited to unavailable/unsupported inputs and never
retries after mutation. Older optional C-helper admission remains independent.
Production batches group only independent queries inside existing physical
transactions. Maximum 64 items and 32,768 primitive candidates/chunk bound work;
charges, ordering, cancellation and publication checks remain authoritative.
Final independent largest-grid cancellation trials peaked at 0.861 ms, publishing no
covers and leaving accepted state unchanged. There is no new concurrency.

## Bounded second profiling pass

The additive frozen simulation/RL manifest is
`manifests/experiments/cpp-acceleration-simulation-v1.json`, SHA256
`7c4eb89e4b1061f2d3ed08d39c53dc1396f549c93b842e8b0dc4c7b157cd177a`.
Six complete public simulation families exercise target treatment, drilling and
thermal effects, progressive CUSA contact/capacity, vascular occlusion, actual
wound/bleeding and active continuum. There are 53 step attempts, six previews and
51 committed actions. Six complete encoded Gym episodes include success,
protected/budget refusal and four independent seeded instances advanced in fixed
sequential order: 44 calls, 38 commits, 2,178 physical substeps. No training or
PyTorch policy operation executes, so no training speedup is claimed.
Exact outputs, observations, rewards, flags, info, histories and restore/RNG
contracts qualified against reference before optimization. Initial qualification
pairs are diagnostics, not final performance evidence. See the
[profile packet](../artifacts/cpp-acceleration-v1/supplement-profile-summary-v1.json).

| Investigated area | Measured attribution and decision |
|---|---|
| Collision, distance, swept motion | Query callers occupy 18.6% simulation / 35.1% RL, but new native kernel self time is <1%. Conversion, repeated admission and crossings dominate surrounding code. Keep current bounded grouping; do not remove integrity checks or change callback visibility. |
| State copying, clones, checkpoints | Checkpoints 9.90% simulation / 10.63% RL inclusive before the copier. Post-copier timed-only planning profile is 2.59%. Retained one bounded alias/cycle-preserving native copier after integrated measurements. |
| Continuum topology, assembly, history | Mesh 0.0032 s, assembly 0.0234 s, pack/unpack 0.0128 s in a 2.222 s profiled simulation suite. Retain effective NumPy/sparse operators and existing history solvers. |
| Drilling, thermal, aspiration, vascular | Apply paths take 0.0024 / 0.0082 / 0.0101 s. Vascular tick 0.1554 s includes SciPy `solve_ivp` 0.1154 s. Broad ports have poor return. |
| Repeated steps and independent environments | 1,333 simulation substeps; real native batch histogram `{1:881,2:263,3:34}`. RL has 2,547 singleton cover queries. No safe measured justification for concurrency or changing physical attempts. |
| RL observations/features | `_observe` 0.0864 s (3.48% inclusive; 2.16% self), reward/codec smaller. Leave the Python interface unchanged. |

Inclusive profile costs overlap and are not speed measurements. Even removing
the pre-copier checkpoint cost entirely estimated only 1.110× simulation
and 1.119× RL; it cannot by itself establish the original 2× target. Required
integrity checks are not a recoverable performance budget. The second experiment
must include binding, conversion, copying, full throughput and fresh memory.

The retained copier uses the same extension and existing Python interfaces, with
262,144-item / 64-depth bounds, private memo publication, owned snapshots and exact
aliases/cycles/float bits/order/component exemptions. Unsupported graphs leave the
caller memo untouched and fall back once; errors propagate. Startup code and
stock-default qualification plus live hook checks preserve both pre-import and
later deepcopy customizations. No broad proof scanner exception was added.
The final focused suites pass **170 checks per backend**. Planning checkpoint and
graph helpers improve 3.31× / 4.05×; smaller helpers improve 1.60–2.42×, below the
3× supporting aim. Near-cap distinct containers take at most 21.9 ms observed.
Late unsupported graphs cost ~23% more than reference; this limit is retained in
the [helper receipt](../artifacts/cpp-acceleration-v1/native-engine-copy-evidence.json).

| Second pass complete workload | Captured first C++ median | Current C++ median | Gain | Timed-work RSS before → after |
|---|---:|---:|---:|---:|
| Six simulation families | 1.216864 s | 1.142910 s | 6.47% throughput | 109,969,408 → 109,969,408 B |
| Six complete Gym episodes | 1.248829 s | 1.160598 s | 7.60% throughput | 101,613,568 → 101,695,488 B |

These are five alternating fresh-process pairs, including bindings/conversion and
production copies/hashes. Every after sample is faster than every before sample;
all exact frozen outcomes and memory gates pass. Simulation: 43.55 → 46.37 public
attempts/s; RL: 4.804 → 5.170 complete episodes/s. Actual engine graph calls:
1,566 simulation / 2,334 RL, zero fallback. Original planning uses 11,901 native
graph copies, zero graph/geometry fallback; JSON detachment has seven documented
unsupported identity-graph fallbacks. Final planning median memory falls
150,192,128 → 141,983,744 B in the current follow-up. Supplementary memory uses
the pre-audit high-water mark; full-process restore audit still enforces the
1 GiB cap. Earlier five-pair
runs remain retained: before the copy startup guard (2.92% simulation / 2.46% RL),
after it (2.59% / 2.87%), and after motion qualification (3.26% / 3.51%). The
previous v3 run includes the guarded occupancy fix; the current v5 run also
includes trajectory batching and shared guard corrections. These supplementary
controls measure the accumulated changes from the first C++ capture and do not
isolate one optimization. The occupancy and trajectory planning increments are
measured separately below. Versions are never pooled, and variation between
versions is not attributed to an individual fix.
No additional port is justified by this bounded pass.

## Residual profile and compatibility follow-up

A fresh post-copier planning profile at `11aaa79c`, before the compatibility
follow-up, covers only the timed fixture and public computation. Earlier planning
profiles also captured untimed imports and result audits, so their attribution
is superseded; ordinary benchmark timing and both frozen manifests are unchanged. The runner asserts that audit functions never
appear in timed profiles. Supplementary profiles already used this boundary.
The 22.584 s profiled run preserves the exact original result and physical trace;
its timings are attribution, not a speed comparison.

In that post-copier profile, checkpoint copying occupies 0.585 s (2.59% inclusive). Swept motion occupies
2.942 s (13.03%), envelope creation 2.022 s (8.95%), owned-state digest work
2.008 s (8.89%), and occupancy lookup 1.369 s (6.06%). These overlap. Occupancy
and rotations already contain compiled NumPy/SciPy work; required live native
integrity checks occupy 2.630 s and cannot be discarded for speed.

A read-only observation of original motion calls found hypothetical 64-entry
cache hit rates of 75.77% for envelopes and 82.77% for complete sweeps. After
retaining existing helper qualification, the optimistic sweep estimate is only
1.087× current throughput before new keys, detached outputs and cold overhead.
Dimension-only keys do not improve the hit rate. This additional cache is deferred;
it cannot plausibly close the remaining fixed 2× gap. See the compact
[residual profile receipt](../artifacts/cpp-acceleration-v1/postcopy-profile-review-v1.json).

The angular-cache pre-import compatibility fix at `a0d2cd05` keeps customized
SciPy calls visible and snapshots keyword defaults without invoking user key
equality. Six focused regressions pass, including stable proof identities.
Current source/wheel/frozen/signed probes still require genuine stock SciPy
qualification. The [prior motion receipt](../artifacts/cpp-acceleration-v1/motion-followup-acceptance-v2.json)
retains its separately audited 23-stage delivery and five-pair measurements.

## Delivery, actual application and defects fixed

Immutable build capture includes native sources/headers, configuration, policy,
licenses and only proof-required SciPy source wrappers. Build/install compiles the
captured wheel before freezing; relocated binary identity is rebound and verified
after signing. Source, external clean wheel, frozen engine and signed-app probes
check native execution, explicit reference, absent-auto fallback, absent-C++
failure, present-corrupt-auto failure, genuine GC within proof scope, and replay.

Actual signed executable workflows use the existing Electron harness through a
pre-main debugger launcher, detached before computation. They run
`app.isPackaged=true` with original sidecar routing and real renderer/preload.
The current complete 23-step verifier passes at `40693c6d`. Source: **747 passed / 77 intentional
older-helper skips**; reference: **276 passed / one skip**. Desktop main/renderer/viewer:
**325 / 716 / 275 passed**; TypeScript and production build pass. All runtime
variants in source, clean wheel, frozen sidecar and signed app pass. Captured
source digest `8bdfc3e2…0bbacb`, source native `bfc5b705…9775a`, signed native
`32fd4fdb…db338`; the current engine is `aade4161…1c76a6` and app CDHash
`763c7f15…ca35d0`. Exact identities are in the current follow-up receipt. The app
is locally ad hoc signed, not notarized.

C++ and reference each complete fresh Generate → Compare → select → replay,
pause/seek → Save → terminate → cold Open. Comparison, insertion order, selected
branch, full view/timeline, actions, ledger and source anatomy restore; reopening
performs no planning. App signature remains valid after execution. The compact
[current delivery receipt](../artifacts/cpp-acceleration-v1/trajectory-followup-acceptance-v5.json)
identifies actual files and retained logs. The [previous delivery receipt](../artifacts/cpp-acceleration-v1/packaged-delivery-evidence.json)
retains the prior increment separately. A narrow legacy-CUSA request correction
omits unsupported duty in legacy mode and retains it in V2. Both warm flows now
prove a well-formed `UNSUPPORTED_EXTRACTION` domain refusal with unchanged accepted
state and no unauthorized action, rather than a malformed-parameter rejection.

Concrete fixes found by these checks: existing batching callback qualification
needed its helper anchor; four transport encoders sorted ordered branch/contact
maps; GC changed accidental proof inclusion of empty uppercase owner caches;
frozen caches wrote under installed sources; PyInstaller omitted the actual
proof-required SciPy wrapper. Fixes retain canonical hashing, size/admission
checks, atomic saves, proof scanning and physical behavior. Native tamper,
callback mutation, replay, transport and GC regressions cover them.

Earlier retained failures also include an initial parallel predicate crossing
regression, a rejected functools wrapper that broke qualification, missing strict
Gym notice pins, local desktop dependency/setup failures and a baseline test that
mutated frozen source. All failed receipts remain under ignored output folders;
no workload, safety threshold or notice guard was relaxed to hide them.

## Reproduction, ownership and next action

Use the project Python 3.12 environment and locked desktop dependencies. In a
fresh checkout, first recreate the original reference capture (the existing capture
was checked byte-for-byte against Git; its only extra file is the portable fixture):

```sh
mkdir -p outputs/cpp-acceleration-v1/baseline/source
git archive 27f59e85c5b0d6f5b918989bec32976789bf391a src | tar -x -C outputs/cpp-acceleration-v1/baseline/source
cp src/resectionlab/generated_cpp_fixture.py outputs/cpp-acceleration-v1/baseline/source/src/resectionlab/
```

For a fresh supplementary baseline, rebuild pre-copier commit `f5a67817` with
the same Python/compiler and use its `src` with `--baseline-backend cpp`. This
run retained an immutable source/binary capture at `supplement-prepass-cpp`;
rebuilt binary identities may differ and are reported separately. A Git rebuild
uses the existing build, without editing the active checkout (verified from Git
with Python 3.12 and an exact frozen simulation result):

```sh
TASK_CHECKOUT="$PWD"
mkdir -p outputs/cpp-acceleration-v1/pre-copier-rebuild
git archive f5a67817 src setup.py pyproject.toml | tar -x -C outputs/cpp-acceleration-v1/pre-copier-rebuild
(cd outputs/cpp-acceleration-v1/pre-copier-rebuild && "$TASK_CHECKOUT/.venv/bin/python" setup.py build_ext --inplace)
```

Then run (substitute the rebuilt baseline path when reproducing on a new checkout):

```sh
PYTHONPATH=src .venv/bin/python scripts/verify_cpp_acceleration.py \
  --output testcases/outputs/<fresh-name> --node /absolute/path/to/node
PYTHONPATH=src .venv/bin/python scripts/benchmark_cpp_acceleration.py \
  --baseline-source outputs/cpp-acceleration-v1/baseline/source/src \
  --output outputs/cpp-acceleration-v1/<fresh-name>
PYTHONPATH=src .venv/bin/python scripts/benchmark_cpp_acceleration.py --supplemental \
  --baseline-source outputs/cpp-acceleration-v1/supplement-prepass-cpp/src \
  --baseline-backend cpp --output outputs/cpp-acceleration-v1/<fresh-name>
PYTHONPATH=src .venv/bin/python scripts/benchmark_cpp_acceleration.py \
  --mode profile --backend cpp --workload generate \
  --output outputs/cpp-acceleration-v1/<fresh-profile-name>
```

The full verification command builds source/wheel/frozen/app sequentially, checks
all runtime variants, runs affected Python/desktop groups and four actual signed
warm/cold workflows. The preceding independent increment passed **82 focused
checks**, including analytic
geometry, generated RL, ownership, final bounded cancellation and genuine-owner
GC proofs. Pre/post-load graph-entrypoint substitution fails with integrity
errors; actual native copies preserve full/light/proof identities. The reviewer
reproduced three delivered runtimes and signed C++ warm/cold execution, then
validated all three native saved cases, seals, full replay views/timelines, ordered
branches and normalized reports. The [final review packet](../artifacts/cpp-acceleration-v1/independent-final-review.json)
records commands and raw evidence; the original 2× performance gate still
prevents goal completion.

Code commits: `b59983b1` foundation, `54afb435` native integration, `235fc77f`
callback ordering, `1c7e0057` transport ordering, `15b17f05` GC-stable owner caches,
`f5a67817` supplementary controls/profile, `0f0a254a` CUSA capability fix,
`a645ac2c` delivery, `01ffb880` measured graph copier and `2d5dde6b` complete
verifier, `11aaa79c` published acceptance and `a0d2cd05` motion-hook compatibility
and timed-only profiling, `37e81bad` its acceptance, and `8438a046` guarded
occupancy lookup. Each uses author/committer `shafsthegoat <shafrir.p@gmail.com>` and the
required `Co-authored-by: skamal23 <sayemkamal12@gmail.com>` trailer.

The preceding independent increments are recorded in the linked historical
receipts. Their samples remain separate from current v3 measurements; unchanged
native kernel, bounded cancellation and proof/tamper evidence is reused because
the native source, header and loaded binaries have not changed. The original 2×
gate remains incomplete. Patient validation stays a separate unavailable task;
no external-volume recheck or patient download is needed for engineering work.

## Occupancy follow-up at `8438a046`

A bounded endpoint-rotation grouping probe was rejected before implementation:
SciPy's scalar and grouped matrix paths produce a one-ULP endpoint difference
(seed 817, brain retractor, case 97). No collision tolerance was introduced.
Raw controls are retained in `outputs/cpp-acceleration-v1/endpoint-batch-exactness-v1`.

The original planner observation confirms 51 mixed Python/NumPy integer owners
cause all 14,346 `np.isin` fallback lookups (20.9% of 68,648 queries). Their IDs
come from the engine's already-owned `int64` retained array. One internal
`retained.tolist()` conversion restores the existing integer lookup strategies;
public caller-supplied unusual scalar types retain their fallback. Review caught
an unconditional-conversion defect with a colliding custom integer key: ownership
changed from `[31, 31]` to `[0, 31]`. The retained fix restricts normalization to
existing exact-builtin-integer cleared sets and retains original equality
operands for unusual keys. The failed reproduction is retained. No new native
surface or numerical algorithm is needed. Two focused ownership/callback
regressions cover the correction; the existing complete verifier now includes
occupancy lookup/key/query tests in both modes.

A separate immediately-prior C++ capture is frozen at
`outputs/cpp-acceleration-v1/occupancy-prepass-cpp-v3`. A five-pair increment-only
Generate experiment uses the original benchmark workers and exact frozen work;
it cannot replace the original pre-port comparison or lower its 2× target.
Focused checks pass: 83 tests in each backend after the custom-key correction,
plus the preceding 184 C++ / 183 reference transport checks. The final planner
matches the exact frozen result, 370 physical calls and 112 actions. Five paired
fresh processes now measure **13.304752 → 12.803883 s**, **1.0391×** against the
immediately prior C++ implementation; every after sample is below every before
sample and memory passes. This increment-only evidence does not mix runs with
the original reference comparison. Code commit `8438a046` passes the complete
delivery checks above. The unchanged original and
supplementary paired comparisons pass exactness, memory and all non-2× latency
gates. Current Generate is 1.3487×, below the unchanged 2× gate. Independent
execution passes 83 tests per backend, 15 source/wheel/signed runtime variants,
actual signed C++ warm/cold workflows, strict signature checks and three native
saved-case seals/full replay comparisons. The reviewer audited all 70 measured
samples and 14 separate warmups, including canonical bytes, raw report/goal seals,
source roles, counters, memory, order and statistics. No code defect remains in
the review; the original 2× acceptance gap remains open. See the
[occupancy receipt](../artifacts/cpp-acceleration-v1/occupancy-fastpath-acceptance-v3.json).

Other investigated seams: the no-copy native state-hash admission estimate used
Python self time only (about 3.6% ideal gain), excluding validation child costs.
It would need new caller/hook qualification; the v4 investigation below measures
it separately. CUSA search repeats a tip-contact query, but
removing it must preserve dynamically replaced callbacks and cancellation checks;
no shortcut is added to its authoritative contact intersection.

The implementation and review owners have finished the occupancy increment.

## Owned-state validation investigation at `aae12ac3`

The fresh timed-only profile preserves the exact frozen Generate result. Its
21.876 s of profiled work includes 3,838 state-ID hashes (1.541 s inclusive;
0.787 s self) and 63,208 small occupancy hashes. The earlier 3.6% estimate used
only Python self time, not all validation children. Current self-time elimination
would imply 1.0373×; removing the entire state-ID hash, including required encoding
and hashing, would imply only 1.0758×. These are ideal profile estimates, not
measured speedups. The original 2× target still requires another 32.6% elapsed-time
reduction from the current integrated result.

Five alternating helper samples (1,000 calls/sample) compare the unchanged owned
hash, existing compiled JSON copy plus hash, and encoding/hash without validation.
The existing copier improves the actual completed 35 KB state 247.9 → 184.8 µs,
but slows the fresh state 11.32 → 12.14 µs and a tiny occupancy key 1.37 → 6.17 µs.
An encoder callback also distinguishes the original root from the detached copy
and produces different hashes. A blanket substitution is rejected. No new native
surface or production code has been added by this investigation.

An isolated C++ no-copy validator, built with the existing setup, preserves the
encoder's original input and passes its bounded type/alias/cycle/error controls.
Five alternating helper samples measure fresh state 11.494 → 11.177 µs, completed
state 248.619 → 155.044 µs, and tiny occupancy 1.330 → 6.646 µs. The fresh-state
saving is only 0.317 µs before caller/builtin/prefix qualification; routing the
63,208 small occupancy hashes through it would regress them. Applying the
completed-input saving to all 3,838 state-ID calls yields a hypothetical 0.359 s
before qualification. Inputs evolve, so this is neither a bound nor a prediction
or an integrated speedup. Production
promotion is declined: the helper win does not justify the required selection and
callback machinery. The isolated prototype is retained only as experimental
evidence; no new adapter, installed/frozen qualification or accepted native API
was added.

Read-only review of the other large categories found no new high-impact numerical
seam. Contract validation mixes compiled hashing/NumPy and required source checks;
its largest simple cell-validation loop is 0.403 s (1.84% of this profile).
Effect validation is 0.528 s (2.41%) of authoritative edge/set checks. Checked
export replay is 6.712 s but includes required physics/proof execution and its
physical calls. Geometry implementation qualification is 1.835 s. The capsule
adapter takes 0.303 s including checks; the actual C++ binding takes 0.055 s.
These inclusive costs
overlap; porting whole categories or skipping replay/checks cannot be justified
from their headline times.

The bounded implementation/review owners have finished and released the heavy
slot. Independent audit passes: 105 helper samples, current timed-only profile,
full frozen Generate result/seal/work, production/capture isolation, source and
binary identities, callback controls and all reported statistics. Current evidence is in the
[ROI receipt](../artifacts/cpp-acceleration-v1/owned-admission-roi-v4.json).
At that v4 investigation, the prior integrated acceptance remained applicable
because production source, headers and binaries were unchanged. The next bounded
increment follows below. The original 2× target remains unmet.

## Retained trajectory increment at `40693c6d`

The next measured seam is repeated scalar SciPy interpolation: 7,816 calls take
0.574 s inclusive in the current timed-only planning profile. This work is
already partly compiled. Bounded batches may remove repeated Python dispatch
while retaining SciPy's numerical implementation and every physical substep.
No new C++ entrypoint or physics algorithm is proposed.

Before implementation, 56,165 quaternion rows in 219 generated trajectories and
1,869 chunks match scalar output bits exactly. Coverage includes the actual
fixture orientations/counts, signed zeros, random/near-zero/near-pi/antipodal
rotations and two full 20,000-step controls. Each returned row is privately owned.
Before the final guard corrections, with selection and helper qualification
repeated per row, five paired helper samples improve about 2.1×. These are
supporting measurements, not the retained helper's performance, integrated
acceptance or evidence for the original 2× gate.

The preceding C++ source/binary is captured at
`outputs/cpp-acceleration-v1/trajectory-prepass-cpp-v5`, with all 334 files checked.
The unchanged workers ran ten alternating fresh-process pairs with one
warmup per role. Before timing, retention required exact frozen work, memory and
hard caps, a positive median gain, and a positive lower 95% paired-bootstrap
geometric-speedup interval. No post-hoc extension of repetitions is planned.
The original pre-port comparison and target remain separate and unchanged.

The motion owner has frozen the small local 32-row buffer in
`surgical_motion.py` and `surgical_engine.py`; the coordinator owns measurement
and delivery, and the independent reviewer checks behavior and fairness. Review
found three concrete issues before timing: a callback can change NumPy underflow
policy after prefetch, a replacement method descriptor can masquerade as the
original function, and instance attribute hooks can be hidden by cached work.
These also expose defects in the existing angular cache. Their failed controls
are retained. Shared source-qualified policy, raw descriptor and bounded class
protocol checks now preserve scalar dispatch for these cases. The fixes remain
required independently of the batching retention decision.

Focused checks pass: 85 C++ tests; 84 reference tests and one intentional skip.
Proof epochs/fingerprints remain qualified. Ten alternating fresh-process pairs
pass the declared retention gate: **12.928910 → 12.627930 s**, **1.02383×** median
speedup, with paired geometric speedup 1.02222× and bootstrap 95% interval
**[1.01633, 1.02736]**. Exact frozen work and memory pass (median RSS
141,934,592 → 141,672,448 B). This measures the combined guard corrections and
batching against the preceding captured C++ implementation; the gain cannot be
attributed solely to batching. Both changes are retained.

A complete public 33-substep runtime probe now observes C++-selected SciPy batches
`[32, 1]` versus 33 scalar reference calls, with identical quaternion bytes and
clock accounting. Source execution passes all five available/unavailable/integrity
variants. Its first diagnostic run used the wrong frame argument name; only the
observer was corrected, and that failed receipt is retained.

Fresh timed-only supplementary profiles retain both exact frozen outputs. They
cover 1.953 s simulation / 2.146 s RL of profiled work. Checkpoint calls take
0.044 / 0.064 s inclusive and transition clones 0.0036 / 0.0022 s after the
existing native copier. Simulation mesh/assembly/history remain 0.0032 / 0.0229 /
0.0121 s; vascular integration includes 0.115 s in SciPy's `solve_ivp`, which mixes
Python integration control and compiled array operations. RL observations take
0.0860 s inclusive (4.01%), with 0.0537 s Python
self time. The new underflow qualification costs 0.0063 / 0.0114 s inclusive.
These overlapping attribution values are not before/after speed evidence. The
six-area inventory above remains applicable: no further numerical port is
justified by these small residual seams. The refreshed 23-stage delivery now
passes at `40693c6d`, including all four runtimes, all four actual signed workflows
and the final strict signature check. The unchanged five-pair original comparison
now measures 1.3674× Generate, below 2×, with all other gates passing. Five-pair
supplementary measurements show 1.0647× simulation and 1.0760× complete RL
throughput against the frozen first C++ capture. Independent execution passes
85 C++ / 84 reference focused tests and one reference skip, 15 source/new-wheel/
signed runtime variants, signed C++ warm/cold flows, strict codesign and three
saved-case native seals/full replays. All 80 measured samples and 14 separate
warmups pass the receipt audit, including original raw report seals, goal
registrations, exact work, source roles, order, counters, RSS and statistics.
The ten-pair improvement is supported by nine faster pairs; no claim is made that
every after sample beats every before sample. No delivered-code defect remains.
The engineering gates pass; the original 2× performance gate remains open and
requires another 31.6% elapsed-time reduction from the current C++ median. See the
[trajectory receipt](../artifacts/cpp-acceleration-v1/trajectory-followup-acceptance-v5.json).

Implementation, measurement, packaging and independent review owners have finished
this increment. No heavyweight job remains. The next port requires a new measured
seam with enough numerical work to justify its complexity; the bounded current
pass does not support a wholesale rewrite or removing authoritative checks.

Delivery target is `origin/shaf/decision-system-v1`, using a normal fast-forward
push. Verify local/remote identity with `git rev-parse HEAD` and
`git ls-remote origin refs/heads/shaf/decision-system-v1`. The final response
records the verified pushed commit; binaries and bulky raw evidence remain ignored.
