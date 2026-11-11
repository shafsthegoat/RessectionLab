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
| M4 source, external clean wheel, frozen app | passed | Final source/wheel/frozen/packaged variants pass, including stock-loader qualification and actual native checkpoint; independently reproduced |
| M5 actual lifecycle and independent acceptance | passed | Four actual signed C++/reference warm/cold flows; valid domain refusal, cancellation, exact restore, final codesign; independent C++ warm/cold and native saved-case replay reproduce |
| Original representative ≥2× throughput | pending; measured below target | Final five-pair result 17.440052 → 13.241774 s, **1.3170×**, unchanged 2× fails |
| Original other latency and memory limits | passed | All four final exact workloads, ≤10% latency/memory gates and 1 GiB hard cap pass |
| Supplementary second-pass improvement | passed | Copier retained: simulation 1.0259× and complete RL episodes 1.0287×; five paired fresh processes, exactness and memory pass |

Host: Apple M2 Pro arm64, 16 GiB, Apple Clang 21.0.0, CPython 3.12.14 arm64.
Use this checkout's `src` explicitly: the available development environment is
shared with a neighboring editable checkout. System Python 3.9 is unsupported.
All numerical pools are fixed to one thread; only one heavy job runs at a time.

## Fixed comparison and results

Baseline/fetched target is `27f59e85c5b0d6f5b918989bec32976789bf391a` on
`origin/shaf/decision-system-v1`. The original manifest SHA256 remains
`f507080eb60f214fd4f050e1595d551bff6222b6fef6cad6d6f7b7a98abb0f67`.
Five alternating fresh-process pairs follow one warmup per backend/workload.
Complete fixture creation plus public computation is timed; imports, audits and
output encoding are separate. Original stress also includes five state digests;
its result is not a mechanics-only speedup. All final samples, variability, exact
hashes, counters, setup/compute splits and memory are retained in the compact
[performance receipt](../artifacts/cpp-acceleration-v1/final-performance-acceptance.json).
The [first-increment receipt](../artifacts/cpp-acceleration-v1/first-increment-acceptance.json)
retains the earlier 1.2313× working increment before the bounded second port.

| Original workload | Pre-port reference median | C++ median | Speedup | Exact / memory / latency |
|---|---:|---:|---:|---|
| Small geometry | 1.832 ms | 1.379 ms | 1.328× | pass |
| Fresh Generate | 17.440052 s | 13.241774 s | 1.317× | exact/memory pass; 2× fails |
| Standard-physics refusal | 1.404199 s | 1.081570 s | 1.298× | pass |
| Geometry + active continuum + state recording | 0.981038 s | 0.515654 s | 1.903× | pass |

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
| State copying, clones, checkpoints | Checkpoints 14.44% planning / 9.90% simulation / 10.63% RL inclusive. Retained one bounded alias/cycle-preserving native copier after integrated measurements. |
| Continuum topology, assembly, history | Mesh 0.0032 s, assembly 0.0234 s, pack/unpack 0.0128 s in a 2.222 s profiled simulation suite. Retain effective NumPy/sparse operators and existing history solvers. |
| Drilling, thermal, aspiration, vascular | Apply paths take 0.0024 / 0.0082 / 0.0101 s. Vascular tick 0.1554 s includes SciPy `solve_ivp` 0.1154 s. Broad ports have poor return. |
| Repeated steps and independent environments | 1,333 simulation substeps; real native batch histogram `{1:881,2:263,3:34}`. RL has 2,547 singleton cover queries. No safe measured justification for concurrency or changing physical attempts. |
| RL observations/features | `_observe` 0.0864 s (3.48% inclusive; 2.16% self), reward/codec smaller. Leave the Python interface unchanged. |

Inclusive profile costs overlap and are not speed measurements. Even removing
checkpoint cost entirely estimates only 1.169× current planning, 1.110× simulation
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

| Second pass complete workload | Captured first C++ median | Retained copier median | Gain | Timed-work RSS before → after |
|---|---:|---:|---:|---:|
| Six simulation families | 1.159233 s | 1.129989 s | 2.59% throughput | 109,756,416 → 109,740,032 B |
| Six complete Gym episodes | 1.221913 s | 1.187788 s | 2.87% throughput | 101,580,800 → 101,564,416 B |

These are five alternating fresh-process pairs, including bindings/conversion and
production copies/hashes. Every after sample is faster than every before sample;
all exact frozen outcomes and memory gates pass. Simulation: 45.72 → 46.90 public
attempts/s; RL: 4.910 → 5.051 complete episodes/s. Actual engine graph calls:
1,566 simulation / 2,334 RL, zero fallback. Original planning uses 11,901 native
graph copies, zero graph/geometry fallback; JSON detachment has seven documented
unsupported identity-graph fallbacks. Final planning median memory falls
151,371,776 → 143,245,312 B. Supplementary memory uses the pre-audit high-water
mark; full-process restore audit still enforces the 1 GiB cap. An earlier five-pair run before the startup hook guard remains retained
(2.92% simulation / 2.46% RL gains). The final five-pair run above includes that
guard and directly measures its fresh-process memory; versions are not pooled.
No additional port is justified by this bounded pass.

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
The complete 23-step verifier passed. Source: **644 passed / 77 intentional
older-helper skips**; reference: **146 passed**. Desktop main/renderer/viewer:
**325 / 716 / 275 passed**; TypeScript and production build pass. All runtime
variants in source, clean wheel, frozen sidecar and signed app pass. Captured
source digest `69ace554…959f5`, source native `bfc5b705…9775a`, signed native
`32fd4fdb…db338`; exact identities are in the linked receipts. The app is locally
ad hoc signed, not notarized.

C++ and reference each complete fresh Generate → Compare → select → replay,
pause/seek → Save → terminate → cold Open. Comparison, insertion order, selected
branch, full view/timeline, actions, ledger and source anatomy restore; reopening
performs no planning. App signature remains valid after execution. The compact
[delivery receipt](../artifacts/cpp-acceleration-v1/packaged-delivery-evidence.json)
identifies actual files and retained logs. A narrow legacy-CUSA request correction
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
```

The full verification command builds source/wheel/frozen/app sequentially, checks
all runtime variants, runs affected Python/desktop groups and four actual signed
warm/cold workflows. Independent final execution passes **82 focused checks**, including analytic
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
verifier. Each uses author/committer `shafsthegoat <shafrir.p@gmail.com>` and the
required `Co-authored-by: skamal23 <sayemkamal12@gmail.com>` trailer.

The final source-version supplementary check also passes; receipts and this
acceptance record are integrated. All code, packaging and independent review
owners have finished; no additional port is
justified by this bounded pass. The fixed 2× target stays incomplete. Required
remaining latency sits in checked orchestration, allocation/hashing and existing
compiled numerical operators; the measured checkpoint ceiling never justified a
claim that this port alone could close the gap. Patient validation remains a
separate unavailable task, with no external-volume rechecks or patient downloads.

Delivery target is `origin/shaf/decision-system-v1`, using a normal fast-forward
push. Verify local/remote identity with `git rev-parse HEAD` and
`git ls-remote origin refs/heads/shaf/decision-system-v1`. The final response
records the verified pushed commit; binaries and bulky raw evidence remain ignored.
