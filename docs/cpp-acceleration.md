# C++ acceleration acceptance record

Goal: integrate a behavior-preserving compiled backend into the existing Python
simulator and Mac Electron application. The fixed fresh planning workload must
reach at least 2× reference throughput. Kernel speed alone does not pass.
Patient validation remains unavailable and is separate from these engineering gates.

## Current state

- Baseline and fetched target: `27f59e85c5b0d6f5b918989bec32976789bf391a`,
  `origin/shaf/decision-system-v1`; clean start on `codex/cpp-acceleration`.
- Host: arm64, 16 GiB RAM, Apple Clang 21.0.0. System Python 3.9 is unsupported;
  use the available arm64 Python 3.12 environment and record its exact ABI in evidence.
- Existing generated legacy-physics cranial control and standard-physics refusal
  remain distinct. Zero completed patient tumor tasks; no clinical claims.
- External data volume is unavailable. No patient acquisition, paid service,
  physics/qualification relaxation, or source mirror modification is authorized.
- Deterministic single-thread execution; one heavyweight job at a time.

| Gate | Status | Current evidence / next dependency |
|---|---|---|
| M1 fixed workloads, contracts and profiles | passed | Portable fixture, exact baseline hashes/programs and frozen manifest; actual profile + pre-port timing; 3 fixture/contract checks passed |
| M2 integrated compiled kernels | passed | Actual fresh Generate uses C++ cover/JSON kernels, exact snapshots and normalized reports/call traces; first increment 15.9964 s |
| M3 production multi-item batching | pending | Search owner locating independent pure geometry queries |
| M4 source, clean wheel, frozen engine | pending | Source native/reference/unavailable/integrity variants passed; clean wheel/frozen/app builds remain |
| M5 actual fresh desktop lifecycle, independent review | pending | Promote existing generated workflow into reusable harness; review after integration |
| Representative ≥2× throughput | pending | Five alternating paired fresh-process samples required |
| Other latency and memory regression gates | pending | ≤10% versus reference; tiny fixed loading allowance must be frozen before tuning |

## Ownership and integration

- Coordinator: this record, shared integration, verification scheduling, commits,
  reconciliation and push. All new commits use author
  `shafsthegoat <shafrir.p@gmail.com>` and coauthor
  `skamal23 <sayemkamal12@gmail.com>`.
- `baseline_contracts`: reusable generated fixtures, benchmark runner, fixed
  `manifests/experiments/cpp-acceleration-v1.json`, focused fixture tests.
- `native_kernels`: C++/binding sources, Python backend/geometry dispatch,
  setuptools configuration, dependency notice, differential kernel tests.
- `batch_search`: production search/geometry batching consumer and deterministic,
  accounting and cancellation checks; coordinate cache ownership with native owner.
- `packaging_workflow`: frozen build capture, actual generated Electron harness,
  runtime probes and packaging checks. No build starts without the coordinator slot.
- Independent review: assign after a testable integrated increment exists.
- `state_kernels`: measured JSON detachment/copying kernels in the same extension,
  `core.py`, dedicated native header and differential tests; engine changes go
  through its existing owner. `independent_review` now audits the design and
  owns independent analytic/native-boundary and generated RL parity checks.

## Decisions and evidence

2026-11-11: read the complete supplied goal and ponytail skill. Fetched origin;
the target remains the inspected baseline. Reused the clean existing acceleration
worktree and preserved the other active worktrees. No nested AGENTS.md or existing
PLANS.md found. Read current decision, continuum and rehearsal performance records;
prior NumPy, cache, C helpers and sparse solvers are part of reference behavior.
Reviewed official pybind11 setuptools, NumPy buffer and GIL guidance. Compile at
installation/build time, never import time; hold Python ownership outside GIL-free
computation and validate the native boundary explicitly.

First actual generated production profile: 36.289 s with profiler overhead,
370 physical calls (129 previews, 241 steps), 112 exported actions, 18/27 target
cells treated and 334 mm³ non-target cost. Capsule coverage: 20,689 calls,
1.995 s inclusive (5.5%); its infinite-speed ceiling is
`1 / (1 - .055) = 1.058×`, insufficient. Repeated scalar segment separation:
106,046 calls / 3.765 s inclusive. JSON detachment: about 32 million recursive
calls / 6.054 s inclusive; semantic digests total 7.005 s. Microstep rollback:
7,814 calls / 3.953 s inclusive; deepcopy: about 6.3 million calls / 4.165 s.
These inclusive values overlap and are attribution, not throughput evidence.
Raw profile: ignored `outputs/cpp-acceleration-v1/baseline/generate-profile.json`.
Decision: finish the first narrow cover path, then add measured copying/scalar
query kernels in the same build rather than converting entire modules.

The first unprofiled baseline attempt met an in-progress missing adapter import.
It is retained as a failed setup attempt; the baseline owner captured original
Git source plus the portable fixture to isolate subsequent pre-port measurements
from concurrent implementation edits. No workload replacement is permitted.

Pre-port unprofiled fresh Generate: 18.5142 s computation + 0.16947 s fixture =
18.68367 s complete; peak RSS 144,932,864 bytes. Retained receipt:
`outputs/cpp-acceleration-v1/baseline/pre-port-unprofiled.json`; original source
under `outputs/cpp-acceleration-v1/baseline/source/src`. Manifest fixes five
alternating pairs, one warmup/backend/workload, 1 GiB worker hard cap, ≤10%
memory/other latency regression, 16 MiB tiny loading allowance and 100 ms native
cancellation budget with 32,768 candidates/chunk. Only explicitly enumerated
measured wall fields and their independently validated derived report seal are
projected from comparison. No floating tolerance fields were introduced.

First compiled increment: 116 passed / 1 skipped in 1.33 s across geometry,
batch, state-copy, semantic and independent acceptance controls. Five alternating
paired complete two-query capsule calls: reference median 11.3343 ms, C++
2.46921 ms (4.590×), exact ordered outputs. This includes binding/conversion
and copies; it does not pass the integrated throughput gate. Receipt:
`outputs/cpp-acceleration-v1/native-capsule-total-call.json`.

Actual first integrated Generate: 15.9964 s complete, versus captured pre-port
18.68367 s (1.168×) and revised reference 17.7218 s (1.108×). This is one
sample each, not final acceptance. C++ geometry: 16,697 bindings / 20,693 items,
maximum three items; histogram `{1:12709,2:3980,3:8}`. JSON-copy bindings: 16,098.
Full snapshots, physical-call traces and normalized reports match pre-port.
First small/refusal/stress controls also match exactly. The frozen stress coupon
has 128 active elements / 840 DOFs, nonzero viscoelastic work, five distinct
states and one removed source cell. Compact initial receipt:
`outputs/cpp-acceleration-v1/first-integrated/initial-increment-summary.json`.

Protocol v2 corrects one pre-existing process-bound identity comparison:
`goalHash` includes limits' monotonic origin and meter object address. Each run
retains actual goals through the supported progress callback, verifies the
registered raw goal/limits seals after timing, and compares complete declarations
normalized only for those two process-local fields. All raw values remain in
diagnostics. No production function is replaced during measurement. The initial
mismatch and original protocol hash remain recorded. Empty-array result encoding
was corrected without changing queries; the 13³ continuum coupon was fixed
before the first manifest freeze, rather than replacing a measured workload.
The legacy fixture exporter/loader now preserves its saved mapping and physics;
four focused fixture/contract/cold-load checks passed in 0.73 s.

Residual C++ profile (33.5965 s with profiler overhead): swept envelopes 6.631 s;
deepcopy 4.284 s; microstep checkpoint 4.046 s; scalar segment distance 3.906 s.
Repeated source validation fell from 6.135 to 1.155 s. Next work targets these
measured residuals. Supporting JSON-copy five-pair microbench: full helper
4.39× on the 49,089-byte source record; tiny 852-byte metadata is slower because
fixed integrity checks dominate. Whole-workload evidence remains authoritative.

Independent reproduction: 19 checks passed in 2.65 s, including actual generated
RL traces, native boundaries, independent scalar oracle, grazing/shear,
pre-load function qualification, sticky invalidation and GC-finalizer lifetime.
The declared RL dependency was installed; no policy training ran. Full/light
seals and proof epoch are available with the compiled backend.

Source runtime probe passed all five variants in about 2 s: real native execution,
explicit reference, absent auto fallback, absent explicit-C++ failure, and present
corrupt-manifest integrity failure. Receipt:
`outputs/cpp-acceleration-v1/source-runtime-v1.json`. Wheel/frozen qualification
must be rerun for the final binary; this source receipt does not qualify them.

Independent design review found and fixed: missing native identity in replay
seals, sticky prefix invalidation on integrity drift, pre-load entrypoint
qualification, preprocessing before native input caps, and borrowed Python
object lifetime across allocations. State-copy traversal now retains owned
snapshots under the GIL. Concrete runtime reproduction remains pending.

## Commands and results

```sh
git fetch origin
git status --short --branch
git log -3 --oneline origin/shaf/decision-system-v1
uname -m
sysctl -n hw.memsize
xcrun clang++ --version
```

All succeeded; target and host identities are recorded above. No acceptance test
has passed yet. Full verification and benchmark commands will be recorded here
when their tracked entry points exist.

## Unresolved / next action

Freeze generated workloads and exact comparison fields; profile the real fresh
production planner, then select the smallest measured native seam. Baseline owner
currently holds the heavyweight slot. Do not expand kernels based on speculation.
