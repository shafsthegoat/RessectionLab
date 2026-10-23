# Surgical RL readiness: evaluation only

This entrypoint checks interfaces and bounded scripted execution. It performs
**no training, no optimizer updates, no policy checkpoint loading and no policy action
selection**. Candidate preparation computes observation-derived features and
options; the script supplies every executed skill and endpoint explicitly.

Run from the existing primary checkout with the prepared Python 3.12 virtual
environment and Gymnasium dependency:

```sh
.venv/bin/python scripts/check_surgical_rl_readiness.py \
  --output testcases/outputs/rl-readiness-20261022/readiness-local \
  --max-wall-seconds 30 --skill-options
```

The default seeds are 17, 29 and 43. At most three distinct seeds are accepted.
They repeat one generated geometry; they do not provide patient diversity or
held-out evaluation. Output must use a new directory below this checkout's
ignored `testcases/outputs/`; existing receipts are never overwritten. The
evaluation deadline is capped at 60 seconds and uses POSIX `setitimer` on
macOS/Linux, in addition to checks inside skill primitives. Interpreter/import
startup and the final small report write are outside the evaluation timer.

`--skill-options` builds the complete declared nominal skill menu from a fresh
published observation before **every** prescribed decision. The script supplies
the selected ID in advance; no actor, ranking or search runs. It reports
observation, option preparation, execution and whole-decision times separately.
The full script and resumed tail use the same 15 declarations and stable IDs.
Episode wall time also includes executor setup, bookkeeping and in-memory
snapshot capture; reset and independent replay/resume costs remain separate.
Omitting this flag retains the historical nine-choice raw-action benchmark,
which is prepared separately and is not a complete surgical skill interface.

An optional prepared patient benchmark adds source-preserving load, reset and
observation timing. It executes **zero patient physics actions** and no patient
skills, policy decisions or arbitrary approach strokes:

```sh
.venv/bin/python scripts/check_surgical_rl_readiness.py \
  --output testcases/outputs/rl-readiness-20261022/readiness-local-patient \
  --max-wall-seconds 30 \
  --patient testcases/environments/meningioma-sub-06/patient.ressectionlab \
  --protection testcases/environments/meningioma-sub-06/rl-protection.json
```

The optional check hashes the supplied workspace and matching protection
JSON/NPZ before and after, and verifies unchanged in-memory source labels.
Referenced original imaging remains in place. It does not write a patient save,
download data, infer anatomy or replay the saved patient rehearsal.

## What the generated check establishes

The explicit surrogate has a 24 × 24 × 20 one-millimeter grid and 27 mm³ of tumor.
Its rescue mode declares a pre-existing modeled bleeding source at a specified
source cell, with 0.1 mm³/s inflow and an illustrative coagulation threshold.
This is a generated engineering source, not acquired vascular anatomy or a
clinical injury model. Critical coverage is explicitly relaxed for this fixture;
the script does not fabricate a brainstem annotation.

The generated scene explicitly opts into `properties.cusaV3` using
`surgical_cusa_v3.default_profile()`: fixed 1.5 mm tip, 0.5 mm shaft and an
**assumed, uncalibrated 2 mm³/s** contact capacity. Whole source cells remain
until their capacity threshold is reached. The optional patient benchmark keeps
the original scene/profile unchanged; the measured sub-06 source has no CUSA v3.

The fixed sequence is inspect → checked approach with bipolar → coagulate →
withdraw → central checked CUSA approach → three repeated resection requests →
withdraw → checked approach with Frazier suction → aspirate → STOP. Bipolar uses
its declared 1 mm radius; CUSA and suction use their declared 1.5 mm radii. The
central approach is at (9, 9, 5.9) mm, followed by the same explicit (9, 9, 11) mm
endpoint for 4.9, 5 and 5 seconds. This fixed schedule removes 9 → 19 → 27 mm³
from the unchanged target. It is not an adaptive retry/search using hidden
material masks, previews or pending-work state. The declared fixture horizon is
64 attempts, 60 simulated seconds and at most 5 seconds per primitive.

`TISSUE_ASPIRATION_IN_PROGRESS` remains a genuine partial outcome in every
receipt and attempt count; it is not relabeled accepted or counted as a
collision. Other partial/rejected outcomes fail this successful-script check.
An earlier off-center x8 request actually hit the retained-body corridor guard;
that failure is preserved under `cli-cusa-v3-seed17`. The guard was not weakened
or its collision relabeled to make the current central fixture pass.

Each seed verifies:

- Identical same-seed reset observations and valid observation-space membership.
- Separate reset, repeated observation, pure candidate-preparation, individual
  skill and complete script timings.
- Explicit rescue declaration; the same injured entry is refused as prevention.
- Target attainment, inactive/withdrawn instruments, released controls, zero
  active bleeding and the configured field-cleanup thresholds.
- Source-label preservation, unique removed source IDs and fluid/debris balance.
  This sequence generates blood but no drilling debris; its zero-debris balance
  does not qualify the drilling model.
- Nonzero pending CUSA capacity captured inside the first genuine work partial.
- Exact native engine JSON prefix restore and sealed-tail execution identity.
- Exact RL JSON prefix restore followed by actual remaining-skill execution,
  matching the uninterrupted final snapshot, exact pending fractions, rewards,
  ledgers, RNG and flags.
- CUSA input = aspirated solid + pending + unused + discarded capacity and
  consistency with the declared activation time and rate.

Separate control probes make one genuine collision-limited approach and one
genuine excessive-duration rejection. An injected callback interrupts a movement
inside its primitive; state identity, committed action count and RL attempt count
must remain unchanged. These probes are counted separately from successful
scripted episodes. No fabricated rejected/partial transitions are inserted.
Separate tip probes verify catalog-cap rejection, refusal of both growth and
shrinkage while inserted, same-radius respecification, and selection of a smaller
compatible tip while withdrawn. These make nine actual raw attempts: six
accepted controls/variant selections and three rejected geometry changes.
Those variant probes explicitly use a separate legacy geometry-only scene and
perform no tissue removal. The CUSA v3 profile fixes its tip at 1.5 mm and does
not admit the smaller CUSA variants used by that legacy control.

`report.json` records `no_training: true`, `optimizer_updates: 0` and
`policy_executions: 0`. Per-seed outputs include native prefix/final and RL
prefix/final JSON snapshots. Snapshots bind to the generated source reconstructed by the
script; they are not patient workspaces. Audits and explicit snapshot writes
happen outside the skill action loop. A deadline or other evaluation failure returns
nonzero and writes an incomplete/failed report with unmeasured gates still open.

## Earlier causal CUSA receipt

Earlier October 22 receipts, including `cli-final-three-seeds-patient`, used
enlarged 3 mm CUSA/bipolar/suction radii. They remain preserved as historical
engineering outputs and are superseded for the current catalog-tip contract.
The later `cli-fixed-tip-final` receipt used legal tips with the legacy instant
removal law. It is also preserved, and is superseded for the current causal
contact/time-capacity fixture.

The October 22, 2026 run at
`testcases/outputs/rl-readiness-20261022/cli-cusa-v3-final-three-seeds/report.json`
completed three seeds plus the optional intact sub-06 reset benchmark in
**9.662 seconds** of evaluation wall time, within an explicit 60-second cap, on
the authoring Mac. These are local
observations, not portable performance promises.

| Measure | Recorded result |
| --- | ---: |
| Generated reset, one per seed | 1.63–2.19 ms |
| Generated observation, per-seed medians of 10 reads | 1.14–1.16 ms |
| Candidate preparation, per-seed medians of 10 calls | 0.371–0.390 ms |
| Complete generated skill script | 0.986–1.023 s |
| Fixed scripted skill requests per second, scope below | 11.7–12.2 |
| Native prefix restore + sealed tail, 15 primitives per seed | 0.912–0.938 s |
| RL prefix restore, 8 attempted primitives per seed | 0.415–0.416 s |
| Remaining six skills after restore | 0.552–0.556 s |
| Optional sub-06 load and construct | 0.569 s |
| Optional sub-06 reset | 68.2 ms |
| Optional sub-06 observation, median of 10 reads | 1.367 ms |
| Optional sub-06 physics actions / skill decisions | 0 / 0 |

For this earlier receipt, `execute_requests` wall time includes each `SkillExecutor.execute` call's fresh
published observation, `propose_skill`, and actual primitive execution. The
clock starts after executor construction and includes the in-memory prefix
snapshot capture; reset/construction and independent replay timings are separate.
The separate `candidates(observation)` option/feature benchmark is **not run per
scripted decision**, and there is no actor scoring or policy selection. The
11.7–12.2 figure describes only these 12 fixed scripted skill requests on this
coupon. It does not qualify a sustained matched online candidate-policy loop
at ≥10 Hz; that gate remains open.

Per seed, the uninterrupted script made **12 scripted skill decisions and 15
primitive attempts: 13 accepted, two work-progress partials, zero rejected**.
Both partials retain `TISSUE_ASPIRATION_IN_PROGRESS`; engine collision count is
zero. The resumed tail made six skill decisions and seven primitives (six
accepted, one work-progress partial). Prefix/native replay counts are separate validation
work. The control probes, run once overall, recorded one partial primitive, one
rejected attempt and one interrupted primitive without a committed outcome.

Each generated endpoint removed 27 mm³ target, generated and aspirated
0.0490244 mm³ modeled blood, retained zero fluid/debris, and ended after
16.3 simulated seconds with successful cleanup. The first partial snapshot
retained 0.8 mm³ exact pending capacity. At the final state, 29.8 mm³ input
capacity = 27 mm³ aspirated solid + 2.8 mm³ unused capacity, with zero pending
or discarded work, over 14.9 activation seconds. Native and RL endpoints matched
`sha256:37008341e44441e3c1180398aa33db2e7f870c85818235db6b4e84a1830c4d52`.

**Patient skill throughput, sustained end-to-end ≥10 Hz, full-patient task
feasibility, held-out generalization, reviewed anatomy/physics calibration and
clinical validation remain open.** Fast inspection or observation does not close
these gates. A successful generated episode does not qualify patient resection.
The generated contact/time-budget and nonzero-pending JSON resume checks pass
for the explicitly enabled CUSA v3 surrogate. **CUSA mechanics calibration and
patient qualification remain open.** The assumed capacity, material response,
generic tip geometry and omitted device physics are uncalibrated. Scenes without
`cusaV3` retain the original instant compatible-cell deletion law. Neither the
generated result nor the unchanged patient reset benchmark makes full mechanics
ready for RL or upgrades the original patient profile.

## Contracts and available skills

`SurgicalRLEnv` uses `resectionlab.surgical-rl.v2`. Relative to v1, observations
explicitly add `local_spacing_mm`, normalized remaining step/time/invalid-action
`budgets`, and physical `cleanup` ledgers. Candidate geometry consumes the
published sampling spacing. Global task/reward ledgers are declared modeled
observations; `info`, engine/world access, snapshots and proposal audit details
are privileged diagnostics, not policy inputs.

Success requires the target fraction plus inactive/withdrawn instruments,
released grips/holds, no active bleeding, and fluid/debris within explicitly
configured limits. Default field limits are zero and are engineering choices,
not clinical thresholds. Time and attempt exhaustion remain truncations.
Configurable attempt horizons now reach the shared engine cap of 8,192;
the ordinary default remains 128 attempts and 60 simulated seconds.

V2 checks each requested working radius against its instrument's declared
catalog radius. An inserted, active or holding instrument cannot grow or shrink;
respecifying its existing radius is allowed. A smaller compatible variant may be
selected only while withdrawn, inactive and released, and must still pass shared
shaft/material guards. The optional CUSA v3 profile is stricter and requires
its fixed 1.5 mm tip/0.5 mm shaft. The action contract and radius caps are bound into the
environment identity and explicitly stored/checked in RL snapshots. Earlier
preliminary v2 snapshots without this action contract are incompatible too.
This stricter wrapper leaves the original desktop/native engine and preserved
legacy episodes separate; it does not silently requalify their geometry.

The versioned `surgical_skills.py` wrapper provides inspect, checked approach,
drill, resect, aspirate, irrigate, coagulate, release, withdraw and STOP. Proposals
accept only the published observation plus explicit request parameters. They
report nominal support and unknown feasibility; the public engine decides exact
collisions and protected-material outcomes. STOP releases/withdraws controls and
ends the executor. It cannot seal wounds or clear remaining fluid by assertion.
Cancellation preserves earlier committed primitives while rolling back the
currently interrupted primitive.

`surgical_skill_options.build_skill_options` exposes these parameterized skills
as an ordered, caller-declared option set. It accepts only the nine published v2
observation arrays, explicit JSON requests, an explicit `SkillTask` and a maximum
primitive duration. It copies the observation and returns detached JSON with
observation/request-set identities. Requests contain unique IDs and explicit
routes/parameters; the adapter does not invent endpoints, inspect engine state,
rank options, execute actions or produce a feasibility mask. Valid declarations
that fail a nominal skill check remain visible as rejected rows with a reason.
Every proposal retains `unknown_until_public_execution` feasibility.

The declaration must include plain release/withdraw controls for both explicit
slots and a route/parameter/instrument-free STOP. STOP can emit zero to six
control primitives and cannot assert field cleanup or success. Limits are 32
options, 64 KiB of compact ASCII-escaped request JSON, six primitives per option,
192 total primitives and 256 KiB of response JSON. Unknown fields, malformed
declarations and missing controls fail before proposals. These limits do not
derive physical action duration from normalized remaining episode budgets.
The old nine-option policy/checkpoint interface is unchanged and remains
unpromoted; this adapter introduces no learner.

Task presets provide explicit horizons: access 128 attempts/300 seconds,
resection 4,096/7,200, and field management 256/300. Callers may supply smaller
explicit fixture budgets, as this CLI does. `prevention` refuses observed entry
injury or active bleeding; `preinjured_rescue` preserves that declaration in
receipts. Task names select horizons and mode; the underlying environment's
resection-and-cleanup success criterion remains authoritative.

V1 RL snapshots must retain their original implementation and semantics; they
are deliberately rejected by v2 rather than silently migrated. The original
source/hash-bound frozen learner remains incompatible with changed observation
and candidate code. **`scripts/train_surgical_pilot.py` is not promoted to this
contract and is not a readiness command.** No training launch is authorized or
provided here.

Targeted evaluation-only regression command, using another fresh local output:

```sh
.venv/bin/python -m pytest \
  tests/test_surgical_rl_contract.py tests/test_surgical_rl_guards.py \
  tests/test_surgical_rl_rewards.py tests/test_surgical_rl_protection.py \
  tests/test_surgical_skills.py tests/test_surgical_anatomy_comparison.py \
  tests/test_surgical_cusa_v3.py tests/test_surgical_skill_options.py \
  --basetemp=testcases/outputs/rl-readiness-20261022/pytest-local -q
```

The earlier catalog-tip regression run passed 93 tests in 7.03 seconds. The new
causal CUSA skill partial/continuation check separately passed in 0.61 seconds.
The separate generated
drill skill test produces 9 mm³ actual drilling debris and checks its conservation;
this is engineering-fixture evidence, not patient drilling calibration.

After the causal CUSA, field-streaming and cleanup changes, the complete surgical
and full-case Python regression selection passed **593 tests in 111.77 seconds**
with two existing Gymnasium space advisories. Its isolated temporary/cache files
are under `testcases/outputs/rl-readiness-20261022/pytest-final-surgical`; the log is
`final-surgical-tests.log`. **That broad selection included an existing pilot
test that made an in-memory Adam update on generated transitions.** An earlier
577-test broad run included it too: two fixture optimizer updates occurred in
total, contrary to this task's no-RL instruction. They produced no patient fit or
learned checkpoint. The dedicated readiness command's zero-update receipts
remain accurate within their stated scope.

Pilot policy/optimizer tests are now marked `rl_pilot` and skipped by default.
They require an explicit `--run-rl-pilot-tests` option, which must not be used for
this task. Ordinary broad pytest selections now retain readiness-only behavior.
Regression coverage does not replace full patient replay or qualify calibration.

Imported patient archives also have a separate read-only prefix replay mode.
It reconstructs requested saved actions from source without promoting imported
checkpoint state, creating an editable engine session, or granting RL resume/
export authority. A prefix matching the old frames is playback evidence only.
The production GUI's three-action qualification and bounded retention contract
are documented in [rehearsal-performance.md](rehearsal-performance.md).

With the opt-in guard active, the earlier default surgical/full-case Python
qualification passed **586 tests, skipped 7 pilot tests, in 107.04 seconds** with
the same two existing Gymnasium advisories. This guarded run performed no pilot
updates; the two earlier fixture updates disclosed above remain part of this
task's history.

After the exact contact-grid, cleared-cell lookup and surface-normal changes at
`7218b080`, the broader surgical/full-case and independent geometry selection
passed **715 tests, skipped 7 pilot tests, in 108.62 seconds**. The same two
Gymnasium advisories remain. A fresh dedicated readiness run repeated all three
fixed fixture seeds and the preserved patient load/reset-only check in
**9.761 seconds**, with zero patient actions, policy executions or optimizer
updates. Its receipt is `cli-geometry-optimized-three-seeds/report.json` below
the same ignored output root. It does not qualify patient replay, an online
candidate-selection policy, or the remaining performance gates.

After adding isolated read-only prefix replay, the same broader selection
passed **733 tests, with 7 pilot tests skipped, in 115.10 seconds**. It adds
source/implementation binding, cancellation, retention and bridge-authority
regressions. The pilot opt-in remained disabled.

The later prefix display-stream change passed **759 tests, with 7 pilot tests
skipped, in 118.48 seconds**, with the same two Gymnasium advisories. New
checks cover generation/field acknowledgments, exact restored views, one hash
per restored state and strict separation from canonical command arguments.
It adds no RL authority or training; imported prefix frames remain read only.

The subsequent bounded capture encoder and exact retained-frame accounting
qualification passed **791 tests, with 7 pilot tests skipped, in 122.93 seconds**,
with the same two Gymnasium advisories. The pilot opt-in remained disabled.
These changes improve read-only archive preparation without changing numerical
laws, observations, rewards or RL authority. The two earlier fixture updates
disclosed above remain part of this task's history; this guarded run performed
no pilot training.

The subsequent stock-engine drill query qualification passed **816 tests, with
7 pilot tests skipped, in 118.18 seconds**. It includes stateful query-override
fallbacks as well as the surgical/full-case and independent geometry selection;
the same two Gymnasium advisories remain. No pilot opt-in was enabled. The
optimization is limited to a single unchanged stock-engine contact search;
read-only prefix engine subclasses retain the original query path.

The subsequent exact-list integer-validation qualification passed **832 tests,
with seven pilot tests skipped, in 119.96 seconds**. The same two Gymnasium
advisories remain. It checks strict type identity, in-place cache invalidation,
subclass/error fallbacks, independent rollback ownership and bound C helpers.
The explicit pilot opt-in remained disabled; this qualification performed no
pilot training.

With that patch, the dedicated three-seed readiness command repeated the same
generated coupon and optional intact sub-06 load/reset-only check in **8.911
seconds**. Its receipt is
`testcases/outputs/rl-readiness-20261022/cli-integer-guard-three-seeds/report.json`.
All measured checks passed with open gates, with zero policy executions,
optimizer updates or patient actions. The three seeds remain one geometry;
this is evaluation-only scripted solvability/resume evidence, not an adaptive
policy, a patient task qualification or a matched throughput comparison.

The subsequent compressed-history/optional-native qualification passed **882
tests, with seven pilot tests skipped, in 149.08 seconds**, with the same two
Gymnasium advisories. The optional native checks used an explicitly supplied
local build in generated subprocess fixtures; the parent kept Python mode.
Public engine/RL history remains complete. Only the separate read-only prefix
shell uses compact metric ancestry, retaining complete canonical receipts in
bounded compressed storage. Selected native mode has its own strict loaded
binary/build/runtime identity and cannot reuse Python-mode cache authority.

The evaluation-only readiness command then passed with open gates in **13.739
seconds**, with that native build configured. The receipt is
`testcases/outputs/rl-readiness-20261022/cli-idle-native-three-seeds/report.json`.
Seeds 17/29/43 still repeat the same generated 27 mm³ coupon; the optional
patient check loads/resets only. Policy executions, optimizer updates and patient
actions were zero. This does not demonstrate learned behavior, genuine online
candidate throughput, another patient, or modern CUSA feasibility on sub-06.
The two earlier fixture updates remain disclosed above.

An independent matched diagnostic of four genuine original late CUSA steps
preserved complete output bytes and the original canonical endpoint while
reducing median step latency 11.67% with optional native idle cooling. It uses
explicit historical checkpoint ancestry and grants no planning, RL resume or
cache admission. The kernel's larger pure-helper gain is a separate component
result. Build/launch commands, strict platform limits and complete measurement
scope are in [rehearsal-performance.md](rehearsal-performance.md).

Separately, the desktop at `cc868257` completed all 73 original vascular rescue
actions and CPU-submitted all 1,149 fresh observations in order. A separate
zero-step audit matched every full historical canonical record and the final
endpoint, with no canonical/session/cache promotion. Full playback including
cold reconstruction measured 0.512×; same-action CPU intervals measured 1.317×.

The later compressed-history implementation, with optional native cooling
explicitly selected, also completed all 73 actions and 1,149 observations. All
complete historical canonical records and the final endpoint matched in the
separate zero-step audit, without canonical/session/cache promotion. Playback
measured **0.493× including cold reconstruction**, with **1.272×** for same-action
CPU intervals. It was slower than the earlier full GUI run, so it establishes
exact saved-replay compatibility, not a GUI speedup or RL performance gate.
Full-primary playback, operative readability, flowing effects, compositor
timing and genuine candidate-inclusive online decision throughput remain open.
This is legacy saved-replay evidence, not patient CUSA v3 qualification, editable
planning/RL resume authority, compositor timing or a training result. The full
primary GUI replay and engineering gates remain open; see
[rehearsal-performance.md](rehearsal-performance.md) for commands and limits.

The subsequent optional prefix JSON snapshot qualification passed **929 tests
in 161.32 seconds**, with the same two Gymnasium advisories. The pilot module
was excluded entirely. Exact bytes, strict bounds, callback/fallback behavior,
native identity drift and prefix integration were covered. External wall time
was 161.721 seconds (<240), sampled process-tree RSS 1,606,320,128 bytes (<3 GiB),
and all 259 bound source/test files stayed unchanged. This capture optimization
grants no learning, planning or canonical authority. No learning path was selected
in this qualification; the earlier two fixture updates remain disclosed above.
The separate v5 GUI run then qualified all 73 saved vascular actions and 1,149
exact ordered observations, with every complete historical canonical record and
endpoint matching in its zero-step audit. It observed 1,369 native JSON
admissions with zero fallbacks. Full playback measured **0.5214×** (388.511 s),
5.43% less wall time than v4; same-action intervals slowed to **1.1781×**.
The 1× full-playback gate remains open. This saved read-only replay grants no
planning or RL authority. Explicit build/launch commands, platform/scratch
limits and comparison caveats are in
[rehearsal-performance.md](rehearsal-performance.md).

The next optional canonical-hash extension passed **952 tests in 182.99
seconds**, with two existing Gymnasium advisories and the pilot module excluded
entirely. Full decoded-state hashing remains mandatory in the read-only prefix
path; custom properties/lookups/digest dependencies retain their original
fallback. The public engine state-ID property and numerical laws are unchanged.
External wall was 183.462 seconds (<240), sampled process-tree RSS
1,212,989,440 bytes (<3 GiB), and all 259 bound source/test files stayed unchanged.
Its single explicit schema-v2 build took 0.636 seconds. This qualification
selected no learning path and grants no planning/RL authority; the earlier two
fixture updates remain disclosed above. A new full GUI receipt is required for
each subsequent implementation.

The separate v6 GUI receipt at `3febf0da` passed all 15 checks, all 73 full
historical canonical records and 1,149 ordered fresh observations. The zero-step
audit matched the saved endpoint; fresh trajectories also matched v3/v4/v5.
Cold-inclusive playback measured **0.6358×** (318.613 seconds), 17.99% less wall
time than v5; same-action CPU intervals reached **1.3729×**. Both actual native
paths admitted with zero fallbacks: 1,369 insertion calls and 2,741 canonical
calls, including one full canonical hash per warm read. Source/build/runtime
bindings and originals stayed unchanged. No policy or optimizer ran in this
replay/audit. These uncontrolled host-run observations grant no planning/RL
authority; full-primary playback, cold-inclusive 1×, delivered-frame timing,
operative readability and genuine candidate-inclusive throughput remain open.

The later [prepared-observation replay mode](prepared-prefix-replay.md) is a
separate read-only application capability. It cannot seed RL reset/resume or
planning, and cache hits do not qualify fresh decision throughput. Fresh
preparation, prepared playback and their validations are reported separately.

The current candidate interface is another concrete readiness gap.
`surgical_observed_policy.candidates` exposes nine legacy left-slot raw options:
seven apply motions, inspect and a raw deactivate labeled STOP. It cannot
express the already successful bipolar → CUSA → suction skill sequence, tool
exchanges, withdrawal or complete cleanup. It remains a separate pilot interface
and is not promoted by the v2 environment or prepared replay. A task-complete
nominal skill-option interface must use only published observations and declared
request parameters; it must retain the shared collision/protection decisions.
Candidate-inclusive fresh skill throughput is still unmeasured. Neither the
whole readiness CLI timer nor cached playback establishes the 10 Hz gate.

The prepare/reopen application pair at `df779989` passed all 73 historical
canonical rows and 1,149 ordered observations. Fresh preparation took 286.184
seconds (0.708×); the filesystem-warm prepared-only reopen took **178.119 seconds
(1.137×)** with **zero physics attempts**, 73 cache loads, no publications or
misses, and unchanged cache/source/native bytes. This improves repeat rehearsal,
while fresh candidate-inclusive throughput needs its own measurement. The later
declared-skill check below addresses the menu and short generated timing.
No policy or optimizer ran in the pair;
the earlier two fixture updates remain disclosed above.

## Declared skill options and empty thermal intervals

The optional native build previously checked its live identity on every thermal
interval, including intervals with no temperatures or powers. A single no-learning
profile of the existing script recorded 944 such checks and 432 ms cumulative
guard cost within 2.308 seconds of profiled execution. These timings include
profiler overhead, overlap other costs, and are not throughput measurements.
The profile's copied launcher incorrectly labeled engine actions as zero;
`readiness-script-hotpath-profile-v1/metadata-correction.json` explicitly corrects
that label to 15 generated primitives and zero patient actions. Original receipts
remain preserved.

The Python thermal path now returns after its existing scalar calculations when
the state has exact string keys and both its temperature map and powers are exact
empty dictionaries. The legacy loop performs no work in this case. It reads no
dose/cooling values and changes no numerical state. Custom mappings/keys and
nonempty inputs retain their original paths. Every actual native call and explicit
identity read still checks drift. **An empty no-op no longer invokes the native
selection or detects its later drift itself**; the first actual use still rejects
drift before mutation. Invalid configured builds still fail at import.

The complete three-seed `--skill-options` receipt is
`testcases/outputs/rl-readiness-20261022/skill-options-cli-v2/evaluation/report.json`.
All 12 fresh decisions per seed construct all 15 declarations before executing
the fixed script ID. Each six-decision resumed tail keeps those same 15 options
and original IDs. Every public skill receipt, final native/RL snapshot and
conservation result matches `cli-idle-native-three-seeds` exactly.

| Seed | Complete script | Requests per episode second | Whole-decision p95 |
| --- | ---: | ---: | ---: |
| 17 | 0.955 s | 12.57 | 142 ms |
| 29 | 1.006 s | 11.93 | 152 ms |
| 43 | 1.102 s | 10.89 | 165 ms |

Per-decision candidate preparation medians were 1.20–1.29 ms; reset took
1.66–2.61 ms and observation-read medians were 1.12–1.15 ms. The full evaluation,
including separate replay/resume/control checks, took 9.181 seconds, and the
external job took 9.825 seconds with 90,570,752 bytes peak sampled process-tree
RSS, under 30 seconds/1 GiB. Monitored totals were 11 resets, 143 public engine
step entries (including validation/cancellation), 56 skill executions and 84
candidate calls. No policy, actor, optimizer or patient action ran. The first
attempt stopped at reset nine because its diagnostic cap omitted restore resets;
that failed receipt is preserved, and only the cap was corrected for v2.

These are 12-request scripts on one generated geometry with three seed values.
They measure genuine fresh execution with option preparation, but do not establish
sustained patient or actor throughput, uniform sub-100-ms decisions, or a matched
10× improvement. Earlier native script times were 1.358–1.428 seconds without
per-decision option construction; host conditions and timer scope differ, so this
is not an isolated speedup attribution. The broader 10 Hz and full-patient gates
remain open. New Python source conservatively invalidates earlier prepared-prefix
packages; the v7/v8 GUI timing remains qualified for its recorded `df779989`
implementation rather than this later source.

Final surgical/full-case/geometry regressions passed **1,081 tests in 181.42
seconds**, with two existing Gymnasium advisories. The entire pilot module was
excluded. The external job completed in 181.853 seconds with 1,085,046,784 bytes
peak sampled process-tree RSS, under 240 seconds/3 GiB. Explicit generated native
fixtures covered the unchanged native guards; no policy or optimizer was selected.
Evidence is `skill-readiness-python-full-v1.{json,log}` under the same ignored
output root. The earlier focused run passed 226 tests before the control-coverage
repair; the final complete run includes that repair and its new regression.

## Generated native workspace companion

The separate evaluation-only exporter creates an openable, explicitly labeled
generated `.ressectionlab` workspace. It uses the same unchanged 27 mm³ coupon,
assumed CUSA v3 profile, catalog tips and fixed 12-request script. It first builds
`CaseData` and adapts it with `from_case`, then executes the case-bound scene; it
does not transplant a standalone snapshot onto a different source identity.
Physical state, action/status sequence and reward/conservation measurements must
match a separate in-memory execution of the original coupon. Scene and state
identities deliberately differ because the portable workspace has its own
source binding.

Use a fresh ignored output directory:

```sh
.venv/bin/python scripts/export_surgical_readiness_demo.py \
  --output testcases/outputs/rl-readiness-20261022/scripted-desktop-companion-local \
  --max-wall-seconds 30
```

The exporter saves complete actions, protection contract, checkpoint and timeline
through the ordinary case artifact format, then uses a fresh `BridgeSession` and
an isolated verified-cache directory to load and canonically inspect the saved
branch. Public seek must reproduce the first partial CUSA boundary, including
nonzero exact pending work, and return to the exact final state. It checks the
workspace bytes remain unchanged. It neither launches the GUI nor changes the
default readiness command or its earlier receipts.

Measured receipt:
`testcases/outputs/rl-readiness-20261022/scripted-desktop-companion-isolated/report.json`.
The command passed in **3.276 seconds**, including the original script (0.940 s),
case-bound script (0.922 s), save (0.003 s), fresh public canonical inspect
(0.959 s), pending-boundary seek (0.353 s), and final seek (0.075 s).
The saved file is
`scripted-desktop-companion-isolated/GENERATED-readiness-cusa-v3-demo.ressectionlab`
relative to the same output root. Its case hash is
`sha256:cc94a2321a1fb006b51fb340dfda32a74cfc3c5872552fa66e640c7097c40316`
and case-bound endpoint is
`sha256:7ee8527cc7e3b56c5948a3e6271aafa9fecb802472e3604245d93b7876620fb3`.
The 15 primitives remain **13 accepted, 2 work partials, 0 rejected**; the partial
boundary is action 8 with **0.8 mm³ pending capacity**. Final removal is 27 mm³,
with zero present fluid/debris, active bleeding or pending controls. The receipt
records zero policy executions, optimizer updates and patient actions, scoped
only to this exporter. This proves generated native binding and public replay;
GUI display, patient qualification and clinical calibration remain open.

A fresh exporter run after those exact geometry/lookup changes passed in
**3.171 seconds** with the same physical/native-binding checks and zero patient
actions, policy executions or optimizer updates. Its receipt and generated
workspace are under `generated-companion-geometry-optimized/`; earlier receipts
remain preserved.

## Integer-predicate integration readiness check

After integrating the bounded exact-integer predicate into the existing optional
native kernel, the same three generated prescribed scripts passed with all
15 skill options prepared before each fresh and resumed decision. The new engine
guard skips the optional predicate below 32,768 removed cells; monitoring
confirmed **zero predicate API entries** on these small generated ledgers.

| Seed | Complete scripted episode | Requests per episode second | Whole-decision p95 |
| --- | ---: | ---: | ---: |
| 17 | 1.050 s | 11.43 | 179.2 ms |
| 29 | 0.946 s | 12.69 | 143.8 ms |
| 43 | 0.952 s | 12.61 | 141.0 ms |

These short generated runs count fresh observation, complete option preparation,
execution and episode bookkeeping; they are not sustained patient/actor
throughput or a 100 ms decision guarantee. Candidate medians were 1.14–1.23 ms.
All 12 saved prefix/final native and RL JSON files matched the previous
`skill-options-cli-v2` bytes. Complete fresh/tail skill receipts, option IDs,
conservation and endpoints also matched. The evaluation took 8.561 seconds;
external wall was 9.254 seconds with 93,224,960-byte sampled process-tree RSS,
below 30 seconds and 1 GiB.

Monitoring recorded 11 resets, 15 engine constructions, 143 public engine
entries, 56 skill executions and 84 option calls across scripted execution,
resume/replay and controls. It recorded zero patient actions, forbidden routes,
policy executions or optimizer updates. Evidence and saved-JSON comparison are
under `testcases/outputs/rl-readiness-20261022/native-integer-readiness-v1/`.
The earlier two unintended fixture updates remain disclosed above; this new
evaluation performed no learning. The complete surgical/full-case/shared-guard/
geometry selection then passed **1,152 tests, skipped two inapplicable alias
rows, in 187.51 seconds**, with two existing Gymnasium advisories. It excluded the
whole pilot module. External wall/RSS were 187.937 seconds/1,483,866,112 bytes,
below 240 seconds/3 GiB; source bindings remained unchanged. The receipt is
`native-integer-python-full-v1.{json,log}` under the same ignored output root.
The subsequent fresh vascular GUI qualification on `3d4d73d7` passed 17 gates,
73 original actions and 1,149 exact observed tuples, including same-renderer-clock
receipt-before-submission proof. Its full/warm rates were 0.707×/1.738×; these
are application measurements, not RL or sustained patient-decision throughput.
The earlier v10 clock-association qualification remains failed/inconclusive.
See [rehearsal-performance.md](rehearsal-performance.md) for evidence and open
performance, mechanics, full-primary, display and patient-readiness gates.

The subsequent recorded-field-transfer implementation remains display-only.
The current Python selection passed **1,173 tests with two intentional skips**
and the existing two Gymnasium advisories, excluding the whole pilot module;
external wall/RSS were 200.060 seconds / 1,612,644,352 bytes, below 240 seconds /
3 GiB. Ordinary headless execution creates no recorder. Generated comparisons
preserve complete physical states, metrics and receipts, including a tiny
positive-duration action whose floating-point clock does not advance.

Fresh vascular v13 on `282525c3` passed all 18 transport/authority/audit checks,
73 original actions and 1,149 exact observations; the original full canonical
rows, endpoint and available prior observed trajectories match. It records
bounded historical transfer indicators over exact endpoint anatomy. Full/warm
CPU rates are 0.682×/1.668×; small glyphs remain obscured in the actual patient
view. The failed compact-wire v12 attempt is preserved and unqualified. This
work ran no policy or optimizer and does not close the remaining patient,
performance, planning or visual prerequisites for RL. See
[recorded-field-transfers.md](recorded-field-transfers.md) and the performance
results for source bindings, controls and limits.

The next display/transport milestone on `8b4964b7` remains preparation only.
An optional pinned local cutaway improves one static operative close-up;
727 frontend checks/build and a 17-check prepared-only actual-app probe pass.
The conservative prefix JSON size proof passed a focused 252-test selection
and the overlapping final 65 helper/fallback tests, excluding the pilot module.
Fresh v14 passed 73 actions / 1,149 observations, complete canonical records,
v13 flow metadata and a separate no-step audit with unchanged bound files.
Its full/warm CPU rates are 0.690×/1.701×. A matched 12-read prepared diagnostic
reduced median handler time from 81.935 to 75.726 ms with complete content hashes
unchanged; it does not measure fresh decisions or compositor delivery.

Python seals invalidate v13 packages; the historical prepared launch used the
fresh v14 cache described in [prepared replay](prepared-prefix-replay.md).
Full-primary playback, responsive visual surgery/flow, patient decision
throughput, remaining mechanics/planning and the other readiness gates stay open.
No policy execution, optimizer update or RL pilot occurred in this continuation.
The earlier two unintended fixture updates remain disclosed above.

The subsequent field-token optimization on `44beb257` remains engineering
preparation. The focused prefix/cache/flow/native/budget selection passed
274 tests in 53.39 seconds, excluding the whole pilot module. Two prior test
attempts exposed constructor-probe isolation/signature issues and remain
preserved; the final run passed. A stateless paired serialization check on one
preserved 1.06 MB field reduced median token cost from 10.061 to 6.937 ms,
with every token exact, 61 actual native admissions and zero physics attempts.
It does not measure decisions or policy selection.

Fresh vascular v15 passed all 18 gates, 73 original public actions and 1,149
observations, with full canonical/endpoint checks and exact v14 complete
receipts, descriptors, flow blocks and compressed physical-state hashes/lengths.
All 557 bound files stayed unchanged. Full/warm CPU rates are 0.705×/1.722×;
fresh 1×, primary playback, responsive tool/field animation and the remaining
patient/mechanics/planning/readiness gates remain open. The historical prepared
launch used the source-matching v15 cache. No policy execution, optimizer update
or pilot ran in this continuation; the two earlier unintended fixture updates
remain disclosed above.

The subsequent opt-in display pipeline adds fixed fresh-producer and
prepared-only consumer roles. It is engineering preparation: the default app
keeps one worker, and patient GUI/throughput qualification for the new source is
pending. The final explicit prefix/cache/bridge/native/JSON selection passes
376 tests; all 284 Electron main-process checks pass. Whole pilot and training
test modules are excluded. Generated real-process checks exercise scripted
saved actions, exact frames/endpoints and source image retention with zero
consumer physics steps. These are neither RL decisions nor training examples.
See [prepared replay](prepared-prefix-replay.md) for bounds, failure handling,
cache-write accounting and the experimental flag. The new Python source
normally invalidates v15 packages. No policy execution, optimizer update or
pilot ran in this pipeline work; the earlier two fixture updates stay disclosed.
The full readiness goal remains active and incomplete.
