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
  --max-wall-seconds 30
```

The default seeds are 17, 29 and 43. At most three distinct seeds are accepted.
They repeat one generated geometry; they do not provide patient diversity or
held-out evaluation. Output must use a new directory below this checkout's
ignored `testcases/outputs/`; existing receipts are never overwritten. The
evaluation deadline is capped at 60 seconds and uses POSIX `setitimer` on
macOS/Linux, in addition to checks inside skill primitives. Interpreter/import
startup and the final small report write are outside the evaluation timer.

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

## Current measured receipt

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

`execute_requests` wall time includes each `SkillExecutor.execute` call's fresh
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
  tests/test_surgical_cusa_v3.py \
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
