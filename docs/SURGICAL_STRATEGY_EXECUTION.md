# Surgical strategy foundation — current execution record

Updated October 31, 2026. Goal: one source-bound, reproducible system for
comparing executable patient-specific strategies. The pasted October 31 steering
supersedes the older training-first priorities. No new training or acquisition
was started. Work began at `72c81843` on `shaf/data-foundation-rebuild`; unrelated
data and experimental retractor changes in the primary checkout are preserved.

## Completed: explicit shared physical clock v1

The current-code reproducer confirmed the audit's CUSA defect: at a fixed
generated contact and an assumed rate of 0.01 mm³/s, a latched left CUSA received
0 versus 0.002 mm³ capacity during 0.2 s of left versus right inspection.
The new mode supplies 0.002 mm³ in either case and gives identical complete
physical states. Two prechange legacy state hashes remain exact regression gates.

`surgical_clock.py` schedules both instruments in left/right order, validates
their contacts before effects, and counts each tool's effect once per interval.
The engine advances elapsed time, deformation, drilling cooling, bipolar heat,
vascular evolution and fluid fields once. Contact/exposure accounting includes
the persistent inserted tools rather than only the inspection owner.

Enable through **Rehearse → Environment and source setup → Enable shared
physical clock**, with both instruments withdrawn/inactive and no holds.
The button records `configure` with `clockRevision: 1`; the version is hashed in
physical state and displayed as `view.physicalClock`. It does not reset anatomy,
pending work, accumulated burdens, mechanics selection or elapsed time. Like
historical mechanics configuration, this is a zero-time setup operation; its
positive action duration is a required envelope field, not a dwell.

No old recording opts in automatically. Replaying or seeking before that action
uses historical behavior, including the earlier thermal-scene exception. Native
save/reopen and verified/observed replay retain the boundary. Scene identity
alone does not identify the clock: compare source plus state/action identity.
No clock downgrade operation is exposed.

Activation/deactivation occurs before the first accepted interval. `apply`,
`hold`, `grasp` and explicit mapping inspection have transient activation;
latched switches continue through other timed actions. Release, deactivation
and withdrawal turn off the selected switch before their interval. Ordinary
motion retains the existing spatial subdivision and maximum 0.1 s temporal
step. Complete microsteps commit atomically; a failed microstep rolls back both
hands and every field, leaving only the accepted prefix. Cancellation discards
the whole unpublished action candidate.

This is deterministic operator splitting for sequential motion with a persistent
stationary other hand. It does not implement continuously simultaneous paths.
Matched stationary time grids are tested for exact state equality. Uneven CUSA
partitions use absolute 1e-14 tolerances for floating quantities, retaining exact
binary capacity thresholds; no tolerance authorizes whole-cell clearance.
Nonlinear coupled fields retain their existing numerical discretization and
qualification limits. Experimental planar-maneuver and surface-retractor clocks
are not admitted to this new mode.

## Executed checks and evidence

- 22 generated scheduling tests pass: legacy hashes, activation, two active
  CUSAs, fixed and uneven partitions, one-shot/latched equivalence, irrigation/
  suction accounting, retractor relaxation, thermal/vascular evolution,
  rollback, cancellation, branching, verified caches and observed frames.
- The first focused regression run passes 281 tests. The broader integration
  run passes 337 with one optional compiled-prefix test skipped. These suites
  overlap; counts are not additive. The final clock run adds the later controls.
- Desktop build and all 1,212 existing desktop tests pass (322 main, 634 renderer,
  256 viewer). Build retains its existing large-bundle warning.
- Actual hidden Electron controls enable the clock and mechanics, activate a
  CUSA, and fork own/opposite 0.2 s inspection from the same boundary. Each
  supplies 0.4 mm³ and reaches the exact same state. Both branches save and
  cold-replay independently; a fresh hidden Electron process verifies the saved
  endpoint. The source archive is unchanged and no window becomes visible.
- Ignored evidence: `testcases/shared-clock-v1/`. The first desktop attempt is
  retained: its comparison also wrote previously absent left-tool parameter
  defaults. The corrected contrast binds both parameter records before the
  timed fork. No simulator tolerance was weakened to obtain a matching state.

Reproduce from the repository root:

```sh
.venv/bin/python -m pytest tests/test_surgical_clock.py -q
.venv/bin/python scripts/prepare_clock_verification.py
cd desktop
pnpm build
pnpm test
CLOCK_OUTPUT="$PWD/../testcases/shared-clock-v1/desktop-fresh" node_modules/.bin/electron electron/verify-shared-clock.cjs
```

Choose a fresh output directory to preserve prior evidence. The preparer reuses
its identical native generated control on subsequent runs. For a cold desktop
reopen, run the same verifier with `CLOCK_REOPEN_ONLY=1`, `CLOCK_CASE` set to the
saved workspace, `CLOCK_EXPECTED_STATE` from its report, and a fresh `CLOCK_OUTPUT`.

The scheduling defect is repaired in an explicit opt-in mode. This is software
and numerical scheduling evidence, not empirical mechanics accuracy or clinical
validation. Legacy episodes intentionally retain the historical defect.

## Completed: bounded reusable patient/task admission

`surgical_task_admission.py` binds the actual native source snapshot, original
image/annotation bytes, native arrays, registered/resampled masks, ownership,
DEVELOPMENT declaration, source review/coverage, models, permitted actions,
annotation-assisted observation contract and exact planning baseline. Cold
admission reads the external native source; a self-resealed verification claim
does not establish proof. An eight-entry process-private cache rechecks every
member's inode, size, mtime and ctime before reusing byte/raster verification.

The workspace owns the contract; case, scene, source arrays and historical branch
identities stay unchanged. The first task requires coherent preoperative source
imaging, a released/undeformed executed baseline, locally removed tissue at a
clear declared entry, retained source tumor, geometry revision 2 and shared
clock 1. Later observations require their own compatible task. Edited entry and
target spheres must stay inside the admission; allowed CUSA programs are checked
as a complete access/withdrawal sequence. The task supports inactive access,
not another removal operation or clinical qualification.

V2 interactive reports validate this admission on generation, load, preview and
installation. Historical V1 reports retain their original source gate and
whole-goal screening. V2 searches monotone swept prefixes to locate first target
contact before rejecting an unreachable deeper shaft pose; protection, shaft,
other-tool and non-target first failures remain explicit. Full public access,
reverse withdrawal and an independent geometry implementation still check every
accepted route. The desktop shows source modalities, estimates versus supplied
annotations, review/coverage, missing evidence and effective model versions.

The same generic preparer and actual hidden desktop interface were run on:

| Acquired source | Outcome at its preserved authored baseline |
| --- | --- |
| UCSF-PDGM-0007 DEVELOPMENT | Three inactive approaches; all three pass independent continuous full-tool, other-hand and reverse-withdrawal checks. Actual preview, installed branch, return to admitted checkpoint, save and fresh Electron reopen pass. |
| UCSF-PDGM-0005 DEVELOPMENT | Admission passes; its historical 0.6-mm corridor does not admit the current 1.5-mm catalog tool. Zero routes are retained; the UI offers no executable preview. Save and cold native replay pass. |
| Exact sub-13 standard-v2 missing-target control | Common interface refuses a missing tumor target and exposes no Generate action. This exact bundle is used; the later native collection has different annotations. |

Both DEVELOPMENT cases explicitly refuse all routes when functional evidence is
required. Brain estimates/registration and supplied tumor annotations remain
unreviewed; skull, vessels, tractography and functional evidence are missing.
The UCSF-0007 baseline already removed 4,940 mm³ non-target and 560 mm³ target
before this task; no new exposure is created or counted as success here.
The 0005 no-route result is preserved, not repaired by changing anatomy or
shrinking the qualified instrument. These are two development integrations,
not held-out generalization or physical/clinical validation.

Checks: the final proposed-commit backend run passes 116 tests; 23 focused admission checks
cover source exchange, cold original-byte rechecks, native ownership, task bounds,
observation scope, stage compatibility, first-contact failure ordering and native
persistence. Desktop build and all 1,212 desktop tests pass. Counts overlap.
Current local evidence and exact run commands are indexed in
`testcases/task-admission-v1/README.md`; acquired data remain ignored. No download,
optimizer update or reserved evaluation was used.

Generic reproduction (from the repository root; choose a fresh output):

```sh
.venv/bin/python -m pytest tests/test_surgical_task_admission.py -q
.venv/bin/python scripts/prepare_admitted_planning.py \
  --source testcases/cases/UCSF-PDGM-0007-operative-demo.ressectionlab \
  --branch original-exposure \
  --native-collection testcases/outputs/integrated-goal-20261026/fresh-patients/UCSF-PDGM-0007/native \
  --output testcases/task-admission-fresh
```

## Completed: public-input boundary for admitted inactive approaches

New V3 reports use a typed immutable `PlanningObservation`. Exact allowlists
project annotated material ownership, target roles, affine, declared protection
cells, both tool states, supported physical ledgers, task/evidence roles and the
remaining native action budget. A fresh public scene/engine has no original
source identity, raw provenance, histories, evaluator arrays, callbacks or
inherited caches. Numeric passive burdens are retained; unsupported profiles,
contact/topology history or pending deformation/vascular state explicitly refuse
this narrow track. Constructor checks prevent bypass by hand-built packets.

The same public model supplies proposals, complete-tool screening, native suffix
previews, deterministic candidate order and COMPARE/STOP. Its complete plan is
sealed before attaching native source/audit bindings. Installation executes the
native engine separately and requires exact predicted physical-state hashes at
access and withdrawal; a disagreement refuses installation without changing the
public decision. V1/V2 records retain their historical behavior and are never
promoted to a V3 information-boundary claim. This migrates the admitted inactive
workflow only; other historical planners/learned components remain outside this
new track. Annotation-assisted planning is not image-only planning.

Generated controls cover private reference/identity/history/provenance swaps,
metadata canonicalization, direct packet construction, public anatomy/physics/
protection changes, passive burden preservation, remaining action budgets,
preproposal time-budget STOP, resealed refusal attacks and post-seal evaluator
mutation. The evaluator diagnostic counts reference membership of proposed
contact cells; it is neither clinical harm nor independent tool verification.

Both acquired development baselines were rerun through the actual hidden desktop:
0007 generates three candidates, previews and installs a native branch, returns
to its exact baseline, saves and reopens in a fresh process. 0005 preserves its
no-route STOP. Both refuse missing functional evidence, and the exact missing-
target control still exposes no Generate action. All three 0007 programs match
native physical-state predictions, cold-replay exactly and pass the separate
continuous complete-tool/other-hand/withdrawal geometry checker. Screenshots
were inspected; no automation window became visible.

Representative local backend timings (single cold/warm pair, not an optimization
claim): generation including public suffix previews 2.28/2.28 s for 0007 and
1.36/1.29 s for 0005; functional-evidence STOP 0.15 s. 0007 native validation/
materialization is 3.05 s cold, 0.72–0.81 s thereafter; fresh suffix replays are
1.09–1.30 s. Independent geometry takes 4.91 s. Sources and original baselines
are unchanged. Full evidence, fingerprints, scripts and run logs are indexed in
`testcases/planning-observation-v1/README.md`.

Reproduce the source-bound checks with existing prepared inputs:

```sh
.venv/bin/python -m pytest tests/test_surgical_planning_observation.py tests/test_surgical_task_admission.py tests/test_surgical_interactive_planning.py -q
.venv/bin/python scripts/check_planning_observation.py \
  testcases/task-admission-v1/acquired-0007-v2/admitted-planning.ressectionlab \
  --output testcases/planning-observation-fresh
```

Choose a fresh output; substitute the existing 0005-v1 input for the no-route
control. No new data, learning, physical calibration or reserved evaluation was
used. Final proposed-commit verification passes 209 backend tests, including 39
observation-boundary controls. All 1,212 desktop tests and the build pass
(existing bundle-size warning). The two saved workspaces retain 19 exact cold-
replayed branches. Exact hidden-native commands are in the ignored inventory.

## Remaining milestones and next dependency

Next: extend this boundary to a new, explicitly versioned executable two-tool
manipulation/reassessment task, then migrate the remaining planner interfaces.
The current admission intentionally permits no active removal or information
acquisition. Source-linked functional consequences need compatible evidence;
end-to-end optimization and frozen uncertainty comparison remain open. Older
lattice resolution dependence and continuum empirical/domain gates remain
qualification holds. No learned advantage, neurological probability or
clinical-readiness claim is made.
