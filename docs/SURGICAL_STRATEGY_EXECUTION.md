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

## Completed: executable CUSA/Frazier strategy comparison

`post-exposure-two-tool-limited-debulking-v1` separately declares a finite action
suffix from the existing source-qualified inactive baseline. It does not extend
historical inactive-task permissions. The typed public baseline drives entry
proposals, full paired-tool screening, finite native-model execution, reassessment,
withdrawal and STOP. Both entry points obey the user's requested patch. The
catalog radii stay 1.5 mm; no new non-target work/removal or target-region escape
is accepted. Caps bound activation, total duration, actions and search. Unsupported
physical regimes still refuse. V1/V2/V3 approach behavior remains versioned.

**Rehearse → Compare two-tool strategies** compares two user-controlled CUSA
activation intervals and STOP. It previews actual inspection, access, suction
positioning, manipulation, reassessment and withdrawal checkpoints, then installs
complete checked branches. Target/entry editors use linked MRI coordinates.
The rule meets a declared research progress threshold, then minimizes modeled
duration. Required missing functional evidence, exhausted computation and failed
bounded geometry have distinct STOP reasons. Visibility and clinical outcomes
remain unassessed; STOP executes no new actions. Existing exposure costs and
residual source target remain attributed, not counted as new task success.

On the unchanged 0007 DEVELOPMENT baseline, 0.25 s produces zero whole-cell removal
and 0.5 mm³ pending CUSA work; 1 s removes 2 mm³ target with no new non-target
removal. The same inactive follow-up pose rejects after the shorter interval and
accepts after the longer one. With a 1-mm³ research progress requirement, changing
the short interval to 0.75 s removes 1 mm³ and reverses the modeled preference to
the shorter sequence. These are consequences of an assumed contact-capacity law
and whole source cells, not measured device rates or surgical recommendations.
Suction removes zero liquid in this dry modeled baseline; no hemostatic benefit
is claimed. Pending work, removal, both tool states and clock survive branching.

Actual hidden Electron controls preview these checkpoints, install two different
branches, restrict the shared entry, force missing-functional-evidence STOP,
save and reopen. 0005 retains its no-access STOP; the exact missing-target case
still offers no comparison action. All automation remains hidden. Native
checkpoints and public predictions agree at every committed suffix action;
manual edits fork protected study branches rather than altering checked evidence.

The separate `verify_two_tool_geometry.py` consumes exported programs, never the
planner. It observes actual native microsteps and uses `geometry.check_route` for
both complete tools, stationary occupancy and shafts. Controlled target working
contact is separate from shaft/non-target exclusions. Axial retreat from contact
requires a whole-cell forward-halfspace proof; it never shrinks tools or clears
extra tissue. Default programs pass 1,595/1,606 independent checks; modified
intervals pass 1,598/1,604. All endpoints cold replay exactly. Falsification controls
reject non-target working contact, target shaft collision, other-tool collision,
forward 'retreat' and clipped submillimeter-grid obstacles.

The native `.ressectionlab` archive is the reproducible plan/evidence export.
Use it for certification: JavaScript diagnostic JSON does not preserve Python's
numeric type identity for seals. Cold verification checks source bytes, frozen
reports, prefix predictions and installed programs. Private-reference changes and
self-resealed program, metric, nested-approach and decision changes are tested.

Measured IPC backend requests on 0007: generation/validation 12.78/12.55 s;
checkpoint preview 13.11–14.09 s; branch installation 13.30–13.38 s; missing-evidence
STOP 1.39 s. These exclude renderer admission and are not a speedup claim. Repeated
canonical execution was the main subsequent profiling bottleneck. Exact outputs,
failed prototypes and reproduction commands are indexed in the ignored
`testcases/two-tool-strategy-v1/README.md`. Final proposed-commit regression:
232 backend checks, all 1,214 desktop tests and build pass. Six frozen studies
and two installed endpoints verify with index simulator sources; all 14 original
source branches remain unchanged.

```sh
.venv/bin/python -m pytest tests/test_surgical_two_tool.py tests/test_two_tool_geometry_audit.py -q
.venv/bin/python scripts/verify_two_tool_geometry.py \
  testcases/two-tool-strategy-v1/desktop-0007-03/strategies-saved.ressectionlab \
  testcases/two-tool-strategy-v1/desktop-0007-03/strategies-saved.ressectionlab \
  --revision 1 --output testcases/two-tool-strategy-v1/independent-fresh.json
```

## Completed: exact reuse of full strategy validation

Profiling found that validation repeated two complete strategy recipes and three
approach previews; materialization repeated that validation before native execution.
Full-volume residual summaries were a smaller cost. A process-private LRU now
retains at most 32 successful complete-validation digests. Every hit still checks
source admission/member fingerprints, the full native checkpoint/history/actions,
report content/seals and freshly projected public inputs, including actual live
guard masks/configuration and the instrument catalog. Reports, observations and
engines are not retained. Native materialization still executes every action,
checks every predicted prefix and receipt, and preserves accumulated burdens.

Three alternating paired runs against `b3da4b10`, using identical source and
physics, give median backend preview-path latency **12.960 → 1.482 s (8.75×)**.
Repeated validation is 5.816 → 0.082 s; materialization of an already validated
report is 7.193 → 1.406 s. First validation (5.866/5.846 s) and generation
(6.323/6.317 s) are unchanged. These use warm source admission; “cold” denotes
an empty canonical-validation cache, not cold storage or source-byte verification.
All six runs have identical complete reports, endpoint snapshots, receipts and
six phase snapshots. One entry estimates 484 bytes; filling 32 digest entries
estimates 6,912 bytes, reaching 8,176 bytes after 10,000 replacements on this
Python as dictionary capacity changes. These exclude other process memory.

Actual hidden Electron preview requests take 1.84–2.16 s; accepted displays are
ready in 2.02–2.61 s including UI dispatch/polling and surface preparation.
Installation takes 2.07–2.09 s. Generation remains 12.75–12.91 s. Duration changes,
the two installed endpoints, missing-functional-evidence STOP and narrow-entry
STOP retain their behavior. All 15 native branch snapshots and both application
receipts equal the prior saved workspace. The narrow-entry wall-budget run
records 21 rejected proposals rather than 20, changing its diagnostic seals;
its controls, STOP decision and alternatives are identical. Full report equality
is claimed for the paired benchmark and other three desktop studies, not that
wall-budget diagnostic. The second acquired case still refuses complete access;
the exact missing-target negative exposes no comparison action. A fresh hidden
process reopens and verifies the positive native save. Source archives remain
unchanged and no verification window becomes visible.

All 241 selected backend controls, 1,214 desktop tests and build pass. New controls
cover warm native execution, equivalent clones, source-byte/state/history/catalog
invalidation, live guards with stale serialized contracts, eviction and resealed
forgeries. Independent bounded review found no cache-specific hole. Physics,
candidate rules, subdivision, replay and report formats are unchanged. Exact
commands, profiles, paired proofs and desktop evidence are indexed under the
ignored `testcases/two-tool-performance-v1/README.md`.

```sh
.venv/bin/python scripts/benchmark_two_tool_validation.py \
  --case testcases/task-admission-v1/acquired-0007-v2/admitted-planning.ressectionlab \
  --output testcases/two-tool-performance-v1/paired-fresh \
  --reference-commit b3da4b10 --isolate-index-physics
```

## Completed: source-linked anatomical encounter through strategy comparison

The optional V2 task binds a deliberately selected supplied categorical context
layer to its verified native collection, original annotation, array, revision,
registration, review status and exact source grid. Public planning receives only
an immutable mask/frame/canonical review declaration. Original identifiers and
provenance remain in the outer native audit binding. A context must share the
verified categorical source/frame with selected anatomy; this version refuses
a mismatch with the supplied source grid, sheared cells and unavailable evidence.
Legacy layer APIs inherit
millimeter units only from that verified shared source series; explicitly unknown
units refuse. No anatomy is relabeled or retroactive native guard attached.

For the acquired DEVELOPMENT cases, supplied label 2 is FLAIR context. It can
include nonenhancing tumor and edema, rather than establish healthy tissue or
function. The source segmentation used algorithmic proposals, human corrections
and expert approval; local import/registration review remains unreviewed. See the
[original UCSF-PDGM paper](https://escholarship.org/content/qt7gt8f8w4/qt7gt8f8w4.pdf)
and [official label definitions](https://www.med.upenn.edu/cbica/brats2021/).
No motor/language localization, injury law or deficit probability is introduced.

In **Compare two-tool strategies**, select supplied context, a research margin
and a maximum unique near-context volume. The exact nominal complete axial
CUSA/Frazier envelope includes both shafts, working parts, the stationary other
hand and withdrawal. Source-cell boxes use the existing 1e-9 mm closed-boundary
convention. Intersected retained source cells count once across the whole sequence,
independent of partition or reverse traversal of an identical fixed-axis path.
Historical context removal,
new removal, working/shaft proximity and representative RAS coordinates stay
separate. These are whole-cell geometric quantities, not continuous overlap
volumes or biological harm. Outside the source image grid remains unassessed.

The same limit filters primary access and paired placement, governs complete
sequence admissibility, changes comparison/STOP and is rechecked at every native
prefix before installation. Anatomical-limit STOP rebuilds its parent shortlist,
replays the complete declared approach/pair family and checks its ordered refusal transcript and fallback;
omitting, duplicating or reordering failures cannot self-seal a refusal. Search
expiry remains COMPUTATION_BUDGET, not proof of geometric infeasibility. Infeasible
counterfactual sequences stay inspectable as evidence and cannot be previewed or
installed. Source-cell markers and **Show supplied context in linked views**
connect the numerical constraint to the actual MRI RAS cursor.

Actual hidden 0007 controls demonstrate:

- Margin 0 / allowed near-volume 0: the existing two timed plans remain admitted.
- Margin 3 mm / limit 0: the three returned approaches encounter 3/7/2 mm³ of
  retained context during access; finite-family validation supports STOP.
- Margin 3 / limit 2: the third approach passes access, but complete maneuvers
  encounter 3 mm³. Both over-limit alternatives are rejected; backend installation
  refuses and the native baseline is unchanged. This distinguishes reaching from
  operating and withdrawing under the same anatomical constraint.
- Margin 3 / permissive limit: both timed strategies encounter the same 3 mm³,
  entirely working-part proximity. Historical source-context removal is 1036 mm³,
  new context/non-target removal is zero. Geometry does not distinguish their
  durations; existing target progress/reassessment still does. The two installed
  endpoint state IDs remain exact. The second acquired case still refuses full
  access, the exact no-target negative exposes no action, and missing functional
  evidence still selects STOP. Fresh native reopen verifies all saved studies;
  the representative context point drives linked MRI coordinates without an action.

Independent literal-tool/box-distance audits verify every exported physical and
context prefix, part counts, representative coordinates and cold replay. The
new third-route counterfactuals also pass 1645/1659 independent complete-tool
geometry checks. The independent endpoint auditor covers native programs and
recorded primary-access refusals; finite-family coverage and other rejection
reasons remain the canonical validator's responsibility. Fifteen source/contract
controls plus seventeen independent-auditor controls cover masks, source bytes,
private-reference invariance, missing/unknown units, exact transforms, false
metrics, source-binding forgeries, stationary hands, shafts, withdrawal and
partition invariance. Existing V1 six studies/two applications replay exactly;
regenerated default report hash remains `sha256:7b0f0d51fbc69e1e8ac83565b98cff8d5d4c92a332d391d01bcdbb8fa7c10d73`.

The 273-test backend regression, latest 32 endpoint/auditor checks, 1,217 desktop
tests and build pass. Final-code cold verification also checks the nine-study/two-
application native save and the second case's three studies without changing them.
The final checks and exact commands are in the ignored
`testcases/anatomical-encounter-v1/README.md`. No acquisition, EVAL read, training
or clinical calibration occurred. This is a DEVELOPMENT integration, not held-out
decision-quality evidence. Positive context previews take 2.23–2.48 s IPC in this
run; the harder complete-family negative required **64.94 s** for generation and
canonical validation before final parent-shortlist hardening. Final cold verification
of the nine-study/two-application workspace takes **87.02 s**. That cold proof is a measured next optimization target;
its constraints were not relaxed. The immutable context mask adds one byte per
source voxel (about 8.5 MiB here), with transient arrays during projection; the
validation LRU still retains only bounded digest entries, not masks or engines.

## Completed: exact public action-prefix reuse in cold proofs

The context-limit proof's profile identifies repeated native compilation as the
main cost:16 recipe compilations,256 public step attempts and12,535 physical
microstep attempts. An invocation-local cache now reuses detached checkpoints of
fully checked, complete ordered action prefixes. Keys include exact durations,
poses, hands and parameters, bound to the full public observation and normalized
controls. Generation and cold validation own separate caches. Failed actions and
speculative reassessment probes never enter. A cold proof still reconstructs the
parent shortlist, checks every declared approach/pair and verifies its ordered
refusal transcript; its already independently rebuilt chosen pair is reused
within that proof. Native installation executes every action again.

One public root supplies immutable scene/guard owners and existing pure geometry
caches. Owner checks cover source/grid/guard metadata, state/ancestry, material
table, task, catalog and a cold occupancy cache. Checkpoints detach state,
ancestry, receipts and encounter sets. All exits release root and payloads;
only numeric diagnostics survive. The separate complete-report digest LRU keeps
its existing live source/native/public-input checks. Cancellation now propagates
through canonical validation, workspace validation/install/read and bridge calls;
a canceled proof cannot publish a study or completed-validation authority.

Three alternating paired runs against281b40dd on the same admitted0007 source and
index physics give these backend medians with warm source/OS caches and an empty
completed-validation cache for each arm:

| Workload | Before | After | Reduction |
| --- | ---: | ---: | ---: |
| Context limit2, generation |29.33s|19.59s|33.2%|
| Context limit2, cold canonical proof |35.26s|21.84s|38.1%|
| Permissive context, generation |6.54s|5.81s|11.1%|
| Permissive context, cold canonical proof |6.03s|5.33s|11.6%|

The full reports, decisions, native endpoints, receipts and every phase snapshot
hash are exact across arms and repeats. The limit2 decision still distinguishes
access from operating: a route passes access at2mm³, but complete maneuvers reach
3mm³ and remain un-installable counterfactuals with STOP. Cold public attempts
fall256→150 and physical microsteps12,535→8,070, including rejected trials.
There are77 prefix hits; the selected canonical pair also avoids duplicate
compilation. This is exact reuse, not a relaxed geometric or physical check.
The permissive long strategy still executes15native actions/642physical
microsteps on every preview or installation; native materialization medians are
1.541s in both arms. Warm completed validation remains about0.13s. Source/OS
cache warmth, backend work and native execution are measured separately; these
figures do not measure rendering.

Payload retention is bounded to64entries and16MiB of accounted key/payload
objects; this case peaks at14.8MiB, evicts48entries and retains none after exit.
One public root contains17,856,000bytes of labels, plus its grid and existing pure
geometry caches. The payload accounting excludes OrderedDict nodes/value tuples,
bookkeeping, observations, active engines and transient copy overlap: it is not
an RSS cap. No persistent checkpoint/scene cache or new dependency was introduced.
The ignored `testcases/context-proof-performance-v1/README.md` preserves exact
commands, source/code hashes, counters, timing scopes and failed intermediate
controls. These are DEVELOPMENT workloads, not generalization or clinical evidence.
All318backend regression tests,1,217desktop tests and the build pass. The45new
controls cover exact V1/V2 uncached equivalence, duration/type-sensitive prefixes,
detachment, bounded eviction, owner/source tampering, failed-check exclusion,
cleanup and cancellation during cold/warm proof and atomic bridge installation.
Six older V1 studies/two applications replay exactly; default regeneration keeps
the same full report hash recorded above.
Actual hidden Electron regenerates8studies, previews/installs both timed branches,
saves and cold-reopens. Long/modified-short endpoint hashes stay exact; paired
comparison of original and optimized8-study and9-study saves also verifies all
reports, applications, branches, cursor position and ancestry in full. Independent
literal-tool/source-cell audits pass the over-limit counterfactuals, permissive
comparison and changed duration. The real full-sequence-limit STOP takes42.156sIPC
and42.228sclick-to-accepted-display; positive previews take2.152–2.499sIPC. These
display intervals include dispatch/polling and surface preparation, not isolated
GPU time. Fresh reopen's first inspect takes63.984s for8studies/2applications;
it is a current application measurement, not a matched speedup experiment.
The source point drives linked MRI RAS[-76,120,62]mm with no native action.
The second acquired case still refuses full access through the same hidden
interface, and the exact missing-target control exposes no strategy action.
Original source files remain unchanged and no window becomes visible.

## Implemented: complete-program execution-offset sensitivity

The desktop can now assess a frozen supplied-context comparison under declared
command offsets, retain the result in its native workspace, select its all-tested-
offset result and inspect a failed commanded pose in linked MRI. Zero offset has
one scenario; nonzero offset has nominal and six±RAS-axis translations. Both
hands' commanded poses share one translation for the complete program. Durations,
orientations, anatomy and the physical model stay fixed. The0–2mm control is a
software stress bound, not measured execution or registration error. Subvoxel
commands change physical pose prefixes directly without image/mask resampling.

`AxisTranslationStress` extends the existing world/frozen-decision infrastructure.
Public program/observation snapshots detach before callbacks. Each action retains
native receipts, physical/context prefix hashes, phases, progress, reassessment,
withdrawal and first failure. A committed partial access is a failed complete
program, not a full successful movement. Public decisions seal before every world
is executed again on native clones; every result must agree exactly. Fresh source
qualification runs again before publication. Canceled/timed-out studies do not
publish. Eight native attachments and32process-local complete-content digest
entries bound persistence/reuse; no engine, scene or dense observation is cached.
One public snapshot costs26,784,000label/context bytes on0007, plus source
projection, one active public model/grid, pure geometry caches and transient copies.
This is not a process RSS cap.

Three policies use the same held candidate family and source constraints:
nominal; fixed nominal geometric margin; and feasibility in every declared
scenario followed by nominal duration/target/ID ordering. Fixed margin expands
all retained shaft clearance, mapped non-target working clearance, supplied
protection and context. It does not predict offset-dependent target progress.
This evaluates existing programs; it is not an uncertainty-aware route-search
optimizer, a continuous-offset guarantee or a matched-budget efficacy benchmark.

The actual hidden0007 run retains nine comparisons, two installed nominal branches
and two sensitivity attachments. Existing long/modified-short endpoint hashes
remain exact. In the existing0.75/1s comparison, zero offset selects short for all
three policies. At declared1mm:

| Nominal selection | Fixed margin | Every tested offset |
| --- | --- | --- |
| Short |STOP|STOP|

At−S, short removes0mm³ of new target while long removes1mm³: preference changes
to long. At+R/+S, both programs stop partially during access, before complete
withdrawal. No new non-target tissue is removed. Fixed margin already rejects
both nominal programs. **No scenario-method selection advantage is established.**
The changed preference and failure location are useful diagnostics, not clinical
outcome predictions. The failed requested pose drives MRI RAS
[-79.394037,118,62.832796]mm without changing native state; it is not the actual
partial-stop/collision location. Source bytes and original baseline remain exact.

Actual desktop assessment costs8.684sIPC/8.823saccepted display at zero and
41.983sIPC/42.044saccepted display at1mm. These include complete public and native
replays and ran alongside regression work; they are not isolated benchmarks or
speedup claims. Cold reopen with all nine comparisons/two applications/two
attachments takes115.480sfirst inspect in the final UI run. This added proof cost remains
a measured optimization target. Native source profiles for unrelated ongoing
retractor work are disabled in these cases.

The independent auditor predicts using literal tools, separate cell-distance
geometry and actual native transitions, without strategy/sensitivity predictor
imports. It verifies both saved studies'24candidate/scenario outcomes, including
partial prefixes, source-byte/context bindings, fixed margin and policy choices.
All400backend regression tests,1,220desktop tests and the final build pass.
The26independent auditor controls include retained-target shaft clearance,
working-target contact, private-owner traps, source corruption and resealed
world/prefix/metric/choice forgeries. Eighteen producer/integration controls
include subvoxel physical prefixes, detached callback inputs, numeric progress,
bounded cache dependencies, cancellation and source-linked save/cold reopen.
The final56producer/world checks also preserve the old registration actor
descriptor while identifying the new sanitized descriptor as execution stress;
execution offsets are not mislabeled as anatomical registration uncertainty.
Fresh hidden reopen restores the studies and scenario STOP selection. The second
acquired case retains no operative candidate and all methods STOP; the exact
missing-target control exposes no planning or sensitivity action. All windows
stay hidden and original source bytes stay unchanged. Seven of eight previous
comparisons and both installed applications remain exact. Restricted-entry
comparison7 examines19rather than20proposals before the same wall cap; both
honestly select COMPUTATION_BUDGET/STOP. The common rejection prefix and every
other field/physical result match; partial inventory seals consequently differ.
All nine current comparisons,
applications, sensitivities, branches and ancestry also match after final reopen.
Six older V1 studies/two applications and default report hash remain exact too.
The current ignored inventory and exact reproduction commands are in
`testcases/execution-sensitivity-v1/README.md`; it retains the weaker exploratory
context-only baseline, STOP empty-receipt mismatch and actual desktop progress-
protocol failure, followed by their fixes. No historical result was rewritten.

Prior art and evidence gates remain explicit. [Fitzpatrick et al.](https://pubmed.ncbi.nlm.nih.gov/9874293/)
distinguish target error from fiducial fit residual; neither supplies execution
calibration here. [Jungo et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC7156850/)
show why aggregate segmentation calibration does not establish subject-level
calibration. [Frisken et al.](https://scholar.harvard.edu/files/sfrisken/files/incorporating_uncertainty_into_path_planning.pdf)
already incorporate segmentation/brain-shift uncertainty into neurosurgical path
planning. This integration does not claim first uncertainty-aware planning.
The explicit calibration result remains unsupported. These DEVELOPMENT cases
are already exposed; no frozen EVAL data, acquisition, training or probability
claim was introduced. Supported anatomical/execution uncertainty, withheld or
continuous perturbations, broader frozen patients and matched robust-search
baselines remain required for the research milestone.

## Implemented: bounded cold-workspace proof reuse

The saved nine-comparison/two-application/two-sensitivity workspace was profiled
before this change. Its780engine steps (300native/480public) perform38,439physical
microstep attempts. Thirteen native restores replay the same seven-action
baseline. Of161.233profiled seconds,126.872are in engine steps and13.701in native
restore. These cProfile numbers include instrumentation overhead and are not
paired timing denominators.

`surgical_strategy_workspace` now restores each exact built-in JSON baseline
once per validation invocation, detaches each consumer's state, ancestry, region
records and protection metadata, and checks source/owner dependencies before
reuse and successful exit. Admission remains rebound per consumer. Non-JSON
inputs retain original native dispatch; forged/changed snapshots cannot borrow
cached authority. Native suffix installation, scenario execution and speculative
reassessment probes still execute independently. The existing public-only prefix
cache now also covers adjacent frozen sensitivity candidates in each world.
All burdens, full-tool/context/fixed-margin checks, phase receipts and decisions
are recomputed. Canonical output order and schema are unchanged.

Native reuse retains at most two roots and16MiB of accounted state/actions/history
objects; the observed single root costs286,841bytes. This excludes source/grid/
guard owners, geometry caches and active consumers and is not an RSS cap. Public
reuse retains at most64JSON checkpoints/16MiB, with one lazy public scene owner.
Both scopes release all retained roots/payloads/callbacks at exit, including
cancellation/failure. Only numeric diagnostics remain. No cross-call native
checkpoint authority or installation bypass is added.

The reusable paired harness `scripts/benchmark_strategy_workspace.py` loads the
retained9352e7c1planning/proof modules without changing the checkout. Both arms
use identical physics, alternating order and empty canonical proof caches against
one frozen DEVELOPMENT workspace. Source/OS caches are warm; source loading and
rendering are excluded, while source qualification and complete canonical proof
are included. It verifies complete comparison/application/sensitivity/branch/
ancestry fields and original source bytes, rather than comparing only choices.
The three unrelated ongoing physics edits remain untouched; backend tests and
paired measurements load those three files from the index in memory. Actual
Electron checks use the shared working checkout and remain hidden.

Three alternating paired cold proofs reduce median108.833→94.179s (13.5% lower;
before108.717–110.488s, after93.462–94.674s). Warm medians124.573→123.668ms
are effectively unchanged. Every run reduces300→216native and480→428public
step attempts (780→644total), and38,439→34,301physical microstep attempts.
The84native steps are12redundant seven-action baseline replays; native scenario
and installation actions still execute. Public reuse skips52exact repeated
steps, while failed prefixes and reassessment probes still execute. The final
seven-world public scope records45hits and15,300,429peak accounted payload bytes;
all scopes retain zero entries after close. Every frozen report, installed
application, sensitivity, branch and ancestry field remains exact. This is
backend proof improvement, not a rendering/GPU or clinical efficacy claim.

All443backend regressions pass, including43new baseline/public-cache controls,
source/metadata forging, private-reference invariance, cold and warm validation,
cancellation, legacy/shared-clock behavior and independent geometry/auditor tests.
The independent native auditor also passes all24outcomes in the two newly saved
studies. Actual hidden desktop controls regenerate all nine immediately prior
comparisons, both installed applications and both sensitivity studies exactly,
with identical branches, ancestry, timeline and task admission. Both installed
physical endpoint hashes remain exact. The second acquired case still selects
STOP and the missing-target control exposes no planning action. New zero/1mm
assessment observations are7.745/36.443sIPC and7.841/36.578saccepted display;
these ran concurrently with regressions/controls and are not paired speedups.
Fresh hidden process reopen then passes with97.459sfirst inspect and0.856swarm
inspect. All nine comparisons, both applications/studies, branches, ancestry,
timeline and task admission remain exact after another native save. These are
application observations, separate from the paired backend timing. All windows
stay hidden and source bytes remain unchanged. No renderer change is required.

A separate bounded metadata/path inventory found only0005/0007currently task
admitted. Historical larger manifests reference absent source/bundle paths;
they do not establish a ready8–12patient benchmark. Existing exposed0004has
source-grid categorical target/context, but lacks the role/provenance declarations
accepted by current admission and a post-exposure checkpoint. Its older context
descriptor also needs semantic qualification. PAT22recovery remains exposed
TRAIN with explicit target-policy/research-episode holds and a derived fractional
annotation, not a new independent evaluation case or calibrated uncertainty.
The ignored inventory, measurements, failures and commands live together in
`testcases/execution-proof-performance-v1/README.md`.

## Implemented: acquired-source preflight and decoded provenance

Older acquired collections can now resolve DEVELOPMENT authority from the trusted
cohort registry and pinned acquisition declaration without changing their native
manifest. Exact patient/visit/collection, acquired originals and modalities must
join; ambiguous identities and alias/derivative role conflicts refuse. Any native
mirror revision still uses the unchanged older staged gate; a broken declaration
never falls back. Receipts bind current authority bytes and native source identity.
They establish declared development use, not official-source byte equivalence,
independent patients, clinical inputs or anatomical qualification. The old staged
and learning holds remain unchanged.

Independent review exposed a pre-existing source-verification gap: genuine
original NIfTI files and arbitrary resealed native arrays could each pass their
own checksums and the derivative ownership reconstruction. The common cold source
verifier now proves original→native float32 MRI pixels/affine/mm/spacing, and
the entire supplied categorical array's exact integer classes/affine/source-label
declaration. Both lossless native uint8/uint16 storage widths remain valid.
Single-file NIfTI1/2 is supported; paired sidecar data needs additional source-
member binding. Unknown annotation units inherit mm only on the identical verified
parent image grid. Edited supplied masks require a separately qualified derivation;
unreviewed model estimates retain their provenance. Source fingerprints fence warm
reuse, and canceled/failed work cannot publish verified cache entries. This adds
source consistency, not anatomical correctness or biological validation.

The generic read-only CLI and actual desktop now show source/checkpoint readiness
for native patients without an admitted task. Source proof, original bytes/replay,
clock/tool versions, actual exposure, withdrawn/static baseline, task and review
are separate checks. Metadata-only desktop inspection explicitly leaves original
bytes unverified there. A preflight adds no action, clears no tissue and grants
no planning authority. Existing admitted controls and public planner inputs stay
unchanged. Automatic inspection excludes generated fixtures, preserving the exact
existing warm replay hash-count/no-clone gate.

Actual0004qualification uses its existing exposed registry/acquisition source.
The older derivative correctly refuses omitted layer-revision metadata. Re-export
through the current generic path preserves MRI, affine and every selected source
mask exactly; the original case and native manifest stay unchanged. Five originals
and seven native arrays pass decoded source checks, but the intact0-action/
0-removal checkpoint still has no exposure/task or modern tool/clock setup.
Its annotations and estimated vessels remain unreviewed. `lesion-context` is still
outside the current context endpoint; no relabeling or functional claim is made.
PAT22still refuses the native source pipeline, with its older holds intact.
This is the third development anatomy with verified source bindings; it remains
outside the admitted task distribution and does not establish an8–12patient benchmark.

Hidden desktop checks cover the current/older0004derivatives, the missing-target
control,0005admission and the full saved0007workspace. Fresh0004save/reopen retains
zero actions/removal/no admission and exact branches/timeline; no planning control
appears. All nine existing0007comparisons, two applications/studies and native
ancestry/admission/timeline remain exact after cold reopen. Its first/warm inspect
observations are97.885/0.899s during concurrent verification, not paired speedups.
The whole0004CLI process takes3.75s/642MBmaximum RSS in the recorded observation;
loading/decompression and bounded transient original decodes are included, rendering
is excluded. No decoded volume is cached by this verifier. The current ignored
evidence, failed strict-width/hot-cursor runs and reproduction commands live in
`testcases/source-preflight-v1/README.md`.

The595-test broad backend run passes. Final source-focused verification passes
186overlapping controls after adding the single-file and overflow-safe header/
shape checks; all three real-source preflights pass against those exact module
hashes. These include104role controls,42decoded-source controls,9preflight
controls and the existing31admission checks. Coordinated reseals, stale authority,
alias/derivative conflicts, noninteger labels, units, paired sidecars, oversized
headers, warm corruption and cancellation refuse as intended. All1,221desktop
tests and the final build pass. Failed initial runs remain in the ignored record;
no previous refusal, replay/hash constraint or research hold was weakened.

## Remaining milestones and next dependency

This is a bounded acquired-anatomy operating task, not the complete strategy
platform. Patient-specific functional consequences still need compatible evidence;
the functional control remains a refusal. The new anatomical endpoint is a
geometric source constraint, not a qualified motor/language map. The new finite
execution stress is integrated but does not complete uncertainty qualification.
Add supported anatomical/execution uncertainty, a frozen comparison with matched
route-search baselines and independent challenge cases; continue reducing cold
proof/display cost and migrate remaining planner interfaces. The next patient
slice is a genuinely executed, source-bound0004exposure and declared task, plus
an explicit generic compatibility rule for its supplied context kind. The new
preflight must continue refusing until those gates pass. Expand the patient
distribution without touching reserved evaluations. Older lattice resolution dependence and
continuum empirical/domain gates remain qualification holds. No RL campaign,
new data acquisition, clinical-outcome probability or novelty claim was added.
