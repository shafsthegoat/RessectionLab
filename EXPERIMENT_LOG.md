# Executed experiment log

All entries below are development evidence. No final held-out patient benchmark
has been run. The patient remains the independent unit for population claims.

## October 4, 2026 — local runtime probe

`scripts/benchmark_runtime.py` ran on the Apple M5 with two PyTorch intra-op
threads and one inter-op thread. Exact workload definitions, warmups, all timed
samples and environment versions are in `docs/runtime-benchmark.json`.

| Synthetic workload | Median | p95 |
|---|---:|---:|
| Distance transform, 96³ grid | 40.34 ms | 41.65 ms |
| Distance transform, 160³ grid | 202.34 ms | 208.51 ms |
| Capsule-distance proxy, 400,000 point/segment pairs | 12.76 ms | 13.14 ms |
| Compact policy-style Adam update, CPU, batch 128 | 0.495 ms | 0.618 ms |
| Same update, MPS, batch 128 | 2.041 ms | 2.284 ms |
| Compact policy-style Adam update, CPU, batch 512 | 0.823 ms | 0.841 ms |
| Same update, MPS, batch 512 | 1.985 ms | 2.343 ms |

MPS execution succeeded, but CPU was faster for this workload. The initial
compact policy will use CPU. These are runtime microbenchmarks, not planning
quality, completed patient RL, or validated interaction-latency results. Distance
transforms must run outside the UI thread and be cached by input version.

## October 4, 2026 — initial independent contract attacks

The orchestrator reran `pytest tests/test_core.py tests/test_adversarial.py -q`
against the in-progress implementation: **26 passed, 10 failed**. The failures
showed that optional string metadata accepted mutable containers and malformed
halo masks silently promoted NaN, infinity, negative or fractional values to
anatomical evidence. Module owners are fixing the implementation; passing unit
tests alone had not exposed these issues. No scientific result relies on this
failed state.

A further review identified that hashing future context values into simulation
seeds could alter primary optimization. Separate permitted-input planning hashes
are being introduced while retaining full provenance hashes for audit/staleness.
The initial sequential fixture exposed only STOP; its insertion-boundary rule
is under correction before any patient learning result can be accepted.

## October 4, 2026 — acquisition findings

TCIA's current UCSF-PDGM v5 fixes diffusion metadata compared with v4 cited by the
specification. Its public transfer service reported unavailable files. A pinned,
hashed public structural mirror unblocks viewer development; equivalence to the
official source bytes remains unverified and is disclosed in the case manifest.

A separately acquired OpenNeuro BTC glioma case includes directional diffusion
and gradients, verified against release git-annex hashes. Its source tumor map
is fractional and has different voxel-axis ordering from T1. Strict categorical
import correctly requires an explicit, recorded derivation/alignment step.
Neither source is treated as clinical functional-outcome ground truth.

## October 4, 2026 — executed search and actual learning

The connected synthetic paid-access fixture requires paying normal-tissue cost
before reaching two target cells. SEARCH reaches the known return 1.74 in seven
transitions; greedy STOP returns zero. Three scratch REINFORCE seeds reach 1.74
after real updates, but require roughly 963–986 optimization transitions and
1.72–2.63 seconds. SEARCH took about 0.0025 seconds. This establishes no RL
advantage. Raw records and exact implementation snapshots are retained in
`artifacts/learning/connected-synthetic-v1/`.

The later branching fixture introduces competing tool paths and harmful
continuations. The development screen saved its configuration before execution,
used seeds 11/23/47 and never opened final/stress worlds. Three bundled
width/learning-rate/entropy configurations ran on this fixture and one UCSF
coarse structural case: **18 runs, 32 actual gradient updates each**. Bundling
settings prevents causal attribution to width alone.

| Development scenario | SEARCH score | Scratch selected scores, configuration A / B / C |
|---|---:|---|
| Branching analytic fixture | 1.30 | [0, 0, 1.26] / [0.08, 1.24, 0.08] / [0, 0, 0] |
| UCSF coarse model, subsequently invalidated | 1473.04 | [1429.82, 1429.82, 1429.78] / [1429.78, 1429.78, 1429.78] / [1473.04, 1429.78, 1429.80] |

Every run changed actor weights. No learned candidate beat SEARCH; one tied it.
SEARCH took about 0.049–0.057 seconds on branching and 0.456–0.480 seconds on
UCSF, compared with 0.339–1.717 and 4.800–7.381 seconds of training respectively.
These are saved development-run timings, not complete end-to-end planning
latency or an externally validated patient benchmark. Raw records:
`artifacts/learning/development-config-screen-v2/`.

Six branching follow-ons increased training to 128 updates and compared entropy
weights 0.01/0.05. Both retained [0, 0, 1.26]. Diagnostics found nonzero actor
gradients and increasing STOP probability in seeds 11/23. Of 256 sampled
optimization-world episodes, the later checkpoints stopped immediately in
235/228 episodes and produced only one positive episode each. Harmful sampled
continuations made local abstention attractive. Extra width, updates and this
entropy increase did not resolve the exploration problem. Records:
`artifacts/learning/branching-budget-entropy-followon-v2/`.

## October 4, 2026 — independent native-grid falsification

The separate checker accepted the UCSF candidates under their declared coarse
grid, but rejected **every non-STOP sequence** when removal was checked against
original 1 mm image cells and the actual distal tool capsule. First action:
92 mm³ of claimed tissue, zero fully contained source-cell volume, first
offending index (180, 150, 102), world RAS (-180, 89, 102) mm. Across retained
sequences, 1604–1820 mm³ of claimed volume lacked footprint containment.

This invalidates the coarse UCSF scores as physical resection performance; it
does not invalidate their use as a negative software experiment. STOP alone
passed this check. The footprint audit does not independently certify the
complete tool trajectory. Exact checker source and raw per-sequence failures are
saved beside the development screen. Native-grid connected removal, causal
microstep checks and a separate native-history evaluator are the corrective
vertical slice. The app must not present these rejected histories as approved
removed/residual-volume replay.

## October 4, 2026 — source diagnostics and packaged UI

BTC `sub-PAT28` raw-data tensor fitting ran on 101,311 voxels, excluding b2800;
FA median/p95 were 0.1939/0.6408 and median normalized signal RMSE was 0.06359.
CSA crop tracking retained 176 unlabeled diagnostic paths. Signal fit and
sampling repeatability are not anatomical accuracy. Strict planning rejected
uncorrected DWI. Reports and inspected overlay:
`artifacts/diffusion/PAT28-diagnostic-v2/`; reproduction and caveats:
`docs/diffusion-pipeline.md`.

The initial standalone arm64 application passed dependency, real MRI/3-D window,
internal-link and native-library checks. Local ad hoc signing is verified;
notarization and another physical Mac are untested. First package size was
1.28 GB; a prior failed Cocoa/VTK render-loop attempt is retained in build logs.
Interactive real-case inspection loaded the UCSF bundle in 0.30 seconds and
generated 54 routes in 1.32 seconds. Twelve geometric Pareto alternatives were
shown with source MRI and full-tool overlay. Screenshot review identified
clipped slice scale labels, opaque route IDs and poor accessibility of route
selection; fixes are underway. These are research usability observations.

## October 4, 2026 — native correction and patient learning

The native engine credits only fully contained original image cells, checks each
shaft microstep against the prior cavity, and records partial contact separately.
The real UCSF two-stroke sequence removes 249 mm³ target and 17 mm³ normal tissue;
the independent checker finds zero unsupported source volume. Cumulative partial
normal contact is 32 mm³: 26 remain and 6 are fully removed later. Partial contact
is never credited as removal at first touch. Residual target is 41,670 mm³.

Three scratch seeds received a 30-second optimization/selection wall budget.
Seeds 11/23/47 made 7/6/5 actual gradient updates; all actor weights changed.
Seed 11 improved from STOP to 245.24, matching SEARCH. Seeds 23/47 retained their
initial selected returns of 171.42/139.62. Cooperative budget overshoot reached
30.51 seconds; setup and independent validation are additional measured costs.
GREEDY and SEARCH reached 245.24 in 0.82 and 7.39 seconds, respectively, excluding
their separately recorded preparation. Nine frozen candidates shared four unique
native audits; all passed. Validation took 19.07 seconds using the preserved
older checker. No final/stress worlds were opened. These results establish actual
patient policy refinement, without an RL advantage or clinical validation.
Records and exact source: `artifacts/learning/native-ucsf0004-v1/`.

The independent checker was then optimized on the same saved two-stroke history.
Sparse connectivity and one whole-grid flood replaced 143 repeated floods.
Elapsed time decreased from 17.638 to 4.310 seconds, with identical geometry
calculations, tolerances and certificates. Ninety-one tests included sealed
pockets, diagonal contact, boundary cuts and randomized connectivity batches.
This measured 4.09× speedup is one before/after probe. The old study timing above
is unchanged. Evidence: `artifacts/native_simulation/ucsf0004-initial-development-v2/independent-optimization/`.

## October 4, 2026 — PPO negative result

Masked clipped PPO with GAE used the same initial weights as REINFORCE.
At 32 actual Adam steps, selected branching scores were [0, 0, 0.04]; SEARCH
achieved 1.30. A 128-step follow-on produced [0, 0, 0.06]. The 32-step PPO arm
used eight fresh rollout batches; the 128-step arm matched REINFORCE's 32 fresh
batches while spending four times its optimizer steps. Transition counts,
sample reuse, clipping and gradient diagnostics are recorded explicitly.
PPO did not resolve the observed STOP/exploration failure. Records:
`artifacts/learning/ppo-branching-v1/`.

## October 4, 2026 — Electron migration and native training bridge

Following the user's interface direction, the current Mac app uses Electron,
React, TypeScript and Three.js with a local Python numerical process. The first
standalone package loaded real MRI, generated 54 routes in 1.16 seconds and saved
route comparison/cursor through the macOS dialog. The 388,588,500-byte bundle
passed strict ad-hoc signing and had 17 internal links, no external/broken links,
and no Qt/VTK dependency. It was tested outside the repository working directory
with Python environment overrides cleared. This first bundle covers imaging and
search; later training verification must accompany its own source snapshot.

The native training bridge's 5-second UCSF run made one real update and retained
STOP. A 30-second run made eight updates and selected the independently checked
249/17-mm³ sequence. Total elapsed times were 8.91 and 37.77 seconds. Reports
bind source/runtime and selected checkpoint hashes; neither opens final worlds.
Adversarial tests exposed mutable resume budgets and unsigned crash recovery.
Both were fixed by binding checkpoint bytes to signed progress metadata and
refusing unverifiable recovery. All 37 focused bridge/refinement tests passed.
Reports: `artifacts/desktop-native-bridge/`.

Screenshot review caught incorrect pre-render camera bounds that mixed voxel and
physical coordinates. The fix is covered by an analytic sign-flipped source-grid
test. Source-derived surface meshes have no degenerate faces and differ in
display volume from source-cell volumes by under 0.6%; quantitative volumes still
use source cells. Binary hydration rejects shape, frame, dtype and accounting
corruption. Model refinement and removal replay are being integrated into the
new interface; the initial Qt workflow is preserved as historical evidence.

## October 4, 2026 — measured throughput and shared-policy comparisons

A bounded cache reuses only the certified initial native geometry inside one
simulator. Each fresh experiment arm still pays cold preparation, and hidden
cost worlds are sampled independently. Five profiled resets decreased from a
0.462-second median to 0.0259 seconds with byte-identical observations, removal
records and rewards. Peak RSS increased by about 35 MiB to 1.20 GiB. Raw profiles,
snapshots and independent certificate: `artifacts/performance/native-reset-cache-v1/`.

The original patient runner was frozen again with only that cache implementation
changed. With the same seeds and 30-second budgets, updates rose from 7/6/5 to
14/11/11 and selected returns were 245.24/171.42/245.16. Seed 47 improved by
105.54; seed 23 still retained its initial checkpoint. All native audits passed.
SEARCH reached 245.24 in 6.22 seconds and GREEDY in 0.75 seconds, excluding their
reported preparation. No RL advantage or final-world evaluation is claimed.
Records: `artifacts/learning/native-ucsf0004-cache-v2/`.

The first complete synthetic population experiment trained one shared policy
across two explicitly procedural development groups, with eight offline updates.
Each of three scratch/adapted seeds then received eight online updates and the
same transition/time limits. Frozen inference made no updates; adapted learners
started from the exact shared weights with fresh optimizers. SEARCH scored 1.30,
scratch [0, 0, 1.24], frozen 0 and adapted [0, 0, 0]. Every online actor changed,
but all adapted runs retained the initial shared checkpoint under selection.
All ten independent synthetic geometry checks passed. Offline cost was 1.374
seconds; the first frozen load/copy cost was not separately timed, a limitation
fixed in the subsequent runner instrumentation without relabeling the run.
No final/stress worlds were opened. This is a negative analytic experiment,
not patient-population generalization. Full records and exclusions:
`artifacts/learning/population-synthetic-v1/`.

## October 4, 2026 — native UI falsifies the route-to-cutting assumption

The complete Electron app trained the selected original Route 02 for 32 updates,
but actor gradients were zero: only STOP was available. The interface correctly
showed unchanged actor weights, zero removal and an accepted zero-action replay;
native JSON export and source-view restoration passed. The dedicated diagnostic
screen then tested all 54 original rays without spending training transitions.
Thirty-one passed static route geometry, but zero admitted the exact native
cutting stroke. The factory also dropped every selected entry/target pair in
favor of its parallel centroid proposal pool. For Route 02, it moved the entry
11.934 mm outside a 4-mm aperture. Restoring that entry still correctly rejected
shaft movement after three microsteps: the declared tip is narrower than its
shaft. A static access check is not a constructive cutting sequence.

An explicit alternative study retained the original rays but tested the two
named native profiles as additional instruments: 108 total combinations, still
zero accepted exact strokes. A separately declared source-grid-aligned window
and ray did pass with those profiles: 19 target + 1 normal mm³ for the fine
profile and 174 target + 11 normal mm³ for the wide profile, with independent
whole-tool and source-cell checks. This changes both access geometry and tool;
it does not repair or certify the original 54 routes. Diagnostics and preserved
source: `artifacts/route-native-screen-v1/`. Exact selected-ray pass-through,
preflight and explicitly labeled alternative generation are the corrective slice.

## October 4, 2026 — evidence QC and native interface boundaries

Four actual MPS extraction runs on PAT28 produced identical main/no-CSF masks
and distance maps. Independent native-grid, output-hash, finite-distance and
annotation-reindex checks passed. No-CSF excludes 214 supplied annotation voxels;
the main envelope includes them all, which is not an accuracy measurement.
Both proposals save/reopen separately from working anatomy and remain unreviewed.
Their addition leaves planning seeds unchanged. Native Electron inspection shows
the warning, threshold-derived source annotation and disabled cortical planning.
Adversarial review caught and fixed free-form provenance bypasses, descriptor
mutation, explicit prohibition precedence and silent enlargement of a reviewed
envelope. Model estimates remain estimated after review.

PAT05 was selected by a metadata-only rule before image acquisition and assigned
to development. Seven pinned creator-source files total 17,862,831 bytes. All
checksums and independent physical reindexing agree on 11,437 threshold-derived
annotation voxels. Full-head planning remains gated and diffusion is missing.
The registry now contains three development groups, two primary-source identities
and no final patients. Source images remain outside Git.

An isolated PyHySCO known-field phantom improved image error from 9.68% to 0.40%
with 0.395-mm field error. Three solver runs totaled 21.4 seconds and stayed below
416 MiB sampled RSS. Header and staggered-field-origin corrections passed tests.
Zero-distortion internal solver failure and BTC's unsupported transported PE
direction remain blockers. A rotated negative control also had low image error,
showing why similarity alone cannot approve unsupported geometry. No patient
correction or FSL execution occurred. Records: `artifacts/pyhysco-phantom-v1/`.

## October 4, 2026 — exact selected-route workflow and second-case extraction

The corrected desktop bridge preserves all 54 original routes and appends two
explicitly separate native-action candidates, retaining their distinct model
receipts. The original generic aspirator is refused before optimization because
its shaft intersects retained tissue. The separate native-wide profile and 6-mm
hypothetical window admit one declared stroke. With a 30-second learning budget,
seed 11 completed 32 actual updates; actor parameters changed and nominal
selection return increased from 0 to 171.58. The selected checkpoint differs
from initialization. Its one stroke removes 174 target and 11 normal mm³;
independent checking found zero unsupported source cells. Export, save and
fresh-process rechecking preserve the exact geometry binding. All source hashes
stayed unchanged during the run.

This is a protocol/workflow test of STOP versus a fixed stroke. Optimization
and selection seeds are separate but use nominal zero-shift anatomy; this does
not measure robustness to anatomical uncertainty or broad route-policy learning.
Concurrent integrated tests and app packaging prevent interpreting its elapsed
time as a comparative benchmark. Final worlds were not used. The complete
receipt is `artifacts/desktop-native-bridge/ucsf0004-exact-selected-route-v1.json`.
Independent adversaries verified complete requested-versus-actual tool/window
identity, no optimizer for actionless routes, unchanged saved runs after refused
resume, and separate search cohorts. The root integrated suite passed 598 tests.

PAT05 extraction used the previously frozen models and settings, with no case
tuning: no-CSF/main child wall times were 7.692/6.015 seconds. Both native masks
and signed distance maps have zero physical corner discrepancy. Independent
threshold, connected-component and hole-fill reconstruction reproduces both
masks exactly. No-CSF excludes 538 of 11,437 threshold-scenario annotation voxels
(4.704%); main excludes none. Inclusion measures overlap, not segmentation
accuracy. Both proposals remain review-required and confer no cortical access.
The six-plane QC was inspected, and source/model/output/runner hashes match.
Frozen records: `artifacts/brain-extraction/PAT05-mps-v1/`; independent audit:
`docs/brain-extraction-pat05-independent-review.md`.

## October 4, 2026 — physical-vector correction feasibility

An isolated, independently written susceptibility prototype represents both
native image affines and physical phase-encoding directions. Five final phantoms
pass their frozen numerical thresholds. On the 0.8-degree native pair, image
relative RMSE improves from 9.6792% to 1.0682%, with 0.3154-mm field error. Every
nonzero solve hits its fixed 40-iteration cap. An exact-identical-input branch
separately certifies zero objective and unit Jacobians before optimization.

Two earlier frame-invariance failures (0.2498 and 0.1044 mm maximum field
differences) are preserved. Using exact reference indices, relative affines and
directions derived from stored headers lowers the final difference to 0.000570
mm, below the unchanged 0.02-mm bound. Three revisions consumed 91.84 seconds
total worker wall time; sampled peak RSS stayed below 500 MiB. Thirty-four
focused tests pass; an independent auditor verifies all 51 generated-file hashes
and reproduces final errors. This one smooth, equal-scaling phantom does not
establish patient performance, solver convergence or a complete diffusion
correction chain. Production gates remain unchanged. Records and limitations:
`docs/vector-susceptibility-phantom.md`, `artifacts/vector-susceptibility-v1/`.

## October 4, 2026 — native Electron interaction verification

The updated complete app executed the exact UCSF stroke workflow, saved all 56
routes and comparison selections, exported the independently checked candidate,
and restored source annotations. A cancelled native run persisted six updates
(four were visible when cancellation was requested). After full app restart,
resuming added 26 updates under unchanged request and contract hashes.

Native accessibility testing caught a real defect: changing the timeline through
an accessibility action updated its label without fetching the corresponding
checked mask. The initial build evidence is preserved. The corrected renderer
uses one active replay request and one replaceable latest request, with separate
requested and displayed steps. Native decrement now shows checked step zero and
0 target/0 normal/41,919 residual mm³; increment shows step one and 174/11/41,745.
The banner, timeline, MRI overlay and quantities agree. Thirty-six renderer and
15 viewer tests pass. Final renderer digest is `5fe94739…`, using the same
verified `377284f6…` Python source snapshot. Receipts and matched screenshots:
`artifacts/electron-refinement-v2/`.

The source-surface display now averages normals only at exactly coincident
vertices. All triangle-position hashes remain unchanged, with finite unit
normals and no resampling or changes to tissue measurements. A fresh probe
measured 332.0 ms source mesh preparation plus 75.5 ms additional worker normal
preparation (+22.8%). It excludes GPU, IPC and visible latency. Matched native
screenshots show improved shading; source voxel stair steps remain. Evidence:
`artifacts/performance/viewer-normal-shading-v1/`.

## October 4, 2026 — source-bound structural proposal inspection

The Electron renderer now loads one selected estimated envelope separately from
working anatomy and source annotations. It verifies current image, physical
frame and mask hashes, then draws a dashed contour on the three original MRI
planes. Clearing, case changes and replay invalidate the estimate display. An
inspection control jumps to an actual supplied annotation cell outside the mask.
Actual PAT28/PAT05 hydration reproduces no-CSF outside counts of 214/538 and main
counts of zero, with both proposals still awaiting review. These counts do not
measure segmentation accuracy or certify cortical access. Local hydration was
245–299 ms for PAT28 and 234–239 ms for PAT05; this excludes native rendering and
visible interaction latency. All 47 renderer, 31 main-process and 19 viewer
checks pass, as does the production build. Native visual checks are pending.
Receipts: `artifacts/desktop-renderer/pat28-proposal-hydration.json` and
`pat05-proposal-hydration.json` in the same directory.

Before the registered procedural transfer experiment, independent review found
that adaptation checkpoint setup consumed the learning allowance, and that an
incomplete initial selection could be extracted as a selected candidate. Both
are being corrected before execution, with separate setup accounting and a
complete-selection gate. No registered training has run yet; construction-only
preflight passed with zero gradients and final/stress worlds closed.

Native follow-up now verifies both proposals on all three MRI planes for PAT28
and PAT05, including the 214/538 outside-annotation jumps at the exact source
cell coordinates (rounded only in display). Source clearing, case switching and
native file-dialog cancellation pass. A stale footer was fixed in a separate
renderer built from exact commit `e70edc1`; the final source digest is
`75d7f69b…`, with the earlier verified numerical engine unchanged. Existing UCSF
replay still independently checks 174/11/41,745 mm³ before source restoration.
No training was needed for this native verification. A same-case BTC transition
to an accepted replay was not exercised because its access gate remains closed.
Initial and corrected receipts are retained in
`artifacts/electron-structural-proposals-v1/`.

The procedural transfer implementation subsequently passed 123 isolated tests,
including actual-case construction without updates, all-arm tiny procedural
tests, checkpoint/source exclusions and corrected timing/selection contracts.
Registered execution was released from exact commit `dfab4c4`, independently
matched to the tested numerical snapshot; heavy background tests were paused.
Results will be recorded after all declared arms and independent audits finish.

## October 4, 2026 — retained procedural-transfer execution failure

Attempt v1 stopped after 158.44 seconds at the first public-patient adaptation
validator. Offline training and three scratch runs completed; adapted seed 11
was rejected before optimizer creation, and seeds 23/47 were not attempted.
There was no final candidate freeze or independent candidate audit, so the
partial records are not a validated comparison. All final/stress worlds stayed
closed. Diagnosis found three generator vectors represented as Python tuples
versus JSON lists in each panel; their values, ordered seeds, identities and
complete canonical hashes match exactly. The test-only path had constructed
both sides in Python and missed this serialized public-data boundary.

The repair compares complete canonical panels without omitting fields or
relaxing values and checks the actual public target/worlds/config before any
offline or online update. A separately declared v2 repeats fresh pretraining
and all arms with unchanged scientific settings. Failure evidence is retained
in `artifacts/learning/procedural-native-to-ucsf-v1/`; the attempt declaration is
`manifests/experiments/procedural-native-to-ucsf-v2-attempt.json`.

The integrated Python sweep passed 743 tests and exposed one orchestration-test
fixture that still bypassed the older checkpoint gate but not the new earlier
preflight. Its deliberately fake target was correctly rejected before the
intended incomplete-selection test. The isolated fixture now explicitly stubs
that upstream check to exercise its intended downstream boundary; the production
gate stays intact. The corrected orchestration and complete procedural runner
suite pass 13 checks in 23.79 seconds, and nine real-public-panel/tamper checks
pass in 1.30 seconds. The original full-sweep failure remains recorded; this is
a focused rerun, not a claim of a second complete-suite pass. Exact repair/test
hashes and results are in
`artifacts/validation/procedural-transfer-repair-v2/test-results.json`.

## October 4, 2026 — completed procedural transfer and independent audit

The fresh v2 attempt completed from committed source `68e4fde` and numerical
snapshot `e1186e12…`, using the original registered scientific settings. Actual
public-case preflight passed before any gradients. Fresh procedural training
made 32 Adam updates, 169 optimization transitions and 188 selection transitions
across two generated families. All six patient actors changed; online scratch
runs made 11–14 updates and adapted runs made 10–12. Each had a 30-second
optimization-plus-selection allowance; initialization and offline costs are
separate, selection transitions additional, and atomic overshoot reached 0.226 s.

SEARCH/GREEDY scored 245.24. Scratch seeds 11/23/47 selected returns
[245.24, 171.42, 245.16]; frozen procedural initialization scored 139.62; adapted
seeds selected [171.42, 171.42, 245.17]. Scratch seed 23 retained its initial
checkpoint despite subsequent real updates. Adaptation shows no advantage over
scratch in this comparison, and no learned method exceeded search. These seeds
are repeated optimizers within one reused development patient, not independent
patients. Pretraining contains zero human patients; functional evidence is
missing and world perturbations are zero.

All 13 frozen candidates passed independent native checks, using eight distinct
sequence audits and zero unsupported removal. A separate artifact reviewer
verified 138 frozen files, exact checkpoint tensors and earliest-best selection,
then independently recomputed every score from source-native labels, removal,
partial contact and tool costs. SEARCH removes 249 target and 17 modeled-normal
mm³; cumulative partial normal contact is 32 mm³, recorded separately. Residual
target is 41,670 mm³. Access and the estimated anatomy remain unreviewed, so this
bounded result does not establish a complete resection or clinical safety.

Launcher time was 288.87 s, with 33.12 s for independent validation and recorded
peak process memory of 3.317 GiB. Heavy background tests were paused, but these
sequential local runs are not controlled latency distributions. Final/stress
worlds remain unopened. Raw results, source identities and the generated report
are in `artifacts/learning/procedural-native-to-ucsf-v2/`; the independent receipt
is in the adjacent `procedural-native-to-ucsf-v2-independent-audit/` directory.
Failed v1 remains preserved and supplies no comparative result.

## October 4, 2026 — seven population-prior controls in Electron

The renderer now exposes seven distinct registered maps, loading one selected
value/coverage pair at a time. Functional concordance uses trilinear sampling;
released structural masks use nearest cells. Covered zero, positive values and
missing interpolation support remain distinct. A neutral hatch marks unavailable
atlas support; source anatomy, source annotations and planning inputs are
unchanged. Population status, pending alignment review and unknown patient
function remain visible. Actual seven-map hydration reads two assets per layer
and preserves source arrays byte-for-byte; local timings were 172–222 ms,
excluding GPU and visible interaction latency.

Production TypeScript/Vite build and 120 checks pass: 62 renderer, 27 viewer and
31 main-process. Source commit `0b4e334` is being packaged with the updated Python
engine for native GPU, cursor, source restoration and case/replay checks.
Previously packaged MRI review/expanded-plane layouts passed on PAT28/PAT05 at
1460- and 1050-point window widths, with exact one-millimeter keyboard movement.
Screenshots and source-bound receipts are in `artifacts/electron-mri-layout-v1/`.

## October 4, 2026 — read-only transfer failure analysis

A post hoc diagnostic compared 21 saved policies on seven common states using
the exact v2 runtime. Four optimization-world transitions reproduced fixed
source-family cuts and the saved patient SEARCH sequence; there were zero new
gradients, searches, checkpoint selections or final/stress samples. All saved
checkpoint bytes remained unchanged. The diagnostic ran in 5.09 s and its five
focused integrity/statistics checks passed.

The shared actor is not collapsed to STOP: its initial patient STOP probability
is 0.083 and entropy is 1.518 of 1.609 nats. It ranks a fine action above a wide
action despite much lower immediate modeled reward. Raw feature scales differ
between procedural and patient anatomy; 87.5–93.75% of hidden units saturate for
the patient wide actions. All initial critic inputs are identical despite
different physical returns, and every online gradient exceeds the clipping
threshold, though actor gradients are nonzero and weights change. These are
conditioning observations, not proof that a proposed normalization will improve
performance. Separately, the saved SEARCH sequence leaves only STOP legal before
the horizon expires: fixed proposal coverage is a distinct planning limit.

The next experiments must separate action-set expansion from model scaling and
retain fresh comparisons. Measurements, immutable policy/source bindings and
executed diagnostic source are in
`artifacts/learning/procedural-native-to-ucsf-v2/diagnostics-v1/`.

## October 4, 2026 — integrated validation and current-image integrity

A fresh complete Python sweep passed 749 checks in 107.54 s with four existing
DIPY basis-deprecation warnings. The earlier orchestration fixture failure and
focused repair remain preserved. The new complete-suite receipt binds tracked
Python and test sources at `5c57ddb` in
`artifacts/validation/integrated-python-2026-10-04-v3.json`.

Independent renderer review found a cache keyed by a mutable image container.
After a first valid evidence load, replacing or changing its MRI array could
reuse the stale source hash. The app had no observed mutation, but this weakened
the integrity boundary. The cache was removed; each evidence inspection now
hashes current bytes. Four adversarial tests reproduced the old acceptance and
now reject before loading another prior or structural mask. All 66 renderer
checks and the production build pass. A descriptive seven-map hydration pass
changed from 1,412.5 to 1,523.6 ms total (median 198.9 to 212.9 ms); this is not an
isolated hash-cost or visible-latency benchmark. The corrected renderer is
being packaged separately from the initial seven-map build.

Independent actual WebGL testing then exposed 18 false-unavailable samples among
64 oblique coverage-boundary landmarks. Float32 inverse-transform residue
introduced a spurious uncovered interpolation corner. The corrected sampler
shares an affine-derived numerical tolerance between CPU and GPU, snaps only
within that band, abstains at ambiguous outer faces and rejects grids requiring
more than 0.001 voxel tolerance. Every positive uncovered contribution beyond
the band remains unavailable; binary ties follow the same convention. Cached
tolerances bind all shape/affine values and refresh after mutation.

The production build and 131 desktop checks pass. An independent real WebGL2
probe passes 487 coverage/value cases on the local Apple M5. Maximum scalar
error is 4.9055e-7; the test uses a documented float32 forward-error bound.
The initial coverage failure and an intermediate overly strict, ad hoc scalar
threshold failure remain recorded, with their exact source and outputs, in
`artifacts/desktop-renderer/prior-gpu-sampling-v1/`. These are numerical display
checks, not clinical atlas-alignment validation.

## October 4, 2026 — expanded native proposal development probe

A separate post hoc probe tested whether the fixed action inventory limited
coverage. It froze a thirteen-column residual-target rule before the comparison,
using the completed v2 runtime, same source, 6-mm access disk, tools, physical
reward and native legality checks. Both old and expanded inventories used nominal
greedy selection with caps of three cuts, 128 previews and 45 search seconds.
No policy was trained or optimization/selection/final/stress panel opened.

The fixed inventory removed 249 target plus 17 normal mm³, with 32 mm³ of partial
normal contact, score 245.24 and 24 checked previews in 1.008 s. The expanded
inventory removed 1,113 target plus 62 normal mm³, with 177 mm³ of partial normal
contact, score 1098.77 and 66 checked previews in 14.951 s. It stopped at three
cuts. Target coverage increased from 0.594% to 2.655%, leaving 40,806 mm³; this is
an action-inventory result, not an RL gain or a complete resection plan. More
modeled normal removal and contact accompanied the increased target removal.

The expanded history passed independent full-tool/prior-cavity/native-cell checks
in 14.750 s with zero unsupported removal. The fixed history exactly matches the
previously audited two-cut sequence, whose certificate was reused explicitly.
Solid and forbidden-barrier phantoms passed separate native audits. The barrier
run stopped at its preview cap with fourteen rays still unchecked and retained
all cells on and beyond the barrier. Seven proposal tests passed; a separate
artifact audit reconstructed source-cell costs, corner containment, dynamic
proposal order, greedy choices, all denominators and compressed history hashes.

The whole probe took 44.353 s. Its first launcher failed on a metadata-key typo
before any comparison; that failure and the corrected frozen attempt are saved.
Review also found overly permissive default relative tolerances and unreported
out-of-image skips in the prototype generator. The actual exact-axis, integer,
in-bounds cases are unaffected; a reusable implementation must correct both.
Production numerical code and the completed learning study remain unchanged.
Evidence: `artifacts/native-frontier-expansion-v1/`, including lossless compressed
histories and the independent audit. Raw histories remain unchanged locally.

## October 4, 2026 — fixed feature-unit implementation, before registered training

The separately committed `procedural-native-feature-units-v1` declaration changes
only actor input units. Physical rewards, critic inputs and outputs, loss weights,
clipping, optimizer and the original proposal set remain unchanged. The shared
learner now carries an immutable named profile, a nontrainable divisor buffer for
FEATURE_UNITS, and separate full-policy versus trainable-tensor identities. RAW
retains its exact arithmetic and parameter state keys. Checkpoint loading,
adaptation and resume reject mismatched profiles before optimizer construction;
the analytic population guards remain separate.

Implementation checks passed: 43 focused tests, 58 broader regressions, five
final edge checks and 24 independently written adversarial tests (these sets
overlap). The independent tests cover exact RAW initialization/forward/gradient
parity, paired trainable tensors, hidden buffer keys, rehashed metadata/buffer
attacks, masked gradients and cross-profile resume/adaptation. Tiny constructed
fixtures exercised actual updates; the registered patient comparison has not
run. Its separate runner and integrated pre-execution validation remain pending.

## October 4, 2026 — complete native Electron prior workflow

The full Python engine was rebuilt from `0b4e334` (source `afd24365…`, executable
`85a92eff…`). Native checks on renderer `5c57ddb` exercised all seven maps, scalar
zero/positive samples, outside/incomplete coverage, structural membership zero
and one, minimum-width 2×2/expanded views, source restoration, case switching and
native save/reopen. Independent frozen-engine inspection verified unchanged MRI,
annotation, prior-value and coverage hashes in the saved case. An older accepted
selection replay rechecked 174 target / 11 normal / 41,745 residual mm³; its
checked initial step showed 0 / 0 / 41,919. No new training was invoked.

The final renderer from `c194a8b` has source digest `762973e2…`, with all 67 desktop
inputs matched to that exact commit. It retains the verified engine and includes
the independently audited GPU precision correction. Native follow-up repeated
the representative motor sample (0.9969 unitless), covered zero, outside and
incomplete support, expanded view and source restoration. The four GPU-audited
source files match the final package; no renderer/shader error was observed and
strict local ad hoc signature verification passed. The earlier seven-map/save/
replay checks are attributed to their earlier renderer rather than claimed as
fully repeated. A same-case prior-to-replay transition remains untested because
the enriched case has a different immutable identity and no accepted run.

Unmodified native screenshots, accessibility readouts, exact package identities
and scope limits are retained in `artifacts/electron-priors-v1/`. The app remains
at the representative MRI review view. Population alignment and patient function
remain unaccepted/unknown; distribution signing and second-Mac validation remain
open. Later experimental learner changes are intentionally absent from this
verified Mac app.

## October 4, 2026 — feature-unit execution gates passed

The full Python suite ran from an immutable archive of committed implementation
`0bffeaa`, independent of concurrent working-tree additions: 794 passed in
176.14 s, with four existing DIPY deprecation warnings and no source-byte change.
The actual public-case preflight then passed both profiles and all three online
seeds before any offline update. Paired initial trainable tensors and critic
outputs agree, distinct profile behavior is explicitly identified, and the two
constructed native source histories pass independent containment checks.
No registered training or final/stress world execution occurred in preflight.

The archive's numerical runtime is `850dab80…`; Git discovery is intentionally
disabled for the preflight archive, while the separate baseline receipt binds
the exact committed revision and archive bytes. Evidence is in
`artifacts/validation/native-feature-units-runner-v1/` and
`artifacts/learning/procedural-native-feature-units-v1-preflight/`. Registered
execution must use these tested frozen bytes; the newer proposal module remains
outside this learning comparison.

## October 4, 2026 — completed actor feature-unit comparison

The declared RAW/FEATURE_UNITS comparison ran from tested archive `0bffeaa`
(runtime `850dab80…`) without outcome-driven changes. Two fresh procedural
pretraining runs each made 32 Adam updates; all twelve patient learners changed
actor weights over 10–15 actual updates. All 23 frozen candidate identities
passed nine distinct native audits. A separate artifact audit passed seventeen
self-checks and verified source/profile/tensor identities, initialization,
selection, costs and physical scores recomputed from source cells. It checked
geometry certificates rather than rerunning their geometry implementation.

SEARCH/GREEDY scored 245.24. RAW scratch seeds scored [245.24, 171.42, 245.16],
while scaled scratch scored [245.24, 171.42, 245.24]. RAW frozen initialization
scored 139.62 versus scaled frozen 245.24. RAW adapted scores were
[171.42, 171.42, 245.17]; scaled adapted scores were all 245.24, with every run
selecting its unchanged initial checkpoint. The scaled frozen policy improved,
but additional adaptation gain was not established and no learner beat search.
Paired trainable initialization does not imply identical initial behavior after
the input transform; the comparison does not isolate initialization from offline
learning as the source of the frozen-policy difference.

Online clocks were 30.027–30.340 seconds, including initial and later selection.
Selection consumed 11.59–15.40 seconds within those clocks. The full offline
calls were 30.630 and 27.773 seconds; SEARCH used 6.127 seconds plus 2.091 seconds
cold setup. Full native validation took 43.904 seconds. Launcher time was
534.183 seconds and peak recorded worker RSS 2.573 GiB. Other heavy work was
paused, while ordinary desktop/OS load remained uncontrolled. These are local
observations, with matched declared caps rather than equal executed transitions.

The experiment remains one reused patient-derived development case, two
procedural families and zero human pretraining patients. Worlds had zero
perturbations; final/stress worlds stayed closed. The best sequence still
removes only 249 of 41,919 target mm³ under unreviewed anatomy and hypothetical
access. Its 17 normal mm³ removed and 32 cumulative partial-normal-contact mm³
are distinct categories. Clinical deficit probabilities remain null.

A separate read-only diagnostic loaded 42 checkpoints on seven archived states
with zero simulator resets, transitions, gradients or optimizer updates. Six
static checks passed. Fixed units removed measured actor saturation; the critic
still receives the same initial six features for different physical reward
scales. Algebraic norm decomposition shows stronger value-gradient contribution
to shared clipping after scaling, but larger actual scaled-actor displacement
means the record does not demonstrate clipping-caused learning harm.

Results, source-derived plots and complete cost tables are in
`artifacts/learning/procedural-native-feature-units-v1/RESULT.md`; the separate
audit and `diagnostics-v1/` preserve their source and limitations. The 50,380,301
byte native replay is retained locally and stored as an 836,680-byte lossless
gzip in Git, with byte/semantic roundtrip checks. Gzip-only report regeneration
reproduces the same Markdown and uncompressed evidence hashes.

## October 4, 2026 — bound proposal integrity and its measured cost

A standalone residual-column provider now binds the source, complete native
model, declared ray rule and current cavity. Independent adversaries exposed
cached-identity holes in its first implementation: source replacement before
preparation and edits to live cavity masks could change endpoints while keeping
old advertised hashes. Eight original failing cases and source bytes remain
preserved. The repaired provider validates actual source contents, reconstructs
the native V2 committed-history chain and all four cavity masks, and rejects
stale or altered exported batches. All 66 focused checks passed; an independent
rerun passed all 15 review cases. The native geometry engine is unchanged.

A subsequent single public-case observation matched the frozen prototype's
initial and post-cut inventories exactly and rejected a stale batch. Native
source setup took 2.324 seconds, provider preparation 0.044 seconds, and each
integrity/proposal/validation call approximately 0.285–0.302 seconds. One native
preview took 0.281 seconds and its paid commit 0.028 seconds. Cumulative process
peak RSS was 1,245,593,600 bytes; it is not an isolated allocation measure. Other
heavy jobs were paused briefly, but these single observations do not establish
a performance distribution or a speedup. No learning or final/stress worlds ran.
The module remains outside selected-route refinement and learning. Exact source,
phase timings and limits are in `artifacts/native-proposer-integrity-v1/`.

## October 4, 2026 — two prospectively selected structural development cases

Metadata-only selection of BTC PAT16/PAT20 was committed as `069c962` before
image access. Their roles and all derivatives are locked to development.
The bounded acquisition implementation passed 77 offline checks and was
committed as `ae1d36b` before either download. Exact pinned object versions,
annex MD5/length checks and measured SHA256 verified 35,727,017 new subject
bytes; 14,891 shared release-metadata bytes were reused. There were no acquisition
or preparation failures, and no diffusion or postoperative data was downloaded.

Independent review verified all seven source files per case, unchanged MRI and
physical frames, exact annotation reindexing/threshold derivation, and identical
case/planning identities after an independent save/reopen. Threshold 0.5 yields
45,400 PAT16 and 12,451 PAT20 target voxels. Maximum reindex residuals were
0.0003265 and 0.0001086 mm. Three-plane overlays were actually inspected;
PAT20 superficial bright features/outer-head irregularities are retained as
engineering observations, without diagnosis or expert anatomical acceptance.
Both cases still refuse full-head nonzero support as cortical access. Working
brain masks and preoperative context remain absent; unknown-timed diagnosis is
retrospective cohort inclusion only.

After independent QC, the registry was extended to five development
representations, four verified primary identities and zero final groups.
All 16 source-evidence checks and 26 cohort tests passed. Historical records,
aliases and final reservations remain unchanged. This is not five UCSF cases,
a tract-aware cohort or an independent clinical validation set. Source imaging
and case bundles remain outside Git; acquisition, independent QC and registry
receipts are linked from `docs/data_acquisition_pat16_pat20.md`.

## October 4, 2026 — isolated dynamic-action adapter, analytic validation

A separately versioned RAW-only axis-column simulator now presents the same
ordered, certified action inventory to search and the shared learner interface.
It retains native geometry/reward/feature semantics, owns reset/fresh/clone,
accounts for every primary/fallback check, refuses partial cancelled inventories,
and preserves existing fixed-route, procedural-checkpoint and feature-unit gates.
Post-commit interruption carries the actual committed transition; resumable
handling needs a separately reviewed runner, and generic learning currently
regards that interruption as a failed run.

Independent review found three defects before patient execution: interrupted
reset left integrity checks uninitialized; a raising geometry call was missing
from the attempt ledger; and a forged transient observation cache could pair
false removal cells with a genuine certificate ID. Five failing cases and both
pre-fix source versions are retained. The repaired adapter passed 29 new tiny
analytic tests (including 19 independent adversaries) and 29 unchanged-native,
selected-route and policy-profile compatibility tests. Independent native replay
passed on the analytic fixture. No patient execution, training or throughput
claim is made. Source, limits and review records are in
`docs/native-axis-adapter-design.md` and `artifacts/native-axis-independent-v1/`.

## October 4, 2026 — native guidance correction and packaged dependency notices

The notice-bearing `bee43c9` renderer passed 142 desktop checks, real-case
search (54 alternatives), estimated-support guidance, cleared A/B guidance,
blocked full-head guidance, minimum-width motor-prior inspection and source
restoration. Native screenshots exposed two overlapping unloaded welcome
messages. That failure is retained. A regression using the actual React App
reproduced it; the shell now owns the unloaded welcome and mounts the viewer
when a case exists.

The corrected `100865f` renderer passed all 143 desktop checks and its production
build. Native inspection confirmed the fix at 1460 and 1050 points, then loaded
the actual UCSF case, displayed the motor prior at the declared view-only
landmark (0.9969), and restored source MRI. The renderer digest is `34285015…`;
71 captured source files match the exact commit. All 2,809 numerical payload
entries and 292 notice-tree entries match the retained engine/capture. The
engine remains `85a92eff…` from `0b4e334`; later experimental adapter/learner
changes were not added to this package. Local ad hoc strict signature checking
passed. Native verification ran no training and opened no final worlds.

Available upstream notice text is now inside the actual app and travels with
it. Two exact-source notice gaps remain explicit; this is not distribution
compliance or notarization approval. The original visual defect, corrected
screenshots, native receipts and exact package identities are retained under
`artifacts/dependency-notices-v1/`.

## October 4, 2026 — integrated axis/preflight verification before public execution

Immutable `100865f` passed 914 tests and failed one historical nonpatient
orchestration fixture. The fixture accidentally copied the evolving cohort
registry into an older frozen declaration; the production byte check correctly
refused it. A scoped test-only repair and explicit real-preserver drift test
passed 63 focused checks. Production source checks and historical declarations
were not changed. The original full failure is retained.

The later immutable `558b2e3` snapshot passed all 959 tests in 196.11 seconds,
with four existing DIPY warnings. All 2,669 baseline files stayed unchanged.
This includes 37 preflight checks, independent adversarial publication/timeout
cases, and six tiny RAW integration tests. The latter also establish a concrete
spatial feature alias and document incomplete generic accounting after an
interrupted committed transition; no patient gradient experiment follows until
that separate accounting slice is tested.

The declared public preflight opens no final/stress worlds and performs no
gradients. Its five episodes, source/runtime identities, complete inventories,
independent certification and 600-second/6-GiB cooperative caps are frozen before
execution. It measures cost rather than selecting a better policy. Launch
baseline is in `artifacts/validation/native-axis-public-v1/`; execution results
will be recorded separately without rewriting that baseline.

## October 4, 2026 — native action preflight V1 stops before simulation

The source-frozen public run used `558b2e3` after the 959-pass full suite and
committed launch baseline. It stopped during native-configuration identity
validation, before constructing the adapter. No patient transition, gradient,
policy initialization or eligible candidate occurred. The launcher took
1.064 seconds; worker preparation recorded 0.600 seconds and a peak RSS of
354,877,440 bytes. Captured and archived source bytes remain unchanged.

The direct native-case helper produces a different identity from the historical
factory. A separate component audit will determine the exact difference; the
strict check was not bypassed and no automatic retry was made. Original failure
records are in `artifacts/preflight/native-axis-v1/`, with root verification in
`artifacts/validation/native-axis-public-v1/execution-result.json`. A later
corrected attempt needs its own committed declaration and output namespace.

Two subsequent configuration-only audits of the exact frozen V1 source agree:
all 11 public constructor fields and the full native fingerprint were compared,
and only
`tissue_support_provenance` differs. Tissue, target labels, affine and hard
exclusions have identical dtype, shape and elements. Access, instruments, source,
case identity, 0.25-mm microsteps and the 4,096-step cap are equal. Both complete
hashes were reproduced; no engine, simulator, policy or transition was created.
The direct helper's full configuration hash is `6d817c5c…`, while the historical
factory remains `0924b7d0…`. Their distinct provenance remains truthful and
unchanged. The independent diagnostic's first JSON export error is retained as
a diagnostic failure, separately from the successfully checked numerical data.
A fresh V2 declaration must bind the direct configuration and all unchanged
physical components. Original V1 evidence is not relabeled.

## October 4, 2026 — frozen extraction repeats on PAT16 and PAT20

The prospectively committed `2e54b98` declaration ran its four sequential CLI
calls/eight model children without failure, retry or tuning. The unchanged
wrapper, MPS runner, pretrained model bytes and source images were checked
before and after. The complete batch took 83.03 seconds; individual model
children took 5.833–8.033 seconds. Maximum observed child RSS was 683,704,320
bytes and maximum MPS driver allocation 5,948,243,968 bytes. These sampled
measurements are not exact total-system memory peaks.

Both repetitions produce identical masks, float32 distance arrays and compressed
output hashes for each case/model pair. Independent saved-array reconstruction
from the distance threshold/component/hole-fill procedure agrees exactly for all
eight results. All native-grid corners agree with the source (0 mm maximum
displacement); masks have one component and no image-face contacts. Both
six-panel overlay sheets were actually inspected. The audit took 10.35 seconds
and 824,393,728-byte peak process RSS; 28 focused checks passed in 0.35 seconds.

No-CSF/main omit 2,045/19 of 45,400 PAT16 annotation voxels and 413/125 of
12,451 PAT20 voxels. All omissions are retained and flagged; masks were not
enlarged to include targets. Within-case repeatability and source-annotation
inclusion do not measure anatomical accuracy. Envelopes remain estimated and
review required, with no cortical access or working-brain promotion. The
original source case bundles stayed unchanged. Recorded runtime versions agree,
but historical interpreter/package binary identities were not preregistered.

Execution snapshots and logs remain under `artifacts/brain-extraction/`;
`docs/brain-extraction-pat16-pat20-independent-review.md` links the independent
array/visual reports, exact source hashes and limitations.

## October 4, 2026 — corrected native axis V2 completes before patient updates

A fresh declaration bound the truthful direct configuration after the V1
provenance-only diagnosis. Immutable `6930417` passed 77 focused checks in
23.07 seconds before its launch baseline was committed. The full V2 launcher
completed in 149.376 seconds with 1,411,072,000-byte peak worker RSS. All 46
captured source files and both original/execution case bundles remain unchanged.
No gradients, final worlds or stress worlds were used.

All five complete episodes passed native checking across two distinct frozen
histories. Greedy removes 1,113 target and 62 normal mm³, with 177 mm³ cumulative
partial normal contact, returning 1,098.77. The unchanged seed-11 RAW policy
removes 445 target and 13 normal mm³, with 72 mm³ partial normal contact,
returning 441.60. Independent source-cell arithmetic and 25 adversarial artifact
checks agree. These are deterministic development replays on one reused case,
not uncertainty samples or evidence of a trained-policy improvement. Every path
stops at the three-cut cap; terminal unpreviewed proposals do not demonstrate
that all feasible actions were exhausted.

The reused-instance greedy replay panel took 43.077 seconds and the unchanged
policy panel 48.194 seconds, including measured resets and episode exports.
The preflight reads observations more often than the generic learner; repeated
clone/factory setup is excluded, so this is not an exact training cost or lower
bound. Native audits took 23.855 seconds. Of 368 primary previews, 112 had unique
model/cavity/proposal/phase identities. Repetition identifies an optimization
candidate but establishes no cache speedup. Full cost scopes and nested counter
limits are in the source-derived RESULT and cost summary.

Both publication authorities and the candidate payload hash verify. Five raw
histories remain unchanged locally; lossless gzip reduces 39,286,826 bytes to
665,699. Byte/JSON roundtrip and gzip-only report reproduction pass. Initial
saved-value argmax can be independently checked, but later observation matrices
and logits were not persisted. A separate one-update pilot requires exact
per-decision logging and executed/returned transition accounting before launch.
Records are in `artifacts/preflight/native-axis-v2/` and the independent
`artifacts/native-axis-v2-result-audit/`; V1 failure evidence remains intact.

## October 4, 2026 — portable envelope proposals for PAT16 and PAT20

Both first-repetition main/no-CSF outputs now travel in separate portable case
copies through the existing tested persistence path. No common source change,
inference or download was needed. Original MRI, thresholded annotations,
physical frame and planning identity remain unchanged; full semantic identity
changes and revision advances from 2 to 4. All four estimates remain review
required with every annotation omission retained. Neither becomes working brain,
cortex or legal access. Exact report text and artifact hashes are embedded;
predicted signed-distance arrays remain external and are not surgical clearance.

Twenty-eight existing structural/persistence checks pass. A separate actual
save/reopen/source audit passed in 11.921 seconds with 636,518,400-byte process
peak RSS and reverified all 37 retained inputs unchanged. Fifteen adversarial
provenance tests passed in 0.25 seconds. Each proposed-as-working-support attempt
is rejected with BRAIN_MASK_REVIEW_REQUIRED. Historical runtime version agreement
does not attest extraction-time binary identity. Native UI inspection remains
separate. Receipts are linked from the portable independent review in `docs/`.

## October 4, 2026 — linked MRI recentering passes in the Electron Mac app

The exact `346df23` renderer passes 148 desktop checks and production build.
The packaged app moves the linked UCSF MRI cursor from [-173, 137, 112] to the
annotation center [-161.9, 84.9, 104.6] mm as rounded in the native display.
Exact RAS/LPS and rotated-source fallback coordinates are covered separately by
source tests. The action preserves selected motor/source mode, A/B routes,
annotation visibility, opacity and MRI layout at 1460 and 1050 points. Camera
preservation is visually observed; internal matrices were not instrumented.
Actual MRI-only import and unloaded states correctly disable the action.

Both newly audited portable BTC cases load, display each estimated envelope,
jump to all four omission examples and restore source MRI. Recentring preserves
the selected estimate. Review-required/view-only labels and blocked cortical
access remain intact. No new numerical engine or training was exercised.
All 72 renderer source inputs, the unchanged 2,809-entry engine payload and
292-entry notice tree match their captures. Source digest is `a6bb6b76…`,
archive `94553760…`, engine `85a92eff…`. Existing chunk-size warning, two
notice-source gaps and distribution-signing limitations remain documented.
Root inspected the wide centered and minimum-width PAT20 screenshots. Exact
records are in `artifacts/electron-annotation-center-v1/`.

## October 4, 2026 — bounded cache and truthful decision logging tested separately

The isolated capsule-cover prototype caches only immutable pure geometry, with
32-MiB payload and entry caps. Actual input-conversion and direct-helper mutation
failures were found, preserved and fixed. Seventy-eight focused checks and 26
independent checks pass. Synthetic reference/cached/reference previews, histories,
masks and independent certificates agree, including causal collision rejection.
These small local timings do not establish a patient or complete-learning speedup.
Production native geometry and the desktop engine remain unchanged.

The optional decision observer and native journal pass 54 compatibility checks
and 30 independent adversaries. Paired recording-on/off training preserves exact
weights, optimizer/RNG state, actions, forward and simulator-access counts.
Actual same-forward inputs/logits/value are captured; forced STOP records absent
outputs when the existing path skips a forward. Records bind to served native
observations and authenticated transition outcomes. Checkpoint identities and
any derived probabilities are explicitly post-run joins, not invented capture.
Two metadata/input-binding gaps were reproduced and fixed. The separately
committed seed-11 one-update pilot remains unexecuted until runner and integrated
checks complete; it is an integration test, not a claim of learning advantage.

## October 4, 2026 — immutable pilot regression and prospective execution release

The final one-update runner passed 53 focused checks, including 38 independent
adversaries, on constructed anatomy. A temporary validator regression incorrectly
required float64 source features instead of the actual float32 capture and was
corrected before public execution. An owner canonical-world fixture mismatch and
a standalone process mock issue were test-only failures, recorded separately
from static publication/checkpoint/audit review repairs.

Immutable `f3a0591` then passed all 1,150 tests in 215.95 seconds, with no skips.
All 2,904 archived baseline files remain identical with no new source files.
The full launcher took 217.46 seconds. Ten intentional metadata-tampering warnings
and four existing DIPY warnings are retained in the full output. Source archive
SHA256 is `e2db9dbd…`; exact source/runtime/input manifests and original output
are in `artifacts/validation/native-axis-pilot-public-v1/`.

A separate committed baseline releases the declared one-update RAW seed-11
integration pilot from that exact archive, with 300 seconds online, 600 seconds
worker time and a cooperative 6-GiB process RSS ceiling. Other agent computation
is held; ordinary Mac activity and OS caches remain uncontrolled. Cache reuse is
disabled. This release is not an execution result; failures or completion will be
preserved separately. No final/stress worlds are permitted.

## October 4, 2026 — one patient update completes without selection improvement

The prospectively released seed-11 RAW pilot ran once from immutable `f3a0591`.
It completed two optimization episodes, both complete selection panels and one
actual actor update. Initial and updated selection both returned 441.60; the
earliest-tie rule correctly retained the initial checkpoint. Optimization returns
were 122.65 and 771.55. All six episodes contain three real cuts, for 18 executed
and returned transitions. A separate zero-transition dimension probe is not a
scored episode. Six accepted certificates cover three unique native histories.

Full launcher time was 221.139 seconds, with 169.590 seconds online and 2.219 GiB
peak worker RSS. Initial and updated panels took 51.889 and 63.424 seconds.
The available timers do not isolate why identical selected actions took different
times. Final-transition counter prefixes record 432 previews in 98.248 seconds
and 66 integrity checks in 19.607 seconds; these are not whole-process totals.
No cache, final worlds or stress worlds were used. Other agent computation was
held, while ordinary desktop load and file caches were uncontrolled.

An independent audit reconstructs all 18 actual saved forward passes directly
from tensors with NumPy. Maximum logit/value discrepancies are 1.19e-7/2.98e-8
within pre-fixed tolerances; all 12 deterministic choices match earliest-row
argmax. Six stochastic choices were eligible, without replaying random draws.
Source-cell rewards, selected/latest checkpoint distinction, publication
authorities and hashes verify. Thirty-five audit tests and three report tests
pass; a test fixture mismatch is preserved. This audit adds no simulation or
gradient execution. Three original JSON payloads remain unchanged locally;
lossless gzip reduces 74,806,029 bytes to 1,248,266 bytes and both report and
audit reproduce from gzip alone. The finalized index binds 22 versionable files.
This is integration evidence and a negative learning result on one reused
development patient, with unreviewed anatomy and hypothetical access.
See `artifacts/learning/native-axis-raw-update-pilot-v1/` and the separate
`artifacts/native-axis-pilot-result-audit/`.

## October 4, 2026 — persistent case identity passes native Mac checks

Renderer `fe25f2a` passes 150 desktop checks and its production build. Native
UCSF → PAT16 → PAT20 switching updates the fixed header correctly; the loaded
identifier stays visible while the evidence sidebar scrolls at 1460- and
1050-point widths. Full accessible labels match the loaded bundles, and actual
BTC identifiers fit without wrapping. The unloaded label remains generic.
Annotation centering still returns the UCSF cursor to the expected displayed
coordinates. No arbitrary extreme-length identifier was loaded; that behavior
has source tests rather than a native claim.

All 73 captured source files match the commit. The 2,809-entry engine payload
and 292-entry notice tree remain unchanged; local ad hoc signature verification
passes. This UI pass performs no training or broader historical evaluation.
Root inspected the minimum-width UCSF and wide PAT20 MRI screenshots. Exact
build, native, screenshot and capture records are preserved in
`artifacts/electron-case-identity-v1/`. Environment and desktop workflow guides
now distinguish this package from separate experimental learning results.

## October 4, 2026 — synthetic spatial information distinguishes one RAW alias

A separate 7×7×8 native fixture probe completes nine transitions in 0.363 seconds,
with no policies, gradients or patient inputs. Eight owner checks pass. The two
first cuts have identical existing action features and reward 4.33, yet their
best two-cut returns are 39.43 and 39.68. Six proposed access-relative coordinates
distinguish the actions; eight cavity/residual volume-and-centroid values
distinguish the resulting states. Translation, RAS/LPS and row permutation
errors are zero on the regular fixture; rotation plus translation is within
1.11e-15 mm. Thirty-seven archived source files verify unchanged.

Seven independent checks reproduce physical units, sheared affine centroids,
the archive and exact cell-based rewards. The 0.25 return gap follows from
66 versus 41 newly contacted normal cells on the final cuts. An additional
counterexample finds a 2.00000000034-mm descriptor change when rotation roundoff
switches tangent-axis choice near its 1e-8 cutoff. This later finding is retained
separately, and documentation qualifies the original fixture-only invariance.
Different abstract masks also share the proposed summary; their native
reachability is unverified. Source reindexing changes the chosen basis, and
partial-contact history, current tool and remaining budget are omitted unless
retained separately. No Markov, learning or clinical benefit is established.
The prototype remains outside the production package. See
`artifacts/native-spatial-feature-probe-v1/` and
`artifacts/native-spatial-feature-independent-v1/`.

## October 4, 2026 — exact dependency-notice provenance remains unresolved

A read-only upstream follow-up finds a complete repository license added in
May 2025, but that commit still declares react-remove-scroll-bar 2.3.7. The
published 2.3.8 metadata is unchanged, dates to December 2024, and names a commit
the official API still cannot retrieve. Current repository text was therefore
preserved as evidence without substituting it for verified 2.3.8 notice text.
This is a provenance limitation, not a legal-permission assessment. Earlier
lookup failures remain unchanged; no package installation or app modification
was made. Small official responses and exact hashes are retained in
`artifacts/dependency-notices-license-followup-v1/`.

## October 4, 2026 — declared public cache configuration is slower with zero hits

After a committed release and independent archive review, the exact `43857b5`
runner replayed the same three V2 greedy cuts in four phases. Reference-before,
cold-cache, warm-cache and reference-after whole-phase times were respectively
21.182, 22.896, 23.050 and 21.670 seconds. These include scientific comparison
and export. The cache produced zero hits in each 22,364-call cached phase;
all calls missed. Cold evicted 16,902 entries and warm evicted another 22,364,
retaining 5,462 entries and 33,553,224 bytes under the declared 32-MiB payload
and 16,384-entry limits. The experiment demonstrates no speed benefit at those
limits; counters alone do not identify exact distinct keys or reuse distances.

All four 23,482,729-byte scientific traces and their deterministic gzip exports
are identical. Each phase executes the same three cuts and inventories with
26/22/18/0 previews, returning 1,098.77. The uncached common constructor's
26 previews are separate. All four native history audits pass. A separate
34-test artifact audit checks certificates, exact source-cell reward and all
five saved mask digests by independent cell/flood reconstruction. It adds no
geometry replay, patient episode or gradient. Full raw-mask byte comparison
and callable restoration are source-bound runner observations; the auditor
distinguishes those from its own saved-evidence verification.

Full launcher time was 158.288 seconds; worker time 157.790 seconds and peak
process RSS 2,289,860,608 bytes. Four native audits took about 59.56 seconds in
total. Nested preview, integrity, proposer and cache timers must not be added.
Certificate capture is not separately timed. Other agent numerical tests,
builds and native UI activity were held; ordinary OS load and caches remained
uncontrolled. No final/stress worlds, optimization or changed scientific model
were used. All 2,930 source files, archive and case remain unchanged.
Original receipts are under `artifacts/benchmarks/native-axis-cache-v1/`, with
the source-derived report and separate audit in their corresponding artifact
directories. Existing production and desktop cache behavior remains unchanged.

## October 4, 2026 — route controls appear when candidates exist

Renderer `c09ee50` passes all 153 desktop checks and production build. Native
inspection at 1460 and 1050 points confirms that unloaded, pre-search UCSF and
blocked full-head PAT20 states display guidance without empty category or A/B
selectors. A real UCSF search restores the controls and compares selected routes.
All three categories switch correctly: 12 retained, 19 dominated and 23 rejected
candidates. Because none was empty, retaining category switching with an empty
active filter is verified by source tests and is not a native observation.
Case identity and annotation centering also pass; full-head access stays blocked.

All 73 captured source inputs match the exact commit; all 2,809 engine entries
and 292 notices are unchanged, with local ad hoc signature verification passing.
The current renderer digest is `834c7ceb…` and archive `bddc5236…`. Root viewed
the minimum-width pre-search and wide populated-search screenshots. No training
or historical full workflow was repeated. Exact tests, accessibility captures,
screenshots and package/native receipts are in
`artifacts/electron-route-empty-state-v1/`.

## October 4, 2026 — explicit spatial reference repairs the bounded frame probe

An isolated V2 descriptor requires an external tangent reference with source and
coordinate identity, rather than choosing a source axis. The declared sine
condition exceeds 0.1 plus a 64-epsilon rejection margin; this is an engineering
condition, not a clinical threshold or universal numerical guarantee. On the
saved V1 rotation counterexample, an explicit well-conditioned Y reference gives
2.22e-16-mm error, while a near-parallel X reference rejects in both frames.

Eleven owner tests initially passed. Independent review then preserved 14
failures and 16 passes involving direct-constructor validation, caller mutation,
nonfinite extreme-coordinate/volume outputs and discarded imaginary components.
V2.1 fixes those isolated validation paths and passes the same 30 independent
tests plus 11 owner checks. Fourteen saved accepted numeric comparisons are
exactly unchanged. The original and repaired source archives, tests and reports
remain separate and verified; V1 bytes are unchanged. Both probes are algebra
only, with zero simulators, transitions, policies, gradients or patient loads.
No learning, state completeness or cross-patient reference claim follows.
See `artifacts/native-spatial-feature-v2/repair-01/` and the separate independent
review; no production profile or public experiment was changed.

## October 4, 2026 — saved decisions change rank without changing the chosen path

A post hoc stdlib-only reader pairs every saved initial/latest selection forward
from the completed RAW pilot. All six pairs match exactly on ordered action IDs,
float32 inputs, state and mask; deterministic repeats yield three distinct states.
The chosen action is unchanged in each, but 8, 17 and 12 rows change ordinal rank
(including STOP). Top-two saved-logit margins change from 0.000252/0.006669/0.018136
to 0.004095/0.010175/0.030040. All three saved values change. None of the
26/24/22 legal non-STOP feature rows is an exact duplicate on this observed path.
Separate synthetic aliases therefore do not establish an alias explanation for
this patient's unchanged selection score.

Twenty-six constructed-record checks pass, including 15 independent tests.
Independent exact-rational arithmetic reproduces all per-action deltas, ranks,
margins and values; mean-centered differences differ by at most 2.78e-17 through
rounding. Source, RAW profile, saved checkpoint bytes and prior forward-audit
bindings verify. Review found duplicate decision IDs could be accepted across
slots; a strict identity gate repairs that validation gap. Actual scientific
fields remain identical across report revisions, and gzip-only reproduction is
byte-identical. Original reports and the initial test/audit harness errors remain
preserved. No model, simulator, geometry, RNG, gradient or new-world execution
was performed. See `artifacts/native-axis-decision-diagnostics-v2/` and
`artifacts/native-axis-decision-diagnostics-independent-v1/`.

## October 4, 2026 — read-only expanded-axis inspection boundary

A separate backend facade exposes the full initial proposal and preview ledger,
including omissions and rejected primaries/fallback attempts, without transitions,
removals or gradients. Exact geometry, support assumptions and source/tool/world/
reward bindings accompany every inventory; candidate and removal authority remain
false. Supported empty inventories retain their full denominator, while unsupported
or interrupted work raises without publishing partial success.

Sixteen owner checks and 33 independent synthetic checks pass. Review retained a
cached-source mutation failure and a valid LPS normal changed by 1.11e-16 during
normalization. Uncached entry/exit source seals and a recorded LPS-only four-epsilon
direction bound repair those issues; centers, radii and IDs remain exact, and larger
geometry drift still rejects. Existing selected-route readiness is unchanged.
The preparation timer excludes serialization and final checks; no patient latency
or desktop workflow claim is made. See `artifacts/native-axis-inspection-independent-v1/`.

## October 4, 2026 — prospective saved-query reconstruction

The frozen cache experiment's accepted certificates can support an exact ordered
projection of variable cover arguments, without new capsule geometry, simulation
or gradients. A prospective declaration caps the read-only public diagnostic at
22,364 queries, 60 seconds, 512 MiB peak process RSS and a 32 MiB uncompressed
trace. Full unsaved frame-key bytes and raw cover payload sizes remain unknown.
Entry-only reuse bounds explicitly retain the unchanged no-bypass assumption.

All 53 synthetic tests pass, including exact bytes/order for 114 independently
instrumented calls and exhaustive small LRU/reuse-distance comparisons. Review
preserved four failures involving duplicate accepted IDs and unchecked runtime
authorities, plus a later wording-only assertion. Final source and evidence are
hash-bound in `artifacts/native-cache-query-review-v1/` and the independent review.
Public reconstruction has not run; commit precedes a separate execution release.

## October 4, 2026 — expanded-axis RAW and fixed-unit compatibility

The exact axis backend and accounting wrapper now expose a source-bound raw
15-action/6-state schema to scratch training. The policy may use RAW or the
unchanged registered FEATURE_UNITS divisor vector; the critic, geometry, rewards,
proposal inventory and procedural/population restrictions remain unchanged.
The same-forward journal verifies actual transformed actor arrays against the
served raw observation without extra policy or simulator calls.

One hundred owner regression checks pass in 54.38 seconds; 25 independent checks
pass in 14.22 seconds on single-thread synthetic fixtures. Initial trainable
tensors match across profiles and prescribed physical histories remain equal.
The independent one-update fixture changes actor weights but retains its initial
checkpoint after a selection tie at −152.98. No learning benefit or public-case
validation follows. A test that incorrectly compared measured timing fields and
an independent missing temporary-directory setup failure are retained. Final
source/docs and compact actual receipts are frozen under
`artifacts/native-axis-input-profile-prerequisites-v1/` and
`artifacts/native-axis-input-profile-review-v1/`; full integrated validation is pending.

## October 4, 2026 — cache argument reuse is substantial

The prospectively declared saved-artifact diagnostic completed once from immutable
`b0f5635`, with zero new cover computations, patient loads, simulator steps or
gradients. All 22,364 ordered argument byte strings match the saved certificates.
There are 8,812 distinct keys and 13,552 first-phase repeated accesses; the four
inventories account for 8,812 / 7,444 / 6,108 / 0 calls. Distinct-key frequencies
are 1,368 once, 1,336 twice and 6,108 three times.

Because the complete key set is smaller than the old 16,384-entry cap, raising
that cap alone cannot help this fixed trace. With unlimited payload and unchanged
no-bypass admission, the entry-only model yields 13,552 first-pass and 22,364
repeat-pass hits. These are hypothetical counts; exact raw cover payload sizes,
byte-weighted reuse and performance under a larger memory cap remain unknown.

External child wall time was 1.108213417 seconds; the producer's 1.022518917-second
field excludes final summary/status writes. Archive setup and final hash checks
are outside both. All 16 copied source/input files remained unchanged. Cooperative
60-second/512-MiB checks passed; exact peak RSS was not retained and no rerun was
performed. The compressed ordered arguments occupy 1,197,360 bytes. An independent
saved-only audit uses different arithmetic/LRU implementations, passes all exact
checks and 33 tamper tests, and records no discrepancies. Original output is under
`artifacts/diagnostics/native-cache-query-keys-v1/`, execution evidence under
`artifacts/validation/native-cache-query-keys-v1/`, and the separate audit under
`artifacts/native-cache-query-result-audit-v1/`.

## October 4, 2026 — official UCSF source access remains unresolved

A metadata-only recheck found the TCIA collection page byte-identical to the
previous capture, still pointing to the same public v5 package. DataCite confirms
the release identity but supplies no per-file checksum or alternative case
manifest. Creator-linked examples and third-party derivatives do not establish
our mirror's source equivalence. The unchanged failed Aspera flow was not repeated;
a read-only NBIA timeout was recorded as inconclusive. No images, accounts,
installers, inference, registry changes or support messages were involved.
The remaining concrete route is provider-supplied case access or release-specific
checksums. Evidence and primary links are in
`docs/ucsf-primary-access-recheck-2026-10-04.md`; source equivalence remains unknown.

## October 4, 2026 — integrated profile attempt preserves 12 failures

A complete suite from immutable `523a72b` ran 1,482 tests: 1,470 passed and 12
failed in 243.45 seconds, with 14 warnings. Source manifests before and after
match exactly and contain no added files. Ten saved-query CLI tests hit the
512-MiB process-peak guard because they execute inside the larger pytest process;
their isolated focused run had passed. Two old pilot audit tests reject the new
axis contract before checking captured forwards or optimizer state. These are
real integration gaps, with all failures and the frozen source retained.

Repairs are separately assigned: isolate resource-sensitive test fixtures without
changing the production cap, and explicitly validate both historical and current
contract forms without discarding nested hashes or changing original experiment
artifacts. No public training or packaging release follows this failed attempt.
See `artifacts/validation/axis-profile-integrated-v1/attempt-01/`.
