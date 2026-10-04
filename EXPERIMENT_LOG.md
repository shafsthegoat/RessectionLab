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
