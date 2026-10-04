# Project status

Updated October 4, 2026. The three revised specification documents remain the
authoritative requirements and have been read completely. This file records
executed work and open gates, not a replacement plan.

**Latest steering (October 4): real-patient learning and probability-aware planning.**
The user supplied `docs/CODEX_STEERING_PROMPT.md` and
`docs/DATASET_ACQUISITION_ADDENDUM.md`; both were read completely and copied
unchanged from the supplied files. They supersede the proposed procedural
training screen. No synthetic population screen or opening-learning run was
executed. Existing analytical fixtures remain software checks,
not human training data. The immediate milestone is one real patient with
independently checked alternative plans and nonzero, evidence-bound functional
uncertainty. Population training must use existing real patients, preserve the
BTC TRAIN/SELECT/unopened roles, and assess transfer on separate patients.

Annotation-assisted and inference-only tracks stay distinct. Reviewable supplied
segmentations are valid inputs in the former; private references cannot influence
inputs or proposals in the latter. Remaining gates include broader executable
route coverage, a fully exercised patient desktop workflow, and actual
real-patient spatial learning with independent-patient evaluation.
Search may remain the strongest research mode;
no learned-policy superiority is established. The unrun procedural runner also
exposed a weak teacher on its development fixture; that negative finding is
preserved and will not be used to justify an RL advantage.

**Latest executed results.** Source-bound functional evidence now persists through
native refinement, bridge run configuration, checkpoints and replay. The real
UCSF-PDGM-0004 prior bundle was attached, saved and reopened in 9.12 seconds;
the original case and seven prior proposals remain unchanged. This derivative
uses unreviewed population priors with uniform ±2 mm / ±1° sensitivity only.
Atlas sampling covers 1,431,949 of 1,432,027 modeled tissue cells, with 78 unknown;
this does not establish patient functional coverage or registration accuracy.
The structural mirror's equivalence to official TCIA bytes is still unverified.
See `artifacts/functional-evidence-integration-v1/real-case-roundtrip.json`.

The independent full-tool event evaluator passes 68 combined controls. The
executed real-case comparison completed eight frozen panels in 28.984 seconds
(worker peak 2.108 GiB). Both retained native histories passed fresh geometry
checks. At map support 0.5, motor encounters were fine 51/64 versus wide 64/64
under Gaussian SD 1 mm / 1°, changing to 43/64 and 53/64 under 2 mm / 2°.
Mean and tail exposure costs increased with broader uncertainty despite lower
binary counts. Missing channels produced 64 unknown outcomes, never zero risk.
These are two instruments on the same axis, not two independently discovered
surgical approaches; three other rays remain rejected for native execution.
An independent saved-record audit recomputed counts, intervals and cost summaries
across 1,344 outcomes. See `artifacts/functional-sensitivity-realcase-v1/RESULT.md`.
This remains unreviewed population-map sensitivity, with no postoperative deficit
probability, new-patient generalization or complete-resection claim.

The local `evaluateCandidate` operation now seals a selected native history
before revealing its predeclared final worlds. Interrupted evaluation can resume
the same assessment; optimizer training cannot resume after sealing. Reopen,
replay and export verify the same full-sequence events. All 87 focused integration
checks pass; the current facade has only three final worlds. The full Python
regression from committed `575eccd` passed 2,134 checks in 286.80 seconds, with
six skips and 14 retained warnings. Five skipped DICOM controls passed in the
separate existing acquisition environment (21 checks); only one unavailable
historical bundle fixture remains skipped. All 3,555 archived source files and
the consulted UCSF bundle stayed unchanged. No repairs or installs were needed.
Receipts: `artifacts/validation/python-regression-575eccd/attempt-01/`.

The fixed-scene discretization control completed all 16 numerical rows. Five
motions were shaft-rejected without state changes; ten accepted histories passed
independent checks and the last audit reached its declared cap. At 1 mm, phase
or orientation alone changed acceptance. Even at 0.125 mm, credited physical
removal remained 13–16% below the analytic reference. Independent audits took
152 seconds versus 17 seconds in native strokes. These are numerical model
limitations, not patient training or clinical accuracy results. See
`artifacts/native-discretization-control-v1/RESULT.md`.

The real spatial adapter now accepts an explicitly bound, provisional research
envelope without promoting expert-review status. The first PAT05 profile from
committed `8aeb3be` failed at the native orthogonal-affine guard in 3.148 seconds,
before candidate previews, policy calls, transitions or optimizer updates.
The original bundle and all 53 numerical source files stayed unchanged; there
was no retry. Its fixed 32³ crop contains 24.36% of the supplied target and also
limits proposal reach. The independently checked, explicitly bounded correction
preserves source images and moves full-cell corners by at most 1.0842e-7 mm.
The separately frozen v2 profile completed in 9.311 seconds at 1.063 GB peak
sampled RSS: 36 initial legal actions, three fixed test actions, three untrained
forwards and no updates. Its full-history geometry audit passed. The shallow
test sequence removed 26.00009 mm³ normal tissue and no target; its actions
were chosen by inventory order, not the policy. Crucially, 10,558 of 11,437
target centers (92.31%) exceed the fixed catalog's distal axial bound. Broader
shared proposals and observed context must be checked before useful learning.
The original v1 failure remains preserved. See
`artifacts/pat05-real-spatial-profile-v2/outcome.json`.

The same frozen SynthStrip main model completed all four additional BTC
development estimates (TRAIN PAT22/PAT25 and SELECT PAT26/PAT27) in 35.024
seconds, without retries, parameter changes or source mutation. All four outputs
were fixed before annotation-overlap checks. All four proposal bundles now save
and reopen with unchanged source arrays, patient roles and planning hashes.
Independent native-grid and distance-to-mask reconstruction checks passed, with
zero omitted annotation cells; 24 sampled planes received nonexpert visual QC.
The estimates remain unreviewed and working support remains absent. No training
result follows from these inferences. PAT29/PAT31 and UPenn remain unopened.
See `artifacts/brain-extraction/BTC-spatial-main-v1-independent-qc/README.md`.

Shared observed-only search and real-patient batch bookkeeping now pass 51
focused checks, including six independent controls. Search preserves costly
opening prefixes, reports incomplete layers, and uses optional actor guidance
only to order legal actions. Bounded beam retention agrees with full-state
materialization across 430 budget/width controls. Independent review reproduced
and repaired two exceptional-exit accounting errors; the original failures are
retained. Six-TRAIN/two-SELECT roles and patient-uniform imitation sampling are
checked without opening images. These helpers performed no patient search or
optimizer updates. Full-field search and cropped actor inputs remain different
representations. See `docs/real-patient-learning.md` and
`artifacts/real-learning-helpers-independent-review-v1/receipt.json`.

An opt-in common nominal/cavity proposal generator now offers exposed openings
and proximal/distal target endpoints across the existing 13 source columns.
It uses permitted annotations/estimates and observed cavity state, keeps engine
target labels zero, and requires a fresh full native preview for every emitted
unique ray. Endpoints are no longer clipped to the actor crop; visibility and
bounded-family coverage remain separate from geometry acceptance. Seven
independent controls pass, including hidden-reference swaps and paid-opening
sequences. Three legacy states/two transitions match the original default
exactly. Real-patient coverage has not yet been measured with this generator.
See `artifacts/nominal-cavity-independent-review-v1/provider-receipt.json`.

The first real TRAIN PAT22 inference profile completed all four declared cells
in 6.247 seconds with unchanged model parameters. One CPU call took 28–30 ms
for a 32³ crop and 256–269 ms for a 64³ crop, with highest observed memory about
1,180 MiB. These are single cold calls, not accuracy, training-fit or complete
planning timings. The image encoder dominated these calls. An optional critic
now also observes legal tool/candidate context; its default preserves the old
model exactly. It has no measured learning advantage yet. See
`docs/real-spatial-policy-scale.md` and the compact execution receipts.

One official ReMIND development case is acquired: 386 preoperative DICOM files,
59,163,552 bytes, with verified object identities and native pixel roundtrips.
The T1 and T2 have different frame identifiers; their automatic cerebrum and
manual tumor annotations do not establish a common planning frame. Independent
native-coordinate checks passed. A bounded rigid-registration attempt worsened
the separate similarity diagnostic and remains unaccepted. No joint planner
case or training result is claimed. Official UCSF v5 access was also retried:
the provider reported missing server files and listing/transfer errors. No
directional diffusion or matching exam-specific gradients were downloaded.

**Current user priority (October 4): spatial RL and surgical decision learning.**
Interface development, packaging, the pending cache-sizing execution and the
older paired feature-unit patient run are deferred. The user wants a policy
conditioned on a new patient's available scan evidence and tool geometry, with
optional bounded adaptation/search, rather than further interface polish.
The immediate research direction is spatial observations, explicit separation
of observed estimates from hidden reference anatomy, and sequential opening /
removal decisions across varied anatomies. Existing clinical and data-provenance
limits remain in force; more simulator training cannot establish surgical safety.

The earlier 15+6-feature actor is annotation-assisted: exact supplied labels and
simulated cavity state affect both its features and candidate inventory. It has
no image encoder. Missing motor/language flags identify array presence; they do
not establish adequate functional evidence or compel abstention. Completed
procedural pretraining used zero human training patients. Unseen-patient,
scan-only and unfamiliar-tool generalization are unproved. The latest steering
replaces the proposed procedural training screen with real-patient learning.
Private reference truth must stay out of proposals as well as tensors; learned
inference and search must use the same permitted observation contract.

The new scan-conditioned spatial policy is implemented: a 24,331-parameter 3D
encoder scores tool rays and STOP, with a spatial value baseline. Its observation,
policy and coordinate suites pass 77 checks. Independent analytic review corrected
trajectory-length bias in the policy gradient; numerical-coordinate checks also
corrected loss of valid image-edge samples. The separate synthetic opening task
passes 16 owner and 19 independent checks. Exhaustive enumeration of its smallest
fixture gives a unique return of 0.503158 after two costly openings; one-step
greedy stops at zero. This verifies a sequential learning opportunity, not that
the learned policy has solved it. No procedural population screen will run under
the current real-data-only training instruction.

The deferred procedural spatial screen uses scan intensities, observed support/cavity, tool
geometry and procedure state. Private target and functional labels cannot enter
proposals, policy tensors or search lookahead. Its primary reward is explicitly
geometric: functional exposure remains unassessed. The small task removes
tip-intersected cells, distinct from native contained-cell execution. Native
integration is now implemented separately in the real native adapter described
above; vessel/function integration and held-out human generalization remain
open requirements. See `docs/spatial-policy-research.md`,
`docs/spatial-task.md` and their linked negative-result receipts.

The frozen BTC expansion acquired PAT22/PAT25 for population training and
PAT26/PAT27 for checkpoint selection: 72,375,710 new bytes, with source checksums
and finite image arrays verified. Native T1/annotation orientation reconciliation
has at most 0.000858 mm corner discrepancy. These are acquisition/integrity
results; brain envelopes and cortical access remain unreviewed. PAT29/PAT31 stay
unopened. Receipts are in `artifacts/btc-spatial-acquisition-v1/`; no new human
training or planning run has been executed from this batch.
All four cases now also have source-bound portable bundles with exact save/reopen
identity checks, prepared in 21.02 seconds from committed code. Native MRI and
annotation frames, missing modalities, and cohort roles are preserved. Peak
worker memory was 430 MB. Brain masks and cortical access remain unavailable;
preparation is not anatomical approval. Execution receipts are in
`artifacts/btc-spatial-preparation-v1/`.

The read-only inspection bridge and separate linked MRI/3D tool display were
committed after 100 bridge checks and 68 viewer checks, respectively. App host
integration is unfinished and preserved as working changes. Its 10 focused
synthetic checks do not establish an integrated build or native validation.
Known deferred issues include logical cancellation preceding actual worker
completion and mismatched support-acknowledgment conditions. The validated
packaged app below has not been replaced. The previous full Python suite's
12 failures have focused repairs, but a new full-suite pass is still pending.

**Interface direction updated by the user:** Electron + React + TypeScript is
the current Mac UI target. Further PySide/PyQt interface work has stopped. The
validated Qt prototype is preserved as an earlier experiment; Python planning,
learning, provenance and evaluation remain the shared backend. The new renderer
communicates through a narrow local sidecar interface. The first standalone
Electron package has passed real MRI, route comparison and native-save checks.
The full training package has passed actual updates, cancellation/resume and
restart checks. The refreshed native patient UI now passes blocked-route
preflight, exact-stroke training, replay/export, cancellation and full-app
restart/resume. Accessible replay controls were fixed after a native regression
and now keep the image, displayed step and quantities synchronized.

The Electron renderer adds separate, explicitly unreviewed brain-envelope
contours on the original MRI planes. Source/frame/mask checks and lazy loading
pass for both BTC cases: the no-CSF estimates exclude 214 PAT28 annotation voxels
and 538 PAT05 annotation voxels. These are overlap checks, not accuracy scores.
All 97 desktop checks and the production build pass. Native inspection now
passes on both cases, including source restoration, case switching and the
outside-annotation cursor jumps. A stale footer was caught and fixed; final
renderer source `75d7f69b…` uses the previous verified numerical engine. Working
anatomy and access gates stay unchanged. Screenshots and exact package identities
are in `artifacts/electron-structural-proposals-v1/`.

Seven registered motor/language prior proposals now persist in portable cases
with separate values and atlas coverage, exact T1 registration/T1c display source
bindings, and unchanged planning inputs. They remain view-only population
evidence requiring alignment review. Their Electron controls now display one
selected layer with separate atlas coverage, scalar/binary sampling and cursor
values. Source restoration, stale loads and case changes clear the layer.
The preceding 143 desktop checks and production build passed after removing an unsafe
source-image hash cache, correcting float32 atlas boundary sampling and adding
context-aware route-comparison guidance. Native inspection caught overlapping
unloaded welcome text; a regression test and shell correction now pass at both
1460- and 1050-point window widths. That notice-bearing renderer is
`34285015…` from `100865f`, with the unchanged verified `85a92eff…` engine. Its
source, complete engine payload and packaged upstream notices match the captured
bytes. Loaded-case guidance, real search, motor-prior inspection and source
restoration pass; screenshots and receipts are in `artifacts/dependency-notices-v1/`.
The preceding native renderer is `a6bb6b76…` from `346df23`, with 148 passing
desktop checks and a successful production build. Its new Center on annotations
action returns linked MRI slices to the source annotation center while preserving
prior/source mode, A/B routes, opacity and layout. Native checks pass at both
window widths; unloaded and MRI-only cases disable the control. PAT16/PAT20
main/no-CSF estimates, all four omission jumps and source restoration also pass.
All 72 source inputs, 2,809 unchanged engine entries and 292 unchanged notice
entries verify. `artifacts/electron-annotation-center-v1/` retains exact receipts
and screenshots. The engine remains `85a92eff…`; new experimental learning and
cache code are not in this package.

The preceding renderer is `bc63bc2e…` from `fe25f2a`, with 150 passing desktop
checks and a successful production build. Its fixed header retains the loaded
case identifier during evidence-panel scrolling. Actual UCSF → PAT16 → PAT20
switching passes at 1460- and 1050-point widths, with correct full accessible
labels and a centering regression check. All 73 desktop source inputs verify;
the engine and notices remain byte-identical. Root inspected the minimum-width
UCSF and wide PAT20 MRI screenshots. Native testing did not introduce an
extreme-length synthetic identifier. Exact receipts and images are in
`artifacts/electron-case-identity-v1/`.

The current native renderer is `834c7ceb…` from `c09ee50`, with 153 passing
desktop checks and a successful production build. Empty category/A/B selectors
now stay hidden until candidates exist. Native unloaded, pre-search UCSF and
blocked PAT20 states show their existing next-step guidance at both window
widths. Actual UCSF search produced 54 candidates; category switching and A/B
comparison work, with breadcrumb and centering preserved. Its three categories
were nonempty, so the empty-filter boundary is source-tested rather than claimed
as a native observation. All 73 source inputs, 2,809 engine entries and 292
notice entries verify. Root inspected pre-search and populated screenshots.
See `artifacts/electron-route-empty-state-v1/`; the engine remains unchanged.

Two notice-source completeness gaps and distribution signing remain open. An
independent actual GPU probe passes 487 cases, with the earlier failures retained.
The new complete engine and renderer passed native inspection. All seven layers,
covered zero, positive values, outside/incomplete support, minimum-width layouts,
save/reopen and historical checked replay were exercised. The preceding renderer
`762973e2…` from `c194a8b` repeats the corrected sampler and source-restoration
checks with engine `85a92eff…` from the verified `0b4e334` Python snapshot.
Later experimental learning changes are not in this app snapshot. Native
receipts and screenshots are in `artifacts/electron-priors-v1/`.

The first declared procedural-to-patient comparison stopped before adaptation
updates because equivalent typed and JSON world vectors compared differently.
Its raw records and failure are retained; it produced no validated comparison.
The implementation repair preserves every declared world field and adds a
public-case preflight before any training. The separately declared fresh v2
completed with original geometry, rewards, seeds and budgets unchanged. All six
patient learners changed actor weights and all 13 frozen candidates passed
independent native checking. SEARCH/GREEDY scored 245.24; scratch seeds scored
[245.24, 171.42, 245.16], frozen procedural initialization 139.62, and adapted
seeds [171.42, 171.42, 245.17]. This is a negative transfer result on one reused
development patient with zero human pretraining patients. Final/stress worlds
remain closed. The best sequence removes only 249 of 41,919 target mm³ under
hypothetical access; this is not a complete resection plan.

The separately declared actor feature-unit comparison completed from the
immutable `0bffeaa` archive after all 794 tests and the real-case preflight
passed. It compares raw inputs with fixed physical reference units while
preserving the old action inventory, rewards, critic, seeds and budgets. Other
heavy work was paused for timing comparability; ordinary desktop/OS load was not
controlled. All 23 frozen candidates passed native checking and a separate
artifact audit. Fixed units raised the procedural frozen-policy return from
139.62 to 245.24, matching SEARCH. All three scaled adapted runs selected their
unchanged initial checkpoint; they establish no additional adaptation gain.
Scratch seed differences were [0, 0, +0.08], and no arm exceeded SEARCH. The full
launcher took 534.18 seconds, including separately reported offline and audit
costs. Read-only diagnostics confirm reduced actor saturation and persistent
critic state aliasing, without proving clipping-induced learning harm. See
`artifacts/learning/procedural-native-feature-units-v1/RESULT.md` and its
separate independent audit and diagnostic records. This is one reused case,
two procedural families and zero human pretraining patients.

A separate post hoc geometry probe expands the proposal inventory while keeping
the native checker, source, tools and access assumptions fixed. Under the same
three-cut cap, it removes 1,113 target and 62 normal mm³, with 177 mm³ of partial
normal contact. Both native and independent artifact audits pass. Coverage is
still only 2.655%, and this new, restricted parallel-column model is not yet part
of the desktop planner or the frozen learning comparisons.

The proposal rule now has a separate source- and cavity-bound implementation.
Independent review caught changed arrays and live cavity edits that retained
cached identities; all eight original failures are preserved. Repairs now verify
source contents, committed history and all four cavity masks before emitting
proposals. All 66 focused checks and a separate 15-case review pass. This module
does not certify tool clearance or integrate with learning yet. In one actual
case observation, integrity/proposal checks took approximately 0.29–0.30 seconds
per call; cumulative process peak RSS was 1.16 GiB. Initial and post-cut proposal
inventories matched the frozen prototype. These are individual local timings,
not a latency distribution or isolated allocation measurement.

A separate RAW-only simulator adapter now supplies that inventory to the common
search/learner interface while retaining the native cutting engine. Twenty-nine
new analytic checks and 29 compatibility checks pass; the independent review's
19 cases are included in the new checks. Failed-reset integrity, preview-error
accounting and transient-cache defects were caught, repaired and preserved as
negative evidence. The first public preflight stopped before adapter construction
on a configuration-identity mismatch; two audits isolated truthful provenance
text as the only difference. The separately declared V2 completed five patient
episodes from immutable `6930417`: greedy return 1,098.77 and unchanged RAW
initialization 441.60, with both distinct histories independently certified.
A separate artifact audit and 25 adversarial checks reproduce all source-cell
counts and rewards. The run took 149.38 seconds and 1.314-GiB peak worker RSS.
Its 48.19-second initial policy panel exceeds the older 30-second allowance,
though it is not an exact generic-learner cost. No gradients or final/stress
worlds were used. Later decision features/logits were not retained in that
preflight; the completed bounded update pilot records them. The adapter remains outside the desktop,
and existing procedural checkpoints and feature-unit gates remain unchanged.
See `artifacts/preflight/native-axis-v2/RESULT.md`.

A supported optional decision observer now records the actual already-used
policy inputs, logits, value and action before execution. Fifty-four focused
compatibility checks and 30 independent checks pass; recording preserves exact
weights, Adam state, RNG and environment behavior. Its native journal binds to
served observations and distinguishes attempted, executed and returned steps.
A separately declared one-update seed-11 pilot completed from immutable
`f3a0591` after 53 focused runner checks and the 1,150-test integrated sweep.
One actual actor update used two complete optimization episodes; initial and
updated two-world selection both returned 441.60, correctly retaining the
initial checkpoint. Six complete histories have accepted certificates covering
three unique native histories. A separate 35-test artifact audit reconstructs
all 18 saved forwards and verifies all 12 deterministic choices without rerunning
patient simulation or gradients. Six stochastic choices are eligibility-checked,
not RNG-replayed. The full launcher took 221.14 seconds, online work 169.59
seconds, and peak worker RSS was 2.219 GiB. This verifies the update pipeline,
not a learning improvement. No cache or final/stress worlds were used.
Original receipts remain unchanged; lossless gzip and gzip-only report/audit
reproduction pass. See the actual report in
`artifacts/learning/native-axis-raw-update-pilot-v1/report-v1/RESULT.md` and
`artifacts/native-axis-pilot-result-audit/`.
A later saved-decision diagnostic pairs all six selection records into three
distinct states. Chosen actions are unchanged, while 8/17/12 rows change ordinal
rank, including STOP, and all saved values/logits change. No exact 15-feature
aliases occur among the 26/24/22 legal non-STOP rows on this path. Twenty-six
constructed-record checks and an independent arithmetic/hash audit pass; a
duplicate-decision-ID validation gap was fixed without changing actual results.
This performs no new model forward or simulation and does not explain the cause
of the unchanged score. See `artifacts/native-axis-decision-diagnostics-v2/`.
An isolated exact capsule-cover cache prototype passes 78 focused checks and
26 independent checks, with its actual invalidation failures preserved. Synthetic
reference/cached/reference outputs agree, including independent native histories.
The separately declared public four-phase probe then completed from immutable
`43857b5` in 158.29 seconds. Cold/warm caching took 22.896/23.050 seconds versus
21.182/21.670 seconds for the bracketing uncached phases. Both cached phases had
22,364 misses and zero hits at the declared 32-MiB/16,384-entry limits; no speedup
was demonstrated. All four scientific traces are byte-identical and all four
native audits pass. A separate artifact audit reconstructs five tissue/contact
masks at all four states and the 1,098.77 reward from source cells; 34 adversarial
checks pass. Process peak RSS was 2.133 GiB, distinct from cache payload size.
This negative fixed-trace result does not establish full-learning throughput.
Production geometry and the packaged engine stay unchanged. See
`artifacts/native-axis-cache-report-v2/` and
`artifacts/native-axis-cache-result-audit/`.

An isolated synthetic descriptor probe reproduces a known RAW action/state
alias: identical first reward 4.33 but best two-cut returns 39.43 and 39.68.
Access-relative coordinates and cavity/residual centroids distinguish that
pair. Eight owner checks and seven independent checks pass. This is nine tiny
probe transitions, no patient or policy execution and no production profile
change. Independent review retains a roughly 2-mm rotation-rounding failure at
the tangent-selection cutoff. Different masks can still share the summaries;
contact history, tool and remaining budget are omitted unless supplied separately.
The proposed summary is not a complete state or an established learning gain.
See `docs/synthetic-spatial-feature-probe.md` and both independent artifact sets.

The isolated V2.1 descriptor replaces automatic tangent-axis selection with an
explicit source/frame-bound reference and a declared engineering conditioning
check. The saved rotation case has 2.22e-16-mm error for a well-conditioned
reference; a near-parallel reference rejects in both frames. Independent review
found 14 initial constructor/mutation/overflow/complex-input failures, now fixed
with their original evidence preserved. Eleven owner and 30 independent checks
pass; 14 saved numeric comparisons are unchanged. This is algebra-only work
with no patient, simulator, policy or production-profile changes. Lossy state
summaries and the absence of a shared cross-patient reference convention remain
explicit. See `docs/synthetic-spatial-feature-v2.md`.

The frozen extraction procedure has now run on PAT16 and PAT20, with two
repetitions of both model variants per case. All eight child inferences completed
without failures or tuning; repeated masks and distance arrays were identical.
Independent array and visual QC passed, with 28 focused checks. No-CSF/main
estimates omit 2,045/19 of 45,400 PAT16 annotation voxels and 413/125 of 12,451
PAT20 voxels. These flags remain; no target voxels were inserted into the masks.
Working anatomy and portable source cases are unchanged. The estimates still
require anatomical review, and runtime version pins are not historical binary
attestation. Exact execution/QC records are linked from
`docs/brain-extraction-pat16-pat20-independent-review.md`.
Separate portable copies now include both r1 envelope proposals and exact report
provenance. An independent 11.92-second audit and 15 adversarial checks confirm
unchanged source/planning inputs, exact mask persistence and enforced review
gates; focused native app inspection now passes. Predicted distance arrays remain
external diagnostic files. See `docs/brain-extraction-pat16-pat20-bundles.md`.

## Verified starting state

- Repository initially contained only the three specifications and two commits.
  A bounded search found no existing implementation to preserve elsewhere.
- Local development: Apple M5, 16 GiB unified memory, macOS 26.6, Python 3.12.
  Hardware probe and architecture evidence are saved in `docs/`.
- An isolated environment now has Qt/PySide6, VTK, NiBabel, SciPy, PyTorch and
  testing/packaging tools installed. The first arm64 app bundle passed an actual
  frozen-runtime and MRI/3-D window smoke check on this Mac.
- Work is on `codex/patient-specific-planner`. Every new commit uses author
  `skamal23 <sayemkamal12@gmail.com>` and co-author
  `shafsthegoat <shafrir.p@gmail.com>`; commits are pushed incrementally.

## Executed vertical slices

- Immutable patient/provenance contracts, time-aware context filtering, physical
  NIfTI QC/import, preserved source annotations, case revision and JSON/NPZ
  roundtrip are implemented. Future context cannot alter planning seeds.
- UCSF-PDGM-0004 structural mirror bytes are pinned and checked; equivalence to
  official TCIA remains unverified. TCIA v5 fixes diffusion metadata relative to
  the specification's v4. The official transfer service failed during acquisition.
- Creator-source OpenNeuro BTC `sub-PAT28`, ds001226 v5.0.1, has 15 verified
  files including T1, fractional tumor annotation, AP DWI, PA references and
  gradients. Its full-head T1 does not provide a reviewed cortical access mask.
- A second creator-source BTC case, `sub-PAT05`, was selected by a recorded
  metadata-only rule before image acquisition. Seven pinned files total 17.86 MB;
  independent checks agree on all 11,437 threshold-derived annotation voxels.
  The registry has three development groups, two primary-source identities and
  zero final patients at that checkpoint. Unknown clinical timing remains
  excluded from planning.
- Two further creator-source cases, PAT16 and PAT20, were selected and committed
  from metadata before image access, then acquired with pinned source versions,
  lengths and annex checksums. Their structural files total 35.73 MB excluding
  shared release metadata. Independent source, native-frame, annotation and
  save/reopen checks pass: threshold 0.5 derives 45,400 and 12,451 target voxels.
  Full-head access gates remain enforced, with no reviewed brain masks or new
  diffusion. The registry now has five development representations, four verified
  primary-source identities and zero final groups; all 16 source-evidence checks
  and 26 cohort tests pass. This does not complete a five-UCSF or tract-aware
  pilot, and engineering QC is not expert anatomical approval.
- Seven licensed motor/language population maps load with source hashes and
  explicit registration-review contracts. Patient alignment remains unaccepted.
- Full-tip/shaft/swept-envelope route search retains feasible, dominated and
  rejected alternatives with failure evidence. The UCSF route run generated 54
  candidates, 31 geometrically feasible, 12 geometric Pareto. The separate
  checker agreed on all 54 under the same limited structural assumptions.
- The standalone Electron app loaded the real UCSF bundle and rendered linked
  physical MRI, annotation surfaces and two full-tool routes. Route generation
  took 1.16 seconds in one native run; the Mac save dialog preserved the routes,
  comparison and cursor. Its initial imaging/search bundle is 388.6 MB and has
  no Qt/VTK dependency. Accessible volume is distinct from removed volume.
- Actual patient policy-gradient updates, resumable checkpoints, scratch/search
  comparators and isolated world roles execute. Eighteen development runs and
  six budget/exploration follow-ons retained their source snapshots and failures.
  No RL advantage over search was demonstrated.
- Independent native-footprint checking **invalidated every non-STOP UCSF
  coarse-removal candidate**. The coarse simulator removed cells beyond the
  active tip's supported footprint. Those runs remain negative development
  evidence. Their coarse returns are not valid resection performance.
- The corrective native-grid engine and separate checker now agree on a real
  two-stroke sequence: 249 mm³ target and 17 mm³ normal tissue removed, with zero
  unsupported source-cell volume. Three patient-training seeds changed weights;
  one improved from STOP to SEARCH's score, two retained their initial policies.
  SEARCH and greedy search were faster. All nine frozen development candidates
  passed native checks. Missing functional evidence remains unknown.
- PPO at 32 and 128 actual optimizer updates failed to outperform SEARCH on the
  branching development fixture. Exact source snapshots, initial weights,
  transition counts and negative results are preserved.
- Shared pretraining, frozen inference and isolated adaptation now execute with
  explicit cohort exclusions and separate offline costs. The first two-group
  synthetic experiment used eight offline updates and eight online updates per
  seed. SEARCH scored 1.30, scratch [0, 0, 1.24], frozen 0 and adapted [0, 0, 0].
  All ten independent synthetic geometry checks passed; no final worlds opened.
  This is a negative analytic experiment, not a clinical population policy.
- A separately declared procedural-native comparison completed fresh pretraining
  and all six scratch/adapted arms after a retained implementation-only failed
  attempt. Offline training made 32 actual Adam updates across two nonpatient
  families; online runs made 10–14 updates within matched 30-second cooperative
  caps. All actors changed, though scratch seed 23 selected its initial policy.
  No learned arm exceeded SEARCH. All 13 candidates passed eight distinct native
  sequence audits, and a separate artifact audit recomputed scores from source
  cells. Total launcher time was 288.87 seconds, including 33.12 seconds of
  independent validation. These development results do not replace real
  patient-population evaluation; full evidence is retained under
  `artifacts/learning/procedural-native-to-ucsf-v2/`.
- The Electron training bridge executes real updates, resumable cancellation,
  independently checked replay and export. A 30-second patient-learning budget
  yielded the two-stroke sequence in 37.77 seconds total including setup/checks;
  a 5-second run retained STOP. Resume binds checkpoint bytes, source/runtime
  contract and original budgets; unsigned crash recovery is refused.
- Native UI testing exposed a selected-route mismatch: the factory discarded
  the selected entry/target in favor of parallel global-centroid proposals.
  Exact-ray diagnostics also found that none of the original 54 straight routes
  could cut their own corridor with the declared generic instruments. Merely
  substituting the two named native profiles did not fix those original rays.
  Two explicitly different, source-grid-aligned research tool/access candidates
  pass independent checks. Exact selection pass-through now freezes the complete
  entry, target, window and tool. Actionless routes perform zero updates. Original
  routes remain alongside two explicitly different native-action alternatives.
  A real UCSF bridge run made 32 updates, changed the actor and improved nominal
  selection return from 0 to 171.58: 174 target + 11 normal mm³ removed in one
  independently checked stroke. Export, save and restart/recheck passed. This is
  fixed stroke-versus-STOP refinement, not free-form route learning or a controlled
  efficacy benchmark; final worlds remained unopened.
- Raw BTC diagnostic tensor fitting ran on 101,311 voxels; bounded CSA tracking
  produced 176 unlabeled diagnostic paths. Uncorrected DWI is correctly refused
  for tract-aware planning. Correction and alignment are current work.
- Pinned SynthStrip main/no-CSF models ran locally on MPS in roughly 6–7 seconds
  per variant with identical outputs in four runs. No-CSF excludes 214
  source tumor-annotation voxels; main includes them all. Neither is reviewed
  brain/cortex or an accepted cortical-access mask. CPU failure is preserved.
- The same frozen extraction settings ran on preregistered PAT05 in 7.69/6.01
  seconds for no-CSF/main. Independent source/model/output hashes, native geometry
  and distance-map reconstruction passed. No-CSF excludes 538 of 11,437 source
  annotation voxels; main excludes none. Both remain unreviewed proposals, and
  annotation inclusion is not an accuracy metric.
- Structural proposals now save/reopen separately from working anatomy, with
  immutable source/model/run provenance and review-required status. Adding unused
  proposals leaves planning seeds unchanged. Explicit anatomy prohibitions take
  precedence over collection defaults; reviews cannot silently enlarge masks or
  relabel model estimates as observations. Native UI inspection confirms the
  BTC proposal inventory leaves route generation disabled.
- Isolated PyHySCO phantom testing reduced a known-distortion image error from
  9.68% to 0.40%, with 0.395-mm displacement error. The zero-distortion solver
  failure and unsupported BTC phase-encoding geometry remain explicit blockers;
  no patient correction or FSL execution was performed.
- A separate experimental physical-vector solver passes a bounded rotated-pair
  phantom and an independently certified identical-input branch. Image error
  falls from 9.68% to 1.07%, with 0.315-mm field error; coordinate-invariance
  failures in two earlier revisions are retained. All nonzero runs reach their
  40-iteration limit. Thirty-four focused checks and independent artifact/error
  verification passed. This does not promote any patient preprocessing gate.

## Validation record

The latest root integrated run passed **1,150 tests in 215.95 seconds** from
immutable `f3a0591`, with no skips and no changes or additions to any of the
2,904 baseline files. Fourteen warnings comprise ten intentional NumPy metadata
adversaries and four existing DIPY warnings. The preserved archive includes the
new observer/accounting, one-update pilot runner, cache prototype and portable
proposal audits. Its committed release baseline permits one patient update,
two optimization episodes and both complete selection panels, with no cache,
final worlds or stress worlds. Results remain separate from the launch record.
See `artifacts/validation/native-axis-pilot-public-v1/`.

The preceding integrated run passed **959 tests in 196.11 seconds** from
immutable `558b2e3`, with four existing DIPY warnings. All 2,669 baseline
files stayed byte-identical. This includes the axis provider/adapter, tiny RAW
learning, preflight/adversarial checks, notice collector and acquisition changes.
The earlier 914-pass/one-failure attempt is retained: its analytic orchestration
fixture incorrectly mixed the evolving registry with historical frozen inputs.
The test-only repair passed 63 focused checks and retains a real production
byte-drift rejection test. Both full attempts are in
`artifacts/validation/integrated-axis-v1/`. The new zero-gradient public
action-model preflight V1 stopped during configuration validation before
adapter construction, simulated cuts or gradients. Its source stayed unchanged. Two actual configuration-only audits found
identical physical inputs, with only the tissue-support provenance text
differing between factories. The fresh V2 declaration binds the direct helper
without rewriting either provenance description or bypassing identities.
Its immutable `6930417` launch passed 77 focused checks in 23.07 seconds; all
2,710 baseline files remained unchanged. Five subsequent public episodes and
two unique native audits completed; a separate artifact audit passed 25 tamper
checks. All 46 captured source files and original/execution case bundles still
match. Five histories compress losslessly from 39,286,826 to 665,699 bytes;
gzip-only report regeneration is identical, and raw originals remain local.
Executed versus returned transition accounting separately passes 19 analytic
checks, including one real tiny gradient update and interrupted native commits.

The preceding complete run passed **794 tests in 176.14 seconds**, with four
existing DIPY basis-deprecation warnings, from an immutable archive of `0bffeaa`.
Its bytes stayed unchanged throughout testing. The real public-case preflight
also passed both policy profiles and all three seeds with zero gradients before
registered feature-unit execution. Exact identities and results are in
`artifacts/validation/native-feature-units-runner-v1/`. The preceding
743-pass/one-failure sweep is retained: its orchestration fixture needed to
isolate a newly added earlier preflight, while production rejection was correct.
The test-only repair passed focused checks before the earlier 749-pass sweep and
this latest complete validation. The newer standalone provider and axis adapter have separate independent
reviews and are included in the latest integrated attempt.
Structural adversarial checks found
and fixed metadata precedence, mask-enlargement and provenance problems;
their focused checks passed before each incremental commit. Native
adversaries also caught temporal shaft borrowing, mutable preview descriptors
and checkpoint-resume tampering. Test counts describe their recorded snapshots.
The independent native checker improved from 17.638 to 4.310 seconds on the same
saved patient history, with identical geometric certificates and 91 regression
tests. These are local single-run timings, not a latency distribution.
Native initial-geometry reuse lowered median profiled reset time from 0.462 to
0.0259 seconds without changing captured outputs. Three further 30-second runs
completed 14/11/11 updates versus 7/6/5 previously; only seed 47 improved its
selected score. SEARCH remained best. Renderer hydration decreased from a
352-ms median to 134 ms in a warm-cache V8 benchmark; this is not GPU latency.
See `EXPERIMENT_LOG.md` for executed experiments, including negative results.
The Qt prototype passed its initial standalone runtime and GUI checks. Its next
build was stopped following the user's Electron decision. The React/Electron
package passed its own imaging/search checks and 31 main-process boundary and
lifecycle tests. The current snapshot passes 38 main-process, 74 renderer and
31 viewer checks covering source
hydration, evidence status, A/B identity, independently accepted removal overlays
and population-prior sampling. The complete
training bundle has its own source-bound verification receipts. Qt results do
not certify Electron. No second-Mac test is claimed.

The previously verified refinement shell (renderer `5fe94739…`, Python `377284f6…`)
recovers a six-update cancelled checkpoint after app restart and adds 26 updates
under the original contract. Accessibility decrement/increment now changes the
actual checked replay between 0/0/41,919 and 174/11/41,745 target-removed,
normal-removed and residual-target mm³. Matching screenshots document the fix.
Display-normal averaging improves shading without moving any source triangles;
a separate local probe adds 75.5 ms of worker preparation to 332.0 ms of source
mesh preparation. This is a display tradeoff, not a GPU speedup claim.

## New backend inspection boundary

A separate read-only facade now returns the complete initial expanded-axis
proposal and native-preview inventory with exact RAS geometry, full tool and
source bindings, and explicit unknowns. It requires separate neighboring-column
and estimated-support acknowledgments. It performs no transitions or policy
updates and grants no removal or candidate authority. Sixteen owner and 33
independent synthetic checks pass. Review reproduced and repaired stale cached
source identity after callbacks and a valid LPS normalization roundoff refusal;
both failures remain preserved. Existing selected-route readiness is unchanged.
No public inspection or Electron integration is claimed for this slice. See
`docs/native-axis-inspection.md` and its independent review artifacts.

## Completed saved-cache diagnostic

The released saved-only reconstruction completed once from immutable `b0f5635`:
22,364 ordered calls contain 8,812 distinct argument keys and 13,552 first-pass
repeats. A separate audit reproduces every byte/order and reuse/LRU statistic;
33 audit tests pass. The full key set is below the existing 16,384-entry cap,
so increasing that count alone cannot help this fixed trace. Full cover payload
sizes and byte-weighted reuse remain unknown; no larger memory capacity or
speedup is established. External child wall time was 1.108 seconds, with all
16 archived inputs unchanged. Cooperative limits passed, but exact peak RSS
was not retained. No geometry, patient, simulation, policy or gradient ran.
See `artifacts/validation/native-cache-query-keys-v1/RESULT.md` and
`artifacts/native-cache-query-result-audit-v1/`.

## Expanded-axis policy input profiles

The scratch learner now accepts explicit RAW or FEATURE_UNITS profiles through
a closed axis observation contract. Accounting version 2 distinguishes raw
simulator features from the policy's actual transformed inputs; physical models,
registered divisors and six critic inputs remain unchanged. One hundred owner
regression tests and 25 independent checks pass, including real tiny updates and
profile/transfer rejection before optimization. The independent tiny update
changed actor weights but tied initial selection return; this is integration
evidence, not learning improvement. Original timing-assertion and test setup
failures remain retained. Full integrated regression and a new frozen paired
public declaration remain prerequisites. No new public training occurred.
See `docs/native-axis-input-profiles.md` and the two focused review directories.

## Current integrated regression attempt

The immutable `523a72b` attempt completed with **1,470 passing tests and 12
failures in 243.45 seconds**, plus 14 warnings. All baseline files remained
unchanged, with no added source files. Ten diagnostic CLI tests inherited the
full pytest process's earlier peak memory and correctly hit the production
512-MiB guard before reaching their intended assertions. Two historical pilot
auditor tests encountered the newly added axis contract field. Scoped test
isolation and explicit auditor contract-version repairs are in progress; public
experiments and the next bundled engine remain gated. Original logs and source
manifests are retained in `artifacts/validation/axis-profile-integrated-v1/attempt-01/`.
This failed attempt does not replace the preceding 1,150-test passing snapshot.

## Completion gates still open

Native-resolution geometric simulation now works under explicit hypothetical
access assumptions; reviewed anatomy and surgical validity remain open.
Corrected/accepted functional reconstruction, fully integrated comparison and
refinement interaction, broader cohort benchmark and release remain incomplete.
Frozen/adapted initialization has been tested on analytic and procedural-native
fixtures, including transfer to one reused patient development simulation;
patient-population training and external evaluation remain open. Developer ID
signing/notarization and independent physical-Mac
validation remain open. No locked final cohort has been opened. Clinical deficit
probabilities remain unavailable; simulator success cannot establish clinical
readiness. Current evidence supports a local research prototype only.

No cloud infrastructure or billing changes have been made. Source imaging stays
outside Git. Numerical arrays and checkpoints are excluded; compact experiment
records and source snapshots preserve the executed development evidence.
