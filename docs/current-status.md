# Current status and working entry points

Repository upkeep follows the [maintenance policy](../CONTRIBUTING.md): keep current-use material,
retire obsolete code/docs/artifacts, and archive useful history outside the tracked tree.

As of November 21, 2026, the maintained application is Electron/React/TypeScript
with the Python engine and optional compiled C++ backend. This tree combines
the November 16 structure-aware harm and canonical endpoint work (`b80d2e9e`,
PR #20), repository cleanup and archive restoration support, and the completed
[EDEN2020 segmented synthetic cohort](eden2020-healthy-synthetic.md).

The EDEN collection contains 15 segmented healthy baselines and 120 organic
tumor scenarios, all DEVELOPMENT. Native checks cover all 135 packages; actual
hidden packaged-app checks cover representative synthetic and healthy cases.
Prepared datasets, model assets and detailed receipts remain on Shafrir's
optional personal HDD and are unavailable to collaborators through a clone.
The [storage ledger](local-storage-ledger.md) records discovery and access limits.

Native patient studies accept scans together in Cases. Prepare anatomy
uses the available modalities and records an outcome for each stage.
The patient reconstruction combines aligned sources while preserving native
slices for review. A selected geometric approach can now be replayed from
bone flap through modeled aspiration wherever corresponding tissue exists.
The acquired BITE-09 MRI and its supplied tumor mask exercised all four stages
at 1440 × 900 and 1050 × 720. A fresh app process reopened the saved route.
The reviewed BITE-09 demonstration removes 39 source-checked peripheral components
across seven SIAM estimates. The supplied tumor, vascular estimates and all route-local
label values match the original study. A representative sheet within the estimated
dural band illustrates opening. The 0.3125 mm display grid samples MRI acquired at
1 × 0.547 × 0.547 mm and model inference at 0.75 mm.
See the [scan-to-rehearsal workflow](desktop-workflow.md#from-patient-scans-to-operative-steps).
Plan & compare binds an explicitly selected lesion to its source and revision.
Hidden desktop runs exercised supplied UCSF glioma labels and a meningioma
model candidate through geometric comparison, portable save and cold reopen.
The migrated CTA opens for source and label inspection. The migrated EDEN
study keeps eight synthetic targets separate from its healthy anatomy.
[Use the native workbench](patient-anatomy-workbench.md#prepare-a-patient-study) ·
[Reproduce the desktop checks](fresh-patient-reproduction.md#desktop-workflow-and-smoke-test) ·
[Resume the versioned migration](data-rebuild.md#native-study-migration).

## Interactive operative practice

A verified generated checkpoint can start a separate editable episode. The child retains its source, target, physical model, action prefix and original target volume. In Rehearse, mouse or keyboard input proposes an instrument pose. Place, move, activate, apply and withdraw send checked actions to the engine. A contact-limited advance leaves the instrument at its accepted endpoint. Saving and reopening restores the independent episode. [Practice from an accepted state](desktop-workflow.md#practice-from-an-accepted-engine-state)

The BITE-09 geometric microscope exposes the registered gray-matter estimate through a 14.03 mm dural window. It reuses 4,152 native surface vertices at their planning positions. The existing source conditioning displaced vertices by at most 0.6993 mm. Both original target components survived at the geometric endpoint. The cavity-facing raw-to-conditioned p95 distance was 0.144 mm. The raw mesh had three edge-connected shells and the conditioned mesh had five, so exact display-mesh topology remains a limitation. In the distinct UCSF-PDGM-0004 episode, stationary brain surfaces carry that study's verified MRI signal throughout recorded removal and withdrawal. The coarse brain envelope limits the visible detail. TopCoW-014 retains its own named vascular classes. Acquired episodes remain available for inspection under their existing permissions. [Acquired comparisons and numerical receipts](local-storage-ledger.md)

On the generated vascular and tissue control, four held cells reached 0.692 mm peak displacement, falling to 0.066 mm one modeled second after release. Two modeled wound ports emitted 5.979 mm³ before cumulative bipolar contact dose sealed them. Irrigation added 5 mm³ and suction removed 5 mm³. A fresh engine process reproduced the committed history and terminal state. These measurements describe the declared control. Held-out tension RMSE was 0.590 mN against 0.485 mN for zero force. Both frozen 16-to-20 mm stress-stability gates still fail. [Donor benchmark](continuum-donor-transfer.md) · [Stability results](continuum-domain-sensitivity.md). Functional risk on acquired studies remains unknown.

On the M2 Pro with 16 GiB at 1440 × 900, a continuous cranial jog produced frames at 20.0/21.3 ms median/p95. The 14 committed manual inputs reached an engine response in 44.3/58.4 ms and a completed frame with the matching state in 79.9/99.3 ms. In the separate recording, the first captured compositor frame measured 117.0/150.23 ms. Three fresh processes restored the saved editable child in 5.49/5.57 s. On the tissue and wound fixture, one-second modeled strokes and withdrawals exceeded the 150 ms target. Reusing unchanged tissue-neighbor bonds reduced input-to-visible p95 from 417.1 to 329.6 ms. All 33 previously captured actions retained their accepted state identities. [Private recordings and benchmark receipts](local-storage-ledger.md)

## Continuous operative rehearsal

The previous milestone established four-channel MRI preparation, revision-bound
playback and a retained acquired refusal. Its films showed that native anatomy
was difficult to read and the recording held one image between action boundaries.
The selected four-sequence route lacked an intact-state decision, and the
generated opening left debris. The desktop now exposes the following outcomes.
[Workflow and reproduction](desktop-workflow.md#reproduce-continuous-rehearsal) · [Private evidence index](local-storage-ledger.md).

| Starting evidence | What the desktop demonstrates | Evidence boundary |
| --- | --- | --- |
| UCSF-PDGM-0004, preprocessed T1CE/T1/T2/FLAIR, supplied enhancing target (label 4), MONAI BraTS 0.5.4 | Three entry windows measured 16.32, 26.36 and 59.15 mm. The shortest left-superior corridor contains 151 mm³ of sampled non-target tissue. Its revision, target, route and preparation bind to an intact STOP that reopens with both tools outside, zero actions and all 15,672 mm³ of engine-grid target remaining. | No physical treatment candidate was evaluated on this revision. No reviewed acquired case qualified for a full procedure. [Admission requirements](supratentorial-procedure-policy.md) |
| Original UCSF-0004 recording, separate revision and trajectory | Over 43.2 simulated seconds, 53 actions removed 66 mm³ of mapped non-target tissue and zero enhancing target. Purple identifies the supplied target. Magenta on the slices identifies a vesselFM estimate. | The historical refusal measures a different preparation. Vascular clearance and postoperative function remain unknown. |
| Generated post-scalp square shell | The planner compares immediate STOP with complete access, treatment, suction and exit. Its accepted 200-action branch charges 285 mm³ bone and 49 mm³ membrane, treats 27/27 mm³ target, clears 236 mm³ debris and withdraws both tools. It satisfies the frozen canonical endpoint. | The original 119-action alternative still fails with 236 mm³ retained debris. Scalp exposure cost is outside this starting state. |
| BITE-09 MRI, supplied tumor and SIAM DS715_NODA fold 1 estimates | The whole-head view links to native slices. Operative Steps offers raw or conditioned surfaces, a registered exterior and an optional teaching cutaway. The original spiral contacts 3,224 display cells (98.39 mm³). Overlapping passes contact 4,741 (144.68 mm³) and preserve both original residual components. | Opening, retraction and aspiration here are geometric illustrations. Most tissue boundaries come from an unreviewed estimate. |
| UCSF-0005 same-patient acquisition addition | Adding T1 and FLAIR to T1CE and T2 preserved 35 masks and marked three routes stale. At RAS (−97, 113, 85) mm the supplied enhancing label is positive at T1CE voxel (97, 126, 85). Each other acquisition explains why registration review is required there. | A newly listed scan need not change a mask. Source grid, coverage and provenance remain available after reopening. [Source comparison](fresh-patient-reproduction.md) |
| TopCoW-001, a separate patient's CTA and supplied masks | The superior sagittal sinus has one 26-neighbor component and ten face-connected components. The geometry retains the sampled diagonal links. | Empty anterior communicating artery masks leave presence unknown. This CTA supplies no evidence for UCSF. [Vascular review](native-vascular-geometry.md) |

**Recorded motion.** Live playback and MP4 export seek accepted observations at
their recorded times. The completed shell has 862 observations, 462 inside
actions, with 309 interior pose changes and 25 interior removal changes. An
attempted exchange with an inserted tool preserves the rejected state. Backward
seeks reproduce the earlier tool, tissue and cumulative ledger.

**Display accuracy and cost.** On the curved generated fixture, mean cap-normal
error went from 7.11° to 6.47°. Mean tangent-plane error increased from 0.069
to 0.107 mm. Groove depth remained underestimated (2.96 mm displayed, 3.48 mm
analytic). All 87 branch samples, the 1 mm membrane, both target components
and 1,377 coincident material-interface vertices were preserved. On an Apple
M2 Pro with 16 GiB, a 15,180-triangle CPU benchmark went from 77.77 to 11.31 ms
for a full rebuild and 52.59 to 13.99 ms for a local update. The curved case
settled in 167.26 ms against a 166.73 ms target with 50 ms sampling. A repeat on the integrated build took
111 ms for the curved case, 1,403 ms for the square and 181 ms for the acquired case. These
measurements assess the display. Anatomical and biomechanical validity require
independent evidence.

**Held-out planning.** Six generated comparisons used a prospectively committed
freeze and the existing seed-17 actor. On both unseen 22 mm³ targets in the
known square shell, the actor and one-ply search chose the central route,
treated 14 mm³ and retained 8 mm³. Each run paid for the same opening, cleared
236 mm³ debris, recorded 1 mm³ of overlapping neural/motor/CST removal, stopped
and withdrew both tools. Opening, cleanup and exit used shared scripted skills.
Across four post-access layouts, the actor treated 459 mm³ versus 297 mm³ for
search, using 50 versus 23 physical requests. On the neural-overlap case it
removed more neural tissue (54 versus 27 mm³). These results do not establish
an overall advantage. Vascular and observed functional outcomes were unavailable.
A stable-source repetition reproduced all 12 trajectories, including refusals and
endpoints. Neither method was tested on a patient.

The annotated CTA comparison preserves the supplied right M2 branches as
separately inspectable layers. All five vessel inference alternatives failed
selection, and the highest-ranked diagnostic alternative failed verification.
The default remains Gaussian patch blending at a 0.50 threshold.
[Measurements and native viewing instructions](data-rebuild.md#annotated-cta-vessel-comparison).

Next work requires independently reviewed acquired operative compartments,
positive and negative coverage, task permissions and swept access and exit
evidence with corresponding admission and refusal controls. Independent geometry
and functional references, source-resolution fold recovery and interactive
frame-time tails remain open.

## Fixed-target procedure comparison

UCSF-PDGM-0004 supports a post-exposure, source-bound comparison on a
supplied enhancing tumor. The prospective objective was 100 mm³ of owned
target removal followed by withdrawal, with the same 160 mm³ non-target
cap and 150 s aspiration budget per arm. A deterministic geometric search
selected two corridors with entries 31.8 mm apart. Each started from the
same zero-action state and used five 15 s scripted CUSA applications.
Neither trajectory advanced to the end of its requested path.

| Engine measure | Short access | Alternate entry |
| --- | ---: | ---: |
| Requested length / recorded tip advance, mm | 15.5 / 10.20 | 24.0 / 10.63 |
| Owned target removed, mm³ | 103 | 116 |
| Whole original annotation remaining, mm³ | 15,569 | 15,556 |
| Non-target brain removed, mm³ | 46 | 33 |
| Elapsed simulated time, s | 77.1 | 77.1 |
| Goal reached and withdrawn | Yes | Yes |

Both arms assumed scalp, skull and dural exposure. Zero removed cells in
mapped vessels and named exclusions describes the available masks. Dedicated
angiography, skull-opening cost and patient-specific functional outcomes
remain unknown. Less than 1% of the original 15,672 mm³ annotation was
removed on either route. The frozen patient CUSA policy requires a different
prepared starting state, action space and budget.

The standardized native viewer comparison favors BITE-09 for cortical folds
with a supplied tumor, TopCoW-014 for arteries and major venous channels,
CADS-465 for CT skull, and IXI265 for deep MRI structures. Meningioma
sub-06 retains a conflict at the MRI envelope and CT skull boundary.
Among reviewed holdings, no accepted glioma case combines tumor MRI, skull
CT and dedicated angiography in a compatible operative state.
[Viewing and comparison](desktop-workflow.md#compare-two-recorded-procedures-on-one-target) ·
[Studies and recordings](local-storage-ledger.md).

## Acquired scan-to-procedure demonstration

IXI077 joins acquired T1 MRI and a partial-coverage MRA to a recorded
post-exposure ultrasonic aspirator (CUSA) rehearsal. The native plan and
the recording identify the same 30 mm approach and synthetic target. A
fixed script configured and placed the instrument, applied aspiration, then
withdrew it. The 10 actions span 211.1 simulated seconds. The engine
removed 27 of 282 target cells (28.48 mm³) and 314 other tissue cells
(331.17 mm³), leaving 255 target cells (268.94 mm³). This episode is a
baseline for testing whether a different action sequence can reduce the
cost to surrounding tissue.

The desktop export captured 1,075 engine observations. An independent
check compared the painted source cells and instrument poses at all 11
boundaries and 42 intermediate observations with the recorded actions.
Separate fresh processes reopened the selected plan, the mid-procedure save
at 42 removed cells and the endpoint at 341 removed cells.

The scalp, bone and dural opening are assumed. The MRA registration
supports visual inspection. Its uncertainty and the partial slab leave
vascular clearance unknown. The evaluated frozen policy has a generated
case contract. The IXI077 episode uses the recorded script.
[Watch or reproduce the episode](desktop-workflow.md#replay-the-ixi077-t1-and-mra-case).

The earlier demonstration at `b05a5534` used preoperative **UCSF-PDGM-0004**, revision
`557b567cecdc3b109cd86873cc74c30c957b35876532a63f882eb8cca62c5844`.
Four preprocessed structural MRI volumes and supplied tumor labels are available.
The four-channel preparation produced revision
`4ffe2dc58f994485f092da208be175dad432f0f1372881448dc33ccad838cd34`.
That portable study holds all four acquisitions, the supplied target, three
additional model estimates and the selected geometric window together.
It cold reopened in the engine at boundary zero. Loading the historical
preparation against this newer revision was refused. The original revision
reproduced all 54 recorded boundaries, including a fresh-process midpoint
at action 26 with 46 mm³ of mapped non-target removal.
The selected target is `enhancing_target` (label 4). Core and FLAIR signal
remain separate. No paired CT, dedicated angiography or patient-specific motor
and second functional maps qualify this case.

| Acceptance item | Demonstrated result | Remaining boundary |
| --- | --- | --- |
| MRI preparation | MONAI BraTS 0.5.4 produced 14,023 enhancing, 22,561 core and 37,595 whole-abnormality voxels. All three estimates reopened alongside the supplied labels. Missing FLAIR, cropped T2 and unreviewed shifted T1 each withheld inference. | The masks are nested. The case overlaps BraTS2021. Membership in the checkpoint's training set is unresolved. |
| Incremental same-patient evidence | On separate meningioma sub-13, adding CT revealed 341,425.75 mm³ of density-labelled tissue. The two CT reconstructions disagreed at 699,679 thresholded voxels. Deliberately mismatched landmarks had residuals of 42.30, 42.30 and 41.40 mm. Refusal preserved the revision. | The proposal includes teeth and extracranial bone. The CT reconstructions are correlated. Cross-frame quantitative fusion remains withheld. Automatic local source precedence currently covers bone. Other structure disagreements remain available for review. |
| Distinct entry windows | The left-superior, posterior-superior and anterior-superior routes measured 16.32, 26.36 and 59.15 mm. Their capsule samples contained 151, 213 and 559 mm³ of estimated non-target brain and 44, 94 and 103 mm³ of target, respectively. The shortest window was selected for inspection and cold reopened. | These geometric intersections do not establish vascular clearance, injury or treatment. |
| Geometric playback | BITE-09 contributed a gadolinium T1 and source tumor mask. SIAM DS715_NODA fold 1 supported illustrative flap, dura, retraction and aspiration. The 0.3125 mm display grid sampled MRI acquired at 1 × 0.547 × 0.547 mm and model output from 0.75 mm inference. | The original spiral removed 98.39 mm³ of local display labels. The overlapping sweep removed 144.68 mm³. Whole-lesion removal and physical response remain unassessed. |
| Acquired engine recording | The separate UCSF preparation on `prepared-task-1` replayed 53 actions over 43.2 simulated seconds. It removed 66 mm³ of mapped non-target tissue and 0 mm³ of target. Both instruments withdrew. All 15,672 mm³ of the original engine-grid target remained. | The comparison stopped because no tested two-tool continuation was admissible. The newly inspected left-superior window has no executed trajectory. |
| Controlled treatment and refusal | A generated post-access superior route treated 54 of 108 mm³ and withdrew. The inferior/posterior attempt hit protected clearance. The rejected preview retained its state. | This control starts after access. Acquired exterior-to-target treatment remains open. |
| Opening through treatment | A separate generated, post-scalp alternative replayed 119 actions. Bone and membrane opening cost 285 mm³ bone and 49 mm³ other tissue. It removed 27/27 mm³ of target and mechanically withdrew. The hidden desktop verified all 120 boundaries and matched all 42 states in the cutaway-off film. | The current planner recommended STOP. The alternative retained 236 mm³ of debris and failed the canonical withdrawal endpoint. Scalp exposure cost was unknown. |
| Vascular consequences | On a generated arterial-occlusion control, the shared evaluator recorded 2.319 supply-deficit seconds and 0.244 drainage-deficit seconds with zero bleeding. Venous occlusion reversed the values. | The reward requires a declared generated graph and regional coverage. The acquired preparations lack qualified vascular and functional outcome evidence. |
| Matched RL and search | On previously evaluated generated parent 200, the frozen seed-17 actor treated 243/243 mm³ in 14 physical calls. One-ply search treated 27/243 in eight calls with two refusals. Both stopped and withdrew. Each had 56 calls and 25 s available and saw the same 28 features and legal-action mask. | This single result establishes no patient-level advantage. Acquired policy inference was refused. |

Full acquired completion remains **0**. It requires an independently reviewed
procedure contract with intracranial coverage, brainstem and cerebellar exclusion,
permitted entry and compartment, treatment and work domains, operative
prerequisites and swept instrument access. The current supratentorial policy
admits generated controls. Adding labels alone cannot lift its acquired-source
qualification hold. Postoperative deficits, perfusion and outcome probabilities
remain unvalidated. The source build passed 277 integrated Python tests
and 1,460 desktop tests. A forward-scrub repair also passed 64 stage and
replay tests and 60 integration checks. [Demonstration files](local-storage-ledger.md) ·
[Reproduction commands](desktop-workflow.md#reproduce-the-acquired-replay-and-generated-comparison).

The November 18 evidence-aware planning milestone adds an opt-in cortical
access assessment and named preservation decisions. The matched generated
overlap control retained the same treatment and residual with less CST
membership removal. The critical-residual control recorded a supported
boundary and checked withdrawal. [Results and reproduction](approach-comparison-checkpoint.md#evidence-aware-planning-milestone).
The subsequent native integration displays source-bound comparisons for
UCSF-PDGM and EDEN, with explicit reassessment and unavailable procedural
outcomes. [Native glioma evidence](approach-comparison-checkpoint.md#native-glioma-evidence-integration).

| Need | Entry point |
| --- | --- |
| Objective, capability boundaries and next dependencies | [Decision-system objective](decision-system-objective.md), [progress and evidence](decision-system-progress.md), [procedure policy](supratentorial-procedure-policy.md) |
| Current architecture and backend limits | [Desktop architecture/workflow](desktop-workflow.md), [C++ delivery](cpp-acceleration.md), [packaging](electron-packaging.md) |
| Native comparison, generated procedure controls and matched baseline | [Verified results and reproduction](approach-comparison-checkpoint.md) |
| Build and ordinary verification | [README development commands](../README.md#local-development); `pnpm --dir desktop test`, `pnpm --dir desktop build`; focused Python tests for the touched subsystem |
| Independent replay/monitor tooling | [Maintained rehearsal support](../scripts/rehearsal_support/README.md) |
| Source collections and data-root workflow | [Local HDD ledger and access limits](local-storage-ledger.md), [EDEN segmented cohort](eden2020-healthy-synthetic.md), [data rebuild](data-rebuild.md); ignored `testcases/README.md` and HDD `catalog/START-HERE.md` |
| Experimental features | [Neighboring paths integration contract](neighboring-paths-panel.md), [continuum](voxel-continuum.md), [research planning](research-planning.md), [interactive approaches](interactive-patient-planning.md) |
| Historical steering, requirements and negative results | [Master plan](../MASTER_PLAN.md), [status narratives](../PROJECT_STATUS.md), [handoff](../IMPLEMENTING_AGENT_HANDOFF.md), [execution ledger](REAL_OBSERVATION_EXECUTION_LEDGER.md) |

SurgicalEngine remains authoritative. Source ownership/admission, anatomical
exclusions, cumulative harm, vascular consequences, STOP/withdrawal, residual
target and replay/save compatibility remain required across both backends.
Historical disabled planners and the optional Qt prototype remain available;
`app/reslice.py` is also used by current registration. Do not reopen guarded
planning/training paths while reproducing earlier work.

The [cleanup execution record](cleanup-execution-20261116.md) explains the verified
historical archives, checkout reduction, restoration and remaining KEEP decisions.
The [earlier support pass](cleanup-20261116.md) remains dated context. Neither
replaces the research goal.
Earlier status documents retain their dated narratives and requirements.
