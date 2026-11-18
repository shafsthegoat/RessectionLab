# Current status and working entry points

Repository upkeep follows the [maintenance policy](../CONTRIBUTING.md): keep current-use material,
retire obsolete code/docs/artifacts, and archive useful history outside the tracked tree.

As of November 18, 2026, the maintained application is Electron/React/TypeScript
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
The inspection grid used 0.3125 mm spacing. Source MRI resolution still limits
the anatomy, and two short dural segments show stepping under close zoom.
See the [scan-to-rehearsal workflow](desktop-workflow.md#from-patient-scans-to-operative-steps).
Plan & compare binds an explicitly selected lesion to its source and revision.
Hidden desktop runs exercised supplied UCSF glioma labels and a meningioma
model candidate through geometric comparison, portable save and cold reopen.
The migrated CTA opens for source and label inspection. The migrated EDEN
study keeps eight synthetic targets separate from its healthy anatomy.
[Use the native workbench](patient-anatomy-workbench.md#prepare-a-patient-study) ·
[Reproduce the desktop checks](fresh-patient-reproduction.md#desktop-workflow-and-smoke-test) ·
[Resume the versioned migration](data-rebuild.md#native-study-migration).

## Acquired scan-to-procedure demonstration

The selected development case is preoperative **UCSF-PDGM-0004**, revision
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
| Geometric playback | BITE-09 contributed a gadolinium T1 and source tumor mask. SIAM DS715_NODA fold 1 supported illustrative flap, dura, retraction and aspiration. The 0.3125 mm display grid sampled MRI acquired at 1 × 0.547 × 0.547 mm and model output from 0.75 mm inference. | The sweep removed 98.4 mm³ of local display labels. Whole-lesion removal and physical response remain unassessed. |
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
