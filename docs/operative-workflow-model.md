# Operative workflow model for glioma research rehearsal

Research date: 2026-10-16. Scope: adult supratentorial glioma, specialist research review. Companion contract: [operative-workflow-v1.json](../manifests/operative-workflow-v1.json). Patient evidence and acquisition priorities belong to [patient-corpus-plan.md](patient-corpus-plan.md).

This is a source-linked simulation specification, not a surgical protocol, clinical validation, or a claim that every listed phase is implemented. It follows the supplied `RessectionLab_POC_First_Codex_Supergoal.md`: the first executable gate remains a bounded **post-exposure episode**. Researching the full operation must not defer that gate. The separately mentioned operative attachment was unavailable; no claim is made to have inspected it. The existing [research dossier](glioma-operative-research-dossier.md) is background, with primary sources rechecked for this document.

## What the clinical evidence changes

The operative objective must retain function and an explicit residual target definition. The current EANS–EANO guideline distinguishes tumor contexts and treats postoperative MRI as the usual extent-of-resection assessment; it does not supply a universal local removal policy. EANO guidance places avoidance of permanent neurological deficit ahead of maximizing removal. A segmentation therefore cannot define a safe resection boundary or a survival reward. [S1, S2]

The original Schupper operation describes image-guided planning and positioning, scalp/bone/dural exposure, awake language mapping as a selected adjunct, exoscopic visualization, debulking, fluorescence assessment, hemostasis, then layered closure. Its edited transcript is an example of sequence, not a continuous action recording or universal method. [S3]

Hebb's original technique report distinguishes ultrasonic fragmentation/aspiration from bipolar treatment of identified feeding vessels, and explains how bleeding obscures anatomy. The described subpial strategy uses a particular anatomical interface; it does not justify assigning every diffuse glioma a capsule. Morshed's reoperation series describes scarred dura, adherent veins, and altered closure needs. Prior surgery must therefore change the scenario rather than merely reuse an intact exposure. [S4, S5]

Serial intraoperative MRI documented regional deformation during surgery. Clinical tool-tracking research also reports limitations from brain shift and interrupted tracking visibility. A historical navigation image, a current observation, an estimated transform, and simulated physical truth are distinct. Touching a region with an instrument is not itself proof that the region was removed. [S6, S7]

RESECT's full methods describe tracked ultrasound probes with sterile covers,
initial imaging often before dural opening, and saline in the cavity for later
acquisitions. Patient positioning helped retain that fluid. An ultrasound
information action therefore needs field preparation, probe/tracking status and
an acquisition phase; it cannot be modeled as a universally available perfect
scan. These are reported cohort practices, not confirmed states or timing for
each source file. The supplied landmarks measure sparse anatomical
correspondence, and the full before/during set extends the shorter set rather
than forming an independent fitting/evaluation split. [S11]

The dataset creator specifies millimeter world coordinates, reference-image
points first, and matching world coordinates between the supplied MINC and
NIfTI representations. Registration tests must retain that direction and check
each image's active header transform before scoring. Filename order alone is
insufficient to establish coordinate semantics. [S14]

## Phase map and simulator contract

The following states, preconditions, and records are **engineering proposals inferred from the cited evidence**, not measured transition laws. Numbering expresses dependencies; phases 7–12 recur and can overlap. No fixed surgical durations are inferred from edited videos. Each phase must retain entry/exit times, actors, evidence references, and a reason for transitions or interruption.

| Phase | Observable state and actions to represent | Preconditions and records | Gate 2 relationship |
|---|---|---|---|
| 1. Intent and preparation | Review available images, baseline function, prior surgery, target compartment, chosen resection/biopsy intent and uncertainty. | Same-person identity and source frames verified; record task baseline and specialist-selected goals. | Initialization metadata; no automatic clinical eligibility decision. [S1, S2, S8] |
| 2. Position and room setup | Head/body/table pose, fixation, operator sides, anesthetic/awake state, monitors, viewing equipment and instrument availability. | Record head-to-room transform, tracking reference and calibration status; preserve team access and equipment occupancy. | Assumed workspace and patient pose, visibly declared. Future room geometry. [S3, S8] |
| 3. Image guidance and registration | Register/check navigation, inspect source slices, identify disagreement or tracking loss. | A pointer requires calibration and valid tracking; observations carry timestamp, modality and uncertainty. | Verified case frame and linked MPR; clinical navigation accuracy remains unvalidated. [S6, S7] |
| 4. Scalp and muscle exposure | Expose bone; represent reflected soft-tissue layers, contact, superficial field contents and access. | Patient-specific scalp/muscle evidence or explicit synthetic layers; instrument–layer compatibility. | Future phase; current post-exposure initialization does not simulate it. [S3, S5] |
| 5. Craniotomy | Plan and execute opening geometry; retain removed bone flap, edges and access constraints. | Acquired skull evidence, local thickness/defects, dura/sinus uncertainty and full-shaft swept checks. | Future engine. A proposed opening can be displayed only with skull evidence; never infer bone from stripped MRI. [S3, S5] |
| 6. Dural opening and initial surface exposure | Open/reflect dura, observe cortical surface and visible vessels, record CSF/field changes. | Dural and adhesion status must be known or labeled assumed; underlying structures are not transparent. | Explicit assumed exposure mask and provenance. No claim of an observed surgical cavity. [S3, S5] |
| 7. Optical orientation | Inspect field, change view/focus/illumination, identify occluded or unobservable regions. | Microscope/exoscope camera pose, field of view, depth and occlusion model; tools and fluid may obstruct view. | Inspect/reassess with declared abstract visibility. A 3D orbit view is not automatically a microscope observation. [S3, S7] |
| 8. Functional mapping/monitoring | Acquire/repeat cortical or subcortical responses and task observations; assess signal reliability. | Appropriate protocol, contact and team readiness; task-specific baseline/participation when awake testing is used. | Unknown unless acquired; scripted responses must be labeled generated. Future functional observation module. [S8, S9] |
| 9. Entry, sampling and progressive resection | Select an accessible region/strategy; sample, dissect or remove supported tissue incrementally. | Active working tip, supported contact, reachable exposed surface, compatible tool and valid collision check. Log sample location/time separately from removal. | Aspirator removal, persistent cavity and alternate action sequences. No histology oracle. [S4, S10] |
| 10. Field management and hemostatic response | Separate mobile field-content evacuation from action on a bleeding source; observe visibility and recurrence. | Source-specific state required for a hemostasis claim; fluid inventory required for volume accounting; existing tissue cannot be erased by a field-clearing tool. | Abstract field manager now; source-level coagulation and fluid physics later. [S4, S5] |
| 11. Shift and anatomical update | Observe displacement or navigation disagreement; acquire updated imaging and review registration. | Acquired intraoperative data or explicitly generated displacement; retain pre-update state and uncertainty. | No implied deformation if unsupported. Future observation/calibration target using paired imaging. [S6, S11, S12] |
| 12. Residual/functional reassessment and stopping | Inspect residual evidence, task/monitor changes, access, visibility and missing information; continue, change route, pause, stop or escalate. | Every decision references observations available at that time; unknowns remain unknown. | Two branches from identical saved state; STOP reason and unresolved target remain visible. [S1, S8, S13] |
| 13. Final field review and closure | Record final field inspection, retained material, dural/bone/soft-tissue closure and unresolved findings. | Completion of resection is separate from field stability and closure; object inventory and closure-layer states required. | Future phase; Gate 2 ends at an explicit episode boundary, not “operation completed.” [S3, S5] |
| 14. Recovery and outcome observation | Record neurological examination, imaging, pathology and follow-up when actually available. | Preserve acquisition times and pre/post comparability; avoid using future evidence in earlier choices. | Future external validation; simulated counts never become patient deficit probabilities. [S1, S5, S9] |

## Instruments and surrounding operative environment

An instrument has a persistent identity, shaft/tip geometry, pose, activation, availability and exchange state. Its motion continues from its prior pose. Two hands share one clock and physical workspace. Passing/exchanging a tool takes a recorded action; activating one must not teleport or erase the other. Geometry checks cover both shafts and tips throughout supported motion, including overlapping intervals.

| Device or object | Proposed effect/observation contract | Evidence still needed for physical realism |
|---|---|---|
| Suction/irrigation | Move mobile fluid/debris into/out of a field inventory; distinguish contact with tissue. | Device flow/pressure and collected-material measurements; calibrated view obstruction. |
| Ultrasonic aspirator | Local fragmentation and removal with aspiration/irrigation states; irreversible tissue/cavity ledger. | Device-specific activation, force, tissue type, trajectories and measured removal. |
| Bipolar forceps | Two-tip contact and activation history; optional source-control and patency transitions. | Temperature, sealing, flow and tissue observations; no borrowed universal injury threshold. |
| Scissors/dissector/grasper | Separation, traction, attachment/release and slip, rather than generic removal. | Contact forces, constraints and independently observed tissue response. |
| Cottonoid/hemostatic material/retractor | Persistent placement, coverage, pressure/contact, retrieval or intended retention. | Material/contact/visibility data; deformation and perfusion evidence for injury claims. |
| Microscope/exoscope | Camera and illumination state, visible surface, working distance and occlusion; observation only. | Synchronized calibrated operative video and camera metadata. |
| Mapping/monitoring devices | Time-stamped signal/task evidence with quality and interpretation; no tissue erasure. | Matched stimulation, task, response and location records; specialist labels. |
| Ultrasound/navigation equipment | A limited acquisition and reviewed spatial update; tracking availability can change. | Probe calibration, tracked images, timing, landmarks and independently scored registration. |

These are simulator contracts, not claims that the published sources measured these fields. Clinical anchors for aspiration/bipolar distinction are S4; real tracking constraints are S7; optical use is S3; mapping is S8–S9. Every material or tool effect beyond available observations starts `generated_uncalibrated`.

The future room model should include surgeon/assistant workspaces, scrub-team exchanges, anesthesia access, navigation line of sight, microscope/exoscope and ultrasound occupancy, display availability, and mapping-team communication. Patient cooperation, monitoring quality, equipment failure and task interruption are independent states. The room must not be only a decorative backdrop: an unavailable image or obstructed view changes the observation offered to the planner. Whether a source supplies any of these variables must be audited, not inferred from a video caption.

## Branches, timing and evidence boundaries

Awake testing is a **case-selected branch**, not a compulsory state for every glioma. The awake-surgery guideline emphasizes team planning, rehearsed tasks, participation and a contingency for inability to continue awake work. Sanai's original series supports individual variability in language sites. Motor and language evidence are not interchangeable; absent or unreliable task responses cannot establish a safe boundary. Specialist review must set the appropriate scenario and interpretation. No anesthetic doses, stimulation prescriptions or clinical conversion rules are encoded here. [S8, S9]

Other explicit branches include initial versus repeat operation, accessible versus inadequately exposed target, observed vessel versus unknown vascular coverage, valid versus lost tracking, reliable versus interrupted mapping, and residual compatible with the selected goals versus stopping with residual. A generated vessel encounter must not be presented as an event observed in the source patient.

Use three clocks: `episode_time` for simulated action durations, `acquisition_time` for actual observations, and `media_time` for an edited source. S3 transcript landmarks are 2:58 positioning, 3:12 skin exposure, 3:26 bone opening, 3:37 dura, 3:48 exoscope, 4:51 specimen, 5:00 fluorescence, 6:07 hemostasis, 6:16 dura closure, 6:19 bone and 6:23 soft tissue. **These are video timestamps, not elapsed surgical durations.** The video itself was not downloaded or frame-reviewed in this research pass; the published full transcript was read.

Each event should record `state_before`, `state_after`, phase, actor/tool IDs, start/end, continuous pose reference, activation, precondition result, observed inputs, evidence status and effect ledger. Replay preserves removal, contacts, field burden, pending events and tool state. Branches share the exact snapshot hash and random-state/version metadata. An observation event may change knowledge without changing anatomy. Ground-truth target/vascular labels may support evaluation but must not leak into deployed action masks or route filtering when unavailable to the policy.

## Immediate Gate 2 boundary and next implementation decision

The integrated v2 episode exposes `aspirate`, `clear_field`, `inspect`, `reassess`, motion, exchange and STOP in `src/resectionlab/rehearsal_episode.py`. Its second persistent instrument is named **Field manager**. It accumulates field-contact history and lowers a dimensionless generated-debris proxy after supported contact; it never removes tissue. The earlier name “Coagulator” was corrected following this source review. Neither action establishes observed bleeding, vessel sealing, thermal injury or calibrated suction/irrigation flow. GUI and independent-check status is recorded separately in the gate ledger.

The first episode compares a short aspiration pass with field management against a deeper pass with reassessment from the same saved snapshot. Compare removal, non-target effects, conflicts, field burden, residual and stopping reason. A source-control mechanic and mobile fluid inventory are prerequisites for a future bipolar/hemostasis claim. Contact history alone is not thermal injury, and these simulator metrics cannot establish that either branch is clinically safer.

Full incision, bone work, dural manipulation, physiological anesthesia, real mapping, calibrated microscope rendering, blood flow/heat, deformable tissue and closure remain later phases. Gate 2 needs honest initialization and limitations, not an entire craniotomy engine. If no vascular anatomy was acquired, vascular status remains unknown. If no source-level bleeding was modeled, show no blood-loss or vessel-sealing endpoint.

## Evidence acquisition and validation still required

1. **Anatomy:** follow the corpus plan for same-person registration and layer QC. Full exposure additionally needs skull/scalp/dura/sinus/vein and prior-operative information where applicable. Visible large arteries do not establish coverage of perforators, small vessels or venous drainage.
2. **Operation observations:** seek consented, appropriately licensed continuous video plus synchronized instrument poses, activation, camera state, navigation calibration, image acquisitions and specialist phase/action annotations. Edited case videos support vocabulary and sequence; they are not paired action–effect datasets.
3. **Shift:** RESECT and ReMIND offer intraoperative imaging appropriate to registration/deformation observation tests. They do not, merely by containing before/after scans, identify which tool action caused each deformation or injury. [S11, S12]
4. **Mechanisms:** obtain independent material/contact/removal, fluid, temperature and vascular measurements before claiming calibrated physical effects. Keep exposure, contact, blood loss, patency, ischemia and neurological outcome separate.
5. **Validation:** software tests establish identity/frame preservation, deterministic branches, swept collision checks and accounting. Specialist review establishes workflow plausibility. External measurements establish specific physical fidelity. Patient outcome validation requires suitable independent cohorts and endpoint definitions; neither screenshots nor passing tests establish it.

No such external mechanism calibration or specialist endorsement was performed in this research pass.

The [fixed Case3 image-only test](../artifacts/poc-resect-case3-image-registration-v1/RESULT.md)
now supplies a measured limitation for phase 11: image NMI improves while
all-17-landmark RMS increases from 2.028542 to 2.087485 mm. This one TRAIN case
does not justify an automatic anatomy update. The earlier
[Case2 conditional comparison](../artifacts/resect-case2-sparse-update-v1/RESULT.md)
uses six observed correspondences as input and ten different points for scoring;
its lower error answers a different, observation-conditioned question. Neither
comparison identifies the surgical action causing displacement, validates dense
tissue response or establishes a safe navigation tolerance.

## Primary source register

All sources checked 2026-10-16 using web search/open. “Indexed text” means the browser search returned source text but a direct open was blocked; it is not claimed as a downloaded full paper. Links are to original publications, professional guidance, or their PubMed/PMC records.

| ID | Source and publication date | Access and limited use |
|---|---|---|
| S1 | Goldbrunner et al., [EANS–EANO guidelines on extent of resection](https://pubmed.ncbi.nlm.nih.gov/40973061/), 2026-01-01; DOI 10.1093/neuonc/noaf217. | PubMed abstract; current scope/EOR context only. |
| S2 | Weller et al., [EANO diffuse-glioma guideline](https://www.nature.com/articles/s41571-020-00447-z), 2021-03 issue. | Indexed surgical-therapy guidance; direct publisher open failed. |
| S3 | Schupper et al., [Contemporary intraoperative visualization for GBM](https://pmc.ncbi.nlm.nih.gov/articles/PMC9555356/), 2022-01-01; DOI 10.3171/2021.10.FOCVID21174. | Full original operative-video transcript read; edited single case, media not frame-reviewed. |
| S4 | Hebb et al., [Sub-pial resection technique](https://pmc.ncbi.nlm.nih.gov/articles/PMC3267372/), 2011-12-13; DOI 10.4103/2152-7806.90714. | Indexed methods/discussion read; original technique/cases, not a universal tumor model. |
| S5 | Morshed et al., [Repeat craniotomies in glioma](https://link.springer.com/article/10.1007/s00701-021-05067-9), online 2021-12-30, 2022 issue. | Publisher article and indexed operative sections; single-surgeon retrospective experience. |
| S6 | Nabavi et al., [Serial intraoperative MRI of brain shift](https://pubmed.ncbi.nlm.nih.gov/11322439/), 2001-04; DOI 10.1097/00006123-200104000-00019. | Original 25-patient study abstract; deformation observation, not tool causality. |
| S7 | Juvekar et al., [Tool-tip tracking during brain tumor surgery](https://pmc.ncbi.nlm.nih.gov/articles/PMC9913508/), 2023-01-29; DOI 10.3390/cancers15030825. | Indexed primary study text; tracking is an imperfect removal surrogate. |
| S8 | Japan Awake Surgery Conference, [Guidelines for Awake Surgery](https://www.awake-surgery.jp/wp-content/uploads/2024/01/Guidelines-for-Awake-Surgery.pdf), 2024-01-15; DOI 10.2176/jns-nmc.2023-0111. | Official society PDF; team/selection/task structure, no dosing model. |
| S9 | Sanai et al., [Functional outcome after language mapping](https://pubmed.ncbi.nlm.nih.gov/18172171/), 2008-01-03; DOI 10.1056/NEJMoa067819. | Original 250-patient study abstract; variability, not transferable safe-distance or deficit rules. |
| S10 | Karschnia et al., [RANO tissue-sampling recommendations](https://pmc.ncbi.nlm.nih.gov/articles/PMC10849105/), 2023-11; DOI 10.1016/S1470-2045(23)00453-9. | Indexed consensus recommendations; specimen–image–time provenance. |
| S11 | Xiao et al., [RESECT clinical database](https://aapm.onlinelibrary.wiley.com/doi/10.1002/mp.12268), online 2017-04-09; [author-hosted full manuscript](https://users.encs.concordia.ca/~hrivaz/Xiao_Database_MedPhys.pdf). | Full acquisition/landmark methods read; reported field preparation and sparse correspondence, not tool-force trajectories or universal case states. |
| S12 | Juvekar et al., [ReMIND database](https://www.nature.com/articles/s41597-024-03295-z), 2024-05-14. | Publisher dataset description; intraoperative imaging, not complete operative trajectories. |
| S13 | Incekara et al., [Randomized intraoperative B-mode ultrasound trial](https://pubmed.ncbi.nlm.nih.gov/34094939/), 2021-05-19; DOI 10.3389/fonc.2021.649797. | Primary trial abstract; evidence for an information action, not perfect residual detection. |
| S14 | Xiao, [RESECT creator's dataset instructions](https://yimingxiao.weebly.com/data-repositories.html), accessed 2026-10-16. | Reference/moving landmark order, millimeter world coordinates and MINC/NIfTI correspondence; no clinical registration tolerance. |
