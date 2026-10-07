# Glioma resection: operative research for simulation design

October 6, 2026. Companion: [Medivis project blueprint](medivis-project-blueprint.md). This extends the [initial feasibility review](glioma-simulation-clinical-feasibility-review.md).

## Scope and method

Three explicitly requested smaller-model research agents (`gpt-6-luna`, high reasoning) investigated operative workflow, instruments/complications, and neuroanesthesia. The parent reconciled their reports with the repository and official product, dataset and interoperability sources. Follow-up research required actual accessible chapter reading and correction of source-access claims. This is a targeted research synthesis, not a systematic review, surgeon endorsement, or clinical protocol. No application behavior or patient data was changed and no experiment was executed.

The clinical scope is adult supratentorial glioma surgery. Pediatric disease, posterior fossa tumors, metastases, meningiomas and emergency operations require different assumptions. The taxonomy below covers major relevant complications, not every possible adverse event. Surgical technique, anesthesia, tumor biology and institutional practice vary; a plausible workflow is not a universal protocol.

### Textbooks and references actually read

| Source | Access achieved | Use and qualification |
|---|---|---|
| Talacchi et al., [Surgical Treatment of Supratentorial Glioma in Eloquent Areas](https://www.intechopen.com/chapters/22649), in *Advances in the Biology, Imaging and Therapies for Glioblastoma*, 2011 | Operative agent read the complete 32-page publisher-linked chapter PDF. | Baseline testing, mapping, approach and navigation limitations. Its age and categorical compartment/infiltration descriptions prevent treating it as a modern tumor-biology model. |
| Narayan et al., [Primary Brain Lesion Resection Complications…](https://www.us.elsevierhealth.com/media/wysiwyg/us/pdf/sample-chapter-9780323509619.pdf), chapter 20, Nanda's *Complications in Neurosurgery* | Instruments agent read the eight-page publisher sample chapter. | Opening pages cover general tumor-surgery complications. The subsequent sagittal-sinus meningioma case is not glioma-specific evidence. |
| [Anesthesia for Awake Craniotomy, StatPearls](https://www.ncbi.nlm.nih.gov/sites/books/NBK572053/) | Neuroanesthesia agent read full reference chapter; parent confirmed accessible source. | Anesthesia phases, monitoring, task participation and team coordination; not a patient-specific control law. |
| [Cerebral Perfusion Pressure, StatPearls](https://www.ncbi.nlm.nih.gov/sites/books/NBK537271/) | Full chapter accessed in the preceding parent review. | Pressure relationships and measurement limitations; does not justify transplanting neurocritical-care targets into open glioma surgery. |
| [Cortical Mapping in the Resection of Malignant Cerebral Gliomas](https://www.ncbi.nlm.nih.gov/books/NBK470008/?report=printable), *Glioblastoma*, 2017 | Parent previously accessed full printable chapter; current operative-agent attempts were blocked and used indexed excerpts. | Functional assessment and mapping workflow; access is not falsely attributed to the subagent. |
| Oxford's [Supratentorial Tumors](https://academic.oup.com/book/24821/chapter-abstract/188472220) and [Operating theatre and neurosurgical instruments](https://academic.oup.com/book/24506/chapter-abstract/187632578) | Public abstracts only. | Scope/context only. No claim to have read subscription chapters or full editions of Youmans, Greenberg or Cottrell. |

## 1. The operative objective

The clinical aim includes obtaining a diagnosis, reducing appropriate tumor burden and mass effect, preserving neurological function and independence, and enabling subsequent treatment. Tumor subtype, location, patient baseline and preferences affect what constitutes an acceptable resection. Current EANS–EANO recommendations distinguish newly diagnosed and recurrent tumors and molecular categories; prevention of permanent deficits is central to EANO management guidance. [EANS–EANO 2026 guideline](https://pubmed.ncbi.nlm.nih.gov/40973061/), [EANO 2021 guideline](https://pmc.ncbi.nlm.nih.gov/articles/PMC7904519/).

**Software implication:** do not optimize a single universal tumor mask or infer an individual survival benefit from virtual removal. Record intended compartment, preservation goals, baseline function, available evidence and stopping rationale. Enhancement, nonenhancing disease, FLAIR abnormality, edema and histology are different evidence categories. Molecular information known only after surgery must not enter a preoperative decision model.

## 2. How the operation unfolds

The table describes clinically recognizable phases and proposed software representations. Actions are categories, not instructions for performing surgery.

| Phase | What the team is resolving | Information and actions | Software representation |
|---|---|---|---|
| Case selection and preparation | Resection versus biopsy; intended benefit; functions at risk; patient's ability to participate | Baseline examination/cognition, MRI and supplied adjuncts, prior treatment, team review | Versioned intent and baseline; missing evidence visible; no automatically generated complete functional/vessel map |
| Position and exposure | Working angles, access, venous drainage, functional exposure and retraction burden | Head/operative position, craniotomy and dural exposure | Reviewed access region and workspace constraints; model these as inputs before implementing bone work |
| Initial orientation | Does the exposed anatomy agree with the preoperative plan? | Microscope view, registration check, optional ultrasound, cortical mapping | State timestamp and coordinate graph; distinguish patient movement/registration error from nonrigid brain displacement |
| Entry and initial tumor work | Which corridor and operative strategy preserve relevant structures? | Functional and vascular observations, cortical/sulcal entry, internal debulking or interface work | Explicit strategy and persistent instrument state; a straight geometric route is only one simplified maneuver |
| Progressive resection | Where is residual target, and what limits the next increment? | Two-handed dissection/aspiration, field clearing, hemostasis, subcortical mapping and monitoring | Cavity chronology, observation quality, interaction history and time |
| Reassessment | Has the anatomy, function or circulation changed enough to alter the plan? | Further imaging, repeated tasks/mapping, monitoring and team assessment | New evidence revises a belief/state estimate; prior plan status becomes unevaluated for that state |
| Completion decision and closure | Is further removal justified? Is the field stable? | Residual assessment, reason for stopping, hemostatic review, closure | Separate finish-with-residual, escalation, pause and aborted/computationally truncated episode outcomes |
| Recovery and follow-up | What was actually achieved, and what deficits or complications developed? | Neurological examination, postoperative imaging, pathology and follow-up | Distinct observed residual, imaging injury, function and recovery records |

The operative chapter by [Talacchi et al.](https://www.intechopen.com/chapters/22649) supports treating anatomy, baseline task performance, mapping and vascular constraints together. Its older simplifications should not be encoded as biological truth. The software should permit disagreement between a radiographic boundary and an operative/functional boundary.

### Resection strategies are not interchangeable brushes

Internal debulking works within the lesion and progressively exposes its margins. Perilesional/circumferential dissection works around a selected interface. Subpial technique uses pial boundaries in suitable anatomical contexts; it is not a guarantee that all vessels or functions are protected. Combinations depend on anatomy and tissue behavior. A retrospective comparison of perilesional and intralesional techniques is evidence of selected clinical practice, not randomized proof of one universally superior approach. [Primary perilesional study](https://pmc.ncbi.nlm.nih.gov/articles/PMC8253299/), [subpial technique account](https://pmc.ncbi.nlm.nih.gov/articles/PMC3267372/).

**Software implication:** define strategy options that change the sequence of exposed surfaces, instruments and observations. Do not represent all strategies merely by a different route endpoint. Do not invent a clean capsule around a diffuse glioma.

### Awake and asleep cases expose different information

Awake task testing can reveal language or other task-specific disruption. Motor mapping and monitoring are possible under appropriate asleep protocols. Task participation depends on baseline ability and anesthetic/airway context. A failed naming response may reflect functional interference, fatigue, sedation, seizure or inadequate testing; it is not automatically permanent damage. [Awake-craniotomy reference chapter](https://www.ncbi.nlm.nih.gov/sites/books/NBK572053/).

The 250-patient language-mapping study found substantial individual variation in functional sites. It supports individual mapping rather than fixed atlas boundaries; its reported surgical margins cannot become a universal distance threshold. [Sanai et al., NEJM 2008](https://pubmed.ncbi.nlm.nih.gov/18172171/).

**Software implication:** store task, baseline performance, stimulus/site, response quality, time and repeated observations. Keep an atlas, tractography estimate and observed stimulation response as separate objects. Unreliable or absent testing means unknown, not negative.

### Information gathering is a surgical action

Fluorescence and ultrasound can affect what the surgeon recognizes as residual target. The randomized 5-ALA study showed an increase in complete contrast-enhancing resection in the studied malignant gliomas. A smaller ultrasound randomized trial also favored ultrasound for its primary resection endpoint. Neither supports a perfect tumor oracle or a function-preservation guarantee. [Stummer et al.](https://pubmed.ncbi.nlm.nih.gov/16648043/), [Incekara et al.](https://pubmed.ncbi.nlm.nih.gov/34094939/).

Serial intraoperative MRI directly documented changing anatomy. Brain shift is therefore a reason to distinguish the actual current scene, the current estimate, and the original navigation image. [Nabavi et al.](https://pubmed.ncbi.nlm.nih.gov/11322439/).

**Software implication:** imaging/mapping actions consume time and return modality-specific evidence with limited field of view, noise and registration uncertainty. A newer scan is not automatically an accepted transform. A prior scan can remain useful while being geometrically unreliable in part of the field.

## 3. Instruments: the minimum credible interaction model

The following response models are **proposals requiring calibration**, not measured laws of living brain tissue.

| Instrument | Required state | Distinct effect | Measurement needed for a physical claim |
|---|---|---|---|
| Suction | Pose, activation, pressure/flow, lumen obstruction, contacted material | Fluid evacuation; possible tissue displacement/aspiration | Pressure/flow, contact force, displacement and removed mass/volume |
| Ultrasonic aspirator | Tip, activation/energy, aspiration/irrigation, dwell and local tissue | Fragmentation/removal distinct from free-fluid suction; hemostasis remains separate | Device-specific settings and trajectories matched to force/removal/thermal observations |
| Bipolar | Two tips/jaws, contact, activation, exposure history | Coagulation, thermal spread, possible vessel occlusion | Temperature field, flow/patency and appropriate tissue assessment |
| Scissors/dissector | Blade/jaw geometry, contact, tissue attachment/separation plane | Sharp separation or blunt mobilization | Force/displacement plus independent separation observations |
| Grasper | Jaw aperture, contact/grasp state and traction | Attachment, traction, release and possible slip | Grasp/slip/force data appropriate to the material |
| Cottonoid/patty | Location, retrieval state, pressure/contact and coverage | Protection, tamponade or obscuration | Controlled contact and optical/flow assessment |
| Retractor | Surface, displacement/load, duration and release | Sustained exposure with deformation and possible perfusion effects | Time-dependent forces/displacements; separate biological injury measurements |
| Microscope | Position, magnification, focus and illumination | Visibility and what is observable | Optical/occlusion tests and realistic working-space constraints |
| Mapping probe | Location, protocol, contact and signal quality | Functional observation and possible stimulation-related event | Matched stimulation/task/response records |
| Ultrasound probe | Pose, contact, acquisition and calibration | Partial updated anatomical observation | Tracked image/calibration data and independent landmarks |

Bimanual coordination includes two persistent poses, tool–tool interference, activation timing, relative force and view management. NeuroTouch/NeuroVR studies measure these behaviors and distinguish aspects of expertise; they do not establish each metric as a predictor of patient outcome. [Bimanual validation study](https://pubmed.ncbi.nlm.nih.gov/25599201/), [continuous bimanual assessment study](https://www.nature.com/articles/s41746-022-00596-8).

A physical brain-tumor simulation study explicitly identified lack of perfusion/bleeding as a limitation, despite realistic tissue handling. This illustrates why visual, mechanical, vascular and educational fidelity need separate evidence. [Physical simulation study](https://pmc.ncbi.nlm.nih.gov/articles/PMC7360639/).

## 4. Complications and failure mechanisms

This is an engineering decomposition informed by the clinical sources. It deliberately contains no borrowed population incidence probabilities or universal injury thresholds. Several mechanisms may explain the same observed deficit; the simulator must not label cause as certain when data do not identify it.

| Category | Mechanism/state to represent | Possible observation and timing | Simulation priority |
|---|---|---|---|
| Direct functional injury | Injury/disconnection affecting motor, language, sensory, visual or cognitive systems | Mapping/monitoring change or later domain-specific deficit; some effects unobserved intraoperatively | Functional exposure now; calibrated outcome prediction later |
| Arterial/perforator compromise | Damage, occlusion, compression or impaired inflow | Bleeding may or may not occur; ischemia and deficits can appear later | Separate vessel patency from bleeding |
| Venous compromise | Impaired drainage/congestion | Delayed swelling, ischemic or hemorrhagic effects | Distinct from arterial model; location and drainage matter |
| Field bleeding/hematoma | Active source, pooling, persistent/recurrent bleeding, accumulation | Obscured view now; postoperative hematoma possible | First interactive tool scenario |
| Thermal injury | Energy/contact/duration-dependent exposure | Limited surface cues; delayed injury | Track exposure; do not claim histological dose-response without data |
| Retraction/mechanical injury | Contact, strain/compression and duration | Deformation visible; downstream injury may be occult | Later mechanics module |
| Seizure/afterdischarge | Stimulation-associated or other electrical event | Task disruption, motor behavior or electrophysiological signal | Recognition/team-response scenario |
| Swelling and brain shift | CSF/volume changes, edema, blood volume, venous congestion, cavity change | Navigation disagreement, altered working space, imperfect monitoring | Anatomical update is a main project target |
| Ventricular/CSF effects | Ventricular entry, altered CSF dynamics, blood/debris, obstruction | Intraoperative anatomy change; delayed hydrocephalus/CSF problems | Case-specific scenario, not automatic harm on entry |
| Awake-patient difficulty | Fatigue, anxiety, pain, movement, reduced cooperation | Unreliable tasks, movement or interrupted mapping | Observation-quality/team state |
| Airway/ventilation disturbance | Obstruction, hypoventilation, aspiration or oxygenation disturbance | Ventilation/oxygenation trends with delay; mapping may fail | Optional team-training extension |
| Systemic hemodynamic disturbance | Volume loss, anesthetic effects, rhythm/cardiac changes | BP/HR trends; cerebral consequences depend on context | Coupled module only where task-relevant |
| Position/equipment error | Pressure/nerve injury, venous drainage compromise, tracking/calibration/dropout | Some visible immediately; others delayed | Tracking/dropout first; positioning injury later |
| Early recovery problem | Hemorrhage, edema, infarction, seizure, altered consciousness or deficit | Exam and imaging; urgent reassessment may be needed | Delayed observation layer |
| Later medical/wound problem | Infection, wound/CSF leak, VTE/PE, pulmonary/cardiac issues, electrolyte disturbance | Days to weeks; strongly context-dependent | Taxonomy and debrief, not first microstep model |
| Treatment-course consequence | Persistent deficits, reduced independence, delayed adjuvant treatment, progression | Longitudinal clinical and patient-reported outcomes | Separate clinical research endpoint |

The broad categories of direct injury, edema, vascular injury and hematoma, and the separation of neurological, regional and systemic complications, are supported by the opening overview in [Nanda chapter 20](https://www.us.elsevierhealth.com/media/wysiwyg/us/pdf/sample-chapter-9780323509619.pdf). Its meningioma example is excluded from glioma event calibration.

Glioma studies link postoperative DWI findings and deficits but also show imperfect correspondence. Rim restriction is not interchangeable with a clinically adjudicated infarct, and an observed DWI lesion does not identify which hypothetical action caused it. [Matched DWI study](https://pmc.ncbi.nlm.nih.gov/articles/PMC4081783/), [rim-restriction study](https://pmc.ncbi.nlm.nih.gov/articles/PMC9329326/).

The continuous suction-mapping study reported persistent motor deficits attributed to vascular injury despite its mapping approach. A dry field and apparently preserved tract geometry are therefore insufficient endpoints. [Raabe et al.](https://pubmed.ncbi.nlm.nih.gov/24628613/).

Ventricular entry must not be an automatic hydrocephalus event; relevant clinical cohorts differ in populations and associated factors. Likewise, a stimulation-related seizure is not a deterministic predictor of long-term seizure outcome. [Grade 2/3 hydrocephalus study](https://pmc.ncbi.nlm.nih.gov/articles/PMC10377207/), [awake-craniotomy seizure cohort](https://pmc.ncbi.nlm.nih.gov/articles/PMC10498660/). Later venous thromboembolic events also require a different time scale from intraoperative injury. [Primary malignant-brain-tumor NSQIP analysis](https://pmc.ncbi.nlm.nih.gov/articles/PMC5754452/).

## 5. Physiological realism without invented patient predictions

A physically meaningful model couples volume, circulation, ventilation and oxygen delivery. A blood-pressure number alone does not identify regional flow. Oxygen saturation is not hemoglobin concentration or cerebral oxygen delivery; end-tidal CO2 is not always arterial CO2. Autoregulation is dynamic and individual. These relationships explain why a single “brain health” scalar would be misleading. [Physiology reference chapter](https://www.ncbi.nlm.nih.gov/sites/books/NBK537271/), [human cerebral-flow physiology synthesis](https://pmc.ncbi.nlm.nih.gov/articles/PMC8576366/).

Opening bone and dura changes the boundary conditions. Surgeon-assessed brain relaxation is not simply a global ICP value. Model `skull_closed`, `bone_open`, and `dura_open` states explicitly; a closed-box pressure–volume rule should not persist unchanged through all three. [Brain-relaxation article](https://pubmed.ncbi.nlm.nih.gov/27121854/) was available at abstract level.

### Proposed minimal coupling

1. A local vessel model produces blood loss and local flow/patency changes.
2. A field-fluid model separately accounts for blood, irrigation, CSF if relevant, pooled fluid and evacuated fluid.
3. Systemic physiology receives vascular blood loss exactly once. Suctioning already-extravasated blood does not remove the same volume from circulation a second time or replenish it.
4. Circulation and ventilation update MAP, HR, oxygenation, CO2 and oxygen-delivery context. Hemoglobin concentration must not fall linearly with every instantaneous milliliter lost; dilution/replacement and measurement timing matter.
5. A local cerebral model combines systemic inputs with regional vessel state and operative boundaries. It does not assume a patent main circulation guarantees every tissue territory is perfused.
6. Separate observation functions produce noisy/delayed monitors, task results and operative cues. Exact latent CPP, regional flow or injury is not automatically visible.

These are proposed modeling constraints, not a validated physiological model. Hemodynamic associations reported in clinical glioma cohorts do not identify a universal causal threshold for injury. [Bette et al.](https://www.nature.com/articles/s41598-017-05767-2), [HGG stroke cohort](https://pmc.ncbi.nlm.nih.gov/articles/PMC9803666/).

Anesthesia is initially a scripted teammate with explicit observations, response latency and actions. The surgeon may request assessment or pause. A medication policy would be a separate research problem. Task testing, electrophysiology and anesthesia observations should retain their different owners and quality limits.

[Pulse](https://pulse.kitware.com/_about_pulse.html) and [BioGears](https://biogearsengine.com/documentation/_nervous_methodology.html) are candidate systemic engines, not verified glioma models. Their official documentation describes systemic physiology and validation contexts. A bounded adapter comparison should check baseline stability, perturbation direction, time-step sensitivity, state serialization, runtime and licensing. Any custom cerebral component needs independent validation. No engine was installed or run for this research.

## 6. What counts as patient benefit

Outcome recording needs baseline and follow-up time. Record motor, language, visual, sensory and cognitive domains separately, alongside independence, quality of life, seizure burden and treatment-course outcomes where actually measured. Distinguish temporary impairment, persistent impairment and recovery. Do not infer domain-specific function from a generic performance-status score.

Early postoperative MRI and diffusion imaging can assess residual and injury-related findings; they do not directly validate every alternative plan. Only the performed operation has observed outcomes. Follow-up may additionally reflect recovery, tumor behavior and subsequent therapy. [EANO guideline](https://pmc.ncbi.nlm.nih.gov/articles/PMC7904519/).

For an initial software project, plausible benefit is narrower and measurable: fewer unrecognized stale-plan assumptions, better identification of missing information, clearer comparison of candidate strategies, or improved decision-task performance. A 2025 randomized study supports studying expert-guided simulation feedback, but its participants were medical students and transfer remained within simulation. [JAMA Surgery trial](https://jamanetwork.com/journals/jamasurgery/fullarticle/2837234).

## 7. Translation rules for the implementation plan

- Every input needs a source, acquisition/availability time, physical frame and evidence class.
- Every action needs duration, persistent state and explicit interaction semantics.
- A model can remain frozen while the episode evolves; reproducibility does not require static anatomy.
- The user/policy receives observations and uncertainty, not hidden ground truth.
- Hypothetical action branches cannot inherit later clinical images as though those images were produced by the hypothetical actions.
- A measured displacement update does not identify which tool forces caused it.
- A valid software state transition, a realistic physical response, a useful training task and a better patient outcome are separate claims.
- Preserve the existing negative learning/mechanics results. They identify where the next claim needs evidence.

The [project blueprint](medivis-project-blueprint.md) uses these constraints to choose a focused Medivis-relevant contribution, with surgical training as an additional task rather than an unsupported whole-operation digital twin.
