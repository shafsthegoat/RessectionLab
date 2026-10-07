# Glioma simulation: clinical feasibility and development priorities

Reviewed October 6, 2026. This is a targeted literature and source-code assessment, not a systematic review or a new validation experiment. Recommendations below are engineering proposals. No application code, patient assignments, checkpoints, rewards, or governing plans were changed. No simulation or training was run for this assessment.

**Conclusion:** RessectionLab can plausibly become a useful platform for patient-specific planning research and surgical decision rehearsal. The present engine models constrained geometric removal; it does not yet model an operation's tissue interactions, changing observations, hemostasis, or physiology. The strongest next product hypothesis is a surgeon-supervised rehearsal of a bounded operative phase, with inspectable consequences and feedback. Its usefulness to practicing neurosurgeons still needs direct testing.

## 1. What the application actually supports

| Area | Evidence inspected | Implication |
|---|---|---|
| Patient anatomy and interface | `README.md`, `docs/desktop-workflow.md`, `desktop/src/App.tsx`, `desktop/src/viewer/inspectionTool.ts` describe MRI inspection, route comparison, instrument display, replay and saving in Electron/React with a Python engine. | Reuse this foundation. This review did not launch or revalidate the packaged app. |
| Tool representation | `src/resectionlab/geometry.py:74` represents tip/shaft capsules, lengths and access angles. Generic suction and aspirator entries differ in dimensions. | A tool name does not establish distinct physical behavior. There are no pressure, energy, jaw, temperature or material-response fields in this geometry class. |
| Resection | `src/resectionlab/native_resection.py` removes connected source cells wholly contained in an active capsule; partial contacts remain occupied. Full shaft clearance is checked. | Valuable geometric accounting, but no calibrated ultrasonic fragmentation, sharp dissection, coagulation or deformation. Resolution sensitivity remains relevant to small tips and vessels. |
| Action semantics | `simulation.py` and `native_simulation.py` use insertion/removal/reverse-path withdrawal macro-actions; reorientation occurs outside modeled tissue. | The active tool does not remain in the cavity between these macro-actions. This cannot yet express sustained bimanual dissection or retraction. Spatial microsteps are not physiological time. |
| Uncertainty and function | `worlds.py` samples rigid registration scenarios; functional modules evaluate supplied-map contact and evidence coverage. | Preserve these distinctions. Static uncertainty is not evolving brain shift, stimulation response or neurological injury. |
| Mechanics | `docs/tissue-mechanics-validation.md` records analytical controls and unpassed physical-validation gates. | Existing FEM preparation does not establish patient-specific retraction or cutting response. Reuse it as a separate research track. |
| Learning evidence | [Prepared TRAIN comparison V2](../artifacts/prepared-training-planner-comparison-v2/RESULT.md) contains four completed cases out of six, with two support blocks. | Greedy search exceeded frozen imitation return in all four completed cases. The policy used less online time. This was not an RL update or an untouched-patient validation. |

That comparison executed only three non-STOP actions per completed active arm: approximately 3.02–5.17% target removal for imitation and 4.31–6.27% for greedy. These are short geometric episodes, not full resections. Motor, language and clinical-deficit outputs were null in that result. A larger positive return would not establish better surgery.

## 2. What the literature supports

The following evidence informs design, not validation of this application. Full text, abstracts and publisher excerpts are identified explicitly.

| Source and access | Finding relevant here | Design consequence and limit |
|---|---|---|
| [Goldbrunner et al., EANS–EANO guidelines, Neuro-Oncology 2026; online 2025](https://pubmed.ncbi.nlm.nih.gov/40973061/) — abstract and indexed recommendation text | Resection goals differ by tumor context, with neurological status, cognition and quality of life relevant alongside extent of resection. | Use separate tumor compartments and functional endpoints. Do not turn one target-volume reward into a universal operative objective. |
| [Alotaibi et al., Operative Neurosurgery 2015](https://pubmed.ncbi.nlm.nih.gov/25599201/) — abstract | NeuroTouch bimanual tasks distinguished aspects of performance among six neurosurgeons and twelve residents using forces, movement, removal and bleeding metrics. | Bimanual tool use and field management are established simulator design targets. Construct validity does not establish clinical transfer or patient outcome prediction. |
| [Giglio et al., JAMA Surgery 2025](https://jamanetwork.com/journals/jamasurgery/fullarticle/2837234) — [author-hosted full text](https://neurosim.mcgill.ca/wp-content/uploads/2025/08/jamasurgery_giglio_2025_oi_250042_1754078206.57075-2.pdf) | In 87 analyzed medical students, AI-augmented personalized expert instruction outperformed AI-only instruction on simulation performance; transfer was to another simulated task. | Prioritize expert debrief and interpretable feedback. This was neither a trial of autonomous surgical RL nor proof of operating-room transfer. |
| [Raabe et al., Journal of Neurosurgery 2014](https://pubmed.ncbi.nlm.nih.gov/24628613/) — abstract | A prospective 69-patient study integrated subcortical stimulation with suction. The two persistent motor deficits reported at three months were attributed to vascular injury. | Mapping can be an information-producing action. Functional proximity and vascular injury require separate models; preserving a tract representation alone cannot certify function. |
| [Stummer et al., Lancet Oncology 2006](https://pubmed.ncbi.nlm.nih.gov/16648043/) — abstract | In the analyzed groups, complete contrast-enhancing resection occurred in 65% with 5-ALA versus 36% under white light. | An observation modality can change decisions. A fluorescent display must not become a perfect ground-truth tumor mask or a guarantee of functional safety. The trial concerns malignant glioma. |
| [Incekara et al., Frontiers in Oncology 2021](https://pubmed.ncbi.nlm.nih.gov/34094939/) — abstract and indexed full text | A 50-patient randomized trial reported complete enhancing resection in 8/23 analyzed ultrasound cases versus 2/24 controls. | Give intraoperative imaging a time cost and imperfect observations. This small selected trial does not establish a survival gain or universal benefit. |
| [Nabavi et al., Neurosurgery 2001](https://pubmed.ncbi.nlm.nih.gov/11322439/) — abstract | Serial intraoperative MRI in 25 patients documented changing anatomy during surgery. | Separate the current anatomical state from the preoperative navigation estimate; observations must update registration and uncertainty. |
| [Bette et al., Scientific Reports 2017](https://www.nature.com/articles/s41598-017-05767-2) — indexed article text | Hemodynamic measures were associated with postoperative infarct volume. | Systemic physiology deserves consideration, but an observational association cannot calibrate a causal MAP-to-injury simulator. |
| [Risk factors and prognostic implications of surgery-related strokes following HGG resection, Scientific Reports](https://pmc.ncbi.nlm.nih.gov/articles/PMC9803666/) — indexed results | This cohort did not show a significant difference in duration below MAP 65 between infarct groups. | Avoid a universal BP threshold that deterministically produces an infarct. Location, local circulation, exposure duration and measurement limitations matter. |

Medical textbook/reference material actually consulted:

- [Zakaria and Prabhu, “Cortical Mapping in the Resection of Malignant Cerebral Gliomas,” in *Glioblastoma*, 2017](https://www.ncbi.nlm.nih.gov/books/NBK470008/?report=printable): accessible chapter. It covers functional assessment, surgical corridors and stimulation. Use it to structure a mapping workflow; its institutional descriptions and older claims are not universal safety rules.
- [Narayan et al., chapter 20 in Nanda's *Complications in Neurosurgery*](https://www.us.elsevierhealth.com/media/wysiwyg/us/pdf/sample-chapter-9780323509619.pdf): publisher's eight-page chapter excerpt, especially the opening overview. It distinguishes neurological, regional and systemic complications and discusses vascular injury, edema and hematoma. The later sagittal-sinus meningioma case must not supply glioma event rates.
- [Mount and Das, *StatPearls: Cerebral Perfusion Pressure*, updated 2023](https://www.ncbi.nlm.nih.gov/sites/books/NBK537271/): accessible reference chapter. It gives the conventional CPP relationship and explains individual physiology and measurement requirements. Its neurocritical-care thresholds are not automatic targets for an elective, open-cranium glioma simulator.
- [Traul and Diehl, “Supratentorial Tumors,” in *Neuroanesthesia: A Problem-Based Learning Approach*, 2018](https://academic.oup.com/book/24821/chapter-abstract/188472220): public abstract only. It frames ICP, cerebral blood flow, metabolism and neuroprotection; detailed chapter protocols were not accessible or used.

Full editions of subscription operative textbooks were not reviewed. This assessment does not claim access to Youmans, Greenberg or Cottrell beyond any publicly available material explicitly identified above.

## 3. Choose the clinical use before adding features

Three applications have different requirements:

| Intended use | Credible near-term direction | Evidence still needed |
|---|---|---|
| Surgeon planning/rehearsal | Compare access, working angles, information needs and stopping decisions in reviewed anatomy. | Surgeon relevance, preparation burden, anatomical accuracy, reproducible decision benefit over ordinary MRI/3D review. |
| Resident/team decision training | Practice recognition, tool selection, field management and escalation in declared scenarios. | Content validity, blinded performance assessment, retention and transfer to an independent task. |
| Patient outcome prediction or operative guidance | A longer-term research objective. | Appropriate clinical data, calibrated relevant models and prospective evaluation of the intended use. Simulation return is insufficient. |

The first two can be useful before every tissue property is known. Claims of instrument-handling skill require appropriate input devices, forces and latency; mouse-based rehearsal can test decisions without establishing psychomotor training value. An attending surgeon's reason to return is likely a difficult case or a meaningful comparison; a resident's reason may be coached practice. Interview both groups separately.

**Initial scope proposal:** one surgeon-reviewed, exposed supratentorial tumor scenario, beginning after dural opening and ending at a documented decision to finish or escalate. Define tumor compartment, operative strategy and awake/asleep context. Bone work, closure and a complete anesthetic induction can remain separate modules. A motor-monitoring scenario and an awake language scenario should not silently share identical observation rules.

## 4. Tools need distinct actions and consequences

This table is a proposed abstraction, not device specifications or operating instructions. Every response parameter must be labeled measured, literature-derived, expert-specified or uncalibrated.

| Tool or adjunct | Minimum state and actions | Consequence the environment should represent | Priority |
|---|---|---|---|
| Suction and irrigation | Pose, activation, suction setting, fluid type/amount; irrigation input | Fluid clearance and visibility; contact-dependent tissue interaction; blood and irrigation tracked separately | First |
| Ultrasonic aspirator | Pose, activation, configured aspiration/irrigation/energy, dwell time and tissue class | A distinct removal-rate model, interaction with vessels and surrounding tissue; no instant guaranteed removal from contact | First, as a declared simplified model |
| Bipolar forceps | Two tips/jaw state, contact region, activation, energy and exposure time | Hemostasis plus possible thermal spread and vessel occlusion; stopping blood flow is not automatically a good outcome | First |
| Cottonoid/patty | Placement, pressure, dwell and retrieval state | Temporary field protection or tamponade, visibility obstruction and contact | Next |
| Mapping probe or suction stimulator | Site, stimulation protocol, contact and response quality | Noisy functional evidence, possible afterdischarge, and reason to alter or stop a maneuver | Next |
| Microscope, fluorescence and intraoperative ultrasound | View pose, illumination mode, acquisition time and registration state | Occlusion, uncertain tumor boundaries and updated observations | Microscope visibility first; additional modalities next |
| Dissector, microscissors and grasping forceps | Blade/jaw geometry, grasp state, traction and separation plane | Sharp/blunt dissection and tissue separation | Later; needs a different interaction model |
| Retractor | Contact surface, displacement/load history and duration | Deformation and time-dependent local exposure | Later; physical-response validation required |

Two-handed operation is more than selecting a second instrument: retain two poses, joint action timing, mutual collisions, shared working space and each instrument's interaction history. Instrument geometry must eventually include relevant jaw states and tip variants. Do not imply manufacturer equivalence from approximate dimensions.

## 5. Physiology: include BP and HR as a coupled model

**Yes, BP and HR can be simulated.** Their value is in how they affect the task and how the simulated team responds. Independent random vital-sign changes teach little about surgical decisions.

A proposed state includes elapsed time, blood volume/loss, HR, MAP, oxygenation, ventilation/CO2, hemoglobin context, anesthesia context and local perfusion. Some can initially be fixed scenario parameters. Add latent intracranial compliance/pressure only with a declared model of skull/dural opening, CSF and brain swelling. Display ICP or CPP as measured only if the scenario actually contains the necessary monitors; otherwise they are latent or labeled model estimates.

The conventional reference relation is `CPP ≈ MAP − ICP`, but it is not an infarct predictor. Open-cranium boundary conditions, regional tissue pressure, venous outflow and measurement reference levels need explicit treatment. Oxygen saturation alone also cannot stand in for oxygen delivery. No universal “safe BP” or injury threshold is proposed here.

Suggested coupling:

```text
instrument interaction -> vessel injury or occlusion
injury -> local bleeding -> field obscuration and accumulated blood loss
suction/irrigation -> field clearance and fluid balance
hemostatic action -> bleeding control, with separate patency/thermal consequences
blood loss + anesthesia/team actions -> systemic physiology
local vessel state + systemic physiology -> regional perfusion scenario
observations -> operator assessment -> next action
```

Local vessel occlusion must be able to matter without dramatic systemic BP/HR change. Otherwise the learner can preserve normal monitor values while performing destructive local actions. Conversely, a mapping/monitoring change should have possible technical or physiological explanations, rather than reveal hidden injury with certainty.

Start with an explicit scenario-driven anesthesia teammate: the surgeon can pause and request assessment, while declared team actions have latency and physiological effects. A drug-dosing agent is a separate task requiring its own evidence. Do not train the resection policy to prescribe medication simply because BP was added.

[Pulse's cardiovascular model](https://pulse.kitware.com/_cardiovascular_methodology.html) and [hemorrhage model](https://pulse.kitware.com/md__hemorrhage.html) are candidates for a bounded integration experiment. They offer a published compartmental model and validation descriptions. Neither establishes glioma-specific microvascular flow, open-cranium ICP, cautery injury or this application's coupled fidelity. Check applicable operating ranges, runtime and licensing before adoption; no dependency was installed here.

## 6. Complications should follow mechanisms and unfold over time

The mechanism-to-event mappings below are proposals for expert review, not estimated patient probabilities.

| Scenario | Trigger/state to model | Observable information | Decision being tested |
|---|---|---|---|
| Bleeding obscures the field | Local injury, source flow, pooling and clearance | Visible blood, reduced visibility, delayed blood-loss estimate | Interrupt removal, restore visibility and assess the source |
| Local ischemia | Arterial/perforator occlusion or sustained local compression | Imperfect monitoring/task changes; possibly no immediate warning | Preserve circulation and reassess despite apparently successful hemostasis |
| Venous congestion | Drainage impairment followed by swelling or delayed injury | Changing tissue appearance, swelling and uncertain monitoring | Recognize that venous and arterial consequences differ |
| Functional boundary encountered | Reproducible mapping response or evolving monitoring change | Site, task, timing and signal quality | Re-map, change the maneuver, pause or leave residual |
| Stimulation-associated seizure | Protocol-dependent scenario event | Clinical/task or electrophysiological observation | Stop the provoking action and coordinate the modeled response |
| Navigation becomes inaccurate | Evolving displacement, cavity change or registration error | Disagreement with current anatomy or new imaging | Acquire information and revise the plan |
| Thermal exposure | Bipolar activation, contact and cumulative dose | Incomplete visual cues; delayed effects | Limit collateral exposure while addressing bleeding |
| Systemic instability | Blood loss, ventilation or anesthesia disturbance | BP/HR/oxygenation/CO2 trends with sensor delay | Pause and communicate with anesthesia |

Early postoperative hematoma, infarction and transient/persistent deficits need a separate delayed-outcome layer if included. Infection, thromboembolism and wound healing are generally outside the first intraoperative task. Do not assign a per-step random “paralysis” event or transplant pooled postoperative complication percentages into action hazards.

## 7. How the RL environment should change

**Preserve the existing geometric benchmark and implement a separately versioned episode type.** The central change is from selecting removal strokes to choosing actions with persistent state, time and imperfect feedback.

Proposed interfaces, not current classes:

- `SurgicalEpisodeState`: current anatomy/cavity, two tools, blood/visibility, vessel state, functional state, physiology, clock and event history.
- `ToolAction`: instrument, pose/motion, activation, settings and duration. Information requests, pause, withdrawal and escalation are first-class actions.
- `ObservationModel`: operative view, permitted navigation, acquired mapping/imaging, monitor signals, timestamps, noise and missingness.
- `InteractionModel`: geometry first, with explicit simplified fluid, thermal and vascular scenarios; calibrated mechanics can later replace individual components.
- `PhysiologyAdapter`: receives physically dimensioned events and returns time-evolved physiological quantities.
- `OutcomeEvaluator`: independently reconstructs removal, exposure, patency scenarios, accumulated disturbances and task completion.

Keep `ToolGeometry` for its geometric role. Extend beyond `MacroAction`/`SimulationObservation` through a new contract; do not reinterpret old checkpoints or source-cell certificates as supporting deformation. After anatomy changes, invalidate/recompute affected geometry. Frozen model parameters and reproducible seeds are compatible with evolving episode state.

Use a partially observed, duration-aware task. The policy sees what the operator could know, while latent anatomy and scenario parameters remain available to the simulator/evaluator. Action masks must not disclose invisible vessels or future injury. A privileged-information planner can be an explicitly labeled oracle ceiling, not the ordinary comparator. Deployment-inaccessible reward components must not leak into the actor's observation history.

RL has a stronger research rationale when an action changes future information or creates delayed consequences: clearing blood before proceeding, acquiring a map, preserving a working corridor, or stopping despite visible residual. These properties do not guarantee RL beats search. Compare a rule-based controller, the existing search baseline where applicable, observation-aware receding-horizon search, imitation and one recurrent policy under the same information and time budgets. Use [LapGym](https://jmlr.org/papers/v24/23-0207.html) as an architectural reference for SOFA-based soft-tissue RL tasks, not evidence of brain-model validity.

Avoid a single score that lets extra tumor removal compensate arbitrarily for a major modeled adverse event. Report a vector of outcomes and use surgeon-reviewed constraints/termination rules. Distinguish simulator-enforced geometric constraints from uncertain clinical consequences. Keep residual compartments, non-target removal, contact, blood loss, local perfusion scenarios, monitoring deviations, recovery and task time separate. “Pause,” “finish with residual,” “escalate” and “time budget exhausted” need different terminal meanings. Include post-action observation time so ending an episode cannot erase delayed consequences.

## 8. Smallest useful next experiment and adoption gates

**First engineering experiment:** a short, surgeon-authored resection-bed task with one removable target, one protected vessel scenario, aspiration, bipolar use and field visibility. Add a small family of bleeding challenges and a stop/escalate option. Keep systemic physiology stable initially unless meaningful blood loss is part of the scenario. This isolates whether distinct tool actions improve decision structure before coupling a whole-body model.

If vessel or material evidence is missing, use explicitly authored experimental geometry and parameters. It is a software/scenario test, not an inferred vessel map of an existing patient or evidence of patient generalization. Preserve current patient roles and sealed measurements. Subsequent mapping and shift scenarios should be added one at a time so their effects can be measured.

| Stage | Deliverable | Evidence needed to move forward |
|---|---|---|
| Clinical task definition | Review operative workflow with a small formative group of tumor surgeons, a neuroanesthesiologist and a monitoring specialist. Identify actual decisions and unacceptable simplifications. | Agreement on what the task tests, acceptable observations and meaningful errors; a small interview group is not clinical validation. |
| Mechanism/software checks | Distinct action effects, replayable clock/events, coherent visibility and fluid accounting | Bleeding persists when its source remains active; suction does not restore circulating blood; irrigation is not counted as blood loss; invisible state does not leak. |
| Task benchmark | Same cases/scenarios for scripted, search, imitation and RL methods | Independent endpoint checks, repeated seeds, failure rates and runtime. Improvement in the optimized score alone is insufficient. |
| Surgeon usability | Clinicians complete and explain the task; compare with current MRI/3D workflow | Setup/review time, task completion, important misunderstandings and desire to reuse; avoid relying on realism ratings alone. |
| Training/rehearsal efficacy | Prespecified blinded assessment, randomized/counterbalanced comparison when feasible | Independent errors, retention and transfer to a different task or suitable phantom. Novice/expert separation alone is insufficient. |
| Patient-specific clinical use | Prospectively defined patient cohort and intended workflow | Anatomical and observation accuracy, calibration where claimed, external evaluation and measured decision benefit. Actual postoperative imaging validates only aspects of the performed operation. |

Data requests should follow the intended claim: time-synchronized tool/video records for interactions; imaging updates and registration uncertainty for shift; mapping/monitoring records for functional observations; anesthesia traces plus interventions for physiology; postoperative imaging and domain-specific follow-up for outcomes. Tool-force claims need force measurements. Public MRI and segmentations alone do not contain these data, and one actual operation does not reveal the outcome of every hypothetical plan.

The immediate development recommendation is **distinct tools plus visibility and hemostasis**, followed by functional observations and anatomy updates, then coupled BP/HR physiology where it changes decisions. A useful initial claim would be: “This application helps surgeons rehearse and inspect specific decisions under declared uncertainty.” Whether it improves those decisions is the experiment to perform.
