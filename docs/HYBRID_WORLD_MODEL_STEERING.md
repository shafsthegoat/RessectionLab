# RessectionLab: evidence-grounded hybrid world model and RL steering

Prepared October 10, 2026. This is a proposed steering addendum for the existing autonomous project, not a replacement for its full destination. Paste the directive below into the running project's next steering message, or provide this file as that message's attachment. No repository changes, training jobs, downloads of surgical videos, or task scheduling were performed by preparing this review.

## Review basis and decision

**Recommendation: continue the project, but add this steering now.** Finish any already-running bounded measurement safely; do not leave the next cycle as another unconstrained iteration of the same tiny RL task. Preserve useful code and negative experiments. Make measurable surgical behavior and independent world-model validation the next organizing milestones.

The reviewed main-branch snapshot is `4d1d0ff193275696c53cc907c8eb8aa79f6f3094`, committed October 10, 2026 at 15:08:47 UTC, or 11:08:47 a.m. EDT. The review inspected recent commit metadata, selected diffs, project status, the active execution ledger, README, and saved experimental reports. It did not rerun the application, native geometry, training, or independent audits. Repository reports describe their own execution evidence; this document does not independently reproduce their numerical results.

The original uploaded October 6 supergoal already asks for extensive operative science, persistent bimanual tools, exposure, hemostasis, uncertainty, and real validation. Those are not newly invented requirements. The proposed change is a concrete implementation sequence and a testable hybrid-model hypothesis, with newer literature and better source-to-test mapping.

### What the inspected results establish

| Area | Evidence at the reviewed snapshot | What remains unestablished |
|---|---|---|
| Native geometry and replay | Full-tool geometry, explicit unknown coverage, post-exposure starts, independently checked saved routes, correction of a false-rotation checker defect | That geometric acceptance implies feasible human surgery or tissue safety |
| Bounded resection progress | Search removes 58 target cells on TRAIN015 and 51 on TRAIN045, respectively 0.74887% and 0.20157% of their full supplied targets; two other cases STOP | A complete or clinically substantial glioma resection |
| Imitation | Original imitation reaches the positive target-cell endpoints; TRAIN045 requires 24 movements versus search's 12 | Surgeon imitation, good efficiency, or useful transfer |
| Scratch RL | Eight shared updates and 32 episodes complete; RL is worse than search and accepted imitation on all four TRAIN cases | Convergence, broad RL failure, or a useful learned planning advantage |
| New-patient comparison | Repaired SELECT013 greedy, imitation, and RL routes all remove zero target. Imitation and RL remove approximately 2 and 27 mm³ outside the supplied target. Beam search hits its 300-second cap | A completed four-arm comparison, useful transfer, or proof no good route exists |
| Performance | One TRAIN045 cache comparison reduces greedy planning time from 16.312 to 11.672 seconds with unchanged non-time outcomes | Sustained benefit, benefit to beam/RL, or a learned-policy speed advantage |
| Surgical continuity | The inspected post-exposure protocol withdraws/reset tools after individual strokes and does not retain a persistent last tool pose | Continuous two-handed operative execution across actions |
| Physical validity | Existing specimen calibration and patient-mesh fidelity tracks retain unresolved/failed gates | Calibrated human tissue interaction, bleeding, perfusion, or patient deficit probabilities |
| Desktop | Shared Python/Electron architecture, image display, replay, persistence, and generated instrument/vascular encounter demonstrations are documented | Every displayed modality being registered and consumed by the planner, or a full clinical surgical rehearsal |

Pinned audit sources:

- [Project status](https://github.com/shafsthegoat/RessectionLab/blob/4d1d0ff193275696c53cc907c8eb8aa79f6f3094/PROJECT_STATUS.md)
- [Active execution ledger](https://github.com/shafsthegoat/RessectionLab/blob/4d1d0ff193275696c53cc907c8eb8aa79f6f3094/docs/REAL_OBSERVATION_EXECUTION_LEDGER.md)
- [Saved scratch-RL result audit](https://github.com/shafsthegoat/RessectionLab/blob/4d1d0ff193275696c53cc907c8eb8aa79f6f3094/artifacts/post-exposure-rl8-result-v1/REPORT.txt)
- [Post-exposure implementation commit](https://github.com/shafsthegoat/RessectionLab/commit/d583008cb1ebc36b4ed34b5f722f3b4f8cc959ec)
- [README and application scope](https://github.com/shafsthegoat/RessectionLab/blob/4d1d0ff193275696c53cc907c8eb8aa79f6f3094/README.md)

---

# STEERING DIRECTIVE

/goal Continue the existing RessectionLab supergoal, preserving its full patient-specific glioma planning and surgical rehearsal destination. Reorganize the next execution cycle around an evidence-grounded hybrid world model, persistent multi-instrument operative behavior, and independently measured validity. Keep the working geometry, search, replay, application integration, data provenance, and useful checkpoints. Do not perform a wholesale rewrite or replace physical state with generated video. Build and evaluate a bounded hybrid-model pilot, then expand only the components that earn their complexity through measured improvements.

## 1. Reconcile current instructions before changing anything

Read the actual checkout, active human steering, current experiment receipts, patient roles, and integration contracts. The reviewed SHA above is a historical baseline; inspect newer work rather than rolling it back.

The October 8 human steering already superseded the original attachment's blanket prohibition on synthetic data and simulator-generated RL experience. Preserve that permission. Generated training, controlled simulation scenarios, search-generated imitation, and appropriate pretrained components may be used under the current provenance and intended-use contracts. They must never be presented as recorded patient interventions, measured biological behavior, or clinical outcomes. Do not waste another workstream rediscovering that old prohibition or treating it as active.

Maintain distinct stores and labels for acquired patient data, measured specimen/device data, observed operative recordings, human annotations, model estimates, and generated experience. A patient's missing vessel or tract is not measured merely because a model predicts one. A stress-test vessel configuration can be explicitly synthetic without being misrepresented as that patient's actual anatomy.

Retain the limited-information deployment hypothesis: rich simulator information may support training, but deployed actors, proposal generators, action masks, search, termination rules, and feature preprocessing must use only the declared available observations. Privileged anatomy must not leak through a legality oracle, candidate rejection, reward preview, cached feature, or hidden file read. Seal plans before private reference evaluation. Preserve the existing TRAIN/SELECT/EVAL roles, exposed development cases, and untouched evaluation reservations.

Keep the project local-first. Inspect actual CPU/GPU, memory, disk, dependency, and concurrency limits before choosing model sizes. Broad subagent reasoning capacity is not unlimited GPU memory or scientific evidence. Do not buy compute, accept new agreements, contact researchers/clinicians, upload patient records, or publish externally without existing authorization. Preserve the latest user-approved author identity, normal commit/push practice, and shared-checkout policy rather than reverting to the older attachment's identity instructions.

## 2. Primary research question

Test this proposition:

> A source-calibrated, uncertainty-aware hybrid transition model, coupled to short-horizon planning and learned action/value proposals, can improve useful multi-stage resection decisions or their computational cost on unseen patient anatomy, under the same test-time information, without increasing independently assessed structural violations or exploiting model error.

Break this into independently falsifiable claims: better transition prediction, better action selection, lower end-to-end planning cost, robustness to missing observations, and external anatomical generalization. None alone establishes clinical benefit. Do not claim novelty merely for combining a simulator with RL; novelty would need a defensible problem formulation, source-grounded mechanisms, independent validation, and convincing comparative results.

The current hand-engineered simulator is already a world model in the broad sense. A learned world model is an additional estimator of action-dependent transitions, observations, or value, not a magic replacement for its specification. Select the smallest learned component that addresses an observed bottleneck.

## 3. Establish one authoritative surgical state

Extend the shared runtime, not a disconnected research notebook. Reuse existing representations where they are correct. Define versioned units, coordinate frames, source identities, uncertainty, and coverage for every component.

The canonical state should be able to retain:

- Remaining tissue, target compartments, cavity, exposed surfaces, and supported deformation. Distinguish target annotation from a clinically approved removal boundary.
- Persistent poses and activation states of both instruments, their swept geometry, motion duration, contact footprint, and tool exchange/withdrawal. Do not silently teleport/reset between strokes in the new continuous mode.
- Vessel identities, graph connectivity where supported, and distinct intact, compressed, occluded, injured, leaking, or transected states. Vessel encounter, tissue removal, perfusion impairment, and clinical injury are different outputs.
- Accumulated contact, mechanical loading, thermal exposure, and fluid/visibility state at the fidelity actually supported. A missing parameter produces an uncertainty or unsupported output, not a fabricated safe value.
- The observation history, camera/navigation state, registration uncertainty, functional observations, and the patient's known versus unknown evidence.
- A separate delayed-outcome layer for systemic/postoperative events. Do not force infection or thrombosis into the same per-tool-step mechanism as local contact injury.

Make actions interruptible and temporally explicit. At minimum, support a declared sequence of inspect, approach, manipulate/resect, acquire information, respond to an adverse modeled event, and continue/change approach/stop. Instrument families should have different effects rather than all being renamed voxel erasers. Introduce one additional physical mechanism at a time, with a parameter source and validation target.

The initial hybrid pilot may begin after a declared exposure. That is an engineering checkpoint, not completion. Keep the later full-head positioning, incision, bone opening, dura, access geometry, and closure objectives active with their own deliverables. Do not invent skull anatomy from a skull-stripped scan to make the full workflow appear complete.

## 4. Separate physical transition, learned residual, observation, and policy

Use four interoperable layers.

**Physical/geometric transition.** Retain exact geometric constraints and a transparent mechanistic baseline. Choose an appropriate numerical approximation for each mechanism. Demonstrate stability, units, time-step behavior, conservation where applicable, and support boundaries before fitting a network. Do not require a full expensive finite-element solve in every RL step if a validated reduced model is sufficient.

**Learned transition or residual.** Fit a compact predictor of the difference between baseline and supported observed response, or a surrogate of an already-qualified solver. Store its applicability range and model uncertainty. Test absolute prediction and multi-step drift, not only training loss. A learned residual must not silently change tissue identity, regenerate removed tissue, violate mass/energy accounting, or bypass collision checks. Training against a numerical solver establishes solver approximation, not agreement with human tissue.

**Observation/belief model.** Distinguish hidden simulated state from what the planner can observe through MRI, ultrasound, navigation, microscopy, mapping, or recorded measurements. Maintain uncertainty across missing/occluded anatomy and registration changes. Information-gathering actions must change the belief state and be able to change the plan. A video world model can help perception or produce a research observation layer; it cannot overwrite authoritative anatomy or manufacture clinical labels.

**Policy and search.** Start with a learned proposal ranker, terminal-value estimator, or short-rollout model inside the existing search/MPC architecture. Evaluate a more fully latent model-based RL approach only if the simpler hybrid establishes a reason for it. Dreamer-style imagined learning and TD-MPC-style local planning are reference designs, not mandatory wholesale ports. Keep a strong nonlearned planner and a stop/withdraw fallback.

For stochastic ensembles, separate uncertainty about model parameters from variability in patient state and execution. Keep correlated consequences coupled through their common causal state. Use short rollouts when model uncertainty grows and require exact checks before accepting motions. Do not rely on visually plausible long video predictions as a safety filter.

## 5. Learn from the current failures before enlarging training

Use the preserved failures as diagnostics, not as proof that all RL is unsuitable. The saved run has eight updates on four TRAIN cases; its positive episodes and low teacher agreement warrant specific investigations, not a convergence claim.

Audit action coverage, initial access, persistent observation channels, source coverage, reward scale, STOP semantics, actor/critic interaction, gradient clipping, temporal credit, and loss of local information in the 64-cube representation. Distinguish representation bandwidth from legal information access: a search reading native geometry and an actor receiving lossy summaries are not a clean equal-representation comparison.

First establish that a small supervised learner can fit an appropriate fixed TRAIN decision set and maintain useful complete behavior. Search-generated examples are allowed but must be labeled as such, not surgeon demonstrations. Where the teacher is locally short-sighted or inefficient, copying it is only a baseline. Train recovery and information actions only from appropriately labeled observed or generated evidence.

Complete or explicitly close capped comparisons; do not score a timed-out partial beam as successful STOP. Report training, construction, proposal generation, exact checking, independent replay, and rendering costs separately. Current previews dominate runtime, so actor forward speed alone is not a deployment speedup. Measure whether the hybrid actually reduces expensive previews or improves useful decisions at a fixed overall budget.

Freeze rewards and relevant world assumptions during each comparison. Do not tune the environment to create a win and then present that as learned improvement. Preserve both blocked and nonproductive cases in denominators. Use paired conditions, multiple seeds when practical, and confidence intervals over independent patients rather than counting thousands of steps as thousands of patients.

## 6. Build an evidence-to-mechanism registry, not another passive bibliography

For every implemented or proposed mechanism, maintain a compact record containing source/DOI, publication status, accessed material, population/specimen, modality, exact observable, units, timing, action availability, uncertainty, license, intended use, relevant equation/figure/section, parameter extraction, fit split, independent validation split, and unsupported claims.

Examples of the required chain:

- A human force-relaxation measurement can constrain a constitutive response over that experiment's conditions. It cannot by itself set a neurological injury threshold.
- Before/during surgery imaging can test displacement prediction at matched landmarks. Without intervening tool actions and boundary conditions, it cannot uniquely identify how a particular maneuver caused that displacement.
- A video can support an observed instrument transition or information-gathering event. Monocular pixels and narration do not supply measured force, exact 3D pose, or an unrecorded counterfactual outcome.
- A cohort can supply event frequencies within a defined population/time window. It does not directly supply a per-action causal probability for a different patient population.

Every newly acquired source must have a named contribution and an executable test. Existing ReMIND/RESECT work should be reused before duplicating ingestion or expanding downloads without a specific use. BraTioUS is a useful newer candidate for multicenter baseline ultrasound perception, not an RL transition dataset. Cohort derivatives and challenge repackagings must not inflate patient counts.

## 7. Surgical video, transcript, and textbook program

Begin with the specific journal videos and chapters in R19–R21 and R26–R28, plus the neurosurgical simulation recordings associated with R29. Confirm licensing and permitted local processing for each actual asset before download or frame extraction. Public viewing is not blanket permission to redistribute a video, textbook, or extracted frame collection. Keep restricted source content outside Git; commit manifests and reproducible permitted processing instructions.

Create a timeline for each eligible recorded operation with observed phases, instruments, tissue interactions, navigation/imaging use, mapping observations, visible complications, responses, and stopping decisions. Retain the original narrated transcript alongside separately authored structured annotations. Record whether a statement is visible, explicitly narrated, inferred, uncertain, or unassessed.

Use event-centered before/during/after frame windows and modest background sampling. Keep timestamp, source asset hash, shot boundaries, and frame index. Video time and actual elapsed surgical time differ when footage is edited, accelerated, or cut. Never reconstruct missing duration by treating an edit as uninterrupted tissue interaction. Do not silently transform inferred 2D motion into measured 6-DoF instrument control.

For R26, the reviewed published transcript provides useful initial anchors at approximately 1:35 planning, 1:42 tractography, 3:12 exposure, 3:26 skull opening, 3:37 dura, 3:48 navigation/exoscope, 5:00 fluorescence, 6:07 hemostasis, and 6:29 postoperative assessment. Recheck alignment against the actual asset before labeling frames. Those timestamps are an extraction starting point, not evidence that the video has already been downloaded or reviewed frame by frame.

Develop a phase/action/observation/hazard/response ontology from these sources. Include interrupted actions, transitions back to information acquisition, maintained versus released retraction, and reasons to leave residual. Prefer a small high-quality annotated set over thousands of unreviewed screenshots. Use independent qualified annotations and disagreement measurement when such reviewers are available; otherwise visibly label the set as nonexpert/unreviewed and do not claim clinical validation.

Use this material for event recognition, workflow constraints, explainability, observation-model training, and appropriate demonstration learning. Admit offline RL transitions only when the actual observation-action-next observation-time-endpoint fields are supported. Do not manufacture missing actions or postoperative rewards from narration. Edited successful teaching cases cannot estimate complication incidence, and expert credentials do not make every frame optimal behavior.

## 8. Complications must have causes, clocks, and distinct endpoints

Implement an extensible event taxonomy covering local vascular injury and ischemia, venous congestion, excessive mechanical/thermal exposure, seizures/mapping interruption, edema/shift, ventricular/CSF events, postoperative hemorrhage, airway/hemodynamic disturbances, infection, hydrocephalus, and venous thromboembolism. Choose a small evidence-supported subset for the first executable module rather than claiming detailed calibration of all of them.

Separate a modeled physical event from a diagnosed clinical complication. For example, predicted vessel compression, predicted flow reduction, imaging-defined ischemia, and persistent motor impairment need different variables and different evidence. Do not display a patient-specific deficit percentage until an appropriate linked outcome model has actually been fitted and independently calibrated.

Treat randomness as conditional uncertainty, not arbitrary bad-luck events. Some variability reflects unobserved individual biology or microscopic anatomy; some reflects execution variability; some is uncertainty or error in the model itself. Repeated independent coin flips per frame are inappropriate for an operation-level complication rate. For a continuous-time hazard model, a proposed implementation is:

`P(event during [t,t+dt]) = 1 - exp(-integral(lambda(context, exposure, latent_state), t, t+dt))`.

This is a modeling choice, not an established fitted law for glioma surgery. Verify invariance to simulation time step, account for competing/correlated events and recovery, and estimate parameters only from suitable data. Oversampling rare generated events is allowed for training, but the artificial sampling frequency is not their clinical prevalence. Reweight evaluation to a justified distribution or report stress-test results separately.

The following rates are reference-cohort observations, NOT default simulator probabilities:

| Source | Population and endpoint | Reported observation | Modeling boundary |
|---|---|---|---|
| R22 | 139 people, 144 diffuse supratentorial glioma surgeries; postoperative confluent diffusion restriction | 64.6% imaging-defined confluent ischemia; new/worsened deficit 30.6% at discharge, 20.9% at first follow-up | Imaging and symptoms differ; this retrospective cohort does not define a universal risk |
| R24 | 581 analyzed awake craniotomies for space-occupying brain lesions | 29 clinically apparent intraoperative seizures, 5.0% | Mixed lesion population; procedure denominator; not an all-glioma per-stimulation rate |
| R25 | NSQIP primary malignant brain tumor operations | 30-day VTE 257/7,376, 3.5%; surgically evacuated intracranial hemorrhage 72/5,699, 1.3% | Hemorrhage denominator is the later subset with reoperation reasons; this excludes lesser hemorrhages and intraoperative bleeding |

Do not pool these endpoints. Do not interpret retrospective associations between vasopressors, anesthetic choices, operative duration, and complications as causal effects of changing those actions. Treatment selection and illness severity can confound those associations. Avoid encoding medication recommendations or doses from an observational coefficient. Unsupported infection or other event rates should stay uncalibrated until suitable evidence is obtained.

For each rate extraction record cohort dates, pathology and location, first/repeat surgery, awake/asleep, baseline deficits, event definition, denominator, follow-up, missingness, ascertainment, and uncertainty. Seek stratified calibration only where sample sizes and raw data access support it. Aggregate publications can constrain broad plausibility but rarely identify an individualized transition model.

## 9. Independent validation ladder

Do not collapse these levels into one green badge.

### A. Software and numerical verification

Verify coordinates, complete swept geometry, persistent state, conservation where relevant, irreversible removal, dependency invalidation, time-step/mesh behavior, replay determinism under a recorded stochastic draw, and the absence of privileged-information leakage. Preserve current successes, failed gates, and independent checker independence. Software tests establish implementation consistency, not biological accuracy.

### B. Measurement agreement

Fit and test specimen/device response on disjoint measurements within the stated material, loading, and acquisition conditions. Keep current failed mechanics results unchanged and issue a new version for any corrected protocol. Report identifiability, residuals, uncertainty intervals, rate/history dependence, and sensitivity to specimen condition. Phantom tests can validate engineering behavior under their own label; they are not human-tissue validation.

### C. Real patient observation prediction

Use suitable ReMIND/RESECT cases and their permitted partitions to compare predictions against withheld acquired observations. Report landmark target-registration error in millimeters, appropriate surface/cavity errors, removal-volume agreement where definitions match, and calibration of uncertainty. Do not reuse held-out landmarks to fit the deformation. Show a static/rigid baseline and a simpler mechanistic comparator.

When actions and boundary conditions are missing, report this as observation/registration validation rather than causal tool-response validation. Before/after agreement alone does not prove the predicted sequence of maneuvers would produce the observed outcome.

### D. Recorded surgical behavior

Test event timing, phase transitions, tool use, information acquisition, and visible response against source-linked recordings. Use measured pose/force only where actually available. Expert review can establish plausibility or educational construct validity; it does not establish correct complication probabilities. Repeated recordings by the same person/procedure cannot cross partitions as independent patients.

### E. Decision usefulness and generalization

Compare classical search, imitation, model-free RL, and the proposed hybrid under declared matched observations and resource budgets. Distinguish transfer to new anatomical geometry from transfer of surgical dynamics and from clinical transfer. Evaluate against an independent checker and, where available, independent measured transitions; do not let the learned world model grade its own policy. Include perturbations and counterexamples that expose model exploitation, with generated stress tests clearly labeled.

Show target removal using both full supplied target and any separately justified reachable/approved target denominator. Report non-target removal, contact, modeled mechanical/thermal exposure, vascular state violations, residual distribution, observation use, action cost, runtime, and unsupported anatomy. Zero non-target removal is not zero tissue contact or neurological harm. A justified decision to leave residual may be better than completing an inappropriate target.

### F. Clinical outcome claims

Keep these closed without appropriate linked patient observations, calibration, external evaluation, and eventually suitable prospective validation. A frozen plan evaluated on withheld scans is not an actually executed intervention. No public data source identified in this review supplies the entire needed chain of same-patient anatomy, actions, forces, evolving vascular/functional state, and alternative clinical outcomes.

## 10. Required next integrated deliverables

Use bounded work packages with ownership and review. Avoid both endless architecture work and hundreds of tiny receipt-only commits.

**Package 1: baseline and information contract.** Reconcile the current code/results, retain the best completed baseline, identify whether any new commits resolve the capped search or bandwidth mismatch, and choose one fixed development task. Define the actual input schema and hidden-reference boundary before another model comparison. Produce a compact executable regression, not a second master plan.

**Package 2: persistent operative episode.** Deliver one replayable shared-backend/desktop episode with persistent tool state, distinct action effects, an information action, and a modeled complication/response or a justified stop. A narrowly supported post-exposure episode is acceptable; its limitations must be visible. Keep unsupported physiological values null. When the scenario is generated, call it generated. When anatomy is acquired, do not relabel invented hidden structures as acquired.

**Package 3: one independent real-observation validation.** Reuse admitted paired imaging or eligible measured traces to execute a complete baseline-versus-model validation at one fidelity level. Deliver prediction, independent reference, errors, uncertainty, and a failure case. If the candidate cannot support the intended claim, narrow the claim to the observable rather than blocking all engineering or inventing measurements.

**Package 4: bounded hybrid comparison.** Fit the smallest justified residual/proposal/value model. Compare it with the retained search and learned baselines using frozen rewards and observations. Count full planning and checking costs. Evaluate held-out data only through existing role rules. Declare a go/no-go result: promote on a prespecified useful difference without worsened independent constraints; otherwise retain the stronger baseline and diagnose the failed hypothesis.

After these packages, expand the full exposure, skull-opening, vascular, functional, and physiological objectives based on the evidence ledger. Keep each next stage connected to the same runtime and app. A visually attractive standalone demo or a large new dataset catalog is not completion.

## 11. Agent coordination and implementation freedom

Use a lead integrator and focused, high-reasoning specialists for operative science/video annotation design, data/measurement validation, geometry/mechanics, world-model/RL, desktop integration, and independent evaluation. Launch additional specialists for genuinely separable subtasks, not duplicate whole-project reading. Respect actual available model settings and resource limits; do not claim unavailable subagents were used.

Each workstream needs a small input/output contract and an observable result. Serialize conflicting edits according to the existing checkout policy. Let independent reviewers challenge data support, hidden-information access, numerical assumptions, simulator exploitation, and overstated claims. Keep them independent of the scoring model they evaluate.

Select dependencies and algorithms autonomously within current permissions. Reuse modules and documentation where possible. Avoid a rigid filename prescription that causes duplicate systems. Maintain only the compact records necessary to reproduce source acquisition, model fitting, independent evaluation, and integrated execution. Commit coherent working milestones after relevant tests; preserve failed experiments without making bookkeeping the primary deliverable.

At each handoff answer: What can now be executed in the app? What changed in actual decisions? What agrees with a real measurement? What is still only a model assumption? Did the hybrid beat the relevant baseline, and at what cost? Which remaining full-supergoal capability is the next executable target?

---

# Annotated source library

Checked for this review on October 10, 2026. Publication status and access descriptions below refer to what was verified here, not to guaranteed ongoing access. An accessible paper is not automatically an accessible dataset or a license to redistribute its assets. Entries marked metadata/abstract or partial access were not treated as fully reviewed chapters, videos, or datasets. Implementation uses and proposed tests are recommendations, not results reported by the source.

## World models, RL, and surgical autonomy

### R01. LapGym: An Open Source Framework for Reinforcement Learning in Robot-Assisted Laparoscopic Surgery

**Status:** Peer-reviewed, Journal of Machine Learning Research, 2023. Official paper/abstract and framework information reviewed.

**Links:** [Paper](https://jmlr.org/papers/v24/23-0207.html), [code](https://github.com/ScheiklP/lap_gym), [SOFA application](https://www.sofa-framework.org/applications/plugins/lapgym/).

**Use:** Environment decomposition, deformable-interaction tasks, observation/action contracts, baseline comparisons. **Boundary:** Laparoscopic simulation infrastructure does not validate glioma physics. **Proposed test:** borrow an environment interface or numerical component only after comparing its relevant interaction with an independent measurement.

### R02. Hafner et al. Mastering diverse control tasks through world models

**Status:** Peer-reviewed, Nature, 2025; DreamerV3. Article material reviewed.

**Links:** [Paper](https://www.nature.com/articles/s41586-025-08744-2), [code](https://github.com/danijar/dreamerv3).

**Use:** Latent dynamics, imagined policy/value learning, and robust training design. **Boundary:** General control performance is not surgical or medical validation. **Proposed test:** small latent-model ablation after baseline dynamics and observation contracts work, not an immediate large-model replacement.

### R03. Hansen, Su, and Wang. TD-MPC2: Scalable, Robust World Models for Continuous Control

**Status:** ICLR 2024; author manuscript reviewed.

**Links:** [Manuscript](https://arxiv.org/abs/2310.16828), [methods text](https://arxiv.org/html/2310.16828v2), [project](https://tdmpc2.com).

**Use:** A particularly relevant pattern combining learned latent dynamics/value with local model-predictive control. **Boundary:** Its benchmark tasks do not establish safe surgical transfer. **Proposed test:** compare short-horizon learned proposals plus exact checking with current search at matched total planning cost.

### R04. Surgical Vision World Model (SurgWM)

**Status:** 2025 author paper; MICCAI-associated Data Engineering in Medical Imaging workshop publication, not a clinical trial. Methods text reviewed.

**Links:** [Manuscript](https://arxiv.org/abs/2503.02904), [full text](https://arxiv.org/html/2503.02904), [proceedings DOI](https://doi.org/10.1007/978-3-032-08009-7_1).

**Use:** Surgical visual representation and latent action-conditioned prediction. **Boundary:** Inferred latent actions are not measured instrument commands or force trajectories. **Proposed test:** evaluate observation prediction separately from physical transition prediction.

### R05. SurgWorld

**Status:** December 2025 preprint; reviewed version v2. No journal acceptance was established in this review.

**Link:** [Full text](https://arxiv.org/html/2512.23162v2).

**Use:** Surgical video generation, procedure annotation, and the use of an embodiment-specific inverse dynamics model to obtain policy-training signals. **Boundary:** Its benchtop robotic needle manipulation evaluation is not glioma resection; expert ratings of generated clips are not clinical success rates. **Proposed test:** keep inferred actions explicitly estimated and compare them with held-out measured robot actions before policy use.

### R06. SAW: Toward a Surgical Action World Model via Controllable and Scalable Video Generation

**Status:** March 2026 preprint; author abstract and reported conditioning approach reviewed. No journal publication verified.

**Link:** [Paper](https://arxiv.org/abs/2603.13024).

**Use:** Conditioning visual prediction on tool trajectories, language, masks, and reference observations. **Boundary:** 2D action control and depth consistency are not proof of correct 3D tissue mechanics. **Proposed test:** challenge predictions with contact/no-contact pairs and independent geometry rather than visual preference alone.

### R07. SurgVista: Long-Horizon Surgical World Modeling with Plausible Instrument-Tissue Dynamics

**Status:** June 2026 preprint; abstract-level review, not independently reproduced.

**Link:** [Paper](https://arxiv.org/abs/2606.19889).

**Use:** Its stated problems of inconsistent instrument-tissue response and long-horizon drift motivate explicit failure tests. **Boundary:** Better visual temporal consistency does not calibrate tissue damage or perfusion. **Proposed test:** track physical state and conservation errors over increasingly long rollouts, independently of video quality.

### R08. SRT-H: A Hierarchical Framework for Autonomous Surgery via Language-Conditioned Imitation Learning

**Status:** Peer-reviewed, Science Robotics, 2025; publication and author manuscript verified.

**Links:** [Journal](https://www.science.org/doi/10.1126/scirobotics.adt5254), [author paper](https://arxiv.org/abs/2505.10251), [demonstrations](https://h-surgical-robot-transformer.github.io/).

**Use:** Hierarchical action organization and recovery behavior. **Boundary:** Imitation learning on an ex vivo gallbladder task, not RL, living-patient autonomy, or brain surgery. **Proposed test:** separate high-level phase choice from interruptible low-level execution and verify recovery transitions.

### R09. Learning Autonomous Surgical Irrigation and Suction with the da Vinci Research Kit Using Reinforcement Learning

**Status:** 2024 author preprint version reviewed; later publication status not verified.

**Link:** [Methods and limitations](https://arxiv.org/html/2411.14622v1).

**Use:** Fluid-task decomposition, curricula, and simulation-to-benchtop evaluation. Its reported suction mismatch is especially relevant: simulated attraction and real pressure-driven liquid removal are not interchangeable. **Boundary:** The physical experiments use benchtop materials, not human brain tissue. **Proposed test:** validate tool-effect locality and fluid accounting before interpreting a trained fluid-control policy as hemostasis competence.

### R10. Alotaibi et al. Assessing bimanual performance in brain tumor resection with NeuroTouch

**Status:** Peer-reviewed neurosurgical simulation study, 2015; abstract reviewed.

**Link:** [PubMed](https://pubmed.ncbi.nlm.nih.gov/25599201/), DOI `10.1227/NEU.0000000000000631`.

**Use:** Tissue removal, force, blood loss, path length, and coordination metrics for comparing surgeon/resident behavior. **Boundary:** Expert-novice discrimination is construct validity, not proof of patient outcome prediction. **Proposed test:** include several interpretable behavior metrics rather than rewarding tumor volume alone.

## Mechanics and real observations

### R11. Flaschel et al. Automated discovery of interpretable hyperelastic material models for human brain tissue with EUCLID

**Status:** Peer-reviewed, Journal of the Mechanics and Physics of Solids, 2023. Author full text and publication metadata reviewed.

**Links:** [Article DOI](https://doi.org/10.1016/j.jmps.2023.105404), [author text](https://arxiv.org/html/2305.16362v1), [code portal](https://euclid-code.github.io/), [software record](https://doi.org/10.5905/ethz-1007-638).

**Use:** Interpretable constitutive-law discovery from human specimen experiments. **Access limit:** The paper states experimental data are available on request; a code release is not a public raw-data release. **Boundary:** Hyperelastic fitting alone does not validate cutting, time dependence, or injury. **Proposed test:** reconcile existing acquired traces, then assess held-out loading modes and identifiable parameters.

### R12. Budday et al. Viscoelastic parameter identification of human brain tissue

**Status:** Peer-reviewed, Journal of the Mechanical Behavior of Biomedical Materials, 2017; institutional metadata/abstract verified.

**Links:** [DOI](https://doi.org/10.1016/j.jmbbm.2017.07.014), [author institution](https://cris.fau.de/publications/118134104/).

**Use:** Rate- and history-dependent response rather than a single stiffness number. **Boundary:** Specimen condition, loading regime, and time after acquisition constrain transfer to living surgery. **Proposed test:** held-out relaxation/history prediction, not merely a good fit to the calibration curve. Confirm raw-data rights and availability before acquisition.

### R13. ReMIND: brain resection multimodal imaging data

**Status:** Scientific Data, 2024; primary descriptor metadata and source documentation verified. Some publisher/source fetches were unavailable during this review.

**Links:** [Descriptor](https://www.nature.com/articles/s41597-024-03295-z), [TCIA collection](https://www.cancerimagingarchive.net/collection/remind/).

**Use:** Paired preoperative/intraoperative imaging, anatomy updates, registration, and observed resection context. Existing repository acquisition and splits take precedence. **Boundary:** Per-patient modality/stage coverage varies; imaging does not supply complete tool actions, forces, or alternative outcomes. **Proposed test:** one independently held acquired-observation prediction with explicit missingness.

### R14. RESECT: retrospective evaluation of cerebral tumors

**Status:** Peer-reviewed Medical Physics dataset publication, 2017; source descriptor identified and existing project provenance retained.

**Link:** [DOI](https://doi.org/10.1002/mp.12268).

**Use:** Preoperative MRI and intraoperative ultrasound for brain-shift/registration evaluation. **Boundary:** Do not infer continuous surgical dynamics from sparse stage images. **Proposed test:** landmark prediction with explicit fit/evaluation separation and a rigid baseline. Reuse current prepared cases; preserve exposed Case4 as development rather than fresh validation.

### R15. RESECT-SEG

**Status:** Peer-reviewed Medical Physics dataset extension, 2024; descriptor information reviewed.

**Link:** [Paper](https://aapm.onlinelibrary.wiley.com/doi/10.1002/mp.17317).

**Use:** Tumor, cavity, and anatomical annotations for relevant ultrasound stages; 23 low-grade-glioma cases in the source cohort. **Boundary:** An extension of RESECT, not an independent new population. **Proposed test:** quantify stage-specific segmentation/cavity agreement and annotation uncertainty rather than treating labels as perfect physical ground truth.

### R16. ReMIND2Reg

**Status:** 2025 challenge report/preprint, reviewed author text.

**Link:** [Paper](https://arxiv.org/html/2508.09649v1).

**Use:** Registration task organization and held-out challenge-style evaluation. **Boundary:** Reuses ReMIND subjects; do not count it as another independent cohort or assume private test data are accessible. **Proposed test:** adopt appropriate benchmark separation without disturbing existing project patient roles.

### R17. BraTioUS: A multicenter dataset of baseline intraoperative brain tumor ultrasound images

**Status:** Peer-reviewed Data in Brief, 2026; PubMed abstract/metadata reviewed.

**Links:** [PubMed](https://pubmed.ncbi.nlm.nih.gov/41657404/), [DOI](https://doi.org/10.1016/j.dib.2026.112478), [dataset DOI cited by the paper](https://doi.org/10.5281/zenodo.16887363).

**Use:** Baseline ultrasound perception across 142 glioma patients, with 1,669 images from six hospitals in five countries. **Boundary:** Baseline 2D images, not longitudinal action-response trajectories. **Proposed test:** site-held-out tumor delineation and uncertainty. Resolve current version, licensing, and overlap with related publications before admitting data.

## Clinical definitions, complications, and textbooks

### R18. EANS-EANO guidelines on the extent of resection in gliomas

**Status:** Neuro-Oncology, 2026 issue, online 2025. Author-institution publication record/abstract verified; full recommendations require section-level reading before implementation.

**Links:** [Journal](https://academic.oup.com/neuro-oncology/article/28/1/38/8256732), [PubMed](https://pubmed.ncbi.nlm.nih.gov/40973061/), [author institution](https://cris.tau.ac.il/en/publications/eans-eano-guidelines-on-the-extent-of-resection-in-gliomas/), DOI `10.1093/neuonc/noaf217`.

**Use:** Define appropriate extent-of-resection endpoints and clinical scope. **Boundary:** Guidelines are not a voxelwise injury model or optimal action trajectory. **Proposed test:** ensure task targets, functional preservation, and residual reporting match the stated disease/clinical setting.

### R19. Nanda, editor. Complications in Neurosurgery, chapter 20

**Status:** Legitimate Elsevier sample chapter reviewed, including rendered pages.

**Link:** [Publisher sample PDF](https://www.us.elsevierhealth.com/media/wysiwyg/us/pdf/sample-chapter-9780323509619.pdf).

**Scope:** “Primary Brain Lesion Resection Complications: An Overview and Malignant Brain Swelling After Resection of Superior Sagittal Sinus Meningioma,” pp. 99–106.

**Use:** Complication taxonomy and mechanisms, including venous-outflow consequences. **Boundary:** The illustrative meningioma case is not a glioma incidence cohort. **Proposed test:** source-bound event definitions and separate edema/venous-congestion states. Link the chapter rather than redistributing it.

### R20. Jackson, Westphal, and Quiñones-Hinojosa. Complications of glioma surgery

**Status:** Handbook of Clinical Neurology, volume 134, Gliomas, 2016, pp. 201–218. Metadata/abstract verified; full chapter not accessed here.

**Links:** [DOI](https://doi.org/10.1016/B978-0-12-802997-8.00012-8), [author institution](https://pure.johnshopkins.edu/en/publications/complications-of-glioma-surgery).

**Use:** Priority lawful textbook acquisition for glioma-specific mechanisms and delayed complications. **Boundary:** Do not claim page-level support or numeric thresholds until the relevant chapter text is actually obtained and read. Preserve unavailable-access status rather than inventing quotations.

### R21. Zakaria and Prabhu. Cortical Mapping in Resection of Malignant Cerebral Gliomas

**Status:** Chapter in Glioblastoma, 2017. Existing-supergoal source; NCBI access was blocked during this review, so full-text review is pending.

**Link:** [NCBI Bookshelf](https://www.ncbi.nlm.nih.gov/books/NBK470008/).

**Use:** Mapping workflow, observation interpretation, and stopping logic. **Boundary:** Reading a mapping chapter does not turn tract proximity into a calibrated deficit probability. **Proposed test:** an observed information event can change a plan without exposing hidden functional truth to the actor.

### R22. van der Boog et al. Occurrence, Risk Factors, and Consequences of Postoperative Ischemia After Glioma Resection

**Status:** Neurosurgery, 2023; retrospective primary study, abstract reviewed.

**Link:** [PubMed](https://pubmed.ncbi.nlm.nih.gov/36135366/), DOI `10.1227/neu.0000000000002149`.

**Use:** Distinguish imaging-defined ischemia from new/worsened symptoms and their assessment times; reference-cohort rates are in section 8. **Boundary:** Associations do not identify causal effects of anesthetic management or justify patient-specific probabilities. **Proposed test:** maintain separate imaging, symptom, timing, and population fields in the risk registry.

### R23. Berger et al. Incidence and impact of stroke following surgery for low-grade gliomas

**Status:** Journal of Neurosurgery, 2021; primary cohort publication/abstract identified.

**Links:** [PubMed](https://pubmed.ncbi.nlm.nih.gov/31881532/), [DOI](https://doi.org/10.3171/2019.10.JNS192301).

**Use:** A complementary lower-grade cohort for examining endpoint/case-mix dependence and recovery. **Boundary:** Do not pool it with another ischemia cohort merely because both use postoperative diffusion imaging. **Proposed test:** compare definitions and covariates before any incidence model; complete full-text extraction before parameterization.

### R24. Paquin-Lanthier et al. Risk Factors and Characteristics of Intraoperative Seizures During Awake Craniotomy

**Status:** Journal of Neurosurgical Anesthesiology, 2023 issue; retrospective study, abstract reviewed.

**Link:** [PubMed](https://pubmed.ncbi.nlm.nih.gov/34411059/), DOI `10.1097/ANA.0000000000000798`.

**Use:** Intraoperative event definition, procedure denominator, and whether an event interrupts mapping. **Boundary:** Space-occupying brain lesions are not exclusively gliomas; anesthesia associations are not randomized treatment effects. **Proposed test:** distinguish transient mapping interruption from persistent deficit and separate event onset from severity.

### R25. Senders et al. Venous thromboembolism and intracranial hemorrhage after craniotomy for primary malignant brain tumors

**Status:** Journal of Neuro-Oncology, 2018; open-access primary NSQIP study. Methods/results text reviewed.

**Links:** [Full text](https://link.springer.com/article/10.1007/s11060-017-2631-5), [PubMed](https://pubmed.ncbi.nlm.nih.gov/29039075/).

**Use:** Time-to-event definitions and different denominators for thrombosis and surgically evacuated hemorrhage, as recorded in section 8. **Boundary:** Not all bleeding qualifies as this hemorrhage endpoint, and registry records are not a public synchronized surgery-imaging dataset. **Proposed test:** separate acute operative mechanisms from delayed outcome monitoring and avoid repeated per-frame sampling of a 30-day incidence.

## Specific surgical videos and expertise evaluation

### R26. Schupper, Roa, and Hadjipanayis. Contemporary intraoperative visualization for GBM with use of exoscope, 5-ALA fluorescence-guided surgery and tractography

**Status:** Neurosurgical Focus: Video, 2022. Open-access article and timestamped published transcript reviewed; video frames were not downloaded or analyzed here.

**Links:** [Article/transcript](https://pmc.ncbi.nlm.nih.gov/articles/PMC9555356/), [video](https://stream.cadmore.media/r10.3171/2021.10.FOCVID21174), DOI `10.3171/2021.10.FOCVID21174`.

**Use:** Source-linked end-to-end phase/observation annotation using section 7's initial anchors. **Boundary:** This is one edited teaching case, not continuous action metrology or incidence evidence. Article CC BY terms were visible; verify the actual video asset's terms separately before extraction or reuse.

### R27. Brain mapping for lower-grade glioma around Wernicke’s area

**Status:** Neurosurgical Focus: Video, 2025. Publication and video target verified; complete video/frame review pending.

**Links:** [PubMed](https://pubmed.ncbi.nlm.nih.gov/39845308/), [article](https://pmc.ncbi.nlm.nih.gov/articles/PMC11748949/), [video](https://stream.cadmore.media/r10.3171/2024.10.FOCVID24101).

**Use:** Language-related information and boundary decisions as a counterpart to a volume-only removal task. **Boundary:** A narrated mapping case cannot provide an individualized language-risk field for a different patient. **Proposed test:** annotate only supported observations and their decision consequences.

### R28. Arcuate fasciculus cortico-cortical evoked potentials and direct cortico-subcortical stimulation during awake craniotomy for debulking of left dominant temporal oligodendroglioma

**Status:** Neurosurgical Focus: Video, 2025. Publication metadata/source target verified; full content access was incomplete here.

**Links:** [Article](https://pmc.ncbi.nlm.nih.gov/articles/PMC11748948/), [publisher video article](https://thejns.org/video/view/journals/neurosurg-focus-video/12/1/article-pV5.xml).

**Use:** Functional observations and debulking decisions. **Boundary:** Do not infer missing stimulation settings, exact forces, or unrecorded outcomes. **Proposed test:** after lawful review, encode an observed measurement-to-decision transition with explicit uncertainty and source timestamp.

### R29. Yilmaz et al. Continuous monitoring of surgical bimanual expertise using deep neural networks in virtual reality simulation

**Status:** Peer-reviewed npj Digital Medicine, 2022; full article text reviewed.

**Link:** [Article and supplementary videos](https://www.nature.com/articles/s41746-022-00596-8).

**Use:** Continuous interpretable metrics for force, motion, tissue removal, bleeding control, and two-handed coordination. **Boundary:** These are human performances in NeuroVR, not measured real patient surgery; expertise scoring and clinical outcome validity remain different. **Proposed test:** evaluate event-level behavior rather than only final return, and treat human simulator recordings as their own evidence class.

### R30. Giglio et al. Artificial Intelligence–Augmented Human Instruction and Surgical Simulation Performance: A Randomized Clinical Trial

**Status:** Peer-reviewed JAMA Surgery, 2025; article verified and reviewed as an educational simulation study.

**Link:** [Article](https://jamanetwork.com/journals/jamasurgery/fullarticle/2837234).

**Use:** Design of human-in-the-loop educational evaluation and feedback studies. **Boundary:** A randomized educational study does not validate an autonomous surgical policy or demonstrate improved outcomes in operated patients. **Proposed test:** reserve an eventual blinded expert/learner evaluation as a separate endpoint rather than presenting attractive replay as clinical usefulness.

---

## Final acceptance rule

A successful next cycle is not “we integrated Dreamer” or “we downloaded more surgical videos.” It is an executable surgical-state improvement, a specific prediction checked against an appropriate real observation, and a transparent result showing whether a small hybrid improves the relevant decision or cost. Preserve negative findings. Keep the full supergoal, but make every new claim point to an independently inspectable measurement.
