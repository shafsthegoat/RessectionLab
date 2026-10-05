# /goal — Transform RessectionLab into a MEDiVIS-caliber patient-specific surgical strategy, neurological-harm, robustness, and learning research platform

Continue directly on `codex/patient-specific-planner` in `shafsthegoat/RessectionLab`.

Do not create a new branch or worktree unless a genuine repository constraint makes it unavoidable. Reconcile this goal against current HEAD before acting. At the time this goal was assembled, the branch head was:

`e477f5e6b00427c3453a7bf8e2d577fc862cd82e`

The branch has moved substantially since the original planning review. Preserve all subsequent work, all useful current research directions, all negative results, all patient-role assignments, and all unrelated changes.

This is a long-horizon research and engineering goal. It is intentionally broad. Treat the details below as a research charter, source of hypotheses, and set of scientific boundaries, not as a brittle checklist that prevents better ideas. The integration owner and specialized subagents have substantial autonomy to reorder work, combine phases, abandon weak ideas, introduce better methods, acquire additional public evidence, or change implementation details when the evidence justifies doing so.

The objective is not to accumulate commits, maximize test count, reproduce papers mechanically, or force reinforcement learning to win.

The objective is to turn the accumulated work into a genuinely technically serious, scientifically defensible, visually compelling research platform that would make sense to engineers and researchers at MEDiVIS and that the project owner can be proud of.

The final system remains a **research prototype**. It is not a medical device, autonomous surgeon, clinical recommendation engine, robot-control system, or validated predictor of postoperative neurological deficit unless a specific component eventually earns a much narrower claim through appropriate evidence.

---

# 0. The central thesis

RessectionLab should become a **patient-specific surgical strategy and robustness research engine for glioma resection**.

The system should answer a much harder question than:

> What path reaches the tumor?

It should investigate:

> Given the patient's actual imaging and evidence, the available instruments, the current surgical state, uncertainty in anatomy and registration, and what is known about neurological function, what resection strategies appear feasible, what might they remove, what functional systems might they endanger, how sensitive are those conclusions to uncertainty and changing anatomy, and what evidence would cause us to change the plan?

The project should eventually compare multiple strategies that differ meaningfully in:

- access location,
- approach direction,
- complete instrument geometry,
- target subregion,
- instrument choice,
- sequential opening and removal choices,
- state of the cavity,
- functional exposure,
- robustness to uncertainty,
- and computational burden.

The planner should preserve rejected strategies and explain why they fail.

The core planning outputs should remain inspectable and should include, where evidence supports them:

- complete instrument sequence,
- source imaging and coordinate frame,
- source evidence and provenance,
- modeled target removal,
- modeled residual tumor,
- modeled non-target removal,
- partial contact,
- complete-tool clearance,
- functional-evidence encounters,
- neurological-harm estimates or surrogates only when validated,
- uncertainty model,
- robustness or fragility of the strategy,
- first failing structure,
- first failing instrument component,
- first failing sequence step,
- unknown or unassessed evidence,
- computational cost,
- and the conditions under which the strategy would need to be re-evaluated.

Do not reduce everything to a single opaque "safety score."

Do not output fake percentages such as "82% safe."

Separate the measurable components and preserve uncertainty.

---

# 1. The most important correction: neurological harm is a first-class objective

The current real-patient learning experiment does **not** yet optimize what the project ultimately cares about.

The recent PAT05 learning work explicitly uses a geometric objective. The current bounded reward mainly reflects:

- modeled target tissue removed,
- modeled other or non-target tissue removed,
- procedural action costs,
- and related geometric quantities.

Motor, language, vascular, cognitive, and other patient-function outcomes are not currently part of the real PAT05 learning reward.

This is not a minor omission.

The point of "maximal safe resection" is not:

> remove as much tumor as possible while removing as little arbitrary non-tumor volume as possible.

One cubic millimeter of tissue is not equivalent to another cubic millimeter of tissue.

A small injury to a critical language, motor, visual, memory, attention, executive, or network hub may matter far more than a larger amount of tissue elsewhere. Ischemic injury in retained tissue may matter even when that tissue was never intentionally "removed." A structurally preserved region can still lose function through network disconnection.

Therefore, **ordinary normal-tissue volume must remain a geometric cost and baseline, but it must not be mistaken for the final neurological-harm objective.**

The research program should explicitly develop a better functional-harm representation and, only after that representation survives appropriate validation, make it available to **both search and learning under the same rules**.

Do not simply add a hand-tuned "neurological penalty" tomorrow and call the problem solved.

First establish what can actually be learned or measured.

---

# 2. A hierarchy of neurological-harm evidence

Treat neurological harm as a hierarchy of increasingly difficult evidence rather than one magical risk number.

## Layer A: geometric and anatomical exposure

This includes things the current planner can reason about relatively directly:

- complete-tool contact,
- non-target removal,
- partial contact,
- distance to supplied structures,
- tract or atlas encounters,
- cavity chronology,
- shaft conflict,
- functional-map coverage,
- and missing evidence.

These are useful planning quantities.

They are **not neurological deficits**.

## Layer B: imaging-observed injury

Where paired postoperative imaging is available, investigate measurable postoperative consequences such as:

- resection cavity,
- residual tumor,
- peri-resection diffusion abnormalities,
- apparent diffusion coefficient abnormalities,
- ischemic injury,
- deformation,
- tissue displacement,
- new structural disconnection,
- network disruption,
- and other image-derived injury representations that are scientifically defensible.

This layer is closer to biological harm than arbitrary normal-tissue volume.

It still does not uniquely determine functional loss.

## Layer C: observed neurological and cognitive outcomes

Where public cohorts provide them, model actual measured outcomes such as:

- new motor deficit,
- language deficit,
- visual deficit,
- sensory deficit,
- cognitive decline,
- independence or performance-status change,
- transient versus persistent deficits,
- severity,
- recovery,
- and assessment timing.

These outcomes are what the project ultimately wants to preserve.

However, observational outcomes from one actual surgery per patient do not magically reveal what every unperformed alternative route would have caused.

That counterfactual limitation must remain explicit.

---

# 3. Neurological outcome data to investigate

The following public sources are now high-priority research leads because they move the project closer to meaningful neurological outcomes.

Do not assume every modality or field is available for every patient. Audit files, identifiers, timing, licenses, missingness, and patient overlap before training.

## 3.1 RHUH-GBM

Primary collection:

https://www.cancerimagingarchive.net/collection/rhuh-gbm/

Dataset paper:

https://arxiv.org/abs/2305.00005

TCIA DOI:

https://doi.org/10.7937/4545-c905

RHUH-GBM contains 40 adult glioblastoma patients with imaging at:

- preoperative,
- early postoperative within 72 hours,
- and recurrence/follow-up.

The collection includes structural MRI and diffusion-derived ADC maps, expert-corrected tumor-region segmentations, clinical variables, extent of resection, preoperative and postoperative KPS, and postoperative neurological-deficit status.

The dataset paper reports the following postoperative neurological categories:

- 26 patients with no deficit,
- 6 with transient deficit,
- 6 with minor persistent deficit,
- 2 with major persistent deficit.

This is one of the strongest immediate public leads for relating actual surgery-associated imaging changes to observed neurological harm.

Important limitations:

- only 40 patients,
- severe class imbalance,
- only two major persistent cases,
- the cohort is selected for gross or near-total enhancing-tumor resection,
- most patients therefore occupy a restricted surgical-outcome distribution,
- observational associations are not counterfactual route effects,
- clinical outcome timing and exact definitions must be inspected carefully,
- raw DICOM access has restrictions while the public brain-extracted NIfTI/segmentations and clinical CSV have different access terms.

Do not report raw accuracy alone.

A classifier that predicts "no deficit" for every patient already gets 65% accuracy.

Use class-specific sensitivity, balanced metrics, uncertainty, calibration, and exact patient counts.

Investigate whether early postoperative imaging, especially diffusion/ADC abnormalities and anatomically localized postoperative change, explains observed deficits beyond crude resection volume.

## 3.2 Brain Tumor Connectomics pre/postoperative cohort

Dataset paper:

https://www.nature.com/articles/s41597-022-01806-4

Preoperative OpenNeuro dataset:

https://openneuro.org/datasets/ds001226

Postoperative OpenNeuro dataset:

https://openneuro.org/datasets/ds002080

The original preoperative dataset contains:

- 11 glioma patients,
- 14 meningioma patients,
- 11 controls.

It includes:

- T1 MRI,
- diffusion MRI,
- resting-state fMRI,
- structural connectivity,
- functional connectivity,
- tumor masks,
- and standardized cognitive assessments.

The OpenNeuro documentation identifies CANTAB tasks including:

- MOT,
- RVP,
- RTI,
- SSP,
- and SOC,

covering aspects of attention, reaction or processing speed, working memory/span, and planning or executive function.

The postoperative dataset contains a subset, including 7 glioma patients, assessed approximately six months after surgery.

This dataset is especially attractive because RessectionLab already uses BTC anatomy.

Audit exact pre/post patient pairings, cognitive missingness, pathology eligibility, image modalities, and acquisition timing before making it an outcome-training source.

Use it to investigate **cognitive change**, not just categorical neurological deficit.

Do not mix meningioma and glioma automatically.

Do not treat follow-up cognitive change as exclusively caused by resection. Recovery, plasticity, disease progression, treatment, medication, and other factors may contribute.

## 3.3 Presurgical structural-connectivity and cognitive-outcome cohort

Primary paper:

https://pubmed.ncbi.nlm.nih.gov/41140809/

Associated public data:

https://doi.org/10.6084/m9.figshare.c.7578326.v1

This 63-patient glioma study used presurgical structural connectivity plus baseline variables to predict postsurgical cognitive impairment.

Reported AUCs were approximately:

- 0.69 for baseline variables,
- about 0.73 for structural-network variables,
- and about 0.75 to 0.76 for network plus baseline information.

Treat those as evidence that presurgical network structure may contain useful predictive signal.

They are not "76% surgical accuracy."

Raw patient MRI is not necessarily public with the associated processed data.

This cohort may be useful for:

- validating network features,
- testing predictive representations,
- studying DMN/FPN vulnerability,
- and building an outcome-model component that later maps a simulated disconnection pattern into a neurological-harm surrogate.

Do not pretend it directly supplies spatial surgical trajectories.

## 3.4 Aphasia Recovery Cohort

Paper:

https://www.nature.com/articles/s41597-024-03819-7

OpenNeuro:

https://doi.org/10.18112/openneuro.ds004884.v1.0.1

The first release includes 230 chronic stroke survivors across 902 imaging sessions with multimodal imaging and behavioral measurements.

This is **not a glioma surgery cohort**.

Its value is different.

It may provide auxiliary supervision or representation learning for relationships among:

- lesion location,
- disconnection,
- language networks,
- structural connectivity,
- functional connectivity,
- and measured language impairment.

Any representation transferred from stroke to tumor surgery requires separate glioma validation.

Do not use stroke labels as if they were direct surgical-outcome labels.

## 3.5 Postoperative DWI and surgically acquired deficits

Primary study:

https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0101805

This matched glioma study compared 42 patients with acquired postoperative dysphasia and/or motor deficits with 42 matched controls.

Postoperative peri-tumoral diffusion abnormalities were observed more often in patients with deficits than controls, approximately:

- 63% in cases with acquired deficits,
- 41% in controls.

This is useful evidence that postoperative ischemic injury can be related to neurological harm.

It also proves the opposite of a simplistic assumption:

**an imaging abnormality is not equivalent to a deficit**, because many control patients without deficits also had abnormalities.

Use postoperative DWI or ADC as an intermediate injury endpoint where available, not as a direct deficit probability.

## 3.6 NANO

Primary publication:

https://pubmed.ncbi.nlm.nih.gov/28453751/

The Neurologic Assessment in Neuro-Oncology scale provides structured ratings over nine neurological domains.

Use NANO as a source for thinking about a structured neurological-outcome schema.

Do not assume every public dataset contains NANO.

Do not fabricate NANO values from insufficient records.

If narrative clinical notes are eventually available under appropriate terms, an LLM may help extract candidate observations for human review, but:

- "not mentioned" must remain unknown,
- extraction is not ground truth,
- and generated plausible labels are forbidden.

---

# 4. Define the neurological outcome representation before training a harm model

Create a general neurological-outcome schema that can represent heterogeneous cohorts without pretending they are identical.

At minimum distinguish:

- patient ID,
- visit or time point,
- preoperative baseline,
- postoperative assessment time,
- domain affected,
- severity,
- transient versus persistent,
- recovery status,
- source of label,
- structured scale if available,
- cognitive test if available,
- missingness,
- and uncertainty or ambiguity.

Potential domains include:

- motor strength,
- language,
- visual function,
- sensation,
- gait/coordination,
- cognition,
- attention,
- processing speed,
- working memory,
- executive function,
- global functional status,
- independence,
- and other documented neurological domains.

Do not collapse all domains into one label until the consequences of doing so are measured.

The system may eventually use a multi-task or hierarchical outcome model.

For example, it may predict:

- probability or ordinal score for new neurological deficit,
- domain-specific impairment,
- persistence,
- cognitive decline,
- KPS change,
- or a latent harm representation.

But the output language must match what the data actually support.

---

# 5. Do not confuse association with counterfactual surgical harm

This is one of the hardest scientific problems in the entire project.

Each real patient usually supplies:

- one actual anatomy,
- one actual operation,
- one postoperative imaging trajectory,
- and one observed outcome.

The data do **not** reveal what would have happened if a different route had been used.

Therefore:

A model trained on real surgery/outcome pairs is primarily an **observational association model** unless a stronger causal design is justified.

Do not automatically call its output:

- causal injury probability,
- counterfactual deficit probability,
- or patient-specific outcome under an unperformed route.

The project should investigate ways of converting real outcome evidence into useful planning signals while preserving this limitation.

Possible research directions include:

- outcome prediction from actual observed injury footprints,
- network-disconnection models,
- lesion-network representations,
- postoperative DWI or ischemia as an intermediate target,
- outcome models conditioned on baseline function,
- ordinal deficit models,
- multi-task domain prediction,
- calibrated uncertainty,
- causal-sensitivity analysis,
- and counterfactual simulation only where assumptions are explicit.

Subagents are encouraged to investigate stronger causal formulations if the data actually support them.

Do not manufacture causal identifiability with terminology.

---

# 6. A practical route from outcomes to a planning objective

The outcome work should proceed in stages.

The order below is a conceptual ladder, not a rigid schedule.

## 6.1 Learn what observed injury correlates with observed harm

Begin from actual surgeries.

Build representations of observed postoperative changes such as:

- resection cavity,
- residual tumor,
- diffusion abnormality,
- ischemic injury,
- structural disconnection,
- network disruption,
- and relevant anatomical changes.

Compare these with neurological and cognitive outcomes.

Use simple, regularized baselines before complex deep models.

A strong result would show that anatomical or network injury features predict observed harm better than simple total non-target volume.

## 6.2 Establish patient-level held-out performance

Keep every visit from a patient together.

Use patient-level splits.

Prevent postoperative information from entering a model that claims to be preoperative.

For classification or ordinal prediction, report metrics appropriate to severe class imbalance.

For continuous cognitive or functional scores, report errors and uncertainty.

Evaluate calibration.

Report confidence intervals where meaningful.

Do not select a model solely from overall accuracy.

## 6.3 Create a simulator-to-outcome interface

Only after an outcome model is credible should the planner experiment with feeding it a **modeled damage representation** from a hypothetical strategy.

The mapping might include things such as:

- modeled removed tissue,
- modeled ischemic-risk surrogate if one exists,
- tract disruption,
- network disconnection,
- function-map exposure,
- or simulated postoperative geometry.

The exact representation should be discovered experimentally.

Do not assume the current voxel-removal mask is the right sufficient statistic.

## 6.4 Measure distribution shift

A hypothetical plan generated by the simulator may produce damage patterns unlike those observed in the training cohort.

The harm model must detect or expose out-of-distribution use where practical.

A plan outside the support of the outcome data should not receive a confident neurological-risk number.

## 6.5 Give search and RL the same harm signal

Once a harm estimator or surrogate earns its place, integrate it symmetrically.

SEARCH, robust SEARCH, policy-only methods, learned-guided search, and RL should all receive the same allowed harm model under the same information contract.

Do not give RL a rich neurological signal while leaving search with tissue volume, or vice versa.

---

# 7. The future reward should represent the real goal without becoming an opaque hack

The eventual decision objective should reflect **maximal useful resection under neurological and other harm constraints**, but do not rush to one arbitrary weighted sum.

Preserve multiple components.

A future planning objective may include:

### Benefit

- modeled target removal,
- reduced residual target burden,
- coverage of relevant target compartments,
- or other evidence-backed oncologic/resection benefits.

### Neurological harm

Only once validated enough for research use:

- predicted or surrogate domain-specific deficit,
- persistent-deficit burden,
- cognitive-harm surrogate,
- network-disconnection cost,
- functional-map exposure,
- or calibrated model-conditioned outcome risk.

### Other tissue/injury costs

- non-target removal,
- partial contact,
- ischemia surrogate,
- vascular exposure where evidence exists,
- and other modeled injury components.

### Robustness

- plan fragility,
- probability of geometric invalidity under a declared uncertainty model,
- tail exposure,
- and evidence missingness.

### Procedure burden

- action count,
- instrument changes,
- travel distance,
- or other meaningful procedural costs.

Keep these components separately visible.

A scalar reward may be required for some algorithms, but the scientific report must show its components.

Experiment with Pareto or constrained formulations rather than assuming all clinical goals should be linearly weighted.

Possible research formulations include:

- constrained optimization,
- lexicographic objectives,
- Pareto planning,
- risk-sensitive objectives,
- CVaR,
- distributionally robust planning,
- or learned utility models.

Use them only if they solve a demonstrated problem.

---

# 8. Why this matters specifically for MEDiVIS

Study MEDiVIS before making product decisions.

## 8.1 MEDiVIS Studio

Official site:

https://www.medivis.com/studio

Studio already covers core medical-image visualization and planning functionality such as:

- multimodal imaging,
- MPR,
- image fusion,
- segmentation,
- 3D reconstruction,
- case planning,
- and saved trajectories.

Therefore:

**MRI viewing, tumor segmentation, 3D rendering, and drawing a path are not the differentiator.**

RessectionLab's viewer is the interface to a deeper research engine.

## 8.2 MEDiVIS Cranial Navigation

Official site:

https://www.medivis.com/navigation/cranial

MEDiVIS Cranial Navigation connects patient imaging and planned trajectories to the operating-room spatial workflow.

Therefore RessectionLab should care deeply about:

- physical coordinate frames,
- registration,
- complete instrument geometry,
- calibration,
- stale planning information,
- updated anatomy,
- millimeter-level tolerances,
- instrument state,
- and plan validity after change.

## 8.3 MEDiVIS Frontier Agents

Official page:

https://www.medivis.com/frontier/agents

The Maia direction is highly relevant.

A model may reason over the case, but validated computational tools should own exact calculations.

Design RessectionLab's backend so that a future agent could conceptually invoke typed operations such as:

- evaluate plan,
- compare plans,
- calculate robustness,
- replay a sequence,
- apply an observation update,
- or explain a geometric failure.

Do not make an LLM responsible for geometry correctness.

Do not claim actual MEDiVIS API compatibility unless a documented integration exists.

## 8.4 MEDiVIS Frontier Robotics

Official page:

https://www.medivis.com/frontier/robotics

The public direction emphasizes surgeon-approved plans, common navigation state, and bounded increases in autonomy.

RessectionLab should therefore be framed as research into:

- which plans deserve inspection,
- which plans are fragile,
- what evidence changes the recommendation,
- and what computational tool can support the surgeon's decision.

Do not build autonomous-surgeon theater.

## 8.5 MEDiVIS engineering culture

The current robotics hiring material emphasizes:

- coordinate frames,
- calibration,
- external tracking,
- latency,
- accuracy,
- repeatability,
- phantom and bench testing,
- safety layers,
- and empirical evidence.

Adopt an internal standard of:

**accuracy numbers, not adjectives.**

Every major claim needs:

- a number,
- a benchmark,
- a failure mode,
- an uncertainty statement,
- or an explicit `UNVALIDATED` state.

---

# 9. Preserve the strongest existing technical work

The project has already built important foundations.

Do not throw them away during a rewrite.

Preserve and continue developing:

- complete tip and shaft geometry,
- swept tool envelopes,
- physical millimeter coordinates,
- explicit RAS/LPS handling,
- sequential cavity legality,
- insertion and withdrawal logic,
- source-cell removal accounting,
- partial-contact accounting,
- STOP,
- independent geometry evaluation,
- provenance and case identity,
- observed-only planning clones,
- spatial observation isolation,
- scan-conditioned CNN work,
- candidate-context critic work,
- functional-evidence objects,
- functional uncertainty evaluation,
- patient-role separation,
- search,
- imitation learning,
- native replay,
- desktop MRI/3D visualization,
- research APIs,
- mechanics verification,
- RESECT/ReMIND work,
- and preserved negative results.

The repository is strongest when new research compounds these pieces.

---

# 10. The latest real-patient learning signal must guide the next experiments

At the latest branch state checked for this goal, the newest experiments provide a very useful diagnosis.

## 10.1 Real PAT05 REINFORCE was negative

The current 30,827-parameter spatial model received two real on-policy REINFORCE updates on TRAIN patient PAT05.

The weights changed.

The selected argmax strategy did not.

Recorded geometric return remained approximately:

`12.7030 -> 12.7030`

with about:

- 17.0001 mm³ modeled target removed,
- 20.0001 mm³ modeled other tissue removed.

A matched greedy three-action strategy achieved approximately:

`410.3124`

with about:

- 493.0017 mm³ modeled target removed,
- 412.0014 mm³ modeled other tissue removed.

That is a valid negative learning result.

Do not respond with a giant hyperparameter sweep.

## 10.2 Small imitation learning produced a real signal

A separate fixed eight-update behavior-cloning experiment from the same initial network used the independently verified greedy trajectory.

The learned policy improved its same-patient geometric return to approximately:

`139.8975`

with about:

- 173.0006 mm³ modeled target removed,
- 164.0006 mm³ modeled other tissue removed.

It still remained substantially below greedy search at approximately:

`410.3124`.

This is a useful same-patient imitation result.

It is not RL success, transfer, or clinical validation.

## 10.3 The failure has been localized

The first learned action became close to the search teacher's first action.

Approximate first-action returns:

- learned policy: `147.3035`
- search teacher: `150.7035`

Most of the remaining teacher-policy gap arose in later decisions.

The learned policy's second and third actions each removed approximately:

- 0 target,
- 18 mm³ other tissue,

and produced negative marginal reward.

The repository estimates that roughly 98.74% of the remaining return gap appeared after the first action.

This is exactly the kind of mechanistic signal the research program should exploit.

The natural immediate hypothesis is:

> fixed demonstration behavior cloning fails after the learned policy enters a cavity/state distribution not represented by the teacher demonstration.

Test that hypothesis before changing the architecture.

The already identified next diagnostic is sensible:

- ask the same permitted search or greedy teacher what it would do in the actual states visited by the learned policy,
- add those visited states to the imitation dataset,
- perform one bounded data-aggregation step,
- re-evaluate under the same task and model.

This resembles a tiny DAgger-style experiment.

Do not automatically continue for many iterations.

The first aggregation step should decide whether that research line earns another one.

If it works, investigate iterative search-policy training or Expert Iteration.

If it fails, investigate other mechanisms such as:

- representation,
- local context,
- candidate-set context,
- crop limitations,
- optimization,
- action-space coverage,
- value estimation,
- or missing historical state.

---

# 11. Search is currently the authority and should remain strong

The current evidence says that search is substantially stronger than the learned policy on the real bounded PAT05 geometric task.

Preserve that.

Do not cripple search to make neural methods look better.

Learning should earn a role by demonstrating one or more of:

- lower planning latency at similar quality,
- fewer expensive expansions,
- better ranking of search nodes,
- better plan quality at matched computation,
- useful transfer across patients,
- useful generalization across instruments,
- better robustness,
- or improved long-horizon decisions.

Policy-only inference does not have to become the final product.

A particularly strong architecture may be:

**learned proposal/ranking/value model + exact search + exact native geometry certification**.

That may be more credible and useful than replacing search entirely.

---

# 12. Action-space design matters more than ceremonial RL

The repository has already shown that changes in proposal coverage can produce larger improvements than learning updates.

An earlier bounded plan removed only a small amount of target.

A broader proposal inventory later enabled much more modeled removal without gradients.

Therefore:

Before tuning learning, ask whether the planner can even represent good strategies.

Measure:

- candidate coverage,
- target-region coverage,
- legal initial actions,
- access diversity,
- approach-direction diversity,
- instrument diversity,
- sequence diversity,
- reachable target fraction,
- and horizon sensitivity.

A policy cannot learn a strategy that does not exist in its candidate set.

Do not credit or blame the policy for proposal-generation limitations.

---

# 13. Build genuinely different surgical-strategy candidates

The final demonstration should not compare only two instruments on the same ray.

Candidates should vary in meaningful ways such as:

- access region,
- entry point,
- target subregion,
- approach vector,
- tool,
- shaft radius,
- active tip,
- opening sequence,
- resection order,
- and stop point.

Use patient-specific anatomy and physical constraints.

Preserve rejected candidates.

For every rejection, keep:

- failure reason,
- location,
- instrument component,
- sequence step,
- evidence state,
- and assumptions.

A rejected route is useful scientific evidence.

---

# 14. Complete-instrument sequential geometry remains a core differentiator

Strengthen the distinction among:

- tip,
- shaft,
- insertion,
- withdrawal,
- previous cavity,
- current cut,
- future cavity,
- partial contact,
- complete source-cell removal,
- and STOP.

A path should not become legal merely because its endpoint is clear.

A shaft should not "borrow" space from tissue that is removed only later in the same action.

An opening can have negative immediate value yet enable a later useful action.

A superficially attractive action can block a later one.

This creates the sequential structure that justifies learning and search research.

Benchmark this explicitly against simpler geometric abstractions such as:

- centerline-only,
- endpoint-only,
- static corridor,
- and complete-tool sequential checking.

Measure when rankings or feasibility differ.

---

# 15. Uncertainty and plan fragility should be a first-class capability

For each plan, investigate:

> How much can the relevant anatomy, registration, functional evidence, or tool state change before the conclusion changes?

Potential uncertainty dimensions include:

- registration translation,
- registration rotation,
- segmentation boundaries,
- functional-map registration,
- brain displacement,
- tool placement,
- evidence coverage,
- and other source-specific uncertainties.

Use coherent uncertainty worlds.

Do not independently jitter every voxel unless that is genuinely the model.

Potential outputs include:

- robust-feasibility frequency,
- minimum-clearance distribution,
- first-invalid-step frequency,
- first failing structure,
- first failing tool component,
- expected exposure,
- upper-tail exposure,
- CVaR,
- ranking reversals,
- and the smallest tested perturbation that changes plan validity or preference.

Do not call these neurological probabilities unless they actually are.

A compelling MEDiVIS-style result would look like:

> Plan A has better nominal target access but loses complete-tool feasibility after a modest measured spatial update, while Plan B retains feasibility across the tested uncertainty envelope.

Whether the exact numerical values support such a story is an experimental question.

Do not manufacture it.

---

# 16. Functional evidence must remain evidence-aware

The project already contains registered population motor/language priors and functional sensitivity work.

Preserve it.

But maintain strict categories:

- patient-specific measured function,
- patient tractography,
- patient fMRI,
- population prior,
- estimated structure,
- simulator state,
- and unknown.

Population maps are not patient-specific cortex.

Structural masks are not neurological deficit probabilities.

Missing functional evidence is not zero risk.

If future patient-specific tractography or functional data become available, integrate them as a different evidence class.

---

# 17. Build the neurological-harm model as a separate scientific object

Do not bury harm prediction inside RL.

Create a separately testable harm-prediction research module.

It should be possible to evaluate it without running the planner.

Its inputs may eventually include combinations of:

- baseline neurological function,
- age and relevant clinical variables,
- tumor anatomy,
- tumor location,
- patient structural connectivity,
- patient functional connectivity,
- modeled or observed resection footprint,
- diffusion injury,
- network disconnection,
- functional-map exposure,
- and other defensible features.

Its outputs may eventually include:

- deficit severity,
- persistence,
- domain-specific outcomes,
- KPS change,
- cognitive impairment,
- or another explicit outcome representation.

Start simple.

Compare against:

- no-deficit majority baseline,
- resection-volume baseline,
- total non-target-volume baseline,
- location-only baseline,
- baseline function alone,
- and other simple models.

Only add complexity when it produces a reproducible signal.

---

# 18. The harm model should be multi-level rather than one monolithic network

Encourage subagents to test different decompositions.

One promising decomposition is:

## Step 1: tissue and network consequence model

Given a realized or modeled damage footprint, estimate things like:

- structural disconnection,
- functional-network exposure,
- ischemic burden,
- or affected anatomical systems.

## Step 2: patient vulnerability model

Given:

- baseline function,
- patient network structure,
- anatomy,
- demographics or clinical variables where justified,

estimate how vulnerable the patient may be to those consequences.

## Step 3: outcome model

Estimate:

- neurological deficit category,
- cognitive decline,
- persistence,
- or another observed outcome.

This decomposition may be easier to validate and interpret than an end-to-end "route to deficit" black box.

It may also help bridge heterogeneous datasets.

This is a hypothesis, not a requirement.

Test it.

---

# 19. Investigate network-disconnection-based harm

The connectivity literature and public data suggest an especially interesting research direction.

A small amount of tissue can have large functional consequences if it disconnects an important network.

Experiment with representations of a plan's modeled impact on:

- structural connectome edges,
- network hubs,
- DMN,
- frontoparietal network,
- motor pathways,
- language pathways,
- and other networks supported by the evidence.

Potential approaches include:

- streamline intersection,
- edge-disconnection burden,
- graph-centrality change,
- network-efficiency change,
- learned network embeddings,
- lesion-network-style mappings,
- or other methods supported by primary literature.

Compare these against simple non-target volume.

The 63-patient glioma connectivity study is particularly relevant for testing whether patient connectivity contains predictive information.

The stroke Aphasia Recovery Cohort may be useful for representation learning or lesion-language relationships, but glioma validation remains mandatory.

---

# 20. Use postoperative imaging as an outcome target, never as leaked preoperative input

For RHUH, BTC, ReMIND, RESECT, or other longitudinal cohorts:

Keep timestamps explicit.

Postoperative scans can be used for:

- training targets,
- injury representations,
- observed outcomes,
- retrospective intraoperative update experiments,
- or final evaluation.

They cannot enter a method that claims to make a preoperative plan unless the experiment explicitly simulates an intraoperative update after those observations become available.

This distinction should be enforced in code and manifests.

---

# 21. ReMIND and RESECT remain central for changing anatomy

## ReMIND

Official collection:

https://www.cancerimagingarchive.net/collection/remind/

Paper:

https://www.nature.com/articles/s41597-024-03295-z

ReMIND contains public multimodal surgical imaging, including preoperative imaging and intraoperative ultrasound/MRI.

Use it for:

- registration,
- displacement,
- changing anatomy,
- postoperative/residual segmentation,
- and plan-update research.

It does not by itself provide a neurological-harm label.

## ReMIND2Reg

Challenge:

https://arxiv.org/abs/2508.09649

Use it as a registration and update benchmark.

Do not claim access to private test landmarks.

## RESECT

Dataset:

https://doi.org/10.11582/2017.00004

Continue the existing conditional-displacement work.

A flagship future experiment should be:

1. freeze a preoperative plan,
2. obtain the retrospective observed intraoperative update,
3. update the anatomy/registration under the experiment's allowed information,
4. re-evaluate the original plan,
5. determine what became stale,
6. replan,
7. visualize the difference.

This directly connects planning to navigation-relevant changing anatomy.

---

# 22. Continue mechanics, but only where measurements support it

The mechanics work is valuable because it prevents "brain physics" from becoming visual theater.

Preserve:

- HBE specimen data,
- FEBio verification,
- solver repairs,
- analytical controls,
- mesh-validation failures,
- RESECT displacement work,
- and all negative results.

The mechanics lane should remain parallel to the geometric/learning lane.

Do not block all useful planning research waiting for perfect brain mechanics.

Do not feed unvalidated mechanics into RL reward.

---

# 23. HBE specimen mechanics

Source:

https://zenodo.org/records/8095559

Primary related work:

https://pmc.ncbi.nlm.nih.gov/articles/PMC10511383/

Continue investigating the current frozen specimen experiment.

The purpose is to determine whether a finite-strain material model can reproduce measured force/torque behavior under the documented fixture.

The current project has already uncovered mesh convergence and runtime limitations.

Do not erase these.

Possible next methods should respond directly to measured failures.

Do not repeatedly rerun a failing expensive configuration without a mechanistic reason.

If a simpler mechanics model performs as well as a complex one, prefer the simpler one.

Do not transfer one ex-vivo specimen's fitted stiffness into a patient and call it patient-specific.

---

# 24. Patient mechanics should initially update geometry, not injury probability

Even if conditional deformation becomes credible, use it first for:

- updating anatomy,
- changing clearance,
- invalidating stale routes,
- or replanning.

Do not jump from displacement accuracy to:

- retraction injury,
- cutting force,
- neurological deficit,
- or tissue viability.

Those require different evidence.

---

# 25. The project should aggressively innovate rather than reproduce papers

Every subagent working on research must treat the literature as a starting point.

For each paper or method, ask:

- what assumption does it make?
- where does that assumption fail for patient-specific glioma resection?
- what useful mechanism can we borrow?
- what does RessectionLab already have that this method lacks?
- what combination of existing ideas might create a stronger method?
- what smallest experiment could falsify the proposed innovation?

Do not implement papers sequentially just because they are cited.

The goal is not:

> reproduce DAgger, reproduce Expert Iteration, reproduce AWAC, reproduce a path-planning paper.

The goal is:

> understand which mechanisms matter here and build something new enough to answer an important question.

---

# 26. High-value innovation directions

These are examples, not mandatory work items.

Use subagents to generate and challenge additional ideas.

## 26.1 Search-policy data aggregation

The current imitation result strongly motivates this.

Use search to label states actually visited by the policy.

Test whether one or a few controlled data-aggregation iterations repair sequential decision failure.

This is related to DAgger but should be adapted to this task rather than treated as a ceremonial reproduction.

## 26.2 Expert Iteration for surgical planning

Alternate between:

- expensive high-quality search,
- policy learning,
- and policy-guided search.

Investigate whether the policy can amortize the best parts of search across patients.

## 26.3 Learned heuristic for exact search

Let the network rank or value search nodes.

Keep exact complete-tool geometry as the legality authority.

Potential success:

- fewer expensive expansions,
- less wall time,
- equal or better plan quality.

This is probably more defensible than neural policy autonomy.

## 26.4 Robust-search distillation

Build a teacher that reasons over:

- uncertainty,
- functional evidence,
- neurological-harm surrogate,
- plan fragility,
- or tail risk.

Then train a model to approximate that expensive robust planner.

This could produce a genuinely interesting research contribution.

## 26.5 Multi-scale candidate-conditioned policy

Combine:

- shared coarse global context,
- high-resolution patches along the tool,
- candidate geometry,
- cavity state,
- procedure history,
- evidence availability,
- and candidate-set context.

Use controlled ablations.

Do not make the model huge without evidence.

## 26.6 Learned action proposals with exact certification

A neural model can propose:

- promising entry regions,
- candidate endpoints,
- or strategy families.

Every proposal must still pass exact native geometry.

This may improve candidate coverage without outsourcing correctness.

## 26.7 Plan fragility optimization

Instead of optimizing only nominal clearance, explicitly optimize how far a strategy is from becoming invalid.

Compare against fixed safety margins.

Potential contribution:

a plan that sacrifices a little nominal performance for much greater robustness to measured spatial uncertainty.

## 26.8 Value of information

Investigate whether the planner can learn when **another observation is worth more than another action**.

Research actions could include:

- request updated registration,
- request intraoperative image,
- request functional review,
- or abstain.

Initially this may be a simulated or retrospective information-action experiment.

Do not claim retrospective information was available prospectively.

## 26.9 Network-preserving planning

Use patient or validated network representations to choose strategies that minimize predicted network disruption rather than arbitrary non-target volume.

Compare with geometry-only planning.

## 26.10 Outcome-aware robust planning

Once a neurological-harm model earns a research role, compare:

- nominal geometric search,
- uncertainty-aware geometric search,
- harm-aware search,
- harm + uncertainty search,
- learned approximation,
- learned-guided robust search.

This may be the eventual heart of the project.

---

# 27. Run research as a portfolio of competing hypotheses

Use many subagents and parallel experimentation.

Do not allocate equal effort to every idea.

Use a scientific portfolio mentality.

## Cheap falsification

For a new idea, start with the smallest experiment that can kill it.

Use:

- existing development patients,
- small horizons,
- fixed task definitions,
- low update counts,
- cached representations,
- analytic fixtures,
- and already generated evidence.

## Mechanism confirmation

If there is a signal, identify why.

Use:

- ablations,
- counterfactuals,
- held-out states,
- alternative baselines,
- representation probes,
- and independent audits.

## Development expansion

Only surviving ideas earn:

- more patients,
- larger budgets,
- stronger architectures,
- more extensive training,
- or more expensive data processing.

## Frozen evaluation

Only after the method and success criterion are frozen should protected evaluation roles be opened.

Do not repeatedly peek.

---

# 28. Cut losers quickly

Do not spend days rescuing a method merely because code has already been written.

Pause or kill a direction when:

- it fails correctness,
- it cannot beat a trivial baseline on the mechanism it claims to improve,
- its gain disappears under matched information,
- it requires hidden truth,
- it has no plausible validation source,
- its cost overwhelms its value,
- it depends on one favorable seed,
- it violates patient split rules,
- or repeated targeted tests fail to support its proposed mechanism.

Preserve the negative result.

Document why it died.

A failed idea that eliminates a research hypothesis is progress.

---

# 29. Compound winners

When an experiment produces a meaningful and reproducible signal:

- understand the mechanism,
- test ablations,
- compare stronger baselines,
- expand to additional development patients,
- test robustness,
- assess generalization,
- then integrate it.

Do not immediately replace a successful simple method with a more fashionable complex one.

Complexity has to earn its keep.

---

# 30. Maintain an innovation ledger

Create or maintain something like:

`docs/INNOVATION_LEDGER.md`

Keep it concise.

For each hypothesis record:

- hypothesis,
- motivating failure or opportunity,
- related literature,
- proposed mechanism,
- baseline,
- predicted result,
- cheapest discriminating experiment,
- result,
- decision such as `KILL`, `HOLD`, `ITERATE`, or `PROMOTE`,
- next experiment,
- and claim status.

The purpose is not bureaucracy.

The purpose is to preserve the scientific logic across hundreds of commits and subagents.

---

# 31. Distinguish baseline, adaptation, and actual innovation

For every substantial method derived from literature, identify whether it is:

## BASELINE / REPRODUCTION

Implemented to provide a known comparison.

## ADAPTATION

An existing method altered to satisfy this project's complete-tool, patient, cavity, evidence, or uncertainty constraints.

## NEW HYPOTHESIS

A genuinely different combination or mechanism being tested.

For adaptations and new hypotheses, document:

- the closest prior work,
- what is different,
- why that difference matters,
- what would falsify it,
- and what evidence supports or rejects it.

Before making novelty claims, perform an updated closest-work search.

Do not infer novelty from absence of an identical GitHub repository.

---

# 32. Use lots of subagents

Use the available agent budget aggressively.

This project benefits from **dozens of specialized subagents** working orthogonally.

The integration owner should delegate deeply.

Possible roles include:

- MEDiVIS product researcher,
- closest-prior-art researcher,
- glioma clinical literature reviewer,
- neurosurgeon-workflow reviewer,
- functional-mapping researcher,
- tractography researcher,
- neurocognitive-outcomes researcher,
- RHUH data auditor,
- BTC longitudinal outcome auditor,
- connectivity-model researcher,
- stroke-transfer researcher,
- causal-inference critic,
- statistics and calibration specialist,
- imaging-registration researcher,
- postoperative injury segmentation researcher,
- brain-shift researcher,
- RESECT specialist,
- ReMIND specialist,
- FEM/mechanics engineer,
- meshing specialist,
- solver-performance engineer,
- complete-tool geometry specialist,
- search/planning engineer,
- imitation-learning engineer,
- RL engineer,
- representation-learning engineer,
- learned-search specialist,
- robust-optimization researcher,
- uncertainty researcher,
- network-neuroscience researcher,
- data-split steward,
- provenance engineer,
- desktop visualization engineer,
- 3D rendering engineer,
- performance profiler,
- adversarial tester,
- independent numerical auditor,
- reproducibility engineer,
- README/demo editor,
- and skeptical "reviewer 2" agents whose only job is to attack the main claims.

Spawn more if useful.

Do not force every subagent to implement code.

Some should challenge the premise, locate better data, review papers, propose experiments, or independently audit results.

Use one clear integration owner or small integration group to prevent architectural fragmentation.

Subagents should share evidence through concise research notes, manifests, or artifacts rather than duplicating entire investigations.

---

# 33. Give subagents research autonomy

Do not overconstrain exact algorithms, hyperparameters, model sizes, or experiment ordering in advance.

Agents are authorized to:

- propose new hypotheses,
- reject hypotheses in this prompt,
- discover better public datasets,
- use new primary literature,
- replace a method with a stronger justified method,
- refactor the architecture,
- optimize bottlenecks,
- design new benchmarks,
- and alter experimental sequencing.

They must preserve:

- source and patient-role integrity,
- honest claims,
- held-out evaluation,
- provenance,
- correctness,
- and the project's core research purpose.

Evidence outranks this prompt's implementation suggestions.

---

# 34. Research questions worth prioritizing

Prefer questions whose answers remain interesting even if negative.

Examples:

- Does complete-tool temporal geometry change plan ranking relative to static or centerline planning?
- Can plan fragility predict which preoperative strategies become invalid after measured spatial change?
- Does the current imitation failure arise mainly from sequential distribution shift?
- Can one-step or iterative search-label aggregation repair it?
- Can a learned heuristic materially accelerate exact search?
- Can neurological-harm features predict observed deficits better than non-target volume?
- Does structural-network disruption predict cognition better than lesion volume alone?
- Can simulated network disruption from a strategy produce a useful, calibrated harm surrogate?
- Does early postoperative diffusion injury improve neurological-outcome prediction?
- Can a harm-aware teacher produce policies that generalize better than a geometric teacher?
- Does robust search produce clinically more plausible tradeoffs than arbitrary fixed margins?
- Does FEM displacement improve held-out landmark prediction beyond simple interpolation?
- Does measured deformation actually change plan validity?
- Can the system identify when it should request more evidence instead of acting?
- Does candidate-local high-resolution context improve later sequential decisions?
- Can outcome-aware planning outperform geometry-only planning without becoming overconfident?

These are stronger questions than:

> Does PPO get more reward if we train longer?

---

# 35. Patient splits and leakage are sacred

Preserve existing BTC patient roles.

Do not casually reshuffle TRAIN, SELECT, later-transfer, or unopened groups.

Keep longitudinal visits from one patient together.

For new outcome cohorts:

freeze patient-level roles before model tuning.

Prevent leakage through:

- postoperative scans,
- outcome-derived segmentations,
- follow-up connectivity,
- visit duplication,
- derivative files,
- preprocessing fit,
- atlas registration,
- feature scaling,
- and any other hidden path.

A model that uses postoperative evidence may be valid for an intraoperative or retrospective task, but not for a preoperative task.

Name the task honestly.

---

# 36. Class imbalance and neurological outcomes need serious evaluation

Do not use overall accuracy as the main metric for RHUH.

The dataset is imbalanced.

Evaluate things such as:

- sensitivity to any deficit,
- sensitivity to persistent deficit,
- class-specific recall,
- balanced accuracy,
- macro F1,
- AUROC,
- AUPRC,
- calibration,
- ordinal discrimination,
- and uncertainty.

Use exact metrics appropriate to the final formulation rather than mechanically computing all of these.

Report the majority baseline.

If there are too few severe cases for a stable model, say so.

Do not hide that limitation with cross-validation statistics.

---

# 37. Categorical severity is not linear physical damage

Deficit outcomes such as:

- none,
- transient,
- minor persistent,
- major persistent

are naturally ordered.

But "major persistent" is not automatically exactly three times "transient."

Use ordinal models or another suitable representation.

Do not encode arbitrary numeric severity ratios unless justified.

Similarly, cognitive test changes have their own units and measurement properties.

Respect them.

---

# 38. Baseline impairment matters

A postoperative impairment should be interpreted relative to preoperative function.

Where data permit, distinguish:

- new deficit,
- worsened pre-existing deficit,
- unchanged impairment,
- improved function,
- transient decline,
- persistent decline.

A patient who already has language dysfunction before surgery should not be labeled as having a new surgical language injury merely because language remains impaired afterward.

---

# 39. Timing matters

Record outcome timing.

Examples:

- immediate postoperative,
- within 72 hours,
- discharge,
- weeks,
- 3 months,
- 6 months,
- longer follow-up.

Transient deficits and persistent deficits should not be conflated.

Postoperative cognition at six months may reflect:

- surgery,
- recovery,
- plasticity,
- radiation,
- chemotherapy,
- tumor recurrence,
- medication,
- and other influences.

Models and claims should reflect this.

---

# 40. Build strong simple baselines before fancy harm models

Before a multimodal transformer or large graph network, test useful baselines such as:

- baseline neurological function,
- age and simple clinical variables,
- tumor location,
- tumor volume,
- resection volume,
- non-target injury volume,
- diffusion abnormality volume,
- structural disconnection burden,
- simple graph features,
- and regularized linear or tree models.

If a simple model performs similarly to a large model, that is an important result.

Use complexity only when it earns a measurable advantage.

---

# 41. The neurological-harm model should have uncertainty and abstention

A research harm model should know when it is outside its evidence.

Potential mechanisms include:

- ensembles,
- conformal methods,
- calibration models,
- distance-to-training-distribution measures,
- Bayesian approximations,
- or simpler explicit coverage rules.

Use what is justified by data scale.

Do not add mathematically ornate uncertainty that is not calibrated.

When evidence is missing or out-of-distribution:

return an explicit unknown or low-confidence state.

The planner should be allowed to abstain.

---

# 42. The ultimate planner should support multi-objective tradeoffs

Do not design the app to pretend there is one objectively correct tradeoff.

Expose alternatives such as:

- maximize modeled target removal,
- minimize neurological-harm surrogate,
- minimize geometric fragility,
- reduce normal injury,
- reduce procedure burden,
- or prioritize robust feasibility.

Generate a Pareto frontier where practical.

Let the research interface show what is being traded.

Do not fabricate surgeon preferences.

---

# 43. The learned model should not own geometric legality

The network may:

- rank,
- propose,
- approximate,
- estimate value,
- or guide search.

The exact native geometry engine remains the authority for:

- complete tool collision,
- temporal cavity legality,
- contained source-cell removal,
- and related physical constraints.

This separation is a strength.

Use it.

---

# 44. A strong future planning architecture

One promising high-level architecture is:

patient evidence  
→ typed case state  
→ candidate generator  
→ learned candidate ranking or proposal model  
→ exact complete-tool feasibility  
→ search / robust search / policy-guided search  
→ functional and neurological-harm evaluation  
→ uncertainty worlds  
→ independent evaluator  
→ Pareto or robustness comparison  
→ sequential replay  
→ desktop/API.

Do not treat this as immutable.

Improve it when evidence supports a better decomposition.

---

# 45. Improve the spatial representation based on observed failure

The current CNN already provides a useful baseline.

Do not replace it reflexively.

A plausible next representation to test includes:

## Shared static context

Encode global patient anatomy once.

Avoid recomputing static image features for every action.

## Candidate-local context

Sample high-resolution patches around:

- entry,
- shaft,
- tip,
- target,
- narrow-clearance regions,
- and potentially functionally important structures.

## Dynamic state

Represent:

- cavity,
- contact history,
- sequence depth,
- remaining action budget,
- previous tool,
- and evidence changes.

## Candidate-set context

Give the model enough context to understand what alternatives currently exist.

## Long-term state

If the current state representation is not Markov enough for sequential planning, investigate recurrence, memory, history encoding, or a better explicit state.

Do not add recurrence merely because sequence models are fashionable.

---

# 46. Train in a ladder rather than jumping straight to RL

A sensible research ladder is:

## Search imitation

Can the network reproduce good search choices?

## Data aggregation

Can it recover when its own actions produce new states?

## Search-guided learning

Can the model guide expensive search?

## Reward refinement

Can reward-based learning improve beyond imitation/search?

## Transfer

Can the learned representation help on separate patients?

Do not claim a later stage until the earlier evidence supports it.

---

# 47. Expert Iteration and DAgger are particularly relevant

DAgger:

https://proceedings.mlr.press/v15/ross11a.html

Expert Iteration:

https://arxiv.org/abs/1705.08439

These provide conceptual tools for the exact problem exposed by the latest PAT05 experiment:

- a strong planner can act as teacher,
- a learned policy visits different states,
- those states can be relabeled by the teacher,
- and the policy can eventually guide the planner.

Do not implement them as religious recipes.

Adapt the mechanism to:

- expensive native geometry,
- patient-specific state,
- candidate actions,
- incomplete evidence,
- and strong search.

---

# 48. Other RL and surgical-learning sources

Use these as references, not commandments.

## Ghesu et al.

"Multi-Scale Deep Reinforcement Learning for Real-Time 3D-Landmark Detection in CT Scans"

PMID:

https://pubmed.ncbi.nlm.nih.gov/29990011/

Useful for volumetric local context.

## AWAC

https://arxiv.org/abs/2006.09359

Useful if search-generated data provide an offline starting distribution before online improvement.

## SurRoL

https://arxiv.org/abs/2108.13035

Useful for structuring surgical-RL experiments and baselines.

## LapGym

https://jmlr.org/papers/v24/23-0207.html

Useful as a model for standardized surgical-learning benchmarks.

Do not import their claimed performance into glioma planning.

---

# 49. Closest neurosurgical prior art must constrain novelty claims

## Frisken et al.

"Incorporating Uncertainty Into Path Planning for Minimally Invasive Robotic Neurosurgery"

DOI:

https://doi.org/10.1109/TMRB.2021.3122357

Uncertainty-aware neurosurgical path planning already exists.

## Bakhshmand et al.

"Multimodal connectivity based eloquence score computation and visualisation for computer-aided neurosurgical path planning"

PMID:

https://pubmed.ncbi.nlm.nih.gov/29184656/

Multimodal eloquence/path scoring already exists.

## Li et al.

"An innovative learning-based framework for automated craniotomy planning in glioma resection"

PMID:

https://pubmed.ncbi.nlm.nih.gov/40889518/

RL-based glioma surgical planning already exists in a related craniotomy task.

## Shan et al.

"Towards Learning-based Surgical Planning of Glioma Resection via a Contrastively Constrained Siamese Neural Network"

DOI:

https://doi.org/10.1109/BIBM58861.2023.10385556

Learning-based glioma resection planning already exists.

## Segato et al.

"Inverse Reinforcement Learning Intra-Operative Path Planning for Steerable Needle"

PMID:

https://pubmed.ncbi.nlm.nih.gov/34882540/

Learning plus deformation-aware neurosurgical path planning exists in another instrument/task domain.

Therefore:

Do not claim novelty for:

- RL,
- uncertainty,
- multimodal risk,
- or surgical path planning in isolation.

Potential novelty must come from the combination and evidence, such as:

- complete-tool temporal legality,
- sequence-dependent cavity state,
- patient-specific imaging,
- neurological-harm modeling,
- plan fragility,
- real intraoperative update,
- exact certification,
- learned search acceleration,
- and rigorous evaluation.

---

# 50. Clinical glioma sources

## EANO diffuse glioma guideline

https://pmc.ncbi.nlm.nih.gov/articles/PMC7904519/

Use for clinical imaging context and appropriate modality expectations.

## EANS-EANO extent-of-resection guideline

DOI:

https://doi.org/10.1093/neuonc/noaf217

Use for the distinction among extent of resection, postoperative imaging, neurological status, and outcome.

Do not equate simulated removal with clinical extent of resection.

## Functional MRI / DTI versus stimulation validation

Use the existing primary study in the repository's evidence notes.

Preserve the distinction between:

- imaging proxy,
- stimulation evidence,
- and true functional outcome.

---

# 51. Mechanics sources

## Budday et al. mechanical characterization

PMID:

https://pubmed.ncbi.nlm.nih.gov/27989920/

## Budday et al. rheological characterization

PMID:

https://pubmed.ncbi.nlm.nih.gov/28658600/

## Hyperelastic Human Brain 1-7

https://zenodo.org/records/8095559

## Hinrichsen et al.

PMID:

https://pubmed.ncbi.nlm.nih.gov/37676609/

## FEBio

PMID:

https://pubmed.ncbi.nlm.nih.gov/22482660/

## DiSECt

https://arxiv.org/abs/2203.10263

Use these to constrain and validate mechanics.

Do not infer patient stiffness or surgical cutting outcomes from them without evidence.

---

# 52. The desktop app should now expose the real research value

The backend is substantially ahead of the validated packaged desktop.

Eventually produce one current, verified Electron application snapshot containing only features that survive their research gates.

The primary workflow should feel like:

**load case → inspect evidence → generate strategies → compare plans → inspect neurological/functional evidence → inspect uncertainty and fragility → inspect sequential replay → refine with search or learned guidance → apply new evidence/update → re-evaluate → save/export.**

---

# 53. Improve A/B comparison

Show meaningful components such as:

- modeled target removed,
- residual target,
- modeled other tissue removed,
- partial contact,
- functional evidence,
- neurological-harm surrogate if validated,
- unknown functional evidence,
- geometric clearance,
- robust feasibility,
- fragility,
- tail exposure,
- computation time,
- and plan completeness.

The app should explain **why** A and B differ.

---

# 54. Add a neurological-impact panel only when justified

Do not build fake clinical gauges.

When outcome models exist, show exactly what they mean.

Possible labels:

- observed-cohort neurological-harm surrogate,
- model-conditioned ordinal deficit estimate,
- cognitive vulnerability estimate,
- network-disconnection burden,
- or evidence-unavailable.

Include:

- model version,
- training cohort,
- domain,
- input coverage,
- calibration state,
- uncertainty,
- and out-of-distribution warning.

Never call an unvalidated surrogate "chance of paralysis."

---

# 55. Add a plan-fragility view

For the selected strategy, visualize:

- nominal tool,
- uncertainty envelope,
- critical clearance point,
- affected sequence step,
- perturbation direction,
- failing structure,
- and whether plan ranking changes.

Use precise labels.

Do not color every uncertainty red and imply injury.

---

# 56. Sequential replay should become a flagship interaction

Allow scrubbing through:

- entry/opening,
- insertion,
- modeled action,
- withdrawal,
- cavity update,
- next legal actions,
- function/harm updates,
- and STOP.

Show what became possible because of earlier actions.

Show what became impossible.

This temporal behavior is one of the strongest technical parts of the project.

---

# 57. Show stale-plan invalidation

When evidence changes, expose:

- old evidence ID,
- new evidence ID,
- frame or transform change,
- affected strategy,
- invalidated result,
- changed metric,
- changed sequence step,
- recomputation,
- and new result identity.

This makes the provenance machinery useful to a human rather than merely an audit artifact.

---

# 58. Build a narrow local planning API

Keep numerical operations behind typed interfaces.

Possible conceptual operations include:

- `load_case`
- `inspect_evidence`
- `generate_strategies`
- `evaluate_strategy`
- `evaluate_robustness`
- `evaluate_functional_impact`
- `evaluate_neurological_harm`
- `compare_strategies`
- `search`
- `policy_guided_search`
- `replay_strategy`
- `apply_observation_update`
- `explain_failure`
- `export_strategy`

Names can change.

Results should carry:

- case identity,
- frame,
- source hashes,
- evidence versions,
- tool catalog,
- uncertainty model,
- strategy ID,
- evaluation role,
- metrics,
- assumptions,
- unknowns,
- and provenance.

---

# 59. Independent evaluation should attack the real claims

Continue adversarial testing, but prioritize scientifically meaningful failure modes.

Examples:

- RAS/LPS mismatch,
- mm/m mismatch,
- affine error,
- oblique grid,
- anisotropic voxels,
- shaft collision,
- endpoint-only false acceptance,
- temporal cavity borrowing,
- stale case result,
- hidden-reference leakage,
- missing evidence interpreted as zero,
- postoperative leakage,
- patient split leakage,
- network feature leakage,
- candidate denominator loss,
- uncertainty-world incoherence,
- class-imbalance metric abuse,
- outcome-timing mismatch,
- mesh local-fidelity failure,
- unsupported interpolation,
- replay mismatch,
- and OOD harm-model use.

Prefer a clear protocol and one independent evaluation over endless duplicated artifact wrappers.

---

# 60. Quantitative planning metrics

Report as appropriate:

- candidate count,
- retained/dominated/rejected counts,
- executable count,
- candidate coverage,
- target reachability,
- target removed,
- residual target,
- other tissue removed,
- partial contact,
- clearance,
- tool violations,
- action count,
- tool changes,
- search expansions,
- and wall time.

---

# 61. Functional and uncertainty metrics

Report:

- evidence coverage,
- missing evidence,
- event counts and denominator,
- uncertainty generator,
- model-conditioned event frequency,
- mean surrogate,
- tail surrogate,
- robust feasibility,
- plan fragility,
- and ranking changes.

Do not imply uncertainty calibration that has not been demonstrated.

---

# 62. Neurological outcome metrics

Choose metrics based on the actual task.

Potential categories:

- deficit sensitivity,
- persistent-deficit sensitivity,
- macro/balanced metrics,
- ordinal performance,
- calibration,
- Brier-type scores,
- AUPRC,
- AUROC,
- KPS change error,
- cognitive-score error,
- domain-specific error,
- and confidence intervals.

Do not mechanically optimize one metric.

The main metric should reflect the clinical question and class distribution.

---

# 63. Learning and search metrics

Report:

- planning quality,
- target/harm tradeoff,
- Pareto quality,
- expansions,
- previews,
- policy forwards,
- gradient updates,
- teacher-generation cost,
- offline training cost,
- online planning cost,
- independent checking cost,
- memory,
- and failure/abstention.

Compare:

- quality at matched time,
- time at matched quality,
- and generalization on separate patients.

---

# 64. Mechanics metrics

For measured specimen work:

- mesh convergence,
- load-step convergence,
- force error,
- torque error,
- Jacobians,
- reaction balance.

For conditional displacement:

- RMS landmark error,
- median,
- maximum,
- upper tail,
- coverage,
- exclusions,
- and improvement over simple interpolation.

---

# 65. Application metrics

Measure actual:

- load time,
- strategy-generation time,
- robust evaluation time,
- harm-evaluation time,
- replay latency,
- update/replan time,
- and memory.

Do not present one local run as a universal performance guarantee.

---

# 66. The flagship demonstration

The project needs one memorable evidence-backed workflow.

Do not cherry-pick a case to create a desired conclusion.

A strong final demonstration might look like this:

## Scene 1: real patient evidence

Load a public glioma case.

Show source MRI, tumor evidence, functional evidence, and what is unavailable.

## Scene 2: genuinely different strategies

Generate several complete-instrument sequential strategies.

## Scene 3: geometric difference

Show a case where complete-tool or temporal geometry matters compared with a simpler abstraction, if the benchmark supports it.

## Scene 4: neurological and functional context

Show what important neurological systems each strategy may affect.

If a validated harm model exists, show its bounded estimate and uncertainty.

Otherwise show functional evidence and explicitly state that clinical harm is unavailable.

## Scene 5: robustness

Perturb the declared uncertainty or apply a measured spatial update.

Show whether strategy validity or ranking changes.

## Scene 6: intraoperative update

Where retrospective data permit, apply a real measured update and re-evaluate the frozen preoperative plan.

## Scene 7: learned assistance

Show the best evidence-backed role of learning:

- imitation,
- policy guidance,
- learned search heuristic,
- robust-plan distillation,
- or patient adaptation.

Compare it honestly with search.

## Scene 8: provenance

Show exactly which data, models, assumptions, and unknowns produced the result.

A MEDiVIS engineer should be able to understand what is real and what is modeled.

---

# 67. What would make this genuinely differentiated

Do not claim differentiation because the repo is large.

Do not claim it because of RL.

Do not claim it because of 3D MRI.

Do not claim it because of tests.

Substantive differentiation could come from evidence that the project does one or more of these:

## A. Temporal complete-tool legality

It catches important failures that centerline/static methods miss.

## B. Neurological-harm modeling beyond volume

A model grounded in observed patient outcomes predicts meaningful harm better than arbitrary normal-tissue volume.

## C. Network-aware functional preservation

Modeled network disruption provides useful outcome information beyond lesion/resection size.

## D. Decision-relevant uncertainty

The system identifies fragile versus robust strategies.

## E. Measured intraoperative updating

Observed spatial change alters plan validity and triggers reproducible replanning.

## F. Learned planning efficiency

Learning reduces search cost or improves quality on held-out patients without hidden information.

## G. Evidence-aware abstention

The system knows when it lacks enough information to make a confident statement.

## H. Reproducibility

Another engineer can reconstruct the result.

The value is the evidence connecting implementation to behavior.

---

# 68. Source-aware harm modeling is potentially a major research contribution

A particularly interesting long-term contribution may be:

> Learn a patient- and network-aware functional-harm representation from real pre/postoperative cohorts, use it to score simulated complete-tool resection strategies, then combine it with exact geometry and uncertainty-aware search.

This would connect several currently separate research worlds:

- surgical planning,
- neuroimaging,
- connectomics,
- neurological outcomes,
- sequential decision-making,
- and robust optimization.

It could be far more interesting than "RL removes more voxels."

But it must be built carefully because the counterfactual gap is severe.

Use skeptical subagents to attack this idea continuously.

---

# 69. Treat the counterfactual gap as a research problem, not an inconvenience

Investigate what can and cannot be learned from one surgery per patient.

Possible approaches subagents may explore include:

- outcome association models,
- inverse-problem formulations,
- causal sensitivity analysis,
- structural causal models if defensible,
- network lesion models,
- virtual lesion augmentation,
- weak supervision,
- multi-dataset representation learning,
- synthetic counterfactuals anchored to real outcomes,
- or partial identification.

Any counterfactual method must clearly state assumptions.

Do not turn simulated labels into evidence merely because they are convenient.

---

# 70. Use stroke data carefully for functional localization

ARC and other stroke cohorts can be useful because stroke provides many real lesion-outcome examples.

Potential uses:

- representation pretraining,
- language network learning,
- lesion-disconnection mapping,
- and sanity checks.

But:

- stroke mechanism differs,
- timing differs,
- plasticity differs,
- lesion geometry differs,
- tumor brains reorganize,
- surgery creates different injury patterns.

Therefore:

Stroke can support a representation.

Glioma data must validate the surgical application.

---

# 71. Consider domain-specific neurological models

A single "neurological deficit" output may be too coarse.

Subagents should test whether separate models or heads are better for:

- motor,
- language,
- cognition,
- vision,
- and global function.

The app can later show multiple dimensions.

This also matches the reality that different tracts and networks matter to different functions.

---

# 72. Consider persistence separately from occurrence

Transient and persistent deficits are not the same outcome.

A useful model may separately estimate:

- immediate deficit,
- persistence,
- or recovery.

RHUH's categories make this especially relevant.

Do not merge them before testing whether the distinction contains usable signal.

---

# 73. Consider cognitive outcomes as a parallel objective

Motor and language are important, but a sophisticated glioma planner should not pretend those are the only patient outcomes that matter.

BTC and the 63-patient structural-connectivity cohort offer an opportunity to investigate:

- attention,
- working memory,
- processing speed,
- executive function,
- and related network-level outcomes.

This may eventually become a second outcome axis alongside focal neurological deficit.

Do not force it into the same model if the data do not support that.

---

# 74. Integrate baseline function into planning research

A patient's acceptable or predicted functional impact depends partly on their baseline condition.

Where available, include:

- preoperative KPS,
- baseline neurological domain,
- baseline cognitive test,
- and related patient context.

Do not imply a tissue region has identical functional consequence across every patient.

---

# 75. Do not let genetics become a magical harm multiplier

Molecular data may be useful for:

- tumor biology,
- prognosis,
- treatment context,
- or research stratification.

Do not invent a rule such as:

> IDH mutation means this voxel is safer to remove.

Molecular information should influence planning only through a defensible mechanism and appropriately timed evidence.

---

# 76. Vascular injury remains an important missing axis

Neurological harm is not only functional cortex and white matter.

Vascular injury and postoperative ischemia can create deficits.

Where public data permit:

- investigate vascular evidence,
- diffusion injury,
- ischemia,
- and uncertainty.

Do not claim a complete safety model while vessels remain unassessed.

Missing vascular evidence should remain explicit.

---

# 77. Keep the geometric objective as a baseline forever

Even after a neurological-harm model exists, preserve the original geometric objective.

It is useful as:

- a baseline,
- an ablation,
- and a way to measure what the harm model changes.

A strong result may be:

> outcome-aware planning chooses a different strategy than volume-only planning and improves an independently measured harm surrogate.

That comparison requires the geometric baseline.

---

# 78. Avoid reward hacking

Once harm enters the objective, watch for new failure modes.

A learned policy may exploit:

- missing coverage,
- unknown regions,
- calibration gaps,
- surrogate weaknesses,
- out-of-distribution inputs,
- or representation blind spots.

Unknown should not have zero cost by default.

Consider conservative or abstaining treatment where justified.

Use adversarial agents to search for reward exploits.

---

# 79. Do not let outcome modeling destroy planning reproducibility

Every harm-model version used by a planner should be pinned.

Record:

- training cohort,
- split,
- features,
- model hash,
- calibration,
- evidence coverage,
- and limitations.

Changing the harm model should invalidate dependent planning results.

The existing provenance machinery is well suited to this.

---

# 80. Update the claim matrix

Create or maintain a central claim matrix such as:

`docs/MEDIVIS_RESEARCH_TARGET.md`

For each desired claim, track:

- claim,
- why it matters,
- current evidence,
- missing evidence,
- data,
- method,
- baseline,
- acceptance condition,
- current status,
- language allowed,
- language forbidden.

Add neurological claims explicitly.

For example:

### "The planner predicts neurological deficit"

Current status:

unsupported.

### "The outcome model distinguishes patients with observed postoperative deficit better than a simple baseline"

Potentially testable.

### "A simulated plan's harm score predicts what would have happened under that unperformed surgery"

Currently counterfactual and unsupported.

This discipline should prevent accidental overclaiming.

---

# 81. Commit cadence

Do **not** spam commits every 10 to 15 minutes.

Commit at meaningful research and engineering checkpoints.

During active multi-hour work, a rough target of **about once per hour** is reasonable when there is a coherent milestone worth preserving.

This is not a timer.

If an experiment takes several hours, do not interrupt it merely to produce a commit.

If a meaningful result appears after twenty minutes, a commit is fine.

If work is unstable or exploratory, wait until the unit is coherent.

Prefer commits that correspond to things such as:

- frozen protocol,
- completed implementation,
- completed experiment,
- preserved negative result,
- independent audit,
- or integrated feature.

Continue pushing coherent progress to the current branch.

---

# 82. Commit authorship

Preserve the current contributor convention on future commits.

Use:

`skamal23 <sayemkamal12@gmail.com>`

as the commit author identity where appropriate to the existing branch convention, and retain:

`Co-authored-by: shafsthegoat <shafrir.p@gmail.com>`

in the commit message.

Do not silently replace these identities with a generic bot author.

If repository configuration already enforces the correct identities, preserve it.

---

# 83. Autonomy

This goal itself authorizes substantial autonomous work.

The agents may:

- inspect the entire repository,
- refactor existing code,
- install free/open-source dependencies in isolated environments,
- read current primary literature,
- download relevant public datasets under their terms,
- run local CPU/GPU experiments,
- use existing Google Cloud credits if genuinely needed and already authorized,
- create local ignored caches,
- generate experiment manifests,
- run search,
- run learning,
- run mechanics,
- run outcome modeling,
- parallelize research,
- spawn many subagents,
- preserve artifacts,
- update documentation,
- commit,
- and push coherent milestones.

Do not stop for routine approval.

Do not spend new real money.

Do not change billing.

Do not use private clinical data.

Do not deploy for live clinical use.

Do not operate a robot.

Do not make medical decisions.

---

# 84. Public-data acquisition autonomy

Agents may acquire more public data when a specific research question needs it.

Before downloading a large dataset, answer:

- what missing variable does it provide?
- which experiment needs it?
- which patients will be used?
- what is the license?
- what is the expected size?
- what split role will it have?
- can a smaller subset answer the question?

Do not hoard medical datasets.

Download the smallest useful cohort first.

---

# 85. Cost discipline

Local-first remains preferred.

Use available compute intelligently.

Before an expensive run:

- profile the bottleneck,
- test on a cheap case,
- confirm the mechanism,
- and freeze a useful protocol.

Do not run massive sweeps merely because compute is available.

Large agent budget should increase research breadth and skepticism, not brute-force waste.

---

# 86. Stop rules against research theater

Stop and reconsider if the project begins doing any of these:

- adding another model because the previous model lost,
- tuning until RL wins,
- weakening a geometry gate,
- weakening a mesh gate,
- treating normal tissue volume as neurological outcome,
- inventing patient-specific risk from population maps,
- treating postoperative DWI as deficit probability,
- treating a stroke model as a glioma model,
- treating different seeds as different patients,
- tuning on validation patients,
- leaking postoperative information into a preoperative claim,
- interpreting missing clinical notes as no deficit,
- creating synthetic clinical labels to fill missing data,
- claiming causal counterfactual effects from observational outcomes without assumptions,
- adding mechanics rewards before mechanics validation,
- claiming successful learning because parameters changed,
- confusing behavior-cloning improvement with RL,
- confusing a visually plausible 3D model with accuracy,
- creating hundreds of audit files without answering a research question,
- polishing the UI while the core result remains undefined,
- or claiming novelty without a closest-work audit.

Negative results are allowed.

Inconclusive results are allowed.

Simple methods winning is allowed.

Search beating RL is allowed.

A harm model being too weak to integrate is allowed.

Preserve the truth.

---

# 87. Final completion gates

This campaign is not complete because the code builds.

Aim for as many of the following as evidence permits.

## Gate A: coherent real planning task

At least one real patient case supports genuinely different complete-instrument strategies and sequential replay.

## Gate B: geometric contribution

A benchmark quantifies the value of complete-tool temporal geometry over simpler abstractions.

## Gate C: uncertainty

Nonzero uncertainty changes measurable strategy properties appropriately.

## Gate D: functional evidence

Missing and available functional evidence are handled honestly.

## Gate E: neurological outcome research

At least one real outcome cohort has been audited and a meaningful neurological-harm baseline has been evaluated against simpler volume/location baselines.

## Gate F: harm integration

If and only if the harm model earns it, the same bounded harm estimate is integrated into both search and learning.

## Gate G: learning role

Learning either provides a measured benefit or is honestly relegated behind search.

## Gate H: held-out patients

Important planning/learning claims are evaluated on patients not used for model selection.

## Gate I: mechanics

Mechanics influences planning only after the relevant validation gate passes.

## Gate J: changing anatomy

At least one retrospective real update experiment evaluates stale-plan validity if the data and mechanics/registration work support it.

## Gate K: desktop

A current Electron snapshot can exercise the actually supported research pipeline.

## Gate L: reproducibility

Another engineer can reproduce the flagship result from documented commands and pinned evidence.

---

# 88. Final deliverables

Produce a polished project, not merely research debris.

## README

Lead with the capability and strongest measured result.

Do not lead with:

"RL for glioma surgery."

Explain:

- the problem,
- the system,
- the technical contribution,
- the benchmark,
- the best result,
- and the limitations.

## `MEDIVIS_TECHNICAL_BRIEF.md`

Answer:

- what does this do beyond ordinary surgical visualization?
- what has actually been measured?
- what failed?
- what is novel or adapted?
- what might conceptually fit a navigation/planning ecosystem?
- how is neurological harm represented?
- what remains unsupported?
- what is the role of learning?
- what is the role of search?
- what is the role of mechanics?

## `INNOVATION_LEDGER.md`

Preserve the research logic.

## Benchmark report

Include strong baselines and every important failure.

## Outcome-model report

If neurological modeling reaches a meaningful result, include:

- cohort,
- split,
- outcome schema,
- class balance,
- baseline,
- model,
- calibration,
- uncertainty,
- and limitations.

## Architecture diagram

Show something like:

patient evidence  
→ case state  
→ strategy proposals  
→ exact complete-tool geometry  
→ sequential simulation  
→ function/network/harm model  
→ uncertainty worlds  
→ search and learned guidance  
→ independent evaluation  
→ Pareto/robustness results  
→ desktop/API.

## Demo

Create one reproducible evidence-backed demonstration.

## Reproduction entrypoint

Provide the smallest reliable command or workflow that rebuilds the flagship result.

## Limitation table

Explicitly include:

- no autonomous surgery,
- no clinical recommendation,
- outcome-model scope,
- counterfactual limitation,
- function/vascular evidence,
- mechanics validation level,
- cohort size,
- data access limits,
- search versus learning outcome,
- and MEDiVIS non-affiliation.

---

# 89. The bar for success

At the end, a MEDiVIS engineer should not think:

> Someone generated an Electron medical app and added an RL loop.

They should think something closer to:

> This person understood that the difficult part is not rendering a tumor. The project treats planning as a patient-specific sequential decision problem with complete instruments, changing cavity state, coordinate rigor, uncertainty, missing evidence, functional networks, and measured patient outcomes. It separates simulation from observation, tests learning against strong search, learns from negative results, uses real longitudinal imaging and neurological labels where available, refuses to pretend normal tissue volume equals functional harm, and makes the evidence inspectable in a polished application. I can see how parts of this could contribute to a future planning or navigation intelligence layer.

That is the target.

Optimize for measured technical substance, scientific originality, and evidence.

Not generated complexity.

---

# 90. Immediate orientation from current HEAD

Before doing anything new:

1. inspect the latest commits after `e477f5e6b00427c3453a7bf8e2d577fc862cd82e` if the branch has advanced;
2. read the latest `PROJECT_STATUS.md`;
3. inspect the current real PAT05 imitation artifacts;
4. preserve the planned visited-state teacher-label aggregation experiment unless newer evidence supersedes it;
5. audit whether any subagent has already begun neurological-outcome work;
6. update the claim and innovation ledgers;
7. spawn research subagents for the neurological-outcome, network, longitudinal imaging, counterfactual, and reward-integration questions;
8. let the integration owner decide the most informative next experiments.

The current momentum should continue.

Do not reset the repository because this goal is larger.

Build from what is already working.

---

# 91. Suggested neurological-outcome subagent swarm

Use a substantial parallel research swarm.

At minimum, consider independent agents for:

## RHUH clinical/outcome audit

Inspect the actual clinical CSV, outcome definitions, patient identifiers, KPS, deficit categories, missingness, imaging linkage, and public-access boundaries.

## RHUH postoperative imaging

Determine what reliable postoperative injury representations can be extracted from:

- T1,
- T2,
- FLAIR,
- T1ce,
- ADC,
- segmentations.

## BTC cognitive audit

Map exactly which pre/post cognitive assessments exist for the glioma subset and which patients have complete multimodal pairs.

## BTC connectivity modeling

Investigate structural/functional connectivity changes and cognitive outcomes.

## 63-patient connectivity cohort

Inspect the public figshare artifacts, reproduce simple reported predictors if possible, and assess how the representation might transfer into plan-harm modeling.

## Stroke transfer

Investigate whether ARC or other stroke lesion-outcome data can provide useful representations without pretending to be direct tumor evidence.

## Outcome schema

Design a unified but honest neurological/cognitive outcome representation.

## Class imbalance/statistics

Design evaluation appropriate to tiny severe-deficit classes.

## Causal/counterfactual critic

Attack every proposal that claims to infer harm under unperformed operations.

## Postoperative DWI/ischemia

Investigate a robust intermediate injury representation.

## Network disconnection

Develop and benchmark structural/connectomic harm features.

## Reward integration

Determine when and how an outcome model could enter planning without leaking postoperative truth.

## Robust optimization

Investigate harm-aware search, Pareto planning, risk-sensitive objectives, and uncertainty.

## Independent clinical-methods reviewer

Attempt to falsify the whole harm-modeling program before large compute is spent.

Let these agents disagree.

The integration owner should promote only the ideas that survive.

---

# 92. A possible long-term research target

A compelling eventual system might do the following:

1. load a patient's preoperative imaging and available functional/connectivity evidence;
2. generate several complete-instrument sequential resection strategies;
3. estimate their geometric feasibility and target benefit;
4. estimate functional/network consequences using the best validated patient-specific and population evidence;
5. run a bounded neurological-harm model grounded in observed postoperative outcomes;
6. propagate uncertainty;
7. compute robust Pareto alternatives rather than one magical best plan;
8. use search as the exact deliberative planner;
9. use a learned spatial model to guide or amortize expensive search;
10. update the plan after intraoperative spatial evidence changes;
11. display exactly why the recommendation changed;
12. preserve every assumption, unknown, model version, and source.

That would be a technically meaningful project.

It does not need to claim clinical readiness.

It needs to show that each layer is earned.

---

# 93. Final research philosophy

Use the enormous agent budget to think broadly.

Use experiments to narrow aggressively.

Read papers.

Challenge them.

Borrow useful mechanisms.

Combine fields.

Invent new hypotheses.

Run cheap tests.

Kill bad ideas.

Concentrate effort on survivors.

Preserve failures.

Keep protected patients protected.

Make the simulator answer to measurements.

Make learned models answer to strong baselines.

Make outcome claims answer to actual outcomes.

Make uncertainty explicit.

Make unknowns visible.

Make the desktop explain the science rather than hide it.

And keep asking the central question:

> Does this piece of work make the system better at producing, evaluating, explaining, or updating patient-specific surgical strategies in a way that is measurable and relevant to preserving neurological function?

If the answer is no, deprioritize it.

If the answer is yes, prove it.

Commit meaningful progress periodically, approximately hourly when a coherent milestone exists, using the existing `skamal23 <sayemkamal12@gmail.com>` author convention and `Co-authored-by: shafsthegoat <shafrir.p@gmail.com>`, and continue autonomously until the strongest defensible version of this research thesis is implemented, evaluated, integrated, and documented, or until a genuine evidence limitation prevents further progress.

If an evidence limitation stops a claim, preserve the limitation and finish the strongest supported subset instead of manufacturing an answer.
