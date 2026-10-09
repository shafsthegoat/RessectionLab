# Brain RL: patient-specific, instrument-aware glioma resection planning

## Research and product master plan

Prepared for Shafrir | Original research snapshot: **October 2, 2026**

**Patient-specific planning revision: October 4, 2026.** This revision makes single-patient simulation-based optimization a core workflow, specifies an implementable uncertainty-halo ladder, and adds time-aware molecular and clinical context. The original detailed plan is retained except where explicitly revised. These are design requirements, not completed experiments. New literature checks are identified in the bibliography; the original source library was not exhaustively re-audited in this revision.

**Repository and project name:** `RessectionLab`, using the owner's exact spelling. This is not a claim about trademark or name availability. Do not rename an existing working package merely to change its display name.

**Goal:** Build an original, technically impressive desktop research application that transforms multimodal glioma imaging into an inspectable planning environment; generates diverse, instrument-aware resection strategies using reinforcement learning and strong classical competitors; and explains the tradeoffs between modeled tumor removal, motor/language hazards, uncertainty and physical feasibility. The public software and evaluation should be credible enough for Medivis engineers to inspect and for a methods-oriented publication to be considered.

**Operating constraints:** Public datasets only. Local-first processing and training. The available $300 Google Cloud credits are an optional fallback, not a required foundation. No AR, robotic execution, hospital integration or clinical deployment in the first scope. No prescribed deadline. Technology recommendations are defaults: the implementing agent has final architectural discretion after measured experiments, while the scientific and data-integrity requirements below remain fixed.

**Evidence status:** Source documentation, research papers and dataset listings were investigated for this plan. Full imaging cohorts were not downloaded, reconstructed or experimentally validated. Some literature entries were verified only at the abstract/metadata level; the annotated bibliography explicitly says so. Dataset counts are release counts, not guaranteed usable training counts. The search is substantial and targeted, not an exhaustive systematic review.

**How to use these three documents:** Read this plan, `ANNOTATED_REFERENCES.md` and `IMPLEMENTING_AGENT_HANDOFF.md` together. They are sufficient to begin. `DATASET_MANIFEST.json`, `EXPERIMENT_PROTOCOL.md` and earlier acquisition checklists are optional supporting artifacts: reuse them when present, or create their minimal equivalents from this plan when absent. Missing ancillary files are not a blocker. Citations such as [D01] resolve to the reference directory and detailed bibliography. For project-design conflicts, this dated revision supersedes older chat prompts; it does not override repository security, permissions or applicable `AGENTS.md` instructions.

---

## 1. The project to build, and the distinction that keeps it credible

Build a **patient-specific surgical strategy explorer**, not a moving dot that reaches a tumor and not an alleged autonomous neurosurgeon. The visible result is an interactive collection of alternative plans. Each plan contains an access window, instrument configuration, executable simulated movements, a sequence of accessible tissue removals, optional information-gathering actions, a stopping decision, a residual-tumor representation, and an evaluation record.

The strongest initial question is:

> Given imperfect knowledge of a patient's motor/language anatomy and a constrained instrument set, which sequential resection strategies preserve the best modeled tradeoff between target removal, functional-network preservation, physical access, and robustness to uncertainty?

This is a constrained, partially observed planning problem. A route is only one part of a plan. Two plans with the same entrance can differ in instrument size, working angle, which tumor portion is removed first, whether additional mapping is requested, and where resection stops.

### 1.1 Three outputs, three meanings

The interface and API must distinguish the following.

| Output | Meaning | Initially supportable? |
|---|---|---|
| Motor/language hazard fields | Spatial evidence that modeled intervention may affect a relevant structure or network | Yes, with clearly identified patient-specific evidence and population priors |
| Model-conditioned encounter probabilities | Fraction of specified uncertain simulation worlds in which a precisely defined structural event occurs | Yes, conditional on a documented world model; not a real-world clinical probability |
| Probability of a new persistent neurological deficit | Calibrated prediction of an actual postoperative functional outcome for this patient and intervention | Not established by the public-data combination verified here |

The public data are strong enough to create a serious planning benchmark and product prototype. They do **not** automatically provide the coupled baseline examinations, actual surgical cavities, detailed intraoperative decisions and timed motor/language outcomes needed for the third row. Survival, tumor grade, KPS and extent-of-resection categories cannot silently stand in for those labels. [D01–D04, F01–F07]

Clinical-risk fields must therefore be nullable, with an explanation, rather than invented percentages. This is not a reason to remove probabilities from the project. It is a reason to make their event definition and evidential basis visible. “The modeled tool envelope encounters the reconstructed motor structure in 18 of 200 sampled worlds” is a meaningful statement about that simulation. “This patient has a 9% chance of permanent weakness” is a different claim.

### 1.2 Narrow the anatomy, not the ambition

Initial inclusion is adult, unilateral, supratentorial glioma with suitable imaging and one or more **user-defined or explicitly hypothetical cortical access windows**. Begin with superficial and moderately deep cases, where the app can demonstrate tool-aware accessible resection without pretending to solve every cranial approach.

Exclude brainstem/posterior-fossa cases, complex skull-base approaches, multifocal disease, major vascular encasement and severely distorted/reoperated anatomy from the first benchmark. These are scope decisions, not statements that those cases are clinically unresectable. Insular and strongly eloquent cases can become a declared challenge subset after the core behaves correctly.

The UCSF core data are skull-stripped. Consequently, an initial plan is conditional on an intracranial access window; it is not a validated scalp-incision, craniotomy or whole-head-position recommendation. A visual skull template must never masquerade as the patient's skull. [D01]

### 1.3 What would make this impressive

The differentiator should be the integrity of the whole chain: native imaging coordinates, uncertainty-aware anatomy, genuine instrument constraints, sequential planning, honest metrics, responsive visualization, and reproducible evaluation. A surgeon or engineer should be able to challenge a result and see which geometric assumption, source image, uncertainty parameter or objective caused it.

A successful demonstration could show a short corridor losing to a slightly longer corridor because the short one admits the tip but not the shaft; a larger instrument reducing the number of removal actions but losing access to a corner; or a policy requesting an observation before deciding whether to continue. These are proposed demonstrations, not claims that any current model has achieved them.

### 1.4 Route planning is the first product; patient-specific rehearsal is the core learning workflow

The immediate user-facing purpose is to **help a surgeon inspect candidate routes for glioma resection**. Deliver a route-comparison workspace before attempting a complete surgical digital twin. It must compare access windows, full instrument envelopes, reachable target compartments, motor/language evidence, uncertainty and missing anatomical information. Static accessible volume is not simulated removed volume; label those quantities separately.

The intended learning workflow is: upload and review the case; create its simulation and uncertainty ensemble; obtain initial candidates from search; train or refine a policy inside that particular patient's environment; independently evaluate the resulting candidates; and present an inspectable collection of alternatives. Actual patient-specific gradient updates are part of the research scope, not merely a renamed inference call. A pretrained policy can accelerate this workflow, but it is not a prerequisite for trying it on one patient.

An initial route planner need not solve tissue forces, hemorrhage, every instrument, or postoperative deficit prediction. It does need correct geometry and explicit limits. Missing vessels, functional evidence or skull anatomy remain unassessed, not automatically safe. Later stages add connected resection sequences, changing access, information gathering and evidence-constrained deformation. Do not let those extensions delay the first useful route comparison.

Return the **best evaluated candidates found under the stated model and compute budget**, not a guaranteed globally optimal or clinically safest operation. A local optimum or a sampled non-dominated set is not the complete clinical decision space.
---

## 2. Medivis alignment without making a copy of Medivis

Medivis Studio's published workflow connects imaging, segmentation, multiplanar visualization and planning. Frontier's Maia is described as a case-aware harness that invokes validated tools. Those are the useful design principles to adopt. The company separately distinguishes its research programs from cleared products. [M01–M03]

Use an original name, layout implementation and visual identity. Do not reuse logos, proprietary illustrations, patient screenshots or wording that suggests affiliation. The aim is recognizable product judgment, not a cloned marketing page.

### 2.1 The architectural translation

Maintain one persistent `CaseState`. A GUI, script or future assistant can call typed operations against it:

```text
import_case -> validate_inputs -> reconstruct_anatomy -> review_anatomy
            -> instantiate_worlds -> propose_plans -> certify_geometry
            -> evaluate_plans -> compare_and_replay -> export_case
```

Every operation takes versioned input objects and returns inspectable artifacts. A future language interface may select a tool or summarize an existing metric, but should not invent geometry, invent injury probabilities or act as the numerical planner.

### 2.2 The signature interaction

The user selects two candidate plans. Linked 3-D and MRI views show both corridors and the changing cavity. Switching instruments recomputes clearance, visibility and reachable target tissue. Increasing uncertainty can reorder the candidates. Selecting a rejected plan reveals the first failed constraint and its location. Selecting an information-gathering action shows the belief before and after a simulated observation.

The visual message is: **the model can explain its own planning assumptions, and the user can inspect and change them.**

---

## 3. Public data strategy: use different datasets for different jobs

Do not look for one miraculous dataset containing every required label. Assemble a small, versioned stack with separate roles and explicit eligibility tests. The most important distinction is between **data for reconstructing a patient**, **data for checking a simulator component**, and **data that could supervise a clinical outcome predictor**.

### 3.1 Acquisition priority and role map

| Priority | Resource | Role in this project | Key restriction |
|---|---|---|---|
| P0 | UCSF-PDGM v4 | Main structural-plus-diffusion patient environments | Repeated patients, BraTS overlap, skull stripping and gradient-space audit |
| P0 | Public DES atlas and language-map releases | Motor/language population priors and provenance | Positive-site sampling; not patient-specific injury labels |
| P0 | TractSeg software/model resources | Practical tract initialization and correction workflow | Tumor-domain failures require QC |
| P1 | UPenn-GBM usable diffusion subset | Independent-site tract-aware evaluation | Original DICOM and processed NIfTI are not in the same frame |
| P1 | ReMIND glioma subset | Intraoperative imaging, registration and shift experiments | No full tool-action stream; sparse stage-specific labels |
| P1 | RESECT / EASY-RESECT | Independent MR/US registration and shift checks | Processed subset is not a new cohort |
| P1 | UTSW-Glioma reviewed subset | External structural segmentation robustness | Not an external diffusion-aware planning test |
| P2 | BTC pre/post OpenNeuro | Small external diffusion/fMRI and longitudinal branch | Very small glioma postoperative sample |
| P2 | MU-Glioma Post | Post-treatment cavity/segmentation morphology | Follow-up imaging is not a simulated action transition |
| P2 | TractoInferno subset | Healthy multi-site tractography QC | Healthy, algorithm-derived reference tracts |
| P3 | BITE | Older external MR/US stress test | Mixed tumors, legacy format, verify reuse terms |
| Excluded core | IMAGO | Potential future structural/longitudinal cohort | Application/agreement access, not the public-only baseline |

The dataset manifest records official URLs, format/size information when verified, exact roles, license status, overlap and exclusions. Never sum these rows into one headline “number of training patients”: many rows are atlases, repeated visits, derived files or different task-specific cohorts.

### 3.2 UCSF-PDGM: first acquisition and main environment source

Use release v4, including the four-dimensional diffusion volumes and gradient files. The official release clarifies 495 unique patients and 501 exams; the complete NIfTI release is listed at 156.5 GB. Tumor labels and molecular information make it useful beyond an arbitrary segmentation example. [D01, D02]

Start with five quality-controlled cases spanning easy access, motor-adjacent anatomy, language-adjacent anatomy, a large lesion and a reconstruction failure. Expand to a development pilot before attempting the full cohort.

For each case, verify gradients, diffusion-to-structural transformations, left/right orientation and the provenance of every supplied derivative. Do not treat scalar FA/ADC images as substitutes for raw directional diffusion. Group repeated patient IDs and use the published BraTS correspondence to detect overlap with any segmentation-model pretraining. Store the source mask separately from the online model's mask.

### 3.3 UPenn-GBM: external evaluation with real diffusion

Use the raw DICOM availability/acquisition tables to select baseline glioma cases with reconstructable diffusion. The cohort includes 630 patients, but that is not the eligible diffusion test count. Its processed NIfTI and source DICOM are deliberately in different spaces. [D03, D04]

A laboratory-maintained derivative resource lists 531 ready-to-track FIB files. These can shorten exploratory work, but they remain UPenn patients and have their own modeling/coordinate conventions. Use them as an optional acceleration path, with a small independently reconstructed comparison set. [D05]

Hold UPenn out of policy development initially. Do not tune repeatedly on its result and continue calling it external validation. Limit downloads to needed radiology files; the pathology package is unrelated to the first project.

### 3.4 Language is the second endpoint

The decision is **motor plus language**, with motor as the first engineering milestone. The reason is the availability of operative stimulation-derived public maps, including language-component maps, rather than a claim that language is simpler than vision. [F01–F04]

Use the multi-domain DES release for functional prior construction and the language-specific release for semantic, phonological and speech-articulation submaps. Keep those submaps inspectable even when the UI presents a combined language-hazard view. The public maps are small, but the clinical outcome cohorts behind the papers are not automatically downloadable training data.

Register population priors into patient space with lesion-aware QC. Present them as priors. Without patient fMRI, nTMS or stimulation, do not claim a personalized functional boundary or assume every person's language system is left-lateralized. Estimate both hemispheres and expose lateralization uncertainty.

### 3.5 ReMIND, RESECT and BITE: dynamic anatomy, not magical physics labels

ReMIND contains 114 tumor cases, including 92 gliomas, with preoperative and intraoperative imaging. Its release is about 44 GB. Segmentation availability varies by structure and operative stage. [D12, D13]

Use it to evaluate MRI/US registration, residual-tumor handling, plausible shift distributions and the ability to update the displayed anatomy. Do not label two images separated by unknown actions as a supervised transition for an arbitrary suction-tip motion.

RESECT's 23 low-grade glioma cases and landmarks are valuable independent registration data. EASY-RESECT is a processed subset. BITE adds a separate older and smaller stress test. [D14–D17]

A learned registration field is an estimate of image correspondence, not automatically tissue mechanics. Evaluate plausibility and held-out landmarks, and avoid forcing a one-to-one deformation through removed tissue.

### 3.6 Structural and longitudinal secondary resources

UTSW-Glioma provides a recent structural external cohort. Use its manually refined subset when evaluating reference segmentation; do not treat all automatic masks as independent ground truth. It does not supply the same diffusion input contract as UCSF. [D06, D07]

BTC provides a small public multimodal complement with pre/post visits, diffusion and resting-state fMRI. Use only the relevant glioma subgroup for the main task, and treat its longitudinal functional analysis as exploratory. [D08–D11]

MU-Glioma Post can support cavity and post-treatment segmentation work. Before reuse, distinguish immediate postoperative morphology from later treatment change and check overlap with pretrained postoperative segmenters. [D20]

TractoInferno is useful for upstream robustness tests, not for learning how tumors displace function. Download a small targeted subset rather than its full roughly 350 GB release. [D18]

### 3.7 What remains unavailable in the verified public stack

No verified core resource couples the complete set of patient-specific multimodal anatomy, continuous instrument identity/pose/force, incremental tissue removal, direct mapping observations and timed motor/language outcomes. Nor is there a verified open dataset that tells us the consequences of every alternative operation on the same patient.

Therefore, the main RL trajectories must be **generated inside a declared simulator**. They are not historical surgeon trajectories. The scientific question becomes whether the proposed learning/planning method performs robustly under increasingly evidence-constrained simulation, with honest limits on clinical transfer.

### 3.8 How one patient's scans become RL training data

A single eligible public case can instantiate many simulator episodes: alternative access windows, feasible tool configurations, action sequences, preferences and coherent plausible anatomical realizations. These episodes supply interaction data for optimizing that case. A large historical collection of surgeon action trajectories is not required to test this form of RL.

This does not turn one patient into many independent patients. Cohorts still support population priors, optional shared-policy training, algorithm selection and evaluation of the entire planning procedure across anatomies. They do not automatically identify the consequences of actions never observed clinically. The current UCSF/UPenn-centered stack remains appropriate; this revision changes its use rather than inventing a new complete surgical-outcome dataset.

Begin with one inspected patient for the closed-loop prototype, then a small multi-case development pilot. Do not bulk-download a cohort or finish population pretraining before the first patient-specific learning experiment. Conversely, do not claim general utility from success on that one case. Source images, derived models and simulated experience must remain separate in the data registry.
---

## 4. Data contracts, provenance and leakage prevention

### 4.1 A case manifest is the first real deliverable

Each case needs a machine-readable manifest containing source collection/version, stable patient and visit identifiers, file hashes, modality mapping, acquisition metadata, native voxel-to-world affines, all transforms and their direction, label conventions, available functional information, QC results, licensing, and split membership.

Separate data by provenance:

```text
observed:     acquired imaging, recorded clinical fields, supplied annotations
estimated:    segmentations, registrations, tract reconstructions, inferred surfaces
prior:        normative DES maps, atlas labels, generic tool assumptions
simulated:    action effects, hidden worlds, generated observations, episodes
```

Every metric and visualization should be traceable through these categories. A simulated stimulation result must not be stored as an observed patient measurement.

### 4.2 Split people, not files or episodes

Create a frozen split by unique patient before model tuning. A reasonable initial proposal is 70/15/15 within the quality-controlled UCSF cohort, stratified coarsely by lesion location/volume and source metadata when possible. The implementing agent may choose another justified split, but repeated patients, longitudinal exams and derived versions must remain in the same group.

For policy research, the supplied tumor annotations may define the environment even when an upload segmenter was pretrained elsewhere. Report this as the **annotation-assisted benchmark track**. Report a separate **end-to-end inference track** using model-generated anatomy. This prevents segmentation failures from being hidden while also avoiding a paper that confounds every planning result with an upstream model change.

Record any overlap between upstream segmentation training and evaluation patients. A novel policy test can still be informative with an existing segmenter, but “fully unseen end-to-end patients” requires a stronger provenance claim.

For patient-specific optimization, distinguish **global method development** from **predeclared adaptation on a new patient's permitted inputs**. A frozen adaptation algorithm may train inside a held-out patient's preoperative simulator as part of its evaluated inference procedure. That is not zero-shot evaluation, and it must not consume postoperative outcomes, hidden evaluation worlds or cross-patient test feedback. Section 15.8 specifies the nested evaluation needed for this distinction.

### 4.3 Exact spatial semantics

Keep the acquired patient frame authoritative. Store LPS/RAS conventions explicitly, use millimeters throughout physical geometry, and test transformations on synthetic landmarks with known answers. Record qform/sform disagreements, anisotropic voxels, oblique acquisitions and handedness.

Plan geometry must not use a nonrigid atlas-normalized shape as though it were the patient's original dimensions. Atlas maps should be brought into the patient frame through documented transforms. For public releases that only supply a normalized representation, label the simulation frame and the unavailable native anatomy accordingly.

Diffusion orientation information needs its own handling. Motion correction, rotations, vector fields and nonlinear warps cannot be treated as ordinary scalar-image interpolation. Choose a supported diffusion pipeline and test it with known synthetic fiber directions before processing the cohort. A plausible-looking CST on the wrong side is a failed case, not a successful render.

### 4.4 Acquisition gates

The download pipeline should first retrieve metadata and a small sample. It should estimate source plus temporary plus derived storage, verify terms, support resumption, and hash files. Do not commit patient volumes into Git. Prefer source fetchers, small redistributable synthetic fixtures and downloadable release assets only where licenses permit.

A failed case receives a named reason such as `MISSING_GRADIENTS`, `TRANSFORM_UNRESOLVED`, `TRACT_QC_FAILED` or `OUT_OF_SCOPE_ANATOMY`. Do not silently replace it with an atlas and continue calling it patient-specific.

---

## 5. From uploaded images to a planning-ready patient representation

### 5.1 Input modes

**Full initial mode:** T1, contrast-enhanced T1, T2 and FLAIR, plus usable raw diffusion with b-values/b-vectors or equivalent DICOM metadata. Import a supplied tumor mask when available. Functional maps, vascular imaging and clinical metadata are optional inputs with explicit provenance.

**Restricted structural mode:** Without usable diffusion, the app can display anatomy and run clearly marked geometry/atlas-prior demonstrations. It should not produce the same confidence or functional-risk labels as the full mode.

**Extended mode:** Patient fMRI, nTMS or stimulation imports, pre/post or intraoperative images, and appropriate angiographic sequences add separate capabilities. Missing inputs should be visible, not imputed to a “normal” finding.

### 5.2 Segmentation and correction

Begin with public reference masks for simulation development and a separately audited pretrained segmenter for real uploads. nnU-Net is a suitable framework candidate, but using its name does not identify a checkpoint, a training population or a license. Record all three. [E01]

Represent enhancing tissue, non-enhancing/necrotic core and FLAIR abnormality separately. The objective should refer to these radiological compartments, not pretend the FLAIR boundary is the boundary of resectable tumor. Permit manual review and correction, retain the original mask, and invalidate dependent plans after edits.

Construct cerebrum, cortical surface, ventricles and other exclusion/constraint structures as supported by the input. Segmentation confidence is not anatomical truth; thin structures and boundary uncertainty need explicit attention.

### 5.3 Diffusion, tracts and networks

A sensible pipeline is an established diffusion preprocessor, an appropriate model for the actual acquisition, tractography/bundle reconstruction, and expert-style quality checks. The agent can choose MRtrix, DIPY, TractSeg or a combination, but should avoid retraining an entire tractography foundation model for this project.

Motor representations should include reconstructed corticospinal pathways, their cortical endpoints and relevant motor cortical priors. Language representations can combine available patient tracts with the DES-derived component priors. Preserve evidence about missing or unreliable segments rather than interpreting them as empty safe space. TractSeg and atTRACTive are useful implementation and correction references. [F09–F12]

Maintain both a spatial representation and a graph. Spatial fields answer where a tool passes; a graph answers which modeled connections are interrupted. A graph assembled from tractography remains a model of connectivity, not direct observation of every axon.

### 5.4 Reconstruction acceptance tests

Before enabling planning, a case must pass image alignment, orientation and unit tests; tumor-mask inspection; diffusion gradient checks; tract endpoint/continuity checks; and provenance completeness. For the first five cases, generate a reproducible QC report with linked views. For the larger cohort, use automated flags plus spot checks and retain all exclusion counts.

---

## 6. Motor and language risk representation

### 6.1 Start with transparent features, not a black-box clinical predictor

For voxel v, retain a feature vector rather than a single unexplained number:

```text
radiological target memberships
motor cortical prior / patient functional evidence
CST evidence, distance, local diffusion features, reconstruction disagreement
language component priors / patient functional evidence
relevant tract evidence and graph membership
registration and segmentation uncertainty
known anatomical exclusions and unknown-data flags
```

A transparent baseline hazard can combine several normalized components:

\[
H_k(v)=\alpha_k A_k(v)+\beta_k B_k(v)+\gamma_k U_k(v),
\qquad k\in\{motor,language\}.
\]

Here A is cortical/functional evidence, B is tract/network-associated spatial evidence, and U is uncertainty. These weights are declared research settings, not clinically calibrated injury coefficients. Keep the components available so the user can see whether a hotspot is patient diffusion evidence, a population prior or simply poor localization.

Do not interpret streamline count, a normalized atlas density or a sigmoid-transformed hazard as an injury probability. Avoid naming raw tract density `P_CST` unless a defined probability model and calibration justify that label.

### 6.2 Add topology to the resection evaluator

Let R be the union of removed tissue and let E contain tool contact, retraction and other modeled exposure. Evaluate spatial intersection and changes to the connectivity representation separately. For example:

\[
C_k(R,E)=a_k I_k(R)+b_k D_k(G,R)+c_k X_k(E).
\]

I is spatial interaction with relevant tissue; D is a defined connection-disruption score; X is a modeled exposure cost. A streamline or graph edge should not be counted as a newly severed connection each time another voxel on the same already-cut trajectory is removed.

This is still a surrogate. Redundant pathways, plasticity and network physiology are not established by a simple graph. Compare graph-aware and purely spatial variants as an ablation, rather than assuming the graph version is biologically correct by construction. Connectivity-based eloquence costs already appear in prior planning research. [P01, F01, F03]

### 6.3 Coherent uncertainty worlds

Create an ensemble of plausible worlds W conditional on the available imaging and assumptions. A world can vary registration alignment, tract location/continuity, segmentation boundaries, observation noise, deformation parameters and instrument tolerances. Perturbations should be spatially correlated and anatomically coherent. Do not independently jitter every voxel or break a tract into biologically meaningless noise.

A candidate policy is rolled out in these worlds. For a clearly specified event V, report:

\[
\widehat p_{model}(V\mid\pi)=\frac{1}{M}\sum_{m=1}^{M}
\mathbf 1\{V\text{ occurs when }\pi\text{ is executed in }W_m\}.
\]

The UI should say what V is, which world generator was used, how many samples were evaluated, and whether the displayed interval represents finite Monte Carlo error or broader model variation. The latter is not measured just by increasing M.

Use mean costs, adverse quantiles and conditional value at risk as separate summaries. Sampling a narrow or wrong world distribution can make any plan look robust. Therefore test deliberately misspecified distributions and report ranking instability. [R04, R06]

### 6.4 When a voxel heatmap can be plan-specific

Besides the static evidence maps, display the marginal change in a plan's surrogate score when an additional local patch is removed or exposed. This is conditional on the current cavity and network state. It can reveal why removing a bridge-like patch matters more than its isolated voxel intensity suggests.

Name this a **marginal modeled consequence map**, not a patient deficit-probability map. Recompute it after material plan changes; do not recycle a static heatmap as a dynamic explanation.

### 6.5 A clinical outcome head is an explicitly gated extension

A future supervised model would need baseline function, actual resection extent/location in a defensible common frame, relevant imaging and mapping, and prospectively specified outcome definitions and follow-up times. A proposed schema could separate transient motor change, persistent motor change and language-domain deterioration from baseline, with the actual study defining the instruments and time windows.

A transparent penalized logistic or hierarchical model should be the first clinical baseline, before a large 3-D multimodal network. Evaluate calibration, discrimination, subgroup support and missingness on independent patients/sites. No such model is required to complete the public-data research project.

Importantly, prediction for historical surgeries does not by itself identify the outcome of surgeries doctors almost never attempt. Selection, confounding and poor support for risky counterfactuals remain even after adding a neural network. Do not use synthetic deficit labels to conceal this gap. [F05–F07]

### 6.6 Implementable halos: geometry first, uncertainty second, consequences third

A useful halo does not require an already validated clinical deficit predictor. Implement three increasingly informative layers and keep their meanings separate.

**Layer H0: geometric proximity.** Given an estimated tract/cortical structure, compute physical distance in millimeters. A reference proximity field is

\[
h(v;s)=\exp\{-d(v,\mathcal A)^2/(2s^2)\},\quad s>0,
\]

with distance zero inside the estimated structure. A signed-distance band or dilation is an equally valid first implementation. This is a chosen distance penalty, not a probability. Its scale is a documented sensitivity parameter, not a universal safe surgical margin. Show the original structure beneath the halo and permit inspection of distance and units.

**Layer H1: uncertain anatomical support.** Reconstruct coherent alternative tract/segmentation/registration realizations, using bootstrap or probabilistic diffusion methods where the acquisition supports them. For structure masks A_m, the empirical support field is `sum_m 1[v in A_m] / M`. It describes this reconstruction ensemble, not the chance of losing motor or language function. Resampling streamlines from one biased fit captures less uncertainty than varying supported reconstruction assumptions. Bootstrap tractography and the official DIPY implementation are concrete starting references. [F13, E06] Simulation-based diffusion inference is a later optional method, not required infrastructure. [F14]

**Layer H2: action- and plan-conditioned events.** For each sampled world, intersect the full swept tool/contact envelope and the proposed removal sequence with the relevant structures; evaluate network interruption separately. Store counts for explicitly named events such as `motor_structure_contact`, `language_bundle_intersection` and `modeled_connection_disconnection`. Do not sum voxel occupancies and call the result a plan-level probability. Nearby voxels and streamline segments are not independent.

Treat instrument size, pose uncertainty, anatomical localization error and tissue-interaction assumptions as different quantities. Do not inflate the tract for shaft radius and then apply the full shaft again without accounting for double counting. Uncertainty is usually spatially correlated and may be directional; a spherical blur is an initial sensitivity model, not a physiological truth.

A missing or implausible tract reconstruction produces an unknown-coverage region or failed eligibility gate, not an empty low-risk field. The known ambiguities in diffusion reconstruction make this essential. [F09] More sampled worlds can reduce Monte Carlo noise without correcting a systematically wrong anatomical model.

### 6.7 Halo and event-model acceptance tests

Verify distance fields against analytic objects at several voxel spacings; report physical rather than voxel distances. A fixed straight tool must fail clearance when its shaft, but not its tip, crosses a known obstacle. For a fixed plan and nested geometric uncertainty envelopes, conservative encounter checks must not decrease merely because the envelope expanded. This monotonicity test does not imply that a newly optimized plan or every ensemble probability must be monotone.

Every rendered halo needs a source, frame, generation method, parameter values, QC state and meaning. Every probability needs an event, world-model version, numerator, denominator and sampling uncertainty. Fit/selection worlds and final evaluation worlds must be disjoint. Inspect alternative perturbation families, since a perfectly calibrated frequency inside the same simulator says little about model misspecification.
---

## 7. Instrument-aware simulation: make the tools change the answer

### 7.1 What “simulate the tool” means in the first serious version

The first useful instrument model should certify geometric feasibility and expose a declared tissue-interaction approximation. It does not need to claim quantitatively accurate tissue cutting, thermal injury or hemostasis.

Represent each tool using:

```text
instrument family and configuration ID
shaft geometry and length; tip shape and active region
allowed pose and orientation range
entry-window / access constraints
working distance and view/visibility requirements
removal footprint or manipulation primitive
contact/retraction and optional thermal-surrogate parameters
paired-tool collision envelope
parameter source, confidence and validation status
```

Use public manufacturer specifications for dimensions only when the exact configuration is verified. Use generic parameterized tools otherwise. A device's marketing claim about tissue selectivity must not become a simulated clinical fact. [I01, I02]

### 7.2 Initial instrument families

**Suction/cannula model:** A rigid shaft and tip with a simple exposed-tissue removal primitive. Vary diameter and reach. Model approach direction, visibility and obstruction, not just the tip center.

**Ultrasonic aspirator model:** A distinct tip/shaft and controllable removal footprint. Initially represent tissue effects with phenomenological parameters and a declared uncertainty range. The useful first comparison is access and removal geometry, not a fabricated estimate of nerve selectivity or heat injury.

**Bipolar/dissection or forceps model:** Represent access, opening/closing envelope, bimanual interference and manipulation capability. A first implementation may use one generic manipulation family. Quantitative coagulation, bleeding control and thermal spread are a later calibrated physics problem.

**Mapping probe and intraoperative imaging probe:** These are information-gathering tools. They incur access/effort costs and produce noisy observations. They must not simply expose ground truth for free.

**Later access systems:** Parameterized tubular retractors and side-cutting devices can extend the tool catalog. Access/retraction and removal are separate roles. Do not begin with dozens of near-identical proprietary models or add laser ablation as though it were just another suction tip.

### 7.3 The geometric rule that prevents a fake simulator

A rigid tool does not follow an arbitrary curved centerline through solid tissue. Check the entire shaft and its swept volume throughout movement, including entry-window constraints and collisions with another tool. A* may supply a coarse accessibility heuristic; it does not certify a rigid surgical instrument trajectory.

Low-level planning should operate in instrument configuration space, with a tested collision checker. OMPL can supply planners, not the brain-specific validity predicate. [P06]

Likewise, a tool cannot remove a tumor voxel merely because the voxel is valuable. Removal requires a feasible pose, an allowed tissue-interaction primitive, appropriate exposure from the current cavity or specified access corridor, and consistent geometry after the action. Account for any normal tissue disturbed while establishing access. Do not mark all intervening brain as empty at episode initialization.

### 7.4 Cavity and tissue bookkeeping

Store remaining target masks, removed tissue, manipulated/exposed tissue, connected free space and the current cavity frontier separately. Removal is irreversible. The active tip must reach the frontier through available access. An enclosed internal voxel cannot disappear without a connecting removal or valid instrument model.

If a tool model produces fragments, the simplest initial version can require removal through an explicit available extraction route or use a suction-removal abstraction. State the abstraction. Do not award benefit for “removed” tissue still physically occupying the cavity unless the model explicitly represents that distinction.

### 7.5 Fidelity ladder

**Level A:** Verified static geometry, cavity updates, reach, swept-volume collision, tool-change costs and repeatable removal primitives.

**Level B:** Coherent uncertainty, simplified displacement/retraction, noisy observations and replanning.

**Level C:** Reduced-order or position-based deformation with parameters checked against public intraoperative image changes; separate validation for registration versus mechanics.

**Level D:** Higher-fidelity contact/resection models on a small subset, potentially with SOFA. Use as a stress-test backend, not necessarily the training environment.

Segato's needle-planning work and the LapGym/SofaGym ecosystem offer relevant ideas, but their tasks and tissue models are not evidence of accurate glioma tool mechanics. [P04, S01–S05]

---

## 8. Define a good plan before training anything

### 8.1 Feasibility first, tradeoffs second

A plan must pass deterministic checks for its stated world/model: input validity, access-window validity, tool reach and shaft clearance, allowed tissue-removal connectivity, temporal consistency, visibility constraints when modeled, and required structures/observations.

Known critical anatomy can impose exclusion constraints. Unknown vascular anatomy should not be assigned a risk of zero. Cases or corridors whose feasibility depends on unsupported vascular information should be marked incomplete or excluded from the relevant claim. Preserving two displayed functional systems does not establish overall surgical safety.

Only feasible candidates enter the main Pareto comparison. A physically impossible “high-resection” plan may be displayed in a rejected-plan gallery with its failure, but it must not sit beside valid candidates as an alternative recommendation.

### 8.2 Objective vector

Use an objective vector rather than a single opaque goodness score:

\[
J(\pi)=\left(
B_{target},
-C_{motor},
-C_{language},
-C_{normal},
-C_{exposure},
-C_{effort},
-C_{uncertainty}
\right).
\]

B is radiological target removal under a declared compartment policy. C_normal distinguishes removed from merely displaced tissue. C_effort is an action-count, motion-length or tool-change proxy unless real operative-duration validation exists. Display benefit by target compartment rather than only a single overall percentage.

Model uncertainty should affect both ranking and confidence. Two plans with similar average performance but different adverse tails are worth showing separately.

### 8.3 A concrete constrained formulation

For a research-selected preference setting w and surrogate budgets b_m, b_l, optimize:

\[
\max_{\pi}\; \mathbb E[B_w(\pi)]-\lambda_e\mathbb E[C_{effort}(\pi)]
\]

subject to geometric validity and, for example,

\[
\mathrm{CVaR}_{\alpha}(C_{motor})\le b_m,
\quad
\mathrm{CVaR}_{\alpha}(C_{language})\le b_l.
\]

These are **surrogate-unit budgets**, not acceptable percentages of patient paralysis or aphasia. The agent may use other coherent risk criteria, but must document units and avoid presenting researcher-selected thresholds as clinical guidelines.

Generate a non-dominated candidate set across a range of declared preferences or constraints. Include diverse entrances and instrument configurations rather than near-duplicate plans that only differ by random seed. Weighted sums alone can miss parts of a non-convex frontier, so an epsilon-constraint or search-based comparator is valuable.

### 8.4 Reward design and STOP

A useful reward can be based on changes in already-defined cumulative quantities:

\[
r_t=\Delta B_t-\lambda_m\Delta C_{m,t}
-\lambda_l\Delta C_{l,t}-\lambda_n\Delta C_{n,t}
-\lambda_e c_{effort,t}.
\]

Take care with discounting, normalization and terminal terms. Do not count the same injury cost repeatedly at every step and again at termination. Do not reward instrument cycling, idle movement, repeated removal of the same tissue, or avoiding the episode's end.

`STOP` is always available. A plan leaving substantial residual target can be valid under the declared constraints. Compare it to a baseline that stops immediately so the reward does not make universal abstention look like a successful learned strategy. Conversely, a reward that always favors complete mask removal is also a failure.

### 8.5 Include tumor biology without inventing a genetics-to-injury equation

Make molecular context part of the case schema now, while keeping a learned biological-benefit model optional. Relevant fields include assayed IDH1/IDH2 status, 1p/19q codeletion, MGMT promoter methylation and the integrated diagnosis where available. MGMT methylation is an epigenetic measurement, not a germline variant. Molecular diagnosis and extent-of-resection decisions are related clinical contexts, but neither guideline supplies a validated genotype-to-voxel-injury function. [C01, C02]

Separate three uses. **Context and stratification:** display verified metadata and report results by supported subgroups. **Research objective scenarios:** compare declared compartment priorities under alternative biological assumptions, labeling these as scenarios. **Clinical benefit prediction:** defer numerical survival or individualized treatment-benefit rewards until suitable data, causal assumptions and validation exist. Retrospective survival/EOR associations are not a license to assign a survival gain to every additionally removed voxel.

The primary benefit term remains declared radiological target removal with residual volume reported by compartment. The EANS-EANO guideline favors absolute residual postoperative tumor volume for describing resection completeness and distinguishes integrated tumor types; it does not make FLAIR abnormality synonymous with disposable tumor. [C01] In a simulation, call the metric **modeled residual target volume**, not an observed postoperative result.

A molecular label alone must never relax motor/language protection constraints, shrink a hazard halo or imply that a more aggressive tumor makes functional tissue less important. Whole-genome or inherited-risk modeling is outside the initial scope unless a specific validated relationship and usable data justify it.

### 8.6 Patient physiology and the preoperative information cutoff

Use a versioned `PatientContext` with `planning_as_of` and a per-field record of value, unit, source, measurement time, availability time and evidence type. Distinguish `observed`, `estimated`, `unknown`, `not_yet_available` and `scenario_assumption`. A dataset column being populated does not prove it was known before the operation.

When actually available, retain baseline motor/language assessments, prior surgery/radiotherapy, edema and other relevant clinical/imaging context. Handedness is not a patient-specific language-localization measurement. Age or a performance score is not a direct mechanical tissue parameter. Do not invent compliance, vascular reserve, functional reorganization or deficit tolerance from an incomplete record.

For an initial operation, a molecular result obtained from the eventual resection cannot be fed retrospectively into the preoperative planner. Results from a prior biopsy can be used only when availability is established. When timing is absent, the primary analysis must treat the value as unavailable; a separate oracle-context or exploratory scenario must be labeled accordingly. An imaging-predicted genotype remains an uncertain prediction, not confirmed pathology.

The ordinary plan must run with missing molecular metadata. Context may justify a declared research scenario or stratification, but geometry, functional evidence and clinician-inspected assumptions remain the basis of route comparison. These proposed data contracts prevent information leakage; they are not claims that the public datasets contain every listed measurement.

---

## 9. The RL problem and a local-first model architecture

The primary, falsifiable transfer question is whether a policy trained with
rich, anatomically and physically informed simulation feedback can make better
decisions for **unseen patients using only realistically available images and
observations** than strong search and learned-model planning under the same
information restrictions. A simulator may use privileged anatomy and physical
parameters to generate training transitions or critic/teacher signals; the
deployed actor, action generator and test-time search cannot read them. Planning
strategies must be frozen before an independent evaluator opens any withheld
patient-linked anatomy. This makes partial observability and leakage control
central to the research question. The [experiment protocol](EXPERIMENT_PROTOCOL.md)
defines the prospective outer split, comparator and evidence levels.

The existing single-patient optimization program remains a necessary
development and product mode. It tests whether training updates improve a
particular modeled case; even excellent performance there does not establish
transfer to a new brain. Conversely, a strong learned-model/search hybrid is a
valid winning result. Simulator reward, anatomical generalization and physical
or clinical validation must be reported as separate levels of evidence.

### 9.1 Single-patient RL is a legitimate optimization mode

**Train or adapt inside the uploaded patient's simulator.** One anatomy can generate enough simulated interactions to test instance-specific RL. Learning a strategy for a known case is different from estimating a clinical injury law from one person. The former is a valid computational formulation; the latter is not established by repeatedly replaying synthetic actions.

This is a substantive correction to the earlier population-policy-first recommendation. Population pretraining may be useful, but it must not be a gate that prevents single-patient experiments. Case-specific training is not automatically a statistical mistake merely because the anatomy repeats. The dangerous overfitting is to reconstruction errors, a narrow uncertainty model, reward loopholes or a repeatedly consulted evaluation set.

An adjacent-domain precedent is the 2025 patient-specific RL proton-therapy replanning study, which trains an agent using one patient's planning CT and augmented anatomies. It supports investigating the architecture, not claiming that radiotherapy results validate surgical tissue simulation. Its scoring and transitions belong to another intervention. [R09]

Compare these four modes without assuming the winner:

| Mode | What happens after a new case is prepared | Role |
|---|---|---|
| `SEARCH` | Per-case search, trajectory optimization or MPC | Strong non-learning reference and initial candidate generator |
| `PATIENT_SCRATCH_RL` | Initialize a compact policy and train using only this case's permitted simulator | Required instance-specific learning comparator; feasible before population pretraining exists |
| `POPULATION_FROZEN` | Run a policy trained on development patients without case-specific gradient updates | Measures amortized inference/generalization |
| `POPULATION_ADAPTED` | Clone the frozen shared policy and refine the clone on this case under a fixed budget | Tests whether warm-started patient-specific learning improves the quality/time tradeoff |

The first two modes can be implemented and compared on one case. Add the population arms once development data and a shared checkpoint exist; the full benchmark should contain all four. Do not represent a missing arm as an evaluated failure. The product can default to the strongest measured mode, with a transparent option to refine further within an explicit budget.

The UI must distinguish policy inference, tree/trajectory search and actual gradient updates. Show the data/model version, optimization budget and checkpoint provenance. A moving progress bar or repeated rollout of a frozen policy is not patient-specific training.

### 9.2 Formal environment specification

Use a constrained partially observable Markov decision process when uncertain anatomy or information gathering is enabled. In the static fully observed version, a simpler MDP is sufficient.

The latent state x_t contains remaining tissue, the cavity, actual sampled anatomy, deformation, instrument configurations, accumulated exposures and an episode-fixed hidden world. The observation o_t contains only the images, estimates and observations available at that stage. The belief b_t summarizes uncertainty over relevant hidden quantities.

The policy receives:

```text
local multi-resolution 3-D fields around the cavity/tool
remaining target compartments and exposed frontier
motor/language spatial evidence and uncertainty
compact graph / tract-disruption features
tool configuration and feasible macro-action mask
remaining surrogate budgets / preference vector
history summary or recurrent state
available observations and their age/quality
```

Do not give the deployed policy latent tract positions, future ultrasound or true tissue-response parameters. Privileged information can be used for a separately declared training critic, but never for the actor during evaluation. Use the same hidden sampled world across the episode; resampling the brain after every action can produce impossible histories.

### 9.3 Hierarchical actions

The high-level policy selects an access configuration, next accessible target patch, instrument family/configuration, local resection direction, information action, or STOP. A low-level deterministic controller verifies and executes a feasible motion primitive.

Initially, a macro-action might be “remove this exposed patch using this legal tip pose” rather than “set three motor torques.” This keeps the task computationally tractable while retaining the important sequential decisions. Later continuous pose refinement can be added where it changes the answer.

A candidate action generator can supply a small set of feasible frontier actions. The policy ranks them, possibly with a graph-attention or set encoder. Include the candidate generator in the benchmark definition: an RL method cannot claim superiority when it secretly receives a better search space than its competitors.

### 9.4 Suggested architecture

Start with a modest 3-D convolutional encoder for a cropped region, a small MLP or graph encoder for tract/network summaries, a tool-state encoder, and a preference/budget embedding. Fuse them into actor and value/cost heads. A recurrent layer becomes useful only if the observation history contains information not represented in the belief summary.

A transformer over millions of whole-brain voxels is not necessary. A small attention block over candidate actions or network nodes may be justified after profiling and an ablation. Make all representations resolution-aware and preserve physical scale.

The agent should begin with a trainable model small enough to run on the available machine, then justify additional capacity by held-out benefit rather than visual sophistication.

### 9.5 Training objective and algorithm ladder

A masked PPO actor with separate reward and cost critics is a reasonable first implementation. A Lagrangian multiplier update can target surrogate constraints, while deterministic validation rejects geometrically invalid actions. CPO is an alternative worth evaluating if the constrained formulation becomes central. SAC is useful only when continuous control genuinely pays for its added complexity. [R01–R03]

A representative optimization is:

\[
\mathcal L = \mathcal L_{policy}
+ c_v\mathcal L_{value}
+\sum_k c_k\mathcal L_{cost,k}
-c_H\mathcal H(\pi)
+ c_{aux}\mathcal L_{aux}.
\]

Auxiliary losses, when used, should predict observable simulator quantities such as reachable-patch validity or local clearance, not invented patient deficits. Start without them unless they solve a measured learning problem.

A preference-conditioned policy is a candidate way to expose tradeoffs continuously. Compare it with separate policies or optimizers at fixed preferences, and consider Pareto Conditioned Networks only after a simpler method works. Original-method and code licenses need separate review. [R07]

### 9.6 Curriculum

**Stage 1: synthetic geometry.** Spheres, branching tubes, narrow access, known targets and exact small-problem solutions. The agent must learn that some short paths are impossible, that removal changes access, and that STOP can be correct.

**Stage 2: one real patient-derived anatomy.** Fixed tool family, supplied labels and deterministic geometric interactions. Train a compact patient-specific policy and compare it with search on this same case under a fixed budget. Use independent simulation realizations for evaluation once uncertainty is enabled. This establishes only within-case optimization, not clinical validity or generalization. Then test the frozen learning procedure across additional development cases; add shared-policy training when justified.

**Stage 3: instrument-aware sequential resection.** Add tool changes, shaft clearance, cavity evolution, bimanual constraints and accessible patch choice. This is the first major project milestone where RL has more to do than route finding.

**Stage 4: robust patient-specific optimization.** Add coherent registration, tract and segmentation perturbations. Compare scratch, frozen and adapted policies with search under matched online budgets. Optimize adverse-tail behavior and evaluate with withheld realizations and alternative perturbation families. Lightweight uncertainty is already needed for robust route comparison; this stage expands it, rather than postponing all uncertainty until after RL.

**Stage 5: information and deformation.** Introduce noisy mapping/imaging actions and evidence-constrained anatomy updates. Replanning must use only revealed observations.

Avoid adding all five stages at once. A policy that fails Stage 1 does not need a more realistic brain mesh; it needs a corrected environment, objective or learning setup.

### 9.7 An algorithmic stop rule for the research itself

If RL fails to beat a strong search/MPC baseline after a bounded, logged comparison, report that result. Keep the best planner in the product. The value of the environment, tool-aware constraints and uncertainty-aware comparison does not disappear because a non-RL optimizer wins.

The most interesting RL advantage may be fast conditional replanning across many patients/preferences, not better one-shot optimization with unlimited compute. Measure both solution quality and total computational cost, including policy training.

### 9.8 The per-patient optimization contract

Implement the following as a headless, resumable procedure callable from the desktop application.

1. **Prepare and review the case.** Validate modalities, transforms, target labels, tract evidence, access assumptions and the preoperative information cutoff. A failed essential input stops that analysis mode, not necessarily the whole viewer.
2. **Freeze the decision model.** Version the geometry, action primitives, reward/objective definitions, uncertainty generator and clinician/researcher-declared constraints before optimizing. Training may update the policy, not make the world easier to obtain a better score.
3. **Partition simulated experience.** Create separate optimization, checkpoint-selection and final-evaluation world/seed manifests. Preserve spatially coherent anatomy within each episode. Withhold independent model-family stress tests. Purely deterministic one-case tests must be described as deterministic optimization, not fabricated uncertainty validation.
4. **Produce initial routes.** Run instrument-aware search so the case has useful candidates even before a policy exists. Search-derived demonstrations or warm starts are optional; record their generation cost and do not secretly give only RL a richer action set.
5. **Train or adapt.** Use bounded wall time, environment steps and checkpoint intervals on a compact macro-action policy. For adaptation, copy the shared checkpoint; never overwrite it with one patient's updates. Log actual gradient steps and evaluate checkpoint selection only on its designated worlds.
6. **Select and freeze candidates.** Retain a diverse non-dominated set across predefined preferences. Reject plans failing physical/model constraints. Compare against initial search and STOP; improved training return alone is not sufficient.
7. **Independently evaluate.** Use untouched worlds and finer or independently implemented geometry checks. Do not send final evaluation rewards back into checkpoint selection or policy improvement. Report all failed checks and unsupported anatomical factors.
8. **Return an inspectable plan package.** Include initial routes, refined routes/sequences, tool settings, objective tradeoffs, model-event frequencies, stopping reasons, assumptions, runtime and reproducible replay. Actual clinical probabilities remain unavailable unless their separate gate has been met.

A user edit or genuinely acquired new image creates a new case version and invalidates dependent results. An observation may update a declared belief model, but that is separate from silently changing the reward or learning clinical physiology from simulated outcomes. Reusing a previously revealed final evaluation set for tuning converts it to development data; it cannot keep its held-out label.

### 9.9 What the agent can learn and what stays supplied

The agent can learn entry/tool selection within permitted access regions, target ordering, local feasible motions, stopping and later observation timing. The simulator supplies legal transitions and measurable surrogate consequences. A learned world model is optional and requires its own validation; it is not necessary to make single-patient RL genuine.

Do not jointly let the optimizer rewrite the motor/language cost definition, shorten tools, reduce uncertainty or alter cohort eligibility in response to poor scores. Research revisions to those assumptions are allowed only as new versioned experiments with their own evaluation, never as hidden changes within a successful run.

The phrase "maximize resection without affecting the patient much" therefore becomes a **declared constrained research objective**, not a complete physiological specification. Explicitly retain what is unmodeled, including unresolved vascular effects, tissue forces, postoperative function and microscopic infiltration. Optimization can identify useful modeled tradeoffs without claiming these unknowns have been solved.
---

## 10. Information gathering and brain shift: the strongest research extension

### 10.1 Why sequential information matters

In the static version, the planner chooses actions using a fixed map. In a partially observed version, it must also decide whether to acquire information before continuing. A possible simulated policy is: remove an accessible portion, inspect a newly exposed boundary, update uncertain tract location, and either continue along another direction or stop.

Direct stimulation and intraoperative imaging motivate this structure, but the simulator must encode their uncertainties and prerequisites. A language-task observation should not be available regardless of operative context or patient participation. A negative observation is not proof of absent function. [F01, F03, F06, I05]

### 10.2 Observation model

For a probe action at location z, define an explicit observation distribution p(o|x,z,mode). Include localization error, imperfect sensitivity/specificity, spatial extent, cost and circumstances in which an observation is unavailable. Use literature to motivate plausible scenarios, but label uncalibrated parameters as scenario assumptions.

Do not reward raw uncertainty reduction by default. Information is useful when it improves subsequent decisions. A policy should sometimes decide not to probe because it will stop anyway or because the observation cannot change a feasible choice.

A simplified belief update may operate on a low-dimensional local tract displacement/occupancy parameter. It does not require a whole-brain Bayesian neural network. Validate it first on synthetic worlds with known hidden truth.

### 10.3 Shift model

Separate three mechanisms: geometric uncertainty before any intervention, deformation during the modeled procedure, and registration error when observing that deformation. Otherwise, the policy can receive an implausibly accurate corrected map even though the actual registration is uncertain.

Use ReMIND to constrain the scale and spatial structure of observed changes and to benchmark image updating. Use RESECT independently to test registration behavior. Fit a cheap spatial displacement basis or reduced-order model only after deciding which measured quantities support it. [D12–D17, S04, S05]

Removal changes tissue topology. Registration should not force missing tissue to correspond to a preoperative voxel through an arbitrary extreme warp. Preserve cavity masks and evaluate landmarks away from removed regions separately from residual-tumor correspondence.

### 10.4 A falsifiable experiment

Test whether a policy with information actions achieves a better removal/hazard frontier than: never acquiring observations, acquiring them at fixed intervals, an uncertainty-threshold heuristic, and an MPC method with the same observation model.

Vary observation reliability and cost. The method should degrade gracefully as observations become less informative, and should not request unlimited free measurements. This creates a stronger methods question than simply adding “uncertainty” to a heatmap.

---

## 11. Baselines that prevent an RL-shaped science project

Use the same anatomies, tool models, action proposals, hard constraints and evaluation worlds for every method. Equalize online compute where possible, and also report an unconstrained-compute reference when useful.

| Baseline | What it tests |
|---|---|
| STOP immediately | Whether the objective rewards trivial abstention |
| Valid straight-access search | Whether elaborate learning beats simple corridor enumeration |
| Distance/voxel-cost route heuristic with instrument certification | Whether anatomical weighting adds value beyond geometry |
| Connectivity-aware deterministic planner | Whether graph features already explain the gain |
| Greedy accessible-patch removal | Whether sequential lookahead is actually needed |
| Beam search or cross-entropy trajectory search | Whether an inexpensive non-learning method is competitive |
| Receding-horizon/MPC planner | Whether RL adds value against strong dynamic planning |
| Patient-specific scratch RL | Whether fresh optimization on this anatomy is useful without population pretraining |
| Frozen population policy | Whether amortized inference already provides competitive plans |
| Population policy plus bounded case-specific adaptation | Whether per-case gradient updates add value over the same starting policy |
| Oracle-information controller in synthetic worlds | A bounded reference for the value of missing information, not a deployable competitor |

P01 and P02 are particularly important prior art for anatomy/connectional risk planning. P03 is a direct learned glioma-planning reference. P04 establishes relevant RL and deformation work in a different neurosurgical instrument task. Read their full available methods before asserting a first-of-its-kind contribution. [P01–P04]

A nearest-equivalent implementation is not an exact reproduction. Mark it as an inspired baseline when code, data or full methods are unavailable, and state the departures.

Compare quality as a function of online planning time and environment evaluations, not only one final reward. Include candidate-generation, adaptation and validation costs consistently. Report shared-policy pretraining separately and give explicit amortization assumptions when claiming speed benefits. Run comparisons under the same declared observability protocol; any privileged training critic or oracle advantage must be separately disclosed and controlled.
---

## 12. Desktop application architecture

### 12.1 Recommended default: a focused custom Slicer application

Prototype a custom Slicer application or extension first, using the custom-app template as a packaging path. This offers a practical native imaging foundation while leaving the interface original. The alternative is PySide6 plus VTK/PyVista with independent imaging services. Decide through a small vertical-slice experiment, not a generic framework preference. [E02, E03]

The comparison should load one real case, display linked MPR and 3-D, run a background operation, replay a cavity change, save/reopen the case and package the result on the user's actual operating system. Choose the option that reaches those goals cleanly.

Do not introduce a browser UI, multiple services, ROS, a database cluster and a remote model server unless a measured requirement justifies them. A local Python core and a native shell can be sufficient.

### 12.2 Proposed module boundaries

```text
resectionlab/
  core/          case schema, transforms, units, provenance
  data/          manifests, fetchers, cohort eligibility, splits
  imaging/       import, registration, segmentation, QC
  anatomy/       surfaces, tracts, graph, functional priors
  instruments/   geometry, parameters, interaction primitives
  simulation/    deterministic environment and world generators
  planning/      search, MPC, RL policy adapters, action proposals
  evaluation/    independent checks, metrics, uncertainty reports
  app/           desktop views, interactions, worker orchestration
  export/        case package, metrics, geometry, segmentation
  tests/         synthetic fixtures, regression and integration tests
```

These are conceptual boundaries, not a requirement to create every folder immediately. Start with a small working slice and extract modules as responsibilities become real.

### 12.3 Case and plan schema

A `Plan` should include the case-version hash, access-window version, anatomy-model version, tool catalog/version, policy/optimizer version, objective settings, uncertainty-world generator/version, action sequence or contingent policy, termination reason, and evaluator outputs.

An example output contract is:

```json
{
  "clinical_use_status": "research_only",
  "case_version": "sha256:...",
  "plan_type": "contingent_resection_policy",
  "motor": {
    "hazard_units": "declared_surrogate",
    "clinical_deficit_probability": null,
    "reason": "no_validated_clinical_outcome_model"
  },
  "language": {
    "clinical_deficit_probability": null,
    "evidence": ["patient_diffusion", "population_DES_prior"]
  },
  "model_events": [],
  "unmodeled_factors": ["unresolved_vascular_anatomy", "uncalibrated_tissue_response"],
  "termination": "surrogate_constraint_or_stop"
}
```

This is an illustrative schema, not an implemented API. All actual scalar results must include units, definitions and source versions.

Extend the contract with `planning_as_of`, `patient_context_version`, `optimizer_mode`, `shared_checkpoint_hash`, `adapted_checkpoint_hash`, `gradient_steps`, `environment_steps`, `optimization_budget`, `world_partition_manifest`, `selection_rule`, `evaluator_version`, `accessible_target_volume` and nullable `simulated_removed_target_volume`. Record source/availability timestamps for molecular and clinical fields. A route-only plan must not populate a removal result by copying its accessibility estimate. Use explicit JSON nulls for unavailable clinical predictions.

### 12.4 Worker model and persistence

The GUI must stay responsive during import, reconstruction and optimization. Use background workers with progress, cancellation and checkpointing. Heavy inference should not block the render loop or require the entire application to restart after one failed case.

Static patient arrays should be shared or cached rather than duplicated for every rollout. Use explicit cache invalidation based on input/model hashes. If the tumor mask or an access window changes, invalidate the affected anatomy/plans and display that change rather than replaying an obsolete result.

Save a portable case bundle with source references, derived data, transformations, plans, settings and a run log. Support a compact export without raw source images when redistribution is restricted.

### 12.5 Formats and interoperability

Prefer common scientific formats: NIfTI/NRRD for volumes, VTK/VTP or another documented format for geometry, JSON for plans/metrics, and a Slicer-compatible scene bundle when appropriate. DICOM-SEG export is a useful later feature if implemented against an established library and verified with independent viewers.

Do not claim Medivis compatibility merely because the app exports a mesh. Interoperability with a specific commercial platform requires documented support and testing.

---

## 13. Interface specification

### 13.1 Main workspace

Use a restrained native application with four coordinated areas. The left sidebar contains the case, sequences, source labels and data-quality state. The center contains linked MRI slices and a 3-D scene. The right panel holds selected plan metrics and alternatives. A bottom strip controls the operative simulation timeline and branching observations.

Orientation labels, scale, current coordinate frame and selected plan should remain visible. A clean interface that hides those essentials is worse than a slightly denser one that prevents interpretation errors.

### 13.2 Required interactions

The user can toggle target compartments, motor evidence, language evidence, uncertainty and tracts independently; inspect the original MRI under every overlay; correct a segmentation; define a hypothetical access window; select available instruments; request plans; and replay tissue removal.

Plan comparison should synchronize cameras, slices and time points. Selecting a point on a frontier plot highlights that plan in the anatomy view. Selecting a failure metric jumps to the offending tool pose or tissue region.

Use restrained categorical colors for distinct anatomy and a separate uncertainty encoding. Do not use one rainbow scale for both uncertainty and hazard. Offer outlines and opacity controls so a compelling volume render does not obscure the actual image evidence.

### 13.3 Plan cards

A card should show modeled target removal by compartment, motor and language surrogate metrics, robustness under the named world model, required tools, reach/visibility limitations, normal-tissue interaction and termination reason. It should show missing clinical probabilities as “not available,” not zero or a dash that looks like a negligible risk.

A candidate may be labeled “lower modeled motor hazard” rather than “safe.” Reserve “feasible” for a specified model and its constraints, not all of surgery.

### 13.4 Riskier and rejected plans

Preserve the user's desire to inspect the aggressive alternatives. Provide an “Explore tradeoffs” control and a separate “Rejected / dominated” view. Explain whether a plan is higher-hazard but feasible, dominated by another candidate, physically impossible, or unsupported because of missing inputs.

This separation is important. A high-benefit feasible plan with a worse surrogate cost is informative. A line through a structure the model cannot see should not appear as a confident risky-but-possible surgical option.

### 13.5 Explanations that are computed rather than narrated into existence

The first explanation system can be deterministic: “larger shaft collides at this pose,” “this patch changes the modeled language connection score,” “the selected plan is unstable under registration perturbation,” or “this observation changes the preferred next action.”

A later command-space explanation can show what preference change would make the policy choose a different action. The 2026 PCN explanation preprint is relevant, but a policy explanation is not a causal explanation of a patient's clinical outcome. [R08]

### 13.6 Presentation quality acceptance targets

Set measured targets for common interactions on the actual machine: smooth manipulation of a cached case, prompt feedback after selecting a plan, cancellable long jobs, and reproducible save/reopen. These are engineering targets, not performance already achieved. Record median and high-percentile timings for named case sizes rather than advertising “real time” without a workload definition.

### 13.7 Patient-specific refinement and surgeon inspection

Expose distinct actions: **Generate candidate routes**, **Refine for this case**, **Compare assumptions**, and **Replay modeled resection** when the sequential mode exists. The refinement view shows the current measured mode, actual updates, elapsed resource use, best selection-set candidate and a cancel/resume control. Label optimization/selection curves as such; keep final benchmark evaluation separate from the tuning loop.

Retain original search candidates next to learned candidates so improvement is inspectable rather than implied. Show the access window, complete instrument geometry, contact locations, remaining target compartments, evidence provenance and unresolved constraints for each alternative. A surgeon can supply or edit a route/access region and compare it under the same model without the system claiming authority over the surgical decision.

For a route-only result, show reachable target volume and geometric exposure. For a simulated resection, additionally show removal order, modeled residual volume and stopping decision. The roadmap aims toward clinician usefulness, but public-data and simulator evaluation alone do not establish readiness for clinical use. A later workflow/usability study with qualified reviewers would test usefulness without substituting for outcome validation.
---

## 14. Local compute and storage plan

### 14.1 Avoid unnecessary training

Reuse supplied annotations for policy-development environments and audited pretrained segmentation/tract models for the upload path. Precompute tract and distance features. Train a compact policy on cropped, multi-resolution representations, not whole 4-D MRI at every step.

A proposed initial observation grid might be 64 cubed or 96 cubed at a coarse physical resolution, with fine-resolution geometry checks around instrument paths. These are starting configurations to profile, not clinical accuracy claims. Coarse training must not allow a small critical structure to disappear from the final collision test.

Run a hardware probe for CPU, RAM, GPU/backend and disk. Scale worker count to memory. An Apple machine may need a different dependency path from a CUDA workstation; no specific hardware was assumed in this plan.

### 14.2 Budgeted experiment ladder

First run deterministic geometry tests and a tiny synthetic policy. Then profile five real cases end-to-end. Expand to a roughly 20–30-case pilot for debugging and model selection. Expand to a larger development cohort only after measuring memory, preprocessing time, rollout throughput and learning curves.

Do not launch a full cross-validation sweep before knowing whether the environment is computationally tractable. Start with three training seeds for a pilot; use more for the final core comparison where affordable. Log total environment steps, gradient updates, CPU/GPU time and preprocessing cost.

Do not confuse population training cost with patient-specific training cost. Benchmark a compact scratch policy on one case before declaring local case adaptation infeasible. Cache anatomy and distance fields, use frontier macro-actions, and profile collisions independently of the neural network. High-fidelity full-brain mechanics is not required for this pilot. If meaningful refinement exceeds a practical local budget, retain search or frozen-policy planning and report the observed limitation; do not claim interactive training without measured evidence.

### 14.3 Storage discipline

The raw data stack can easily become the dominant resource. UCSF, UPenn and ReMIND alone are substantial downloads, and transforms, tractograms and temporary files add further space. Use subject-level fetching where available, compressed/cache-aware derivatives, and explicit storage forecasts before bulk acquisition. [D01, D03, D13]

Do not download pathology, full normative collections or every derivative because they are free. “Public data only” is not “download everything.” Keep source data immutable and delete only reproducible temporary artifacts under a logged policy.

### 14.4 Google Cloud fallback

Use cloud compute only after identifying a concrete bottleneck that materially changes the achievable project. Potential uses are a bounded policy comparison or a one-off imaging batch, not an always-running GUI server.

The standard non-billable Google Cloud Free Trial does not permit adding GPUs. The user's credits may have a different status; verify before relying on them. Upgrading billing changes the charge exposure and must not be done automatically by the coding agent. [E04]

A cloud job should have explicit machine/disk choices, a maximum runtime, checkpoints, termination behavior and cleanup. Budget alerts are notifications, not a guaranteed spending cap. Keep all cloud data public and permitted for that use. Do not upload a new private patient case by default.

---

## 15. Validation: what the project must prove at each layer

### 15.1 Imaging and geometry

Evaluate segmentation with overlap and boundary metrics, stratified by compartment and difficult anatomy. Evaluate registration with held-out landmarks and visual overlays, not only image similarity. Record transform direction, physical units and inverse consistency where meaningful.

For tools, use analytically solvable fixtures: a shaft through a narrow aperture, a tip that fits while the shaft does not, crossing instrument envelopes, a blocked line of sight, and a target accessible only after removing an intervening exposed patch. Test swept volumes between poses, not only sampled endpoints.

For removal, assert monotonic remaining volume, no double-counted benefit, correct tissue accounting, connected access and deterministic replay. A policy exploiting any violation is an environment bug until shown otherwise.

### 15.2 Functional representation

Use available DES priors and upstream tract resources to evaluate consistency and localization, with held-out data where feasible. Do not split individual stimulation points at random when patient-level grouping is available; related points from the same operation are not independent observations. Do not validate a map by comparing it only to itself after a transform.

Where independent patient functional labels are absent, report what was checked: geometric alignment, reconstruction plausibility, perturbation sensitivity or agreement with a population prior. Do not rename those checks “validated motor risk.”

### 15.3 Planning outcomes

Primary outcomes should include the non-dominated removal/hazard frontier, removal achieved at predeclared surrogate budgets, modeled hard-constraint violations, and robustness to independent perturbations. Report tool configuration, online planning latency and resource consumption.

Use hypervolume only with fixed objective normalization and a preregistered reference point. A changing scale can manufacture an improvement. Show the underlying per-patient metrics as well as an aggregate score.

Evaluate all candidates with an independent checker, preferably at finer geometry resolution or under a different implementation from the training reward where feasible. Optimizing the simulator's reward and reporting only that reward is not sufficient.

### 15.4 External evaluation

Freeze the core method before evaluating the eligible UPenn diffusion subset. Use UTSW for the structural upload path, not as proof of tract-aware external planning. Use RESECT for independent registration/shift evaluation and BTC for a small separate multimodal exploration. A patient cannot be made fully observed by combining unrelated patients' modalities without marking the resulting world as synthetic.

The final paper should include a cohort flow diagram with original counts, exclusions, usable modalities, split membership and overlap handling. Report missingness and failure rates, not just successful cases.

### 15.5 Uncertainty and calibration

For simulated structural events, check predicted event frequency against realized events in held-out synthetic worlds. This tests internal calibration under that world family. Repeat under alternative world families to expose misspecification.

This does not calibrate clinical motor/language impairment. A future clinical risk head requires its own Brier scores, calibration curves, discrimination, confidence intervals and external testing on the actual outcome labels.

### 15.6 Statistics and leakage

The independent unit is the patient or declared patient cluster, not the number of rollouts. Use paired patient-level comparisons and patient-level bootstrap intervals. Treat different random seeds as repeat experiments, not new patients. Report effect sizes, uncertainty and failure distributions rather than only p-values.

Do not select “good-looking” test cases after seeing results. A demo can contain curated cases, but disclose their selection and include an honest failure case. Keep a separate final test set and a locked analysis script.

### 15.7 Required ablations

Compare removal planning with and without instrument geometry, graph features, uncertainty, explicit STOP, information actions and deformation. Include a sensitivity analysis over surrogate weights and observation assumptions. Test whether a policy's advantage survives changes in tool diameter, registration error, tract dropout and tumor location.

Ablations should test claims, not create a combinatorial experiment explosion. Choose a small primary set before the final evaluation, then label additional investigations exploratory.

Include a primary `POPULATION_FROZEN` versus `POPULATION_ADAPTED` comparison and a matched-budget `PATIENT_SCRATCH_RL` comparison. When molecular/context-conditioned objectives are studied, compare known-as-of metadata with the same method lacking that metadata and separate scenario-conditioned changes from validated benefit. An unchanged plan is an acceptable result; biology must not be forced to change geometry simply to make a demonstration more dramatic.

### 15.8 Nested evaluation for a planner that learns on each patient

**Outer split:** Freeze global algorithms, shared weights, hyperparameters, adaptation budget, stopping/checkpoint rule, uncertainty-model families and primary endpoints using development patients. Keep repeated visits, derivatives and relevant pretraining overlaps grouped. The test evaluates the entire frozen procedure, not just a neural checkpoint.

**Within a test case:** The frozen procedure may use the permitted preoperative images to create that patient's simulator and perform its predeclared scratch training or adaptation. Those training worlds are distinct from the case's checkpoint-selection worlds and final evaluation worlds. This is case-adapted evaluation, not zero-shot evaluation. No actual postoperative cavity, timed clinical outcome, future intraoperative image or held-out simulator realization may guide adaptation unless the experiment explicitly evaluates a later information state with an appropriate cutoff.

**Isolation:** Restart every test case from the specified initialization. Do not carry adapted parameters, hyperparameter changes or discoveries from one test patient into the next. Keep the shared model and uncertainty generator frozen. Genuinely online cross-patient learning is a different protocol and cannot be silently mixed into this benchmark.

**Independent checks:** Randomly withheld worlds from one generator test performance under that generator. Separately test reconstruction/model-family misspecification and high-resolution geometry. Neither establishes real postoperative safety; that requires other evidence. A planner capable of exploiting one simulator should be evaluated with assumptions it was not rewarded for exploiting.

**Analysis:** Use paired patient-level effects and uncertainty intervals, with prespecified repeated optimization seeds as within-patient replicates. Report target removal or accessibility at matched functional-surrogate budgets, invalid-plan rates, robust tails, retained-plan diversity and quality versus total online time. Count all cases and failures. A million episodes on one patient still supply one patient for between-patient inference.

**First acceptance experiment:** On one verified case, demonstrate actual learning, legal actions, independent candidate evaluation and a comparison with search; make no population claim. Repeat the frozen pilot procedure across a small development set before committing to the larger four-arm external benchmark. No rule requires population pretraining to precede the first patient-specific experiment.
---

## 16. Publication strategy and a defensible novelty claim

### 16.1 The most promising paper

A candidate title is:

> Patient-specific learning for instrument-aware glioma access and resection planning under anatomical uncertainty.

The primary question is whether privileged training experience improves
limited-information planning on independent patients, compared with strong
classical and learned-model planners that use the same permitted test-time
inputs. Bounded patient-specific optimization remains an important secondary
question and application mode. Any claim of transfer needs independent
patient-linked withheld anatomy; any claim of physical or clinical fidelity
needs its own measurements and observed outcomes. Instrument conditioning and
uncertainty are substantive parts of the task, not decorative input channels.

Close risk-map and learned glioma-planning work already exists, and patient-specific RL is not new across all medicine. Do not claim novelty merely from those phrases. A defensible contribution needs an actual method, evaluation or benchmark advance at their intersection. [P01–P04, R09]

Use two supporting questions: whether full-tool constraints change the accessible/resectable frontier, and whether uncertainty-aware optimization reduces adverse-tail surrogate costs at matched target benefit. Active information gathering and brain-shift updating remain strong later extensions, not prerequisites for the first route-planning or case-adaptation paper.

Select one primary claim and preregister the comparison before final testing. Do not attempt to invent segmentation, tractography, tissue mechanics, a new RL algorithm and a clinical outcome model simultaneously.

### 16.2 What would count as an actual result

A result could be a robust improvement at matched surrogate budgets, a clear speed/quality advantage after amortized training, a well-characterized failure of RL compared with search, or a validated benchmark that exposes instrument-feasibility errors in simpler methods.

A polished video alone is not a methods paper. A statistically significant change in an invented reward is not evidence of improved patient outcome. The evaluation should support exactly the level of claim made.

### 16.3 Venue positioning

IJCARS is a plausible match for a rigorous image-guided planning, instrumentation, simulation and interface contribution; these topics are explicitly in its scope. [E05] A relevant medical-image computing or computer-assisted intervention workshop may fit an earlier benchmark release, while a mature algorithmic contribution could be considered for broader medical-imaging venues. Verify current venue calls when the experiments exist.

Do not optimize the project around a prestige label before knowing the result. A credible reproducible paper and a strong repository are more useful for the user's goal than a manuscript making unsupported clinical claims.

### 16.4 Required paper artifacts

Prepare a registered or timestamped protocol; frozen dataset/split manifests; a closest-work comparison; formal environment definitions; calibration and sensitivity analyses; compute accounting; source/version/license records; negative results; and a reproducibility package.

The paper should explicitly distinguish validity of anatomy reconstruction, validity of instrument geometry, validity of the tissue simulator, validity of surrogate outcomes and clinical effectiveness. Evidence for one does not automatically establish the next.

---

## 17. Milestones with concrete completion criteria

### Milestone 0: Evidence and execution contract

**Goal:** Prevent the project from becoming dependent on nonexistent labels or unbounded compute.

**Data:** Metadata and sample files from UCSF plus the small DES releases.

**Deliverables:** Data dictionary; source/license manifest; initial cohort filters; hardware report; storage forecast; scientific claims table; architecture decision record.

**Completion:** A five-case acquisition plan is executable, every field has a provenance category, and absent clinical probability labels are represented explicitly. No cloud spending or bulk downloads are required to finish this milestone.

### Milestone 1: A real case in a real desktop workspace

**Goal:** Produce the earliest useful vertical slice.

**Data:** One public UCSF case and a supplied mask.

**Deliverables:** Linked MPR/3-D, tumor compartments, source inspection, save/reopen and background-job status in an original focused layout.

**Completion:** Coordinate landmarks match between slices and 3-D; units/orientation are visible; the case reopens reproducibly; the UI stays responsive. No RL is necessary yet, but the schema already supports plans.

### Milestone 2: Patient-derived anatomy with QC

**Goal:** Make motor and language representations inspectable rather than decorative.

**Data:** Five UCSF cases, TractSeg resources and DES priors.

**Deliverables:** Diffusion reconstruction, candidate bundles, cortical/ventricular surfaces, prior registration, correction workflow and failure reports.

**Completion:** Each case passes or fails named checks; failed diffusion is not silently replaced; uncertainty and population priors are visually distinguishable from observed imaging.

### Milestone 3: Risk representations and model-world sampling

**Goal:** Compute interpretable spatial and connectional surrogate costs.

**Data:** The QC-passing pilot cases and functional priors.

**Deliverables:** Motor/language fields, graph representation, uncertainty worlds, plan-specific marginal-cost inspection and metric definitions.

**Completion:** A synthetic connection-cut test behaves correctly; repeated intersection of one severed connection is not double-counted; every probability field has a named simulated event; clinical deficit probability remains unavailable.

### Milestone 4: Route-planning product milestone

**Goal:** Produce an inspectable, instrument-aware glioma route-comparison workspace before complete surgical simulation exists.

**Data:** QC-passing pilot cases, source annotations and generic documented instruments.

**Deliverables:** Full-tool swept-envelope checks, specified access windows, reachable-target analysis, distinct candidate corridors, motor/language evidence, uncertainty halos, missing-data warnings and rejected-plan explanations.

**Completion:** Analytic fixtures pass; a real case shows how tool geometry changes access; alternatives can be compared and saved. Accessible volume is not labeled removed volume, modeled feasibility is not called clinical safety, and absent vascular/functional inputs remain explicitly unassessed. The product remains useful for research inspection without claiming whole-operation prediction.

### Milestone 5: A sequential resection environment

**Goal:** Make the next decision depend on previous actions.

**Data:** Synthetic fixtures and one verified real case first. Expand toward a roughly 20–30-case development pilot only after profiling and a reproducible single-case loop; that cohort is not a prerequisite for first training.

**Deliverables:** Cavity evolution, accessible frontier, removal bookkeeping, tool changes, STOP, deterministic replay and greedy/beam baselines.

**Completion:** No teleporting removal, free corridor creation, impossible shaft motion, duplicated rewards or nondeterministic replay. A small toy problem has a known optimum or exhaustive reference.

### Milestone 6: Genuine patient-specific learning and fair comparison

**Goal:** Train inside a patient's environment and establish whether that optimization improves useful modeled decisions.

**Data:** Start with one development patient and independently partitioned simulated worlds. Add other development patients, optional population training and the frozen nested-evaluation protocol as capacity permits.

**Deliverables:** Actual scratch-policy updates, checkpoints, resource/selection curves, independent event/geometry evaluation, initial-search comparisons and replay. Add frozen-population and adapted-population arms for the complete benchmark once their prerequisite checkpoint exists.

**Completion:** Legal actions and STOP remain enforced; no optimizer can weaken the world or reward; selection and evaluation remain separate; successes and failures are retained. A single-case demonstration is labeled single-case. The final method is judged at the patient level across held-out cases using the permitted adaptation protocol, not by insisting that every test case be zero-shot.

### Milestone 7: Uncertainty, observations and dynamic replanning

**Goal:** Extend the route/adaptation core with active information and dynamic anatomy when the earlier comparison is trustworthy.

**Data:** ReMIND for intraoperative-image/shift work, RESECT for independent registration checks, simulated observation worlds.

**Deliverables:** Belief updates, noisy information actions, limited deformation model, replan timeline and robustness experiments.

**Completion:** Observation value varies sensibly with cost/reliability; held-out perturbation families are tested; real imaging snapshots are not mislabeled as action-response supervision.

### Milestone 8: Product-level polish and broader upload behavior

**Goal:** Turn the technical core into a credible software experience.

**Data:** Curated public cases plus UTSW structural cases and deliberately malformed input fixtures.

**Deliverables:** Case loading, missing-modality behavior, clean plan cards, linked comparisons, geometry-aware tool switching, exports, packaging and a documented demonstration.

**Completion:** Invalid inputs fail helpfully; edits invalidate old plans; the app is cancellable and recoverable; a fresh user can run the synthetic demo without data access or cloud services.

### Milestone 9: Locked external evaluation

**Goal:** Produce defensible evidence beyond the development site.

**Data:** Eligible UPenn diffusion subset, UTSW structural holdout, RESECT registration holdout and optional BTC exploration.

**Deliverables:** Cohort flow, leakage audit, primary statistical analysis, ablations, resource report, failure analysis and independent evaluator results.

**Completion:** No post-hoc changes to the global method or predeclared adaptation rule based on final test feedback. Per-case updates use only permitted inputs and optimization worlds under Section 15.8. Every claimed improvement has a comparator, unit of analysis and interval; clinical outcomes are not fabricated.

### Milestone 10: Public release and manuscript package

**Goal:** Make the work easy to inspect, reproduce and discuss with Medivis.

**Deliverables:** Clean repository, release build, concise authentic demo, source citations, method documentation, benchmark scripts, model/environment cards, limitations and a manuscript draft aligned with actual results.

**Completion:** Another machine can reproduce a synthetic case and a documented public case; all displayed results come from saved runs; licensing is clear; the README explains exactly what is novel and what remains unvalidated.

Milestones 1–4 yield the first route-planning product. Milestones 5–6 establish the patient-specific RL investigation; Milestone 7 is a staged extension rather than a blocker. Milestones 9–10 establish the strongest publication case. The interface develops throughout, not after all research is finished.

---

## 18. Failure register and contingency plans

| Failure mode | Detection | Contingency |
|---|---|---|
| Raw diffusion or gradients unusable | Per-case contract and orientation tests | Exclude from tract-aware cohort; retain separately for structural display |
| Tumor distorts automatic tracts | Endpoint, continuity and manual QC; disagreement maps | Correction workflow, conservative unknown region, no automatic low-risk claim |
| Atlas fails near lesion | Registration inspection and deformation checks | Reduce prior confidence; retain native patient evidence; mark unsupported functional localization |
| Most datasets lack neurological labels | Field-level manifest audit | Complete surrogate/simulation project; do not fabricate a clinical head |
| Tool action teleports through tissue | Analytic fixtures and independent swept-volume checker | Fix environment before further training |
| RL learns reward exploits | Replay highest-return episodes and adversarial fixtures | Correct reward/physics and invalidate affected results |
| RL cannot beat MPC | Matched-compute held-out comparison | Use the better planner in the app; report the result; emphasize benchmark/system contribution |
| Deformation is too expensive | Profile worker throughput and memory | Reduced-order training model, small high-fidelity evaluation subset |
| Shift model is not identifiable from snapshots | Lack of action/force supervision | Restrict claim to observed deformation/registration robustness, not tool-response prediction |
| No complete vascular imaging | Modality/region capability gate | Explicit unknowns; restrict corridor claims and difficult cases |
| Test leakage through BraTS or duplicates | ID mapping, source hashes and model provenance | Regroup/re-split; separate annotation-assisted and end-to-end tracks |
| App resembles a research toolbox dump | Task-based usability checks | Hide irrelevant controls; prioritize one coherent case workflow |
| Local GPU absent or insufficient | Hardware probe and five-case profiling | CPU/pretrained/cached path first; bounded authorized cloud fallback only when useful |
| Public download or license unclear | Source/terms audit | Keep resource optional; use a verified alternative; never stall all milestones |
| Case-specific optimizer shrinks hazards or changes rewards | Hash frozen objective, world generator and tool models at every checkpoint | Reject incomparable runs; treat model revisions as new experiments |
| Same-patient simulation score is mistaken for clinical validation | Separate optimization, selection and evaluation reports | Report model-conditioned evidence and retain null clinical probabilities |
| Molecular result was learned only after surgery | Audit measurement and availability timestamps | Exclude from the preoperative primary analysis; label oracle/scenario tracks |
| Reachable volume is reported as tissue removed | Separate route and sequential-plan schemas | Require legal removal replay before reporting modeled resection |

Do not respond to a blocked component by building unrelated infrastructure. Continue on an independent milestone, keep a clear unresolved item, and return when the missing evidence or resource genuinely becomes necessary.

---

## 19. Repository, demo and hiring-oriented presentation

The README should open with a factual one-sentence purpose and an authentic short demonstration. Show the actual application loading public imaging, generating plans, changing tools, exposing a failure and replaying a decision. Avoid a montage that could be mistaken for an unimplemented concept video.

Include a small synthetic demo with no external downloads, a public-data setup guide, tested platform/dependency information, architecture overview, reproducible experiments, benchmark results, explicit limitations and citations. Add `CITATION.cff`, an appropriate code license, a dataset/weights license inventory and a security/privacy note. Data and code licenses are separate.

Useful screenshots or clips would demonstrate: a tip-versus-shaft feasibility conflict; a motor/language tradeoff; sensitivity to uncertain registration; a changing cavity; and a decision to acquire information or stop. Every number should be traceable to a saved case/run.

The Medivis-facing explanation should emphasize that the project complements a case-centered imaging/planning workflow. It should not imply integration, endorsement or device clearance. A grounded summary is stronger than “I built an AI that performs brain surgery.”

The research and hiring goals can reinforce one another: the same provenance, spatial correctness and failure visibility that improve the paper also make the application more credible to an engineering team.

---

## 20. Implementing-agent decision authority

The implementing agent may change the GUI framework, image-processing stack, representation, optimizer, RL algorithm, model size, curriculum details and packaging strategy. It should choose through small measured experiments and record the reason in an architecture decision log.

The agent may not change the user goals without approval, replace the core RL investigation with an unrelated segmenter, hide negative results, relabel surrogates as clinical probabilities, fabricate missing data, silently spend money, or claim patient safety from simulator success.

For each major choice, record the alternatives, local resource cost, empirical result, chosen option and reversal condition. Prefer the simplest design that passes the acceptance criteria. The plan is an execution scaffold, not a demand to use every cited technique.

The first execution should complete Milestone 0 and the beginning of Milestone 1, not scaffold every module or train a large model. The earliest artifact should be a real, correctly aligned public case in an inspectable application.

The revised core requirements include a route-first deliverable and a genuine single-patient optimization experiment. The agent may choose the best measured default, but must not quietly delete the scratch-RL comparison or make population pretraining a mandatory entry gate. Preserve working repository code and reconcile it with this revision before refactoring. The original three Markdown files are the necessary specification; create missing lightweight manifests/protocols rather than blocking on ancillary files mentioned by an older package.
---

## 21. Research coverage and how to extend it responsibly

The annotated bibliography prioritizes original data releases, direct clinical/mapping studies, close planning/RL prior art, simulation frameworks and official engineering documentation. It deliberately includes negative access findings: attractive papers with nonpublic data are not usable public training sources.

Before manuscript submission, repeat the closest-work search using exact titles and citation chaining. Search concepts separately: glioma resection planning; neurosurgical risk maps; instrument configuration-space planning; active functional mapping; uncertainty-aware resection; brain-shift simulation; clinical deficit prediction; and constrained/preference-conditioned RL. Track search date, source, screening reason, data/code availability and whether full methods were actually read.

Treat titles and abstracts as leads, not evidence of reproduction. A paper whose dataset is “available on reasonable request” is not a required input under the present public-only constraint. An apparently new method may already exist under “access planning,” “eloquence cost,” “connectome-guided surgery” or “task and motion planning.”

The deliverable below is therefore a substantial starting research library with use-specific annotations, not a guarantee of complete literature coverage or an established novelty claim.

---

## Reference directory

For per-source limitations, access status and proposed uses, see `ANNOTATED_REFERENCES.md`.

- **[M01]** Medivis (2026). *Medivis Studio*. https://www.medivis.com/studio
- **[M02]** Medivis (2026). *Frontier: Agents / Maia*. https://www.medivis.com/frontier/agents
- **[M03]** Medivis (2026). *Clinical research index*. https://www.medivis.com/clinical-research
- **[D01]** The Cancer Imaging Archive (2023). *UCSF-PDGM collection, version 4*. https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=119705830
- **[D02]** Calabrese et al. (2022). *The University of California San Francisco Preoperative Diffuse Glioma MRI Dataset*. https://pmc.ncbi.nlm.nih.gov/articles/PMC9748624/
- **[D03]** The Cancer Imaging Archive (2022). *UPENN-GBM collection*. https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=70225642
- **[D04]** Bakas et al. (2022). *The University of Pennsylvania glioblastoma (UPenn-GBM) cohort: advanced MRI, clinical, genomics, & radiomics*. https://www.nature.com/articles/s41597-022-01560-7
- **[D05]** Fang-Cheng Yeh laboratory (2026). *UPenn-GBM diffusion MRI reconstruction / ready-to-track derivatives*. https://brain.labsolver.org/upenn_gbm.html
- **[D06]** The Cancer Imaging Archive (2026). *UTSW-Glioma collection*. https://www.cancerimagingarchive.net/collection/utsw-glioma/
- **[D07]** Reddy et al. (2026). *The University of Texas Southwestern Glioma Dataset*. https://www.nature.com/articles/s41597-026-07274-4
- **[D08]** Aerts and colleagues / OpenNeuro (2026). *BTC preoperative dataset, ds001226*. https://github.com/OpenNeuroDatasets/ds001226
- **[D09]** Aerts and colleagues / OpenNeuro (2026). *BTC postoperative dataset, ds002080*. https://github.com/OpenNeuroDatasets/ds002080
- **[D10]** Aerts et al. (2018). *Modeling Brain Dynamics in Brain Tumor Patients Using the Virtual Brain*. https://doi.org/10.1523/ENEURO.0083-18.2018
- **[D11]** Aerts et al. (2020). *Modeling brain dynamics after tumor resection using The Virtual Brain*. https://doi.org/10.1016/j.neuroimage.2020.116738
- **[D12]** Juvekar et al. (2024). *ReMIND: The Brain Resection Multimodal Imaging Database*. https://www.nature.com/articles/s41597-024-03295-z
- **[D13]** The Cancer Imaging Archive (2023). *ReMIND collection*. https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=157288106
- **[D14]** HealthX Laboratory (2026). *RESECT and EASY-RESECT resources*. https://www.healthx-lab.ca/databases.html
- **[D15]** Xiao et al. (2017). *REtroSpective Evaluation of Cerebral Tumors (RESECT): a clinical database of pre-operative MRI and intra-operative ultrasound in low-grade glioma surgeries*. https://archive.sigma2.no/dataset/5D6BFC33-F58D-4F56-88E8-C40AF269D6F2
- **[D16]** Montreal Neurological Institute (2026). *BITE: Brain Images of Tumors for Evaluation*. https://nist.mni.mcgill.ca/bite-brain-images-of-tumors-for-evaluation-database/
- **[D17]** Mercier et al. (2012). *On-line database of clinical MR and ultrasound images of brain tumors*. https://nist.mni.mcgill.ca/bite-brain-images-of-tumors-for-evaluation-database/
- **[D18]** Poulin et al. (2022). *TractoInferno: A large-scale, open-source, multi-site database for machine learning dMRI tractography*. https://www.nature.com/articles/s41597-022-01833-1
- **[D19]** Wasserthal and colleagues (2018). *TractSeg training/reference bundle data*. https://doi.org/10.5281/zenodo.1088277
- **[D20]** Mahmoud et al. (2025). *MU-Glioma Post: A comprehensive dataset of automated MR multi-sequence segmentation and clinical features*. https://www.nature.com/articles/s41597-025-06011-7
- **[D21]** Wamelink et al. (2026). *The Amsterdam IMAging and Clinical GliOma Dataset (IMAGO)*. https://www.nature.com/articles/s41597-026-07424-8
- **[F01]** Coletta et al. (2024). *Integrating direct electrical brain stimulation with the human connectome*. https://academic.oup.com/brain/article/147/3/1100/7458468
- **[F02]** Coletta and colleagues (2023). *Public data accompanying Integrating direct electrical brain stimulation with the human connectome*. https://zenodo.org/records/10439149
- **[F03]** Coletta et al. (2025). *Integrating direct electrical stimulation with brain connectivity predicts lesion-induced language impairment and recovery*. https://www.nature.com/articles/s43856-025-01121-0
- **[F04]** Coletta and colleagues (2025). *Public language DES/connectivity maps and figure resources*. https://zenodo.org/records/16418628
- **[F05]** Denker et al. (2024). *Navigated Transcranial Magnetic Stimulation and Diffusion Tensor Imaging Tractography in Insular Glioma Surgery*. https://pubmed.ncbi.nlm.nih.gov/39508607/
- **[F06]** Sanai, Mirzadeh and Berger (2008). *Functional Outcome after Language Mapping for Glioma Resection*. https://www.nejm.org/doi/full/10.1056/NEJMoa067819
- **[F07]** Honeyman et al. (2026). *Speech mapping in awake high-grade glioma resection: subcortical tract proximity as a predictor of language outcomes*. https://pubmed.ncbi.nlm.nih.gov/41825076/
- **[F08]** Nandakumar et al. (2021). *Automated eloquent cortex localization in brain tumor patients using multi-task graph neural networks*. https://www.sciencedirect.com/science/article/pii/S1361841521002486
- **[F09]** Maier-Hein et al. (2017). *The challenge of mapping the human connectome based on diffusion tractography*. https://www.nature.com/articles/s41467-017-01285-x
- **[F10]** Wasserthal et al. (2018). *TractSeg: Fast and accurate white matter tract segmentation*. https://arxiv.org/abs/1805.07103
- **[F11]** MIC-DKFZ (2026). *TractSeg implementation and pretrained-model workflow*. https://github.com/MIC-DKFZ/TractSeg/
- **[F12]** Peretzke et al. (2023). *atTRACTive: Semi-automatic white matter tract segmentation using active learning*. https://arxiv.org/abs/2305.18905
- **[P01]** Bakhshmand, Eagleson and de Ribaupierre (2017). *Multimodal connectivity based eloquence score computation and visualisation for computer-aided neurosurgical path planning*. https://pubmed.ncbi.nlm.nih.gov/29184656/
- **[P02]** Kunz et al. (2021). *Multimodal Risk-Based Path Planning for Neurosurgical Interventions*. https://cris.fau.de/publications/293044349/
- **[P03]** Shan et al. (2023). *Towards Learning-based Surgical Planning of Glioma Resection via a Contrastively Constrained Siamese Neural Network*. https://ieeexplore.ieee.org/document/10385556/
- **[P04]** Segato et al. (2022). *Inverse Reinforcement Learning Intra-Operative Path Planning for Steerable Needle*. https://pubmed.ncbi.nlm.nih.gov/34882540/
- **[P05]** Pehlivanoğlu (2024). *A new surgical path planning framework for neurosurgery*. https://pubmed.ncbi.nlm.nih.gov/37773772/
- **[P06]** Kavraki Laboratory (2026). *The Open Motion Planning Library*. https://ompl.kavrakilab.org/
- **[R01]** Schulman et al. (2017). *Proximal Policy Optimization Algorithms*. https://arxiv.org/abs/1707.06347
- **[R02]** Haarnoja et al. (2018). *Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor*. https://arxiv.org/abs/1801.01290
- **[R03]** Achiam et al. (2017). *Constrained Policy Optimization*. https://proceedings.mlr.press/v70/achiam17a.html
- **[R04]** Chow et al. (2015). *Risk-Sensitive and Robust Decision-Making: a CVaR Optimization Approach*. https://arxiv.org/abs/1506.02188
- **[R05]** Chua et al. (2018). *Deep Reinforcement Learning in a Handful of Trials using Probabilistic Dynamics Models*. https://arxiv.org/abs/1805.12114
- **[R06]** Tobin et al. (2017). *Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World*. https://arxiv.org/abs/1703.06907
- **[R07]** Reymond, Bargiacchi and Nowé (2022). *Pareto Conditioned Networks*. https://arxiv.org/abs/2204.05036
- **[R08]** Chulev and Baier (2026). *Command-Space Counterfactual Explanations for Pareto-Conditioned Reinforcement Learning*. https://arxiv.org/abs/2608.14963
- **[S01]** Scheikl et al. (2023). *LapGym: An Open Source Framework for Reinforcement Learning in Robot-Assisted Laparoscopic Surgery*. https://arxiv.org/abs/2302.09606
- **[S02]** Scheikl and colleagues (2026). *LapGym source repository*. https://github.com/ScheiklP/lap_gym
- **[S03]** Schegg et al. / SofaDefrost (2023). *SofaGym: An Open Platform for Reinforcement Learning Based on Soft Robot Simulations*. https://github.com/SofaDefrost/SofaGym
- **[S04]** Li et al. (2025). *Fully automated image updating for brain shift compensation after dural opening*. https://pubmed.ncbi.nlm.nih.gov/40911919/
- **[S05]** Rivaz et al. (2015). *Automatic Deformable MR-Ultrasound Registration for Image-Guided Neurosurgery*. https://nist.mni.mcgill.ca/bite-brain-images-of-tumors-for-evaluation-database/
- **[S06]** Pérez-García et al. (2020). *Simulation of Brain Resection for Cavity Segmentation Using Self-Supervised and Semi-Supervised Learning*. https://arxiv.org/abs/2006.15693
- **[I01]** NICO Corporation (2026). *NICO integrated access and resection system*. https://niconeuro.com/our-integrated-system/
- **[I02]** Integra LifeSciences (2026). *CUSA Clarity product family*. https://products.integralife.com/cusa-clarity/category/cusa-tissue-ablation-cusa-clarity
- **[I03]** Kalavakonda et al. (2019). *Autonomous Neurosurgical Instrument Segmentation Using End-To-End Learning*. https://openaccess.thecvf.com/content_CVPRW_2019/html/WiCV/Kalavakonda_Autonomous_Neurosurgical_Instrument_Segmentation_Using_End-To-End_Learning_CVPRW_2019_paper.html
- **[I04]** Luo et al. (2023). *Fast instruments and tissues segmentation of micro-neurosurgical scene using high correlative non-local network*. https://www.sciencedirect.com/science/article/pii/S0010482522012392
- **[I05]** Eyüpoglu et al. (2012). *Improving the Extent of Malignant Glioma Resection by Dual Intraoperative Visualization Approach*. https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0044885
- **[E01]** MIC-DKFZ (2026). *nnU-Net source and documentation*. https://github.com/MIC-DKFZ/nnUNet
- **[E02]** Kitware Medical (2026). *SlicerCustomAppTemplate*. https://github.com/KitwareMedical/SlicerCustomAppTemplate
- **[E03]** 3D Slicer contributors (2026). *3D Slicer extension development documentation*. https://slicer.readthedocs.io/en/latest/developer_guide/extensions.html
- **[E04]** Google Cloud (2026). *Google Cloud free features and trial restrictions*. https://docs.cloud.google.com/free/docs/free-cloud-features
- **[E05]** Springer Nature (2026). *International Journal of Computer Assisted Radiology and Surgery: aims and scope*. https://link.springer.com/journal/11548/aims-and-scope

- **[R09]** Madondo et al. (2025). *Patient-Specific Deep Reinforcement Learning for Automatic Replanning in Head-and-Neck Cancer Proton Therapy*. https://proceedings.mlr.press/v298/madondo25a.html
- **[C01]** Goldbrunner et al. (online 2025; issue 2026). *EANS-EANO guidelines on the extent of resection in gliomas*. https://academic.oup.com/neuro-oncology/article/28/1/38/8256732
- **[C02]** Sahm et al. (2023). *Molecular diagnostic tools for the WHO 2021 classification ...; an EANO guideline*. https://pubmed.ncbi.nlm.nih.gov/37279174/
- **[F13]** Campbell et al. (2014). *Beyond Crossing Fibers: Bootstrap Probabilistic Tractography Using Complex Subvoxel Fiber Geometries*. https://pmc.ncbi.nlm.nih.gov/articles/PMC4211389/
- **[F14]** Manzano-Patrón et al. (2025; correction 2026). *Uncertainty mapping and probabilistic tractography using Simulation-based Inference in diffusion MRI*. https://pubmed.ncbi.nlm.nih.gov/40311303/ ; correction: https://pubmed.ncbi.nlm.nih.gov/41986194/
- **[E06]** DIPY contributors. *Probabilistic tractography example*. https://docs.dipy.org/stable/examples_built/fiber_tracking/tracking_probabilistic.html
