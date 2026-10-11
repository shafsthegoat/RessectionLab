> Active update, October 10: follow the [hybrid world-model steering](docs/HYBRID_WORLD_MODEL_STEERING.md) supplied by the user after main `4d1d0ff`. It changes the next implementation sequence, not the historical results or patient roles. Prioritize the matched information contract, persistent operative episode, independent real-observation prediction, then a bounded hybrid comparison. Preserve separately labeled generated training permission. The source review and proposed mechanisms in the attachment are not new experimental results.
>
> Active update, October 9: read the [integration-first multimodal steering](docs/INTEGRATION_FIRST_MULTIMODAL_STEERING.md) first. Deliver shared state/actions, typed desktop integration and executable replay together. Keep acquired evidence, estimated planning state and private evaluation separate. Coherent integration milestones supersede micro-commits and routine review campaigns; scientific claims and patient admission still require appropriate independent checks.
>
> October 8: read [the October 6 supergoal](docs/SUPERGOAL_REAL_OBSERVATIONS.md)
> and [October 8 human steering](docs/REAL_OBSERVATION_EXECUTION_LEDGER.md#active-human-steering-october-8) first.
> The later user instruction permits separately labeled synthetic/simulator RL
> development and training, superseding the older real-only restriction. Preserve
> splits and history; evaluate transfer on held-out real patients and physical
> measurements. Continuous acquisition and separate QC must proceed independently.
> A scoped [generated opening-task learner](docs/native-opening-learning.md) is
> now enabled; legacy patient/model guards remain. Read current results before
> launching additional training or claiming transfer.

# Implementing-agent handoff

Original brief October 2, 2026; **patient-specific planning revision October 4, 2026**. Read the revised `MASTER_PLAN.md` and `ANNOTATED_REFERENCES.md` before executing. This is an implementation brief, not completed experiments. Use the exact repository name `RessectionLab`. The dated revision supersedes conflicting older project prompts, not repository security or permission rules.

## Mission

Build a local-first desktop research application that helps a surgeon inspect candidate glioma routes, then supports training or refining an RL policy inside that particular patient's imaging-derived simulation. Route comparison is the first product milestone; connected sequential resection is the next layer, not a reason to postpone a useful route explorer. Include a genuine reinforcement-learning investigation, strong classical/search competitors, motor and language hazard representations, explicit uncertainty, and visible rejected alternatives. The intended outcomes are a technically excellent public portfolio, a credible Medivis-facing demonstration, and an appropriately bounded research publication.

The user has no fixed deadline, prefers public data only, and has up to $300 in Google Cloud credits as a fallback. Local hardware is not specified. Inspect the available machine rather than assuming CUDA, a particular GPU, memory capacity or storage. Do not create cloud resources, upgrade billing or incur charges without explicit permission. No AR, robotic control, live patient use or undocumented commercial-platform integration belongs in the initial scope.

## Authority and invariants

You choose the final stack, GUI framework, representations, algorithms, model sizes and implementation order through measured prototypes. The recommendations in the master plan are defaults, not mandatory dependencies. Record material departures and the evidence for them. Do not build every suggested extension.

The following are fixed requirements:

1. Anatomical coordinates, gradients, patient identity, units and provenance must be correct and testable.
2. A normalized anatomical map or simulated event frequency is not a postoperative-deficit probability. Keep clinical probability fields null until a separately validated clinical model exists.
3. Public source images are not surgeon action demonstrations. Simulator-generated actions, observations and outcomes remain labeled simulated.
4. Instrument feasibility concerns the complete tool and its swept volume, not only the tip or a centerline.
5. Removal requires connected access and a legal interaction. Do not teleport into solid tissue, erase intervening normal brain without cost, or let inaccessible internal tumor voxels disappear.
6. All methods receive the same observations, action primitives, hard constraints and evaluation contract. Hidden simulation truth must not leak into the deployed policy.
7. A genuine single-patient RL experiment and serious comparison are required, but the product may use search/MPC if it wins. Population pretraining is not a prerequisite for the one-case experiment. Do not hide negative results.
8. The native app must remain responsive, support cancellation and saving, and make failures and unsupported inputs visible.
9. Adopt Medivis's case-centered product principles without copying branding or claiming affiliation, device clearance or compatibility.
10. Preserve source licenses, dependency/model licenses and user work. Do not put large medical datasets or restricted assets in the repository.
11. Freeze reward definitions, tool parameters and the uncertainty generator during optimization. A policy cannot earn a better score by making its world easier.
12. Respect the preoperative information cutoff. Molecular/pathology results available only after surgery are not valid preoperative inputs.
13. Keep geometric accessibility separate from simulated tissue removal, and all modeled outcomes separate from clinical outcomes.

## Begin with evidence and one real case

First, inventory compute, storage, current repository state and any installed medical-imaging libraries. Create a short architecture/evidence log. Reuse `DATASET_MANIFEST.json` when present; otherwise create a minimal manifest from the three Markdown documents. Confirm the authoritative source record and acquire one UCSF-PDGM case with gradients, source masks and necessary metadata before expanding to a five-case pilot. Include a documented failure case where possible. Inspect actual file contents before assuming completeness.

Acquire the small DES map releases only after checking their terms and template space. These are functional priors, not patient-level clinical outcome labels. Use motor first; add language once the motor pipeline is inspectable. Resolve repeated patients and BraTS overlap before freezing splits or importing a pretrained segmenter.

Build the earliest vertical slice around one genuine public case: linked MRI slices and 3-D anatomy, inspectable transforms, editable source annotations, one hypothetical access window, two instrument geometries, a few independently geometry-checked candidate routes, and save/reopen. The first slice may use supplied annotations and deterministic planning. Mark it annotation-assisted; do not present it as autonomous segmentation or a trained RL result.

The first meaningful success is not a complicated repository scaffold. It is a reproducible case where changing a tool's geometry changes what the planner can reach, and the visualization shows why.

## Decisions to settle empirically

**Desktop foundation:** Compare a custom Slicer application/extension with a PySide6 plus VTK-style implementation using the same small case. Test MPR/3-D alignment, asynchronous computation, cavity replay, save/reopen and packaging on the real machine. Prefer the option that makes the imaging workflow reliable with less custom infrastructure.

**Patient representation:** Keep authoritative geometry in a declared physical frame. A cropped, multiresolution grid can serve the policy, but high-resolution geometry must independently check final actions. Choose sparse/dense fields, meshes and graph features by profiling rather than assuming one representation serves every task.

**Segmentation and tract reconstruction:** Audit model pretraining, gradients and tumor-domain failures. A framework installation is not a suitable pretrained model. Maintain source-annotation and inference-only benchmark tracks. Implement correction/QC before pretending every reconstruction is trustworthy.

**Simulation:** Start with repeatable geometry, exposed-frontier removal and explicit tool envelopes. Add coherent uncertainty before expensive tissue mechanics. ReMIND/RESECT can constrain image updating and shift scenarios, but do not directly supervise arbitrary tool-action forces or tissue destruction.

**RL:** Begin with masked macro-actions and a compact trainable policy on one case. Compare actual scratch-policy updates with per-case search under a fixed compute budget. Then compare frozen population inference and a clone of that population policy refined for the new case. Add recurrence, preference conditioning or cost critics only when required. Keep learning curves separate from independent evaluation. Reserve continuous low-level control for a demonstrated need.

## Required sequence of deliverables

Complete the master-plan milestones through vertical slices. Keep a working application while developing headless simulation and training. Every milestone should produce an artifact that can be inspected without trusting a narrative summary.

The core progression is: source manifest and eligibility checks; aligned patient representation; motor/language evidence and lightweight uncertainty halos; a working instrument-aware route comparator; connected sequential simulation and strong baselines; actual one-patient training and independent validation; optional population pretraining and bounded per-case adaptation; broader frozen evaluation; reproducible release. Active observations and deformation extend this core rather than blocking it. Keep the desktop workflow working throughout.

A milestone is not complete because tests were written. Run them, retain the result, and include a small real-case inspection in addition to synthetic fixtures. Separate known limitations from completed behavior.

## Experiment discipline

Use `EXPERIMENT_PROTOCOL.md` if present; otherwise create a concise protocol from Sections 9.8 and 15.8 of the revised master plan. Freeze patient-level splits, method settings and the within-patient adaptation rule before final comparisons. Split repeated visits and derivatives together. Hold UPenn out of global method development initially; later evaluation may run the already frozen patient-specific adaptation procedure on its permitted preoperative inputs; UTSW tests structural generalization only unless its input contract changes. Use RESECT independently of ReMIND for the corresponding registration task.

Log code revision, data/model hashes, hyperparameters, hardware, random seeds, training and inference cost, failures and all attempted variants. Report statistical uncertainty across patients, not across thousands of correlated rollouts from one patient. Use a separately implemented or independently configured geometry checker for final validation.

Before large training, adversarially test rewards and action legality. Include immediate STOP, repeated removal, tool cycling, through-tissue travel, missing gradient input, left/right mistakes, stale plans after edits, and uncertain anatomy that cannot be resolved. No policy should gain reward from a bookkeeping error.

## Patient-specific execution contract

The central workflow is **case upload/review -> fixed patient simulation -> initial search routes -> actual case-specific learning -> independent candidate evaluation -> desktop comparison**. One patient supplies many simulated episodes but only one independent patient. No complete patient-level surgical action dataset is required to test this optimization; no quantity of simulated trials establishes real neurological injury probabilities.

Implement `SEARCH` and `PATIENT_SCRATCH_RL` first. Add `POPULATION_FROZEN` and `POPULATION_ADAPTED` when a shared checkpoint exists. The final benchmark compares all four; the early prototype need not wait for the latter two. Use the best measured mode as the product default, without assuming warm-starting or scratch training wins.

For each case, version the input cutoff, geometry, objective, tool catalog and world generator. Create independent optimization, checkpoint-selection and final-evaluation worlds. Sample a coherent hidden anatomy per episode. Train with actual policy updates, using a fixed resource/stopping rule; keep the simulator and objective frozen. For adaptation, clone the shared checkpoint and do not carry one test patient's updates into another. Independently check the frozen candidate set at finer geometry resolution and under additional model assumptions. Do not tune on final evaluation feedback.

Track code/data/model hashes, initialization, action/gradient counts, preprocessing cost, online compute and all failures. Label routes by geometric accessibility until legal sequential removal is simulated. Preserve initial search routes next to refined candidates so improvement is demonstrated, not implied. A frozen adaptation procedure may use held-out patients' preoperative scans during case adaptation; it may not use postoperative outcomes, future information or held-out world truth. See Section 15.8 for the full nested protocol.

## Halos and patient context

Implement a physical-distance buffer first, clearly labeled a surrogate. Extend it to coherent tract/segmentation/registration ensembles and then action-conditioned contact/disconnection events. A normalized blur or streamline frequency is not a postoperative-deficit probability. Missing reconstruction coverage must remain unknown, not zero-risk. Separate shaft size from anatomical and pose uncertainty so safety buffers are not double-counted. Use Sections 6.6–6.7 and F13/E06; F14 is optional.

Include molecular and baseline-function fields with provenance and availability timestamps. IDH, 1p/19q and MGMT-related evidence may support stratification and clearly declared biological scenarios when actually known. Do not infer missing pathology, shrink functional protection because a tumor is aggressive, or train causal survival rewards from retrospective correlations. The core planner works without molecular metadata. Unknown physiology and vascular anatomy remain explicit limitations.

## Completion and release standard

The portfolio release should provide installation instructions, a small redistributable fixture or reproducible public-data downloader, a scripted real-case demonstration, configuration files, experiment tables, a limitations/model card, data lineage, source/license notices and tests. All displayed numbers must derive from saved evaluator outputs.

Show multiple candidate plans, rejected-plan explanations, instrument sensitivity, a changing cavity, uncertainty-sensitive rankings, and at least one failure or abstention. Publish RL results whether positive or negative relative to the predefined competitors. Do not call a demonstration clinically validated or claim a first-of-its-kind method without renewed prior-art screening.

End each implementation session with a factual handoff: what changed, what ran, what failed, artifact paths, unresolved risks and the next highest-value test. Do not substitute a large new architecture proposal for finishing the current vertical slice.
