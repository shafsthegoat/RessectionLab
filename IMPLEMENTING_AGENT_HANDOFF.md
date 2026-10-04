# Implementing-agent handoff

Prepared October 2, 2026. Read `MASTER_PLAN.md` before executing. This is an implementation brief, not a record of completed experiments.

## Mission

Build a local-first desktop research application that takes multimodal adult glioma imaging, constructs an inspectable patient-specific simulation, and compares instrument-aware sequential resection strategies. Include a genuine reinforcement-learning investigation, strong classical/search competitors, motor and language hazard representations, explicit uncertainty, and visible rejected alternatives. The intended outcomes are a technically excellent public portfolio, a credible Medivis-facing demonstration, and an appropriately bounded research publication.

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
7. A serious RL comparison is required, but the product may use search/MPC if it wins. Do not hide negative RL results.
8. The native app must remain responsive, support cancellation and saving, and make failures and unsupported inputs visible.
9. Adopt Medivis's case-centered product principles without copying branding or claiming affiliation, device clearance or compatibility.
10. Preserve source licenses, dependency/model licenses and user work. Do not put large medical datasets or restricted assets in the repository.

## Begin with evidence and one real case

First, inventory compute, storage, current repository state and any installed medical-imaging libraries. Create a short architecture/evidence log. Load `DATASET_MANIFEST.json`, confirm the current authoritative source record, and acquire only five UCSF-PDGM cases with gradients, source masks and necessary metadata. Include a documented failure case where possible. Inspect actual file contents before assuming completeness.

Acquire the small DES map releases only after checking their terms and template space. These are functional priors, not patient-level clinical outcome labels. Use motor first; add language once the motor pipeline is inspectable. Resolve repeated patients and BraTS overlap before freezing splits or importing a pretrained segmenter.

Build the earliest vertical slice around one genuine public case: linked MRI slices and 3-D anatomy, inspectable transforms, editable source annotations, one hypothetical access window, two instrument geometries, a few certified candidate plans, and save/reopen. The first slice may use supplied annotations and deterministic planning. Mark it annotation-assisted; do not present it as autonomous segmentation or a trained RL result.

The first meaningful success is not a complicated repository scaffold. It is a reproducible case where changing a tool's geometry changes what the planner can reach, and the visualization shows why.

## Decisions to settle empirically

**Desktop foundation:** Compare a custom Slicer application/extension with a PySide6 plus VTK-style implementation using the same small case. Test MPR/3-D alignment, asynchronous computation, cavity replay, save/reopen and packaging on the real machine. Prefer the option that makes the imaging workflow reliable with less custom infrastructure.

**Patient representation:** Keep authoritative geometry in a declared physical frame. A cropped, multiresolution grid can serve the policy, but high-resolution geometry must independently check final actions. Choose sparse/dense fields, meshes and graph features by profiling rather than assuming one representation serves every task.

**Segmentation and tract reconstruction:** Audit model pretraining, gradients and tumor-domain failures. A framework installation is not a suitable pretrained model. Maintain source-annotation and inference-only benchmark tracks. Implement correction/QC before pretending every reconstruction is trustworthy.

**Simulation:** Start with repeatable geometry, exposed-frontier removal and explicit tool envelopes. Add coherent uncertainty before expensive tissue mechanics. ReMIND/RESECT can constrain image updating and shift scenarios, but do not directly supervise arbitrary tool-action forces or tissue destruction.

**RL:** Begin with masked macro-actions, a compact spatial encoder and a simple policy/value baseline. Compare against greedy, beam/CEM search and MPC using the same legal actions. Add recurrence, preference conditioning or separate cost critics only when the experiment requires them. Reserve continuous low-level control for a demonstrated need.

## Required sequence of deliverables

Complete the master-plan milestones through vertical slices. Keep a working application while developing headless simulation and training. Every milestone should produce an artifact that can be inspected without trusting a narrative summary.

The core progression is: source manifest and eligibility checks; aligned patient representation; motor/language evidence fields; tool-aware deterministic simulation; objective and baseline suite; RL environment tests and training; uncertainty and observation-driven replanning; integrated desktop comparison; frozen external evaluation; reproducible release and manuscript.

A milestone is not complete because tests were written. Run them, retain the result, and include a small real-case inspection in addition to synthetic fixtures. Separate known limitations from completed behavior.

## Experiment discipline

Use `EXPERIMENT_PROTOCOL.md` as a starting specification. Freeze patient-level splits and evaluation settings before final comparisons. Split repeated visits and derivatives together. Hold UPenn out of development initially; UTSW tests structural generalization only unless its input contract changes. Use RESECT independently of ReMIND for the corresponding registration task.

Log code revision, data/model hashes, hyperparameters, hardware, random seeds, training and inference cost, failures and all attempted variants. Report statistical uncertainty across patients, not across thousands of correlated rollouts from one patient. Use a separately implemented or independently configured geometry checker for final validation.

Before large training, adversarially test rewards and action legality. Include immediate STOP, repeated removal, tool cycling, through-tissue travel, missing gradient input, left/right mistakes, stale plans after edits, and uncertain anatomy that cannot be resolved. No policy should gain reward from a bookkeeping error.

## Completion and release standard

The portfolio release should provide installation instructions, a small redistributable fixture or reproducible public-data downloader, a scripted real-case demonstration, configuration files, experiment tables, a limitations/model card, data lineage, source/license notices and tests. All displayed numbers must derive from saved evaluator outputs.

Show multiple candidate plans, rejected-plan explanations, instrument sensitivity, a changing cavity, uncertainty-sensitive rankings, and at least one failure or abstention. Publish RL results whether positive or negative relative to the predefined competitors. Do not call a demonstration clinically validated or claim a first-of-its-kind method without renewed prior-art screening.

End each implementation session with a factual handoff: what changed, what ran, what failed, artifact paths, unresolved risks and the next highest-value test. Do not substitute a large new architecture proposal for finishing the current vertical slice.
