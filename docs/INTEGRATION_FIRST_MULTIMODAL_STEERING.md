RessectionLab: Integration-First, Multimodal Surgical Simulation and RL Steering Directive
0. Authority, scope and intent
This is a steering directive for executing the existing RessectionLab supergoal. It is not a new supergoal, a replacement research plan, or permission to abandon any previously specified capability.

Continue implementing the full existing surgical planning, rehearsal, anatomical reconstruction, reinforcement learning, vascular, functional, mechanical, physiological, validation and desktop application objectives.

Read and honor the existing repository instructions, current execution ledger, MASTER_PLAN.md, EXPERIMENT_PROTOCOL.md, and docs/SUPERGOAL_REAL_OBSERVATIONS.md, including the later explicit human steering that superseded the original prohibition on simulator-generated RL experience.

Do not rewrite or reduce the supergoal. Do not quietly remove advanced requirements because an intermediate prototype is easier.

The purpose of this directive is to correct an emerging execution problem:

The project is accumulating independently developed scientific and engineering components faster than those components are becoming part of one functional application.

That must change.

From this point forward, the agents must treat multimodal patient reconstruction, instrument-aware simulation, RL planning, independent evaluation and the desktop UI as interacting components of a single system.

Research should continue aggressively, but integration must happen continuously.

The ultimate goal remains a technically exceptional, patient-specific glioma surgical-planning and rehearsal research platform that could demonstrate meaningful engineering and scientific capability to Medivis.

The system must not claim clinical validity that has not been established.

1. Central architectural principle: One patient, one anatomical world, one evolving simulation
The most important instruction
Do not develop separate segmentation, vascular, tissue mechanics, instrument, surgical simulation, RL and UI systems that will eventually need to be connected. Design their shared interfaces now and integrate them incrementally.

Every subsystem must operate on compatible representations of:

The patient's actual acquired imaging.

The estimated and observed anatomical structures derived from that imaging.

The spatial registration and coordinate systems relating those structures.

The current modeled surgical state.

The available surgical instruments and their geometry and capabilities.

The sequence of actions already performed.

The consequences and uncertainty associated with those actions.

The information available to the planning algorithm at each decision.

The independent information reserved for scientific evaluation.

The data needed to reconstruct and visually replay the same operation in the desktop application.

These representations should share stable identities, physical coordinates, units, provenance, versioning and reproducible state transitions.

Avoid incompatible parallel implementations of these concepts.

Shared patient case representation
Develop or consolidate a canonical patient-case abstraction that can represent:

Structural imaging and its acquisition metadata.

Multimodal image registration.

Skull and bone structures.

Brain tissue and relevant anatomical compartments.

Tumor regions and subregions.

Arterial and venous anatomy.

Ventricles and cerebrospinal fluid spaces.

Functional cortical regions.

White-matter pathways and tractography.

Tissue properties and their uncertainty.

Anatomical surfaces and volumetric representations.

Surgical access geometry.

Observed and estimated anatomical boundaries.

Derived evidence from segmentation and reconstruction models.

Missing or unsupported evidence.

The representation should accommodate both volumetric and surface-based structures without requiring every subsystem to maintain its own incompatible patient model.

The same registered anatomy must be usable by the renderer, planner, simulator, evaluation system and training environment.

Do not duplicate anatomical truth into disconnected systems merely because doing so is convenient for a particular subagent.

Separate three different worlds
Maintain a strict distinction between:

A. Acquired patient evidence

What was actually measured, observed or annotated in the source datasets.

B. The estimated planning world

What can legitimately be reconstructed from permitted preoperative images and other available information. This includes uncertainties, incomplete segmentations and model-derived anatomical estimates.

C. The privileged simulation and evaluation world

Additional information available for simulator development, training critics, privileged teachers, sensitivity studies or sealed post-planning evaluation, according to the experimental protocol.

Never allow information from C to enter B through action generation, legality masks, proposed corridors, tissue support, procedural constraints or other indirect channels.

The separation must be implemented and tested through the complete software stack, not merely by masking one observation tensor.

2. The desktop application must develop alongside the science
UI development is no longer a downstream task
The existing Electron/React/WebGL application is a central deliverable, not a presentation layer to be constructed after the research is finished.

When a meaningful new backend capability becomes available, the responsible agents must coordinate with the integration/UI team to make it inspectable through the application.

Examples:

If vascular segmentation becomes available, provide a way to load and visualize vascular structures.

If a new anatomical dataset is admitted for display, make its supported modalities and labels accessible through the case workspace.

If a new instrument simulation is implemented, make the instrument selectable and its modeled actions replayable.

If tissue deformation is supported at a particular validated or experimental fidelity, show its changing state and scope of validity.

If a new planning algorithm becomes executable, expose its candidate strategies and comparison results.

If uncertainty is quantified, show the actual coverage and uncertainty information rather than an invented clinical-risk percentage.

Every completed major capability must have both a tested backend contract and an appropriate application-facing integration path.

Research-only mechanisms may remain internal while incomplete, but they must not accumulate indefinitely without an integration plan.

Medivis-inspired imaging workspace
Develop the desktop application toward the visual and interaction quality of professional medical-imaging software, including the general level of anatomical detail and clarity associated with Medivis.

Do not copy proprietary assets or misrepresent the product as Medivis software.

The interface should clearly communicate that RessectionLab is a multimodal, patient-specific surgical planning and simulation platform, not simply an MRI viewer with two geometric route suggestions.

Prioritize:

High-quality 3D volume and surface visualization.

Linked axial, sagittal and coronal image views.

A synchronized interactive 3D anatomical scene.

Camera orbit, zoom, clipping planes, slice controls and opacity adjustment.

Anatomical structure isolation and visibility controls.

Meaningful legends for anatomy, estimates, uncertainty and missing coverage.

Clearly differentiated tool geometry and surgical states.

Alternative plan comparison.

Instrument selection and replay controls.

An understandable sequence of surgical actions.

Reproducible inspection of why different plans diverge.

Preserve the existing renderer's physical-coordinate, source-integrity and replay safeguards.

Improve the existing application rather than replacing functional infrastructure unnecessarily.

3. Multimodal imaging upload and reconstruction must become a first-class workflow
The user should be able to create a patient case and import multiple relevant imaging acquisitions.

The architecture must support realistic combinations of:

T1-weighted MRI.

Contrast-enhanced T1 MRI.

T2-weighted MRI.

FLAIR.

CT.

CTA.

TOF-MRA and other appropriate MRA acquisitions.

Diffusion imaging and derived tractography.

Functional MRI when suitable evidence is available.

SWI and other useful vascular or susceptibility-sensitive sequences.

Intraoperative ultrasound or other intraoperative observations as a separate later-timepoint category.

Support the appropriate research formats, including NIfTI and DICOM where feasible, using vetted imaging libraries and preserving source metadata and spatial calibration.

A modality's presence must not imply that it provides clinically usable segmentation or quantitative functional evidence.

Desired case-import experience
A user should be able to:

Create or open a patient workspace.

Import multiple imaging series.

Identify the modality of each series.

Review alignment and image geometry.

Select an appropriate reference coordinate system.

Register compatible acquisitions where scientifically supported.

Inspect whether registration succeeded or requires review.

Import source-provided segmentations or annotations.

Request available automatic segmentations.

Inspect and selectively display the resulting anatomy in a synchronized 3D scene.

The system should preserve originals and derived outputs separately.

Handle mismatched resolution, orientation, field of view, modality coverage and acquisition times explicitly.

Do not silently assume that two datasets or modalities occupy the same physical coordinate frame.

Automatic segmentation
Build a modular segmentation system that can support, when eligible methods and evidence exist:

Whole-brain extraction.

Tumor and tumor-subregion segmentation.

Skull/bone segmentation.

Vascular segmentation.

Ventricular segmentation.

Relevant anatomical compartments.

White-matter tract estimates.

Functional-region estimates or prior overlays.

Segmentation modules should be interchangeable and source-aware rather than hardcoded into the renderer.

Each segmentation should carry:

Source image identity.

Algorithm or model identity.

Model weights and training-lineage information.

Spatial transform and coordinate frame.

Prediction and/or annotation status.

Anatomical coverage.

Confidence or uncertainty where meaningfully available.

Quality-control results.

Whether it is display-only, eligible for research planning, or eligible for independent evaluation.

If a qualified dataset already supplies segmentations, use those source annotations where appropriate.

Do not unnecessarily rerun a model simply to recreate a source-provided label.

But do not confuse provided labels, automatic estimates and independent evaluation truth.

Layer visibility
The application should expose a clear anatomical layer inventory.

For example, users should be able to independently toggle:

Original MRI or CT.

Brain surface.

Skull.

Tumor.

Arteries.

Veins.

Ventricles.

Cortical functional regions.

White-matter tracts.

Surgical corridor and instrument geometry.

Simulated cavity.

Modeled deformations.

Estimated hazard or uncertainty fields.

Include opacity and color controls where appropriate.

Do not show missing anatomy as an empty, risk-free space.

If an artery segmentation is absent, display vascular anatomy unavailable or unassessed, not an apparently vessel-free brain.

Realistic 3D reconstruction
The goal is a spatially coherent, anatomically detailed representation of the individual patient's brain, derived from actual imaging and explicitly labeled estimates.

Use suitable volume rendering, surface extraction, spatial registration, smoothing and GPU acceleration.

Optimize large image handling, streaming and memory usage without discarding meaningful anatomy or falsifying fine structures.

Do not invent small vessels or functional tracts simply to improve visual appearance.

The reconstructed anatomical representation must be compatible with simulation and planning, rather than being an unrelated decorative model.

4. Instrument-aware simulation is a primary architectural priority NOW
The central scientific potential of this project is not merely finding a path through a brain.

It is learning how different surgical instruments interact with different anatomies over sequences of actions.

That requires reusable instrument behavior models.

Patient-independent instrument mechanics
Instrument behavior should be defined by the tool and its interaction with local anatomy, not by a hardcoded patient ID, tumor location or manually scripted successful procedure.

The same action machinery should work on different patient geometries, subject to evidence and model limitations.

The simulator should receive the patient's geometry, tissue state, available observations, instrument parameters and current surgical state.

It should then calculate the modeled consequences of the selected action.

This is the mechanism by which RL may learn transferable behavioral patterns.

Do not assume such transfer will automatically occur. Test it.

Establish a shared instrument registry
Develop a reusable representation of each surgical instrument, including:

Instrument type.

Geometry and working dimensions.

Active tip or operating region.

Reachability and access requirements.

Applicable material interactions.

Allowed motions and operating modes.

Contact conditions.

Tool-specific action parameters.

Preconditions and contraindicated modeled interactions.

Collision and exclusion rules.

State-transition behavior.

Data source and physical-model fidelity.

Uncertainties and unvalidated effects.

The registry should be consumed by simulation, planning, RL action construction, geometric validation and desktop visualization.

Avoid one-off instrument models that cannot participate in the same simulation episode.

Instrument families to support progressively
The architecture should accommodate:

Suction/aspiration instruments

Removal of exposed material, tip-contact restrictions, shaft geometry, modeled removal volume and effects of suction-related tissue interaction.

Dissectors and microinstruments

Separation or manipulation of modeled tissue boundaries, contact forces, displacement constraints and access changes.

Retractors

Persistent tissue displacement, retraction geometry, loading duration, changing visibility and mechanical consequences when supported by an appropriate model.

Bipolar coagulation instruments

Localized energy delivery, hemostatic effects and potential adjacent-tissue consequences where adequately modeled.

Ultrasonic aspirators and other tumor-removal instruments

Instrument-specific removal behavior, material selectivity, geometry and operating parameters where empirical support exists.

Scissors and cutting instruments

Explicit cutting operations, geometric interactions and resulting tissue-state changes.

Irrigation and suction combinations

State changes relevant to modeled visibility, fluid clearing and exposure.

Vascular clips or hemostatic tools

Manipulation or occlusion of explicitly modeled vascular structures, with physiological effects distinguished from unsupported assumptions.

Exposure and bone instruments

Initial access and craniotomy-related operations, including skull interactions and geometric exposure changes.

These are target capabilities, not assertions that their physiological behaviors are presently understood or validated.

Implement the best-supported subset first while ensuring the interface is extensible to the full planned instrument inventory.

Instruments must produce different consequences
Two tools must not merely have different names, colors or dimensions while sharing an identical generic removal operation.

When mechanistic evidence supports different behavior, those differences must change the actual simulation transition.

For each action, define what it can:

Move.

Contact.

Remove.

Separate.

Displace.

Expose.

Obstruct.

Damage in the model.

Leave unchanged.

Make newly accessible or inaccessible.

If a mechanism is not yet supported, retain an explicit unsupported or simplified state rather than inventing a precise interaction.

5. Build one persistent, sequential surgical environment
The RL environment must represent an evolving operation, not a collection of independent static routes.

The existing suction-based simulator, geometric collision machinery, action histories and replay functionality should be reused and progressively generalized.

A surgical episode should maintain persistent state such as:

Current anatomical geometry.

Remaining tumor.

Removed tissue.

Surgical cavity.

Exposed surfaces.

Active instruments.

Instrument poses and contact history.

Access constraints.

Known and unknown structures.

Modeled deformation when supported.

Vascular-contact or vascular-state information.

Observations obtained during the episode.

Previously performed actions.

Procedure status and stopping conditions.

An action changes this state, and the next action is chosen based on the resulting state and available observations.

Examples include:

An instrument removes tissue, changing the cavity.

The altered cavity changes reachability and visibility.

A subsequent instrument accesses a previously inaccessible region.

A retraction action changes geometry for later actions.

An observation action reduces a particular uncertainty.

A modeled vascular event changes available options when the underlying mechanism supports that consequence.

A planner decides to stop because its constraints or uncertainty no longer justify additional intervention.

Not all examples need to be scientifically complete immediately. But the shared state-transition architecture must support them now.

Avoid a permanently simplified simulator
A six-cell or two-action environment is useful for numerical software tests.

It is not an adequate long-term representation of this project's central learning problem.

Develop progressively longer episodes with multiple state-changing actions, tool transitions, varying geometries and uncertainty.

Do not indefinitely optimize RL performance on a toy environment while leaving the intended multi-instrument environment disconnected.

Shared simulation semantics
The renderer must not invent a visually appealing procedure different from the one the numerical simulation actually executed.

The displayed instrument trajectory, target removal, cavity changes and other modeled effects must be reconstructed from the same authoritative transition history used by the simulator and evaluator.

The environment should provide a reproducible action record and state snapshots or deltas that the UI can replay.

This is essential for both engineering correctness and scientific credibility.

6. Design for unfamiliar patients, not a small set of hardcoded brains
The simulator should work with arbitrary supported patient-specific anatomical inputs within its declared geometric, data-quality and computational constraints.

Instrument interactions should derive from geometry, material assumptions, operating parameters and the current state.

Avoid patient-specific branches that silently encode expected outcomes.

Cross-patient learning
The long-term RL objective is to learn reusable surgical decision behavior across different anatomies.

During eligible training, expose the model to variation in:

Brain and tumor geometry.

Tumor position, shape and compartment structure.

Vessel distributions and visibility.

White-matter anatomy where available.

Surgical access constraints.

Instrument dimensions and availability.

Tissue properties.

Model uncertainty.

Observation quality.

Consequences of previous actions.

Use legitimate acquired anatomical data and explicitly labeled simulated variations according to the active training policy.

Maintain clear separation between empirical anatomy, simulator assumptions and generated training experience.

The desired learned behavior is the ability to reason about what an instrument is likely to do in the current anatomical state, not memorization of one patient's preferred route.

This must be demonstrated through held-out cases and controlled changes in instrument geometry and anatomy.

Scientific caveat
General-purpose computational support for different patient brains does not itself establish generalizable biological accuracy.

The simulator may be able to execute on many anatomical reconstructions while still having insufficient evidence for a particular tissue or vascular effect.

Track those two claims separately.

7. Reinforcement learning must use this shared environment
The agents should actively move RL development toward the full sequential simulator as the environment becomes capable of supporting it.

Do not maintain one simplified training simulator and another independent surgical rehearsal simulator unless there is an explicit, tested equivalence or fidelity relationship between them.

Prefer a shared environment interface with clearly identified fidelity modes.

Core training question
Can reinforcement learning, using privileged training feedback and a broad collection of anatomical simulation experiences, discover useful multi-instrument planning policies that transfer to unseen patient-specific cases when only realistically available observations are provided?

This remains a falsifiable hypothesis.

The policy must not gain access to held-out anatomy through hidden geometry, action masks, candidate construction or planner-specific shortcuts.

Privileged critics, teachers and simulator feedback may be used in specifically declared training arms.

RL actions
The RL action representation should be designed for sequential surgical behavior, including:

Choosing an instrument.

Selecting a legal access or workspace region.

Choosing an action type.

Specifying bounded motion or operating parameters.

Acting on a visible or estimated target region.

Requesting additional information where modeled.

Changing or withdrawing an instrument.

Stopping.

Consider hierarchical policies or structured action spaces when appropriate.

Do not expand the action space so aggressively that training becomes computationally intractable before the environment is stable.

Use curriculum learning, carefully defined abstractions, imitation, model-based planning, constrained RL or hybrid methods where scientific evidence and benchmarking justify them.

The agents have autonomy to select the most suitable algorithms.

Rewards and constraints
The scientific objectives remain:

Maximize appropriate modeled target removal.

Preserve healthy tissue.

Avoid supported anatomical hazards.

Respect instrument feasibility.

Consider modeled functional structures.

Minimize unnecessary damage and procedure burden.

Account for uncertainty.

Preserve the option to stop or abstain.

Do not confuse arbitrary reward weights, anatomical proximity penalties or simulator-derived event frequencies with calibrated clinical neurological injury probabilities.

Use explicit constraints and measured or model-conditioned outcomes where appropriate.

Strong competitors remain mandatory
Continue matched evaluation against classical search, greedy heuristics, model-predictive approaches, imitation and hybrid planning.

These methods must receive the same legally available test-time information.

Do not require RL to win in advance.

A hybrid or search method that is empirically superior should remain available in the product.

The intended innovation is useful patient-specific planning behavior, not the presence of an RL algorithm for its own sake.

8. Integrate vascular, mechanical and physiological research through shared interfaces
Current specialized research programs must continue.

But they must have a planned connection to the surgical state.

Vasculature
The existing full-tool vascular-contact work should become part of a reusable evaluation interface and, where permitted, the observable estimated planning representation.

Design for:

Registered arterial and venous layers.

Explicit vessel coverage and segmentation uncertainty.

Geometric instrument-vessel encounters.

Modeled vessel displacement where supported.

Differences between proximity, contact, compression, occlusion and rupture.

Future hemodynamic or bleeding state when independently justified.

Do not immediately equate contact with vessel rupture or a clinical complication.

The UI should be able to display vascular layers and independently evaluated encounters even before full vascular physiology becomes scientifically defensible.

Tissue mechanics
The numerical mechanics program should expose a stable interface capable of receiving a supported local tissue state, instrument interaction and physical boundary conditions, then returning a modeled deformation or force response.

This should connect to the shared surgical state, rather than exist as an entirely unrelated solver demonstration.

Where patient-specific mechanical parameters are unavailable, represent parameter uncertainty and clearly identify the fidelity and scope of the model.

Do not block the initial integrated simulator while waiting for perfect physical calibration.

Functional risk
Motor and language evidence should remain anatomically and clinically distinct.

Display the actual type of supporting evidence, such as registered tractography, observed mapping or population priors.

Avoid treating a population atlas as an observed patient-specific functional map.

An event that intersects an estimated functional structure may be reported as a model-conditioned anatomical encounter, but not automatically as a probability of permanent disability.

Physiology
Maintain extensible connections for anesthesia, perfusion, blood loss, pressure and physiological consequences required by the supergoal.

Do not fabricate detailed systemic physiological responses simply because an instrument action occurred.

Integrate supported intermediate mechanisms incrementally and retain unknown consequences.

9. The UI must visualize the exact science that exists
Establish a strict UI/backend evidence contract.

For every new scientific capability, the application should know whether it is:

Available and scientifically admitted for the stated use.

Available as an experimental research feature.

Display-only.

Generated or simulated.

Awaiting anatomical or numerical review.

Unsupported because required data or validation are missing.

The interface must not present all of these categories as equivalent.

Scientific feature integration rule
Whenever an agent develops a meaningful new capability, it should provide:

An executable backend operation or reusable interface.

Its required inputs and produced outputs.

A typed application-facing contract.

A reproducible software test.

A way for the existing desktop application to display, control or inspect the result where appropriate.

A statement identifying unsupported scientific conclusions.

The dedicated UI/integration team should consume these contracts continuously.

Do not wait until the entire supergoal is finished before connecting scientific progress to the frontend.

Avoid cosmetic-only progress
Do not implement nonfunctional controls or staged screenshots merely to suggest that a feature exists.

For example:

A Segment vessels action must either run a real supported segmentation path or visibly explain why it is unavailable.

A Simulate surgery action must execute and replay modeled state transitions.

An instrument selector must affect the underlying action semantics.

A tumor removal visualization must correspond to actual modeled removal.

An RL comparison panel must identify the policy used, its checkpoint, the observation contract and the completed strategy.

A vascular warning must reference real geometry and a defined modeled encounter, not a fabricated medical prediction.

The UI should be technically impressive because it exposes a genuinely functioning system.

10. Redefine immediate development priorities
These priorities are about execution order, not removing or postponing the complete supergoal.

Priority 0: Establish and enforce the integration architecture
Immediately audit the existing imaging, geometry, simulation, RL, evaluation and Electron/React boundaries.

Reuse the current resectionlab modules and desktop infrastructure wherever possible.

The following existing areas are especially relevant:

geometry.py

simulation.py

native_spatial_task.py

research_estimate_planning.py

matched_private_vascular.py

private_vascular_evaluation.py

vascular_contact_streaming.py

desktop_bridge.py

desktop/src/App.tsx

desktop/src/viewer/

desktop/src/ReplayStepControl.tsx

desktop/electron/preload.cjs

Verify current paths and interfaces before editing.

Define a minimal set of shared contracts covering patient cases, registered anatomical layers, tool actions, evolving simulation state, planning observations, complete strategy records and evaluation results.

Do not force every existing subsystem through a new abstraction if a compatible interface already exists.

Deliverable: a tested shared integration pathway, not another large standalone architecture document.

Priority 1: Create a functional multimodal case workspace
Make the existing desktop application visibly support multiple imaging inputs, registered anatomical layers, imported segmentations and eligible automated estimates.

Begin with supported modalities and data already available.

The interface should show clear distinctions between acquired images, source annotations, model estimates and missing evidence.

Deliverable: an actual running application in which a user can load a supported multimodal case and inspect its anatomical structures in 2D and 3D.

Priority 2: Integrate a minimally viable multi-instrument surgical episode
Build on the existing suction simulation and rigid-tool geometry.

Implement at least two genuinely different modeled instrument interactions, chosen for scientific support and implementation feasibility.

Execute a sequence with multiple decisions and persistent state changes.

Demonstrate that a previous action changes the next action's feasible geometry, modeled outcome or observation.

Deliverable: reproducible multi-step execution and actual desktop replay, not just independently rendered static instrument trajectories.

Priority 3: Connect RL, search and hybrid planning to that same environment
Use the shared action and state interfaces.

Run controlled generated development tasks first when required.

Move to eligible patient-derived planning inputs through the appropriate source, registration and model-admission gates.

Preserve the private evaluation boundary and compare complete sealed strategies.

Deliverable: a working comparison on an evolving multi-instrument environment, plus an explicit record of what is and is not demonstrated on real patients.

Priority 4: Expand anatomical richness and instrument effects continuously
Bring the existing vascular-contact system into the shared workflow.

Advance skull/exposure, tissue mechanics, functional fields, visibility, tool interactions and physiological mechanisms through their respective scientific workstreams.

Integrate each accepted capability incrementally.

Deliverable: the same running application becomes anatomically and mechanically richer, rather than spawning new isolated simulators.

Priority 5: Complete increasingly demanding generalization experiments
Expand simulator diversity, eligible patient coverage, action horizons, instrument variety and uncertainty.

Evaluate on frozen held-out patient clusters.

Measure whether the learned policy genuinely transfers to different geometries and produces better supported outcomes than matched competitors.

Preserve negative results.

Deliverable: reproducible evidence for or against the primary RL generalization hypothesis.

These priorities should overlap wherever dependencies permit.

11. Mandatory near-term vertical slice
I want the integration lead to prioritize a working vertical slice during the next 24–48 hours of active development, or report the specific blockers preventing it.

This is a development target, not permission to manufacture evidence or bypass scientific admission rules.

The slice should demonstrate as much of the following as the current accepted components permit:

Input

Load an eligible development case from actual imaging, or use an explicitly labeled generated software-development fixture for portions that cannot yet legally or scientifically operate on patient data.

Anatomy

Display the available acquired image and its source or estimated anatomical layers in the same registered workspace.

Planning

Construct an allowed estimated planning state without hidden anatomical truth.

Simulation

Execute at least a small multi-step state-changing sequence using instrument-aware actions.

Comparison

Generate search and, where actually executable, learned-policy alternatives under matched information restrictions.

Evaluation

Seal strategies and inspect the available independently computed modeled outcomes. Use real withheld patient evidence only after every relevant admission requirement is satisfied.

Visualization

Replay the exact executed sequence in the desktop application, including tool motion, changes in modeled tissue/cavity state and supported anatomical encounters.

Evidence

Save the source identities, experiment configuration, executed actions, state transitions, numerical outputs, testing results and known limitations.

If the full slice cannot be completed, deliver the largest honest executable subset and identify the next missing connection.

Do not turn the blocked pieces into visually convincing placeholders.

12. Subagent coordination must become integration-driven
Use the available Astra extra-high subagents extensively, subject to actual tool, compute and concurrency limits.

The agents should operate as coordinated contributors to one system, not as independent research laboratories with their own incompatible representations.

Organize around these responsibilities:

Lead architect and integrator
Own shared contracts, integration order, dependency resolution, acceptance tests and final system consistency.

Actively identify disconnected projects and redirect their outputs toward the working application.

Imaging and reconstruction team
Own multimodal import, registration, anatomical reconstruction, segmentation interfaces, provenance, spatial QC and usable anatomical layers.

Surgical world and instruments team
Own the common evolving state, instrument registry, action semantics, geometry and reusable material-interaction interfaces.

RL and planning team
Own learning methods, policy architecture, training environments, search comparators, constrained planning, generalization experiments and fair accounting.

Mechanics, vasculature and physiology teams
Continue their detailed scientific work while delivering callable mechanisms compatible with the shared environment.

Desktop and visualization team
Own the medical imaging workspace, anatomical layers, instruments, surgical animation, controls, alternative strategies and scientific result inspection.

This team should run continuously, not be activated only after the other groups finish.

Independent validation team
Verify source integrity, patient-role restrictions, numerical correctness, information leakage, physical fidelity, strategy replay and scientific claims.

End-to-end systems testing team
Exercise the complete path from data input through reconstruction, simulation, planning, evaluation and visualization.

Independently verify that what the desktop displays matches the underlying backend state and recorded scientific output.

The lead may create more teams, divide work differently, or merge responsibilities where that improves throughput.

Subagent delivery requirements
Every major assigned task should identify:

Which shared subsystem it extends.

The inputs and outputs it requires.

The existing interfaces it must preserve.

The end-to-end behavior it is intended to unlock.

Its scientific assumptions and evidence level.

The tests needed to accept it.

How the application or simulator will consume it.

Its handoff to the integration lead.

An isolated component without a plausible integration pathway is incomplete from a delivery perspective, even if its own tests pass.

Do not permit independent subagents to create divergent patient schemas, instrument registries, coordinate conventions or definitions of surgical state.

Respect the existing single-writer repository discipline. Parallelize research, independent computation, and isolated proposals; coordinate source edits and integration. Preserve running downloads, unrelated work and the active scientific evidence ledger.

13. Engineering standards and progress measurement
Maintain scientific rigor
Do not weaken the real-patient evaluation protocol, source lineage, patient separation, geometry checks, withheld-information protections or uncertainty requirements.

Do not claim clinical effectiveness, predicted postoperative neurological outcomes, vascular rupture probabilities or accurate physical simulation without supporting evidence.

Simulator-generated training experience remains permissible under the active human steering, but never becomes independent patient validation.

Avoid excessive serialization
Not every routine implementation correction needs a separate elaborate review campaign.

Use suitable focused tests and automated regression suites throughout development.

Reserve deeper independent scientific review for consequential numerical results, patient-data admission, benchmark claims, novel mechanisms and changes affecting evidence integrity.

Continue rigorous safety and scientific boundaries, but do not allow repetitive bookkeeping to become the dominant project deliverable.

Integration acceptance
A capability should be considered integrated only when:

Its backend contract is implemented.

A caller in the shared system actually exercises it.

Relevant tests pass.

Its outputs use the shared anatomy/state conventions.

Its provenance and limitations are preserved.

Its effects can be inspected in the application's relevant workflow, or a specific documented limitation explains why not.

Distinguish:

Research completed.

Code implemented.

Backend integrated.

Desktop integrated.

Patient-data qualified.

Scientifically validated.

Do not merge these statuses into a single claim of completion.

Measure real progress
Track:

Supported imaging modalities.

Qualified patient reconstructions.

Registered anatomical layers.

Segmentation methods actually executable.

Instrument types with real distinct state-transition implementations.

Supported multi-step surgical actions.

End-to-end simulated episodes.

Successfully reproduced UI replays.

Working planning algorithms.

Matched strategy comparisons.

Generalization results on eligible unseen patients.

Independent validation status.

Failed integration or scientific gates.

Commit counts, dataset bytes, passing microtests and the number of active agents are supporting operational metrics, not the primary definition of progress.

Progress demonstration
At meaningful milestones, provide:

A concise description of the newly working capability.

The exact command or workflow needed to reproduce it.

The case or fixture used.

Actual test and execution results.

The part of the desktop application where it is visible.

A genuine screenshot or recording of the working application when practical.

Remaining scientific limitations.

The next integration dependency.

If the desktop UI has not changed despite meaningful backend progress, explicitly explain why and identify the assigned integration task.

14. Autonomy, preservation and completion philosophy
You have broad technical autonomy to research and implement better approaches, including adapting published methods, optimizing existing algorithms, introducing suitable libraries, changing internal architectures and developing new RL or simulation methods.

Use that autonomy to make the overall system better.

Do not unnecessarily restart working components.

Do not replace the original supergoal with a narrower demonstration project.

Do not silently postpone complex scientific requirements indefinitely.

Do not introduce clinically unsupported mechanisms merely to make a demonstration appear complete.

Continue permitted data acquisition and independent research concurrently.

Use existing local-first infrastructure and actual authorized compute resources. Do not initiate unapproved purchases, paid services, external data transfers or clinical outreach.

Preserve existing results, original data, negative experiments, running processes and reproducibility records.

Maintain coherent milestone commits with genuine integration progress, rather than committing every tiny subagent output.

The guiding question for every subagent
How does this work make RessectionLab better at reconstructing a patient's anatomy, simulating a sequence of instrument interactions, learning or comparing surgical strategies, or explaining those strategies through the actual desktop application?

If the relationship is unclear, investigate whether the task is on the critical path or belongs to a longer-horizon scientific workstream.

Do not abandon necessary foundational research. But do not let it displace integration indefinitely.

15. Final directive
The complete RessectionLab supergoal remains unchanged.

The platform should ultimately support a user who can:

Upload a multimodal collection of patient imaging.

Inspect the scans in a coherent 2D/3D workspace.

Import or automatically derive eligible anatomical segmentations.

Turn anatomical structures and evidence layers on or off.

Construct an explicitly uncertain patient-specific planning environment.

Select or compare available surgical instruments.

Generate multiple sequential surgical strategies.

Simulate how supported instrument interactions change modeled anatomy over time.

Compare RL against strong planning alternatives using equivalent limited information.

Replay the complete modeled procedure visually.

Inspect modeled removal, anatomical encounters, constraints, unknowns and uncertainty.

Evaluate sealed strategies against independent patient-linked evidence wherever it exists.

Distinguish a successful software simulation from an experimentally supported biological prediction or validated clinical outcome.

The architecture should be designed around this complete workflow now.

Do not wait for every individual scientific subsystem to be perfected before developing the shared environment and UI.

Build the platform as an integrated system from the beginning. Expand its biological fidelity and anatomical sophistication as evidence allows. Allow RL to learn from the same instrument-aware environment that the desktop application visualizes.

The next major achievement should not be another isolated component. It should be a measurable expansion of an actual working, multimodal, adaptive, instrument-aware surgical planning and rehearsal system.

Continue autonomously, use subagents aggressively, preserve scientific integrity, and prioritize integrated executable progress.