# Tissue-mechanics validation contract

Status: prospective, 2026-10-04. No mechanics gate below has been executed or passed.
Keep the current geometric planner and patient splits unchanged. Use an established FEM package; compare frameworks in [tissue-mechanics-frameworks.md](tissue-mechanics-frameworks.md).

## Three separate claims

| Gate | Supported comparison | Remaining limitation |
| --- | --- | --- |
| Specimen response | Force/torque under measured imposed displacement/rotation | Ex-vivo specimens do not establish patient properties or retractor contact |
| Conditional patient displacement | Held-out retained-tissue motion given sparse observed motion | An observed displacement fit does not identify the surgical action that caused it |
| Surgical action-response | Independent tissue response to recorded instrument trajectory/loading | Requires suitable interaction measurements; currently an unresolved dependency |

The initial patient proof of concept is conditional displacement propagation through real retained anatomy. A simulated retraction scene must be labeled simulated and unvalidated for action-response.
Do not use these gates interchangeably or introduce mechanics-based policy comparisons before the relevant task gate passes.

## Evidence and prospective specimen selection

[Hyperelastic Human Brain 1–7](https://zenodo.org/records/8095559) supplies specimen geometry and displacement–force / angle–torque data. Its filtered, averaged curves approximate quasi-static hyperelastic response; they do not validate cutting, hysteresis or surgical rate dependence.
Acquisition pins and permitted schema-only inspection are recorded in [the acquisition manifest](../manifests/hbe_8095559_acquisition.json). Force/torque values remain sealed until exact roles and the bounded experiment are committed.
Immediate specimen-PoC selection rule, to resolve from actual schema before reading curves:

1. Inventory exact donor/specimen IDs, region metadata, geometry, mode and cycle filenames. Eligibility requires usable declared geometry and the prospectively required files; do not inspect outcomes to exclude specimens.
2. Sort metadata-eligible specimens by their actual numeric donor ID, then numeric specimen ID; if the released identifiers cannot be parsed this way, resolve the schema before proceeding. Select the first specimen, with no outcome-based replacement.
3. Mark its entire donor DEVELOPMENT permanently; all sibling specimens, cycles, loading levels and signed branches share that donor role. Other donor values remain sealed. Never randomly split curve rows or treat processed samples as independent experiments.
4. Fit only the selected specimen's third-cycle compression/tension. Keep its low-amplitude l1 third-cycle torsion (both signs) sealed as the within-specimen held-out-mode check until model and parameters are frozen; first-cycle and l2 curves are outside this first experiment.
5. Freeze the exact ID, region, file hashes, geometry, loading modes, evaluation points, solver/model candidates, budgets and acceptance tolerances in a small experiment manifest before fitting. Schema-derived specimen height is an input with provenance, not independently validated geometry.
6. Any later donor-transfer study needs a separate prospective donor-grouped split. The development donor cannot become a final-validation donor; region-matched selection is preferable, or regional heterogeneity must be explicit.

Metadata resolves the first specimen as HBE_01_03; [the prospective role manifest](../manifests/experiments/hbe-01-03-specimen-roles-v1.json) binds exact archive/member identities without opening curves. This first PoC tests within-specimen cross-mode prediction, not donor transfer; held-out-mode results cannot reopen fitting under the same claim.
The specimen step does not replace the subsequent real-patient anatomy displacement PoC and cannot establish patient forces.

## Patient measurement partitions

Use B for supplied boundary/observation measurements, C for calibration and model selection, and V for sealed validation correspondences. Freeze the allowed image/observation access for every method.
V may not determine registration, load placement, material selection, stopping or outcome-directed mesh refinement. Whole follow-up images used for fitting change the claim to image-conditioned updating; disclose that access.
Preserve patient and longitudinal family splits. Overlapping landmark sets remain linked, not independent validation cohorts.
[RESECT](https://aapm.onlinelibrary.wiley.com/doi/10.1002/mp.12268) provides retrospective correspondences and repeated-picking variability. Its two ultrasound landmark sets overlap; published picking distances are not total coordinate covariance or tracking uncertainty.
Retraction-specific validation additionally needs recorded instrument motion/contact and independent tissue measurements; a force prediction claim needs force/torque measurements. Imaging snapshots do not supply these automatically.

## Explicit mechanics and numerical verification

Record constitutive law, finite- versus small-strain regime, boundary/contact assumptions, units, mesh and solver version. Freeze a small justified parameter/scenario set before V is opened.
The specimen force test must reproduce the documented bonded-plate fixture and finite-strain loading range; a free-slip analytical formula or small-strain propagation model is not a substitute. Match the constitutive energy convention before transferring any fitted parameters.
For homogeneous linear displacement-only equilibrium with no body load and traction-free remaining surfaces, absolute stiffness scaling does not change displacement. Treat that scale as a nondimensional convention, never measured patient Young's modulus; absolute forces remain unsupported ([primary study](https://pmc.ncbi.nlm.nih.gov/articles/PMC3600363/)).
Do not infer patient stiffness from age, tumor label or genotype. Ex-vivo fits remain specimen/population estimates with explicit transfer limitations.
Check zero-load behavior, rigid motion and an appropriate analytical patch test before real-anatomy evaluation. Analytical solver controls are not synthetic patient training data.
Use at least three mesh resolutions and successive load-step refinements, comparing predictions at fixed physical points. Record solver failures, residuals, minimum deformation Jacobian and boundary/constraint violations.
Predeclare a numerical error budget small relative to measured spatial resolution and documented measurement uncertainty; require both finest-level displacement change and load-step change below it. Record actual units and values before validation, without inventing missing measurement uncertainty.
Check force/torque convergence separately for the specimen gate. A converged displacement field alone does not certify reactions.
Verify near-incompressible behavior using the actual element/formulation; a “mixed” label alone is insufficient. Require a justified stable/stabilized formulation and locking/refinement controls ([FEBio methods](https://febio.org/site/uploads/maas_jbme_2012.pdf)).

## Independent outcomes, baselines and acceptance

Export immutable physical rest mesh and forward displacement in the original canonical frame. The independent evaluator locates each V source point in the rest mesh and interpolates displacement itself; it does not trust a solver-provided error summary.
Compare predicted and observed positions in millimetres. Retain outside-mesh, removed-tissue and unsupported observations as counted exclusions; never turn them into zero errors or force correspondence through a cavity.
Use identical B/C information for no-displacement/geometric, rigid-alignment and simple displacement-interpolation baselines. Tune interpolation only on C.
Primary patient metrics: held-out 3D RMS error and paired improvement over each baseline. Also report median, maximum, empirical upper-tail error, included/excluded counts and spatial coverage.
Primary specimen metrics: force RMS/absolute error in N and torque RMS/absolute error in Nm, separately by loading mode and declared role; report every normalized-score denominator. In the [initial one-scale baseline](hbe-specimen-mechanics-poc.md), axial force is calibration and only torsion is held out.
Propagate documented annotation, frame, geometry and instrument uncertainties, including shared correlations. If only approximate ranges exist, report sensitivity bounds rather than calibrated confidence intervals.
Patient conditional-displacement acceptance requires numerical convergence and improvement over the strongest simple baseline that persists under the declared uncertainty analysis; predeclare any absolute application tolerance and tail noninferiority margin before V is read. The first specimen model is itself the simple mechanics baseline; report descriptive held-out agreement when metrology or an application tolerance is unavailable.
If uncertainty prevents discrimination, record inconclusive; if FEM loses to interpolation, preserve the negative result. One patient demonstrates conditional feasibility, not clinical or population validation.
Sparse observed-motion agreement cannot license retractor controls or force rewards. A controlled-retraction phantom precedent used tracked displacement boundaries and independent CT beads; publication does not establish our access to its raw measurements ([primary study](https://pmc.ncbi.nlm.nih.gov/articles/PMC4082653/)).

## Later decision-making boundary

Only after the relevant gate passes compare search, imitation and RL on the same observations, actions, frozen mechanics and budgets. Keep geometric planning as a baseline.
Deployment inputs may contain available images, measured instrument state and uncertain estimated mechanics. Hidden simulator material values, unobserved loads, withheld displacements and solver truth must not enter the policy.
Preserve every failure, exclusion and dependency; passing specimen or conditional-displacement validation does not mark the surgical action-response gate complete.
