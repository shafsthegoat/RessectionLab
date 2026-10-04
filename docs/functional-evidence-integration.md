# Explicit population-prior sensitivity mode

`FunctionalEvidence` is an immutable, source-bound planning input. It carries
motor/language arrays, their separate atlas sampling coverage, physical RAS+
affine, original source/transform records, review status and frozen coherent
uncertainty. Its first version supports population-prior sensitivity only.
Absent evidence continues to select the existing structural-only path.

`population_prior_sensitivity(case, uncertainty=...)` explicitly copies the
already imported functional concordance proposals. The three language component
fields use a declared maximum aggregation and intersection of sampling coverage.
All original functional and structural proposals remain separately inspectable,
view-only and unreviewed. No import, engineering test or save operation confers
expert approval. Atlas support does not establish patient function, even where
the released map is zero. Patient tracts and vessels remain unassessed.

The selected evidence changes the case and planning identities. Historical
registration proposals retain a separate identity of their unchanged parent
inputs; changing the source image, labels, frame or provenance still invalidates
them. Portable cases preserve arrays and metadata without pickle. Reopening
checks their complete manifests. Existing cases without this field retain their
prior semantic identities.

Native planning receives the nominal fields, coverage and world generator.
Unknown atlas samples receive an explicit unit-upper-bound surrogate for
optimization, while raw maps and missingness remain available to independent
evaluation. This conservative research cost is not imputed patient function.
Sampled motor/language maps share one episode-fixed coherent transform. Unknown
coverage moves with that transform and is retained in removal reporting.

Native inspect/train/recheck accepts explicit `world_generator`,
`hard_exclusion` and `hard_exclusion_provenance`. Evidence, uncertainty and
constraint identities persist through the request, decision model, checkpoint,
route binding and replay. Changing them rejects resume or reopening. Known
exclusion masks reach the complete-tool engine and independent checker.
No omitted vascular mask establishes vascular clearance.

The desktop bridge automatically forwards the selected case evidence and its
frozen generator through native training, disk replay and candidate export.
Expanded-axis inspection preserves the same fields and coverage through fresh
simulators. Raw per-call functional or world overrides are rejected: a changed
scenario requires an explicit new source-bound evidence version. This integration
adds no vascular source and no new interface control. Six independent bridge
controls include actual updates, clearing the in-memory report, disk checkpoint
rechecking, replay and export with matching evidence/world identities.

The separate `functional_events.py` evaluator reports full-tool, whole-sequence
events using raw evidence, preserving the distinction between a known atlas hit,
unknown coverage, and missing patient functional assessment. Map integrals are
surrogate costs; only explicit scenario-event counts have Monte Carlo intervals.
Clinical probabilities remain null.

The local bridge operation `evaluateCandidate` accepts `caseHash` and `runId`
after a completed native checkpoint selection. It seals that exact full-tool
history, source evidence, footprint and event definitions before revealing the
run's originally declared final worlds. The current native facade has **three**
final worlds; this small conditional sensitivity panel does not establish robust
clinical accuracy. Optimization and selection worlds remain excluded.

The immutable seal and a shared ledger are saved before event computation.
Cancellation retains this state: repeating evaluation resumes the same frozen
assessment, while optimizer resume is blocked. No replacement worlds are chosen
automatically. A different candidate cannot reuse revealed worlds from another
run in the same local run directory. Headless callers should keep related runs
under one parent directory to retain this ledger scope.

Replay and export include the complete-sequence report, counts, denominators,
coverage unknowns, conditional intervals and surrogate mean/tail costs. These
events describe the whole frozen sequence even when the displayed removal mask
shows an earlier replay step. Reopening independently recomputes the same events;
changing reported counts cannot pass merely by recomputing a content hash.
The bridge signs the seal and report alongside its existing run artifacts.
Structural-only runs remain unsealed and retain their previous resume behavior.

The local preparation helper `scripts/prepare_functional_sensitivity.py` creates
a new bundle and receipt from existing proposals with explicit sensitivity
scales. It performs no new registration, acquisition, expert review or training.
Uncertainty values are declared research scenarios, not measured patient errors.

Validation includes independent actual native gradient updates on an analytical
software fixture, checkpoint/resume/replay, changed-world rejection, a known
obstacle that blocks the same complete-tool route, raw unknown versus known zero,
physical-frame changes and source-bound case round trips. These controls are not
population training data. The first regression exposed a tuple/list mismatch
when uncertainty was saved to JSON; canonical JSON records corrected the failure.
