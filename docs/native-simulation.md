# Native source-cell sequential simulation

The native backend is a distinct model from the earlier coarse-cell pilot. A
coarse cell touched by a small tip cannot be reported as fully removed native
tissue. The coarse pilot remains useful for bounded algorithm experiments, and
its failed native-footprint checks remain part of the evidence record.

`native_resection.py` executes a declared geometric aspiration primitive on the
source image grid. `native_simulation.py` supplies the same reset, observation,
action, replay, and frozen-world interface used by patient-specific learning.
Both are research models; neither predicts postoperative function or establishes
clinical safety.

## Geometric contract

Each macro-action specifies an instrument, entry point within the hypothetical
access disk, and terminal tip position. The rigid tool inserts on one straight
line, then retracts on that line before reorientation. No corridor is cleared at
initialization.

The insertion is divided into small physical steps. At each step:

1. The active distal capsule may contact exposed tissue.
2. A source cell is removed only when all eight physical corners are contained
   by the active capsule sweep and the cell connects to exterior/cavity through
   the eligible removal patch.
3. Partially contacted cells remain occupied. Their exposure is recorded
   separately and receives no target-removal credit.
4. The complete shaft must clear the remaining tissue. Hard exclusions and the
   access aperture remain constraints on the complete tool.

The cell-containment rule makes removal conservative at the chosen image
resolution. A valid continuous shape can be rejected by the cell model. Those
failures are retained instead of enlarging the cavity to admit the shaft.

Preview is transactional: an invalid stroke changes no cavity. A successful
preview has a state-bound certificate. Stale, altered, foreign, or deserialized
previews cannot be committed directly. Saved histories require replay and
independent checking.

## Observation and objective

The actor receives nominal anatomy/evidence and candidate features. A coherent
functional-registration world is sampled once per episode. Hidden anatomy may
change surrogate reward but never the actor's features, proposals, or mask.
Optimization, selection, and final evaluation use the separate world manifests
specified by `EXPERIMENT_PROTOCOL.md`.

Native candidate features add partial normal, motor, and language contact to
the coarse adapter's twelve features. STOP is always action zero. Target and
normal removal are counted separately on source cells. Partial contacts receive
a separately declared surrogate weight, charged once per cell in that exposure
category; they do not become removal or clinical injury probabilities.

Missing functional evidence produces null motor/language result fields and
explicit unavailable flags. Structural-only optimization does not establish
motor/language preservation. Unknown vascular anatomy, tissue forces, microscopic
infiltration, and postoperative function remain unassessed.

The native generic instruments have new IDs and explicit dimensions. They are
not verified commercial instruments and are not silent replacements for the
coarse pilot's tool catalog. Tool, world, objective, action proposals, and native
engine fingerprints are frozen during optimization.

## Source and frame gates

The factory preserves source cells and converts physical geometry to RAS+ when
an input case is declared LPS+. It does not resample anatomy to perform that
conversion. Explicit access coordinates declare their own RAS+/LPS+ convention.

A supplied brain mask cannot be expanded silently to include conflicting target
annotations. Full-head or unknown images need a reviewed brain mask. The UCSF
structural case may use its declared skull-stripped MRI support as an explicitly
estimated, unreviewed intracranial envelope. An explicit `skull_stripped: false`
overrides that collection-level assumption. BTC full-head images cannot use
nonzero intensity as a brain mask.

## Local API

- `make_native_patient_simulator(case, ...)` prepares source-grid proposals.
- `NativeSequentialSimulator.reset(seed)` samples the fixed episode world.
- `observation()`, `proposed_actions()`, and `step(action)` implement the shared
  masked-policy contract.
- `clone()` preserves the current rollout and its valid proposal certificates.
- `fresh(max_steps=None, world_generator=None)` creates an isolated arm with
  independent dynamic caches, preserving backend type and evidence availability.
  Explicit changes create a new decision-model hash.
- `native_beam_search` and `native_greedy_search` rank the same proposals using
  nominal evidence and the declared partial-contact costs.
- `independent_check_native_history` separately checks source identity, physical
  containment, connected removal, swept shaft, hard exclusions, aperture, and
  partial-contact records.

Cancellation is checked during preparation and between proposal previews. Native
preparation raises `InterruptedError` on cancellation; desktop callers should
show a cancelled job rather than a clinical or geometry failure.

## Development evidence and limits

The saved UCSF-PDGM-0004 development probe is
`artifacts/native_simulation/ucsf0004-initial-development-v2/result.json`.
It records two parallel entries in one hypothetical aperture, 249 mm³ of modeled
source-cell target removal, 17 mm³ of normal removal, and 32 mm³ of cumulative
partial normal contact. These quantities come from a restricted structural
simulation, not an observed operation. The pinned structural mirror is not
verified as byte-equivalent to the official TCIA package.

The separate `independent_audit.json` in that directory passed the source-grid
prefix audit: 266 mm³ of claimed tissue was fully contained, with zero
unsupported removal. Both the complete-tool and connected-frontier checks
passed across the two strokes. This checks the declared geometric primitive;
tissue mechanics and clinical safety remain unvalidated.

Initial oblique proposals failed shaft clearance. Adding an explicit paid
axis-aligned opening and separate aperture entry points supplied legal
alternatives; old failures were not converted to successful plans. Candidate
sampling remains small and deterministic, so the result is a best-found plan
within this action space, not a globally optimal operation.

A single transition profile on this source case took 0.839 seconds, of which
0.336 seconds were spent rehashing immutable array contents. Checking immutable
buffer identity, shape, dtype, strides, pointer, and all metadata reduced the
same profiled transition to 0.516 seconds. Mutation tests cover replacement and
reinterpretation. These are exploratory timings, not a cross-machine benchmark.

Exactly deterministic worlds now preserve the source fields directly. This
avoids unnecessary interpolation and prevents floating-point inverse-affine
error from falsely creating unknown boundary coverage on oblique images.

## Bounded initial-geometry cache

The native adapter retains one prepared initial engine snapshot per factory
instance. Episode resets clone its cavity state and its original in-memory
geometry certificates, then sample the episode's hidden world independently.
The cache contains no hidden costs and does not admit deserialized certificates.
The frozen-model guard runs before reuse. An isolated `fresh()` arm performs its
own cold preparation, so its setup cost remains visible to experiment budgets.

The UCSF development profile in
`artifacts/performance/native-reset-cache-v1/comparison.json` measured five
resets before and after this change. Median profiled reset time fell from
0.462 seconds to 0.0259 seconds (17.85×). Cold preparation stayed near 2.5 seconds;
later cavity proposals and history-copy costs remain. All captured observations,
rewards, removal records, source hashes, and metrics were byte-identical. The
bounded snapshot raised peak process RSS from 1.168 GiB to 1.202 GiB on this
16 GiB Mac. This is one development profile with uncontrolled concurrent load,
not a general wall-time training benchmark or clinical result.
