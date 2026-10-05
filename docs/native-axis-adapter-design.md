# Separate native axis-column adapter

Status: the isolated adapter is implemented in `native_axis_simulation.py` after
the [standalone proposer's](experimental-native-proposals.md) repair tests and
the preceding registered study completed. Tiny analytic and independent
adversarial tests exercise the adapter; patient training and desktop integration
remain unstarted. This document records one integration slice, not a replacement
project plan.

The separately versioned, RAW-only `AxisColumnNativeSimulator` uses the
existing native engine and contained-cell reward/observation semantics. Preserve
`NativeSequentialSimulator`, its fixed-route model, shared learners and existing
study contracts. The new adapter needs its own constructor; passing dummy target
points through the fixed-route constructor would falsely describe its derivation
and model identity. Its action and state features may retain the existing 15+6
ordering, but its action-model version and experiment declaration must be new.

## Shared inventory and bounded checking

Both SEARCH and scratch RL must consume the same deterministic, ordered inventory
from `PreparedAxisColumnProposer.propose(engine)`. Preview every emitted primary
with the native engine. Try the paired fallback only after primary rejection;
never substitute it because a feasible primary has a low reward. Hidden-world
evidence and comparator identity must not influence generation, ordering or
feasibility. Only accepted, state-bound native previews become non-STOP actions.

Use `max_actions = max_primary_rays + 1`, including STOP, so the old eight-action
limit cannot silently truncate this different inventory. Require
`max_primary_rays <= 128`: the current engine retains 128 feasible preview
certificates, and at most one endpoint per primary/fallback pair can be accepted.
The default 26 primary rays require at most 52 preview attempts. Changing engine
certificate retention requires reviewing this bound.

Keep the complete provider slot ledger, including omissions, plus each primary
and fallback attempt, rejection reason, selected phase/endpoint and preparation
cost. Distinguish proposed rays, actual geometry checks and certified actions.
Hash the adapter model identity, provider proposal ID and endpoint phase into a
bounded action ID; retain full provenance separately. IDs must reproduce across
equivalent resets/clones and become invalid after their cavity changes.

## State integrity and lifecycle

Cached actions must still regenerate or validate the current batch before reuse.
An unchanged ancestry string alone cannot detect direct edits to live masks or
history. Reuse geometry certificates only for an identical complete batch on the
corresponding engine lineage; proposal validation itself grants no clearance.

Implement `reset`, `fresh` and `clone` explicitly. The existing initial-geometry
cache lacks dynamic batch/ledger state, and inherited `fresh()` rejects
subclasses. Reset must clear episode accounting and bind the reset cavity;
clone must isolate mutable ledgers and masks while preserving valid engine-local
preview lineage. Fresh instances rebuild their own engine and preview state.
Share only immutable source preparation, and record setup costs for both methods.

Check cancellation before/after integrity preparation, around every primary or
fallback preview, and before commit. Publish a complete inventory atomically;
an interruption must not expose a partial action set or manufacture STOP-only
termination. Current integrity scans and native previews have no internal
cancellation hook, so responsiveness is limited by one such operation. Existing
search checks wall limits after proposal materialization; any new deadline
handling must abort explicitly rather than truncate one comparator's inventory.

Freeze and hash the adapter version, native fingerprint, proposer/rule hashes,
reward/world configuration, feature schema, caps, ordering, fallback policy and
partial-contact semantics. Export the new adapter version in metrics. Preserve
source arrays, genuine removed-cell accounting, and the distinction between
partial contact and removal.

## Compatibility gates and first verification

`FEATURE_UNITS`, procedural observation schemas and transfer validation currently
require the exact `NativeSequentialSimulator` class. Preserve those gates: this
slice supports RAW scratch RL only. Broader acceptance needs a separately
declared and tested schema change; existing population checkpoints cannot be
silently relabeled. New headless modules also change future numerical source-hash
receipts, although the current registered worker uses its own frozen archive.

Keep this adapter outside the desktop and exact selected-route refinement: its
neighboring columns deliberately broaden the action model. Use a separate
experimental factory and output namespace.

Before any new training, retain checks for SEARCH/RL inventory equality, strict fallback
eligibility, action/certificate caps, stale-ID and tampered-cavity rejection,
clone isolation, reset/fresh replay, cancellation without partial publication,
frozen model identity, and independent native replay of executed histories.
Measure integrity and preview costs separately; no performance claim follows
from the implementation or tiny-fixture checks.

## Experimental API and current limits

Construct `AxisColumnNativeSimulator(native_config, proposal_config=rule, ...)`
with an explicit native configuration and optional nominal evidence, frozen
reward/world configuration, horizon and cancellation callback. It exposes the
existing observation/step protocol and native search methods can consume its
clones. `fresh()` preserves missing evidence and the dynamic backend; changing
the horizon or world configuration creates a different model identity. No
factory silently broadens a selected patient route.

`metrics()` exports complete inventory receipts with the original provider
ledger, attempted primary/fallback previews, accepted action IDs and interrupted
status. Proposal accounting separates integrity calls/time from preview
calls/time. Metrics reads themselves verify the cavity and therefore increment
integrity accounting. Timings are diagnostics, not action identifiers or rewards.
One observation validates its inventory once and reuses that tuple during the
three synchronous inherited feature passes; subsequent public accesses validate
again. Adapter history, partial-contact accounting, derived volume arrays and
the sampled episode world are also guarded against mutation.

An interrupted reset leaves the episode unavailable until an explicit successful
reset. Observation, action, clone, fresh and metrics access cannot continue with
partially initialized guards. The transient feature-pass inventory is usable
only inside its guarded synchronous observation read; injecting it outside that
context or executing a transition during the read is rejected.

Cancellation before commit leaves tissue unchanged and publishes no incomplete
inventory. If cancellation arrives after a native commit while the next
observation is being prepared, `CommittedTransitionInterrupted` explicitly
carries the committed transition and reward. The history and removal remain
available; rollback is never implied. The unchanged generic learner treats this
exception as a failed run, so resumable cancellation needs a separately reviewed
experimental runner before patient training.

Before declaring any patient budget, measure a complete initial inventory,
complete episode and complete selection panel under this new model. Do not
inherit the earlier 30-second study window or claim policy updates when its
initial selection/episode cannot finish. The separate provider observation
already found material whole-grid integrity cost; there is no patient-adapter
throughput result yet.
