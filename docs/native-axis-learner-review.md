# RAW learner review of the native axis adapter

Reviewed the separately committed adapter `072dd5e` and the existing masked
REINFORCE learner. This review changes neither implementation. Tiny analytic
checks exercise integration; no patient training or patient timing is performed.
The separate public preflight, including its untrained seed-11 policy panel,
must complete before an experimental training budget is chosen.

## Action representation and identity

The adapter emits an observation with **K actual certified actions**, bounded
by `max_primary_rays + 1`; it does not pad to that cap. STOP is always row zero.
Every emitted mask entry is true because rejected or omitted proposals are kept
in the ledger rather than exposed as actions. The shared candidate scorer uses
each row's 15 action features plus the six repeated state features. Its output
size follows K, and REINFORCE retains each transition's own distribution rather
than stacking incompatible action arrays. A focused test also verifies that
optional masked padding cannot acquire probability or change legal scores.

Action IDs bind the adapter model, proposal, cavity and endpoint phase. After a
cut the IDs can change, and an old nonSTOP ID must be rejected. The learner uses
the current observation's integer row for its immediate step and records that
row's current ID for replay; it does not treat an index as a persistent spatial
action. Permuting rows permutes logits, and exact cloned histories reproduce the
new IDs. Greedy ties remain sensitive to provider ordering, which is explicitly
frozen and shared with search. A horizon-terminated episode may end on its final
cut without an additional STOP transition; retain actual transition counts.

The default provider has 26 primary slots and at most 52 native preview attempts,
with at most one accepted primary/fallback endpoint per slot. STOP makes the cap
27. The constructor refuses primary caps above the engine's 128-certificate
capacity. A fallback is eligible only after rejection of its paired primary,
never because the primary has low reward.

## What is learnable, and a concrete alias

The RAW generic learner completes real actor updates on a small dynamic native
fixture. This establishes callable gradient integration, not improved patient
planning. FEATURE_UNITS and procedural transfer remain closed to this distinct
adapter through their exact-class validation; changing those gates requires a
separate declaration and compatibility review.

There is a specific representation limit even before patient timing. On the
7×7×8 analytic fixture, certified fine-tool cuts from entries `(3,3,-.5)` and
`(4,3,-.5)` have identical 15-feature rows and the same immediate reward 4.33.
Their exact best two-cut nominal returns are **39.43 and 39.68**. The same state
features are repeated for both, so this scorer necessarily assigns equal logits
to them for every parameter setting. Physical entry/target coordinates, cavity
layout and inter-candidate spatial relationships are absent from its input.
The stochastic policy can sample either, but cannot learn a different relative
probability for this pair. This example does not prove that the fixture's best
overall sequence is unrepresentable, nor establish a patient bottleneck.

The existing critic also retains the six fraction/availability features, so the
previously measured cross-domain state alias remains. More actions alone do not
add information to either network. The expanded physical proposal model and any
future feature changes must be analyzed separately.

## Frozen source, reward and interruption boundaries

The adapter identity includes its version, native anatomy/tool/access hash,
provider and rule hashes, ordering/fallback policy, cap, horizon, physical reward,
partial-contact semantics, feature names and world configuration. The common
learner binds the actual source, sampler, model and optimization/selection
partitions on every episode. A focused test rejects a factory that keeps the
same source but substitutes another proposal cap before any completed update.
Missing motor/language evidence remains missing, and actual contained native
removal stays distinct from retained partial contact.

**Resumable patient training is not ready.** If cancellation arrives after a
native commit while its next inventory is being prepared, the adapter raises
`CommittedTransitionInterrupted` with the actual reward and committed history.
The unchanged generic learner catches it only as a failure: its transition
counter is incremented after `step()` returns, and its generic failure receipt
does not serialize the exception's committed history/reward. This is honest
failure, but incomplete accounting and no resumable state for that interruption.
The analytic regression preserves this limitation explicitly. A dedicated
experimental runner must retain raw committed state and executed-transition
counts; any resume claim additionally needs a tested continuation contract.

## Timing and the next declaration gate

Network inference over at most 27 rows is small, while constructor/reset,
complete previews, whole-grid integrity scans, clones, next-state inventories
and metrics reads all incur physical-model work. The cooperative learner checks
its clock around these operations; it cannot interrupt an atomic preview or
scan. An initial selection panel can exhaust the allowance before any gradient
is possible. The prior 30-second fixed-candidate budget must not be inherited.

The public preflight should report initial complete inventory cost, every cut's
next-inventory cost, complete greedy and untrained policy panels, failed/omitted
slots, peak memory and cancellation latency. Its reused-instance reset/panel
timing is not automatically a generic-learner timing: the common learner also
constructs an integrity-checked clone/factory instance before each episode.
That cost must be measured separately or the panel estimate labeled accordingly.

Only after the preflight is complete, declare a bounded RAW-only development
comparison with the same axis model and physical actions for STOP, untrained
seeded policy, greedy/search and scratch RL. Preserve the complete selection
panel, source snapshot, every failed attempt, actual transition/update counts,
setup/clone cost and atomic overshoot. A wall allowance and update/episode cap
must be based on complete measured operations, with enough explicit allowance
for initial selection and at least one whole optimization batch; no numeric
budget is chosen in this review. Keep independent native geometry checks after
candidate freeze and leave final/stress worlds unopened. No new spatial
features, input-profile support or population arms are implied by this review.
