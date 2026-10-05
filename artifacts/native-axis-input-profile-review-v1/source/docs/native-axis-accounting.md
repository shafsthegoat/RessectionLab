# Experimental axis training accounting

`native_axis_accounting.py` is a separate scratch-training boundary for the
[axis adapter](native-axis-adapter-design.md). It addresses the executed-transition
loss identified in the [learner review](native-axis-learner-review.md). It retains
the shared learning algorithm, native engine, fixed-route experiments, preflight,
and procedural profile gates. No public training is authorized by this slice.

## API and required receipts

Construct `AxisTrainingAccounting(factory, optimization_manifest,
selection_manifest, receipt_path=..., expected_model_hash=..., input_profile="RAW")`, then call
`train_axis_policy(accounting, config=..., output_dir=...)`. The helper uses the
unchanged generic learner with `accounting.recorded_factory`, scratch
initialization and the accounting session's declared profile. RAW remains the
default; version 2 also accepts the existing registered FEATURE_UNITS transform
through the separate [axis input-profile contract](native-axis-input-profiles.md).
The raw simulator observations and model identity remain unchanged. Resume and
transfer options are deliberately absent. Only the exact
`AxisColumnNativeSimulator` is accepted; existing exact-class procedural and
population contracts remain closed to this backend and wrapper.
Optional `record_decisions=True` uses the supported
[decision observer](learner-decision-observer.md) to join exact pre-action inputs
and already computed policy outputs to these transition receipts. Version 2
separately binds `observation_encoding: RAW` and the actual policy profile/hash;
it checks captured transformed actor features against the served observation
and immutable divisors. State/critic inputs are unchanged.

Consumers must read **both** the generic learner directory (`contract.json`,
`result.json` when present, checkpoints and `failures.jsonl`) **and** the separate
accounting receipt. The common learner's counters are retained exactly. Its
exception receipt alone does not include a native transition that committed before
preparation of the next observation failed. The accounting receipt supplies those
executed-transition counts and histories. Neither receipt supplies independent
geometry evaluation, clinical probabilities or candidate eligibility. The
accounting file always states `candidate_eligible: false`.

Constructor observations have no experimental role. Only explicit resets using
validated, disjoint optimization/selection seeds bind episodes. Source, sampler
and frozen decision-model identities must agree. Resetting records a new episode;
previous histories remain in the journal. A failed reset does not overwrite the
status of a previously completed episode. Factory reuse and valid transitions
performed outside the wrapper are rejected.

## Transition meaning and interruption

Each role separately reports executed transitions, steps returned to the learner,
native commits, committed-but-unreturned transitions and executed reward. An
explicit successful STOP counts as one transition and zero native commits. Native
cuts count only after authentication against sealed adapter history, native
history ancestry, cavity integrity, exact requested action, reward delta and
removed-cell accounting. Authentication runs no additional geometry previews.
An exception's `committed` flag alone is never sufficient evidence.
If a real authenticated cut is followed by false reported reward or history,
the receipt keeps the actual cut and reward, records the protocol mismatch, and
fails the run. Incorrect reported metadata cannot erase already executed work.

The wrapper re-raises the original `CommittedTransitionInterrupted` object after
recording its authenticated cut. It synthesizes no next observation, performs no
rollback, and cannot continue that session. Precommit cancellation counts no
transition. An unauthenticated state change sets `counts_complete: false` globally
and for the affected role; numerical totals then mean verified lower bounds. Raw
before/after state and the verification failure remain inspectable.

Every complete JSON export uses a temporary file, flushed file contents and atomic
replacement. If exporting fails after a real cut, the cut remains committed. The
session closes with `AccountingReceiptError`; its `receipt` and
`axis_accounting_receipt` attributes, plus `accounting.snapshot()`, retain the
in-memory evidence. If an original transition exception already exists, that same
exception is re-raised with `axis_accounting_receipt` and an export-failure note.
The last durable file can predate this failure and must not be treated as proof of
completion. A generic learner result produced before a failed final accounting
export also cannot stand alone as a complete combined run.

## Cost and validation scope

Accounting adds history copies, counts, JSON serialization/file flushing and one
full provider integrity scan after each attempted step. Provider counters include
that scan; it does not generate an inventory or independently recertify geometry.
Live-instance reuse checks use weak references, allowing completed simulators and
their volume copies to be released. Receipt histories still grow with executed
work and must be included in later bounded resource measurements.
The generic learner's measured clock includes wrapper work at the same call sites
as simulator work. Constructor/setup remains in the learner's separately defined
initialization scope. No overhead estimate, public-case throughput, or latency
claim is inferred from tiny tests.

The original version-1 checks covered a real tiny RAW gradient update, successful cut/STOP counts,
actual postcommit and precommit cancellation, the initial-selection failure before
any checkpoint, export failure, and then-closed non-RAW gates. Those frozen
receipts and historical RAW runs retain their original scope. Version-2 profile
checks add paired tiny RAW/FEATURE_UNITS updates, actual same-forward array
binding, unchanged physical transition results, and before-optimizer rejection
of profile mismatches and unsupported transfer. Independent adversarial review
supplements these checks. All new prerequisite fixtures are synthetic; this
slice performs no patient gradients and changes no source arrays, action
inventories or rewards.
