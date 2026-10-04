# Optional learner decision observer

The optional `decision_observer` on `train_patient_policy` and `rollout_policy`
receives a `DecisionRecord` after the existing choice and before `simulator.step`.
The default is `None`. The callback is synchronous: recording/export failures
propagate and prevent that action from executing. There is no second forward,
random draw, simulator observation, or replacement training algorithm.

`MaskedPatientPolicy.forward` has an optional `capture_inputs` keyword for this
implementation. It copies the arrays and tensors already used in that same call,
including FEATURE_UNITS transformed actor features. The normal return remains
`(logits, value)`. The public decision callback receives detached evidence only;
it receives neither a policy nor simulator reference. Source feature copies keep
their original dtype/shape. `action_features` and flattened `state_features` are
the actual float32 inputs; `actor_action_features` is the actual transformed tensor.
The mask is the actual boolean policy mask. Arrays have immutable backing bytes;
serialization additionally checks dtype, shape, strides and content to reject
mutable NumPy metadata changes. Changing a callback's copy cannot change training.

`DecisionRecord.to_dict()` exports schema `learner-decision-observer-v1`:

- `role`, completed-update count `update`, learner `episode`, selection `panel`,
  `seed`, and zero-based `step`.
- `inputs`: source features, actual policy features, transformed actor features,
  mask and ordered action IDs. Array descriptors retain dtype, shape and values.
- Already computed masked `logits`, state `value`, chosen index and action ID.
  Masked negative infinity is the explicit string `"-inf"` for strict JSON.
- `decision_rule`, `forced_reason` and `forward_evaluated`.

Optimization records `sampled_categorical` unless the existing episode/transition
budget forces STOP. That forced optimization STOP retains the logits/value already
computed but does not draw an action. Selection records `deterministic_argmax`;
its last-step forced STOP skips the forward exactly as before. Such a selection
record contains source observation copies, with actual policy features, logits
and value null. The immediate STOP baseline has the same no-forward behavior,
labelled `stop_baseline`. No fabricated policy output fills these nulls.

Training supplies optimization/selection labels. Standalone rollout accepts a
`DecisionContext`; its default labels are unset and cannot establish an evaluation
partition. For training, `update` is the number of completed gradient updates,
optimization `episode` is its global zero-based counter, selection `episode` is
the index within a panel, and `panel` is the number of previously completed
selection panels. An incomplete panel is not assigned a score.

## Native accounting integration

`train_axis_policy(..., record_decisions=True)` declares recording before factories
run and passes the supported observer into the unchanged training loop. The
accounting wrapper binds source features/mask and RAW float32 policy inputs to the
observation already returned by reset/step. It does not request another observation.
Events of kind `decision` hold `decision_id`, accounting `instance`/`episode`,
role/seed, status and the serialized record under `payload`. The payload's
`episode` is the learner counter; the outer episode identifies the accounting
lifecycle. This distinction retains the zero-transition initialization reset
without pretending it is a completed optimization episode.

Each step requires a matching persisted decision, and its transition/failure event
carries the same `decision_id`. A successfully returned step marks the decision
`step_returned`; an authenticated commit followed by failure marks it
`executed_unreturned`. Failed or unverified execution remains explicit. A pending
record cannot be silently reset or replaced. Failed exports close the session;
in-memory receipts retain already committed work, as described in
[native accounting](native-axis-accounting.md). Accounting never publishes an
eligible candidate, including after incomplete decisions or selection panels.

Policy/checkpoint identity is joined by the completed-update count to preserved
checkpoint tensors. The record does not claim a separately captured policy hash
or action probabilities. Any probabilities reconstructed later from saved logits
must be labelled derived. Logging, copies, hashing and export consume elapsed
wall time at their call sites; numerical parity tests use generous time limits
and do not claim identical measured timing or throughput.

Validation uses tiny synthetic fixtures. Independent paired training compares
actual environment traces, policy forwards, observation accesses, exact policy and
Adam tensors, private/global RNG, and selection/optimization results. Separate
checks exercise transformed inputs, mutable source buffers, forced STOP, strict
serialization, callback/export failures and committed native-transition joins.
No public execution or patient learning is part of this change.
