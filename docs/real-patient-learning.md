# Real-patient learning helpers

These are reusable planning and bookkeeping components, not a trained model or
an efficacy result. They open no images, run no optimizer, and do not alter
geometry, observations, reward, or patient roles. The first population runner
still requires a frozen declaration and a valid common decision problem.

`observed_search.observed_beam_search(task, max_calls=..., beam_width=...,
seconds=..., policy=None, objective_source=...)` uses the task's observed-only
planning clone and its complete legal inventory. It returns a sequence and
accounting. All evaluated prefixes have the existing zero-increment STOP
baseline. Negative opening prefixes remain eligible; beam pruning and incomplete
layers are reported separately. Call-cap STOP results do not establish that STOP
is optimal. Time exhaustion raises `ObservedSearchLimit` with partial accounting
and the best prefix, preserving failure rather than manufacturing a result.
Attempted model transitions are charged even when they fail; the implicit STOP
count includes only the initial state and successfully evaluated finite prefixes.
Next-layer native states are retained incrementally by the exact existing
`(-return, action-ID path)` ranking, with at most `beam_width + 1` during insertion.
Every legal transition is still evaluated. Logical counts of discarded negative
prefixes are preserved independently of which state objects remain in memory.

`transition_mode="eager"` remains the default. The explicit `"lazy_planning"`
option requires a nominal-only planning clone with `advance_planning(action)`;
unsupported tasks fail rather than silently falling back. This commits the same
certified transition but defers its successor inventory until that state is
expanded. Attempted eager/lazy transitions and explicit observation requests are
counted separately. Observation requests are not inventory-build or preview
counts: task setup, caching, and eager transitions can perform additional work.
Use the native preview profiler and the full wall/RSS budget for cost comparisons.

Optional guidance orders all legal expansions by actor logits. It does not use
the critic, alter physical reward, or discard candidates using a policy top-k.
Every attempted actor forward is counted, including failures, and runs without
gradient recording. Exceptional search exits retain partial accounting. Before/after
state-dictionary hashes check parameters and registered buffers, including timeout
and other exceptional exits; mutation raises `ObservedPolicyChanged`. Check costs
are included in total planning time. Ordinary Python counters are outside this
state check. A BC-trained policy's critic is untrained and cannot justify value
guidance.

Shared inventory is not identical information representation. Native search may
read the full permitted nominal target while the actor receives a cropped image.
Report crop/path/target coverage and this limitation before comparing quality or
latency. Private reference labels remain outside planning clones and proposals.

`real_patient_learning.read_development_cohort(path)` verifies the original BTC
manifest bytes. `require_development_role(cohort, subject, role=...)` checks the
six TRAIN and two SELECT canonical identities before a caller loads a case.
PAT29/31 and external final patients are not admitted. SELECT is not a source of
population gradients. Actual bundle, source, support and model provenance checks
remain the real runner's responsibility.

`patient_uniform_bc_indices(counts, batch_size=..., generator=...)` draws a TRAIN
patient uniformly, then a teacher state within that patient. All six groups must
have nonempty histories; STOP-only histories remain eligible. Missing or invalid
cases cannot be silently dropped or replaced by fabricated STOP samples.
`training_patient_schedule(rounds=..., generator=...)` shuffles all six once per
round. Input failures must remain in experiment denominators; the runner must
report actual contributing patients rather than label partial training as six-case
pretraining.

`validate_on_policy_batch(episodes, parameter_hash=...)` accepts complete,
finite-reward TRAIN episodes with one recorded behavior hash and reports per-case
contributions. The rollout caller must measure unchanged weights before/after;
this helper validates the supplied records. Pass the complete trajectories to
the existing `reinforce_loss`, whose actor term sums within an episode and then
averages episodes. No per-step normalization, hidden length weighting, stale
off-policy reuse, optimizer changes, or new architecture are introduced here.

The legacy screen wrapper delegates to this search helper and preserves its
`ScreenLimit` timeout interface. That deferred procedural runner is not required
by the standalone helper tests and is not a real-patient training path.
