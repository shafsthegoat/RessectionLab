# Real-patient learning helpers

The current runnable development experiment is
`scripts/run_real_patient_learning.py`, following
`CORE_IDEA_VALIDATION_STEERING.md`. It uses only TRAIN PAT05 and the existing
annotation-assisted 64³ spatial actor, unchanged geometric objective and at most
three decisions including STOP. Two REINFORCE updates use two complete episodes
each; initial and fixed-latest policies are compared with three random episodes
and greedy search through the same horizon. Initial/latest readouts are on the
training anatomy, not an independent patient or checkpoint-selection set.

The actor ranks certified geometric proposals. Generating those proposals
already requires native tool previews; this is not direct path generation from
a scan without geometric computation. Report that shared preparation and each
successor-inventory cost as well as network latency. Greedy search scores all
current legal certificates against the permitted annotation and frozen objective;
it can miss a valuable route requiring a negative opening action. No method is
claimed globally optimal. Search uses the full nominal field while the actor uses
the fixed crop; all PAT05 target annotations fit that crop, but representation
and computational access are not identical.

The binary positive-target threshold is a reachability check. In the initial
PAT05 inventory, 52/70 non-STOP actions remove some target and only 26/70 have
positive geometric reward. Continuous target removal, other-tissue removal,
contact bounds, path length and runtime therefore carry the quality comparison.
The reward weights are research assumptions, not calibrated clinical injury
costs. Functional evidence is unavailable in this task. Independent complete-tool
replay and separately recomputed reward/accounting are required for every accepted
episode; neither establishes cutting or retraction fidelity.

The next controlled diagnostic keeps the BC8 weights and Adam state fixed at
initialization in both branches. Each receives eight additional updates and six
examples per update: the original three teacher states repeated twice, or the
original three plus three states visited by the frozen learned policy. The common
initial state appears twice in the augmented data; six examples therefore contain
at most five unique states. Additional labels use only the permitted nominal
objective at each live pre-action state, and collection cannot change the
behavior policy or its actual simulated trajectory. Terminal states are excluded.

This is a small application of the dataset-aggregation idea in
[Ross, Gordon and Bagnell (2011)](https://proceedings.mlr.press/v15/ross11a.html):
request expert labels at states reached by the learner to address the shift from
expert trajectories. Here the labeler is a geometric greedy planner, not a
surgeon; one bounded iteration does not inherit a clinical or generalization
guarantee. The matched control separates added state coverage from merely
performing more optimizer updates. It remains imitation learning, not RL.

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
