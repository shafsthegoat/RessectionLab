# Frozen-representation action-value component

The new additive `spatial_continuation_value.py` captures the existing IL actor's
public input features in one frozen forward. It fits a separate copy of the actor
head; the original encoder, actor, STOP head and critic remain unchanged. No
existing planner, patient factory or reward is modified.

The target forces the recorded action, then permits stopping after its best
recorded continuation: `q[i] = reward[i] + max(0, q[i+1])`. A negative mandatory
action therefore retains its cost; STOP is a distinct zero-valued action. Labels
are behavior-specific nominal simulation returns, not optimal Q-values or
observed surgical outcomes. Complete admitted traces and source/observation
bindings are required; saved scalar files alone cannot authorize fitting.

The fixed intended experiment includes all 36 existing traces / 497 states from
the four TRAIN subjects, equal subject/trace/state weights, and 64 head-only Adam
updates at 0.001 with gradient norm cap 5. The corpus contains no observed
negative-cost preparation followed by a positive continuation. Deployment in
this comparison will rank only independently scored positive-immediate-reward
candidates and preserve the full-inventory greedy fallback and STOP rule. It
tests productive continuation ranking; the nominal guard supplies avoidance of
negative actions. No opening-credit, optimal-policy or clinical claim is supported.

Root verification passes **18 generated tests in 1.66 s**. The first batch had
17 passes and one test-fixture refusal: an arbitrary action mask violated the
existing observation contract. The repaired test explicitly expects that refusal
and uses a valid terminal-budget mask for its positive check. Production code
and the observation contract were unchanged; original logs remain. Independent
source review found no material blocker.

The [index](source-index.json) binds source and checks. No existing patient
checkpoint, acquired image, real-case feature extraction or experiment fitting
has run for this component. The next milestone is the fixed four-patient
comparison, including all reconstruction/training/inference costs and independent
route evaluation. A lower fitting loss will not establish a useful improvement.
