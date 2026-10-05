# Explicit input-profile compatibility for axis scratch training

This is a tested integration prerequisite, not a public-case experiment or a
claim that fixed units improve planning. The existing masked REINFORCE learner
can use the registered RAW or FEATURE_UNITS actor transform with the exact
`AxisColumnNativeSimulator`, including its exact accounting wrapper. Other
subclasses and lookalike wrappers do not gain FEATURE_UNITS compatibility.
The existing fixed-native, procedural-transfer, population and PPO interfaces
are unchanged. The axis branch permits fresh scratch training only; it does not
authorize transfer or resume after interrupted native commits.

`AxisTrainingAccounting(..., input_profile="FEATURE_UNITS")` declares the
profile before any factory or optimizer exists. `train_axis_policy` uses that
declaration; its default remains RAW. A common-learner call requesting a
different profile fails before Adam. The declaration is checked against the
immutable registry, so resealing a different divisor list does not authorize it.
Checkpoints retain the existing profile-aware load and buffer checks.

## Observation and policy identities

The adapter, inspector, proposal engine, physics, reward, source data and model
hash are unchanged. Their existing `input_profile: RAW` denotes the raw feature
stream under this new compatibility contract; it is not the learned policy's
transform. The new learner contract explicitly records the raw axis schema,
physical model identity and this interpretation. Accounting version 2 records
`observation_encoding: RAW` separately from the actual policy profile, its
registry manifest and hash. Historical receipts and frozen source archives
are unchanged and retain their original version and RAW-only interpretation.

The axis schema binds an already served observation: no extra observation,
preview, episode or policy forward is requested. Actions have the existing 15
features and six state features, with STOP first and only actual certified rows.
The names alone are insufficient to establish semantics: the exact backend,
source hashes and implementation bytes remain part of the training contract.
In particular, `depth` is the fraction of the action budget used; the adjacent
target feature is supplied-label occupancy in the clipped endpoint neighborhood,
not remaining-target occupancy. Partial-contact features count newly encountered
retained contact, excluding previously charged contact and cells removed now.

FEATURE_UNITS divides the action vector by the unchanged registry vector
`[1,648,648,648,648,120,2.25,1.10,1,1,1,1,648,648,648]`. The 648 mm³ reference
comes from procedural support; it is not a universal bound, a fitted statistic,
or a clipping threshold. There is no centering, clipping, return transform,
critic transform or new coordinate. All six state inputs remain unchanged.

The supported observer still captures the actual transformed actor arrays
inside the existing policy forward. Accounting independently binds those arrays
to its served raw observation and declared float32 divisors, and binds the
chosen current action ID to the transition. It adds no inference or random draw.
Forced STOP records without a forward retain null policy arrays. One profile
cannot be silently logged as another.

## Verification scope and next gate

Focused checks use only a 7×7×8 synthetic fixture. They cover paired initial
trainable tensors with distinct behavioral hashes, unchanged critic outputs,
variable action counts, row permutations and masked padding, stale-ID rejection,
identical prescribed physical transitions under both profiles, before-optimizer
profile/transfer rejection, and one actual update per profile with same-forward
decision receipts. These updates establish callable integration, not a learning
benefit. Independent review and root-owned integrated regression remain separate
gates before any new public execution declaration.

The separately proposed one-update paired public comparison is not executed or
authorized by this code. Its full initial and updated selection costs, cold
setup, accounting and independent validation still require explicit budgets
and a frozen source. Scaling preserves exact feature aliases and does not make
the six-field critic or action representation Markov complete.
