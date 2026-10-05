# Completed feature-unit mechanism diagnostic

The fixed units removed measured actor saturation on the archived common
states. Critic state aliasing remains, and value gradients dominate the shared
clipping norm more strongly in the scaled model. The records do **not** establish
that clipping caused a learning failure: scaled actors actually moved farther in
parameter space, and all three scaled adapted runs retained an initial policy
already scoring 245.24 under this fixed candidate model.

Six static diagnostic tests passed. The analysis loaded 42 policy snapshots on
seven already archived observations. It performed **zero simulator resets,
transitions, backward passes, optimizer updates or searches**. All checkpoint
bytes and policy hashes remained unchanged. Final and stress worlds were not
opened. The original study and its independent geometry records were preserved.

## Actor input sensitivity

These are means over legal nonSTOP actions at each initial state, using the
actual profile transform and the final shared pretrained policy.

| Initial state | RAW fraction with absolute tanh ≥ .99 | FEATURE_UNITS fraction | RAW mean tanh derivative | FEATURE_UNITS mean derivative |
| --- | ---: | ---: | ---: | ---: |
| Procedural slab | .71875 | 0 | .04739 | .88902 |
| Procedural split lobes | .84375 | 0 | .04842 | .90592 |
| UCSF structural simulation | .671875 | 0 | .10764 | .88704 |

On the patient initial state, the RAW shared policy ranks `native-fine-aspiration:3`
first; FEATURE_UNITS ranks `native-wide-aspiration:0` first. Entropy remains high
in both: 1.51829 and 1.54542 nats respectively, with a five-action maximum of
1.60944. This is a ranking/conditioning difference, not a STOP-collapse finding.
The archived states do not describe each learned policy's visitation frequency,
and scaling also changes initial action probabilities; this does not isolate
saturation as the cause of any outcome difference.

## Critic representation and scale

Both procedural families and the patient have exactly the same initial critic
input `[1, 0, 0, 0, 0, 0]`. Every one of the 42 snapshots therefore predicts
exactly the same initial value across those three domains.

For the exact RAW shared checkpoint, the saved complete source panel return is
28.675 and the patient frozen panel return is 139.62, while its initial value
prediction is .81902 everywhere. For the exact FEATURE_UNITS shared checkpoint,
the source panel is also 28.675, the patient panel is 245.24, and the initial
value prediction is .77760 everywhere. These are complete deterministic greedy
panels for the same respective checkpoint; they are not the stochastic
per-transition training targets. The logs cannot reconstruct those targets.

This is a concrete representational alias across domains. During adaptation
within one fixed patient, a value head could still learn a patient-specific
intercept without new state fields. The latest adapted initial-state values
remain about 1.14–1.20 RAW and 1.18–1.19 FEATURE_UNITS. Thus the saved evidence
also leaves limited value-fitting time and physical target scale as possible
contributors, distinct from the cross-domain alias itself.

## Shared gradient clipping

The learner clips the combined actor/value gradient norm at 5. From the saved
total preclip norm and actor postclip norm, the diagnostic infers group norms
using the pinned Torch coefficient `min(1, 5 / (total + 1e-6))`. No gradients
were recomputed. The inference allows float32 rounding and refuses inconsistent
norms.

Across online runs, median inferred value contributions to squared global norm
span 18.86–50.00% for RAW and 93.85–98.47% for FEATURE_UNITS. For scaled adapted
seeds 11/23/47, the joint norm reduces actor gradients to approximately
.1233/.1256/.1982 of what the same threshold applied to the actor alone would
retain. This comparison is arithmetic context, not an executed optimizer arm.

However, actual initial-to-latest actor parameter displacement is
.435/.404/.458 for scaled adapted runs versus .350/.281/.260 for RAW. Adam's
moment normalization means scalar gradient shrinkage need not produce matching
parameter-update shrinkage. The diagnostic therefore identifies coupling but
does not show that removing it would improve learning. The three scaled adapted
initial policies already obtain 245.24 and remain the earliest selected ties.

## Binding and next decision

The study runtime is
`sha256:850dab806bca81be6d18f25cc39643e76d7cf1621d7cbad1f96669228fed65fa`.
`diagnostic.json` records every checkpoint and policy hash, the exact preserved
script/helper hashes, completed summary/declaration bindings, the old observation
receipt hash and the installed clipping-function source hash. A separate
`source-receipt.json` records the completed diagnostic and its validation.

A next study could separately test a prospectively available physical summary
for the critic, or separately test actor/value clipping groups while holding
representation fixed. Current evidence does not select a clipping change or new
numeric scales. Either needs its own declaration and bounded comparison. The
fixed geometric proposal ceiling is unchanged and remains a separate issue.
