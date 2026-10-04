# Saved-record diagnostic after feature-unit completion

Prepared while the registered run was executing, without inspecting partial
rankings. After full completion and root release, six static tests passed and
the diagnostic executed with zero resets, transitions or backward passes.
Results and source receipts are preserved under
`artifacts/learning/procedural-native-feature-units-v1/diagnostics-v1/`.
This is a descriptive diagnostic design, not another optimization declaration.

`scripts/diagnose_feature_unit_mechanisms.py` waits for the complete 12-arm,
two-pretraining, two-frozen study and all 23 candidate geometry records. It then
imports the exact frozen runtime and loads the actual initial, selected, latest,
and shared policy tensors. The seven common raw observations already saved in
the completed procedural-native v2 diagnostic supply the two procedural source
initial/cut states and three patient states. Their world seeds, model hashes and
feature order must match the unchanged physical declaration. No simulator is
constructed, reset or stepped. The script performs no backward pass or update.

| Question | Measurement from retained evidence | What it cannot establish |
| --- | --- | --- |
| Does actor input saturation persist? | Apply each checkpoint's actual input profile to the same raw observations. Record masked logits/probabilities, entropy, first-layer preactivation range, fraction of tanh activations with absolute value at least .99, and mean local tanh derivative. Compare paired initial tensors separately from trained checkpoints. | Seven common states are not the learned policy's occupancy distribution. Saturation or a change in saturation alone does not prove a causal effect on learning or selection performance. |
| Does the critic alias different domains? | Check exact equality of the six initial state inputs across both sources and the patient. For every fixed checkpoint, measure the spread of its three value predictions. Compare saved complete source and patient selection panels only when they belong to the same shared checkpoint; missing latest-source panels stay null. | Deterministic greedy panel returns are not the stochastic per-transition return targets used in REINFORCE. Logs do not retain those individual targets, so exact training-value error cannot be reconstructed. |
| Does the critic increase clipping of actor gradients? | From each saved total preclip norm, actor postclip norm and the recorded threshold, infer the common clipping coefficient and preclip actor norm. Since this policy has only actor and value parameters, infer the value-gradient norm by subtracting squared group norms. Report the additional actor shrinkage attributable to using the joint norm instead of the same threshold on the actor alone. | This is arithmetic on logged norms, not a counterfactual optimizer run. Float32 rounding limits precision; gradient directions and optimizer moment history are insufficiently logged. Adam can offset common gradient rescaling, so these coefficients do not predict proportional parameter-update loss. |

The clipping calculation is bound to the installed, version-matched Torch source
rule `min(1, threshold / (total_norm + 1e-6))` and the frozen learner's recorded
actor norm after clipping. Unexpected parameter groups or inconsistent logged
norms cause refusal instead of a guessed decomposition. All checkpoint hashes
are checked again after inference. The report preserves its own script/helper
hashes, the study source/declaration hashes and the archived-observation digest.

Existing gradient history, selected-versus-initial status and physical selection
returns remain descriptive context. No diagnostic changes checkpoint selection,
feature units, learning rates, loss weights, clipping, candidate geometry, world
panels or rewards. Final and stress worlds remain unopened.

Evidence for a separately declared follow-up would need to identify a specific
remaining limitation. Equal critic inputs together with divergent completed
returns for one fixed policy would justify testing a small, prospectively
available physical state summary for the value head, while preserving the actor
and current loss/clipping settings. A large inferred value contribution to the
joint norm would justify a separate clipping-group ablation with the feature and
state representation fixed. Either intervention changes a different mechanism
and should be declared and tested separately; neither is automatically selected
by this diagnostic. Saturation findings should first be interpreted alongside
the completed RAW-versus-FEATURE_UNITS ablation, without fitting new scaling
constants to patient outcomes.

After explicit execution release, validate the completion gates, saved-profile
forward statistics and clipping arithmetic on small static fixtures before
running this diagnostic. Preserve any failure and its receipt; do not modify the
registered runtime to make a diagnostic succeed.
