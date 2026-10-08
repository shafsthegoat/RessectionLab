# Diagnose the failed opening-task fit

October 8, 2026. Saved traces show that RL sampled all 16 terminal action sequences
in 64 episodes, including three exact-optimal, four near-optimal and three other
target-removing episodes. The teacher labels are three tool actions and two STOPs,
with no conflicting observations. Neither absent successful exploration nor a
STOP-label majority explains the failed final policies.

A separately bounded readout reconstructed only the original root and four nominal
prefix states, matching all five saved observation/action fingerprints. It performed
15 model forwards across initial, final BC and final RL checkpoints, with zero
updates or new evaluation rollouts. All checkpoint and source/input hashes stayed
unchanged. Runtime was 3.1264 seconds overall, with 339,165,184 bytes sampled peak RSS.

BC chooses STOP in every state. Mean cross-entropy improves from 1.402509 to
1.290096, but it worsens in all three tool-label states; improvement comes from the
two correctly stopped dead ends. After both useful openings, BC ranks the correct
distal cut highest among tools, but STOP exceeds it by roughly 0.416 logit.
RL ranks the correct distal cut in both post-opening states with small margins
of 0.002632 and 0.002643, while choosing a destructive narrow cutter at the root.

Only one RL update clipped its gradient (multiplier 0.85146). BC never clipped
and has zero critic gradients. Critic/clipping dominance is therefore not an
explanation for this BC failure. STOP logits barely distinguish the four
post-action states. The STOP head lacks candidate features supplied to tool heads,
but this is a representation hypothesis, not proof that the current network cannot
learn the distinction from its image/procedure inputs.

The next discriminating test is one fixed longer BC fit using the same architecture,
labels, loss and initialization. Measure each state's teacher probability/margin,
not only average loss, and use the fixed final checkpoint. This remains a generated
training-capacity check; it cannot establish patient transfer or clinical usefulness.

`teacher-fit-attempt-01/readout.json` has SHA-256
`555bcc56f333e9ca505735e6ed85855e2cb034ed7179d3ac4ec06f5d873d75cd`.
The retained scripts, settings, release, supervision and evidence index reproduce
the diagnostic and bind its original local inputs. No patient or holdout was opened.
