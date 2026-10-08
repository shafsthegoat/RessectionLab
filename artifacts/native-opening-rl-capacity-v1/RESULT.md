# Longer scratch RL learns the fixed generated opening task

The unchanged policy trained for 256 updates from the original initial weights. Its final argmax rollout scored **1.098**, close to complete search's **1.100**. Both remove 2 mm³ of target and 4 mm³ of normal tissue with one tool change. RL travels 12 mm rather than search's 10 mm, accounting for the remaining 0.002 reward gap. The original initial score of −0.464 and failed 16-update score of −0.896 remain unchanged.

This is one generated six-cell task, with fixed geometry, tools, reward, seed and a two-action horizon. There are no real patients in this experiment. It demonstrates learning capacity under these assumptions, not anatomical generalization, physical fidelity or surgical performance. Search remains the exact reference optimum for this finite task.

Independent saved-output review verified all 256 updates and 1,024 audited training episodes, checkpoint weights and optimizer moments. The first 16 updates reproduce the original 64 trajectories, losses, gradients and optimizer state exactly. Training uses fresh on-policy experience; the complete search tree is used only for assessment. All 1,026 training and inference histories passed their native geometry audits. These are execution checks, not clinical validation.

| Fixed readout update | Expected return under stochastic policy | Probability of any target removal |
|---|---:|---:|
| 0 | −0.275463 | 23.55% |
| 16 | −0.270151 | 22.04% |
| 32 | −0.263068 | 22.47% |
| 64 | −0.251587 | 23.26% |
| 128 | −0.192246 | 23.77% |
| 256 | +0.546023 | 67.99% |

These probabilities are exact finite-tree calculations for this generated task, not clinical probabilities. Final full-target probability is 67.35%; the near-optimal argmax therefore does not imply a reliable stochastic policy. Appropriate STOP probability in the two dead ends remains only about 4.84% and 5.67%. Intermediate readouts did not select the fixed final checkpoint.

The run completed in 93.6124 seconds, with 334,151,680 bytes sampled peak worker RSS and no retry. Fitting with readouts and persistence took 88.3922 seconds. Episode collection before auditing took 45.5161 seconds, audit callbacks 24.6023 seconds, and loss/backward/optimizer work 4.1403 seconds. The remaining time includes persistence and other overhead that was not separately isolated. Native previews consumed 27.2113 seconds within collection; these overlapping timers must not be added together. Geometry and auditing dominate the measured cost. Final online inference took 0.0650 seconds, plus 0.0426 seconds for its independent audit; single ordered timings are not a robust deployment benchmark.

Original full records and weights remain local, bound by the output inventory. The compact publication summary, update ledger and independent verification preserve reproducible accounting without committing bulk trajectory files or weights. The next useful diagnostic is frozen input compatibility on a declared TRAIN patient's acquired anatomy, followed by matched transfer evaluation once its task and evidence requirements are met. More fitting on this same fixture would not answer that question.
