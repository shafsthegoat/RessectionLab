# Fixed longer imitation fit: search matched on the training task

The unchanged model, labels, reward, seed and initial weights now fit all five teacher decisions after the fixed 256 updates. The final independently audited argmax rollout uses the same two-tool sequence as complete search: return **1.1**, target removal **2 mm³**, normal removal **4 mm³**, and one tool change. The original 16-update imitation policy stopped at zero; that negative result is unchanged.

| Update | Correct teacher decisions | Mean cross-entropy |
|---|---:|---:|
| 0 | 0/5 | 1.402509 |
| 16 | 2/5 | 1.290096 |
| 32 | 2/5 | 1.153211 |
| 64 | 4/5 | 0.417345 |
| 128 | 5/5 | 0.277706 |
| 256, fixed primary | 5/5 | 0.103085 |

Checkpoint 16 reproduces the original tensor weights, Adam moments and complete gradient/loss history exactly. Intermediate readouts did not select the result. Independent review verified all 256 updates, 1,280 loss forwards, 30 diagnostic forwards, four inference forwards, 84 native previews and the two accepted inference histories. Actor, encoder and STOP weights changed; imitation left critic weights fixed. All 278 indexed outputs of the original study remain unchanged.

The parent observed 7.7639 seconds and 329,711,616 bytes sampled peak RSS. Actual loss/update work took 2.4609 seconds; fitting with intermediate readouts/exports took 3.6030 seconds. Fixed-final online inference took 0.0600 seconds, plus 0.0373 seconds for its independent audit. Earlier teacher generation and validation costs remain explicit inherited costs. These single ordered timings are not a robust deployment benchmark.

This demonstrates capacity on one generated six-cell training task. It is not anatomical generalization, held-out patient performance, physical fidelity or an RL improvement. Exact saved-tree arithmetic also shows that stochastic BC256 still assigns substantial probability to inferior actions, despite its optimal argmax. The next controlled question is whether the unchanged scratch RL learner improves with more optimization; no architecture change is justified by the earlier short BC failure alone.
