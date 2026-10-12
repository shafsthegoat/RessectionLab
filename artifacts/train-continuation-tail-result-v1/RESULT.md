# Exact immediate cost repairs most learned-route fragmentation

Separating the known immediate reward from predicted continuation reduces the
learned planner from 42 to 25 movements and from 30 to 3 tool changes, with the
same target removal, outside-target removal and contact bounds. This improves
the previous direct-value model, but fails the frozen acceptance gate against
greedy search. Greedy remains the retained planner.

All twelve fresh routes complete and pass the independent full-tool geometric
checker. All eight GREEDY/DIRECT_CALIBRATED routes reproduce the prior comparison
exactly, and both prior head fits reproduce. A separate saved-result auditor
passes 5,357 checks in 0.347 seconds, including independently enumerated tail
labels, scalar prediction errors, selection rules, complete-route outcomes,
historical failures and cumulative costs. It does not rerun neural inference.

| Four TRAIN cases combined | Greedy search | Previous direct value | Exact reward + learned tail |
|---|---:|---:|---:|
| Target removed, mm³ | 103.950561 | 103.950561 | 103.950561 |
| Non-STOP movements | 25 | 42 | 25 |
| Tool changes | 0 | 30 | 3 |
| Complete-tool travel, mm | 152.382837 | 243.446920 | 153.489088 |
| Declared simulated return | 103.048178 | 101.547114 | 102.957072 |
| Online planning previews | 2,231 | 3,730 | 2,245 |
| Online planning transitions | 29 | 46 | 29 |
| Online planning time, seconds | 25.374 | 52.008 | 32.852 |

Timing is one ordered, instrumented run, not a repeated latency benchmark.
The new learned planner uses 39.81% fewer online previews than the previous
learned model, but 14 more than search. Training, source preparation and fresh
independent replay remain separately charged; online figures are not total cost.

ReMIND-002 and 018 stop without removal in every arm. The unchanged
actual-positive-action guard enforces those stops; they are not evidence of
learned negative-action avoidance. On 015, TAIL matches greedy's 58 target cells,
13 movements, return and travel, with different ordering and 14 extra previews.
On 045, both remove 51 cells in 12 movements, but TAIL makes three tool changes
and travels 1.106252 mm farther. Its return is 48.090857 versus 48.181963.
No case has increased outside-target removal or cumulative contact bounds.

## Change, failure and next experiment

The original encoder and hidden head stay frozen. One fixed weighted SVD solve
fits the best recorded subsequent prefix, excluding the current action reward,
on the same 497 authenticated rows from 36 simulator traces. Deployment adds
the already-computed nominal immediate reward exactly once to the nonnegative
predicted tail. Rewards, patient roles, complete candidate inventory, geometry,
horizon, guard and acceptance criteria stay fixed. There are no new training
rollouts or parameter sweeps.

Reconstructed logged Q error decreases from 87.912479 to 83.379598 (5.16%).
Raw tail MSE is 83.461201; clipped tail MSE is 83.379599, with 70 negative raw
predictions. These are fitting errors, not held-out accuracy. The logged design
still has rank 9 of 33. Lower loss and fewer movements do not establish useful
improvement over search.

Saved decisions isolate the remaining 045 loss: its 0.091106 return deficit
equals three 0.03 switching costs plus 0.001 times the extra path length. At
steps 9 and 12, predicted tail advantages of 1.301331 and 1.888038 override
immediate losses of 0.030553 each. The final productive actions predict tails
of 17.5804 on 015 and 15.9263 on 045, yet the next decision stops with zero
additional return. Those exact observation/action pairs are absent from the
logged training rows; this does not establish feature-space novelty or prove
all alternative continuations worthless.

The next bounded diagnosis is a common-state endgame comparison at 045 step 9:
take the learned choice or the immediate-best choice, then run the same frozen
continuation policy to completion in both branches. Charge complete branches
and independent geometry checks. This tests whether the forecasted switching
benefit exists in this simulator before changing the model, its training labels
or its inputs. No new dataset is justified by the current evidence.

## Cost and evidence boundary

The process completes cleanly in 270.440 seconds parent time (269.887 seconds
worker), with 17,856 native previews, 67 frozen encoder forwards and
1,406,976,000 bytes sampled peak process-group RSS. It loads the single existing
670,261-byte feature cache without copying or reconstructing it, reproduces the
64-update head and previous direct readout, then performs the one tail solve
(0.011804 seconds). Two extra head-only batches save predictions for independent
scalar checking. The parent confirms zero exit, a reaped child and no remaining
owned processes. Source pins remain unchanged.

The prior cache-publication failure and direct-calibration recovery remain paid.
Together with this contrast they total 84,990 native previews, 654 frozen
encoder forwards and 1,941.253 seconds parent time. The older original negative
Q64 experiment remains an additional historical cost. Nothing is amortized away.

These are annotation-assisted simulated decisions in four exposed TRAIN
anatomies, not recorded surgeon actions. Productive routes still remove less
than 1% of either full target. SELECT/EVAL remain closed. Independent geometric
and scalar checks support software correctness under the declared model; no
new physical measurement validates cutting, neurological safety, generalization
or patient benefit. Existing measured mechanics and observation limits remain.

The dated storage receipt accompanying this result records extension growth and
free space after compact evidence copying. No new dataset was acquired. Patient
arrays, the feature cache and all model/head weights remain outside Git.

## Reproduction

Canonical source is commit `5944847fe474675b7df37950c9f1e14927edecf5`.
The fixed runner, configuration and 28 model / 19 runner / 414 independent
generated controls are archived in `../train-continuation-tail-source-v1/`.
`source-index.json` binds the actual run. `inventory.json` maps these compact
result files to their original locations and hashes; the head-only checkpoints
remain local and are identified by hashes in the report.

For the saved-only audit, restore the archived files to their recorded `build/`
paths without overwriting existing files. Its prior auditor dependencies and
saved inputs are recorded in `independent/audit-01.json` and the earlier
readout/value result archives. Run `independent/audit_saved.py` with the report,
launcher and index hashes recorded in the receipt, and a new exclusive output
inside its own directory. The audit reads JSON/scalars only: it neither decodes
the cache/weights nor substitutes for independent physical validation.
