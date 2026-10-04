# Synthetic four-arm development result

October 4, 2026. The prespecified eight-update experiment completed from its saved exact source snapshot. **Shared pretraining and adaptation showed no benefit on this task.** These are modeled development scores, with zero independent human patients and no final/stress worlds opened.

| Method | Selection return, seeds 11 / 23 / 47 | Actual online gradient steps |
|---|---|---|
| SEARCH | 1.30, deterministic nominal result | 0 |
| Scratch REINFORCE | 0 / 0 / 1.24 | 8 per seed |
| Frozen shared policy | 0, one fixed policy | 0 |
| Adapted shared policy | 0 / 0 / 0 | 8 per seed |

Every scratch/adapted actor changed during optimization. All adapted runs started from the same shared hash and retained that initial checkpoint under the complete-selection-panel rule; updates did not imply improvement. Scratch seed 47 improved from 0.02 to 1.24. SEARCH reached 1.30 in 110 transitions and approximately 0.048 seconds, plus 0.016 seconds of preparation. Online learner times, including selection, were approximately 0.070–0.099 seconds per seed, plus separately recorded preparation/extraction. This tiny run is a software comparison, not a desktop performance guarantee.

Shared pretraining used two procedural anatomy groups. Both contributed eight completed episodes to gradient batches: eight optimizer updates total, 42 optimization transitions and 20 selection transitions. Learner time was 1.329 seconds; total offline time before writing the exported checkpoint was 1.373 seconds, including measured preparation and postprocessing. This offline cost is additional to the frozen/adapted online cost.

The first run's frozen-policy elapsed field measures selection rollouts only. Its arm preparation is separate, and target-side shared-checkpoint validation/copy/loading was included in total experiment time but not timed separately. Therefore this field is not end-to-end frozen-inference latency. The runner now records that initialization phase for future runs; these saved results were not rerun or retroactively relabeled.

All ten frozen candidates passed the independent geometry checker on the declared synthetic simulation grid. Validation took 0.134 seconds. Geometry passing does not validate patient anatomy, clinical outcome probabilities or tissue mechanics. The branching task was excluded from this shared checkpoint but was already an algorithm-development fixture; it is not an untouched clinical test case. One shared initialization cannot estimate population-pretraining-seed variation.

`summary.json` contains the measured rows and limitations; `prespecified-design.json` contains the budgets and source hashes. `frozen-source/` preserves executable source, `pretraining/` preserves shared-training provenance, and `comparison/` preserves initialization/selected checkpoints, candidate freeze, independent audit and per-arm records. The copied shared checkpoint remained unchanged. No parameters were tuned or additional runs made in response to these results.
