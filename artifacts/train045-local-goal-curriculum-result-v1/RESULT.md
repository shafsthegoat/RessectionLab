# Reset-curriculum PPO comparison: actual uniform512 midpoint

**Interim result, October 12: the uniform policy stops immediately and fails the
full task.** Its return improves from -0.438688 at 256 decisions to zero at 512
because it avoids acting. This is not useful planning improvement. The retained
search still succeeds in five preparation/goal actions, with return 9.811680 and
zero outside-target removal. The campaign remains in progress; both scheduled
1,024-decision endpoints and the reverse-reset midpoint are pending.

![Actual interim learning curves](snapshot/plots-curriculum-01/learning-curves.png)

The fixed task, information, reward, starts and resource limits are recorded in
[the prospective source archive](../train045-local-goal-curriculum-source-v1/README.md).
This is one exposed TRAIN045 geometric simulation using supplied/estimated
anatomy, not observed surgery, physical validation or evidence of generalization.
Search uses charged model queries under the retained protocol; it does not share
PPO's exact feature representation. Final patient evaluation remains closed.

## What actually trained

The uniform branch resumes the exact frozen-encoder PPO256 weights and Adam
state. At its saved midpoint:

| Quantity | New segment | Shared-parent lineage |
| --- | ---: | ---: |
| Policy decisions | 256 | 512 |
| Actual completed Adam updates | 46 | 204 |
| Completed episodes | 35 | 61 |
| Successful sampled episodes | 5 | 9 |
| Distinct native source states | 219 | 409 |
| Distinct state/action pairs | 250 | 465 |
| Unique patients | 1 reused | 1 |

All five new sampled successes start closer to the goal: four at start 16 and
one at start 14; none at start 12, 13 or 15. They are saved training trajectories,
not independently replayed full-task successes. The lineage's 492 decisions in
completed episodes, 12 retired unfinished decisions and 8 live unfinished
decisions sum to 512. Unfinished episodes are not converted into failed episodes.
The reverse branch will share the same original 256 decisions; cross-branch
totals must count that parent once. The retained trainable-encoder256 baseline
and original imitation model remain separate.

The standard SB3 checkpoint, including optimizer state, remains local at
`build/train045-local-goal-ppo-v1/curriculum-1024-01/uniform-seed0/ppo-0512.zip`:
248,811 bytes, SHA256
`f299a593ff0a07cf1694e07410ccda020f3ee46b91f1231eebe2f1070346a24e`.
The encoder remains unchanged, as required by this reset-only comparison.

## Independent midpoint check and cost

The evaluator verified the exact inherited twelve-action prefix, the final STOP,
compressed artifact hashes, semantic plan/history seals, accepted full-tool
geometry and reconstructed local reward. The full history has thirteen rows and
the policy makes one STOP decision: no new movement, goal removal or outside
removal. Its initial native state, observation and action catalogue match the
retained search comparison. Available preparation actions were not removed from
the benchmark. The review used saved records; the worker had already performed
the fresh native replay and independent geometric evaluation.

Checkpoint hash, 204 completed optimizer hooks, unchanged encoder and decision
partitions were checked separately. The midpoint isolation receipt verifies
unchanged policy, optimizer, training summary, last observation and native state,
with random generators and policy mode restored. Subsequent saved progress at
561 decisions confirms training resumed. The running report's initial
`source_unchanged: false` is pending the final guard, not a terminal failure; no
terminal parent receipt exists yet.

The new segment through this checkpoint costs 1,515.756 seconds, including its
evaluation. Evaluation, replay, geometry and isolation together cost 25.434
seconds. The evaluated policy constructs one catalogue in 3.553 seconds; that
catalogue-only duration is not total inference latency. The live report has not
yet published final phase costs. Retained search costs 118.055 seconds online
and 194.241 parent seconds including replay/validation. A failed STOP policy
does not establish an acceptable speed advantage. Shared upstream model/teacher
preparation costs remain in the preceding campaign records.

The compact renderer reads only saved scalar reports and training-episode
records; it neither executes policies nor decodes checkpoints. The figure was
rendered and visually inspected. Endpoint points, sampled training returns,
actual optimizer steps and paid costs are separated. Its latest training rows
may extend past the immutable512 endpoint; inputs are sequential snapshots with
their hashes retained in the JSON. Pending endpoints are not plotted as failures.
Rerun from the original runtime location:

```sh
.venv/bin/python -I -B build/train045-local-goal-ppo-v1/plots-curriculum-01/render_curves.py
```

The archived renderer preserves that script; it expects the original stage
layout. `INVENTORY.json` binds the exact interim report, curve, renderer and plots.
Weights, anatomy arrays and bulk trajectories are not archived in Git.

## Decision still pending

Keep search as the working baseline. The demonstrated failure is immediate
abstention at the full-task start despite sampled successes near the goal; this
does not yet isolate representation capacity, optimization or sparse-credit
learning. Finish the frozen uniform/reverse comparison without changing rewards
or legal actions mid-run, then decide whether the curriculum improves complete
outcomes. No additional model sweep is launched alongside it.

Before this compact archive, measured extension growth is 1,839,554,560 bytes
(1.713 GiB), with 351,302,254,592 bytes free (327.176 GiB), no reservation
overruns and no new dataset. Runtime artifacts remain within the existing
192 MiB stage reservation; compact archival writes use the existing root
bookkeeping reservation. Physical measurements continue as a separate evidence
stream.
