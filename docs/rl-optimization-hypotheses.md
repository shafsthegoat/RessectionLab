# RL optimization hypotheses and decision criteria

Reviewed October 4, 2026: `spatial_policy.py`, `spatial_task.py`, the spatial
observation contract and existing surgical/encoder evidence notes. This is a
focused interpretation of primary literature, not a novelty review, new study
declaration or claim of state-of-the-art performance. No training ran for this
review; the first spatial screen remains the required baseline.

The purpose of RL is to **amortize sequential planning across anatomies**: learn
which feasible openings enable later benefit, interpret permitted imaging, and
propose useful actions with little online computation. It may also guide bounded
search. An exact optimizer cannot be beaten on its own fully specified objective;
the practical comparison is quality versus total online cost under imperfect
observations. Search can simulate cavity changes and complete tools too. RL
cannot compensate for incorrect mechanics merely by receiving more training.

The current 24,331-parameter CNN learns from scan pixels and observed cavity,
access, tool and budget state. Its search clone is a finite-horizon deterministic
planning model with a fixed scan estimator. Private reference anatomy makes true
reward partially observed, but this task acquires no new anatomical measurements.
Missing vascular/function information therefore does not justify recurrence by
itself. The geometric screen does not assess clinical functional preservation.

Each follow-up below needs one separately declared ablation, frozen objective,
TRAIN-only updates and SELECT-only choices. Do not alter the first screen.

1. **Few positive trajectories or premature STOP:** inspect initial action
   probabilities, costly-opening frequency, completed returns and exact tiny-task
   regret. Compare the existing scratch and BC→RL arms before increasing model
   size. If BC learns teacher states but fails after its own actions, aggregate
   observed-model search labels at learner-visited TRAIN states and compare this
   with equal-label-budget BC. This follows [DAgger](https://proceedings.mlr.press/v15/ross11a.html).
   Charge every teacher query. Retain only if full-episode return improves; action
   accuracy alone is insufficient. Surgical [DEX](https://arxiv.org/abs/2302.09772)
   supports demonstration-guided exploration as a direction, not glioma transfer.

2. **High rollout variance or expensive fresh samples:** compare clipped
   [PPO](https://arxiv.org/abs/1707.06347) with current one-update REINFORCE,
   holding encoder, reward and trajectory budgets fixed. Start with complete
   returns; test [GAE](https://arxiv.org/abs/1506.02438) separately, including
   lambda=1 before a lower-lambda bootstrap. Log old/new log probabilities, KL,
   clipping fraction, advantage variance and all reuse costs. Short episodes
   reduce the urgency of bootstrapping. PPO cannot discover absent successful
   examples simply through repeated updates. Require improved return per total
   second, with failures and seed variability retained.

3. **Value learning overwhelms the actor:** current encoder gradient norms combine
   actor, value and entropy contributions. Measure their separate encoder norms
   and gradient alignment before asserting interference. Compare one lower value
   weight or a separate critic encoder; change one factor. Reject if reduced
   value error fails to improve episode return, or online cost erases the gain.
   A concrete architecture gap precedes these measurements: the critic omits
   tool geometry and available-action summaries, although the actor uses them.
   Identical image/procedure inputs can therefore predict identical value for
   different tools. This remains a valid REINFORCE baseline, but can increase
   variance; test tool-conditioned value error before introducing GAE, then
   consider a pooled tool context as a separate ablation.

4. **Scan perception fails:** compare original, disabled-image-branch and
   intensity-shuffled TRAIN/SELECT probes while preserving geometry. Stratify by
   contrast/noise and examine target-estimation errors separately from planning.
   If perception is limiting, add training-only segmentation supervision or a
   small endpoint patch alongside ray features. Hidden labels remain targets,
   never inference inputs. Keep the strong 0.35 observed-model search prior.
   Require gains on disjoint anatomy, not only auxiliary loss or repeated shapes.
   These diagnostic ablations do not authorize inference without a required scan.

5. **Policy is useful but imperfect:** evaluate policy-guided expansion or learned
   leaf values against unguided search at identical expansion and wall-time
   budgets, plus frozen direct inference. [Expert Iteration](https://arxiv.org/abs/1705.08439)
   explicitly couples planning and generalization. Test value ranking on unseen
   reachable states; retain geometry checks and STOP. A learned value is not an
   admissible bound or a clearance certificate. Report quality lost to pruning.

6. **History actually matters:** first construct two identical current
   observations with different permitted histories requiring different decisions.
   Add missing observed contact/tool state explicitly when possible. Compare
   recurrent and feedforward policies only after timestamped observations or
   measured history dependence exists. [DRQN](https://arxiv.org/abs/1507.06527)
   motivates memory for partial observations; it cannot identify anatomy absent
   from every measurement.

Profile scan preparation, candidate certification, cloning, encoder, ray sampling,
backward pass and selection separately. Static-feature caching requires a
separate dynamic-cavity branch; stale cavity features change the task. Check
CPU/MPS numerical agreement before throughput comparisons. Report training cost
separately and preprocessing through legal-plan output end to end.

Use paired anatomy effects and uncertainty, preserving all seeds and negative
results; small-run point estimates are fragile ([Agarwal et al.](https://arxiv.org/abs/2108.13264)).
After evaluation informs a redesign, that development set cannot support a new
untouched-test claim. Full native-tool and real-patient transfer remain separate
gates; improving this small synthetic task alone does not meet the user goal.
