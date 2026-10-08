# Fixed longer scratch RL fit on the native opening fixture

The first 16 RL updates produced a slightly higher stochastic expected reward
(−0.275463 to −0.270151), while any-target probability fell from 23.55% to 22.04%
and the argmax trajectory worsened. The separate BC256 experiment fit all five
teacher states and reproduced the optimal deterministic return, 1.1. This shows
representational capacity on the generated fixture; it does not show that longer
RL will succeed. The saved failures remain part of the evidence.

This experiment changes only the number of scratch RL updates, from 16 to a fixed
256. Seed 11, the original initial tensors, fresh Adam at 0.001, four on-policy
episodes per update, REINFORCE loss, gamma 1, entropy weight 0.01, value weight 0.5
and global gradient clip 5 remain unchanged. The existing candidate-aware 3D
policy, source, tools, two-decision horizon and geometric reward also stay fixed.
No BC weights, teacher actions, cached trajectories or nominal tree returns enter
the training loss. Every update uses four newly executed, independently audited
episodes from unchanged current weights. The first 16 action/reward/observation
trajectories and loss/gradient/hash records must exactly reproduce the original
run or this attempt stops.

All 1,024 training episodes have the existing uninterrupted online envelope of
10 seconds and 2,048 actual native preview entries, including clone, inventory,
policy and execution. Independent audits are separate measured work within the
whole worker limit. There are at most 2,048 training transitions, the same number
of repeated loss forwards, and separately counted action-collection forwards.
Failed updates keep their completed and partial episode records; an incomplete
final rollout has no outcome score.

At updates 0, 16, 32, 64, 128 and 256, five no-gradient policy forwards assess the
complete saved finite tree. The 16 terminal route returns are recovered from the
nominal planning tree and matched to original independently audited episodes.
Each path probability is the product of its state-dependent action probabilities;
saved float32 probability vectors are normalized by their sums for this arithmetic
only. Report nominal expected reward, generated any-target/full-target
probabilities, expected removal volumes, root STOP probability, action logits,
critic values and gradient/clipping records. This is privileged model assessment,
not fresh sampled evaluation and not a clinical probability. The four reconstruction
transitions, previews and 30 readout forwards are separately charged. None affects
the action RNG or selects a checkpoint. Initial and fixed-final argmax executions
also receive fresh independent geometry/removal audits.

Update 256 is the primary result even if worse than an earlier checkpoint. There
is no successful-result stopping, retry, seed sweep or budget extension. Original
RL16 took 4.02 seconds for fitting; linear extrapolation is about 64.4 seconds,
not a throughput guarantee. The prospective allowance is one CPU thread, 180
seconds for the externally supervised worker and 2 GiB sampled worker RSS. Prior
teacher/search/audit evidence is credited as inherited assessment cost; new native
preparation, collection, audits, loss updates, readouts and exports are measured.

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
PYTHONPATH=src .venv/bin/python scripts/run_native_opening_rl_capacity.py \
  --declaration manifests/experiments/native-opening-rl-capacity-v1.json \
  --execute --output artifacts/native-opening-rl-capacity-v1
```

An exact-tree expectation-gradient optimizer would change the estimator and its
model access, so it is not mixed into this duration test. This single generated
fixture can diagnose optimization and credit assignment. It cannot establish
seed robustness, anatomical generalization, patient transfer, physical realism,
clinical benefit or an advantage over search. No patients or held-out cases are
opened, and existing numerical sources and historical artifacts remain unchanged.
