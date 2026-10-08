# Fixed longer imitation fit on the native opening fixture

This prospective diagnostic tests under-optimization before changing the model,
reward, tool actions or label weighting. The earlier 16-update fit selected STOP
at all five teacher states: its average cross-entropy improved, but loss worsened
at each of the three states labeled with a tool action. The other two labels are
STOP. The preserved diagnostic is `build/native-opening-learning-diagnostic-v1`.

One seed-11 model starts from the exact original initial weights, with a fresh
Adam optimizer, learning rate 0.001 and gradient clip 5. Every one of 256 updates
uses the same five equally weighted states in the original order. The model,
physical normalization, horizon two, reward, source, tools and production code
remain unchanged. There are no RL updates, additional seeds or patient inputs.

The runner authenticates the original declaration, execution index, source
closure, teacher labels and all five accepted independent demonstration audits.
It reconstructs the four original one-action prefixes using nominal planning
clones, then matches every observation fingerprint, action inventory and label.
It performs no new search or teacher-label selection. Update 16 must reproduce
the original parameter hash and full gradient/hash trajectory; otherwise the run
stops and retains the failure.

Readouts at updates 0, 16, 32, 64, 128 and 256 report all five states' teacher
cross-entropy, probability, rank, teacher margin and STOP-versus-best-tool margin.
These are 30 separate no-gradient forwards. The training workload is exactly
1,280 loss forwards. The fixed final checkpoint at update 256 is the primary
result, even if an earlier checkpoint performs better. No readout selects,
extends or stops optimization based on performance.

Initial and fixed-final argmax episodes use the existing durable-history writer
and independent native geometry/removal evaluator. Each online episode has a
10-second, 2,048-actual-preview envelope including clone, inventory and execution;
the audit is separately timed. Shared preparation, four-prefix reconstruction,
optimization, readouts and exports are also measured. Earlier teacher-generation
and demonstration-validation costs remain explicit inherited costs, not free
training data and not newly executed work. An external supervisor caps this
single attempt at 60 seconds and 2 GiB sampled worker RSS, with one CPU thread.
The cap is an engineering allowance, not a performance target. No automatic retry.

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
PYTHONPATH=src .venv/bin/python scripts/run_native_opening_bc_capacity.py \
  --declaration manifests/experiments/native-opening-bc-capacity-v1.json \
  --execute --output artifacts/native-opening-bc-capacity-v1
```

This is a generated training-fit diagnostic on one six-cell native task. It can
show whether the unchanged architecture fits these five labels with additional
updates. It cannot establish anatomical generalization, patient transfer,
mechanical realism, clinical benefit or RL improvement. The original negative
fit, checkpoints, source snapshots and diagnostic receipts remain unchanged.
