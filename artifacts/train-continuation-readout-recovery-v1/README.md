# Exact cache-only recovery

The original `runner/`, source index and `run-01/` remain immutable. Its 36 complete traces / 497 states produced a valid saved cache; execution then failed because the exclusive output writer was called twice for `logged-targets.json`. No fit, calibration or evaluation route occurred. Preserve the failed parent status, 1285.243199 s elapsed, 45,560 previews, 497 reconstruction transitions and 497 frozen-policy forwards. Original corpus reconstruction itself cost 1279.018679 s.

This successor authenticates that exact failed attempt and its external source index, cache binding, write receipt and reconstruction evidence before loading only the existing IL64 checkpoint and cache. It neither replays training traces nor rewrites/copies the cache. Canonical feature-source identities and original corpus/encoder/world bindings must match. `logged-targets.json` is published once with the prior cache reference and correct saved-feature status.

The scientific continuation is unchanged: reproduce the original64 head/hash/loss rows and case MSE; one fixed final-Linear calibration; fresh GREEDY/Q64/CALIBRATED planning and independent replay on the same four TRAIN cases; exact fresh GREEDY/Q64 historical route reproduction. Primary useful gain remains calibrated versus fresh GREEDY, allcase target/return nonregression and outside/contact nonincrease plus strict aggregate work/route improvement. Ties are negative. No new trajectories for training, tuning, held-out role, reward, geometry or action-set changes.

The only new diagnostic is `logged-predictions.json`: 497 Q64 and calibrated scalar predictions in the existing manifest order, STOP fixed0, joined to targets/weights by exact file hash. It is at most64KiB and costs exactly two extra head-only forwards, separately charged; no encoder/native calls. It permits saved arithmetic checking of actual fit residuals without reading cached features or head weights.

Caps remain one thread, 3600s worker /3630s parent,3GiB sampled group RSS,100,000 previews,2048 frozen-policy forwards,12MiB recovery output within the32MiB shared stage. Approximate new cost forecast400–700s and25–35k previews; the cache-fill failure remains separately paid and cumulative preview/forward counts are reported. No automatic retry.

Root removes only its completed-control convenience symlinks with a receipt, retaining target files, then freezes once:

```sh
.venv/bin/python -I -B build/continuation-next-iteration-v1/recovery-runner/freeze_sources.py --release
```

Use the emitted SHA for the sole attempt, with fresh absent cache prefix:

```sh
.venv/bin/python -I -B -X pycache_prefix=build/continuation-next-iteration-v1/absent-root-recovery-cache build/continuation-next-iteration-v1/recovery-runner/run_owned.py --source-index-sha256 <emittedSHA> --execute
```

Fresh exact output: `build/continuation-next-iteration-v1/recovery-01`. Existing cache remains at `run-01/authenticated-features.zip` (670261B, SHA256 `2ebd7bd2ad777492bec95fa910ee2fb1c4405da04a30f69219b448ae20d236f7`). Binding/receipt/evidence pins are explicit in `common.py`; source freeze also rechecks every original cache-producing source file.

`controls-01.json` retains the exact17-test source pins and clean lifecycle. Controls include a real `OutputBudget` duplicate-write regression, one correct publication, exact saved metadata origin and negative substitutions, unchanged useful gate/baseline checks, false-release refusals and an isolated actual runtime/bootstrap import. No cache features, checkpoint, patient arrays, native calls, training or model forwards were used. Root owns actual recovery release.
