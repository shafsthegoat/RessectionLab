# One recorded-continuation-tail contrast

Source preparation only. Root owns the generated test slot, current-source freeze and sole acquired comparison. Previous runner/source indices/outputs and the 670261-byte feature cache remain untouched; no cache, images or base weights are copied.

Fixed setup: original IL64 encoder/actor, authenticated497 features and weights from36 simulator-generated recorded traces in annotation-assisted four TRAIN anatomies (002/015/018/045). Reproduce the original64-update head and direct calibration, then one fixed weighted final-Linear SVD tail solve. Frozen hidden layer, cutoff1e-8, no tuning or new training trajectory. Tail labels use the original manifest's best subsequent recorded prefix (STOP0); current reward is excluded from the fitted tail.

Fresh GREEDY, DIRECT_CALIBRATED and TAIL routes use the same worlds, h24, native actions/reward and actual-immediate-positive guard. Direct Q already includes current reward. TAIL ranks exact certified immediate reward plus max(0,predicted raw tail), adding current reward exactly once. Future recorded labels/corpus stay outside deployment. Both original GREEDY and direct-calibrated full routes/outcomes must reproduce exactly; improvement over the direct head alone does not qualify.

Useful gain requires all12 complete independently accepted routes; percase TAIL target/return no-loss and outside/contact no-increase versus fresh full-inventory GREEDY, plus strict aggregate improvement in previews, transitions, complete path or nonSTOP actions. Ties are negative. This does not measure surgical or clinical outcomes and does not demonstrate learned negative-preparation recovery.

The original cache's external origin, encoder, feature-producer hashes, corpus seal, world bindings and scalar labels remain exact. Current executing sources are independently frozen; unrelated historical planner files are provenance rather than an executable closure that must remain unchanged forever.

Outputs include compact reports/certificates, two head-only checkpoints,497 original-Q labels,497 tail labels/rewards/weights/STOP flags, and497 raw DIRECT_CALIBRATED/TAIL scalar predictions. Two extra logged-row head forwards are charged. Tail planning rows retain actual reward, raw/clipped tail and combined score for independent saved choice reconstruction. No bulk histories or source images are duplicated.

Forecast based on the completed384.716s direct recovery: about400–700s and25–35k new previews. Fixed ceilings remain3600s worker /3630s parent,3GiB sampled group RSS,100,000 previews,2048 frozen-policy forwards,12MiB output,32MiB whole stage. The1285.243s cache failure and direct recovery retain their paid costs; current and cumulative previews/forwards are explicit. No automatic retry.

After the root-owned test batch and convenience-link cleanup, freeze current sources once:

```sh
.venv/bin/python -I -B build/continuation-tail-v1/runner/freeze_sources.py --release
```

Root's sole acquired run (fresh output `build/continuation-tail-v1/run-01`):

```sh
.venv/bin/python -I -B -X pycache_prefix=build/continuation-tail-v1/absent-root-launch-cache build/continuation-tail-v1/runner/run_owned.py --source-index-sha256 <emittedSHA> --execute
```

Generated controls entry: `runner/test_runner.py`; isolated bootstrap imports the current cache/calibration/tail/planner under the actual read-only boundary, without patient arrays, cache feature decode, checkpoint loads or native/model calls. No tests or acquired execution occurred during this source preparation.
