# One fixed final-readout comparison

Root executes one run only after freezing this runner and the canonical sources. The original negative run remains unchanged. Four existing TRAIN worlds (002/015/018/045), 36 authenticated traces / 497 rows and all signed targets/weights are fixed. One native reconstruction per trace feeds one detached feature extraction per state. The derived cache is saved and reloaded against its external original binding before fitting; it contains no image arrays or full encoder/actor checkpoint.

The runner loads only the fixed IL64 checkpoint. It reproduces the original 64-update head hash, all loss/gradient rows, case MSE and fresh GREEDY/Q64 routes exactly. It then makes one predefined final-Linear weighted least-squares calibration, leaving the original hidden layer and Q64 scorer unchanged. GREEDY, Q64 and CALIBRATED each receive fresh planning and independent authoritative replay on all four worlds. Historical IL64/RL8 outcomes are pinned references only. No selected/evaluation patients, new training rollouts, reward changes or parameter sweep.

Primary useful gain is CALIBRATED versus fresh full-inventory GREEDY: complete certified routes on all four worlds; target/return no loss and outside/contact no increase per case; then strict aggregate improvement in planning previews, transitions, complete path length or executed nonSTOP actions. Exact ties are negative. Improvement over the poor Q64 baseline alone is not success. Guarded immediate-positive action selection is identical to the previous experiment; no claim about learned negative-preparation recovery.

## Fixed budget and output

One thread, 3600 seconds worker / 3630 seconds parent including cleanup, 3 GiB sampled group RSS, 100,000 native preview entries, 2,048 frozen-policy forwards. Exact output `build/continuation-next-iteration-v1/run-01` is limited to 12 MiB; whole shared stage 32 MiB. Parent retains the tested owned-child helper and Darwin sampler, 50 GiB free-space floor, no retries.

Prior paid reconstruction was about 1117 seconds / 45,199 previews. Removing repeated IL64/RL8 evaluation and adding one calibrated arm gives a conservative expectation of 1500–1900 seconds / about 70,000 previews; this is a forecast, not a measured speed result. Every reconstruction, policy forward, cache write/read, 64 optimizer updates, calibration call and fresh route/replay is charged separately. Original failed/negative costs remain visible.

Outputs: `report.json`, `launcher.json`, `trace-manifest.json`, `logged-targets.json`, `reconstruction-evidence.json`, `cache-binding.json`, `cache-receipt.json`, one bounded `authenticated-features.zip` (at most 2 MiB), two head-only checkpoints (at most 128 KiB each), 12 compact route/certificate JSONs and existing source/admission receipts. The cache and original binding stay local; archive compact scalar evidence rather than copying derived feature arrays. `logged-targets.json` explicitly marks the saved feature cache. No full source images or original weights are copied.

## Root commands

After root removes only completed test convenience symlinks with its existing receipt process, leaving all target evidence, freeze once:

```sh
.venv/bin/python -I -B build/continuation-next-iteration-v1/runner/freeze_sources.py --release
```

Use the emitted SHA256 without replacing the source index. Check-only:

```sh
.venv/bin/python -I -B build/continuation-next-iteration-v1/runner/run_owned.py --source-index-sha256 <emitted-sha256>
```

Single released run, with a fresh never-created cache path:

```sh
.venv/bin/python -I -B -X pycache_prefix=build/continuation-next-iteration-v1/absent-root-launch-cache build/continuation-next-iteration-v1/runner/run_owned.py --source-index-sha256 <emitted-sha256> --execute
```

A preexisting output or cache prefix refuses. No actual source index or run output was created during preparation.

## Generated controls

`controls-01.json` binds the exact executing runner and records 15 passing controls, including current saved-JSON joins and a real isolated import/Adam-constructor bootstrap under the actual input boundary. It performed zero acquired-array/checkpoint reads, native calls, model forwards or optimizer steps. Root owns actual dispatch.
