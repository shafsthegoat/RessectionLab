# Direct retention attribution pair

One prospective pair on the identical generated six-cell opening task: existing `return_plus_opening_depth_volume_v1` versus `return_plus_certified_evacuation_v1`. This checks whether the earlier benefit over `return_only` requires the new certified-evacuation retention signal. No additional task, sweep, training or data.

All five Python files are byte-identical copies of the root-tested original runner. Only config `arms` and `output_directory` differ. Horizon8/beam2/200calls/120seconds per arm, lazy planning, terminal-microstep parent closure, public information, target, reward, action inventory, independent history checking, positive control, outcome gate and complete cost accounting are unchanged. Ties remain negative. The original run, source index, supervision receipt and scientific report are untouched. Original preparation volume worsened3.625→4mm³; its useful criterion passed through fewer transitions, not a preparation-volume improvement.

Root alone freezes and executes once, from repository root:

```sh
.venv/bin/python -I -B build/native-opening-coverage-v1/ablation-runner/freeze_sources.py --release
```

Use its printed digest as `INDEX_SHA256`:

```sh
.venv/bin/python -I -B -X pycache_prefix=build/native-opening-coverage-v1/absent-ablation-parent-cache-01 build/native-opening-coverage-v1/ablation-runner/run_owned.py --source-index-sha256 INDEX_SHA256 --execute
```

The fresh output is `build/native-opening-coverage-v1/ablation-run-01`; no automatic retry. Parent300seconds/work290seconds,2GiB sampled group RSS plus worker self-peak, one worker thread,≤2MiB output and shared≤4MiB whole build. Existing original results remain counted in that shared cap. Exact lifecycle helper and sampler are reused by their unchanged fixed hashes. No new tests, imports or native execution occurred during this preparation.
