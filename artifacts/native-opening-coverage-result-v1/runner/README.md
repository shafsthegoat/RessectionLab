# One generated opening decision pair

Preparation only. Root owns the short controls batch, source freeze and single run. The exact six-cell `make_native_opening_task` uses the declared terminal-microstep parent-closure model and exact binary public target mass. No previous 18-row matrix is executed or loaded. The only changed search setting between the two arms is retention mode.

Both arms: horizon8, beam2, at most200 planning transitions, lazy planning,120 seconds including initial inventory and chosen-route replay/checking. The search API still receives its exact120-second bound; an outer per-arm cancellation bound includes preparation and replay as well. Both begin from fresh equal generated state. The first arm is return-only; the second reserves one certified-evacuation retention lane. Fixed order is descriptive, not a speed claim. All preview calls, including initial/query inventories, failed previews, replay and the fixed positive route, are counted. Inclusive nested timings are explicitly non-additive. Search state-verification time and task/snapshot integrity time are reported separately. No actor, optimizer, fitter, model weights or acquired images.

Each selected route must independently pass full-tool/history/terminal-closure/partial-volume checking. The two tiny full histories remain in the report. The checker independently reconstructs every committed macro's partial volumes. Since each generated parent target fraction is exactly0 or1, its target interval collapses; source target and reference target coincide only in this declared fixture. “Outside supplied target” is not a clinical healthy-tissue or injury claim.

Useful gain requires positive target removal and return for both selected routes, complete search layers without time/call truncation, accepted geometry, equal source/model/reward/candidate identity, no whole-cell target loss/outside increase/return loss, no certified target loss/final outside evacuation increase, and strictly fewer **search model transitions** or lower preparation outside volume. Preparation is certified outside volume strictly before the first action removing a positive whole-cell target amount. If no target action exists it is final outside volume. Final outside volume is separately constrained so moving preparation into the target-bearing action cannot hide removal. Numerical comparison allowance is fixed1e-10mm³/reward; transition counts are integers. Exact ties and completed STOP-only outcomes are negative; truncated or incomplete searches are inadmissible. This is a bounded beam comparison, not proof of horizon-global optimality.

The sole positive route uses the existing public inventory: short-wide-opener at[4,4,1], long-narrow-cutter at[4,4,5], STOP. Unique feasible IDs are resolved in order. It must remove both target cells, earn positive return and pass the independent checker. Its costs are charged separately in the same worker.

## Root commands

From the repository root, in the single granted control slot:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 .venv/bin/python -B -m pytest -q build/native-opening-coverage-v1/runner/test_runner.py
```

Seven pure/fake controls plus one initial-inventory/STOP/independent-volume interface control (12 generated previews; no search or trained actor). Expected under three seconds. Root can include them in the existing bounded test harness. Then freeze only once:

```sh
.venv/bin/python -I -B build/native-opening-coverage-v1/runner/freeze_sources.py --release
```

Use its printed exact index digest in the one owned command (replace `INDEX_SHA256`; choose a never-created parent cache prefix):

```sh
.venv/bin/python -I -B -X pycache_prefix=build/native-opening-coverage-v1/absent-parent-cache-01 build/native-opening-coverage-v1/runner/run_owned.py --source-index-sha256 INDEX_SHA256 --execute
```

Without `--execute`, the launcher only verifies source pins. `run-01` must not exist. No automatic retry. Parent wall300seconds includes10seconds reserved cleanup; work stops at290seconds. Worker is single-threaded with2GiB sampled RSS cap; output≤2MiB, total shared build≤4MiB, log≤64KiB, disk floor50GiB. Source freeze includes the unchanged tested `OwnedChildExit` helper and exact Darwin sampler. Neither is copied or modified. It accepts an observed exit only with exact child/reap/membership evidence; unknown live RSS or unverifiable cleanup still fails. RSS is sampled and is not a hard bound on between-sample spikes; whole-worker self-peak is also retained.

Expected work: at most400 search transitions plus ≤16 chosen-route transitions and3 fixed-positive transitions; each inventory has12 declared tool/cell slots. Rejected preview work counts. No native runtime forecast beyond the fixed240-second paired allowance is claimed. The six-cell geometry and≤8-step histories should fit comfortably within2MiB, but a cap refusal is retained rather than increasing limits. Full search failure/partial costs and cleanup evidence remain in `report.json`, `progress.json`, `worker.log`, `launcher.json` as available.
