Explicit terminal-microstep parent closure: same fixed18 comparison

Source only; root schedules controls and one attempt. No automatic retry.
Typed API changes only scheduling:
  CutUnionSpec(exposure_mode='same_step_parent_closure',
               closure_schedule='terminal_microstep_only', original depth/caps)
The macro's actual last microstep uses same-capsule parent exposure closure;
intermediate microsteps retain prior parent exposure. This is uniform for all
actions/tools and does not inspect target labels or evaluator outcomes.
Mode: cumulative_affine_terminal_parent_closure_cut_union_v4.
Evacuated footprint: cumulative_affine_terminal_parent_closure_evacuation_v4.
Witness/volume schemas,8192 witness cap,depth3,source target mass bounds,shaft
obstacle rule,geometry/actions and reward rule remain unchanged.

Primary reference and criterion are exactly the retained b16ffec partial-volume
arm: all13 completions retained; all18 rows/180 planned slots; all nonempty
history checks accepted; zero budget refusals; all6 fine physical target volumes
nonregressing within1e-8mm3 and at least1 strict gain above1e-8; every physical
cap/target mass enclosure/interval holds. The actual criterion function is
byte-identical to the preceding comparison. Only accepted checker sparse unions
reach independent physical clipping. No baseline history is rerun.

All-capsule run02 remains a negative. It completed18 rows in64.8601s with128
started/returned calls, but2 witness-budget refusals lost2 baseline completions.
It is a secondary descriptive reference only. New row summaries report target,
tissue,reward and completion differences versus it, never use it to replace the
original criterion. The interrupted monitor attempt01 also remains intact:
11 calls started/returned before parentSIGTERM. The new report records both
prior progress bindings and139 prior calls plus this attempt's counts. Refused
preview leaf/witness work remains in retained progress. Baseline matrix compute
is a separate retained-reference cost, not silently repeated or dropped.

Reused repaired monitor/lifecycle is byte-identical to comparison-repair-v1.
Only known live writer temporary ENOENT is tolerated; strict final checks and
explicit failure acceptance reset remain. Fresh output is terminal-run-01.
Limits stay240s worker/250s parent,10s cleanup,2GiB sampled group RSS,1thread,
8MiB output,180 native calls/18 audits,50GiB free disk. No cap/depth increase.
Forecast90-220s and3-6MiB, with actual cost/caps recorded and negative retained.
No speed claim from saved unpaired timing. Source remains under100KiB.

Root controls:16 comparison cases include one prescribed coarse row and saved
cost/secondary-reference negatives;12 unchanged monitor-only cases are reusable.
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 .venv/bin/python -I -B -m pytest -q -p no:cacheprovider build/native-cut-residual-diagnostic-v1/comparison-terminal-v1/test_comparison.py build/native-cut-residual-diagnostic-v1/comparison-terminal-v1/test_monitor.py
After root freezes canonical producer/checker and passes focused controls:
  .venv/bin/python -I -B build/native-cut-residual-diagnostic-v1/comparison-terminal-v1/freeze_sources.py --release
Use printed SHA with a never-created parent cache prefix:
  .venv/bin/python -I -B -X pycache_prefix=build/native-cut-residual-diagnostic-v1/absent-terminal-parent-cache-01 build/native-cut-residual-diagnostic-v1/comparison-terminal-v1/run_owned.py --source-index-sha256 PRINTED_SHA256 --execute
No numerical run/test/import of model code has been performed by this author.
