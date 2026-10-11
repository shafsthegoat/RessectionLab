Same-capsule parent exposure closure: fixed 18-row comparison

SOURCE PREPARATION ONLY. Root schedules generated controls, freezes source,
begins output storage, and dispatches exactly one matrix. No automatic retry.

Explicit new API:
  replace(original_cut_spec, exposure_mode='same_step_parent_closure')
The producer's mode is cumulative_affine_parent_closure_cut_union_v3; volume
readout remains the existing explicit EvacuatedVolumeSpec. The checker receives
the complete new cut_union_spec and emits the same independently reconstructed
certified_evacuated_geometry. Only that accepted geometry reaches the frozen
independent physical plane evaluator. No new planes enter the native model.

The baseline is the saved b16ffec partial-volume result and its additive exact
JSON parity certificate. No baseline action, history, or geometry is re-executed.
All18 original scenes, 180 action slots, tool definitions, microstep, source
support, shaft rule, depth, preview limits and reward rule are retained.
Actual removals/rewards/actions may differ because exposure is the changed model.
Each row includes the original compact action/result projection, current action
results, matching fixed-input identities, and volume/reward/completion changes.
The prospective NumPy scalar comparison defect is fixed by finite JSON
normalization; equality is descriptive and is not a success requirement.

Primary predeclared criterion: all13 baseline completions retained; all18 rows
and180 action slots accounted; all nonempty audits accepted; no budget refusal;
all six0.25mm physical target volumes nonregressing within1e-8mm3, with at least
one strict gain greater than1e-8mm3; every row's physical lower/upper relation,
target parent-mass enclosure, cumulative target interval and analytic cap pass.
The target cap is zero unless fixed target action10 commits: preparation actions
1-9 do not reach z>=2.8 for either retained tool. Do not use an unexecuted suffix
as removed geometry. Other tissue-volume changes and original whole-cell reward
changes are descriptive; there is no speed claim from an unpaired baseline.

Root's focused controls (14 cases, one prescribed coarse row; not run by author):
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 .venv/bin/python -I -B -m pytest -q -p no:cacheprovider build/native-cut-residual-diagnostic-v1/comparison/test_comparison.py
Expected controls cost: a few seconds and below512MiB; no fine matrix here.

After controls and canonical producer/checker source are frozen:
  .venv/bin/python -I -B build/native-cut-residual-diagnostic-v1/comparison/freeze_sources.py --release
This exclusively creates source-index.json and prints its SHA256. Use that
digest (and a never-created parent cache path) in the one actual invocation:
  .venv/bin/python -I -B -X pycache_prefix=build/native-cut-residual-diagnostic-v1/absent-comparison-parent-cache build/native-cut-residual-diagnostic-v1/comparison/run_owned.py --source-index-sha256 PRINTED_SHA256 --execute
Omit --execute for source-only preflight. config.execution_released=false is
the preserved preparation state; the root-created source index is the separate
actual release. Do not overwrite a prior index or output directory.

Output: build/native-cut-residual-diagnostic-v1/comparison-run-01, bounded8MiB.
One worker/thread;240 seconds work,250 seconds parent including10s cleanup;
sampled2GiB group RSS;180 primitive attempts and18 independent audits maximum;
50GiB free-disk floor. Full proof history is audited in memory, never saved.
Only final sparse unions, explicit parent bounds, per-row scalar report and
owned lifecycle/work receipts persist. All partial/failure output remains.

Resource forecast: prior partial run96.42s/528.4MB/3.3MB output. Same-capsule
closure adds bounded parent revisits, so plan roughly120-220s and4-6MiB output,
with uncertainty bounded by the declared caps rather than a promised runtime.
Native started/returned calls, attempted coverage work including refusal,
history audits and clipping started/returned counts/times are all reported.
If the frozen cap is reached, retain the negative; do not enlarge depth or cap.

No training, patient reads, acquisition, empirical cutting validation, or
clinical claim. This tests an explicit idealized geometric evacuation model.
