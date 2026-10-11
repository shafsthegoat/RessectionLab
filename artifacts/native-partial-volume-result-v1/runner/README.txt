Fixed 18-row partial-evacuation comparison

Status: root's combined generated batch passed 168 tests in 4.80 seconds,
including all four runner controls and one prescribed coarse native row.
Root alone dispatches the single frozen matrix; no automatic retry.

From repository root, after the source freeze:
  .venv/bin/python -I -B build/native-partial-volume-v1/runner/freeze_sources.py --release
The command exclusively creates source-index.json and prints its SHA256.
Use that exact digest in the following command (the cache path must be absent):
  .venv/bin/python -I -B -X pycache_prefix=build/native-partial-volume-v1/absent-parent-cache build/native-partial-volume-v1/runner/run_owned.py --source-index-sha256 PRINTED_SHA256 --execute
Omitting --execute performs only source/reference verification. The immutable
config's execution_released=false records its preparation state; the separate
root-created source index is the actual one-run release. Do not overwrite it.

The run preserves the previous 18 scenes, ten prescribed action slots per row,
tools, 0.0625 mm microsteps, conservative whole-parent shaft obstacles and
whole-cell reward. Only the explicit v2 evacuated-volume declaration is added.
Public float32 parent target fractions become outward-enclosing rounding-bin
mass intervals, with no analytic target plane supplied to the native engine.
The unchanged baseline action/reward projections are compared after excluding
only new volume fields, bound state/history identities, and elapsed timing.

Every nonempty execution history is independently checked in memory before
discarding its bulk proof history. Only the checker's accepted final whole-cell
mask and sparse partial-leaf ledger enter the independent plane-volume evaluator.
The saved sparse geometry and parent-mass bounds permit a later independent
physical-volume reconstruction. Planes z>=0 and z>=2.8 are evaluator-only.

Limits: one child, one thread, 170 seconds work plus up to 10 seconds cleanup,
2 GiB sampled process-group RSS, 8 MiB total output, 180 native action calls,
18 independent audits, and a 50 GiB free-disk floor. Output must be a fresh
build/native-partial-volume-v1/run-01. Partial/capped/failed evidence is retained.
Expected output is a few MiB: scalar report, sparse geometry per accepted row,
one bounded console and lifecycle receipts. No full proof histories or arrays.

Resource forecast before execution: the retained union matrix took 99.42 seconds,
including 34.57 seconds of independent audits. The added volume reconstruction
and clipping may bring this run to roughly 120-160 seconds; this is an estimate,
not a guarantee. Whole-parent classifications avoid expanding all certified
leaves. Explicit visit/clip bounds and the cancellation callback protect the
remaining worst case; reaching any cap is a negative result, not a reason to
extend the frozen budget. Timing comparisons remain descriptive.

The useful criterion requires unchanged old projections and completion outcomes,
no loss of any prior default/union completion, at least one completion above the
default baseline, accepted audits, physical lower/upper consistency, independent
parent target-mass enclosure, aggregate target containment, and strictly tighter
volume bounds in every one of the six 0.25 mm rows. It is a generated geometric
representation result, not biological evacuation, clinical safety, learned-policy
improvement, or measured-mechanics validation.

Focused runner controls (already included in root's passing batch):
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 .venv/bin/python -I -B -m pytest -q -p no:cacheprovider build/native-partial-volume-v1/runner/test_runner.py
The separate evaluator's 12 controls also passed before this matrix release.
