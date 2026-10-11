# Reproduce the retained exposure experiments

Use the existing project `.venv`; no new dependency or dataset is needed.
`bc78bbb` contains all-capsule closure; `7ed4ef8` contains its optional
terminal-only successor. Defaults remain unchanged. The archived source
indexes bind exact executed source/reference bytes. This result's root archive
index verifies copied payloads and points to local sparse geometry; it does not
pretend those omitted local files are distributed with Git.

In a fresh checkout only, restore this archive's subdirectories at the matching
paths under `build/native-cut-residual-diagnostic-v1/`. Restore the previous
`artifacts/native-partial-volume-result-v1/` archive according to its
`REPRODUCE.md`, including its independent plane evaluator, retained original
report/parity correction and shared `build/goal-conditioned-policy-v1/` runtime.
Restore prior cut-union references from `artifacts/native-cut-union-result-v1/`.
Never overwrite an existing attempt or its source index.

Each runner's `README.txt` describes its original bounded command. Its config
records source-preparation `execution_released: false`; the separately frozen
source index contains the actual execution release. Preserve that distinction.
Fresh outputs and exact source bindings are required. A future experiment must
have a fresh source freeze/output, retain all costs and negatives, and validate
the prospective supervisor fix; do not silently edit a historical source index
or adopt a new result as the old attempt.

The canonical model/source controls are in `tests/test_native_leaf_exposure.py`
and `tests/test_independent_leaf_exposure.py`, with broader native geometry,
volume, evaluation, runner and monitor regressions recorded in root logs.
Original failed fixtures and both successful batches are retained. Test-source
commits, reports and indexes provide the authoritative versions.

The independent v4 auditor at `terminal-validation/audit_saved_matrix.py` checks
canonical bytes using the executed Git commit, so unrelated later additions
do not require rewriting prior source. It authenticates all saved report,
launcher, baseline and geometry hashes before accepting arithmetic. Running it
on this original workspace requires its locally retained sparse geometry and a
fresh output filename; use `--help` for its exact required digest arguments.
It deliberately distinguishes saved scientific acceptance from failed resource
supervision. A clone containing only compact Git artifacts cannot rerun the
sparse-geometry audit without restoring or regenerating those local inputs.

The plot is regenerated from the scalar terminal report with:

```text
.venv/bin/python -I -B artifacts/native-terminal-exposure-result-v1/plot_result.py \
  artifacts/native-terminal-exposure-result-v1/terminal-run-01/report.json \
  /a/fresh/local/output.png
```

All scenes and interactions here are generated. This archive supplies neither
patient outcomes nor measured surgical forces.
