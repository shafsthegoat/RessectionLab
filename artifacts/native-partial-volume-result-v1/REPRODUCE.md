# Reproduce the bounded generated result

Use the existing project `.venv` at canonical commit `52b2a62`. No new dataset or
dependency is needed. Restore the following archived files to their exact ignored
workspace paths in a fresh checkout; do not overwrite existing run outputs:

- This archive's `runner/`, `independent-review/`, and `runner-repair/` map to
  `build/native-partial-volume-v1/` with the same subdirectory names.
- This archive's `runtime/` files map to `build/goal-conditioned-policy-v1/`.
- The existing `artifacts/native-cut-union-result-v1/` copies of `probe_union.py`,
  `declaration.json` and `run-01/report.json` map to `build/native-cut-union-v1/`.

Other dependencies are already tracked. `runner/source-index.json` binds every
executed module and reference. The archived config's `execution_released: false`
records source-preparation state; the separately frozen source index contains
the actual root execution release. Preserve it to reproduce the exact attempt.

`runner/README.txt` contains the bounded command. With the restored index already
present, use its hash directly with `run_owned.py --execute`; do not regenerate
or overwrite the index. The owned launcher requires a fresh `run-01` directory,
one thread, 8 MiB output, 2 GiB worker RSS and 170/180 s deadlines. A reproduced
original run will also reproduce the original scalar-type comparison defect.
The saved-only correction is bound to this original report's exact bytes; it
cannot be applied unchanged to a new run with different timings. The proposed
normalization patch is retained for a future newly frozen run, not retroactively
applied to the executed source.

For saved-only evaluation on this original workspace, keep the original geometry
directory, report and launcher. Choose a fresh output filename, for example:

```text
.venv/bin/python -I -B build/native-partial-volume-v1/independent-review/audit_saved_result.py \
  --report-sha256 c9265daaf186b025403ee499a333f0e2811feab66f65e10a98a3d97f17300ee9 \
  --launcher-sha256 d9be4f777fdf07567c5aa9ce4ed45a1c4ac0c96ae5b071b089d4150fd4ec177a \
  --output build/native-partial-volume-v1/independent-review/audit-rerun-03.json
```

The artifact plot uses only the archived scalar report and can be regenerated
with `plot_result.py`.

Do not substitute other patient inputs, unlock evaluation patients, or infer
physical cutting validation from this generated geometry experiment.
