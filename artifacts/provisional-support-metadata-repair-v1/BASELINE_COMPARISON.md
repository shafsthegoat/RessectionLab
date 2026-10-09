# Exact structural-evidence selection versus pre-edit source

Read-only verification on 2026-10-09. The baseline is commit `936cf2acb13cd01a80e88d8d5791afff9cf846f3`, before the narrow provisional-support repair. It was extracted with `git archive` to `/tmp/resectionlab-preedit-0aYXO4`; no managed worktree, patient payload, native solver, training, or tracked edit was used. The same project `.venv` Python ran both selections. `PYTHONPATH` pointed explicitly to each checkout's `src` and root, and an import-path probe confirmed the baseline loaded `/tmp/resectionlab-preedit-0aYXO4/src/resectionlab/data_policy.py`, while current loaded this checkout's `src/resectionlab/data_policy.py`.

Baseline command, from the isolated snapshot:

```sh
PYTHONPATH="$PWD/src:$PWD" /Users/sayemkamal/.codex/.chatgpt-projects/g-p-6ac1ef0d1fc481919c65bf30beca2549/RessectionLab/.venv/bin/python -B -m pytest -q tests/test_structural_evidence.py tests/test_structural_evidence_adversarial.py tests/test_structural_evidence_io.py tests/test_native_spatial_task_review.py --tb=no -p no:cacheprovider --junitxml=/Users/sayemkamal/.codex/.chatgpt-projects/g-p-6ac1ef0d1fc481919c65bf30beca2549/RessectionLab/build/weight-lineage-boundary-audit/base-structural.xml
```

Current command, from the project checkout:

```sh
PYTHONPATH="$PWD/src:$PWD" .venv/bin/python -B -m pytest -q tests/test_structural_evidence.py tests/test_structural_evidence_adversarial.py tests/test_structural_evidence_io.py tests/test_native_spatial_task_review.py --tb=no -p no:cacheprovider --junitxml=build/weight-lineage-boundary-audit/current-structural.xml
```

**Result in both:** 14 failed, 26 passed, 31 setup errors, 71 collected. JUnit comparison found exactly the same 71 test IDs, pass/fail/error classifications, and exception types; **zero new failures or errors**. All 31 setup errors and 13 direct policy failures had byte-identical JUnit message strings. The remaining failure is the same CLI test raising `subprocess.CalledProcessError` with exit status 2 in both; its message differs only because the isolated script/temporary paths differ. The 31 setup errors stem from the already-disabled `create_synthetic_case` pathway; the direct failures chiefly exercise the already-disabled `import_brain_extraction_evidence` pathway. The narrow repair did not introduce these failures.

This comparison does not make the older tests pass or reopen either policy gate. It only resolves regression attribution for this exact selection. The independently reviewed provisional-support repair remains a separate 178-pass focused current-source result; the frozen limited-input canary remains a seven-pass historical-source replay and intentionally refuses a current-source hash drift.
