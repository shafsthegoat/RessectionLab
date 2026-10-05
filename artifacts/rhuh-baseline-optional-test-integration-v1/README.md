# Optional RHUH test integration

Six function-local `pytest.importorskip("sklearn")` guards cover nine test cases: two owner functions (including four parametrized failure cases) and four independent-review functions. All other tests remain active. No production source, protocol, adapter, runtime installation, completed experiment, or historical independent-review artifact changed.

The original root-run failure is retained separately in `receipt.json`: 90 passed, 9 failed in 4.98 seconds, all due to missing `sklearn`. This is root-reported evidence from tool session 43441, which has no durable raw log; it was not replayed or assigned an invented log hash.

The same six-file scope now yields:

- Default environment, `PYTHONPATH` unset: **90 passed, 9 skipped in 4.60 seconds**.
- Isolated optional runtime enabled: **99 passed in 5.36 seconds**.

Skipped cases are **not validation**. The optional-enabled invocation is required evidence for the scaler, convergence, runtime-origin and fit-cap controls. Its only fitting is 24 small analytical logistic fits; there are no clinical refits or patient-value loads. This integration test does not reproduce the completed clinical experiment.

From the repository root, use the same command in each environment:

```sh
.venv/bin/python -m pytest -q -rs tests/test_pat25_access_diagnostic.py tests/test_pat25_access_diagnostic_review.py tests/test_rhuh_outcomes.py tests/test_rhuh_outcomes_review.py tests/test_rhuh_preoperative_baseline.py tests/test_rhuh_preoperative_baseline_review.py
```

For the optional run, prefix that command with `PYTHONPATH=build/rhuh-baseline-runtime-v1/site-packages:src`. Both recorded runs set the five numerical thread variables to `1` and disabled bytecode writes. The receipt contains full output, commands, source hashes and before/after preservation checks. The isolated dependency tree is local and ignored; this does not claim a clean-clone runtime installation.
