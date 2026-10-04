# Nonpatient orchestration fixture repair

The immutable integrated run at commit
`100865f037b37708f4f0d0d5a48950de22ba4e5f` recorded **914 passed, 1 failed** in
171.74 seconds, with four existing diffusion warnings. Its original log remains
at `build/validation/integrated-axis-v1/100865f037b37708f4f0d0d5a48950de22ba4e5f/pytest.log`.
The baseline SHA-256 record here identifies that log; the full failed receipt is
retained by the integrated validation workflow. It was not rewritten or rerun
for this repair.

`test_all_arm_orchestration_uses_explicit_nonpatient_test_scope` substitutes tiny
synthetic geometry and reduced budgets for the registered patient study. It
already substitutes the declaration, source and target validators, but still
called `preserve_declaration_inputs` on the live historical cohort inputs. The
PAT16/PAT20 registry extension correctly caused that production preservation
gate to refuse the old declaration's input hash.

The test now substitutes the preservation side effect too, matching the existing
incomplete-selection orchestration fixture. Actual tiny-fixture trainers,
candidate selection and native audits still execute. A separate regression uses
the real preservation function with isolated test registry bytes: unchanged
bytes are preserved, and even one added whitespace byte is rejected. The copied
registered declaration stays unchanged and passes its original identity check.

Only `tests/test_procedural_transfer_runner.py` changed. `repair.patch` plus the
baseline commit reconstructs the test change; before/after SHA-256 records bind
the exact test bytes and confirm unchanged production runner, historical
declaration and current cohort registry. No clinical study or historical
experiment result was altered.

The focused command completed with **63 passed in 46.87 seconds**, exit code 0:

```sh
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python -m pytest -q \
  tests/test_procedural_transfer_runner.py \
  tests/test_procedural_comparison_review.py \
  tests/test_procedural_learning.py \
  tests/test_procedural_native_transfer_adversarial.py \
  --junitxml=artifacts/validation/procedural-orchestration-fixture-v1/focused-junit.xml
```

`focused-pytest.log` and `focused-junit.xml` retain that execution. This is focused
validation of the repair; it does not replace the failed immutable full-suite
receipt or claim that a subsequent full-suite run passed.
