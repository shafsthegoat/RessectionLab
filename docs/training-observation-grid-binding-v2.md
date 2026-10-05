# PAT05 exact task-grid binding correction

This is a wrapper/schema correction, with no geometry, tolerance, tool, support, objective, horizon, patient-role or representation change. The V1 coverage attempt remains incomplete: PAT05 failed after its initial previews and has no representation result. Its actual full grid record was not saved, so the historical record cannot establish that attempt's numerical equality retrospectively.

V1 compared a runtime record containing 20 fields with an inherited declaration containing eight. The historical 20-field record projects exactly onto all eight original values. V2 keeps that original member intact and adds a separate `complete_native_grid_binding`, authenticated by:

- `artifacts/pat05-real-geometric-learning-v1/receipt.json`, SHA-256 `fdc575e6695a7f949f65de93e7165f7833b514fcd193094a24ac3e0f94e7ee77`;
- JSON pointer `/initial_task_metrics/native_grid_reconciliation`;
- canonical sorted compact JSON SHA-256 `a9b1a2f220e2a0137e16aa31b5e59146ea0c54d8d3b5c2fd7242764d2fb4005a` for the complete record.

The [runner](../scripts/diagnose_training_observation_coverage.py) authenticates these saved bytes, the exact 20-key schema, the original eight-field projection and the image/support hash joins before any decoder. Matrix-only affine hashes remain distinct from the composite structural-frame hash. Runtime checking compares all 20 JSON values/types exactly, plus the unchanged objective and horizon. There is no approximate comparison or ignored extra field. Complete expected/actual records and changed/missing/extra field lists are retained in `failure.binding_mismatch` on failure. The historical grid receipt also joins the existing before/after original-record closure.

The runner now names `fixed-training-observation-coverage-v2`, its V2 release schema and the default `training-observation-coverage-v2.json` declaration path. V1 declarations, release, archive and failed outputs are untouched. No V2 declaration or execution release is created by this preparation; root must freeze reviewed source and explicitly release any later attempt.

The 47 [owner controls](../tests/test_training_observation_grid_binding.py) passed using small saved JSON and fake task/decoder objects: every grid field, missing/extra/type changes, all eight original fields, receipt/pointer corruption, image/support mismatch, objective/horizon drift, predecode refusal and durable failure reporting. The old 20-versus-eight failure is explicitly retained as a negative assertion. No patient arrays, native previews, policy, training or rerun were executed.

Independent review added 75 controls. The combined owner, independent and existing
runner guards passed **155 checks in 4.73 seconds**, with no patient experiment
rerun. Source closure must be frozen from a committed checkpoint before any V2
declaration; unrelated in-progress package modules must not be sealed accidentally.
