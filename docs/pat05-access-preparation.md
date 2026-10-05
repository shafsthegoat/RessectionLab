# PAT05 canonical access metadata

This adapter joins two existing historical schemas without changing either record or selecting another access. `load_pat05_historical_records()` reads only three hash-pinned JSON files, each bounded to2MiB:

- `artifacts/pat05-real-spatial-profile-v3/nominal64/declaration-input.json` supplies the historical access derivation and selected geometry.
- `artifacts/pat05-real-geometric-learning-v1/declaration-input.json` supplies the executed physical task definition and original eight-field grid projection.
- `artifacts/pat05-real-geometric-learning-v1/receipt.json` supplies the executed source/model identities and complete twenty-field grid record.

`canonical_pat05_access_metadata(task, records)` requires an already authenticated, initial PAT05 `NativeSpatialTask`. Upstream bundle, CaseData and support validation remain mandatory. It reauthenticates the immutable record bytes, checks the profile-to-learning source/access joins and executed source/model identities, and reuses the corrected observation diagnostic's `pat05_complete_grid_binding` and `verify_pat05_task_binding`. All twenty grid fields, objective and horizon must match exactly. The profile's objective/adapter definitions differ historically from the learning definition; they are not substituted for the executed task.

The adapter calls the unchanged shared `derive_access` on the permitted binary nominal annotation, unchanged observed support and original source affine. It compares the complete historical centroid, representative, selected boundary and depth; all six ordered axis/sign/distance triples; and the exact selected access. The task must still use that original access. Final grid/source/state checks precede returning recursively immutable canonical metadata.

The returned six per-exit boundary coordinates are labeled **newly rederived**, because the historical profile omitted them. The `rule` and `selection_uses_reward_or_native_preview` keys are labeled current shared-rule metadata. Historical `procedure`, track and source-grid-only claims remain separately preserved. The adapter never inserts new fields into the historical files, relaxes equality, chooses another access or creates a task/inventory. It does not load images, checkpoints or patient arrays; future authorized callers supply their existing task.

The focused controls use pinned metadata and mocked typed task boundaries. They test every historical exit triple and selected-access component, all twenty grid fields, source/model pins, final-state integrity and immutable output. Their positive mocked result is not evidence that actual PAT05 source-array rederivation has run. Two initial test-collection errors were incorrect test-only `RewardSpec` import paths; the correct existing simulation import fixed collection without a production change.
