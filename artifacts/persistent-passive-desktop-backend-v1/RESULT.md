# Persistent passive desktop backend

The local desktop bridge now executes the fixed paired generated passive episode, binds its displayed image/support/nominal target and both complete tool geometries, and retains its five actions/six operative frames. Saving, closing the engine and reopening reconstructs the episode from the fixed source before accepting its poses, timing and frame selection. Rewritten saved positions, clocks, dimensions or source bindings are refused. Returned tools remain recorded at their access poses; they do not disappear.

The existing passive v1 episode identity remains `sha256:33cbb8bdea5873acece2461fbf1c992791ff7cae18ba2099c4bc1724cd76717a`. Legacy atomic stroke replay is unchanged. The new renderer workflow is still being integrated; this result establishes the backend path, not a completed live UI check.

Canonical integration: **50 tests passed in 31.45 seconds**:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m pytest -q tests/test_passive_v1_transport.py tests/test_workspace_passive.py tests/test_workspace_persistence.py tests/test_persistent_passive.py tests/test_desktop_development_episode.py
```

The source index preserves the original three-file transport handoff. Canonical transport tests remove the staged import-path override and add live bridge, invalid request and cancellation controls. Saved-workspace integration adds fresh engine replay and tamper controls. The desktop agent's first staged workspace invocation loaded the old canonical module and failed at legacy `replayFrames`; its correctly isolated five tests then passed, followed by the canonical run above.

A broader historical bridge suite returned **37 passed, 27 failed and 8 setup errors in 14.07 seconds**. Its old generic synthetic/training fixtures expect operations now excluded by the existing source-admission policy. A direct committed-597eae7/current-source comparison confirms identical rejection codes and messages for both legacy entry points; see `legacy-gate-baseline.json`. These failures are retained, and the whole historical suite is not claimed green. No production admission rule was weakened.

All new execution uses generated arrays and declared software timing. No acquired patient, checkpoint, fit or training run occurs. No physical force, deformation, neurological safety, sensor accuracy or learned improvement is established. Pair clearance uses the primary full-tool checker; the separate geometry check covers tissue motion and does not independently certify tool-pair contact.
