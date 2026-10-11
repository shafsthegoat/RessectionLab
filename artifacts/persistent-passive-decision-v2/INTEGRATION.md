# Canonical generated decision integration

Exact repaired source `ae6ce9ad09c6f95a419defaf6bb97d8ee17fdb072d2e6109fac1352ea490128c` is integrated into `src/resectionlab/persistent_passive.py`. The author test imports the canonical module instead of the staged snapshot. **113 tests passed in 10.40 seconds**, covering the new decision, unchanged v1, public information boundary, desktop transport, native task and geometry:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m pytest -q tests/test_passive_decision.py tests/test_persistent_passive.py tests/test_hybrid_information_boundary.py tests/test_passive_v1_transport.py tests/test_native_spatial_task.py tests/test_geometry.py
```

Independent review preserved the earlier state defects and exact before/after source hashes; both focused corrections pass. This generated endpoint-uncertainty test is separate from the completed real-patient landmark evaluation and the still-unexecuted learned proposal pilot. No policy fit, patient action, measured mechanics, desktop v2 admission or performance improvement is claimed.

The compact archive stores the state-repair patch instead of duplicating the complete module. The integrated source has the recorded successor hash; reversing `independent/state-integrity-repair.patch` on a disposable copy reconstructs the earlier hash. Original full snapshots remain in the local build handoff. Independent scripts take the chosen source file as their first argument; their historical default build paths are not required.

Archived independent scripts discover the repository root from their nested archive location and default to the canonical module. Only this path setup differs from the original build scripts; numerical controls are unchanged. Copy the scripts into a disposable directory under the repository before reproducing so saved receipts are not overwritten.
