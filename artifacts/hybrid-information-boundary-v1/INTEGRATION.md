# Information-boundary regression integration

The independent generated controls now live in `tests/test_hybrid_information_boundary.py`. Canonical integration with the existing passive and observed-landmark controls passes **47 tests in 4.20 seconds**:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m pytest -q tests/test_hybrid_information_boundary.py tests/test_persistent_passive.py tests/test_observed_landmark_replay.py
```

The original independent runner, source, source pins and result are preserved here; its original `build/` invocation is historical. The canonical command above is the maintained regression. These controls qualify the existing generated APIs, not the pending raw-proposal scorer or registration-conditioned motion extension. They establish no learned, physical or clinical improvement.

The October 11 steering reattachment has SHA256 `97a4c5d1f82ca255a3ebccf363d91a169c514512c51c2906c0ee123761ca2b15`. Its content matches `docs/HYBRID_WORLD_MODEL_STEERING.md` except for the final newline; no replacement specification is introduced.
