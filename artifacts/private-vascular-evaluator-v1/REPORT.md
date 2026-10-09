# Independent vascular evaluator integration review

Decision: **GO for the reviewed tracked-module relocation and generated-only regression set.** No blocking findings. Patient/model admission, model execution, native mechanics, network access and clinical use remain outside this approval.

`src/resectionlab/private_vascular_evaluation.py` is byte-for-byte identical to the independently reviewed ignored implementation: SHA-256 `30e9efaac9d0f4e311bedf9075362fcd488f1b31f76a3938f4224736d39e2eb7`. Its unchanged `Path(__file__).resolve().parents[2]` still resolves to the repository. The original source review is `build/private-vascular-evaluator-independent-v1/REPORT.md`, SHA-256 `b8624bd0b092e4cf4785b6e8e3430a19ef8e1a21d40714ddd2e01aff861a758b`; all its scope limitations continue to apply.

## Reviewed test-only changes

The tracked author tests have SHA-256 `ce3573a4f381baaf01b50ff0346ef883acf814a8193e14e1602c5821f431cec9`. The full relocation diff is saved in `test-relocation.diff`, SHA-256 `35b753ccfa37fd5fbf8189b604a6deb6596eaca6df39ee5cb81f2b4ea4007f69`.

- Repository ROOT changes from `parents[2]` to `parents[1]`, and the implementation is imported through `resectionlab.private_vascular_evaluation` instead of the ignored file loader.
- The existing permanent Python audit hook is gated by a module-local flag enabled by an autouse fixture and reset in `finally`. Thus unrelated test modules do not inherit an active patient/model suffix guard. The vascular tests retain the same forbidden-payload checks.
- A module-local `tmp_path` fixture creates a temporary directory under the actual repository `build/` and cleans it afterward. This exercises the unchanged build-only production boundary without requiring an external pytest basetemp setting.

No assertions, generated geometry, reference fixtures, tolerances, or implementation behavior changed. The independent test copy now imports the tracked author fixture and asserts the actual module path is `src/resectionlab/private_vascular_evaluation.py`; its 14 original controls remain intact. It also asserts Torch is not imported during those controls. Review writes are confined to this ignored directory.

## Result and exact regression scope

One command ran the tracked 30 author tests, 14 independent controls and 81 focused regression cases: **125 passed in 10.45 seconds**, process exit 0, harness elapsed 10.619315499905497 seconds. The exact argument list and resource environment are preserved in `run-receipt.json` and `run_review.py`. The command used `.venv/bin/python -B -m pytest -q`, no pytest plugin autoload/cache, one numerical-library thread, and a fresh ignored basetemp.

The selected files were:

1. `tests/test_private_vascular_evaluation.py`
2. `build/private-vascular-evaluator-integration-review-v1/test_independent.py`
3. `tests/test_research_estimate_planning.py`
4. `tests/test_research_estimate_strategy_record.py`
5. `tests/test_native_spatial_evaluation.py`
6. `tests/test_functional_events.py`
7. `tests/test_functional_events_review.py`

The actual strategy-record filename was verified before execution. `tests/test_limited_observation_boundary.py` was deliberately excluded because it imports Torch and includes policy forwards; this task prohibited model execution. Its reusable source interfaces were reviewed in the original source review. No patient payload, model, native solver or network operation was used by this integration review. The independent audit guard rejects subprocess launches, network connections and patient/model array payload extensions inside the test process.

All 86 snapshotted source/test files remained unchanged across the run. `source-before.json` and `source-after.json` both have SHA-256 `462a30433db10ef6c29daeb66cfb8a462f09bd7437a7c987a30a3659c76b91cc`. The observed post-run HEAD was `4669c7a70e64be113b9f464734d60b27acee5265`; byte hashes, not this contextual HEAD, bind the reviewed files.

| Evidence file | SHA-256 |
| --- | --- |
| `test_independent.py` | `5ca33a8cd84b45cc021c178b9abb35cadcacda51a096e870825d6f2c12d94440` |
| `run_review.py` | `bafa1167706124e38c9937347e382b16060e868a24ad52fd2775cfadd4d765a8` |
| `run-receipt.json` | `44b6f5d855c433f44bde4ec5cad8d06c88e1695015edaff076b1b0e86d957552` |
| `pytest-output.txt` | `071776b736fe1bb427a64b1b4caf2d968663fabbfe2c1d26db3d23f693ad0675` |

This approval does not enlarge the original generated-only scope or turn cooperative checkpoints into external wall/RSS containment. Further implementation changes require a new review.
