# HBE v5 stream-local cache: independent generated-only check

Status: **source-only numerical/performance control**, not an HBE specimen fit or patient result. Reviewed commit: `fafbe50923f4dc79ce52b6ffb6111c41d6278fc7`.

| Committed file | SHA256 |
| --- | --- |
| `scripts/mechanics_hbe_v5_frame.py` | `8431c7ae293befdf75bbccc54ce3481ddbd2e42b70daf1b6f826be96bf68e0d9` |
| `scripts/mechanics_hbe_v5_stream.py` | `a353ca0d96d1c31b80302ce3239796e622b1b6adc15c97bf318ffb9246c3b70a` |
| `tests/test_mechanics_hbe_v5_stream.py` | `b72ccf0c830df8c62b5db83479f513a271d8618f0cdf63ca63f8dff5f64ef6ca` |
| `docs/hbe-v5-stream-evaluator.md` | `b460cede1e76cb996d9d6300c428a5bbd5548d4a54d2d7a1fb1cb76d15a97683` |

On macOS 26.6, arm64, Python 3.12.14, NumPy 2.5.3, I ran `.venv/bin/python -m pytest -q tests/test_mechanics_hbe_branch_calibration_v5.py tests/test_mechanics_hbe_v5_source_bindings.py tests/test_mechanics_hbe_v5_frame.py tests/test_mechanics_hbe_v5_stream.py`: **104 passed**. With `PYTHONPATH=tests .venv/bin/python`, I used `generated_run` from `test_mechanics_hbe_v5_stream`, parsed each node/element stream through `_iter_data_records`, and compared the complete output dictionaries from `evaluate_prepared_frame` and the standalone `evaluate_generated_frame`. They were **exactly equal in all 18 comparisons**: tension and compression; full S60, reflected-half S60, and reflected-half S120; initial, middle and endpoint frames (`0/30/60` or `0/60/120`). This is exact output equivalence for generated fixtures, not numerical convergence against an external solution. Independent mutation checks also found that caller-owned mesh/mapping edits after preparation did not change the prepared result, relevant geometry arrays rejected writes, and changed run/schedule contracts were refused.

For a separate single-frame profile, I read only the hash-bound historical N36 *geometry* manifests: full mesh `6ba012cc042ec0a537a6c61283d444488ccf69de0bc9c385f319a827d69c8e74`, half mesh `8e7ad0e636151e007ab9fd8af0d043abfcf889b960c53e726766126691c3c075`, and reconstruction wrapper `9e0265495f47ece2d315746f79f40e71f717e0720f6f1621fa9c7f9d2d9eba8f`. For `compression:N36:S120:reference`, I generated a rest-only frame (`current = rest`, zero displacement/reaction/stress/energy, logged `J = 1`) and timed calls with `time.perf_counter()`; `resource.getrusage(RUSAGE_SELF).ru_maxrss` recorded process peak RSS. Preparation took **3.393 s**; three prepared-frame evaluations took **1.030, 0.953 and 0.957 s**; one standalone evaluation took **4.300 s**. The prepared and standalone dictionaries were exactly equal. Process peak RSS was **1,124,843,520 bytes**, with both prepared and standalone models resident; this is not a steady-state stream memory measurement. These timings are local observations for one generated rest frame, not a 121-frame runtime or native FEBio budget.

No native solver output, measured HBE response, patient anatomy, observed surgical action, or clinical outcome was opened for this check. The stream's physical validation and native-run release remain closed; full-stream throughput, memory and physical fidelity still need independent evidence.
