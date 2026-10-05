# Full-history comparison V1: harness failure

The single authorized attempt failed after the first scalar audit passed. Batch and the second scalar phase did not start. This attempt provides no complete-history performance comparison or speedup result.

The frozen runner compared the in-memory `NativeRemovalAudit.to_dict()` certificate, whose empty `failures` field is a tuple, with the historical JSON certificate, whose equivalent field is a list. Python dictionary equality rejected that representation difference. The saved JSON certificates and their exact canonical JSON bytes agree; no numerical tolerance was used in the diagnosis. V1 remains `failed_or_incomplete` and will not be retried.

The reconstructed source, configuration and 97-microstep history matched the original certified 20-degree, 0.125 mm numerical-control row before timing. The scalar audit recorded 97 active-contact scans, 97 committed prefixes and 679 scalar cell queries, with a feasible complete-tool/frontier certificate and restored trace hooks. Its 43.897 s includes 0.590 s of trace capture; trace export took a further 0.930 s. Setup before that audit took 6.879 s. These observations do not establish batch performance and are not a direct comparison against the earlier uninstrumented audit.

The parent launcher ended with exit 1 after 51.898 s (outer wrapper 51.949 s), with sampled and worker peak RSS both 300,793,856 bytes. The original 140 s cooperative, 150 s parent and 3 GiB limits were unchanged, with one worker and numerical thread settings of one. No kill, retry, cap increase, patient case or learning update occurred. RSS sampling can miss brief excursions; the thread environment is not a CPU-affinity assertion.

Execution used the full immutable Git archive of `2e6ea98c60e0871f97efd1cfeb1414d7667bef50`. All 3,762 tracked archive files and the archive tar hash remained unchanged; added runtime files were Python bytecode only. The archived worker asserts its numerical module import paths before setup. This is source-bound evidence, not an independent process-memory capture of imports.

The original receipts are preserved in [attempt-01](attempt-01/). [failure-diagnostic.json](failure-diagnostic.json) reproduces the representation failure using saved data and the frozen validation function only; it performs no geometry replay. [archive-verification.json](archive-verification.json) and [execution-baseline.json](execution-baseline.json) bind the unchanged archive and launch. [artifact-index.json](artifact-index.json) records file sizes and hashes.

A separately declared V2 may repair certificate comparison through exact canonical JSON normalization and test the actual dataclass serialization roundtrip. That would be a harness-only change, with the scalar evaluator still the default. V2 has not been executed.
