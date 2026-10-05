# Native axis RAW one-update pilot

[RESULT.md](report-v1/RESULT.md) and [cost-summary.json](report-v1/cost-summary.json) report the completed attempt. One actor update completed, but initial and updated selection returns both equal 441.60; the earliest initial checkpoint was retained. All six episodes completed and have independent native receipts, with three exact-history checks shared where appropriate.

The run used the tested immutable `f3a0591` archive and the committed prospective declaration. Its [execution baseline](../../validation/native-axis-pilot-public-v1/execution-baseline.json) binds the full source and public derivative. Both launcher and worker authorities completed. The run used only optimization and selection worlds.

The [independent final audit](../../native-axis-pilot-result-audit/audit-02.json) checked source cells, all 18 recorded forwards from saved tensors, the 12 deterministic first-argmax decisions, saved checkpoint selection, and all receipt hashes. It checked eligibility of the six stochastic choices without replaying RNG. [Verification](../../native-axis-pilot-result-audit/verification.json) records 35 passing adversarial tests; the independent gzip-only audit preserves every scientific field.

Three large raw JSON files are retained unchanged locally and have lossless gzip copies for version control. [compression-roundtrip.json](compression-roundtrip.json) verifies every byte and JSON value. [gzip-only-report-verification.json](gzip-only-report-verification.json) confirms that the source-derived report is byte-identical when those raw files are absent from a private copy. The report and independent audit refuse disagreement between raw and compressed copies.

Reproduce the report with `scripts/report_native_axis_pilot.py` using this attempt and its execution baseline. The exact executable source snapshot, patient bundle and tensor checkpoints remain local; their identities are retained in the tracked receipts. The reporting script performs no simulator calls, policy forwards, random draws or gradients. The independent audit uses direct NumPy operations on saved tensors and does not invoke the simulator, policy class, optimizer or RNG.

This is one previously studied development case with unreviewed structural support and hypothetical access. Deterministic seed repeats do not establish clinical uncertainty or efficacy. Setup, online, logging, clone, validation and publication scopes are distinguished in the report; unmeasured components are not assigned invented costs.
