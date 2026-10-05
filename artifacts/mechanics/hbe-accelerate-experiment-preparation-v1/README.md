# Prepared fixed Accelerate specimen experiment

Prepared once after the [independent eight-control saved review](../../mechanics-accelerate-controls-saved-review-v1/review.json) accepted the actual repaired runtime outputs. No specimen solver, calibration member or held-out member was accessed.

* [source-archive.json](source-archive.json): fourteen files from committed `6486dbcfb0395a4c5b9df3f140b08141d0d96fd7`, extracted read-only; actual archive 317,440 bytes, SHA `fd3a72a995dd38fc1107b83ec71f1b0911ed060ecac451245637c1559e598d34`.
* [backend-profile.json](backend-profile.json), SHA `c9fafd50be2ed0c824b6ac15e8055b0f916d1331cc3b0e1e7e1d52a1ed1cbfc1`: actual runtime `13c4f60c…`, actual eight-control summaries and committed combined repair. [profile-verification.json](profile-verification.json) records the archived verifier’s successful replay and complete input inventory.
* [preparation-result.json](preparation-result.json) and [raw-input-index.json](raw-input-index.json): eighteen solver-only copies, original mesh/deck associations, unchanged original input hashes. New manifest is `outputs/mechanics/hbe-01-03-poc-v1/backend-inputs-accelerate-csc-v1/manifest.json`, SHA `0fe58b97e29bf8d712463ae0eac09b2102129a6af016216b89eb9f8bcb25217c`. The 37 generated input files and their directories are read-only; original geometry was not regenerated.
* [release-draft.json](release-draft.json), SHA `9f21f130d51911c52faf03bcaba0a8a24c058339451794809a387ea2a4191c10`: all actual source, interpreter, mesh, profile, deck, CSV and prerequisite bindings. It uses `hbe-calibration-release-draft-v1` and `authorized:false`. [draft-refusal.json](draft-refusal.json) confirms the current runner refuses this schema before any execution. A false authorization label alone would not be an enforced gate in the existing release schema, so this draft uses the unsupported schema deliberately.

After root accepts preparation and explicitly authorizes the one experiment, preserve the draft and create a separate final release under `artifacts/mechanics/hbe-accelerate-experiment-v1/release.json`. Change only the draft schema to `hbe-calibration-release-v1`, `authorized` to true, the authorization text/time, and add the actual preparation-review binding. Retain source commit/archive, all scientific fields and actual input bindings. Hash that final file. No old continuation or elapsed-time debit applies to this new runtime experiment.

Run metadata preflight from the frozen source with the final root-relative release path and its actual hash; then the same invocation with `--execute --explicit-backend` only after root’s execution release:

```sh
"$HBE_REPO/.venv/bin/python" \
  "$HBE_REPO/outputs/validation/hbe-accelerate-6486dbc/frozen-source/scripts/mechanics_hbe_experiment.py" \
  --root "$HBE_REPO" \
  --protocol manifests/experiments/hbe-01-03-mechanics-poc-v1.json \
  --protocol-sha256 ab4385f5ad315d2444ca3aedc87b455db0d36539cea4e2119114d803fba49035 \
  --release artifacts/mechanics/hbe-accelerate-experiment-v1/release.json \
  --release-sha256 "$HBE_RELEASE_SHA"
```

The exclusive new namespace is `outputs/mechanics/hbe-01-03-poc-v1/experiment-accelerate-csc-v1`; its marker and directory remain absent. Existing source reviews cover this unchanged runner. The fixed sequence is eighteen reference/scaling solves → numerical gates → two axial calibration members → one positive modulus fit → two fitted axial confirmations → twenty-run raw replay and durable parameter/prediction freeze → exactly two held-out torsion members. Keep 20 calls, 90 seconds/call, 900 seconds aggregate worker including readout/replay, one thread, 3 GiB sampled group RSS, 256 MiB active and 1 GiB common output. First failure terminates the attempt; no retries, substitutions or CSV/schema/tolerance changes. Results, if completed, describe one specimen and do not establish clinical validation.
