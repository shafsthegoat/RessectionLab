# Historical pilot auditor contract integration repair

The two assigned failures in root's frozen `523a72b` run came from the auditor's
fixed legacy checksum field list. Current tiny-fixture training correctly hashes
the added `axis_observation_contract`; reconstructing the old body therefore
failed before forward validation or the intentionally corrupted optimizer check.
The original two failure excerpts and full-log hash are retained here. Root owns
the complete failed integrated receipt and the other ten failures.

The auditor now explicitly supports the original RAW contract and the known
RAW axis-observation-v1 extension. It validates the exact field inventory, the
full nested schema and its own checksum, semantic definitions, physical units,
proposal/model/reward/budget bindings, then hashes the complete learner body.
Unknown fields or versions fail. Presence of the axis schema must agree with
its module in the numerical source inventory, preventing a simple resealed
downgrade to the historical form. Frozen source comparison remains required in
the public audit entry point; this does not authorize a new public experiment.

The private synthetic fixture previously copied a public proposal hash and
inventory limit while replacing the native and decision model hashes. Its
declaration now records its actual tiny proposal hash, rule, and inventory. The
public declaration and all tracked historical actual artifacts remain unchanged.

The final focused run passed **68 tests in 5.55 seconds**. It covers both formerly
failing tests, the exact retained historical contract bytes, unknown-version and
coherently resealed semantic attacks, full nested hashing and downgrade guards.
Existing tests still verify saved actual forwards without policy-class calls or
RNG changes, and reach the intended nonfinite optimizer rejection. Only the
existing constructed tiny fixture was executed; no public case was rerun.

Attempt 01 passed 66 checks before the two downgrade/source-generation tests
were added. Both logs, JUnit records, original/intermediate/final sources and the
unchanged historical contract bytes are preserved. `receipt.json` gives source
hashes and validation scope; `historical-file-hashes.json` binds the tracked
historical output bytes checked against Git. No full integrated suite was run
by this agent.

Reproduce the focused checks with the installed development dependencies:

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m pytest -q tests/test_native_axis_pilot_artifact_audit.py tests/test_native_axis_pilot.py
```
