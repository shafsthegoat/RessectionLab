# Maintained rehearsal support

These are supported copies of the October 20 review helpers. Their historical
originals remain unchanged in
`testcases/outputs/full-case-rehearsal-20261020/review/`. The original scientific
identities, refusal predicates, scene/source hashes and protection contracts are
retained. Only locations, CLI bindings and default report destinations changed.
`readonly_step_monitor.py` is an exact byte copy of its original. Its generated
controls now run as `tests/test_readonly_step_monitor.py` in ordinary discovery.

Use the current wrapper from the checkout:

```sh
.venv/bin/python scripts/verify_shared_rehearsal.py --data-only
.venv/bin/python -m pytest tests/test_readonly_step_monitor.py tests/test_shared_rehearsal_support.py -q
```

The first command checks every one of the 581 original data files, source arrays,
saved state hashes, protection, accounting and withdrawal, with zero physics
actions. It does **not** prove execution under the historical producer source.
The unchanged default/`--files-only` whole-source check still rejects changed
source bytes; it already failed before this cleanup because development changed
131 pinned files. The original manifest must not be updated to make it pass.

`--bundle-root /absolute/data-root` chooses an existing, checkout-shaped **data**
root independently of the executing source checkout. The corresponding environment
variable is `RESSECTIONLAB_REHEARSAL_BUNDLE_ROOT`; CLI values take precedence.
Keep these paths together below that root:

```text
testcases/cases/meningioma-sub-06-standard-v2.ressectionlab
testcases/environments/meningioma-sub-06/rl-protection.json
testcases/environments/meningioma-sub-06/<its referenced arrays and metadata>
testcases/outputs/full-case-rehearsal-20261020/<complete historical payload>
```

There is no archive or relocation in this pass. The default remains the existing
checkout. A directory containing only `final/` is not a valid complete data root.
Strict historical-source checks still read executing-checkout source paths;
`--bundle-root` redirects only data. No authority or physics guard is relaxed.

The wrapper's explicit slow `--canonical`, `--prepare-frames`, and `--audit-frames`
modes now invoke these maintained copies and inherit the selected data root.
They retain their original cache-only, local-fingerprint and source-hash checks.
Canonical acceptance and new reports go under ignored
`testcases/outputs/shared-rehearsal-verification/`; existing acceptance reports
are refused. Do not start a full patient replay to test path wiring.

The prefix profiler accepts the same `--bundle-root` and a separate `--baseline`
for its independent playback report. Vascular benchmark processes accept the
environment variable; the standalone vascular spool audit also accepts
`--bundle-root`. The normal generated prefix/guard tests need neither the patient
bundle nor its historical `review/` directory. The old helpers remain available
for exact historical inspection and must not be overwritten by these copies.
