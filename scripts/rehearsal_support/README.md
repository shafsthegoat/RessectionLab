# Maintained rehearsal support

These are supported copies of the October 20 review helpers. Their historical
originals are preserved unchanged in the optional rehearsal archive under
`testcases/outputs/full-case-rehearsal-20261020/review/`. The original scientific
identities, refusal predicates, scene/source hashes and protection contracts are
retained. Only locations, CLI bindings and default report destinations changed.
`readonly_step_monitor.py` is an exact byte copy of its original. Its generated
controls now run as `tests/test_readonly_step_monitor.py` in ordinary discovery.

After restoring the optional bundle, use the current wrapper from the checkout:

```sh
.venv/bin/python scripts/verify_shared_rehearsal.py --data-only --bundle-root /absolute/restored-root
.venv/bin/python -m pytest tests/test_readonly_step_monitor.py tests/test_shared_rehearsal_support.py -q
```

The first command checks every one of the 581 original data files, source arrays,
saved state hashes, protection, accounting and withdrawal, with zero physics
actions. It does **not** prove execution under the historical producer source.
The unchanged default/`--files-only` whole-source check still rejects changed
source bytes. The earlier support-pass baseline had 131 mismatches; the pushed
pre-migration revision `6547ad74` has 135 of 1,030 historical source files changed
by subsequent development. The original manifest must not be updated to make it
pass. All 581 original data files still pass their separate exact-byte check.

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

The default data root remains the executing checkout for compatibility with a
locally restored bundle. The archived historical payload is optional; ordinary
application use and generated regressions do not require it. A directory
containing only `final/` is not a valid complete data root.

Restore the versioned archive from the mounted external data root into a new,
absent destination. The standard restore tool refuses overwrites, validates
paths and verifies every restored file against the archive manifest:

```sh
archive_root='/Volumes/shafrir/old coding projects/RessectionLabData/archive/closed-evidence/rehearsal-20261020-v1'
python3 scripts/archive_historical_evidence.py restore \
  --archive "$archive_root/archive.tar.gz" \
  --manifest "$archive_root/manifest.json" \
  --destination "$PWD/testcases/restored-rehearsal-20261020"
export RESSECTIONLAB_REHEARSAL_BUNDLE_ROOT="$PWD/testcases/restored-rehearsal-20261020"
.venv/bin/python scripts/verify_shared_rehearsal.py --data-only
scripts/open_shared_rehearsal.sh --bundle-root "$RESSECTIONLAB_REHEARSAL_BUNDLE_ROOT"
```

The destination contains all 581 manifest-selected files with their original
checkout-relative paths, including source case, protection arrays and historical
review code. Current helper imports always come from this maintained directory.
The archive manifest/receipt and execution cleanup record identify the exact
archive hashes and preservation Git reference. Existing unique ignored October
review/test evidence remains local and is not removed by this migration.
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

The independent October 22 playback baseline is a separate, optional profiler
component. Restore it independently into a new destination, then explicitly
select its report together with the October 20 bundle. These input checks read
the episode, three-action trace and baseline with zero physics actions:

```sh
baseline_archive_root='/Volumes/shafrir/old coding projects/RessectionLabData/archive/closed-evidence/rehearsal-profiler-baseline-20261022-v1'
baseline_root="$PWD/testcases/restored-rehearsal-profiler-baseline-20261022"
python3 scripts/archive_historical_evidence.py restore \
  --archive "$baseline_archive_root/archive.tar.gz" \
  --manifest "$baseline_archive_root/manifest.json" \
  --destination "$baseline_root"
rehearsal_baseline="$baseline_root/testcases/outputs/rl-readiness-20261022/desktop-baseline/playback-authenticated/report.json"
.venv/bin/python scripts/profile_surgical_replay_prefix.py --check-inputs \
  --bundle-root "$RESSECTIONLAB_REHEARSAL_BUNDLE_ROOT" \
  --baseline "$rehearsal_baseline"
```

The restore tool refuses an existing destination; reuse a previously verified
restore or choose a new path. Omitting `--check-inputs` still requires `--output`
and explicitly profiles the bounded three-action prefix; it does not qualify
the full patient episode. For example:

```sh
.venv/bin/python scripts/profile_surgical_replay_prefix.py \
  --bundle-root "$RESSECTIONLAB_REHEARSAL_BUNDLE_ROOT" \
  --baseline "$rehearsal_baseline" \
  --output "$PWD/testcases/outputs/restored-rehearsal-prefix-profile" \
  --max-wall-seconds 120
```

The vascular monitor and benchmark can also validate restored input bindings
without launching Electron or executing any episode actions:

```sh
PYTHONPATH=src .venv/bin/python desktop/electron/vascular_playback_monitor.py \
  check-inputs --bundle-root "$RESSECTIONLAB_REHEARSAL_BUNDLE_ROOT"
node -e 'console.log(JSON.stringify(require("./desktop/electron/benchmark-surgical-vascular.cjs").checkRehearsalInputs(), null, 2))'
```
