# Bounded BTC queue execution

The selection declaration in `manifests/development_acquisition_queue.json`
was committed before image access. Its SHA256 is
`a768aaefd78146540cf20c13193e22268344206a4a396ffb654e7d40af7b8898`.
The downloader binds that exact declaration and permits only the structural
files of PAT16 and PAT20. It does not activate the queue's optional diffusion
files. Existing PAT28 and PAT05 entry points retain their acquisition scopes.

Before acquisition, inspect the fixed seven-file budgets without network or
output-directory creation:

```sh
.venv/bin/python scripts/acquire_btc_case.py --queued-subject sub-PAT16 --dry-run
.venv/bin/python scripts/acquire_btc_case.py --queued-subject sub-PAT20 --dry-run
```

After the implementation checkpoint is committed and the orchestrator releases
the image-acquisition hold, omit `--dry-run` to acquire one case at a time.
Each source image must match its published annex MD5 and byte count before
atomic publication; the measured SHA256 is recorded only afterward. Partial
transfers can resume, existing files are verified without rewriting, and an
unexpected object version, response length or checksum fails closed. No image
hash is inferred from a filename or server header.

Successful acquisition writes
`data/diffusion_source/ds001226-v5.0.1/btc_sub-PAT16_verified_manifest.json`
(or PAT20), separate from the immutable pre-image declaration. This manifest
retains the development lock and exact source bindings, adds measured hashes,
and can be supplied to the existing preparation CLI:

```sh
.venv/bin/python scripts/prepare_btc_case.py \
  --manifest data/diffusion_source/ds001226-v5.0.1/btc_sub-PAT16_verified_manifest.json
```

Preparation rejects a changed source, changed role, changed queue or pending
image hash. It preserves fractional source values, requires explicit threshold
derivation, records unavailable clinical context, and keeps full-head signal
from supplying cortical access. Structural-only cases retain missing
directional diffusion. Append acquired development records to the cohort
registry only after source and independent geometry QC; final assignments
never change.

The focused offline suite passed **77 tests in 2.15 seconds**, covering both
existing cases, immutable selection, exact source binding, checksum failure,
wrong object version, resume ranges, null pre-acquisition SHA256, and synthetic
preparation of each new subject with the full-head and context gates intact.
No real PAT16 or PAT20 image was fetched or opened during these tests.
