Canonical cache optional bounded disk allowance — integration receipt

Source changes are limited to surgical_verified_cache.py; tests update test_surgical_verified_cache.py and the prior observed-cache canonical-policy assertion in test_surgical_observed_cache.py; docs/integrated-rehearsal.md documents both settings. No engine/director/observed production implementation or patient/source/native data edits.

RESSECTIONLAB_VERIFIED_CACHE_MAX_DISK_BYTES=8589934592 opts the canonical cache into 8 GiB aggregate disk. Default remains 512 MiB when unset. An explicit integer max_disk_bytes argument wins over environment, including invalid environment. Noninteger/zero/negative/above8GiB values fail cache construction. max_file_bytes=min(configured disk,512MiB); statistics exposes maxFileBytes. Manifest, boundary descriptor and state-part reads use this separate ceiling. Existing 128 MiB transient decode/combined-state bounds and source/code/action/checkpoint/HMAC/completeness/atomic publication/eviction semantics are unchanged.

Observed settings remain separately default512MiB/optin8GiB, per-file64MiB, frameRAM64MiB, per-actionstaging64MiB, decoded128MiB. Its private owner always receives explicit validated observed bytes, so even an invalid canonical environment does not affect observed construction.

Final pipeline must pass both RESSECTIONLAB_VERIFIED_CACHE_MAX_DISK_BYTES=8589934592 and RESSECTIONLAB_OBSERVED_CACHE_MAX_DISK_BYTES=8589934592 consistently to canonical verification, observed export and Electron launch. Disk headroom does not prove an episode fits: inspect actual publication and retained entry bytes after final preparation.

Validation: 50 passed in19.39s. Sparse512MiB+1 malformed canonical manifest/descriptor/part each fail before stream.read can receive oversized data; no huge memory allocation. Existing canonical complete replay admission, tamper fallback, source identity, cancellation, fresh-process equality, field deduplication, observed signedprefix retention/read bounds and editable frame history tests pass. git diff --check passes.

Command:
.venv/bin/python -m pytest tests/test_surgical_verified_cache.py tests/test_surgical_observed_cache.py tests/test_surgical_frame_replay.py -q --basetemp=testcases/outputs/full-case-rehearsal-20261020/review/canonical-budget-pytest-20261020 --junitxml=testcases/outputs/full-case-rehearsal-20261020/review/canonical-budget-tests-20261020.xml

Only tiny engineering engines were run. No patient scene/engine/replay/preview, live process/control, GUI, commit or push. Independent reviewer notified; root owns review and integration.
