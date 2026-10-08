# RESECT original TRAIN acquisition preparation

Ready for root review and launch. The separate manifest fixes 104 files / 672,336,857 bytes for the existing 14 TRAIN people. The original 25 image/mask pairs and completed 890-file queue remain unchanged. Nine original metadata/documentation/helper proofs are archived under source-proofs, with exact hashes; the three new implementation files are frozen under source-snapshot.

All 43 focused offline controls passed in 1.45 seconds. The actual CLI preflight passed metadata identity checks without network or patient payload reads. Controls cover correct and malformed 206 responses, ignored 200 responses, checksum/TLS failures, partial mutation, interrupted/torn append recovery, storage quota, provider cooldowns, persisted attempt limits, continuous progress beyond eight objects, and verification of previously completed bytes. Preparation failures and repairs are recorded in verification.json and the task tool history.

Run from the repository:

```sh
.venv/bin/python -B scripts/acquire_resect_train_originals.py run
```

The full queue runs continuously with one worker. It stages every resumed suffix separately, validates exact Content-Range and remaining length before appending, and never appends a 200 response to a retained prefix. Journals recover torn merges without duplicated bytes. After a verified durable merge, redundant suffix staging is compacted; its exact bytes remain in the retained prefix and its hash/merge record remains immutable. Failed unaccepted suffixes, old prefixes and failure receipts are preserved. Complete originals require published MD5 and size, plus recorded local SHA256.

Bounds: 1 GiB across the new cache, completed payloads and retained staging; 295-second workers within 300-second supervisors; at most three attempts per object across restarts; provider-wide 429/Retry-After cooldowns. Metadata-only status is available with the same command ending in `status`. Terminal failures remain in the denominator and return unsuccessful completion.

No source data was acquired or decoded in preparation. Image, landmark, geometry and anatomical QC remain separate. This does not admit training, spatial planning or RL. No existing helper, shared document or UI file was edited.
