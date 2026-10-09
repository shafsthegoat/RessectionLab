# Exact downloader allowance delta — prospective only

The independently reviewed no-allowance contract is preserved byte-for-byte
as `pair-contract-original-no-acquisition.json` (SHA-256
`58a7365cb929334280ad3a481f5c1119c7b3bd1c18718b4f0b81a8d5a8c8bb06`).
The active `pair-contract.json` is SHA-256
`4c6d17d8287f7ad791ef1002cf931beed7ee561dafb3c99547020993e0f93374`.
Its **only** changed keys are `acquisition_allowlist_path` and
`acquisition_allowlist_sha256`. The path points to the previously reviewed
full128 tile1-v2 exact-process manifest, SHA-256
`4ab567045e91d70e040c799b70df0b8c45fe4c9cf2f8aa07156c88aabf82f629`.

The copied supervisor already loads this manifest and the copied slow
inventory checks live PID, PGID, start time, executable, exact command hash,
source/declaration bytes and a 512 MiB RSS ceiling on every sample. A mismatch
becomes blocking overlap. The sampler never signals the downloader. Fresh
read-only checks observed PID/PGID `92434`, start identity
`Fri Oct  9 05:46:44 2026`, all exact hashes matched and RSS `40,736 KiB`
within `524,288 KiB`. See `live-acquisition-verification.json` and
`readiness-acquisition-v2.json`. The identity must still match at each future
preflight and run sample; this preparation is not a permanent exemption.

All 11 runner source hashes and the earlier 22 guard/input fields are
unchanged. Neither model arm exists or has run. This contract delta requires
independent exact-source review before root launches the baseline. The
no-allowance wording in `READINESS.md` describes the preserved original
contract; this delta governs the active contract.
