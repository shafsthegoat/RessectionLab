# Full TRAIN source index and bounded intake acceptance

October 8, 2026. This milestone indexes original acquisitions and verifies the
intake controls. It does not train a model or establish spatial validity.

- All 199 frozen TRAIN people / 210 sessions are indexed: 420 original NIfTI
  images and 420 source JSONs, **10,020,802,851 bytes**. The index SHA256 is
  `6f1fc7812af0d66550076aa08d37d7f36f08d764fdab701bbbc9ca0608629e66`.
- Initial metadata acquisition resolved 201 sessions and failed nine requests
  through actual connection/reset/timeout errors. Resume resolved all nine.
  [The original failure record](index-attempt-1-failures.json) is preserved.
- A strictly existing-files-only run rechecked sub-000/ses-20110101 in
  **1.1797565 seconds**, including fresh scalar/header QC in one worker. All four
  original file hashes agree with the earlier acquisition. No acquisition
  function ran on this path. The other **209 sessions were explicitly excluded**.
- A second invocation verified the cached receipt in **0.1488955 seconds**,
  launched zero workers and attempted zero source bytes. It did not repeat
  scalar QC. Its explicit session filter still excluded all 209 other sessions.
- Both runs retained the full 210-session denominator, unchanged source/index
  bindings, completed closure records, no spatial admission and zero optimizer
  updates/recorded surgical transitions. The replay adds **zero unique people**.
- **14 focused controls pass** (0.95 seconds). These include actual worker timeout
  and supervisor SIGTERM cleanup, immutable atomic publication, source fixity,
  historical snapshot verification, missing-original refusal, setup/closure
  failures, fixed TRAIN membership and rejection of promoted spatial claims.
  Controls use real source metadata/receipts or process/file operations; no
  patient images or clinical events were generated.

Independent source review found and prompted repairs to scope, interruption,
provenance and closure behavior before replay. Scope filtering now precedes
cache/payload access; historical receipts resolve to retained code snapshots;
SIGTERM reaches cleanup; closure-probe errors do not erase the original failure.

Receipts: [source index](source-index.json), [replay declaration](replay-declaration.json),
[replay batch](replay-batch.json), [replay acquisition](replay-acquisition.json),
[cache verification](cache-resume-batch.json), [executed source](replay-source.json),
[transport diagnostic](transport-diagnostic.json). Full source snapshots and
worker logs remain under the paths recorded in the local receipts.

The source-size budget is not measured network transfer. A decoded-image size
limit is not a hard process-memory limit. SIGKILL/unwritable storage cannot
guarantee closure. Byte/header/scalar checks cannot verify patient anatomy,
scanner-frame provenance, registration, vessel coverage or clinical accuracy.
The original pilot's 1.70-degree orientation mismatch remains unresolved.

Next: acquire the remaining TRAIN originals in bounded, resumable batches.
Preserve all failures and patient roles; actual component learning additionally
requires eligible source-linked labels or an independently justified real-data
learning task. These scans contain no recorded surgical action trajectories.
