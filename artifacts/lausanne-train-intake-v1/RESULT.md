# Full TRAIN source index and bounded intake acceptance

Latest audited cumulative state: **90 people / 95 sessions / 380 files /
4,315,283,541 bytes**. Batches 17–18 added 19 sessions from 16 people; repeated
visits remain grouped. All 95 receipts, originals and retained source snapshots
passed root and independent checks. [Batch17](acquisition-batch-17-summary.json)
and [batch18](acquisition-batch-18-summary.json) preserve individual closures.
Each stayed within 600 seconds and 536,870,912 declared source bytes.

The image-only total is 190 files / 4,314,984,782 bytes. **115 TRAIN sessions from
109 people remain incomplete** at batch18 closure. Earlier sub208 and sub220
timeouts later completed; their historical failures remain. Batch18 accounts
for every indexed session: 82 cached, thirteen successful workers, one timeout
(sub251) and 114 byte-budget deferrals. Later batch19 files remain outside this
closure. No spatial admission or learning contribution follows from intake QC.
Earlier milestone totals below retain their historical meaning.

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

## First bounded expansion batch

The first acquisition batch closed after **599.1025 seconds** under its
600-second / 134,217,728-source-byte declaration. It verified the prior cached
pilot, completed **sub-001/ses-20101222** (32,714,941 bytes; worker 536.9822
seconds), then reached the time budget during sub-002. The latter retained a
**6,291,456-byte partial T1** without a completed-source or QC claim.

Both worker processes were reaped. The exact index/source bindings remained
unchanged. Root independently rehashed all completed originals against the
published fixity and receipt SHA256 and verified retained source snapshots.
The new worker recorded header/scalar passes for T1 (30×512×512) and TOF
(384×512×90). Anatomical/frame/registration acceptance remains absent.

Totals are **two people / two sessions / 69,376,670 complete source bytes**;
**208 TRAIN sessions remain incomplete**. The full 210 denominator includes the
timeout. There are still zero optimizer updates, recorded RL transitions and
spatial admissions. Resume the retained partial in the next bounded batch.

[Batch summary](acquisition-batch-01-summary.json), [full closure](acquisition-batch-01-batch.json),
[declaration](acquisition-batch-01-declaration.json), [new acquisition](acquisition-batch-01-sub-001.json),
[executed source](acquisition-batch-01-source.json).

## Second bounded expansion batch

The next serial batch completed four additional TRAIN people in **521.3543
seconds** under the same 600-second/134,217,728-source-byte limits. It resumed
sub-002 and completed sub-005, sub-006 and sub-015. Attempted source sizes total
134,090,621 bytes; this is not an independent network-byte measurement. Sessions
that did not fit the remaining byte budget are deferred, not excluded from the
full TRAIN source index. The next invocation considers them again.

Root reverified all six complete source receipts, published MD5/SHA fixity and
retained source snapshots. Current totals are **six people / six sessions /
203,467,291 complete source bytes**; **204 TRAIN sessions remain incomplete**.
All workers exited. Header/scalar checks passed, with no independent anatomical,
scanner-frame or spatial-planning acceptance. No optimizer updates or eligible
recorded RL transitions were added. The first batch timeout remains preserved.

[Summary](acquisition-batch-02-summary.json), [closure](acquisition-batch-02-batch.json),
[declaration](acquisition-batch-02-declaration.json), [source](acquisition-batch-02-source.json).

A separate three-request transport diagnostic compared one first-MiB T1 request
with concurrent first-MiB T1/TOF requests from the already acquired pilot. Exact
range, version and local-prefix hashes matched. Observed aggregate throughput
was 48,871 versus 151,153 bytes/second. This single fixed-order trial has large
connection/header-time confounding and cannot establish sustained throughput or
a 3.09-fold speedup. The acquisition runner remains serial. A bounded two-request
full-object trial, with balanced timing, would be required before a performance
claim. [Diagnostic record](transport-concurrency-diagnostic.json).
# Third bounded TRAIN expansion, October 8

The [batch03 summary](acquisition-batch-03-summary.json) records four additional
people: sub-007, sub-021, sub-022 and sub-450. All four workers exited zero and
their original-image/sidecar receipts pass source fixity and saved QC checks.
The run took 527.264424209 seconds of 600 and used 133,217,934 bytes of its
134,217,728-byte source-size budget. This budget measures declared source size,
not network traffic. The remaining 200 sessions were deferred, not failed.

Root independently repeated `validate_receipt` for all ten acquired sessions;
a separate read-only reviewer also rehashed all originals and execution
snapshots. Totals are ten unique people, ten sessions, 40 files and 336,685,225
bytes. Batch receipt SHA256 is
`da8b6cded6f4e3afd8b57dffa4eec576c10d0c1eeca5b6b1d876c82441bb31fc`.
All patient roles and source-index hashes are unchanged. Current imaging/core
edits postdate this execution; its retained historical snapshots are authoritative.
These checks establish acquisition and worker QC, not registration, vascular
annotation admission, component training or recorded surgical experience.
