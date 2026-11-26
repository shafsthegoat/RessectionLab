# Cleanup execution · November 16, 2026

The active tracked checkout is substantially smaller. The complete historical
rehearsal and desktop capture collections were archived, independently verified,
restored and consumed before explicit tracked files were removed. This follows
the [earlier support pass](retired-docs-20261126.md); its original register is reused
with a separate execution section, rather than replaced by another audit.

<!-- execution-metrics-start -->
| Measurement | Before | After |
| --- | ---: | ---: |
| Tracked files | 17,608 | 16,742 |
| Tracked checkout bytes | 1,885,429,107 (1798.085 MiB) | 389,883,950 (371.822 MiB) |
| Net tracked reduction | — | **1,495,545,157 bytes (1.393 GiB; 79.32%)** |
| Git object files, separately (baseline → archival implementation snapshot) | 379,840,512 bytes | 380,401,664 bytes |
| Installed app payload | Existing package | Unchanged |

Largest remaining tracked path groups:

| Path | Files | Bytes |
| --- | ---: | ---: |
| `artifacts/` | 14,109 | 351,703,241 |
| `tests/` | 768 | 8,324,262 |
| `scripts/` | 430 | 7,117,699 |
| `src/` | 367 | 6,890,821 |
| `manifests/` | 185 | 5,522,456 |
| `desktop/` | 525 | 5,402,110 |
<!-- execution-metrics-end -->

The reduction is in tracked working-file bytes, **not Git history**. Original
objects remain reachable; no history rewrite, force-push, prune or main merge was
used. No installed app payload change is claimed for this pass. Dependencies,
model/runtime caches, EDEN source/prepared/case deliverables and unrelated ignored
work were retained. Temporary restore copies remain ignored under `testcases/`.

## What left the checkout

- **581 rehearsal files / 1,312,252,037 bytes:** 576 files under
  `testcases/outputs/full-case-rehearsal-20261020/`, the intact source case at
  `testcases/cases/meningioma-sub-06-standard-v2.ressectionlab`, and four matching
  files under `testcases/environments/meningioma-sub-06/`. The unchanged historical
  [manifest](../artifacts/full-case-rehearsal-20261020/manifest.json), filtered to
  its `testcases/` rows, is the exact path/size/hash list. Intermediate checkpoints,
  endpoints, original actions, failed attempts, review code and replay evidence
  are preserved together.
- **291 visual media files / 183,382,322 bytes:** exact paths are in
  [the removal list](../manifests/archived-visual-media-20261116.txt). The complete
  original 550-file collection is archived. All 253 nonmedia files and six proof
  images remain locally, including both live numerical inputs. Duplicate image
  names and immutable report-relative media mappings remain intact in the archive.
  [Visual evidence guide](visual-evidence-archive-20261116.md).

Gross removal is **872 files / 1,495,634,359 bytes**. Small source, tests, index and
documentation additions account for the difference from the net reduction above.
Git removal used only reviewed literal file paths, never recursive directory
deletion. All **5,539 ignored original files and 339 symlinks** in the rehearsal
tree remain unchanged; they were also copied into the archive. Their removal was
not authorized. A third small archive preserves the independent later profiler
baseline, whose ignored original also remains.

## Archives, durable copies and restoration

The existing HDD was verified writable as ExFAT `/dev/disk8s1`, mounted at
`/Volumes/shafrir`, UUID `9CC3623C-E38B-3EEE-B6BB-B9A06C193700`. `diskutil`
independently confirmed that identity. The archive builder checks the mounted
independent filesystem, existing root marker and archive containment; it does not
itself query the UUID. About 1,363,114,065,920 bytes were available initially.

All three archives are under:

```text
/Volumes/shafrir/old coding projects/RessectionLabData/archive/closed-evidence/
  rehearsal-20261020-v1/
  desktop-visual-evidence-20261116-v1/
  rehearsal-profiler-baseline-20261022-v1/
```

Each directory contains `archive.tar.gz`, `manifest.json` and `receipt.json`.
The small [archive index](../manifests/historical-archives-20261116.json) records
their exact locations, archive/manifest/receipt SHA-256 values, sizes, member
counts, purposes and complete restore commands. No new storage service was used.

Tracked originals have two independent durable copies: the verified external
archive and pushed tag **`preservation/cleanup-execution-20261116`**, pointing to
**`6547ad746787283f9f9159ddf3235d0820c9c382`**. This clean pre-removal revision
combines the latest EDEN work (`9660c8fc`) and previous cleanup implementation.
The original source-history commits remain reachable. Git preservation is not
claimed for ignored files; those originals remain on their original device.

From the repository, restore into a **new, absent** directory:

```sh
rsl_archive='/Volumes/shafrir/old coding projects/RessectionLabData/archive/closed-evidence/rehearsal-20261020-v1'
rsl_restore="$PWD/testcases/restored-rehearsal-20261020"
python3 scripts/archive_historical_evidence.py restore \
  --archive "$rsl_archive/archive.tar.gz" \
  --manifest "$rsl_archive/manifest.json" \
  --destination "$rsl_restore"
.venv/bin/python scripts/verify_shared_rehearsal.py \
  --data-only --bundle-root "$rsl_restore"
```

The standard gzip/PAX process verifies the archive and receipt before extraction,
rejects extra/missing/duplicate/unsafe members, and compares the complete restored
inventory. Existing targets, including symlinks, are refused. Original relative
paths, file bytes, modes, nanosecond mtimes and symlink targets are exact;
ownership restores as the current user. Failed operations never remove originals.
[Supported rehearsal commands](../scripts/rehearsal_support/README.md) also show
the separately restored profiler baseline and launcher/monitor options.

Restoration succeeded to separate roots under ignored
`testcases/cleanup-execution-20261116/`. Independently checked totals:

| Archive | Restored members | Content bytes |
| --- | ---: | ---: |
| Rehearsal | 6,122 files + 339 links + 941 directories | 6,750,822,938 |
| Visual collection | 550 files + 37 directories | 194,640,539 |
| Profiler baseline | 1 file + 5 directories | 389,091 |

Sparse generated fixtures inflate logical rehearsal bytes. These content totals
are not archive sizes or additional checkout savings.

## Implementation and verification

- `4d9ad383`: simple exact archive/restore tool; 14 focused controls pass.
- `caa08d2b`: restore-aware optional consumers and 581-file rehearsal retirement.
  The launcher, shared verifier, prefix profiler, vascular benchmark/monitor and
  other named historical helpers use selected data roots with actionable missing
  data errors. Maintained support stays in the checkout; original review code is
  preserved in the archive.
- `b55ddf5a`: 291-file visual retirement, retained live references, restoration
  links and narrow capture ignores. New interaction/rehearsal output defaults use
  unique ignored runs; deliberate evidence promotion remains available.
- Final records update current navigation and the existing register. Scientific
  `src/` code and frontend implementation remain unchanged from the preservation
  revision. Both backends, admission/exclusions, cumulative harm, vascular effects,
  STOP/withdrawal, residual handling and replay/save contracts remain intact.

Post-removal checks passed:

| Check | Result |
| --- | --- |
| Desktop main / renderer / viewer | 330 / 739 / 276 passed, zero failures or skips; same discovery as baseline |
| TypeScript and production build | Passed; existing bundle-size advisory retained |
| Compiled policy/support/archive checks | 364 passed (311 core + 39 support + 14 archive controls) |
| Reference-backend policy checks | 311 passed |
| Vascular generated monitoring | 42 passed, zero skips |
| Independent mocked packaging | 18 passed; no installed package replacement claimed |
| Restored historical data and saved records | All 581 files / 1,312,252,037 bytes; zero new physics actions |
| Actual restored bounded prefix profiler | Three actions, all 12 frame IDs exact, 29.002 seconds, zero timed-read physics steps |
| Current app without optional archive | Hidden startup, renderer/preload/IPC, native Save and separate cold reopen with exact generated state |

Audit hooks rejected reads from every old payload path and the old profiler
baseline during restored checks. Optional workflows were separately checked for
actionable missing-data refusals. Removing the files exposed a real generated
monitor dependency: injected reference receipts still hashed the absent historical
file. The fix explicitly identifies provided records; historical spool validation
still requires the original file hash. A new regression passes without archives.

The preservation baseline already differed from 135 of 1,030 historical source
hashes (the earlier audit had 131). The strict whole-source verifier remains
strict. No immutable scientific/source identity or historical manifest was
rewritten to force a pass. One initial archive test assumed short seek-created
holes were sparse; both original and restore were eagerly allocated by this
filesystem, so allocation assertions now apply when the source is actually sparse.
Byte/mode/mtime/link checks were never relaxed. Corrected harness attempts remain
locally recorded and are not counted as successes.

A separate Sol 6.1 High preservation reviewer independently streamed all three
archives, checked complete restores, compared every removed file against pushed
Git/original/archive/restore bytes, and rechecked ignored/retained originals after
removal. Final source/provenance/test review found no blocking issue. Full patient
replay, GUI vascular qualification and new training are outside these claims.
Raw receipts and failed attempts are under ignored
`testcases/cleanup-execution-20261116/`; the register records key evidence hashes.

## Remaining scope and rollback

Keep old source snapshots and HBE generations: exact source hashes, pathname
consumers and ancestry tests still use them. Shared content-addressed bundles for
future experiment producers remain deferred; migrating those sealed producers
without changing old receipts needs a separate bounded change. No new snapshots
were generated by this cleanup. Keep the three distinct demo packagers, guarded
historical planners and Qt/reslice utilities; source/reflection and frozen package
contracts make speculative pruning riskier than its small benefit.

Keep the diagnostic display/barrel and neighboring-paths feature as explicit
experimental contracts, including the independent shared fixture. No current
code was removed merely because it is unmounted or appears unused. The permanent
EDEN deliverables and model/anatomy runtimes remain untouched.

Review `preservation/cleanup-execution-20261116..shaf/cleanup-execution-20261116`.
On a clean checkout, a normal revert restores the tracked tree without rewriting
history (it needs space for the original payload):

```sh
git revert --no-commit preservation/cleanup-execution-20261116..shaf/cleanup-execution-20261116
# Review and commit the inverse changes normally.
```

At the November 16 delivery, the execution branch was pushed separately and
had not been merged into main. The primary checkout returned to the yielded
EDEN branch. The November 17 consolidation incorporates both lines of work;
see the [current status](current-status.md). The measurements and checks above
remain the results of the original cleanup pass.
