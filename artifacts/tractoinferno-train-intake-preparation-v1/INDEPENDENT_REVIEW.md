# TractoInferno full published TRAIN byte acquisition: independent review

GO for root to freeze the person roles/scope, rebind the declaration/source snapshots to that frozen record, and release byte acquisition. This exact candidate remains unreleased; no acquisition was launched by this review. No detailed anatomical QC gate is needed for this byte-only scope.

Frozen candidates independently checked before and after controls:

| Candidate | SHA256 |
| --- | --- |
| run-intake.py | e1db984939c5b342b754650e7f42b7df57d6c1993d2ffb8add36da373d58b5bc |
| proposed-cohort.json | 18da6cfead3146184b656258f8f5635081d893fbfdcc42d8ff9aa654c9c8d0bc |
| proposed-scope.json | e21eeabf9d03521534253a6f1c15c4c278fec423e8ec9db5417f9d2a5eb07c9c |
| prepared-declaration.json | e77b37f7d9608633f94544a33b9ede0137ee9954ec5f1649738a6d64315f86f0 |
| train-exact-object-manifest.json | 151441d5eb6fe923ad3b1f978edf49b04843956f3209e216a95340466003a846 |

Independent metadata reconciliation verified the official OpenNeuro ds003900 tag 1.1.1 resolves to bf5b266e9964881b4ba3695a8374c29808f0c4da, the complete saved API tree agrees with the local Git tree, and all 7,622 files beneath derivatives/trainset/ are selected exactly once: 273,788,367,039 bytes and 198 TRAIN source identifiers. Publisher validset 58 SELECT and testset 28 MEASUREMENT_EVAL identities are preserved and excluded from payload scope. Top-level duplicates are excluded. The pinned shared dataset description states CC0; its bytes, README and CHANGES match their Git blobs. Unknown biological overlap, site crosswalk and pretrained exposure remain unknown; source roles are not proof of biological independence.

Read only 7,229 shared-metadata/selected-annex-pointer blobs, with Git lazy fetch disabled; no gradient, image or tract payloads. Verified all 7,226 selected annex pointers against their pinned Git IDs, pointer SHA256, MD5E sizes and MD5 identities. All 396 gradient objects retain their pinned regular Git blob SHA1 and lengths. Reconciled every selected object to the exact version in nine chained S3 listing pages (7,820 version rows), including single-part ETag MD5, length, key and version URL. Verified 26 source-proof hashes and all 13 staged-source hashes. The three transport helpers exactly match previously reviewed ReMIND SEG helper bytes.

The adapter uses the established streaming acquire_file transport. TLS peer/hostname verification and redirect refusal remain active; exact source URL, If-Match ETag, S3 version header, size, identity encoding and strict Content-Range bind resumption. The gradient Git blob callback verifies framing plus bytes before the unchanged atomic publication. Local SHA256 is recorded afterward. Existing verified files and successful root completion history remain immutable; mismatches do not overwrite/publish or silently restart. Durable intents/results enforce three attempts across restarts, orphan intents require reconciliation, and transient provider cooldown is retained. ExitStack retains predecessor and own exclusive queue locks throughout execution. Two workers are fixed.

Storage checks reserve the queued/inflight unwritten bytes plus 64 GiB model-output allowance and a 100 GiB free reserve. Completed bytes are not charged again; current inflight reservations are conservatively retained until terminal. Checks occur before queue execution and each new attempt. The declared 512 MiB transport allowance is explicitly not an enforced or measured peak. Model overlap/process monitoring remains root's separate coordination responsibility.

Independent validation: metadata reconciliation passed; all 13 generated eight-byte transport/storage controls passed; check-only passed; false release refused before helper import, queue/payload directory creation or network. Generated controls include wrong Git identity before publication, wrong version/ETag/MD5, strict and truncated resumption, ignored Range refusal, cross-restart attempt exhaustion, orphan refusal, existing completion preservation, completed-partial verification and remaining-storage arithmetic. Test harness changes only redirected evidence output into this review directory; exact production runner/helper bytes were used. Author evidence was preserved. Test execution used one-thread environment settings and completed in under one second per command.

No live network requests, payload downloads, protected payload access, header reads, decoding, label admission, training, original MAT access, tracked edits or commits occurred. Outputs remain unreviewed acquisition candidates; derivative tract bundles are algorithm-produced references, not functional ground truth or proof of glioma/retraction mechanics. Rebinding must preserve this reviewed selection and runner, with only the explicit frozen role/scope/release metadata changes.
