# NFBS saved member inventory: independent result review

PASS for the saved names/sizes/types inventory and its pairing summary. No archive was reopened, no inventory was rerun, and no image or label data were decoded by this review.

| Evidence | SHA256 |
| --- | --- |
| Saved inventory | 2e4db48a19f55bb4f06e3fca88200829d130948746bc1c9922534c105b39d37f |
| Terminal receipt | 7c27256339cb858f1a4c6c5c757ff83274971fb130819739202f899e681df2ea |
| Names-only summary | 1b24ad293a7ff334d7cb1b054100d21174fda665b2d4de40449b7edf8c3775ab |
| Root release | 892a2e2933815db33cf0b230b7e4666ebffc41c42df0d97a830c59be9494026b |

Exact release/template changes, child protocol, command, launcher/reader/review hashes, CPython and stdlib source bindings, historical/current supervisor helper bytes, frozen cohort and original acquisition receipt all reconcile. The original archive identity agrees across saved records: 1,751,464,473 bytes, SHA256 fd616b9ea21aad3d0d4cd05a25a71f4a3c251140df38188e58e499254440d2cc, publisher MD5 06c467ac9ab6cbdb8185e6643879721d. These source hashes were not recomputed against the archive during this audit.

The result has semantic status member_metadata_only. Independently reconciled all 501 unique entries: 126 zero-byte directories (archive root plus 125 source-ID directories), and 375 regular files. Every source ID has exactly one T1w, one T1w_brain and one T1w_brainmask filename; all use ses-NFB3 and directory/file source IDs agree. The author pairing summary maps every exact path and byte size correctly. Declared member bodies total 1,778,826,347 bytes; saved inflated traversal is 1,779,179,520 bytes, consistent with headers, per-member block padding and zero end/trailing records. Every member record contains only name, byte length and type. The reviewed reader returns this status only after strict tar termination, bounded trailing-zero traversal and gzip CRC validation; no additional decompression was performed by this audit.

One worker attempt, PID 95700, exited zero without kill reason. Reader work took 5.599938459 seconds, supervised stage 5.722246375 seconds and lifecycle 5.742511667 seconds. Sampled peak process-group RSS was 26,460,160 bytes, under the 256 MiB sampled guard; sampling is not a full high-water measurement. The expected four regular output files total 70,297 bytes, below 4 MiB. Cleanup records direct child reaped, no remaining members, no fallback and no errors. All recorded time, output and sampled-memory caps passed.

All source IDs inherit the unchanged dependent TRAIN-family assignment. Source IDs and name pairing do not establish 125 biologically independent people, absent cross-source overlap, NIfTI integrity, shared geometry, pixel alignment or valid masks. Per-member hashes and inner NIfTI gzip integrity were not checked; extraction, header/array decoding, anatomy/label QC, training and independent-evaluation admission remain false. BEaST/library lineage remains unresolved. Any later extraction or header QC requires its own bounded intended-use step; this result grants no such execution automatically.

Evidence: saved-audit.json records independent reconciliation. Its audit hook recorded zero source archive/image/MAT open attempts. No tracked edits, commits, payload downloads or changes to the live TractoInferno intake occurred.
