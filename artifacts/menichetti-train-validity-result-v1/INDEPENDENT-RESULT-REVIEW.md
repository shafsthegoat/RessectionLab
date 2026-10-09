# Independent saved TRAIN validity result audit

PASS for the declared TRAIN numerical validity screen. The saved result contains 72 matrices and 357 trial columns from the six frozen TRAIN brain aliases. All 3,694,950 sample positions are finite and all trial series are nonconstant. This audit read only saved QC summaries, metadata, source files, runtime files, and receipts. It did not reopen the original MAT, run the scientific worker, or decode protected numerical arrays.

Exact evidence:

| Artifact | SHA-256 |
| --- | --- |
| Root release | 29dbc55b39fd2e7ede6c94f9adf356cd5679c654d58e146eced2aecc363fe98c |
| validity.json, 302,935 bytes | 4977406ecb0f52bbe44478d438323cdf56657ef94bf32ccc997301ebbc4335b5 |
| receipt.json, 5,526 bytes | 0910bb39a7dce40e7aeaa54e9e12a2d37962f57c320f1a78a333fee20526f64a |
| Released child protocol | 68ba439afb73700c908a8e52b786e626601e6523c10aea6a2442c8ef3e383978 |
| Runtime closure | a8584f66e51cdf45a36cdbe879c210c36577bb31a5ab651bcd6fa18367b8a6fc |
| Writer aggregate | f00925508f15678680f67fa892c305bf33b20cd0349cc162cf09693c7f15271b |

Execution files remain under `build/menichetti-train-validity-v1/`; root release/runtime under `build/menichetti-train-validity-launch-preparation-v1/`. The launcher is the reviewed `9b172b3e93375fb2dfc918da3f67896e0b4ea977520e3aa30f5d1b25b958d689`. Release and launch both record HEAD `86b3cdbf334a1dffff67b07d65788906d4c42b75`. The root release changes only its permitted three template fields. The child protocol differs from the false prepared protocol only by its execution bit.

All 14 source bindings and 413 distinct runtime/interpreter files match their recorded hashes and sizes. The three supervisor helpers also match their historical Git blobs; frozen roles match the execution HEAD. The saved exact command, successful pinned entry control flow, source rechecks, and unchanged absent cache prefixes support the recorded runtime checks before and after QC. The loaded closure has 411 module origins and 406 module files. OS shared-cache libraries remain identified by platform release rather than byte-pinned. Original-file fixity is supported by the executed pinned worker and provenance; the reviewer deliberately did not recompute the original MAT hash.

Independent saved-summary reconciliation reproduced the semantic result, every brain/region/trial membership, count partition, flag, and disposition. It also exactly reproduced the writer's aggregate and every per-brain aggregate. Brain1 contributes 57 trials; Brain4, Brain5, Brain7, Brain8, and Brain10 each contribute 60. Each trial has 10,350 positions. The four protected aliases account for 48 skipped matrices; reported protected numerical arrays materialized, raw force samples emitted, and fitted models are all zero. Opaque compressed traversal of protected bytes was permitted; it is not protected numerical decoding.

The useful QC findings are:

- No NaN or infinity positions and no quarantined nonfinite trials.
- No all-zero or finite-constant trial, no exact-zero suffix of at least two samples, and no exact constant terminal run of at least 100 samples.
- 306 trials have both positive and negative finite values; 51 are nonnegative. Across TRAIN there are 3,687,575 positive, 4,611 negative, and 2,764 exactly zero positions. These counts do not determine the force-sign convention or justify sign changes.

One attempt completed with worker exit 0 and accepted QC semantics. Worker time was 0.663306 seconds; lifecycle time was 0.685649 seconds. Peak sampled group RSS was 63,717,376 bytes; final aggregate output was 313,961 bytes. These are below the 35/45-second, 512-MiB, and 4-MiB caps. The saved cleanup records PID 376 reaped, no remaining group members, no fallback, and no errors. RSS/output sampling can miss brief peaks; the independent scan verifies current final output size.

Passing these finite/count/flag checks does not establish absence of padding by other patterns, contact/hold alignment, physical validity, constitutive behavior, or transfer accuracy. No crop, sign flip, model fitting, or physical-validation claim was made. Brain aliases retain the creator's specimen grouping, without independent donor authentication; cross-release biological overlap remains unknown and existing protected roles remain unchanged.

Reproducible saved-only evidence is in `audit_saved.py`, `saved-audit.json`, and `writer-aggregate-check.json`. The audit hook recorded zero original-MAT open attempts. No tracked files or execution artifacts were modified.
