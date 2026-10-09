# NFBS inventory launcher: independent adaptation review

GO for later root release of this exact one-attempt launcher, after root's host-recovery check. The current template remains false. No actual NFBS archive pass occurred during review.

| Candidate | SHA256 |
| --- | --- |
| launch.py | e73953b40a33e77bbae823acfb9d5759cf962c7d9fc979855bcd9a82531abef4 |
| authorized-template.json | 8236e87e763327456842f555777cc0afa8dc0b90c2b9da2b215933c849b196f5 |
| released child protocol bytes | 097e84b282fa8b45a86ea7759de70f2e8a9e0426368b39ae641f44bdf2725286 |

Source diff against the reviewed Menichetti launcher 2f93c7fd… is confined to NFBS reader/protocol/review/output bindings, metadata/role semantics, CPython and tarfile/gzip binding, and declared caps. Independently compared ASTs: require, digest, bound, child_protocol and OwnedWorker are exactly unchanged. Reused supervisor/helper hashes and their historical Git blobs at 0f7eee3a17b6743fe15549cf8aba396fa4e98a2c validate. The reviewed reader 0488f36b… and false prepared protocol fcee2570… remain exact.

The root gate requires the exact release hash, launcher and source bindings, explicit true release/status and an absent output directory. Child protocol differs from the prepared document only in execution_released=true. Its output path agrees with the supervised directory. The fixed command uses the project Python interpreter, no bytecode writes and the existing single-thread environment. Parent and reader check CPython 3.12.14 plus tarfile/gzip hashes; the frozen dependent TRAIN-family manifest is bound. No source archive is opened by launcher validation.

Limits agree across release/protocol: one attempt, one numerical thread, 120-second worker stage and 10-second cleanup/finalization reserve inside the checked 130-second lifecycle, 256 MiB sampled process-group RSS and 4 MiB aggregate output. Binding preflight precedes that lifecycle. Existing OwnedWorker retains the exact child handle, handles group-signal EPERM through direct-child kill/reap, checks remaining members and prevents failed/unresolved cleanup from succeeding. Clean exits receive no unconditional group signal. Final aggregate output is checked including receipt/protocol; semantic inventory acceptance remains a separate saved-result review. RSS/output are sampled, not hard kernel quotas.

Independent validation: all nine generated process controls passed in 1.132 seconds at ../nfbs-member-inventory-launch-preparation-v1/generated-process-controls-100ipze7. Controls cover success, wall/RSS/output caps, observer failure, actual tiny-process EPERM cleanup/reap, check-only, false release and prepared-protocol preservation. Additional adaptation-controls.json records AST equality, exact source/history checks, release-only child protocol change, consistent caps and zero actual archive open attempts. Tests ran only after root released the model quiet window.

No original archive access, extraction, image headers/arrays, label use, training, changed source roles, payload downloads, tracked edits or commits occurred. TractoInferno intake was untouched. GO applies to this launcher candidate; actual execution remains root-controlled and the original prepared protocol stays false.
