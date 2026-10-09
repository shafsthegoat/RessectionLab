# Versioned calibration runtime migration and failed first data read

The first HBE branch-calibration attempt remains failed before any measured access or native solve. Its pinned Python executable changed after successful preflight. The [exact predecessor receipts](../hbe-branch-calibration-runtime-drift-v1/RESULT.md) are preserved at their original paths; v2 authenticates them before reading curves.

Commit `2054fb265da36c36aeb276ad30b889c7347bcaf9` adds a separate v2 study and branch modules. All 23 scientific study fields, eight historical declarations, 26 inherited Python modules, geometry, material assumptions, exact axial/held-out member split, FEBio runtime/profile and one-hour resource bounds remain unchanged. Eighty-two owner and 15 independent controls pass. The reviewer verified a 38-file source archive against that commit. A separate real metadata/source preflight passed in 4.968568 seconds with no curve or solver access.

Root authorized release SHA-256 `d286f46e616036be399c2acd69741ce2fcb6a629ab13d62ebbe4fc74e9ac94f7` and started one supervised experiment at approximately 01:22 UTC October 9. It failed while parsing the allowed axial calibration files: the frozen schema tried to convert the column label `displacement` to a number. The ledger records one calibration access **attempt**, while `calibration_responses_accessed` remains unknown. The held-out torque access flags are false. No fit, deck preparation, mesher or native solver call occurred. The worker exited 1 after 6.739999 seconds, with 228,818,944 bytes sampled peak process-group RSS. The outer harness finished after 7.149769 seconds. This is a real negative execution result, not a material-model performance result; there was no automatic retry.

Correcting the observed CSV framing requires a separately versioned study and release. Physical agreement, patient properties, tissue-contact forces and surgical benefit remain unestablished.

The active archive download is independent. No patient images, learned weights, native bulk logs or measured curves are included here.
