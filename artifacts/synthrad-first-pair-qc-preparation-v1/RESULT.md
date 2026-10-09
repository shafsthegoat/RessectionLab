# SynthRAD first TRAIN-pair QC preparation

This preparation freezes a single restricted CT/MR structural-QC attempt for
SynthRAD Task1 brain subject `1BA336`, the first subject in the existing TRAIN
order. It binds the verified 14.47 GB archive, exact central-directory member
records, role and identity joins, source/rights proofs, resource limits, a
fixed display policy, and an initially unauthorized execution declaration.
Only the selected pair may be extracted by a separately frozen execution.

The runner keeps staged images and all preview pixels in restricted local
storage. Portable receipts contain hashes, scalar/geometry summaries, stage
and failure lineage, and explicit pending human review. They do not release
image views or grant anatomy, scanner-frame, training or planning admission.
CT values are not assumed to be calibrated HU; MRI values remain source units.

The [independent source review](final-source-review.md) verified all seven
top-level proof pins and all 28 nested source proof entries, including tracked
copies of two acquisition completion receipts. Its isolated metadata preflight
required no ignored proof file or image payload. The 115 combined offline
controls passed. No ZIP image member, patient voxel, model or optimizer was
accessed in preparation. The archive itself remains an external, stat-bound
verified source and cannot be relocated without a new fixity review.

The next bounded step is one current-checkout execution after a root declaration
and resource/output checks. Even a mechanically successful run remains
`review_pending` until restricted visual privacy, anatomy, overlap and source
scaling reviews are completed. No broader cohort queue is released here.

Implementation: [runner](../../scripts/synthrad_first_pair_qc_v1.py) and
[tests](../../tests/test_synthrad_first_pair_qc_v1.py).
