# One RHUH image: bounded offline inspection helper

Prepared only. No patient header or voxel was opened to develop or test this helper. `scripts/inspect_rhuh_single_image.py` has no network/download function, no case-import operation and no scientific-use promotion. The only prospective source is the already selected RHUH-0001 visit-0 T1 filename in acquisition proposal V2. That filename remains a source label, not verified modality, identity or anatomy.

## Authorization and provenance

Default invocation accepts a hash-pinned request JSON and performs **metadata-only preflight**. It does not open, hash, decompress or allocate from the image. It verifies the exact proposal and checksum-index SHA256 constants, the index's exact source/token row, accepted parent transfer summary, successful launcher and worker records, three acquisition sources, transfer release and an independently accepted byte-reconciliation record. All files have repository-relative paths and SHA256 bindings; symlinks and traversals are rejected. JSON inputs require `.json`, the original index `.sums`, and bound implementation sources `.py`.

Request schema `resectionlab.rhuh-image-inspection-request.v1` has exactly: `schema`, `source`, `proposal`, `checksum_index`, `parent_summary`, `parent_execution`, `worker_receipt`, `reconciliation`, `payload`, `acquisition_sources`, `transfer_release`. Each file binding is `{path, sha256}`. `acquisition_sources` uses `image_helper`, `public_transfer`, `checksum_helper`. `payload` must equal both parent and worker payloads and the independent reviewer payload, including actual compressed length, SHA256, candidate MD5, fixed published token and original quarantine path. The unknown prior length stays null; MD5 compatibility does not confirm the publisher's algorithm.

The reconciliation schema is `resectionlab.rhuh-compressed-byte-reconciliation.v1`. Its required keys are `schema`, `accepted`, `source`, `proposal_sha256`, `checksum_index_sha256`, `resolved_source_sha256`, `parent_summary`, `parent_execution`, `worker_receipt`, `payload`, `immutable_original`, `publisher_algorithm_confirmed`, `prior_length_match_claim`, `reviewer_source`, `acquisition_sources`, `transfer_release`, `decoded`, `scientific_use_released`. The last two remain false. A fixed documentary allowlist also permits `acquisition_source`, `checker_development_note`, `consulted_inputs`, `history_checks`, `immutable_original_meaning`, `independent_computation`, `limits`, `original_mode`, `source_archive`; these never substitute for required joins or enter the image report. Unknown keys fail. A preserved worker-receipt copy can have a different path: the parent must name the original receipt derived from the exact quarantine directory, and both original and copy must match the independently bound SHA256. `immutable_original=true` means independently reconciled preserved compressed bytes; read-only mode is not proof of irreversible filesystem immutability.

A distinct `resectionlab.rhuh-image-inspection-release.v1` must bind the request, this inspector's exact source SHA256, fixed source, `released=true`, and action `bounded_header_and_voxel_inspection_once`. Its exact keys are `schema`, `released`, `action`, `request`, `inspector_source_sha256`, `source`. No such execution release is created by this preparation. The surrounding root workflow must enforce one attempt; this stateless helper does not maintain a persistent release-consumption ledger.

Future default syntax, after genuine accepted receipts exist:

```text
python scripts/inspect_rhuh_single_image.py --request <relative-request.json> --request-sha256 <exact-sha256>
```

Execution additionally requires `--execute --release <relative-release.json> --release-sha256 <exact-sha256>` and a separate root release. No actual CLI invocation against the source was performed here.

## Decode limits and supported subset

The released path first reads the complete bounded original compressed file, checks its regular-file type, observed length, inode/stat stability and independently bound SHA256/MD5, and then inspects those same in-memory bytes. This prevents any header interpretation before the original-byte identity join.

| Limit | Frozen bound |
|---|---:|
| Original compressed payload | 64 MiB |
| Initial decompressed header/extension marker | 352 bytes for NIfTI-1; 544 for NIfTI-2 |
| Header, extension region and voxel offset | 1 MiB |
| Total decompressed bytes | 256 MiB |
| Calculated voxel storage before allocation | 255 MiB |
| Dimensions | Exactly scalar 3D; each axis 1–512 |
| Voxel product | At most 64,000,000 |

Installed NiBabel supplies header layout, byte order, data type, scaling and qform/sform semantics. Accepted formats are single-file NIfTI-1/2, either byte order, with standard signed/unsigned 8/16/32/64-bit integers or 32/64-bit floats. Complex, RGB/RGBA, extended-precision, paired header/image and non-3D layouts are unsupported. Integer dimensions/product/storage and finite integer offset are checked before voxel storage allocation. Unused dimensions must be zero or one. NIfTI-2 newline checks and datatype/bitpix agreement are enforced.

Streaming zlib decompression bounds each output request and rejects malformed streams, CRC/footer failures, truncation, extra gzip members (including empty members), compressed trailing bytes, and any discrepancy between declared and actual uncompressed length. Voxel bytes are interpreted **only after** footer and exact EOF validation. Extensions stay opaque: only bounded 16-byte-aligned framing/count is checked; no extension text/DICOM parsing occurs. Unmarked padding must be zero. This intentionally conservative subset can refuse a file accepted by a permissive reader.

Voxel summaries use bounded float64 chunks: finite/nonfinite counts, finite minimum/maximum/mean, and explicit effective NIfTI scaling. Overflowing summary sums produce a null mean. Float64 summaries of wide integers may lose integer precision. Nonfinite voxel values are counted, not silently replaced or treated as known tissue absence. No percentiles, masks, normalization, registration or anatomical inference is performed.

## Meaning and tests

The result whitelist contains shape, scalar dtype, storage counts, units/zooms, coded qform/sform matrices, finite/invertible flags, orientation labels, a full-cell-corner numeric comparison and intensity summaries. Arbitrary `descrip`, `aux_file`, intent names and extension bodies never appear. Fixed source/receipt bindings are provenance, not clinical claims. Unknown physical units, absent/invalid transforms and differing qform/sform frame codes are explicit unresolved geometry issues, separate from malformed binary storage. No fallback affine is invented. The 0.001 mm comparison threshold is an engineering reporting tolerance, not anatomical registration acceptance. Scanner/atlas provenance, anatomical alignment and scientific/clinical usability remain unaccepted even when binary structure passes.

Owner controls use only small generated format/receipt fixtures. Thirty-eight checks passed in 0.26 s, including both formats/endians, scaling/nonfinite data, unknown/conflicting frames, malformed headers, all storage caps, exact initial peek, opaque extensions, CRC/truncation/trailing-member refusal, receipt/source/release failures and no-image-open preflight. An early NIfTI-2 magic-field assumption was corrected using NiBabel's separate `magic`/`eol_check` fields; an extension fixture length was corrected. Independent review added 82 analytical controls: 120 combined checks passed in 0.51 s on unchanged source `268d23d8560c963461f633fe2d66ef581b8bd47e74e6431b89c23d7058979a6f`. The receipt is `artifacts/rhuh-single-image-inspection-review-v1/verification.json`, SHA256 `ef93e68a74d5277d23b5ed8bbbe8074defa53722e503d4c936cc679c92bef11d`. Its initial `.txt`/`.sums` fixture mismatch is retained as a test-fixture correction, not a production defect. This documentation update follows the frozen-source review. These checks do not establish patient-image correctness or measured native-image resource performance.
