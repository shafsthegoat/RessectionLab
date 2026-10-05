# BTC PAT16 and PAT20 structural development cases

Both cases were selected from metadata before image access, with permanent
development roles in the queue committed as `069c962`. The bounded acquisition
and preparation implementation was committed as `ae1d36b` after 77 focused
offline tests. The exact queue SHA256 remains
`a768aaefd78146540cf20c13193e22268344206a4a396ffb654e7d40af7b8898`.

Acquisition used [BTC_preop ds001226 v5.0.1](https://openneuro.org/datasets/ds001226/versions/5.0.1),
Git `359d372c5e972a161966312128adb365870df949`, DOI
`10.18112/openneuro.ds001226.v5.0.1`, CC0. Each native T1 and supplied annotation
matched the published annex MD5 and source length before publication. Computed
SHA256 values and exact object-version URLs are recorded in
`manifests/btc_pat16_acquisition.json` and `manifests/btc_pat20_acquisition.json`.
The four shared release metadata files were reused after verification.

| Result | PAT16 | PAT20 |
| --- | ---: | ---: |
| New subject source bytes | 18,195,685 | 17,531,332 |
| Seven-file scope including shared metadata | 18,210,576 | 17,546,223 |
| Bundle bytes | 9,615,059 | 8,885,710 |
| Shape | 160 × 256 × 256 | 160 × 256 × 256 |
| Source target voxels at threshold 0.5 | 45,400 | 12,451 |
| Threshold 0.25 / 0.75 sensitivity, voxels | 47,772 / 43,048 | 13,811 / 11,192 |
| Save/reopen identity | Passed | Passed |

The two downloads added **35,727,017 subject bytes**; the shared 14,891 bytes
were already present. No diffusion, fMRI or postoperative images were acquired.
Original images and annotations were not rewritten. The 0.5 threshold remains
the predeclared research scenario, with source-value sensitivity reported
separately; none of these values is a clinical probability or resection result.

The prepared bundles are `outputs/cases/BTC-sub-PAT16.ressectionlab` and
`outputs/cases/BTC-sub-PAT20.ressectionlab`. Each preserves full-head coverage,
an absent reviewed brain mask, blocked automatic cortical access, missing
directional diffusion, unknown patient function and unavailable clinical
context. No pathology or molecular field was admitted to the optimizer.

Source-bound acquisition/preparation receipts, exact commands, source hashes,
case hashes and copied QC reports are in
`artifacts/validation/btc-queued-structural-v1/`. No acquisition or preparation
failure occurred, and implementation hashes match the preceding test receipt.
Independent source/geometry and three-plane overlay review passed for both
cases. The native T1 axes are RAS and annotation axes LAS; explicit first-axis
reindexing gives maximum corner residuals of 0.0003265 mm (PAT16) and
0.0001086 mm (PAT20). Independent threshold masks match the saved bundle masks
at every voxel, and independent save/reopen preserves semantic and planning
hashes. Both nonzero full-head shortcuts were rejected by the planner.
The receipts are `docs/btc-pat16-independent-qc.json` and
`docs/btc-pat20-independent-qc.json`. They explicitly leave whole-volume
annotation accuracy, reviewed brain/cortex, patient function and tract or
clinical validation unverified.

The acquired-case registry was extended after these checks. It now records
five development representations, four with verified primary-source identities,
and no final patients. PAT16/PAT20 are linked to their frozen development queue,
participant rows, exact source hashes and independent review evidence.
Historical records, aliases and final reservations remain unchanged. The local
registry audit verifies sixteen evidence records; unknown-timed diagnosis is
excluded from policy inputs, and both new cases retain no reviewed brain mask
and no acquired directional diffusion.

These two cases broaden development only. The UCSF primary-source transfer and
five quality-controlled UCSF milestone remain unresolved, as recorded in the
frozen queue. Related visits and derivatives must remain grouped with the same
development person; neither candidate can later become outer-final evaluation.
