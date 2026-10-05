# RHUH-GBM postoperative imaging: source audit

Audit: October 5, 2026. **The public processed-image inventory exists, but no
patient images were downloaded or inspected. Injury labels are not established.**
Clinical outcome counts, missingness and definitions are owned by the separate
clinical audit; this note covers imaging and the smallest useful next experiment.

## Access and actual inventory

[TCIA version 1, June 9, 2023](https://www.cancerimagingarchive.net/collection/rhuh-gbm/)
lists brain-extracted NIfTI images and segmentations under **CC BY 4.0**: 40
patients, 720 entries, advertised 2.9 GB. Its raw DICOM download is controlled;
that route was not accessed. The public
[series digest](https://www.cancerimagingarchive.net/wp-content/uploads/RHUH-GBM_v1_20230606-nbia-digest.xlsx)
contains 40 IDs, 120 studies, 600 MR series, 37,425 instances and 15,964,322,448
bytes. Metadata access does not grant raw-image access.

The collection's public Faspex package **684**, `RHUH-GBM-nii-v1`, returned
`files_on_server=yes` and successful directory listings. A bounded metadata-only
traversal verified exactly `RHUH-0001` through `RHUH-0040`, each with visits
`0`, `1`, `2`, and six names per visit:

`RHUH-0001/1/RHUH-0001_1_{adc,flair,t1,t1ce,t2,segmentations}.nii.gz`

There is one exact naming exception: `RHUH-0035/2/segmentation.nii.gz` is singular
and lacks the patient/visit prefix. It needs an explicit manifest mapping; the
source was not renamed. The remaining 719 names follow the pattern above.

Every image entry is a **symbolic link with blank size**. The package's reported
43,065 bytes is not a valid imaging-download budget. The listed checksum file,
`/RHUH-GBM-nii-v1.sums`, is 61,787 bytes; its contents were not retrieved. Thus
the 720 names are verified, while payload lengths, checksums and successful
transfer remain unverified. No link target was followed, transfer specification
requested, or image bytes downloaded. Compact receipts and the compressed full
listing are in `artifacts/rhuh-postoperative-imaging-audit-v1/`.

## What the release can and cannot label

The [dataset paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC10551826/) assigns
visits to preoperative, early postoperative **within 72 hours**, and progression.
All five modalities are inclusion requirements. The public inventory provides
ADC maps, **not a separate raw DWI/b-value/b-vector file set**. Diffusion-related
names in the restricted DICOM digest do not establish accessible DWI or gradients.
Exact postoperative scan hours have not been verified.

The described expert-corrected labels are necrosis, peritumoral signal change
(edema plus nonenhancing tumor), and enhancing tumor. Early postoperative
enhancement can support a residual-enhancement measurement. No separate cavity,
ischemia, infarct, diffusion-restriction or neurological-injury annotation is
documented in the inspected sources/listing. Necrosis is not a cavity label;
peritumoral signal change is not an injury label. These would require a defined,
independently reviewed derivation or new annotation.

## Two source inconsistencies block premature injury thresholds

The [author repository, pinned at `46cf3825`](https://github.com/smcch/RHUH-GBM-dataset-MRI-preprocessing/tree/46cf3825278243036c08131b62dcfcfb1d3bba39)
disagrees with the paper in scientifically important ways:

| Field | Paper | Inspected author code |
| --- | --- | --- |
| Enhancing label | `3` | README/Python use `4` |
| ADC intensity | Unnormalized, physical mm²/s | Both shell and Python normalize ADC by Z-score; optional ADC calculation also applies a ×1,000 scale first |

The scripts were inspected, not executed. They may differ from the pipeline
that produced the released files. Actual label values, NIfTI scaling and ADC
distributions therefore require verification; neither a literature ADC threshold
nor a contralateral ADC ratio is valid merely because a file is named `adc`.
Source hashes and exact code URLs are recorded in `source-audit.json`.

## Grid and longitudinal limits

The paper describes T1ce-to-SRI24 registration, within-visit coregistration,
1 mm resampling, skull stripping and structural intensity normalization. These
are processed atlas-space derivatives, not native scanner samples. The inspected
shell uses 12-parameter affine registration and trilinear interpolation; the later
Python variant uses translation/rigid/affine stages. Neither inspected output
inventory contains saved transform matrices or a cavity-aware longitudinal
registration. A shared atlas grid does not prove voxelwise correspondence after
resection. Affine scaling and interpolation also prevent treating exported
voxel volumes or narrow ADC boundaries as untouched native measurements.

## Smallest defensible next experiment

First declare one development patient by metadata/ID, with outcome labels hidden
from imaging review. Retrieve its **two visits only**, subject to a source-bound
byte limit once link sizes/checksums are resolved. Verify label values, ADC units,
qform/sform, modality coverage and registration visually and numerically. Draw or
review a postoperative cavity independently; preserve disagreement and uncertain
boundaries. This establishes measurement feasibility, not predictive performance.

If units and review support it, the first association test should ask whether
**observed peri-cavity low-ADC burden/location adds information beyond available
pre/post tumor-volume measures and baseline KPS**. Compare with contralateral
and preoperative controls; do not label low ADC as confirmed ischemia without
appropriate image/reference evidence. Aggregate deficit labels cannot validate
motor/language-specific injury. The full small cohort needs patient-grouped
evaluation and severe-class uncertainty; all visits stay together. A one-pair
pilot cannot answer that association question.

This is retrospective postoperative evidence. It cannot enter a preoperative
actor, become a reward for an unperformed route, or establish causal route harm.
The direct dependencies are bounded source transfer, ADC/label reconciliation,
injury-region review and outcome-timing clarification—not a larger neural network.

Dataset attribution: Cepeda and colleagues (2023), *RHUH-GBM*, TCIA,
[DOI 10.7937/4545-C905](https://doi.org/10.7937/4545-c905).

## October 5: fixed one-pair transfer preparation

The integration owner fixed **RHUH-0001, visits 0/1**, before this preparation;
this preparation did not consult clinical outcome rows. Cohort outcomes were
previously inspected and a baseline fit completed, so this is neither a blinded
nor held-out cohort. Exact-path requests for the 12 images and,
separately, the checksum file produced supported Connect transfer specifications
(HTTP 201). Specifications contain transfer credentials, which were kept in
memory and omitted from receipts. No transfer was started.

The provider's public configuration reports `http_gateway_url=null` and prompts
users to install Connect. An HTTP-Gateway specification returning 201 therefore
does **not** establish an available HTTP download service. No Connect/ascp client
was found locally. A selected-file query to the directory-browse endpoint failed
with HSTS code 1202; it neither resolved the symbolic link size nor proved that
the file is unavailable.

`artifacts/rhuh-one-pair-acquisition-preparation-v1/bounded-pair-manifest.json`
fixes the 12 source paths, ignored output destination and **256 MiB compressed
image ceiling**. Acquisition remains disabled: lengths/checksums are unknown,
the 61,787-byte checksum file is still unfetched, and root review is required
before image acquisition. The next concrete dependency is a supported
[Connect/FASP client](https://www.ibm.com/docs/en/aspera-faspex/5.0.x?topic=packages-downloading-package-connect)
for checksum-only retrieval, or a provider-offered HTTP service. No access grant,
account signup, controlled DICOM request, billing change or outcome lookup was
attempted. Every eventual image must pass size/checksum checks before decoding.
