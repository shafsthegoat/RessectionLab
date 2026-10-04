# RessectionLab: targeted public-data acquisition

Prepared October 4, 2026 for the branch reviewed at `5ca02451d63412d074281f8fde5407420a2f0903`.

These are verified source listings and proposed uses, not a claim that new imaging was downloaded or reconstructed during this review. Most collections were already in the master plan. The change is prioritizing specific files that unblock the working planner, including an alternative official download route. Check each release's license, access terms, modality completeness, source hashes and preprocessing before use. Download selected cases, not every collection.

## 1. UCSF-PDGM v5: primary glioma diffusion route

Official collection: https://www.cancerimagingarchive.net/collection/ucsf-pdgm/
Canonical DOI: https://doi.org/10.7937/tcia.bdgf-8v37

The collection lists 495 patients and 501 studies. Version 5, updated May 30, 2025, fixes `DTI_eddy_noreg` headers and supplies per-exam rotated b-vectors. Its description identifies these NIfTI volumes as original orientation/spacing after DICOM conversion and FSL eddy correction, without further processing. This is not evidence that every remaining correction or alignment requirement has been satisfied.

Acquire one selected case's structural sequences, annotations, directional diffusion, matching b-values and that exam's rotated b-vectors. Verify gradient frame and image identity; do not substitute the generic b-vector download for a per-exam corrected vector file. Establish the DWI-to-structural transform, reconstruct motor pathways, retain failure/coverage information, and then extend to supported language bundles. Use corrected source derivatives to avoid unnecessarily making a novel MRI correction solver the critical path.

The collection currently lists Aspera access. A failed transfer remains a source-access problem, not permission to declare an unrelated mirror byte-equivalent. Record unavailable files and advance the independent structural work below.

## 2. BTC: use the already pinned cohort, modalities and derivative documentation

Preoperative release: https://openneuro.org/datasets/ds001226/versions/5.0.1
Postoperative collection: https://openneuro.org/datasets/ds002080
Primary paper: https://www.nature.com/articles/s41597-022-01806-4
Accessible paper record: https://pmc.ncbi.nlm.nih.gov/articles/PMC9637199/

The paper describes T1, diffusion and BOLD imaging, tumor masks, connectivity derivatives, and behavioral data. Its overall tumor sample includes more than glioma; apply the repository's version-specific pathology eligibility rather than treating every tumor as a diffuse glioma. Raw acquisitions are not the same as executed corrected reconstructions.

Preserve the prospective roles in `manifests/experiments/btc-spatial-development-cohort-v1.json`: PAT22/PAT25 additional training; PAT26/PAT27 checkpoint-selection development; PAT29/PAT31 unopened later development-transfer cases. PAT05/PAT16/PAT20/PAT28 were already consulted. Later development transfer is not external final validation.

Inspect `derivatives/dmriqc` and `derivatives/TVB` for existing quality reports and connectome resources before rebuilding every auxiliary product. Regional connectivity matrices can inform network representations but cannot localize a fine surgical corridor by themselves. Postoperative images and tests must not become preoperative planning inputs. Use the author's raw-data preprocessing description for patient reconstruction and label any synthetic imaging separately.

## 3. DES-derived motor/language resources: finish integration before accumulating more maps

Author-linked DES coordinates/demographic release: https://zenodo.org/records/10439149
Language structural/functional maps, version 2: https://zenodo.org/records/16418628
Primary paper and availability statement: https://www.nature.com/articles/s43856-025-01121-0
Author implementation: https://github.com/FBK-NILab/White-Matter-DES-Brain-Maps

The language release lists `pub_release.zip` (19.1 MB), including `maps_paper_release` and data/scripts for selected figures. The paper links the earlier coordinates release. The older record was not directly retrievable during this review; its URL is supported by the authors' availability statement. Inspect release-specific license information before redistributing data or derived assets.

The repository already has seven prior proposals. Use their source records rather than creating duplicate downloads. Preserve template space, map meaning, interpolation and coverage when registering to a patient. Integrate explicit population-prior sensitivity analyses into plan evaluation; do not silently certify them as individualized functional anatomy. Use patient diffusion when available to refine the representation. The paper explicitly does not publicly release its glioma raw MRI, derivatives and clinical scores, so these maps cannot alone train clinical injury probabilities.

## 4. ReMIND through NCI Imaging Data Commons: anatomical cases and later updating

Official IDC collection: https://portal.imaging.datacommons.cancer.gov/collections/remind/
TCIA: https://www.cancerimagingarchive.net/collection/remind/
Canonical DOI: https://doi.org/10.7937/3rag-d070
Primary paper: https://www.nature.com/articles/s41597-024-03295-z
IDC download guide: https://learn.canceridc.dev/data/downloading-data

IDC lists 114 tumor patients with MRI, segmentations and intraoperative ultrasound. The collection includes preoperative tumor, pre-resection cerebrum and residual tumor segmentations where available. Select glioma cases using the accompanying clinical metadata; do not assume every case is glioma.

First select a small declared development subset with suitable preoperative anatomy and available cerebrum/tumor labels. Use these as a separate geometry/segmentation-assisted track rather than spending all effort deriving a brain envelope from full-head BTC scans. Cerebrum segmentation still does not establish a clinically appropriate cortical access window or functional safety.

Later use the intraoperative images to evaluate registration, modeled displacement and observation updates. They are snapshots, not a labeled sequence of surgical instrument actions. Never imitate the observed cavity as if it were the proven optimal resection.

IDC provides direct downloads and `idc-index` access. Select patients/series through the portal or supported client filters. Its collection-wide example is `idc download remind`, but do not use this as the default acquisition command for a small local pilot. Pin DICOM series identities and preserve reference-image/frame relationships during conversion.

## 5. UPenn-GBM through IDC: independent structural evaluation

Official IDC collection: https://portal.imaging.datacommons.cancer.gov/collections/upenn_gbm/
TCIA: https://www.cancerimagingarchive.net/collection/upenn-gbm/
Primary paper: https://www.nature.com/articles/s41597-022-01560-7

IDC lists 630 GBM patients and MR/SEG data. The collection description includes multiparametric imaging, tumor and brain segmentations, and clinical/genomic metadata.

Use a prospectively selected independent-site cohort for structural robustness and annotation-assisted route evaluation. Keep development and final evaluation roles explicit. Verify the exact selected raw and processed files: having diffusion scalar maps does not establish access to full directional diffusion or valid gradients. Do not assume all 630 patients support tractography. Processed NIfTI and raw DICOM may require explicit coordinate reconciliation.

The IDC portal/client provides an alternative to TCIA transfer for the hosted DICOM collection; supplementary NIfTI/clinical resources may still use separate collection links. Download only selected cases and verify DICOM SEG references and the exported physical frame.

## 6. TractoInferno: upstream tractography validation and initialization

Public dataset: https://openneuro.org/datasets/ds003900
Version cited by the dataset paper: https://openneuro.org/datasets/ds003900/versions/1.1.1
Primary paper: https://www.nature.com/articles/s41597-022-01833-1

This resource contains 284 healthy-subject samples from six sites, T1/diffusion-derived data, fODFs, tissue masks and reference streamlines for 30 bundles derived from multiple tractography algorithms. Start with a small sample respecting its published splits.

Use it to validate tract handling, physical-coordinate conversion, bundle extraction, coverage semantics and optional tract-model initialization before transferring to glioma anatomy. Reference streamlines are algorithm-derived rather than histological truth. Healthy anatomy cannot establish tumor-domain accuracy or clinical risk calibration. Synthetic lesions inserted into these cases remain simulated cases, not human glioma training patients.

## Common acquisition contract

Each acquired case needs patient/visit/group identity, release and source URLs, checksums, sequence/gradient identity, processing history, frame transforms, license, expected use, unavailable fields and split assignment. Track human patients separately from synthetic environments and optimizer seeds. Preserve expert-review status separately from source annotations and engineering QC. No dataset listed here supplies the complete patient-specific surgery-action/functional-outcome supervision needed to equate simulated encounters with postoperative deficit probabilities.
