# Verified diffusion-source fallback

Checked and acquired October 4, 2026. This is a targeted acquisition result, not a completed tract-reconstruction or clinical validation result.

## Decision and source

Use **one BTC_preop development case, `sub-PAT28`**, to unblock the diffusion pipeline while the preferred UCSF-PDGM archive is unavailable. BTC is already the P2 multimodal resource in the master plan. This is an explicit departure from UCSF-first acquisition; it does not complete the UCSF five-case milestone and this case must not later be presented as untouched external evaluation. Keep UPenn reserved for the frozen external evaluation.

The data-acquisition investigation observed the official UCSF-PDGM Aspera packages for both the current v5 and historical v4-linked release reporting `files_on_server=no`, with failed browse/download requests. The v5 official release fixes diffusion headers and adds per-exam rotated gradients, so substituting an old image plus generic gradients would also need a frame audit. See the [official TCIA collection](https://www.cancerimagingarchive.net/collection/ucsf-pdgm/) and the acquisition log for that component. The structural mirror already acquired in the project is not a source of raw directional diffusion.

The replacement is the creators' [BTC_preop OpenNeuro archive](https://openneuro.org/datasets/ds001226/versions/5.0.1), identified by the [creator dataset descriptor](https://doi.org/10.1038/s41597-022-01806-4). The [pinned repository description](https://github.com/OpenNeuroDatasets/ds001226/blob/359d372c5e972a161966312128adb365870df949/dataset_description.json) records:

- Dataset DOI: `10.18112/openneuro.ds001226.v5.0.1`.
- Snapshot: `5.0.1`; Git commit `359d372c5e972a161966312128adb365870df949`.
- Authors: H. Aerts and D. Marinazzo.
- License: **CC0**, as recorded in that snapshot. Data license and article license are separate.
- The supplied acknowledgment asks users to acknowledge the OpenNeuro database.

OpenNeuro's [official Git documentation](https://docs.openneuro.org/git.html#git-annex-special-remote) describes public annexed objects being available through its S3 remote. Images were downloaded from the public `openneuro.org` S3 bucket, with each response version pinned. Every imaging object matched both the byte count and MD5 recorded in the pinned Git annex pointer. SHA-256 hashes were then computed locally. No anonymous third-party imaging mirror was used for this case.

## Downloaded artifacts

All volumes are ignored by Git. The local root is `data/diffusion_source/ds001226-v5.0.1/`.

| Artifact | Downloaded bytes | Source-content verification |
|---|---:|---|
| Native T1-weighted MRI | 7,444,400 | Annex MD5 `2ba1aa890588b55f777b88da805d428e` |
| AP 4-D diffusion MRI | 45,750,278 | Annex MD5 `a593165993d2b18d04e61a6856c4ec15` |
| Reverse PA b0 MRI | 1,106,184 | Pinned annex MD5 and length checked; see manifest |
| Native-T1 source tumor mask | 10,486,112 | Annex MD5 `15531588aad08c214c4e0613412a6502` |
| Gradients, acquisition JSON and release documents | 21,789 | Pinned Git content; local SHA-256 recorded |
| **Total** | **64,808,763** | All 15 selected files downloaded |

`acquisition_manifest.json` records every relative path, exact versioned download URL, byte count, source MD5 when annexed, SHA-256, release and retrieval time. `initial_qc.json` records the actual image headers and first content checks. The small acquisition prototype is `data/diffusion_source/fetch_verified_btc.py`; it intentionally lives in the ignored acquisition workspace pending incorporation into the maintained downloader. It requests only this subject's anatomy, DWI and native tumor mask, plus four release documents. It does not download fMRI, other subjects or postoperative data.

## Initial QC findings

| Input | Observed shape | Voxel spacing | Orientation |
|---|---|---|---|
| T1 | 160 × 256 × 256 | approximately 1 mm isotropic | RAS |
| Source tumor mask | 160 × 256 × 256 | 1 mm isotropic | LAS |
| AP DWI | 96 × 96 × 60 × 102 | 2.5 mm isotropic | LAS |
| PA reference | 96 × 96 × 60 × 2 | approximately 2.5 mm isotropic | LAS |

All arrays contain finite values. Physical units are millimeters. Each input's qform and sform agree within `1.4e-5` in their matrix entries. This is a header consistency check, not evidence of registration accuracy.

AP has **102 matching b-values and 3 × 102 b-vectors**: 6 b0, 16 at b=700, 30 at b=1200 and 50 at b=2800 s/mm². The b0 vectors are zero, and weighted gradient norms differ from 1 by at most `6.3e-7`. PA has two b0 volumes and matching zero vectors. JSON phase-encoding directions are `j-` for AP and `j` for PA; both record `TotalReadoutTime=0.0266003` seconds. Their spatial affines differ slightly, so a preprocessing implementation must check/reconcile reference grids before treating them as a correction pair.

The T1 and mask arrays have opposite first-axis orientation. The measured mask-voxel-to-T1-voxel transform is approximately `x_T1=159-x_mask`, with the other axes unchanged. Array indexing alone would put the mask on the wrong side; use the physical affine. The largest departure from the ideal flip matrix is about `3.2e-4` voxel across the grid.

**The source mask is not a categorical integer segmentation.** Its stored dtype is uint8, but the NIfTI scale is approximately 1/255, yielding all 256 fractional levels from 0 to approximately 1. Preserve this supplied source. Do not cast scaled values directly to integers. A derived target needs an explicit threshold and resampling policy, with sensitivity analysis: the native-grid voxel counts are 10,859 at threshold 0.25; 10,269 at 0.5; and 9,628 at 0.75 (approximately mm³ at this spacing). Nonzero support is 15,178 voxels. These values are not radiological tumor compartments or observed resection volumes.

## Eligibility and remaining gates

The [pinned participant table](https://github.com/OpenNeuroDatasets/ds001226/blob/359d372c5e972a161966312128adb365870df949/participants.tsv) identifies `sub-PAT28` as a 44-year-old participant with a frontal oligodendroglioma II and a reported tumor size of 11.49 cm³. Selection used a source-confirmed glioma with frontal location, a modest released file size and the required modalities; no planner score was used. The diagnosis supports cohort eligibility only. Its preoperative availability is not established, so it must not enter the primary preoperative planner as known molecular/pathology context. No postoperative data were acquired.

The source's T1-only structural set does not satisfy the preferred four-contrast structural contract. It can support an explicitly annotation-assisted geometry/diffusion development case; do not silently invent contrast-enhanced, T2 or FLAIR compartments. The creator paper's methods describe manual T1 delineation, automatic lesion-segmentation refinement and Gaussian smoothing with a 3 mm standard deviation for tractography threshold modulation. Its Data Records section identifies native-T1 and MNI derivatives. These methods are consistent with a fractional annotation, but do not prove that every released file underwent that exact smoothing. The values are not established probabilities. Full methods and Data Records were read through the [Europe PMC article XML](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC9637199/fullTextXML).

Required before accepted motor/language reconstruction: diffusion motion/distortion preprocessing, documented rotated gradient handling, diffusion-to-T1 registration and visual/landmark QC, suitable shell/model selection, side/endpoint/continuity checks, and named unknown-coverage regions. A quick tensor/FA or principal-direction demonstration should be labeled uncorrected exploratory output until those steps are done. No CST, language tract, functional localization or neurological-deficit probability has been validated by this acquisition.

The current evidence establishes **verified source files and a usable raw-diffusion input candidate**, with two useful real import hazards (opposite array orientations and fractional source labels). It does not establish a clinically usable route or a complete cohort.
