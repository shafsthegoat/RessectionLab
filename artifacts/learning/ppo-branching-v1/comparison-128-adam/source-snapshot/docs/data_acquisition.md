# Public-case acquisition evidence

Checked October 4, 2026. This records acquisition evidence and failures; the three
project specification documents remain unchanged.

## Source revision correction

The current [official TCIA collection](https://www.cancerimagingarchive.net/collection/ucsf-pdgm/)
lists **version 5, May 30, 2025**, while the supplied October 4 specification calls
version 4 current. Version 5 corrects the `DTI_eddy_noreg` NIfTI orientation and
spacing, retaining original post-DICOM-conversion, post-FSL-eddy geometry, and
adds each exam's eddy-rotated b-vectors. Corrected diffusion should use v5; a
shared historical b-vector file is insufficient evidence of correct per-exam
gradient orientation. The official current imaging package is listed as 142 GB;
no full cohort was downloaded.

The official release and its metadata remain CC BY 4.0. Source data attribution:
Calabrese, Villanueva-Meyer, Rudie, Rauschecker, Baid, Bakas, Cha, Mongan, and Hess,
*The University of California San Francisco Preoperative Diffuse Glioma MRI
(UCSF-PDGM)*, dataset DOI [10.7937/tcia.bdgf-8v37](https://doi.org/10.7937/tcia.bdgf-8v37),
descriptor DOI [10.1148/ryai.220058](https://doi.org/10.1148/ryai.220058).
TCIA's [usage policy](https://www.cancerimagingarchive.net/data-usage-policies-and-restrictions/)
and [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) apply to the data.

## Observed official transfer failure

The collection's public Faspex package link successfully authenticated using
its normal public-link OAuth flow; no personal account or secret was used.
The server's own [public API documentation](https://faspex.cancerimagingarchive.net/aspera/faspex/api)
and public client configuration document the relevant package operations.

| Attempt | Observed result |
| --- | --- |
| Current package 1065, package details | `UCSF-PDGM Version 5`, `files_on_server: no` |
| Package 1065, `/files/received` and `/files/received/page` | HTTP 500, code 1202, HSTS directory-browsing failure |
| Package 1065, HTTP download transfer specification | HTTP 400, code 1228, source files unavailable on HSTS |
| Historical v4 link, package 758 | Title `UCSF-PDGM-v3-20230111`, `files_on_server: no`; file browsing also HTTP 500/code 1202 |
| Current server configuration | `http_gateway_url: null` |

These are observed service failures, not a license or patient-eligibility
failure. Installing a transfer client would not establish that missing source
files exist. The package's numerical `total_bytes` was inconsistent with its
own description, so it was not used as a storage forecast. No source bytes from
the official imaging package were available for comparison in this attempt.

## Small structural case obtained

To keep the structural viewer moving, one public mirror case was acquired from
[MedOtter/UCSF-PDGM](https://huggingface.co/datasets/MedOtter/UCSF-PDGM) at immutable
revision `e9372219cf1cd2fdd52260cd45f7514b4aa7638e`. The mirror declares CC BY 4.0
and provides structural images with supplied tumor annotations.

**Status: public structural mirror; equivalence to official source bytes and
the original imaging release are unverified.** The application must not label
these files as an official v5 download, complete patient reconstruction, or
patient-specific functional localization. Structural integrity and provenance
checks support restricted, annotation-assisted development. They do not satisfy
the required corrected-diffusion acquisition milestone.

The preselected first baseline identifier is `UCSF-PDGM-0004`. Its four structural
images and source annotation total **11,559,726 bytes**, about 11 MB compressed.
Each downloaded SHA256 and size matches the pinned mirror's Git LFS object.
No images were modified locally. Local files are in:

```text
data/public_mirrors/MedOtter-UCSF-PDGM/e9372219cf1cd2fdd52260cd45f7514b4aa7638e/UCSF-PDGM-0004/
```

The machine-readable record is `manifests/data_acquisition_ucsf.json`. The
selection preceded any route-planning result; the case is permanently a
development/demo case. Official metadata maps its shorter source identifier
`UCSF-PDGM-004` to **BraTS2021_00097, Training**. Do not call a pipeline using an
overlapping pretrained segmenter unseen end-to-end evaluation.

All five images have identical 240 × 240 × 155 geometry with 1 mm spacing,
finite arrays, and agreeing qform/sform (both code 1). The NIfTI world convention
is RAS+, while increasing array axes are L, P, S. The voxel-to-world matrix is:

```text
-1  0  0    0
 0 -1  0  239
 0  0  1    0
 0  0  0    1
```

This is the downloaded preprocessed grid; native scanner anatomy and the full
preprocessing transform chain are not available. Array orientation is not a
license to invent anatomical laterality after a different transform.

| Source label | Radiological compartment | Voxels / mm³ |
| --- | --- | ---: |
| 1 | Non-enhancing or necrotic core | 7,593 |
| 2 | FLAIR abnormality | 18,654 |
| 4 | Enhancing target | 15,672 |

These numbers describe a supplied annotation, not removed tissue. In particular,
FLAIR abnormality is not established to be disposable tumor. No patient-specific
motor, language, or vascular evidence is present in the structural subset.
Clinical deficit probabilities remain unavailable.

The official clinical CSV and glossary were downloaded separately as audit
references. They include final diagnosis, molecular measurements, survival,
and extent of resection without per-field availability times. They are **not
permitted preoperative optimizer inputs** on that basis. The downloader does
not inject those columns into a planner. An unknown date remains unknown.

## Reproduction and checks

Run from the repository after installing its normal Python environment:

```sh
.venv/bin/python scripts/acquire_public_case.py --dry-run
.venv/bin/python scripts/acquire_public_case.py --inspect-nifti
.venv/bin/python scripts/acquire_public_case.py --verify-only --inspect-nifti
.venv/bin/python -m pytest tests/test_data_acquisition.py -q
```

The downloader verifies pinned sizes and SHA256, enforces the manifest's
20 MB ceiling, resumes interrupted partial files with exact HTTP range checks,
and refuses to overwrite existing files whose bytes differ. Completed arrays
and QC output remain under ignored `data/`. It requires no cloud service or
account. An existing source file with a mismatch is preserved for inspection.

Thirteen download regression tests passed, covering idempotence, corruption,
resume handling, ignored ranges, size bounds, hash mismatch, duplicate paths,
and destination/symlink escape. The real-case SHA256 and NIfTI inspection
also passed. `data/UCSF-PDGM-0004_acquisition_report.json` records the source
manifest hash and checked arrays. Header alignment is not a clinician's review
of registration or anatomical plausibility; those remain separate gates.

The independently acquired creator-linked BTC diffusion case is documented in
`docs/diffusion_source.md`; it is a different patient and must never be fused
with this UCSF anatomy as though the modalities belonged to one person.
