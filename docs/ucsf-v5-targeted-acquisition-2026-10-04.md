# Official UCSF-PDGM v5 targeted acquisition: live access result

After reading the October 4 dataset addendum and steering prompt, a fresh **official Aspera service check still found the imaging unavailable**. The selected patient remains the previously consulted development case `UCSF-PDGM-0004`; no new evaluation patient, official image, gradient file or client installer was downloaded. This replaces no earlier source evidence or experiment.

The precise declaration is `manifests/data_acquisition_ucsf_v5_official_0004.json`. It was written before live package browsing, fixes the development role, requests one baseline exam only, and caps initial imaging at 1 GiB. Its empty file inventory explicitly means that acquisition has not occurred. The prior structural mirror manifest remains unchanged and its official source equivalence remains unverified.

## Primary release and permitted scope

The [official TCIA collection](https://www.cancerimagingarchive.net/collection/ucsf-pdgm/) returned HTTP 200 with 342,016 bytes and SHA-256 `418408a5734ad96db3df297e6dd7a4eb1c64290b65fe5344df72ac0e6bf24c99`, identical to the earlier retained page. It still identifies **version 5, May 30, 2025**, public package **1065**, and **CC BY 4.0**. The declared full package is 142 GB; no full-package transfer was started. Dataset attribution remains Calabrese et al., dataset DOI [10.7937/tcia.bdgf-8v37](https://doi.org/10.7937/tcia.bdgf-8v37), descriptor DOI [10.1148/ryai.220058](https://doi.org/10.1148/ryai.220058), under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) and the linked [TCIA usage policy](https://www.cancerimagingarchive.net/data-usage-policies-and-restrictions/).

The selected metadata identity was checked against the current [official v5 CSV](https://www.cancerimagingarchive.net/wp-content/uploads/UCSF-PDGM-metadata_v5.csv): 58,149 bytes, SHA-256 `afc1c23a0eb0597b56e603741e72f2308b4860281187cd3808e05b7a0116fd78`, also identical to the retained file. Exactly one row has `UCSF-PDGM-004`, linked to `BraTS2021_00097` and the BraTS training cohorts. This is an already-used development patient, never an independent held-out case. The exact downloadable baseline exam path cannot be established while directory listing fails. No molecular, postoperative or unknown-timed clinical field is supplied to planning.

The v5 release description distinguishes native `DTI_eddy_noreg` images from processed structural space. It supplies revised orientation/spacing and each exam's eddy-rotated b-vectors. The collection's processing description states that FSL 6.0.2 eddy used outlier replacement and **TOPUP was off**. Thus a released eddy derivative could bypass unnecessary reinvention of that processing step, but does not prove complete susceptibility correction, valid alignment or anatomical coverage. Collection preprocessing statements are source claims, not patient-specific QC results.

## Actual live operations

The normal public-link OAuth flow succeeded using the collection-published context and current public client configuration. No personal login or paid service was used. Token creation returned HTTP **201**, which is success. Bearer tokens are absent from the recorded artifact; the temporary token file was removed.

| Operation | Actual response | Meaning |
|---|---|---|
| `GET /api/v5/packages/1065` | HTTP 200; title `UCSF-PDGM Version 5`; `files_on_server: no` | Current server storage state reports unavailable files |
| `POST /api/v5/packages/1065/files/received`, root path | HTTP 500, code `1202`; HSTS directory-browsing error | No case/exam file inventory can be verified |
| Documented `POST /api/v5/packages/1065/transfer_spec/download?transfer_type=connect&type=received` | HTTP 400, code `1228`; files unavailable on HSTS | Even the actual Aspera Connect path cannot obtain a transfer specification |

The final Connect error trace is `06929152-a4a6-40b4-af65-952d6e70031d`; the directory error trace is `37f91bec-ce9d-440c-a7ae-92f4afc29a79`. Sanitized complete responses, timestamps and response hashes are in `artifacts/ucsf-v5-targeted-access-2026-10-04/access-report.json`.

Two client-probe mistakes are retained separately from those source failures. An initial wrapper expected HTTP 200 for token creation and stopped on valid HTTP 201. An initial GET transfer-spec request returned HTTP 404/code 1002 because the documented method is POST. Neither is treated as evidence of absent source files. The subsequent correct public authentication and documented Connect POST establish the reported result.

No `ascp` or `aspera` executable was found on PATH; the normal system/user Connect application locations and user Aspera support directories were absent. Installing a client would not repair this observed server-side failure, so no installer was fetched. The package's own `total_bytes=885200` conflicts with its 142-GB collection description and 12,030-file count; it is not used as a storage forecast.

## Gradient and frame contract remains unfulfilled

The requested scope is that exam's structural sequences, matching annotation, v5 `DTI_eddy_noreg`, **its own rotated b-vectors**, and b-values demonstrably associated with the same 4D volume. The historical shared b-vector link is not a substitute. No case-level filenames, original arrays/affines, gradient frame, per-file checksums or gradient-count pairing could be inspected.

When official access returns, retain original bytes and source lengths/checksums, then check volume/vector/value counts, b0 handling, non-b0 vector norms, the source FSL convention/handedness, and the complete image-to-gradient frame. Native diffusion must be aligned independently to the structural/annotation space with failure and coverage reporting. Merely matching patient labels or broad collection metadata is insufficient. Supplied source annotations and engineering QC remain separate from expert review.

## Practical next step

The necessary provider action is to restore package 1065's files/listing or expose an official bounded `UCSF-PDGM-0004` baseline-exam download with the corresponding rotated vectors and b-values. The [TCIA support channel](https://www.cancerimagingarchive.net/support/) can use the package, trace IDs and exact errors above. No support message was sent.

In parallel, the separately assigned selected IDC ReMIND/UPenn structural work can proceed under its own identities and roles. It must not be fused with this UCSF patient's unavailable diffusion. This acquisition task adds no correction research, does not reopen BTC/final cohorts, and does not promote any patient functional evidence or mirror to official-v5 status.
