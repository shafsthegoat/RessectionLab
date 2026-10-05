# RHUH-GBM public clinical/outcome audit

The public clinical table supports a small **observational postoperative-outcome study**, with all 40 deficit labels and paired KPS scores available. It does not establish new surgery-caused deficits or effects of unperformed routes. No model was fitted and no patient images were downloaded in this audit. Existing BTC patient roles are unchanged.

## Verified source and access

The [TCIA collection](https://www.cancerimagingarchive.net/collection/rhuh-gbm/) lists clinical CSV and brain-extracted NIfTI/segmentations under **CC BY 4.0**, separately from controlled raw DICOM. Its version-1 table is dated June 9, 2023. The [actual public CSV](https://www.cancerimagingarchive.net/wp-content/uploads/clinical_data_TCIA_RHUH-GBM.csv) downloaded without credentials: **7,087 bytes**, SHA-256 `32d638906d34aaf8f66f5ec41c53c044216aed73bac22c776fb399bf2f741728`. Downloaded bytes, source pages, descriptive script and complete field inventory are frozen in [`artifacts/rhuh-outcome-audit-v1`](../artifacts/rhuh-outcome-audit-v1/).

TCIA requires collection attribution and its data DOI under the [data-usage policy](https://www.cancerimagingarchive.net/data-usage-policies-and-restrictions/); the [CC BY 4.0 terms](https://creativecommons.org/licenses/by/4.0/) require attribution, a license link and indication of changes. This audit acquired no controlled DICOM. Cite: Cepeda, S., García-García, S., Arrese, I., Herrero, F., Escudero, T., Zamora, T., & Sarabia, R. (2023), *The Río Hortega University Hospital Glioblastoma dataset: a comprehensive collection of preoperative, early postoperative and recurrence MRI scans (RHUH-GBM)* [Dataset], The Cancer Imaging Archive, [doi:10.7937/4545-c905](https://doi.org/10.7937/4545-c905).

## What the actual table contains

Comma-delimited UTF-8 text has **40 unique `Patient ID` values, `RHUH-0001`–`RHUH-0040`, and 23 columns**. There are zero empty or explicit NA-like cells among 920 cells. This does not fill structurally absent clinical fields. Original headers, including spelling and trailing spaces, are retained.

| Source field | Observed values/counts |
|---|---|
| `Postoperative Neurological Deficit` | `No` 26; `Transient` 6; `Minor Persistent` 6; `Major Persistent` 2 |
| `Preoperative KPS` | 60–90, observed increments of 10; median 80 |
| `Postoperative KPS` | 50–90, observed increments of 10; median 80 |
| Derived postoperative minus preoperative KPS | 5 decreased; 22 unchanged; 13 increased; range −30 to +20 |
| Previous treatment | 38 `no`; 2 `surgery + QT/RT` |
| Recorded IDH / resection category | 4 mutant, 36 wild type; 27 GTR, 13 NTR |

The 26/6/6/2 counts agree with both [arXiv v2, Table 1](https://arxiv.org/html/2305.00005v2) and the [published paper, Table 2](https://pmc.ncbi.nlm.nih.gov/articles/PMC10551826/). Other columns provide age/sex, days from earliest imaging to surgery, pathology, operative adjuncts, pre/post contrast-enhancing and T2/FLAIR volumes in cm³, EOR percent, adjuvant treatment, radiotherapy, PFS/OS days and one right-censoring flag. Earliest-imaging-to-surgery intervals are 1–33 days. These are not clinical outcome-assessment intervals.

## Label and integrity limits

The paper describes electronic-record extraction and early postoperative **MRI within 72 hours**. It does not specify the clinical deficit/KPS assessment time, transient/persistent duration threshold, minor/major severity rubric, affected neurological domain or baseline domain deficit. Preserve those fields as unknown. `Transient` is a recorded category, not a released recovery time series. `No` cannot be promoted to normal baseline function or no new injury. The observed KPS range is not the scale's formal allowable range.

KPS and categorical deficit are distinct: one `No` patient has a lower postoperative KPS; two `Minor Persistent` patients have higher KPS. Neither score change nor categorical persistence supplies a domain-specific new/worsened-deficit label.

Two integrity issues matter before eligibility filtering. RHUH-0017 is labeled NTR with EOR 94.6%; its rounded volumes also imply 94.5946%, conflicting with the stated >95% definition. RHUH-0023 displays 95.0%, but rounded volumes imply 95.0205%, a rounding-boundary ambiguity. Preserve both records and flags. Also, the publication table repeats approximately 35 cm³ for pre/post T2/FLAIR volume; the CSV yields means **73.14125/34.99975 cm³**, consistent with the arXiv narrative. Radiotherapy category counts in the CSV also differ from the summary table. Use the pinned row data, retaining these discrepancies.

## Smallest defensible next experiment

Before fitting, declare patient-level roles and one primary endpoint: **any recorded postoperative deficit category versus `No` (14/26)**. Keep the four original categories and KPS change as separate descriptive outcomes. Compare a training-only prevalence predictor and majority-label baseline with a small regularized preoperative model, then assess whether preoperative volume and separately declared postoperative injury features add information. A constant `No` predictor has 65% apparent accuracy but zero positive sensitivity; it is not a validated model.

Exclude postoperative KPS, residual volume/EOR, adjuvant treatment, survival and pathology with unverified presurgical availability from a model claiming preoperative prediction. Keep all visits of each patient together. Two major cases cannot support reliable stand-alone major-deficit prediction. Outcome definitions/timing, selection for extensive resection and treated follow-up limit interpretation. All 40 labels were audited for QA; no future partition is claimed untouched. Postoperative injury-outcome association and preoperative prediction require separate tasks; neither identifies counterfactual route risk or yet earns a planner reward.
