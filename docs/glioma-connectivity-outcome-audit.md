# Glioma connectivity and cognitive outcome: public cohort audit

Audited 5 October 2026. **Useful processed outcome data exist, but exact paper reproduction and a strictly prospective clinical-risk claim are not currently supported.** No model was fitted and no patient roles were assigned. The release contains 63 patients, complete pre/post connectivity matrices, seven cognitive-domain scores per visit, and author-supplied impairment labels. A baseline-label discrepancy, incomplete test-level outcome reconstruction, and pooled longitudinal preprocessing require explicit treatment before an experiment.

## Sources and acquisition

Smolders et al., *Presurgical structural connectivity predicts postsurgical cognitive impairment in glioma patients*, Brain Communications 2025, [PMID 41140809 / DOI fcaf346](https://doi.org/10.1093/braincomms/fcaf346). The complete primary article and supplement were read through the [Europe PMC full-text service](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC12550501/fullTextXML), not only the abstract. The public [collection version 1](https://doi.org/10.6084/m9.figshare.c.7578326.v1) has exactly two dataset items, both **CC BY 4.0**:

| Release | Verified files | Bytes | Actual contents |
|---|---:|---:|---|
| [Structural matrices v1](https://doi.org/10.6084/m9.figshare.28000559.v1) | 126 | 13,346,928 | IDs 1–63, one `_pre.npy` and one `_post.npy` each |
| [Baseline and cognitive scores v1](https://doi.org/10.6084/m9.figshare.28001120.v1) | 1 | 7,844 | `baseline_variables_cognitive_scores.csv`, 63 rows × 31 columns |

All 127 data files were downloaded under those terms; advertised byte counts and MD5 values matched, and local SHA-256 values were recorded. The matrices remain in the ignored `outputs/datasets/glioma-connectivity-7578326-v1/` directory. The reproducible [audit and source receipt](../artifacts/glioma-connectivity-outcome-audit-v1/source-receipt.json) pin metadata, data, article, supplement and inspected author-code versions. Attribution remains to the original authors; our audit does not alter their released values.

## Patient and visit identity

The retrospective source cohort is from Elisabeth-TweeSteden Hospital, Tilburg, Netherlands, with surgery between July 2016 and July 2023. Eligibility required presurgical and approximately three-month MRI and cognitive assessments; prior intracranial surgery, cranial radiotherapy/chemotherapy, and neurological/psychiatric history were exclusions. Actual grades in the release are II:34, III:4, IV:25. These WHO-grade counts are not a reconstruction of the paper's molecular LGG/HGG grouping: individual IDH data are absent.

Use the namespace `figshare:7578326:v1:<ID>`; IDs 1–63 are unique only within this release. There is no cross-cohort identity key, exact acquisition/surgery/assessment date, or per-patient day offset. No overlap with BTC, RHUH, or another cohort has been established or excluded. Both visits, all matrices, and all derived features from a patient must have one role.

The paper places T0 acquisition within the week before surgery and T3 at about three months; cognitive testing and MRI occurred on the same day at each visit. These are study-level timing statements, not individual timestamps. Radiotherapy occurred between visits for 37/63 and chemotherapy for 35/63. T3 is an outcome under observed surgery, recovery and subsequent care, not an isolated surgical effect.

## What the files support

The 126 arrays load without pickle as **115×115 float64** matrices. All are finite, nonnegative, exactly symmetric, zero-diagonal and nonempty; each visit covers all 63 CSV IDs. They represent SLANT-region connectivity weighted with SIFT2; weights are neither injury volumes nor deficit probabilities. Whole-brain row sums can be computed directly. The original [analysis repository](https://github.com/larssmolders/predicting-postop-impairment/tree/1712a43b3a5af68b6d125330e2be9ece6c826c20) supplies region-index/name mappings and modeling code, with limitations below.

The CSV provides:

- Baseline context: `age`, `education` (Dutch Verhage scale), `sex`, `left_hemi`, `right_hemi`, five `loc_` lobe indicators, and `tumor_volume`. The volume column has no unit label; its magnitude and the paper's 1-mm voxel-volume calculation suggest mm³, but conversion must remain explicitly inferred until confirmed.
- Seven Z-score fields at both `T0_` and `T3_`: `verbmem_Z`, `vismem_Z`, `psymotor_Z`, `reacttime_Z`, `complatt_Z`, `cogflex_Z`, `procspeed_Z`—verbal memory, visual memory, psychomotor speed, reaction time, complex attention, cognitive flexibility and processing speed. The paper describes healthy normative standardization for age, education and sex; lower scores indicate worse performance. Exact norm tables and all raw test measurements are absent.
- Author-supplied `T0 2 imp` and `T3 2 imp`, plus postoperative context `tumor_grade`, `radiotherapy`, `chemotherapy`. Exclude the latter three from a preoperative predictor. Individual information-availability timestamps are absent.

All 63×31 entries are present. A case-insensitive check for empty/NA/NaN/null/none/N/A/missing tokens found none, and all numeric columns are finite. This does not verify measurement validity or undocumented sentinel conventions. Reaction-time Z scores reach +10.15 and other domains extend below −10; retain these as released values and do not silently winsorize or reinterpret them as missing.

## Outcome discrepancy: preserve, do not repair by assumption

The paper defines impairment using at least two of **nine test variables** below Z=−1.5. Its supplement lists nine test outputs; the public CSV contains seven domain-score columns per visit. Applying that threshold to only those seven columns is a diagnostic check, not a valid replacement endpoint.

| Check | T0 | T3 |
|---|---:|---:|
| Paper Table 1 impaired | 30/63 | 34/63 |
| Public supplied label positive | **42/63** | 34/63 |
| At least two of seven public domain scores below −1.5 | 29/63 | 24/63 |
| Rows disagreeing with supplied label | 13 | 10 |

The T0 paper/public count discrepancy remains unresolved. The 13/10 differences are retained by ID in [audit.json](../artifacts/glioma-connectivity-outcome-audit-v1/audit.json); they could reflect unavailable test variables or another provenance issue, and do not establish which labels are correct. The supplement's practice-effect statement does not supply enough information to reconstruct them.

Supplied binary transitions are 0→0:16, 0→1:5, 1→0:13, 1→1:29. These are transitions between the released composite labels, **not validated new-deficit, recovery, or reliable-change endpoints**. The paper separately reports 21 patients with decline; no corresponding decline field or operational rule is supplied. Arithmetic T3−T0 domain differences are available, but clinically meaningful decline would require a prespecified reliable-change definition.

## Published AUC scope and reproduction limits

The reported AUCs—baseline .69, location .62, DMN/FPN .73, baseline+network .75/.76—are internal repeated stratified 75/25 random splits of the same 63 patients, averaged across 100 iterations, using depth-3 random forests. They are not external validation, calibrated individual probabilities, or percentages of surgical accuracy. Patient recurrence across splits does not increase the independent sample size. The public splitting code floors class counts, producing 46 training/17 validation patients for the released T3 class balance; it does not pin the random seeds. The paper describes 1,000 permutations and FDR correction; public permutation code redraws splits rather than preserving a saved fold inventory.

Additional limitations prevent an exact reproduction claim:

1. **Longitudinal preprocessing:** white-matter response functions were averaged across all patient scanning sessions before connectome estimation. T0 image acquisition therefore does not imply preprocessing isolated from held-out patients or T3 scans. The released matrices cannot remove this dependence without raw imaging. Any experiment must be described as validation on fixed, cohort-processed features, not a prospective end-to-end imaging pipeline.
2. **Predictor inventory:** the paper reports 33 DMN and 20 FPN variables with subcortical additions. The pinned public code contains 23 DMN/10 FPN indices plus only two thalamic indices (25/12 predictors in those combinations). Do not invent the missing selection or claim exact feature replication. The paper's own count descriptions also do not reconcile transparently with its listed cortical/subcortical regions.
3. **Location comparator:** hemisphere/lobe indicators and volume are public. The granular per-ROI tumour-overlap predictors used for the reported location-only AUC are not supplied, so a coarse location/volume baseline is a new, explicitly weaker comparator.
4. **Runnable provenance:** author code expects local spreadsheets, directory paths and column names that differ from the public CSV; its default target is T0, requiring deliberate modification for T0→T3. No frozen splits or fitted model are published. No reuse license was found in the inspected code repository; no author code is incorporated into our executable pipeline.

## Smallest useful next experiment and stop conditions

**Do not fit yet.** First freeze patient roles, endpoint and the handling of these discrepancies. The preferred narrow, independently reproducible experiment is prespecified T3 domain-score prediction with its same-domain T0 score as the first baseline, then age/sex/education, coarse location+volume, and a small fixed connectivity summary extension. Choose the domain prospectively or report all seven without selecting the most favorable result; keep standardization, imputation if ever needed and regularization selection inside training folds. Use patient-level out-of-fold errors, paired baseline differences and uncertainty; describe the fixed cohort-processed feature limitation.

A separate prediction of the supplied `T3 2 imp` is technically possible (34 positive/29 negative), but must be labeled **author-supplied composite impairment**, exclude the disputed T0 binary from the primary baseline or resolve it first, and retain raw T0 domains as an explicitly different baseline. A prevalence predictor and coarse location/volume model are necessary comparators. ROC-AUC, PR-AUC, class sensitivities and Brier score would be appropriate outputs; tiny-sample calibration or clinical utility would remain unproven. This is a proposal, not an executed protocol.

Stop exact paper reproduction if endpoint/predictor mapping cannot be resolved. Stop clinical-deficit or reliable-decline claims unless suitable definitions become available. No released MRI, tumour masks, spatial streamline paths, resection cavities, complete-tool trajectories, vascular injury images, or counterfactual surgeries are present. Consequently these data **cannot validate a hypothetical surgical route's neurological harm**, localize a simulated removal into this patient's network, or train the planner's reward directly. At most they support bounded observational cognitive-outcome research.
