# BTC longitudinal cognitive/connectivity metadata audit

Audit: 2026-10-05. Metadata supports **five candidate longitudinal TRAIN pairs**, not five verified complete outcome records. No cognitive score, individual assessment date, image array or connectivity payload was opened in this audit. No cohort role changes are proposed.

## Access and source versions

Read public release tags/recursive file lists, READMEs, dataset descriptions, change logs, selected Methods/Data Records prose in the [creator paper](https://doi.org/10.1038/s41597-022-01806-4), the OSF dictionary, and **only the first line** of three participant tables and one behavioral CSV. Header reads used bounded `readline(16384)` followed by close; no participant rows were parsed, logged or saved. Article outcome tables/results and OSF analysis-code/data bodies were not inspected. Existing local metadata supplied the already-declared cohort roles; this audit did not reread its outcome rows.

| Release | Exact Git commit | Source license statement |
|---|---|---|
| [BTC_preop ds001226 v5.0.1](https://openneuro.org/datasets/ds001226/versions/5.0.1) | `359d372c5e972a161966312128adb365870df949` | Root description: CC0 |
| [BTC_postop ds002080 v4.0.1](https://openneuro.org/datasets/ds002080/versions/4.0.1) | `4e7326e5c0bd01464cb8573f2bb277500c3a5881` | Root description: CC0 |

Both TVB derivative descriptions instead state PDDL v1.0; their identical file is CP1252 and malformed JSON. Preserve that discrepancy rather than silently applying the root license to every derivative. The [OSF behavioral supplement](https://osf.io/5kfw3/) declares CC-BY 4.0 through its license metadata.

## Actual cognitive fields

Both pinned root `participants.tsv` headers use `participant_id` and contain these same **28** CANTAB columns. Spelling is source-exact; commas separate individual fields.

| Task | Released columns |
|---|---|
| MOT | `MOT_latency_mean`, `MOT_latency_md`, `MOT_error_mean` |
| RTI | `RTI_simple_accuracy`, `RTI_simpleRT_mean`, `RTI_simpleRT_md`, `RTI_simpleRT_sd`, `RTI_simpleMT_mean`, `RTI_simpleMT_md`, `RTI_simpleMT_sd`, `RTI_five_accuracy`, `RTI_fiveMT`, `RTI_fiveRT`, `RTI_fiveRT_sd`, `RTI_fiveMT_mean`, `RTI_fiveMT_md`, `RTI_fiveMT_sd` |
| RVP | `RVP_A`, `RVP_probhit`, `RVP_falsealarms`, `RVP_latancy_mean`, `RVP_latency_md` |
| SOC | `SOC_prob_minmoves`, `SOC_meanmoves2`, `SOC_meanmoves3`, `SOC_meanmoves4`, `SOC_meanmoves5` |
| SSP | `SSP_spanlength` |

Sources: pinned [preop header source](https://raw.githubusercontent.com/OpenNeuroDatasets/ds001226/359d372c5e972a161966312128adb365870df949/participants.tsv) and [postop header source](https://raw.githubusercontent.com/OpenNeuroDatasets/ds002080/4e7326e5c0bd01464cb8573f2bb277500c3a5881/participants.tsv). `RVP_latancy_mean` is a source typo; do not silently rename it. Baseline demographic headers also include leading spaces in ` sex` and ` age`. No `participants.json` or phenotype dictionary was present in either complete release tree.

The creator-linked [OSF Legend.csv v1](https://osf.io/download/js8q4/?version=1) and [Data.csv v2 header](https://osf.io/download/z3sk7/?version=2) provide a clearer longitudinal schema: `subID`, `date_t1`, `date_t2`, plus each of `MOT_latency_mean`, `RTI_fiveRT`, `RVP_A`, `SOC_prob_minmoves`, `SSP_spanlength` suffixed `_t1` and `_t2`. The dictionary identifies preoperative versus postoperative assessment dates and respectively describes motor-screening response latency, five-choice reaction time, attention/inhibition sensitivity, minimum-move planning performance and visuospatial memory span. It also includes paired emotional questionnaires and lesion-volume fields. **Units, score ranges, direction, normalization, missing-value encoding and reliable-change thresholds remain unverified.** Do not infer milliseconds, impairment or binary neurological deficits from names alone. Calendar values/formats and event-to-assessment delays remain unopened; a date is not a UTC availability timestamp.

The postoperative `derivatives/TVB/participants.tsv` header is byte-identical to the baseline root header, including baseline-style lifestyle/height/weight columns. This does not prove duplicated values, but blocks treating that table as an authenticated follow-up outcome source.

## Linkage, roles and missingness

The [postop creator README](https://raw.githubusercontent.com/OpenNeuroDatasets/ds002080/4e7326e5c0bd01464cb8573f2bb277500c3a5881/README) explicitly describes follow-up of the baseline cohort: seven glioma participants from eleven, with aggregate reasons of one dropout, one without resection and two beyond the study period. It does not assign those reasons to particular TRAIN IDs. The paper describes baseline acquisition the day before surgery and follow-up at approximately six months; those are cohort-level timing statements, not audited patient dates.

| Existing TRAIN patient | Same-ID preop/postop folders; raw T1/DWI/rest-fMRI and five TVB files listed | Audit disposition |
|---|---|---|
| PAT05, PAT16, PAT20, PAT25, PAT28 | Present at both visits | Five candidate glioma TRAIN pairs under existing baseline eligibility |
| PAT22 | Preop present; postop folder absent | Keep in TRAIN denominator; paired outcome availability unknown |

The five TVB paths checked per visit are `FC.mat`, `SC.zip`, `SCthrAn.mat`, `ROIts.dat`, `HRF.csv`. File presence is not score completeness, image quality or usable matrix validation. The creator documentation establishes related cohorts and uses `ses-preop`/`ses-postop`; the audited material does **not** explicitly certify an individual cross-dataset ID map. OSF defines `subID` only as subject identification, with no audited mapping to OpenNeuro `participant_id`. Therefore no outcome rows may be ingested yet. Obtain a creator-supported ID crosswalk or explicit stable-ID statement before joining; do not identify people by score/date similarity.

Roles remain TRAIN **05/16/20/22/25/28**, SELECT **26/27**, unopened **29/31**. Every linked visit/derivative inherits its existing patient group and role. Unknown linkage is quarantined, never counted as an independent subject. No protected outcome/image/connectivity content was inspected. Endpoint-specific paired counts, actual interval distributions and informative dropout cannot be measured from this header-only audit.

## Connectivity and leakage limits

Raw diffusion and resting-state image paths and processed matrices/time series exist. Connectivity itself is estimated. The paper describes 68 cortical Desikan–Killiany regions; `SC.zip` packages weights, tract lengths and regional geometry, while `SCthrAn.mat` is thresholded/normalized SC. Its Methods describe Fisher-z FC, but Data Records call `FC.mat` Pearson coefficients: payload representation needs verification, not assumption.

The documented diffusion pipeline pools response functions/intensity normalization across subjects. Released SC is therefore **not automatically a split-clean feature set** for our existing roles. Audit contributing subjects and normalization provenance, or recompute within TRAIN-defined preprocessing before predictive evaluation. Postoperative connectivity, residual lesion volume and cognitive scores are follow-up observations, unavailable to a preoperative policy. Cognitive change does not identify route-caused injury without addressing treatment, recovery, practice effects and confounding. No route-specific risk calibration is supported here. [Primary methods and data records](https://doi.org/10.1038/s41597-022-01806-4)

## Compact fixity record

SHA-256 values are of exactly the audited bytes, unless explicitly marked provider-only:

- Preop root header, 765 bytes: `f7b6b3e6b4fbcf57a398fc0aa1a99e3130d84f088936aca28fcdafbea926a6a5`; postop root header, 803 bytes: `2f32f482104349b34cdbf369269abfbc18447888e9741d1910556ad02b91803c`.
- OSF Legend v1, 2232 bytes, provider hash independently matched: `4f35873d0fb8eb49a3b7ef2c2ffc5f23491db58f258eb55f7efaa043557d0fbb`; CP1252 decoding required. Data v2 header: `53078853633a5475b76ebd5ca23d8588c3f40d437aa0439af19401562fb005b5`. Entire Data v2 is **unread**; provider-reported SHA only: `0bd830758c13832b5c86bc9b656e2c6b9908cdd8da600fb2588948464317cd9a`.
- Complete Git tree API responses: preop `593f6a60454a40ba97ba100c29ff90e583c8020ef676082329f2168ccf47165d`, postop `1fcdc1624c738549d742540c91572b4e28ffc4002f7c1c7dbfb326150767beab`; both reported `truncated=false`.
- Pinned postop README: `e6a4d7e184a02e7cca59817d5b7a2520f33809622ff7a89cfd07d9d5e07444c6`. Shared TVB description: `69f1e94f142110b5e1c5525319c3aca9b11bcf9c872aa9bf6b6ea9779dd4508c`.

Next bounded dependency is linkage and score-definition verification, followed by a separately declared TRAIN-only row audit. This document authorizes no outcome ingestion, training, bulk download or held-out consultation.
