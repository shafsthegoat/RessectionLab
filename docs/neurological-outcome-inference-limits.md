# Neurological outcome inference: limits and first falsification

**Current recommendation:** test a small observational outcome association outside the planner. No audited cohort currently establishes neurological risk under an unperformed route. This review proposes analyses; it fits no model, reads no patient images or protected BTC outcomes, and changes no patient roles.

## Three different questions

Let `X` contain information available before planning, `I` actual postoperative imaging changes, `Y_t` a defined outcome at time `t`, and `S` cohort inclusion.

| Question | Estimand | Supported interpretation |
|---|---|---|
| Preoperative prognosis | `P(Y_t | X, S=1)` under observed care | Association in the selected cohort and its treatment setting; not the effect of choosing a route. |
| Postoperative injury association | `P(Y_t | X, I, S=1)` | Relationship to observed postoperative changes. If outcome timing relative to imaging is unknown, this is not established prospective prediction. |
| Hypothetical route comparison | `E[Y_t(r) − Y_t(r′) | X=x]` | A causal contrast requiring defined interventions and defensible identification/transport assumptions. A predictor supplied a simulated lesion does not estimate this automatically. |

Treatment-aware prediction requires an explicit estimand ([van Geloven et al.](https://pubmed.ncbi.nlm.nih.gov/32445007/)). One surgery per patient does not itself make causal inference impossible. Here, however, the audited sources do not establish recorded comparable route choices, adequate treatment overlap or control of route–outcome confounding. Consistency, exchangeability and positivity cannot be supplied by a simulator or virtual-lesion augmentation ([Hernán and Robins, chapters 3/19](https://content.sph.harvard.edu/wwwhsph/sites/1268/2024/04/hernanrobins_WhatIf_26apr24.pdf)). Even identified conditional average effects would not reveal an individual's two unrealized outcomes.

For RHUH, replace `Y_t` with **the recorded postoperative category at an unspecified assessment time**. The [verified CSV audit](rhuh-outcome-audit.md) finds 26 `No`, six `Transient`, six `Minor Persistent`, two `Major Persistent`, but no domain, baseline focal-deficit assessment, clinical assessment date or persistence/severity definition. MRI within 72 hours does not date the clinical label. Neither category nor KPS change establishes a new surgery-caused deficit. The [source paper](https://arxiv.org/html/2305.00005v2) also selects extensive resections with treatment and recurrence imaging available, limiting transport to other operations/patients.

## Failure modes to exclude first

- Baseline function, tumor location/biology, surgeon decisions and monitoring can affect both actual injury and outcome. Conditioning on resection success, follow-up availability or postoperative variables can introduce selection bias; indiscriminately adding covariates is not causal adjustment. Six-month cognition additionally reflects recovery and subsequent care.
- Postoperative KPS, EOR/residual, follow-up connectivity, survival and molecular results unavailable at planning cannot enter `X`. Keep postoperative association tasks separate. All visits/derivatives stay with one patient; audit cross-cohort reuse. Fit normalization, feature selection, harmonization and any calibration within training data. An independently fixed atlas is different from an atlas/template fitted using test patients.
- Imaging change is not automatically injury, and injury is not automatically deficit. The [matched DWI study](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0101805) found abnormalities in both cases and controls; its outcome-selected sampling cannot supply population absolute risk. Cavity, ischemia and retained tissue need separate validated measurements; removed tumor volume is not measured non-target injury.
- Challenge charter §37: transient deficits have unspecified severity. A total ordering against “minor persistent” is an assumption, not a released clinical definition. Preserve raw categories and course; do not assign equal damage increments, infer NANO scores, or silently repair uncertain cognitive labels.

## Smallest useful evaluation

Propose one endpoint before fitting: **any recorded postoperative deficit versus `No` (14/26)**. This coarsening deliberately answers a narrower question; it does not validate major-deficit or persistence prediction. Retain original categories descriptively. Always predicting `No` gives 65% accuracy and zero positive sensitivity. Major-class recall changes by 50 percentage points per patient: resampling, regularization and synthetic oversampling do not turn two events into a severe-risk validation cohort. Sample adequacy depends on candidate parameters and outcome frequency, not simply a successful fit ([Riley et al.](https://www.bmj.com/content/368/bmj.m441)).

Freeze a small candidate set: training-fold prevalence; preoperative KPS alone; then KPS plus one prespecified preoperative volume. Use fixed regularization and patient-level out-of-fold evaluation without searching splits or penalties. Treat this as exploratory internal evaluation: all RHUH labels have already been inspected for QA, and cross-validation is not external validation. A later postoperative experiment can compare a justified volume measure with one additional location/injury feature on the same eligible patients; it must not inherit the preoperative claim.

Keep per-patient predictions and denominators. Report paired Brier-score change against the prevalence baseline, confusion counts, sensitivity/specificity and balanced accuracy at a declared threshold; discrimination is secondary. No class metric when its denominator is absent. Do not present fold standard deviations as independent patient uncertainty; any resampling interval must repeat the fitted pipeline at patient level and disclose its small-sample limits. Calibration is an evaluation question, not a guaranteed deliverable: no severe-class calibration claim, ten-bin reliability theater or post-hoc calibrator fitted on evaluation labels. Explicitly report non-estimability.

The cheapest falsifications are (1) permuting forbidden postoperative fields leaves preoperative features, permitted-input hashes, seeds and predictions unchanged, (2) patient-level label permutation reruns the entire declared pipeline, including any selection, and (3) removing each positive patient in turn exposes influential results. These are diagnostics, not extra independent patients. An apparent gain that depends on one case or vanishes after leakage correction ends that claim; it does not trigger a larger model sweep.

## Gate before search/RL can optimize it

Require a frozen endpoint/population/time contract, provenance and patient roles; reproducible incremental evidence beyond simple baselines; and a common observed-versus-simulated feature definition with checked units, evidence coverage and out-of-distribution rejection. Separately validate the proposed mapping from route/tool effects to the injury representation. Good observational AUC or calibration does not validate that mapping or confer causal route risk.

An initial integration, if earned, must be explicitly an **experimental association-based surrogate**, with unknowns/abstention rather than invented safety probabilities. Freeze its preprocessing, weights, rewards and uncertainty assumptions during optimization; give search and RL identical access. Retain geometric legality independently. Evaluate on evidence independent of the optimized score and test whether optimization exploits missing evidence or unsupported damage patterns. Lowering the learned score alone demonstrates optimization of that score, not reduced neurological harm. These gates do not delay the separately authorized geometric learning experiments.
