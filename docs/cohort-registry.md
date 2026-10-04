# Cohort identity and eligibility registry

`manifests/cohort_registry.json` records **five development representations and
no final evaluation patients**. Four have verified primary-source identity;
the UCSF structural mirror remains attributed to an unverified source release.
The registry also records collection-level future roles. A release
total is not an eligible, acquired, or evaluated patient count. All eligible
collection counts remain null until individual screening is performed.

## Verified source facts

| Source | Checked release and intended role | Current status |
| --- | --- | --- |
| [UCSF-PDGM](https://www.cancerimagingarchive.net/collection/ucsf-pdgm/) | v5, May 30, 2025; primary development source | Official imaging transfer unavailable; one structural mirror used for development |
| [BTC_preop](https://openneuro.org/datasets/ds001226/versions/5.0.1) | v5.0.1, Git `359d372c5e972a161966312128adb365870df949`; development fallback | Creator-linked `sub-PAT28` imaging and gradients; predeclared structural cases `sub-PAT05`, `sub-PAT16` and `sub-PAT20` |
| [UPENN-GBM](https://www.cancerimagingarchive.net/collection/upenn-gbm/) | v2, October 24, 2022; reserved external final candidate | Collection description only; no patient data or availability tables opened or downloaded |
| [UTSW-Glioma](https://www.cancerimagingarchive.net/collection/utsw-glioma/) | v1, March 18, 2026; future structural validation | Collection description only; no patient records instantiated |

The UCSF release reports 495 people and 501 exams. Its official clinical CSV
maps `UCSF-PDGM-004` to **BraTS2021_00097, Training**. The mirror directory uses
the padded identifier `UCSF-PDGM-0004`; source-byte equivalence remains
unverified. This known correspondence conservatively blocks independent
evaluation of overlapping BraTS derivatives or population-training records.
The six repeated-exam relationships published by TCIA are also retained as
aliases; none of those additional images was acquired for this registry.

The pinned [BTC description](https://github.com/OpenNeuroDatasets/ds001226/blob/359d372c5e972a161966312128adb365870df949/dataset_description.json)
identifies `BTC_preop`, its DOI, creators, and CC0 license. Its participant table
and the acquired T1 bytes support the selected subject's source identity.
Related BTC postoperative data cannot be treated as independent new patients.
No postoperative correspondence was guessed: a future import must supply its
source-supported linkage or remain ineligible for independence claims.

UPenn's published 630 patients are not 630 diffusion-eligible cases. Its raw
DICOM and processed NIfTI require a coordinate-frame audit. Keep the collection
untouched by global development until the method and adaptation procedure are
frozen. UTSW is a structural test candidate: its
[descriptor](https://www.nature.com/articles/s41597-026-07274-4) distinguishes 362
manually refined masks from the wider 625-case release. That does not establish
an external diffusion planning cohort.

## Run the audit

From the repository root:

```sh
.venv/bin/python -m resectionlab.cohort
.venv/bin/python -m pytest tests/test_cohort.py -q
```

The audit performs no downloads. It hashes local evidence and checks specified
CSV/TSV rows or release assertions. Missing source evidence, changed bytes, or
an identity asserted against another patient's evidence cannot establish
independence. A fresh clone without downloaded evidence reports that evidence
as unavailable; it does not trust a cached success flag.

The current local audit passed all sixteen evidence checks. It reports five
registered development groups, **four with verified primary-source image
identity**, and zero final groups. The UCSF mirror remains an explicit
limitation. These are bookkeeping results, not anatomical acceptance or a
between-patient statistical result. PAT05 was registered as development before
opening its images, using the selection in `manifests/btc_pat05_selection.json`.

PAT16 and PAT20 were permanently assigned to development in the committed
`manifests/development_acquisition_queue.json` before image access. Their
acquired-case records were added only after independent source/geometry checks
passed. Each binds the participant row, T1, source annotation, queue hash and
independent review receipt. Both retain full-head coverage without a reviewed
brain mask, blocked automatic cortical access, missing acquired diffusion and
unknown patient function. Their clinical context remains excluded. Neither
they nor related visits or derivatives can become outer-final evaluation.
The original records, UCSF/BraTS aliases and final-source reservations are
unchanged. Five development representations do not satisfy the five-UCSF
milestone, establish five tract-aware cases or imply anatomical approval.

Each acquired record has a namespaced identity, visit, optional derivative
parent, outer split, declared uses, and evidence references. Repeated visits,
explicit aliases, and derivative chains form one patient group. The audit
rejects development/final overlap, a final patient's presence in population
pretraining, global development mislabeled as final evaluation, unresolved
derivative parents, and contradictory or cyclic lineage. A reported but
unverified alias is grouped conservatively and flagged.

Per-case optimization, checkpoint selection, and final scoring may belong to
the same outer-final patient. That is the intended frozen adaptation procedure;
it does not move the patient into population pretraining. This utility checks
patient identity and declared use. Separate experiment checks must enforce
world/seed partitioning, frozen objectives, checkpoint rules, and hidden-world
isolation. Upstream model training subjects must be entered in the registry;
an undisclosed training population cannot be certified by this audit.

## Information availability

Inputs retain provenance references, operative phase, availability basis,
measurement time, availability time, and whether the primary optimizer used
them. Source-declared preoperative images can support an imaging-only research
replay when calendar dates were removed. Supplied labels require the
annotation-assisted track. This exception never establishes availability of
molecular or clinical results.

Molecular/clinical context needs a verified availability time at or before the
planning cutoff. Unknown or naive timestamps, later results, and postoperative
outcomes are excluded. Retrospective tumor diagnosis may establish glioma
cohort membership while remaining unavailable to the preoperative optimizer.
An unavailable field marked as used produces a registry error. No context
values or clinical probabilities are generated by this utility.

Twenty-six tests passed, including alias and derivative leakage, unknown
identity, source corruption, evidence bound to the wrong patient, legitimate
within-patient adaptation, future molecular results, and preservation of null
eligible counts. The two new source-bound record checks also verify frozen
development roles, independent review evidence, annotation identity and
excluded unknown-timed context. The registry is intentionally incomplete; completing external
evaluation requires newly verified individual records and a frozen procedure.
