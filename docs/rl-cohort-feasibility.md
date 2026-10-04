# Scan-conditioned RL cohort feasibility

Checked October 4, 2026, using local provenance records and primary-source metadata only. **Expand BTC_preop first**, as a small annotation-assisted development pilot. Its verified source and modest storage cost make it practical; it does not yet provide reviewed working anatomy or population-level clinical validation. No new images were downloaded or opened during this audit.

## Fixed prospective split

The [creator release](https://openneuro.org/datasets/ds001226/versions/5.0.1), Git `359d372c5e972a161966312128adb365870df949`, declares **CC0**. Its [participant table](https://github.com/OpenNeuroDatasets/ds001226/blob/359d372c5e972a161966312128adb365870df949/participants.tsv), SHA-256 `2ae814b3e90e5c45ea95342a3658e58ba21d3f2d7bf1caa83ef8f1c74986e71d`, contains ten records matching our existing glioma/astrocytoma inclusion rule. The README's broader eleven-glioma count includes an ependymoma record outside this diffuse-glioma selection.

Keep previously consulted PAT05, PAT16, PAT20 and PAT28 in training/method development. Sort the six remaining matching IDs lexicographically and assign consecutive pairs before image access:

| New subject | Fixed development role | T1 + annotation + T1 JSON bytes |
| --- | --- | ---: |
| PAT22 | Population training | 18,036,963 |
| PAT25 | Population training | 18,327,917 |
| PAT26 | Checkpoint selection | 18,166,177 |
| PAT27 | Checkpoint selection | 17,844,653 |
| PAT29 | Unopened later frozen transfer pilot | 18,540,023 |
| PAT31 | Unopened later frozen transfer pilot | 17,478,092 |

These are source-supported acquisition candidates; geometry, quality and anatomy eligibility remain unchecked. No images, route scores, pathology values or outcomes determined split placement. Keep failures in their assigned role; no convenient replacements. Selection patients cannot contribute population-training gradients. Later transfer patients remain unopened until the complete pilot procedure is frozen; they are still development patients, not the reserved external final cohort.

The committed declaration is `manifests/experiments/btc-spatial-development-cohort-v1.json`, commit `9881c17`, SHA-256 `962d964e1d71427f3625cdbebc0f7e4759e5d2345d8f95cb211ed45810ed2985`. It binds exact Git-annex pointer bytes, expected MD5/length, official S3 version IDs verified by HEAD, metadata hashes, roles and information cutoffs. Each distinct `BTC:participant_id` groups every visit, derivative and augmentation. No repeated visit is selected. The related postoperative ds002080 cohort is unopened; unknown correspondence cannot establish independence.

## Input strata and gates

1. **Structural, annotation-assisted:** native T1, affine/mm, source fractional lesion annotation and explicit 0.5 threshold scenario. Encode image intensity/spatial features with training-only normalization; preserve channel-missingness indicators. BTC supplies neither four structural contrasts nor separable enhancing/core/FLAIR labels. Conditioning on its supplied lesion annotation is not an end-to-end segmentation result.
2. **Diffusion extension, later:** BTC exposes AP multi-shell diffusion, reverse PA references and gradient/acquisition metadata. All six candidates have pinned AP/PA pointers and accompanying metadata, but none was fetched. Only PAT28 currently has acquired diffusion; preprocessing, gradient-frame/registration QC and tract validation remain separate. Resting-state fMRI is not task-localized motor/language evidence.
3. **Brain support:** all four acquired BTC cases are full-head. Their frozen extraction proposals remain estimated, review-required and view-only; working brain masks are absent. Additional acquisition cannot enable cortical access. Keep full-head nonzero-intensity shortcuts blocked. Molecular, pathology, cognitive and other context fields with unknown preoperative availability stay outside actor, critic and reward inputs.

Six new structural pairs would total 108.39 MB before metadata. The permitted first four need **72,375,710 new bytes**, or **72,390,601 bytes** including shared metadata once. Six AP/PA pairs would add 287.18 MB, deferred. About 744 GB was free locally; reserve a bounded 5 GB workspace initially and measure preprocessing/cache expansion.

## Other sources and overlap

[UTSW-Glioma](https://www.cancerimagingarchive.net/collection/utsw-glioma/) is the next structural diversity candidate: four contrasts, 625 patients, 22.9 GB, CC BY 4.0. Its [descriptor](https://www.nature.com/articles/s41597-026-07274-4) distinguishes 362 manually refined tumor masks from automated outputs and describes skull-stripped variants; individual usable files and transport remain unverified. Preserve its future structural-validation role until a separate split is declared.

[EGD](https://xnat.health-ri.nl/REST/projects/egd) offers 774 structural cases but currently requires a provider-issued account and data-use agreement; license acceptance and manual-versus-automatic label provenance need review. It is not the immediate anonymous-download option. UCSF official access remains blocked as documented separately; its working mirror overlaps BraTS2021_00097 training. UPenn remains untouched and reserved. Audit all pretrained segmentation/encoder training populations: unknown overlap cannot be reported as proven absent. Supplied labels support annotation-assisted planning, not independent segmentation accuracy.

## Bounded executable acquisition

The tested extension permits only `--spatial-subject sub-PAT22`, `sub-PAT25`, `sub-PAT26`, or `sub-PAT27`; PAT29/PAT31 are rejected through every manifest/CLI route. After root's implementation checkpoint and release, run `.venv/bin/python scripts/acquire_btc_case.py --spatial-subject sub-PAT22`, then the other three explicit IDs. `--dry-run` is offline. Existing modes remain unchanged; pending image SHA-256 is established only after source MD5/size verification. Preparation needs the same cohort binding before these new IDs can enter bundles.

Initial owner checks passed 106/106 in 2.20 seconds. Independent review then reproduced **three failures** in established-SHA reacquisition: wrong/missing S3 version and changed final URL were accepted. It also identified automatic redirect following before response validation. These negative results are retained in `artifacts/btc-spatial-acquisition-review-v1/`.

The BTC transport now refuses redirects before any follow-up request and applies the same exact URL, version, length and checksum checks to both pending- and known-SHA images. The shared mirror downloader is unchanged. Combined owner, historical acquisition/preparation, and independent adversarial checks passed **135/135 in 2.11 seconds**. Repaired source SHA-256: `2978b5c7756f46aac9dc8aa9f308fdf6f84a054326a48bd704fe3addc5bca0bf`. No real image GET occurred during this audit; prospective roles and acquisition do not grant anatomical acceptance.
