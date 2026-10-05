# Second creator-verified structural development case

On October 4, 2026, `sub-PAT05` was selected from the pinned
[BTC_preop v5.0.1 release](https://openneuro.org/datasets/ds001226/versions/5.0.1).
`manifests/btc_pat05_selection.json` locked the rule at 08:05:59 UTC, before any
PAT05 image download or inspection: sort participant IDs lexicographically,
retain source diagnoses containing glioma, glioblastoma, or astrocytoma, exclude
PAT28, and select the first remaining ID. The source metadata SHA256 is retained.
The cohort registry assigned PAT05 to **development before image acquisition**.

The archived diagnosis `Oligo-astrocytoma II` supports source cohort selection;
it is not reinterpreted as a modern molecular diagnosis or supplied to the
preoperative optimizer. Every clinical/context availability timestamp remains
unknown. No postoperative data, other subjects, DWI, or fMRI were acquired for
this addition.

The seven case files total **17,862,831 bytes**, including four already cached
release documents. New subject files total 17,847,940 bytes: native T1 MRI,
its acquisition JSON, and the creator's source tumor annotation. Both imaging
objects match the pinned Git-annex lengths and MD5 checksums; their exact
official S3 version IDs and SHA256 are in `manifests/btc_pat05_acquisition.json`.
The release is CC0 and the original source bytes are unchanged.

Reproduction retains the PAT28 defaults:

```sh
.venv/bin/python scripts/acquire_btc_case.py --manifest manifests/btc_pat05_acquisition.json
.venv/bin/python scripts/acquire_btc_case.py --manifest manifests/btc_pat05_acquisition.json --verify-only
.venv/bin/python scripts/prepare_btc_case.py --manifest manifests/btc_pat05_acquisition.json --annotation-threshold 0.5
```

The acquisition script accepts only the two reviewed subjects and their fixed
file scopes. PAT05's selection checksum and metadata hash must match the locked
declaration. It reproduces the metadata-only selection during acquisition and
enforces a 100 MB ceiling. PAT28 still requires all 15 diffusion-case files.

Preparation produces `outputs/cases/BTC-sub-PAT05.ressectionlab` and
`outputs/qc/BTC-sub-PAT05.json`; the bundle reopens with identical case and
planning hashes. The source T1 has a 160 × 256 × 256 grid. Source annotation
thresholds 0.25, 0.5, and 0.75 retain 12,147, 11,437, and 10,766 native voxels,
respectively. These are declared source-intensity scenarios, not clinical
probabilities, removed tissue, or validated compartment labels. Source images
and fractional annotation remain separate from the derived target.

The case remains **full-head structural imaging without a reviewed brain
mask**. Automatic cortical access is blocked; nonzero image intensity cannot
become a cerebral surface. Missing diffusion is explicitly reported, motor and
language localization are unavailable, and clinical deficit probability is
null. This acquisition does not extend PAT28's diffusion evidence to PAT05.
Independent visual alignment review is a separate imaging task.
