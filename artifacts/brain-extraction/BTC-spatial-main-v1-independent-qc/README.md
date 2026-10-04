# Independent QC: four fixed BTC main-only estimates

The retained PAT22/PAT25 TRAIN and PAT26/PAT27 SELECT estimates pass the independent engineering and portability checks. All four inference outputs were fixed before annotation-overlap QC. This audit performed no inference, training, variant selection, threshold change or access selection.

| Subject | Frozen role | Envelope volume (mL) | Source annotation voxels | Omitted voxels |
|---|---|---:|---:|---:|
| PAT22 | TRAIN | 1458.75 | 13,915 | 0 |
| PAT25 | TRAIN | 1506.34 | 16,526 | 0 |
| PAT26 | SELECT | 1767.17 | 55,312 | 0 |
| PAT27 | SELECT | 1794.08 | 11,983 | 0 |

Mask and predicted-distance grids match their native T1 affines exactly, including all eight full-cell corners. All distances are finite. Independently applying SDT < 1 mm, retaining the largest component and filling holes reproduces each delivered mask exactly. Each mask has one connected component and no field-of-view face contacts. Numeric QC took 9.56 seconds and peaked at 529.0 MiB.

All 24 rendered native planes were visually inspected. No obvious gross displacement or axis reversal was seen in those sampled views. Small anterior/inferior contour protrusions in PAT25 warrant anatomical review; this inspection does not determine whether they are correct. Predicted whole-brain envelopes include inferior structures and do not define cortical entry sites. The SDT exterior fill at 100 mm is not a validated clearance measurement.

Independent save/reopen checks preserved MRI, affine, active/source threshold-derived annotations, metadata, fixed patient roles and planning identities. Revision changed from 2 to 3 only through the additive estimate. Each embedded mask equals the fixed output, and embedded QC-report text matches the separately saved derivative byte for byte. Original reports remain unchanged. SDTs remain external hashed artifacts. All four estimates remain `estimated` / `review_required`, working brain remains absent, and explicit use of each estimate as working support is rejected. Portable QC took 21.24 seconds and peaked at 484.3 MiB.

`independent-qc.json`, `visual-review.json` and `portable-qc.json` are separate immutable receipts. `audit.py` and `audit_portable.py` reproduce their numerical checks. The first portable-audit attempt incorrectly expected the original array index to change; it correctly remained unchanged. The tightened assertion passed on the second audit attempt. The first source snapshot and error log are retained; no model inference was repeated.

Zero annotation omission is not a measure of segmentation accuracy. No expert anatomical review, cortex localization or surgical-access approval was performed. Current source/model/runner/interpreter bindings were checked; historical PAT05 installed-package binary equivalence and pretrained-model cohort disjointness remain unverified. No PAT29/PAT31 or external final cohort was opened.

Commit the compact text receipts, audit sources and this record. The four PNG overlays and independent `.ressectionlab` roundtrip copies are local binary QC artifacts.
