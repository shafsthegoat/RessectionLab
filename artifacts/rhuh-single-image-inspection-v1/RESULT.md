# RHUH-0001: first bounded image inspection

The one released inspection completed with inspector and outer launcher exit codes 0. The parent accepted **format inspection only**; scientific use, anatomical registration and clinical validation remain false. This interpretation reads saved JSON only. The [independent saved-artifact audit](independent-result-audit.json) passed on its first run: 23 archived files, release history, result bindings and arithmetic agreed, and original compressed bytes were unchanged. It did not re-decode the image or independently recompute intensities or anatomy. No additional image reads, decoding or visual inspection were performed for this note.

| Saved finding | Interpretation |
|---|---|
| Scalar NIfTI-1, little-endian float32, 240 × 240 × 155 | Declared storage matches the supported format; 8,928,000 voxels |
| 1 × 1 × 1 mm spacing; spatial units mm | Header-declared physical sampling, not independently verified anatomy |
| qform code 1, finite/invertible; voxel-axis directions L/P/S | A usable numerical voxel-to-world transform is recorded; the code does **not** establish scanner-native provenance in this processed cohort |
| sform code 0, absent | qform-only representation; no qform/sform agreement test is available |
| Gzip footer, single member and exact EOF valid; no extensions | Stream integrity and exact declared length passed |
| 35,712,000 voxel bytes; offset 352; total 35,712,352 uncompressed bytes | Consistent bounded storage |
| 8,928,000 finite scaled values; zero nonfinite | Numerical values are finite, without establishing image quality or tissue identity |

The recorded affine is `diag(-1, -1, 1, 1)` with translation `(0, 239, 0)` mm. L/P/S describes increasing voxel-axis directions under the NIfTI convention; it is not a verified patient-registration or atlas label. The report contains no geometry issue from the implemented checks, while explicitly retaining unverified scanner/atlas provenance and unaccepted anatomical alignment.

Effective intensity scaling is slope 1, intercept 0. The descriptive range is **−8.976039 to 10.492467**, with finite-voxel mean **−7.312971**. These values alone do not identify background, brain, tumor, tissue mechanics or a physical MRI unit. In particular, nonzero intensity must not become an intracranial/support mask.

The original compressed payload remains bound to 6,337,221 bytes and SHA256 `b3b9fa69b87221062261b8b16fbddfdac2c678104b354ba371b069fd784e3625`. Its MD5 matches the predeclared index-token candidate; the publisher's algorithm remains unconfirmed. This format result adds no new source-identity or outcome-linkage evidence.

One execution took 0.328035 s at the root and 0.272976 s in the parent. Three RSS samples recorded a combined peak of 192,675,840 bytes (183.75 MiB). These are single-attempt observations, not a latency distribution or continuous memory maximum.

The next data-validation gap is **source-processing/frame interpretation and controlled anatomical visual QC**, under a separate release. Check the documented resampling/normalization provenance and inspect fixed orthogonal planes for coverage, orientation and gross artifacts without changing the source. No such view has yet been inspected. A single structurally readable T1 does not supply a reviewed brain/cortex/target mask, vessels, patient function, surgical injury labels or a neurological-risk model. Case import and scientific use remain unreleased.

Evidence: release commit `dd633a21edfb48b27cacb72da1c2e042f6cd7c21`; saved `stdout.json` SHA256 `7a0bdb595034c519967c7ce044f130d2119d0a361e550d5fd424ba03fbf3c9b1`; `parent-summary.json` SHA256 `a28dfcf3fd96c70cbf5c0e361dc3fda9a64b935ceba07ec22fae4a50c4c0f59e`; `root-terminal.json` SHA256 `8d9287aed3301dc8e93014b970b9dcfef0034fa541b0a285566fbe6316ca0040`. Paths are under this artifact directory except the original inspector output at `outputs/rhuh-single-image-inspection-v1/run-first/stdout.json`.
