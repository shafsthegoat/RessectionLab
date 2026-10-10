# ReMIND-008: usable geometry with material annotation disagreement

The first preselected TRAIN patient completed header and native-resolution crop conversion. Header work verified367 objects/195,025,118bytes in0.716seconds; conversion verified253 selected objects/135,002,220bytes in1.688seconds at1,045,741,568bytes sampled peak RSS. Both owned workers exited0, were reaped, and hit no limit. An initial supervisor dependency import failed before any child launched; the replacement uses standard process-group sampling.

The public cerebrum extent reduces95,420,416MRI voxels to20,770,250without changing source samples. The derived planning grid is explicitly tied to the public cerebrum grid: the maximum crop-frame difference is0.0001075mm, with original affines preserved. Independent stored-output checks match all9artifact hashes, label domains and source placements. Three orthogonal overlays show no obvious gross misalignment; this is not expert segmentation accuracy. No explicit source-SOP links exist; correspondence rests on shared frame, source descriptions and the checked physical grids.

The important negative is semantic:6,394 of 18,509whole-tumor positives (34.545%) lie outside automatic cerebrum support. The original simulator demands containment. Neither clipping the tumor nor filling support is justified silently. The next explicitly declared condition is partial target progress on the unchanged estimated occupancy, retaining the full target denominator and separately reporting unsupported target volume. It does not claim full tumor removal.

294,560 of 320,127ventricular positives (92.013%) occupy public support zeros. This supplies anatomical cues, but only 2.108% of all  support zeros are ventricular positives.309 tumor and ventricular positives overlap. These are automatic source-reference relationships, not neurological harm labels. Withheld ventricular arrays remain excluded from public task preparation. Patient model training has not yet run.

Patient images, masks, overlays and detailed source headers stay local outside Git. All previous cohorts and roles are preserved.

## Reusable remaining-TRAIN preparation

The executed first-case geometry and sample checks are now consolidated into
`resectionlab.remind_planning_qc` and `scripts/prepare_remind_planning.py`. Exact
case/header bindings preserve the original TRAIN roles and source objects. The
canonical imaging environment passes all nine generated controls in 0.010 seconds;
the CLI loads successfully and independent source review passes. An initial pytest
invocation could not run because pytest is absent in that environment; the tests
use standard-library unittest and ran directly without changing dependencies.
This source integration performs no further patient reads or planning admission.
