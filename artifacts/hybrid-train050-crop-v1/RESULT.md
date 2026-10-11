# Public TRAIN050 geometry preparation

The next fixed patient now has a converted public MRI/annotation crop with its
source domains preserved. Both bounded phases complete: conversion in 1.540 s
and saved-domain checking in 0.438 s, 1.979 s overall. Maximum sampled worker
memory is 850,821,120 bytes, with no remaining process or cleanup error. The
patient remains TRAIN and has not been used to refit or evaluate a policy.

| Quantity | Result |
|---|---:|
| Crop dimensions | 134 × 168 × 137 |
| Crop cells | 3,084,144 |
| Full source/placed tumor-positive cells | 19,557 / 19,557 |
| Source-positive centres outside crop | 0 |
| Tumor in automatic cerebrum / known-label zero | 9,791 / 9,766 |
| Tumor in unknown support coverage | 0 |
| Unknown support-domain cells | 183,804 (5.96%) |

Automatic cerebrum is a source estimate, not complete physical occupancy. Its
zero label does not mean air or harmless tissue. All tumor cells are retained;
49.94% fall outside that automatic label. Missing support coverage remains
unknown. The raw MRI, support estimate, supplied whole-tumor region and their
two domains remain separate. No material or patient-specific injury parameters
are inferred from them.

The existing converter verifies the source pixels, uses native MRI samples and
nearest-neighbor label placement, and preserves the source grids/affines. It
reopens 137 MRI slices plus two public SEG objects; the other 55 MRI slices need
only the preceding verified header record. The independent saved-domain worker
checks all domain voxels, saved hashes, source-centre coverage and counts. A
separate reviewer verifies the saved bindings, accounting and arithmetic without
reopening arrays. All 21 generated controls pass.

Root inspected the three existing deterministic previews at crop indices
107/62/83. The automatic support contours broadly follow the visible brain;
the supplied tumor overlaps the visible lesion, with no obvious gross displaced,
reversed or cropped overlay in these views. This is a limited visual check of
oblique source-array planes, not expert segmentation review, complete-volume
anatomical validation or independent registration evidence. Neither SEG names
explicit source MRI SOP references; pairing uses the shared frame and source
description. The original preview hashes and orientation limits are recorded.

The exact five-array public manifest is now prepared and independently checked
against the saved geometry, counts, affines and provenance. It explicitly grants
no runtime admission. The next capability is a narrowly scoped frozen-model
comparison; these QC receipts do not silently enlarge the four-patient fitting cohort. Held037
and closed067 remain untouched. No planner, model, optimizer, private reference,
patient substitution or clinical-outcome inference occurs in this preparation.

`preservation-index.json` binds source, generated controls and result receipts.
The original crop receipt SHA256 is
`d0715fe192ba2723b713f137309266325d6e6d281aa017c2ba1f6c1e29b77c78`.
Raw arrays, acquired files and patient preview images remain local and are not
included in this Git record. Reproduction needs the preceding verified header
bindings, exact local acquired bytes and sources named by `crop-source-index.json`.
Restore this archive's staged scripts to their recorded build paths; retain the
historical attempt rather than overwriting it. The source stage adds only 050
to the existing conversion whitelist; decoding and geometry remain unchanged.
