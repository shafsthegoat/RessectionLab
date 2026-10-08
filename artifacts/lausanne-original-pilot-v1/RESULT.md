# Acquired original T1 / TOF pilot

October 8, 2026. One prospectively TRAIN person, Lausanne sub-000, was acquired
from the pinned official release. The four source files total **36,661,729 bytes**.
Both image sizes/MD5 and both sidecar SHA-256 match the published/pinned source.

| Image | Acquired shape | Voxel spacing, mm | Value range |
|---|---|---|---|
| T1 | 37 × 420 × 448 | 3.900 × 0.558 × 0.558 | 0–2362, all finite |
| TOF | 350 × 448 × 160 | 0.469 × 0.469 × 0.700 | 0–843, all finite |

[Acquisition receipt](acquisition.json), [independent recomputation](independent-check.json)
and [display receipt](display.json) bind exact source/output hashes. An independent
agent also recomputed byte checks, geometry and finite-value statistics directly
from these four local files, without the importer or other patient access.
The original download/QC took 157.357 seconds and stayed below its 240-second
cooperative deadline. All source checks passed; numerical geometry passed.

**Physical-frame provenance remains unresolved.** The TOF DICOM sidecar is
approximately 1.70° oblique; the NIfTI frame is axis-aligned. T1 axes agree with
their sidecar to rounding precision, and qform/sform corner disagreement is
0.000421857 mm. No cross-scan registration or source repair was attempted.

The six central planes were rendered and inspected locally. Labels and physical
aspect are readable; T1's coarse sampling is visible and TOF supplies a limited
slab. This is display QC, not a full-volume anatomical or clinical review.

Independent source review found that the original runner could catch its own
timeout during QC as an ordinary file error. The actual run completed before
that condition occurred. The subsequent [two-line correction](post-acquisition-timeout-fix.diff)
propagates TimeoutError. The unchanged receipt retains the original executed
source SHA; reversing this diff on the current script reconstructs those exact
bytes, also retained locally under `build/lausanne-original-pilot-v1/`.
The focused source-role and deadline-regression suite passes **2 tests in 0.47 s**.

No patient segmentation, scanner-world correspondence, vascular coverage,
mechanical fidelity, surgical outcome, model accuracy or clinical benefit was
validated. There were **zero training updates and zero recorded RL transitions**.
The fixed full TRAIN cohort has 199 people / 210 sessions; this pilot does not
satisfy full-cohort acquisition or use.
