# IBSR20: prospective one-record tissue diagnostic

**Decision: the metadata protocol is reviewable; acquisition and execution are blocked.**
This fixes a later noncommercial component diagnostic for record `2_4`. It creates
no cohort role and grants no source, anatomical, host, training or planning admission.
The active location feature is unaffected. The machine-readable contract and local
code hashes are in [protocol.json](protocol.json); request evidence is in
[source-metadata.json](source-metadata.json).

## Source, access and rights

Use the **20 Normals** data release
[2308](https://www.nitrc.org/frs/shownotes.php?release_id=2308) and updated README
release [3124](https://www.nitrc.org/frs/shownotes.php?release_id=3124), file
[8151](https://www.nitrc.org/frs/download.php/8151/20Normals.README.txt).
This is separate from IBSR18 and its skull-stripped NIfTI conversion.

| Proposed archive | Official route | Publisher display size | Unauthenticated HEAD, 2026-10-16 |
|---|---|---|---|
| `20Normals_T1_8bit.tgz` | [5680](https://www.nitrc.org/frs/download.php/5680/20Normals_T1_8bit.tgz) | 45 MB | 302 to login |
| `20Normals_T1_seg.tgz` | [5682](https://www.nitrc.org/frs/download.php/5682/20Normals_T1_seg.tgz) | 1 MB | 302 to login |

No HEAD redirects were followed, HEAD bodies read, or archive GETs made. Neither
HEAD returned Content-Length, ETag or an archive checksum. Display sizes are
rounding, **not byte bounds**. Additional terms behind login are unknown. The README
was public: 8,906 bytes, SHA-256
`464b08abf07331a8882a31ad2e5cefcf1badccb74ae61b9a08292194089fbf10`.
Other official archive names/routes are recorded as alternatives in metadata only;
this protocol selects neither brain-only input nor a combined image/label archive.

The [project](https://www.nitrc.org/projects/ibsr/) says noncommercial use only.
[Andrew Worth's 2018 clarification](https://www.nitrc.org/forum/message.php?msg_id=23479)
says NITRC's license prevails over the older MGH README. Preserve that historical
notice and the earlier [qualification](../ibsr-source-qualification-v1/qualification.json).
No commercial, redistribution or image-publication permission is inferred, and
the IBSR18 conversion's public-domain statement is not imported into IBSR20.
The current project's noncommercial eligibility and existing authorized access
have not been established.

## Fixed record and frame

The [README](20Normals.README.txt) identifies `2_4` as a normal woman aged 34.
Its image is source-scaled uint8, 256 columns × 256 rows, with documented slices
1–65. Per-record spacing is **1.171751 × 1.171751 × 3.125 mm**. These are positionally
normalized scans, not untouched scanner-native images. The generic narrative's
3.0-mm reslicing statement must not override the per-record table.

Freeze **offset 6**: label slice 1 corresponds to image slice 7; zero-based label
index `j` maps to image index `j+6`. Actual label depth is unknown and must fit
within the image (at most 59 slices). Codes are **0 other, 128 CSF, 192 GM, 254 WM**.
Zero and unannotated slices are ignored for tissue scoring, never certified
background. The labels were produced by investigators using contours and
histograms; call them a source reference, not wholly independent manual voxel truth.

Raw byte-to-axis order, world directions/origin, archive members and actual
headers remain unverified. Do not invent a RAS affine. A future decoder must
establish these before image-only support inference. Do not optimize alignment
against labels. The README's contradictory `111_2` example does not change the
selected `2_4` offset.

## One fixed blind ordering

1. Resolve access, noncommercial scope, exact transfer limits and a separately
   reviewed source-bound decoder, support/fitting adapter and scorer. Freeze them
   on generated controls before label access. The existing runner is IXI265-only.
2. After a separately authorized transfer, materialize only `2_4` from the full
   image archive; keep other records unopened and labels inaccessible. Use the
   pinned **SynthStrip main** image-only method already used for IXI265: border
   1 mm, two threads, existing MPS adapter and exact code/weight pins in JSON.
   This is proposed support generation, not a released model run. Its training
   overlap with IBSR is unknown. No label-derived mask or brain-only archive.
3. Seal image `I`, predicted binary support `B`, fit domain `D=B∩(I>0)`, and UNKNOWN
   set `U=B∩(I=0)`. Freeze actual UNKNOWN indices from image-only preparation;
   do not copy IXI265's eight cells. Reject nonfinite/negative image values, empty
   support, invalid masks or frame mismatches without repair or fallback.
4. Run one future frozen v3 fit, then seal predictions, provenance and numerical
   checks **before** obtaining/decoding the selected labels. Use float32 N4 masks,
   ANTsPy 0.6.3, N4 shrink 4 / iterations 30,20,10 / tolerance 1e-7 / spline 150,
   and Atropos `KMeans[3]`, `[0.1,1x1x1]`, `[10,0]`, prior weight 0, `r=0`.
   All parameters are copied into JSON. Preserve the full publisher grid and
   intensities; no additional cropping, normalization, resampling or priors.
5. Require nonempty raw classes and corrected-intensity means `μ1<μ2<μ3`.
   Map **1→128, 2→192, 3→254**. Ties/reversals fail; no relabeling or Hungarian
   matching. Preserve raw membership probabilities; they are not physical PVE.
6. Only then decode `2_4` labels and score once using offset 6. Any geometry/code
   inconsistency makes the diagnostic invalid, without fitting again or searching
   shifts. Retain failures as results; no best-case selection or tuning.

## Frozen scoring

Primary domain `E` is all represented voxels with reference code 128/192/254;
**do not intersect it with predicted support**. Predictions outside `D` are
ABSTAIN and count as false negatives for their reference class. Report the
3×4 confusion matrix (three tissue rows, three predicted tissues plus ABSTAIN).
For each class report Dice, precision, recall, predicted/reference volume in mm³,
signed volume error and relative volume error. Zero denominators are NA.
Macro Dice is the unweighted mean only when all three values exist. No acceptance
threshold is set for this single-record diagnostic.

Also report support coverage per tissue, missed support, UNKNOWN counts,
annotation depth, ignored zero voxels and unannotated slices. A secondary score
on `E∩D` is explicitly conditional on estimated support and cannot replace the
primary score. JSON fixes the formulas and exclusions. No surface-distance,
posterior-calibration, physical-PVE or brain-extraction accuracy claim is made.
Oracle label support is **disallowed** here; using it requires a newly named
oracle-support protocol that explicitly disallows extraction accuracy.

## Remaining limits

Execution remains blocked by login/unknown extra terms, unresolved project-use
eligibility and byte limits, unverified decoding geometry, and the absent
IBSR-specific adapter/scorer preparation. No acquired IBSR receipt or cohort role
was found in the searched current records. IBSR18/20 and other-cohort aliases,
including support-model training overlap, remain unknown.

Even a good result would measure agreement for one differently acquired person.
It cannot validate IXI265's tissue anatomy, confirm skull/vessels/function, establish
a normal host, or establish independent multi-patient generalization. This work
performed only public metadata reads, two archive HEADs and local text/code reads;
no scientific payload, source/model run, agreement or shared-document edit occurred.
