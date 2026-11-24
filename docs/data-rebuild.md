# RessectionLab data collection

## Multimodal anatomy pilots

The [source catalog](DATASET_ACQUISITION_ADDENDUM.md) lists rights, access,
annotations and known overlaps. On the marked HDD, start with
`catalog/inventory/multimodal-anatomy-baseline-20261118-v1/README.md`. The
baseline checked paths and sizes for 368 accepted packages across 130 source
families, including 208 synthetic cases. It verified 175 private migration
manifest checksums and located the separate EDEN v2 collection of 15 baselines
and 120 variants. See the HDD manifest for imaging and annotation counts,
the first-party acquisition receipts and the scope of file verification.

The development addition has six newly acquired source families, seven acquired
CT-family series, one publisher-registered CTA derivative, three native
showcases on previously acquired TopCoW families, and 12 generated scenarios.
The checkout lists 22 CTA development entries in the core addition, including
a publisher-geometry companion for TopCoW 006. The route-conditioned CTA014
and IXI077 studies each add a pristine case and a separate recording. IXI265
adds an MRI anatomy study, and RESECT Case3 adds three observation studies.
CADS case 465 adds one acquired noncontrast head CT study. Two further
published-geometry companions reuse TopCoW 012 and 014. The recordings and
companion studies reuse existing people. All 33 additions resolve in Cases.
Acquired identities are distinct within their source
namespaces. Cross-publisher subject independence is unverified.

| Retained publisher payload | Bytes |
| --- | ---: |
| TAIR, Head CTA and ISLES’24 acquisitions | 354,652,995 |
| CoW geometry and selected TopCoW source and annotation members | 1,652,113 |
| CoW geometry and exact original CTA masks for 012 and 014 | 3,689,826 |
| RESECT Case3 during-resection cavity annotation | 28,518 |
| RESECT Case3 after-resection cavity and paired landmark files | 46,148 |
| RESECT Case3 two tumor annotations and two MRI–ultrasound landmark files | 145,205 |
| CADS 465 CT, three model-label volumes and publisher metadata | 94,793,190 |
| Total | 455,007,995 |

The three new CTA source directories, prepared pilot and scenario directories,
both recorded workspaces and the Case3 stage benchmark occupy 6.48 GB of
allocated HDD file space. CADS source and preparation add 0.85 GB. The two
new paired companions and their selective source directory add 1.24 GB.
The Case3 target-to-cavity workspaces, report and overlays use 187 MiB.
These figures exclude the baseline inventory, model caches, hidden desktop
captures and portable demonstration copies.

### Selective CTA intake

```sh
export RESSECTIONLAB_DATA_ROOT="/absolute/path/on/mounted/RessectionLabData"
.venv/bin/python scripts/acquire_tair_cta.py 00273 00244 00410
.venv/bin/python scripts/prepare_tair_cta.py 00273 00244 00410
```

The three TAIR scans occupy 183,437,613 source bytes. The downloader reuses
complete objects, resumes partial transfers and verifies the publisher's
full-object CRC64NVME and object length before writing a local SHA256 receipt.
Open `prepared/tair-v1-cta-native/<id>/patient-anatomy.json` through Cases.
The publisher originals retain their full grids. The display uses every second
slice at its original physical location, giving 0.6 mm through-plane sampling
and 0.44–0.50 mm native in-plane spacing. Review the calvarium, brain, skull
base and enhanced structures at `qc/tair-v1-cta-native/<id>/source-planes.png`.
The release is defaced and supplies no segmentation objects for these scans.

TAIR 00273 also has source-bound brain and dense skull estimates. The existing
TotalSegmentator CT adapter ran on the display grid with 3 mm inference. The
worker exited zero, retained both masks and wrote its completion record. Its
inherited logging pipe triggered the supervisor failure. Intake independently
verified the retained binary masks, exact grid, source and pinned model. The
failed attempt receipt remains in the model cache. Its masks were reused without
repeating inference. The skull and brain surfaces have about 2.07 million and
719,000 triangles respectively. Their conditioned display vertices move by at
most 0.42 mm. Anatomy coverage and vascular segmentation remain unqualified.

```sh
export RESSECTIONLAB_MODEL_ROOT="$RESSECTIONLAB_DATA_ROOT/cache/models"
export RESSECTIONLAB_POC_OUTPUT_ROOT="$RESSECTIONLAB_DATA_ROOT/cache/poc-model-results"
# For a fresh CTA study with available model assets.
.venv/bin/python scripts/prepare_tair_cta.py --estimate-skull-brain 00273
```

Ordinary CT preparation checks the existing zero-exit attempt before running
a model. It validates the completion evidence, input and checkpoint hashes,
and both native masks. The original failed attempt stays in the cache. The
explicit `--recover-ct-worker` option can still select a particular receipt.

A bounded VesselFM run on the 0.75 mm CTA crop returned 21,282 positive
voxels across 104 components. The largest component held 17.6% of positives,
and 13.1% intersected the dense skull estimate. Source overlays showed
predictions in frontal sinus regions. The crop and outputs are retained as a
diagnostic experiment in `qc/tair-v1-cta-native/00273/vesselfm-crop-v1/`.
Reproduce its quality check with `.venv/bin/python scripts/prepare_tair_vessels.py`
under the same model-cache environment. TAIR 00273 carries no new vessel layer.

The [Annotated Head CTA release](https://zenodo.org/records/15697196) supports
selective ZIP64 extraction. Development subjects 189 and 013 have 276 and 289
DICOM instances. Subject 013 includes one supplied aneurysm STL. Member CRC and
local SHA256 checks cover the selected originals. The reported whole-archive
MD5 identifies the publisher release. Both native study affines passed an
independent check against DICOM orientation and slice positions.

```sh
.venv/bin/python scripts/acquire_head_cta.py --prepare 189 013
```

Open `prepared/head-cta-15697196-native/<id>/patient-anatomy.json`. The source
and reconstruction checks are under `qc/head-cta-15697196-native/<id>/`.

The subject 013 diagnostic at `qc/head-cta-15697196-native/013/stl-frame-v1/`
compares the original STL against the native CTA under eight axis sign
conventions. The closed 992-triangle mesh measures 109.11 mm³. Four conventions
are in bounds. Only the LPS hypothesis overlaps enhanced tissue, with a median
of 214 HU inside the voxelized surface and 71 HU in its two-voxel shell. The
other three in-bounds placements have interior medians of 36–38 HU. The LPS
candidate lies on the right of the approximate native axial skull midline,
while the publisher metadata specifies left MCA. The original STL and report
remain available for review. The study has no admitted aneurysm layer.

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python \
  scripts/audit_head_cta_aneurysm.py --output-name stl-frame-reproduction
```

### Same-patient NCCT and CTA

The [ISLES'24 v7 release](https://zenodo.org/records/17652035) supplies an acute
admission pair and an automated Circle of Willis annotation for `sub-stroke0146`.
Selective solid-stream extraction used 654,353,001 bytes of reserved transfer
budget, with 71,915,829 bytes of source files retained. The acquisition checks
the exact publisher archive index, member CRCs and local hashes. The optional
original CTA uses a verified LZMA2 reset and has a separate receipt bound to the
unchanged records for the other three members. CC BY-NC-SA 4.0 applies.

```sh
.venv/bin/python scripts/acquire_isles24_cta_pair.py
.venv/bin/python scripts/acquire_isles24_cta_pair.py --native-cta
.venv/bin/python scripts/prepare_isles24_cta_pair.py
```

Open `prepared/isles24-v7-native/sub-stroke0146/patient-anatomy.json`. Acquired
NCCT, publisher-registered CTA and the supplied automated mask share an exact
512 × 512 × 87 grid at 0.451 × 0.451 × 2.0 mm. The acquired CTA has its own
512 × 512 × 207 grid at 0.470 × 0.470 × 0.75 mm and opens independently.
The visually inspected montage and verification receipt are in
`qc/isles24-v7-native/sub-stroke0146/`. The mask contains 2,848 positive voxels
across nine arterial classes and two components. Exact categorical data remain
in the publisher original. The app shows the union as an automated estimate.
The exploratory CTA-minus-NCCT subtraction left substantial dense residuals
and remains a diagnostic image. Independent registration error and downstream
vessel or surgical authority remain unqualified.

The hidden desktop test selected all three scans. NCCT and registered CTA
showed distinct signals on the same physical grid with the same rendered
publisher CoW mask. Switching back restored the exact NCCT signal. The acquired
CTA opened on its independent grid with no transferred layers. See
`qc/hidden-app/isles24-v7-0146-paired-cta-20261118-v1/report.json` and its
three app captures. The six new source-study cards are indexed locally in Cases.

The acquired NCCT also supports a 3 mm model estimate of the brain envelope
and a dense-signal-gated skull estimate. The independently verified retained
worker outputs were used after the same inherited-log condition observed in
TAIR. Their surfaces contain 487,124 and 1,085,448 triangles. Conditioned
displacement stayed below 1 mm on the anisotropic NCCT grid.

```sh
receipt="$RESSECTIONLAB_DATA_ROOT/cache/poc-model-results/anatomy-d9c4252897c3723a/totalsegmentator-skull/356208c07c8b9d6e6e9d/inference.json"
.venv/bin/python scripts/prepare_isles24_cta_pair.py --recover-ct-worker "$receipt"
```

### Tumor-bearing studies on the new CTA subjects

ISLES 0146 and TAIR 00273 came from the selective intake above. The showcases
and scenarios reuse those source people and do not add to the accepted
research release. This preparation reuses the acquired originals and
TotalSegmentator brain and skull masks on their existing physical grids.
Both studies use the retained 3 mm outputs from TotalSegmentator 2.12.0,
Dataset297 v2.0.0, labels 90 and 91, with the density gate at 250 HU for
the skull. The original QA receipts record the checkpoint and mask hashes.
Head CTA 189 lacks retained brain, skull and vessel masks. The 013 aneurysm
STL remains withheld because of the laterality conflict described above.

**ISLES24-0146-CTA-ANATOMY-v1** uses the acquired NCCT, publisher-registered
CTA and categorical publisher CoW mask. The rendered 2,848-voxel union
retains its two components. The acquired CTA remains independently
selectable. The 2 mm NCCT spacing and the 1,111 skull-mask voxels touching
the inferior source face limit skull-base reconstruction. The automated
CoW provides selected arterial context. Distal vessels, veins and independent
registration landmarks remain unavailable.

**TAIR-00273-CTA-ANATOMY-v1** retains exact every-second-slice samples at
their original physical coordinates. The reviewed enhancement segment
follows a visible intracranial tube, with 146 display voxels and 451 voxels
in the unsampled CTA. Its endpoints depend on the threshold. The layer is
an intensity-derived, context-only estimate. Vessel identity and the rest
of the vascular tree remain unresolved. The earlier failed VesselFM output
remains outside the study.

Original masks and mesh triangle indices are preserved. The ISLES skull
and brain already have protected vertices beyond the 0.9 mm limit for
further custom surface attachment. TAIR's first additional skull fit
reduced normal-turn variation by 65.1%, while the 95th-percentile
categorical-boundary residual rose from 0.413 to 0.500. A bounded fit with
the categorical guard brought that residual to 0.397 and preserved 99.94%
of matched CT contrast, with an 11.2% roughness reduction below the preset
15% gate. The showcases retain the existing source-bound display surfaces.
Compared with raw categorical positions, these skull surfaces reduce mean
squared normal turn in the measured upper vault by 63.2% on ISLES and
86.5% on TAIR. Their maximum displacements are 0.999 and 0.420 mm.
Cortical detail, skull-base detail and functional anatomy remain unqualified.

```sh
export RESSECTIONLAB_DATA_ROOT="/absolute/path/on/mounted/RessectionLabData"
export PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
.venv/bin/python scripts/prepare_tair00273_showcase.py --register "$PWD/testcases/cases"
.venv/bin/python scripts/build_new_cta_gliomas.py --host isles --register "$PWD/testcases/cases"
.venv/bin/python scripts/build_new_cta_gliomas.py --host tair --register "$PWD/testcases/cases"
```

To reproduce the numeric display trials, run
`scripts/probe_new_cta_surface_conditioning.py --case <case> --output-name <fresh-name>`
with the same Python environment. The case selectors are `isles0146-skull`,
`isles0146-brain` and `tair00273-skull`. Add `--guard-categorical` for the
second TAIR trial.

Each portable scenario keeps the source scan arrays, baseline layers and
one approximate simulated contour. Exact core, rim and edema truth are
separate NIfTIs. The recipes record physical placement, deterministic seed,
disjoint membership, distances to known exclusions and independent shape
measurements. Only the 46-person UCSF TRAIN summary informs calibration.
The sites span both left and right, anterior and posterior, and higher and
middle-height tissue in the estimated brain envelope. Low-attenuation and
bright-CTA exclusions proxy some unsupported tissue. CSF borders and distal
vessel clearance remain unverified. The cases retain the acquired image
intensities and omit mass effect.
The earlier TopCoW centers were at 0.667–0.810 of the retained brain height.
The new centers span 0.425–0.765. Two from each host lie below 0.60. Heights
use the full retained brain estimate for each source person.

| Site and EDEN profile | ISLES core + rim (mL) | TAIR core + rim (mL) |
| --- | ---: | ---: |
| Left anterior, circumscribed | 11.01 | 9.33 |
| Right posterior, lobulated | 19.98 | 19.68 |
| Right anterior, broad edema | 26.25 | 24.45 |
| Left posterior, EDEN v2 | 25.65 | 26.87 |

Median core-and-rim volumes are 22.81 mL on ISLES and 22.06 mL on TAIR,
compared with 8.31 mL in the earlier TopCoW cohort and 28.27 mL in UCSF TRAIN.
Medians including edema are 49.39 and 52.46 mL, compared with 17.69 and
96.76 mL respectively. All eight core-and-rim masks are single connected
components. Dice between the approximate contours and exact core plus rim
ranges from 0.965 to 0.980. The two `collection.json` files bind images,
masks, standalone native packages and distribution comparisons.

The hidden desktop opened all eight scenario cards through Cases and verified
the actual scalar and contour transfers. ISLES variant 02 switched to the
independent acquired CTA and restored the registered CTA and selected contour.
TAIR variant 03 and ISLES variant 02 both restored their saved source signal,
contour, cursor and selection in fresh processes. For ISLES use
`qc/hidden-app/isles0146-cta-gliomas-{warm-v1,cold-v4}/`. For TAIR use
`qc/hidden-app/tair00273-cta-gliomas-{warm-v1,cold-v1}/`.
The independent verification in `qc/new-cta-showcases-independent-verification-v1.json`
binds those runs to the original scan hashes, exact label grids and cohort receipts.

Matched anterior and oblique desktop pairs are in
`qc/hidden-app/{isles0146,tair00273}-native-surface-pair-v1/`. Each `source-*`
and `smooth-*` pair uses an identical camera, native input and three linked
slice crops. The 3D images change between raw categorical positions and the
retained conditioned surfaces.

For the standard rehearsal check on ISLES variant 02, a temporary copy maps
the exact simulated contour to the standard environment's tumor field. The
contour bytes and simulated provenance are preserved. The production engine
reopens an intact scene with 19,036 mm³ of sampled target. Planning refuses
with `PROCEDURE_POLICY_REQUIRED_FOR_PATIENT_PLANNING`. The full-case check
also requires supplied brainstem and critical-structure coverage. The scene
stays at zero actions. The result is `qc/isles0146-cta-guarded-refusal-v1.json`.
The desktop's unmodified setup independently refuses the same guarded
planning command and preserves its zero-action state.

```sh
.venv/bin/python scripts/build_new_cta_gliomas.py --guarded-rehearsal
```

### Published masks and acquired-host scenarios

TopCoW 006, 012 and 014 reuse same-person TopBrain v3 CTA, existing skull and
brain estimates, and six publisher annotations. Display uses three V1 masks,
with V2 retained separately. Reuse avoided three vessel model runs and three
brain/skull reruns. All three supplied arterial layers reach the inferior scan
boundary. The 006 brain estimate spans the head field, with scenario placement
restricted to the reviewed superior support region. See
`qc/multimodal-anatomy-20261118-vessel-v1/README.md` for exact grids, label
semantics, continuity, and source overlays.

The [published Circle of Willis geometry](https://zenodo.org/records/17358162)
was qualified for 006, 012 and 014 on their original CTA lattices. Every edge
midpoint and all but one graph point were supported by their original CoW
masks. The unsupported point was inside the 006 acquisition. The original
publisher meshes, graphs, label files and notices retain their exact bytes.
The retained 012 and 014 CTAs matched each official archive member's length
and CRC and were reused. Neither geometry required a frame transform.

| TopCoW pilot | Supported graph points | Supported edge midpoints | Mesh vertices | Mesh triangles |
| --- | ---: | ---: | ---: | ---: |
| 006 | 914 of 915 | 914 | 47,202 | 94,404 |
| 012 | 814 | 815 | 61,860 | 123,724 |
| 014 | 773 | 771 | 47,462 | 94,932 |

The 006 proof and source receipt are under
`qc/multimodal-anatomy-20261118-cow-vtp-v1/`. The 012 and 014 receipts are
under `sources/topcow-cow-vtp-selective-v1/`. Their acquisition retained
3,689,826 bytes through 2,874,142 bytes of selective transport. These source
families retain their frozen TRAIN roles.

Search **PUBLISHED-COW** for the native companions. The
**Circle of Willis · publisher mesh** layer uses the original triangular
surface, while the exact original CoW segmentation supplies linked CTA slice
overlays. The wider TopBrain arterial and venous unions remain separate.
The same-person TopBrain MRA and its V1 arterial union open on their own source
grid. Each study keeps both VTP originals, publisher metadata and licenses.
The centerline graph and its attributes remain available as source evidence.

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python \
  scripts/build_cta_published_cow_study.py --register "$PWD/testcases/cases"
.venv/bin/python scripts/acquire_topcow_cta_geometry.py 012 014
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python \
  scripts/build_cta_published_cow_study.py --subject 012 --subject 014 --register "$PWD/testcases/cases"
```

The hidden desktop verified all 47,202 projected publisher vertices, 94,404
triangles and the native CoW mask byte for byte. Hiding the publisher layer
changed the 3D image. Source and conditioned modes preserved the same mesh
and three linked slice crops. The app's portable Save retained the exact VTP,
and a fresh process reopened it with identical transfers. See
`qc/hidden-app/cta006-published-cow-warm-20261118-v1/` and
`qc/hidden-app/cta006-published-cow-cold-20261118-v1/` on the HDD.
The portable MRA check rendered its 429 × 507 × 139 source grid and
122,060 supplied arterial voxels. It verified independence from the CTA
overlays and restored the original CTA mesh on return. The supplied MRA
annotation header omits units, so the app offers native-grid preview.
See `qc/hidden-app/cta006-published-cow-paired-mra-20261118-v1/`.

All three companion packages carry the refined superior skull and brain
surfaces alongside the untouched author vessels. Each fresh-process check
restores the publisher surface, switches to the separate acquired MRA and
returns to the CTA. The 006 update copies only the two previously verified
display attachments. It preserves the preceding native revision, original
CTA and MRA, acquired-grid labels and VTP geometry without rerunning inference
or mesh reconstruction. The hidden save and cold reopen verify the exact
brain and skull positions in the same study as the original publisher mesh.

```sh
subject=012
study="$RESSECTIONLAB_DATA_ROOT/prepared/cta-published-cow-v1/TopCoW-$subject-CTA-MRA-PUBLISHED-COW-v1"
warm="$RESSECTIONLAB_DATA_ROOT/qc/cta$subject-published-cow-warm-reproduction"
SURFACE_VERIFY_NATIVE_PRESENTATION=1 SURFACE_VERIFY_DIRECT_SOURCE="$study/transfer-expectation.json" \
  SURFACE_VERIFY_OUTPUT="$warm" desktop/node_modules/.bin/electron \
  desktop/electron/verify-surface-quality.cjs --anatomy "$study/patient-anatomy.json" --automation-hidden
SURFACE_VERIFY_NATIVE_PRESENTATION=1 SURFACE_VERIFY_DIRECT_SOURCE="$study/transfer-expectation.json" \
  SURFACE_VERIFY_DIRECT_PRIOR="$warm/report.json" SURFACE_VERIFY_DIRECT_MRA_AFTER_REOPEN=1 \
  SURFACE_VERIFY_OUTPUT="$RESSECTIONLAB_DATA_ROOT/qc/cta$subject-published-cow-cold-reproduction" \
  desktop/node_modules/.bin/electron desktop/electron/verify-surface-quality.cjs \
  --anatomy "$warm/publisher-geometry.rsworkspace" --automation-hidden
```

The retained output directories are `qc/hidden-app/cta<id>-published-cow-warm-v1/`
and `qc/hidden-app/cta<id>-published-cow-cold-paired-v1/` for 012 and 014.
For 006, use `qc/hidden-app/cta006-published-cow-refined-warm-v1/` and
`qc/hidden-app/cta006-published-cow-refined-cold-v1/`. The refresh command
reuses the existing verified attachments and is safe to repeat.

```sh
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python \
  scripts/build_cta_published_cow_study.py --subject 006 --refresh-reviewed-surfaces
```

To repeat the paired desktop checks for 006, set `subject=006` in the previous
commands and export `SURFACE_VERIFY_DIRECT_CONDITIONED` with the path to
`qc/hidden-app/cta-showcase-006-refined-surfaces-v1/report.json` on the HDD.

Build or reopen 12 development scenarios and register them in the checkout's
Cases library. The EDEN v2 generator provides the lobulated geometry. A
checksummed 46-person UCSF TRAIN summary guides sizing and excludes four people
listed in BraTS validation. The scenarios span 3.90–12.87 mL across four recipes
per host. Each preserves acquired CTA and source anatomy and exposes an
approximate simulated contour. Exact core, rim, and edema labels stay in
separate research files. The cases omit mass effect and synthetic CTA signal.

```sh
.venv/bin/python scripts/build_cta_organic_gliomas.py --register "$PWD/testcases/cases"
```

Search **CTA-ORGANIC** in Cases. Package checksums, seeds and a contact sheet are
under `prepared/cta-organic-gliomas-v1/`. The global accepted release is unchanged.
The cohort shape audit compared the 12 private masks with the existing
46-person UCSF TRAIN summary using the same physical volume, surface,
and covariance measurements. All 12 core-and-rim masks are single connected
components, with sphericity 0.632–0.828 and axis ratios 0.367–0.720. Their
median volume is 8.31 mL versus 28.27 mL in the reference. Including edema,
five scenarios fall below the smallest reference volume and two exceed the
reference sphericity range. The engineering cohort emphasizes small targets
and compact edema. All regions stayed within the modeled brain and excluded
the retained bone and vessel memberships. Dice between each approximate
contour and private core-and-rim mask was 0.942–0.971. The app packages
preserve the original host attachments, with exact tumor truth stored separately.
The report and plot are in `qc/cta-organic-cohort-shape-audit-v1/`.

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python \
  scripts/audit_cta_organic_cohort.py --output-name cta-organic-shape-audit-reproduction
```

The hidden desktop opened all 12 actual Cases entries, verified the native CTA
and five layer transfers against the sealed receipts, and prepared a GPU frame
on the correct source grid for each. Every case entered and canceled an
inspection preview at zero accepted actions. All project hashes and simulation
states were unchanged. Clinical planning remained unavailable. The report and
two captures are in `qc/hidden-app/cta-organic-cohort-20261118-v1/`.

```sh
cohort="$RESSECTIONLAB_DATA_ROOT/prepared/cta-organic-gliomas-v1"
DATA_REBUILD_CASE="$cohort/TopCoW-006-CTA-ORGANIC-01-v1/TopCoW-006-CTA-ORGANIC-01-v1.ressectionlab" \
  DATA_REBUILD_CTA_COLLECTION="$cohort/collection.json" DATA_REBUILD_CTA_COHORT=1 \
  DATA_REBUILD_OUTPUT="$RESSECTIONLAB_DATA_ROOT/qc/cta-organic-cohort-app-reproduction" \
  desktop/node_modules/.bin/electron desktop/electron/verify-data-rebuild.cjs --automation-hidden
```

The separate CTA014 variant 01 save restored the same CTA and contour in a
fresh process. See `qc/hidden-app/cta-organic-014-warm-20261118-v2/report.json`
and `qc/hidden-app/cta-organic-014-cold-20261118-v2/report.json`. The earlier
reviewed CTA014 replay restored a 48-state run with 15.81 mm³ of target effect
and withdrawal. Its report is `qc/hidden-app/cta-pilot-baseline-20261118-v1/report.json`.

The three **NATIVE-SHOWCASE** entries use the native conditioned surfaces.
Each shows the matched CTA, broader mineralized skull estimate, brain envelope,
separate publisher artery and vein/sinus unions, and its existing variant 01
contour. Fifteen meshes were verified and cached. Conditioning changed 3D
pixels while preserving all three linked source-slice views in each study.
The skull and brain remain estimates. The opaque skull views show anterior
discontinuities and partial skull-base coverage. The brain envelopes lack
qualified cortical or ventricular boundaries. The
arterial unions reach the scan boundary and cover only the supplied branches.
These studies carry display and inspection scope.

| TopCoW pilot | Brain triangles | Skull triangles | Artery / vein connected components | Skull voxels touching image boundary |
| --- | ---: | ---: | ---: | ---: |
| 006 | 730,496 | 1,923,752 | 9 / 7 | 37,183 |
| 012 | 927,588 | 2,087,724 | 5 / 2 | 23,913 |
| 014 | 590,488 | 1,543,300 | 8 / 3 | 23,968 |

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python \
  scripts/build_cta_native_showcases.py --register "$PWD/testcases/cases"
showcase="$RESSECTIONLAB_DATA_ROOT/prepared/cta-native-showcases-v1/TopCoW-014-CTA-ORGANIC-01-v1-NATIVE-SHOWCASE/patient-anatomy.json"
SURFACE_VERIFY_NATIVE_PRESENTATION=1 SURFACE_VERIFY_OUTPUT="$RESSECTIONLAB_DATA_ROOT/qc/cta-showcase-reproduction" \
  desktop/node_modules/.bin/electron desktop/electron/verify-surface-quality.cjs --anatomy "$showcase" --automation-hidden
```

Open `qc/hidden-app/cta-native-showcase-014-surfaces-20261118-v2/smooth-oblique.png`
for the combined view. The 006 and 012 proofs are in the same `hidden-app/`
directory under `cta-native-showcase-006-surfaces-20261118-v1` and
`cta-native-showcase-012-surfaces-20261118-v1`.

The source-bound refinement adds a physical-coordinate low-pass to the
superior skull and brain envelopes. Vertices on the protected lower surfaces
and within two cells of any scan clipping face are fixed at stock positions.
The unchanged native CTA, masks and raw triangle connectivity stay bound to
each layer. No new mesh intersections were found. The median CTA contrast
across matched boundary samples remained within 0.52% of stock.

| TopCoW pilot | Skull normal-turn reduction | Brain normal-turn reduction |
| --- | ---: | ---: |
| 006 | 46.3% | 63.1% |
| 012 | 71.0% | 61.5% |
| 014 | 40.8% | 48.4% |

The metric is the decrease in mean squared angular variation between adjacent
faces on the measured superior surface. The evidence and candidate hashes are
in `qc/cta-showcase-conditioned-surfaces-v1/`. The same directory includes
the three paired superior skull crops. Matched desktop comparisons
use the same camera and linked source slices. Only the conditioned display
positions change. They are in `qc/hidden-app/cta-showcase-<id>-refined-surfaces-v1/`.
The underlying skull-base crop and brain-envelope uncertainty remain visible.
The live Cases snapshots, A/B records and cold paired companion checks are
indexed in `qc/cta-showcase-conditioned-surfaces-v1/cohort-verification.json`.

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python \
  scripts/probe_cta_showcase_conditioning.py 006 012 014 --output-name cta-showcase-surface-reproduction
subject=014
trial="qc/cta-showcase-surface-reproduction/$subject-staged-study"
.venv/bin/python scripts/install_cta_showcase_conditioning.py "$subject" \
  --proofs qc/cta-showcase-surface-reproduction --trial "$trial"
stock="$RESSECTIONLAB_DATA_ROOT/qc/hidden-app/cta-native-showcase-$subject-surfaces-20261118-v2/report.json"
SURFACE_VERIFY_NATIVE_PRESENTATION=1 SURFACE_VERIFY_REFINED_LAYER=broader_skull_display,brain \
  SURFACE_VERIFY_EXPECTED="$stock" \
  SURFACE_VERIFY_OUTPUT="$RESSECTIONLAB_DATA_ROOT/qc/cta014-refined-surface-reproduction" \
  desktop/node_modules/.bin/electron desktop/electron/verify-surface-quality.cjs \
  --anatomy "$RESSECTIONLAB_DATA_ROOT/$trial/patient-anatomy.json" --automation-hidden
```

CTA014 organic variants 01 and 03 and CTA012 variant 03 received fresh read-only
preparation screens. All 10,752 sampled straight stock-tool paths were refused.
Eighty-seven reached the final gate, where the proposed drill cut failed to
cover the CUSA bone blockers within the source-tissue constraints. The three
scenarios remain intact and preview-only. See `qc/cta-organic-rehearsal-preflight-v1/`
for CTA014 variant 03, and `qc/TopCoW-014-CTA-ORGANIC-01-v1-static-preflight/` and
`qc/TopCoW-012-CTA-ORGANIC-03-v1-static-preflight/` for the other results.

```sh
.venv/bin/python scripts/probe_cta_organic_rehearsal.py --root "$RESSECTIONLAB_DATA_ROOT" \
  --case-id TopCoW-014-CTA-ORGANIC-03-v1 --output-name cta-organic-rehearsal-preflight-reproduction
```

The next bounded screen on CTA014 variant 01 evaluated 60 stock craniotome
lanes across three of the straight screen's final-gate routes. Fifty-six
lanes predicted new retained-anatomy contact on return and four crossed the
native field boundary. Each route retained all of its bone blockers. A single
unobstructed CUSA sweep on those paths would contact at most 87 of 15,270
visible contour cells. This evaluates fixed-attitude lanes with passive
return. The source case and initial simulated state remained intact.

```sh
.venv/bin/python scripts/probe_cta_organic_multilane.py \
  --output-name cta014-organic-01-multilane-reproduction
```

**TopCoW-014-CTA-ORGANIC-FIXED-ROUTE-01-v1** is a separate development
case on the same acquired CTA and unchanged tissue masks. Its 0.21 mL
generated target is conditioned on a point 5 mm upstream of a previously
traced endpoint. The 0.8 mm contour blur retains a proxy-to-truth Dice of
0.886 for this small target. The 1.3 mm blur used by the larger cohort gave
0.694 on the same target, below the source contour quality gate. The exact
core, rim and edema stay outside the app package.

Using the historical geometry only, 950 opening and 782 kerf sweeps miss the
719-cell contour. The stock CUSA working envelope along the historically
attained part of the line intersects 375 contour cells. The fresh study
contains no actions or opening. These results qualify a candidate for an
independently checked rehearsal from the starting state. They leave scalp,
dural, and unmapped anatomy assumptions explicit.

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python \
  scripts/probe_cta014_route_conditioned.py --register "$PWD/testcases/cases"
```

See `prepared/cta014-organic-fixed-route-v1/preflight.json` and the source
shape QA beside the package.

The separate **TopCoW-014-CTA-ORGANIC-FIXED-ROUTE-01-v1-RECORDED** card
starts from the pristine case and preserves 45 newly checked actions. The
simulator repeated 24 opening actions and removed 552 estimated skull cells.
Five kerfs and their returns removed 22 more. Two stock CUSA applies produced
the first target effect, followed by withdrawal along the same line with no
additional removal. The recorded totals are 131.52 mm³ of estimated skull,
47.66 mm³ of non-target brain and 8.02 mm³ of target. The edited target count
is 35 of 719 visible contour cells. Cold backend reopen restored the exact
acquired image, source masks, action history and final state.

Scalp and dural exposure remain assumed. The modeled action-boundary drill
temperature reached 124.25 °C. That is a sampled numerical result. It supplies
no measured thermal history or clinical clearance. The partial vessel layers
show zero modeled encounters in this run, with unobserved vessels outside the
scope of the check. Complete removal remains unqualified.

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python \
  scripts/rehearse_cta014_route_conditioned.py --output-name cta014-organic-replay-reproduction
```

The active replay and guarded action log are in
`qc/cta014-organic-fixed-route-rehearsal-v1/` on the HDD.
The second backend run checked all 17 guarded bone blockers and reproduced
all nine archive members byte for byte after decompression. Its report and
restore pointer are in `qc/cta014-organic-fixed-route-rehearsal-reproduction-20261118/`.
The hidden desktop verified completed GPU frames at pristine state, the
552-cell drilled opening, first target-effect boundary and withdrawal. Play,
Pause and scrubbing used native replay observations. The selected frame at
29.8 s within the second 30 s apply shows 7.79 mm³ of target effect. The
accepted boundary shows 8.02 mm³. All stages preserve the acquired CTA signal.
The report and five captures are in
`qc/hidden-app/cta014-organic-conditioned-replay-20261118-v1/`.

```sh
trial="$RESSECTIONLAB_DATA_ROOT/qc/cta014-organic-fixed-route-rehearsal-v1"
DATA_REBUILD_CASE="$trial/replay.ressectionlab" \
  DATA_REBUILD_OUTPUT="$RESSECTIONLAB_DATA_ROOT/qc/cta014-replay-app-reproduction" \
  DATA_REBUILD_REPLAY_ONLY=1 DATA_REBUILD_REPLAY_ACTION=43 \
  DATA_REBUILD_REPLAY_OPENING_REPORT="$trial/report.json" \
  DATA_REBUILD_REPLAY_START_STATE_ID="$(jq -r '.[43].before_state_id' "$trial/actions.json")" \
  DATA_REBUILD_EXPECT_OPERATING_STATE_ID="$(jq -r '.target_effect_state_id' "$trial/report.json")" \
  DATA_REBUILD_EXPECT_STATE_ID="$(jq -r '.final_state_id' "$trial/report.json")" \
  DATA_REBUILD_EXPECT_REPLAY_ACTIONS="$(jq -r '.committed_action_count' "$trial/report.json")" \
  desktop/node_modules/.bin/electron desktop/electron/verify-data-rebuild.cjs --automation-hidden
```

### Noncontrast head CT and supplied brain tissue estimates

Search **CADS-BRAINCT-465-NATIVE-v1** in Cases. The [pinned BrainCT-1mm release](https://huggingface.co/datasets/huggingface/CADS-dataset/blob/eee3c0a993636264fd04104f7e2a090c507c0c43/0042_new_brainct_1mm/README_0042_new_brainct_1mm.md)
provides the acquired noncontrast CT and three segmentation parts under CC BY
4.0. The selection uses the finest in-plane spacing among published 512 × 512
grids with 180–225 slices at 1 mm. Case 465 is 512 × 512 × 208 at
0.426 × 0.426 × 1 mm. The pinned publisher spacing index establishes the unit
for a display copy that changes only the NIfTI spatial-unit bits. The original
CT and all three labels retain their exact bytes and affine.

The seven displayed layer definitions preserve the publisher's brain, compact
and spongy head bone, brainstem, gray matter, white matter, CSF and carotid
memberships. Every label is a model estimate. The source planes show broad
vault and posterior skull-base coverage. The head-bone union includes neck
bone and meets the lower scan edge at 8,948 voxels. Brain and bone estimates
overlap at 51,715 voxels, retained for inspection. Dedicated ventricular
subdivisions, cerebellar parcellation and detailed intracranial vessels are
unavailable.

The conditioned head-bone, brain and brainstem surfaces contain 2,132,164,
664,100 and 43,220 triangles. Desktop conditioning changed the 3D rendering
and preserved the linked source-slice views. The linked desktop panes use
the 256 × 256 × 104 preview grid. Surface extraction and the saved acquired
source use the full lattice. The study has inspection scope.

```sh
.venv/bin/python scripts/acquire_cads_brainct.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/prepare_cads_brainct.py \
  --register "$PWD/testcases/cases"
study="$RESSECTIONLAB_DATA_ROOT/prepared/cads-brainct-1mm-v1/CADS-BRAINCT-465-NATIVE-v1/patient-anatomy.json"
SURFACE_VERIFY_NATIVE_PRESENTATION=1 \
  SURFACE_VERIFY_OUTPUT="$RESSECTIONLAB_DATA_ROOT/qc/cads-465-surfaces-reproduction" \
  desktop/node_modules/.bin/electron desktop/electron/verify-surface-quality.cjs --anatomy "$study" --automation-hidden
DATA_REBUILD_CASE="$study" DATA_REBUILD_INSPECT_STUDY=1 DATA_REBUILD_NATIVE_VIA_CASES=1 \
  DATA_REBUILD_OUTPUT="$RESSECTIONLAB_DATA_ROOT/qc/cads-465-cases-reproduction" \
  desktop/node_modules/.bin/electron desktop/electron/verify-data-rebuild.cjs --automation-hidden
```

The source receipt and overlay are with the prepared study. The captures and
desktop reports are in `qc/hidden-app/cads-brainct-465-surfaces-20261118-v1/` and
`qc/hidden-app/cads-brainct-465-cases-20261118-v1/`.

### Source-supported anatomical detail

The field dimensions below are voxel counts multiplied by acquired spacing.
The CTA registered to ISLES NCCT and the displayed TAIR slice selection are
listed separately from the acquired grids.

| Candidate | Spacing in mm | Sampled field in mm | Retained anatomical evidence |
| --- | --- | --- | --- |
| ISLES 0146 | CTA native 0.470 × 0.470 × 0.750. Registered pair 0.451 × 0.451 × 2.000 | CTA native 240.8 × 240.8 × 155.2. Registered pair 230.7 × 230.7 × 174.0 | Acquired CTA cuts through the upper vault. The registered skull mask reaches the lower boundary. The automated CoW mask has nine classes and 2,848 voxels across two components. No reviewed cortical labels or qualified transfer of the registered contours to native CTA. |
| TAIR 00273 | 0.496 × 0.496 × 0.300 native. 0.600 through plane in the exact display selection | 254.0 × 254.0 × 226.2 | Finest grid. Defaced anterior field and 3 mm brain and skull model lattice. Native unclassified enhancement occupies 451 voxels and provides limited vascular evidence. No independent tissue or arterial annotation. |
| Head CTA 189 | 0.478 × 0.478 × 0.800 | 244.7 × 244.7 × 220.8 | Defaced head and upper neck control with wider vault and posterior base coverage. The retained source has no admitted cortical, bone or vascular masks. The DICOM nominal slice thickness is 1.0 mm. |
| TopCoW 014 | CTA 0.553 × 0.553 × 0.750. TOF MRA 0.297 × 0.297 × 0.600 | CTA 136.5 × 181.3 × 158.2. TOF MRA 144.0 × 188.9 × 104.4 | Publisher CoW categorical mask, mesh and centerline graph. Same-person TopBrain CTA artery and vein/sinus masks, plus separate native TOF annotation. The CTA skull estimate touches all faces. No reviewed cortical labels or qualified CTA-to-TOF registration. |
| CADS 465 | 0.426 × 0.426 × 1.000 | 218.0 × 218.0 × 208.0 | Native noncontrast CT and exact publisher brain, tissue and skull-base model categories. The posterior base is visible. The bone union reaches the inferior face at 8,948 voxels, and the brain and bone estimates overlap at 51,715 voxels. Tissue and carotid estimates lack independent annotation. |

Open **TopCoW-014-CTA-PUBLISHER-BRANCHES-v1** for the 11 named partitions of
the publisher CoW mesh. Each class has an independently selectable 3D surface
and a native categorical slice overlay. Reassembling the partitions recovers
all 94,932 triangles and 47,462 vertex positions exactly. The original
surface has two components, zero open or nonmanifold edges and a 0.569 mm p95
distance from mesh vertices to the nearest original mask boundary voxel
center. The original mask has 6,115 voxels. All 773 centerline points and 771
edge midpoints land in the mask, with the exact class present at 768
midpoints. The two graph components and six junctions are preserved. The
broader artery, vein/sinus and simulated target remain available for context.

Open **CADS-BRAINCT-465-SKULL-BASE-DETAIL-v1** for the full 512 × 512 × 208 linked
CT grid and eight named classes from the retained categorical files. Compact
and spongy head bone have 1,884,026 and 393,037 voxels. The left and right
cochlea estimates have 837 and 1,298 voxels, with median source values of
918 and 990.5 CT units. The optic nerves, chiasm and pituitary are available
as additional model overlays. The source array and affine are identical to
the retained acquisition. The source CT shows fine bone detail, and cortical
folds, small soft-tissue boundaries and ventricular subtypes remain uncertain.

A single posterior skull-base edge-fit trial was rejected. Across 486 held-out
CT edges in 17 spatial blocks, equal-block error increased from 0.446 to
0.448 mm. Four blocks improved, and p90 error increased from 0.727 to
0.747 mm. The trial preserved 1,021,772 protected vertices and introduced
zero new triangle contacts. The report is in
`qc/cads465-skull-base-edge-v1/report.json`. The same directory holds the
archived probe script and its checksum. To reproduce the trial, check
`SHA256SUMS` and run the script with `PYTHONPATH="$PWD/src:$PWD/scripts"`,
`--root`, and a fresh `--output-name`.

```sh
.venv/bin/python scripts/prepare_topcow014_publisher_branches.py --register "$PWD/testcases/cases"
.venv/bin/python scripts/prepare_cads465_skull_base_detail.py --register "$PWD/testcases/cases"
```

Matched hidden desktop views are in
`qc/hidden-app/topcow014-publisher-branches-v1/` and
`qc/hidden-app/cads465-skull-base-detail-v1/`. Each overlay pair records the same
camera, slice cursor, and source signal. The vessel pair swaps the complete
publisher mesh and the 11 rendered classes, using the companion's colors
through the layer controls where needed. The CADS pairs show the temporal
bone/cochlea and gray matter estimate against the same CT. All 20 earlier CTA
scenario packages match their original cohort sizes and SHA-256 digests in
`qc/anatomical-fidelity-cta-cohort-integrity-v1.json`.

### Annotated CTA vessel comparison

Search **VESSEL-COMPARISON** in Cases for TopCoW 014 and 018. Each portable
study opens the acquired CTA at full native resolution with the supplied
TopBrain arterial and venous landmarks. The vascular layer list also includes
the baseline and diagnostic comparison within the publisher's Circle of Willis
box, the reference labels in that box, and a separate right M2 layer. Right M2
keeps every voxel of publisher class 17. The baseline missed 2,100 of 3,540
of these labelled voxels on 014 and 2,488 of 4,635 on 018. The original
annotations are included in the archives. Their headers omit spatial units,
so the viewer uses the exact acquired CTA lattice for inspection.
[TopBrain release](https://zenodo.org/records/21972006) ·
[TopCoW release and ROI definitions](https://zenodo.org/records/15692630).

TopCoW 006 and 012 selected settings. TopCoW 014 and 018 verified the frozen
comparison. The five alternatives used lower or higher thresholds, constant
patch blending, connected low-score growth from higher-score seeds, and
high-attenuation exclusion. All failed the selection criteria. Constant
blending ranked highest as a diagnostic alternative and failed verification.
The existing Gaussian blending and 0.50 threshold remain the default.

| Verification subject | Centerline F1 | Recall for estimated diameter under 2 mm | Complete branch segments | Boundary HD95 |
| --- | ---: | ---: | ---: | ---: |
| 014 | 0.747 → 0.712 | 50.2% → 45.6% | 39.1% → 39.1% | 3.61 → 4.69 mm |
| 018 | 0.770 → 0.775 | 57.5% → 55.2% | 59.1% → 63.6% | 4.91 → 4.52 mm |

Arrows compare Gaussian with constant blending at the same threshold. The
central metrics use 1 mm centerline tolerance and the union of the two
publisher masks inside the TopCoW box. Beyond that box, only agreement with
positive TopBrain landmarks is scored. The four scans are publisher training
subjects and pretrained exposure is unknown. These results quantify agreement
with the supplied annotations within their coverage.

The private `qc/annotated-cta-vascular-benchmark-v1/` directory contains the
frozen protocol, source receipts, cached-inference bindings, measurements and
paired hidden desktop captures. The studies and cold-reopen receipts live in
`prepared/annotated-cta-vascular-comparison-v1/`.

```sh
export PYTHONPATH=src:scripts OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
.venv/bin/python scripts/benchmark_annotated_cta_vessels.py --root "$RESSECTIONLAB_DATA_ROOT" verification
.venv/bin/python scripts/prepare_annotated_cta_vessel_comparison.py \
  --root "$RESSECTIONLAB_DATA_ROOT" --subject 014 --subject 018 --register "$PWD/testcases/cases"
subject=014
case_id="TopCoW-$subject-CTA-VESSEL-COMPARISON-v1"
archive="$RESSECTIONLAB_DATA_ROOT/prepared/annotated-cta-vascular-comparison-v1/$case_id/$case_id.rsworkspace"
SURFACE_VERIFY_VASCULAR_COMPARISON=1 \
  SURFACE_VERIFY_OUTPUT="$RESSECTIONLAB_DATA_ROOT/qc/annotated-cta-vessels-$subject-reproduction" \
  desktop/node_modules/.bin/electron desktop/electron/verify-surface-quality.cjs \
  --anatomy "$archive" --automation-hidden
```

### Combined supplied CTA arterial coverage

When a CTA workspace includes a compact arterial view and a complementary wider supplied annotation on the same acquisition, prepare a separate workspace for both views. Pass the native layer IDs from the manifest. The focus list follows the authored opening, and each selected layer must match its supplied positive labels on the exact CTA grid. Either source alone can omit positives contained in the other. The combined display retains all native positive cells from both and leaves the input workspace intact.

```sh
PYTHONPATH=src .venv/bin/python scripts/prepare_cta_arterial_coverage.py \
  "$manifest" "$new_workspace_directory" --series "$cta_series_id" \
  --focus "$focused_layer_id" --wider "$wider_layer_id"
```

Open the new `patient-anatomy.json` in RessectionLab. The two arterial buttons reuse the current camera and linked scan. The combined layer is display-only. For branch tracing or source-specific measurements select one of the original input layers. The archived coverage receipt records both input counts and the exact union.

### Meningioma sub-13 native CT and T1 review

Open **Meningioma-sub-13-CT-T1-EVIDENCE-v2** in Cases. The [paired release](https://zenodo.org/records/17486320)
provides sharp and soft CT reconstructions and contrast T1 on a shared
414 × 414 × 302 publisher grid at 0.5 mm. The grid spacing describes sampling.
The source does not establish the effective acquisition resolution.

Fourteen anatomical transects evaluated the frozen registration candidates.
The signed edge profiles report the displacement along each transect. They
cover five vault landmarks, three ventricular landmarks and six skull-base
landmarks. The soft CT identity map has median absolute errors of 0.49 mm
at the vault, 0.79 mm at ventricular boundaries and 2.34 mm at the skull base.
The retained sharp CT map has 2.31 mm median skull-base error and 4.07 mm
maximum measured discrepancy. The receipts flag search-window boundary
peaks and retain the source-specific profiles. Differences in CT and T1
tissue contrast also affect these edges. The measurements do not establish
3D landmark error or qualify a shared frame.

The companion opens the modalities independently. T1 shows the supplied image
beside the cached 1 mm SynthSeg cortical labels. Sharp CT shows gross
mineralized skull-base anatomy and the retained density-derived bone layer.
The bone layer exactly matches the sharp CT threshold at 300 HU. The largest
26-connected component contains 98.0% of its 2,731,406 voxels. The remaining
components include disconnected structures and density fragments. Over 99.5%
of the cortical label voxels sample within the separate brain estimate.
These checks support inspection of the cortical pattern and visible bone.
Thin foraminal margins, tissue thickness, and precise skull–brain contact
remain unverified. Both retained tumor outputs are empty. The experimental
T1 vascular map has no independent angiographic annotation or completeness
evidence. The subject has no measured dural label.

The numeric report and annotated image pairs are in
`prepared/meningioma-sub13-anatomical-review-v2/`. Hidden desktop receipts
are in `qc/hidden-app/meningioma-sub13-independent-{t1-cortex,ct-skull-base}-v1/`.
The paired views hold the camera, cursor and source signal fixed within each
modality and show only layers from the active scan.

### Five-layer CTA inspection cases

Search **FIVE-LAYER** in Cases for 18 portable studies from six acquired
subjects. All five anatomy groups are enabled when each case opens. Each
combines an acquired CTA or CT reference, retained brain and skull estimates, source-linked
vascular annotations, a CT-derived inner-cranial dural proxy, and a retained
simulated tumor contour. The measured dural anatomy and patient tumor
status remain unknown. The proxies use the existing 1.5 mm enclosed-ring
algorithm and preserve open and acquisition-limited regions as unknown.
The case receipts report the fraction of tumor voxels at supported proxy
slice levels. This measures axial overlap, with no inference about local
dural thickness, contact, vessel absence, or surgical access.

The stronger TopCoW set includes supplied arteries and veins or sinuses.
ISLES 0146 has a restricted publisher-provided automated CoW estimate. The
TopCoW 004 companion retains the local skull estimate. TopCoW 018 uses its
previously qualified wider density estimate. The TAIR 00273 enhancement component is too restricted for the five-layer
collection. The EDEN MRI-based cranial estimates and registered angiographic
surfaces require separate evidence review before comparable admission.

```sh
.venv/bin/python scripts/review_meningioma_sub13_alignment.py \
  --root "$RESSECTIONLAB_DATA_ROOT" --register "$PWD/testcases/cases"
.venv/bin/python scripts/prepare_five_layer_cta_cases.py \
  --root "$RESSECTIONLAB_DATA_ROOT" --scratch "$PWD/testcases/outputs" --register "$PWD/testcases/cases"
```

The collection index is `prepared/five-layer-cta-inspection-v1/collection.json`.
Each archive was opened into a fresh native-study directory and verified
against its original tumor contour. The TopCoW 014 representative also has
matched hidden desktop captures in `qc/hidden-app/five-layer-cta-topcow014-v1/`.
These studies carry inspection permissions.

### CADS superior-vault surface verification

The superior-vault refinement preserves the raw mesh, its triangle indices,
the CT, and the labels. A physical-coordinate low-pass tapers in from native
slice index 116 to 136. All 681,110 lower vertices at or below 116 retain
their stock positions. The geometry audit found zero newly intersecting
triangle pairs after returning 140 vertices to their stock positions. Maximum
motion from stock is 0.505 mm. Across the measured superior vault, squared
normal turn drops 62.9%. The median CT contrast between samples ±0.5 mm from
matched mesh points changes from 275.02 to 274.91 CT units. Lower skull,
brain and vessel estimates retain their earlier limitations.

Matched hidden desktop runs compare the same scan, camera, slice pixels,
source surface and indices. The new vault changes only the conditioned 3D
render. A portable Save carries the attachment and its source receipt. The
fresh desktop process restores the same display geometry and 3D capture.
The numeric report and matched upper-vault crops are in
`qc/cads465-vault-conditioning-v1/`. The desktop
reports are in `qc/hidden-app/cads465-conditioned-vault-warm-v2/` and
`qc/hidden-app/cads465-conditioned-vault-cold-v1/`.

To repeat the refinement, run the probe and stage the selected CADS snapshot
under `qc/`. For a locally refined study, use the immutable `stock_manifest`
from its receipt as the staged manifest. The surface verifier compares the
staged candidate against the stock desktop report.

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/probe_cads_vault_conditioning.py \
  --output-name cads465-vault-reproduction
trial="$RESSECTIONLAB_DATA_ROOT/qc/cads465-staged-study"
mkdir "$trial"
rsync -a --exclude='._*' "$(dirname "$study")/" "$trial/"
stock_manifest=$(jq -r '(.stock_manifest // .manifest).path' "$(dirname "$study")/receipt.json")
cp "$RESSECTIONLAB_DATA_ROOT/$stock_manifest" "$trial/patient-anatomy.json"
.venv/bin/python scripts/install_cads_vault_conditioning.py \
  --study qc/cads465-staged-study --probe qc/cads465-vault-reproduction/report.json
stock="$RESSECTIONLAB_DATA_ROOT/qc/hidden-app/cads-brainct-465-surfaces-20261118-v1/report.json"
warm="$RESSECTIONLAB_DATA_ROOT/qc/cads465-vault-app-reproduction"
SURFACE_VERIFY_NATIVE_PRESENTATION=1 SURFACE_VERIFY_REFINED_LAYER=head-bone \
  SURFACE_VERIFY_EXPECTED="$stock" SURFACE_VERIFY_SAVE_REFINED=1 SURFACE_VERIFY_OUTPUT="$warm" \
  desktop/node_modules/.bin/electron desktop/electron/verify-surface-quality.cjs \
  --anatomy "$trial/patient-anatomy.json" --automation-hidden
SURFACE_VERIFY_NATIVE_PRESENTATION=1 SURFACE_VERIFY_REFINED_LAYER=head-bone \
  SURFACE_VERIFY_EXPECTED="$stock" SURFACE_VERIFY_REFINED_PRIOR="$warm/report.json" \
  SURFACE_VERIFY_OUTPUT="$RESSECTIONLAB_DATA_ROOT/qc/cads465-vault-cold-reproduction" \
  desktop/node_modules/.bin/electron desktop/electron/verify-surface-quality.cjs \
  --anatomy "$warm/conditioned-vault.rsworkspace" --automation-hidden
```

The live Cases entry uses the refined revision. Its post-install desktop check
is in `qc/hidden-app/cads465-conditioned-vault-cases-v1/`.

### Higher-detail CT and MRI anatomy

The retained Meningioma sub-06 study preserves sharp-kernel CT, soft-kernel CT
and contrast-enhanced T1 MRI on the same publisher-supplied 439 × 439 × 361
grid at 0.5 mm. Scanner resolution remains unspecified. The existing CT
density surface shows a more continuous mineralized vault than the CTA pilot
estimates. The MRI separately supplies cortical, cerebellar, brainstem and
ventricular estimates. Existing desktop source-switch captures and their
source identities are in `qc/engine-operative-integration-20261118/multimodal/`.

A checksum-bound audit sampled both CT images against retained MRI masks at
1 mm. On soft-kernel CT, 8.72% of the sampled brain-envelope locations reached
300 CT units. The retained sharp-kernel alignment gave 7.42%. In both, more
than 96% of those dense-signal samples lie within 4 mm of the extraction
boundary. The selected SynthSeg tissue regions show 2.83% and 1.82%,
respectively. The masks describe different tissue supports. These results
localize the conflict and leave its registration and segmentation causes
unresolved. The combined frame remains unreviewed, with operative use of the
CT-derived bone withheld. Reuse follows the retained private DEVELOPMENT
scope and frozen CC BY-NC 4.0 receipt.

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python \
  scripts/audit_meningioma_skull_brain.py --output-name meningioma-sub06-conflict-reproduction.json
```

The measured source, transform and mask bindings are in
`qc/meningioma-sub06-skull-brain-conflict-v1.json` on the HDD.

### MRI anatomy and diffusion inspection

Search **IXI265-Guys-0845** in the checkout's Cases library or open
`prepared/adaptive-anatomy-migration-20261117-v1/IXI265-multimodal/patient-anatomy.json`
directly. The local entry reuses the checksum-bound migration study and cached
1 mm T1-derived SynthSeg labels. The acquired T1 has a 256 × 256 × 150 grid at
0.938 × 0.938 × 1.20 mm. The study also retains the acquired MRA and diffusion
b0 reference, a partial MRA vessel annotation, and an explicitly derived FA
map. The MRA and diffusion-to-T1 transforms are unreviewed. The diffusion
gradient frame remains unresolved, and diffusion is limited to inspection.
The retained view opens on FA. Select **T1** under Scans to inspect the
estimated cerebral cortex, white matter, ventricles, brainstem and cerebellum.
The hidden desktop proof in `qc/hidden-app/ixi265-multimodal-20261118-v3/`
records conditioned surfaces for 11 selected T1 anatomy labels and changes
in all three linked source panes. It also compares the acquired MRA and b0
and the derived FA against exact separate-grid readouts. The captures can
show the viewer's estimated alignment preview. The report records that mode
separately from the independent source checks.

```sh
study="$RESSECTIONLAB_DATA_ROOT/prepared/adaptive-anatomy-migration-20261117-v1/IXI265-multimodal/patient-anatomy.json"
DATA_REBUILD_INSPECT_IXI=1 DATA_REBUILD_CASE="$study" \
  DATA_REBUILD_OUTPUT="$RESSECTIONLAB_DATA_ROOT/qc/ixi265-app-reproduction" \
  desktop/node_modules/.bin/electron desktop/electron/verify-data-rebuild.cjs --automation-hidden
```

The **IXI-077-T1-MRA-ORGANIC-ROUTE-01-v1** card is a separate DEVELOPMENT
scenario on existing acquired T1 and MRA. It preserves the source brain estimate
and the reviewed partial vascular overlay. The EDEN geometry is conditioned
on the endpoint of a previously traced stock-tool line. The new simulated
contour contains 282 cells (297.42 mm³) and differs from the old spherical
fixture by 103 cells. The exact core, rim and edema labels are kept outside
the app case.

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python \
  scripts/build_ixi077_organic_rehearsal.py --register "$PWD/testcases/cases"
```

See `prepared/ixi077-organic-rehearsal-v1/` for the pristine package and static
receipt. The stock working envelope intersects 96 of 282 contour cells and no
mapped vessel cells. The geometry assumes prior scalp, local skull and dural
exposure. Unmapped vasculature and unknown tissue remain outside the evidence.

Search **IXI-077-T1-MRA-ORGANIC-ROUTE-01-v1-RECORDED** for the separate saved
simulation. Stock CUSA reached 87.7% of the fixed line. At the first accepted
apply boundary with a contour effect, the modeled removal totaled 27 of 282
visible contour cells (28.48 mm³) and 314 non-target brain cells (331.17 mm³).
The next action withdrew on the same line with no additional removal. The
source remained unchanged, and all ten actions, effects and final tool states
matched after cold backend reopen. The result covers conditional partial
rehearsal. Complete treatment, intact-head access, automatic planning,
vascular clearance beyond the partial overlay and clinical validity remain
unqualified.

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python \
  scripts/rehearse_ixi077_organic.py --output-name ixi077-organic-replay-reproduction
```

The committed actions and preserved workspace are in
`qc/ixi077-organic-conditional-replay-v1/` on the HDD.
The hidden desktop verified the withdrawn and inserted target-effect boundaries
with completed GPU frames. Play and Pause displayed a recorded frame at 0.2 s.
Scrubbing to a distinct recorded frame at 29.8 s of the seventh 30 s apply
showed the 28.48 mm³ target effect before the action boundary. The reader
then returned to the exact withdrawn state. The report and three captures are
in `qc/hidden-app/ixi077-organic-replay-20261118-v2/`.

```sh
trial="$RESSECTIONLAB_DATA_ROOT/qc/ixi077-organic-conditional-replay-v1"
DATA_REBUILD_CASE="$trial/replay.ressectionlab" \
  DATA_REBUILD_OUTPUT="$RESSECTIONLAB_DATA_ROOT/qc/ixi077-app-reproduction" \
  DATA_REBUILD_REPLAY_ONLY=1 DATA_REBUILD_REPLAY_ACTION=8 \
  DATA_REBUILD_REPLAY_START_STATE_ID="$(jq -r '.[8].before_state_id' "$trial/actions.json")" \
  DATA_REBUILD_EXPECT_OPERATING_STATE_ID="$(jq -r '.target_effect_state_id' "$trial/report.json")" \
  DATA_REBUILD_EXPECT_STATE_ID="$(jq -r '.final_state_id' "$trial/report.json")" \
  DATA_REBUILD_EXPECT_REPLAY_ACTIONS="$(jq -r '.committed_action_count' "$trial/report.json")" \
  desktop/node_modules/.bin/electron desktop/electron/verify-data-rebuild.cjs --automation-hidden
```

### Observed surgical change

```sh
.venv/bin/python scripts/benchmark_resect_surgical_change.py \
  --output-name RESECT-Case3-benchmark-reproduction
```

`qc/surgical-change/RESECT-Case3-benchmark-verified-20261118/README.md` indexes
the checksum-bound evidence. The same-patient intraoperative cavity is an
observed display layer. Its 20,005 voxels and 160.04 mm³ survived workspace
save and cold reopen at Dice 1.000. The verified 28,518-byte during mask is
recorded in `qc/surgical-change/RESECT-Case3-benchmark-20261118/cavity-source-mirror-attempt.json`.
Case4's frozen update reproduced held-out
ultrasound landmark RMS of 4.045 mm unchanged, 1.020 mm after rigid correction,
and 1.104 mm with inverse-distance weighting. Case3 image-only registration
increased landmark error from 2.029 to 2.087 mm RMS. No frame-qualified predicted
cavity was available for a simulated-versus-observed removal comparison.

The Case3 two-stage extension recovers 46,148 bytes of checksum-pinned
publisher data, comprising the after-resection cavity mask and two short
landmark files. It reuses the acquired US scans. The original T1 remains the
primary reference in **RESECT-Case3-observed-cavity-stages-v1**. The two
ultrasound annotations open on their separate native grids as display-only
observations. Both masks restore exactly. FLAIR and before-resection US remain
available in the original source-stage workspace.

The visible cavity measured 160.04 mm³ during resection and 2,044.01 mm³
afterward. The labels represent the cavity visible at each ultrasound
acquisition. Among the two files' 11 before-US landmarks, ten match exactly.
The remaining coordinates differ by 0.408 mm and their destinations were
excluded from fitting and scoring. Across the five matched points withheld
from the fit, RMS error was 3.050 mm for the unchanged before positions,
1.905 mm for the during positions carried forward, and 1.391 mm with a rigid
update conditioned on the other five correspondences. The two cross-stage
cavity overlap scores use separate acquired-signal supports and describe
visible anatomy under each alignment hypothesis. No causal removal or dense
tissue deformation is inferred.

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python \
  scripts/benchmark_resect_case3_stages.py --output-name RESECT-Case3-observed-stages-reproduction
```

The workspace, source overlay, frozen held-out predictions and report are in
`qc/surgical-change/RESECT-Case3-observed-stages-20261118-v1/`.
The hidden desktop checks in `qc/hidden-app/resect-case3-native-stage-pair-20261118-v1/`
bind each selected ultrasound to its native source and cavity checksum. Showing
each mask changes all three linked ultrasound panes. Returning to the original
MRI restores its exact scalar identity, cursor and slice pixels.

```sh
stages="$RESSECTIONLAB_DATA_ROOT/qc/surgical-change/RESECT-Case3-observed-stages-20261118-v1"
DATA_REBUILD_CASE="$stages/RESECT-Case3-observed-cavity-stages-v1.ressectionlab" \
  DATA_REBUILD_RESECT_PAIR_REPORT="$stages/report.json" \
  DATA_REBUILD_OUTPUT="$RESSECTIONLAB_DATA_ROOT/qc/resect-case3-app-reproduction" \
  desktop/node_modules/.bin/electron desktop/electron/verify-data-rebuild.cjs --automation-hidden
```

### Case3 target and after-resection cavity

The [RESECT image and landmark release](https://doi.org/10.11582/2017.00004)
and [RESECT-SEG annotations](https://doi.org/10.17605/OSF.IO/JV8BK) supply
same-person preoperative FLAIR, before-resection ultrasound and after-resection
ultrasound observations. This comparison recovered the FLAIR and before-US
tumor masks and two MRI–US landmark files. It reused the three images, the
before–after landmark file and the after-US cavity mask. The source report
records the exact member versions and checksums. The MRI points use the FLAIR
world frame described in the [source methods](https://doi.org/10.1002/mp.12268).
Case3 retains its TRAIN role.

The script fits each proper rigid transform on odd landmark ranks in
lexicographic source RAS order and scores only even ranks. It writes the
transforms and held predictions before opening the label volumes. The overlap
scores use the same support for both hypotheses, with positive ultrasound
signal as a visibility proxy. A separate sparse pullback reproduces the rigid
intersection counts.

| Transport and observed label | Held landmarks | Landmark RMS, identity → rigid | Dice on fixed support, identity → rigid |
| --- | ---: | ---: | ---: |
| FLAIR tumor → before-US tumor | 7 | 9.620 → 0.991 mm | 0.00013 → 0.557 |
| FLAIR tumor → after-US visible cavity | 7 | 7.410 → 2.146 mm | 0.053 → 0.363 |
| Before-US tumor → after-US visible cavity | 5 | 3.012 → 0.843 mm | 0.312 → 0.288 |

The native annotation volumes are 1,322 mm³ for FLAIR tumor, 614 mm³ for
before-US tumor and 2,044 mm³ for the observed cavity. The nearest withheld
point to each source tumor centroid is 9.5–12.0 mm away. The opposite movement
of landmark error and Dice in the ultrasound pair leaves local cavity
agreement uncertain. These comparisons quantify image registration and visible
label overlap. Residual tumor, extent of resection, and dense tissue motion
remain unmeasured. No saved simulator output for this patient has an admitted
frame and matching cavity observation.

A CPU replay of the publisher-configured Raidionics LGG model reproduced the
saved FLAIR score volume byte for byte. All 1,322 publisher tumor voxel centers
lay inside the brain mask, retained nonzero signal after 0.75 mm RAS resampling
and reached all eight inference patches. The highest score inside the tumor was
0.161. The whole-volume peak of 0.213 lay 18.0 mm from the nearest annotated
voxel center. At the published 0.5 cutoff the output remained empty (Dice 0).
The preparation result and layer inspector report the cutoff and highest score.
Importing the publisher label on its own FLAIR restored the 1.32 mL annotation
as a separate layer. T1-frame planning still requires review of FLAIR alignment.
The ultrasound cavity and surgical-change comparisons remain separate observations.
The personal HDD review guide at `qc/patient-strategy-comparison-20261119/resect/README.md`
links the recovered study, patch-level receipt, score overlays and desktop captures.

```sh
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python \
  scripts/benchmark_resect_case3_target_cavity.py --output-name RESECT-Case3-target-cavity-reproduction
study="$RESSECTIONLAB_DATA_ROOT/qc/surgical-change/RESECT-Case3-observed-target-cavity-20261119-v1"
PYTHONPATH=src .venv/bin/python scripts/benchmark_resect_case3_target_cavity.py \
  --output-name RESECT-Case3-target-cavity-reproduction \
  --saved-localization-run "$study/historical-localization"
```

The current study directory contains both Cases workspaces, the source-linked
report and the overlay figure. The unchanged T1 reference opens with the
annotated before/after ultrasound pair or with the annotated FLAIR. Splitting
the previews keeps the complete native grids within the desktop display budget.
All three masks are supplied inspection layers in independent frames. Both
hidden desktop proofs recorded pixel changes in all three slice panes for each
mask and exact primary T1 pixel restoration.

```sh
DATA_REBUILD_RESECT_PAIR_REPORT="$study/report.json" \
  DATA_REBUILD_CASE="$study/RESECT-Case3-observed-target-cavity-v1.ressectionlab" \
  DATA_REBUILD_OUTPUT="$RESSECTIONLAB_DATA_ROOT/qc/resect-case3-us-target-cavity-app-reproduction" \
  desktop/node_modules/.bin/electron desktop/electron/verify-data-rebuild.cjs --automation-hidden
DATA_REBUILD_RESECT_PAIR_REPORT="$study/report.json" \
  DATA_REBUILD_CASE="$study/RESECT-Case3-observed-flair-target-v1.ressectionlab" \
  DATA_REBUILD_OUTPUT="$RESSECTIONLAB_DATA_ROOT/qc/resect-case3-flair-target-app-reproduction" \
  desktop/node_modules/.bin/electron desktop/electron/verify-data-rebuild.cjs --automation-hidden
```

## Native study migration

Use the private frozen inventory in the configured `testcases/` collection.
These commands display dispositions, migrate eligible studies serially, and
report the resulting counts. The data root must identify the mounted, marked HDD.

```sh
export RESSECTIONLAB_DATA_ROOT="/Volumes/shafrir/old coding projects/RessectionLabData"
export RESSECTIONLAB_MODEL_ROOT="$RESSECTIONLAB_DATA_ROOT/cache/models"
export RESSECTIONLAB_POC_OUTPUT_ROOT="$RESSECTIONLAB_DATA_ROOT/cache/poc-model-results"
inventory=testcases/adaptive-anatomy-inventory-20261117-v1.json
destination="$RESSECTIONLAB_DATA_ROOT/prepared/adaptive-anatomy-migration-20261117-v1"
PYTHONPATH=src .venv/bin/python scripts/migrate_anatomy_library.py "$inventory"
PYTHONPATH=src .venv/bin/python scripts/migrate_anatomy_library.py "$inventory" --run "$destination"
PYTHONPATH=src .venv/bin/python scripts/migrate_anatomy_library.py "$inventory" --results "$destination" \
  --ledger-output testcases/adaptive-anatomy-migration-20261117-v1-results.json
```

Repeat the run command to resume. Use `--only CASE_KEY` for a representative
and `--retry-failed` to retry failed stages. Open the resulting
`patient-anatomy.json` in Cases. Migration writes a private versioned study.
After a verified SynthStrip MPS allocation failure, the same retry command
offers one CPU attempt with a ten-minute timeout and an 8 GiB process-memory
limit. Its checkpoint retains the original GPU failure and records the device.
Catalog publication requires a separately admitted native release. EDEN
healthy acquisitions and synthetic target options share one study per person.
The joined results include every frozen row, its saved outcome, and any stage
findings. The report links reasons to their exact inventory or receipt entries
and records both SHA-256 digests. Running, cancelled, and unprocessed rows keep
the report pending. A complete report can include failures and protected or
retained studies. `--only` filters the printed counts. The joined export always
includes the full inventory and can be regenerated from the same frozen inputs.
When later source review changes an alignment, inspect the affected row's
`migration.postmigration_review`. It identifies the reviewed working archive,
the original snapshot it came from, and the checksummed decision. Open the
linked working archive to see the corrected registration and saved comparison.

For a hand-authored native study, use a plan with schema
`resectionlab.local-standardization-plan.v2`. Its `cases` contain `id`,
`patient_id`, `collection`, and a list of `series` with their source paths.
Optional `layers` declare the source series, label value and its meaning.
Optional `prepare_options` include `indication`, `full_head_confirmed`,
and `include_dural_proxy`. Run the plan from the checkout.

```sh
PYTHONPATH=src .venv/bin/python scripts/standardize_case_library.py testcases/plan.json
```

Repeat the command to resume. A new plan filename targeting the same collection
adds scans and preserves the earlier readiness receipt. Version 1 retains its
explicit simulator-export workflow.

### Resume a native preparation batch

Run the same native plan again after SIGINT or SIGTERM. Both signals ask
the script to checkpoint and stop with exit status 130. Each completed stage
is committed independently. A completed CT attempt writes its zero-exit
checkpoint before the native masks are read, so an interruption at that point
can resume at validation. Later runs verify cached artifacts and report
missing inputs, empty outputs, and model failures per study. Use
`--retry-failed` when you intend to repeat failed model stages.

The worker supervisor cleans up its process group after exit and drains the
remaining log for a bounded time. A descendant retaining stdout is recorded
in the log. CT worker completion and native mask checks determine whether
that finished attempt can be admitted.

The ten-study local batch uses the frozen
`qc/anatomy-preparation-unattended-v1/batch.json` with copies under
`prepared/anatomy-preparation-unattended-v1/`. All ten studies finished with
documented limits. Eight gained 24 available layers beyond the originals.
ISLES 0146 and TAIR 00273 recovered both historical CT masks without a new
worker for either source. Eight fresh workers finished despite an inherited
log pipe. The recorded SIGTERM preserved the first completed CT stage, and
both repeat passes published identical study snapshots without launching new
workers. The per-case `results.json` details the available layers, missing
inputs, empty outputs and registration status.


The collection supports source inspection and explicitly scoped surgical-research
controls. Acquired observations, supplied annotations, algorithmic estimates and
synthetic components remain separate. Synthetic variants add no real people.
Independent review is AI engineering/source-data review, not clinical or
neuroradiologist validation. No RL training, medical-data upload or remote
publication belongs to this cycle.

## Open accepted cases

The latest frozen compatibility release is `2026-11-05-integrated-rehearsal-v2`,
with 368 packages from 130 source-person families, including 208 synthetic
variants. Its CTA cases have bounded conditional instrument workflows and the
healthy MRI/MRA cases have reviewed approximate overlay inspection. See the
[integrated-suite instructions](data-integrated-rehearsal.md) and the HDD
`integrated-v1/START-HERE.md` for capabilities, reproduction commands,
independent decisions and app captures.

The original frozen `2026-10-31-data-foundation-v1` release remains unchanged:
355 native packages, 130 source person families and 200 synthetic variants.
Its 11-case core, final report and retrospective coverage matrix remain in
`catalog/START-HERE.md` and `catalog/reporting/`. The current accepted index
preserves all 355 original rows and receipts.

Set `RESSECTIONLAB_DATA_ROOT` to your actual mounted data directory, then launch
from the existing primary checkout:

```sh
export RESSECTIONLAB_DATA_ROOT="/absolute/path/on/mounted/RessectionLabData"
./scripts/launch_poc.sh
```

Open **Cases**; reviewed external entries appear automatically. Committed app defaults
contain no machine-specific bulk-data location. The current checkout app checks
the mounted volume's identity through fresh native Disk Arbitration properties and the live filesystem
UUID on every call, without cached leases, and reports an unavailable drive
rather than treating missing anatomy as a valid empty case. Preparation reserves the larger of 50 GiB and 10% of the
volume. Twelve recorded native guard probes had a 6.6 ms median
(`qc/native-volume-guard-probe.json`); this replaces diskutil probing that stalled
under ExFAT writes. These app proofs cover the checkout app, not a packaged build;
a packaged engine without the checkout helper retains a bounded diskutil fallback.
Source archives, extraction/download partials, model caches, derivatives,
native projects, review images and archived evidence stay on the data drive.

`catalog/reporting/coverage-<stage>.json` and its compact Markdown companion summarize actual saved synthetic preparation metadata retrospectively. They separate generated variants from source people and bind usable counts to the published release. Recipe margins, radii and volumes do not establish clinical depth, vessel clearance, corridor safety, skull constraints or functional anatomy.

`catalog/release.json` is the current accepted index. Each entry binds an exact
package SHA256 to a checksummed `qc/case-decisions/` sidecar containing allowed
capabilities and limitations. `staging/candidates/` and `quarantine/` are not
accepted catalogs. Native `metadata.data_case` preparation status is historical;
it cannot override the linked independent decision. Published packages use the
existing self-contained `.ressectionlab` format and production save/reopen reader.

Read `catalog/reporting/report-preview.md` while work continues; use
`report-final.md` only after the lead has published the final snapshot. The JSON
companions preserve exact counts, source-family aliases, component provenance,
resource evidence and current pending work. Synthetic controls, healthy acquired
hosts, clinical CTA/MRA patients and observed surgical-change cases have distinct
classifications. Clinical vascular patients are not healthy controls. Source
person-family counts do not prove independence across unrelated cohorts or
pretrained-model exposure.

## Resume preparation and review

Run serially; do not overlap memory-heavy registrations, inference or case builds.
Commands below reuse verified source bytes and completed receipts:

```sh
# Current healthy-host plan, one unfinished host at a time.
.venv/bin/python scripts/data_rebuild_batch.py --limit 1

# Fifty prospectively frozen UCSF TRAIN sources; independent pilot gate required.
.venv/bin/python scripts/data_rebuild_real.py batch --count 50 \
  --review-receipt "$RESSECTIONLAB_DATA_ROOT/qc/independent/UCSF-PDGM-0009-structural-scale-review-v1.json"

# Selected paired clinical CTA/MRA people, preserving frozen roles and pilot gate.
.venv/bin/python scripts/data_rebuild_topbrain.py batch --count 25 \
  --review-receipt "$RESSECTIONLAB_DATA_ROOT/qc/independent/TopBrain001-native-scale-review-v1.json"

# One already-exposed RESECT TRAIN family: MRI and before/during/after US.
.venv/bin/python scripts/data_rebuild_surgical_change.py

# Automated engineering checks and representative hidden app save/reopen checks.
.venv/bin/python scripts/data_rebuild_proxy_units.py
.venv/bin/python scripts/data_rebuild_qc.py
.venv/bin/python scripts/data_rebuild_verify_suite.py --expansion

# Read-only preview; no source-array decoding or bulk rehashing.
.venv/bin/python scripts/data_rebuild_report.py --stage preview
```

The one-cycle inventory, real-source, surgical-change, reporting and retirement
helpers additionally retain this run's verified volume UUID and canonical-root
bindings. They deliberately refuse an absent/replacement volume; moving the
corpus requires reviewing those bindings and restoration manifests, not changing
portable app defaults or inventing an internal fallback directory.

The resumable ledger is `catalog/work-ledger.json`. Source rights/access receipts
are in `catalog/source-research.json`; cohort assignments remain in repository
manifests. All modalities, timepoints, annotations and synthetic descendants
inherit their person's frozen TRAIN/SELECT/EVAL role and exposure history.
UPENN remains reserved external evaluation. Known TopBrain aliases identify
legacy clinical person 001 as existing DEVELOPMENT: 25 clinical families therefore
represent 24 newly materialized families and 1 previously exposed recovery.
RESECT Case3 is another already-exposed TRAIN recovery, not a never-seen person.

Original healthy `plan-v5.json` remains frozen. Its explicit
`plan-v5-resource-extension.json` preserves the original plan hash/roles and failed
IXI051 source while predeclaring TRAIN replacement 084 for the generation slot.
IXI051 hit the 6 GiB MPS allocator cap; its CPU model subprocess later received
SIGKILL 9 after about 289 seconds with `no_saved_masks`. An 8 GiB supervisor-limit cause
was not established. Lausanne197's 50 million voxel deformation was withheld before
allocation because the conservative additional-array estimate exceeded the 4 GiB
budget. The qualified mass-field scope is 016/021/022/091; never infer five completed
fields from five original requests. Final usable counts come from publication.

Automated QC is not independent acceptance. Inspect actual source planes,
transforms, masks, provenance and app behavior; failed cases must be repaired,
restricted, quarantined or rejected. `qc/independent/release-decisions-v1.json`
records the independent decisions. Local catalog publication uses:

```sh
.venv/bin/python scripts/data_rebuild_release.py
```

Only the lead freezes a reviewed release. Reporting uses checksum-linked case
decisions, never native preparation flags or a directory's package count.

## Interpret the data correctly

Source MRI can be skull-stripped, cropped or registered upstream. A displayed
brain/skull/vessel mask may be an estimate; missing function, veins or outside-FOV
anatomy is unknown rather than safe. No verified accepted person combines acquired
tumor MRI, useful skull CT and dedicated angiography in a compatible surgical
state: the exact-match count remains zero. Synthetic appearance/deformation is a
controlled engineering model, not biological glioma realism or outcome evidence.

TopBrain's source terms are noncommercial; retain its original attribution and
restrictions. IXI share-alike and other source-specific terms are recorded in the
source registry and case sidecars. Do not bundle restricted source data into a
commercial deliverable or infer image rights from a software/paper license.

Large MRA and acquired US sources default to native MPR with 3D raycasting disabled
where required by the observed Metal failure. This display policy preserves the
original NPY payloads. Operative US stages have explicit captions. Lausanne's typed
adapter distinguishes NIfTI observations from JSON metadata; oversized auxiliary
views use an explicitly labelled 1 mm display derivative while full native primary
and originals remain available. It is display LOD, not anatomical registration.

Historical internal placement proxies with unknown spatial units retain their
original bytes. Header-only `v2-mm` companions declare mm without changing masks,
affines or native anatomy. Their independent byte proofs are in
`qc/independent/placement-proxy-unit-repairs-independent-v3.json`; current case
sidecars link the importable companions. Never silently reinterpret a v1 helper
as an acquired image or a validated tissue segmentation.

## Inventory and reversible restoration

The initial local `testcases/` plus `data/` inventory recorded 82,899 entries,
41,147,174,770 logical regular bytes and 27,983,089,664 lstat allocated bytes.
78,129 files were checksummed; 70 protected paths were skipped. Existing shared
symlinks, provenance and unique malformed/failure controls remain recorded under
`catalog/inventory/`. AppleDouble `._*` helpers are metadata, not images or cases.

Only the closed `testcases/performance/rehearsal-2026-10-19` campaign was retired.
Its complete PAX tar was verified, a real native workspace reopened directly from
it, and all remaining native/JSON/code/index/symlink references and live open files
were checked before removal. 1,167,580,626 logical bytes and 1,170,530,304 lstat allocated
bytes were removed; immediate system disk-free increased 1,171,329,024 bytes. APFS
sharing, snapshots and concurrent work limit interpretation of physical savings.
Active patient sources, cases, models and history were preserved. No Git history
or object shrinking is claimed.

The exact archive SHA, source hashes, scope and restore command are in
`catalog/inventory/migration/rehearsal-2026-10-19-migration-receipt.json`. The local
pointer is `testcases/performance/ARCHIVED-EVIDENCE.md`. Restore to an internal
filesystem while the original campaign directory is absent:

```sh
.venv/bin/python scripts/data_rebuild_migrate.py restore \
  --plan "$RESSECTIONLAB_DATA_ROOT/catalog/inventory/migration/rehearsal-2026-10-19-plan.json" \
  --destination "$PWD/testcases/performance"
```

Restoration refuses overwrites and verifies the plan/archive SHA, file bytes and
link targets. If a new benchmark recreated that output directory, restore into a
different ignored internal parent. Do not extract symlink archives onto ExFAT.
Disposable generated fixtures verified restoration, relative links, native reading
and corruption/UUID/scope guards; no other broad cleanup is implied.
