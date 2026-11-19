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
The checkout lists 22 new CTA development entries in Cases, including a separate
publisher-geometry companion for TopCoW 006. The three acquisition
sets retain 354,652,995 bytes of publisher payloads. At the final check the
scoped source files, studies, scenarios, inference caches, inventory and QA
used 5.80 GB of allocated HDD file space before the publisher-geometry companion.

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
# Recover only the retained outputs of the specific zero-exit, log-pipe case.
receipt="$RESSECTIONLAB_POC_OUTPUT_ROOT/anatomy-410401c34a17e131/totalsegmentator-skull/e0bedd63e0d8502d8985/inference.json"
.venv/bin/python scripts/prepare_tair_cta.py --recover-ct-worker "$receipt" 00273
```

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
was qualified separately for CTA006. The 915 graph points use the original CTA
RAS frame and all 914 edge midpoints overlap the original CoW annotation. The
surface contains 47,202 vertices and 94,404 triangles. The proof used the
published VTP files directly in VTK and retained the original mask, license and
hashes. See `qc/multimodal-anatomy-20261118-cow-vtp-v1/README.md`.

Search **PUBLISHED-COW** to open the CTA006 native companion. The
**Circle of Willis · publisher mesh** layer uses the original triangular
surface, while the exact original CoW segmentation supplies linked CTA slice
overlays. The wider TopBrain arterial and venous unions remain separate.
The same-person TopBrain MRA and its V1 arterial union open on their own source
grid. The study keeps both VTP originals, publisher metadata and licenses.
The centerline graph and its attributes remain available as source evidence.

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python \
  scripts/build_cta_published_cow_study.py --register "$PWD/testcases/cases"
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
The hidden desktop verified all 12 Cases entries and checked CTA014 variant 01
against its original CTA identity. Its scan and generated contour restored in a
fresh process, and rehearsal inspect preview left the modeled state intact.
See `qc/hidden-app/cta-organic-014-warm-20261118-v2/report.json` and
`qc/hidden-app/cta-organic-014-cold-20261118-v2/report.json`. The separate
reviewed CTA014 replay restored a 48-state run with 15.81 mm³ of target effect
and withdrawal. Its report is `qc/hidden-app/cta-pilot-baseline-20261118-v1/report.json`.

The three **NATIVE-SHOWCASE** entries use the native conditioned surfaces.
Each shows the matched CTA, broader mineralized skull estimate, brain envelope,
separate publisher artery and vein/sinus unions, and its existing variant 01
contour. Fifteen meshes were verified and cached. Conditioning changed 3D
pixels while preserving all three linked source-slice views in each study.
The skull and brain remain estimates, with visible stepping where the source
or estimate is coarse. These studies carry display and inspection scope.

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
save and cold reopen at Dice 1.000. Case4's frozen update reproduced held-out
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
