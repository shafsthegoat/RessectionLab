# Native patient anatomy workbench


## Prepare a patient study

Launch the desktop with `pnpm --dir desktop build` and `pnpm --dir desktop start`.
In **Cases**, create a patient study or open a native manifest or portable
`.rsworkspace` file. In **Scans**, add NIfTI files or DICOM folders, attach
annotations to their source scan, review sequence roles, and confirm patient
and visit. Untagged images require an explicit local association. Each nonzero
source label keeps its supplied meaning. Diffusion series and projections are
retained as source assets. Use separate studies for separate visits.

Choose **Review preparation plan** and **Prepare anatomy**. The plan lists
eligible models and explains unavailable anatomy. Each completed stage has
a checkpoint. Cancellation leaves admitted layers available. Retry acts on
failed or empty stages. Adding scans keeps the reference fixed and marks
dependent comparisons stale.

Use **Anatomy → Structures** to view registered layers together. The source
selector changes the linked scan for slice review. A layer can extend beyond
that scan's field of view in 3D. The study-wide intensity backdrop uses the
brightest separately windowed contribution at each location. Read quantitative
signal values on the original scan. An unaligned series stays independently
inspectable until its transform is resolved.
In **Plan & compare**, explicitly select a nonempty lesion label. The selection
binds the mask, source and registration chain. Missing vascular and functional
coverage remains unassessed. Quantitative comparison needs an admissible
source-to-reference transform. **Preview operative steps** opens a reversible
geometric animation for the selected approach. It uses the supported tissue
on the line from entry to target. **Rehearse → Open instrument environment**
accesses the separate research mechanics workflow for compatible studies.
The [desktop walkthrough](desktop-workflow.md#from-patient-scans-to-operative-steps)
describes stage controls, source evidence and approach selection.

**Save patient study file…** includes native data, preparation checkpoints, view
settings, the selected target and comparison receipts. For hidden verification
commands, see the [fresh patient reproduction](fresh-patient-reproduction.md).

### Direct MRI tissue estimates with SIAM v2

On an Apple Silicon host, install the pinned inference environment and the
SIAM weights explicitly from the project root. An existing original archive
can be supplied with `--archive /path/to/DS715_NODA.tar.gz`.

```sh
.venv/bin/python scripts/poc_model_setup.py
.venv/bin/python scripts/install_siam_v2.py
```

The installer verifies Romain Valabregue’s 5.31 GB archive and retains the
1.13 GB fold-1 checkpoint, dataset ontology and nnU-Net plan. The default
asset directory is `data/models/siam-v2/DS715_NODA`. Set
`RESECTIONLAB_SIAM_HOME` to an existing installation or
`RESECTIONLAB_SIAM_PYTHON` to the interpreter of the pinned runtime.

For intact structural MRI, confirm full-head coverage under **Preparation
options**, review the **SIAM v2** stage for each acquisition and choose
**Prepare anatomy**. Each source has its own coverage screen and checkpoint.
A cropped reference can remain open while another acquisition supplies
cranial tissue estimates. Each positive class appears on its acquisition grid. Classes include dura mater,
skull, vascular tissue, residual head tissue, CSF and brain tissues. The
separate geometric dural envelope has its own optional stage and provenance.
Inspect boundaries against the source slices. The vascular class combines
vessel types, and the skull label does not measure bone density.

The default SIAM display hides predictions beyond a coarse head envelope
derived from that acquisition. The envelope connects the dominant MRI tissue
signal in each axial section and includes a physical margin. In the outer
margin, dark estimates more than 2 mm from continuous native MRI tissue
signal are suppressed. Above-threshold signal, the acquisition borders and
the filled interior stay protected, including dark skull and disconnected vessels. Anatomy,
slice overlays and the operative field use the same native categorical result.
**Anatomy → Display & fragments → Unfiltered SIAM estimates (QC)** shows the
original predictions. Supplied tumor masks and manual corrections retain
their source voxels. The study archives the original model labels and source
MRI. An older study rebuilds its display cache locally when opened.

The bounded Metal adapter uses one of the five publisher folds, 128³
float32 inference tiles, CPU logit accumulation and Gaussian blending. Its estimates are
unreviewed inspection context. Training used synthetic contrasts derived
from six anatomical templates. Layer provenance records the source scan,
model assets, adaptation and exclusion from training and benchmark inputs.
[Publisher implementation](https://github.com/romainVala/SIAM/tree/52baf64d51275a949d46ee36967b76f8ceaa99d8) · [SIAM v2 weights, CC BY 4.0](https://doi.org/10.5281/zenodo.15780983)

In a development check on MNI BITE case 09, the preoperative contrast MRI
produced all 16 classes. The study authors' tumor mask, on its original
cropped grid, supplied the rehearsal target. The chosen corridor intersected
the estimated skull, dura and brain as well as the annotated tumor. The raw
predictions included detached skull and dura in six sampled background regions.
The supported display removed those skull/dura predictions and preserved
every raw label in a 2.38-million-voxel central intracranial region. All
16,096 supplied tumor voxels remained in place. Thin tissues and small
vessels still need anatomical review.
[MNI BITE source collection](https://nist.mni.mcgill.ca/bite-brain-images-of-tumors-for-evaluation-database/)

### Evidence from the adaptive workflow

The UCSF development study retained its supplied core, edema-like and enhancing
labels. Selecting the enhancing label produced three geometric comparisons that
survived a cold reopen. The scanner-acquired meningioma T1 postcontrast study
reached the same point using an explicitly selected, unreviewed model candidate.
Both studies lacked a source-supported vascular assessment.

In an independent source-image check on EDEN CTRL_0866, readers could identify
both lens centers in the T1 and TOF volumes. The archived rigid transform
placed the two pairs 12.89 mm and 11.49 mm apart. The other three prespecified
points could not be localized reliably in both scans. Reapplying the archived
pull transform reproduced 40 sampled registered voxels exactly. The estimate
failed the workbench's 2 mm landmark criterion. Use the separate corrected
working revision for this volunteer. It opens the TOF in its own frame, keeps
the rejected transform in history and withholds quantitative fusion. The local
testing inventory links the revision, audit and source-plane evidence.

## EDEN source anatomy and saved approaches (2026-11-17)

The normal **Cases → Anatomy → Plan & compare** workflow now opens prepared
EDEN2020 healthy-volunteer studies. The primary demonstration is
`EDEN2020:CTRL_1277`; `CTRL_0866` checks portability. Preparation reuses existing
T1, original TOF (not MIP), FLAIR, SWI, head labels, registration receipts,
and seven publisher-named tract bundles. The separate synthetic target reuses
an existing core/rim annotation. Healthy source scans and streamlines remain
unchanged; no disease displacement, edema or postoperative recovery is inferred.

The paired evaluation uses CTRL_1277, CTRL_0866 and CTRL_0933, with two existing
target options per study. Variants 01 and 08 have independent target-layer IDs
and share the healthy acquired anatomy. The [comparison checkpoint](approach-comparison-checkpoint.md)
links the prepared collection, execution receipts and verification commands.

Private studies live under the configured HDD root at
`prepared/eden2020-planning-v1/<subject>/patient-anatomy.json`. Local prepared
library links end in `.anatomy.json`; opaque library IDs resolve their canonical
manifest without duplicating recent cards. The preparation command is:

```sh
.venv/bin/python scripts/prepare_eden2020_planning.py --subjects CTRL_1277 CTRL_0866
```

It verifies the mounted data-root marker, reuses existing conversions, and writes
source-to-package maps beside each study. Existing completed native studies are
reused. Inventory, checksums, private review evidence and captures stay under
ignored `testcases/eden2020-planning-v1`, linked to the HDD. No imaging or weights
belong in Git. Raw 4D diffusion, gradients and reverse-phase acquisitions remain
source assets, never flattened structural scans. SWI, SWI magnitude and SWI phase
are separate import choices; EDEN's processed SWIp does not supply separate
magnitude/phase or a validated venous segmentation. T1 vein estimates retain their
T1 provenance and do not become SWI-derived veins.

Use the structure legend to show, isolate, recolor, change opacity or focus a
named bundle. Source slices share a physical RAS cursor with 3D and retain that cursor when
switching scans (contrast and camera remain scan-specific); slab-clipped
fiber previews provide slice context. Reversible scalar previews and complete
streamline subsets bound display load; measurements use original grids and
streamline segments. Source-bound imported registrations survive automatic
opening; **Estimate** is still an explicit replacement operation. Their
independent anatomical accuracy remains unmeasured.

In **Plan**, choose entry/target at the cursor or edit their coordinates, set a
corridor radius and declared clearance/length constraints, then create a manual
path or **Suggest approaches**. Suggestions enumerate a bounded selected entry
patch, exclude inferior entries and known cerebellar/brainstem conflicts, and
retain at most three distinct alternatives. Automatic suggestions keep the sampled
ledger and freeze up to eight finalists for shared sensitivity checks. Supported
objectives include length, source proximity, target reach and non-target traversal.
Pareto filtering groups paths with comparable evidence. A detected hard violation
in a declared scenario vetoes the route. The report explains any shortfall and
counts nominally eligible paths outside the sensitivity budget.

**Suggest distinct entries** searches five superior-facing directions from the
selected target for an observed skull or brain surface. It records the source
layer and native voxel for each entry, applies the same comparison checks,
and saves the resulting group alongside the existing patch alternatives.
A cropped field ending inside a label cannot supply its exterior boundary.

Each comparison card separates the observed geometric finding (detected
intersection, no detected intersection in assessed anatomy, or unavailable
measurement) from evidence completeness and declared constraint failures. Missing
labels, limited coverage and unmeasured registration remain explicit uncertainties.
Coverage for a critical source includes the declared clearance margin and any
scenario boundary padding. Contact is measured using the capsule and that
scenario's boundary padding.

Inspect nominal, worst assessed, and empirical adverse fifth results where
perturbed measurements are complete. The default design translates registered
sources in coherent 1 mm scenarios and pads critical voxel boundaries by 0.5 mm,
distances near the source grid scale. Every registration root is stressed in
each signed cardinal direction. Three or more roots use joint translations to
keep the evaluation budget bounded. Registration distributions, model
completeness and postoperative outcomes remain unmeasured. [Uncertainty in path planning](https://scholar.harvard.edu/files/sfrisken/files/incorporating_uncertainty_into_path_planning.pdf) · [Registration uncertainty visualization](https://link.springer.com/article/10.1007/s11548-025-03407-z)

**Compare input sets** freezes rich and restricted nominations on the same task
and sample budget. The restricted arm receives the reference, selected target,
brain, head surface and listed brainstem/cerebellar estimates. The same richer
reference evaluator may veto either arm. Every refusal and abstention is retained
without replacement. Nominations with no hard veto still show incomplete coverage
when source evidence is missing. Known hard failures and incomplete coverage can
appear together in the same scenario. Execution authorization remains with the
procedure engine. Coverage statements apply to the supplied labels, acquisition
fields and declared missing anatomy. Structures absent from those inputs remain
unassessed.

A selected approach shows its capsule, entry region and illustrative virtual
axis. **Along corridor** and **Corridor oblique** reuse the existing camera.
The entry region is hypothetical, with no measured CT skull opening. Edits
recompute source-bound comparisons, including edits to the saved entry-patch
radius. Selecting a path initializes all editor fields from that object; switching
paths and saving flush valid pending edits. The patch centre and plane stay fixed
when its entry moves, and disk violations are displayed as constraint failures.
Selecting a concern highlights and centres that anatomy. Measurements include full capsule field-of-view containment,
per-label conservative voxel-cell proximity/intersections, supplied streamline
proximity, target voxel-centre reach and supported brain traversal. These are
static geometric quantities, not clinical margins, tissue removal, functional
outcomes or execution permission. Unlabeled anatomy, limited sensitivity and
unmeasured alignment remain unknown; empty labels stay unassessed. Existing
physical/simulator refusals are unchanged.

**Save study & comparison** preserves view state, the active target, per-target
approach selections and exact comparison receipts;
**Save patient study file…** includes them in the existing portable `.rsworkspace`
format. Source, assessment-engine or parameter changes mark cached assessments
stale. Geometry lives in `anatomy-approaches.json`, separately from immutable
anatomy revisions. The `anatomyPlan` operation supports target selection,
patch and distinct-entry suggestions, comparison, editing and assessment.
Its `operativeField` action samples tissue from the saved route and selected
target at their recorded revision.


### Verify a saved operative rehearsal

From the repository root, build the desktop app and run the hidden Electron
verifier against a writable test copy with a selected, assessed approach.
The example uses the retained BITE-09 fixture. Each output directory must
be new. The video option requires `ffmpeg` on PATH.

```sh
pnpm --dir desktop run build
output_root="$PWD/testcases/outputs"
study="$output_root/operative-preview-20261118"
manifest="$study/MNI-BITE-09-with-SIAM/patient-anatomy.json"
workspace="$study/MNI-BITE-09-supplied-tumor-SIAM.rsworkspace"
run=$(mktemp -d "$output_root/rehearsal-check.XXXXXX")
electron=desktop/node_modules/.bin/electron
verify=desktop/electron/verify-native-operative-preview.cjs
"$electron" "$verify" "$manifest" "$run/preview" preview --reuse-selection --source-support --video
"$electron" "$verify" "$manifest" "$run/reopen" reopen --open-archive "$workspace"
```

Check `report.json` in both directories for a passing result, the same selected
route and source snapshot, and an empty renderer error list. The preview also
checks MRI identity and the filtered skull against the original estimates, and
records motion in each available chapter. For another prepared study, supply
its matching manifest and workspace and omit the BITE-specific
`--source-support` assertions. The optional drive holds the recorded examples
and fixture restore instructions. [Storage ledger](local-storage-ledger.md)

`desktop/electron/verify-eden-planning.cjs` drives the actual React/preload/Python
workflow with hidden windows. Supply `RESSECTIONLAB_DATA_ROOT`, `EDEN_STUDY` and a
fresh `EDEN_OUTPUT`; `EDEN_PHASE=reopen` with `EDEN_EXPECTED` pointing to the
preceding successful report checks exact geometry, measurements, selection,
camera, layers and source signal in a distinct process. Select changes dispatch
React-compatible change events. Source verification requires controlled React
state, the requested source response, matching rendered/GPU-completed identities,
source-signal fingerprints and visibly different source-pane pixels. A successful
API call or a changed DOM selector alone cannot establish a modality switch.
The private `testcases/eden2020-planning-v1/README.md` identifies the verified
app build, recording, primary case and second-study results. The packaged
walkthrough exercises library opening, named tract focus, three proposals,
radius-driven recomputation, saving, cold reopening, concern focus and along-axis
viewing; the independent source/display and geometry reviews are stored beside it.

Workflow references: [EDEN2020 source](https://zenodo.org/records/3994749),
[Medivis Studio](https://www.medivis.com/studio), and
[Brainlab cranial planning](https://www.brainlab.com/surgery-products/overview-neurosurgery-products/cranial-planning/).


## Automatic alignment and skull display (2026-10-15)

Native workspace opening, imports, and scan selection now ensure source-bound
automatic rigid alignment. CT soft/brain reconstruction anchors to MRI; other CT
reconstructions connect by measured CT-to-CT registration and compose into the
same RAS-mm frame. Identity, head-centre, and historical candidates are compared
against independent samples, anatomical conflicts, reverse registration, and
initialization stability. Reports and historical transforms are retained in
manifest revisions. A warning still permits the best valid display estimate;
automatic consistency never grants planning review. Changed source hashes,
parent transforms, geometry, or algorithm versions invalidate the cache.

The default skull mode is opaque exterior. Transparent and cutaway modes change
both mesh and CT-volume presentation; they never change source anatomy. Legacy
views without an explicit skull mode migrate to exterior. The default vessel
scope is **Full source estimate**, preserving all source components and retained
probability surfaces. Legacy saved scopes migrate once to full source; saved
preferences with `vesselDisplayVersion: 2` retain explicit scope choices.
The optional **Skull-supported only** filter uses enclosed, brain-seeded CT cranial rings. A ring must be closed within
a 2 mm morphological tolerance, avoid the acquisition border, cover at least
70% of its brain seed, and have no more than three times its area. Supported
intracranial space can extend beyond the brain mask. Missing/open skull regions
remain unknown; they are not completed with a convex hull or synthetic anatomy.
The UI separately reports unknown coverage, estimated outside-skull voxels, and
bone conflicts, with a direct **Show full source** action. Unsupported slices can
remove whole bands of source vessels; filtering is never evidence of missing vessels. Quantitative source masks remain unchanged.

Privacy removal can be recorded explicitly as user-reported coverage metadata.
Privacy-redacted regions are excluded from image similarity and are not treated
as alignment failures or reconstructed. They can still limit skull-supported
vessel display; that limitation is reported separately from transform quality.

Reproducible checks (add `--commit` only to save a new patient revision):

```sh
.venv/bin/python scripts/verify_anatomy_alignment.py PATIENT_DIR --output outputs/alignment-verification/report.json --privacy-redacted
cd desktop
pnpm exec electron electron/verify-alignment-display.cjs --anatomy PATIENT_DIR/patient-anatomy.json
```

The Electron check captures actual pixels in mesh and volume views, from two
camera directions, for opaque/transparency/cutaway modes. The 1 mm synthetic
recovery and CT-reconstruction checks and 2 mm consistency warning thresholds
are software acceptance criteria, not clinical accuracy guarantees.

Verification on 15 October: 66 focused Python tests, 599 desktop tests, and the
production build passed. Known-transform fixtures recovered landmarks within
1 mm, including mirrored, oblique and anisotropic grids. Actual Electron pixel
checks passed 12 views each on sub-06 and independent sub-13 (two viewpoints,
mesh/volume, three skull modes). CT reconstruction disagreement was 0.317 mm
p95 on sub-06, down from 10.834 mm, and 0.460 mm on sub-13. The sub-06 alignment
was saved as a revision with user-reported privacy metadata and historical
transforms retained. Source series and layer descriptors were unchanged.

Remaining anatomical warnings are intentionally visible: sub-06 retains a
brain/bone conflict and competing image-similarity result; sub-13 has 2.04 mm
forward/reverse disagreement. No independent anatomical landmark validation was
available. Reports and screenshots are in `outputs/alignment-verification/`.
Broader legacy synthetic bridge suites remain incompatible with the existing
`SYNTHETIC_CASE_DISABLED` policy; that policy was not changed for these checks.

Implemented in the primary checkout on 2026-10-14. Use **Open** or the case library in the main desktop workspace and select
`/Users/shaf/Downloads/RessectionLab-Native-Anatomy-sub-06/patient-anatomy.json`.
The historical `.ressectionlab` workbench and sub-06 baseline remain readable.

This implementation preserves supplied anatomy and supports geometric research.
It does not establish complete vascular coverage, segmentation accuracy, clinical
planning validity, or learned surgical dynamics. The full acceptance milestone
remains open until anatomical review and the remaining qualification below are
complete.

## Implemented behavior

* `PatientImagingCollection` stores original NIfTI/DICOM files, source hashes,
  acquisition/timepoint metadata, native arrays, separate preview pyramids and
  explicit source-to-patient transforms. Equal grids never imply registration.
  Rigid estimates remain unreviewed until explicit landmark review. Checkerboard
  and oblique sections show sampled coverage; landmark residuals are not asserted
  to be a whole-image uncertainty bound.
* Anatomy layers retain their own grids, probabilities/scores, roles, provenance,
  review state and immutable revisions. Categorical regions share a single array;
  masks are materialized only when needed. Brain/skull/parcellations are context,
  tumor is a target, and vessels are critical constraints.
* VTK Flying Edges extracts continuous probability boundaries; Surface Nets
  produces categorical boundaries. Indexed original and display meshes are
  cached independently of masks. Smoothing is bounded in physical millimeters
  and limited by vascular radius. Inverted/collapsed candidates are pinned or
  rejected. Zero-area threshold-tie triangles are omitted from both meshes;
  no vessel components are discarded or bridged. Connectivity is retained.
* The renderer retains full native textures, adapts screen resolution/sampling
  during movement, and refines after interaction. It has editable intensity
  windows/transfer points, modality presets, gradient lighting, local volumetric
  shadows, volume-pass translucent labels, opaque depth testing, clipping boxes,
  orthogonal/oblique cuts, tissue isolation and linked slices. Labels and scan
  density have separate controls. This is not a reproduction of Medivis internals.
* Correction tools provide physical 3D paint/erase brushes on slices, probability
  threshold preview, seeded/component selection, bounded undo/redo, original
  comparison and explicit commit/reject. Commits preserve model outputs and
  create a new snapshot identity. Derived geometry is keyed by content/revision;
  stale drafts/planning requests are refused. Source-frame planning is disabled
  until registration is reviewed.
* Reconstruction adapters share one local job lock and support cancellation,
  sampled RSS limits, cached stages and preserved failed attempts. FastSurfer is
  gated to noncontrast T1. SynthSeg v1 is an explicit manual-QC choice; v2 requires
  its robust/parcellation/QC assets. VesselFM angiographic and experimental T1CE
  entry points are separate. Raidionics adapters validate publisher-declared
  input sequences; missing sequences are never substituted.
* Tractography imports preserve streamlines in source RAS coordinates. Functional
  maps retain their continuous values, appear on oblique sections and can be
  sampled numerically. They remain context until an explicit reviewed anatomical
  derivative is provided; an arbitrary positive score is not an exclusion mask.

## Patient reconstruction

The collection contains three supplied 439 × 439 × 361 volumes on 0.5 mm grids:
T1CE, CT bone and CT brain. The supplied spacing does **not** establish acquisition
resolution. CT-to-MRI fusion remains withheld pending registration review.

| Reconstruction | Actual result | Interpretation |
|---|---|---|
| Native vesselFM, Gaussian tile blending | 673,582 foreground voxels; 84.198 mL; 1,167 components | Experimental T1CE proposal; no angiographic validation |
| Native vesselFM, constant tile blending | 690,756 voxels; 86.345 mL; 1,248 components | Retained comparison, not a recall/precision verdict |
| Native meningioma rerun | 360,280 voxels; 45.035 mL; probabilities exactly reproduce retained native output | 58.27 seconds; sampled peak RSS 3.54 GB |
| Native SynthStrip | 13,341,986 voxels; one component | Extraction proposal; no cortical parcellation claim |
| SynthSeg v1 | Shared categorical regions on the model's own 1 mm output grid | 165.71 seconds; sampled peak RSS 7.59 GB; automatic QC unavailable |
| CT boundary / head envelope | Native CT density boundary and full-head MRI envelope proposals | Includes non-skull bone; envelope is not histological scalp segmentation |

The previous displayed vessel mask had 19,134 voxels on a 1.25 mm grid. Comparing
voxel counts across these grids does not measure recall. The native vascular
proposal retains 223,362 voxels outside the brain proposal and 1,472 overlapping
the tumor. Separate review layers expose those disagreements; zero voxels are
deleted by ROI clipping or tumor subtraction.

Sub-13 remains a negative regression case: its retained native tumor probability
has maximum 0.219622 and no voxels at the publisher's 0.5 threshold. The package
records its input/probability hashes and does not manufacture a tumor label.

## Snapshot and planning interface

```python
from resectionlab.patient_anatomy import PatientImagingCollection, PatientAnatomySnapshot
from resectionlab.anatomy_planning import planning_case, removal_preview

collection = PatientImagingCollection("/path/to/patient-directory")
snapshot = PatientAnatomySnapshot.capture(collection)
occupancy = snapshot.sample([0., 0., 0.])
distance = snapshot.distance("vessels", [0., 0., 0.])
vascular_graph = snapshot.vessel_geometry("vessels")
clearance = snapshot.clearance(start_ras_mm, end_ras_mm, radius_mm=1.)
case = planning_case(snapshot)
removed = removal_preview(snapshot, point_ras_mm, radius_mm=3.)
preview_case = planning_case(snapshot, removed=removed)
```

`planning_case` feeds the existing source-bound critical-evidence resolver and
`generate_candidate_routes`. The tested vessel-intersecting route is rejected;
the controlled fixture without that constraint permits the same route. Missing
vessels/functions remain unknown. Display smoothing leaves planning hashes and
quantitative masks unchanged. Removal is a reversible geometric target preview;
it changes the planning identity while preserving vessels and original anatomy.

Simulation exports are explicit derivatives with snapshot identity, transforms,
mask hashes, volume changes, sampling bounds and conservative obstacle dilation
when resampling is needed. These quantities are not validated boundary errors.
Keep the native collection alongside an exported planning derivative: the latter
contains target/critical arrays, while the collection supplies full native context.

Pretrained anatomical estimates use policy
`pretrained-anatomy-estimates-2026-10-14`. Historical results and their original
learning/admission context are not rewritten. This policy permits anatomy
proposals; it does not convert imaging data into recorded surgical RL transitions.

## Reproduce locally

Install the base project plus `[anatomy]` dependencies. The tested local runtime
uses VTK 9.7.0. Model runtimes are separate from the desktop sidecar.

```sh
.venv/bin/python -m resectionlab.poc_models \
  --image /Users/shaf/Downloads/RessectionLab-Single-Patient-Anatomy-sub-06/scans/CE-T1W.nii.gz \
  --case ZENODO-17486320-meningioma-sub-06-native \
  --model vesselfm-base --modality T1CE --experimental-t1ce --blend gaussian
```

Repeat with `--blend constant` and a different case directory for the comparison.
`scripts/rebuild_native_anatomy.py --help` describes packaging the three native
scans, retained tumor probability and the vessel/SynthStrip inference receipts.
`scripts/complete_native_patient.py` adds the patient-specific previews, overlap
review, head-envelope proposal, archived unreviewed CT transform and sub-13
regression evidence. `scripts/compare_native_vessels.py` records threshold/blending
comparisons. Successful inference and content-keyed geometry are reused.

Local reconstruction controls use:

* `.tools/poc-model-runtime/venv/bin/python` and `data/models/` for vesselFM/SynthStrip.
* `.tools/SynthSeg`, commit `2a2aa3bbfccb83f8253a51ca8b329b9938a2646d`, with
  `.tools/synthseg-runtime/bin/python` (Python 3.10, TensorFlow/Keras 2.15,
  NumPy 1.23.5). v1 weights are present. v2 robust/parcellation/QC weights are not.
* `.tools/raidionics-packages` (RaidionicsSeg 1.5.2 / ONNX Runtime 1.31.0) and
  `data/models/raidionics/MRI_Meningioma`. Meningioma model SHA-256:
  `b74764e90722b1071dab8b085be0d916f8f318d5bed987470fe15198febb81c8`.
  Glioma assets are not installed; its gated adapter has not been qualified.
* FastSurfer supports configured `RESECTIONLAB_FASTSURFER_HOME` and
  `RESECTIONLAB_FASTSURFER_PYTHON`. Its runtime is not installed and this patient
  has no supplied noncontrast T1. No FastSurfer output is claimed.

Runtime/model paths can be overridden by the named `RESECTIONLAB_*` environment
variables in `anatomy_models.py`. Model weights and patient data remain local and
are not committed. Preserve their licenses; the source collection records
Zenodo 17486320 / CC-BY-NC-4.0.

## Earlier native-workbench verification

`tests/test_patient_anatomy.py` exercises mirrored/oblique grids, coverage,
registration review, content integrity, thin separate vessels, smoothing bounds,
compact regions, correction persistence/staleness, cancellation/memory limits,
functional/tract imports, graph caching and the real route generator. Combined
with the focused critical-evidence suites: 27 passed, 13 skipped for unavailable
source fixtures. The desktop component suites pass 122 main, 344 renderer and
105 viewer tests. A combined run hit a temporary-directory cleanup race; the
affected viewer suite passed on rerun. Older bridge tests invoking the deliberately
disabled synthetic-case entry point remain incompatible with the existing policy;
that gate was not relaxed for this work.

`outputs/native-anatomy-verification/` contains the real Electron receipt,
fixed-view screenshots, measured geometry, vessel settings comparison and a
same-data Slicer 5.12.4 reference. `desktop/electron/verify-native-anatomy.cjs`
checks the full main/preload/sidecar/renderer path and rejects its edit drafts.
Frame submission, frame delivery and GPU completion are distinct measurements;
the receipts retain their raw timings. Performance applies to this patient,
selected layers, viewport and machine, not every patient.

The final captured run delivered median frame cadences of **30.17 FPS** for
surfaces and **30.25 FPS** for volume rendering; 95th-percentile delivery
intervals were 45.68 / 44.89 ms. Render-submission cadences were 54.95 / 58.48 FPS.
Capture has observer overhead, and these results do not establish a 30 FPS floor.
Measured GPU completion after interaction ranged from **22.9 to 146.6 ms**, within
the two-second refinement target for the tested views. `performance.json` retains
these definitions. `native-rotation.mp4` is a separate capture demonstration,
not an FPS benchmark. The real-patient handoff receipt confirms that geometric
removal changes the target while retaining the original vessel mask.

Outstanding acceptance is explicit:

* No reference tumor/vessel annotations or independent anatomical reviewer were
  available. Accuracy metrics are therefore unreported. The evaluation module
  supports boundary distances, centerline precision/recall, branch recall and
  diameter-stratified failures when suitable annotations/coverage are supplied.
* CT registration requires landmark review. Cortical/deep-region proposals,
  skull extent, false-positive/missing vessels and tumor-distorted anatomy need
  review. The MRI head envelope is only a coarse anatomical proposal.
* FastSurfer, SynthSeg v2 QC and the glioma adapter require appropriate inputs,
  assets and qualification. DICOM/tract/functional imports have software tests;
  they have not been validated on a broad clinical cohort here.
* The local implementation caps each array at 512 MiB, 64 series and 512 layer
  descriptors. The unified viewer supports 128 resident descriptors and packs
  shared categorical label sets into four channels (up to eight independent
  masks or 255 categorical labels per channel). The current 38-layer collection
  fits without manual unloading. Larger datasets still need an out-of-core/bricked
  rendering extension. It cannot claim unrestricted
  “any patient” operation or universal anatomical completeness.
* Full mesh self-intersection auditing, independent anatomy review and broad
  performance qualification remain open. Tissue mechanics, bleeding, perfusion,
  brain shift and learned surgical dynamics are not supplied by these renderings.

Primary implementation references: [VTK Surface Nets](https://vtk.org/doc/nightly/html/classvtkSurfaceNets3D.html),
[FastSurfer on macOS](https://deep-mi.org/FastSurfer/stable/overview/install/MACOS.html),
[SynthSeg](https://github.com/BBillot/SynthSeg),
[Slicer volume rendering](https://slicer.readthedocs.io/en/latest/user_guide/modules/volumerendering.html),
[Medivis feature reference](https://www.medivis.com/studio/personal).

## Unified workspace integration

Native collections now use the main application header, library, navigation and viewer. The separate native-anatomy screen has been removed. The default is scan plus smooth anatomy with three linked slices; Layout switches to four equal views. Appearance and Clip contain advanced display controls. The sidebar switches between Scans and Anatomy, with grouped searchable regions and details only for the selected structure.

Native view preferences live in `workspace-view.json`, bound to patient, snapshot and source series. Historical `.ressectionlab` sessions remain readable and keep their original save format. Opening native data does not silently convert historical masks, accept registration, or change research/planning evidence. The main launch supports both `--anatomy` and `--case`; subsequent ordinary launches and file-open events are routed into the existing instance.

**Edit segmentation → Review fragments** lists connected components in physical volume order. Selection highlights components, centers the linked slices for a single selection, and leaves original anatomy intact. Preview removal, undo/redo, original comparison and rejection precede an explicitly authored acceptance. Acceptance creates an immutable anatomy revision and invalidates stale review requests. No size threshold or component deletion is selected automatically.

The Electron integration verifier exercises the actual main/preload/Python/renderer path. Its optional mutation checks require a disposable copy whose directory includes `ressection-unified-patient-review-`; never point mutation verification at the supplied patient collection. Use `UNIFIED_REVIEW_TEST=1` for the correction/save/reopen checks and `UNIFIED_LEGACY_CASE` for an additional historical-case Open check. The previous native verifier entry point delegates to this verifier.

```sh
cd desktop
pnpm build
pnpm test
pnpm exec electron electron/verify-unified-workspace.cjs \
  --anatomy /path/to/patient-anatomy.json
```

### Unified verification, 14 October 2026

The focused anatomy, imaging and persistence suites passed **41 Python tests**.
Desktop validation passed **123 main, 344 renderer and 108 viewer tests**, and the
production TypeScript/Vite build passed. `outputs/unified-workspace-verification/`
contains screenshots and the real Electron receipt: native/legacy opening,
38-layer inventory, simultaneous brain regions, CT/MRI switching without
unreviewed fusion, four-view layout, component selection, preview, undo/redo,
original comparison, rejection, revision acceptance, save/reopen and the
1050 × 720 minimum window. Correction checks used a disposable collection;
the supplied original was not edited.

During the measured default MRI rotation, Electron delivered approximately
**30.2 displayed frames/second on average**; the median interval corresponds to
32.7 FPS. Full-detail GPU refinement completed in **99.3 ms** after interaction.
Raw frame-delivery and submission timings are separate in `report.json`, with
derived measurements in `performance.json`. This short local sample meets the
target average and refinement budget, but does not establish a sustained 30 FPS
floor across all datasets, visible layers or window sizes. Smooth display does
not certify prediction accuracy; remaining fragments require explicit review.


### Vessel display gap repair, 15 October 2026

The sub-06 skull filter hid 332,542 of 673,582 source vessel voxels, including
203,723 with unknown skull coverage. The full-source default restores those
existing estimates without editing anatomy, registration, thresholds or planning
identity. Its probability surface also avoids the binary conversion used by
filtered modes. Full-source estimates remain unreviewed and may include
extracranial tissue or false positives.

Migration preserves other view preferences and writes the version marker only
through the existing Save flow. Incremental requests are invalidated before a
scope replacement, preventing stale filtered layers from entering the new view.

Reproduce the data and actual Electron checks on disposable patient copies:

```sh
.venv/bin/python scripts/verify_vessel_display.py PATIENT_DIR \
  --output outputs/vessel-display-verification/PATIENT-data
cd desktop
VESSEL_VERIFY_OUTPUT=../outputs/vessel-display-verification/PATIENT \
  pnpm exec electron electron/verify-vessel-display.cjs \
  --anatomy /private/tmp/ressection-vessel-display-PATIENT/patient-anatomy.json
```

The Electron verifier requires a disposable `ressection-vessel-display-` path
because it exercises saving and reopening view preferences. It compares both
scopes in Structures and Volume, captures the linked slices, measures delivered
rotation frames, and checks that the manifest remains unchanged. Data checks
compare the transmitted full-source mask with native-source resampling and
verify source-array hashes. Residual source-gap candidates, their physical
coordinates and original-image/probability crops are recorded separately;
nearby endpoints do not establish that two vessels should connect.


Gap-repair verification passed **49 focused Python tests**, **601 desktop tests**
(123 main, 356 renderer, 122 viewer), and the production build. The viewer suite
passed on rerun after an unrelated temporary-directory cleanup race. Real
Electron checks passed on sub-06 and sub-13, including saved-scope reopening and
both render modes. Source counts were 673,582 and 33,208 respectively, with no
source changes. Full-source volume rotation delivered about 31.9 and 30.7 FPS;
scope replacement including settling took 11.9–12.6 and 9.0–9.3 seconds.
Receipts and remaining source-gap coordinates/crops are in
`outputs/vessel-display-verification/`. These timings describe this machine and
these two cases, not general performance or anatomical validation.
