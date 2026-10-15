# Native patient anatomy workbench

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
