# POC A: local patient anatomy workbench

October 14, 2026. This milestone uses the existing Electron/React/WebGL viewer and
Python imaging engine. It opens saved acquired imaging with cached segmentation
surfaces and linked axial, coronal, and sagittal source views. The application
starts in Cases / Data / 3D Workspace. Earlier experimental workflows remain
under the small research control; the three POC cases cannot be submitted to
route generation, patient training, or surgical replay operations.

## Open the prepared app

From the repository root:

```sh
./scripts/launch_poc.sh
```

This builds the current interface and launches Electron with the local Python
engine. It restores the last opened saved case. **Cases** also lists the three
prepared cases in `outputs/poc-cases/`. To choose the canonical glioma case
explicitly:

```sh
./scripts/launch_poc.sh --case ../outputs/poc-cases/UCSF-PDGM-0004.ressectionlab
```

The case argument is relative to `desktop/`, where Electron runs. The launcher
uses Node on PATH or the existing bundled Codex Node installation. Dependencies
are `.venv` and `desktop/node_modules`; on a fresh checkout, install the Python
environment and desktop dependencies described in the main README first.
Patient data and weights are intentionally absent from Git.

## Short walkthrough

1. Open **UCSF-PDGM-0004** from **Cases**. The default view shows T1CE, three linked
   source-image slices, the extracted brain envelope, and source-provided tumor
   annotations. The seven layer entries distinguish imported annotations from
   MONAI/SynthStrip estimates. Turn any combination on; use each layer's color
   and opacity controls. **Hide all** permits inspection of the source alone.
2. Click or drag a source slice to move the linked RAS cursor. Use the slice
   arrows/wheel to move through the volume. **Source scan** switches among T1CE,
   T1, T2, and FLAIR. Each scan retains its own view settings.
3. Drag in 3D to orbit, scroll to zoom, or choose **Pan** and drag to move the
   camera sideways. **Orbit** restores rotation; right-drag also pans.
   **Fit 3D** and the anatomical camera presets restore useful framing. The
   cutaway axis, position, and flip controls clip displayed surfaces without
   changing masks or source images. **Image review** and **3D focus** change
   the pane layout; all views use the same physical coordinates.
4. Open **Data**. With the reference T1CE selected, choose FLAIR under registered
   blending and adjust its contribution. These four MRI scans have an explicit
   source-provided registration declaration and matching physical grids.
   Newly imported images stay in separate frames and cannot enter this blend.
   Expand the source/model entries to see licenses, original-file hashes,
   local weight paths, mask hashes, and saved run details.
5. **Save case**, quit the application, and run the same launch command. The
   saved camera, cursor, selected scan, visibility, colors, opacity, cutaway,
   blend, and layout are restored. The `.ressectionlab` file includes exact
   downloaded originals and cached masks; it does not need to rerun a model.
6. In **Cases**, open **SynthRAD2023-1BA336** to inspect CT with estimated skull
   and brain, then **Lausanne-sub-000-ses-20110101** for TOF-MRA with estimated
   vessels. These are different people. They are never fused with the glioma
   case. Missing structures remain named as missing in the layer panel.

The local [screenshot walkthrough](../outputs/poc-acceptance/walkthrough.html)
records actual app interactions. Images and the reviewed saved session are kept
locally in `outputs/poc-acceptance/` and `outputs/poc-cases/`.

## Acquired cases and coordinate limits

| Case | Imaging and saved anatomy | Provenance and limits |
| --- | --- | --- |
| UCSF-PDGM-0004 | T1CE, T1, T2, FLAIR; 3 source tumor labels, 3 MONAI tumor estimates, SynthStrip brain extraction | [UCSF-PDGM public mirror](https://huggingface.co/datasets/MedOtter/UCSF-PDGM/tree/e9372219cf1cd2fdd52260cd45f7514b4aa7638e), CC BY 4.0. Exact downloaded preprocessed files retained. Equivalence to official TCIA bytes remains unverified. |
| SynthRAD2023-1BA336 | Acquired CT; TotalSegmentator skull and brain | [SynthRAD2023](https://zenodo.org/records/7260705), CC BY-NC 4.0. Released CT is defaced, cropped, and preprocessed. This is the acquired CT, not synthetic CT. |
| Lausanne-sub-000-ses-20110101 | Acquired TOF-MRA; vesselFM vessels | [OpenNeuro ds003949 v1.0.1](https://openneuro.org/datasets/ds003949/versions/1.0.1), CC0. Regional slab; T1/TOF alignment is unresolved. The session label is preserved; a complete acquisition datetime is unavailable. |

Each saved bundle retains original file bytes and their SHA-256, NIfTI affine,
modality, source identity/license, and known acquisition metadata. Unknown dates
remain unknown. The source's released RAS coordinates are retained; original
scanner DICOM and its full transform chain are not available for these examples.
The existing renderer converts declared LPS frames to RAS for display. This
cycle's native import workflow was exercised with NIfTI, not DICOM ingestion.

Registration eligibility requires a source declaration tied to the same patient
and exact scans, plus a compatible physical grid. Matching dimensions or a
plausible appearance alone does not enable fusion. Source-provided registration
is labeled and still needs anatomical review. The viewer performs source-image
blending on MPR and the 3D source plane; this is not a newly estimated
registration or full volume raycasting.

Source volumes and binary masks remain the quantitative geometry. Worker-built
surfaces are display derivatives. Color, visibility, camera, and clipping never
change source voxels or launch inference. Plans and historical replay records
retain the existing workspace format but are not created for these POC cases.

## Actual pretrained inference on this Mac

All four runs used MPS on the 16 GB Apple Silicon Mac. Times below are single
observed local runs, excluding downloads. RSS was sampled, and Metal allocation
is recorded separately where available; the numbers are not exact whole-device
peak memory or accuracy measurements.

| Model | Required input / output shown | Time | Sampled process RSS | Terms |
| --- | --- | ---: | ---: | --- |
| TotalSegmentator 2.12.0, Dataset297 | CT; estimated skull and brain, 3 mm fast model returned to original grid | 9.46 s | 3.44 GiB | Apache 2.0 model task; CT data noncommercial |
| vesselFM base, pinned `dd70bbdd…` | TOF-MRA; estimated vessels, 128³ tiles | 81.34 s | 1.82 GiB | Open RAIL++-M, Attachment A restricts research/noncommercial use |
| SynthStrip v1 | Structural MRI; brain extraction | 4.21 s | 0.90 GiB | MIT weight option; FreeSurfer runner license |
| MONAI BraTS 0.5.4 | Aligned T1c/T1/T2/FLAIR; whole tumor, core, enhancing estimates | 7.23 s | 1.29 GiB | Apache 2.0 |

Full version, SHA-256, **real local weight path**, source/mask paths, device,
parameters, and timings are recorded in:

- [`manifests/poc-models-v1.json`](../manifests/poc-models-v1.json)
- [`docs/poc-brats-model-manifest.json`](poc-brats-model-manifest.json)
- `outputs/poc-models/<case>/<model>/<cache-key>/inference.json`

The cache binds input bytes, model weights, parameters, and runner identity.
Successful masks are validated before reuse. The app imports the saved results;
it does not run segmentation automatically. Inference is currently an explicit
local command workflow, not a model-launch button in the app.

Important limits:

- Glioma MRI is already preprocessed and skull stripped. SynthStrip demonstrates
  brain extraction, not skull bone or anatomical parcellation.
- MONAI used 128³ fp32 tiles with CPU accumulation instead of the publisher's
  larger CUDA/AMP inference window. The source has known BraTS training overlap;
  this proves runnable inference, not held-out predictive performance.
- Fast CT inference gives coarse bone margins. Vessel estimates include possible
  false positives and extracranial signal; coverage is incomplete and arteries
  are not named. All model estimates remain unreviewed.
- No demonstrated patient has glioma MRI, CT skull, and angiography together.
  Ventricles, detailed parcellation, functional maps, physiology, and surgical
  outcomes are unavailable. No patient-specific safety probabilities are shown.

## Reproduce the saved model cases

The prepared Mac already has the input files. Acquisition identities and hashes
are in `manifests/data_acquisition_ucsf.json`, `manifests/poc-synthrad-1ba336.json`,
and `manifests/poc-lausanne-sub-000.json`. Existing acquisition scripts include
`acquire_public_case.py`, `acquire_lausanne_pilot.py`, and the pinned SynthRAD
archive acquisition. Do not download the entire archive again when it is already
present; the selected CT member is `Task1/brain/1BA336/ct.nii.gz`.

Install the isolated inference runtime and pinned public weights:

```sh
.venv/bin/python scripts/poc_model_setup.py --download-weights
.tools/poc-model-runtime/venv/bin/python scripts/poc_brats_inference.py --download
```

Run or reuse the cached estimates:

```sh
.venv/bin/python -m resectionlab.poc_models --model totalsegmentator-skull --modality CT --case SynthRAD2023-1BA336 --image data/qc/synthrad-task1-first-pair-v1/restricted/ct.nii.gz
.venv/bin/python -m resectionlab.poc_models --model vesselfm-base --modality MRA --case Lausanne-sub-000-ses-20110101 --image data/anatomy/ds003949-v1.0.1/sub-000/ses-20110101/anat/sub-000_ses-20110101_angio.nii.gz
.venv/bin/python -m resectionlab.poc_models --model synthstrip-main --modality T1c --case UCSF-PDGM-0004 --image data/public_mirrors/MedOtter-UCSF-PDGM/e9372219cf1cd2fdd52260cd45f7514b4aa7638e/UCSF-PDGM-0004/UCSF-PDGM-0004_T1c.nii.gz
.tools/poc-model-runtime/venv/bin/python scripts/poc_brats_inference.py --device mps
```

The following uses the successful cache keys from this milestone. A changed
runner or parameter set creates a different key; use that run's reported paths.

```sh
.venv/bin/python scripts/prepare_poc_case.py --brain-mask outputs/poc-models/UCSF-PDGM-0004/synthstrip-main/fe693f2aed5cb0bb90aa/brain_mask.nii.gz --brain-inference outputs/poc-models/UCSF-PDGM-0004/synthstrip-main/fe693f2aed5cb0bb90aa/inference.json --tumor-inference outputs/poc-models/UCSF-PDGM-0004/monai-brats-0.5.4/af24e4d8f13e49e37ff8/inference.json
.venv/bin/python scripts/prepare_poc_model_case.py --inference outputs/poc-models/SynthRAD2023-1BA336/totalsegmentator-skull/32e4c06b003e5b904387/inference.json --source-metadata manifests/poc-synthrad-1ba336.json
.venv/bin/python scripts/prepare_poc_model_case.py --inference outputs/poc-models/Lausanne-sub-000-ses-20110101/vesselfm-base/0cf5010a6c39c81d7f4b/inference.json --source-metadata manifests/poc-lausanne-sub-000.json
```

## Validation

The actual desktop acceptance ran on the target Mac on October 14. The final app
was left open on UCSF-PDGM-0004. Its reviewed session is
`outputs/poc-cases/POC-A-final-review-reopened.ressectionlab`.

| POC A acceptance requirement | Result | Observed evidence |
| --- | --- | --- |
| Open/import an identified public case with source bytes intact | Pass | All three prepared cases opened; native UCSF NIfTI import also saved/reopened with its exact original. |
| Save, quit, reopen without automatic inference | Pass | Electron and Python processes exited; fresh launch restored the final session. Native second save retained identical imaging state and session hash. All 16 watched model/cache files were unchanged. |
| Linked 3D and three source views; registered blend and alignment status | Pass | Integrator exercised linked cursor movement; independent tester switched T1CE/FLAIR and blended the source-registered MRI. Separate CT/MRA views worked; unregistered fusion was refused through the bridge. |
| Individual/combined layers, color, opacity, clipping and camera | Pass | Hide-all/single/all layers, cutaway axis/position/flip, orbit/zoom/Pan/Fit, contrast and restored camera were operated in the app. |
| Real weights and saved modality-compatible inference | Pass | Four MPS model runs; independent SHA-256, nonempty mask, exact affine/shape and archived-voxel checks. Three separate patient identities. |
| Reproducible launch and actual UI walkthrough | Pass | Launcher above; 8-step screenshot walkthrough and full GUI result record saved locally. Remote publication is identified in the delivery message. |

The independent GUI report lists **15 exercised checks**, all passed. The exact
staged application build passed along with **560 desktop tests** (122 host,
344 renderer, 94 viewer). The focused Python persistence/imaging/model suite
passed **44 tests**. One viewer-test cleanup race was repaired and that suite
rerun successfully.

The older, broader simulation bridge suite is **not green**: 27 failures and
8 fixture errors also occur unchanged on pristine base commit `1e0d853`, with
13 passing tests on both snapshots. Independent per-test comparison found zero
new failures; existing synthetic-case/training policy refusals explain the
legacy failures. This milestone did not relax those guards or resume simulation
work. The local comparison is
`build/poc-a-baseline-check/comparison.json`.

The independent local audit `tests/poc_acceptance.py` exercises the actual Python
bridge, Electron asset registry, and frontend workspace hydration. It verifies
all source/weight/output hashes, exact archived voxel arrays and affines, source
versus model labels, saved settings, fresh-runtime reopen, and refusal to blend
an unregistered CT. It rejects process spawning during save/reopen and checks
that model artifacts remain unchanged. A separate import regression verifies
that a saved original survives removal of the input file.

Run it with Node available as `NODE_EXECUTABLE`:

```sh
NODE_EXECUTABLE="$(command -v node)" .venv/bin/python tests/poc_acceptance.py
```

Its report is `outputs/poc-acceptance/independent-backend-hydration.json`.
Actual GUI interactions, process exit/relaunch, and screenshots are recorded
separately in `outputs/poc-acceptance/`; backend tests alone do not count as
desktop acceptance. The ordinary desktop test suite and TypeScript/Vite build
also cover the changed application boundaries.

Small portable result metadata is retained in
[`poc-a-acceptance.json`](poc-a-acceptance.json); images, patient bundles, model
weights, and temporary test snapshots stay out of Git. Earlier unrelated
neighboring-path edits in the working tree are preserved outside this commit.
