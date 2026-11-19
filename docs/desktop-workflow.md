# Using the Mac workspace

Repository upkeep follows the [maintenance policy](../CONTRIBUTING.md): keep current-use material,
retire obsolete code/docs/artifacts, and archive useful history outside the tracked tree.

RessectionLab is a local research app. This guide follows the current Electron interface; [project status](../PROJECT_STATUS.md) and the [packaging records](electron-packaging.md) identify which app snapshots have been tested. A modeled geometry check does not establish clinical suitability.

## Open and inspect a study

With no study open, the app shows **Cases**. Search by case, source or anatomy, open a row to load its study, or choose **Details** to inspect its evidence. **New patient study** starts a study with a patient ID and timepoint.

In a native study, **Scans** shows the active acquisition and its source thumbnail. **Anatomy** lists structures by group. Select a row to inspect the layer or change its appearance, and use its eye control to toggle visibility. **Scan** shows the acquired scalar signal with subdued overlays. **Structures** and **Vessels** emphasize the corresponding anatomy. Reopening a saved study restores its recorded view.

Use the right-hand slice controls to move the linked cursor. In the 3D pane, **Clip → Section at cursor** exposes the source image at the chosen plane. **Exit section** restores the prior cut and plane settings. **Plan** and **Rehearse** show the operations and evidence available for the current study.

### From patient scans to operative steps

1. In **Cases**, choose **New patient study**. In **Scans**, add the patient's DICOM folders or NIfTI volumes together. Attach any label maps to their acquisition and assign the meaning of each label. Confirm the study association.
2. Open **Prepare anatomy**. Confirm intact full-head MRI coverage where applicable and choose a tumor indication if you want a lesion estimate. Review the proposed stages, then run preparation. Calibrated CT supports skull estimates. Volumetric CTA, TOF and MRA support vascular candidates. Eligible structural MRI can contribute dural, skull, vascular and other tissue estimates through the installed SIAM adapter. Review each stage's outcome. [Setup, coverage and models](patient-anatomy-workbench.md#direct-mri-tissue-estimates-with-siam-v2) · [SIAM study](https://arxiv.org/abs/2605.02737) · [vesselFM study](https://openaccess.thecvf.com/content/CVPR2025/html/Wittmann_vesselFM_A_Foundation_Model_for_Universal_3D_Blood_Vessel_Segmentation_CVPR_2025_paper.html)
3. In **Anatomy**, use **Structures** to inspect the combined patient reconstruction. Change the source scan to check a contour against its original slices. Inspect a layer's source and coverage, including where an acquisition ends. An estimated alignment is labeled in the viewer. Resolve a questionable registration under **Scans → Alignment & registration** before using it in a quantitative comparison. Published cranial planning cases used different acquisitions for different structures and frequently required manual segmentation. [107-case imaging study](https://pmc.ncbi.nlm.nih.gov/articles/PMC11008664/)
4. In **Plan**, select the lesion. Use **Suggest distinct entries** to compare paths from separate skull or brain surface points in the supported ipsilateral supratentorial search. To explore variations within one entry patch, set the patch and choose **Suggest approaches**. Compare measured length, labeled target reach, estimated brain traversal, intersections and missing coverage. Select a path and choose **Preview operative steps**. **Rehearse** also opens the current assessed path. A study with no supported lesion can still be inspected in Anatomy.
5. Press **Play steps** or scrub the **Operative timeline**. The field shows the bone flap lifting, the dural flap opening, optional brain retraction and a modeled ultrasonic aspirator advancing into the labeled lesion. Stages need corresponding tissue on the path. A dural envelope proxy is identified at the opening step. Toggle **Include retraction** to explore the same path without a held blade. **Inspection cutaway** exposes the instrument and cavity. The tissue motion and cavity are illustrative geometry. Patient-specific forces, bleeding, functional outcomes and surgical feasibility remain unassessed. [Operative simulation study](https://link.springer.com/article/10.1007/s00701-020-04359-w) · [Retraction in selected deep lesions](https://pmc.ncbi.nlm.nih.gov/articles/PMC6529054/)
6. Use **Return to plan** to select and preview another path. **Save study & comparison** retains the selected approach with its source study. The choice is yours. The metrics describe geometric tradeoffs within the available evidence.

For a cranial study with a cortical gray-matter label, **Structures** shows the folds inside the outline of the head. The selected tumor keeps its cyan planning color. **Layout → Camera preset → Left** shows the lateral profile. Use the linked slices to check an inset, fold or lesion against the scan. The layer details identify the model or supplied annotation. The scan's field of view determines where the head ends. On angiography with an arterial label from that acquisition, the brain and skull recede so you can trace the supplied vessels.

During a geometric preview with anatomy in the patient frame, the intact head and selected route remain beside the local operative field. The adjacent source slice shows the route plane. With a matching surrounding surface, playback starts at the patient extent and approaches the operative site. The exterior and **Inspection cutaway** share the same timeline. The cutaway uses the source MRI to modulate the cut faces before lighting. Cyan identifies the remaining local target and the indentation at the aspirator shows the modeled cavity. Select the grayscale source slice to inspect the underlying signal. In BITE-09, isolated inferior and superficial colored islands also occur in the source segmentation estimates. The original study has no angiogram to adjudicate those islands.

In **Scans → Acquisition contributions**, inspect the acquisition role, registration
and processing outcomes. Compare a moving scan with its reference in
**Alignment & registration**. Pick corresponding anatomical points on both
source views and inspect their residuals near the route. Opening a study
or selecting another acquisition preserves its source revision.

For a preprocessed glioma study with aligned T1CE, T1, T2 and FLAIR, choose
the four-channel MONAI option under **Prepare anatomy**. It retains enhancing,
core and whole abnormality as nested unreviewed estimates alongside supplied
labels. Incomplete inputs keep the compatible single-sequence options.
The four inputs must be distinct brain-extracted, spatially qualified images
on the supported common grid. An arbitrary set of four uploads can fail
that check. Inspect the reason for each withheld stage in the study.
The BraTS bundle requires four channels. Methods evaluated on missing-modality
subsets require a separate verified implementation. Supplied tract bundles
can be inspected. Raw DWI, MRV and SWI alone do not produce named tracts,
veins or sinuses in this build. [MONAI bundle](https://huggingface.co/MONAI/brats_mri_segmentation/blob/main/docs/README.md)
· [mmFormer subset evaluation](https://arxiv.org/abs/2206.02425).

### Watch a recorded engine procedure

In **Plan**, select the target and approach. In **Rehearse**, choose **Open recorded engine episode** and select the saved environment. The app checks the patient, study revision, target, source masks, reference pixels, instruments and recorded branch. When the recording identifies a native approach, its entry, target, radius and measurement identity must match the selected plan. An older recording can establish the patient and target without establishing an approach match.

The intact reference shows the patient and the approach carried by the recording. The operative viewer shows the engine's tool poses and tissue state. **Operated anatomy** sections the working grid through the effect. Residual target, removed space and mapped protected structures have separate colors. Use the grid and slice controls to inspect a specific location, or **Follow effect** to follow the removal. The section marks the recorded tool tip when it intersects the plane. Volume readouts cover the whole modeled target.

Play, pause, step or seek to an observation. The 3D field, section and quantities follow that observation on the same branch. In the procedure, **Save** writes the engine environment and playback position. In the native study, **Save** retains the plan and source view. Returning to Plan restores the native study. Opening a saved procedure on its own uses its embedded source grid for the intact reference.

**Export observed replay** saves a captioned MP4. It waits for the tool and surface state and the corresponding operated section, then captures the joined view. The recording uses engine observation times and holds the endpoint for one second. Export restores the prior cursor. FFmpeg with H.264 and subtitle support is required.

Choose **Source edges** or **Conditioned edges** in the operative legend to
compare display surfaces at the same camera and engine state. Inspect a legend
entry or double-click a surface for its layer and linked source coordinates.
On the historical UCSF preparation the violet source label is supplied contrast
enhancement. Magenta on the MRI slices identifies the experimental vesselFM
estimate. The microscope shows that estimate in muted rose with a separate
slice swatch. Its positive signal does not measure vascular clearance.
[VesselFM's published scope](https://github.com/bwittmann/vesselFM).

To inspect the current proposal in the engine, choose **Open instrument
environment** from the preview and **Inspect approach in engine**. The derivative
records the approach, target, source revision and physical geometry. A full
procedure requires reviewed entry, operative compartment, brainstem, cerebellum,
coverage and swept instrument workspace evidence. A selected path by itself
authorizes zero actions. [Acquired case and measured results](current-status.md#acquired-scan-to-procedure-demonstration).

The operative field reports its display spacing above the 3D view. On the
MNI BITE 09 study the field uses a 0.3125 mm grid. The renderer smooths the
tissue surfaces and samples the acquired scan to shade them, including cut
faces. The inspection plane sections the vessels along with the surrounding
tissues. In the exposed-skull field, superficial vascular estimates above a
supported outer bone surface are hidden with the scalp. **Field evidence**
reports that count and identifies the scan supplying the tissue contrast.
The full vascular estimates are available in **Anatomy**. Original scan
spacing and supplied contours are available in **Scans**.

Use **Show target** in Plan to reveal the selected annotation and center the
linked slices at the editable endpoint. In a deep geometric rehearsal,
**Follow operative depth** opens a longitudinal section through the working
field. Its surrounding grayscale plane comes from the matching acquisition.
**Exterior** returns to the surface view. The BITE-09 example uses the supplied
tumor contour and MRI-derived tissue estimates.

### Compare two recorded procedures on one target

Open the native study and select its target in **Plan**. In **Rehearse**,
choose **Compare two recordings** and select both procedural files in the
same file chooser. The app independently checks each saved route, replays
both action histories, and compares endpoints on the same patient revision,
engine scene, initial state, physics and instrument catalog. When a bound
reference scan is available, **Along A** and **Along B** change the plane in
both endpoint images together. Cyan traces the original target. Amber marks
removed source cells intersected by that plane. The table
reports the whole original annotation remaining, tissue assigned to the
target and removed, other mapped tissue removal, elapsed time and
withdrawal. An unavailable material appears as **Unmeasured**. Close the
table and select **Review comparison** to inspect it again.

To watch one arm, select its own approach in **Plan**, then choose **Open
recorded engine episode** in **Rehearse**. Choose **Inspect result in 3D**
from the operated section to rotate the amber cast of the empty cavity
and the cyan surface of the remaining owned tumor. Choose **Cavity** or
**Residual** to isolate and fit that surface. **Both** shows their relative
positions. The residual is opaque, and the portion of the cast behind it
appears in muted amber. Mesh conditioning affects the display. The cell
ledger supplies the measurements. **Return to operative field** restores
the microscope. In a bound recording, **Inspect recorded corridor** sections
the working field along the recorded approach. **View down approach** returns
to an exterior view. **Save**
retains the current frame, result mode and orbit. The adjacent full-target
readout includes original annotation cells assigned to overlapping labels.

On UCSF-PDGM-0004, the study's supplied enhancing contour was frozen
before searching for approaches. To reproduce both arms, run the following
from the checkout with the local fixtures listed in the storage ledger.
The first command creates a new directory and writes the protocol.

```sh
output=testcases/outputs/ucsf-fixed-target-repeat
.venv/bin/python scripts/compare_patient_cusa_routes.py plan "$output"
.venv/bin/python scripts/compare_patient_cusa_routes.py execute "$output"
.venv/bin/python scripts/compare_patient_cusa_routes.py bind "$output"
```

For a different study, `scripts/bind_native_engine_recording.py --help`
lists the inputs to bind a recorded environment to a measured native
approach. The application supports exact source grids and its verified
resampled research derivative. The binder checks physical results through
cold replay, including effective tool radius and direction, initial state,
time, tissue membership and residual. Generator attribution comes from
the producer and includes a source digest and, for RL, a checkpoint digest.
The current route contract covers straight treatment strokes and preceding
opening actions. [Procedure results](current-status.md#fixed-target-procedure-comparison).

### Replay the IXI077 T1 and MRA case

Open `selected-approach.rsworkspace` for the intact study. Its Plan view
identifies the simulated target and the recorded 30 mm post-exposure
approach. In Rehearse open `recorded-engine.ressectionlab` to see the
intact patient alongside the working field. For an immediate inspection of
the final cavity, open the procedure file directly and step back to see
earlier observations. The local testing guide indexes the saved files,
walkthrough and matched captures. [Research data locations](local-storage-ledger.md).

To reproduce from the verified original IXI077 archives on the optional
research drive, build the desktop app and choose a fresh output directory.
The preparation checks the input files, saved execution report and engine
endpoint. The following commands run the application in hidden windows.

```sh
pnpm --dir desktop build
run=$(mktemp -d "$PWD/testcases/outputs/ixi077-replay-XXXXXX")
.venv/bin/python scripts/prepare_ixi077_study_replay.py \
  --data-root "/Volumes/shafrir/old coding projects/RessectionLabData" --output "$run/source"
verify="$PWD/desktop/electron/verify-integrated-engine-replay.cjs"
native="$run/source/native/patient-anatomy.json"
episode="$run/source/replay.ressectionlab"
for phase in warm --cold-plan --cold-midpoint --cold --full-movie; do
  if [ "$phase" = warm ]; then
    desktop/node_modules/.bin/electron "$verify" "$native" "$episode" "$run/result" synthetic_target_observation
  else
    desktop/node_modules/.bin/electron "$verify" "$native" "$episode" "$run/result" synthetic_target_observation "$phase"
  fi
done
```

The full movie pass reopens the portable plan, attaches the recording,
captures the observations through **Export observed replay** and checks the
restored endpoint. `observed-engine-replay.mp4` follows simulation time.
`patient-engine-continuous.mp4` adds the planning views, accelerates
playback fourfold and holds the final inspection. The shorter sampled
verification movie identifies its individual observations.

In studies with a cortical mesh and matching source map, switching the
slice panel to an aligned CT keeps the MRI-derived cortex registered to the
patient and textured by its own MRI. The scan label identifies the active
slices. A transform accepted only for overlay inspection retains that
qualification in the anatomy view and in planning. Unknown vascular
clearance remains visible on the IXI077 route.

### Reproduce the acquired replay and generated comparison

Run from the repository root on the configured Mac with the Python environment,
desktop dependencies and FFmpeg installed. The local test guide locates the
UCSF source and original episode. The output link below resolves to the verified
personal HDD. The commands create a new run and a disposable APFS clone of the
source. Both Electron entry points use hidden windows.

```sh
set -e
pnpm --dir desktop build
evidence="$PWD/testcases/outputs/scan-to-engine-20261118"
test -f "$evidence/README.md"
run=$(mktemp -d "$evidence/reproduce-XXXXXX")
scratch=$(mktemp -d "$PWD/testcases/outputs/scan-to-engine-input-XXXXXX")
cp -cR testcases/environments/native/UCSF-PDGM-0004 "$scratch/native"
native="$scratch/native/patient-anatomy.json"
episode="$PWD/testcases/matched-exposure-v1/desktop-reopen-01/strategies-saved.ressectionlab"
electron="$PWD/desktop/node_modules/.bin/electron"
plan=desktop/electron/verify-native-operative-preview.cjs
replay=desktop/electron/verify-study-engine-replay.cjs
"$electron" "$plan" "$native" "$run/plan" preview --distinct --inspect-shortest --export "$run/selected.rsworkspace"
"$electron" "$plan" "$native" "$run/inspect" reopen --open-archive "$run/selected.rsworkspace" --engine-inspection
"$electron" "$replay" "$native" "$episode" "$run/recording" enhancing_target
"$electron" "$replay" "$native" "$episode" "$run/recording" enhancing_target --cold
"$electron" "$replay" "$native" "$episode" "$run/recording" enhancing_target --cold-midpoint
.venv/bin/python scripts/demonstrate_approach_procedure.py --output "$scratch/generated-approaches"
cp -R "$scratch/generated-approaches" "$run/generated-approaches"
.venv/bin/python scripts/compare_observed_skill_policies.py "$run/generated-search-and-rl"
```

To repeat the comparison on the four-channel revision saved after inference,
open its portable source in the same harness. The manifest supplies the
expected revision for the verifier. The app opens the full archive through
its normal study chooser.

```sh
bundle="$scratch/brats.rsworkspace"
qc="/Volumes/shafrir/old coding projects/RessectionLabData/qc"
ln -s "$qc/brats-native-0004-20261118/UCSF-PDGM-0004-BraTS-compartments.rsworkspace" "$bundle"
unzip -p "$bundle" patient-anatomy.json > "$scratch/brats-manifest.json"
"$electron" "$plan" "$scratch/brats-manifest.json" "$run/four-channel" preview --open-archive "$bundle" --distinct --inspect-shortest --export "$run/four-channel.rsworkspace"
"$electron" "$plan" "$scratch/brats-manifest.json" "$run/four-channel-inspection" reopen --open-archive "$run/four-channel.rsworkspace" --engine-inspection
```

The plan check saves the shortest of three measured windows. The recording
check opens the historical preparation through **Rehearse**, captures every
accepted boundary, verifies the visible tools and cumulative ledger, and writes
`engine-preparation.mp4`. The two fresh processes restore the endpoint and the
midpoint. The generated comparison reads the frozen seed-17 checkpoint under
`testcases/overnight-sequential-v1/benchmark-task-v2`. After checking the
outputs, discard the disposable input clone. The saved study contains the scan
arrays and planning selections. [Case results and qualification](current-status.md#acquired-scan-to-procedure-demonstration) · [Private asset locations](local-storage-ledger.md).

### Reproduce continuous rehearsal

Open the generated `generated-cleanup/final/complete.ressectionlab` from
**Cases → Open** and choose **Rehearse → Surgeon / Microscope**. Leave the
inspection cutaway off. Play from boundary zero, pause during action 108
to inspect the first measured target removal, and scrub between boundaries
119 and 200 to see 236 mm³ of debris cleared. The endpoint records both tools
outside. Export from the same view with **Export observed replay**. For the
acquired decision, open `selected-decision/UCSF-0004-intact-stop.ressectionlab`.
The four-channel source study is the earlier
`UCSF-0004-four-channel-selected-window.rsworkspace`. The older acquired MP4
uses a separately bound trajectory. The [storage ledger](local-storage-ledger.md)
locates these private assets and the reproduction guides for comparing them.

From the project root, choose a new, empty ignored output directory. When
using the personal HDD, create it below the verified archive link. The generated
case can be recreated from source. Hidden desktop runs use the installed
Electron binary and the same controls as the application.

```sh
run=testcases/outputs/continuous-rehearsal-rerun
mkdir "$run"
pnpm --dir desktop build
.venv/bin/python scripts/demonstrate_generated_kerf_cleanup.py --output "$run/cleanup"
.venv/bin/python scripts/generate_curved_render_case.py "$run/curved"
.venv/bin/python scripts/verify_curved_render_geometry.py "$run/geometry.json"
electron=desktop/node_modules/.bin/electron
"$electron" desktop/electron/verify-engine-contours.cjs "$run/curved/curved-field.ressectionlab" "$run/contours" 2 1,2
replay=desktop/electron/verify-study-engine-replay.cjs
casefile="$run/cleanup/complete.ressectionlab"
"$electron" "$replay" --workspace "$casefile" "$run/movie" generated-target
"$electron" "$replay" --workspace "$casefile" "$run/movie" generated-target --cold
"$electron" "$replay" --workspace "$casefile" "$run/movie" generated-target --cold-midpoint
```

For the frozen comparison, the previous run's `freeze.json` and
`checkpoints/17/selected.pt` must be present under
`testcases/overnight-sequential-v1/benchmark-task-v2/`. The HDD review
copy under `evaluation/reproduction-inputs/` includes checksums. The evaluation
checks the protocol and weights on entry. Each selector receives the same 28
features and legal skill mask. Each post-access arm has up to eight selections,
56 physical calls and 25 online seconds. Each closed-shell arm has one route
selection, 600 public calls, 300 accepted actions and 180 online seconds.
The original engine independently replays each recorded sequence.

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTHONPATH=src \
  .venv/bin/python scripts/compare_complete_generated_plans.py "$run/matched"
```

Historical capture receipts remain available at the links below. Most images
and recordings now require the [visual evidence archive](visual-evidence-archive-20261116.md),
which restores their original filenames beside the unchanged receipts.

The [workspace persistence follow-up](../artifacts/integrated-workspace-persistence-v1/RESULT.md)
passes 310 desktop and 55 relevant backend/imaging checks. Actual native Save and
fresh-process reopen were checked for generated replay and public RESECT T1/FLAIR.
Its exact staged desktop builds; the app has not been repackaged.

The [October 9 shared-episode integration](../artifacts/integration-first-shared-episode-v1/RESULT.md)
was built and exercised in the actual Electron app with generated replay and
public RESECT T1/FLAIR display. Its exact staged desktop builds; the worktree
desktop suite passed 282 checks. Packaging is separate and has not been repeated
for this source. The trained-policy smoke is headless; it is not a learned option
in the desktop episode selector.

An earlier verified renderer was `c09ee50` (`834c7ceb…`) with the unchanged
`0b4e334` numerical engine (`85a92eff…`). Its 153 desktop checks passed, and all
73 captured desktop source inputs match the commit. The
[latest native checks](../artifacts/electron-route-empty-state-v1/native-workflow.json)
cover unloaded/pre-search/blocked guidance, actual UCSF search and category/A/B
comparison at two window widths, plus case identity and annotation centering.
All three actual categories were nonempty; the empty-filter boundary has source
tests. Previous identified packages separately verified three-case identity,
PAT16/PAT20 view-only proposals, all-seven-map inspection and full training/resume;
those broader checks were not repeated in full here. Experimental feature-unit
learning and the RAW-only axis pilot are not included in this app. See the
[environment card](environment-card.md) for evidence boundaries and the
[exact capture](../artifacts/electron-route-empty-state-v1/capture-verification.json).

## Open and inspect the source

Choose **Open case** (⌘O) for a saved `.ressectionlab` workspace. **Import MRI** first asks for a structural NIfTI image, then a tumor segmentation; cancel the second chooser to open MRI only. **Open generated episode** provides a labeled software example for learning the controls.

The imaging workspace can attach additional 3D scalar NIfTI images, optionally
with aligned source-provided or estimated annotations. Select a modality and use
**Inspect scan** to view each native grid. Source association, acquisition time,
registration and annotation coverage remain unverified. Primary routes and
simulation overlays are hidden while a separate grid is displayed. These
attachments are now preserved by **Save**, together with their native frames,
source/estimated label provenance, selected image, individual cursors and layer
visibility. Reopening restores those settings without requiring the original
image paths. Generated Episode history and its selected replay frame are also
saved and revalidated. Camera pose, layout, contrast and global opacity are not
persisted by this extension. See the [save/reopen checks](../artifacts/integrated-workspace-persistence-v1/RESULT.md).

Check the loaded identifier in the fixed central header; it stays visible when the left panel scrolls and changes when another case opens. It identifies the research bundle, not a verified clinical patient match. The left panel shows the source badge, image dimensions and target annotation labels. **Inspect evidence** shows source provenance, transforms, unresolved inputs and any recorded annotation threshold. A public mirror remains identified as a mirror.

Choose **MRI review** for the linked axial, coronal and sagittal images beside the 3D view. Click an MRI to move the shared cursor; scroll or use arrow keys in a focused MRI pane to change slices. A pane's expand button enlarges that MRI; **Restore linked views** returns to the layout. Use **Contrast**, target checkboxes and **Overlay opacity** to inspect the original image beneath annotations.

**Center on annotations** moves the linked MRI cursor to the annotation center while retaining the current evidence layer, route choices, opacity and layout. It is disabled without annotations. Centering does not accept an estimate or change the source labels.

**3D focus** gives more space to the spatial view. Drag to rotate and scroll to zoom. **Focus anatomy** may crop instruments; **Fit instruments** frames the whole selected tools. **MRI plane on/off** controls the image plane inside 3D without hiding the linked MRI panels.

Route category and A/B controls appear after candidate routes exist. Before search, the right panel explains the next step or the unresolved access requirement. If an active category is empty after search, use the category selector to inspect another set.

## Keep the evidence layers distinct

| Layer | How to inspect it | What the display means |
| --- | --- | --- |
| Source annotations | Target checkboxes and opacity | Supplied or explicitly threshold-derived labels; not a predicted removal. |
| Brain-envelope estimate | **View on MRI** under Structural proposals | A separate lavender contour. **Inspect annotation outside (…)** moves to an actual annotation cell outside the estimate; the count is not segmentation accuracy. |
| Population prior | Choose one map under **Population priors** | Registered population evidence awaiting alignment review; not patient-specific function and not used in route scoring. |
| Modeled removal | **Inspect selection replay** after an accepted simulation check | The declared simulated effect and residual annotations; the original MRI and source labels are preserved. |

**Hide estimate**, **Return to source view**, or the banner's **Source view** clears an inspection layer. Changing cases or entering removal replay also clears estimates and priors.

The seven stored priors distinguish motor, phonology, semantics, and speech arrest/articulation. Functional maps show unitless atlas concordance; structural maps show released-mask membership, not patient tracts. At the cursor, covered zero still means patient function is unknown. Hatching and an unavailable-sample message indicate missing atlas or interpolation support. Language dominance remains unknown.

**Inspect evidence → Add brain-envelope proposal** imports the original extraction MRI, predicted mask and recorded report. Cancelling any chooser leaves the case unchanged. Viewing or importing an estimate does not accept it as working anatomy or certify cortical access; there is no acceptance shortcut in this interface.

## Compare complete-tool routes

In **Routes**, open **Search settings**, choose an instrument configuration and select **Generate candidate routes**. Where offered, estimated image support is an explicit opt-in for hypothetical access windows. Full-head cases can remain blocked for missing reviewed research support; displaying a brain-envelope estimate does not unblock them.

Select mint **A** and amber **B** to compare accessible target, route length, clearance, normal exposure and assessment together. **Accessible target** and **Normal exposure** are geometric quantities, not removed tissue. Missing anatomy remains **Unassessed** or **Incomplete**.

Switch between **Retained alternatives**, **Dominated alternatives**, and **Rejected candidates**. A rejected candidate retains its reason and **Inspect failure location** moves the cursor to the recorded failure. Additional research models keep their own assumptions and retained sets; they are not one shared ranking.

## Refine the selected route

Choose a feasible route **A**, open **Refine**, and select **Check modeled cutting**. A route can pass static inspection yet have no legal initial cutting action. In that case training stays disabled and the reason remains visible.

**Need another research geometry? → Generate additional research routes** adds alternatives with different hypothetical access windows, routes and instruments. Return to **Routes** to select one, then check it again.

The historical **Learning budget**, **Seed**, and **Freeze assumptions & train**
flow is currently guarded off at the patient-training bridge. Earlier snapshots
performed fixed-world prototype updates, but those historical checks do not enable
patient learning in this build. Current generated learning and its actual results
are described in the [opening-task guide](native-opening-learning.md).

## Execute a generated sequential episode

Choose **Open generated episode**, select **Scripted** or **SEARCH**, and choose
**Execute generated episode**. This opens a generated source case, not a simulation
derived from the previously displayed patient. The scripted sequence demonstrates
aspiration and non-removing probe contact; SEARCH chooses its own actions under
the same modeled task.

Use the action timeline, **Previous frame**, **Next frame** and replay slider to
inspect recorded insertion and withdrawal poses, full-tool geometry and changing
tissue. **Source view** restores the original source. Probe contact counts are
geometric history, not force measurements. Modeled removed volume is distinct
from accessible target volume. No trained mixed-mode policy or clinical outcome
is represented by this workflow.

After execution, **Evaluate annotated encounters** checks the completed generated
strategy against a separate generated vessel reference. The readout follows the
selected action and distinguishes shaft, tip and full-tool contact, positive
annotation cells, unknown cells and missing field of view. Counts cover the entire
recorded action and return path; they are not contact at the single displayed
frame. The initial frame has no action, and STOP has no sweep. These annotations
were not planning inputs. Contact does not establish vessel removal or injury.
This evaluation is transient: a new or reopened episode clears it, and Save does
not persist it. See the [integrated checks](../artifacts/integrated-episode-vascular-v1/RESULT.md).

## Replay, save and resume

If the independent check accepts the sequence, use **Inspect selection replay**. The timeline and volumes describe the last checked step. While another step loads, the pending message identifies both the requested step and the still-displayed step. STOP with zero removal is a valid recorded result. **Return to source annotations** or **Source view** restores the source display.

**Save** (⌘S) writes the case and view settings locally. After an app restart, saved routes may require a fresh search; their presence in a file is not renewed validation. **Export checked candidate** writes a JSON research record after the replay checks; it does not export a clinically approved plan.

Use **Cancel** in the bottom operation bar to stop a running operation, and wait
for acknowledgement before quitting. Historical training checkpoints remain in
local run storage, separately from the case bundle. The retained resume controls
do not bypass the current patient-training guard. Any saved replay requires its
matching source and completed checks; incompatible or altered records can be withheld.
