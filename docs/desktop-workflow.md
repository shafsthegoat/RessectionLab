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

In **Scans → Acquisition contributions**, inspect the acquisition role, registration
and processing outcomes. Compare a moving scan with its reference in
**Alignment & registration**. Pick corresponding anatomical points on both
source views and inspect their residuals near the route. Opening a study
or selecting another acquisition preserves its source revision.

For a preprocessed glioma study with aligned T1CE, T1, T2 and FLAIR, choose
the four-channel MONAI option under **Prepare anatomy**. It retains enhancing,
core and whole abnormality as nested unreviewed estimates alongside supplied
labels. Incomplete inputs keep the compatible single-sequence options.
The BraTS bundle requires four channels. Methods evaluated on missing-modality
subsets require a separate verified implementation. Supplied tract bundles
can be inspected. Raw DWI, MRV and SWI alone do not produce named tracts,
veins or sinuses in this build. [MONAI bundle](https://huggingface.co/MONAI/brats_mri_segmentation/blob/main/docs/README.md)
· [mmFormer subset evaluation](https://arxiv.org/abs/2206.02425).

### Watch a recorded engine procedure

Select the study's target in **Plan**. In **Rehearse**, choose **Open recorded
engine episode** and pick the matching recorded environment. The app checks
the native patient, revision, target, source masks, registered reference image,
instrument configuration and branch replay. The selected geometric route stays
in the native study. The episode uses its recorded path. **Surgeon / Microscope**
shows the operative field. Leave **Inspection-only cutaway** and **Inspection-only
transparency** off to see the actual modeled exposure. Play, pause, step or
scrub to a recorded state. Use **Save** to persist the episode and view, and
**Open** to restore them. Each frame carries its cumulative effects.

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
