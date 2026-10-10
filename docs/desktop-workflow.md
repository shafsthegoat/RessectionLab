# Using the Mac workspace

RessectionLab is a local research app. This guide follows the current Electron interface; [project status](../PROJECT_STATUS.md) and the [packaging records](electron-packaging.md) identify which app snapshots have been tested. A modeled geometry check does not establish clinical suitability.

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
attachments are session-only in this version; a save/reopen extension is in progress.

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

## Replay, save and resume

If the independent check accepts the sequence, use **Inspect selection replay**. The timeline and volumes describe the last checked step. While another step loads, the pending message identifies both the requested step and the still-displayed step. STOP with zero removal is a valid recorded result. **Return to source annotations** or **Source view** restores the source display.

**Save** (⌘S) writes the case and view settings locally. After an app restart, saved routes may require a fresh search; their presence in a file is not renewed validation. **Export checked candidate** writes a JSON research record after the replay checks; it does not export a clinically approved plan.

Use **Cancel** in the bottom operation bar to stop a running operation, and wait
for acknowledgement before quitting. Historical training checkpoints remain in
local run storage, separately from the case bundle. The retained resume controls
do not bypass the current patient-training guard. Any saved replay requires its
matching source and completed checks; incompatible or altered records can be withheld.
