# Using the Mac workspace

RessectionLab is a local research app. This guide follows the current Electron interface; [project status](../PROJECT_STATUS.md) and the [packaging records](electron-packaging.md) identify which app snapshots have been tested. A modeled geometry check does not establish clinical suitability.

The current verified renderer is `fe25f2a` (`bc63bc2e…`) with the unchanged
`0b4e334` numerical engine (`85a92eff…`). Its 150 desktop checks passed, and all
73 captured desktop source inputs match the commit. The
[latest native checks](../artifacts/electron-case-identity-v1/native-workflow.json)
cover the persistent loaded identifier across UCSF, PAT16 and PAT20 at two
window widths, plus annotation centering. The preceding `346df23` package had
148 checks and [actual PAT16/PAT20 view-only proposal checks](../artifacts/electron-annotation-center-v1/native-workflow.json);
`100865f` had 143 checks and a guidance/motor-prior refresh. Earlier full
training/resume and all-seven-map checks retain their original build identities;
they were not repeated in full here. Experimental feature-unit learning and the
RAW-only axis pilot are not included in this app. See the
[environment card](environment-card.md) for evidence boundaries and the
[exact capture](../artifacts/electron-case-identity-v1/capture-verification.json).

## Open and inspect the source

Choose **Open case** (⌘O) for a saved `.ressectionlab` workspace. **Import MRI** first asks for a structural NIfTI image, then a tumor segmentation; cancel the second chooser to open MRI only. **Explore synthetic fixture** is a labeled fixture for learning the controls.

Check the loaded identifier in the fixed central header; it stays visible when the left panel scrolls and changes when another case opens. It identifies the research bundle, not a verified clinical patient match. The left panel shows the source badge, image dimensions and target annotation labels. **Inspect evidence** shows source provenance, transforms, unresolved inputs and any recorded annotation threshold. A public mirror remains identified as a mirror.

Choose **MRI review** for the linked axial, coronal and sagittal images beside the 3D view. Click an MRI to move the shared cursor; scroll or use arrow keys in a focused MRI pane to change slices. A pane's expand button enlarges that MRI; **Restore linked views** returns to the layout. Use **Contrast**, target checkboxes and **Overlay opacity** to inspect the original image beneath annotations.

**Center on annotations** moves the linked MRI cursor to the annotation center while retaining the current evidence layer, route choices, opacity and layout. It is disabled without annotations. Centering does not accept an estimate or change the source labels.

**3D focus** gives more space to the spatial view. Drag to rotate and scroll to zoom. **Focus anatomy** may crop instruments; **Fit instruments** frames the whole selected tools. **MRI plane on/off** controls the image plane inside 3D without hiding the linked MRI panels.

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

Choose a **Learning budget** and **Seed**, then **Freeze assumptions & train**. The app performs actual policy updates with fixed image, geometry, rewards and optimization/selection worlds. Preparation and independent checking take additional time. This prototype chooses between stopping and one declared modeled stroke; it does not optimize a free-form trajectory or entry. Selection curves are used to choose a checkpoint, not as final evaluation evidence.

## Replay, save and resume

If the independent check accepts the sequence, use **Inspect selection replay**. The timeline and volumes describe the last checked step. While another step loads, the pending message identifies both the requested step and the still-displayed step. STOP with zero removal is a valid recorded result. **Return to source annotations** or **Source view** restores the source display.

**Save** (⌘S) writes the case and view settings locally. After an app restart, saved routes may require a fresh search; their presence in a file is not renewed validation. **Export checked candidate** writes a JSON research record after the replay checks; it does not export a clinically approved plan.

Use **Cancel** in the bottom operation bar to stop a running operation, and wait for acknowledgement before quitting. Available training checkpoints stay in the app's local run storage, separately from the case bundle. After restarting, reopen the same case version, open **Refine → Local run history**, and use **Resume cancelled run**. Resume preserves the saved run's original geometry and budget, even if another route is currently selected. **Recheck and inspect saved replay** reruns the saved replay gate. Incompatible or altered records can be withheld.
