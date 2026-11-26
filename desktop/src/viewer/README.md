# Patient imaging viewer

Repository upkeep follows the [maintenance policy](../../../CONTRIBUTING.md): keep current-use material,
retire obsolete code/docs/artifacts, and archive useful history outside the tracked tree.

One WebGL2 context shares original MRI and annotation textures across the 3D scene and three linked MRI views. Source arrays use NumPy XYZ C order (Z fastest); texture dimensions reverse this order. Source affine transforms are retained, including oblique spacing. LPS inputs convert once to the RAS display frame. Cursor positions and complete instrument dimensions are millimeters.

The default 3D focus layout and optional equal-pane image review layout reuse the same renderer, camera, source textures, and linked physical cursor. Each source view can expand and restore within either layout. Controls and pane headers sit outside the rendered image areas; actual image-area bounds drive GPU scissoring, slice aspect ratio, scale bars, and pointer mapping. Hidden panes submit no draw calls. Layout changes do not reset the camera or alter routes, images, masks, or measurements.

Temporary inspection modes appear as a mode chip at the top-left of their pane (`ModeChip.tsx`; registry and Escape order in `viewerModes.ts`). The chip names the mode and keeps its provenance wording; its × and Escape run the same restore. One Escape leaves only the highest-priority mode, and an Escape already handled by a popover, field or dialog leaves none. The expanded operative field keeps its unconditional Escape. The chips cover the expanded field or source pane, the 3D result, the corridor view, the source section, the vessel slab, the corridor/box/branch crops, the CTA cut, vessels through a lesion, the unexecuted tool preview, and the host's estimate, population prior, modeled replay and generated episodes. While a slider that changes the 3D image is dragged, a read-only HUD at the top centre names it and shows its value; it hides 600 ms after release, without a fade under reduced motion. Range inputs opt in through `data-viewport-hud`, or by a known accessible name in `viewportHud.ts`. The 3D scale bar at bottom right is measured at the orbit target from each rendered camera (RAS millimetres) and rounds to 10, 20, 50 or 100 mm. In perspective, nearer anatomy appears larger than the bar.

The bottom-left **Body orientation guide** is a generic, naturally proportioned mannequin with a mint head and patient-right/left labels. It follows the actual 3D camera rotation (including roll and microscope views), with a fixed screen size independent of pan, zoom or source voxel orientation. Appearance can hide it; the choice persists per scan and old workspaces start enabled. Its two original smooth meshes render in a separate scene through the same WebGL context, after anatomy compositing and inside their own depth/scissor bounds. It participates in no anatomy bounds, fitting, clipping or picking.

When a displayed surgical state supplies valid body/head-to-room rotations, the head stays aligned with the brain and the torso uses `inverse(headRasToRoom) × bodyRasToRoom` about the neck. The caption names the recorded posture and **Down** shows room gravity in head RAS; ⊙ means toward the viewer and ⊗ away. The rigid posture model keeps limbs neutral. Labels and transforms follow the committed tissue display, including previews and replay; missing/invalid transforms return to **Anatomical reference** without inferring a patient setup. The guide does not change the recorded posture or source camera.

From `desktop/`, `pnpm exec electron electron/verify-orientation-guide.cjs --report-dir testcases/outputs/<fresh-run>` checks actual hidden GPU views and pointer pass-through. `ORIENTATION_CASE` selects an existing local `.ressectionlab` or `.rsworkspace`; native archives are copied into the run before opening because header Save updates the opened archive. `ORIENTATION_REOPEN=1` checks a saved hidden guide in a fresh process; `ORIENTATION_RECOVER=1` also exercises graphics recovery. `ORIENTATION_APP_BUNDLE` selects the built app's actual main code, renderer and frozen engine while keeping the verifier hidden.

Source binary annotations produce display surfaces in a worker. The 0.5 isosurface rounds voxel corners; quantitative volumes always come from native voxel cells, never mesh volume. No generic brain surface is substituted for missing patient anatomy. The source-image plane starts visible in 3D and can be toggled; the linked source views remain visible.

## Anatomical workspace display

The **Scan volume** control composites CT or MRI scalar data by GPU ray marching, independently of annotation surfaces. CT bone, CT soft tissue, CTA and sequence-aware MRI presets map the source values to linear color and extinction per millimetre. The 3D toolbar has one preset menu next to **Volume**. **Vessels** selects the CTA transfer, or the bright-blood MR transfer on MR angiography. **Bone** and **Soft tissue** select the CT transfers and are disabled on MR. **Brain** selects the MR tissue transfer. Stored preset IDs are unchanged. The primary ray uses linear source interpolation, affine-correct gradients and ray steps, and front-to-back Beer–Lambert compositing. Explicitly styled CT skull context combines the configured color with the underlying HU to retain variation inside the bone. Opaque scene depth stops rays at anatomy and instruments. Depth peeling retains four foreground translucent mesh boundaries at rest, using the mesh materials and clipping. The remaining transparent fragments are composited as a tail at its nearest depth, with approximate ordering inside the tail. Interaction uses that bounded tail alone, and release restores the foreground peels. Colors are tone mapped once at final display.

The same RAS cutaway clips the scalar volume and surfaces, exposing the acquired intensity field at the cut. Camera movement uses coarser ray steps and a smaller framebuffer. Pointer release restores four progressively jittered samples accumulated in linear space. A camera, geometry, appearance, instrument, clipping or layout change invalidates accumulation. Static refinement reuses the mesh/depth passes. The volume framebuffer's long edge is capped at 1024 pixels when settled. During interaction it ranges from 240 to 720 pixels, adjusted from asynchronous GPU completion. Drag and final-quality render targets are cached separately and disposed on case replacement with the volume textures. The shared drawing buffer and linked slices retain their display resolution throughout gestures. Wheel bursts, slice navigation, and appearance changes stay in interaction mode until 150 ms of quiet. A single GPU frame may be in flight, with later input coalesced to its latest state. Scan-volume settings persist per source. Modeled removal and passive replay hide the original scalar volume so accepted cavities remain visible. Scan volume and annotation visibility have independent controls.

Camera persistence does not reapply unchanged layer materials. Cursor, visibility and opacity updates independently invalidate their own uniforms. The volume pass visits visible layer indices in original order, preserving overlap compositing and isolation behavior, including opaque and zero-opacity selections. Native visibility loads retain previously admitted layers from the same scan in document order, so hiding one while loading another does not evict its masks and meshes. Scan changes, revisions, corrections and original/preview changes still hydrate their own view.

The host supplies each scan's `ViewerWorkspaceSettings`. Layer color and opacity update existing surfaces and slice uniforms without inference or mask changes. The 3D cutaway clips segmentation meshes and the source plane in RAS millimeters, leaves original slices intact, and excludes clipped or fully transparent surfaces from cursor picking. It has no tissue-removal meaning. Camera position, target and up are emitted after direct interaction; saved cameras suppress late worker-driven fitting. Layout, plane visibility/orientation and contrast are also host-persistable.

Structure surfaces render as **Glass** or **Solid** (`structureGlass.ts`); the selected structure's inspector offers the choice, saved as `layerSurfaceStyle`, and a missing entry means the kind's default. Lesion and target kinds, ventricles, brain regions and context shells (brain, dura, CSF, skull, bone, scalp and head envelopes) default to glass. Vessels, cortex and unknown kinds stay solid. Glass alpha rises with the Fresnel term (1 − |n·v|)^2.5 from a facing alpha to the silhouette, and the rim emits in the structure's own color. A target uses its opacity as the rim alpha, with a facing alpha of 22% of it, so an opaque lesion becomes a glass shell. Other glass structures change only below full opacity, so **Opaque exterior** and other opaque choices stay solid. Transparent layers composite outer shells first, then inner shells and cortex, then other structures, then targets and vessels. Without the scan volume, each double-sided glass layer draws its back faces before its front faces and writes no depth; with it, the existing depth peels order them. Non-vessel surfaces add a camera-relative key and rim light and darken concave folds using screen-space curvature. Operative views and engine-modeled tissue keep their own finish. Picking, source triangles, masks and measurements do not change.

For acquired CT and CTA, **Scan** starts with bone visualized from the selected acquisition. **Appearance → Show 3D bone overlay** restores the skull and bone meshes for comparison at the current cut, camera and linked cursor. Other annotation surfaces and linked slice contours remain available in both settings. The preference persists per scan. Workspaces saved before this control retain their previously composed view when opened and pick up the acquired CT default when Scan is selected. Label isolation and structures-only display keep their bone overlays.

An optional `ViewerRegisteredFusion` is admitted only for the displayed reference case with a source-provided or verified registration and an explicit source-to-reference RAS transform. The second original grid is sampled through inverse source affine × inverse registration transform. Equal shapes or plausible appearance never authorize fusion. Outside the second scan's coverage, the primary image remains unchanged. Invalid replacements clear prior fusion before returning an error. This is grayscale source-image blending on MPR and the 3D plane, not volume raycasting or a registration algorithm. Source-provided registration remains labeled as such.

The **Appearance → Smooth surfaces** control starts on and persists per scan. Native meshes and the independent single-mask worker use the versioned Hann-windowed Chebyshev sinc policy in `src/resectionlab/surface_policy.json`. Tumors use 40 iterations / 0.03 passband and the smaller of 1 mm or 1.5 voxel altitudes. Independent-worker vessels use 20 / 0.1, bounded by half a voxel altitude and one quarter of local radius. Worker-generated context uses 20 / 0.1 with a half-altitude bound. Native vessel walls use local foreground depth, limited displacement and local balancing of normal motion to preserve caliber on the source contour. Shallow score crossings on one-sample strands and acquisition boundaries stay fixed. Unsupported or sheared radius estimates use a conservative one-eighth-altitude bound. The [native vessel geometry guide](../../../docs/native-vascular-geometry.md) describes source evidence, section measurements and the shared research wall.

In the sinc filter, only existing mesh neighbours interact. Open/nonmanifold edges stay fixed. Actual float32 output is checked for reversed/collapsed faces and new nonadjacent-triangle intersections, including coplanar intersections. Local backoff precedes connected-component rollback; rollback is rechecked against retained components. No components are deleted, bridged or decimated. Worker-generated context meshes above two million triangles retain their original source geometry with an explicit high-complexity fallback, bounding cold preparation; vessel and tumor meshes still receive the full safety checks. Subvoxel limits cannot repair false-positive or missing anatomy. Probability contours require agreement with the exact mask at the stored 0.5 threshold; edited or inconsistent fields fall back to the binary source.

Authoritative engine rehearsal uses the shared source ownership partition and the current accepted material masks. Raw mode draws the exact binary marching-tetrahedra surface. Conditioned mode places its same crossings on a piecewise-linear implicit field with a shared positive weight at each lattice node. The same coordinate transformation therefore applies to both sides of each material interface, including junctions across worker chunks. The interface-sharing requirement follows the discrete isocontouring literature. Here the existing triangles and source sample identities also survive the transformation. [VTK Surface Nets](https://vtk.org/doc/nightly/html/classvtkSurfaceNets3D.html)

The field uses local physical Gaussian coverage with a curvature correction. Short opposing-material runs retain source support at narrow creases. Weights around original vessel and membrane labels preserve their source boundary positions. Every crossing stays on its original source edge within half the smallest voxel altitude and 1 mm. Display normals combine the geometry normal with a local gradient of the current material indicator. The owner field stays fixed across a recorded cut or move, and the current masks determine which triangles exist. An ownership grid above the 32 MiB worker allowance restores source geometry for all its layers and reports the fallback. Picking, measurements and planning use the source representations. [3D Slicer segmentation representations](https://slicer.readthedocs.io/en/latest/user_guide/modules/segmentations.html)

On the generated folded-tissue control, mean normal error was 25.12° raw, 7.11° with the previous sinc filter and 6.47° with the shared field. The corresponding mean positional errors on the specified curved cap were 0.202, 0.069 and 0.107 mm. The shared field retained coincident positions at the 1,377 sampled material-interface crossings. These are analytic display-control measurements. Run `scripts/verify_curved_render_geometry.py` for the separate surface, groove and branch metrics.

Normal and mesh preparation stays in the worker. Smooth display vertices are an approximation of the voxel boundary: original images, annotation buffers, affines, slice overlays and native-cell measurements remain unchanged. Cursor raycasting always uses the original isosurface, even in smooth mode. Modeled-removal replay also retains its exact surfaces. Fully opaque anatomy writes depth so rear triangles cannot paint over its front; translucent layers retain their opacity controls. Coarse acquisition/segmentation steps can remain under the subvoxel displacement limit; smoothing adds no anatomical detail.

For an unedited Anatomy brain estimate on the same acquired CT or structural
MR grid, display conditioning can make a second pass of up to 0.4 mm per
vertex. It fixes crop-edge vertices, checks original signal contrast and label
support, retains the triangle indices, and rejects new self-intersections and
reversed faces. Prepared publisher surfaces and Rehearse ownership fields use
their existing cues and geometry.

An ordinary local native-wall rehearsal can use acquired CT or structural MR
signal to shade its brain estimate in the illustrative microscope. A
same-acquisition crop uses its original physical frame. A separate vascular
acquisition uses the reference signal sampled into the native wall lattice
under the saved review for that exact crop. Both paths require full sampled
brain coverage and the accepted wall binding. The albedo window uses the 15th
and 90th percentiles of the brain-mask samples. Trilinear sampling interpolates
only the windowed acquired signal. Gray end faces use those same samples and
identify where the model ends at the source volume boundary. Source-colour
mode retains the label palette and saved vessel colours.

Conditioned edges can round the free brain display surface on qualifying
native-wall grids. The worker pins the crop, protected tissue interfaces and
the vicinity of accepted edits. The candidate keeps the accepted triangle
indices and uses a physical displacement bound of at most 0.4 mm from the raw
source surface. Reversed faces or new intersections restore the existing
surface. **Source edges** displays the original indexed extraction. Linked
slices, physical tool contact, and the source-indexed vessel wall retain their
existing geometry.

An already-modeled vessel keeps its score-derived lighting and the gray source-volume boundary markers when the viewer starts or rebuilds. The local matched anatomy and microscope record checks the indexed position, triangle, and owner binding when the case supplies that binding.

## Estimated envelope inspection

A selected structural proposal uses a separate texture and a lavender dashed contour on the three original MRI planes. It never enters the target compartment list, removed-tissue mask, working brain mask, 3D anatomy, or planning metrics. The display gate requires an estimated, view-only record bound to the exact case version, native dimensions, binary cells, and canonical RAS affine. LPS conversion preserves source voxel indexing. Review-required, accepted, and rejected estimates retain distinct status labels; a free-form description cannot change those labels or permit cortical access. Clearing/changing evidence disposes the previous texture before validation; replay suppresses the contour. Source-coordinate sampling determines the contour's physical alignment without resampling the stored mask.

## Population prior inspection

A selected registered population prior has separate value and atlas-coverage textures on the original MRI planes. The viewer preserves its explicit RAS affine, accepting only the backend's 0.01 mm corner tolerance relative to the native source grid. It does not replace that affine with the source affine. Functional concordance uses trilinear samples only when every contributing corner has coverage; released binary masks use nearest-cell samples. A covered zero is a real atlas value. Missing coverage or incomplete interpolation support produces no value and a neutral hatch. The shared CPU cursor sampler follows the same rules as the GPU shader.

The muted blue-to-sand palette is unitless, from zero to one. Atlas concordance is not an individual patient's functional status, injury risk, or clinical deficit probability. The alignment-review-required label and cursor coverage readout remain visible in the MRI review and expanded layouts. Priors never enter target compartments, brain estimates, removed tissue, route scoring, or 3D anatomy. The host validates saved source and evidence hashes; the viewer separately checks case version, source frame/grid, value semantics, and the view-only flags. Source changes, rejected props, cleared selection, replay, or structural-envelope inspection clear the previous prior texture before any replacement is shown.

CPU and GPU sampling share an explicit display precision convention. Each installed grid gets a cached, per-axis float32 tolerance derived from its inverse affine and maximum absolute physical coordinates over the native cell field. A signature of all 19 shape/affine numbers invalidates the cached budget after replacement or in-place edits; unchanged cursor samples reuse the preparation. The allowance is `16 × 2⁻²⁴ × (1 + |inverse translation| + Σ |inverse coefficient| × maximum |world coordinate|)`, with a floor of `2⁻²⁰` voxel; the resulting tolerance itself is stored as float32. This accounts for coordinate/coefficient quantization and matrix arithmetic. If any axis needs more than 0.001 voxel, the prior is withheld instead of widening the band.

Within this bounded band, functional samples snap to integer source centers; binary samples snap to half-cell boundaries with upward ties. Every positive interpolation weight outside the axis bands still requires atlas coverage, even when the weight product is tiny. Both samplers abstain within the tolerance of an outer half-cell field face and identify this as display precision uncertainty; this deliberately withholds the exact lower face that an ideal half-open field would include. No source cell or affine is changed. This numerical display convention is not anatomical registration accuracy or clinical uncertainty.

## Unexecuted neighboring-tool inspection

`inspectionTool` is a separate `ViewerInspectionTool` prop: the host-validated complete backend report, selected accepted action ID, and expected case, planning, binding and inspection hashes. The host validates the response and user-selected access/tool assumptions; the viewer checks source-grid and canonical RAS affine equality, model/cavity joins, accepted action membership, proposal/attempt/ledger geometry, and the exact bound instrument catalog. The current volume must supply its planning hash for this feature. This is a display integrity gate, not another geometry checker or a clinical certificate. Partial/rejected/STOP records, stale identities, execution or removal claims are withheld. The display snapshot is detached and deeply frozen.

The dedicated blue tool group contains the entire proximal shaft capsule and distal active-tip capsule. Linked MRI uses the same physical capsule endpoints and radii in the existing two-slot shader. Inspector coordinates are already RAS and bypass route/source coordinate conversion, including for LPS source images. The unexecuted-preview label and that selected preview's outside-image uncertainty remain visible; another action's warning is not attached to the selected tool. No cells or tissue-effect meshes are drawn. The host can clear through `onClearInspection`; a local clear also removes the overlay immediately.

A/B selections and their display geometry remain separate and restore when inspection clears. Updates to A/B while inspecting are retained. Invalid replacements dispose the old tool before validation. Case changes reject stale reports. Replay, structural-envelope inspection and prior inspection suppress the tool and notify the host to clear it; leaving those modes does not resurrect the same inspection identity, even in a cloned prop wrapper. Pass a null selection before explicitly selecting that same action again. Selection does not move the camera or linked cursor. The existing fit-instruments control includes the inspection tool when it is displayed. No application wiring or route-selection behavior is changed in this viewer-only slice.

`inspectionTool.test.mjs` exercises source identity, exact RAS/LPS alignment, oblique/mirrored grids, whole-tool endpoints and mesh bounds, immutable snapshots, accepted/rejected/fallback joins, loss of uncertainty, A/B restoration and replay suppression using small analytic fixtures. These checks do not establish GPU appearance, patient access feasibility or independent motion certification.

## Modeled removal display

The host first verifies the independent native replay certificate. `validateReplay` then checks the accepted effect's case version, grid, RAS affine, binary cells, and target/normal/residual cell accounting. It is a display integrity gate, not an independent trajectory checker.

- Original MRI and source annotations remain unchanged.
- Colored replay surfaces show source annotations minus accepted removed cells.
- The mint mesh and MRI overlay show only accepted modeled removal. They do not show accessible volume or partially contacted tissue.
- Candidate approach tools are hidden during replay because their terminal poses do not establish the replayed motion.
- A new replay cancels pending surface work. Failed or cleared replay restores source anatomy; stale worker results cannot replace current geometry.

`coordinates.test.mjs` covers interpolation, orientation, physical bounds, watertight surface geometry, union accounting with overlapping annotations, immutable source data, LPS replay alignment, rejected invalid effects, and zero-removal STOP.

The October 4, 2026 integration check used `outputs/desktop-bridge/native-ucsf0004-v2/candidate.json` and the UCSF-PDGM-0004 source arrays: two accepted strokes, 249 mm³ target plus 17 mm³ normal modeled removal, 41,670 mm³ residual target. Viewer checks agreed exactly. On the local development machine, validation took 79 ms, residual copies 42 ms, and removal mesh generation 39 ms for 8,928,000 source voxels. These are one-run implementation timings, not clinical or throughput benchmarks. The underlying geometric and tissue assumptions remain those declared in the saved replay.

## Unified native workspace

The main application accepts native patient collections and historical cases through the same Open and library flow. Native collections hydrate their original scalar grid, shared categorical label sets and indexed source/display meshes. The anatomy inventory is independent of residency: visibility changes on resident layers do not retransfer source images or rebuild the renderer. Missing layers load serially in the background, with per-layer progress/retry and latest-selection handling. Label packing uses a worker and bounded source-buffer copies. Validated additions update the current renderer atomically; failures retain the admitted view. Native source/revision/preview generations invalidate residency and reject late responses. One scalar GPU texture is shared by source slices and volume rendering. Camera-only frames composite cached source slices; layout or image/overlay changes invalidate the affected cache. A new source or revision invalidates pending loads; one scalar image is retained for reuse during same-scan edits.

The native label worker uses one or two bytes per voxel when the resident labels
fit in R8 or RG8. Every bit mask and categorical value retains its original
membership, including overlaps. Larger inventories use RGBA8. When new resident
layers require more channels, the renderer updates the source slices, volume and
vessel thickness sampling together. Modeled tissue states retain the RGBA8
layout for sparse updates.

The layer texture packs independent binary masks in byte banks and maps categorical regions through shared byte palettes. Four channels retain overlapping groups without allocating a full mask per brain region; the current 38-layer patient collection fits in two channels. The metadata limit is 128 layers; four texture channels bound resident label groups. Invalid groups are reported rather than silently omitted.

Layout, Appearance, Clip and Fit are the primary toolbar controls. Per-scan view state is saved beside the native collection in `workspace-view.json`, separately from anatomy identity. Fragment review is bound to the exact starting mask and snapshot; it highlights user-selected components and requires a correction commit to change anatomy. Quantitative masks and model outputs remain unchanged by rendering, visibility, framing and preview rejection.


## Patient studies and scan-independent structures

Structures and Vessels share one patient RAS reconstruction across the study. Switching linked acquisitions preserves the registered structures, their appearance, the shared cutaway and the patient camera. Each structure uses its full source-grid mesh. The nearest-neighbor overlay on a selected slice covers only that scan’s acquired field.

All presentations use the selected acquisition’s scalar samples and physical spacing for 3D ray marching and linked slices. The shared patient bounds support clipping and framing. The source list records acquisition hashes, native geometry, transform status and exclusions. Scan-specific transfer controls stay with their acquisition when switching sources.

Older saved Structures and Vessels views with shared patient context used a normalized composite signal. When opened, their authored scalar preset, window and transfer points revert to the selected acquisition’s automatic transfer. Density, shadows, appearance and physical cutaways are retained. An older shared view set to show structures only keeps its scalar volume hidden. Scan and independent-review transfer settings keep their original values.

On a native CTA with vessel layers, new Structures and Vessels views start with **Vessel shading → CTA contrast · by vessel type**. Declared arteries use red, veins and sinuses use blue, and unclassified vessels use muted violet. Lower clearcoat and specular intensity reduce pale highlights in both direct surface and volume-composited views. **Source layer colors** restores the branch palette. Editing a vessel’s label color selects that palette. Linked slice outlines use the same vessel-type or layer color as the 3D surface, and the acquired Scan view keeps its own appearance.

**Appearance → Vessel depth lighting** shades a vessel where a nearby course
passes in front of it. The viewer caches the nearest opaque vessel depth at
the current physical camera and uses it in both the live view and the PNG
export. An acquired CT cut uses an angled key to bring out ridges in the
source signal. The lighting leaves source slices, contours, vessel type,
opacity, picking and geometry intact. The control can be turned off per scan.

The conditioned publisher CTA arteries use an additional bounded pass on the
lighting normals to soften fine highlight banding. The exact publisher view
uses the acquired source geometry and its source normals.

On first open, a native CTA with a compact visible arterial field and bone from the same scan can start at a superior axial cut. The plane clears the measured extent of the original and conditioned arterial meshes. The linked slice cursor remains at the selected source location. **Full head** clears the cut and frames the complete acquisition. A broad or incomplete arterial field, missing bone support, or a real visible lesion above the plane leaves the full field on first open. Saved camera and cutaway settings take precedence.

Source-bound unreviewed rigid transforms are labeled as alignment previews. Missing, stale or incompatible transforms exclude a scan from the patient reconstruction. The scan remains available in its independent frame. Corrections, planning and measurements use their own geometry checks.

In mixed **Vessels** with sourced cortical context and a structural MRI
selected, **Show scan volume with structures** starts at a density of 0.02.
This lets the acquired outline remain visible while reducing overlap with the
displayed vessels. **Appearance** offers manual density and transfer controls.

New multi-scan studies, studies with an identified cortical label, and angiograms with a source-matched artery label start in **Structures**. Other single acquired scans start in **Scan**. A saved study restores its recorded appearance. **Scan** displays the original scalar field with modality-specific transfers for MRI, CT and bright-blood imaging. Linked slice panes display the original scan and the outlines described below.

Linked slices outline the study target and the selected structure. The target is any layer with a declared lesion kind: the engine's lesion candidates and the source tumor compartments. A layer without a declared kind is itself the supplied annotation and is outlined too. Names and colors are not evidence. Other visible layers keep their 3D surfaces but stay off the slices, so a brain envelope, skull or vessel tree does not trace over the acquired image. The acquired CT and CTA signal is unchanged. **Appearance → Show all structures on slices** draws every visible layer again, with non-target outlines fainter; the choice persists per scan as `sliceStructures` and follows linked scans. Outlines are solid and unfilled, about 1.5 CSS pixels wide in whole device pixels, and use the color the 3D surface shows, including user color overrides and CTA vessel types. The overlay opacity control can still fade them. Views saved without a slice style open with outlines; **Filled labels** remains an explicit choice. The 3D source plane keeps its existing label tint. Slice pane notes use sentence case, for example *Source CTA · modeled effect overlay*.

**Structures** recognizes a cortical gray-matter label from its declared type or model ontology. With that label, it displays the folded surface and target inside a subdued head outline. The renderer computes the nearest visible wall of the head and cortex separately, using the same physical camera and clipping planes for each group. This avoids building opacity from successive walls of one tissue. Surface shading uses local depth changes over a projected 2 mm neighborhood and samples each cortical layer’s own acquisition at its physical position. Switching linked slices from MRI to CT keeps the cortical MRI samples, contrast window and camera. Large sources use the same reversible display pyramid in either view. An independent source shows its own supported layers. The structures-only toolbar control retains those boundaries with scalar volume rendering disabled. On a CTA or MRA with a source-matched artery label, Structures leaves the brain and skull translucent so vessels remain visible. For a study with only whole-brain and bone envelopes, Structures shows those surfaces. **3D surface** compares the original and conditioned surfaces at the current camera. Display lighting and depth selection operate on the visual mesh. Planning, measurements and collision checks continue to use their source representations.

At an explicit cut, the displayed 3D source-image plane writes its actual depth so scalar tissue behind the section cannot veil the acquired image. Primary, box and oblique clips apply consistently to that plane and its picking. Uncut reference planes retain their translucent appearance; linked 2D image panes remain unclipped.

In **Vessels**, choose **Vessel slab at cursor** in the 3D toolbar to focus on the depth around the linked cursor. Use **Clip** to change the anatomical plane or thickness in millimeters. The slab retains the full transverse field and the linked slices keep their full coverage. Toggle the toolbar button or choose **Exit slab** to restore the prior 3D cut.

Choose **Faint context** to see distant courses from the currently displayed vessel surfaces. This background follows any prior cut and fades with distance from the slab. The foreground stays fully shaded, and 3D picking remains inside the slab. The same presentation is available in the 3D PNG export.

On a native TOF or MRA with a source-bound vascular layer, **Vessels** balances
the automatic full-depth backdrop using the acquired signal. It estimates
brightness along 243 sampled rays in physical millimeters and blends the volume using
a density bounded to 0.008–0.035. The estimate is fixed per selected source
and transfer, so it stays steady while orbiting. **Balance TOF context** in
Appearance restores the preset after a manual density adjustment. Scan,
physical cuts and slabs, isolation, and authored transfers use their stored
volume settings. Choosing Vessels retains the selected source or conditioned
surface.

When Vessels has visible, admitted MR cortex and a primary vascular layer, it
uses the nearest cortical surface for context and samples that surface from
its own structural acquisition. The 3D volume starts disabled. **Show scan
volume with structures** enables the active acquisition at the same camera and
source cursor. Saved views restore their recorded volume and appearance choices.

With one translucent source-bound lesion in mixed MR **Vessels**, **Appearance →
Show vessels through lesion** displays its closest surface with a visible rim.
This reveals more of the vessels in projection while retaining the lesion
geometry, contours and estimated-alignment status. The inspection resets on
leaving the source view. The warning banner keeps a failed forward/reverse
check visible when multiple acquisitions report warnings.

Settled Scan uses ray steps of one source spacing and at most 640 intervals. Lighting uses two scales of cached optical density for softer key-light shadows and ambient occlusion. The cache maps source values to extinction before averaging at roughly 4 mm spacing, capped at 56 cells per axis. It includes source support and the physical gradient weighting used for automatic MRI shells. Preparation runs in short tasks after the first draw, then restarts accumulation for the current acquisition and transfer. Export completes any pending cache before its first sample. Shadow probes honor active clipping planes. Label isolation and explicit CT skull overrides use exact source membership for each probe. The largest cache occupies 1.76 MiB, including the transfer row. Primary ray samples and linked source slices keep their existing resolution. The colored fill approximates scattered light.

Transparent CT context uses a centered optical footprint when a single visible, independent bone mask shares the selected acquisition's display grid. A symmetric six-face filter centered on each voxel reduces bands from the discrete mask. Face offsets are bounded to 1.45 cells and 1 mm along each source axis. The ray interpolates the resulting byte coverage grid and the shadow atlas uses the same grid. Geometry, measurements and linked source panes retain the original mask and acquired samples. Scan, exterior bone, isolated labels, multiple bone contexts and scans with native support keep their existing path. The optical texture needs one byte per displayed voxel and reuses the auxiliary sampler.

In a physical CT cutaway in **Structures** or **Vessels**, the declared skull mask still controls where to style bone. Opacity builds over physical distance from the acquired mineral signal. For a single same-grid skull mask, the cutaway also uses the centered coverage grid to avoid abrupt opacity steps. Its shadow probes sample the retained side directly. Linked slices and the source-image section keep the selected acquired image.

With standard CTA transfer settings and one physical cut, an independent
same-grid skull mask also supports a shallow band of acquired soft-tissue
contrast. The volume samples an additional point 0.75 mm into the retained
side. The resulting gradient weights tissue extinction over the first 6 mm,
with little contribution from uniform signal. Tagged skull context keeps a
separate color response to acquired attenuation. Eye and shadow rays use the
same term. This is a display treatment, with no tissue masks or categories
derived from the cut shading.

The same standard CTA cut keeps a density-scaled contribution from high
acquired attenuation outside the tagged skull fraction. The contribution
ramps from 440 to 850 HU and is capped at 0.32 per mm. This helps dense
source detail remain visible where the estimated skull coverage falls away.
The skull hue still belongs to the tagged fraction. The visible ray and its
scalar shadow probes share the transfer. Custom transfers and uncut views
keep their own settings.

With **Vessel depth lighting** enabled on a qualifying acquired CTA cut, the
volume and the 3D surfaces share the same directional key. Vessels from the
selected scan receive a bounded direct-light shadow from two probes into the
retained CTA and its same-source skull context. Their visible opaque surfaces
also cast soft shadows onto the acquired volume. Four light-space depth
samples soften the shadow, with a 28 mm maximum caster distance and the same
physical clip as the displayed meshes. This affects only direct illumination.
The independent skull mask lets this pass use the packed-label sampler slot
for light-space depth. Opacity, geometry, and linked source slices keep their
existing paths. Samples inside the exact stock
transfer's zero-opacity range can skip compositing when the cut's mineral and
soft-tissue terms are also zero.

With a ready source-bound atlas and no cut, two fixed RAS light directions
integrate acquired optical depth to the edge of the field of view. The atlas
stores the total and separable context contributions so the scalar light
still follows bone opacity or the MR boundary weighting. A single atlas
lookup supplies both directions. Cut and isolated views use the local probes
and clip-aware paths so removed signal cannot cast a shadow into the cut.

When that atlas is ready and the view is uncut, the secondary probes read
it directly. For non-CT scalar views without label isolation, the ray skips
transfer lookups only inside the exact zero-extinction prefix of the active
transfer. It still samples the acquired signal on the original ray schedule.
The shortcut respects the authored transfer and source support, including at
physical cuts. CT and label-isolated views retain their context evaluation.

An acquired CT or CTA can qualify a detached posterior sheet for exclusion
from the scalar context in the full-depth Structures and Vessels views. The
component must separate from the main foreground at −500 HU, have a broad
boundary-spanning shape and clear the same-grid anatomical labels. The
detector leaves connected, dense or ambiguous components visible. Use
**Appearance → Show detached CT signal** to restore the full CT signal in
the same view. Scan, linked slices, cuts, isolated labels and custom transfer
curves use the full acquisition. The eye ray and optical cache share the
same display-only air-filled derivative. The native values and structure
surfaces remain available through their original paths. The derivative is
allocated when first used and capped at 128 MiB in each of CPU and GPU memory.

With shadows enabled, eligible transparent CT views reuse the same optical
atlas for total extinction and the fraction associated with the bone mask.
The renderer applies the visible opacity after interpolation, so changing it
does not rebuild the atlas. The settled view integrates a directional key
over physical distances and averages two short ambient directions. A probe
whose cache footprint crosses a cut uses the acquired signal and exact
context treatment at that retained location. The uncut shader specializes
its secondary probes to the atlas. The visible ray still uses native samples.
Opaque, isolated and multi-mask overrides keep the exact shadow path.

**Export 3D PNG** in the 3D toolbar captures the current anatomy camera, clipping, and layers at a 2048-pixel long edge, bounded by the GPU's attachment limits. Scalar-volume exports average 32 jittered samples per pixel using finer physical ray steps. Surface-only exports use the same full resolution. Guarded image bands keep the additional render attachments below 160 MiB. The displayed 3D pane, source slices and their interactive settings stay at their usual resolution. The export can be cancelled from the same toolbar button, and editing the view cancels a capture in progress.

The viewer emits `anatomy-export` upon completion with image dimensions, sample count, CPU submission time, GPU query results where available, fence-poll wall time, readback, PNG encoding and estimated peak export attachment bytes. The attachment estimate covers color, depth and multisample buffers owned by the export. Browser image storage, existing anatomy resources and driver overhead are outside that estimate. `anatomy-frame` reports the live volume dimensions and sample count in `scanQuality`.

**Structures** and **Vessels** leave the active acquisition as the scalar volume and display the selected registered anatomy alongside it. When viewing CTA, the acquired volume supplies fine bone and contrast detail while opaque vascular meshes remain selectable. Live scalar draws settle at a 1024-pixel long edge. New vessel-focused CT views with one bone layer from the same acquisition start with volume shadows enabled. They retain the existing shadow control and saved view settings. In native TOF, subdued context lets unclassified vessel candidates use a separate palette from supplied arteries, veins and sinuses. Conditioned vessel walls have wider, weaker highlights and a camera-depth cue. Acquisition edge closures keep their neutral material and geometry.

In **Anatomy → Combine supplied arteries**, build a display from focused and wider supplied annotations on the same CTA. **Focused branches** displays the original selected layers at the same camera, slice cursor and cut. A fresh CTA opening follows the focused branches when supported skull context is available and at least 90% of both combined mesh vertex sets fit below the physical plane. **Full head** exposes the full field. The display uses the exact positive cells from the two groups. An accepted correction to any input retires its combined view. [Prepare the display from supplied annotations](../../../docs/data-rebuild.md#combined-supplied-cta-arterial-coverage).

To reproduce a matched capture, pass one native `.json` study and the same view recipe to both desktop builds:

```sh
cd desktop
pnpm exec electron electron/capture-vascular-surfaces.cjs \
  --anatomy "$manifest" --views "$recipe" --output "$new_capture" \
  --expected "$baseline/report.json"
```

The hidden harness verifies the reference source, exact layer selection, camera, cursor and clipping before comparing views. Its receipt includes GPU completion, CPU submission, asynchronous GPU time where available, sampled Electron memory and acquisition hashes. Use `--export-view` with a recipe name to exercise the full PNG workflow, or `--width` and `--height` for a compact-window check. [Local comparison index](../../../docs/local-storage-ledger.md).

New native studies choose Pre-op or Post-op. Import rejects a different timepoint before adding source files; the same patient can have separate study files. The portable `.rsworkspace` archive includes the native manifest, referenced original imaging and annotation arrays, and saved view. It omits model runtimes and mesh caches. Open verifies source checksums and unpacks into an editable local working copy; Save updates the opened portable file. Native collection folders remain supported.

**Vessel display → Within brain** intersects the displayed vessel estimate with a filled, aligned brain mask. It does not delete the source, join components, certify vessel identity or change planning masks/volumes. Without an aligned brain mask the UI reports that it is showing the full estimate. Fragment inspection restores the full source, highlights chosen components and retains preview/undo/reject/accept correction behavior. Native context meshes use the existing categorical Surface Nets or validated continuous probability isosurfaces. Windowed-sinc movement is bounded to 0.7 source-cell units and 1 mm of physical displacement. The cell-relative bound accommodates coarser axes without increasing acquired resolution. Native lesion meshes retain the 1.5 minimum-altitude or 1 mm cap and restore any closed component whose final displayed mesh volume differs from its source mesh by more than 1%. Vessel smoothing keeps its radius-dependent cap. The source shape and triangle connectivity are retained separately. The contact and face checks reject newly intersecting nonincident triangles and locally reverse changes that flip or collapse a face. A native context mesh above 4.5 million triangles retains its source shape and reports the audit budget fallback. Mask volumes, intersections and planning clearances use the original analysis geometry. Smoothed native brain and skull use an oblique key and bounded physical depth falloff to make surface shape easier to see. Original surface mode retains its original lighting.

**Attached vessel uncertainty** supports a reviewer-selected local erasure
in editable native TOF studies. The draft and its accepted revision reuse
the original score field, with excluded source positives reflected below the
0.5 contour. Edges without an edited endpoint keep the same native crossing.
The viewer compares both states at the same camera. Conditioned display can
still move nearby vertices within its usual bounds.

Retina slice-cache targets use physical texture dimensions and logical viewport dimensions; multiplying the pixel ratio twice would crop the original slices. The regression test checks this at 2× scale.


## Source quality and validation

Anatomy quality findings are bound to the source snapshot, mask, affine and probability field. Empty masks, 26-connected components, image-edge contact, threshold inconsistency and unreviewed alignment are shown beside native layer details and under Appearance → Anatomy quality. Surface diagnostics identify local backoff and source-restored components. No finding approves anatomy or performs a correction. Existing explicit preview/undo/reject/accept revision tools remain authoritative.

The engine reuses one verified native mask for quality checks, volume, contour extraction and transfer when the displayed layer uses that mask. Large source-only contours use a lossless uncompressed cache with one stored position array. The viewer receives the full indexed contour and acquired grid. Existing compressed source caches load through the same view path.

`scripts/audit_anatomy_display.py` audits local collections, portable studies and saved cases, including symlinked case libraries. Mesh derivatives and checkpointed receipts go under the requested output folder; source file hashes must remain unchanged. `scripts/verify_rendering_rl_parity.py` compares fixed headless RL episode fingerprints, retaining admission rejection for missing targets. `desktop/electron/verify-surface-quality.cjs` uses a hidden window and disposable study copy. It fits the source geometry before each pair, waits for matching GPU completion, checks that source-slice pixels stay identical, and reports renderer timing and peak memory. Provide a fresh local output directory and a native study or portable case. Set `SURFACE_VERIFY_FOCUS=1` for isolated structure captures, `SURFACE_VERIFY_RETAIN_CACHE=1` to retain the disposable mesh cache, and `SURFACE_VERIFY_CACHE_FROM` to repeat a warm-cache run with the retained cache. A report supplied through `SURFACE_VERIFY_EXPECTED` checks matched camera, source slice signal, original mesh and mask hashes, topology and source volumes against the baseline. The figures demonstrate software and display behavior. Anatomical accuracy is unmeasured.


In native **Operative Steps**, **Operative microscope** approaches the saved entry on the registered exterior, then holds the optical field through opening, retraction and aspiration. Drag to orbit, scroll to zoom or press Home to restore the current framing. The scale bar measures the focus plane. **External anatomy** fits the surrounding registered surface. **Inspection cutaway** shows a longitudinal section at the same timeline position. Its optional registered scan context ends at the sampled operative field. The linked route slice marks the original target and the current station.

**Field evidence** offers two geometric aspiration paths. **Overlapping passes** is the default. **Original spiral** replays the earlier path. Switching paths pauses playback at the current timeline position. Model-derived unclassified vessels use the matte plum **Vascular estimate** swatch. Other unclassified labels use **Unclassified vessel label**. Identified artery, vein and sinus classes retain their own materials. A model-derived dural layer is labeled **Dural estimate**, and a proxy-only layer is labeled **Dural proxy**. **Field voxel faces** shows the cells of the sampled display grid, including the full dural label thickness. Click a tissue surface, or focus the canvas and press Enter, to inspect its source and rest-space coordinates.

For an engine recording with a verified approach, **View down approach** frames the opening from the entry side. **Inspect recorded corridor** places a longitudinal display cut through the recorded entry and target. Both use the currently displayed engine state. The whole-head pane and source slice beside it remain intact references. **Inspect result in 3D** opens the cavity and residual from the displayed engine observation. The geometric preview and the engine recording retain separate histories.
