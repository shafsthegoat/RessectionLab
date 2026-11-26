# Acquired vascular source geometry and connectivity checks

## Native vessel wall reconstruction

Native anatomy review keeps the acquired grid, mask, score array and extracted
source contour. Label layers use VTK Surface Nets. Mask-compatible score layers
use VTK Flying Edges at the stored half-score crossing. The conditioned surface
uses the same triangle indices. A physical-space fairing balances locally
coherent normal motion to reduce caliber drift while smoothing the silhouette.
Local foreground depth bounds the movement and protects one-sample strands.
Weak, shallow score crossings on those strands keep their incident faces.
Acquisition edges and nonmanifold mesh edges stay pinned. The physical audit
checks for reversed or collapsed faces and newly intersecting nonincident
triangles after final position quantization.

The research wall keeps ownership from the original native crossings. Its
positions and triangle indices provide the canonical wall for admitted
collision and rendering. Newly prepared native-wall rehearsals use
`source-conditioned-v2`. It keeps the original crossings for a wholly interior
component with at most two 26-connected positive sample centers when the
triangle audit accepts the replacement. Eligibility comes from the complete
original interior component, so it excludes crop edges and fragments created
by edits. Existing `source-conditioned-v1` cases and replays keep their original
reconstruction. The source shape remains available for inspection. Measured
sections describe the mesh derived from the score field. Vessel identity and
lumen caliber remain unresolved at the scale of two positive samples.

Use `scripts/measure_vessel_surface_sections.py --compare-previous` with the
native study and matching source-branch JSON. The receipt pairs sections on
the same centerline stations and records displacement, geometric normal turns,
surface area and newly introduced contacts. Its temporary mesh data stays
beside the requested receipt. The HDD pilot in
`qc/native-vessel-geometry-20261124-v1/` includes unlit, wireframe and flat
shaded comparisons for the supplied TopCoW-014 artery and vein/sinus and two EDEN
TOF crops. The same receipt directory holds a per-case table for all 29 acquired
EDEN TOFs at the frozen native crop bounds. Across the crops, the median case
95th-percentile absolute section area change fell from 18.5% to 2.6%. The
physical audit found zero introduced nonincident contacts.

## Supplied CTA surface display

**Smooth 3D surfaces** applies bounded fairing to the complete publisher CTA
surface in its native physical frame. Named branches take their display vertices
and normals from that same surface using the publisher face and vertex IDs, so
the class seams stay coincident. Disabling the control draws the exact supplied
positions. Missing or mismatched partitions are shown in their exact source form.
The native mask supports the displacement bounds. Original mask, mesh indices,
source picking and the research wall use their existing source representations.

On orthogonal CTA grids with same-series venous or sinus labels, the
arterial display can use a second pass guided by adjacent face normals. The
pass holds fixed vertices near native venous samples, nonmanifold edges,
source borders and small isolated mesh components. It checks the rounded
display vertices against the existing source displacement envelope and both
triangle intersection controls. An unavailable adjacent label leaves the
earlier conditioning available.

Across TopCoW-006, 012 and 014, the upper decile of measured face-angle turns
decreased. The 556 sampled sections still matched, with small increases in
section-area differences from the exact supplied mesh. The three cases had
zero newly introduced arterial–venous triangle contacts. The viewer uses a
longer, angularly bounded lighting-normal pass on the newly refined surfaces.
At a fixed camera this reduces mottled highlights, most visibly on close-ups.
The coarse stair steps still follow the supplied contour.
[Matched publisher validation](local-storage-ledger.md)

## Default connected vessel display

The desktop opens vessel layers as a **Connected tree** (`vesselScope:
'rooted'`). `resectionlab.vascular_rooting.root_vessels` derives it per layer;
nothing in the source mask, labels or manifest changes. Every source voxel is
in exactly one of three outputs: shown, review (components that reach no root
and were not bridged) or hidden by level of detail. Bridge voxels are a fourth,
separate output with their own provenance. Tests check the exact partition and
that the shown display has no component without a root.

- **Roots.** TopBrain v1 named labels where the layer has them: basilar (1),
  internal carotids (4, 6) and vertebrals (23, 24), plus on CT the vein of Galen
  (35), straight sinus (36) and superior sagittal sinus (40). Label numbers
  come from the TopBrain label maps (`labels_topbrain_CT.json` and
  `_MR.json`); CT and MR maps are never applied to the other modality. On MR the
  external carotid branches (35 to 42) are not roots; those that do not reach a
  root stay in review and the receipt lists them. A supplied TopBrain
  annotation is bound only when the layer equals the imported original's listed
  label values on the scan grid and the original is identified as a TopBrain v1
  map by its label dictionary or by its file name on the TopBrain record; when
  both are present they must agree on the modality. A layer may also carry `named_labels: {data, scheme}` for model
  ensembles. Without named labels the roots are the largest 26-connected
  component plus any component that enters through the inferior face of the
  field of view with a calibre of at least 2 mm (widest over an even sample of
  the component) and an extent of at least 10 mm (the other carotid tree or a
  separate posterior circulation); the receipt records the fallback, each
  accepted root and each refused entering component with its calibre.
- **Bridging.** A detached component of at least 0.5 mm³ is joined when a
  geodesic path of at most 3 mm (voxel centre to centre) reaches the shown tree
  through voxels that are all supported. A voxel is supported when its signal
  is at least 0.5 of the local lumen contrast above background, or when a model
  probability reaches 0.25. Lumen and background are fixed physical
  quantities, not properties of the search window: within 6 mm of the midpoint
  of the closest fragment-to-tree voxel pair, the lumen is the median signal of
  the tree and fragment voxels and the background the median of the 1 to 3 mm
  shell outside the mask; the signal criterion is used only when their
  difference exceeds three robust noise widths of that shell. The search
  window always contains this neighbourhood, and a test checks that the
  decision and the recorded contrast are unchanged in a larger window. A supported voxel is then vetoed when its signal is
  above twice the local contrast, above 700 HU on CT, or inside a same-grid
  bone or skull layer, so probability cannot carry a bridge through bone. A
  fragment joins only tree voxels of its own class (artery or vein, from named
  labels or the layer kind); an unclassified fragment may join either. The
  search window covers only the fragment voxels within 3 mm of the tree, padded
  by 6 mm. The path is widened to the smaller end radius (the deepest lumen
  within 0.6 mm of each end, less half a voxel, at most 1 mm) through supported
  voxels, keeping only the piece that holds the path. Up to three rounds let a
  chain attach one gap at a time. Bridges take the class of the tree voxel they
  join. Bridge paths are face-connected (6-connected), because the drawn
  surface joins voxels only across faces. Thresholds were frozen on TopCoW-014
  and PTNT_4509 only.
- **Face connectivity.** Every shown part must be face-connected to a root.
  Where shown parts touch only at an edge or a corner, a display-only
  connectivity fill adds the minimal face-connecting voxel (one for an edge,
  two for a corner) with the highest probability or signal, vetoed by the
  bridge rules; if every candidate is vetoed the far side goes to review. Fill
  voxels are counted separately in the receipt and never enter a source mask.
- **Level of detail.** Calibre is lumen diameter from the lumen volume inside a
  1.5 mm sphere, inverted for a cylinder through the centre, plus 0.1 mm, and
  taken as the widest within 1.5 mm. It does not depend on how the centreline
  sits on the voxel lattice: 1.0 to 1.3 mm branches are kept at 0.3 to 0.6 mm
  spacing and 0.6 mm reads thin at 0.3 and 0.45 mm. At 0.6 mm spacing a 0.6 mm
  vessel can read up to about 0.85 mm. Thin branches within about 2 mm of a
  wider vessel read wide. Calibre 1 mm and up is always shown; 0.5 to 1 mm only
  within 10 mm of a tumour or of a saved corridor capsule. A tumour drawn on
  another registered series of the same session is resampled into the vessel
  grid; without a registration the receipt says so (`lod.tumourReason`). Named
  vessels and roots are exempt. Only thin terminal subtrees are hidden: a thin
  piece that joins two kept parts stays, so the shown tree never disconnects.
- **Colour.** Every vessel is drawn in one red, in the standard and Volume
  lighting paths (owner's decision, 2026-11-26). Arteries and veins are still
  told apart in the receipt, and when a layer holds both classes its
  other-class voxels (and bridges joining them) still form a separate
  compartment, `__vessel_veins:<layer>` or `__vessel_arteries:<layer>`, in the
  same colour. The receipt notes when vessels are unclassified.

**Show detached and thin parts** (`'rooted-review'`) adds each layer's hidden
voxels as an amber overlay, `__vessel_review:<layer>`; a version 3 view saved
with it on reopens with the overlay on. Derived compartments
follow their layer's visibility, select their layer when picked, are never
saved as layers and give way before the viewer's 128-compartment budget. Saved
views now carry `vesselDisplayVersion: 3`. Views without a version, and version
2 views that stored `'all'`, open on the connected tree; version 2 views that
stored another filter keep it. Native-adaptive TOF studies, which used to open
on the TOF exterior-air review, also open on the connected tree; that scope
remains selectable. **Show full source** remains available. If the derivative
fails for one layer, that layer shows its full source with the reason in its
display filter. The derivative and its receipt are cached per session like the
TOF exterior-air filter; saved corridors and tumour registrations are part of
the key, and a view built while a supplied label original was missing is not
cached. Memory-mapped signal and probability are read once in large sequential
blocks over the box around candidate fragments, as float16. A bridge voxel
outside the source mask makes that layer's wall use the categorical contour
rather than the score contour.

CPU check on the supplied annotations (named roots, CT signal, same-grid skull
as bone, no corridors), 2026-11-26, with the fixed 6 mm statistics
neighbourhood. Floating components count the union of shown layers against
the root labels.

| | TopCoW-014 (development) | TopCoW-018 (held out) |
| --- | --- | --- |
| Floating components, annotation / default view | 6 / 0 | 4 / 0 |
| Arterial components bridged (gap, contrast fraction) | 3: R-M2 with R-M3, R-M2, R-M3 (1.7 to 2.4 mm, 0.52 to 0.62) | 1: an R-M3 piece (1.0 mm, 0.74) |
| Arterial components in review | 3: R-OA, L-OA (too far), an R-AChA piece (no supported path) | 3: R-M2 with R-M3 (no supported path), R-OA, L-OA (too far) |
| Venous components in review | 1: L-BVR, more than 10 mm from the venous tree (it touches an artery) | 0 |
| Hidden by level of detail | 0 (every annotated voxel is named) | 0 |
| Peak footprint of the check | 0.83 GB | 0.99 GB |

The named segments hidden by default are the ophthalmic segments, part of the
right anterior choroidal artery and, on TopCoW-014, the left basal vein of
Rosenthal; on TopCoW-018 also its R-M2 with R-M3 segment. Ophthalmic segments
would join only at 8 to 10 mm gaps; the 3 mm limit was kept. TopCoW-018 had
been run twice before this definition was fixed, while the lumen and
background statistics still followed the search window (first one bridge, R-M2
with R-M3 at 1.4 mm; then none after the window was cropped). The result above
is the single run after the fix; nothing was adjusted after any TopCoW-018 run.

On the merged vesselFM TOF layer of EDEN PTNT_4509 (development; no named
labels; TOF signal and vesselFM probability), the fallback accepted the largest
component and 20 components entering through the inferior face at 2.1 to
2.9 mm calibre (10 to 112 mm long). It refused 124 entering components: 117
shorter than 10 mm and 7 of 1.2 to 1.95 mm. 250 of 2,476 components joined
across bridges, 2,205 stayed in review (1,262 below 0.5 mm³) and the level of
detail hid 2,689 voxels in 126 thin end pieces. The default view has 21
components, each holding a root. The 672 × 672 × 210 grid took 51 s on the CPU
with a 1.89 GB peak footprint for the whole check. In stage B the hero TOF
studies get named MR labels, whose ICA, vertebral and basilar roots replace
this fallback.

### Phase 2 model layers in the hero studies

`scripts/capture_vascular_parity.py wire --ensemble <folder>` adds a Phase 2
ensemble to each hero study that has one: a `vessels` layer with the named
TopBrain labels (`named_labels: {data, scheme}`, the CT map only on CT studies
and the MR map only on MR) and the soft vessel probability, plus the connected
vesselFM score as a separate unnamed layer whose `rooted_parent` is that tree.
The addition is shown only where it reaches the tree's shown display, under the
calibre rule: directly, or across a signal-supported bridge of at most 3 mm.
It never joins the named labels; each of its pieces takes the class (and
colour) of the tree part it joins. A face-connected piece of the addition is
shown only when at least one of its voxels lies outside the tree dilated by
two voxels (26-connected), that is, when it extends the tree by more than its
wall. Pieces that only hug the tree wall go to review (amber overlay) and are
counted in the receipt as `review.wallThickening` (pieces, voxels, volume and
the reason). The shown pieces' displayed mask also holds the tree voxels
face-adjacent to them (a one-voxel overlap, in the class of the tree part it
belongs to), so the branch surface is drawn into the tree's rather than
beside it. The overlap is display-only: it is counted as
`display.overlapVoxels` and is not part of the named labels, the review set or
the shown, review and hidden partition. If its tree is missing, on another grid,
fails, shows nothing or is hidden, the addition shows nothing and its voxels
stay under review. Both carry an
`image_derived_estimate` provenance with the model, teams, ensemble folder and
file hashes. The newest generation becomes the study's initial vessel layers
in place of the previous initial vessel layers; supplied annotation layers stay
in the study, unchanged and selectable, and non-vessel initial layers are
untouched, so the desktop default, the capture recipe and the geometry gate
show the same layers. A binding of the hashes names the layers: a second run
changes nothing, and a re-wire after the ensemble changed retires the previous
wired generation (its entries and arrays, reproducible from the folder; older
revision snapshots of the manifest still name them). Layers wired from a folder
that has since changed are refused until re-wired. The write is all or nothing
and refuses if saving would change anything else; source-linked review
manifests are written as their own document beside the pinned source. Each
wiring writes a receipt (manifest hash and snapshot before and after, the
previous initial layers) under the evidence folder, and `unwire` undoes it. `geometry` and `capture` take the same
`--ensemble` and `--scope` (default `rooted`): the geometry measures the rooted
union the default view shows. Its floating count is taken on the surfaces the
desktop receives: each displayed part is meshed as the viewer meshes it, and
pieces of different parts are joined where they share a vertex position or
where a vertex of one lies inside the displayed voxels of the other (a vertex
on a boundary between voxels counts only when all of them are set, so a
surface that only abuts another part stays separate). Every piece with no
root-label geometry counts as floating; face-connected voxel components are
reported beside it. The command prints the mesh count with the voxel count
beside it and exits non-zero when any case has a floating mesh piece in the
default view. The major-vessel gate covers the basilar, ICA, M1, A1-A2 and the
third A2, M2, P1-P2 and vertebrals, and on CT the vein of Galen, straight and
superior sagittal sinuses (TopBrain merges A1 with A2 and P1 with P2). Where a
reference annotation exists, a major vessel counts as lost when less than 90 %
of its reference centerline lies within one voxel of the default display
(declared before measuring); 90 to 100 % is listed as partly in review. Without
a reference the gate uses the prediction-side status and says so; without named
labels it reports `not-measured`.

Mesh gate, 2026-11-26 (CPU; each displayed part meshed as the desktop meshes
it). "Pieces" are outer surface pieces; "floating" are pieces with no root-label
geometry within one voxel; face components are counted beside them.

| | Full source: pieces / floating (face components / floating) | Default view: pieces / floating (face components / floating) | Connectivity fill voxels | Major vessels against the reference (coverage below 100 %) |
| --- | --- | --- | --- | --- |
| TopCoW-014 annotation | 12 / 9 (16 / 13) | 3 / 0 (3 / 0) | 6 | none lost; R-M2 98.9 % |
| TopCoW-014 UZH+ARG+KDH ensemble | 544 / 438 (37 / 34) | 3 / 0 (3 / 0) | 15 | none lost; R-M2 98.4 %, L-M2 97.7 % |
| TopCoW-018 annotation (held out) | 16 / 11 (19 / 14) | 5 / 0 (5 / 0) | 8 | none lost; R-M2 92.7 % (19.4 mm uncovered) |
| TopCoW-018 ensemble (held out) | 1,192 / 966 (55 / 50) | 5 / 0 (5 / 0) | 30 | none lost; R-M2 92.7 %, L-M2 94.5 % |
| PTNT_4509 vesselFM cleanup (no names) | 3,367 outer pieces, no root labels (3,367) | 21 / 0 against the fallback root seeds, plus 28 inner walls (21 / 0) | 312 | not measured (no named labels, no reference) |

With the combined `combined-base-UZH-ARG-KDH` ensemble and its connected
vesselFM addition (2026-11-26, after the wall-thickening rule and the overlap):
TopCoW-014 default view 3 mesh pieces / 0 floating (3 / 0 face components);
the addition keeps 6 pieces (9,683 voxels), sends 215 wall-thickening pieces
(546 voxels) to review and draws 233 overlap voxels. TopCoW-018: 6 / 0 (5 / 0);
3 pieces kept, 613 wall-thickening pieces (2,069 voxels) to review, 27 overlap
voxels. The two Eden2020 TOF studies: 5 / 0 and 3 / 0 outer pieces (1 / 0 face
components each); 63 and 82 pieces kept, 6,479 and 6,755 wall-thickening pieces
to review, 4,878 and 5,380 overlap voxels. The major-vessel gate is met on all
four. Before the rule, the TopCoW-014 addition drawn as its own meshes gave 207
pieces, 187 of them floating.

The full-source ensemble is drawn from its probability contour, which splits
into many more pieces than its face-connected voxel components. Named segments
left in review on the ensembles: pieces of R and L M2 and M3, R-AICA, R-PICA,
L-AChA, R-OA, L-OA (014); pieces of R and L M2 and M3, R-P3P4, R and L AChA,
R-OA, L-OA (018). TopCoW-018 was rerun in this round for the face-connectivity
fix and the mesh gate (annotation and ensemble, full source and default view);
each run is logged with the evidence.

## Centerline tube model

`vessel_tubes.py` (`centerline-tubes-v2`) turns a binary vessel layer, with an
optional soft probability on the same grid, into one geometric model for the
wall mesh, contact queries, rod sites and a soft tag volume. A local native
wall uses it when prepared with `vessel_geometry` set to `vessel_tubes.MODEL`
(the IPC field `vesselGeometry` passes it through; no desktop control sets it
yet), which becomes `operative_vascular_geometry`. The option is refused
without V2 tissue physics and the working-axis constraint, which carry its
shell contact. The model is stored once in case metadata
(`vessel_tube_models`), its digest in the wall receipt, and tube metadata under
any other policy is refused. Derived from it: the collision and display wall,
the desktop `vesselTags` volume and `vesselClasses`, and the positions and
calibres of the rod sites (`vesselRodGeometry`). Which sites exist, their bonds
and branches still come from `_local_chains`. An edit never refits: removed
cells join the model's ordered edit record (`removedCells`, with the faces open
to non-vessel), the stored field is cut by them, and the edited wall, its
per-vertex classes and its digest come from that model and record alone. Wall
vertices away from the cut are bit-identical, so vertex transport keeps
treating them as unchanged. Walls prepared
without the option, and every saved scene and replay, keep their earlier
geometry; a run of the default path against the previous code gave identical
report bytes, case hashes, engine snapshots and desktop arrays. Tools keep their whole
working shell out of a tube wall (see the shell projection in
[surgical physics v2](surgical-physics-v2.md)): on the fixture strokes the
largest shell overlap at an accepted step was 0.018 mm, with the vessel
displaced by up to 2.2 mm. Changing what `extract` or `mesh` produce without a
new `vessel_tubes.MODEL` makes saved tube cases refuse to reopen, because the
rebuilt wall no longer matches the stored binding.

- **Graph.** Centerlines are the `build_source_branches` Lee skeleton paths
  (certified H₀/H₁ representative edges). Spurs whose tip lies inside the
  inscribed ball of their junction are pruned and degree-two nodes merged.
  A component whose skeleton is empty becomes one ball. Each edge carries the
  majority class and named label of its source cells.
- **Samples and radii.** Samples are smoothed on the graph (Taubin, tips
  pinned) and recentred from opposite pairs of 16 rays in the normal plane.
  Each ray stops at the 0.5 crossing of the trilinear soft signal (the
  probability, or the mask itself without one), refined by bisection. A
  sample's starting radius is the median ray length (near a junction or tip,
  the distance to the source half-level wall), then the median axis distance
  of the wall points its tube is nearest to. Tips extend along the axis to the
  measured lumen end. Radii have a floor of half the smallest spacing.
- **Sections.** Each edge sample also carries angular harmonics of order 1 to
  6 (`Re zⁿ`, `Im zⁿ`, `z` the unit direction from the axis in a
  rotation-minimising frame of the edge) and, at a tip, two end-cap terms in
  the axial component (`x²`, `x⁴`) that flatten or sharpen the cap. Radii and
  sections are fitted in three rounds of regularised linear least squares to
  the source half-level wall points each segment owns (the segment with the
  smallest field within 1.5 voxels; node balls excluded): every point asks
  that the interpolated offset equal its round-cone distance. Neighbouring
  samples are tied (weight 1), order `n` has ridge 0.05 `n`, and a section's
  largest offset is capped at 0.6 of its radius. This follows elliptical and
  fused sections, flares, crotches and flat-cut ends.
- **Field.** A segment's field is its round-cone distance less the offset
  interpolated between its two samples, `F = d − (1 − t) h_a(w) − t h_b(w)`.
  Junctions are the hard union of the incident tubes and the inscribed node
  ball: with fitted sections a smooth blend did not help (`k` is kept in the
  format and is 0). Edges that do not share a node are never blended. The
  field is clipped to the acquired box, half a voxel past the outer sample
  centres. Tubes of different 26-connected source components are shrunk where
  they come within one mesh step.
- **Consumers.** `mesh` evaluates `F` per segment box on a grid of smallest
  spacing / 2, runs Lewiner marching cubes in 96-cell blocks, joins block
  seams by grid-edge identity, orients faces outward and takes normals from
  the field gradient. Wall components that enclose no centerline sample are
  dropped and counted. `tag_volume` writes `clip(0.5 − F / spacing, 0, 1)` on
  the source grid. `rod_sites` gives `[x, y, z, r]` per source cell, the
  closest centerline point clamped into the cell. `audit` checks open,
  non-manifold and inconsistently oriented edges, non-manifold vertices,
  degenerate faces and edge-through-triangle self-intersections, and with the
  model the largest field value over vertices, edge midpoints and face
  centroids (`wall_outside_max_mm`). `deviation` measures both directions
  against the source half-level wall, in millimetres, smallest-spacing voxels
  and index-space voxels (each axis over its own spacing).
- **Collision bound.** `query` and `capsule_clearance` use a lower bound on the
  distance to the wall, not `F`. Per segment, `L` bounds the gradient of `F`
  outside the wall: the angular slope of the section over the wall radius
  (the round cone contains the ball of the interpolated radius about each axis
  point, so outside the wall the axis distance exceeds it), the change of the
  offset along the axis and the taper. Slopes are sampled on 256 axis-normal
  and about 2,100 cap directions per end, each sample standing for its arc or
  cap through second-derivative bounds. Inside, the value is `F / L`; outside
  it is the largest `r` with `r L(r) ≤ F`, where `L(r)` uses the axis distance
  over the ball of radius `r` (between `F / L` and `F`). `capsule_clearance`
  is the smallest bound over axis samples less half the sample step (the true
  distance is 1-Lipschitz) and less `wall_tolerance`, one mesh grid step,
  which covers marching-cubes triangles that cut across concave creases and
  narrow grooves between adjacent tubes (largest measured excursion 0.77 of a
  step, 014 CTA). A positive clearance therefore means the capsule meets
  neither the implicit nor the meshed wall; a caller holding an audited mesh
  may pass its `wall_outside_max_mm` as `tolerance`. Tests check both on
  synthetic tubes, and on 014 CTA 300 capsules beside the wall gave no
  violation. The bound is conservative: the median slack of clear capsules
  within 1 mm of the 014 CTA wall is about 0.5 mm (0.28 mm of it the
  tolerance), and `L` is computed on first query (about 2 s per model).

TopCoW 014 and 018 (CTA and MRA), Phase 2 ensemble probability as the vessel
set and soft signal, measured 2026-11-26 on CPU. Deviation is mean / 95th
percentile in smallest-spacing voxels (index-space p95 in brackets), arteries,
points more than one voxel from the acquisition edge; 018 was run once per
modality after the method was fixed on 014. Source wall points take the
source's own artery/vein class and tube points their edge's class; a point no
tube owns within four voxels keeps its full distance (none on the tube side;
70 source points on 018 CTA, none elsewhere). Re-measured with that metric by
`scratchpad/phase4-fit/real2.py` on the `ensemble-UZH-ARG-KDH` probabilities
(the selected combined ensemble was not written to the drive):

| Case | Tube to source | Source to tube | Veins (tube, source) | b0 / b1 source, tubes | Faces | Seconds, peak |
| --- | --- | --- | --- | --- | ---: | --- |
| 014 CTA | 0.08 / 0.25 (0.23) | 0.06 / 0.18 (0.16) | 0.09 / 0.25, 0.10 / 0.28 | 16 / 22, 16 / 21 | 738,128 | 22, 1.3 GB |
| 018 CTA | 0.09 / 0.28 (0.25) | 0.08 / 0.25 (0.21) | 0.08 / 0.22, 0.13 / 0.26 | 27 / 23, 27 / 22 | 788,124 | 26, 1.4 GB |
| 014 MRA | 0.11 / 0.33 (0.28) | 0.12 / 0.38 (0.30) | none | 11 / 11, 11 / 7 | 1,675,296 | 65, 1.7 GB |
| 018 MRA | 0.12 / 0.34 (0.29) | 0.13 / 0.39 (0.32) | none | 27 / 31, 27 / 33 | 2,829,478 | 101, 2.4 GB |

The half-voxel gate is met on all four: arterial p95 0.18 to 0.39 voxels
(0.10 to 0.14 mm), means 0.06 to 0.13. Every wall is closed, edge- and
vertex-manifold, consistently oriented and has no self-intersection; b0 is
kept and no named label is lost (lowest tag coverage of a label's voxels
0.71, 018 MRA R-ECA; tag Dice against the mask 0.98 to 0.99). Seconds are
extract plus mesh; peak is the whole evaluation process. The remaining
misses sit mostly at tips: on 014 MRA the p95 (tube to source, source to tube)
is 0.48 / 0.56 voxels near tips, 0.37 / 0.41 near junctions and 0.31 / 0.36
along vessels. The sections follow the source's lumpy half level, so the
wall undulates where the source does.
Tubes close source holes smaller than their radii: the four cycles 014 MRA
loses are sub-millimetre holes (half-level probability 0.3 to 0.45 at their
centres); the circle of Willis loops are kept. On 018 MRA the tube wall has
two more cycles than the mask (not localised). Attempt one (circular sections,
smooth junction blend) had p95 0.46 to 0.85 voxels on CTA and 1.06 to 1.35
on MRA, with veins at 1.2 to 1.7. The annotation label maps alone give CTA
arterial p95 0.55 to 0.94 voxels, because the binary half level is
stair-stepped. Tests: `tests/test_vessel_tubes.py`.

## Source-owned branch geometry

The existing TopBrain CTA001 now has reproducible, source-owned centerline
geometry and a separately certified H₁ representative subgraph. Original voxel
IDs, native physical coordinates and publisher artery/vein/sinus classes remain
traceable. No source mask, registration, clinical boundary condition, perfusion
territory or patient outcome has been created or changed.

This is source-geometry qualification on a separate vascular development case.
It does not supply vascular anatomy to UCSF-PDGM-0007, which remains unsupported
for patient-specific vascular evaluation. It does not pass acquired brain-shift
prediction or measured-flow validation.

## Exact source and frame

The source is **TopBrain Challenge Organizers, TopBrain Challenge Data Release,
Batch1 July30 2025**, [Zenodo 16623496](https://zenodo.org/records/16623496).
Its TopCoW lineage is described by the [creator challenge paper](https://arxiv.org/abs/2312.17670).
The release requires attribution and permission from the data owner for
commercial use. This local work uses existing noncommercial research inputs;
there is no new acquisition or redistribution of patient volumes.

The [publisher's CTA class table](https://topbrain2025.grand-challenge.org/data/)
distinguishes arteries 1–34, veins 35/37/38/39 and sinuses 36/40. These CTA semantics
must not be applied to MRA, whose values 35–40 denote different structures.
The execution requires the exact frozen existing patient manifest, source label
map and label dictionary hashes before opening arrays. It checks that the two
native binary masks exactly equal the corresponding original categorical unions.

The CTA header explicitly declares millimetres. The associated annotation header
leaves its units unset, although shape and sform affine are exactly identical.
The first strict header check correctly refused this combination before decoding
any array. A tested named-reference rule now binds this annotation's indices to
the explicitly associated CTA grid; it records the inherited unit and leaves
both original headers unchanged. Contradictory explicit units, absent association,
missing sform or a different grid still refuse. This is indexed label-to-image
association, not a registration fit or independent anatomical acceptance.

## From source voxels to representative edges

`anatomy_geometry.vessel_graph` crops only empty exterior workspace, pads for
numerical operations and applies Lee thinning. Every retained point carries its
original C-order source-cell ID, exact float64 RAS coordinate, source6/source26
component IDs and an annotation-to-background-center distance. That distance is
not a measured lumen radius. Original image-boundary contact is reported
separately from node-level boundary-limited distances; thinning can move all
nodes off a truncated source face.

The source voxel union, thinned voxel union and all-26-neighbor edge graph are
different mathematical objects. In particular, a triangle of mutually touching
voxel cells has no voxel hole but creates a cycle in the adjacency edge graph.
The implementation reports their components, Euler counts, cavities and cycle
ranks separately. It preserves source6 island omissions and unresolved local
face-path contacts in the coverage denominator. A local diagonal shortcut may
still belong to one larger global face-connected component; the two findings
are not conflated.

`vascular_homology.representative_subgraph` retains every skeleton node and uses
only a subset of its original edges. Complete26 adjacency is first verified.
A deterministic spanning forest identifies fundamental-cycle chord coordinates.
Exact GF(2) elimination of triangle boundaries chooses pivot chords to remove;
the nonpivot cycle space is a complement of those boundaries. The output includes
forest/chord indices, original independent triangle generators, echelon-basis
words, pivot columns and retained source-edge indices. A verifier can enumerate
all triangles to check membership, completeness and the quotient complement.
Caps return no partial representative.

The closed-convex [nerve theorem](https://arxiv.org/abs/2203.03571v6) supplies the
voxel-cell interpretation: pairwise-intersecting axis-aligned boxes have common
intersection. The resulting inclusion certifies H₀/H₁ over GF(2), not a
deformation retract, homotopy equivalence, route length, lumen or flow. Higher
homology is checked separately. A hollow voxel shell therefore cannot be
certified by a one-dimensional graph merely because its H₁ is zero. Raw edges
are preserved alongside the representative, and physiological admission remains
false.

## Acquired-source results

| Quantity | Arterial union | Venous/sinus union |
|---|---:|---:|
| Original positive voxels | 44,951 | 34,226 |
| Retained skeleton nodes | 2,057 | 1,098 |
| Source components6 /26 | 17 /2 | 22 /6 |
| Raw adjacency edges /cycle rank | 2,107 /52 | 1,184 /92 |
| H₁ representative edges /cycle rank | 2,080 /25 | 1,140 /48 |
| Triangle-boundary rank | 27 | 44 |
| Source26 components omitted | 0 | 0 |
| Source6 islands /voxels unrepresented | 8 /8 | 9 /12 |
| Edges lacking a local face path | 22 | 43 |
| Edges crossing global source6 components | 8 | 10 |
| Source voxels touching image extent | 115 | 0 |

Independent positive-voxel BFS, explicit cubical cell counting, local-path checks
and direct affine evaluation verified the source geometry. Separate dense GF(2)
elimination verified every native representative certificate. All 3,155 retained
source nodes remain, with no new edges. Source masks and image files retain their
original hashes. The final two-group run took about 3.12 s with approximately 718 MiB
peak RSS; this is an extraction measurement, not an RL or viewer speedup.

The public native inspector exposes source IDs, separate topology checks, FOV
contact, island loss and representative status. Its existing radius field is
explicitly marked as an annotation estimate, and `flow_model_admitted` is false.
The partial annotations omit known source classes and unannotated branches;
missing labels do not establish absent anatomy. Local expert acceptance,
physiological boundary conditions, lumen/flow measurements and glioma-specific
vascular coverage remain open. The [RC model](vascular-graph.md) is a separate
declared numerical mechanism; these thousands of adjacency nodes are not silently
converted into its lumped segments or blood-flow parameters.

### Same-source class review in the newer TopBrain release

A separate review of the TopCoW-001 CTA development migration used the
August 17, 2026 TopBrain v3 release. Both supplied label versions used the
same native image lattice, 0.508 × 0.508 × 0.625 mm. The reference hashes,
native overlays and per-class measurements are in the private HDD review
under `qc/continuous-rehearsal-20261118/source-review/`. The acquisition is
privacy-redacted. The two releases share subject and annotation lineage.
[Publisher dataset](https://zenodo.org/records/21972006)

| Structure | Source result |
|---|---|
| Basilar artery | The 2,975 positive voxels match across versions. The v1 label is one face-connected component. Median twice-centerline-to-background distance is 3.71 mm. |
| Right posterior communicating artery | Both versions label the same 218 voxels. The v1 label is one face-connected component. The median width proxy is 1.25 mm, close to two source slices. |
| Superior sagittal sinus | The v1 label has one 26-neighbor component and ten face-connected components. Twenty-seven skeleton edges lack a local face-connected path. |
| Vein of Galen | The v1 label has 524 positive voxels and one face-connected component. One diagonal skeleton edge lacks a local face-connected path. |
| Anterior communicating artery | Both supplied masks are empty. Presence and caliber remain unknown. |

The two venous structures have no matching v2 class in this comparison. The width proxy
measures the sampled label. Lumen diameter and patency would require independent
source review. Vascular evidence belongs to this CTA patient.

## Reproduce

The final [protocol](../manifests/experiments/native-vascular-graph-v4.json) binds
exact source and code hashes. Earlier protocol versions preserve the initial
header refusal, raw-graph diagnostic correction and pre-reduction reference.
Detailed coordinates, source primitives, negative attempts and independent review
receipts remain under ignored `testcases/outputs/simulator-qualification-20261029/native-vascular-graph-v1/`.

```sh
.venv/bin/python -m pytest tests/test_vascular_homology.py \
  tests/test_vascular_source_graph.py tests/test_patient_anatomy.py -q
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python \
  scripts/qualify_native_vascular_graph.py \
  outputs/poc-angio-fidelity-v2/native-cta \
  testcases/outputs/simulator-qualification-20261029/native-vascular-graph-v1/new-run
```

No force curves or reserved patient payloads are accessed. Surgical model and
recording semantics remain separate from this source-geometry cache.
