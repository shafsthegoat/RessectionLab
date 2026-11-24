# EDEN2020 local intake and prospective qualification

The public [Zenodo record](https://zenodo.org/records/5714812), DOI
[10.5281/zenodo.5714812](https://doi.org/10.5281/zenodo.5714812), was inspected on
October 19, 2026. The page describes 30 adult glioma patients imaged at 3 T,
version 1.0, and restricted files. It requires an intended-use access request.
Only the public metadata page was read; no file, login, access request, or
restricted download was attempted. No public redistribution license was
established from that page. Access approval and compatible use terms must be
obtained and reviewed independently.

The published modalities are post-gadolinium 3D T1 PROSET, volumetric FLAIR,
TOF-MRA arterial signal, SWI magnitude/phase, 32-direction b=1000 DTI plus b0,
DTI-derived AD/RD/MD/FA, advanced diffusion/NODDI (60 directions at b=3000,
35 at b=711, 11 b0), and reverse-phase b0. The source says DICOM was converted
with dcm2niiX v1.0.20200331. Availability and naming must be checked per patient;
the description does not establish that every patient has every sequence.

## Native TOF vessel review

The restricted local collection is indexed by the publisher acquisition
receipt and the prepared native study manifests on the verified personal
volume. It contains 30 unique subjects and 29 acquired TOF volumes. The TOF
volumes have corresponding native vesselFM masks and uncalibrated scores. The
TOF-to-T1CE transforms are unreviewed, so vascular review uses each original
TOF lattice. The [storage ledger](local-storage-ledger.md) describes who can
access this volume.

The vessel review preserves the original model layer. It proposes visibility
exclusions for separate findings with weak TOF evidence, records uncertain
components, and keeps uncertain components visible. The source head
envelope is estimated from the TOF acquisition with a physical margin.
Pial vessels, basal cisterns, bright extracranial structures and cropped
vessels remain eligible for review. The companion extracts the filtered
surface from the original score field while retaining the whole prediction
as a selectable layer. The component decision changes display evidence only.

Source branch IDs bind to the original acquisition and source cells. Branch
paths retain supported junctions and cycles and report incomplete graph
coverage. Gap proposals record geometric and source-signal evidence for
review, and add zero voxels. The physical surface audit compares local
transverse section area and vertex movement before and after fairing.
The TOF lumen estimate, vessel type, and assumed simulator wall and support
properties are independent evidence.

In **Anatomy**, select a vessel layer on its acquired TOF and open **TOF gap
review**. **Find nearby branch endpoints** checks short gaps between compatible
source-bound paths. One TOF signal window applies across all findings and
sections. The view shows two sections along the proposed path and one across
its midpoint. Amber marks the sampled path and cyan outlines the source
estimate. Use **Show gap overlay** to
inspect the acquired signal by itself or **Center linked slices** to check the
original planes. Weak-signal candidates and partial graph coverage remain
visible. The reconstruction and quantitative anatomy continue to use the
source-bound mask. [Evidence location](local-storage-ledger.md)

New patient import workspaces start with **TOF exterior-air review**. The
view uses a native TOF head envelope with a physical margin and hides a
detached vesselFM component only when the entire finding is sufficiently
dark and its measured 2 mm neighborhood contains no signal above the
envelope threshold. Acquisition edges, attached findings and ambiguous
components stay visible. **Full source estimate** restores the original
prediction and score surface. Supplied layers and operative preparation use
their existing source evidence.

Applying this same screen to the selected review layers for all 29 acquired
TOFs proposed no additional exclusions. In three matched full-head views,
the remaining peripheral specks follow the selected native score surfaces.
Source-plane checks found weak signal in some remote groups and coherent
signal along nearby vessels. The tested scans showed no consistent offset
along the phase-encoding direction to support a shared ghosting explanation.
In **Vessels**, the 3D toolbar's **Vessel slab at cursor** control reduces
through-depth overlap for local inspection. **Faint context** shows the
out-of-slab vessel courses behind the focused foreground, with visibility
falling off with physical distance. Linked source slices and slab picking
retain their coverage. Exit restores the prior 3D cut.

The available attached-wall trials found that dark-signal subtraction can
create pits, split adjacent geometry or narrow sections with plausible source
support. The full-volume COSTA, SPOCKMIP and Vessel-MAYON comparisons also
missed bright native TOF paths. VesselFM remains the reconstruction estimate.
The restricted local review tracks the source comparisons and failures.

In **Anatomy**, select the native TOF vessel estimate and open **Attached
vessel uncertainty** in its layer inspector. The on-demand review finds
physically thick low-signal regions with a nearby face-adjacent brighter
part of the estimate. Three native sections show the original TOF signal,
the estimate contour and the review footprint. The signal window is shared
across the three views. **Show review overlay** switches the same sections
back to acquired signal alone. A weak-signal region can contain a genuine
vessel. This inspection preserves reconstruction geometry, source labels,
saved views, correction drafts and rehearsal inputs.

In an editable workspace, pick an amber voxel on an acquired section and
use **Center sections at selection** to inspect all three planes at the
same native location. **Preview local erasure** subtracts only selected
footprint cells within the chosen physical radius. The 3D comparison keeps
the native model scores and their crossings outside the edited cells. Use
**Show original**, **Undo**, or **Reject** to inspect or discard the draft.
Accepting a correction records the reviewer, exact native pick, radius and
source checksums in a new revision. The operation uses the native TOF grid
and leaves cross-acquisition registration status in force.

The screen samples the low decile of foreground TOF signal, uses a 1 mm
distance-to-background core and requires at least 2.5 mm³ after a physical
opening. It measures nearby signal against the foreground interquartile
range. VesselFM score summaries are uncalibrated. Thin joins can fall
outside this screen, and sections near the acquisition faces have limited
context. The operation runs only on a source-bound, unedited TOF estimate.

A separate connected exterior-air trial yielded one small candidate for
exact geometry comparison. The strictest local deletion preserved the
audited voxel topology and the measurable nearby source section, while
changing the indexed score surface's Euler characteristic. The source
review and matched silhouettes support retaining the region for inspection.

Run the review and package steps from the primary checkout.
Replace the repeated subject selection with `--cohort` to run or resume the
component, branch and attached reviews for the whole restricted source inventory.

```sh
.venv/bin/python scripts/run_eden2020_vascular_review.py --subject PTNT_4509
.venv/bin/python scripts/run_eden2020_vascular_review.py --graph --subject PTNT_4509
.venv/bin/python scripts/run_eden2020_vascular_review.py --attached --subject PTNT_4509
.venv/bin/python scripts/build_eden2020_vascular_companion.py --subject PTNT_4509 --linked-source
```

The private run is under
`qc/eden2020-glioma-5714812/vascular-reconstruction-20261123-v1/` on the data
volume. Read `workflow.json` for the frozen model and cleanup policy,
`processing.json` for per-subject processing, `graphs.json` for branch coverage
and `attached.json` for the attached-region screen and its exact native cell
receipts. Open the source-linked `vascular-review.anatomy.json` or a standalone
companion's `patient-anatomy.json` in Cases, then select the original or cleanup
under **Vessels**. Linked views reuse the canonical source study's checked
members and open read-only. The native score surface is the vascular review
default. Conditioned surfaces are available as a separate presentation choice.

A scoped source review can restore a proposed exclusion to the uncertain set
with `--restore COMPONENT --reason REASON` or hide a reviewed artifact with
`--exclude COMPONENT --reason REASON`. The reviewed partition is saved beside
the initial partition and keeps its source binding. The source review
and application captures provide the private comparison evidence. The
companions are for research display. Patient-specific surgical admission
requires its own source registration and mechanics evidence.

### Matched app review and source-linked rehearsal

The continuity run reuses the published native TOF companions and branch
graphs. Build the desktop renderer from the current checkout before recording
evidence so its submitted geometry matches the backend. Capture the full cohort
with `--match-baseline` to keep the camera, source cursor and lighting fixed
between the original and selected layers. Use `--subject PTNT_4509` to retry a
single case. The capture status records the workspace snapshot and image paths.

```sh
cd desktop && pnpm build && cd ..
campaign=qc/eden2020-glioma-5714812/vascular-reconstruction-continuity-20261123-v2
data='/Volumes/shafrir/old coding projects/RessectionLabData'
.venv/bin/python scripts/capture_eden2020_vascular_cohort.py --match-baseline \
  --output "$campaign/app/cohort" --display-settings "$data/$campaign/app/cohort-display-settings.json" \
  --reviewed-manifests "$campaign/reviewed-manifests.json"
```

Use `scripts/measure_vessel_surface_sections.py` with the exact linked manifest
and compatible native branch JSON for physical section comparisons. Its receipt
records the manifest SHA-256 and snapshot it opened.

Source review can separate a complete posterior component into a dedicated
display layer. Use the component anchor from a structural screen tied to the
selected TOF review.

```sh
.venv/bin/python scripts/separate_eden2020_vascular_compartment.py \
  --selected REVIEW --screen SCREEN --anchor CELL
```

Use paths relative to the verified HDD. The linked companion keeps the original
estimate and the selected union available beside the two disjoint groups. Each
group uses the original native score field. The structural alignment and
superficial tissue labels remain research estimates.

For reviewed field-edge signal, pass `--compartment field-edge` and a native
field-edge screen that binds the acquired TOF, selected mask and candidate
component anchors. Inspect the native planes before approving anchors. The
field-edge estimate opens muted and selectable, with wider vessels emphasized.
Use **Vessel shading** to review fine detail or select the intact union to
compare the full estimate.

Stage reviewed posterior, field-edge and skull-exterior splits in that order,
using the current priority layer as the source for each split. A case may skip
a split where its source evidence is insufficient. For `--compartment
skull-exterior`, supply a source-bound MRI skull screen of whole exterior
components. The startup view focuses on the remaining network. Earlier review
layers and the original union are selectable. Open or ambiguous skull sections
retain their original vessel evidence.

Use the acquired TOF to judge vessel continuity. Structural MRI provides
context for reviewing detached groups, and native TOF-supported paths stay in
the selected reconstruction.

For a detached estimate inside the head, `--compartment source-disagreement`
stages a final display split from an acquired TOF and bound structural review.
The reviewed component remains available in a muted layer. The split uses the
current native priority layer, including one produced by the skull screen.

### Prepare a local native wall from a patient study

In the desktop anatomy workbench, place the cursor on a vessel in the reviewed
patient frame. Open **Rehearsal**, select the patient brain layer and the native
angiography vessel layer with saved model scores, and choose **Prepare native
wall at cursor** in the limited research environment. The local mode uses an
approximately 24 mm box in the original angiography lattice. It keeps the
source mask, scores, affine and branch cell identities, and samples the chosen
patient tissue layers into that box. Acquired images and annotation revisions
stay in the source study. The saved scene records the crop, input checksums,
qualified local traces, and the exact conditioned vertices and source owners
used by the viewer and by contact. When vascular inspection has built a source
branch graph, the local scene keeps those branch IDs. Otherwise its traces
have local IDs tied to original-grid cells. Save and reopen through the usual
rehearsal workspace controls.

Preparation requires reviewed or source-provided alignment for every selected
acquisition, an unchanged native score layer, adequate source coverage around
the cursor, a continuous local vascular trace, nearby patient-derived tissue
samples, and a clear initial wall. The selected layer must match its material
role. Display and inspection annotations are excluded. Tissue sampling respects
acquired and declared label coverage, with uncovered cells supplying no
mechanical material. The entire vascular crop must lie within the score field's
declared coverage. The engine models motion around the retained traces. Other
vessel surfaces in the box remain visible and participate in contact.
Stiffness, tissue springs and caliber estimates are research assignments. The
local scene declares zero pressure and no physiological flow boundaries.
New ordinary preparations let intact springs transfer accepted vessel motion
back to their assigned tissue cells. The preparation receipt records the
attachment model, and historical recordings retain the model they saved.
Field-edge faces indicate where coverage stops. A selected saved approach still
opens in its separate inspection mode.

The primary restricted native inventory has 30 studies and 29 acquired TOF
score layers. Each TOF-to-patient alignment is marked `unreviewed`, so these
head studies currently abstain from this local cross-acquisition mechanics
path. Estimated alignment remains available through **Prepare standard
environment** with its explicit research choice. The independent TOF view
continues to support native reconstruction and source inspection.

In the full-depth native TOF **Vessels** view, the automatic volume backdrop
uses the acquisition's signal and affine to balance context brightness.
Choose **Balance TOF context** in Appearance to compare, or set Scan density
manually. Source overlays, physical sections, and vessel surfaces retain
their exact acquired and segmented inputs.

The current cohort reconciliation binds all 30 acquired studies and 29
chosen vascular reviews to their live source identities. Hidden desktop
recapture reproduced all 29 selected native surface hashes at the archived
physical cameras. Each TOF review includes an acquired-source cut and a
separate scalar context view. The study without acquired TOF opened in
structural MRI. A second pass using the renderer in the signed build matched
all 30 selected Anatomy panes pixel-for-pixel while reducing known label
texture storage. The completion table also records the six native section
checks and three cold research-workspace opens. The current preflight found
zero qualified ordinary EDEN native-wall layer pairs. The same source catalog
now drives **Rehearse → Local native wall**. A read-only run on all 30 prepared
study manifests found 29 acquired TOF volumes and no selectable local pair.
The catalog records the tissue and alignment reasons by case. Signed cold
verification of the desktop refusal used PTNT_7195.
[Local evidence index](local-storage-ledger.md)

In **Scans → Alignment & registration**, choose the moving scan and a **Review
position**, then select **Check local alignment evidence**. The 24 mm RAS
field reports overlap of the acquired grids. Twenty-seven image comparisons
report edge agreement and search shifts while allowing contrast inversion. Competing matches,
peaks at the edge of the search, and uncovered regions remain visible. Open
**Local measurements** and load paired source slices at any probe for manual
landmark review. The screen retains the current transform and review status.

Five prospective fields used the frozen graph anchors for PTNT_4509,
PTNT_0952, PTNT_9937, and PTNT_5387 and the earlier PTNT_4509 teaching
junction. The entire sampled field was within both acquisitions for each
comparison. Only 9–13 of 27 probes per field had a distinct peak. The
PTNT_4509 teaching junction had a 0.50 mm median shift across its 13 distinct
peaks, with nine peaks at the search limit and five competing matches. These
shifts describe image-texture correspondence under the current transform.
Landmark residuals throughout the intended domain and positive mechanical
tissue evidence remain unestablished. See the [private storage ledger](local-storage-ledger.md).

In the PTNT_5387 follow-up, nearby candidate landmarks in the native TOF
fit the existing transform about as well as the selected candidates, and a
diagnostic fit missed a held-out point by 2.37 mm. The provisional TOF points
also fell short of the 3D spread requirement. The review retains the raw
landmark and alternative crops with the unchanged transform.

PTNT_7195 had positive estimated tissue support near 71 of 77 sites on the
original native trace. An alternative field preserved six measurable native
ventricular boundaries across three directions. The three closest edges
belonged to the same lateral wall, and a deliberate 1.5 mm superior shift
still matched five edges within 1 mm. The local rehearsal review remains
withheld. [Private source evidence index](local-storage-ledger.md)

The Scans review panel also supports a review for one native-wall field. Select
the reference brain estimate, moving angiography scores, and review position.
Pair at least four spatially distributed landmarks. The review checks nearby
correspondence in the proposed field and the structural and score coverage.
Open **Rehearse → Local native wall** and choose **Prepare reviewed local wall**.
Choose the source pair first when more than one is offered. The receipt binds
the exact native crop, candidate transform, original
scans, and selected label and score revisions. General planning and display
retain the scan's existing alignment status.

TopCoW-012 exercised this route with same-person CTA brain estimates and native
MRA vessel scores. Four arterial junctions withheld from fitting had 0.34–1.00
mm landmark residuals. The three retained native MRA traces were within 0.34 mm
at the 95th percentile of the independent CTA model positives. The left
PCom–ICA crop had complete sampled coverage in both acquisitions. A shifted
crop was refused. Acquisition timing is unreported, and anatomical boundaries
and tissue mechanics carry the limitations of the acquired images and models.

The TopCoW-014 CTA study qualified in limited research mode using its existing
CTA-derived brain estimate and the stock vesselFM mask and scores on the same
acquired lattice. The model scores came from the frozen annotated CTA
comparison. The publisher contour served as same-voxel inspection guidance.
The simulated domain contains the selected CTA-derived brain and vessel
estimates. Both contacted voxels had strong acquired contrast. An exploratory
aspiration stopped at the material guard. The final recording used a shorter
stroke that accepted all 12 actions, reproduced from a cold process at every
action boundary.

The modeled aspiration removed three brain-estimate cells (0.687 mm³),
releasing two assumed supports. The same dissector movement displaced 505 wall
vertices, with maximum endpoint displacement of 0.046 mm before the release
and 0.047 mm after it. At the final withdrawal the residual was 0.0013 mm.
An equal-duration inactive control retained the supports and reproduced the
0.046 mm peak with the same two contacted cells. Passive advance into the
retained tissue encountered the collision guard, so the control waited at
the last clear pose before repeating the same dissector stroke. Access,
attachment and constitutive assumptions apply throughout. The six
other inspected CTA studies used brain masks declared for display context and
were refused by the local mode. The IXI265 T1 and MRA showed good central
ventricular alignment, with unresolved cross-modality correspondence across
the prospective working domain, so its transform stayed unreviewed.

The [private storage ledger](local-storage-ledger.md) locates the TopCoW-014
source study, action record and local source sections. To reproduce the
preparation, select **Estimated brain envelope** and **Unclassified vessel
candidates · vesselfm** in the native study and place the patient cursor at the
RAS position recorded in its preparation receipt. Use **Prepare native wall
at cursor** in the Rehearsal panel. The saved replay opens directly in the
operative workspace.

The PTNT_4509 teaching rehearsal preserves the acquired TOF lattice and
source-bound vessel triangles. The exposure banks, vessel attachments and
mechanical properties are modeled. The private continuity run index links the
portable workspace, recorded actions, source and collision checks, and hidden
replay commands. See the [private storage ledger](local-storage-ledger.md).

The open-field display contour audit uses all 29 available stored native TOF
fields. It compares exact closed section polygons, fixed open rims, and new
contacts between triangles without shared vertices. The per-case results and
plane-sensitive junction measurements are indexed in the [private storage ledger](local-storage-ledger.md).

Gray faces in a score-bound operative view mark where the source volume ends.
The marker follows the rest coordinates of the wall through deformation and
distinguishes the source boundary from interior transection caps. These faces
retain the same opaque geometry for contact and source inspection.

For a new score-bound rod rehearsal, pass `--guard-native-wall` to
`scripts/prepare_tof_bifurcation_rehearsal.py`. The saved materials policy
checks the continuous motion of nonadjacent score triangles at each rod step.
The engine checks new caps at a topology edit, including steps with no rod
motion. A rejected stroke leaves the tool and vessel at the pre-stroke state.
The verified native wheel accelerates the same check available through the
Python reference implementation.

This policy uses the native contact and display owner map. It protects against
intersections between nonadjacent faces along a linear wall interval, including
across source labels. Closely spaced adjacent faces can still fold under load.
The preparation flag records the policy in the scene identity so earlier
recordings remain reproducible with their original materials declaration.

Pass `--vertex-wall-transport` to use the guarded native wall with per-vertex
motion. Native contact uses the transported triangles, and the renderer binds
the same offsets to the native score mesh or to the delivered, capped cut
mesh. Very short native mesh edges share their displacement weights to reduce
local folds. That step preserves the original vertex positions and source
owners. An edit that requires a cuboid fallback is rejected by this profile.

Research staging can use `--source-graph-fork --vertex-wall-transport` with
`--subject` and `--junction-native I J K`. Use a degree-three junction in
the subject's saved native source graph. The procedure checks input hashes
and the native grid, then builds a physical-size crop with three contiguous,
face-connected source legs. The saved workspace centers the microscope at
the junction and faces it toward the exposed side of the modeled banks. The
scene receipt identifies the graph selection and its research-only simulation
status. Omit `--source-graph-fork` to use the manually reviewed source-region
workflow.

## Prospective intake receipt

`resectionlab.eden2020.inspect_local_case` produces a versioned evidence
manifest from already permitted local NIfTI and gradient files. It performs no
network access, inference, conversion, registration, resampling, acquisition,
training, or automatic collection insertion. It returns both
`planning_admitted=False` and `training_admitted=False` even when intake checks
pass. This is a prerequisite receipt for a future patient collection adapter,
not an alternative to existing planning admission or a claim of anatomical QC.
It cannot authenticate a human permission or quality-review assertion; its
caller must verify those records and retain their original evidence.

### Inspecting a native diffusion baseline in Scans

Import a 4D NIfTI diffusion acquisition with its matching same-name `.bval`
file and any `.bvec` file. Under **Import into this study**, select **Extract
diffusion baseline** and open the result in Scans. The study retains the raw
acquisition and companion identities. The inspection scan takes the first
measured volume at the minimum b-value (at most 50 s/mm²), preserving its
native samples and coded millimeter grid. A structural reference takes
precedence when available. The separate baseline opens in its native frame
until the usual registration workflow estimates and reviews a transform.

The baseline adds no motion or susceptibility correction, gradient-frame
declaration, tract reconstruction, functional localization, or segmentation.
Those uses require their own evidence. The private verification exercised
ordinary intake, baseline extraction and native views for both the DTI and
NODDI acquisitions of two archived EDEN controls. The private storage catalog
indexes those verification studies.

## Declaration interface

Call `inspect_local_case(local_root, declaration, permission=permission)`.
`local_root` should be a patient directory under Git-ignored
`testcases/downloaded-datasets/`. Check existing content hashes before adding
files; retain provenance and permission records locally. Generated tests use
only pytest temporary directories.

The permission mapping requires `status="granted"`, the exact `doi`,
`intended_use="research_planning"`, `license_status="reviewed_permitted"`, a
raw 64-character `evidence_sha256`, and a `subjects` list containing the declared
subject. Missing approval fails before local path resolution or file access.
Do not put account credentials or access tokens in receipts.

The declaration requires `doi`, `subject`, `role` (`TRAIN` or `SELECT`),
`prospective_before_inspection=True`, `split_evidence_sha256`, `series`, optional
`annotations`, and `intended_use_review`. Preserve all visits and derivatives
under `patient_group="EDEN2020:<subject>"`. `EVAL` is refused. This adapter
neither allocates nor changes splits; the supplied split evidence must come
from a separately frozen prospective role declaration. SELECT never authorizes
optimizer updates.

`series` maps `T1_POST`, `FLAIR`, `TOF_MRA`, `SWI`, `DTI_32`, optional `NODDI`
and `B0_REVERSE` to rows with `subject`, root-relative `path`, raw `sha256`,
`metadata_review`, `registration_review`, and `coverage_review`. A missing
required sequence or unfinished review becomes an explicit blocker. SWI phase,
DTI scalar maps and additional acquisitions should be preserved separately in
the source inventory until a versioned extension gives them explicit semantics.
No filename heuristic infers modality or patient identity.

DTI/NODDI additionally require `bval`/`bvec` file records (`path`, `sha256`) and
`gradient_frame` (`voxel`, `scanner_RAS`, or `scanner_LPS`). Bvals must have one
finite nonnegative value per volume; bvecs must be 3×N with finite unit vectors
for b>50 and near-zero vectors for b0. Both baseline and weighted volumes are
required. DTI_32 requires 32 weighted volumes for intake qualification. These
checks do not establish valid directions, expected shell distribution,
eddy-current correction, susceptibility correction, or reorientation history.
Those remain acquisition/diffusion QC work.

`annotations` optionally maps `brain`, `tumor`, `vessels`, `tracts` to same-subject
file records plus `provenance` (`supplied_annotation` or `model_estimate`) and
`review_status`. Model estimates also require a reviewed permitted
`model_license_status`. Masks must be nonempty binary arrays on the reference
T1 physical grid. Brain and tumor masks and independent intended-use review are
required to clear intake blockers. Segmentation provenance, model/weight/run
hashes and detailed QC evidence must be retained by the owning review workflow;
this intake does not create them or certify estimates as acquired truth.

## Quality-control checklist before downstream admission

- Authenticate access approval for the particular subject and intended use;
  review dataset terms, derived-asset redistribution, and model licenses.
- Freeze prospective patient roles before opening files; reconcile aliases,
  visits, related releases, and prior exposure against existing splits.
- Verify source hashes, NIfTI millimeter units, finite invertible affine,
  explicit agreeing qform/sform, orientation, spacing, and acquisition metadata.
  Conflicting or missing forms fail closed. Never repair headers silently.
- Compare physical fields of view, distortion and same-subject anatomical
  alignment. Equal dimensions/affines do not prove alignment. Different grids
  remain separate and blocked until an independently reviewed source-bound
  transform and a future explicit resampling adapter are supplied.
- Validate DTI and NODDI bval/bvec counts, norms, coordinate convention,
  direction/shell metadata, motion/eddy and susceptibility correction,
  reverse-phase acquisition and gradient rotation. The receipt never rotates
  gradients or claims successful tractography.
- Independently review brain/tumor annotation semantics and coverage; the
  release description does not promise clinician-reviewed tumor masks.
- Treat TOF as incomplete arterial signal. Review vessel extraction, SWI phase
  semantics, false positives, small-vessel visibility, venous coverage and
  alignment. Do not infer comprehensive venous topology from TOF or SWI.
- Review tract extraction and coverage separately from functional localization;
  DTI/NODDI and distance to a tract do not calibrate neurological injury.
- Preserve missing skull CT, functional mapping and anatomy outside supplied
  coverage as unknown. No CT, fMRI, complete vessel map or operative tool
  trajectory is assumed available. The images do not supply offline RL
  observation/action/next-state/reward labels or clinical counterfactuals.
- Pass a distinct downstream case/evidence contract before any opt-in
  post-exposure simulation. Keep acquired images, inferred anatomy, assumed
  exposure and simulator-generated actions/rewards explicitly distinguished.

Reproduce the bounded generated controls:

```sh
.venv/bin/python -m pytest -q tests/test_eden2020.py
```
