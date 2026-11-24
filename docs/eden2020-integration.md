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

Run the component, branch and package steps from the primary checkout.
Replace the repeated subject selection with `--cohort` to resume the first
two steps for the whole restricted source inventory.

```sh
.venv/bin/python scripts/run_eden2020_vascular_review.py --subject PTNT_4509
.venv/bin/python scripts/run_eden2020_vascular_review.py --graph --subject PTNT_4509
.venv/bin/python scripts/build_eden2020_vascular_companion.py --subject PTNT_4509 --linked-source
```

The private run is under
`qc/eden2020-glioma-5714812/vascular-reconstruction-20261123-v1/` on the data
volume. Read `workflow.json` for the frozen model and cleanup policy,
`processing.json` for per-subject processing, and `graphs.json` for branch
coverage. Open the source-linked `vascular-review.anatomy.json` or a standalone
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

The PTNT_4509 teaching rehearsal preserves the acquired TOF lattice and
source-bound vessel triangles. The exposure banks, vessel attachments and
mechanical properties are modeled. The private continuity run index links the
portable workspace, recorded actions, source and collision checks, and hidden
replay commands. See the [private storage ledger](local-storage-ledger.md).

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
