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
