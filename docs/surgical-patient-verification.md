# Patient-derived surgical toolbox

The toolbox now runs on an intact, real same-patient CT/MRI reconstruction with
brain, bone, tumor, and vessels. The prepared local case is
`outputs/surgical-patient-sub06/patient.ressectionlab`. Open it in the desktop,
choose **Rehearse → Prepare instrument scene**, and use Preview/Apply. No anatomy
is automatically exposed. The original collection and imaging files are unchanged.

## Source and reconstruction

The source is the local revision-7 `ZENODO-17486320-meningioma-sub-06` collection,
from [A Paired Head CT-MRI Dataset for Cross-Modality Image Synthesis](https://zenodo.org/records/17486320).
The publisher supplies paired CT/MRI; the four segmentations used here are cached
estimates, not publisher-provided ground truth. No additional patient or generated
anatomy was needed:

| Material | Existing segmentation | Exclusive simulation volume |
|---|---|---:|
| Brain / soft tissue | SynthStrip brain extraction | 1,389,147 mm³ |
| Tumor | Raidionics meningioma proposal | 43,573 mm³ |
| Bone | CT ≥300 HU density proposal | 732,802 mm³ |
| Vessels | Experimental vesselFM structural-MRI estimate | 141,681 mm³ |

The 220×220×181 grid uses 1 mm cells. MRI is linearly resampled; brain and tumor
use nearest-neighbor samples. Bone and vessels use any-hit samples at ≤0.5 mm
spacing within each output cell. This expands coarse occupancy and can still
miss finer structures; it does not establish vessel completeness. Four overlapping
annotation masks remain stored. Exclusive material ownership follows the saved
precedence brain < tumor < bone < vessels, avoiding double-counted simulation
volume. Priority survives bundle JSON key sorting and reopening.

CT registration remains **unreviewed**, including the recorded 8.9% brain/dense-bone
conflict. Export requires an explicit research opt-in and does not change source
review status or relax the native adapter's registration gate. The report retains
source manifest, transform graph, layer provenance, checksums, sampling method,
annotation overlaps, and ownership volumes. Bone includes other density structures;
the vessel estimate includes unvalidated components. These are limitations of this
research environment, not evidence of complete or correct anatomy.

## Material behavior

Bone remains rigid. Drill and craniotome remove compatible bone cells and produce
debris; removal refreshes connected components. Large components use bounded
summaries instead of huge cell lists. Separability remains unknown without supplied
anchors. There is no calibrated cutting-force, fracture, or bone-density law.

Brain/tumor retain the existing cell connectivity, CUSA removal, and reversible
lattice-patch retraction. This is not elastic FEM, brain shift, or patient-calibrated
mechanics. Missing membranes, attachments, functional evidence, and inflow anatomy
are not manufactured.

Vessels have an explicit opt-in `resectionlab.tissue-physics.v1` profile. A micro-scissor
cut severs local vessel-cell adjacency and retains both pieces. Each newly severed
adjacency creates one modeled wound outlet; intact anatomy does not spontaneously
bleed. Wounds maintain independent continuity/patency, thermal exposure, bleeding
status, generated fluid, and reservoir fields. Bipolar seals the wound without
repairing the cut. Frazier suction removes available fluid/debris, not vessel tissue.

The outlet uses a capped [Poiseuille flow approximation](https://openstax.org/books/college-physics-2e/pages/12-4-viscosity-and-laminar-flow-poiseuilles-law):
`Q = min(cap, π r⁴ ΔP / (8 μ L))`, using mm, Pa, Pa·s, and mm³/s.
The export explicitly assumes an **equivalent wound aperture** radius 0.15 mm,
pressure difference 1000 Pa, viscosity 0.0035 Pa·s, length 5 mm, cap 50 mm³/s,
and coagulation threshold 2 exposure units. These are model settings, not inferred
vessel diameter or measured physiology. One outlet per cut adjacency makes totals
grid-dependent. There is no pressure network, pulsatility, collateral circulation,
wall elasticity, or tissue perfusion model. Global perfusion is explicitly unknown.

## Reproduce

```sh
PYTHONPATH=src .venv/bin/python scripts/prepare_surgical_patient.py \
  /Users/shaf/Downloads/RessectionLab-Native-Anatomy-sub-06 \
  outputs/surgical-patient-sub06 \
  --brain brain --tumor tumor --bone bone --vessels vessels \
  --resolution-mm 1 --allow-unreviewed-registration

PYTHONPATH=src .venv/bin/python scripts/verify_surgical_patient.py \
  outputs/surgical-patient-sub06/patient.ressectionlab outputs/surgical-patient-sub06

PYTHONPATH=src .venv/bin/python scripts/verify_patient_vessels.py \
  outputs/surgical-patient-sub06/patient.ressectionlab \
  outputs/surgical-patient-sub06/vessel-verification.json
```

The first verifier searches bounded straight access rays as a software smoke test,
not a clinical route recommendation. It uses public transitions only, with no
pre-removed cells. The successful sequence removes 698 mm³ bone, stops at incompatible
tissue, withdraws/exchanges to CUSA, and removes 27 mm³ target tumor with zero vessel
removal. Preview matches Apply; replay and branch checks pass. The saved demonstration
is `patient-tool-demo.ressectionlab`, with action log and sparse checkpoint.

The second verifier uses an already exposed component of the experimental vessel
estimate near the grid edge. It does **not** establish that this component is a real
anatomical vessel. Five local connections are cut, five modeled wounds form and are
sealed by bipolar, and suction removes 2.7083 mm³. Fluid accounting and exact replay
pass, with no solid removal. Generated fluid is a model output, not predicted patient
blood loss. The source MRI and label-array hashes remain unchanged in both tests.

## Verification and next use

The focused surgical, geometry, and rehearsal tests pass; desktop main, renderer,
and viewer suites pass (678 tests), as does the production build. Backend timings
on this Mac for the successful six-step ray were 6.8–2401 ms **for preview plus Apply**,
including repeated state copies and component summaries; this is not frame latency.
The search took 73.4 seconds, including rejected candidate paths, with 449 MB peak
Python RSS. UI and GPU memory are not included in that figure.

The deterministic `reset/observe/preview/step/snapshot/restore/replay/branch` API
remains the interface for future agents. Observations and detailed world inspection
stay separate. No RL training, reward policy, or clinical outcome claims are added.
Before using this as a credible RL benchmark, anatomy/registration review, vessel
coverage assessment, material calibration, deformation and hemodynamic qualification,
and task-specific reward/termination definitions remain necessary.

Actual Electron interaction reproduced the same 698 mm³ bone / 27 mm³ tumor
removal using numeric pose controls, Preview, Apply, withdrawal, and tool exchange.
All four layers are identified as unreviewed estimates in the anatomy panel. The
native Save dialog wrote `patient-desktop-verified.ressectionlab`; a fresh Python
process verified its five-action checkpoint and state identity
`sha256:1c0c2b70b774381fa6c8f708a3ebbb85f43a8f2246ab4b96db8f75b96d52fec3`.
The restarted desktop also reopened this file. The screenshot
`outputs/surgical-patient-sub06/patient-tool-desktop.png` records the 3D reconstruction,
linked source slices, and modeled removal metrics. All exported patient assets are
local ignored outputs rather than repository fixtures.

A third public-action check, `scripts/verify_patient_retraction.py`, held one
exposed brain-estimate cell, displaced it outward by 2 mm, verified that collision
occupancy and reported moved geometry agreed, and released it back to its original
location. Held and released states replayed exactly; no source cells were injected
or pre-removed. Its receipt is `retraction-verification.json` in the output directory.
This is a constrained-patch software check, not validation of brain mechanics.

Compatibility audit: an older local `surgical-desktop-verified.ressectionlab`
engineering artifact has a pre-existing replay identity mismatch. Replaying with
the previous committed engine and with this change produces the same current state,
which differs from that old artifact's stored identity. The old file is untouched
and remains rejected, rather than silently migrated. New patient saves replay
exactly; legacy bone-component checkpoint field shapes are regression-tested.
