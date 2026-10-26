# Fresh glioma import and reconstruction

This is the first integration milestone for the October 26, 2026 rehearsal goal.
Two previously unused development cases, UCSF-PDGM-0005 and 0007, now pass the
same import, brain reconstruction, simulation preparation and save/reopen path.
They are public structural mirror cases, not independent clinical evaluation.
The supplied enhancing label is the selected target. Skull, vessels, functional
mapping and tracts are unavailable; no missing anatomy is synthesized.

## Reproduce the two cases

Use the primary checkout and its installed base/anatomy Python environment.
All patient bytes and execution receipts remain under Git-ignored `testcases/`.
The declaration is metadata-only and precedes image access. A conflicting
declaration is refused; repeating the matching declaration preserves it.

```sh
.venv/bin/python scripts/prepare_fresh_ucsf_demo.py --declare-development \
  --case UCSF-PDGM-0005 --case UCSF-PDGM-0007 \
  --selection-rationale "First two ascending pinned mirror IDs after previously used 0004, before pixel access; development only."
.venv/bin/python scripts/prepare_fresh_ucsf_demo.py --case UCSF-PDGM-0005 --infer-brain
.venv/bin/python scripts/prepare_fresh_ucsf_demo.py --case UCSF-PDGM-0007 --infer-brain
.venv/bin/python scripts/verify_fresh_ucsf_demo.py --case UCSF-PDGM-0005
.venv/bin/python scripts/verify_fresh_ucsf_demo.py --case UCSF-PDGM-0007
```

The source is [MedOtter/UCSF-PDGM](https://huggingface.co/datasets/MedOtter/UCSF-PDGM/tree/e9372219cf1cd2fdd52260cd45f7514b4aa7638e)
at revision `e9372219cf1cd2fdd52260cd45f7514b4aa7638e`, declared CC BY 4.0.
Attribution: Calabrese et al., *The University of California San Francisco
Preoperative Diffuse Glioma MRI Dataset*, [dataset DOI](https://doi.org/10.7937/tcia.bdgf-8v37).
LFS SHA-256 and size checks bind each downloaded file to that mirror; equivalence
to official TCIA bytes remains unknown. Pretrained model exposure is unknown.
The mirror images are skull-stripped, co-registered T1/T1CE/T2/FLAIR. Their
released affine is retained. Source-provided registration is not a landmark
validation or scanner-native reconstruction.

The scripts retain label 1 (core/necrosis) and 2 (edema) as source context and
select label 4 (enhancing tumor) as this demonstration's operative target.
This is an explicit software task definition, not a resection recommendation.
The source mask remains intact. Brain is a separate unreviewed SynthStrip
estimate. The scene and native manifest identities are checked after reopening.

Outputs are in
`testcases/outputs/integrated-goal-20261026/fresh-patients/<case>/`:
`native/patient-anatomy.json`, `scene/patient.ressectionlab`, preparation and
independent-verification receipts. A separate `--package-root` beneath
`testcases/` supports an isolated reproduction. Existing same-sized source
files are hash-checked for deduplication before new imaging is downloaded.

## Desktop workflow and smoke test

In **Cases → New patient study**, create a pre-op study. **Data** now exposes
explicit T1, T1CE, T2, FLAIR, CT and angiographic sequence choices for NIfTI and
DICOM. A DICOM folder with multiple MR series requires UID-keyed typing through
the backend; the desktop single-series control asks for one series per folder.
Raw DICOM `MR` is an acquisition family and is refused as a semantic sequence.

For a segmentation, list every nonzero numeric label and choose its operative
role. Missing, duplicate and unsupported mappings are refused. Background 0 is
preserved. Imported anatomy with unknown or non-mm units cannot silently enter
the mm simulation grid. **Reconstruct anatomy** includes brain, CT skull,
sequence-specific glioma, vessel and existing parcellation adapters.

Inspect the linked slices and 3D anatomy, save a portable patient study, then
choose **Rehearse → Prepare standard environment**. Select existing layers;
missing structures remain unknown. Estimated alignment requires an explicit
research opt-in and retains its review status. Save and reopen the prepared
simulation as a `.ressectionlab` file.

The actual Electron smoke test selects local source files through native-dialog
stubs; all React controls, preload IPC, Python operations and anatomy are real.
It creates a study, imports four MRI sequences and source labels, invokes the
real cached brain adapter, saves/reopens the study, prepares the simulation,
and saves/reopens it. Use a new output directory on each run:

```sh
pnpm -C desktop build
FRESH_PATIENT_SOURCE="$PWD/testcases/downloaded-datasets/UCSF-fresh-development-20261026/UCSF-PDGM-0005" \
FRESH_PATIENT_OUTPUT="$PWD/testcases/outputs/integrated-goal-20261026/electron-new-run" \
desktop/node_modules/.bin/electron desktop/electron/verify-fresh-patient.cjs
```

Use the identical command with source `UCSF-PDGM-0007` and another output
directory for the second case. Screenshots wait for source loading and surface
preparation. The report distinguishes UI checks from headless source checks.

## Model weights, runtime and licenses

Weights stay outside Git. Existing `scripts/poc_model_assets.py` installs
SHA-pinned SynthStrip, vesselFM and TotalSegmentator assets. Their current
adapter declarations live in `src/resectionlab/poc_models.py`:

| Adapter | Input | Installation and license |
|---|---|---|
| SynthStrip | Structural MRI | `scripts/poc_model_assets.py`; offered MIT weights, separate FreeSurfer runner license |
| TotalSegmentator skull | Acquired CT in HU | Same installer, pinned Dataset297 fast weights; Apache-2.0 declaration retained |
| vesselFM | MRA/TOF or CTA | Same installer; vesselFM Open RAIL++-M research/non-commercial restriction retained |
| Experimental vesselFM | Explicit T1CE | Same weights; experimental domain, vascular completeness unvalidated |
| Raidionics GBM | T1CE plus explicitly selected same-scan brain mask | Command below; publisher BSD-2-Clause notice retained |
| Raidionics LGGlioma | FLAIR plus explicitly selected same-scan brain mask | Command below; same publisher notice |

```sh
.venv/bin/python scripts/poc_model_assets.py
.venv/bin/python -m resectionlab.reconstruction_assets \
  --model glioma-gbm --model glioma-lgg
```

The [official Raidionics 1.2.0 release](https://github.com/raidionics/Raidionics-models/releases/tag/1.2.0)
archives are pinned by exact size and locally recorded SHA-256. These are local
fixity pins of official release bytes, not published publisher checksums. The
installer preserves original model configurations and refuses different existing
files. It saves per-file hashes and the
[BSD-2-Clause notice](https://github.com/raidionics/Raidionics-models/blob/1.2.0/LICENSE).

The tested runtime is RaidionicsSeg 1.5.2/ONNX Runtime 1.31.0 in
`.tools/raidionics-packages`, separate from the SynthStrip/vesselFM/CT runtime
at `.tools/poc-model-runtime/venv/bin/python`. Runtime discovery honors
`RESECTIONLAB_RAIDIONICS_PACKAGES` and `RESECTIONLAB_RAIDIONICS_MODELS`.
The adapter binds all runtime Python sources, model/config hashes, input image
and selected brain mask. It stages the publisher root-level ONNX file as one
hash-checked fold for the installed runner, without modifying installed assets.
The original brain preprocessing is retained. Wrong sequence, absent brain,
cross-scan brain or mismatched geometry is refused.

A real GBM-model run on fresh 0005 produced an unreviewed proposal in 85.66 s
with 2.86 GB peak sampled process-family RSS. It did not change the original
native manifest or simulation target. The descriptive comparison script is
`scripts/compare_fresh_glioma_estimate.py`; generic model Tumor and supplied
enhancing label 4 are not proven equivalent classes. Model scores are not
calibrated confidence. Newly imported model tumor proposals have context roles
until an environment is explicitly prepared with a selected target layer.

SynthSeg/FastSurfer and their existing runtime gates remain available. Missing
weights or unsupported input produces an error, never substitute anatomy.

## Remaining integration gates

This milestone establishes fresh-case software integration. The later
[integrated demonstration](integrated-goal-demonstration.md) adds actual corridor
comparison and a trained continuation on fresh 0007 while retaining a failed
frozen assessment on 0005. Three complete
access approaches, matched multi-seed RL assessment and learned execution on
the fresh glioma scenes remain separate gates. Skull access, complete vascular
coverage, functional anatomy, neurological/perfusion probabilities and clinical
validation remain unavailable. Simulator-generated experience must retain its
label and must never be presented as patient observation.
