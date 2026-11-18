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

Create a study in **Cases → New patient study**. In **Scans**, choose the available
NIfTI files or DICOM folders together, review ambiguous sequence assignments, and
confirm the patient and timepoint. Attach supplied annotations to the scan they
describe and set the meaning of each nonzero source label. The UCSF example
uses label 1 for nonenhancing core or necrosis, 2 for edema-like abnormality,
and 4 for the enhancing lesion.

Use **Review preparation plan → Prepare anatomy**, then **Anatomy** to inspect
the source slices and estimates. Explicitly select a candidate in **Plan &
compare**. Geometric comparisons retain source coverage and review status.
**Save patient study file…** packages the study and comparisons in one
`.rsworkspace` file. **Rehearse → Prepare standard environment** remains a
separate capability with its own geometry requirements.

Build the desktop once from the repository root. The verifier uses hidden
Electron windows and supplies local paths to the file dialogs. It exercises
the renderer, preload, Python engine and actual patient arrays.

```sh
pnpm -C desktop build
source_case="$PWD/testcases/downloaded-datasets/UCSF-fresh-development-20261026/UCSF-PDGM-0005"
evidence="$PWD/testcases/outputs/adaptive-ucsf-0005"
mkdir -p "$evidence"
ADAPTIVE_SOURCE="$source_case" ADAPTIVE_OUTPUT="$evidence/create" \
  desktop/node_modules/.bin/electron desktop/electron/verify-adaptive-patient.cjs
ADAPTIVE_PHASE=reopen ADAPTIVE_EXPECTED="$evidence/create/report.json" \
  ADAPTIVE_SOURCE="$source_case" ADAPTIVE_OUTPUT="$evidence/reopen" \
  desktop/node_modules/.bin/electron desktop/electron/verify-adaptive-patient.cjs
ADAPTIVE_PHASE=incremental ADAPTIVE_EXPECTED="$evidence/create/report.json" \
  ADAPTIVE_SOURCE="$source_case" ADAPTIVE_OUTPUT="$evidence/incremental" \
  desktop/node_modules/.bin/electron desktop/electron/verify-adaptive-patient.cjs
```

The create phase imports T1 post-contrast and T2 with the supplied annotation.
The separate reopen process reads the saved file. The incremental phase copies
the same study into its own evidence directory, adds T1 and FLAIR, and re-adds
T2 to check deduplication. Use fresh phase directories to preserve prior captures.
If verification stops after preparation, a new create run with
`ADAPTIVE_RESUME=/absolute/path/to/the/preceding/report.json` continues through
the normal **Open** control using a copy of the exact prepared study.

Use the `create` and `reopen` commands with
`source_case="$PWD/testcases/downloaded-datasets/RessectionLab-Full-Anatomy-Patient/candidate-sub-06"`
and a separate evidence directory to exercise the acquired meningioma T1
post-contrast profile. That profile uses the source-session-unknown timepoint
and explicitly requests the pinned meningioma adapter. Its candidate keeps
the model's unreviewed status. Set `ADAPTIVE_CANCEL=1` during a create run
with pending work to exercise cancellation and resume.

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
