# Independent PAT28 extraction check

October 4, 2026. The separate imaging/QC agent reviewed the implementation,
native outputs, pinned source/model records, original-versus-MPS-runner diff, and
six-plane overlay. The machine-readable result is
`brain-extraction-independent-qc.json`; regenerate it with
`.venv/bin/python scripts/audit_brain_extraction.py`.

The audit checks the final `PAT28-mps-v4` artifacts against the unchanged creator
T1 and source annotation, the separate pinned model manifest, and the preceding
three complete runs. Source, model, runner, implementation snapshot, and output
hashes agree. Both mask and predicted-distance artifacts retain the original
160 × 256 × 256 RAS millimeter grid with finite values. An independent NiBabel
orientation transform reproduces the source-annotation flip and 10,269-voxel
threshold scenario. Independent component selection and hole filling of each
predicted distance field below the declared 1-mm border reconstruct its saved mask
with **zero differing voxels**. Masks and predicted-distance arrays are identical
across all four stored MPS runs; this audit did not run another network inference.

| Independent measurement | Main | No-CSF |
| --- | ---: | ---: |
| Estimated envelope volume, mL | 1529.550 | 1337.639 |
| Connected components | 1 | 1 |
| Input-grid face contacts | 0 | 0 |
| Threshold-derived annotation voxels excluded | 0 | 214 |

The 214-voxel discrepancy remains a review flag. Complete source-annotation
inclusion by the main model is **not accuracy**, and the no-CSF model has not been
rejected or corrected solely to increase that overlap. Source intensities are a
fractional annotation, not probabilities. The learned distance output includes an
upstream exterior fill of 100 mm and is not an independently measured surgical
clearance field.

The sampled views make the intensity comparator's extracranial/neck support and
missing brain regions visible. Model envelopes are more plausible in those
planes, but retain inferior structures and provide neither a pial surface nor a
reviewed cerebral access region. This was limited engineering inspection, not
expert anatomical review or whole-volume accuracy certification.

Two audit findings were repaired by the extraction owner without changing prior
artifacts: the wrapper now validates predicted-distance shape/frame/finite values
before reporting success, and every omitted source-annotation voxel is counted
and flagged instead of hiding omissions below 2%. Actual-child malformed-output
tests cover these repairs. Nine independent audit tests additionally check source
tampering, axis flips/translations, nonfinite/fractional outputs, orientation
reindexing, physical volumes, one-voxel omissions, and mask/SDT disagreement. The
combined extraction/audit suite passed 26 tests.

## Saved evidence and review contract

Store each estimated envelope separately from `CaseData.brain_mask` and target
compartments. A structural-evidence record needs the immutable native-grid mask;
case-image/frame identity; source-file, mask, checkpoint and run/report hashes;
model variant and parameters; QC flags; and `review_status=review_required`.
Persist its array and metadata together in the portable case bundle. Importing or
saving it must not enable planning or mark it reviewed.

The interface should show “Estimated brain envelope · review required,” offer
main/no-CSF and discrepancy overlays over the original MRI, and link the source
and QC record. A deliberate user review records reviewer identity, time, decision,
scope and the exact evidence hash. Corrections create a new mask version and
invalidate that review. A whole-brain-envelope review does not establish cortex
or permit a cortical window: retain `cortex_localized=false` and
`cortical_access_permitted=false` until the separate access contract is satisfied.
Keep the full-head nonzero-intensity shortcut disabled.

The separate evidence contract and portable bundle support are implemented. The
tested import below creates two `review_required` proposals while retaining the
existing MRI, annotations, empty working brain mask, planning-input hash and
source files. It checks the source file against both case provenance and current
MRI values/frame; mask bytes/grid and the pinned checkpoint against the report;
and case identity after saving and reopening.

```sh
.venv/bin/python scripts/attach_brain_evidence.py \
  --case outputs/cases/BTC-sub-PAT28.ressectionlab \
  --source-image data/diffusion_source/ds001226-v5.0.1/sub-PAT28/ses-preop/anat/sub-PAT28_ses-preop_T1w.nii.gz \
  --report artifacts/brain-extraction/PAT28-mps-v4/brain_extraction_report.json \
  --output outputs/cases/BTC-sub-PAT28-structural-evidence.ressectionlab
```
