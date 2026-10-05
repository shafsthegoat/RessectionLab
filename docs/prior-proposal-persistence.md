# Registered prior proposals: local persistence and inspection

The seven saved UCSF-PDGM-0004 atlas previews can be imported into a separate
`CaseData.prior_proposals` collection and reopened without the original atlas
cache. They are **population priors, view only, alignment review required**.
This collection has no acceptance operation and is never an input to planning.
Viewing an overlay, changing its display threshold, importing or saving a case
does not approve alignment or establish patient function.

Each immutable record stores the registered values and a separate binary atlas
sampling-coverage array. Its manifest binds the raw map source, source-grid and
array hashes, registered-array and coverage hashes, physical template-to-patient
transform, interpolation, method/version, original registration-case hash,
report/inventory/preview hashes, registration template identity/hash, current
display image/frame and exact compartment identity. Source atlas/template arrays
are not embedded. Map values are unitless; physical coordinates are millimeters.

The UCSF experiment used the supplied **T1** for registration and **T1c** for the
desktop case display. The importer checks their distinct pinned files and exact
common physical grid, compares displayed pixels with T1c, and checks the source
segmentation against every original and active compartment. Swapping tissue
labels fails even when the total lesion union is unchanged. The historical
registration-case hash is retained as provenance and is not compared with the
new case hash after proposal attachment. The current planning-input hash, source
image/frame and exact lesion-label hashes provide the ongoing binding.

Before import, the saved transform must match one unchanged candidate in the
registration report. Every source map is loaded from its hash-pinned manifest;
its resampling is reconstructed and compared with the saved registration hash,
inventory, preview values, coverage and target affine. This applies a saved
transform; it does not fit another registration. Nonfinite or nonbinary data,
source substitutions, altered preview arrays, reflections, stale cases, forged
review states, path escapes and excessive source/preview sizes are rejected.
Cancellation discards the entire pending import without attaching partial data.

Portable bundles retain only the registered values/coverage plus provenance.
An empty proposal collection is omitted from the manifest, preserving existing
case hashes. Adding proposals changes the full case identity but leaves the
planning hash unchanged. Original source arrays and original case files remain
unchanged. JSON/NumPy storage never uses pickle.

## Reproduce the local seven-layer import

```sh
.venv/bin/python scripts/attach_prior_proposals.py \
  --case outputs/cases/UCSF-PDGM-0004.ressectionlab \
  --registration-directory outputs/prior_registration/UCSF-PDGM-0004/final_v3 \
  --source-cache-directory data/functional_priors \
  --prior-manifest manifests/functional_priors.json \
  --source-image data/public_mirrors/MedOtter-UCSF-PDGM/e9372219cf1cd2fdd52260cd45f7514b4aa7638e/UCSF-PDGM-0004/UCSF-PDGM-0004_T1c.nii.gz \
  --registration-image data/public_mirrors/MedOtter-UCSF-PDGM/e9372219cf1cd2fdd52260cd45f7514b4aa7638e/UCSF-PDGM-0004/UCSF-PDGM-0004_T1.nii.gz \
  --lesion data/public_mirrors/MedOtter-UCSF-PDGM/e9372219cf1cd2fdd52260cd45f7514b4aa7638e/UCSF-PDGM-0004/UCSF-PDGM-0004_tumor_segmentation.nii.gz \
  --output outputs/cases/UCSF-PDGM-0004-prior-proposals.ressectionlab
```

The actual seven-layer import, save and reopen succeeded on October 4, 2026.
Choose a new output filename when repeating this command; existing artifacts are
never overwritten by this helper.
[`prior-proposal-roundtrip.json`](prior-proposal-roundtrip.json) records the exact
local bundle and layer hashes. All layers remain `alignment_review_required`;
no patient functional localization, clinical deficit probability, or independent
anatomical alignment accuracy is available. Sampling coverage means atlas field
of view, and a zero map value does not establish absent patient function. The
layer labels and precise scalar interpretations are documented in
[`prior-registration.md`](prior-registration.md).

Validation: 31 focused prior-proposal tests and 111 tests across prior proposals,
core, imaging, structural evidence persistence and existing anatomy passed.
No registration optimizer or declared benchmark anatomy/evaluation source was
changed by this storage slice.
