# Portable PAT16/PAT20 structural proposals

The completed frozen extraction outputs are attached as separate, unreviewed
structural proposals in these local bundles:

- `outputs/cases/BTC-sub-PAT16-structural-evidence.ressectionlab`
- `outputs/cases/BTC-sub-PAT20-structural-evidence.ressectionlab`

Both use the first declared repetition, `r1`, consistently. There was no selection
by overlap, appearance or score. Each contains separate main and no-CSF masks,
both `provenance=estimated`, `review_status=review_required`, with no accepted
review. The working brain mask remains absent and cortical access remains
disabled. The original source bundles are byte-for-byte unchanged.

| Patient | Main omitted annotation voxels | No-CSF omitted annotation voxels |
| --- | ---: | ---: |
| PAT16 | 19 | 2,045 |
| PAT20 | 125 | 413 |

The source and current-target omission flags are retained. These counts describe
overlap with a thresholded source annotation, not anatomical accuracy or removed
tissue. No estimated mask was corrected to improve that overlap.

The established `import_brain_extraction_evidence`, `save_case` and `load_case`
pipeline required no source changes. Both saved cases reopened at their new full
semantic identities, while their planning hashes, MRI, physical frame, active
and original annotations, source references and existing metadata remained
unchanged. Revision increased from 2 to 4 because two proposals were attached.
Twenty-eight existing structural-evidence and persistence tests passed.

[`brain-extraction-pat16-pat20-bundle-integration.json`](brain-extraction-pat16-pat20-bundle-integration.json)
records original and derived bundle hashes, full case identities, model/source/run
provenance, proposal hashes and omission flags. The subsequent [independent audit](brain-extraction-pat16-pat20-portable-independent-review.md)
passed source equality, persistence and review/access gates for both cases. The
integration receipt retains its historical audit-pending state; the later audit
records completion without rewriting that earlier receipt.

## Portable provenance and limits

Each case embeds the MRI, source and active annotations, and the two proposal-mask
arrays. Its `structural_proposal_integration` artifact also retains the exact
extraction-report UTF-8 text with its SHA256, external mask and predicted-distance
file hashes, and references to the independent extraction-QC records. The report
text hash agrees with both proposals' `run_sha256` identities.

Original NIfTI files and predicted-distance arrays are not embedded. Distance
outputs remain local diagnostic artifacts at the frozen extraction paths; their
hashes and provenance are retained. They are model predictions, including the
upstream exterior fill, and do not measure surgical clearance.

Historical interpreter and installed-package binary hashes were **not
predeclared** for extraction. Matching recorded version strings, source hashes,
model hashes and output hashes does not attest execution-time runtime binary
identity. This limitation is carried inside both bundles as well as their
integration receipt. Retained timestamps and logs also cannot establish the
absence of unrecorded runs.

Packaging these artifacts performed no inference or downloads and grants no
anatomical, functional, cortical-access or clinical approval.
