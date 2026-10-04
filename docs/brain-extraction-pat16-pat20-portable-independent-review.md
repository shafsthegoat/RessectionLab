# Independent portable-proposal audit

Both PAT16 and PAT20 proposal-enriched bundles passed independent engineering checks on October 4, 2026. This does not approve anatomy, cortical access, or clinical accuracy. The audit performed no inference, downloads, training, or source modifications.

The original source bundles, all seven acquisition files per patient, source MRI samples, affine, active and original threshold-derived annotations, metadata, and planning hashes are preserved. The fractional source annotation is still an external immutable file; the declared binary threshold remains 0.5. Both cases have no working brain mask or patient context with unknown availability. Full-head nonzero MRI support remains disallowed.

The first declared extraction repetition supplies both proposals consistently. All four embedded mask arrays match retained native-grid masks exactly. Their exact source, model, runner, report, configuration, runtime-version records, and omission flags agree with the retained extraction artifacts. PAT16 retains 19 main-envelope and 2,045 no-CSF-envelope omitted annotation voxels; PAT20 retains 125 and 413. No omitted tissue was filled or repaired.

Both proposals remain `estimated`, `review_required`, and separate from working anatomy. Direct attempts to use each proposal as working support raise `BRAIN_MASK_REVIEW_REQUIRED`. Semantic identity changes with the additions; planning identity remains unchanged. Original revision 2 becomes revision 4. Independent save/reopen reproduces every manifest field and proposal mask, including the original artifacts and the newly embedded exact report text.

Masks and the report are portable. Predicted signed-distance arrays remain external artifacts with verified lengths and SHA-256 hashes; they are neither embedded nor treated as measured surgical clearance. The earlier independent extraction overlays were visually inspected and are referenced by exact audit hashes; this audit establishes exact mask identity with those inspected outputs. Historical interpreter/package binaries were not predeclared, so runtime version agreement is not historical binary attestation.

Validation:

- `.venv/bin/python scripts/audit_pat16_pat20_portable_proposals.py`: both cases passed in 11.9207315 seconds; one numerical worker, no GPU; macOS peak resident memory 636,518,400 bytes. All 37 retained input files were hashed again after the audit and remained unchanged.
- `.venv/bin/python -m pytest tests/test_pat16_pat20_portable_audit.py -q`: 15 passed in 0.25 seconds. These small adversarial checks reject altered embedded reports, missing/duplicate source artifacts, swapped run paths or hashes, false distance-portability claims, invented binary attestation, and anatomy/clinical claims.

Source-bound records:

| Item | SHA-256 |
| --- | --- |
| `brain-extraction-pat16-pat20-portable-independent-qc.json` | `19f9ff9ef0a2783368a53642021210fb8195d358cde6693c8b49abfd70107ace` |
| `scripts/audit_pat16_pat20_portable_proposals.py` | `4b996b16760f368606ef9d3d7e516fb84095ad79ef76e58c49c606c65c2c1d57` |
| `tests/test_pat16_pat20_portable_audit.py` | `9028613bbdd5f2823f563c1f3c9a9022bf18cb06b2fcf5dfff4e5ada42267082` |
| Integration receipt | `006e223b8b6a2b689173c264584e007fdbce0e96da611b67fb74f4e841d88946` |

Independent roundtrip bundles are retained under ignored `outputs/portable-proposal-independent/`. This review does not modify the cohort registry or final-patient reservations.
