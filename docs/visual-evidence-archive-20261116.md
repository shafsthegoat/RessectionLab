# Historical desktop visual evidence

The November 16 cleanup preserves the complete original desktop visual evidence
collection in a verified external archive. Following independent review, 291
historical images and recordings (183,382,322 bytes) were removed from the active
checkout. The archive includes all 550 original
files (194,640,539 bytes), including their receipts, numerical reports, source
producers and original filename mappings. Exact duplicate images retain every
original path. Historical reports, manifests, source hashes and captured
accessibility text are unchanged.

The archive covers `artifacts/desktop-renderer/` and the original
`artifacts/electron-*` files and directories. It preserves renderer fidelity v1/v2,
source quality-control records, GPU prior-sampling probes, native source/frozen/
packaged checks, cancellation, annotation centering, case identity, MRI layout,
population priors, route guidance and refinement captures. These records retain
their original package and verification scope; they do not qualify newer builds.

All 253 nonmedia files remain at their original paths in the checkout. In
particular, `desktop/electron/verify-priors.cjs` still reads the original
`artifacts/desktop-renderer/ucsf-seven-prior-hydration.json`, and real training-case
preparation still reads `artifacts/electron-btc-inventory-validation.json`.
Ordinary application use and tests require no visual archive installation.

## Restore the complete collection

Archive directory on the external data volume:

```text
/Volumes/shafrir/old coding projects/RessectionLabData/archive/closed-evidence/desktop-visual-evidence-20261116-v1/
  archive.tar.gz
  manifest.json
  receipt.json
```

The manifest records original relative paths, sizes, SHA-256 hashes and metadata;
the receipt records archive verification. Keep these files together. From the
repository root, restore to a new directory:

```sh
python3 scripts/archive_historical_evidence.py restore \
  --archive '/Volumes/shafrir/old coding projects/RessectionLabData/archive/closed-evidence/desktop-visual-evidence-20261116-v1/archive.tar.gz' \
  --manifest '/Volumes/shafrir/old coding projects/RessectionLabData/archive/closed-evidence/desktop-visual-evidence-20261116-v1/manifest.json' \
  --destination '/absolute/new-root'
```

The restore command refuses an existing destination and verifies restored bytes.
Open reports under `/absolute/new-root/artifacts/`; original relative media
filenames then resolve beside their original receipts. Repository-local receipts
can refer to images that are now optional archive members. For example,
`medivis-fidelity-v2/media-manifest.json` retains its original eight media entries;
restore the collection before checking those entries. Do not rewrite historical
receipts to describe a partial local collection.

The archive is 174,994,429 bytes. Its SHA-256 is:

```text
5c1c26ec69db9ac4a58e95b1885d945027c58390e0130542693ead437fd24241
```

The exact `manifest.json` SHA-256 is:

```text
4d3d5438736772670422ca7dfa94bfeccf0a49edf47990e21ff816a1758f0c72
```

Restoration to
`testcases/cleanup-execution-20261116/restored-visual-v1/` passed verification of
all 550 files and 194,640,539 content bytes, including SHA-256, file modes and
nanosecond modification times. The archive also preserves 37 parent directory
records. Original tracked content remains independently recoverable from pushed
tag `preservation/cleanup-execution-20261116`, commit
`6547ad746787283f9f9159ddf3235d0820c9c382`.

The pushed pre-cleanup preservation reference and measured restoration results
are recorded in the [cleanup execution record](cleanup-execution-20261116.md).

## Representative images retained locally

These six original images provide compact examples from the newest presentation
and native UI checks in the archived collection (3,978,011 bytes total):

- [MRI segmentation presentation](../artifacts/desktop-renderer/medivis-fidelity-v2/mri-final/segmentation.png)
- [CTA default presentation](../artifacts/desktop-renderer/medivis-fidelity-v2/cta-final/cta-polished-default.png)
- [Selected routes, wide layout](../artifacts/electron-route-empty-state-v1/ucsf-selected-routes-wide.jpg)
- [Selected routes, minimum layout](../artifacts/electron-route-empty-state-v1/ucsf-selected-routes-minimum.jpg)
- [Population priors ready](../artifacts/electron-priors-v1/final-prior-ready.jpg)
- [Case identity in MRI review](../artifacts/electron-case-identity-v1/pat20-final-mri-review-wide.jpg)

## New captures

Shared Electron verification and the interaction benchmark create unique runs
under ignored `testcases/outputs/desktop-verification/`. `--report-dir` selects an
explicit root; `INTERACTION_OUTPUT` selects the benchmark output directory, and
`FIDELITY_OUTPUT` selects the visual fidelity capture directory. Existing explicit
output options remain available. Review a run and explicitly promote selected
receipts and images when durable repository evidence is needed.
