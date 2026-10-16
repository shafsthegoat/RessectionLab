# Published tumor sites are searchable in Cases

The actual Electron case browser now searches the reported tumor location and
side for six existing BTC TRAIN cases. The [additive evidence](evidence.json),
SHA256 `f5d9db8e86f7ecc8343b5c302715f93d65727246dfd94a735d5db94d56c64362`,
binds the frozen twelve-source catalog, exact case/bundle/source-file identities,
dataset participants rows and the [primary paper's Table 1](https://www.eneuro.org/content/5/3/ENEURO.0083-18.2018/tab-figures-data).
The paper hash identifies its Europe PMC archival XML, not publisher HTML.
Only participant ID, location and lateralization facts are copied.

| Existing participant | Reported site |
| --- | --- |
| PAT05 / PAT28 | Left frontal |
| PAT16 | Right fronto-temporal |
| PAT20 / PAT22 | Right parietal |
| PAT25 | Right temporal |

Each label says **Dataset / paper report · review pending**. These are
patient-level source reports, not reviewed native-mask localization, functional
boundaries or automatic seed coordinates. Canonical region/lobe/hemisphere
fields remain UNKNOWN. No location is inferred for UCSF004, other patients or
saved versions with similar names. Catalog IDs, twelve source identities,
qualification receipts, source arrays and protected roles remain unchanged.
This adds zero people, model/solver runs or anatomical/training acceptance.

The [source verification](verification.json) passes one positive and twelve
negative metadata controls and checks all six primary-table rows against the
fixed archival XML. It preserves 89 historical files and all twelve current
source receipts without opening image or bundle payloads.

The [native interaction record](ui-demonstration.json), SHA256
`4cbe51b3038c579df8ca4db4901bfa23960240c390e5438d6cbc2622bcbda023`,
hashes six screenshot/accessibility pairs and the exact app source/build.
Actual searches find two parietal cases, two left-frontal cases and four
right-sided cases; “IXI frontal” finds none. Opening the right-parietal result
loads BTC20's acquired T1w and its threshold-derived source target annotation,
with linked slices and source-preserved status. The app exits zero; no case is
saved or modified during this feature check. The library's 24 saved versions
remain twelve qualified source identities, not 24 people.

Twenty-one focused host tests, five renderer tests, TypeScript and production
build pass. Controls cover exact case binding, unchanged IDs/anatomy, unrelated
saved variants, missing optional evidence, absent-catalog fallback and corrupt
or redirected evidence. Present invalid metadata is refused before a bundle
open. Missing evidence preserves ordinary browsing. The existing large-renderer
chunk warning remains; this is not a full historical Python-suite result.

The [independent metadata/primary-source audit](../btc-six-source-location-independent-v1/independent-qa.json),
SHA256 `12805f8e224faf0c668b4ed4d72da6b13bf2632d32733e657261a009be6c6959`,
passes 30 independent tampering controls, 15 reader rejection controls and nine
metadata-only integration checks. It independently extracts the six primary
table rows by column names and verifies the TSV, acquisition bindings and all
twelve receipt records. No source image/bundle payload is opened by this audit.
Reviewed anatomical diversity and segmentation accuracy remain open; published
site coverage alone cannot close them.
