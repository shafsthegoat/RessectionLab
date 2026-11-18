# Optional local HDD ledger

**Owner/access:** this is Shafrir's local HDD, not shared project storage.
Sayem Kamal (the commit coauthor) does not have the drive or access to its files.
Cloning this repository does not supply these assets. Use documented acquisition
instructions or an explicitly agreed transfer for collaborator workflows.

- Mount: `/Volumes/shafrir`
- RessectionLab root: `/Volumes/shafrir/old coding projects/RessectionLabData`
- Volume UUID: `9CC3623C-E38B-3EEE-B6BB-B9A06C193700`

Last location check: **November 18, 2026**. The drive may be disconnected.
Verify the real mount and `.ressectionlab-data-root.json` before writing; never
create a substitute mount directory. Ignored `testcases/` links may be dangling
while it is absent. Ordinary development must not require this private drive;
report missing inputs explicitly for optional data/reproduction workflows.

Native studies prepared from the November 17 frozen inventory live under
`prepared/adaptive-anatomy-migration-20261117-v1/<case>/`. Its
`migration-receipt.json` records progress and source-bound results.
Each native study can open directly in Cases. External library entries
require separate admission. `catalog/native-release.json` lists the reviewed
TopCoW-001 CTA workspace and binds its allowed uses to the checksummed decision
under `qc/native-study-reviews/`. [Run and resume the migration](data-rebuild.md#native-study-migration).

The private corrected EDEN CTRL_0866 study and its review receipt are in
`prepared/adaptive-anatomy-qa-20261118-v1/`. Open
`EDEN2020-CTRL_0866-tof-refused-v1.rsworkspace` for independent TOF inspection
and the refreshed T1-based comparison.
Its migration row links the acquired source snapshot to this reviewed version.

Paths below are relative to the RessectionLab root. This compact ledger covers
that area, not unrelated personal files elsewhere on the HDD. Presence does not
establish scientific readiness or a completed build; consult each run's receipts.

| Area | Locations and contents |
| --- | --- |
| Acquired originals | `sources/`: RESECT, UCSF-PDGM, EDEN2020 healthy record 3994749, IXI, IXI vascular record 17393202, Lausanne, meningioma and TopBrain collections, with acquisition provenance. |
| EDEN projects/work | `prepared/eden2020-synthetic-v1/` and `prepared/eden2020-anatomy-v2/`: original synthetic cohort and completed segmented/organic-tumor projects. `prepared/eden2020-planning-v1/collection.json`: native patient-anatomy studies for CTRL_1277 (primary) and CTRL_0866, retaining native T1/TOF/FLAIR/SWI grids, cached estimates, seven supplied named tract bundles each and separate existing synthetic targets; per-study `source-map.json` records exact reuse and limits. Permanent requested deliverables remain distinct from temporary run output. |
| EDEN approach comparison | `prepared/eden2020-approach-comparison-goal-v1/collection.json` binds CTRL_1277, CTRL_0866, and CTRL_0933 with variants 01 and 08. The ignored `testcases/approach-comparison-goal-v1/eden-studies` link and per-study source maps locate the packages. Saved evidence workspaces can restage imaging from this collection. Checksum-verified local comparison metadata is backed up at `archive/closed-evidence/eden-approach-comparison-goal-v1-sidecars/`. |
| Other preparation | `prepared/`: `header-precision/`, `placement-support/`, `recipes/`, `synthetic-fields/`, `topbrain/`. |
| Case library | `cases/real/`, `cases/synthetic/`, `cases/controls/`: native projects; use their manifests for roles and supported capabilities. |
| Model assets | `cache/models/`: `synthstrip-v1/`, `poc-vesselfm-base/`, `poc-totalsegmentator/`, `supersynth-source/`; availability, terms and installation state are in local receipts. |
| Historical archives | `archive/closed-evidence/`: October 19 rehearsal tar, generated-control tar, October 20 rehearsal, desktop visual evidence and profiler-baseline archives. [Exact newer archive identities/restoration](../manifests/historical-archives-20261116.json). |
| Archived desktop builds | `archive/desktop-builds-20261117/`: older macOS app bundles, packaging inputs and staging directories. `README.md` provides restore commands beside the per-archive checksums. |
| Archived sidecar and test runs | `archive/closed-evidence/local-build-and-test-archives-20261117/`: prior orientation-guide sidecar and 16 older vascular test runs. `README.md` links the verified archives and the existing evidence archives used to retire the local restore-test copies. |
| November 18 operative rehearsal examples | Open `archive/closed-evidence/operative-rehearsal-20261118/README.md` for the BITE-09, UCSF-0005, EDEN CTRL_1277 and synthetic workspaces, recordings and comparison images. Its `SHA256SUMS` covers the archived media, earlier diagnostic captures and test logs. The guide lists the working-study restore paths and identifies the five fixtures kept locally for verification. |
| November 18 scan-to-engine demonstration | `qc/scan-to-engine-20261118/README.md` links the UCSF-0004 portable geometric study, 53-action engine recording, cold saves, BITE-09 preview, paired CT registration refusal and generated controls. The ignored local `testcases/outputs/scan-to-engine-20261118` link resolves here. |
| Four-channel glioma preparation | `cache/models/poc-monai-brats-0.5.4/` contains the pinned 18,840,620-byte weight and release configuration. `qc/brats-native-0004-20261118/` holds the portable UCSF-0004 multimodal estimate, source/model identities and verification receipt. The application requires the matching local MPS runtime. |
| Integrated research | `integrated-v1/`: cases, anatomy hosts, replay experiments, QC, reports and reproduction records; start at its `START-HERE.md`. |
| Catalog and review | `catalog/START-HERE.md`, `catalog/releases/`, `catalog/inventory/`, `catalog/work-ledger.json`; `qc/`, `quarantine/`, `staging/` hold review/recovery/work areas. Catalog releases and ledgers have their own dates and scope. |

When adding, moving, archiving or retiring relevant HDD assets, update this ledger
and the affected local catalog pointers. Keep detailed file inventories, hashes
and run logs on the HDD; retain only concise discovery metadata here. Check
existing collections before downloading again. See [repository upkeep](../CONTRIBUTING.md).
