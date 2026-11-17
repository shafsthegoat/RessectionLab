# Optional local HDD ledger

**Owner/access:** this is Shafrir's local HDD, not shared project storage.
Sayem Kamal (the commit coauthor) does not have the drive or access to its files.
Cloning this repository does not supply these assets. Use documented acquisition
instructions or an explicitly agreed transfer for collaborator workflows.

- Mount: `/Volumes/shafrir`
- RessectionLab root: `/Volumes/shafrir/old coding projects/RessectionLabData`
- Volume UUID: `9CC3623C-E38B-3EEE-B6BB-B9A06C193700`

Last location check: **November 17, 2026**. The drive may be disconnected.
Verify the real mount and `.ressectionlab-data-root.json` before writing; never
create a substitute mount directory. Ignored `testcases/` links may be dangling
while it is absent. Ordinary development must not require this private drive;
report missing inputs explicitly for optional data/reproduction workflows.

Paths below are relative to the RessectionLab root. This compact ledger covers
that area, not unrelated personal files elsewhere on the HDD. Presence does not
establish scientific readiness or a completed build; consult each run's receipts.

| Area | Locations and contents |
| --- | --- |
| Acquired originals | `sources/`: RESECT, UCSF-PDGM, EDEN2020 healthy record 3994749, IXI, IXI vascular record 17393202, Lausanne, meningioma and TopBrain collections, with acquisition provenance. |
| EDEN projects/work | `prepared/eden2020-synthetic-v1/` and `prepared/eden2020-anatomy-v2/`: original synthetic cohort and ongoing segmented/organic-tumor preparation, verification and prior attempts. Permanent requested deliverables remain distinct from temporary run output. |
| Other preparation | `prepared/`: `header-precision/`, `placement-support/`, `recipes/`, `synthetic-fields/`, `topbrain/`. |
| Case library | `cases/real/`, `cases/synthetic/`, `cases/controls/`: native projects; use their manifests for roles and supported capabilities. |
| Model assets | `cache/models/`: `synthstrip-v1/`, `poc-vesselfm-base/`, `poc-totalsegmentator/`, `supersynth-source/`; availability, terms and installation state are in local receipts. |
| Historical archives | `archive/closed-evidence/`: October 19 rehearsal tar, generated-control tar, October 20 rehearsal, desktop visual evidence and profiler-baseline archives. [Exact newer archive identities/restoration](../manifests/historical-archives-20261116.json). |
| Integrated research | `integrated-v1/`: cases, anatomy hosts, replay experiments, QC, reports and reproduction records; start at its `START-HERE.md`. |
| Catalog and review | `catalog/START-HERE.md`, `catalog/releases/`, `catalog/inventory/`, `catalog/work-ledger.json`; `qc/`, `quarantine/`, `staging/` hold review/recovery/work areas. Catalog releases and ledgers have their own dates and scope. |

When adding, moving, archiving or retiring relevant HDD assets, update this ledger
and the affected local catalog pointers. Keep detailed file inventories, hashes
and run logs on the HDD; retain only concise discovery metadata here. Check
existing collections before downloading again. See [repository upkeep](../CONTRIBUTING.md).
