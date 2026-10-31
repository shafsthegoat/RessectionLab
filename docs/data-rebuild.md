# RessectionLab data collection

The collection supports source inspection and explicitly scoped surgical-research
controls. Acquired observations, supplied annotations, algorithmic estimates and
synthetic components remain separate. Synthetic variants add no real people.
Independent review is AI engineering/source-data review, not clinical or
neuroradiologist validation. No RL training, medical-data upload or remote
publication belongs to this cycle.

## Open accepted cases

Set `RESSECTIONLAB_DATA_ROOT` to your actual mounted data directory, then launch
from the existing primary checkout:

```sh
export RESSECTIONLAB_DATA_ROOT="/absolute/path/on/mounted/RessectionLabData"
./scripts/launch_poc.sh
```

Choose the external Case Library and open a reviewed case. The portable app has
no committed machine-specific bulk-data default. It checks the mounted volume's
identity through fresh native Disk Arbitration properties and the live filesystem
UUID on every call, without cached leases, and reports an unavailable drive
rather than treating missing anatomy
as a valid empty case. Preparation reserves the larger of 50 GiB and 10% of the
volume. Twelve recorded native guard probes had a 6.6 ms median
(`qc/native-volume-guard-probe.json`); this replaces diskutil probing that stalled
under ExFAT writes. Source archives, extraction/download partials, model caches, derivatives,
native projects, review images and archived evidence stay on the data drive.

`catalog/release.json` is the current accepted index. Each entry binds an exact
package SHA256 to a checksummed `qc/case-decisions/` sidecar containing allowed
capabilities and limitations. `staging/candidates/` and `quarantine/` are not
accepted catalogs. Native `metadata.data_case` preparation status is historical;
it cannot override the linked independent decision. Published packages use the
existing self-contained `.ressectionlab` format and production save/reopen reader.

Read `catalog/reporting/report-preview.md` while work continues; use
`report-final.md` only after the lead has published the final snapshot. The JSON
companions preserve exact counts, source-family aliases, component provenance,
resource evidence and current pending work. Synthetic controls, healthy acquired
hosts, clinical CTA/MRA patients and observed surgical-change cases have distinct
classifications. Clinical vascular patients are not healthy controls. Source
person-family counts do not prove independence across unrelated cohorts or
pretrained-model exposure.

## Resume preparation and review

Run serially; do not overlap memory-heavy registrations, inference or case builds.
Commands below reuse verified source bytes and completed receipts:

```sh
# Current healthy-host plan, one unfinished host at a time.
.venv/bin/python scripts/data_rebuild_batch.py --limit 1

# Fifty prospectively frozen UCSF TRAIN sources; independent pilot gate required.
.venv/bin/python scripts/data_rebuild_real.py batch --count 50 \
  --review-receipt "$RESSECTIONLAB_DATA_ROOT/qc/independent/UCSF-PDGM-0009-structural-scale-review-v1.json"

# Selected paired clinical CTA/MRA people, preserving frozen roles and pilot gate.
.venv/bin/python scripts/data_rebuild_topbrain.py batch --count 25 \
  --review-receipt "$RESSECTIONLAB_DATA_ROOT/qc/independent/TopBrain001-native-scale-review-v1.json"

# One already-exposed RESECT TRAIN family: MRI and before/during/after US.
.venv/bin/python scripts/data_rebuild_surgical_change.py

# Automated engineering checks and representative hidden app save/reopen checks.
.venv/bin/python scripts/data_rebuild_qc.py
.venv/bin/python scripts/data_rebuild_verify_suite.py --expansion

# Read-only preview; no source-array decoding or bulk rehashing.
.venv/bin/python scripts/data_rebuild_report.py --stage preview
```

The one-cycle inventory, real-source, surgical-change, reporting and retirement
helpers additionally retain this run's verified volume UUID and canonical-root
bindings. They deliberately refuse an absent/replacement volume; moving the
corpus requires reviewing those bindings and restoration manifests, not changing
portable app defaults or inventing an internal fallback directory.

The resumable ledger is `catalog/work-ledger.json`. Source rights/access receipts
are in `catalog/source-research.json`; cohort assignments remain in repository
manifests. All modalities, timepoints, annotations and synthetic descendants
inherit their person's frozen TRAIN/SELECT/EVAL role and exposure history.
UPENN remains reserved external evaluation. Known TopBrain aliases identify
legacy clinical person 001 as existing DEVELOPMENT: 25 clinical families therefore
represent 24 newly materialized families and 1 previously exposed recovery.
RESECT Case3 is another already-exposed TRAIN recovery, not a never-seen person.

Original healthy `plan-v5.json` remains frozen. Its explicit
`plan-v5-resource-extension.json` preserves the original plan hash/roles and failed
IXI051 source while predeclaring TRAIN replacement 084 for the generation slot.
IXI051 hit the 6 GiB MPS allocator cap; its CPU model subprocess later received
SIGKILL 9 after about 289 seconds with `no_saved_masks`. An 8 GiB supervisor-limit cause
was not established. Lausanne197's 50 million voxel deformation was withheld before
allocation because the conservative additional-array estimate exceeded the 4 GiB
budget. The qualified mass-field scope is 016/021/022/091; never infer five completed
fields from five original requests. Final usable counts come from publication.

Automated QC is not independent acceptance. Inspect actual source planes,
transforms, masks, provenance and app behavior; failed cases must be repaired,
restricted, quarantined or rejected. `qc/independent/release-decisions-v1.json`
records the independent decisions. Local catalog publication uses:

```sh
.venv/bin/python scripts/data_rebuild_release.py
```

Only the lead freezes a reviewed release. Reporting uses checksum-linked case
decisions, never native preparation flags or a directory's package count.

## Interpret the data correctly

Source MRI can be skull-stripped, cropped or registered upstream. A displayed
brain/skull/vessel mask may be an estimate; missing function, veins or outside-FOV
anatomy is unknown rather than safe. No verified accepted person combines acquired
tumor MRI, useful skull CT and dedicated angiography in a compatible surgical
state: the exact-match count remains zero. Synthetic appearance/deformation is a
controlled engineering model, not biological glioma realism or outcome evidence.

TopBrain's source terms are noncommercial; retain its original attribution and
restrictions. IXI share-alike and other source-specific terms are recorded in the
source registry and case sidecars. Do not bundle restricted source data into a
commercial deliverable or infer image rights from a software/paper license.

Large MRA and acquired US sources default to native MPR with 3D raycasting disabled
where required by the observed Metal failure. This display policy preserves the
original NPY payloads. Operative US stages have explicit captions. Lausanne's typed
adapter distinguishes NIfTI observations from JSON metadata; oversized auxiliary
views use an explicitly labelled 1 mm display derivative while full native primary
and originals remain available. It is display LOD, not anatomical registration.

Historical internal placement proxies with unknown spatial units retain their
original bytes. Header-only `v2-mm` companions declare mm without changing masks,
affines or native anatomy. Their independent byte proofs are in
`qc/independent/placement-proxy-unit-repairs-independent-v1.json`; current case
sidecars link the importable companions. Never silently reinterpret a v1 helper
as an acquired image or a validated tissue segmentation.

## Inventory and reversible restoration

The initial local `testcases/` plus `data/` inventory recorded 82,899 entries,
41,147,174,770 logical regular bytes and 27,983,089,664 lstat allocated bytes.
78,129 files were checksummed; 70 protected paths were skipped. Existing shared
symlinks, provenance and unique malformed/failure controls remain recorded under
`catalog/inventory/`. AppleDouble `._*` helpers are metadata, not images or cases.

Only the closed `testcases/performance/rehearsal-2026-10-19` campaign was retired.
Its complete PAX tar was verified, a real native workspace reopened directly from
it, and all remaining native/JSON/code/index/symlink references and live open files
were checked before removal. 1,167,580,626 logical bytes and 1,170,530,304 lstat allocated
bytes were removed; immediate system disk-free increased 1,171,329,024 bytes. APFS
sharing, snapshots and concurrent work limit interpretation of physical savings.
Active patient sources, cases, models and history were preserved. No Git history
or object shrinking is claimed.

The exact archive SHA, source hashes, scope and restore command are in
`catalog/inventory/migration/rehearsal-2026-10-19-migration-receipt.json`. The local
pointer is `testcases/performance/ARCHIVED-EVIDENCE.md`. Restore to an internal
filesystem while the original campaign directory is absent:

```sh
.venv/bin/python scripts/data_rebuild_migrate.py restore \
  --plan "$RESSECTIONLAB_DATA_ROOT/catalog/inventory/migration/rehearsal-2026-10-19-plan.json" \
  --destination "$PWD/testcases/performance"
```

Restoration refuses overwrites and verifies the plan/archive SHA, file bytes and
link targets. If a new benchmark recreated that output directory, restore into a
different ignored internal parent. Do not extract symlink archives onto ExFAT.
Disposable generated fixtures verified restoration, relative links, native reading
and corruption/UUID/scope guards; no other broad cleanup is implied.
