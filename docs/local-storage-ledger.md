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
| Continuous operative rehearsal | `qc/continuous-rehearsal-20261118/README.md` indexes two observed engine films, the BITE-09 geometric walkthrough, matched raw/conditioned fields, the UCSF intact decision, source review, canonical generated success and the repeated frozen-policy comparison. The local ignored link is `testcases/outputs/continuous-rehearsal-20261118`. Use the root checksums and sample-restore receipt before copying. |
| Four-channel glioma preparation | `cache/models/poc-monai-brats-0.5.4/` contains the pinned 18,840,620-byte weight and release configuration. `qc/brats-native-0004-20261118/` holds the portable UCSF-0004 multimodal estimate, source/model identities and verification receipt. The application requires the matching local MPS runtime. |
| Integrated research | `integrated-v1/`: cases, anatomy hosts, replay experiments, QC, reports and reproduction records; start at its `START-HERE.md`. |
| Catalog and review | `catalog/START-HERE.md`, `catalog/releases/`, `catalog/inventory/`, `catalog/work-ledger.json`; `qc/`, `quarantine/`, `staging/` hold review/recovery/work areas. Catalog releases and ledgers have their own dates and scope. |
| Multimodal library baseline | `catalog/inventory/multimodal-anatomy-baseline-20261118-v1/README.md` indexes the pre-acquisition files, receipts, role restrictions and source families. |
| New CTA source studies | `sources/tair-v1-cta-selective/` and `sources/head-cta-15697196/selective/` hold original scans, checksums and the publisher STL. `prepared/tair-v1-cta-native/` and `prepared/head-cta-15697196-native/` hold the development studies. Matching `qc/` directories contain source and display checks. |
| Head CTA 013 aneurysm frame review | `qc/head-cta-15697196-native/013/stl-frame-v1/` keeps the diagnostic overlay and measured axis hypotheses. The only contrast-supported placement conflicts with publisher laterality, so the original STL remains outside the native study layers. |
| ISLES'24 v7 acute CT pair | `sources/isles24-v7/acute-pair-selective/sub-stroke0146/` holds acquired NCCT and CTA, the publisher-registered CTA and automated CoW mask. `prepared/isles24-v7-native/sub-stroke0146/` holds the study. `qc/isles24-v7-native/sub-stroke0146/` links source and paired-display evidence. |
| New CTA showcases and gliomas | `prepared/isles0146-cta-gliomas-v1/` and `prepared/tair00273-cta-gliomas-v1/` index the source grids, contours, separate truths and overlays. `prepared/tair00273-cta-showcase-v1/` contains the restricted enhancement segment and exact wider-grid evidence. Under `qc/hidden-app/`, the `*-native-surface-pair-v1/` and `*-cta-gliomas-*/` directories contain the matched views and portable desktop checks. `qc/isles0146-cta-guarded-refusal-v1.json` binds the attempted rehearsal to the sealed contour. Numeric trials are under `qc/isles0146-{brain,skull}-conditioning-v1/` and `qc/tair00273-skull-{conditioning,categorical-guard}-v1/`. |
| CTA anatomy and glioma scenarios | `prepared/cta-organic-gliomas-v1/collection.json` indexes 12 development projects, overlays, the contact sheet and separate truth labels. The 46-person TRAIN summary is in `prepared/cta-organic-glioma-guidance-v1/`. The shape audit and Cases preview evidence are in `qc/cta-organic-cohort-shape-audit-v1/` and `qc/hidden-app/cta-organic-cohort-20261118-v1/`. TopBrain reuse and CoW geometry frame checks are in `qc/multimodal-anatomy-20261118-vessel-v1/` and `qc/multimodal-anatomy-20261118-cow-vtp-v1/`. |
| Native CTA showcases | `prepared/cta-native-showcases-v1/` contains the three locally registered studies and their verified mesh caches. Hidden source/conditioned captures are in `qc/hidden-app/cta-native-showcase-*-surfaces-20261118-v*/`. The CTA014 organic route screen is in `qc/cta-organic-rehearsal-preflight-v1/receipt.json`. |
| CTA014 route-conditioned organic case | `prepared/cta014-organic-fixed-route-v1/` holds the pristine workspace, separate truth labels and static screen. `qc/cta014-organic-fixed-route-rehearsal-v1/` holds the fresh opening, first target effect and withdrawal. `qc/hidden-app/cta014-organic-conditioned-replay-20261118-v1/` holds opened-source, drilled opening, aspiration, and withdrawal captures. |
| Publisher CoW CTA/MRA companions | `prepared/cta-published-cow-v1/` contains the 006, 012 and 014 direct publisher CTA meshes, original labels and graphs, paired independent MRAs and verified superior skull and brain display geometry. The selective 012/014 receipts and exact publisher assets are in `sources/topcow-cow-vtp-selective-v1/`. Hidden verification directories start with `qc/hidden-app/cta<id>-published-cow-`. The 006 `refined-warm-v1` and `refined-cold-v1` checks bind the conditioned surfaces and native MRA to the portable study. The CTA014 multi-lane screen is in `qc/cta014-organic-01-multilane-v1/receipt.json`. |
| IXI265 structural MRI and diffusion | `prepared/adaptive-anatomy-migration-20261117-v1/IXI265-multimodal/` retains the T1, MRA, diffusion b0, diagnostic FA and cached T1 tissue estimates. The checkout Cases alias points to this existing study. Hidden app verification and two captures are in `qc/hidden-app/ixi265-multimodal-20261118-v3/`. |
| Meningioma sub-06 CT and MRI | `prepared/adaptive-anatomy-migration-20261117-v1/Meningioma-sub-06/` retains the two CT kernels, T1CE and cached tissue estimates. `qc/meningioma-sub06-skull-brain-conflict-v1.json` localizes dense CT signal against the MRI-derived masks and leaves the existing transform review status unchanged. |
| CADS BrainCT-1mm case 465 | `sources/cads-brainct-1mm/465/` retains the pinned CT and three model-label originals, publisher index, label map and acquisition receipt. `prepared/cads-brainct-1mm-v1/CADS-BRAINCT-465-NATIVE-v1/` contains the native study, stock revision pointer and source-plane review. `qc/cads465-vault-conditioning-v1/` holds the numeric evidence. The portable study, paired captures and restore reports are in `qc/hidden-app/cads465-conditioned-vault-{warm-v2,cold-v1}/`. |
| IXI077 organic route-conditioned scenario | `prepared/ixi077-organic-rehearsal-v1/IXI-077-T1-MRA-ORGANIC-ROUTE-01-v1/` holds the pristine study, separate exact labels and static screen. `qc/ixi077-organic-conditional-replay-v1/` holds first contour effect and withdrawal with cold backend verification. `qc/hidden-app/ixi077-organic-replay-20261118-v2/` holds the desktop playback proof and captures. |
| Fixed-target strategies and patient reconstruction comparison | `qc/patient-strategy-comparison-20261119/README.md` indexes the two UCSF-PDGM-0004 recordings, joined study and result-inspection film, standardized anatomy captures, saved sub-06 CT/MRI study and RESECT Case3 model/annotation comparison. The directory includes source-bound contract checks and a sample restore. |
| IXI077 integrated engine replay | `qc/engine-operative-integration-20261118/README.md` indexes the source-bound T1/MRA plan, procedural saves, observed films, independent state checks and matched BITE-09 and meningioma rendering comparisons. Use the checksums and endpoint sample-restore receipt to verify a copy. |
| Cranial rendering and operative rehearsal | `qc/cranial-reconstruction-20261118/` contains the MNI BITE-09 saved MRI reconstruction, matched planning and operative captures, and the full geometric rehearsal. It also retains separate CTA006 publisher-mesh and independent MRA checks. |
| Observed surgical change | `sources/RESECT-SEG/v1-20240215/Case3/` retains the publisher cavity annotation and rights notice. `qc/surgical-change/RESECT-Case3-benchmark-verified-20261118/README.md` links the portable workspace, overlay and landmark results. |
| RESECT Case3 observed stage pair | `sources/RESECT-SEG/v1-20240215/Case3/` now includes during and after cavity labels. The paired landmarks are in `sources/RESECT/v1/Case3/Landmarks/`. `qc/surgical-change/RESECT-Case3-observed-stages-20261118-v1/` contains the workspace, cavity overlay, held-out predictions and report. See `qc/hidden-app/resect-case3-native-stage-pair-20261118-v1/` for separate source and display proof. |
| RESECT Case3 observed tumor and cavity | `qc/surgical-change/RESECT-Case3-observed-target-cavity-20261119-v1/` binds the two acquired tumor annotations, after-US cavity, source landmarks, two native workspaces, overlay figure and held-out comparisons. `historical-localization/` retains the exact earlier FLAIR model output and its preoperative score. Hidden display evidence is in `qc/hidden-app/resect-case3-target-cavity-native-20261119-v1/` and `qc/hidden-app/resect-case3-flair-target-native-20261119-v1/`. |

When adding, moving, archiving or retiring relevant HDD assets, update this ledger
and the affected local catalog pointers. Keep detailed file inventories, hashes
and run logs on the HDD; retain only concise discovery metadata here. Check
existing collections before downloading again. See [repository upkeep](../CONTRIBUTING.md).
