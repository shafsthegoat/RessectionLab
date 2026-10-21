# Patient corpus construction after the working rehearsal gate

Audit date: 2026-10-16. This plan follows the latest user-supplied
`RessectionLab_POC_First_Codex_Supergoal.md`: finish Gates 1–2 first. It does not
start acquisition, cohort expansion, PDE generation, PPO or a new training run.
Earlier real-only/synthetic-disabled and learning-first steering does not cancel
that newer authorization or reorder these gates. Patient identity, existing
roles, provenance and unsupported-anatomy guards remain mandatory.

## What is actually available

The reproducible [inventory](../scripts/patient_corpus_inventory.py) found **291
materialized source-family patient groups** in the five requested families.
This is deduplicated within each source family; biological independence across
unlinked cohorts is not proven. It is not 291 reconstructed or validated cases.
Protected SELECT payloads contribute only their existing acquisition-receipt
claims; this audit did not open or stat them. UPENN, UTSW, BTC 29/31 and other
measurement-evaluation imaging remain outside the reader allowlist.

| Source | Materialized people and preserved roles | Prepared or qualified evidence | Missing evidence / useful next capability |
|---|---|---|---|
| UCSF-PDGM | 1 development person, `004` / local `0004` | Four registered structural contrasts, source tumor compartments, SynthStrip brain estimate and real MONAI inference in a POC bundle | Source is the pinned mirror, not independently verified official-v5 byte equivalence. Known `BraTS2021_00097` training overlap prevents independent MONAI accuracy claims. Skull, angiography and patient function absent. |
| BTC | 8: TRAIN 05/16/20/22/25/28; SELECT 26/27 | Eight saved structural-source bundles and estimated brain-envelope derivatives exist; six TRAIN bundle manifests inspected here. All six TRAIN raw anatomy/diffusion/resting-state paths were acquired earlier. | Source fractional lesion annotations are a declared 0.5-threshold research target, not categorical glioma compartments. Brain estimates still need intended-use review; source/brain support conflicts must survive. No acquired CT/angiography. Diffusion acquisition is not accepted tract reconstruction; resting-state BOLD is not task function. |
| ReMIND | 82: 79 TRAIN, 001 DEVELOPMENT, SELECT 013/037 public-series intake | TRAIN receipts bind 59,520 MR/US objects plus 242 SEG objects; SELECT receipts bind 372 objects. 001 has native converted MRI/SEG evidence. Existing 045/050 and SELECT 013 planning experiments are retained separately. | Not all 82 are gliomas or converted planning cases. The 79-person byte-intake receipt is historical, not a blanket current QC claim. Source automatic Brainlab cerebrum/ventricular labels are estimates; per-series SOP/frame/phase linkage and tissue-domain coverage need case-specific admission. No new `.ressectionlab` POC bundle was found for ReMIND in the inspected case roots. |
| SynthRAD2023 | 1 materialized pair, TRAIN `1BA336` | Acquired CT and MRI extracted; matching coded grids; TotalSegmentator CT brain/skull estimates saved in a POC bundle | One opaque verified Task1 archive also contains 180 declared brain people (126 TRAIN /27 SELECT /27 EVAL), but the other 179 are not extracted or reconstructed cases. Data are defaced/cropped; skull coverage, HU interpretation and anatomical CT/MR registration need review. Source outline mask is not bone, brain or tumor truth. Tumor status and angiography absent. |
| Lausanne | 199 TRAIN people /210 T1–TOF sessions | 840 original files; 414/420 image-level QC passes; 204 passing pairs from 193 people. 144 acquired masks from 106 people/117 sessions; 136/144 mask-grid passes. Sub000 has a real vesselFM estimate in the POC. | Six T1 conflicts, eight mask-grid conflicts and four metadata exclusions remain. A passing header is not scanner-world registration. Sub000 T1/TOF alignment is unresolved; TOF is regional and includes extracranial signal. Existing manual-region masks have limited/subtype-dependent semantics, not comprehensive vessel truth. |

Counts are supported by the frozen role manifests and acquisition/QC receipts:
[UCSF source](../manifests/data_acquisition_ucsf.json),
[BTC roles](../manifests/experiments/btc-spatial-development-cohort-v1.json),
[BTC raw coverage](../artifacts/btc-train-modalities-intake-v1/RESULT.md),
[ReMIND roles](../manifests/experiments/remind-component-cohort-v1.json),
[ReMIND001 conversion](remind-001-development-acquisition.md),
[SynthRAD roles](../manifests/synthrad-component-cohort-v1.json),
[SynthRAD restricted-pair review](../artifacts/synthrad-first-pair-qc-execution-v1/RESULT.md),
[Lausanne completed QC](../artifacts/lausanne-deferred-qc-execution-v1/RESULT.md), and
[Lausanne annotation semantics](../artifacts/lausanne-annotation-full-intake-v1/RESULT.md).
The compact machine-readable [snapshot](patient-corpus-inventory-2026-10-16.json)
records fresh metadata hashes and measured geometry; no patient arrays are added to Git.

The inspected development/TRAIN bundle roots represent **9 people**: UCSF 1,
BTC 6, SynthRAD 1 and Lausanne 1. Across those saved bundles, **7 have brain and
tumor layers, 1 brain and skull, 0 registered brain and vessels, and 0 complete
same-person brain+tumor+skull+vessel cases**. These are layer-presence counts,
including estimated/unreviewed support, not accepted anatomy counts. Only
**three** are the current dedicated POC source patients. The protected BTC
SELECT bundles exist but were not opened or included in those nine.

There are **0 source-bound virtual-glioma bundles in the inspected named case
roots**. Unknown-named workspace copies are reported, not opened or guessed.
This is not a global assertion that the repository contains no synthetic data:
the retained [analytic population experiment](../artifacts/learning/population-synthetic-v1/comparison/population-provenance.json)
reports two synthetic training families and zero independent patients.
Procedural worlds, crops, branches, sessions, repeated visits and replayed
operations must never increase the patient count.

Other existing sources are outside the five-family denominator. RESECT has
14 TRAIN people with 70 acquired original images and 59 correspondence files;
25 cavity masks cover 13 of them, and the separately protected Case4 experiment
is not a new component-training record. See its
[completed original intake](../artifacts/resect-train-originals-execution-v1/RESULT.md).
IXI265 already has acquired T1/MRA and a derived vessel label; its
[paired-header result](../artifacts/ixi265-paired-header-result-v1/RESULT.md)
leaves T1/MRA registration, anatomy and label coverage unresolved. Reuse these
sources before any new IXI/RESECT download. NFBS is an unextracted archive;
125 publisher-reported people are not125 reconstructed local cases.

## Complementary datasets from the supplied screenshot

The screenshot is a source suggestion, not evidence that every modality is
available for every person. This metadata-only check made no imaging downloads,
accepted no new terms and changed no cohort roles. All new intake remains after
Gate2; existing acquired sources take priority.

| Candidate | Primary-source check on 2026-10-16 | Corpus role and restrictions |
|---|---|---|
| Olfactory-groove meningioma, OpenNeuro ds007345 | [Pinned v1.0.2 repository](https://github.com/OpenNeuroDatasets/ds007345/tree/c83e1d3ca8883d3695b6528441d255cc12b39e96), commit `c83e1d3ca8883d3695b6528441d255cc12b39e96`; [dataset description](https://raw.githubusercontent.com/OpenNeuroDatasets/ds007345/c83e1d3ca8883d3695b6528441d255cc12b39e96/dataset_description.json) says CC0. README documents T1 before/after contrast, T2, raw 64-direction b=1500 diffusion with opposite-PE b0, resting-state BOLD, topup/eddy derivatives and CSD tractography. | Separate **meningioma** diffusion/tract-displacement and longitudinal-observation track. It does not add glioma location diversity or acquired vascular/skull truth. Tractography is estimated; resting-state BOLD is not task-localized motor/language ground truth. Unknown-timed clinical/outcome fields stay outside planning. |
| ReMIND | [TCIA collection](https://www.cancerimagingarchive.net/collection/remind/) describes 114 people with preoperative MRI, intraoperative MRI and 3D ultrasound plus typical segmentations; release rights are CC BY 4.0 in the pinned existing manifests. Source summary was retrievable through search; direct page fetch failed. | Reuse the already acquired 82 source groups. Qualify paired observations/frames to test anatomy change and registration. These scans are not continuous surgical actions, tool-force trajectories, or causal evidence for alternate routes. Do not count all 114 as gliomas or local usable cases. |
| TopCoW | [Official data page](https://topcow24.grand-challenge.org/data/) states 125 training CTA/MRA patient pairs with annotations,5 validation pairs without public annotations and70 private test pairs. [Permanent release/API](https://zenodo.org/api/records/15692630) describes CoW masks, region boxes and graph labels; core data terms require source attribution and permission for commercial use. | Paired arterial **Circle of Willis** component/registration benchmark; not whole-brain arterial/venous ground truth or a glioma cohort. The5 validation images lack public labels; the 70 private tests cannot be assumed available. Per-archive rights and annotation ancestry require verification; CC0 Lausanne images do not automatically license new annotations. |
| Human Connectome Project, Young Adult S1200 | [Official release](https://www.humanconnectome.org/study/hcp-young-adult/document/1200-subjects-data-release) reports 1,206 healthy participants,1,113 with structural MRI, with diffusion/resting and task-functional resources, retests and smaller 7T subsets. [Data-use terms](https://www.humanconnectome.org/study/hcp-young-adult/data-use-terms) require an account and agreement to open-access terms; restricted variables have a separate process. | Healthy structural/diffusion/functional priors or methodological controls only. Never transplant HCP function as observed tumor-patient function. Group relatives and retests for independence. No login, acceptance, restricted-data application or new agreement was undertaken; any required new terms need user approval. |

A complete recursive **metadata tree**, not image payloads, was read for
`ds007345` at the pinned commit. It lists 37 baseline participant IDs and 23
`fu` IDs that refer to repeat visits. Baseline folders contain structural-image
pointers for 37, diffusion for 35, and resting-state fMRI for 34; follow-up folders
contain structural pointers for 23 and diffusion/resting fMRI for 22 each. Missing
baseline diffusion IDs are `olf25`/`olf36`; resting-state additionally lacks
`olf20`. Follow-up `olf06fu` has structural data only. These are listed-file
counts, not decoded/QC-passed modalities. Pair `olfNNfu` with `olfNN` before any
split: this is 37 people, not 60. The tree response SHA-256 is
`4c6dcc2071cc13ed6ec8db50768ec0c564c5dfca96cb763debe19e16a4b9ed4c`.

The [author's data report](https://www.frontiersin.org/journals/radiology/articles/10.3389/fradi.2026.1809871/full)
has a retained discrepancy: its methods report 23 follow-ups while its data-records
section says 22. Use the pinned per-file inventory and verify timing/linkage
before admitting a longitudinal pair. Do not silently assert uniform six-month
follow-up, all 37 complete diffusion/fMRI cases, or a released FLAIR channel merely
because a clinical protocol mentioned FLAIR. MONAI's four-contrast glioma model
is therefore not automatically applicable to this meningioma cohort.

TopCoW's 20 Lausanne external cases are already known source people, and its IXI
external subset also needs origin-cohort linkage. The core CTA/MRA pairs offer a
plausible **future** same-person bone-and-artery display route if CT bone coverage
and segmentation are independently qualified, but no observed tumor is supplied.
That is an inference from available modalities, not a demonstrated complete
patient bundle. Upstream vessel-model/benchmark overlap must be audited before
using TopCoW as independent evaluation.

A bounded later extension could qualify **two paired meningioma people** for
diffusion/longitudinal interoperability and **two TopCoW training pairs** for
same-person CTA/MRA source switching. Record those as separate component pilots;
they do not replace the 12-glioma target or increase its claimed location coverage.
HCP is deferred unless an unmet prior-validation need justifies its access and
family-aware split work. None of these additions is a prerequisite for Gate2.

## Current tumor location diversity

This audit independently computes union voxel count, affine-volume determinant
and centroid from the saved **source-label** arrays of seven permitted people.
It verifies the saved array-container SHA-256 before reading those arrays and
excludes MONAI estimates from the source-label union. It does not establish
external anatomical accuracy or original-source fixity by this self-consistency
check. Source-frame coordinates cannot establish a lobe, hemisphere, cortical
depth or functional boundary, especially with an arbitrary world origin.
Those fields therefore remain **UNKNOWN**.

| Person | Source-label union, mL | Centroid RAS+ mm | Anatomical region |
|---|---:|---|---|
| UCSF-PDGM004 | 41.919 | −161.892,84.878,104.565 | UNKNOWN |
| BTC05 | 11.437 | −29.180,25.740,8.792 | UNKNOWN |
| BTC 16 | 45.400 | 30.716,36.210,−13.186 | UNKNOWN |
| BTC 20 | 12.451 | 48.652,−27.422,28.296 | UNKNOWN |
| BTC22 | 13.915 | 15.603,−7.561,82.611 | UNKNOWN |
| BTC25 | 16.526 | 27.932,32.169,−17.573 | UNKNOWN |
| BTC28 | 10.269 | −6.684,8.655,51.421 | UNKNOWN |

BTC volumes use the fixed fractional-label threshold. UCSF includes core,
enhancement and FLAIR abnormality; the union is not histological tumor volume.
These differing label definitions prevent treating the table as a common
clinical tumor-size distribution. SELECT/EVAL tumor locations were not measured
or used to choose cases.

The separate [six-case source-report sidecar](../artifacts/btc-six-source-location-evidence-v1/evidence.json)
binds exact TRAIN identities to the pinned BTC participants table and
[primary paper, Table 1](https://www.eneuro.org/content/5/3/ENEURO.0083-18.2018/tab-figures-data).
It reports PAT05/PAT28 left frontal, PAT16 right fronto-temporal, PAT20/PAT22
right parietal and PAT25 right temporal. These are six patient-level reports
across four named site categories, with two left and four right reports. They
do not establish native-mask localization, reviewed anatomical diversity,
functional boundaries or automatic seed locations. The canonical catalogs'
region/lobe/hemisphere fields remain UNKNOWN; the additive labels carry
`source_reported_unreviewed` status and grant no training or anatomy acceptance.
No location is guessed for UCSF004 or unrelated saved variants.

## Construction driven by the recording

The supplied recording was previously sampled and documented in
[the functional reference audit](SUPERGOAL_REAL_OBSERVATIONS.md#appendix-b-functional-reading-of-the-supplied-medivis-recording).
The 32.67-second recording was sampled at 0.65, 5.55, 10.78, 16.34,
21.89, 27.44 and 31.69 seconds. The inspected frames show session/source selection
and linked views, bone/vessel volume presentation, primary/overlay fusion,
segmentation controls, implant templating, and an operating-room device view.
Those are workflow references, not evidence of glioma physiology, complete
vascular topology or registration accuracy. Implant templating motivates physical
annotation tools; it is not evidence that this recording demonstrates a glioma
entry plan or an executable resection timeline.

| Visible behavior | Required corpus product | Existing source route |
|---|---|---|
| Linked 3D and axial/coronal/sagittal views | Native volume, declared physical frame, slice cursor and source-linked mask; saved source identities | UCSF registered four contrasts; BTC T1; one CT and one MRA POC |
| Bone cutaway and vessels | Independent acquired CT/angiographic volumes, segmentation coverage and uncertainty; display clipping distinct from removal | SynthRAD for bone; Lausanne/IXI for vessels; separate people until a same-person link is proven |
| Fusion and source/layer controls | Source selector, unchanged originals, transform graph, registration-QC record, provenance and visible unsupported regions | SynthRAD CT/MR; Lausanne T1/TOF only after registration validation |
| Named tumor/brain structures | Source annotation separate from model estimate, measured label semantics, reviewed parcellation/ventricles | UCSF/BTC/ReMIND; no invented function or vessels |
| Entry/opening, chronology and branch comparison | One saved episode on one source-bound patient; snapshots and actor-visible evidence remain reproducible | Current Gate2 runtime; skull-dependent openings only on supported CT anatomy |

No combination of the three demonstration patients may be labeled one patient.
Missing skull/angiography/function remains unknown and visible. A labeled virtual
lesion on one real tumor-free host adds a generated target to that host; it does
not make a multi-patient composite legitimate.

## Quantitative targets, staged after Gate2

These are prospective engineering targets, not achieved scientific sample sizes.
Freeze eligible IDs and failure denominators before new processing. Keep failure
records; do not replace a failed case based on model scores or desired results.

1. **First construction tranche: up to 12 distinct real people.** Process the
   existing 6 BTC TRAIN tumor cases, 3 SynthRAD TRAIN CT/MR pairs (one per anonymous
   center where eligible), and 3 Lausanne/IXI TRAIN vascular hosts. Target 6
   `brain+tumor`, 3 `brain+skull`, 3 registered `brain+vessels`; report each achieved
   count separately and permit fewer passes. Count overlapping same-person work
   once. UCSF 004 remains a useful additional development demonstration with
   declared BraTS overlap, not an independent segmentation test.
2. **Second tranche: up to 24 real people.** Add 6 previously acquired, metadata-
   eligible ReMIND TRAIN tumor cases and 3 more subjects in each component track.
   Target 12 tumor-bearing, 6 CT/MR and 6 registered vascular people. Same-person
   complete acquired tumor+CT+angiography remains a **source-discovery target of
   1**, with present count0; there is no supported promise the existing public
   cohorts can supply it. Retain the incomplete tracks when it cannot be found.
3. **Measure real location coverage before asserting it.** Seek at least three
   reviewed supratentorial regions and both hemispheres among the 12 tumor cases,
   with at least two people in each of three declared annotation-volume bands
   (<10,10–30,>30mL) when those bands share a label definition. These are research
   stratification bins. Report observed gaps and UNKNOWN strata; do not move
   protected patients, alter real labels or call source cohorts balanced to meet
   a quota. Record insular/deep and near-vessel cases only when actual evidence
   supports localization and coverage.
4. **Virtual target after the one-host pilot succeeds:96 descendants from 4
   qualified TRAIN hosts**, not96 patients. Per host: four reviewed seed regions
   (frontal, temporal, parietal, and a supported deep/insular region), three target
   density-contour volumes (5,15,40mL), and two diffusion/proliferation parameter
   settings. Balance seed side across the four hosts. Retain ineligible or failed
   combinations; do not force seeds into missing tissue support. Generate this
   bounded set in resumable batches only after pilot review. Report96 targets
   separately from successful solver outputs and zero added patients.

Source candidates are selected by frozen role and imaging/annotation availability,
then declared ID/hash order; performance does not select the next patient.
Existing post-exposure evidence may be reused without calling the cohort broadly
admitted. Expansion does not require new paid services, agreements or uploads.

## Exact source-to-bundle procedure

1. **Bind identity and rights.** Namespace source/person/visit, check canonical
   aliases and origin cohorts, preserve frozen role, pin release and per-file
   hashes, and store license/citation/restrictions. Source data, labels and model
   weights have separate rights and provenance. A “control” label is not by itself
   adequate proof of a tumor-free generator host.
2. **Retain originals and qualify acquisition.** Reuse the existing cache. Check
   immutable byte identities, patient/series/SOP links, contrast, acquisition
   phase, spacing/units, qform/sform, orientation, coverage and de-identification.
   Keep slice/frame failures in the denominator. Downstream copies never replace
   originals. Do not reinterpret a SynthRAD outline as skull or an aneurysm-region
   annotation as complete vasculature.
3. **Register only the same person.** Preserve native volumes. Create a versioned
   transform graph with source/target frame hashes, algorithm/version/parameters,
   interpolation and inverse direction. Start with rigid CT/MR or T1/TOF where
   justified; deformable registration requires separate validation. Inspect
   linked MPR overlays and independent landmarks; common affines or a visually
   convincing blend are insufficient. Unresolved Lausanne frames stay unfused.
4. **Run compatible frozen models.** MONAI BraTS requires registered T1/T1ce/T2/
   FLAIR; do not feed single-T1 BTC as equivalent. Run SynthStrip/SynthSeg only on
   supported images, TotalSegmentator on acquired suitable CT, and vesselFM on
   compatible angiography under its research/noncommercial checkpoint terms.
   Keep weight checksum, training overlap, preprocessing, runtime, peak memory,
   mask grid/coverage and failures. An inferred segmentation is `estimated`,
   distinct from source-supplied and reviewed labels.
5. **Review anatomy and measurable diversity.** Check brain/tumor support,
   compartments, ventricles, surface topology, skull defects/thickness coverage,
   vessel components/centerlines and boundary truncation. Store centroid/volume
   in a named frame. Lobe/hemisphere/depth requires reviewed parcellation or
   source-supported localization plus its transform; point coordinates alone
   do not populate those fields. Acquired tract/functional data need separate
   preprocessing, timing and interpretation; atlas priors remain estimates.
6. **Package one immutable version.** Bundle the actual source references,
   transforms, masks/surfaces, coverage/unknown maps, licenses and review status.
   Record model and code revisions, source/derived hashes and parent bundle.
   Open/save/reopen in Electron; exercise source switching, linked slices,
   opacity/cutaways and provenance. Plan/episode snapshots bind that exact version.
   Re-run the inventory and publish counts by people, visits, bundles, variants,
   anatomy combination, QC disposition and role.

The next real scaffold slice now uses the exact next two TRAIN people in the
frozen SynthRAD center order: `1BB033` and `1BC077`, following center A's existing
`1BA336`. A fresh full archive checksum precedes extraction of only their four
named acquired CT/MR members; ZIP/gzip CRC, exact length, physical frame and
finite-value checks precede reconstruction. The unchanged pinned TotalSegmentator
fast model estimates brain/skull on CT and restores the native grid. Each saved
case retains both original acquisitions, model outputs and receipt, with MRI as
a separate unreviewed display source and fusion disabled. Source checks and both
bounded model runs passed, followed by the
[independent source/mask/bundle audit](../artifacts/poc-synthrad-bc-integration-v1/independent-qa.json).
These radiotherapy-source cases are not declared healthy or tumor-free, and
missing tumor labels do not permit them to become virtual-glioma hosts. Their
model/GUI integration is separate from the frozen ten-person catalog.

The [later twelve-source release](../artifacts/patient-corpus-twelve-recovery-v1/RESULT.md)
qualifies those two CT/MRI candidates with the exact prior ten. Its retained cold
execution fails the memory guard; separately declared fresh workers complete the
two pending receipts under the unchanged cap, followed by cached verification.
An independent metadata audit preserves the earlier releases and source records.
The actual Cases browser exposes all 12 source identities and three skull cases,
with canonical bundle checks and separate MRI inspection. This is a source
engineering milestone: location fields remain UNKNOWN and the intended three
registered vascular hosts, reviewed tumor-location diversity and independently
accepted anatomy remain open. Saved versions and virtual descendants add no people.

## One bounded PDE generator pilot, only after Gate2

Use the author-maintained [TumorGrowthToolkit](https://github.com/m1balcerak/TumorGrowthToolkit)
single-cell Fisher–Kolmogorov solver as the first candidate. The primary README,
checked 2026-10-16, exposes single-cell, two-cell/nutrient and DTI variants and an
MIT code license. Pin a commit and inspect its exact license/solver interface
before installation; the repository name alone establishes no data rights or
physiological validation. No generator was installed or run during this audit.

**Execution update, October 16:** the [subsequent IXI integration result](../artifacts/ixi265-virtual-pilot-v1/RESULT.md)
now records a pinned FK adapter and one homogeneous-domain geometry control on
the publisher-described healthy IXI265 host, after the actual first-release
gate. It preserves acquired MRI and passes numerical/source and Electron
save/reopen checks. Individual tumor-free review and tissue maps are still
missing; three bounded SynthSeg failures are retained. This control does not
close the tissue-conditioned pilot specified below or release the 96-descendant
campaign. The [separate ten-person catalog release](../artifacts/patient-corpus-ten-person-v1/RESULT.md)
now includes the acquired IXI candidate; the frozen nine-person catalog remains
unchanged and the generated descendant is excluded.

The [later independent quality checks](../artifacts/ixi265-quality-independent-v1/RESULT.md)
retain a first N4/Atropos rejection before fitting: eight zero-intensity cells
within the original estimated brain mask. A separate positive-intensity fitting
support declares those exact cells UNKNOWN and preserves the source mask,
intensities and native grid. Prospective review passes; the subsequent single v2
attempt passes those guards but crashes at native N4 entry, with no completed
fitting or candidate. That failure and unchanged source bytes are independently
verified and retained before any generated-image compatibility diagnostic.
Any resulting class posteriors are fitted membership probabilities, not physical
partial-volume tissue fractions, and cannot automatically enter the GM/WM solver
contract. A successful fit still requires anatomical and host review.

The [later strict admission guard](../artifacts/virtual-glioma-tissue-review-guard-v1/RESULT.md)
now enforces a review bound to the exact real GM/WM pair, image/frame, source,
method, known support and UNKNOWN masks. Mapped categorical coverage remains
separate from interpolated class weights. This does not create a review for the
IXI candidate or turn memberships into physical tissue fractions; the existing
uniform numerical control retains its prior qualification identity.

The [subsequent generated comparison](../artifacts/poc-ants-mask-dtype-generated-v1/RESULT.md)
isolates our N4 mask representation defect: unsigned-byte and float masks have
equal values/frames, but the unsigned-byte pointer is interpreted incorrectly by
the native float-mask interface. The float arm completes the generated pipeline;
a separately prepared typed kernel changes only that API representation.
This supports a targeted adapter repair while preserving both failed real
protocols, source support and the tissue/host acceptance gates.

The [subsequent typed-mask v3 run](../artifacts/poc-ixi265-atropos-typed-mask-run-v3/RESULT.md)
now completes one bounded N4/Atropos fit and passes independent checks on all ten
native outputs. The acquired T1 and original brain are unchanged; eight exact
cells remain UNKNOWN and raw memberships remain unmodified. This is an unreviewed
intensity-class candidate, not accepted tissue fractions or a qualified host.
Separate display packaging and actual Electron save/reopen follow the numerical
result; the tissue-conditioned campaign remains gated by the review below.

The [later native display integration](../artifacts/poc-ixi265-tissue-display-package-v1/RESULT.md)
now preserves the acquired T1, original brain, primary/planning identities and
both references while adding one four-layer estimate display. Independent saved
mask/source checks and actual Electron controls/save/fresh reopen pass. The raw
memberships remain archived, and no anatomical or normal-host acceptance follows.

**Manual tissue-reference lead, checked October 16:**
[IBSR's publisher](https://www.nitrc.org/projects/ibsr/) provides expert-guided
segmentations; its [file list](https://www.nitrc.org/frs/?group_id=48) separates
18-subject and 20-normal-subject releases. The
[18-subject release notes](https://www.nitrc.org/frs/shownotes.php?release_id=2316)
document rotated, bias-corrected T1 and trinary tissue labels. This is a possible
independent tissue-component reference, not raw native imaging, a glioma cohort
or evidence about IXI265. Published resource/release rights differ and require
reconciliation before use. The general README download redirected to login;
no account, agreement or image acquisition occurred. Freeze version, label
mapping, participant roles and preprocessing before any benchmark or host use.

The [subsequent IBSR20 component prospectus](../artifacts/ibsr20-component-prospectus-v1/PROTOCOL.md)
finds the separate updated 20-Normals README publicly accessible and fixes
record `2_4`, source slice offset, label codes and image-first scoring. Both
selected image/label archive HEAD requests redirect to login; exact transfer
bounds, additional terms and project-use eligibility remain unresolved. No
payload or model run is released. The protocol forbids reference-label support
and label-driven class remapping; a later component score could not establish
IXI265 tissue accuracy or qualify its host.

Select exactly **one existing TRAIN host** with reviewed tumor-free status,
brain/tissue support and accepted within-person frames. Prefer a qualified
Lausanne/IXI T1–MRA host so retained vessels can constrain the same scene. If no
host passes, record the failure and continue real-case work; do not erase an
observed tumor or attach another person's brain.

The pilot solves `dc/dt = div(D(x) grad(c)) + rho*c*(1-c)` in the accepted brain
mask with declared no-flux boundaries, one reviewed seed, tissue-conditioned
isotropic coefficients, units, initial radius/amplitude, duration and seed.
Do not infer a DTI tensor without acquired qualified diffusion. Keep the full
continuous density and every derived contour threshold. Start with a bounded
64³ grid and a single 96³ refinement check, at most 15 minutes and 3GiB per run on
one worker; stop on resource/numerical failure and preserve the receipt. Pin
actual parameters after checking the solver's units/stability conditions,
rather than copying a dimensionless example into millimeters and days.

Verify finite/nonnegative density, no tumor outside allowed support, zero-growth
and zero-diffusion controls, deterministic replay and coarse/fine volume/front
sensitivity with independent calculations. The result is **generated density and
geometry**, not patient-calibrated infiltration. Do not name contours enhancing
core/necrosis/edema without an explicit supported mapping. Do not claim edema or
mass effect: the first pilot leaves acquired host geometry unchanged. Any later
mass-effect model must consistently update brain, ventricles, vessels and source
mapping or remain labeled `geometry_only`.

Each descendant records `host_patient_group`, host bundle/hash, inherited split,
solver/commit, tissue-mask hashes, coefficient units/maps, seed/coordinates,
random seed, discretization/time-step, stopping rule, density/contour hashes,
QC, and `generated` status. Its anatomy stays linked to the original host. All
seeds, growth times, masks, crops, episodes and teacher trajectories from that
host inherit the same role. Never use a SELECT/EVAL host for training or count a
synthetic descendant as an independent patient or external-validation sample.

## Validation and stopping rules

Software controls and external accuracy evidence are reported separately.
The inventory's ten controls cover protected access, canonical aliases, role
conflicts, duplicate/virtual counting, source-presence rules, source-label union,
physical affine volume, LPS/RAS conversion and corrupt/incorrect bundle refusal.
Those tests do not validate clinical anatomy or mechanics.

For segmentation, compare against suitable source manual/expert annotations
where available; preserve model-assisted label ancestry and checkpoint training
overlap. Report per-person Dice plus surface-distance/95th-percentile Hausdorff
metrics, failures and coverage, not just an aggregate overlap. Vessel validation
also needs centerline/connectivity errors and small-vessel coverage; a high Dice
on an aneurysm region does not validate all vasculature. If no independent skull
or vessel truth exists, label accuracy unvalidated and obtain qualified blinded
annotation/review before making an accuracy claim.

For registration, freeze independent anatomical landmarks not used by the
optimizer and report target-registration error in millimeters with region/voxel
uncertainty, including failed/missing landmarks. RESECT paired MRI/US and released
correspondences, and ReMIND staged imaging, can support observed registration/
brain-shift tests after their own admission. They provide no per-tool causal
forces, alternative-route outcomes or simulated-action truth. Predeclare
intended-use tolerances with specialist input; do not invent a universal clinical
millimeter threshold from this inventory.

For generated anatomy, numerical convergence, mask consistency and realistic
looking overlays establish software/geometric plausibility only. Compare
morphology and density-front statistics with appropriate independent real data
without tuning on protected patients; do not infer tumor-growth accuracy from
single timepoint masks. External physiology claims require independent
longitudinal measurements and uncertainty analysis. No current endpoint is a
calibrated probability of neurological deficit.

Preserve **UPENN external final**, **UTSW structural validation**, **BTC 29/31
unopened transfer**, all SELECT/MEASUREMENT_EVAL roles, the UCSF/BraTS alias,
Lausanne/TopCoW overlap and RESECT Case4 protected observations. The later latest
steering permits synthetic training data; it does not make those groups usable
for development. Freeze model/generator/registration choices before a separately
released evaluation. Public labels, hidden references and later outcomes must
not leak into deployed policy inputs or candidate filtering.

## Reproduce this audit

From the checkout, use the existing environment and explicit output location:

```sh
PYTHONPATH=src .venv/bin/python scripts/patient_corpus_inventory.py \
  --output-root ../RessectionLab/outputs \
  --measure-tumor-geometry --output /tmp/patient-corpus-inventory.json
PYTHONPATH=src .venv/bin/python -m pytest tests/test_patient_corpus_inventory.py -q
```

Omit `--measure-tumor-geometry` for metadata-only operation. The report is created
exclusively, never overwritten. Metadata hashes are freshly measured; source
hashes are copied from named receipts and explicitly **not rehashed**. Presence/
size checks do not prove current byte integrity. The default bundle root is the
checkout's `outputs`; the explicit argument above is needed in the isolated POC
checkout to inspect the existing output cache. The script does not scan arbitrary
workspaces, open protected payloads, establish gate completion, or execute models.

Primary-source links checked 2026-10-16 include the
[SynthRAD release](https://zenodo.org/records/7260705) (acquired CT/MR; outline-mask
semantics and source preprocessing),
[IXI source](https://brain-development.org/ixi-dataset/) (source lead), and the
[TumorGrowthToolkit repository](https://github.com/m1balcerak/TumorGrowthToolkit)
(generator scope). The current TCIA ReMIND page fetch failed; ReMIND claims above
use the pinned local primary-source metadata and receipts rather than asserting
that a live refresh succeeded. TopCoW annotation rights/initializer ancestry
remain unresolved in the existing source audit; its Lausanne derivatives add
zero people and are not promoted by this plan.
