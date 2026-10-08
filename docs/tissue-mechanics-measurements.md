# Measured tissue mechanics: available evidence and first test

Checked October 4, 2026 using primary sources, metadata, archive previews and HTTP headers. No image/force arrays, bulk archives, registrations or mechanics fits were opened or executed by this audit. **Start with a measured specimen-force benchmark; evaluate patient displacement separately.** None of the imaging releases below supplies a verified synchronized retractor pose/load history, contact law or cutting-force reference.

| Resource | Measurements and scope | Access verified in this audit |
|---|---|---|
| [RESECT, original release](https://archive.sigma2.no/dataset/5D6BFC33-F58D-4F56-88E8-C40AF269D6F2) | 23 low-grade glioma patients; preoperative MRI and before/during/after-resection US. Manual US–US correspondences exist for 17 patients. | Public **CC BY 4.0**, DOI `10.11582/2017.00004`; landing page HTTP 200, archive listed as 3.3 GB. A 16-KiB table-of-contents range returned HTTP 206 and exposed individual file URLs, byte sizes and fixity values; image transfer not tested. |
| [BITE, creator release](https://nist.mni.mcgill.ca/bite-brain-images-of-tumors-for-evaluation-database/) | 14 tumor patients. Group 1: selected pre/post US pair with ten expert landmarks. Group 2: preoperative MRI/pre-resection US with multiple raters. Group 4: MRI/post-resection US, about 15 landmarks; repeat tagging in six patients. | HTTPS HEAD 200: `group1.tar.gz` **4,882,291,030 bytes**, `group4.zip` **591,217,459 bytes**, both advertise byte ranges. Creator page/readmes were readable, but no explicit reuse license was found there; do not assume another dataset's license. |
| [ReMIND, original TCIA release](https://www.cancerimagingarchive.net/collection/remind/) | 114 patients; preoperative MRI, US before/after dural opening and after substantial resection where available, plus intraoperative MRI. | **CC BY 4.0**; collection page HTTP 200, DICOM release listed as 43.53 GB. Earlier repository work already verified selected ReMIND-001 structural DICOMs through IDC. This audit did not acquire its intraoperative series. |
| [ReMIND2Reg, processed release](https://zenodo.org/records/11387725) | Same ReMIND patient source: 99 training and five validation patients, post-resection US paired with ceT1/T2 when available. | API HTTP 200, **CC BY 4.0**; ZIP HEAD 200, **3,199,189,599 bytes**, provider MD5 `050887d39a48fa07c60a71ddf9fb3bf7`. Public images do not imply public reference landmarks. |

RESECT's `.tag` rows contain paired world coordinates in **mm**, with matching MINC/NIfTI world frames; bind each file to its actual source/destination images before converting to RAS. The online guide's wording about the reference image is ambiguous across pair types, so filenames, headers and point overlays must settle direction. T1 was rigidly aligned to FLAIR upstream. Its US–US Set 2 includes Set 1: these cannot become independent calibration/evaluation partitions merely by set name. Four repeated destination picks were averaged while the reference point stayed fixed. Before/during inter-rater distance was 0.27±0.05 mm; that is not a three-dimensional covariance, total tracking uncertainty, or permission to divide error by √4. These landmarks assess retained-anatomy correspondence, not force or missing-tissue motion. [Creator methods](https://users.encs.concordia.ca/~hrivaz/Xiao_Database_MedPhys.pdf)

BITE Group 2 has a rigid Talairach-like orientation/position change, without scaling; its separate MRI and US `.xfm` transforms return to their different native spaces. Group 1 and Group 3 cannot be combined by assuming identical coordinates. [Creator frame instructions](https://www.bic.mni.mcgill.ca/uploads/Services/00_readme_group2.txt) EASY-RESECT and RESECT-SEG are derivatives of RESECT, not independent patients; segmentation-only OSF files are not original images or load measurements. [CuRIOUS data guide](https://curious2022.grand-challenge.org/data/)

ReMIND2Reg is less suitable for a locally independent first mechanics test: its 2025 validation landmarks and test set are private. Reference construction includes pre-dural-US-to-MRI affine transfer, expert consensus, then T2-to-ceT1 landmark transfer. The quoted 1.89±0.37 mm variability comes from an earlier similar protocol, not direct uncertainty measurements for every released pair. A field fitted to those correspondences cannot independently validate itself. Native ReMIND-001's existing T1/T2 frame mismatch and unsuccessful, unselected rigid diagnostic remain unchanged. [Challenge protocol](https://arxiv.org/html/2508.09649v1)

**Available force measurements.** [Hyperelastic Human Brain 1–7](https://zenodo.org/records/8095559), version 1.0, is **CC BY 4.0**. API and archive HEAD returned 200: `HBE_Data.zip` is **12,968,598 bytes**, MD5 `fef23ea6291afd10bb668f763d25863f`; README is 1,593 bytes and the region lookup is 14,839 bytes. Seven postmortem donors supplied approximately 8-mm cylindrical specimens, with per-specimen heights in `geometry.yaml`. Files contain displacement **m**/force **N**, or rotation **rad**/torque **N·m**, for compression/tension to 15% and torsional shear to 0.15/0.3, cycles one/three. Curves are filtered and averaged approximations to quasi-static hyperelastic response, not raw time histories.

The actual fixture glues both specimen faces to sandpaper-covered plates; lateral slip cannot be substituted silently. Tests used hydrated tissue at 37°C within 72 hours postmortem. A published modified one-term Ogden/deal.ii inverse fit exists, but its implementation is available only on request. Published fitted parameters are not independent validation targets. This source does not establish live glioma, surgical retraction, viscoelastic relaxation or cutting behavior. [Primary experiment and boundary conditions](https://pmc.ncbi.nlm.nih.gov/articles/PMC10511383/)

**Narrow prospective recommendation:** use the first metadata-eligible specimen under the [frozen selection rule](tissue-mechanics-validation.md) as a separate mechanics-development record (the creator preview lists **HBE_01_03**; the acquired inventory must confirm eligibility), with no change to BTC, ReMIND-001 or reserved external patient roles. Before fitting, freeze geometry/unit conversion, no-slip fixture, material law, calibration curves and a withheld loading mode. For example, calibrate on third-cycle compression/tension and assess third-cycle low-amplitude torsional torque, retaining failure if the shared material law does not transfer. That is within-specimen prediction, not independent-donor generalization. Keep all specimens/modes from one donor grouped for a later donor-held-out study; do not tune using published aggregate parameters fitted on that held-out donor.

A later patient check should freeze the model before opening reserved RESECT images/correspondences. Measurements used to impose boundary displacement must be disjoint from withheld interior correspondence targets, with spatial grouping to limit near-neighbor leakage. Sparse landmarks are not necessarily surface boundary measurements; if no defensible boundary inputs exist, restrict the claim to displacement interpolation/image updating. Report physical endpoint errors against identity and rigid baselines, point coverage, missing/resected anatomy, numerical convergence and failures. Never force correspondence through a cavity, use a registration-generated warp as mechanical ground truth, or infer reaction-force accuracy from low landmark error.

## October 8 real-observation follow-up

These are source-admission findings, not new measurements or successful mechanics
validation. Existing HBE convergence failures and RESECT mesh-fidelity gates
remain unchanged.

**RESECT-SEG visible-cavity labels.** The [creator annotation release](https://osf.io/jv8bk/),
DOI `10.17605/OSF.IO/JV8BK`, has explicit **CC BY-NC-SA 4.0** annotation rights
in its [README](https://osf.io/download/4mkfg/); original RESECT images separately
use CC BY 4.0. Noncommercial component research is the intended use here;
commercial annotation rights are not assumed. The All-Labels v1 release dated
2024-02-15 uses revision-2 eight-bit masks. Metadata identifies 21 during and
22 after cavity volumes, representing 22 people. Case11 has neither label;
Case15 lacks the during label. Missing labels cannot become negative masks.

[Primary methods, sections 2.3 and 2.7](https://doi.org/10.1002/mp.17317) describe
manual contours on roughly every fifth slice, morphological interpolation,
manual refinement and review/revision by two neurosurgeons. Record that actual
annotation chain; do not describe every voxel as hand-drawn. No learned teacher
appears in this documented cavity-label workflow. Exact source identity and
image/mask correspondence still require acquisition QC.

These labels describe the visible dark ultrasound cavity and can omit ambiguous
blood-filled regions. They cannot establish complete removed tissue, cutting
forces or a surgical reward. The component task would use actual during/after
ultrasound at its recorded acquisition phase; later anatomy is not available to
a preoperative planner. Exact event times and tool actions remain absent.

The [family-level role manifest](../manifests/resect-component-cohort-v1.json)
freezes 14 TRAIN, four SELECT, four locally payload-held-out MEASUREMENT_EVAL
people and the previously exposed Case4 DEVELOPMENT person. Case3 is the
metadata-selected TRAIN ingestion pilot (9,184,649 source image/mask bytes).
Case4 is excluded from this learner; its protected during/after anatomy and
motion partition remain under their earlier contract. Earlier anatomical
inspection of its baseline MRI/US does not release those protected endpoints.
RESECT-SEG and verified CuRIOUS/EASY-RESECT derivatives inherit original people;
unmapped aliases remain quarantined. This is component evaluation, not an
untouched external clinical cohort. No new image/mask payload was read to
assign these roles.

**Measured interaction candidate: MULTIS donor004, run005.** The [creator donor
configuration](http://archive.simtk.org/multisdelta/SMULTIS004-1/Configuration/SMULTIS004-1.cfg)
marks runs001–004 rejected and run005 accepted. Preserve that correction to the
earlier run001 candidate before acquiring response curves. Proposed canonical
donor `MULTIS004` is DEVELOPMENT, grouping SMULTIS/CMULTIS acquisitions of the
same human; donor-role acquisition has not yet been enacted. Donors005–012
remain unopened for future prospective evaluation. The release is
[CC BY 4.0](http://archive.simtk.org/multisdelta/license.txt).

Read-only source inspection identified **234 core files / 343,532,158 bytes**:
one TDMS load/motion record, five configuration/XML files, acquisition/project
metadata, 224 surface-deformation CSV exports and calibration metadata. The
trial prefix is `005_SMULTIS004-1_SXX_IND_SKN-5` under the [official donor
directory](http://archive.simtk.org/multisdelta/SMULTIS004-1/). Each core size was
checked by HEAD; six transient metadata requests succeeded on one retry.
Underlying stereo images add 300 pairs and 21 calibration pairs. Their combined
876-file size is approximately 3.56 GB, estimated from sampled image sizes.
The 76 image times without deformation exports cannot become zero displacement.

Before any force/displacement comparison, inspect the actual run's channels,
units, source validity flags, clock pairing and spatial transforms. [Creator
code](https://simtk.org/svn/multis/app/InstrumentedSurgicalTools/PythonScripts/surgicalTools_experimentDataCheck.py)
expects motion in mm/degrees, configuration offsets in m/rad, TDMS time in ms
and six loads in N/N·m. Its rotation order is RzRyRx. Do not silently combine
raw-sensor, tip and femur-oriented quantities. Camera calibration alone does not
provide the camera-to-femur transformation.

[Testing notes](http://archive.simtk.org/multisdelta/TestingNotes.txt) describe
extra initial force data for donor004 and terminal-overlap alignment; the exact
trim needs observed timing, not an assumed fixed offset. The [primary
descriptor](https://www.nature.com/articles/s41597-020-0359-0) also reports an
indenter timing discrepancy of 71.5 ± 32.5 ms. Retain it as a limitation, not a
run005 correction. Donor004 had an extra freeze–thaw cycle. Calibration projects
reference CMULTIS filenames while released images use SMULTIS names; verify
the mapping explicitly. This candidate can validate a human cadaver **leg**
interaction, not live brain retraction, glioma material properties or injury.

A subsequent four-request transport check could not establish authenticated
run005 transfer: the static archive's HTTPS connection was refused; a DOI
metadata lookup timed out; the [creator HTTPS SVN donor directory](https://simtk.org/svn/multis/app/InstrumentedSurgicalTools/SMULTIS004-1/)
loaded at repository revision1079; the exact run005 filename beneath its Data
folder returned404. The directory revision/weak ETag does not establish file
equivalence, and no creator-published run005 checksum was found. Local SHA256
after HTTP acquisition would establish local integrity, not independent source
authentication. Keep that dependency explicit. A first schema-only sample
would be the TDMS plus its three configurations and donor CFG/XML: six files,
7,022,670 bytes. Resolve source/fixity and freeze donor roles before opening it;
keep the much larger surface exports closed until actual channel/timing checks.
