# RessectionLab dataset research catalog

Publisher listings and access terms checked November 18, 2026. The counts below
describe the cited releases. For acquired originals, source-person families,
studies, series, segmentation files, prepared or accepted packages, and synthetic
descendants, use the inventory and release receipts in the [collection guide](data-rebuild.md)
and [storage ledger](local-storage-ledger.md). Established cohort roles and
exposures follow the [cohort registry](cohort-registry.md) and each source-specific
frozen manifest.

## Local reuse baseline

The private HDD audit is indexed in
`catalog/inventory/multimodal-anatomy-baseline-20261118-v1/README.md` under
`RESSECTIONLAB_DATA_ROOT`. It found these reusable source components:

- TopBrain v3 had 25 TopCoW people, 50 CTA/MRA scans and 100 supplied masks.
  Selected exact archive members were preserved. The full release ZIP and
  separate CoW centerline archive were outside the verified local baseline.
- The pinned UCSF structural mirror had 50 development-training people, 200
  structural images and 50 supplied tumor files. The mirror carried no official
  v5 diffusion or perfusion payload. The IXI source area had 49 T1/MRA pairs
  plus a separately sourced 100-ID vascular annotation archive. Four of those
  IDs overlap the 49 pairs. The checkout also retains selected IXI265 diffusion.
- The EDEN2020 healthy source had 15 archives and their 15 extracted hosts.
  The separate v1 and v2 derived collections each had 15 baselines and 120
  synthetic scenarios on the same 15 families.
- The scoped RESECT baseline held two structural MRI and three intraoperative
  US observations for Case3. The checkout BITE-09 preview had preoperative
  T1CE and a supplied tumor mask. ReMIND and RHUH had existing source evidence
  and role records; their matched raw payloads were not verified in this audit.
  An unrelated paired CT fixture belongs to the Slicer Registration Library.

This was a pre-acquisition snapshot. The current collection/index receipts
record subsequent downloads, spatial review and case admission separately.

| Selective addition | Retained coverage | Case admission |
| --- | --- | --- |
| TAIR v1.0, 00273, 00244 and 00410 | Three acquired CTA series | Three native inspection studies. Pilot brain and skull estimates for 00273. |
| Head CTA, 013 and 189 | Two acquired DICOM studies and the 013 aneurysm STL | Two native inspection studies. The STL remains withheld from display pending frame qualification. |
| ISLES’24 v7, 0146 | Acquired NCCT and CTA, publisher-registered CTA, automated CoW mask | One study with independent acquired grids and separately identified derivative. |
| TopCoW/TopBrain 006 | Original CTA CoW mask, publisher VTP surface and graph, reused native MRA and its V1 annotation | One companion study on a previously acquired person. |
| RESECT Case3 | Two observed cavity masks and paired landmark files, reused MRI and staged ultrasound | One paired observational study on a previously acquired person. |
| CADS BrainCT-1mm, 465 | One acquired noncontrast head CT and three publisher model-label volumes, with source metadata | Separate native inspection study for head-bone and brain tissue coverage. |

## A. CTA and paired cranial imaging

| Source and publisher release | Access and rights | Anatomy and reuse decision |
| --- | --- | --- |
| [Annotated Head CTA](https://zenodo.org/records/15697196), Zenodo, 2025-06-17, DOI `10.5281/zenodo.15697196` | Direct CC BY 4.0 ZIP64, 25.58 GB. Verified HTTP ranges support selective retrieval of DICOM series. | 172 defaced CTA studies (149,329 DICOM slices), comprising 90 aneurysm-free controls and 82 cases with aneurysms. The archive supplies 122 aneurysm STL files. Some series use JPEG Lossless, and slice positions can imply a different spacing from nominal thickness. Review cranial coverage, defacing, slice geometry and STL frame for each selected scan. [Publisher descriptor](https://www.mdpi.com/2306-5729/11/4/74) |
| [ISLES’24](https://zenodo.org/records/17652035), Zenodo v7, 2025-11-20, DOI `10.5281/zenodo.17652035` | Direct public 99.0 GB `train.7z` under CC BY-NC-SA 4.0. The verified 0146 pilot uses bounded prefixes and one validated LZMA2 reset. Other subjects may require decoding more of a solid stream. | 149 challenge training patients. Defaced acute CT, CTA, CTP and perfusion maps, with NCCT-space derivatives, automated CTA CoW labels, LVO masks and follow-up DWI/ADC infarct masks. Local 0146 retains acquired NCCT/CTA, registered CTA and automated CoW only. The TopCoW 26-case external ISLES subset covers separate publisher IDs. |
| [TAIR cerebrovascular imaging](https://tair.kitware.com/clinical-data/), Kitware/ARPA-H v1.0, 2026-11-03 | Direct CC BY 4.0 defaced NIfTIs in public S3. Select the subject and series through the [publisher manifest](https://multimodal-imaging-for-cerebrovascular-disease.s3.us-east-1.amazonaws.com/manifest.csv) and object listing. A 70.45 GB combined ZIP64 is also available. | The site describes 500 combined CTA/MRA studies. The scan-object listing yielded 580 CTA NIfTIs across 414 subject prefixes, 20 MRA prefixes and nine shared CTA/MRA prefixes, for 425 imaging prefixes in the scan-object listing. The separate scan/fluoroscopy manifest uses a different scope. The listing exposed no current masks, centerlines or anatomical models. Verify coverage per series. [Public S3 registry](https://registry.opendata.aws/arpa-h-air/) |
| [MCA aneurysm CT/CTA and arterial geometry](https://doi.org/10.6084/m9.figshare.26965450.v1), Figshare v1, 2024-11-13 | Direct CC BY 4.0 archives. The controls and patient packages are 4.82 and 10.74 GB. The nonsolid RAR5 layout and validated HTTP ranges support selecting stored NIfTI payloads. | 44 controls and 99 patients with 105 MCA aneurysms. The authors describe CT/CTA acquisition, CTA-derived artery and aneurysm STL, and hemodynamic data. Spot-checked archive members contained head CTA, local STL geometry and simulation assets. Separate NCCT files remain unverified. [Publisher descriptor](https://www.nature.com/articles/s41597-024-04056-8) |
| [CADS BrainCT-1mm](https://huggingface.co/datasets/huggingface/CADS-dataset/blob/eee3c0a993636264fd04104f7e2a090c507c0c43/0042_new_brainct_1mm/README_0042_new_brainct_1mm.md), CADS subset 0042, Hub revision `eee3c0a993636264fd04104f7e2a090c507c0c43` | Public ungated file-level access. The subset card specifies CC BY 4.0. | 484 anonymized noncontrast adult head/neck CTs in 1 mm NIfTI from Istanbul. The per-image segmentation parts include brain, tissue, bone and vascular classes and are curated model outputs. [Publisher label map](https://github.com/murong-xu/CADS/blob/main/resources/info/labelmap.md) |

## B. Supplied vascular labels and geometry

| Source and publisher release | Access and rights | Anatomy, overlap and reuse |
| --- | --- | --- |
| [TopBrain](https://zenodo.org/records/21972006), challenge organizers, Zenodo v3, 2026-08-17, DOI `10.5281/zenodo.21972006` | Direct 2.0 GB ZIP. Attribution is required and commercial use requires prior owner permission under the release’s OpenData Swiss terms. | 25 paired CTA/MRA people drawn from TopCoW. Original artery/vein label sets have 40 CTA and 42 MRA classes. V3 includes a separate 36-class TopAneu-compatible representation. Reuse the matching labels and existing preparations, preserving label-map version. |
| [TopCoW training and external test sets](https://zenodo.org/records/15692630), challenge organizers, Zenodo v1, 2025-06-18, DOI `10.5281/zenodo.15692630` | Direct per-cohort ZIPs. The main release requires attribution and prior permission for commercial use. External components retain their respective source terms. | 125 paired CTA/MRA people with CoW masks, ROIs and topology in the main set. The external archives annotate 26 ISLES’24/TUM CTA cases, 20 LargeIA CTAs, 20 Lausanne MRA controls and 20 IXI Hammersmith MRA cases. Link those identities before assigning roles or totaling cohorts. The LargeIA component requires original images separately. |
| [Circle of Willis Centerline Graphs and Morphometric Features](https://zenodo.org/records/17358162), Musio et al., Zenodo v1, 2025-10-15, DOI `10.5281/zenodo.17358162` | Direct 373.1 MB `CoW_Centerline_Data.zip`, CC BY-NC 4.0 per [publisher metadata](https://zenodo.org/api/records/17358162). | Covers 250 scans of the same 125 TopCoW people. Each modality/person has a labeled VTP graph and mesh with JSON nodes, variants and morphology. Match the TopCoW ID and modality, then verify the physical transform and exact scan grid before reusing supplied geometry. |

## C. Multimodal brain MRI and glioma anatomy

| Source and publisher release | Access and rights | Anatomy and reuse decision |
| --- | --- | --- |
| [UCSF-PDGM](https://www.cancerimagingarchive.net/collection/ucsf-pdgm/), TCIA v5, 2025-05-30, DOI `10.7937/tcia.bdgf-8v37` | CC BY 4.0. TCIA lists a public Aspera package. Reuse verified existing case files and receipts, preserving official and mirror provenance. | 495 people across 501 exams, with preoperative structural, diffusion and perfusion MRI and supplied glioma segmentations. Processed structural images are brain-extracted. Use official metadata to group follow-ups and reconcile BraTS identifiers before claiming independence. See [v5 case acquisition](ucsf-v5-targeted-acquisition-2026-10-04.md). |
| [IXI](https://brain-development.org/ixi-dataset/), IXI project public archive | Direct modality tarballs under CC BY-SA 3.0. The publisher gives no versioned dataset DOI. Join the [separate CC BY 4.0 vascular release](https://zenodo.org/records/17393202) by source ID and preserve component terms. | Nearly 600 healthy subjects with T1, T2, PD, MRA and 15-direction diffusion. Reuse existing hosts. Match modalities and hospital IDs, preserve the limited angular resolution and check overlap with the TopCoW external set and VesselVerse. |
| [UPENN-GBM](https://www.cancerimagingarchive.net/collection/upenn-gbm/), TCIA v2, 2022-10-24, DOI `10.7937/tcia.709x-dn49` | Public CC BY 4.0. TCIA Data Retriever and [NCI IDC](https://portal.imaging.datacommons.cancer.gov/collections/upenn_gbm/) provide DICOM routes. | 630 patients. Multiparametric MRI, diffusion/perfusion products, tumor and brain masks. **Reserved external evaluation.** Preserve that assignment for every representation and descendant. Check directional diffusion and gradient availability for any eventual selected cases. |
| [CFB-GBM](https://www.cancerimagingarchive.net/collection/cfb-gbm/), TCIA v4, 2026-09-11, DOI `10.7937/v9pn-2f72` | Public CC BY 4.0 NIfTI via Aspera. Selective candidate for a longitudinal or radiotherapy anatomy gap. A previous directory probe timed out. | 264 people with longitudinal MRI, GTV and brain masks. The publisher lists planning CT for 195 and RT dose for 194 at baseline. MRI is skull-stripped and CT is defaced. Check actual CT cranial coverage before using bone. |
| [EDEN2020 healthy MRI](https://zenodo.org/records/3994749), Castellano et al., Zenodo v1.0, DOI `10.5281/zenodo.3994749` | Direct per-subject archives under CC BY-NC-ND 4.0. Retain those terms on the source and check any redistribution of adapted material against them. | 15 healthy adults with T1, FLAIR, SWI, TOF-MRA, DTI, multishell diffusion, maps and supplied tractography. Reuse the existing anatomy and [synthetic pipeline](eden2020-healthy-synthetic.md). Derived or synthetic records retain the healthy source-person identity. |
| [BraTS 2021](https://wiki.cancerimagingarchive.net/display/DOI/RSNA-ASNR-MICCAI-BraTS-2021), challenge/TCIA, TCIA DOI `10.7937/jc8x-9874` | Deferred. [Challenge access](https://www.med.upenn.edu/cbica/brats2021/) uses registration. Apply the terms for the specific TCIA or challenge component. Face-reconstructable DICOM has separate controls. | Skull-stripped MRI and tumor labels support annotation and model-exposure research. Use published crosswalks for UCSF-PDGM and other included TCIA sources. UCSD-PTGBM's supplement belongs to the distinct BraTS-GLI 2024 track. |

## D. Surgical observations and registration

| Source and publisher release | Access and rights | Observations and reuse |
| --- | --- | --- |
| [ReMIND](https://www.cancerimagingarchive.net/collection/remind/), TCIA v1, DOI `10.7937/3rag-d070` | Public CC BY 4.0 DICOM via TCIA Data Retriever or [NCI IDC](https://portal.imaging.datacommons.cancer.gov/collections/remind/), with separate NRRD segmentation package. Reuse existing selected data and series bindings. | 114 tumor patients with preoperative MRI, intraoperative MRI and tracked 3D ultrasound. Supplied labels vary by patient and may include preoperative tumor, prior cavity, cerebrum, ventricles or intraoperative residual. Keep the stage and frame of each observation. |
| [RESECT](https://doi.org/10.11582/2017.00004) and [RESECT-SEG](https://osf.io/jv8bk/), NIRD and OSF, annotation release DOI `10.17605/OSF.IO/JV8BK`, [image descriptor](https://doi.org/10.1002/mp.12268), [annotation descriptor](https://doi.org/10.1002/mp.17317) | Images and original landmarks CC BY 4.0. RESECT-SEG masks and the [combined mirror](https://huggingface.co/datasets/MedOtter/RESECT-SEG) are CC BY-NC-SA 4.0. Reuse pinned components from [selective acquisition](resect-selective-acquisition.md). | 23 shared patient families with preoperative MRI, 3D iUS at surgical stages and homologous landmarks. Masks may identify tumor, falx, sulci and the resection cavity. Verify the actual stage and source grid. |
| [ReMIND2Reg](https://zenodo.org/records/12700312), Dorent et al., Zenodo v2.3, 2024-07-09, DOI `10.5281/zenodo.12700312` | Direct public CC BY 4.0 ZIP for training and validation. The [benchmark](https://github.com/ReubenDo/ReMIND2Reg) maintains a private test set and separate code terms. | 99 training and five validation people come from ReMIND. Preoperative ceT1/T2 and post-resection 3D US are resampled and cropped for registration. Preserve that preprocessing and report deformation as a registration estimate. |
| [BITE](https://nist.mni.mcgill.ca/bite-brain-images-of-tumors-for-evaluation-database/), McGill MNI, [descriptor DOI](https://doi.org/10.1118/1.4709600) | Publisher offers research downloads by group. Sample archive headers returned HTTP 200. An explicit redistribution license is absent from the landing page. Reuse locally available imaging first and verify terms for any later publication of derivatives. | 14 tumor patients with pre/postoperative MRI, intraoperative US and selected homologous landmarks. Availability and formats are group-specific. Use paired landmarks where both images are available. |
| [RHUH-GBM](https://www.cancerimagingarchive.net/collection/rhuh-gbm/), TCIA v1, 2023-06-09, DOI `10.7937/4545-c905` | Public CC BY 4.0 brain-extracted NIfTI, segmentations and clinical table. Raw DICOM falls under NIH Controlled Data Access. Reuse [existing acquisition evidence](rhuh-postoperative-imaging-audit.md). | 40 patients at preoperative, early postoperative and recurrence timepoints, with masks and resection-related measurements. Use linked timepoints and source label definitions for residual or cavity comparisons. |
| [RIRE](https://rire.insight-journal.org/), Vanderbilt project v3.0 | Public IPFS links for [one training](https://rire.insight-journal.org/download_training_data) and [18 test](https://rire.insight-journal.org/download_data) patient entries. The two download pages specify CC BY 3.0 US. Sample published gateway URLs returned HTTP 403 during this check. | Small matched CT/MR and PET/MR registration benchmark. Marker-based reference transforms remain sequestered and online scoring is retired. Use for physical coordinate and registration checks within available references. |

## E. Selective or reserved follow-on collections

| Source and publisher release | Access and rights | Scope and disposition |
| --- | --- | --- |
| [UCSD-PTGBM](https://www.cancerimagingarchive.net/collection/ucsd-ptgbm/), TCIA v3, 2026-03-13, DOI `10.7937/fwv2-dt74` | Public CC BY 4.0 NIfTI via Aspera in separate packages. | 178 people with 243 timepoints and MRI labels for enhancing, necrotic, FLAIR and cavity compartments. Preserve the identities and published roles of the 136-person main release and 42-person BraTS-GLI 2024 test supplement. |
| [MU-Glioma-Post](https://www.cancerimagingarchive.net/collection/mu-glioma-post/), TCIA v1, 2025-03-21, DOI `10.7937/7k9k-3c83` | Public CC BY 4.0 NIfTI via Aspera. | 203 people and 596 post-treatment timepoints. Skull-stripped MRI with tumor, FLAIR abnormality and cavity labels for postoperative comparisons. |
| [UCSD-BMETS-Longitudinal](https://www.cancerimagingarchive.net/collection/ucsd-bmets-longitudinal/), TCIA v1, 2026-10-05, DOI `10.7937/5ejb-9g94` | Public CC BY 4.0 NIfTI via Aspera. | 300 people with pre/post-treatment metastasis MRI and reviewed compartment masks. The publisher describes 1,065 timepoints and separately lists 1,066 studies. Deferred beyond the glioma pilot. |
| [UTSW-Glioma](https://www.cancerimagingarchive.net/collection/utsw-glioma/), TCIA v1, 2026-03-18, DOI `10.7937/dfae-1b86` | Public CC BY 4.0 NIfTI via Aspera. **Reserved external evaluation.** | 625 glioma patients with multicontrast MRI and segmentation. The [descriptor](https://doi.org/10.1038/s41597-026-07274-4) reports 362 manually refined masks. Preserve evaluation intent for all representations and descendants. |
| [ADAM](https://adam.isi.uu.nl/details/), Utrecht 2020 challenge | Restricted by a signed [challenge confidentiality agreement](https://adam.isi.uu.nl/wp-content/uploads/2020/03/ADAMConfidentialityAgreement.pdf), with restrictions on non-challenge use and redistribution of images and derivatives. Deferred. | 113 training cases with TOF-MRA and structural MRI, with aneurysm annotations for detection and segmentation. |

## Deferred acquisitions requiring permission or further verification

| Source | Publisher evidence and disposition |
| --- | --- |
| [EDEN2020 glioma MRI](https://zenodo.org/records/5714812), Zenodo v1.0, DOI `10.5281/zenodo.5714812` | Restricted high-resolution MRI and diffusion-related files for 30 adults with glioma. Publisher access requests and terms apply separately from the healthy release. |
| [RSNA-ICA](https://doi.org/10.1148/dataset.ica.2025), RSNA 2025 | The [descriptor](https://arxiv.org/abs/2610.01135) reports 4,278 people and 7,202 series (2,566 CTA, 2,166 MRA, 2,470 MRI), with 178 vascular segmentations and 5,041 public series through [MIRA](https://mira.rsna.org/dataset/7). The [RSNA AWS listing](https://aws.amazon.com/marketplace/pp/prodview-rzsnz3mxm46rg) documents anonymous S3 and terms requiring noncommercial use without redistribution. The sampled S3 listing returned HTTP 403. Kaggle challenge terms are separate. Deferred under current acquisition priorities. |
| [TopAneu](https://topaneu-26.grand-challenge.org/), MICCAI 2026 challenge | Multicenter vessel-specific aneurysm CTA/MRA classification and segmentation. The organizers report over 400 training cases across two challenge batches. The open [Zenodo record](https://zenodo.org/records/19848807) contains the proposal PDF. The official challenge uses a verified account and join request. An anonymous imaging release with standalone terms is unverified. The compatible masks in TopBrain cover TopBrain scans. |
| [LargeIA](https://zenodo.org/records/6801398), Zenodo v1, DOI `10.5281/zenodo.6801398` | 1,338 internal and 138 external 3D CTA images with voxel-wise aneurysm annotations. Noncommercial terms and institutional access request apply. Preserve the identity of the 20 LargeIA cases annotated separately by TopCoW. |
| [SemanticVessel](https://proceedings.mlr.press/v340/lin26a.html), MLHC 2026 | Around 360 CTA scans with venous and 20 arterial label classes. The [authors require an institutional request](https://github.com/alceballosa/robust-vessel-segmentation#dataset-access). Inspect granted data terms separately from code and weights. |
| [DynaVessel](https://doi.org/10.1007/978-3-032-38085-2_43), MICCAI 2026 | Artery and vein masks from dynamic 4D CTA. The paper reports 110 training volumes from 29 patients and 165 test volumes from 14 patients. The [authors require a request](https://github.com/alceballosa/robust-vessel-segmentation#dataset-access). Their repository separately identifies CC BY-NC-SA 4.0 for the DynaVessel-trained models. |
| [VesselVerse annotations](https://github.com/i-vesseg/VesselVerse-Dataset), MICCAI 2025, [descriptor DOI](https://doi.org/10.1007/978-3-032-05169-1_63) | The authors require a request. The 950 annotated TOF-MRA, T1-MRA and CTA images reuse IXI, TubeTK and TopCoW scans. Preserve upstream identity and terms. Manual, model and consensus masks have distinct provenance. |
| [Human Connectome Project Young Adult](https://www.humanconnectome.org/study/hcp-young-adult/data-releases), S1200 and 2025 repackaging | The 1,206-participant S1200 release includes structural imaging, motor/language task fMRI and diffusion. The 2025 update repackages existing people. Open imaging requires a ConnectomeDB account and [Data Use Terms](https://www.humanconnectome.org/study/hcp-young-adult/data-use-terms). Restricted demographic and family data require approval. Group maps remain population priors. |
| [BIND](https://bdsp.io/content/n1vba1x5qt62frfjem65/1.0/), Brain Data Science Platform v1.0, DOI `10.60508/mby8-3a26` | About 39,000 people with heterogeneous clinical MRI, CT, PET/SPECT and linked EEG/polysomnography. Access requires a signed agreement, CITI certification, an AWS account and approval under the credentialed health data license. Some diffusion scans lack gradient metadata. |

## Existing targeted acquisition routes

For current resumable preparation commands, see the
[collection guide](data-rebuild.md#resume-preparation-and-review).

### UCSF-PDGM v5 diffusion

Official collection: https://www.cancerimagingarchive.net/collection/ucsf-pdgm/
Canonical DOI: https://doi.org/10.7937/tcia.bdgf-8v37

Version 5 fixes `DTI_eddy_noreg` headers and supplies per-exam rotated b-vectors in the post-conversion, post-FSL-eddy image frame. A November 9, 2026 supplemental release documents per-exam ADC scaling to mm²/s. Verify diffusion gradients, scaling and alignment against the exact exam before use.

Acquire one selected case's structural sequences, annotations, directional diffusion, matching b-values and that exam's rotated b-vectors. Verify gradient frame and image identity; do not substitute the generic b-vector download for a per-exam corrected vector file. Establish the DWI-to-structural transform, reconstruct motor pathways, retain failure/coverage information, and then extend to supported language bundles. Use corrected source derivatives to avoid unnecessarily making a novel MRI correction solver the critical path.

The collection currently lists Aspera access. A failed transfer remains a source-access problem, not permission to declare an unrelated mirror byte-equivalent. Record unavailable files and advance the independent structural work below.

### BTC preoperative and postoperative MRI

Preoperative release: https://openneuro.org/datasets/ds001226/versions/5.0.1
Postoperative collection: https://openneuro.org/datasets/ds002080
Primary paper: https://www.nature.com/articles/s41597-022-01806-4
Accessible paper record: https://pmc.ncbi.nlm.nih.gov/articles/PMC9637199/

The paper describes T1, diffusion and BOLD imaging, tumor masks, connectivity derivatives, and behavioral data. Its overall tumor sample includes more than glioma; apply the repository's version-specific pathology eligibility rather than treating every tumor as a diffuse glioma. Raw acquisitions are not the same as executed corrected reconstructions.

Preserve the prospective roles in `manifests/experiments/btc-spatial-development-cohort-v1.json`: PAT22/PAT25 additional training; PAT26/PAT27 checkpoint-selection development; PAT29/PAT31 unopened later development-transfer cases. PAT05/PAT16/PAT20/PAT28 were already consulted. Later development transfer is not external final validation.

Inspect `derivatives/dmriqc` and `derivatives/TVB` for existing quality reports and connectome resources before rebuilding every auxiliary product. Regional connectivity matrices can inform network representations but cannot localize a fine surgical corridor by themselves. Postoperative images and tests must not become preoperative planning inputs. Use the author's raw-data preprocessing description for patient reconstruction and label any synthetic imaging separately.

### DES-derived motor and language resources

Author-linked DES coordinates/demographic release: https://zenodo.org/records/10439149
Language structural/functional maps, version 2: https://zenodo.org/records/16418628
Primary paper and availability statement: https://www.nature.com/articles/s43856-025-01121-0
Author implementation: https://github.com/FBK-NILab/White-Matter-DES-Brain-Maps

The language release lists `pub_release.zip` (19.1 MB), including `maps_paper_release` and data/scripts for selected figures. The paper links the earlier coordinates release. The older record was not directly retrievable during this review; its URL is supported by the authors' availability statement. Inspect release-specific license information before redistributing data or derived assets.

The repository already has seven prior proposals. Use their source records rather than creating duplicate downloads. Preserve template space, map meaning, interpolation and coverage when registering to a patient. Integrate explicit population-prior sensitivity analyses into plan evaluation; do not silently certify them as individualized functional anatomy. Use patient diffusion when available to refine the representation. The paper explicitly does not publicly release its glioma raw MRI, derivatives and clinical scores, so these maps cannot alone train clinical injury probabilities.

### ReMIND through NCI Imaging Data Commons

Official IDC collection: https://portal.imaging.datacommons.cancer.gov/collections/remind/
TCIA: https://www.cancerimagingarchive.net/collection/remind/
Canonical DOI: https://doi.org/10.7937/3rag-d070
Primary paper: https://www.nature.com/articles/s41597-024-03295-z
IDC download guide: https://learn.canceridc.dev/data/downloading-data

IDC lists 114 tumor patients with MRI, segmentations and intraoperative ultrasound. The collection includes preoperative tumor, pre-resection cerebrum and residual tumor segmentations where available. Select glioma cases using the accompanying clinical metadata; do not assume every case is glioma.

First select a small declared development subset with suitable preoperative anatomy and available cerebrum/tumor labels. Use these as a separate geometry/segmentation-assisted track rather than spending all effort deriving a brain envelope from full-head BTC scans. Cerebrum segmentation still does not establish a clinically appropriate cortical access window or functional safety.

Later use the intraoperative images to evaluate registration, modeled displacement and observation updates. They are snapshots, not a labeled sequence of surgical instrument actions. Never imitate the observed cavity as if it were the proven optimal resection.

IDC provides direct downloads and `idc-index` access. Select patients/series through the portal or supported client filters. Its collection-wide example is `idc download remind`, but do not use this as the default acquisition command for a small local pilot. Pin DICOM series identities and preserve reference-image/frame relationships during conversion.

### UPenn-GBM through IDC for reserved external evaluation

Official IDC collection: https://portal.imaging.datacommons.cancer.gov/collections/upenn_gbm/
TCIA: https://www.cancerimagingarchive.net/collection/upenn-gbm/
Primary paper: https://www.nature.com/articles/s41597-022-01560-7

IDC lists 630 GBM patients and MR/SEG data. The collection description includes multiparametric imaging, tumor and brain segmentations, and clinical/genomic metadata.

After the evaluation method and adaptation procedure are frozen, select the independent-site cohort prospectively. Check the selected raw and processed files for directional diffusion and valid gradients. Processed NIfTI and raw DICOM may require explicit coordinate reconciliation.

The IDC portal/client provides an alternative to TCIA transfer for the hosted DICOM collection; supplementary NIfTI/clinical resources may still use separate collection links. Download only selected cases and verify DICOM SEG references and the exported physical frame.

### TractoInferno tractography validation and initialization

Public dataset: https://openneuro.org/datasets/ds003900
Version cited by the dataset paper: https://openneuro.org/datasets/ds003900/versions/1.1.1
Primary paper: https://www.nature.com/articles/s41597-022-01833-1

This resource contains 284 healthy-subject samples from six sites, T1/diffusion-derived data, fODFs, tissue masks and reference streamlines for 30 bundles derived from multiple tractography algorithms. Start with a small sample respecting its published splits.

Use it to validate tract handling, physical-coordinate conversion, bundle extraction, coverage semantics and optional tract-model initialization before transferring to glioma anatomy. Reference streamlines are algorithm-derived rather than histological truth. Healthy anatomy cannot establish tumor-domain accuracy or clinical risk calibration. Synthetic lesions inserted into these cases remain simulated cases, not human glioma training patients.

## Common acquisition contract

Each acquired case needs patient/visit/group identity, release and source URLs, checksums, sequence/gradient identity, processing history, frame transforms, license, expected use, unavailable fields and split assignment. Track human patients separately from synthetic environments and optimizer seeds. Preserve expert-review status separately from source annotations and engineering QC. No dataset listed here supplies the complete patient-specific surgery-action/functional-outcome supervision needed to equate simulated encounters with postoperative deficit probabilities.
