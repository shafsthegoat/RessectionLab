# Brain RL: annotated sources

Research snapshot: October 2, 2026. These are primary papers, original dataset records, author-maintained resources and official documentation. This is a targeted research map, not a claim that every relevant publication has been found. Entries distinguish accessible content from metadata-only leads. No full imaging cohort was downloaded or tested for this planning deliverable.

The master plan cites the stable IDs below. “Use in project” is a proposed adaptation, not a claim made by the cited author. Dataset sample counts describe source releases, not the final number eligible after quality control.

## M01 | Medivis Studio

**Source:** Medivis (2026). Official product documentation.

**Reference:** https://www.medivis.com/studio

**Use in project:** Product inspiration: a case-centered workspace joining imaging, segmentation, multiplanar views and planning. Translate the workflow into an original desktop interface.

**Important limit / verification:** A product description is not evidence that this proposed planner is clinically effective. Do not copy branding or claim interoperability with an undocumented API.

## M02 | Frontier: Agents / Maia

**Source:** Medivis (2026). Official research-program documentation.

**Reference:** https://www.medivis.com/frontier/agents

**Use in project:** Use a persistent case state and typed, validated tools for reconstruction, planning, evaluation and export. Keep the planning engine independent of any language model.

**Important limit / verification:** Research direction, not an endorsement of this project or a cleared autonomous surgical-planning capability.

## M03 | Clinical research index

**Source:** Medivis (2026). Official publication index.

**Reference:** https://www.medivis.com/clinical-research

**Use in project:** Includes the March 2025 cranial-surgery paper on head-mounted fiber tractography. Useful for understanding the company’s interest in spatial anatomy and tract visualization.

**Important limit / verification:** Index verified; linked cranial-paper PDF retrieval failed in this search. Do not infer its clinical findings from the title.

## D01 | UCSF-PDGM collection, version 4

**Source:** The Cancer Imaging Archive (2023). Official dataset release.

**Reference:** https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=119705830
**DOI:** 10.7937/tcia.bdgf-8v37

**Use in project:** Primary patient-derived RL environment cohort. The release documents 495 unique patients, 501 exams, four-dimensional diffusion data and gradient files.

**Important limit / verification:** 156.5 GB NIfTI release; skull-stripped; repeated-patient IDs and BraTS overlap require grouping. Dedicated postoperative motor/language labels are not documented.

## D02 | The University of California San Francisco Preoperative Diffuse Glioma MRI Dataset

**Source:** Calabrese et al. (2022). Primary dataset descriptor.

**Reference:** https://pmc.ncbi.nlm.nih.gov/articles/PMC9748624/
**DOI:** 10.1148/ryai.220058

**Use in project:** Acquisition, imaging, segmentation and molecular-metadata provenance for UCSF-PDGM. Use the current release, rather than the initial paper alone, for file availability.

**Important limit / verification:** Patient/exam accounting was clarified by the release. Molecular and extent-of-resection metadata are not instrument trajectories or neurological outcome labels.

## D03 | UPENN-GBM collection

**Source:** The Cancer Imaging Archive (2022). Official dataset release.

**Reference:** https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=70225642
**DOI:** 10.7937/TCIA.709X-DN49

**Use in project:** External-site glioma evaluation and DICOM import. Provides per-subject availability and acquisition tables; 630 patients; DICOM imaging 139.4 GB and processed NIfTI 69 GB.

**Important limit / verification:** CC BY 4.0. Processed NIfTI is atlas-registered and explicitly does not align directly with original DICOM. Select baseline patients with usable diffusion, not all records.

## D04 | The University of Pennsylvania glioblastoma (UPenn-GBM) cohort: advanced MRI, clinical, genomics, & radiomics

**Source:** Bakas et al. (2022). Primary dataset descriptor.

**Reference:** https://www.nature.com/articles/s41597-022-01560-7
**DOI:** 10.1038/s41597-022-01560-7

**Use in project:** Check raw DTI availability, acquisition heterogeneity and expert-reviewed segmentation subsets before constructing an external planning benchmark.

**Important limit / verification:** Clinical metadata includes other outcomes, but does not supply a verified motor-plus-language injury model training set. Baseline and follow-up exams must remain grouped.

## D05 | UPenn-GBM diffusion MRI reconstruction / ready-to-track derivatives

**Source:** Fang-Cheng Yeh laboratory (2026). Author-laboratory data resource.

**Reference:** https://brain.labsolver.org/upenn_gbm.html

**Use in project:** Optional compute-saving source: the page lists 531 ready-to-track FIB files with QC resources. Compare a small subset against independently processed diffusion.

**Important limit / verification:** Derived from UPenn-GBM, not an independent cohort. GQI/FIB representation and transformation conventions differ from a generic NIfTI pipeline; native patient IDs must be retained.

## D06 | UTSW-Glioma collection

**Source:** The Cancer Imaging Archive (2026). Official dataset release.

**Reference:** https://www.cancerimagingarchive.net/collection/utsw-glioma/
**DOI:** 10.7937/DFAE-1B86

**Use in project:** Structural-domain-shift and tumor-segmentation testing: 625 adults, four conventional MRI contrasts; use the 362 manually refined masks for the reference-label evaluation.

**Important limit / verification:** 22.9 GB NIfTI; CC BY 4.0; release dated March 18, 2026. Do not treat all 625 masks as independently expert-refined, or assume raw diffusion is present.

## D07 | The University of Texas Southwestern Glioma Dataset

**Source:** Reddy et al. (2026). Primary dataset descriptor.

**Reference:** https://www.nature.com/articles/s41597-026-07274-4
**DOI:** 10.1038/s41597-026-07274-4

**Use in project:** Source for segmentation curation and clinical/molecular fields in UTSW-Glioma.

**Important limit / verification:** Structural external evaluation is distinct from external validation of tract-aware RL plans.

## D08 | BTC preoperative dataset, ds001226

**Source:** Aerts and colleagues / OpenNeuro (2026). Public dataset and repository.

**Reference:** https://github.com/OpenNeuroDatasets/ds001226

**Use in project:** Small independent multimodal cohort: 11 glioma patients, 14 meningioma patients and 11 controls. Includes structural, diffusion and resting-state functional MRI.

**Important limit / verification:** Use glioma cases for the main question. Verify actual per-participant released files and license at the pinned OpenNeuro snapshot; acquisition of a behavioral measure does not prove release of usable postoperative deficit labels.

## D09 | BTC postoperative dataset, ds002080

**Source:** Aerts and colleagues / OpenNeuro (2026). Public dataset and repository.

**Reference:** https://github.com/OpenNeuroDatasets/ds002080

**Use in project:** Longitudinal extension with 7 glioma, 12 meningioma and 10 control participants, useful for exploratory post-resection network analyses.

**Important limit / verification:** Overlaps the preoperative cohort. Small glioma sample; postoperative exams and controls do not create independent extra surgical patients.

## D10 | Modeling Brain Dynamics in Brain Tumor Patients Using the Virtual Brain

**Source:** Aerts et al. (2018). Primary research.

**Reference:** https://doi.org/10.1523/ENEURO.0083-18.2018
**DOI:** 10.1523/ENEURO.0083-18.2018

**Use in project:** Patient-specific network modeling inspiration and BTC provenance.

**Important limit / verification:** Network simulation is not a mechanical resection simulator or a validated voxelwise injury probability.

## D11 | Modeling brain dynamics after tumor resection using The Virtual Brain

**Source:** Aerts et al. (2020). Primary research.

**Reference:** https://doi.org/10.1016/j.neuroimage.2020.116738
**DOI:** 10.1016/j.neuroimage.2020.116738

**Use in project:** Design an optional before/after network-consistency experiment on BTC rather than inferring function solely from local distance.

**Important limit / verification:** Exploratory network changes do not identify the causal neurological outcome of arbitrary counterfactual resections.

## D12 | ReMIND: The Brain Resection Multimodal Imaging Database

**Source:** Juvekar et al. (2024). Primary dataset descriptor.

**Reference:** https://www.nature.com/articles/s41597-024-03295-z
**DOI:** 10.1038/s41597-024-03295-z

**Use in project:** Best-fit core resource for intraoperative anatomy, residual tumor and registration: 114 tumor patients, including 92 gliomas.

**Important limit / verification:** Snapshots constrain deformation/observation models, not the effect of every tool action; no continuous tool-action and force stream is supplied.

## D13 | ReMIND collection

**Source:** The Cancer Imaging Archive (2023). Official dataset release.

**Reference:** https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=157288106
**DOI:** 10.7937/3RAG-D070

**Use in project:** 44 GB DICOM collection, NRRD/DICOM-SEG labels, metadata and CC BY 4.0. Audit stage-specific segmentation availability.

**Important limit / verification:** Labels are incomplete by design: only structures needed during the actual case were segmented. A prior-operation cavity is not automatically the current-operation final cavity.

## D14 | RESECT and EASY-RESECT resources

**Source:** HealthX Laboratory (2026). Creator-laboratory dataset page.

**Reference:** https://www.healthx-lab.ca/databases.html

**Use in project:** RESECT has 23 low-grade glioma cases with preoperative MRI, intraoperative ultrasound and corresponding landmarks. Use as independent registration/shift evaluation.

**Important limit / verification:** EASY-RESECT is a processed 22-case subset, not another independent cohort. RESECT is CC BY 4.0; verify formats and transforms of the exact release.

## D15 | REtroSpective Evaluation of Cerebral Tumors (RESECT): a clinical database of pre-operative MRI and intra-operative ultrasound in low-grade glioma surgeries

**Source:** Xiao et al. (2017). Primary dataset descriptor.

**Reference:** https://archive.sigma2.no/dataset/5D6BFC33-F58D-4F56-88E8-C40AF269D6F2

**Use in project:** Creator-linked archival data entry for RESECT; informs landmark-based target registration error evaluation.

**Important limit / verification:** Does not give a ground-truth arbitrary-action tissue mechanics model or calibrated functional injury labels.

## D16 | BITE: Brain Images of Tumors for Evaluation

**Source:** Montreal Neurological Institute (2026). Creator-laboratory dataset page.

**Reference:** https://nist.mni.mcgill.ca/bite-brain-images-of-tumors-for-evaluation-database/

**Use in project:** Optional older external registration stress test: 14 tumor cases with MRI, intraoperative ultrasound and landmark resources.

**Important limit / verification:** Mixed tumor types; MINC conversion and exact reuse terms require verification. The download groups are not four independent cohorts.

## D17 | On-line database of clinical MR and ultrasound images of brain tumors

**Source:** Mercier et al. (2012). Primary dataset descriptor.

**Reference:** https://nist.mni.mcgill.ca/bite-brain-images-of-tumors-for-evaluation-database/

**Use in project:** BITE scientific provenance and independent MR/US registration evaluation. Medical Physics 39:3253–3261.

**Important limit / verification:** Bibliographic information and dataset description verified through the creators’ page; full paper was not retrieved.

## D18 | TractoInferno: A large-scale, open-source, multi-site database for machine learning dMRI tractography

**Source:** Poulin et al. (2022). Primary dataset descriptor.

**Reference:** https://www.nature.com/articles/s41597-022-01833-1
**DOI:** 10.1038/s41597-022-01833-1

**Use in project:** 284 healthy samples across six sites, 30 reference bundles; use a small subset to test diffusion/tractography robustness before tumor-domain evaluation.

**Important limit / verification:** About 350 GB for the complete release. Reference streamlines are algorithm-derived, not histological truth. Healthy anatomy is not a substitute for infiltrated glioma anatomy.

## D19 | TractSeg training/reference bundle data

**Source:** Wasserthal and colleagues (2018). Dataset record.

**Reference:** https://doi.org/10.5281/zenodo.1088277
**DOI:** 10.5281/zenodo.1088277

**Use in project:** Optional reference bundle data linked by TractSeg; useful for interface tests and tract segmentation benchmarking.

**Important limit / verification:** Not patient-specific functional ground truth. Verify the exact record version and associated code/weight/data licenses separately.

## D20 | MU-Glioma Post: A comprehensive dataset of automated MR multi-sequence segmentation and clinical features

**Source:** Mahmoud et al. (2025). Primary dataset descriptor.

**Reference:** https://www.nature.com/articles/s41597-025-06011-7
**DOI:** 10.1038/s41597-025-06011-7

**Use in project:** Optional postoperative morphology and segmentation robustness extension.

**Important limit / verification:** Not required for the first RL system. Audit the release before using any cavity labels; this plan does not assert that it contains neurological outcome labels.

## D21 | The Amsterdam IMAging and Clinical GliOma Dataset (IMAGO)

**Source:** Wamelink et al. (2026). Primary dataset descriptor; access-gated resource.

**Reference:** https://www.nature.com/articles/s41597-026-07424-8
**DOI:** 10.1038/s41597-026-07424-8

**Use in project:** Useful future external structural/longitudinal resource; included to document why a seemingly attractive dataset is not a core dependency.

**Important limit / verification:** Raw access requires an application/agreement. Open metadata or a paper is not unrestricted imaging access. Excluded under the user’s public-data-only baseline.

## F01 | Integrating direct electrical brain stimulation with the human connectome

**Source:** Coletta et al. (2024). Primary research.

**Reference:** https://academic.oup.com/brain/article/147/3/1100/7458468
**DOI:** 10.1093/brain/awad402

**Use in project:** Motor and language functional priors from 612 glioma patients and thousands of cortical/subcortical stimulation observations; publication appeared online in 2023.

**Important limit / verification:** Aggregated positive stimulation sites reflect surgical sampling and mapping tasks. They are not a uniform sample of voxels or postoperative impairment probabilities.

## F02 | Public data accompanying Integrating direct electrical brain stimulation with the human connectome

**Source:** Coletta and colleagues (2023). Public data record.

**Reference:** https://zenodo.org/records/10439149
**DOI:** 10.5281/zenodo.10439149

**Use in project:** Small 8.4 MB release of mapping data supporting F01. Good first acquisition for functional-atlas provenance and transformations.

**Important limit / verification:** Record/files verified at the listing level. Verify the file-level license before redistribution; coordinate priors are not individual functional localization.

## F03 | Integrating direct electrical stimulation with brain connectivity predicts lesion-induced language impairment and recovery

**Source:** Coletta et al. (2025). Primary research.

**Reference:** https://www.nature.com/articles/s43856-025-01121-0
**DOI:** 10.1038/s43856-025-01121-0

**Use in project:** Language is the recommended second endpoint: the study links subcortical stimulation to semantic, phonological and speech-articulation networks.

**Important limit / verification:** 297-patient mapping cohort; public maps are usable, but the associated patient-level glioma MRI/clinical outcome data are not fully open. No clinical risk-head training claim follows.

## F04 | Public language DES/connectivity maps and figure resources

**Source:** Coletta and colleagues (2025). Public data record.

**Reference:** https://zenodo.org/records/16418628
**DOI:** 10.5281/zenodo.16418628

**Use in project:** 19.1 MB release, version 2, for the language maps in F03. Keep component maps separately before building a combined display.

**Important limit / verification:** Normalized map values are not probabilities of injury. Verify license and avoid treating this as an independent patient cohort from related DES studies.

## F05 | Navigated Transcranial Magnetic Stimulation and Diffusion Tensor Imaging Tractography in Insular Glioma Surgery

**Source:** Denker et al. (2024). Primary clinical research; abstract inspected.

**Reference:** https://pubmed.ncbi.nlm.nih.gov/39508607/
**DOI:** 10.1227/ons.0000000000001414

**Use in project:** Motor-feature motivation: tract proximity, integrity measures and mapping information. Supports modeling more than geometric tumor volume.

**Important limit / verification:** Small, selected clinical cohort. Its association with a distance threshold is not a universal safe margin and must not become a voxel-to-paralysis lookup table.

## F06 | Functional Outcome after Language Mapping for Glioma Resection

**Source:** Sanai, Mirzadeh and Berger (2008). Primary clinical research.

**Reference:** https://www.nejm.org/doi/full/10.1056/NEJMoa067819
**DOI:** 10.1056/NEJMoa067819

**Use in project:** Clinical motivation for preserving function and treating mapping as consequential information, rather than an aesthetic overlay.

**Important limit / verification:** Does not provide the open imaging/action/outcome tuples required to train this planner’s clinical risk model.

## F07 | Speech mapping in awake high-grade glioma resection: subcortical tract proximity as a predictor of language outcomes

**Source:** Honeyman et al. (2026). Primary clinical research; indexed abstract.

**Reference:** https://pubmed.ncbi.nlm.nih.gov/41825076/
**DOI:** 10.3171/2025.10.JNS251541

**Use in project:** Recent language-risk feature and endpoint reference. Useful when designing a future clinical outcome schema.

**Important limit / verification:** Not verified as a publicly downloadable patient-level training cohort. Do not transfer cohort-specific numerical thresholds without their context.

## F08 | Automated eloquent cortex localization in brain tumor patients using multi-task graph neural networks

**Source:** Nandakumar et al. (2021). Primary research; indexed abstract.

**Reference:** https://www.sciencedirect.com/science/article/pii/S1361841521002486
**DOI:** 10.1016/j.media.2021.102203

**Use in project:** Optional resting-state fMRI graph model for motor/language localization. Relevant to a later BTC-based functional branch.

**Important limit / verification:** Do not assume the private clinical validation cohort is publicly released. Not needed to make the structural-plus-diffusion first version work.

## F09 | The challenge of mapping the human connectome based on diffusion tractography

**Source:** Maier-Hein et al. (2017). Primary tractography benchmark.

**Reference:** https://www.nature.com/articles/s41467-017-01285-x
**DOI:** 10.1038/s41467-017-01285-x

**Use in project:** Essential failure-mode reference: bundle reconstruction can contain false connections and omissions. Build adversarial missing-tract and false-positive-tract tests.

**Important limit / verification:** Consult the subsequent author correction as well. A tractogram is not an axonal census or a clinically calibrated function map.

## F10 | TractSeg: Fast and accurate white matter tract segmentation

**Source:** Wasserthal et al. (2018). Primary methods paper.

**Reference:** https://arxiv.org/abs/1805.07103

**Use in project:** Practical pretrained bundle segmentation starting point; reduce local training burden.

**Important limit / verification:** Tumor anatomy is a distribution shift. Verify bundle endpoints, side, continuity, and plausibility rather than trusting attractive streamline renders.

## F11 | TractSeg implementation and pretrained-model workflow

**Source:** MIC-DKFZ (2026). Official source repository.

**Reference:** https://github.com/MIC-DKFZ/TractSeg/

**Use in project:** Implementation, available bundles, processing modes and model/data links.

**Important limit / verification:** Pin a tested version and record model provenance. Do not silently substitute atlas-derived bundles for patient diffusion evidence.

## F12 | atTRACTive: Semi-automatic white matter tract segmentation using active learning

**Source:** Peretzke et al. (2023). Primary preprint.

**Reference:** https://arxiv.org/abs/2305.18905

**Use in project:** Human correction and interactive tract selection inspiration for difficult tumor cases.

**Important limit / verification:** Use as an optional workflow reference, not evidence that all public glioma tract reconstructions can be automatically certified.

## P01 | Multimodal connectivity based eloquence score computation and visualisation for computer-aided neurosurgical path planning

**Source:** Bakhshmand, Eagleson and de Ribaupierre (2017). Primary research; abstract inspected.

**Reference:** https://pubmed.ncbi.nlm.nih.gov/29184656/
**DOI:** 10.1049/htl.2017.0073

**Use in project:** Close prior art for combining gray matter and connectional costs and displaying access-path costs. Reproduce a simplified connectivity-aware baseline.

**Important limit / verification:** Demonstrates that risk maps plus route selection alone are not a novel project claim.

## P02 | Multimodal Risk-Based Path Planning for Neurosurgical Interventions

**Source:** Kunz et al. (2021). Primary research; author-university abstract.

**Reference:** https://cris.fau.de/publications/293044349/
**DOI:** 10.1115/1.4049550

**Use in project:** Close prior art for multimodal risks, modular GUI, entry-point and trajectory planning. Compare with a deterministic risk-based corridor planner.

**Important limit / verification:** Do not describe the proposed desktop tool as the first risk-map-driven neurosurgical planning interface.

## P03 | Towards Learning-based Surgical Planning of Glioma Resection via a Contrastively Constrained Siamese Neural Network

**Source:** Shan et al. (2023). Primary conference paper; publisher metadata/abstract indexed.

**Reference:** https://ieeexplore.ieee.org/document/10385556/
**DOI:** 10.1109/BIBM58861.2023.10385556

**Use in project:** Direct learned-glioma-planning prior art. Include in the closest-work table and inspect full methods before making a novelty claim.

**Important limit / verification:** Full text was not accessible in this research pass. Do not invent its architecture details, dataset openness, or a numerical comparison.

## P04 | Inverse Reinforcement Learning Intra-Operative Path Planning for Steerable Needle

**Source:** Segato et al. (2022). Primary research; abstract inspected.

**Reference:** https://pubmed.ncbi.nlm.nih.gov/34882540/
**DOI:** 10.1109/TBME.2021.3133075

**Use in project:** Direct neurosurgical RL/deformation precedent. Borrow the separation of preoperative planning, deformable simulation and intraoperative replanning.

**Important limit / verification:** Steerable needle navigation is not open glioma resection with rigid instruments. Reported simulation success does not transfer to this task.

## P05 | A new surgical path planning framework for neurosurgery

**Source:** Pehlivanoğlu (2024). Primary research; indexed record.

**Reference:** https://pubmed.ncbi.nlm.nih.gov/37773772/

**Use in project:** Additional neurosurgical trajectory-planning comparator candidate.

**Important limit / verification:** Metadata-level review only in this search. Read the accessible methods before implementing or reproducing it.

## P06 | The Open Motion Planning Library

**Source:** Kavraki Laboratory (2026). Official software documentation.

**Reference:** https://ompl.kavrakilab.org/

**Use in project:** Sampling-based configuration-space planning and benchmark infrastructure for full instrument poses.

**Important limit / verification:** OMPL is not a brain-specific collision checker, tissue model or clinical safety layer; those must be implemented and tested separately.

## R01 | Proximal Policy Optimization Algorithms

**Source:** Schulman et al. (2017). Primary methods paper.

**Reference:** https://arxiv.org/abs/1707.06347

**Use in project:** Practical initial policy-gradient baseline for masked discrete macro-actions and a small local policy.

**Important limit / verification:** PPO alone does not enforce safety constraints; compare fairly against non-RL search and add explicit action validity checking.

## R02 | Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor

**Source:** Haarnoja et al. (2018). Primary methods paper.

**Reference:** https://arxiv.org/abs/1801.01290

**Use in project:** Optional continuous-control alternative when the action representation genuinely needs continuous poses.

**Important limit / verification:** Not a reason to start with raw continuous force control. Continuous invalid actions still require constraint handling.

## R03 | Constrained Policy Optimization

**Source:** Achiam et al. (2017). Primary methods paper.

**Reference:** https://proceedings.mlr.press/v70/achiam17a.html

**Use in project:** Constrained-MDP formulation and an alternative to a simple Lagrangian baseline.

**Important limit / verification:** Theoretical guarantees rely on assumptions and approximations; no guarantee about patient safety follows from using the algorithm.

## R04 | Risk-Sensitive and Robust Decision-Making: a CVaR Optimization Approach

**Source:** Chow et al. (2015). Primary methods paper.

**Reference:** https://arxiv.org/abs/1506.02188

**Use in project:** Tail-risk objectives over coherent uncertain anatomy worlds. Keep mean and adverse-tail performance separate.

**Important limit / verification:** Tail risk is only as meaningful as the simulator’s uncertainty distribution. It is not a substitute for clinical calibration.

## R05 | Deep Reinforcement Learning in a Handful of Trials using Probabilistic Dynamics Models

**Source:** Chua et al. (2018). Primary methods paper.

**Reference:** https://arxiv.org/abs/1805.12114

**Use in project:** Probabilistic ensembles and model-predictive control inspire uncertainty-aware rollouts and a strong planning comparator.

**Important limit / verification:** Do not infer that ReMIND snapshots identify action-conditioned tool mechanics. A learned dynamics model needs actual suitable transition supervision.

## R06 | Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World

**Source:** Tobin et al. (2017). Primary methods paper.

**Reference:** https://arxiv.org/abs/1703.06907

**Use in project:** Randomize coherent anatomy/observation/geometry conditions to probe robustness.

**Important limit / verification:** Domain randomization alone does not establish sim-to-clinical validity or correct uncertainty probabilities.

## R07 | Pareto Conditioned Networks

**Source:** Reymond, Bargiacchi and Nowé (2022). Primary methods paper.

**Reference:** https://arxiv.org/abs/2204.05036

**Use in project:** Candidate single-policy representation of multiple tumor-removal versus hazard tradeoffs. Compare against a grid of independently optimized baselines.

**Important limit / verification:** The original code has GPL-3.0 licensing; inspect compatibility before reuse. A preference-conditioned PPO policy is a simpler first implementation.

## R08 | Command-Space Counterfactual Explanations for Pareto-Conditioned Reinforcement Learning

**Source:** Chulev and Baier (2026). Primary preprint, August 2026.

**Reference:** https://arxiv.org/abs/2608.14963

**Use in project:** Optional explanation feature: determine what preference change makes a different action attractive to the same policy.

**Important limit / verification:** This explains a policy’s command-to-action behavior, not the causal medical effect of an operation. Recent preprint, not clinical validation.

## S01 | LapGym: An Open Source Framework for Reinforcement Learning in Robot-Assisted Laparoscopic Surgery

**Source:** Scheikl et al. (2023). Primary framework paper.

**Reference:** https://arxiv.org/abs/2302.09606

**Use in project:** Design patterns for surgical RL environments, observations, task variation and deformable simulations.

**Important limit / verification:** Laparoscopy tasks and tissue models do not automatically represent brain resection. Reuse interfaces selectively.

## S02 | LapGym source repository

**Source:** Scheikl and colleagues (2026). Official source repository.

**Reference:** https://github.com/ScheiklP/lap_gym

**Use in project:** Implementation reference for a Gym-style surgical environment and the SOFA ecosystem.

**Important limit / verification:** Dependency compatibility and performance on the user’s hardware must be tested before choosing it as a foundation.

## S03 | SofaGym: An Open Platform for Reinforcement Learning Based on Soft Robot Simulations

**Source:** Schegg et al. / SofaDefrost (2023). Primary framework paper and implementation.

**Reference:** https://github.com/SofaDefrost/SofaGym
**DOI:** 10.1089/soro.2021.0123

**Use in project:** Optional higher-fidelity soft-body backend and RL integration patterns.

**Important limit / verification:** Not a validated glioma simulator; older pinned dependencies may be incompatible with a modern app. Use a separate worker if adopted.

## S04 | Fully automated image updating for brain shift compensation after dural opening

**Source:** Li et al. (2025). Primary research; indexed record.

**Reference:** https://pubmed.ncbi.nlm.nih.gov/40911919/
**DOI:** 10.3171/2025.4.JNS242786

**Use in project:** Image-update and brain-shift modeling reference for dynamic replanning.

**Important limit / verification:** Post-dural-opening updating is not the same validation problem as extensive tissue removal. Publication timing: online 2025, later print issue.

## S05 | Automatic Deformable MR-Ultrasound Registration for Image-Guided Neurosurgery

**Source:** Rivaz et al. (2015). Primary research; creator reference list.

**Reference:** https://nist.mni.mcgill.ca/bite-brain-images-of-tumors-for-evaluation-database/

**Use in project:** Registration baseline family; evaluate on independent landmarks and report failures, not only image similarity.

**Important limit / verification:** Referenced via the BITE creators’ bibliography; full methods not inspected in this pass.

## S06 | Simulation of Brain Resection for Cavity Segmentation Using Self-Supervised and Semi-Supervised Learning

**Source:** Pérez-García et al. (2020). Primary methods preprint.

**Reference:** https://arxiv.org/abs/2006.15693

**Use in project:** Optional synthetic-cavity augmentation reference.

**Important limit / verification:** Its epilepsy/resection imaging context is not a validated tool-contact mechanics model. Do not confuse synthetic image augmentation with surgical physics.

## I01 | NICO integrated access and resection system

**Source:** NICO Corporation (2026). Manufacturer documentation.

**Reference:** https://niconeuro.com/our-integrated-system/

**Use in project:** Distinguish access/retraction devices from tissue-removal devices. Use manufacturer geometry as a source for optional parameterized instrument models.

**Important limit / verification:** Marketing descriptions cannot calibrate injury likelihood, tissue selectivity, or a clinical superiority claim.

## I02 | CUSA Clarity product family

**Source:** Integra LifeSciences (2026). Manufacturer product listing.

**Reference:** https://products.integralife.com/cusa-clarity/category/cusa-tissue-ablation-cusa-clarity

**Use in project:** Ultrasonic aspiration as a separate instrument family, with tip/shaft/working-envelope parameters to be verified from the exact public specification.

**Important limit / verification:** The listing was indexed but direct retrieval was incomplete. Do not invent operating settings, tissue-response constants or compatibility across tips.

## I03 | Autonomous Neurosurgical Instrument Segmentation Using End-To-End Learning

**Source:** Kalavakonda et al. (2019). Primary conference paper.

**Reference:** https://openaccess.thecvf.com/content_CVPRW_2019/html/WiCV/Kalavakonda_Autonomous_Neurosurgical_Instrument_Segmentation_Using_End-To-End_Learning_CVPRW_2019_paper.html

**Use in project:** NeuroID-related instrument-perception reference for a future surgical-video branch.

**Important limit / verification:** Current downloadable dataset availability was not verified. Segmentation frames do not provide 3D pose, contact force, removal volume and patient outcomes.

## I04 | Fast instruments and tissues segmentation of micro-neurosurgical scene using high correlative non-local network

**Source:** Luo et al. (2023). Primary research; indexed abstract.

**Reference:** https://www.sciencedirect.com/science/article/pii/S0010482522012392
**DOI:** 10.1016/j.compbiomed.2022.106531

**Use in project:** NeuroSeg instrument/tissue taxonomy and an optional video-perception extension.

**Important limit / verification:** Meningioma scenes are not glioma resection action supervision. Dataset download access was not verified; not a required dependency.

## I05 | Improving the Extent of Malignant Glioma Resection by Dual Intraoperative Visualization Approach

**Source:** Eyüpoglu et al. (2012). Primary clinical research.

**Reference:** https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0044885
**DOI:** 10.1371/journal.pone.0044885

**Use in project:** Motivates modeling fluorescence and intraoperative imaging as complementary observations, not omniscient tumor truth.

**Important limit / verification:** Clinical study findings do not determine a universal fluorescence sensitivity or a mechanical tissue-removal rule.

## E01 | nnU-Net source and documentation

**Source:** MIC-DKFZ (2026). Official source repository.

**Reference:** https://github.com/MIC-DKFZ/nnUNet

**Use in project:** Default segmentation framework candidate; start with supplied dataset labels for planning research and a separately audited pretrained checkpoint for uploads.

**Important limit / verification:** nnU-Net is a framework, not a guarantee of a suitable leak-free pretrained glioma model. Pin the checkpoint, training provenance and license.

## E02 | SlicerCustomAppTemplate

**Source:** Kitware Medical (2026). Official source repository.

**Reference:** https://github.com/KitwareMedical/SlicerCustomAppTemplate

**Use in project:** Strong native desktop starting point: a custom-branded application on Slicer’s imaging and rendering foundation.

**Important limit / verification:** Prototype packaging, startup, responsiveness and platform support before committing. The app needs an original focused workflow, not a crowded default Slicer layout.

## E03 | 3D Slicer extension development documentation

**Source:** 3D Slicer contributors (2026). Official documentation.

**Reference:** https://slicer.readthedocs.io/en/latest/developer_guide/extensions.html

**Use in project:** Reference for modular imaging app integration and separating core processing from the interactive shell.

**Important limit / verification:** Keep long-running inference and RL out of the GUI event loop.

## E04 | Google Cloud free features and trial restrictions

**Source:** Google Cloud (2026). Official documentation.

**Reference:** https://docs.cloud.google.com/free/docs/free-cloud-features

**Use in project:** Verify whether the available $300 credits can be used for GPU jobs. Non-billable standard Free Trial accounts cannot add GPUs.

**Important limit / verification:** The user’s specific credit type is unknown. Upgrading billing can incur charges beyond remaining credits; never upgrade or spend automatically.

## E05 | International Journal of Computer Assisted Radiology and Surgery: aims and scope

**Source:** Springer Nature (2026). Official journal documentation.

**Reference:** https://link.springer.com/journal/11548/aims-and-scope

**Use in project:** Plausible target for a rigorous image-guided planning/simulation/software methods paper with reproducible evaluation.

**Important limit / verification:** Venue fit is not an acceptance forecast. This project alone would not justify a clinical efficacy paper.

