This dataset was obtained from the Lausanne University Hospital (CHUV).
Accession #: ds-aneurysms-lausanne
Description: This dataset set is composed of 284 TOF-MRA subjects, of which 127 are healthy controls and 157 are patients with brain aneurysm(s).

Please cite the following reference if you use this dataset:

Di Noto, et al. "Towards Automated Brain Aneurysm Detection in TOF-MRA: Open Data, Weak Labels, and Anatomical Knowledge", Neuroinformatics, 2022, https://doi.org/10.1007/s12021-022-09597-0

Here is an overview of the dataset organization: every original folder (i.e. non-derivative) contains the TOF-MRA and the T1w volume of the subjects. Instead, the "derivatives" folder contains 3 sub-folders:

1) "manual_masks": for controls, it only contains the skull-stripped TOF-MRA volume; for patients, it contains the skull-stripped volume and the binary manual mask(s) of the aneurysm(s).
2) "N4_bias_field_corrected": for every subject, it contains the N4-bias-field-corrected TOF-MRA volume and the corresponding N4-bias-field-corrected skull-stripped volume.
3) "registrations": it contains 3 sub-folders
	3.1) "reg_metrics": for every subject/session, it contains the registration quality metrics. We performed 2 registrations: MNI_2_T1w, and T1w_2_TOF. For each of these registrations, we save the ANTsNeighborhoodCorrelation and MattesMutualInformation of the ANTs package. The metrics can be used to check which are the subjects for which the registration was not accurate.
	3.2) "reg_params": for every subject/session, it contains the parameters used in the registration (i.e., warp fields and .mat files).
	3.3) "vesselMNI_2_angioTOF": for every subject/session, it contains the probabilistic vessel atlas (Mouches et al., 2019) co-registered to TOF-MRA subject space.
