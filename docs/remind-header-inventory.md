# ReMIND header inventory

`resectionlab.remind_header_inventory` checks declared TRAIN objects before a
separately released header inventory. It has no acquisition or patient-run CLI.
The caller supplies the frozen series/object contract and an immutable output
location; source promotion alone does not authorize a dataset sweep.

`read_verified_header` verifies original file identity and checksums before a
bounded, allowlisted DICOM header read. Pixel decoding is excluded. It rejects
source mutation, links, excess read budgets and unsupported pixel tags.
`inspect_headers` checks patient/study/series identity and classic single-frame
MR consistency, reusing the existing physical-grid converter. Enhanced or
multiframe MR and ultrasound remain explicitly unsupported for this geometry
path. Dates and study descriptions are observations; they do not establish
availability at a surgical decision. Header success does not establish anatomical
coverage, preoperative eligibility or training admission.

NumPy, nibabel and pydicom load only when needed. Generated controls run in the
existing optional acquisition runtime; the base environment skips them cleanly
when those packages are absent. The exact source review and root validation are
in `artifacts/remind-header-inventory-integration-v1/`.
