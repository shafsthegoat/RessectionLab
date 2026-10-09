# BTC training modalities: byte acquisition complete

The continuation acquired another **36 files, 371,219,704 bytes**: AP/PA diffusion images and sidecars for PAT05/PAT22/PAT25, plus raw resting-state BOLD and metadata for all six existing TRAIN patients. Exact public source identities and checksums were frozen before image transfer; metadata fetched during binding was reused, not downloaded again. Root independently rehashed all 36 files against source and completion bindings, without decoding images.

Together with the earlier PAT16/PAT20 diffusion intake and existing PAT28 diffusion, all **72 catalogued raw preoperative anatomy/diffusion/resting-state paths** for the six BTC TRAIN patients are present. This is directory/source coverage, not a new full rehash of every historical original. The latest source snapshot is OpenNeuro ds001226 v5.0.1, CC0. Source URLs and hashes are preserved in [ROOT_FIXITY.json](ROOT_FIXITY.json); the [coverage inventory](raw-source-coverage.json) distinguishes older and new files.

The complete acquisition continuation totals **452 new source files, 878,317,000 bytes**, including the separately documented ReMIND001 extension. This is a cumulative total, not additional bytes on top of those prior results. No transfer remains live at the recorded completion time. Approximately 647.69 GiB remains free, above the 100 GiB reserve. [Consolidated receipt](continuation-summary.json).

Patient splits and existing role files are unchanged. All new files remain `byte_verified_unreviewed`; no anatomical, diffusion, functional or registration QC has been performed. Resting-state images are not reviewed functional maps. No new independent patients, direct surgical trajectories, optimizer updates, transfer results or training admissions are counted. Data remain local and excluded from Git.

Next acquisition work checks full-cohort eligibility and patient-role declarations for additional sources; absent new roles alone is a preparation task, not a requirement for new human permission. Anatomical QC is a separate later phase.
