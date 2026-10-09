# BTC PAT16/PAT20 diffusion intake

All 16 declared AP/PA diffusion images and sidecars were acquired: **94,311,092 bytes**. Verified TLS, public source revision, publisher image MD5 and frozen text SHA-256 checks passed. Root independently rehashed all 16 payloads and compared every checksum with the declaration and acquisition record; role and queue files still match HEAD.

PAT16 and PAT20 retain their canonical **TRAIN** assignments within the existing development cohort. These add modalities for two existing patients; they are not new patient counts. The intake receipt's development role describes outer-use restrictions and does not change TRAIN/SELECT assignments.

Payloads remain `byte_verified_unreviewed`. Image and anatomical QC, diffusion reconstruction and downstream admission have not run. No policy update, withheld evaluation, or physical/clinical result follows. Data remain local and excluded from Git.

The [declaration](declaration.json) preserves exact public URLs and source checksums; the [summary](summary.json) preserves local hashes and transport facts. The acquisition agent's post-transfer check is supplemented by the distinct [root fixity audit](ROOT_FIXITY.json). The original transfer used an ignored adapter around the existing guarded downloader; this artifact does not claim a newly shipped acquisition CLI.
