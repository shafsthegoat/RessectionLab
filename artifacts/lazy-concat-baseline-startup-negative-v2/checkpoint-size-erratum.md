# Erratum: extracted GlioMODA checkpoint size

The preserved independent v1 matched64 prospective review, SHA-256 `93f858205b1e4ac0664b765b2e72a96d0d04219e2403a4f5a069769a11645190`, mistakenly calls the checkpoint a “3.3 GB checkpoint.” That figure belongs to the separately published `weights.zip` archive, not the extracted file used by the worker.

The pinned local `data/models/gliomoda-v1.0.2/t1c-t2f-pinned-v1/fold_all/checkpoint_final.pth` is **250,184,062 bytes** by filesystem metadata. Its frozen SHA-256 in the experiment contract is `0e29f882310fe8cb076d6cadb982067ef53c6a32231f40ae17d9c173aa4307b3`. This erratum did not read checkpoint contents or run the model; the worker rehashes the file before and after each controlled run.

The original review bytes are intentionally unchanged so the prior failure-evidence package remains reproducible. This correction changes no experiment criterion or outcome. The still-prospective v2 review states the size correctly.
