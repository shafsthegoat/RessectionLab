# Optional scan preprocessing integrated

The canonical adapter now uses the pinned nnU-Net 2.5.2 preprocessing and inverse probability resampling for SHA-bound T1c/FLAIR channels and a matching support grid. Seven generated canonical tests pass in 0.154 seconds; ordinary collection skips the unavailable optional runtime without importing it. Independent algorithm and canonical source reviews pass.

The anisotropic partial-patch control preserves unknown output outside supplied coverage, including where a full-grid sentinel would have predicted positive labels. Cropping, normalization, channel order, spacing and frame refusals are checked. No patient image, model forward or checkpoint was used by these tests. The new desktop type is a contract only; it is not yet a rendered diagnostic layer.

A separate bounded runner still needs to bind actual model outputs to the accepted source records. Patient registration, anatomy, contrast identity, pretrained overlap and prediction accuracy remain unverified. All outputs are unreviewed diagnostics with planning and evaluation eligibility false. No clinical or route claim follows.
