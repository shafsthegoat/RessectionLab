# Fixed SRI-24 atlas acquisition

October 9, 2026. Acquired the CC-BY-4.0 `brats_sri24.nii` asset from [immutable Zenodo release 15927391, v2.0.0](https://zenodo.org/records/15927391), using verified TLS, HTTPS-only redirects and a resumable transfer. Source content: `https://zenodo.org/api/records/15927391/files/brats_sri24.nii/content`. The BrainLes concept URL `15236131` resolves to this release; the inference workflow must use the fixed local asset rather than a mutable automatic fetch.

- Original bytes: **17,856,352**.
- Publisher MD5, independently reproduced: `605c659c2976f055cf3a43a88a7ad6f3`.
- Local SHA-256, independently reproduced: `50ceb5c1f1b727bf578110bea92d80ed801cf6d3207e08b619db548a27023b1b`.
- Ignored original and acquisition metadata/receipt: `data/models/sri24-v2.0.0/`.
- Independent header check: `build/scan-target-estimator-research/root-atlas-verification.json`, SHA-256 `01c6c8b1ddfb6d5cf43384d280c76299329148deae84c518601f6dad769a94f4`.

The actual NIfTI shape is **(240, 240, 155, 1)**, with a singleton fourth axis. Spatial spacing is 1 mm, axis codes LPS, and the finite affine is `diag(-1,-1,1,1)` with a 239 mm y translation; spatial determinant is +1. Any conversion to three dimensions must explicitly preserve this physical frame and original bytes. This header check does not validate anatomical registration.

The reference is the modified, non-skull-stripped `brats_sri24.nii` selected by the pinned BrainLes preprocessing interface. It is neither a new patient nor withheld patient-specific truth. No patient registration, target inference, planning admission or transfer evaluation occurred. A separate isolated nnU-Net/ANTs runtime is being prepared; code rights, exact model weights, preprocessing, inverse mapping and scan-only QC remain required before the development pilot.
