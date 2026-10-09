# Independent post-run review: Case4 support-map attempt 01

**Decision: HOLD.** The one-shot worker correctly stopped on an independent nearest-neighbor discretization disagreement. It did not save a support map or a success result. This is a numerical support-map preparation failure; it provides no anatomical, inference, planning, or clinical result.

## Frozen evidence

- V2 release SHA-256: `36ba92b204746131b86cc56e3f13d46db3a6e3a1d664c486e6f681dab40e5759`; V2 supervisor SHA-256: `5c768868ec642bb6807363698174aa64dc82c1723717d1ca2111be464609873c`; generator SHA-256: `82e205d2c47b46d0d65e15055a73ec15ba65f0010352fb236597623ccf2ddd6d`.
- Supervisor receipt `support-map-supervision-result.json` SHA-256: `01870da186e6c37a707cea60a304d4febcfef9a5d2ab72a704d2511cad0f4756`. It records `discretization_hold`, worker exit 1, no stop action, no cleanup/finalization/receipt error, completed cleanup verification, no unresolved owned PIDs, and a nonliving worker. Its output inventory binds the hold hash and records null map/result hashes. Peak sampled process-group RSS was 1,527,513,088 bytes, below the frozen 3 GiB stop; the worker was not stopped for resources.
- Worker log `support-map-worker.log` SHA-256: `b37c439f0ee1d54856c5d1bfb5da7e755064d53b84ab7d14a3858bd3bd9e688c`. It shows the deliberate `ValueError` from `make_support_map.py:152` after voxelwise parity failed.
- Hold receipt `../case4-support-contract-v1/discretization-hold.json` SHA-256: `3c64662024efccbfc990922460294d2c9cb6cfc830ad562d9d9f9174268856f1`. It binds the original preparation and transform hashes and reports **49,293 T1** and **24,923 FLAIR** voxel mismatches between ANTs and independent SciPy reconstruction. Direct filesystem checks confirm `support-bits-atlas.nii.gz` and `support-map-result.json` are absent.

## Generated-only diagnostic

No patient arrays were reopened. In a fresh 4×4×4 all-ones generated NIfTI with an identity saved ITK transform, atlas shifts of ±0.25 and ±0.49 voxel produce exactly one border-slice (16-voxel) mismatch: ANTs nearest-neighbor includes the source border while SciPy `mode="constant"` excludes it. In the same generated control, SciPy `mode="grid-constant"`, order 0, includes these half-voxel-border samples and drops the border at +0.5 voxel. This demonstrates a plausible boundary-semantics cause for the observed failure, but does not prove every real-scan mismatch has that cause.

The writer should verify the replacement rule with generated oblique and nonidentity ITK affine controls, including half-voxel ties, then prepare a **new frozen one-shot release** with the original strict voxelwise parity and source bindings. Preserve this failed attempt and its hashes. Any patient-array replay needs a separate root-owned compute window and independent release review. No tolerance, silent clipping, or promotion of the hold to success is justified.
