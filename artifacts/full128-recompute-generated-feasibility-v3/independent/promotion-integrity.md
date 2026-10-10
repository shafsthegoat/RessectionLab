# Independent compact-promotion integrity review

**Decision: GO for metadata-only promotion.** This review did not open the generated logit array, checkpoint, model source, or patient data, and did not run a model.

The package index SHA-256 is `55949cd79b76575d9fe57d7f608d1270260f3974c33c4449693ff60e90769dd9`; `RESULT.md` SHA-256 is `1e30e2acd2b3de0a321e70bbe74d46173752c41f9108ce525475b15ef4cdb96a`. All 17 indexed metadata/log/review files (224,856 bytes in total) exist as regular files. Each size and SHA-256 matches the index, and each copy is byte-identical to its declared original saved receipt or review. There are no missing or unindexed files. No `.npy`, checkpoint, patient image/label, or executable `.py` file is packaged.

The index's 16 runner/test source hashes, model metadata/checkpoint hashes, and generated-input hash exactly match the copied V3 contract. Its output NPY digest matches both the copied worker result and independent saved-array metadata. The copied supervision says accepted, exit 0, peak sampled process-group RSS 2,577,743,872 bytes under the 3 GiB cap, with no watchdog/post-guard stop. The finalization receipt records no residual group/detached processes or cleanup errors. Independent array metadata records FP32 shape `[1,3,128,128,128]`, zero nonfinite values, matching raw/NPY hashes, a 57,884,672-byte auditor RSS peak, and `rlimit_as_enforced=false`.

The result correctly keeps the V2 project-shell attribution uncertain and separates that negative from V3's successful generated-input run. It also distinguishes prior 64³ exact parity from the **absence of an accepted native full128 comparator**. The evidence establishes one bounded generated full128 feasibility run with finite output, not patient accuracy, full128 numerical parity, route quality, or clinical utility. The hard address-space limit was unavailable on macOS; the report does not claim otherwise.

Minor wording suggestion, not a promotion blocker: replace “native tiled control” for the 64³ comparison with “tiled control with native full-map InstanceNorm” to avoid implying an untiled native full-network arm. This does not change the stated 64³-only claim.
