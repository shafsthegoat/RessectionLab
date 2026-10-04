# Case4 source-only landmark partition

Completed under root's source-only measurement release, following helper
`6766a2a`, header QC `77f41c8` and access contract `cb9f71b`. The exact committed
helper, core dependency and package initializer were copied into an isolated
local source snapshot. Import origins and all source/input hashes were checked
before and after parsing. The snapshot manifest makes those files reproducible
from Git; mutable workspace code was not imported.

**The frozen rule passed without relaxation:** 19 unique source landmarks;
six B inputs selected in order **1, 14, 8, 17, 19, 7**. V contains the other
13 rows: **2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 15, 16, 18**.
The centered B singular-value ratio is **0.289608142874653**, above the fixed
`1e-6` eligibility threshold. Source extents are **12.868 × 34.622 × 30.963 mm**
in the declared world axes. This local spatial extent and algebraic rank do not
establish whole-brain coverage.

All 19 sources lie inside the original before-US full-cell bounds under the
creator-described common-world interpretation; the smallest grid margin is
41.095 voxels. This is a header-based diagnostic, not visual anatomy or frame
acceptance. The typed frame remains `world_mm_unverified`, with no frame-QC
object or forward-input object created.

Only first-triplet coordinates were numerically parsed. The helper tokenizes
the paired ASCII file but does not convert destination tokens in this path.
**All destinations, including the six future B observations, remain unopened.**
During-US and MRI-before landmark files were blocked from access. The before-US
image was read only as compressed bytes for identity verification. No image
array, registration, inference, mesh or model was created.

The full source coordinate/row audit and exact split are retained in
`partition-receipt.json`, SHA256
`de329ceb9fd964bc3b09a5f75d930a9a1bba2b235c233511cd69831bef1ff6aa`.
Partition identity is
`sha256:cc9d7776cbb4bb6ba745519e24f008c53e556bfe60d3c0509917377c9db02982`.
Full-file/destination-image identities stay audit metadata, separate from that
source-only partition identity and future solver inputs.

Exact wrapper `partition.py` SHA256:
`dbe672caa60af24a5ea37e0a8b701b60bf55922309fe3e6d1ae79988319e2c8f`.
The bounded run completed in **0.145 s**, with **37.6 MiB** child peak RSS and
**58.5 MiB** maximum sampled combined RSS, below 55 seconds/1 GiB. Monitoring
is sampled rather than continuous. Original tag/US bytes, acquisition/header
receipts, wrapper and all three frozen source files retained their hashes.
No retries, alternate patient or selection-rule changes occurred.

Next access requires its separate release: baseline MRI-before pairs and their
frame/registration QC precede any B displacement use. V destinations and the
during image remain withheld from fitting.
