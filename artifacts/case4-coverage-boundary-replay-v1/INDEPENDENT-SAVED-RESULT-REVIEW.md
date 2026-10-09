# Independent review: Case4 corrected support-map v2

**Prospective GO for one root-owned, supervised support-map preparation attempt only.** The numerical boundary defect in the preserved failed v1 attempt is explained by a read-only replay. This release still creates only an unreviewed diagnostic scan-support bitfield; it does not qualify anatomy, infer tumor, admit planning, or provide clinical evidence. No patient arrays or annotations were opened by this reviewer.

## Saved read-only replay audit

The comparison JSON SHA-256 is `e989c562e5296093d711f6526da61f2809dc90f1a69e14495c83dcae7e852f58`; its supervisor receipt SHA-256 is `c05ee4b946b38654725c1816e86c920b29245e5023480bf0f60b74d46a2342e9`. The result reproduces the original v1 mismatch counts exactly: **49,293 T1** and **24,923 FLAIR** voxels under SciPy `constant`. Under `grid-constant`, both modalities have **zero** ANTs voxel mismatches, and all old disagreements are reported as corrected boundary samples. Both source coverage hashes match the saved preparation inventory. The replay used Case4 **DEVELOPMENT**, the saved transform, and no new registration or model/planner; its receipt records worker exit 0, verified cleanup, no stop/errors, and no anatomy assessment. This establishes parity for these saved coverage inputs, not anatomical validity.

The original v1 hold remains at SHA-256 `3c64662024efccbfc990922460294d2c9cb6cfc830ad562d9d9f9174268856f1`; its failed v1 support map and success result remain absent. The corrected v2 attempt outputs, supervision receipt, worker log, and provisional negative were all absent at review.

## Frozen corrected-map release

- Release SHA-256 `9bf59ec526f0ba7477dcbd023d1fa30be7b1f7ffdcf69097db65560e2bdc27fa`.
- Supervisor SHA-256 `13690df551128277a95cf90a7097f28f9b455ae8302fa05f23475c80e88baac8`.
- Generator SHA-256 `61c1604283a94ac9cb0dc683386e644880d7082de5f762f7222d0eb465b9f672`; helper SHA-256 `3c7a9bbce5042cc9ca0ab5c51086045ddebc4590ccbc508160681bc4111f335a`.

The generator differs from the preserved failed v1 generator in **one line only**: SciPy `mode="constant"` changes to `mode="grid-constant"`. The same ANTs nearest-neighbor replay and exact voxelwise parity gate remain; source/shape/frame checks, original-scan-FOV bit meanings, joint-FOV preparation check, and unreviewed/admission-false result fields remain unchanged. The copied V2 monitor changes only fixed paths, run/schema labels, generator hash, and an exact preflight binding to the successful read-only replay result. Its resource caps and identity-bound cleanup are unchanged: 30-second stable-host preflight, 3 GiB sampled process-group RSS, 180-second wall, 256 MiB output, 0.2-second sampling, no automatic retry. All exact source/helper/runtime/manifest/preparation/replay-result hashes checked against the release.

Independent source-only verification: **13/13** supervisor mocks pass, including substitution and cleanup denials. The CLI without `--execute` refused before creating attempt artifacts. The previously reviewed generated oblique/nonidentity/half-voxel controls passed **2/2**. No patient run was made here.

The only approved invocation, from repository root during a root-owned quiet compute window, is:

```sh
python3 -B build/scan-target-estimator-research/case4-support-grid-constant-v2/support_map_supervise.py --execute --release-sha256 9bf59ec526f0ba7477dcbd023d1fa30be7b1f7ffdcf69097db65560e2bdc27fa
```

After any terminal run, independently review the supervisor receipt, map/result or new hold, hashes, bit counts, affine, FOV and mask omissions, resource/cleanup evidence, and no model/planning admission. Even a numerically successful support map remains unreviewed until that separate anatomical phase.
