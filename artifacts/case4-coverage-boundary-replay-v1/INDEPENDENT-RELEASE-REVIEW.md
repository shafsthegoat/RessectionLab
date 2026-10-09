# Independent release review: Case4 read-only boundary replay

**Prospective decision: GO for one root-owned, supervised diagnostic replay only.** No Case4 array or annotation was read by this reviewer. The exact replay below compares saved coverage resampling using ANTs nearest-neighbor and SciPy `constant`/`grid-constant`; it writes one small JSON comparison. It does not write a support map, infer tumor, optimize registration, plan a route, or qualify anatomy. The original attempt-01 discretization hold remains preserved.

## Exact frozen bytes

- `replay-release.json` SHA-256 `481897b822a095adc2117f4660aa462a86684ae3e152f105357047f77726a961`.
- `supervise_replay.py` SHA-256 `0c5590dcde4296195d1ca644f2c9584f3144f9b9f97db61724733750113a196c`.
- `replay.py` SHA-256 `85fbf2928d54cd7713a770c0548ec144cc1abf0bcb9b5d8b57a3c29015179767`.
- Generated test SHA-256 `de39381f2c8a119c3750f4d10be06170241043e3187008dde0e2b92ed0783702`; supervisor mock test SHA-256 `5633be6fda476256f3b362188c29e722cab7219a26629ae4fb3f9729be7349de`.

The release binds the original negative hold, preparation receipt, source manifest, helper sources, pinned runtime lock, sampler, and reviewed monitor helpers by exact file paths and matching hashes. The worker independently rehashes every saved preparation artifact before reading the two coverage volumes. It uses the saved affine only, with no new registration. The copied V2 monitor preserves 30-second stable-host preflight, 0.2-second runtime sampling, 3 GiB sampled process-group RSS, 180-second wall, 256 MiB output, one session, no retry, and identity-bound per-PID cleanup. It strips ambient `PYTHON*` variables and fixes ITK thread count at two. `killpg` is absent. All attempt output, supervision receipt, log, and provisional negative files were absent at review.

## Verification

- Independently ran **2/2** generated-only controls in the pinned scan runtime: oblique NIfTI with nonidentity ITK translation, and exact half-voxel/beyond-boundary cases. `grid-constant` matched ANTs voxelwise in those generated cases. A prior overly exact old-mismatch assertion failure is retained in `GENERATED-EDGE-RESULT.md`.
- Independently ran **14/14** source-only supervisor mocks: release substitution refusals, ambient Python isolation, identity/cleanup failures, durable provisional negative, and result-status admission. Positive supervisor status now requires the inner diagnostic JSON to reproduce both old mismatch counts **and** show zero new mismatches. Incomplete parity and unreproduced old counts are distinct diagnostic holds; a mere worker exit 0 cannot become positive parity.
- CLI without `--execute` raised `ValueError: explicit reviewed one-shot execution required` and created no attempt artifacts.

The only reviewed invocation, from repository root and during a root-owned quiet compute window, is:

```sh
python3 -B build/scan-target-estimator-research/case4-discretization-diagnosis-v1/supervise_replay.py --execute --release-sha256 481897b822a095adc2117f4660aa462a86684ae3e152f105357047f77726a961
```

Afterward, independently verify the supervisor receipt, comparison JSON hash/status, exact old mismatch reproduction, new per-modality mismatch counts, output inventory, and no surviving process. A process-complete replay is a numerical diagnosis only. Even exact parity does not accept the estimated mask, support map, tumor prediction, or any planning input.
