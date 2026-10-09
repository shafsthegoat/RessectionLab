# Independent tiled vascular contact feasibility review

Decision: **GO for the frozen source candidate and tiny generated controls.** The profile worker is suitable for a separately released, externally supervised root benchmark with the conditions below. **The realistic-size benchmark has not run and remains unreleased by this review.** No patient/model/native-solver/network work or tracked edits were performed here.

## Frozen identities

All candidate paths below are under `build/private-vascular-streaming-preparation-v1/`.

| File | SHA-256 |
| --- | --- |
| `streaming_contact.py` | `c85cca298b20190be865e23cb3d0bde3dc142e69f182e93e65c008895cec3d75` |
| `test_generated.py` | `de7d1fe8f26344dd516aea18e90e28c22ff8dc71399226f8b7ff896c0e7501c4` |
| `RESULT.txt` | `5c2bcac4b5adc2fd197dc31c9eaf04aed05a23f48a38c407e5cc3cb60170fbea` |
| `profile_generated.py` | `1b149489ee2dfb0338c08518fba5835b689de3cace7f696796b08b42e27b4eda` |
| `repository-source-pins.json` | `264f0e811dbb117357de979c2748ac687b7108e56e1f183d8c6582f4b4a1861d` |
| `test_profile_boundary.py` | `2d1677c0e11bae3be398e169a6c521f23a401e24f9c66beb8a46737d897d3db5` |

The kernel, author tests and RESULT were unchanged throughout review. Profile-worker corrections supersede its earlier unexecuted `247d5fc4...` and `5c5ad266...` versions. Only the final worker identity above is approved by this report.

## Independent result

The kernel run combined 28 author controls and 12 independent controls: **40 passed in 1.29 seconds**, exit 0, harness elapsed 1.449674041941762 seconds. All 81 source/test snapshot entries were unchanged. Both `source-before.json` and `source-after.json` have SHA-256 `9ee7f3f99db727ee4b73b301c83d2d1081d5ed7d07f1a7a632d9f72de7ab1da1`.

The separate worker-boundary run combined six author controls and four independent controls: **10 passed in 0.05 seconds**, exit 0, harness elapsed 0.20653712493367493 seconds. All 82 snapshot entries were unchanged. Both `boundary-source-before.json` and `boundary-source-after.json` have SHA-256 `6f70453156872478dd979d88b292cc363346bec2a13cc960f53dff075c82072f`. These tests check release refusal, bounded nonblocking regular-file reads, file growth beyond a stale size check, symlinks, inventory traversal/source mismatches, stale cache configuration, shadow import locations and loader-origin mismatch. No profile evaluation occurs in these tests; the check-only branch remains stdlib/source-only.

Exact commands, environment settings and results are preserved in the two runners, receipts and outputs. Numerical-library thread counts were one, Python bytecode writing and pytest plugin autoload/cache were disabled, and scratch paths stayed under this ignored review directory.

| Independent evidence | SHA-256 |
| --- | --- |
| `test_independent.py` | `15fc822c9b8c90222bc2d1c3086d01a568f80565bc45b766c33278e986f94065` |
| `run_review.py` | `0f833f272519767b3cda2ccdde046d0b32391ba86b3f992208b2dc67d46694ca` |
| `run-receipt.json` | `e9b24d4231d3353fe56f70c3e2ebc9c086ccd8d05790a0d269f93f5a27b24f3d` |
| `pytest-output.txt` | `8067552e48bd8ea9cb8ccdf6b143dff72eb1c3a192c61b4d7e6ff82f03e7b5cb` |
| `test_profile_independent.py` | `dc45954d3d5c104d5c14edb9020bd0b3bec60b545f64d23d9e35f6fd571fe1ac` |
| `run_boundary_review.py` | `8caa5304290f65184be53d4329cd703625c7c21c64a9a88449b07ed702c6c4b6` |
| `boundary-run-receipt.json` | `9ca1fb10781dd1c80a3d59ec95aaf46557d532d7871eba10f4bf9ce43b75172b` |
| `boundary-pytest-output.txt` | `e8b981c8b9010bb633ad4e4d591190d80aad92cbe20d00915232255ce3e7fd94` |

## Geometry and accounting findings

No remaining blocker was found. Independent full-grid scalar-oracle comparisons cover uneven tile boundaries, multiple tile sizes, rotated anisotropic/reflected physical frames, a positive-radius near-tangent contact across a tile face, duplicate capsules and actions, outside-FOV no-contact unknowns, zero budgets and cancellation. The largest independently generated oracle grid has 210 cells; author geometry fixtures have at most 512 cells. Merely constructing a 256×256×192 grid descriptor in an author validation test does not query that grid or allocate its volume.

The disjoint tile partition makes accumulation independent of tile boundaries: each reference cell belongs to one tile, and local shaft/tip/action masks are unioned before counting. Full-FOV exposure is not cropped by nominal estimated support. Coarse pruning uses stop-exclusive physical boundary boxes with inclusive padding only at that stage; final cell decisions retain the existing `1e-10 mm²` squared-distance tolerance. Tiny oracle equivalence supports these cases, not a formal floating-point proof for every admitted input.

The prefreeze review corrected three accounting gaps. Refused charges now retain current/requested/projected/limit values plus the actual budget, coarse and cell geometry-batch maxima are both recorded, and successful output binds the passed budget. Work counters represent charged/reserved work and may exceed completed work after cancellation. Budget failures expose no completed or partial scientific outcome.

Generated procedural patterns and explicit coverage are the only reference provider. Whole-strategy counts do not multiply when strokes repeat. No-hit partial coverage or outside-FOV motion stays unknown; positive encounters can coexist with unknown coverage. Outputs remain discrete mask-cell encounters and cell-volume surrogates, with biological vessel-free and clinical injury fields null.

## Profile-worker conditions

The final worker addresses the release and provenance findings. It reads a regular release file once with `O_NOFOLLOW`, `O_NONBLOCK`, and a bounded read; the receipt hashes exactly the validated bytes. It verifies the 76-file repository source inventory before and after evaluation, checks actual loaded `resectionlab` `__file__` and `__spec__.origin` paths and bytes before/after, and requires `-B` plus a fresh absent output-local `-X pycache_prefix` to avoid stale project bytecode. Later lazy imports are included in the final origin check. A source/cache mismatch cannot retain a successful profile status.

The worker still needs a root-issued exact release and fresh ignored output directory, with a separately reviewed external supervisor controlling wall time, RSS, outputs, threads, stop/reap behavior and terminal evidence. Its own 20-second checks are cooperative, and it supplies no hard RSS containment. Import/setup failure, process death or final write failure can leave an incomplete attempt; the external supervisor must preserve that as a negative and must not infer completion from a reserved attempt. Scientific-runtime binary closure is explicitly not authenticated by this feasibility worker. No automatic retry or limit increase is approved here.

The full256 run must bind these exact final hashes and the source inventory. Check-only output and disabled templates do not release it. Source changes, broader inputs or integration into the live evaluator require review.

This candidate is a contact-kernel feasibility adapter. It does not seal or replay a strategy, certify full-tool kinematics, load patient/reference images, evaluate removed-cell overlap, or change existing admission and congruence gates. Future integration must preserve those duties. Tiny tests do not establish realistic-size performance; that measurement remains pending the separate root benchmark.
