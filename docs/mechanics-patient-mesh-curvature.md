# One curvature-aware Case4 candidate — preparation only

This separately identified candidate uses the analytical control’s joint profile: Gmsh curvature sizing **24 elements per 2π** and a compatible **3 mm global minimum**, with the existing **24 mm maximum**. The Distance/Threshold field still requests 12–24 mm, Sampling100, DistMin2 mm and DistMax24 mm. The minimum is applied after that field’s setup so its legacy 12 mm floor cannot overwrite the intended 3 mm floor.

The one analytical ellipsoid pair demonstrated a joint profile effect; it did not isolate curvature from the floor change or forecast patient mesh counts. Its 64 curvature samples established limited nonzero availability, not curvature accuracy or coverage. [Saved analytical evidence](../artifacts/mechanics/gmsh-discrete-curvature-capability-v1/RESULT.md) and its [independent audit](../artifacts/mechanics/gmsh-discrete-curvature-capability-independent-review-v1/verification.json) remain unchanged.

The source is the identical saved estimated Case4 native T1 envelope, including its known inferior/cerebellar exclusions. No image extraction, source repair, smoothing, dilation, B/V landmarks, solver or learning is involved. Discrete classification, parametrization, straight tet10 generation and the physical gates remain unchanged: 2 mm bidirectional full surface bounds, 3% relative volume error, positive Jacobians, exact shared midsides, minimum mean ratio .05, topology and nonoverlap checks. Both old rejected attempts remain rejected.

| Scope | Fixed limit |
|---|---:|
| Eligible returned candidate | 64,000 nodes / 80,000 tet10 elements |
| Complete diagnostic packet | 80,000 nodes / 100,000 elements / 16 MiB |
| Whole attempt / single file | 32 MiB / 8 MiB; 32 files |
| Supervised process group | 180 s / 3 GiB / one numerical thread |
| Native generation / retries | One / zero |

The diagnostic maximum requires 11,368,704 bytes including four NPY headers and the 8,192-byte metadata reserve. The largest connectivity file is 8,000,128 bytes. Over-limit packets are omitted with their reason; arrays are never truncated. Complete available diagnostics are preserved before mesh-count eligibility is assessed. Whole-output limits are sampled by the existing supervisor and rechecked on completion; the child also has a hard per-file limit. Wall/RSS guards, rather than node counts, control actual resource use.

The higher count allowance is a new geometry-research budget, not a forecast or solver admission. A simple two-level surface-subdivision scenario would have 46,848 quadratic boundary nodes before any new interior refinement; this only explains why the old 6,000-node allowance may be unsuitable. Even an accepted mesh would require separate solver resource and formulation review.

## Explicit reuse and preserved defaults

`scripts/mechanics_patient_mesh_candidate.py` gains optional profile/generator callbacks, with its old implementations remaining the defaults. `scripts/mechanics_patient_mesh_candidate_run.py` gains an explicit immutable `ExecutionSpec` passed through release checks, worker, CLI, disk observer and supervisor. Its default v2 manifest, helper, entrypoint, 4 MiB per-file and 8 MiB whole-output limits remain unchanged. No module globals or entrypoints are patched for the new candidate.

The new helper and small launcher wrapper pin their shared source hashes. They require the exact new declaration and a ten-file committed execution closure; the old v1/v2 declarations are included unchanged. Prior analytical and diagnostic receipts are additional input bindings. Historical runs retain their original Git archives and byte identities, rather than being rerun against this refactor.

Before a future attempt, root must separately authorize the exact committed archive, fresh output path, source/runtime and context hashes. The launcher checks the original source/runtime ancestry, records all inputs before/after, charges before the sole native generation, and preserves the first failure. The preparation manifest has `execution_release.authorized=false`. No patient execution was performed during this preparation.

Owner preparation checks: **82 passed in 0.64 s**, including existing candidate/launcher controls and new curvature-specific cases. Independent preparation review is pending. These are mocked or analytic software controls; they make no patient fidelity or runtime claim.
