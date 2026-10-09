# Independent source review: non-patient tet10 sparse feasibility v1

**Decision: GO for a source-only preparation commit. NO-GO for native execution, patient mechanics admission, or clinical inference.** This report reviews four source files in their final state. It does not authorize a deck writer, solver call, geometry acceptance, material estimate, RL reward, or comparison against observed patient outcomes.

## Exact reviewed bytes

| File | SHA-256 |
| --- | --- |
| `manifests/experiments/nonpatient-tet10-sparse-feasibility-v1.json` | `718088d41d9c7c8a7dd7dce79ffa887e7a7e39aa47c96403e7ac2ab9c088faaa` |
| `scripts/mechanics_nonpatient_sparse_feasibility.py` | `9ea400e1ac7f3e72dd084deb2ef1138c71c11a686cf0f9d16e8e9a1d242090c5` |
| `tests/test_mechanics_nonpatient_sparse_feasibility.py` | `3658b5dc201f372c6ddeb8acdf7909708d646b44f11acf29cafb5364ace4b168` |
| `docs/mechanics-nonpatient-sparse-feasibility.md` | `19e2b06bf0b5cac47787e651427719f8df7640581c6886cde8153005efc32be3` |

I copied those four files plus the four declared bound runtime/fixture files and two existing analytic fixture modules into `build/nonpatient-sparse-benchmark-independent/source/`, verified identical SHA-256 for the four reviewed files, and ran only fixture code there. The focused isolated test command was `.venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_mechanics_nonpatient_sparse_feasibility.py`: **23 passed in 0.12 s**. No patient data, mesher, FEBio process, or native deck was opened or run.

## Independent checks and findings

- The manifest explicitly fixes SI dimensions (m, N, Pa, J), a 10 mm **numerical** cube, `mu=1000 Pa`, `K=9666.666… Pa` as a software gauge, seven calls, one attempt per case, and resource/error gates before execution. No part is a patient property. The pure script does not launch a native tool.
- The homogeneous Ogden `alpha=2` energy, Cauchy stress, first Piola stress, and reaction convention match the pinned analytic FEBio fixture. At the declared final deformation, `J=1.029897`, total energy `5.307676290870663e-6 J`; independent central finite difference of energy gave maximum first-Piola discrepancy `2.49e-7 Pa`. Pinned six-tet fixture boundary reactions gave work `-1.0524168725192568e-5 J`, exactly the analytic signed boundary-work value at displayed precision. Discrete net force was `4.95e-18 N`; moment using **deformed** coordinates was `7.85e-20 N m`.
- Each of the six local tetrahedra has positive unit-cube orientation determinant `+1`; their volumes sum to the unit cube. Independent straight-reference edge-midpoint enumeration yielded 27, 125, 1,331, 6,859, and 19,683 unique nodes for subdivisions 1, 2, 5, 9, and 13. The declared level counts are consistent combinatorial predictions, still requiring actual incidence/Jacobian checks in a future implementation.
- An initial validator silently accepted drift in boundary coordinates, face conditions, case overrides, and the residual rule. The writer repaired this before this GO decision. Twelve independently constructed mutations of the final isolated declaration—top/bottom `z`, side-face condition, affine boundary, early-case step override, residual rule, mesh recipe, mesh acceptance, runtime binding policy, material gauge, numerical threshold, and extra section—were all rejected. The four declared bound-file hashes and runtime-profile identity matched in the isolated copy.
- The documentation now correctly notes that homogeneous affine strain cannot establish integration-point placement or distinguish quadrature rules that integrate constants identically. A future native slice must verify exact TET10G8 identity separately. The pure `affine_check` and `convergence_check` take supplied arrays and intentionally do not prove mesh completeness, boundary conditions, backend provenance, residuals, or physical fidelity. Their pass flags cannot be used alone as release gates.

## Execution gate still closed

A separately reviewed, source-bound native implementation and root-authored release must verify generated incidence, midnode sharing, element reference and deformation Jacobians, boundary tags, deck/material/quadrature, actual runtime binary/linkage/patch, source and deck hashes, solver backend choice and final residuals, complete saved states, independent force/moment balance, exact output provenance, and the frozen seven-call resource/stop rules. This benchmark can at most establish idealized solver feasibility near Case4 algebraic size. It cannot repair the rejected Case4 surface, validate tissue deformation or retraction force, or establish patient-specific harm.
