# Non-patient tet10 sparse-solver preparation

The [fixed declaration](../../manifests/experiments/nonpatient-tet10-sparse-feasibility-v1.json),
[pure numerical checks](../../scripts/mechanics_nonpatient_sparse_feasibility.py),
[tests](../../tests/test_mechanics_nonpatient_sparse_feasibility.py), and
[method note](../../docs/mechanics-nonpatient-sparse-feasibility.md) define a
seven-call, three-level numerical cube experiment for the pinned FEBio sparse
backend. The material values are software gauges, not patient estimates.
Native meshing and solving remain disabled: a deterministic deck writer,
native-result parser, complete residual/backend checks and a separately
reviewed execution release are still required.

The repaired declaration rejects drift in boundary conditions, mesh recipe,
material gauge, case order, numerical tolerances and resource limits. The
[independent source review](independent-review.md) passed 23 isolated fixture
controls and twelve adversarial declaration changes. It independently checked
the affine Ogden stress/work formula and structured tet10 counts. The writer's
larger adjacent suite passed 94 tests; a separate root subset passed 54.
No patient data, native mesh, FEBio solve, material fit or physical-fidelity
result was produced.
