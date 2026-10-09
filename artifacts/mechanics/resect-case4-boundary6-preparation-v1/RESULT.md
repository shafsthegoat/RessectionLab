# RESECT Case4 boundary-6 mesh preparation

This separately versioned **source-only geometry** candidate requests one
Gmsh generation with a 6 mm boundary and 24 mm interior target size, the same
2–24 mm transition, and curvature sizing off. It preserves the retained
estimated T1-derived surface, physical frame, straight-reference tet10 format,
and all existing count, topology, Jacobian, overlap, bidirectional 2 mm
surface-distance, and 3% volume-difference gates. One attempt, no retry,
180 seconds, one numerical thread, sampled 3 GiB process-group RSS and bounded
output are fixed. These are allowances, not evidence that a mesh will pass.

The [independent source review](source-review.md) passed 176 patient-mesh
software controls, including 20 v4 controls in an isolated source export.
It approved a preparation commit only. Actual source-surface access and one
Gmsh call require a separate committed-source archive and root release.
No patient surface, B/V landmark destination, during-surgery image, mesher,
solver or policy was accessed during this preparation.

The [feasibility audit](feasibility-audit.md) preserves three prior terminal
attempts: v1 exceeded its node cap; v2 failed both 2 mm surface gates despite
acceptable total volume; v3 timed out before a volume mesh. Projecting saved
tet10 midsides onto the source would violate the current straight-reference
element, operator and geometry checks and is not substituted here.

Case4's 13 comparison landmark outcomes were already revealed in a prior
rigid/IDW study. Any later FEM comparison on this case is retrospective
**DEVELOPMENT** evidence, not untouched patient validation. An accepted mesh
would establish only geometry relative to a provisional envelope; it cannot
validate retraction forces, injury or even deformation predictions by itself.

Source: [manifest](../../../manifests/experiments/resect-case4-patient-mesh-boundary6-v4.json),
[helper](../../../scripts/mechanics_patient_mesh_boundary6.py),
[launcher](../../../scripts/mechanics_patient_mesh_boundary6_run.py), and
[tests](../../../tests/test_mechanics_patient_mesh_boundary6.py).
