# Case4 rejected-mesh fidelity-witness preparation

The [fixed declaration](../../../manifests/experiments/resect-case4-boundary6-fidelity-witness-v1.json),
[source](../../../scripts/mechanics_patient_mesh_boundary6_witness.py),
[controls](../../../tests/test_mechanics_patient_mesh_boundary6_witness.py), and
[method note](../../../docs/resect-case4-boundary6-fidelity-witness.md) prepare
one saved-geometry diagnostic of the rejected Case4 v4 mesh. They do not
authorize execution. No patient array, mesh generation, solver, landmark,
during-US image or policy was accessed for this preparation.

The diagnostic would replay the original bidirectional 1 mm lattice and 2 mm
surface gate before measuring where discrepancies occur. A fixed 30 mm cube
test classifies whether exceeding witness-face area is spatially concentrated
under an explicit whole-face proxy. The candidate remains rejected regardless
of that classification. A future run requires an exact committed-source
archive, separate root release, bounded private output and independent
saved-result review.

The final adjacent patient-mesh suite passed 186 tests. The
[independent source review](independent-review.md) found and repaired two
supervision failure paths, then passed ten isolated synthetic controls and
100 randomized localization checks. It approved this non-executable
preparation only. Case4 remains retrospective DEVELOPMENT, with anatomical
source suitability and sparse-solver gates still closed.
