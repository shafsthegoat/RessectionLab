# Fixed reference-volume observation operator

This software slice implements the 5 mm tent functional in
[the conditional-displacement specification](resect-conditional-displacement-poc.md).
It reads no patient files or displacement destinations and runs no mechanics
solver. Existing tet10/MPC solver controls used nodal averages; those controls
do not establish this volume operator or its eventual patient performance.

`scripts/mechanics_patient_observation.py` accepts finite `nodes_m[N,3]`,
zero-based `tet10_indices[E,10]` in FEBio order (vertices, then edges
12/23/31/14/24/34), and at most six source centers in the same reference frame.
The caller must bind the anatomical frame and perform the mm-to-m conversion
once. Midsides must be linear reference midpoints within 1e-12 m. Positive
Jacobians, complete node use and connected node topology are checked here;
conforming, nonoverlapping, closed anatomical domain QC remains the mesh
producer's responsibility. No kernel relocation, mesh refinement driven by
validation landmarks, or nearest-node substitution is performed.

`integrate_tent_observations` returns dimensionless rows over explicit sorted
source node indices. The numerator, denominator and first moment use the same
physical-volume quadrature. Quadratic negative nodal coefficients are retained.
`apply` accepts a complete finite source-nodal field; `prepare_elimination`
selects source-only parents and child coefficients without destinations.
Input arrays are copied, returned arrays have immutable byte storage, and
operator content hashes, including nested support/convergence reports, are
checked before application or elimination. Editing a diagnostic therefore
invalidates the object rather than silently changing its scientific evidence.

## Frozen numerical policy

The kernel is `max(0,1-|X-p|/0.005)` in meters. Integration is restricted to
candidate tetrahedra, with an exact-in-real-arithmetic closest-point test on
linear tetrahedra. Each retained tetrahedron is subdivided into eight equal
subcells. Complete local levels use positive degree-two cubature. A level is
accepted only after agreement with its preceding level **and** with separate
positive degree-five cubature on the preceding subcells. The latter rule is
independently checked against every barycentric monomial through degree five.
The two comparisons must each satisfy:

- weighted mass difference ≤ 1e-3 of the refined mass;
- sum of per-original-element normalized coefficient differences ≤ 0.002;
- kernel-centroid difference ≤ 5 micrometers.

These are empirical integration-convergence tests, **not certified error
bounds**. A possible support missed by all quadrature points keeps refining or
abstains; zero samples are never taken as proof of empty support. An accepted
weighted mass must be at least 1e-4 of the analytic full-tent mass `pi*r^3/3`.
No tolerance or cap depends on landmark destinations.

Caps can only be lowered: 200,000 nodes, 250,000 elements, six rows, 2,048
AABB candidates per row, 262,144 quadrature points per row, eight subdivisions,
and 4,096 active source nodes across all rows. The AABB cap is deliberately
conservative and can reject a case before exact intersection filtering. All
quadrature samples, including cross-checks and discarded previous levels,
count toward the cap. Failure returns no partial operator.

Each row records unclamped weighted support fraction, unweighted support-volume
estimate, inside/possible-subcell volume interval, weighted centroid and first
moment, and the complete refinement ledger. The geometric interval is separate
from integration discrepancies and assumes valid mesh topology and floating
point geometric tests. Its potentially wide extent is retained. The
`truncated_support_estimate` flag uses weighted mass below 0.999 of full mass;
a false flag is not proof of full support. Small fractions above one due to
quadrature are retained and are not probabilities.

## Elimination and limitations

Rows must be linearly independent at relative singular threshold 1e-8. Their
restriction to translations and rotations about the mean weighted centroid
must have rank six, with rotation scale 0.005 m. Disconnected domains are
refused separately. Deterministic twice-projected QR column selection resolves
exact ties by lower source-node index; the pivot block must have condition
number ≤ 1e8. Library solves produce all simultaneous child coefficients, with
original-row residual ≤ 1e-10 and no parent appearing as a child. At most
20,000 scalar child coefficients (60,000 over three displacement components)
are allowed. No row is dropped and no anchor is added.

The operator is a declared smoothing approximation to point-picked motion,
not a measured volume-average displacement. These controls do not establish
patient mesh resolution, nonlinear solvability, mechanics validity or prediction
accuracy. All public patient construction and solves remain separately gated.

## Analytic control evidence

The 10 mm analytical cube uses the fixed 5 mm radius. Full-sphere mass error
was 1.82e-5 relative, using 106,014 samples. The half-space control had 7.93e-5
relative mass error and a centroid error below 0.5 micrometers against the exact
inward shift `3*r/10`, using 54,445 samples. Tests also compare a quadratic field
against independent Cartesian tensor integration, test constant/affine fields,
negative coefficients, rigid coordinate reexpression, support failures,
source/rank/term caps and original-row elimination.

The initial absolute-local-discrepancy design and subsequent two-h-level designs
exhausted the same sample cap and correctly abstained. These development
negatives are retained in `artifacts/mechanics-patient-observation-v1`; no patient
outcome or solver result guided the numerical rule. First owner test failures
were a differing floating point barycentric reconstruction and an arbitrary
quadratic-field check tighter than the declared policy; their original output
is retained, and corrected tests explicitly state the shared reference
coordinates and 1e-3 relative comparison policy. Independent review passed seven additional controls, including closed-form
octant integrals, nonzero constraint offsets and both reproduced report-integrity
negatives after repair. The combined gate passed 36 checks; see the
[review receipt](../artifacts/mechanics-patient-observation-independent-review-v1/verification.json).
The initial two report-integrity failures and exact original source are retained
by the reviewer. No patient arrays or solver were accessed.
