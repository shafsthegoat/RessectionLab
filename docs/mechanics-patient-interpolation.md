# Independent saved-field point evaluator

`scripts/mechanics_patient_interpolation.py` evaluates a complete, previously
frozen displacement field. It performs no physics solve, fitting, registration,
landmark disclosure or anatomical acceptance. Only numerical fixtures and
temporary archives have been used; no patient B/V coordinates or arrays were
opened for implementation.

`FrozenTet10Field` copies native MRI RAS nodes and nodal displacement vectors in
metres into immutable arrays. Connectivity is zero-based: vertices 0,1,2,3,
then edge nodes 01,12,20,03,13,23. Positive, adequately conditioned corner
Jacobians and straight reference edges are required. Midside deviation must be
at most 1e−12 m and 1e−8 times the smallest corner-map singular value; the
relative condition prevents the absolute allowance hiding curvature in tiny
cells. The recorded geometry approximation bound is 1.5 times maximum midside
deviation, from the sum of the six nonnegative quadratic edge basis functions.
Original mesh positions are retained. Duplicate geometry, incompatible shared
edge IDs and invalid shared-face topology receive explicit refusal statuses.

The hash-bound `geometry_frame` JSON uses schema
`native-rest-to-common-ras-v1`, native frame `MRI_RAS+`/metres and common frame
`RAS+`/millimetres. `native_m_to_common_mm` contains exactly 1000 times a proper
rotation plus millimetre translation. Point queries use its inverse;
displacement vectors use only its linear part. No voxel indices or implicit
frame identity are used. The record also requires exact structural-source and
baseline-alignment hashes and `provisional_baseline_diagnostic` status.

Point location uses a SciPy centre KD-tree, conservative corner bounding boxes
and NumPy matrix algebra. Candidate corners define barycentric coordinates;
quadratic Lagrange values interpolate the saved nodal vectors. A fixed 1e−12 m
boundary band controls candidate expansion and face-distance roundoff checks.
Coordinates and barycentric values are never clamped. A point with slightly
negative barycentric coordinates that qualifies only through this band remains
`uncertain_reference_boundary` with null displacement. Consequently, a rounded
oblique face point can remain uncertain rather than receive artificial support.

On a shared face, edge or vertex, containing cells must share that topological
simplex, with exactly zero off-simplex barycentric coordinates. Contact that
can only be claimed through the roundoff band stays uncertain, so a small
positive interior overlap is never promoted to supported contact. Every pair
of the proven contact's predicted vectors must agree within 1e−12 m before
their equal traces are averaged. Overlapping interiors are
`ambiguous_reference_location`; incompatible topology or discontinuous traces
are `nonconforming_reference_mesh`. Clearly exterior points are
`outside_reference_domain`; curved, inverted, ill-conditioned or over-budget
geometry is `unsupported_reference_geometry`. These exclusions remain distinct
and null in the comparison report. A supported sample establishes interpolation
coverage only, not valid mechanics or retained tissue.

The portable interface is `sample_frozen_field(root, manifest, FieldQueries)`.
It returns the existing comparison `FieldSamples` bound to the exact query hash.
The strict manifest names this exact entrypoint and five artifacts: reference
mesh NPZ (`nodes_m`, `tet10_indices`), displacement NPZ (`displacement_m`), frame
JSON, upstream numerical evidence and executing interpolator source. All bytes
are hash-checked; arrays are inspected before allocation, pickle is disabled,
and unknown arrays/options fail. Limits are 10,000 nodes, 20,000 elements,
4,096 queries/candidates per point and 64 MiB per encoded/expanded artifact.

Call `load_frozen_field` for input validation before the comparison freeze.
Persist the entire mesh, nodal field, transform and numerical evidence before
V sources and destinations are jointly revealed by the existing landmark
helper. The sampler then receives only source queries, never destinations.
Hashes provide reproducibility on a cooperative host, not authorization by
themselves. Upstream numerical evidence is bound rather than reinterpreted as
solver proof; global mesh overlap, deformation validity and mesh convergence
remain separate gates.

Twenty-six evaluator, seventeen independent and fourteen existing comparison
controls passed together in 0.45 seconds: constant/affine/quadratic patches, shared-face continuity,
physical rotations and units, conservative boundaries, overlap/geometry refusal,
immutable fields, portable bindings and pre-allocation archive limits. These
are software checks, not patient displacement validation. Three initial review
failures are preserved: band-only contact incorrectly accepted a tiny overlap,
and two malformed forward hashes passed the direct loader. The repaired loader
checks identity syntax and retains the existing prefixed-hash convention.
Independent final verification passed all 57 checks in 0.41 seconds with all six bound files unchanged; see `artifacts/mechanics/patient-interpolation-independent-review-v1/verification.json`. Patient query access remains unreleased.
