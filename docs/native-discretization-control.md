# Fixed-scene native discretization control

Prospective numerical unit control required by the October 4 steering; no matrix
result yet. This is not synthetic training data. Existing native tests already
cover joint rigid transforms, affine corners, partial-cell retention and causal
shaft clearance. Jointly rotating cells and the instrument does not test the
effect of changing how fixed physical anatomy is sampled.

The [declaration](../manifests/experiments/native-discretization-control-v1.json)
fixes one physical tissue half-space, one unmodified native fine instrument,
entry, endpoint, aperture and 0.0625-mm microstep. Sixteen grids vary resolution,
actual subvoxel phase and source-grid orientation while anatomy and tool stay in
the same physical coordinates. The source image field encloses the part of the
half-space reachable by this stroke. The proximal shaft extends outside that
field; the native engine's outside-image uncertainty remains recorded.

Occupied cells conservatively cover any positive-volume intersection with the
half-space. They include some volume outside the analytical tissue boundary.
The report therefore separates native whole-cell volume from true half-space
volume inside those cells, calculated by integrating an affine cell clipped by
the physical plane. An independent convex-hull construction tests that integral.
No boundary sliver becomes free space or earns extra native removal credit.

The complete active-sweep/half-space intersection is exactly
`pi * 1.25² * 4 + (2/3) * pi * 1.25³` mm³. This is an ideal geometric reference,
not an asserted physiological resection volume. Static contained-cell mass is
a lower bound, and contacted-cell mass an upper bound, after physical clipping.
Those static bounds are explicitly non-executable. Transactional native removal
is separately checked against them, retaining rejection and partial contact.
Do not require monotonic observed convergence: report the actual finite-grid
bracket and the analytical cell-diameter bounds, including phase-dependent
failures and remaining uncertainty at the finest sampled resolution.

The [runner](../scripts/probe_native_discretization.py) first saves all numerical
rows, then applies the existing independent native-history checker to successful
strokes in the declared order. The worker has a 170-second cooperative allowance;
its parent kills it at 180 seconds or a polled 2-GiB RSS limit. These are different
mechanisms. An individual native preview has no internal cancellation callback;
the parent bound remains necessary. Audit cancellation and unaudited rows are
explicit, and never count as independent acceptance. Source identities, wall
costs, peak RSS and every completed/rejected row persist on failure. No cap will
be enlarged after results. The intended command after source freeze/release is:

```sh
.venv/bin/python scripts/probe_native_discretization.py --output artifacts/native-discretization-control-v1/run-01
```

This isolates the native source-cell model, bypassing proposal generation only.
The separate source-axis proposer intentionally abstains for some oblique access
frames; its action-space restriction is not a measured engine geometry error.
No patient data, policy update, final world, UI or production model changes enter
this control.
