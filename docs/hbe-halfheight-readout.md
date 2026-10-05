# Half-height numerical readout

`scripts/mechanics_hbe_halfheight_readout.py` implements the separate
[prospective equivalence contract](../manifests/experiments/hbe-01-03-halfheight-equivalence-v1.json).
It has no solver launch, measured-curve reader or calibration path. The original
full-domain readout and its accepted input limits remain unchanged.

`read_halfheight_equivalence(root, *, half_bindings, full_bindings,
reconstruction_binding, declaration_binding, expected_branch,
expected_mesh_N)` returns one JSON-compatible numerical receipt. Each run has
six hash-bound primitives: mesh, deck, loading, nodes, elements and solver.
The full inputs must equal the declaration's selected original native run.
The reconstruction binds both meshes, extraction source and exact declaration.
Inputs are verified before reading and again before returning the report.

The adapter independently checks original coordinates, reflected connectivity,
oriented local cube permutations, complete node/cell coverage and mesh counts.
It retains original full coordinates. At shared midplane nodes, displacement
contributions must agree before merging; raw reaction contributions are summed.
Stress components follow the same vector-basis reflection on both tensor axes.

Native half checks prescribe bottom xyz and midplane z only; midplane xy remain
free and their reactions must vanish. Energy uses the existing independent
three-field quadrature. The existing full boundary, balance, Jacobian and work
checks run on reconstructed full geometry, with original H and all 75 probes.
The saved original full run is independently re-read as the comparison.

Signed plate force has scale one, with independent bottom/midplane estimates.
Energy has scale two. Half work uses d/2; full work uses d. Original work-energy
gates apply separately; cross-run work difference is a retained diagnostic.
All 61 states, every nodal reaction component and every node/probe displacement
participate in the declared equivalence limits. Logged stress/J/energy-density
differences are diagnostic; logged energy density never substitutes for
independently integrated three-field energy.

The receipt separates `full_native`, `half_native`, `reconstructed_full` and
`equivalence`. Reconstructed fields explicitly disclaim a full native solver
run. Passing this comparison would establish only the tested numerical branch,
not physical material fidelity or suitability for patient intervention.

Preparation tests use a labelled analytical two-cell fixture and generated
text logs. No saved specimen output or measured response was evaluated to
develop these checks. Actual equivalence remains unexecuted pending review
and a separate execution release.
