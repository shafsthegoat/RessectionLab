# HBE axial half-height equivalence

Execution update, October 8, 2026: the separately released preparation and four
solve cases have completed, with independent saved-output review. See the
[result and scope](../artifacts/mechanics/hbe-halfheight-execution-v1/RESULT.md).
All N8/N12 axial equivalence criteria passed; the original spatial-convergence
and material-validation gates remain unresolved. The specification below
preserves the prospective preparation contract; it is not a release for more runs.

This preparation tests a computational cost hypothesis for the same declared axial specimen model. It does not authorize a solver run, measured-curve access, calibration or promotion to finer meshes. The original mesh-gate failure and subsequent full N24 timeout remain unchanged.

The [exact-quarter draft](../artifacts/mechanics/hbe-quarter-abandoned-preparation-v1/README.md) was abandoned before solving: picometre x/y mesh-coordinate differences exceeded its prospective exact-reflection criterion. That is not a clinically meaningful geometry error or an invalidation of the full model. The [saved eligibility check](../artifacts/mechanics/hbe-quarter-eligibility-diagnosis-v1/README.md) found that the distinct z=H/2 cut passes the unchanged criterion on N8 and N12, with no straddling cells, complete node/cell reflection matches and maximum discrepancy below 4.34e-19 m. Geometric eligibility is not yet mechanical equivalence.

The [new declaration](../manifests/experiments/hbe-01-03-halfheight-equivalence-v1.json) fixes four comparisons: compression and tension, each at N8/N12, μ=1000 Pa and 60 equal load steps. Extract only whole original cells in the lower half, using original coordinates and connectivity; do not remesh, snap or average geometry. Record explicit original/local IDs, reflection maps and canonical hex8 permutations. N8 contains 627 half nodes/384 cells; N12 contains 1828/1296. The full radius, original height H, constitutive law, three-field formulation, solver settings and physical load d(t)=±0.15Ht remain authoritative.

The physical bottom stays bonded with all three displacements zero. At the artificial midplane prescribe only u_z=d(t)/2; u_x/u_y are free. The curved side remains traction-free. The modeled domain height H/2 must never replace H in the full loading coordinate or physical normalization.

For a reflected upper point, lift displacement as (u_x,u_y,d−u_z), and use the original full rest coordinate plus that displacement. Shared-midplane displacements must agree before keeping one value. Reflect raw reaction vectors by diag(1,1,−1) and **sum every contribution** into original full node IDs. This cancels artificial normal-plane reactions; averaging, dropping cut reactions or summing displacements is incorrect. Reconstruct full fields and stresses using the original full mesh. Canonical full-top axial force equals both +ΣR_z on the half physical bottom and −ΣR_z on the half midplane: force scale is one. Full energy and work equal twice half energy/work, with half work integrated over d/2.

All four cases must pass at every one of the 61 states. Compare every lifted nodal displacement, every raw nodal reaction component and all 75 original physical probes. Motion error uses Euclidean norms and a fixed series-wide limit 1e-6R+1e-5 max‖u_full‖. Raw reactions use 1e-7F₀+1e-4|R_full| componentwise; each signed force uses 1e-7F₀+1e-4|F_full|; energy uses 1e-7E₀+1e-4|E_full|. Both lifted and independently scaled half totals must agree with full results. Original per-run residual, boundary, balance, Jacobian and work-energy gates remain required. These are numerical equivalence limits, well below the existing refinement budgets, not experimental uncertainty.

Pure preparation has 60 s/3 GiB and no Gmsh or solver calls. A separately released solve phase permits at most four calls, 90 s each/420 s aggregate, one numerical thread, 3 GiB sampled process-group RSS, 128 MiB active output and 512 MiB total new output. Original inputs stay external and are not duplicated. Failures stop without retries or revised criteria.

Passing would support the tested midheight-symmetric axial branch only. The constraint can exclude asymmetric instability. Ordinary torsion is excluded; no speedup, fine-mesh convergence, material fidelity or patient relevance is established by preparation.
