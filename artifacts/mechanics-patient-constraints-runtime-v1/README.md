# Prepared tet10 and multipoint-constraint software controls

**Prepared, not executed.** This declaration permits no solver execution until root review, commit and separate release. It uses no patient data, HBE curves, anatomy meshing or synthetic training examples.

Three exact cases, in order, use the analytic 10 mm cube with six tet10 elements, 27 nodes, and a genuinely interior body-diagonal midpoint. The affine case leaves that node's three displacement DOFs free. The other cases impose only nine linear MPC equations from three overlapping nodal averages; no additional fixation or anchor is present.

1. `tet10_affine`: mixed stretch and xy/yz shear, checking volumetric and deviatoric response, signed quadratic-face reactions, actual interior displacement and all sampled Jacobians.
2. `mpc_translation`: identical target translation at three noncollinear overlapping nodal averages. The expected whole field is rigid translation, with zero strain/energy.
3. `mpc_nonrigid`: a small fixed nonrigid set of average displacements. There is no assumed equilibrium displacement field. Acceptance requires the actual original constraints, pinned-law stress/energy consistency, complete nonlinear convergence and independent feasible virtual work.

The nodal averages are explicit software fixtures, **not** the proposed patient 5 mm volume-tent observation operator. The geometric integration accuracy and medical interpretation of that future operator remain unverified. This fixture does not change the prior specimen or patient model.

The established FEBio material is uncoupled alpha=2 Ogden, pressure_model1, mu=1000 Pa (arbitrary numerical scale), K/mu=29/3 (initial nu=.45). Domain is explicitly `elastic-solid`, element `TET10G8`. The eight-point rule and ten-node ordering are taken from the pinned FEBio v4.13 source. No material solver or mesher is implemented here.

The frozen settings are four steps of .25 with no cutback/retry, BFGS, symmetric stiffness and Skyline, dtol/etol/rtol1e-10 and squared force residual floor1e-20 N². Plot output is disabled. The launcher must reuse the reviewed `scripts/febio_runtime.py` process-group supervisor, execute from an exact committed archive, bind the accepted executable/libraries and all sources/decks before and after, initialize all case statuses, and stop after the first solver/checker failure. **Aggregate60s,3GiB sampled process-group RSS,one numerical thread,one attempt.** No case or tolerance changes after observing solver output.

The exact values, deck hashes, observation matrix, parent/free node IDs, displacement targets and numerical tolerances are in [decks/manifest.json](decks/manifest.json). The eventual worker must rehash that declaration and require complete initial plus all four load states. Root's execution receipt will record the actual command, runtime, environment, timing, memory, solver outputs and any partial failure; this preparation is not such a receipt.

For both MPC cases, the output audit evaluates energy at positive/negative perturbations in **all72 feasible basis directions**, normalized so a unit perturbation has maximum nodal displacement1. The predeclared central-difference steps are2e-7,1e-7,5e-8m. Every finest derivative must be within1e-7N of zero; the last step difference must be at most2e-8N and no larger than the previous difference plus1e-9N roundoff allowance. The derivative arrays and all gates are retained, not collapsed into a hardcoded pass. This complements FEBio's actual reduced nonlinear residual. Parent Rx/Ry/Rz are saved only; they do not report eliminated MPC forces and cannot pass a force-balance gate.

Independent energy uses the pinned reference-volume quadrature applied to actual returned deformation. Element log stress, J and sed are compared against **unweighted** integration-point means, matching FEBio's primitive output convention. The global integrated energy is calculated separately; element-mean sed times volume is not assumed correct for nonuniform deformation. Sampled Jacobian witnesses include all eight Gauss points, vertices, edge midpoints and centroid; this does not certify positivity at every point of arbitrary curved elements.

Owner checks include a deliberately constraint-compatible but unrelaxed displacement field, which passes H u=d and has zero parent reaction log values but fails the independent virtual-work gate. Constructed text fixtures are clearly labelled and are not solver evidence. Unit checks do not replace the future actual runtime attempt.
