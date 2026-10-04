# Conditional patient displacement: FEBio constraint feasibility

Status: **static source review and control specification only**. No patient arrays, landmark destinations, new mesh, model parsing or FEBio solve were accessed for this review. Source is FEBio v4.13 commit `32ae206ff4881dfb54f62296cd1558e58ed9fcc6`; selected local file hashes are in [the source receipt](../artifacts/mechanics-patient-constraints-v1/source-receipt.json). The earlier five successful homogeneous patches did not exercise multipoint constraints or tet10 elements.

A minimal existing-framework path is available: keep the volume mesh and material in FEBio; express the frozen observation operator as exact inhomogeneous linear multipoint constraints; let FEBio minimize its hyperelastic energy on the remaining degrees of freedom. No new physics solver or artificial fixed node is needed. Whether a particular anatomy/operator yields a well-posed, tractable solve remains untested.

## Freeze the observation functional first

For reference-coordinate landmark source `p_b`, define `k_b(X)=max(1−||X−p_b||/0.005 m,0)` and

`H_bi = integral_Omega[k_b(X) N_i(X) dV0] / integral_Omega[k_b(X) dV0]`.

Then impose `sum_i H_bi u_i = d_b` independently for x, y and z. Use the fixed reference anatomy; never move the kernel with the current deformation. Each row is dimensionless and sums to one. Convert physical RAS millimeters to meters exactly once for mesh coordinates, radius and observed displacement. Retain the source-frame transform and its inverse for returned predictions.

A point-picked ultrasound displacement is **not an observed 5 mm tissue-volume mean**. Treating it as this mean is a declared measurement-support/regularization assumption. Radius, support clipping, integration accuracy, pivot choices and element formulation must be frozen without validation landmarks or their destinations. A nodal-distance average is a different, mesh-density-dependent operator and must not silently replace this volume integral.

Integrate numerator and denominator with the same frozen reference-volume rule. Check positive support volume, row sum and constant/affine-field reproduction about the actual kernel centroid. The compact kernel is not polynomial; the solver's ordinary element quadrature alone may miss or poorly resolve a small support on a coarse element. A separate, prospectively bounded integration-refinement check is required. Quadratic tet10 basis functions can give negative nodal coefficients: do not clip them or renormalize a different operator. Empty support, nonfinite input, unresolved integration error or invalid reference Jacobian must fail before any solve.

## Exact FEBio 4.0 XML

Version 4.0 delegates `Boundary` to `FEBioBoundarySection3`. It recognizes `bc type="linear constraint"`; `node` and `dof` select the **dependent parent**, and `child_dof` provides the independent terms. The exact implemented equation is:

`u_parent = sum(child.value * u_child) + offset`.

`offset` accepts a load-controller attribute. For the illustrative equation `0.25*u1_x + 0.75*u2_x = 0.001 m`, the correct boundary fragment is:

```xml
<Boundary>
  <bc type="linear constraint">
    <node>1</node>
    <dof>x</dof>
    <offset lc="1">0.004</offset>
    <child_dof>
      <node>2</node>
      <dof>x</dof>
      <value>-3</value>
    </child_dof>
  </bc>
</Boundary>
```

A standard linear load controller with points `(0,0)` and `(1,1)` ramps the offset from rest. This fragment is **not a complete or well-posed model**. The parser accepts node IDs, resolves them to mesh indices and uses displacement names `x`, `y`, `z`. It does not require a `node_set` for this special boundary type. Keep coefficients fixed; attach the load controller only to the offset.

Source support: [version dispatch](https://github.com/febiosoftware/FEBio/blob/32ae206ff4881dfb54f62296cd1558e58ed9fcc6/FEBioXML/FEBioImport.cpp#L274), [XML fields](https://github.com/febiosoftware/FEBio/blob/32ae206ff4881dfb54f62296cd1558e58ed9fcc6/FEBioXML/FEBioBoundarySection3.cpp#L157), [implemented update](https://github.com/febiosoftware/FEBio/blob/32ae206ff4881dfb54f62296cd1558e58ed9fcc6/FECore/FELinearConstraintManager.cpp#L595).

## Dependent DOFs and rank

The manager rejects any active parent DOF that occurs among **any** active constraint's children. It does not support recursive parent chains. Duplicate parent definitions can overwrite its lookup table, so the adapter must reject duplicates explicitly. Parent DOFs must not also carry prescribed/fixed boundary conditions or belong to rigid bodies. Use deformable volume nodes only; the source warns that constraint matrix profiling is not generic for contact connectivity. This POC has no contact.

For `m` scalar rows and `n` nodes, choose a deterministic, nonsingular pivot block `H_P` from anatomy/operator coefficients only, with the other columns `H_F`. Export

`u_P = −solve(H_P,H_F) u_F + solve(H_P,d)`.

This is ordinary constraint-coordinate elimination, not a mechanics solver or a change to `H u=d`. Use a stable library factorization and record its rank/condition evidence; do not form an explicit inverse. Parent columns occur nowhere among the children. The same scalar pivot block can be used for all three displacement components. Overlapping kernels are allowed after this simultaneous elimination; they must not be exported as naive chained constraints.

A simple overlap control is `H=[[1/2,1/2,0],[0,1/2,1/2]]`: parents 1 and 3 give `u1=2*d1−u2` and `u3=2*d2−u2`. Both original rows remain exact. The adapter must verify its exported coefficients by reconstructing the original rows. It must refuse a deficient or prospectively ill-conditioned block, never drop a row or add a guessed anchor. Exact elimination can create more nonzero coefficients and a larger matrix profile; bound exported terms and memory before launching.

Also test the restriction of `H` to the six infinitesimal rigid-body modes. Scale rotation columns by a declared characteristic length before recording singular values. Full row rank of `H` alone does not remove all rigid motions. Insufficient rigid-mode rank, disconnected unconstrained components or missing support must abstain. This is a necessary local well-posedness check, not proof of global finite-strain uniqueness. No skull fixation, zero normal motion, gravity or pressure boundary may be invented to make the system solvable.

Source support: [initialization/activation](https://github.com/febiosoftware/FEBio/blob/32ae206ff4881dfb54f62296cd1558e58ed9fcc6/FECore/FELinearConstraintManager.cpp), [parent fixed activation](https://github.com/febiosoftware/FEBio/blob/32ae206ff4881dfb54f62296cd1558e58ed9fcc6/FECore/FELinearConstraint.cpp), [matrix profile caveat](https://github.com/febiosoftware/FEBio/blob/32ae206ff4881dfb54f62296cd1558e58ed9fcc6/FECore/FELinearConstraintManager.cpp#L150).

## Nonlinear solve and output caveat

FEBio's solid solver calls constraint `PrepStep` before Newton iterations and `Update` after nodal updates. The manager transforms residual and stiffness terms, including the nonzero-offset contribution. A conservative hyperelastic model with fixed linear constraints retains a symmetric reduced tangent in exact arithmetic; actual convergence and symmetry must still be verified with the fixed solver configuration. Use the already reviewed explicit symmetric stiffness/Skyline path only within a prospectively bounded mesh; do not infer patient scalability from an eight-element patch.

**Ordinary `Rx/Ry/Rz` do not provide the eliminated constraint forces.** Parent DOFs are assigned fixed status (equation ID −1), while the ordinary reaction path stores only prescribed IDs ≤−2. Zero parent reaction output therefore does not certify force balance. For verification, independently evaluate the frozen-law internal force/virtual work from actual returned nodal deformation, then check equilibrium in feasible directions (`T^T f≈0`) and the original `H u−d` residual. If constraint multipliers are reconstructed from full internal force, retain their sign, units and reconstruction residual. This is an output audit, not a new solver. A narrower independent check can evaluate the correctly integrated frozen-law energy at `u ± h*T*v`, compare the central directional derivative with zero and demonstrate a prospectively declared step-size convergence window. That must accompany genuine reduced FEBio residual evidence; it cannot substitute for an unconverged solution. Element-averaged stress/energy alone is insufficient to reconstruct arbitrary nonuniform internal forces. Alternatively, report unavailable force evidence and do not claim full mechanical verification.

Keep actual element Gauss-point/extra-point Jacobian positivity, constraint residual, fixed-step nonlinear convergence and deformation outputs independent of landmark prediction scores. Three-field energy uses cell-average volume ratio; the existing affine-patch energy shortcut must not be reused for nonuniform anatomy.

Source support: [solid update](https://github.com/febiosoftware/FEBio/blob/32ae206ff4881dfb54f62296cd1558e58ed9fcc6/FEBioMech/FESolidSolver2.cpp#L517), [residual transformation/reaction IDs](https://github.com/febiosoftware/FEBio/blob/32ae206ff4881dfb54f62296cd1558e58ed9fcc6/FEBioMech/FEResidualVector.cpp#L48), [three-field average pressure](https://github.com/febiosoftware/FEBio/blob/32ae206ff4881dfb54f62296cd1558e58ed9fcc6/FEBioMech/FE3FieldElasticSolidDomain.cpp#L509).

## Element choice at conditional Poisson ratio 0.45

For the same uncoupled alpha=2 Ogden numerical law, `c1=2*mu`, `m1=2`, `pressure_model=1`, finite initial `nu=0.45` means `K/mu=29/3`. This differs from the specimen `nu=0.49` assumption and requires its own declared model identity. It is not a fitted patient constitutive property.

For a conforming anatomical tetrahedral mesh, the narrow candidate is **tet10 with eight-point quadrature and explicit `elastic-solid`** at this finite compressibility:

```xml
<SolidDomain name="anatomy" mat="conditional_material"
             elem_type="TET10G8" type="elastic-solid"/>
```

The pinned builder defaults tet10 to eight-point quadrature and turns three-field tetrahedra **off**. The factory supports ordinary tet10, selective-reduced-integration variants and explicit three-field tet10. Availability is not evidence that all have equivalent stability. Do not switch between them after evaluating landmark errors. Tet10 midside ordering, curved reference Jacobians and domain support must be validated before solving. A linear tet4 anatomy mesh must not silently be treated as tet10.

If a geometrically faithful, quality-controlled hex8 volume mesh already exists, the separately viable path is explicit `three-field-solid`, `HEX8G8`, `laugon=0`; our actual patch evidence covers that formulation only at its frozen numerical inputs. A voxel staircase does not become a verified anatomy surface by choosing hex8. Neither path has established anatomy mesh convergence or freedom from locking; those remain separate numerical controls. Tet10 ordinary versus three-field choice is a prospective decision, not a performance/accuracy conclusion from this static review.

Source support: [element defaults](https://github.com/febiosoftware/FEBio/blob/32ae206ff4881dfb54f62296cd1558e58ed9fcc6/FEBioXML/FEModelBuilder.cpp#L56), [supported domains](https://github.com/febiosoftware/FEBio/blob/32ae206ff4881dfb54f62296cd1558e58ed9fcc6/FEBioMech/FESolidDomainFactory.cpp#L67), [explicit element/domain parser](https://github.com/febiosoftware/FEBio/blob/32ae206ff4881dfb54f62296cd1558e58ed9fcc6/FEBioXML/FEBioMeshDomainsSection4.cpp#L115).

## Required small controls before patient execution

The first two algebra specifications and the feasible-direction/rank/unit identities have executable controls in `tests/test_mechanics_patient_constraints.py`; their result receipt is separate. Operator integration and all solver controls below remain **not executed**. Freeze numerical tolerances, term/memory caps and exact solver decks before release.

| Control | Required independent result |
| --- | --- |
| Nonzero scalar equation above | XML equation reproduces 0.001 m with the correct offset sign and load fraction; mismatched mm/m fails. |
| Two overlapping rows | Simultaneous elimination preserves both rows for arbitrary free coefficients; no parent appears as a child. Duplicate pivots and rank deficiency are rejected. |
| Fixed tent operator | Constant reproduction; affine reproduction at the volume-weighted centroid; bounded integration refinement; no source-only support yields no model. |
| Free body with three noncollinear kernel supports conditioned on a common translation | No other BC; returned field is that rigid translation with zero strain/energy within fixed tolerances. Deficient supports must refuse before solve. |
| Small nonrigid conditional deformation | Original constraint residual, reduced virtual-work equilibrium, positive actual Jacobians and complete fixed-time output pass. Reaction-log zeros are explicitly not accepted as evidence. |
| Proposed tet10 domain patch | Finite affine stretch/shear against the established law, plus node-order/inverted-element negative; free interior DOFs required. Existing hex8 evidence cannot substitute. |

Current blockers to a verified patient run: no actual MPC control has run, no tet10 control has run, the fixed integral operator and pivot/cap policy are not implemented/frozen, and eliminated constraint forces require separate audit logic. These are numerical readiness gaps, not permission to add anatomical boundary conditions or access held-out destinations.
