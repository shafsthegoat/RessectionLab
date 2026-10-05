# FEBio numerical patch controls

Prepared for tagged FEBio 4.13 commit `32ae206ff4881dfb54f62296cd1558e58ed9fcc6`.
No solver execution or mechanics validation has occurred. The thin
`scripts/mechanics_febio_verification.py` builds XML and checks primitive outputs;
it never launches FEBio or solves equilibrium. The separate isolated-runtime
workflow owns acquisition, executable identity, process supervision and release.

Five fixed cases share a 10 mm cube of eight hex8 elements, 27 nodes and three
free center-node displacement DOFs. Every boundary component is prescribed,
including zero values, to retain signed reactions. The cases are zero load,
translation `(0.3,-0.2,0.1)` mm, finite stretch `diag(1.10,0.95,1)`, simple shear
`Fxy=0.1`, and the same shear with all stiffness doubled. Four load fractions
`0.25,0.5,0.75,1` are required. There is no contact, body load, inertia, rigid
connector, patient mesh or measured specimen curve.

The explicit material is FEBio `Ogden`: `c1=2μ`, `m1=2`, other coefficients zero,
`k=149μ/3`, `pressure_model=1`. The arbitrary reference μ is **1000 Pa**, not a
measured or fitted tissue property. With `J=det(F)>0`, `B=FFᵀ`, this convention is

- `W = μ/2 [J^(-2/3) tr(B)−3] + K/4 [J²−1−2 ln(J)]`, per reference volume;
- `σ = μ J^(-5/3) dev(B) + K/2 [J−1/J] I`, Cauchy stress;
- `P = J σ F^(-T)`, first Piola stress.

The [tagged Ogden implementation](https://github.com/febiosoftware/FEBio/blob/32ae206ff4881dfb54f62296cd1558e58ed9fcc6/FEBioMech/FEOgdenMaterial.cpp)
and [volumetric convention](https://github.com/febiosoftware/FEBio/blob/32ae206ff4881dfb54f62296cd1558e58ed9fcc6/FEBioMech/FEUncoupledMaterial.h)
bind these equations. This choice differs from FEBio's default logarithmic
volumetric penalty. The domain is explicitly `three-field-solid` with `laugon=0`;
augmentation would change the selected finite compressibility assumption.

Use SI metres, newtons, pascals and joules. The conservative no-contact problem
uses explicit symmetric stiffness and Skyline, with BFGS, at most 15 stiffness
reformations, `dtol=etol=rtol=1e-10`, and `min_residual=1e-20` N². These strict
patch settings are prospective, not validated specimen-solver settings.

The deck requests current coordinates, displacement and `Rx;Ry;Rz`, plus
Cauchy stresses, `J` and `sed`. Tagged `DataRecord` default output has 12
significant digits; avoid its lower-precision custom `%g` format. Element data
are integration-point averages. The checker reconstructs deformation gradients
and determinants from returned node positions at all eight Gauss points,
eight corners and the center of every element. It compares logged average J
with the actual Gauss average and reports the minimum sampled J; this is not
an everywhere-positivity proof for arbitrary warped elements.

For this no-external-load fixture, FEBio's raw reaction is **body on constraint**,
`R = −∫ Nᵃ P n dA`, not the required applied nodal force. This sign follows the
[tagged residual/solver assignments](https://github.com/febiosoftware/FEBio/blob/32ae206ff4881dfb54f62296cd1558e58ed9fcc6/FEBioMech/FESolidSolver2.cpp)
and [output accessor](https://github.com/febiosoftware/FEBio/blob/32ae206ff4881dfb54f62296cd1558e58ed9fcc6/FEBioMech/FEBioMechData.cpp).
Constant nominal tractions integrated against boundary bilinear shape functions
give independent signed nodal expectations, including edge/corner contributions.
The checker never chooses the sign from the observed result. Global force and
moment balance use all reactions and **current** positions; the free center's
force supplies an additional equilibrium-residual check.

All output states must be complete, uniquely indexed, finite and synchronized.
Acceptance checks actual boundary/interior motion, displacement consistency,
signed reactions, stress, reference energy, positive sampled Jacobians, force and
moment balance, and final nonlinear residual evidence at every load fraction.
Normal termination alone is insufficient. Raw residual norms and tolerances
remain in the report. A separate cross-case helper checks unchanged motion/J
and doubled reactions/stress/energy. Reactions of prescribed-zero nodes must be
retained: [the official boundary documentation](https://febiosoftware.github.io/febio-docs/features/features/solid_bc_prescribed_displacement/)
explains why fixed-zero boundary conditions cannot replace them here.

Owner controls cover closed-form shear/dilatation, XML/free DOFs, units,
reaction sign, hidden interior warp/inversion, averaged-J corruption, force and
moment imbalance, wrong energy volume measure, extraction damage and incomplete
solver residuals. Parser-positive fixtures are analytically generated test
records, explicitly not FEBio outputs. Tagged supporting source and receipts
are under `artifacts/mechanics-febio-verification-v1/`.

Before any solver run, freeze the builder, generated deck hashes and runtime
identity, then obtain the separate execution release. The external driver must
preserve process logs, exit status, actual executable identity, and input/output
hashes; the output checker alone does not prove which executable generated a
file. Failure must remain recorded without changing a sign, tolerance or material
parameter to fit it. Passing these patches would not establish locking resistance,
mesh/load-step convergence, bonded-cylinder fidelity, measured specimen response,
or patient tissue mechanics. Those remain separate gates.
