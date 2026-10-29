# Opt-in voxel continuum for mechanics research

`brainContinuumV1` is an explicit scene property. Historical scenes and saved
recordings retain their lattice semantics. The new model addresses the observed
resolution dependence and infinitesimal rotation stiffness of that lattice;
it does not establish living-patient mechanics or surgical injury prediction.
The [frozen protocol](../manifests/experiments/continuum-mechanics-v1.json)
defines numerical controls and the limited exposed-specimen comparison.

## Material and geometry

Each retained source voxel owns one affine hex8 element occupying exactly its
original physical cell. Full 2×2×2 Gauss integration assembles symmetric-strain
elasticity. For equilibrium shear modulus μ and Poisson ratio ν,

```text
λ = 2 μ ν / (1 − 2 ν)
ε = sym(∇u)
W = μ ε:ε + λ tr(ε)² / 2
```

Material inputs are Pa, converted explicitly to N/mm². Positions are RAS
millimetres and energies N·mm. This release permits ν≤0.45; near-incompressible
locking and finite rotations/strains remain unqualified. The displacement
gradient is bounded at all element corners and Gauss points. Exceeding the
declared bound rejects the attempted action instead of projecting its solution.
That bound is a numerical model-domain choice, not a tissue injury threshold.
The enclosing V2 profile still supplies patch radius, source-cell cap, fixed time
step and displacement/force domain bounds. Its historical spring stiffness,
nodal drag, hold stiffness and divergence-penalty coefficients do not determine
this continuum response; material and handle coefficients come from the
explicit continuum profile.

Corner DOFs share only through uncut face connectivity among incident source
cells. Edge/corner-only touching cells are not glued. A cut splits corner
connectivity; removal deletes exactly its source-owned element. Tissue remains
continuous across a tumor-label interface until explicitly cut. The radiographic
mask is not a capsule or a calibrated glioma infiltration probability.

[Hinrichsen et al., Table 11](https://pmc.ncbi.nlm.nih.gov/articles/PMC10511383/)
provides postmortem region-dependent hyperelastic parameters. Their initial
shear limits inform explicit material scenarios; they are not patient estimates
or validation of this reduced law. Using a quasi-static value as the equilibrium
branch is itself an assumption; this comparison does not measure its long-time
limit. A homogeneous brain assignment does not
identify that patient's cortical regions. Different tumor coefficients remain
assumed sensitivity scenarios unless suitable measurements support them.

## History, support and energy

Up to two Maxwell branches augment equilibrium elasticity. Each branch has an
explicit relaxation time and shear modulus. Bulk and shear share the same
assumed spectrum, keeping each material's Poisson ratio constant. Internal
displacement histories reproduce the corresponding linear strain-history
convolution; they are stored per source-cell corner for exact cut/removal/replay
identity. Processed quasi-static HBE curves cannot identify these times.

For each fixed substep, backward Euler eliminates the branch history and uses
a cached local sparse factorization. The matrices rebuild when topology,
materials, support, contact operator or time step changes. There is no global
patient FEM solve at every action. Separate source-cell and DOF caps remain in
force; static matrix audits exceeding a runtime cap are labeled as such.

Artificial patch faces against retained inactive soft tissue are fixed; other
faces have zero declared traction. No skull fixation, ICP, CSF pressure or
buoyancy is inferred. Every connected component must have all six rigid modes
constrained. Unsupported free components cause a refusal, not an artificial
foundation force. The contact operator distributes a fixed total handle
stiffness over held source cells, avoiding stiffness proportional to voxel count.

The fixed-topology, constant-target energy identity includes equilibrium and
Maxwell storage, handle stiffness, viscous dissipation and numerical dissipation.
Stored energy carried out with removed active tissue has a separate export
ledger. It is not converted to blood, heat or injury. Changes of handle/target
have unmeasured actuator work outside that identity. Displaced or remembered
tissue cannot silently become rigid/inactive, and patch activation cannot add
unjustified displacement/history. Such transfers require a separately justified
model and currently refuse the action.

Two cumulative, source-label-specific quantities integrate ε:ε and σ:σ over
volume/time. They describe modeled mechanical exposure in mm³·s and
Pa²·mm³·s. They have no neurological threshold, deficit probability, or calibrated
biological damage meaning. A longer hold may reduce instantaneous force while
increasing these integrals.

## Checks and working integration

`tests/test_surgical_continuum.py` checks six rigid modes without hourglass
modes, physical affine energy on oblique/reflected grids, resolution-invariant
affine energy/reactions, cut-face separation, exact source removal, memory and
time partitioning, one/two-time relaxation including a zero branch, exposure
integrals, removed-energy export, unsupported transfers and domain refusals.

`scripts/qualify_continuum_mechanics.py` adds traction-free uniaxial controls and
an optional comparison with already exposed acquired HBE development values.
That comparison freezes a published material scenario, fits nothing, reads no
reserved modes, and preserves source hashes. Cohort overlap, voxel geometry,
boundary-representation differences and the absence of metrology/acceptance
thresholds prevent an independent physical-validation claim. Negative results
remain in the report.

A separate [prospectively frozen donor-transfer test](continuum-donor-transfer.md)
now compares a development-calibrated continuum with the default lattice on
one acquired, previously unopened motor-cortex specimen. Compression improves,
but the zero-force baseline wins in tension; this mixed result leaves physical
acceptance open. It does not replace the replay case's original assumed
material scenarios or validate relaxation.

`scripts/prepare_continuum_rehearsal.py` reuses the admitted acquired-source
development archive and historical exposure. It executes native single-step
blade acquisition, two timed holds, actual finite-CUSA work, release/withdrawal,
canonical replay and a domain-limited STOP with checked tool cleanup. Original
MRI and source labels are unchanged. Exact local commands, inventories and
receipts remain in ignored `testcases/`.

The existing desktop microscope can replay these source-bound branches. Its
research inspector identifies the actual model and exposes dissipation,
removed-energy export and mechanical exposure. The hidden
`verify-continuum-rehearsal.cjs` harness checks native endpoints, HUD/surface
identity, GPU completion, fixed optics and save without showing a window.

Remaining gates include non-affine/patch-radius convergence, measured transient
response, large deformation, actual cavity/CSF boundaries, independent withheld
anatomical prediction, device contact calibration and linked neurological
outcomes. Numerical agreement and the exposed specimen comparison do not admit
large-scale physiological RL.
