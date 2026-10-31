# Finite planar blade: bounded numerical verification

The corrected standalone kernel passes all frozen numerical gates on 24 generated
histories / 960 prescribed steps, with final work changes of 1.2765% and 2.6517%
against the unchanged 5% limit. See the [temporal correction report](planar-blade-temporal-correction.md)
for the algorithm, three-resolution before/after results, canonical native action
study, runtime and remaining limits. General physical, patient-specific and
clinical validation and planning/training eligibility remain unavailable.

The original BE algorithm and its failed qualification are retained below as the
`2e88f45` baseline. Saved declarations without the new integration revision retain
that algorithm and its exact recorded behavior.

## Model and admission

One rigid rectangular plane has fixed orthonormal tangents and an inward tissue
normal `n`. Only its prescribed scalar normal offset changes. The topology,
explicitly selected source-cell faces, material assignments and supports stay
fixed. Contact faces must border in-image void, be coplanar, and align with the
rectangle on an orthogonal affine source grid. Unknown image boundaries,
retained material interfaces, duplicate faces, nonplanar surfaces and empty or
over-budget footprints refuse. This release supports elasticity or one Maxwell
branch, reusing the existing fully integrated Q1 stiffness operators and
distributed memory equations. It adds no contact spring or foundation.

Explicit support faces clamp all three displacement components. Every tissue
component must lose all six rigid modes **with contact absent**. Held cells are
never supplied to the support check. These are declared numerical boundary
conditions, not inferred skull fixation. Source cells/corner owners remain
attached to the mesh; immutable operator/geometry buffers and source/model
binding checks reject altered arrays, metadata or operators under an old ID.
Stricter source-profile DOF and gradient limits remain binding, and source
identity is checked again before a candidate is published.
Each step returns a detached candidate; refusal or cancellation leaves its input
history unchanged. There is no `SurgicalEngine` integration or scene-property
switch that silently enables this model.

The finite footprint is a **fixed reference rectangle**, consistent with the
small-strain equations. It is not a current-configuration sliding/edge-contact
model. Projected deformed footprint migration is reported separately; its
maximum is 0.00162926 mm on the coarsest partial fixture. Friction, tangential
blade motion, large deformation, cutting, CUSA and patient fitting are outside
this release. Zero traction outside the reference footprint does not establish
current-configuration edge contact.

## Gap, enforcement and energy

For surface interpolation `N`, reference point `x`, initial blade origin `o` and
positive inward offset `d`, define:

```text
C = N n                         b = d - n·(x-o)
g = C u - b >= 0                 lambda >= 0 N
K0 u + sum Kj(u-zj) - Cᵀlambda = 0   [free DOFs]
zj_new = (tau_j zj_old + h u)/(tau_j+h)
```

Positive gap means separation; negative gap means penetration. Rectangle/face
intersections are clipped exactly in reference coordinates. Gauss interpolation
rows and the four vertices of every clipped rectangle constrain the same Q1
trace. A bilinear gap attains its extrema at those vertices, bounding the whole
clipped patch between samples. The positive-semidefinite dual bound QP is solved
with L-BFGS-B and bounded rank-revealing refinement. Admission requires the actual
primal, dual, stationarity and complementarity residuals, regardless of an
optimizer's success flag. Unsupported or uncertified solutions refuse.

All multipliers are **integrated forces in N**, including corner guards with
zero area weight. Every force enters `Cᵀlambda`; no guard force is discarded or
divided by zero area. Reactions are normal-only, so tangential traction is zero.
Redundant constraints can have nonunique multipliers; displacement and aggregate
force are compared, not point-pressure peaks. Reported contact area is reference
Gauss area with `gap <= 1e-9 mm`, distinct from the full footprint area. The
quadrature-order sweep primarily checks area integration and algebraic consistency:
the vertex guards already define the same feasible displacement set.

Using displacement/memory in mm and stiffness in N/mm, all energies are N·mm
(`1 N·mm = 1e-3 J`). With fixed zero supports and no body load:

```text
E = 0.5 uᵀK0u + 0.5 sum (u-zj)ᵀKj(u-zj)
W = lambda_newᵀ (b_new-b_old)                 [signed actuator work]
Dvis = sum tau_j/h * Delta_zjᵀ Kj Delta_zj
Dnum = 0.5 Delta_uᵀK0 Delta_u
       + 0.5 sum Delta(u-zj)ᵀ Kj Delta(u-zj)
Dactivation = lambda_newᵀ g_old
W = E_new-E_old + Dvis + Dnum + Dactivation
```

Withdrawal can return negative work. Separation has zero contact force while
distributed Maxwell memory continues to relax. `Dnum` and contact activation
loss are explicit backward-Euler accounting, separate from assumed viscous
dissipation. Tiny signed roundoff values are retained, not silently clamped.
Neither dissipation quantity is an injury measure.

## Original frozen controls and results (before temporal correction)

[Protocol v1](../manifests/experiments/planar-blade-contact-v1.json) froze the
fixtures, tolerances and limits before numerical probes. Pre-qualification review
caught an inconsistent deformed-footprint refusal and operator-mutation loopholes;
their development evidence is retained. [Method clarification v2](../manifests/experiments/planar-blade-contact-v2.json)
was committed before the qualification runs and explicitly defines corner guards
and the reference footprint without changing fixtures, tolerances or limits. Its SHA256 is
`b1d8e7c18ddbc1288cc50585b2f280e4c57c3eed20caf524ccfe2b61e0525b15`.

Generated 4 mm cubes use 2³/4³/8³ source cells, bottom supports, equilibrium shear
1,000 Pa, and either no branch or 500 Pa / 0.5 s Maxwell relaxation. Uniform
compression uses Poisson ratio zero and a 4×4 mm blade; partial indentation uses
Poisson ratio 0.3 and a 2×2 mm blade. The prescribed ramp/hold/withdrawal lasts
3 s, with 0.1/0.05/0.025 s steps. Limits are 512 cells, 2,187 DOFs, 512 contact
rows, 120 steps, 15 s per step, 600 s per run and 768 MiB fresh-process peak.
The numerical thread count is one.

| Check | Maximum observed | Frozen limit |
| --- | ---: | ---: |
| Penetration over complete clipped reference patches | 1.083e-10 mm | 1e-9 mm |
| Tensile contact force | 0 N | 1e-10 N |
| Free-equilibrium residual | 3.47e-19 N | 1e-10 N |
| Complementarity | 1.42e-15 N·mm | 1e-10 N·mm |
| Discrete energy imbalance | 5.79e-15 N·mm | 1e-10 N·mm |
| Tangential contact force | 4.15e-21 N | 1e-12 N |
| Independent dense partial-fixture displacement difference | 4.56e-11 mm | 1e-7 mm |
| Independent dense aggregate-force difference | 2.88e-13 N | 1e-9 N |
| Uniform analytical displacement relative error | 9.53e-14 | 1e-7 |
| Whole-fixture rotation displacement difference | 5.14e-16 mm | 1e-8 mm |
| Whole-fixture rotation reaction difference | 8.67e-19 N | 1e-10 N |

The uniform elastic control has analytical force `2 mu A d/H = 0.00032 N`
and stored energy `0.0000064 N·mm` at 0.04 mm indentation. The independently
implemented dense oracle assembles Q1 matrices and surface rows separately and
uses a primal constrained solve/KKT recovery, unlike the production dual solve.
It checks complete coarse partial elastic and Maxwell histories. Withdrawal
ends with zero force and zero contact area. Seven qualification refusal controls
pass, including a touching disconnected component lacking independent support.

Partial elastic hold reactions are 0.000220337, 0.000165076 and 0.000150598 N
on the 2³/4³/8³ meshes, with 4 mm² reference contact area. The 4-to-8 reaction
and stored-energy changes are 9.6138%, below the frozen 25% gate but still material
resolution dependence. Orders 1–4 preserve the guarded feasible set; order 2-to-4
area changes are at roundoff. Singular sharp-edge peak stresses are not scored.

Hold reaction changes by 0.11354% from 0.05 to 0.025 s (2% limit). Against the
continuous uniform Maxwell hold solution, absolute force errors decrease from
1.47215e-6 to 7.62324e-7 to 3.87925e-7 N. Signed net cycle work fails its gate:

| Coupon | Work at 0.05 s · N·mm | Work at 0.025 s · N·mm | Change / 5% limit |
| --- | ---: | ---: | ---: |
| Uniform Maxwell | 4.11916477e-6 | 3.74182482e-6 | **10.0844% / fail** |
| Partial Maxwell | 1.12171646e-6 | 1.00680727e-6 | **11.4132% / fail** |

The ledgers close while backward-Euler numerical dissipation remains time-step
sensitive. No alternate work norm, extra time steps or looser limits replace
these failures. The first run also exceeded the memory cap at 771.5625 MiB due
to whole-report serialization copies. A streaming writer preserves identical
numerics. The final source additionally rejects nonfinite energy from underflowing
time steps and respects stricter source-profile limits; its repeated run takes
21.910 s with 190.406 MiB peak on an Apple M2 Pro,
Python 3.12.12 / NumPy 2.5.3 / SciPy 1.18.1. All 24 histories pass their step
checks; the overall qualification still returns exit 1 for the two work gates.

## Reproduction and limits

```sh
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 .venv/bin/python -m pytest \
  tests/test_surgical_planar_contact.py tests/test_planar_blade_contact_qualification.py -q
.venv/bin/python scripts/qualify_planar_blade_contact.py \
  --output testcases/outputs/planar-blade-contact-NEW.json
```

Use a fresh output name. The runner generates its fixtures locally, uses a fresh
single-thread worker and retains every failure, raw state/force/ledger, source
snapshot and hardware identity under ignored `testcases/`. Final evidence is
`testcases/outputs/planar-blade-contact-20261030/qualification-v4.json`, with its
compact summary and README beside it; the first failed run is preserved.
The focused continuum/kernel regression selection passes 112 tests. An isolated
35-test dependency control also passes with the committed continuum module,
confirming no dependency on the checkout's separate unfinished live-contact work.
No desktop or live surgical integration was tested or enabled in this milestone.

This is numerical verification of generated coupons, with failed work-sensitivity
gates and unqualified current-edge geometry. It establishes no physiological,
patient-prediction, material-fitting or RL claim. Existing historical continuum
boundary/contact failures remain unchanged.

The [Zimmerman–Ateshian paper](https://doi.org/10.1115/1.4040497) and
[FEBio sliding-elastic documentation](https://febiosoftware.github.io/febio-docs/features/features/solid_surfaceinteraction_sliding-elastic/)
provide the broader finite-surface contact reference. FEBio describes penalty
and augmented-Lagrangian enforcement; this bounded linearized kernel uses the
convex constrained solve above. It does not reproduce their finite-deformation,
frictional examples or claim a FEBio cross-solver validation.

The subsequent [canonical maneuver integration](planar-blade-maneuver.md) exposes
one opt-in engineering sequence through native actions and desktop replay. It
retains the failed work-sensitivity gates above and additionally refuses loaded
footprint migration and nonplanar rendered boundary faces. It changes no
historical episode or physiological admission.
