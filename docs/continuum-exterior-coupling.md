# Fixed-domain exterior preparation

The [distributed-history qualification](continuum-history-solve.md) now passes
its numerical gate. The next mechanics milestone targets the rigid artificial
core clamp, while retaining the already studied 8 mm core and 20 mm outer
domain. Both earlier 16-to-20 mm stress-stability gates remain failed. No new
radius, material fit, source response, patient role or live cap is selected.

The first step is executable geometry preparation, not a mechanical coupling
solver. `scripts/prepare_continuum_exterior.py` reads only six geometry/support
arrays from the two SHA256-bound, already exposed reference exports. The
[preparation declaration](../manifests/experiments/continuum-exterior-preparation-v1.json)
binds these exact inputs. It does not decode displacement, compliance, basis or
measurement response arrays, or load another patient archive.

## Source ownership changes the induced core topology

The partition maps each original core node through its source-cell/corner
owners into the complete reference. Matching coordinates alone is insufficient:
distinct node sectors can occupy the same geometric corner across a cut or
missing connection. A core node whose owners map to different reference nodes
refuses explicitly. All original-to-reference joins are exported, alongside
the full-reference-induced core connectivity and disjoint source-element cover.

| Geometry quantity | Prepared value |
| --- | ---: |
| Core source elements | 1,203 |
| Exterior source elements | 12,008 |
| Complete reference elements | 13,211 |
| Original core nodes | 1,836 |
| Reference-induced core nodes | 1,832 |
| Shared interface nodes | 956 |
| Free interface DOFs under the outer support convention | 2,868 |
| Exterior-exclusive free DOFs | 27,864 |

Four pairs of original core boundary sectors join through exterior connectivity.
All eight old local nodes were constrained in the clamped 8 mm problem; they
become four shared nodes in the larger model. A coupling witness must use the
reference-induced core topology and retain these explicit identifications. It
cannot insert an interface operator into the original core matrices and assume
that their independent node numbering is already compatible. This preparation
does not merge nonzero episode displacement or Maxwell memory; such history
transfer remains unsupported and requires its own evidence.

At float64 precision, a dense free-interface matrix requires 65,803,392 bytes
(62.76 MiB). Retaining every exterior harmonic-response column would require
639,311,616 bytes (609.69 MiB). Batching 64 columns limits combined RHS/response
arrays to 28,532,736 bytes (27.21 MiB), over 45 batches. These are array-size
estimates, excluding sparse matrices, factor fill, assembly, copies and solver
workspaces; they do not establish runtime or process-memory feasibility.

Seven generated controls pass, including exterior-induced joins, inconsistent
source-owner splits despite equal coordinates, malformed ownership, changed
supports, detached exports and batch bounds. Independent code reproduces the
source partition, all four joins, support counts and storage estimates.

Run from the primary checkout with a fresh ignored output path:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  .venv/bin/python scripts/prepare_continuum_exterior.py \
  testcases/outputs/simulator-qualification-20261029/continuum-exterior-v1/NEW-PREPARATION
```

## Next executable mechanics milestone

Implement and independently verify fixed-topology exterior condensation before
another source response run. With equilibrium stiffness partitioned into
retained core `I` and exterior-exclusive free DOFs `E`, the intended operator is

```text
H = K_II - K_IE K_EE⁻¹ K_EI
```

The declared proportional Maxwell spectrum also requires the corresponding
history forcing and exterior field recovery. An equilibrium-only replacement
does not establish transient or full-energy equivalence. Primitive controls
must check interface force cancellation, core/exterior source ownership,
history recovery and full-domain energy/dissipation against independent dense
block algebra. Then profile exterior assembly/factorization and bounded operator
application on the fixed prepared source partition before freezing a full run.

The initial witness is restricted to the two already exposed fixed-contact
17-action-derived load histories, zero initial fields, unchanged proportional
material spectrum, no body loads and the existing 20 mm outer supports. It must
compare recovered core displacement, memory, reactions and fixed-probe
instantaneous/cumulative strain/stress against the complete 20 mm equations,
with prospective numerical thresholds and independent primitive exports. Cuts,
CUSA removal, changing contacts and nonzero history transfer are later gates;
this preparation grants no shortcut around their ownership checks.

The existing offline reference ceilings remain the proposed upper bounds
(900 seconds, 3 GiB, 32 MiB output, 16,000 source cells, 60,000 DOFs, one numerical
thread, no retry). Live 2,000-cell/24,000-DOF limits remain unchanged. A successful
coupling comparison would reproduce a declared numerical reference; it would
not validate its 20 mm outer boundary, material/device response, patient shift,
vasculature, neurological consequences or physiological RL. Those evidence
gates and the broader scientific goal remain open.

## Exterior-history primitive and setup pilot

`surgical_continuum_exterior.py` now provides an offline batched exterior
operator and proportional-history advancement. It preserves arbitrary
distributed fields in the numerical controls; the forthcoming source witness
remains restricted to its declared fixed topology/contact and zero initial
fields. Its source adapter checks each input's own source-affine corner identity
before assembling the same full-integration equilibrium matrices. Core node
joins still come from the complete reference, never from coordinate-only
history merging.

For common branch ratios `αj`, `rj=τj/(τj+h)`, `β=1+Σαj rj`, and
`a=Σαj rj zj_old`, core advancement uses

```text
(β H + Cᵀ D C) uI = fI − KIE KEE⁻¹ fE + H aI + Cᵀ D target
uE = −KEE⁻¹ KEI uI + (aE + KEE⁻¹ KEI aI + KEE⁻¹ fE) / β
```

Exterior history cancels from the core force but remains in recovered fields
and stored energy. Operators retain no episode state. Exterior support is
checked through symmetric sparse LDL pivots and factor identities; full support
additionally requires positive-definite core Schur admission. Contact springs
cannot supply missing tissue support. Matrix asymmetry, indefinite/singular
support, contacts outside the core and nonfinite data refuse explicitly.

The new primitive reports the actual declared spring potential
`½(Cu−target)ᵀD(Cu−target)`, rather than the live engine's existing force-gauge
quadratic. Each step requires the previous target. Target work is
`[D(target_new−Cu_new)]ᵀ Δtarget`, body work is `f_newᵀ Δu`, and contact numerical
loss is `½Δ(Cu−target)ᵀ D Δ(Cu−target)`. Tissue/Maxwell energy, dissipation and
numerical loss include both core and exterior. A change of contact binding has
its own spring-energy jump; it is not silently included in a fixed-binding
step. These are declared model identities, not measured actuator work. No live
engine equation, historical recording or accepted history-gate result changes.

The focused primitive/continuum regression selection passes 82 tests. Independent
code compares the implementation with complete dense equations over 24 systems
and 288 steps, including zero/one/two branches, arbitrary nonzero exterior
histories, changing body loads/targets, contactless states and reordered core
DOFs. Maximum displacement disagreement is 2.19e-14; physical work-energy
residual is 5.69e-14 in the generated controls' declared units. Five unsupported
matrix controls refuse. The initially identified force-gauge/contact-potential
distinction and its correction remain in ignored review receipts.

The [prospective setup pilot](../manifests/experiments/continuum-exterior-pilot-v1.json)
uses the same frozen 8/20 mm source exports and partition. It assembles and
factors the exterior, including factor-certificate temporary storage, then
times only one 64-column batch. Its ceilings are 180 seconds, 3 GiB sampled
process-group RSS, 8 MiB outputs, one numerical thread and zero retries. It does
not advance a contact-target history or certify the full core problem.

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  .venv/bin/python scripts/profile_continuum_exterior.py \
  testcases/outputs/simulator-qualification-20261029/continuum-exterior-v1/NEW-PILOT
```

This implementation constructs a dense **5,496-DOF core** Schur matrix, not just
the interface block: 241,648,128 bytes for the matrix, plus its Cholesky copy.
Those allocations, the complete 45-batch path, sparse factors/certificates and
field/energy verification require full-path resource admission after the pilot.
The 62.76 MiB interface-only estimate alone cannot establish that feasibility.

The first committed pilot (`9cebb77b`) completes in 5.85 seconds. Assembly/free
partition takes 0.35 seconds, the exterior factor/certificate 4.56 seconds, and
one 64-column batch 0.881 seconds, with maximum residual 6.07e-18 N. A simple
45-batch extrapolation is 39.64 seconds, excluding core factorization and fields.
Peak process RSS is 2,589,409,280 bytes before any dense core allocation. This
admits the pilot, not the complete path: transient full-size `U-D Lᵀ` certificate
storage contributes to that risk. The certificate now checks identical
coefficients in bounded 64-column blocks with the same positivity, permutation,
symmetry and residual tests. Subsequent resource receipts distinguish retained
resident factors from transient peak memory before the full path is admitted.
