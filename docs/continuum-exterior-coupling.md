# Fixed-domain exterior preparation

The [distributed-history qualification](continuum-history-solve.md) now passes
its numerical gate. The next mechanics milestone targets the rigid artificial
core clamp, while retaining the already studied 8 mm core and 20 mm outer
domain. Both earlier 16-to-20 mm stress-stability gates remain failed. No new
radius, material fit, source response, patient role or live cap is selected.

An offline fixed-domain exterior/history witness now passes its numerical
qualification. Its geometry preparation remains a separate step without a
mechanical response: `scripts/prepare_continuum_exterior.py` reads only six geometry/support
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

## Declared coupling and scope

The verified fixed-topology condensation partitions equilibrium stiffness into
retained core `I` and exterior-exclusive free DOFs `E`, giving

```text
H = K_II - K_IE K_EE⁻¹ K_EI
```

The declared proportional Maxwell spectrum requires the corresponding
history forcing and exterior field recovery. An equilibrium-only replacement
does not establish transient or full-energy equivalence. Primitive controls
check source ownership, history recovery and full-domain energy/dissipation
against independent dense block algebra. Frozen source samples independently
check interface force cancellation and full-domain energy. Assembly/factor and
bounded operator application are profiled before the complete source run.

The completed witness is restricted to the two already exposed fixed-contact
17-action-derived load histories, zero initial fields, unchanged proportional
material spectrum, no body loads and the existing 20 mm outer supports. It
compares recovered core displacement, memory, reactions and fixed-probe
instantaneous/cumulative strain/stress against the complete 20 mm equations,
with prospective numerical thresholds and independent primitive exports. Cuts,
CUSA removal, changing contacts and nonzero history transfer are later gates;
this witness grants no shortcut around their ownership checks.

The qualification retains the existing offline reference ceilings
(900 seconds, 3 GiB, 32 MiB output, 16,000 source cells, 60,000 DOFs, one numerical
thread, no retry). Live 2,000-cell/24,000-DOF limits remain unchanged. The passed
coupling comparison reproduces a declared numerical reference; it does
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

The distinct second pilot at `2a53b11f` completes in 5.98 seconds. Peak process
RSS falls to 1,533,493,248 bytes with the same 6.07e-18 N basis residual. Retained
RSS is 1,290,256,384 bytes after factor admission and 1,533,444,096 after the
first batch. That batch takes 0.930 seconds, implying 41.86 seconds across 45
batches before core/field costs. These measurements support a separately frozen
full-path witness under the original offline 900-second/3-GiB ceiling.

The [complete source witness](../manifests/experiments/continuum-exterior-qualification-v1.json)
binds both original 60/276-step schedules, complete-20 mm reference fields,
geometry and partition hashes before execution. It compares all 336 full free
displacement/memory states, per-contact and aggregate reaction, actual
tissue/Maxwell/spring energy, work/dissipation and the fixed 257-cell probe.
Instantaneous and cumulative quantities retain separate unit-specific absolute
budgets; the failed 5% radius gates are not changed. Six predetermined
before/after field samples support independent quadrature, interface-force and
energy checks. Identical common-prefix sample payloads are reused by SHA256.

The qualification selection passes 85 tests, including a homogeneous affine
physical-gradient control in Cartesian and oblique reflected coordinates and
an analytical moving-spring energy example. Volume integration uses only the
eight Gauss points; eight additional corner points guard the gradient domain
without adding volume. The controls caught and corrected an initial doubled
exposure integral before any full source execution; its failed log is retained.

The independent export checker and its generated selftest are SHA256-bound in
the source protocol before execution. It independently integrates full Q1
tissue/Maxwell and actual spring energy, backward-Euler work/loss, free
equilibrium and core/exterior interface traction cancellation at the six
predetermined steps. It also checks all 336 energy/reaction/probe-rate records
against the existing complete-reference coordinates and compliance, plus exact
JSON/NPZ agreement, schedule counts, unit-specific budgets and cumulative
accounting. Its generated selftest includes an analytical shear field and four
corruption/refusal controls. Shared identical sample files retain separate
branch/step records and must have the same content hash.

The worker releases its large factors before the independent export-only stage.
Both run serially within one 900-second wall and 32-MiB output budget, with each
process group capped at 3 GiB. The checker receives at most 60 seconds from the
remaining wall budget. A completed mechanics worker alone cannot accept this
gate; the frozen independent checker must also pass on the final report hash.

## Completed fixed-domain source qualification

The frozen witness at `2e2de647` passes in **140.40 seconds**, including the
independent checker. The mechanics worker takes 137.03 seconds; independent
verification takes 2.78 seconds. Peak worker RSS is **1,620,672,512 bytes**
(1.51 GiB); sampled worker group peak is 1,614,282,752 bytes. The checker group
peaks at 86,523,904 sampled bytes after the mechanics process has exited.
Retained output is **4,482,443 bytes**, within the shared 32-MiB ceiling.

All 336 complete free displacement/memory states reproduce the original 20 mm
reference, including every per-contact/aggregate reaction, actual model energy,
work/loss and fixed-source exposure comparison. Source inputs and code hashes
remain unchanged. The six predetermined sample worlds reuse four identical
payloads where their source histories have a common prefix; branch/step
receipts remain separate. No field projection or history reduction is fitted.

| Quantity | Maximum disagreement | Frozen absolute budget |
| --- | ---: | ---: |
| Full free displacement | 3.05e-16 mm | 1e-9 mm |
| Full free Maxwell memory | 2.64e-16 mm | 1e-9 mm |
| Per-contact/aggregate reaction | 4.34e-19 N | 1e-10 N |
| Energy/work/loss quantities | 7.46e-20 N mm | 1e-10 N mm |
| Instantaneous fixed-probe strain² volume | 4.51e-17 mm³ | 1e-9 mm³ |
| Cumulative fixed-probe strain² | 6.94e-17 mm³ s | 1e-9 mm³ s |
| Instantaneous fixed-probe stress² volume | 5.10e-11 Pa² mm³ | 0.001 Pa² mm³ |
| Cumulative fixed-probe stress² | 1.17e-10 Pa² mm³ s | 0.001 Pa² mm³ s |

Independent source-owner Q1 quadrature passes all six samples. Maximum free
equilibrium residual is 6.44e-19 N, core/exterior interface imbalance 2.40e-19 N,
and actual spring/tissue/Maxwell work-energy residual 4.58e-20 N mm. Independent
coordinate/compliance algebra checks all 336 energy/reaction/probe records; its
maximum stored-energy difference is 6.10e-20 N mm. The JSON and NPZ records agree
exactly, and all sampled state statistics bind to their respective history rows.
A final receipt-only audit passes 26 provenance/scope checks, including all six
committed source hashes, immutable inputs, the executed checker/selftest binding,
resource ceilings, four node joins and unchanged live/history code. The checker
itself is retained under ignored evidence and bound by the committed protocol's
SHA256; its contents are not claimed to be a Git blob.

The source setup includes 5.58 seconds for exterior factor/certificate, 43.29
seconds for all Schur columns and 1.44 seconds for core factor/contact basis.
The 60/276-step histories take 15.36/70.78 seconds including full field guards,
reference comparisons, energy accounting and exports. These research-verifier
costs do not establish live instrument latency or planning throughput.

The already exposed clamped-8 reference remains a descriptive control:

| Program | Clamped-8 reaction | Coupled/reference-20 reaction | Fixed-probe stress² change |
| --- | ---: | ---: | ---: |
| Short hold | 0.8329 mN | 0.7254 mN | −26.97% |
| Sustained hold | 0.7397 mN | 0.6375 mN | −28.67% |

Those contrasts reuse the frozen original reference report and the same 257
source probe cells; they are neither new measurement units nor patient evidence.
Numerical coupling removes the arbitrary inner clamp relative to the declared
20 mm model. The 20 mm *outer* support convention remains unaccepted, and both
original 16-to-20 mm stress-stability failures are still failures.

## Next executable gate

Extend the offline witness to the complete original public programs with
changing contacts, CUSA removal, relaxation and release. First verify
source-corner history transfer across splits/joins, retained-interface support
identity, removed tissue energy and explicit contact-binding energy jumps on
small controls. Profile operator updates before freezing a complete source run;
all original actions and physical refusals must remain counted. No parameter,
radius, threshold or action adjustment is justified merely to obtain completion.
Prescribing events captured from native programs is a mechanics witness; actual
larger-domain public-action/collision admission requires separate verification.

The complete operator has 33,360 free DOFs, including 27,864 exterior-only DOFs.
Counting only the 5,496 core DOFs would conceal its state/resource cost and
cannot admit it under the live 24,000-DOF cap. Any later live integration needs
its own explicit domain/resource admission; this offline witness provides none.
Measured tissue/device response, source-supported vascular physiology,
functional consequences and acquired-anatomy validation remain open. Both
physiological and large-scale RL stay on hold, and the scientific goal remains
active.
