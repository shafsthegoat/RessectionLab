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
