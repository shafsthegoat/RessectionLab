# Larger-domain contact-compliance reference

An offline reference now advances the existing voxel continuum through a small
contact system, while retaining the full source-owned displacement field. It
reproduces the detailed equations under explicit restrictions. The larger-domain
study still **fails its frozen radius-stability criterion**: this makes the
boundary error measurable without promoting the model to physical acceptance.

The live engine, its caps, original recordings and physiological RL hold are
unchanged. This reference is executable through
`scripts/qualify_continuum_reference.py`; it is not a new native rollout mode.

## Exact reduction and limits

For a fixed retained mesh and supports, let `K0` be equilibrium stiffness, `C`
the source-cell corner-average contact operator, and `D` the spring stiffness.
When every Maxwell branch satisfies `Kj = αj K0`, initial displacement/memory
fields are zero and there is no body load, define:

```text
B = K0_ff⁻¹ C_fᵀ       S = C B
u = B q                zj = B wj
rj = τj / (τj + h)     β = 1 + Σ rj αj
(β I + D S) q_new = D target + Σ rj αj wj_old
wj_new = (τj wj_old + h q_new) / (τj + h)
```

This is an algebraic invariant subspace, with no fitted response or truncated
modes. The seven held source cells require 21 contact coordinates. Recovered
fields retain every corner and source label. Construction uses bounded temporary
assembly storage and one sparse equilibrium factorization; subsequent history
updates use the small system. A separately bounded detailed solver remains the
disagreement control.

The builder rejects nonproportional spectra, unsupported rigid components,
nonzero initial fields, rigid fragment holds, absent source contacts and body
loads. Operators remain fixed during integration. Changes to cuts, removal,
contact ownership or arbitrary distributed history require another treatment;
this reference does not transfer history through such changes. Its outputs
do not admit native instrument actions or certify their collision feasibility.

## Frozen acquired-source study

Commit `812c6492` freezes the [protocol](../manifests/experiments/continuum-contact-reference-v1.json),
code and tests before the larger-domain response. Geometry-only feasibility
counted domains without a mechanics solve. Every derivative remains the same
DEVELOPMENT UCSF-PDGM-0007 patient. The original archive is reused by SHA256;
there is no new acquisition, material fit or training.

The exact original 15-action prefix has zero initial continuum fields. A
transient observer captures the accepted movement/dwell clock through action
17, requiring equality with the original 4 mm endpoint snapshot. Both complete
target histories are prescribed to the larger fixed domains. Material, contact,
relaxation spectrum and time step remain unchanged assumptions. The reference
does not enlarge the live engine's 2,000-cell budget.

| Offline radius | Source elements | Corner DOFs | Short-dwell reaction | Sustained-dwell reaction |
| --- | ---: | ---: | ---: | ---: |
| 8 mm | 1,203 | 5,508 | 0.8329 mN | 0.7397 mN |
| 12 mm | 3,301 | 13,620 | 0.7758 mN | 0.6852 mN |
| 16 mm | 7,095 | 27,588 | 0.7449 mN | 0.6558 mN |
| 20 mm | 13,211 | 49,260 | 0.7254 mN | 0.6375 mN |

All eight worlds completed. The same 257 source cells define displacement and
Gauss strain/stress probes. Time integrals cover the exact post-prefix load
clock, not a changing whole-domain volume. The original 560 mm³ target and
4,940 mm³ non-target removal remain counted; 18,808 mm³ of target remains.
Endpoints still hold a retractor and are not complete surgical plans.

The declared 16-to-20 mm comparison requires center-field RMS difference
≤0.005 mm and all reaction/instantaneous/cumulative probe changes ≤5%.

| History | Center-field RMS difference | Instantaneous stress² change | Cumulative stress² change | Frozen result |
| --- | ---: | ---: | ---: | --- |
| Short dwell | 0.001873 mm | 5.292% | 5.131% | **Fail** |
| Sustained dwell | 0.001969 mm | 5.679% | 5.484% | **Fail** |

Reaction and strain criteria pass, while both stress criteria fail in both
histories. These are engineering radius-stability limits, without clinical or
metrology meaning. The failed 5% criterion is preserved; neither a new radius
nor a looser threshold is selected from the result. Even a pass would describe
only two finite artificial boundaries on the same voxel grid.

## Verification, timing and storage

At 8 mm, recovered native endpoint displacement/memory errors are at most
2.50×10⁻¹⁶ / 1.81×10⁻¹⁶ mm. Three detailed-versus-reduced history repetitions
also agree within roundoff. Median core integration times, after construction
and with the detailed factor prepared, are 45.83 versus 4.21 ms for 60 steps,
and 210.56 versus 21.00 ms for 276 steps: about 10× faster state advancement.
Reconstruction, field guards, construction and export have separate costs.
These are not complete engine-action, renderer or RL-throughput measurements.

On Apple M2 Pro with 16 GiB RAM and one numerical thread, all reference work
took 55.97 s, with 1.70 GB peak process RSS under the frozen 900 s / 3 GiB caps.
The largest matrix/basis construction took 8.57 s. Compressed geometry, bases,
histories and endpoint fields retain about 13 MB under ignored `testcases/`;
the patient archive is not copied.

The 94 targeted tests pass. Independent review passed 49 numerical and 21
runner-contract controls, then checked all four source domains and 1,344 steps
from exported primitives without importing the engine's element operators.
It independently derived geometry, Q1 stiffness action, compliance, histories,
energy/dissipation, probe rates, reactions and field limits, and reproduced both
failed radius gates. Native archived-endpoint equality is separate runner
evidence; the reviewer did not reopen the original archive or prior artifacts.

The local `testcases/outputs/simulator-qualification-20261029/continuum-reference-v1/README.md`
records exact commands, immutable inputs, all timings, independent hashes and
the scored suite. Run the supervisor with a **new** output path:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  .venv/bin/python scripts/qualify_continuum_reference.py \
  testcases/outputs/simulator-qualification-20261029/continuum-reference-v1/NEW-RUN
```

This establishes numerical reduction fidelity and a usable larger-domain
reference. Boundary adequacy remains failed; measured material/contact response,
accepted acquired-anatomy prediction and patient physiological consequences
remain unqualified. The existing donor force comparison remains mixed/negative.
The environment is not ready for meaningful physiological RL comparisons.
