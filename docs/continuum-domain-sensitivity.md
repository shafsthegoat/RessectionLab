# Artificial-boundary sensitivity of recorded retraction

The acquired-source continuum's retractor force and stress depend materially on
its artificial patch boundary. The recorded comparison remains conditional on
its original 4 mm patch and declared relaxation scenarios. Neither exact replay
nor the prior affine-element checks establish that this boundary is adequate.

The new audit also found and fixed silent source-cell budget truncation: an
opt-in continuum action now rejects and rolls back if activation cannot cover
its declared physical radius. The legacy lattice retains its previous behavior.
The continuum is now explicitly refused by the legacy scalar reward tracker,
which omits its cumulative strain/stress consequences and qualification limits.

## Frozen experiment

The [prospective protocol](../manifests/experiments/continuum-domain-sensitivity-v1.json)
and runner were committed in `3edd9956` before this replay. They bind the
unchanged UCSF-PDGM-0007 continuum archive by SHA256 and scene identity. The
experiment changes only the physical patch radius to 4, 8 and 12 mm, within the
existing profile domain. Material, relaxation spectrum, source labels/affine,
seven held cells, tool geometry, actions, time steps, collision guards and all
cell/DOF budgets remain fixed. This is one DEVELOPMENT patient in six numerical
worlds, with no new acquisition, fit or training.

Each of the two recorded histories runs from intact anatomy through action 17:
the exact 15-action exposure/acquisition prefix, the original 0.1 mm movement,
and either the short or sustained dwell. These endpoints still hold a retractor.
They are field probes, not complete surgical plans. Completed probes undergo a
separate canonical replay with exact state, action and history equality.

All 257 active source cells in the original common prefix define a fixed probe,
including that patch's artificial boundary cells. The audit compares center
displacements and instantaneous Gauss-integrated strain²/stress² on these same
cells. Whole-patch cumulative exposure is retained separately: changing its
integration domain prevents treating it as the same fixed-support quantity.
All probes retain the shared 560 mm³ target and 4,940 mm³ non-target removal;
18,808 mm³ of annotated target remains.

The prospective stability test compares 8 with 12 mm, requiring center-field
RMS difference ≤0.005 mm and reaction/fixed-support strain²/stress² relative
changes ≤5%. These are engineering numerical criteria, with no clinical or
experimental metrology meaning. Passing would establish only stability across
those two finite radii. Every declared world must remain present; a refusal
cannot become zero or a successful smaller experiment.

## Executed result

Both 4 mm and both 8 mm probes completed and replayed exactly. Active source
cells increased from 257 to 1,203. Both 12 mm worlds refused acquisition at
action 15 with `CONTINUUM_PATCH_TRUNCATED`; their accepted 14-action prefixes
were preserved exactly. The unchanged 2,000-cell cap was not expanded. The
prespecified 8-to-12 mm stability result is therefore **not estimable** for
both histories.

The completed 4-to-8 mm contrast is descriptive, separate from that unfulfilled
stability test:

| Probe endpoint | Reaction, 4 → 8 mm | Fixed-source stress² integral, 4 → 8 mm | Center-field RMS difference |
| --- | ---: | ---: | ---: |
| Short dwell | 0.9815 → 0.8329 mN | 177,244 → 99,214 Pa²·mm³ | 0.008567 mm |
| Sustained dwell | 0.8855 → 0.7397 mN | 143,932 → 77,573 Pa²·mm³ | 0.009178 mm |

Reaction decreased 15.1–16.5%, and the fixed-source instantaneous stress²
integral decreased 44.0–46.1%. Fixed-source strain² decreased 7.4–9.8%.
These are modeled responses to artificial constraints, not observed patient
forces, displacement errors, biological damage or calibrated uncertainty.
The audit does not select a new radius or retune the model from these results.

Execution took 44.98 s with 639 MB peak process RSS; this includes replay and
export and is not a rollout-throughput benchmark. Compressed primitive fields
and native snapshots are retained under ignored
`testcases/outputs/simulator-qualification-20261029/continuum-domain-v1/`.

## Verification and remaining qualification

An independent checker passed 24 synthetic analytical/adversarial controls and
reconstructed all four completed worlds from saved primitives. It derives Q1
quadrature without importing the engine's element operators. Center fields and
spring reactions agree exactly; strain² and stress² integrals agree within
1.53×10⁻¹⁶ mm³ and 6.70×10⁻¹⁰ Pa²·mm³, respectively. All six declared worlds,
source/action/support bindings and both negative outcomes remain counted.

The targeted mechanics, tools, position, reward, protection and bridge-data
regressions pass 276 tests. The local README records exact commands, immutable
inputs, independent evidence and complete original-recording replay results.

Numerical patch adequacy remains open. A later extension needs its own frozen
domain/reference, resource and field-convergence study. The existing donor
force comparison remains mixed/negative, and no new measured response,
accepted acquired-anatomy prediction or patient-specific vascular/functional
validation was added. Patient mechanics and physiological RL remain on hold.

A separately frozen [offline contact-compliance study](continuum-contact-reference.md)
now computes 8/12/16/20 mm reference domains without changing the live caps.
The reduction reproduces detailed histories, but both 16-to-20 mm stress tests
exceed the unchanged 5% stability limit. This resolves reference execution,
while leaving artificial-boundary adequacy failed.
