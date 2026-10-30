# Finite-blade temporal correction and rehearsal cleanup

The review of `2e88f45` reproduced on the current branch: duplicate rehearsal
workflows and accepted-state cards during preview, two frozen work gates failing
at 10.0844% / 11.4132%, and canonical work changing by 56.7599% on timestep halving.

The corrected **frozen numerical protocol passes all 24 histories / 960 prescribed
steps**, including its original work, force, displacement, rotation, dense-oracle,
energy and resource gates. The actual canonical maneuver passes both adjacent
work comparisons across three resolutions. This is bounded numerical verification
of generated coupons. General physical qualification, patient prediction, clinical
validation and RL admission remain unavailable.

## Rehearsal UI

Environment setup retains readiness, protection, mechanics configuration, patient
position, previous environments and material mapping. Duplicate planning, manual
actions, exchange, withdrawal evidence, preview/result/stimulation records, timeline,
undo/reset controls and legacy metric cards were removed from that disclosure.
The authoritative outcome summary reads `preview ?? view`; missing measurements
remain **Unknown**. Mechanics, provenance and diagnostics retain their disclosures.
Preview/cancel, replay, blocked readiness, partial outcomes and refusals stay visible.

The regression renders the whole panel, including mounted children of closed
disclosures. It checks unique workflows and every tagged measurement in accepted,
preview and missing-measurement states, including 1.50 mm³ preview removal over
0.00 mm³ accepted removal. It failed on duplicate manual actions before the cleanup.

## Cause and correction

Original endpoint backward-Euler work is `sum(lambda_new * Delta offset)`. Its
discrete energy identity closes but includes first-order numerical loss. Rapid
unloading and separation turn recoverable elastic energy into that loss. In the
canonical 0.025 s run, numerical dissipation was 1.63713338e-6 N·mm out of
2.20059979e-6 N·mm net work; activation dissipation was roundoff. The 56.76%
figure describes **work sensitivity**: hold reaction changed only 0.11046%, while
peak displacement remained 0.02 mm.

The correction retains the material equations, Signorini constraints and signed
endpoint work. Bounded step doubling integrates each prescribed linear blade
segment and accepts an actual finer BE history. Every internal solve passes the
original KKT, source, support, gradient and energy checks. Work and dissipation
are summed over that accepted history; endpoint forces, gaps and state come from
its last solve.

Work agreement alone is insufficient at separation: coarse and doubled trials
can both land after release and falsely agree at zero work. A regression reproduces
this alias. The controller independently bounds numerical plus activation loss,
which exposes the missed loaded interval. Per-node vector norms keep state-error
decisions invariant under rigid RAS rotation. A development controller using
coordinate-component norms failed six frozen rotation gates; those failures remain
in the evidence directory.

At the 0.025 s reference interval, work-error and artificial-loss budgets are
`1e-9 N·mm + 1e-4 * max stored energy`; displacement/memory error budget is
`1e-6 mm + 1e-4 * max nodal motion`. Budgets scale with
`(requested_dt / 0.025)^2`, matching BE local-error order and requiring finer
internal accuracy as requested dt shrinks. These controls supplement the original
qualification limits. Trials refuse after 2,048 subdivisions or **15 s total per
prescribed step**. The existing 120 prescribed-step, 600 s run and 768 MiB limits
remain binding. Receipts expose accepted subdivisions, all trial evaluations,
internal dt, error estimates, tolerances, artificial loss and wall time.

New declarations bind `schedule.integration = work-controlled-backward-euler-v1`.
Declarations without that field retain original BE and exact model identity, so
saved states and replay retain their recorded numerics. Source arrays, material
assumptions, supports, footprint, physical trajectory, durations and frozen
thresholds are unchanged.

## Frozen benchmark

4³-cell Maxwell coupons. Work is N·mm; force is the final hold reaction in µN.
Runtime includes assembly and checks for that history.

| Coupon | dt (s) | Work before | Work after | Hold force before → after (µN) | Runtime before → after (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Uniform | 0.1 | 4.87077142e-6 | 3.53163174e-6 | 330.833722 → 330.084142 | 0.085 → 0.802 |
| Uniform | 0.05 | 4.11916477e-6 | 3.44421719e-6 | 330.123896 → 329.728549 | 0.151 → 1.677 |
| Uniform | 0.025 | 3.74182482e-6 | 3.40080749e-6 | 329.749497 → 329.539312 | 0.270 → 3.469 |
| Partial | 0.1 | 1.35453889e-6 | 9.95164371e-7 | 170.664899 → 170.273069 | 0.055 → 0.260 |
| Partial | 0.05 | 1.12171646e-6 | 9.34323421e-7 | 170.298725 → 170.094999 | 0.091 → 0.557 |
| Partial | 0.025 | 1.00680727e-6 | 9.10187925e-7 | 170.105587 → 169.998156 | 0.164 → 1.154 |

Maximum nodal displacement remains 0.04000000 mm for uniform and 0.02000194 mm
for partial coupons at all three resolutions, before and after. Absolute differences
use `abs(Wcoarse-Wfine)`; relative differences divide by `abs(Wfine)`.

| Coupon / refinement | Absolute work difference before → after (N·mm) | Relative difference before → after |
| --- | ---: | ---: |
| Uniform 0.1→0.05 | 7.516067e-7 → 8.741455e-8 | 18.2466% → 2.5380% |
| Uniform 0.05→0.025 | 3.773400e-7 → 4.340970e-8 | **10.0844% → 1.2765%** |
| Partial 0.1→0.05 | 2.328224e-7 → 6.084095e-8 | 20.7559% → 6.5118% |
| Partial 0.05→0.025 | 1.149092e-7 → 2.413550e-8 | **11.4132% → 2.6517%** |

The frozen 5% work criterion scores **0.05→0.025 s**. The coarser partial pair
still changes by 6.5118%; it is reported without relabeling it as a scored pass.
Both final hold-force refinement changes are below 0.058% (2% gate).
Maximum complete-patch penetration is 1.324e-10 mm (1e-9 limit), equilibrium
residual 3.97e-19 N (1e-10 limit), and discrete energy imbalance 3.55e-14 N·mm
(1e-10 limit). Dense-oracle, analytical uniform recurrence on the accepted partition,
rotation, source ownership and seven refusal controls pass.

Fresh-process wall time is **25.106→50.084 s**; peak memory **189.016→284.859 MiB**.
The frozen manifest SHA256 remains
`b1d8e7c18ddbc1288cc50585b2f280e4c57c3eed20caf524ccfe2b61e0525b15`.

## Actual canonical approach/indent/hold/withdraw

The diagnostic invokes actual `SurgicalEngine.step` actions at three resolutions,
with unchanged 0.1/0.2/0.2/0.2 s phases and −0.1→0→0.02→0.02→−0.1 mm offsets.
Only an isolated worker varies observation dt; production retains 0.025 s and
28 prescribed increments. An independent scalar BE oracle checks reaction,
displacement and work on the accepted partitions.

| Method | dt (s) | Net work (N·mm) | Peak / hold reaction (µN) | Final displacement (mm) | Max step energy imbalance (N·mm) | Runtime (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Original | 0.025 | 2.20059979e-6 | 224.632128 / 203.745568 | 0.00235031 | 2.70e-21 | 0.121 |
| Original | 0.0125 | 1.40380299e-6 | 225.275013 / 203.970876 | 0.00237830 | 4.61e-21 | 0.159 |
| Original | 0.00625 | 1.00247435e-6 | 225.603185 / 204.084296 | 0.00239274 | 3.93e-21 | 0.294 |
| Corrected | 0.025 | 5.82989459e-7 | 225.904593 / 204.232197 | 0.00240896 | 1.66e-18 | 4.010 |
| Corrected | 0.0125 | 5.79162694e-7 | 225.920940 / 204.215696 | 0.00240913 | 7.53e-19 | 7.918 |
| Corrected | 0.00625 | 5.77480483e-7 | 225.928628 / 204.207079 | 0.00240917 | 1.29e-18 | 16.201 |

Peak displacement remains 0.02 mm in every run. Every run ends separated with
zero reaction. Corrected hold reaction changes by 0.00808% and 0.00422%.

| Refinement | Absolute work difference before → after (N·mm) | Relative difference before → after |
| --- | ---: | ---: |
| 0.025→0.0125 s | 7.967968e-7 → 3.826765e-9 | **56.7599% → 0.6607%** |
| 0.0125→0.00625 s | 4.013286e-7 → 1.682211e-9 | **40.0338% → 0.2913%** |

At production dt, corrected final stored energy is 6.96367721e-8 N·mm, viscous
dissipation 5.04742323e-7, numerical dissipation 8.61036424e-9 and activation loss
2.23e-18. Their sum equals signed work within 2.21e-18 N·mm.

An independent continuous-time uniaxial solution integrates the contact ramps
and hold analytically, locates release at 0.02828496193 s after withdrawal starts,
and integrates unloaded Maxwell recovery. It gives net work **5.75953235e-7 N·mm**,
hold force 204.198216 µN and final displacement 0.00240918060 mm. Corrected work
errors against this solution are **1.2217%, 0.5572%, 0.2652%**. Its independently
integrated viscous loss and stored energy close within 1.02e-20 N·mm. This reference
checks whether adjacent refinements merely agree with each other.

## Evidence and remaining limits

Verification passes 219 selected Python tests and a separate 61-test control
using committed versions of the checkout's unrelated unfinished continuum,
engine and motion modules. Desktop verification passes 318 main-process,
627 renderer and 254 viewer tests, plus the production build. The native bridge
check passes preview/commit isolation, source preservation, exact restore/replay,
save/cold reopen and the independent scalar endpoint oracle. Historical integration
declarations also replay their original full snapshots exactly.

Original and development failures remain under ignored
`testcases/outputs/qualification-correction-20261030/`. Final numerical reports are
`before-frozen.json`, `after-frozen-final.json` and `temporal-final.json`, with raw
states, forces, ledgers and content-addressed implementation snapshots. The native
bridge check uses a separate generated derivative carrying the new method binding;
source arrays and original archives remain unchanged.

Legacy declarations retain their original failing work gates. No scored failure
remains in the corrected bounded protocol. The coarsest partial-work refinement
is still above 5%, so 0.1 s is not established as work-converged. Acquired-source
admission, current-configuration edge migration, broader continuum boundary/stress
limitations, patient material fitting and physiological/clinical validation remain
unresolved. The reference footprint and fixed small-strain topology have not expanded.
Refinement materially increases runtime and can explicitly refuse at its existing
time/cache limits or new subdivision cap. No real-time claim, route generation,
additional physics or RL training is included. Existing unqualified-model disclosures
remain in place.

```sh
.venv/bin/python scripts/qualify_planar_blade_contact.py \
  --integration backward-euler --output testcases/outputs/NEW-before.json
.venv/bin/python scripts/qualify_planar_blade_contact.py \
  --output testcases/outputs/NEW-after.json
.venv/bin/python scripts/check_planar_blade_temporal.py \
  --output testcases/outputs/NEW-temporal.json
```

Use new output names to preserve evidence. Runners use fresh single-thread workers
and generated coupons. Wall times include instrumentation and are observations,
not performance guarantees.
