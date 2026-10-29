# Explicit bipolar heat storage and conduction

`bipolarThermalV1` adds an opt-in, source-owned heat model to V2 scenes. Actual
closed-electrode contacts deposit a declared total heat power; retained tissue
conducts heat and continues warming or redistributing it after energy stops.
Coagulation, vessel patency and neurological injury remain separate, unvalidated
mechanisms. The new model is not enabled in historical recordings.

## Evidence and assumptions

[Mohammadi et al., Table 3](https://pmc.ncbi.nlm.nih.gov/articles/PMC8235733/)
reports measured ex vivo calf-brain thermal properties. The generated scenario
uses the 33°C values: volumetric capacity 3.83 MJ/(m³·K), converted to
0.00383 J/(mm³·K), and conductivity 0.553 W/(m·K), converted to
0.000553 W/(mm·K). Their ratio gives diffusivity 0.144386 mm²/s. These are
source-backed material scenarios, not live human, glioma or vessel estimates.
Constant coefficients, a 33°C baseline and uniform properties across all patch
labels—including vessel labels—remain explicit assumptions.

[Delorme et al.](https://doi.org/10.3233/978-1-60750-706-2-166) separated
nominal bipolar settings from the heating needed to reproduce their bovine
experiment. Their result does not calibrate this simulator's generic forceps.
Our **0.02 W per activation unit** is declared as an assumed delivered heat
coefficient; the existing `power` control is not silently relabeled as watts.
Heat is divided across the contacted source cells with a fixed total, rather
than increasing with voxel count. No electrical-current or impedance solve is
claimed.

[Chen et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC3779387/) describe
device-dependent thermal spread and delayed remote warming after activation
ends. Their tissue preparation, instruments and cooling differ from this
coupon. No published curve was digitized or fitted, and no quantitative
device-response agreement is claimed.

The [protocol](../manifests/experiments/bipolar-thermal-v1.json) separates
these sources, material scenarios, assumed delivery, executable tests and
unsupported claims. Device calibration and patient thermal response remain
Level B/C gaps.

## Source geometry and conservation

A scene explicitly lists at most 4096 original tissue cells and assigns each
patch material a heat capacity and conductivity. Heating outside that coverage
refuses the transition. Rotated, reflected and anisotropic orthogonal grids are
supported; sheared grids refuse the two-point face approximation.

For excess temperature θ relative to the declared baseline:

```text
Cᵢ = volumetric_capacityᵢ × source_voxel_volume        [J/K]
Gᵢⱼ = harmonic_conductivityᵢⱼ × face_area / distance   [W/K]
Cᵢ (θᵢ,new − θᵢ,old) / Δt
    = Pᵢ − Σⱼ Gᵢⱼ(θᵢ,new − θⱼ,new) − Hᵢ θᵢ,new

deposited_energy = retained_excess_heat + bath_export + removed_excess_heat
```

Backward Euler gives a passive implicit solve. Cut or removed source faces
cannot conduct through a cavity. Warm removal exports its excess heat once;
cumulative temperature exposure retains original source identity. Optional
surface exchange and a retained exterior fixed at ambient are explicit bath
assumptions with separately counted energy. An ambient exterior is an infinite
reservoir, so patch-size sensitivity remains necessary before patient use.
The demonstration instead uses a completely insulated 15-cell patch.

The operator uses fixed reference geometry. Deformation-dependent transport,
perfusion, irrigation, evaporation and electrode heat removal are absent. A
declared maximum temperature refuses the action without clipping; it is a
numerical domain limit, not an injury threshold. Temperature exposure is
integrated in mm³·K·s with no biological threshold or deficit probability.

Persistently activated instruments share the same clock during inspection
regardless of which hand owns the inspect action. This correction applies to
the new opt-in model; historical scheduling semantics remain unchanged.
Direct and indirect protected-source heating, invalid coverage and temperature
domain failures preserve the exact accepted action prefix. Unqualified thermal
scenes are refused by the existing RL reward.

## Executed comparison

The generated source-bound case has four replayable branches at a common
40-second observation horizon. All three resecting branches remove the same
nine original target cells (9 mm³), no non-target tissue, and leave 41 mm³.
STOP removes nothing and leaves all 50 mm³. Tools finish withdrawn.

| Branch | Deposited heat | Peak modeled temperature | Final maximum | Proxy arterial occlusion |
| --- | ---: | ---: | ---: | --- |
| No heat | 0 J | 33.00°C | 33.00°C | No |
| Continuous, 2.1 s | 0.042 J | 36.70°C | 33.7817°C | Yes |
| Two 1.05 s pulses, 6 s apart | 0.042 J | 35.73°C | 33.7917°C | Yes |
| STOP with residual | 0 J | 33.00°C | 33.00°C | No |

Both heated branches retain essentially all 0.042 J: the lower pulsed peak is
redistribution and timing, not net cooling. Its final maximum is slightly
higher. Their cumulative volume-temperature-time values are 414.517 and
381.046 mm³·K·s; the difference follows delivery timing in this insulated,
uniform-capacity patch and does not independently validate spatial spread.
Maximum energy-budget residual is below 8×10⁻¹⁷ J.

The separate existing power-time coagulation proxy causes the reported
occlusion. Those modeled temperatures do not establish coagulation, tissue
injury or successful hemostasis. The discrepancy exposes a missing device
calibration; temperature and patency have not been made interchangeable.
Differences in vascular flow deficits also include the later timing of the
second pulse.

## Verification and remaining gates

An independently assembled dense finite-volume oracle checked 18 heterogeneous,
anisotropic, rotated/reflected, bath, cut, removal and cache-reuse transitions.
Maximum temperature difference was 1.12×10⁻¹⁵ K. Nine independent public
controls covered shared clocks, two active hands, accepted-prefix rollback,
unknown coverage, indirect protection, previews/cancellation, restoration,
source mutation and RL refusal. A present-null declaration check was added
after review; both discovered defects were resolved with evidence preserved.

Against an analytic two-cell transient, timestep errors decrease from
0.01536 to 0.00389 K for 0.1 → 0.025 s. A manufactured insulated rod's spatial
error decreases from 0.00356 to 0.000234 K for 1 → 0.25 mm at a fixed 0.001 s
step. These are numerical diffusion checks, not measured temperature errors.

Targeted Python checks, all 1119 desktop tests and desktop build pass. Separate
export/telemetry checks retain heat units and the unknown neurological endpoint.
Hidden native checks verify all branch endpoints, source counts, GPU/HUD state,
fixed microscope optics, heat readouts, save and cold reopen. The original
120-action acquired CUSA episode and 29-action continuum episode retain their
exact full snapshots and source arrays.

Local cases, failed setup attempts, commands, numerical controls, independent
review and native captures remain under ignored
`testcases/outputs/simulator-qualification-20261029/bipolar-thermal-v1/`.
No new scientific dataset was acquired. Two closed temporary browser profiles
were removed, reclaiming about 26 MB; recordings and research evidence remain.
Electrical delivery, patient thermal response, temperature-to-coagulation,
injury and neurological outcomes remain unvalidated. Physiological RL stays on
hold.
