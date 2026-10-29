# Simulator qualification before physiological planning

The qualification runner preserves the current engine and recordings while
checking what an episode can support. Its frozen
[contract](../manifests/experiments/simulator-qualification-v1.json) specifies a
post-exposure, finite-CUSA development task, mechanism gaps, validation questions
and numerical tolerances. It grants no held-out payload access, training
admission, physiological calibration or clinical outcome claim.

`scripts/qualify_surgical_simulator.py` inventories native-study metadata without
decoding image arrays. Patient versions remain separate. Series/layer identity,
declared units/affines, registration status, source hashes, review status,
vascular modality and typed labels are inspectable. Missing layer units remain
unknown even when the reference image declares millimetres. Diffusion and
diagnostic tensor scalars do not establish functional pathways. This inventory
does not requalify absent historical catalogs or legacy-only bundles.

An optional baseline replay admits the entire frozen archive by SHA-256 before
loading any primary or auxiliary payload. It reconstructs public actions from
intact source anatomy and compares the full final state. An independent checker
counts original source labels, checks uniqueness/ownership/monotonic removal
after every action, and reports exposure, setup, finite work and cleanup costs.
The historical authored opening remains a cost and an explicit assumption.
It is not a measured or clinically planned exposure.

Run with existing local data and a new ignored output directory:

```sh
.venv/bin/python scripts/qualify_surgical_simulator.py \
  --native-root "$CASE_ROOT" --native-root "$LEGACY_NATIVE_ROOT" \
  --case "$FROZEN_BASELINE_ARCHIVE" --mechanics \
  --output "$NEW_OUTPUT_UNDER_TESTCASES"
```

The local `testcases/README.md` points to exact case paths, commands, report
versions and test logs. Outputs are compact JSON; the runner copies no patient
datasets or workspaces. `coverage.json`, `baseline.json`, `mechanics.json` and
`validation-suite.json` distinguish numerical passes, partial evidence, failed
grid invariance and unvalidated physical mechanisms. A baseline pass applies
only to canonical replay and source accounting. An episode with zero active
deformation substeps has not exercised deformation, even if all diagnostic
residuals are zero.

## Observing the existing lattice

`surgical_deformation_v2.diagnostic_scope()` collects constant-size aggregate
counts outside canonical state: target/force/displacement caps, strain
projection, collision rejection, free-node residual and discrete energy defect.
Its scope includes attempted and rolled-back solves; it is not an injury ledger.
Nested scopes and exceptions restore the prior observer. Unobserved engine
arithmetic and recorded state remain unchanged.

The residual uses the actual system, including hold-spring stiffness. For a
constant target and zero fixed anchors, with `A = K_tissue + K_hold`,
`d = u_new - u_old`, and nodal drag `c`, the independent identity is:

```text
r = A u_new + (c/h) d - f
f·d - Δ(½uᵀAu) - (c/h)||d||² - ½dᵀAd = -r·d
```

The quadratic hold potential is included. This is solver verification, not
measured retractor work or a constitutive energy-damage law. Collision rejection
can have zero displacement/work but nonzero residual, so residuals and
projection counts must be read together. Changing prescribed anchor positions
would require separate constraint-work accounting.

`scripts/qualify_surgical_mechanics.py` checks a one-node response against both
an independently derived backward-Euler recurrence and its continuous solution.
A fixed-physical-cube energy check uses independent bond/foundation counts at
three spacings. It deliberately exposes the unchanged profile's grid dependence;
it does not tune coefficients to hide it. The cube excludes the bulk penalty to
isolate these terms. It is not a complete mesh-convergence assessment of patient
mechanics; its finest static kernel audit exceeds the runtime patch-node cap
and is not offered as a runnable patient patch. Tests separately force all five
cap/projection/rejection mechanisms.

## What the human measurements support

[Budday et al.](https://doi.org/10.1016/j.jmbbm.2017.07.014) measured postmortem
human tissue under controlled shear, axial loading and relaxation. Their finite
viscoelastic identification supports testing loading history and force relaxation.
It does not identify this app's absolute nodal drag or a surgical CUSA rate.
[Hinrichsen et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC10511383/) provide
region-dependent, bonded-specimen axial/torsional measurements; processed
hyperelastic curves do not identify relaxation time constants. Neither provides
glioma cutting/separation or living-patient retractor-injury calibration.

The existing [specimen calibration result](../artifacts/hbe-v5-axial-calibration-result-v1/RESULT.md)
retains large axial residuals. Its separate FEBio numerical controls do not
validate the app lattice. The preserved
[torsion format failure](../artifacts/hbe-v5-torsion-format-failure-v1/REPORT.txt)
is not a held-out torque-validation pass.

A separate [donor-held-out continuum comparison](continuum-donor-transfer.md)
now uses a prospectively selected motor-cortex specimen and a development-only
effective stiffness fit. The calibrated continuum beats both default-lattice
variants, but loses to a zero-force baseline in tension. Numerical admission,
independent scoring checks and this mixed measured result remain distinct;
physical-response acceptance is still open.

[ReMIND](https://www.cancerimagingarchive.net/collection/remind/) and
[RESECT](https://doi.org/10.1002/mp.12268) can support appropriately framed paired
anatomical observation tests. Missing maneuver records prevent interpreting
anatomical agreement as causal validation of an instrument. Existing
TRAIN/SELECT/MEASUREMENT_EVAL and Case4 boundary/calibration/validation
reservations remain intact. Prospective evaluation candidates in the contract
are metadata-only; actual acquisition/admission, measurement coverage and metric
freezes remain open.

An [opt-in voxel continuum](voxel-continuum.md) now provides a source-owned
discretization, material/history scenarios, numerical controls and a limited
exposed-specimen comparison. Independent measured-response and patient-anatomy
gates remain open. An [opt-in vascular graph](vascular-graph.md) now separates
compression, occlusion, transection, sealing and declared supply/drainage
signals, with independent numerical checks and matched generated maneuvers.
Acquired vascular connectivity and measured flow response remain unvalidated.
Separate motor/language evidence, device calibration, responsive viewer and
bounded RL comparison remain subsequent open stages. Large-scale RL stays on hold.

The [opt-in bipolar heat model](bipolar-thermal.md) now budgets delivered heat,
passive conduction, declared bath exchange and heat exported with removal.
Continuous/pulsed generated controls replay with distinct temperature histories.
Temperature-to-coagulation, device response and injury remain unvalidated;
independent numerical checks do not promote those gates.
