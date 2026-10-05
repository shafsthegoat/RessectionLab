# Prospective paired scratch input-unit contrast

This declaration compares two fresh policies on the existing native axis action
model: RAW inputs and the existing FEATURE_UNITS registry. It is one seed and one
update on one previously studied development patient. It cannot establish
learning efficacy, convergence, clinical usefulness or a runtime advantage.
Public execution remains held pending review, an immutable tested source and an
explicit execution release.

The machine-readable declaration is
[`native-axis-scratch-feature-units-v1.json`](../manifests/experiments/native-axis-scratch-feature-units-v1.json).
It references the completed RAW pilot's immutable physical, world and six-episode
protocol. **The historical RAW result is cost evidence only. Both comparison
arms are new runs from the same source snapshot.**

Run RAW, then FEATURE_UNITS, in separate cold worker processes. Each uses seed 11,
15 action features, six state features, 16 hidden units and fresh Adam state.
Initial trainable tensors must have the same declared digest. Behavior hashes
remain distinct because FEATURE_UNITS adds the registered nontrainable float32
divisor buffer. The runner authenticates the actual fresh initial checkpoint at
the first scored factory call, before any transition or gradient, without an
extra policy forward or random draw. Profile metadata, divisor bytes, observation
semantics and complete optimization/selection seed vectors are checked together.
This file check occurs after fresh Adam construction, before its first update;
its time is charged inside the initial selection panel. Model, registry and
world mismatches are rejected before the learner is called.

The exact axis backend, source cells, 13-column/26-primary proposal rule, fallback,
ordering, three-cut horizon, tools, access, reward, partial-contact cost and
zero-perturbation world generator stay fixed. Cache is OFF. Inputs remain RAW in
the simulator; the actor alone applies the registered fixed divisors. No running
statistics, centering, clipping, return scaling, critic scaling, coordinates,
pretraining, new patient or new world family enters this contrast. The 648 mm³
unit is a fixed reference from the prior registry, not an upper bound on actions.

Each arm performs:

1. One unscored zero-transition shape probe.
2. Initial deterministic selection on both prescribed selection seeds.
3. Two complete stochastic optimization episodes on the first two prescribed
   optimization seeds, then exactly one Adam update.
4. Updated deterministic selection on the same two selection seeds.
5. Freeze actual initial/latest/selected tensors and all six histories, then
   independently audit every history. Exact-history reuse is allowed within an
   arm only; there is no audit cache shared between arms.

All panels and the update must finish. Earliest checkpoint wins a selection tie.
Latest traces bind latest tensors even when initial is selected. Record initial,
latest and selected panel means; report latest−initial and selected−initial
separately, plus their paired contrasts. A worsening update remains visible even
when selection retains the initial checkpoint. An incomplete arm prevents a
completed paired comparison; preserve both planned-arm denominators and every
attempted receipt. No retries, resumption or automatic cap enlargement occur.

Each arm keeps the original 300-second online cap, 600-second complete worker cap,
6 GiB measured RSS limit and 15-second termination grace. The worker cap includes
independent audits and publication. The prior RAW pilot used 169.590 seconds
online, 221.139 seconds overall and 2.219 GiB peak RSS. Those measurements support
feasibility for its observed path only. Scaled routes and the new schema/profile
checks can cost differently. Preserve actual overruns and failures.

Report cold construction, setup, the seven factory clones, initial/latest
selection, online work, full trainer call, final checkpoint export, independent
audits, worker and launcher times separately. These intervals overlap; do not add
nested measurements. The fixed order, OS file cache and ordinary desktop load
limit timing comparisons. Two deterministic selection replays are not two
independent patients or uncertainty samples.

The runner is
[`run_native_axis_feature_units.py`](../scripts/run_native_axis_feature_units.py).
Default invocation writes the pinned declaration only; `--execute` is required.
No result file alone grants eligibility: completed arm worker/launcher authorities
and a final paired authority must bind the exact candidate, checkpoint, journal
and source bytes. Missing motor/language evidence, estimated unreviewed support
and hypothetical access retain their existing limitations.
