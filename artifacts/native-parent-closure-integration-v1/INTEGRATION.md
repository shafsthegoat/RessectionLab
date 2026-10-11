# Same-capsule parent exposure closure

The optional `CutUnionSpec(exposure_mode='same_step_parent_closure')`
revisits partial tissue exposed by a fully cleared parent within the same
executed active capsule. A newly completed parent can expose its neighbor;
partial parents and unconnected empty pockets cannot. This addresses the
saved aligned-suction diagnosis: 63.65% of its remaining target-volume gap is
geometrically contained but uncertified. Final-state exposure alone does not
reconstruct the exact original microstep timing.

The default mode and its serialization remain unchanged. Depth, work limits,
reward rules, prior-cavity shaft checks and whole-parent removal connectivity
stay fixed. The new mode has a distinct declaration, history footprint and
state identity. The independent checker verifies every added leaf against the
executed capsule and requires a supporting cleared parent to have complete
geometric and prior-free-space connectivity evidence before state mutation.

Root's focused batch passes **203 tests in 4.84 s**. It covers default parity,
same-step exposure, complete-parent chains, reversed witness order, false
bridges, future cuts, isolated removals, shaft collisions, volume accounting,
clone/reset and rollback after resource refusal. The command and both root
logs are retained locally under `build/native-cut-residual-diagnostic-v1`.
The first batch had seven invalid-fixture step-size failures; a focused repair
then exposed two additional fixture mistakes (unreachable endpoint and a cap
before any productive cut). Corrected fixtures preserve the production limits;
all original failed logs remain. No acceptance threshold was relaxed.

This is source/numerical verification of an idealized generated cutting model.
No learned-policy improvement, patient result, force validation or clinical
safety claim follows from these controls.

## Measured failure and targeted scheduling change

The first full comparison completed in 64.860 s but failed the original useful
criterion. Four fine-grid target volumes improved, including aligned suction
from 0.280835 to 0.363794 mm³. Two fine-grid aspirator first strokes exceeded the
unchanged 8,192-witness limit, reducing completed sequences from 13 to 11. The
independent saved audit passes 21,571 checks and preserves this negative result.
This is not a speed improvement: the failed sequences executed less work.

The next optional declaration adds
`closure_schedule='terminal_microstep_only'`. Intermediate microsteps use the
legacy exposure rule; only the actual final microstep performs same-capsule
closure. It addresses the measured final-exposure lag while avoiding many
small intermediate witness additions. It never raises or redefines the limit,
uses target-dependent scheduling, or relaxes shaft checks. The checker derives
terminal position from the concrete history and rejects early closure even if
a witness claims to be terminal. Earlier modes remain available unchanged.

The combined terminal-schedule batch passes **233 tests in 5.04 s**, including
exact intermediate witnesses and all-capsule predecessor parity, independent
early-closure attacks, and the original comparison criteria. Its fixed18 result
is still pending at this source checkpoint. The original interrupted monitor
attempt and complete negative comparison remain local and unmodified.
