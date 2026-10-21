At saved primary position **566**, generated bone debris is **5,192 mm³**:
**5,014.897 mm³** remains and **177.103 mm³** has been aspirated. The ledger
balances to roundoff; fluid and wounds are zero. The currently active right
Frazier field at `[-9.464996337890625, 30.441162109375, -418.25]` contains
**zero remaining debris**, so continued stationary activity there cannot clean
the newer field.

The smallest useful repair is a bounded resampling of that same already-opened
right-hand ray, after guarded left-hand withdrawal/parking. Keep the existing
Frazier and its original zero-force, radius-3 mm, flow-20 mm³/s research settings.
Public preview/step remains mandatory for every motion, dwell and withdrawal.

1. Deactivate instruments, finish guarded left CUSA withdrawal, and park its
   complete shaft clear. Preserve the right instrument's existing insertion.
2. Move the right Frazier upward on its existing XY ray to these candidate
   heights **in increasing Z order**. Use ordinary checked `apply` actions,
   keeping quaternion `[1, 0, 0, 0]`:

   | RAS Z (mm), XY unchanged | Current nearby debris (mm³) |
   |---|---:|
   | -415.25 | 134.769 |
   | -411.25 | 250.769 |
   | -407.25 | 197.744 |

   Start with one station; allow at most three. Remeasure each field after the
   preceding action because the spheres overlap. Use at most 10 seconds of
   stationary `apply` per station and at most 30 seconds total dwell. The
   natural dwell bound is current local debris / 20 mm³/s plus 0.2 seconds,
   capped at 10 seconds. Stop a station when the actual removed-debris ledger
   does not increase.
3. Deactivate right, inspect the remaining field and actual generated/removed/
   present/escaped debris ledger, shrink its research field radius to 0.5 mm,
   and perform the normal checked withdrawal. Inspect once both hands are
   inactive and withdrawn. Report actual aspiration and retained debris.

The static geometry filter found zero retained source cells in the 0.67 mm
superior tip/shaft ray to the exterior for all three positions, and no current
other-hand envelope conflict. This is candidate evidence, **not** an admitted
public transition. The current engine and protection checks still decide.

The three spheres contain a union of approximately **425.282 mm³** of current
debris. This is an overlap-adjusted geometric upper bound, not a promise that
the bounded dwell will remove all of it, and it addresses only part of the
remaining 5,014.897 mm³. The JSON station list is ranked by incremental debris;
the execution order above is sorted along the superior ray and requires fresh
dwell measurements. Unreachable or unattempted debris must remain recorded.

Irrigation is unnecessary to aspirate dry debris under the existing law:
suction directly spends its shared flow budget on fluid first, then debris.
Adding irrigation would consume part of that budget and may redistribute
sediment through fluid relaxation. There is no bulk-clear operation and none
is proposed.

The [JSON note](field-cleanup-recommendation-20261020T130825Z.json) records the
captured native hash/state, per-height masses and static checks. This review
executed no patient physics, preview, step, replay or GUI and changed no
production code, source arrays or historic field ledger. All values are
uncalibrated model quantities, not clinical suction/force/outcome predictions.
