# Explicit partial-evacuation volume

The existing cumulative-cut model certifies evacuated subcells but only reports
whole-parent removal volume. The new separately opt-in volume declaration reports
the certified source subset and cumulative target-mass bounds without changing
whole-cell rewards or the conservative shaft obstacle. A target fraction describes
unlocated parent mass, never a target location within a leaf.

For parent volume V, evacuated volume R, and target mass [Tlo, Thi], the reported
target intersection is [max(0, R - (V - Tlo)), min(R, Thi)]. Bounds are recomputed
per parent cumulatively; they are not independent action rewards. Actual action
volume deltas count each certified leaf once, including later parent clearance.
Rejected previews roll back all credit. Probe contacts add none. Old default and
v1 configurations retain their prior serialization and behavior.

The separate checker reconstructs executed leaf geometry and cumulative volumes,
binds explicitly supplied target arrays, and exports checked final geometry only
after a successful complete-history audit. Source-support volume remains distinct
from physical volume clipped by the generated reference evaluator.

## Validation

Root ran the following focused batch on October 11, 2026. Exit status 0:
**168 passed in 4.80 s**. Single numerical thread, plugin autoload disabled,
bytecode/cache output disabled, temporary output inside the shared reservation.

```text
.venv/bin/python -I -B -m pytest -q -p no:cacheprovider \
  --basetemp=build/native-partial-volume-v1/root-controls-01-tmp \
  tests/test_native_evacuated_volume.py tests/test_independent_evacuated_volume.py \
  tests/test_native_cut_union.py tests/test_independent_cut_union.py \
  tests/test_native_cut_union_digest_cache.py tests/test_native_resection.py \
  tests/test_native_immutable_state.py tests/test_native_spatial_task.py \
  build/native-partial-volume-v1/runner/test_runner.py
```

Coverage includes ambiguous target location, rollback, clone/reset, forged
certificates, old configuration parity, independent rejection, and one complete
coarse generated scene through the producer, checker, physical clipping and
compact serialization. Separately, the independent clipping evaluator passed
12 generated controls in 0.003 s. The full fixed 18-row comparison is the next
experiment; no accuracy gain is inferred from these controls.

This remains an idealized cut-and-evacuation assumption. No patient, training,
held-out evaluation, measured cutting-force or neurological-safety evidence is
created by these software tests. No new data or dependencies were acquired.
