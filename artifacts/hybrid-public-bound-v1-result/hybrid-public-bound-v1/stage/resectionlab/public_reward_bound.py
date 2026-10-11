"""Optimistic nominal target potential for straight, full-cell aspiration.

This is a reward upper bound, not a clearance/legality or tissue model. Inputs
are explicit public arrays; no scene, engine, case, reference target or learned
parameters enter here. The index is a snapshot of one committed observation.

Every removed cell lies in a microstep active capsule. All such capsules lie
inside the interval envelope of physical_start -> tip, extended backwards by
tip_length * axis and in every direction by tip_radius. Counting every positive
public-target cell whose entire-cell AABB *intersects* that envelope overcounts
possible removal. Fractional memberships are optimistically counted as one.
Normal-tissue and movement penalties are dropped; action/switch costs remain.
The resulting deliberately loose bound preserves the canonical nominal reward.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Mapping

import numpy as np

VERSION = "public-full-cell-target-envelope-bound-v1"
_EPS = np.finfo(np.float64).eps
MAX_INDEX_CELLS = 1_000_000  # Bounded workspace; refusal retains exact preview.
_FIELDS = {"target_per_mm3", "normal_per_mm3", "motor_per_mm3",
           "language_per_mm3", "action_cost", "motion_per_mm",
           "tool_change_cost", "graph_edge_cost"}


@dataclass(frozen=True)
class RewardUpperBound:
    reward_upper: float | None
    potential_target_cells: int | None
    reason: str

    def cannot_change_winner(self, incumbent_reward: float) -> bool:
        """STOP wins zero ties; all positive ties retain exact raw-order checks."""
        if (not math.isfinite(incumbent_reward) or incumbent_reward < 0
                or self.reward_upper is None):
            return False
        return (self.reward_upper <= 0 if incumbent_reward == 0
                else self.reward_upper < incumbent_reward)


def _immutable(a):
    a = np.ascontiguousarray(a, dtype=np.float64)
    return np.frombuffer(a.tobytes(), dtype=a.dtype).reshape(a.shape)


def _cell_boxes(indices, affine):
    """Outward interval affine images of full index cells, not point samples.

    The extra 64-epsilon arithmetic envelope covers the small fixed sequence
    of binary64 center/corner and distance operations in the native checker.
    It is numerical outward rounding, never a fitted spatial/reward tolerance.
    Inputs outside the declared finite range are refused by prepare/query.
    """
    lower = np.broadcast_to(affine[:3, 3], (len(indices), 3)).copy()
    upper = lower.copy()
    magnitude = np.abs(lower) + 1.
    for axis in range(3):
        a = (indices[:, axis:axis+1] - .5) * affine[:3, axis]
        b = (indices[:, axis:axis+1] + .5) * affine[:3, axis]
        lo, hi = np.minimum(a, b), np.maximum(a, b)
        lower = np.nextafter(lower + np.nextafter(lo, -np.inf), -np.inf)
        upper = np.nextafter(upper + np.nextafter(hi, np.inf), np.inf)
        magnitude += np.maximum(np.abs(a), np.abs(b))
    pad = 64 * _EPS * magnitude
    return _immutable(np.nextafter(lower-pad, -np.inf)), _immutable(np.nextafter(upper+pad, np.inf))


@dataclass(frozen=True)
class PublicTargetBoundIndex:
    target_lower: np.ndarray
    target_upper: np.ndarray
    unknown_lower: np.ndarray
    unknown_upper: np.ndarray
    unavailable_reason: str | None = None

    @classmethod
    def prepare(cls, *, affine, public_target, target_domain, observed_cavity,
                cancelled: Callable[[], None] = lambda: None):
        """Validate once per observed state and snapshot only public cell boxes.

        Cancellation is propagated. Unsupported inputs return an unavailable
        index: callers must retain ordinary exact preview. No hidden support
        membership or coverage inference is permitted.
        """
        cancelled()
        empty = _immutable(np.empty((0, 3)))
        def unavailable(reason):
            return cls(empty, empty, empty, empty, reason)
        if any(x is None for x in (affine, public_target, target_domain, observed_cavity)):
            return unavailable("missing_explicit_public_input")
        a, t, d, c = map(np.asarray, (affine, public_target, target_domain, observed_cavity))
        if (a.shape != (4, 4) or a.dtype.kind not in "fiu" or
                not np.isfinite(a).all() or not np.array_equal(a[3], [0, 0, 0, 1])
                or np.max(np.abs(a)) > 1e6):
            return unavailable("unsupported_affine")
        a = a.astype(np.float64)
        basis = a[:3, :3]
        spacing = np.linalg.norm(basis, axis=0)
        if (np.any(spacing < 1e-6) or np.any(spacing > 1e4)
                or not np.allclose((basis/spacing).T @ (basis/spacing), np.eye(3), rtol=0, atol=1e-8)):
            return unavailable("unsupported_native_frame")
        if (t.ndim != 3 or t.size > 2**31 or min(t.shape, default=0) < 1
                or t.dtype.kind not in "bifu" or d.shape != t.shape or c.shape != t.shape
                or d.dtype != bool or c.dtype != bool):
            return unavailable("unsupported_public_grid")
        if not np.isfinite(t).all() or np.any(t < 0) or np.any(t > 1):
            return unavailable("membership_outside_unit_interval")
        if float(np.max(np.abs(basis)))*max(t.shape)*3 + float(np.max(np.abs(a[:3, 3]))) > 1e9:
            return unavailable("unsupported_coordinate_magnitude")
        cancelled()
        if int(np.count_nonzero((t > 0) & d & ~c)) + int(np.count_nonzero(~d)) > MAX_INDEX_CELLS:
            return unavailable("public_index_workspace_limit")
        target = np.argwhere((t > 0) & d & ~c)
        unknown = np.argwhere(~d)  # Unknown never becomes free through cavity.
        tl, tu = _cell_boxes(target, a)
        cancelled()
        ul, uu = _cell_boxes(unknown, a)
        cancelled()
        return cls(tl, tu, ul, uu)

    def bound(self, *, physical_start_mm, tip_mm, axis_unit, tip_length_mm,
              tip_radius_mm, reward: Mapping[str, float], reward_voxel_volume_mm3,
              tool_change: bool, interaction_mode="aspirate",
              cancelled: Callable[[], None] = lambda: None) -> RewardUpperBound:
        """Full active-sweep AABB potential; outside-image cells earn no reward.

        Unknown *within-grid* target coverage intersecting this envelope makes
        the bound unavailable. Outside-image space cannot contain a native-grid
        removed cell; this says nothing about outside-image physical clearance.
        Motion penalties remain charged by the real scorer, but are not used to
        tighten this bound. Unsupported geometry always retains exact preview.
        """
        cancelled()
        def unavailable(reason):
            return RewardUpperBound(None, None, reason)
        if self.unavailable_reason:
            return unavailable(self.unavailable_reason)
        if interaction_mode != "aspirate" or type(tool_change) is not bool:
            return unavailable("unsupported_interaction")
        if (not isinstance(reward, Mapping) or set(reward) != _FIELDS or any(type(v) not in (float, int) or not math.isfinite(v) or v < 0
                                         for v in reward.values())):
            return unavailable("unsupported_reward")
        try:
            p, q, axis = map(lambda v: np.asarray(v, dtype=np.float64),
                             (physical_start_mm, tip_mm, axis_unit))
        except (TypeError, ValueError):
            return unavailable("unsupported_trajectory")
        if any(v.shape != (3,) or not np.isfinite(v).all() or np.max(np.abs(v)) > 1e6
               for v in (p, q, axis)):
            return unavailable("unsupported_trajectory")
        if (not math.isclose(float(np.linalg.norm(axis)), 1., rel_tol=0, abs_tol=1e-12)
                or any(type(v) not in (float, int) or not math.isfinite(v) or v <= 0
                       for v in (tip_length_mm, tip_radius_mm, reward_voxel_volume_mm3))
                or max(tip_length_mm, tip_radius_mm) > 1e4):
            return unavailable("unsupported_tool_or_volume")
        # Interval of every computed microstep p + (q-p)*t, 0 <= t <= 1.
        delta = q-p
        dl, du = np.nextafter(delta, -np.inf), np.nextafter(delta, np.inf)
        lo = np.nextafter(p + np.minimum(dl, 0), -np.inf)
        hi = np.nextafter(p + np.maximum(du, 0), np.inf)
        back = -float(tip_length_mm)*axis
        lo = np.nextafter(lo + np.minimum(np.nextafter(back, -np.inf), 0), -np.inf)
        hi = np.nextafter(hi + np.maximum(np.nextafter(back, np.inf), 0), np.inf)
        pad = 64*_EPS*(np.abs(p)+np.abs(q)+abs(tip_length_mm)+abs(tip_radius_mm)+1)
        lo = np.nextafter(lo-float(tip_radius_mm)-pad, -np.inf)
        hi = np.nextafter(hi+float(tip_radius_mm)+pad, np.inf)
        intersects = lambda lower, upper: np.all((lower <= hi) & (upper >= lo), axis=1)
        if np.any(intersects(self.unknown_lower, self.unknown_upper)):
            return unavailable("unknown_target_coverage_in_envelope")
        count = int(np.count_nonzero(intersects(self.target_lower, self.target_upper)))
        cancelled()
        # Match canonical multiplication/subtraction ordering, rounded upwards;
        # dropping nonnegative normal/motion penalties can only raise reward.
        upper = np.nextafter(float(count)*float(reward_voxel_volume_mm3), np.inf)
        upper = np.nextafter(float(reward["target_per_mm3"])*upper, np.inf)
        upper = np.nextafter(upper-float(reward["action_cost"]), np.inf)
        upper = np.nextafter(upper-float(reward["tool_change_cost"])*tool_change, np.inf)
        if not math.isfinite(upper):
            return unavailable("nonfinite_reward_bound")
        return RewardUpperBound(float(upper), count, "public_full_cell_envelope")
