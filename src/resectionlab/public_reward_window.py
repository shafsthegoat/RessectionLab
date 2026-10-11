"""Conservative public target potential using bounded native-index windows.

This replaces global unknown-cell box enumeration with two immutable boolean
snapshots. It accepts only explicit public target, coverage and observed cavity.
Unknown coverage never becomes empty, even when cavity is true. The window is
an overestimate of the full active sweep; this is not a geometry/clearance test.
Frozen policy features and learned weights are not changed.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Mapping

import numpy as np

from .public_reward_bound import RewardUpperBound

VERSION = "public-full-cell-target-window-bound-v2"
_EPS = np.finfo(np.float64).eps
_FIELDS = {"target_per_mm3", "normal_per_mm3", "motor_per_mm3",
           "language_per_mm3", "action_cost", "motion_per_mm",
           "tool_change_cost", "graph_edge_cost"}
MAX_SNAPSHOT_BYTES = 64 * 1024**2
MAX_WINDOW_CELLS = 1_000_000
MAX_QUERY_WORKSPACE_BYTES = 8 * 1024**2
VALIDATION_CHUNK_CELLS = 65536
# Immutable conversion briefly retains both mutable and immutable snapshots.
MAX_PREPARE_WORKSPACE_BYTES = 2 * MAX_SNAPSHOT_BYTES + 8 * 1024**2


def _immutable(a):
    a = np.ascontiguousarray(a)
    return np.frombuffer(a.tobytes(), dtype=a.dtype).reshape(a.shape)


@dataclass(frozen=True)
class PublicTargetWindow:
    target_positive: np.ndarray
    target_domain: np.ndarray
    inverse_lower: np.ndarray
    inverse_upper: np.ndarray
    origin: np.ndarray
    target_cells: int = 0
    unknown_cells: int = 0
    snapshot_bytes: int = 0
    unavailable_reason: str | None = None

    @classmethod
    def prepare(cls, *, affine, public_target, target_domain, observed_cavity,
                cancelled: Callable[[], None] = lambda: None):
        """Validate and copy bounded public state; no scene/private fields.

        Resident payload is two bool grids plus 21 doubles (168 bytes). Numeric
        validation uses fixed 65,536-cell chunks; it never enumerates unknown
        coordinates or allocates an array of global cell boxes. Unsupported input
        returns an unavailable snapshot. Cancellation always propagates.
        """
        cancelled()
        empty = _immutable(np.empty((0, 0, 0), bool))
        empty3 = _immutable(np.empty((0, 3), float))
        def unavailable(reason):
            return cls(empty, empty, empty3, empty3, _immutable(np.zeros(3)),
                       unavailable_reason=reason)
        if any(v is None for v in (affine, public_target, target_domain, observed_cavity)):
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
        if (t.ndim != 3 or min(t.shape, default=0) < 1 or t.dtype.kind not in "bifu"
                or d.shape != t.shape or c.shape != t.shape or d.dtype != bool or c.dtype != bool):
            return unavailable("unsupported_public_grid")
        size = int(t.size)
        if 2*size + 168 > MAX_SNAPSHOT_BYTES:
            return unavailable("public_snapshot_byte_limit")
        if float(np.max(np.abs(basis)))*max(t.shape)*3 + float(np.max(np.abs(a[:3, 3]))) > 1e9:
            return unavailable("unsupported_coordinate_magnitude")
        # Certify a coefficient interval for the exact inverse of the supplied
        # binary64 affine. E = I - B*A; ||A^-1-B||inf <= ||E||inf*||B||inf/(1-||E||inf).
        # The 64-epsilon envelope includes the three-term dot/subtraction/sums;
        # near-singular or poorly resolved inverses are refused, never guessed.
        try:
            inv = np.linalg.inv(basis)
        except np.linalg.LinAlgError:
            return unavailable("uncertified_inverse")
        if not np.isfinite(inv).all():
            return unavailable("uncertified_inverse")
        residual = np.abs(np.eye(3)-inv@basis) + 64*_EPS*(np.abs(inv)@np.abs(basis)+1.)
        rho = float(np.nextafter(np.max(np.sum(residual, axis=1)), np.inf))
        norm = float(np.nextafter(np.max(np.sum(np.abs(inv), axis=1))*(1+64*_EPS), np.inf))
        condition = norm*float(np.max(np.sum(np.abs(basis), axis=1)))*(1+64*_EPS)
        if not math.isfinite(rho) or rho >= 1e-8 or condition > 1e8:
            return unavailable("uncertified_inverse")
        error = float(np.nextafter((rho/(1-rho))*norm*(1+64*_EPS), np.inf))
        il = _immutable(np.nextafter(inv-error, -np.inf))
        iu = _immutable(np.nextafter(inv+error, np.inf))
        positive = np.empty(t.shape, bool)
        domain = np.empty(t.shape, bool)
        # flat slicing makes bounded copies even for non-contiguous input views.
        pf, df = positive.reshape(-1), domain.reshape(-1)
        for offset in range(0, size, VALIDATION_CHUNK_CELLS):
            cancelled()
            stop = min(offset+VALIDATION_CHUNK_CELLS, size)
            tv, dv, cv = t.flat[offset:stop], d.flat[offset:stop], c.flat[offset:stop]
            if not np.isfinite(tv).all() or np.any(tv < 0) or np.any(tv > 1):
                return unavailable("membership_outside_unit_interval")
            pf[offset:stop] = (tv > 0) & dv & ~cv
            df[offset:stop] = dv
        target_cells = int(np.count_nonzero(positive))
        unknown_cells = size-int(np.count_nonzero(domain))
        cancelled()
        return cls(_immutable(positive), _immutable(domain), il, iu,
                   _immutable(a[:3, 3]), target_cells, unknown_cells, 2*size+168)

    def _window(self, world_lo, world_hi):
        """Outward interval inverse of world AABB, expanded by half a cell.

        A removed cell is fully contained in an active capsule, hence its center
        is in this interval; the additional half-cell expansion also retains
        cells that merely intersect it. Rotated/reflected grids are permitted.
        No rounded point or sampled-path approximation can shrink the window.
        """
        wl = np.nextafter(world_lo-self.origin, -np.inf)
        wh = np.nextafter(world_hi-self.origin, np.inf)
        lower, upper = np.zeros(3), np.zeros(3)
        for j in range(3):
            products = np.stack((self.inverse_lower[:, j]*wl[j], self.inverse_lower[:, j]*wh[j],
                                 self.inverse_upper[:, j]*wl[j], self.inverse_upper[:, j]*wh[j]))
            lower = np.nextafter(lower+np.nextafter(np.min(products, axis=0), -np.inf), -np.inf)
            upper = np.nextafter(upper+np.nextafter(np.max(products, axis=0), np.inf), np.inf)
        lower = np.nextafter(lower-.5, -np.inf)
        upper = np.nextafter(upper+.5, np.inf)
        if not np.isfinite(lower).all() or not np.isfinite(upper).all():
            return None, "nonfinite_index_window"
        starts, stops = [], []
        for j, n in enumerate(self.target_domain.shape):
            start = max(0, min(n, math.ceil(float(lower[j]))))
            stop = max(0, min(n, math.floor(float(upper[j]))+1))
            starts.append(start); stops.append(max(start, stop))
        cells = math.prod(b-a for a, b in zip(starts, stops))
        # Counting/coverage operations use bool views (no coordinate boxes).
        # This explicit pessimistic 8B/cell ceiling bounds reduction workspace.
        if cells > MAX_WINDOW_CELLS or 8*cells > MAX_QUERY_WORKSPACE_BYTES:
            return None, "public_window_workspace_limit"
        return tuple(slice(a, b) for a, b in zip(starts, stops)), None

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
        window, reason = self._window(lo, hi)
        if reason is not None:
            return unavailable(reason)
        cancelled()
        # The inverse AABB window includes every intersecting full affine cell.
        # Unknown coverage anywhere in that overestimate retains exact preview.
        if not np.all(self.target_domain[window]):
            return unavailable("unknown_target_coverage_in_window")
        count = int(np.count_nonzero(self.target_positive[window]))
        cancelled()
        # Match canonical multiplication/subtraction ordering, rounded upwards;
        # dropping nonnegative normal/motion penalties can only raise reward.
        upper = np.nextafter(float(count)*float(reward_voxel_volume_mm3), np.inf)
        upper = np.nextafter(float(reward["target_per_mm3"])*upper, np.inf)
        upper = np.nextafter(upper-float(reward["action_cost"]), np.inf)
        upper = np.nextafter(upper-float(reward["tool_change_cost"])*tool_change, np.inf)
        if not math.isfinite(upper):
            return unavailable("nonfinite_reward_bound")
        return RewardUpperBound(float(upper), count, "public_full_cell_window")
