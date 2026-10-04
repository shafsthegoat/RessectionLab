"""Conservative rigid-instrument geometry in the image's physical millimeter frame.

Voxel occupancy means the *whole affine voxel cell*, not just its center. This
checker computes exact capsule distances to orthogonal affine cells and
circumscribes sheared cells by physical spheres. Exposure is a union-of-cells upper bound,
never removed tissue. Independent final evaluation uses a separate checker.

Motion is linear tip translation and shortest-arc axis rotation. Every interval
is enclosed by a midpoint tool with a proved displacement inflation. Sampling
alone is never used to certify clearance. No curved centerline is a rigid tool.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Any, Iterable

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.spatial import cKDTree

GEOMETRY_VERSION = "rigid-capsules-exact-orthogonal-cells-v2"
_EPS = 1e-9
_CLEARANCE_CAP_MM = 10.0


class _ImmutableArray(np.ndarray):
    """Protect both buffer data and interpretation metadata of frozen inputs."""

    def __setattr__(self, name: str, value: Any) -> None:
        if name in {"shape", "dtype", "strides", "data"}:
            raise ValueError(f"Immutable geometry array {name} cannot be reassigned")
        super().__setattr__(name, value)


def _immutable(array: NDArray) -> NDArray:
    """Use an immutable backing buffer, so writeability cannot be re-enabled."""
    value = np.ascontiguousarray(array)
    return np.frombuffer(value.tobytes(), dtype=value.dtype).reshape(value.shape).view(_ImmutableArray)


def _vector(value: ArrayLike, name: str, *, unit: bool = False) -> NDArray[np.float64]:
    result = np.array(value, dtype=float, copy=True)
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite three-vector")
    if unit:
        norm = float(np.linalg.norm(result))
        if norm < _EPS:
            raise ValueError(f"{name} must have a nonzero direction")
        result /= norm
    return _immutable(result)


def _positive(value: float, name: str) -> None:
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")


def _scalar(value: Any, name: str) -> float:
    array = np.asarray(value)
    if array.shape != () or array.dtype.kind not in "iuf" or not np.isfinite(array):
        raise ValueError(f"{name} must be a finite numeric scalar")
    return float(array)


def _axis_angle(first: NDArray, second: NDArray) -> float:
    # atan2 retains tiny rotations that acos(dot) rounds to zero. Their distal
    # effect can still matter for a long shaft near a small obstacle.
    return float(np.arctan2(np.linalg.norm(np.cross(first, second)), np.clip(first @ second, -1, 1)))


@dataclass(frozen=True)
class ToolGeometry:
    """Generic two-capsule tool; dimensions are research assumptions.

    ``working_length_mm`` is the centerline distance from proximal shaft end
    to distal tip. The active distal capsule has ``tip_length_mm`` centerline
    length. Spherical capsule ends extend beyond their centerline endpoints.
    """

    tool_id: str
    tip_radius_mm: float
    shaft_radius_mm: float
    working_length_mm: float
    max_access_angle_deg: float = 45.0
    tip_length_mm: float = 2.0
    parameter_source: str = "generic research geometry; not a verified commercial device"

    def __post_init__(self) -> None:
        if not isinstance(self.tool_id, str) or not self.tool_id:
            raise ValueError("tool_id is required")
        for name in ("tip_radius_mm", "shaft_radius_mm", "working_length_mm", "tip_length_mm"):
            object.__setattr__(self, name, _scalar(getattr(self, name), name))
            _positive(getattr(self, name), name)
        object.__setattr__(self, "max_access_angle_deg", _scalar(self.max_access_angle_deg, "max_access_angle_deg"))
        if not isinstance(self.parameter_source, str) or not self.parameter_source:
            raise ValueError("parameter_source must be a nonempty string")
        if self.tip_length_mm >= self.working_length_mm:
            raise ValueError("tip_length_mm must be shorter than working_length_mm")
        if not np.isfinite(self.max_access_angle_deg) or not 0 <= self.max_access_angle_deg < 90:
            raise ValueError("max_access_angle_deg must be in [0, 90)")

    @property
    def envelope_radius_mm(self) -> float:
        return max(self.tip_radius_mm, self.shaft_radius_mm)


GENERIC_TOOLS = (
    ToolGeometry("generic_suction", 1.0, 1.4, 100.0, tip_length_mm=3.0),
    ToolGeometry("generic_aspirator", 1.7, 2.7, 90.0, tip_length_mm=4.0),
)


@dataclass(frozen=True)
class ToolPose:
    """Tip and normalized axis pointing from the access side toward the tip."""

    tip_mm: ArrayLike
    axis_unit: ArrayLike

    def __post_init__(self) -> None:
        object.__setattr__(self, "tip_mm", _vector(self.tip_mm, "tip_mm"))
        object.__setattr__(self, "axis_unit", _vector(self.axis_unit, "axis_unit", unit=True))

    def proximal_mm(self, tool: ToolGeometry) -> NDArray[np.float64]:
        return self.tip_mm - tool.working_length_mm * self.axis_unit


@dataclass(frozen=True)
class AccessWindow:
    """Hypothetical circular intracranial aperture, with its inward normal.

    This is an aperture constraint only: it does not create a free corridor.
    Whole-head/skull feasibility is not implied by this model.
    """

    center_mm: ArrayLike
    normal_inward: ArrayLike
    radius_mm: float
    window_id: str = "hypothetical-access"

    def __post_init__(self) -> None:
        object.__setattr__(self, "center_mm", _vector(self.center_mm, "center_mm"))
        object.__setattr__(self, "normal_inward", _vector(self.normal_inward, "normal_inward", unit=True))
        object.__setattr__(self, "radius_mm", _scalar(self.radius_mm, "radius_mm"))
        _positive(self.radius_mm, "radius_mm")
        if not isinstance(self.window_id, str) or not self.window_id:
            raise ValueError("window_id must be a nonempty string")


@dataclass(frozen=True)
class SphereObstacle:
    center_mm: ArrayLike
    radius_mm: float
    obstacle_id: str = "analytic-obstacle"

    def __post_init__(self) -> None:
        object.__setattr__(self, "center_mm", _vector(self.center_mm, "center_mm"))
        object.__setattr__(self, "radius_mm", _scalar(self.radius_mm, "radius_mm"))
        _positive(self.radius_mm, "radius_mm")
        if not isinstance(self.obstacle_id, str) or not self.obstacle_id:
            raise ValueError("obstacle_id must be a nonempty string")


@dataclass(frozen=True)
class GeometryFailure:
    reason: str
    position_mm: tuple[float, float, float]
    detail: str
    motion_fraction: float | None = None


@dataclass(frozen=True)
class GeometryResult:
    feasible: bool
    failures: tuple[GeometryFailure, ...]
    clearance_mm: float
    swept_voxel_indices: NDArray[np.int64]
    exposure_volume_mm3: float | None
    unknowns: tuple[str, ...] = ()
    geometry_version: str = GEOMETRY_VERSION
    exposure_definition: str = "unique affine cells in conservative full-tool envelope; upper bound, not removal"
    clearance_definition: str = "conservative lower bound capped at 10 mm; contact is rejected"

    def to_dict(self, *, include_voxels: bool = False) -> dict[str, Any]:
        result = {
            "feasible": self.feasible,
            "failures": [vars(failure) for failure in self.failures],
            "clearance_mm": self.clearance_mm,
            "exposure_volume_mm3": self.exposure_volume_mm3,
            "unknowns": list(self.unknowns),
            "geometry_version": self.geometry_version,
            "exposure_definition": self.exposure_definition,
            "clearance_definition": self.clearance_definition,
        }
        if include_voxels:
            result["swept_voxel_indices"] = self.swept_voxel_indices.tolist()
        return result


@dataclass(frozen=True)
class GeometryScene:
    """Frozen masks and their authoritative affine (voxel centers to mm).

    ``forbidden_mask`` identifies modeled hard exclusions. Ordinary tissue is
    separately supplied as ``exposure_mask``; this class never clears/removes it.
    Bounds gate the distal tip by default. Proximal geometry outside the image
    is explicitly unassessed because an intracranial window is not a skull model.
    """

    forbidden_mask: ArrayLike
    affine: ArrayLike
    exposure_mask: ArrayLike | None = None
    sphere_obstacles: tuple[SphereObstacle, ...] = ()
    enforce_tip_in_bounds: bool = True
    unknowns: tuple[str, ...] = ()
    _inverse: NDArray[np.float64] = field(init=False, repr=False, compare=False)
    _cell_radius_mm: float = field(init=False, repr=False)
    _forbidden_indices: NDArray[np.int64] = field(init=False, repr=False, compare=False)
    _forbidden_centers: NDArray[np.float64] = field(init=False, repr=False, compare=False)
    _tree: cKDTree | None = field(init=False, repr=False, compare=False)
    _orthogonal_spacing: NDArray[np.float64] | None = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        mask = np.array(self.forbidden_mask, dtype=bool, copy=True)
        if mask.ndim != 3 or min(mask.shape) < 1:
            raise ValueError("forbidden_mask must be a nonempty 3-D mask")
        affine = np.array(self.affine, dtype=float, copy=True)
        if affine.shape != (4, 4) or not np.all(np.isfinite(affine)):
            raise ValueError("affine must be a finite 4x4 matrix")
        if not np.allclose(affine[3], [0, 0, 0, 1], atol=1e-10, rtol=0):
            raise ValueError("affine must have homogeneous last row [0,0,0,1]")
        if abs(np.linalg.det(affine[:3, :3])) < 1e-12:
            raise ValueError("affine must be invertible")
        exposure = None
        if self.exposure_mask is not None:
            exposure = np.array(self.exposure_mask, dtype=bool, copy=True)
            if exposure.shape != mask.shape:
                raise ValueError("exposure_mask and forbidden_mask must share a grid")
            exposure = _immutable(exposure)
        mask = _immutable(mask)
        affine = _immutable(affine)
        inverse = _immutable(np.linalg.inv(affine))
        corners = np.array(list(product((-0.5, 0.5), repeat=3))) @ affine[:3, :3].T
        radius = float(np.linalg.norm(corners, axis=1).max())
        indices = _immutable(np.argwhere(mask))
        centers = _immutable(indices @ affine[:3, :3].T + affine[:3, 3])
        spacing = np.linalg.norm(affine[:3, :3], axis=0)
        directions = affine[:3, :3] / spacing
        orthogonal_spacing = _immutable(spacing) if np.allclose(directions.T @ directions, np.eye(3), rtol=0, atol=1e-10) else None
        object.__setattr__(self, "forbidden_mask", mask)
        object.__setattr__(self, "affine", affine)
        object.__setattr__(self, "exposure_mask", exposure)
        object.__setattr__(self, "sphere_obstacles", tuple(self.sphere_obstacles))
        object.__setattr__(self, "unknowns", tuple(self.unknowns))
        object.__setattr__(self, "_inverse", inverse)
        object.__setattr__(self, "_cell_radius_mm", radius)
        object.__setattr__(self, "_forbidden_indices", indices)
        object.__setattr__(self, "_forbidden_centers", centers)
        object.__setattr__(self, "_tree", cKDTree(centers) if len(centers) else None)
        object.__setattr__(self, "_orthogonal_spacing", orthogonal_spacing)

    @property
    def voxel_volume_mm3(self) -> float:
        return abs(float(np.linalg.det(self.affine[:3, :3])))

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.forbidden_mask.shape

    def world_to_voxel(self, points_mm: ArrayLike) -> NDArray[np.float64]:
        return np.asarray(points_mm) @ self._inverse[:3, :3].T + self._inverse[:3, 3]

    def voxel_to_world(self, indices: ArrayLike) -> NDArray[np.float64]:
        return np.asarray(indices) @ self.affine[:3, :3].T + self.affine[:3, 3]

    @classmethod
    def from_case(cls, case: Any, *, forbidden_mask: ArrayLike | None = None,
                  exposure_mask: ArrayLike | None = None, **kwargs: Any) -> GeometryScene:
        mask = np.zeros(case.mri.shape, dtype=bool) if forbidden_mask is None else forbidden_mask
        return cls(mask, case.affine, exposure_mask=exposure_mask, **kwargs)


def point_segment_distances(points: ArrayLike, start: ArrayLike, end: ArrayLike) -> NDArray[np.float64]:
    """Euclidean distances to a finite segment, in the supplied physical units."""
    points, start, end = np.asarray(points, float), np.asarray(start, float), np.asarray(end, float)
    delta = end - start
    length2 = float(delta @ delta)
    fraction = np.zeros(points.shape[:-1]) if length2 <= _EPS**2 else np.clip((points - start) @ delta / length2, 0, 1)
    return np.linalg.norm(points - (start + np.expand_dims(fraction, -1) * delta), axis=-1)


def segment_segment_distance(a0: ArrayLike, a1: ArrayLike, b0: ArrayLike, b1: ArrayLike) -> float:
    """Exact finite-segment separation, including parallel/degenerate cases."""
    a0, a1, b0, b1 = map(lambda x: np.asarray(x, float), (a0, a1, b0, b1))
    u, v, w = a1 - a0, b1 - b0, a0 - b0
    aa, bb, cc, dd, ee = float(u @ u), float(u @ v), float(v @ v), float(u @ w), float(v @ w)
    candidates = [float(point_segment_distances(p, q0, q1)) for p, q0, q1 in
                  ((a0, b0, b1), (a1, b0, b1), (b0, a0, a1), (b1, a0, a1))]
    determinant = aa * cc - bb * bb
    if determinant > _EPS**2 * max(aa * cc, 1.0):
        s, t = (bb * ee - cc * dd) / determinant, (aa * ee - bb * dd) / determinant
        if 0 <= s <= 1 and 0 <= t <= 1:
            candidates.append(float(np.linalg.norm(w + s * u - t * v)))
    return min(candidates)


def tool_capsules(tool: ToolGeometry, pose: ToolPose, inflation_mm: float = 0.0
                  ) -> tuple[tuple[NDArray, NDArray, float, str], ...]:
    """Proximal shaft and distal active tip capsules forming one rigid object."""
    if not np.isfinite(inflation_mm) or inflation_mm < 0:
        raise ValueError("inflation_mm must be finite and nonnegative")
    shoulder = pose.tip_mm - tool.tip_length_mm * pose.axis_unit
    return ((pose.proximal_mm(tool), shoulder, tool.shaft_radius_mm + inflation_mm, "shaft"),
            (shoulder, pose.tip_mm, tool.tip_radius_mm + inflation_mm, "tip"))


def _segment_cell_distances(scene: GeometryScene, indices: NDArray, start: NDArray,
                            end: NDArray) -> NDArray[np.float64]:
    """Distances to orthogonal voxel boxes; a lower bound for sheared cells.

    In a box-aligned physical frame the squared segment-to-box distance is a
    convex piecewise quadratic in segment fraction. Crossing the six box faces
    partitions [0,1]; minimize each interval's quadratic and keep the minimum.
    This uses full cells and handles segments, points, and parallel faces alike.
    """
    spacing = scene._orthogonal_spacing
    if spacing is None:
        return point_segment_distances(scene.voxel_to_world(indices), start, end) - scene._cell_radius_mm
    count = len(indices)
    if not count:
        return np.empty(0)
    origin = (scene.world_to_voxel(start) - indices) * spacing
    delta = (scene.world_to_voxel(end) - scene.world_to_voxel(start)) * spacing
    half = spacing / 2
    crossings = np.zeros((count, 6))
    for axis in range(3):
        if abs(delta[axis]) > _EPS:
            crossings[:, 2 * axis] = (-half[axis] - origin[:, axis]) / delta[axis]
            crossings[:, 2 * axis + 1] = (half[axis] - origin[:, axis]) / delta[axis]
    times = np.sort(np.concatenate([np.zeros((count, 1)), np.clip(crossings, 0, 1), np.ones((count, 1))], axis=1), axis=1)
    left, right = times[:, :-1], times[:, 1:]
    midpoint = (left + right) / 2
    positions = origin[:, None, :] + midpoint[:, :, None] * delta
    outside = np.abs(positions) > half
    offsets = origin[:, None, :] - np.where(positions > half, half, -half)
    numerator = np.sum(np.where(outside, offsets * delta, 0), axis=2)
    denominator = np.sum(np.where(outside, delta**2, 0), axis=2)
    optimum = np.divide(-numerator, denominator, out=midpoint.copy(), where=denominator > _EPS**2)
    optimum = np.clip(optimum, left, right)
    positions = origin[:, None, :] + optimum[:, :, None] * delta
    excess = np.maximum(np.abs(positions) - half, 0)
    return np.sqrt(np.min(np.sum(excess**2, axis=2), axis=1))


def capsule_voxel_indices(scene: GeometryScene, start_mm: ArrayLike, end_mm: ArrayLike,
                          radius_mm: float) -> NDArray[np.int64]:
    """Cell supercover for a capsule; exact for orthogonal, conservative for shear.

    Bounding-box slabs limit temporary arrays even for long oblique instruments.
    Returned cells are an upper bound on intersection, not an exact occupancy.
    """
    start, end = _vector(start_mm, "start_mm"), _vector(end_mm, "end_mm")
    if not np.isfinite(radius_mm) or radius_mm < 0:
        raise ValueError("radius_mm must be finite and nonnegative")
    radius = radius_mm + scene._cell_radius_mm
    extent = np.linalg.norm(scene._inverse[:3, :3], axis=1) * radius
    endpoints = scene.world_to_voxel(np.stack([start, end]))
    lo = np.maximum(0, np.ceil(endpoints.min(axis=0) - extent - _EPS).astype(int))
    hi = np.minimum(np.array(scene.shape) - 1, np.floor(endpoints.max(axis=0) + extent + _EPS).astype(int))
    if np.any(lo > hi):
        return _immutable(np.empty((0, 3), dtype=np.int64))
    slab_size = max(1, 131072 // int((hi[1] - lo[1] + 1) * (hi[2] - lo[2] + 1)))
    found = []
    for x in range(int(lo[0]), int(hi[0]) + 1, slab_size):
        ranges = (np.arange(x, min(x + slab_size, hi[0] + 1)), np.arange(lo[1], hi[1] + 1),
                  np.arange(lo[2], hi[2] + 1))
        grid = np.stack(np.meshgrid(*ranges, indexing="ij"), axis=-1).reshape(-1, 3)
        distances = point_segment_distances(scene.voxel_to_world(grid), start, end)
        candidates = grid[distances <= radius + _EPS]
        exact_distances = _segment_cell_distances(scene, candidates, start, end)
        found.append(candidates[exact_distances <= radius_mm + _EPS])
    return _immutable(np.concatenate(found).astype(np.int64, copy=False))


def _failure(reason: str, position: ArrayLike, detail: str, fraction: float | None) -> GeometryFailure:
    return GeometryFailure(reason, tuple(float(x) for x in position), detail, fraction)


def _unique_indices(indices: Iterable[NDArray[np.int64]]) -> NDArray[np.int64]:
    arrays = [array for array in indices if len(array)]
    result = np.unique(np.concatenate(arrays), axis=0) if arrays else np.empty((0, 3), dtype=np.int64)
    return _immutable(result)


def _result(scene: GeometryScene, failures: list[GeometryFailure], clearance: float,
            cells: Iterable[NDArray], unknowns: Iterable[str]) -> GeometryResult:
    indices = _unique_indices(cells)
    exposure = None
    if scene.exposure_mask is not None:
        exposure = float(np.count_nonzero(scene.exposure_mask[tuple(indices.T)])) * scene.voxel_volume_mm3
    # Retain one location per reason; a conservative sweep location is an envelope
    # witness, not a claimed time of physical tissue contact.
    unique = {failure.reason: failure for failure in reversed(failures)}
    ordered = tuple(reversed(tuple(unique.values())))
    return GeometryResult(not failures, ordered, float(clearance), indices, exposure,
                          tuple(sorted(set(unknowns))))


def _check_envelope(tool: ToolGeometry, pose: ToolPose, scene: GeometryScene,
                    access: AccessWindow | None, other_tools: tuple[tuple[ToolGeometry, ToolPose], ...],
                    *, inflation: float, angular_bound: float, tip_translation_bound: float,
                    fraction: float | None,
                    capsule_override: tuple | None = None) -> tuple[list[GeometryFailure], float, list[NDArray], set[str]]:
    failures: list[GeometryFailure] = []
    clearance = _CLEARANCE_CAP_MM
    cells = []
    unknowns = set(scene.unknowns)
    if scene.exposure_mask is None:
        unknowns.add("normal_tissue_exposure_unassessed")

    tip_voxel = scene.world_to_voxel(pose.tip_mm)
    tip_extent = np.linalg.norm(scene._inverse[:3, :3], axis=1) * tip_translation_bound
    if np.any(tip_voxel - tip_extent < -0.5 - _EPS) or np.any(tip_voxel + tip_extent > np.array(scene.shape) - 0.5 + _EPS):
        unknowns.add("tip_outside_image_volume")
        if scene.enforce_tip_in_bounds:
            failures.append(_failure("TIP_OUTSIDE_IMAGE", pose.tip_mm, "Distal tip leaves the known image cell domain.", fraction))
    endpoints = np.stack([pose.proximal_mm(tool), pose.tip_mm])
    voxel_endpoints = scene.world_to_voxel(endpoints)
    radial_extent = np.linalg.norm(scene._inverse[:3, :3], axis=1) * (tool.envelope_radius_mm + inflation)
    if np.any(voxel_endpoints - radial_extent < -0.5) or np.any(voxel_endpoints + radial_extent > np.array(scene.shape) - 0.5):
        unknowns.add("tool_geometry_outside_image_unassessed")

    if access is not None:
        alignment = float(pose.axis_unit @ access.normal_inward)
        angle = float(np.arccos(np.clip(alignment, -1, 1)))
        max_angle = angle + angular_bound
        allowed = np.deg2rad(tool.max_access_angle_deg)
        if max_angle > allowed + _EPS:
            failures.append(_failure("ACCESS_ANGLE", pose.tip_mm, "Tool orientation exceeds the declared access angle.", fraction))
        depth = float((pose.tip_mm - access.center_mm) @ access.normal_inward)
        alignment_lower = float(np.cos(min(np.pi, max_angle)))
        if alignment_lower <= _EPS or depth - tip_translation_bound < -_EPS:
            failures.append(_failure("ACCESS_DIRECTION", pose.tip_mm, "Tip must remain inward with a forward-facing tool.", fraction))
        elif (depth + tip_translation_bound) / alignment_lower > tool.working_length_mm + _EPS:
            failures.append(_failure("WORKING_REACH", pose.tip_mm, "The complete shaft cannot span the access plane at this depth.", fraction))
        if alignment > _EPS:
            entry = pose.tip_mm - (depth / alignment) * pose.axis_unit
            radial_offset = float(np.linalg.norm(entry - access.center_mm))
            # The infinite cylinder bounds every capsule-plane section, including
            # oblique ellipses and spherical tips. This can conservatively reject
            # a large shaft that has not yet entered the aperture.
            aperture_clearance = access.radius_mm - radial_offset - (tool.envelope_radius_mm + inflation) / alignment
            clearance = min(clearance, aperture_clearance)
            if aperture_clearance <= _EPS:
                failures.append(_failure("ACCESS_APERTURE", entry, "Full instrument envelope does not clear the circular aperture.", fraction))

    capsules = tool_capsules(tool, pose, inflation) if capsule_override is None else capsule_override
    for start, end, radius, part in capsules:
        cells.append(capsule_voxel_indices(scene, start, end, radius))
        if scene._tree is not None:
            midpoint = (start + end) / 2
            query_radius = float(np.linalg.norm(end - start)) / 2 + radius + scene._cell_radius_mm + _CLEARANCE_CAP_MM
            ids = scene._tree.query_ball_point(midpoint, query_radius + _EPS)
            if ids:
                centers = scene._forbidden_centers[ids]
                margins = _segment_cell_distances(scene, scene._forbidden_indices[ids], start, end) - radius
                worst = int(np.argmin(margins))
                clearance = min(clearance, float(margins[worst]))
                if margins[worst] <= _EPS:
                    failures.append(_failure("FORBIDDEN_COLLISION", centers[worst], f"Conservative {part} envelope intersects a forbidden affine voxel cell.", fraction))
        for sphere in scene.sphere_obstacles:
            margin = float(point_segment_distances(sphere.center_mm, start, end)) - radius - sphere.radius_mm
            clearance = min(clearance, margin)
            if margin <= _EPS:
                failures.append(_failure("SPHERE_COLLISION", sphere.center_mm, f"{part} intersects analytic obstacle {sphere.obstacle_id}.", fraction))
        for other_tool, other_pose in other_tools:
            for other_start, other_end, other_radius, _ in tool_capsules(other_tool, other_pose):
                margin = segment_segment_distance(start, end, other_start, other_end) - radius - other_radius
                clearance = min(clearance, margin)
                if margin <= _EPS:
                    failures.append(_failure("TOOL_COLLISION", (start + end) / 2, f"{part} intersects tool {other_tool.tool_id}.", fraction))
    return failures, clearance, cells, unknowns


def check_pose(tool: ToolGeometry, pose: ToolPose, scene: GeometryScene,
               access: AccessWindow | None = None,
               other_tools: Iterable[tuple[ToolGeometry, ToolPose]] = ()) -> GeometryResult:
    """Check the complete static tool, aperture, reach, and simultaneous tools."""
    failures, clearance, cells, unknowns = _check_envelope(
        tool, pose, scene, access, tuple(other_tools), inflation=0, angular_bound=0,
        tip_translation_bound=0, fraction=None)
    return _result(scene, failures, clearance, cells, unknowns)


def interpolate_pose(start: ToolPose, end: ToolPose, fraction: float) -> ToolPose:
    """Linear translation and constant-angular-speed shortest-arc axis SLERP."""
    if not np.isfinite(fraction) or not 0 <= fraction <= 1:
        raise ValueError("fraction must be in [0,1]")
    angle = _axis_angle(start.axis_unit, end.axis_unit)
    if angle >= np.pi - 1e-7:
        raise ValueError("antipodal axes have no uniquely specified rigid motion")
    if angle < 1e-7:
        axis = (1 - fraction) * start.axis_unit + fraction * end.axis_unit
    else:
        axis = (np.sin((1 - fraction) * angle) * start.axis_unit + np.sin(fraction * angle) * end.axis_unit) / np.sin(angle)
    return ToolPose((1 - fraction) * start.tip_mm + fraction * end.tip_mm, axis)


def check_motion(tool: ToolGeometry, start: ToolPose, end: ToolPose, scene: GeometryScene,
                 access: AccessWindow | None = None, *, max_surface_step_mm: float = 0.5,
                 other_tools: Iterable[tuple[ToolGeometry, ToolPose]] = (),
                 max_intervals: int = 20000) -> GeometryResult:
    """Certify a continuous rigid motion with conservative midpoint envelopes.

    Over an interval, every centerline point moves at most translation/2 plus
    ``2*length*sin(rotation/4)`` from its midpoint position. Adding that bound
    to both capsule radii encloses the entire motion, including endpoints.
    The step parameter changes conservatism/runtime, not whether gaps are tested.
    Other tools are fixed throughout this call; simultaneous motion needs its own
    synchronized trajectory contract and is not silently certified here.
    """
    _positive(max_surface_step_mm, "max_surface_step_mm")
    if not isinstance(max_intervals, int) or max_intervals < 1:
        raise ValueError("max_intervals must be a positive integer")
    angle = _axis_angle(start.axis_unit, end.axis_unit)
    if angle >= np.pi - 1e-7:
        raise ValueError("antipodal axes have no uniquely specified rigid motion")
    translation = float(np.linalg.norm(end.tip_mm - start.tip_mm))
    if translation == 0 and np.array_equal(start.axis_unit, end.axis_unit):
        return check_pose(tool, start, scene, access, other_tools)
    others = tuple(other_tools)
    displacement = end.tip_mm - start.tip_mm
    lateral = displacement - float(displacement @ start.axis_unit) * start.axis_unit
    if np.array_equal(start.axis_unit, end.axis_unit) and np.linalg.norm(lateral) < _EPS:
        # Axial insertion/retraction has an exact continuous capsule union. This
        # avoids hundreds of redundant pose checks for the common route primitive.
        swept_capsules = []
        for first, last in zip(tool_capsules(tool, start), tool_capsules(tool, end)):
            points = np.stack([first[0], first[1], last[0], last[1]])
            projection = points @ start.axis_unit
            swept_capsules.append((points[np.argmin(projection)], points[np.argmax(projection)], first[2], first[3]))
        first = _check_envelope(tool, start, scene, access, others, inflation=0,
                                angular_bound=0, tip_translation_bound=0, fraction=0,
                                capsule_override=())
        last = _check_envelope(tool, end, scene, access, others, inflation=0,
                               angular_bound=0, tip_translation_bound=0, fraction=1,
                               capsule_override=tuple(swept_capsules))
        return _result(scene, first[0] + last[0], min(first[1], last[1]), last[2], first[3] | last[3])
    intervals = max(1, int(np.ceil((translation + tool.working_length_mm * angle) / max_surface_step_mm)))
    if intervals > max_intervals:
        raise ValueError(f"motion requires {intervals} intervals, exceeding max_intervals={max_intervals}")
    translation_bound = translation / (2 * intervals)
    angular_bound = angle / (2 * intervals)
    inflation = translation_bound + 2 * tool.working_length_mm * np.sin(angle / (4 * intervals))
    failures, all_cells, unknowns = [], [], set()
    clearance = _CLEARANCE_CAP_MM
    for index in range(intervals):
        fraction = (index + 0.5) / intervals
        pose = interpolate_pose(start, end, fraction)
        current, margin, cells, flags = _check_envelope(
            tool, pose, scene, access, others, inflation=float(inflation),
            angular_bound=angular_bound, tip_translation_bound=translation_bound,
            fraction=fraction)
        failures.extend(current)
        clearance = min(clearance, margin)
        all_cells.extend(cells)
        unknowns.update(flags)
    return _result(scene, failures, clearance, all_cells, unknowns)


def check_route(tool: ToolGeometry, poses: Iterable[ToolPose], scene: GeometryScene,
                access: AccessWindow | None = None, *, max_surface_step_mm: float = 0.5,
                other_tools: Iterable[tuple[ToolGeometry, ToolPose]] = ()) -> GeometryResult:
    """Check piecewise rigid poses; tip points alone cannot certify a route."""
    poses = tuple(poses)
    if not poses or not all(isinstance(pose, ToolPose) for pose in poses):
        raise ValueError("a route requires explicit ToolPose objects, including orientations")
    others = tuple(other_tools)
    if len(poses) == 1:
        return check_pose(tool, poses[0], scene, access, others)
    results = [check_motion(tool, start, end, scene, access,
                            max_surface_step_mm=max_surface_step_mm, other_tools=others)
               for start, end in zip(poses[:-1], poses[1:])]
    return _result(scene, [failure for result in results for failure in result.failures],
                   min(result.clearance_mm for result in results),
                   (result.swept_voxel_indices for result in results),
                   (flag for result in results for flag in result.unknowns))
