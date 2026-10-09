"""Pure voxel-domain geometry for a *future* Case4 source-only support check.

This module reads no files and has no command-line entry point. Its caller must
first independently accept local T1/FLAIR and MRI/before-US correspondence and
the unchanged retained-tissue domain. Synthetic tests do not grant that gate.
All coordinates and volumes here use physical RAS millimetres.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from math import pi

import numpy as np
from scipy.spatial import cKDTree


RADIUS_MM = 5.0
FULL_BALL_MM3 = 4.0 * pi * RADIUS_MM**3 / 3.0
FULL_TENT_MASS_MM3 = pi * RADIUS_MM**3 / 3.0
B_ROWS = (1, 14, 8, 17, 19, 7)
V_ROWS = (2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 15, 16, 18)
MASS_RELATIVE_TOLERANCE = 0.01
CENTROID_TOLERANCE_MM = 0.1
BOUNDARY_AMBIGUITY_MM = 1e-6
MINIMUM_WEIGHTED_SUPPORT_FRACTION = 1e-4
MAXIMUM_EXPOSED_FACES = 2_000_000
MAXIMUM_CANDIDATE_CELLS = 100_000
MAXIMUM_QUADRATURE_POINTS = 4_000_000


def _point3(point: object) -> np.ndarray:
    p = np.asarray(point, dtype=np.float64)
    if p.shape != (3,) or not np.isfinite(p).all():
        raise ValueError("finite three-vector required")
    return p


def _parallelogram_distance(point: np.ndarray, centre: np.ndarray,
                            edge_a: np.ndarray, edge_b: np.ndarray) -> float:
    """Exact Euclidean distance to a bounded affine voxel face."""
    corner = centre - (edge_a + edge_b) / 2.0
    q = point - corner
    gram = np.array([[edge_a @ edge_a, edge_a @ edge_b],
                     [edge_b @ edge_a, edge_b @ edge_b]])
    uv = np.linalg.solve(gram, [q @ edge_a, q @ edge_b])
    if np.all((uv >= 0.0) & (uv <= 1.0)):
        return float(np.linalg.norm(q - uv[0] * edge_a - uv[1] * edge_b))
    vertices = (corner, corner + edge_a, corner + edge_a + edge_b,
                corner + edge_b)
    distances = []
    for a, b in zip(vertices, vertices[1:] + vertices[:1]):
        edge = b - a
        t = np.clip(((point - a) @ edge) / (edge @ edge), 0.0, 1.0)
        distances.append(float(np.linalg.norm(point - (a + t * edge))))
    return min(distances)


class VoxelDomain:
    """Immutable-copy binary voxel union, including the half-cell boundary.

    The full 3x3 native affine is retained, including small obliquities and
    shear; faces are parallelograms. This is geometric mask support only.
    """

    def __init__(self, mask: np.ndarray, affine_ras_mm: np.ndarray):
        array = np.asarray(mask)
        affine = np.asarray(affine_ras_mm, dtype=np.float64)
        if array.ndim != 3 or array.dtype != np.bool_ or not array.any():
            raise ValueError("nonempty three-dimensional binary mask required")
        if array.size > 20_000_000:
            raise ValueError("mask exceeds preparation cap")
        if affine.shape != (4, 4) or not np.isfinite(affine).all() or not np.allclose(
            affine[3], [0, 0, 0, 1], rtol=0, atol=1e-12
        ):
            raise ValueError("finite homogeneous RAS-mm affine required")
        matrix = affine[:3, :3]
        if abs(np.linalg.det(matrix)) < 1e-9 or np.linalg.cond(matrix) > 1e6:
            raise ValueError("singular or ill-conditioned voxel affine")
        self.mask = np.array(array, copy=True)
        self.mask.flags.writeable = False
        self.affine = np.array(affine, copy=True)
        self.affine.flags.writeable = False
        self.matrix = matrix.copy()
        self.inverse = np.linalg.inv(matrix)
        self.origin = affine[:3, 3].copy()
        self.voxel_volume_mm3 = abs(float(np.linalg.det(matrix)))
        exposed_faces = 0
        for axis in range(3):
            low = [slice(None)] * 3
            high = [slice(None)] * 3
            low[axis] = 0
            high[axis] = -1
            first = int(np.count_nonzero(self.mask[tuple(low)]))
            last = int(np.count_nonzero(self.mask[tuple(high)]))
            low[axis] = slice(None, -1)
            high[axis] = slice(1, None)
            transitions = int(np.count_nonzero(self.mask[tuple(low)] !=
                                                self.mask[tuple(high)]))
            exposed_faces += first + last + transitions
        if exposed_faces > MAXIMUM_EXPOSED_FACES:
            raise ValueError("exposed voxel faces exceed preparation memory cap")
        centres, axes = [], []
        for axis in range(3):
            for sign in (-1, 1):
                neighbour = np.zeros_like(self.mask)
                target = [slice(None)] * 3
                source = [slice(None)] * 3
                if sign < 0:
                    target[axis] = slice(1, None)
                    source[axis] = slice(None, -1)
                else:
                    target[axis] = slice(None, -1)
                    source[axis] = slice(1, None)
                neighbour[tuple(target)] = self.mask[tuple(source)]
                indices = np.argwhere(self.mask & ~neighbour).astype(np.float64)
                indices[:, axis] += sign / 2.0
                centres.append(indices @ matrix.T + self.origin)
                axes.append(np.full(len(indices), axis, dtype=np.int8))
        self.face_centres = np.concatenate(centres)
        self.face_axes = np.concatenate(axes)
        self.tree = cKDTree(self.face_centres)
        self.face_radius_bound = max(
            (np.linalg.norm(matrix[:, j]) + np.linalg.norm(matrix[:, k])) / 2.0
            for j, k in ((1, 2), (0, 2), (0, 1))
        )

    def voxel_index(self, point_ras_mm: object) -> np.ndarray:
        coordinate = self.inverse @ (_point3(point_ras_mm) - self.origin)
        return np.floor(coordinate + 0.5).astype(np.int64)

    def within_full_cell_bounds(self, point_ras_mm: object) -> bool:
        coordinate = self.inverse @ (_point3(point_ras_mm) - self.origin)
        return bool(np.all(coordinate >= -0.5) and
                    np.all(coordinate <= np.asarray(self.mask.shape) - 0.5))

    def contains(self, point_ras_mm: object) -> bool:
        index = self.voxel_index(point_ras_mm)
        return bool(np.all((index >= 0) & (index < self.mask.shape)) and
                    self.mask[tuple(index)])

    def signed_distance_mm(self, point_ras_mm: object) -> float:
        """Positive inside, negative outside, zero at the voxel-union boundary."""
        point = _point3(point_ras_mm)
        first_distance, _ = self.tree.query(point)
        candidates = self.tree.query_ball_point(point, first_distance +
                                                self.face_radius_bound + 1e-9)
        distance = min(
            _parallelogram_distance(point, self.face_centres[i],
                                    self.matrix[:, (axis + 1) % 3],
                                    self.matrix[:, (axis + 2) % 3])
            for i in candidates for axis in (int(self.face_axes[i]),)
        )
        return distance if self.contains(point) else -distance

    def _nearby_voxels(self, point: np.ndarray, quadrature_order: int) -> np.ndarray:
        fractional = self.inverse @ (point - self.origin)
        reach = RADIUS_MM * np.linalg.norm(self.inverse, axis=1) + 0.5
        lo = np.maximum(0, np.floor(fractional - reach).astype(int))
        hi = np.minimum(self.mask.shape, np.ceil(fractional + reach).astype(int) + 1)
        if np.any(hi <= lo):
            return np.empty((0, 3), dtype=np.int64)
        candidate_box_cells = int(np.prod(hi - lo, dtype=np.int64))
        if candidate_box_cells > MAXIMUM_CANDIDATE_CELLS:
            raise ValueError("candidate voxel box exceeds preparation memory cap")
        if candidate_box_cells * quadrature_order**3 > MAXIMUM_QUADRATURE_POINTS:
            raise ValueError("quadrature work exceeds preparation cap")
        local = np.argwhere(self.mask[tuple(slice(a, b) for a, b in zip(lo, hi))])
        return local + lo

    def integrate_tent(self, centre_ras_mm: object, *, order: int = 4,
                       method: str = "midpoint") -> "SupportIntegral":
        """Integrate the fixed 5 mm tent over occupied native voxel cells.

        `midpoint` and `gauss` are distinct deterministic voxel-cell rules.
        They must be compared at prospectively frozen resolutions before any
        patient support status is accepted; neither is an anatomy assessment.
        """
        centre = _point3(centre_ras_mm)
        if type(order) is not int or not 1 <= order <= 12:
            raise ValueError("quadrature order must be an integer in [1,12]")
        if method == "midpoint":
            nodes = (np.arange(order, dtype=float) + 0.5) / order - 0.5
            weights = np.full(order, 1.0 / order)
        elif method == "gauss":
            nodes, weights = np.polynomial.legendre.leggauss(order)
            nodes, weights = nodes / 2.0, weights / 2.0
        else:
            raise ValueError("unknown quadrature method")
        offsets = np.array(list(product(nodes, repeat=3)))
        cubature = np.array([a * b * c for a, b, c in product(weights, repeat=3)])
        voxels = self._nearby_voxels(centre, order)
        mass = volume = 0.0
        first = np.zeros(3)
        active: set[tuple[int, int, int]] = set()
        for start in range(0, len(voxels), 128):
            block = voxels[start:start + 128]
            points = (block[:, None, :] + offsets[None, :, :]) @ self.matrix.T + self.origin
            relative = points - centre
            r = np.linalg.norm(relative, axis=2)
            inside = r < RADIUS_MM
            kernel = np.maximum(0.0, 1.0 - r / RADIUS_MM)
            weighted = kernel * cubature
            per_voxel = weighted.sum(axis=1)
            mass += float(per_voxel.sum()) * self.voxel_volume_mm3
            volume += float((inside * cubature).sum()) * self.voxel_volume_mm3
            first += np.einsum("ijk,ij->k", relative, weighted) * self.voxel_volume_mm3
            active.update(map(tuple, block[per_voxel > 0]))
        components = 0
        unseen = set(active)
        while unseen:
            components += 1
            stack = [unseen.pop()]
            while stack:
                p = stack.pop()
                for axis in range(3):
                    for direction in (-1, 1):
                        q = list(p)
                        q[axis] += direction
                        q = tuple(q)
                        if q in unseen:
                            unseen.remove(q)
                            stack.append(q)
        centroid = first / mass if mass > 0 else np.full(3, np.nan)
        return SupportIntegral(
            unweighted_volume_mm3=volume,
            weighted_mass_mm3=mass,
            weighted_mass_fraction=mass / FULL_TENT_MASS_MM3,
            first_moment_mm4=tuple(float(x) for x in first),
            centroid_offset_mm=tuple(float(x) for x in centroid),
            active_component_count=components,
            centre_component_present=tuple(self.voxel_index(centre)) in active,
            method=method,
            order=order,
        )


@dataclass(frozen=True)
class SupportIntegral:
    unweighted_volume_mm3: float
    weighted_mass_mm3: float
    weighted_mass_fraction: float
    first_moment_mm4: tuple[float, float, float]
    centroid_offset_mm: tuple[float, float, float]
    active_component_count: int
    centre_component_present: bool
    method: str
    order: int


def numerical_convergence(coarse: SupportIntegral, fine: SupportIntegral,
                          independent: SupportIntegral) -> bool:
    """Frozen 1% mass and 0.1 mm centroid checks, including another rule."""
    if (coarse.method != fine.method or coarse.order >= fine.order or
            independent.method == fine.method or independent.order < fine.order):
        raise ValueError("two resolutions and an independent quadrature rule required")
    if not (coarse.weighted_mass_mm3 > 0 and fine.weighted_mass_mm3 > 0 and
            independent.weighted_mass_mm3 > 0):
        return False
    return all(
        abs(other.weighted_mass_mm3 - fine.weighted_mass_mm3) /
        fine.weighted_mass_mm3 < MASS_RELATIVE_TOLERANCE
        and np.linalg.norm(np.array(other.centroid_offset_mm) - fine.centroid_offset_mm)
        < CENTROID_TOLERANCE_MM
        for other in (coarse, independent)
    )


@dataclass(frozen=True)
class RowSupport:
    row_id: int
    signed_distance_mm: float
    integral: SupportIntegral
    converged: bool
    surface_mask_agreement: bool | None
    full_cell_bounds: bool | None

    @property
    def disposition(self) -> str:
        if self.full_cell_bounds is None:
            return "hold_full_cell_bounds_unchecked"
        if self.full_cell_bounds is False:
            return "unsupported_outside_grid"
        if not np.isfinite(self.signed_distance_mm) or abs(self.signed_distance_mm) <= BOUNDARY_AMBIGUITY_MM:
            return "hold_boundary_ambiguous"
        if self.signed_distance_mm < 0 or self.integral.weighted_mass_mm3 <= 0:
            return "unsupported"
        if not self.converged:
            return "hold_nonconverged"
        # This is the previously frozen observation-operator numerical support
        # floor, not a clinically meaningful minimum amount of tissue.
        if self.integral.weighted_mass_fraction < MINIMUM_WEIGHTED_SUPPORT_FRACTION:
            return "unsupported_insufficient_mass"
        if self.surface_mask_agreement is not True:
            return "hold_surface_disagreement_or_unchecked"
        if not self.integral.centre_component_present or self.integral.active_component_count != 1:
            return "unsupported_disconnected"
        return "supported"


def rigid_average_rank(centres_ras_mm: object, offsets_mm: object) -> tuple[int, float]:
    """Rank of six rigid modes averaged at six kernel centroids.

    This is a geometric operator control, not a finite-element solver check.
    """
    centres = np.asarray(centres_ras_mm, dtype=float)
    offsets = np.asarray(offsets_mm, dtype=float)
    if centres.shape != (6, 3) or offsets.shape != (6, 3) or not np.isfinite(centres).all() or not np.isfinite(offsets).all():
        raise ValueError("six finite source centres and offsets required")
    p = centres + offsets
    p = (p - p.mean(axis=0)) / RADIUS_MM
    rows = []
    for x, y, z in p:
        rows.extend(((1, 0, 0, 0, z, -y),
                     (0, 1, 0, -z, 0, x),
                     (0, 0, 1, y, -x, 0)))
    singular = np.linalg.svd(np.asarray(rows, dtype=float), compute_uv=False)
    ratio = float(singular[-1] / singular[0])
    return int(np.count_nonzero(singular > singular[0] * 1e-8)), ratio


def fixed_partition_decision(rows: dict[int, RowSupport], *,
                             frame_accepted: bool, anatomy_accepted: bool,
                             rigid_rank: int) -> dict[str, object]:
    """Preserve every frozen B/V row and the original all-13 denominator.

    Refuses to assess patient rows before independent expert frame and anatomy
    acceptance. V coordinates and destinations are never an input here.
    """
    if not (frame_accepted and anatomy_accepted):
        raise PermissionError("independent local frame and anatomy acceptance required")
    if set(rows) != set(B_ROWS + V_ROWS) or any(key != row.row_id for key, row in rows.items()):
        raise ValueError("exact fixed 19-row source-only partition required")
    b = {row_id: rows[row_id].disposition for row_id in B_ROWS}
    v = {row_id: rows[row_id].disposition for row_id in V_ROWS}
    b_supported = all(status == "supported" for status in b.values()) and rigid_rank == 6
    v_supported = sum(status == "supported" for status in v.values())
    return {
        "B_status": "supported" if b_supported else "blocked",
        "B_row_dispositions": b,
        "rigid_average_rank": rigid_rank,
        "V_row_dispositions": v,
        "V_fixed_denominator": 13,
        "V_geometrically_supported": v_supported,
        "all_13_RMS_geometrically_estimable": bool(b_supported and v_supported == 13),
        "interpretation": "source-only geometry, not observed motion, surgery, force, injury or independent validation",
    }
