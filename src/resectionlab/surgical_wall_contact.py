"""Unilateral contact between disconnected triangles on the accepted native wall."""

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree

from . import surgical_vessel_surface as surfaces
from . import surgical_wall_sweep as wall_sweep
from .surgical_wall_sweep_reference import _axes


# Numerical separation for a projected wall, in millimeters.
_GAP_MM = .0002
_SEPARATION_MM = 1e-9
_MAX_NEIGHBORS = 131_072
_MAX_PAIRS = 4_096
_CONSTRAINTS_PER_PASS = 16
_CONTACT_PASSES = 4
_SOLVER_PASSES = 16


@dataclass(frozen=True)
class _Geometry:
    rest: np.ndarray
    faces: np.ndarray
    components: np.ndarray
    translation: sparse.csr_matrix


def _geometry(engine, frames, bonds):
    key = wall_sweep.topology(frames), bonds
    cached = getattr(engine, '_vessel_wall_contact_geometry', None)
    if cached is not None and cached[0] == key:
        return cached[1]
    rest, faces, translation = [], [], []
    offset = 0
    for frame in frames:
        triangles, indices, _, shifts = surfaces.sweep_mesh(frame)
        points = np.zeros_like(shifts)
        points[indices] = triangles
        rest.append(points)
        faces.append(indices + offset)
        translation.append(frame[0].vertex_motion.binding.translation)
        offset += len(points)
    faces = np.concatenate(faces)
    links = faces.ravel(), faces[:, [1, 2, 0]].ravel()
    graph = sparse.coo_matrix((np.ones(len(links[0])), links), shape=(offset, offset)).tocsr()
    _, components = sparse.csgraph.connected_components(graph, directed=False)
    present, components = np.unique(components[faces[:, 0]], return_inverse=True)
    geometry = None
    if len(present) > 1:
        geometry = _Geometry(np.concatenate(rest), faces, components, sparse.vstack(translation, format='csr'))
    engine._vessel_wall_contact_geometry = key, geometry
    return geometry


def _near_pairs(before, after, components):
    low = np.minimum(before.min(axis=1), after.min(axis=1))
    high = np.maximum(before.max(axis=1), after.max(axis=1))
    indices = np.arange(len(low))
    count = components.max() + 1
    if count <= 8:
        # Most faces in a compact wall patch only neighbor their own vessel.
        # Cull them by the other components' swept bounds before the tree query.
        comp_low, comp_high = np.full((count, 3), np.inf), np.full((count, 3), -np.inf)
        np.minimum.at(comp_low, components, low)
        np.maximum.at(comp_high, components, high)
        facing = ((low[:, None] <= comp_high + _GAP_MM)
                  & (high[:, None] >= comp_low - _GAP_MM)).all(axis=2)
        facing &= components[:, None] != np.arange(count)
        indices = np.flatnonzero(facing.any(axis=1))
    center = (low + high) / 2.
    radius = (high - low).max(axis=1) / 2.
    dirty = np.any(before != after, axis=(1, 2))
    active = indices[dirty[indices]]
    if not len(active):
        return active, active
    tree = cKDTree(center[indices])
    reach = radius[active] + radius[indices].max() + _GAP_MM
    counts = tree.query_ball_point(center[active], reach, p=np.inf, return_length=True)
    if counts.sum() > _MAX_NEIGHBORS:
        return None
    left = np.repeat(active, counts)
    right = indices[np.concatenate(tree.query_ball_point(center[active], reach, p=np.inf, return_sorted=True))]
    relevant = ((right > left) | ~dirty[right]) & (components[right] != components[left])
    left, right = left[relevant], right[relevant]
    overlap = ((low[left] <= high[right] + _GAP_MM) & (low[right] <= high[left] + _GAP_MM)).all(axis=1)
    left, right = left[overlap], right[overlap]
    if len(left) > _MAX_PAIRS:
        return None
    return left, right


def _contacts(geometry, before, after):
    pairs = _near_pairs(before, after, geometry.components)
    if pairs is None:
        return None
    left, right = pairs
    if not len(left):
        return ()
    a0, a1, b0, b1 = before[left], after[left], before[right], after[right]
    axes, _, usable = _axes(a0, b0)
    length = np.linalg.norm(axes, axis=2)
    axes /= np.maximum(length[:, :, None], 1e-30)
    axes = np.concatenate((axes, -axes), axis=1)
    origin = a0[:, :1]
    pa0, pa1, pb0, pb1 = (np.einsum('nvi,nki->nvk', points - origin, axes)
                          for points in (a0, a1, b0, b1))
    first_gap = pb0.min(axis=1) - pa0.max(axis=1)
    last_gap = pb1.min(axis=1) - pa1.max(axis=1)
    separating = np.tile(usable, (1, 2)) & (first_gap > _SEPARATION_MM)
    # Keep a separating side from the already certified start. A normal chosen
    # from an intersecting end pose can point the wrong way after tunneling.
    axis_index = np.argmax(np.where(separating, last_gap, -np.inf), axis=1)
    rows = np.arange(len(left))
    gap = last_gap[rows, axis_index]
    active = np.flatnonzero(separating.any(axis=1) & (gap <= _SEPARATION_MM))
    selected = active[np.argsort(gap[active], kind='stable')[:_CONSTRAINTS_PER_PASS]]
    axis = axis_index[selected]
    neg = geometry.faces[left[selected], pa1[selected, :, axis].argmax(axis=1)]
    pos = geometry.faces[right[selected], pb1[selected, :, axis].argmin(axis=1)]
    return tuple(zip(neg.tolist(), pos.tolist(), map(tuple, axes[selected, axis])))


def _project(geometry, position, predicted, constraints, solve):
    negative, positive, normal = zip(*constraints)
    negative, positive, normal = np.asarray(negative), np.asarray(positive), np.asarray(normal)
    weights = (geometry.translation[positive] - geometry.translation[negative]).toarray()
    # The inverse damped rod stiffness distributes an equal/opposite normal
    # load according to the local support, axial and bending compliance.
    response = solve(weights.T).T
    coupling = (weights @ response.T) * (normal @ normal.T)
    gap = np.einsum('ij,ij->i', predicted[positive] - predicted[negative], normal)
    multipliers = np.zeros(len(weights))
    usable = np.flatnonzero(coupling.diagonal() > 1e-14)
    for _ in range(_SOLVER_PASSES):
        for row in usable:
            correction = (_GAP_MM - gap[row] - coupling[row] @ multipliers) / coupling[row, row]
            multipliers[row] = max(0., multipliers[row] + correction)
    return position + np.einsum('ij,ik,i->jk', response, normal, multipliers)


def resolve(engine, previous, proposed, bonds, position, solve):
    """Propose translational contact motion for a separate sweep to certify."""
    geometry = _geometry(engine, previous, bonds)
    if geometry is None:
        return None
    prior_points = geometry.rest + np.concatenate([surfaces.sweep_offsets(frame) for frame in previous])
    predicted = geometry.rest + np.concatenate([surfaces.sweep_offsets(frame) for frame in proposed])
    before = prior_points[geometry.faces]
    constraints = {}
    candidate = position
    for _ in range(_CONTACT_PASSES):
        points = predicted + geometry.translation @ (candidate - position)
        contacts = _contacts(geometry, before, points[geometry.faces])
        if contacts is None:
            return None
        if not contacts:
            break
        constraints.update(dict.fromkeys(contacts))
        candidate = _project(geometry, position, predicted, constraints, solve)
    return candidate if np.any(candidate != position) else None
