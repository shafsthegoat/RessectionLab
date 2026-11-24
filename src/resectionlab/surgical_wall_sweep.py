"""Transactional clearance of the source-bound score wall between rod poses."""

from itertools import combinations

import numpy as np

from . import native_backend
from . import surgical_vessel_surface as surfaces


PROPERTY = 'vesselWallMotionGuard'
POLICY = 'continuous-source-triangles-v1'
_MIN_AREA_RATIO = .2
_LOCAL_SEARCH_DEPTH = 8


def topology(frames):
    return tuple((wall.source.label, edited.identity if edited is not None else None)
                 for wall, _, _, _, _, edited in frames)


def _shared_edge_quads(faces):
    edges = np.sort(np.concatenate((faces[:, :2], faces[:, 1:], faces[:, ::2])), axis=1)
    opposite = np.concatenate((faces[:, 2], faces[:, 0], faces[:, 1]))
    order = np.lexsort(edges.T[::-1])
    edges, opposite = edges[order], opposite[order]
    starts = np.r_[0, np.flatnonzero(np.any(np.diff(edges, axis=0), axis=1)) + 1]
    counts = np.diff(np.r_[starts, len(edges)])
    paired = starts[counts == 2]
    regular = np.column_stack((edges[paired], opposite[paired], opposite[paired + 1]))
    irregular = [(*edges[lo], a, b) for lo, n in zip(starts[counts > 2], counts[counts > 2])
                 for a, b in combinations(opposite[lo:lo + n], 2)]
    return np.vstack((regular, np.asarray(irregular, dtype=int).reshape(-1, 4)))


def _face_normals(vertices):
    edge = vertices[:, 1] - vertices[:, 0]
    return np.cross(edge[:, None, :], vertices[:, 2:] - vertices[:, :1])


def _facet_areas_clear(rest, before, after):
    first = _face_normals(before)[:, 0]
    middle = _face_normals((before + after) / 2.)[:, 0]
    last = _face_normals(after)[:, 0]
    area = np.linalg.norm(first, axis=1)
    axis = first / np.maximum(area[:, None], 1e-30)
    half = np.einsum('ij,ij->i', middle, axis)
    end = np.einsum('ij,ij->i', last, axis)
    a = 2. * (end + area - 2. * half)
    b = end - area - a
    t = np.clip(-b / (2. * np.maximum(a, 1e-30)), 0., 1.)
    lowest = np.minimum(np.minimum(area, end), np.where(a > 0., area + t * (b + t * a), area))
    reference = np.linalg.norm(_face_normals(rest)[:, 0], axis=1)
    return bool(np.all(lowest >= _MIN_AREA_RATIO * np.maximum(area, reference)))


def _hinges_clear(before, after, depth=0):
    """Certify that edge neighbors cannot overlap across their common edge."""
    midpoint = (before + after) / 2.
    middle = _face_normals(midpoint)
    unit = middle / np.maximum(np.linalg.norm(middle, axis=2)[:, :, None], 1e-30)
    # Same-side coplanar faces have parallel normals when both use the same
    # shared-edge winding. Opposite projections exclude that event.
    axis = unit[:, 0] - unit[:, 1]
    at_mid = np.einsum('ijk,ik->ij', middle, axis)
    if np.any(at_mid[:, 0] <= 0.) or np.any(at_mid[:, 1] >= 0.):
        return False
    at_start = np.einsum('ijk,ik->ij', _face_normals(before), axis)
    at_end = np.einsum('ijk,ik->ij', _face_normals(after), axis)
    # Normal projections are quadratic. Their Bernstein coefficients bound
    # them for the entire interval, including intermediate coplanar poses.
    control = 2. * at_mid - .5 * (at_start + at_end)
    lower = np.minimum(np.minimum(at_start[:, 0], at_end[:, 0]), control[:, 0])
    upper = np.maximum(np.maximum(at_start[:, 1], at_end[:, 1]), control[:, 1])
    uncertain = (lower <= 0.) | (upper >= 0.)
    return not uncertain.any() or (depth < _LOCAL_SEARCH_DEPTH
                                   and _hinges_clear(before[uncertain], midpoint[uncertain], depth + 1)
                                   and _hinges_clear(midpoint[uncertain], after[uncertain], depth + 1))


def _local_face_guard(rest, faces, owners):
    points = np.empty((int(faces.max()) + 1, 3))
    point_owners = np.empty(len(points), dtype=int)
    points[faces] = rest
    point_owners[faces] = owners
    adjacent = _shared_edge_quads(faces)
    nearby = points[adjacent]
    adjacent_owners = point_owners[adjacent]

    def clear(prior, candidate, static):
        increment = candidate - prior
        changed = np.any(increment != 0., axis=1)
        if not static and not changed.any():
            return True

        def moving(bindings):
            active = np.flatnonzero(changed[bindings].any(axis=1))
            delta = increment[bindings[active]]
            return active[(delta[:, :1] != delta[:, 1:]).any(axis=(1, 2))]

        facets = np.arange(len(owners)) if static else moving(owners)
        if len(facets):
            start = rest[facets] + prior[owners[facets]]
            end = rest[facets] + candidate[owners[facets]]
            if not _facet_areas_clear(rest[facets], start, end):
                return False
        hinges = np.arange(len(adjacent_owners)) if static else moving(adjacent_owners)
        if len(hinges):
            start = nearby[hinges] + prior[adjacent_owners[hinges]]
            end = nearby[hinges] + candidate[adjacent_owners[hinges]]
            if not _hinges_clear(start, end):
                return False
        return True

    return clear


def _build(frames, *, local_guard=False):
    triangles, vertices, owners = [], [], []
    vertex_start = owner_start = 0
    for frame in frames:
        edited = frame[5]
        if edited is not None and len(edited.cuboids):
            raise ValueError('VESSEL_WALL_SWEEP_UNSUPPORTED: cut has no closed score-wall cap')
        rest, faces, bindings, offsets = surfaces.sweep_mesh(frame)
        triangles.append(rest)
        # Drop unused indices so multiple source layers have disjoint adjacency.
        unique, inverse = np.unique(faces, return_inverse=True)
        vertices.append(inverse.reshape(faces.shape) + vertex_start)
        owners.append(bindings + owner_start)
        vertex_start += len(unique)
        owner_start += len(offsets)
    geometry = (np.ascontiguousarray(np.concatenate(triangles), dtype=np.float64),
                np.ascontiguousarray(np.concatenate(vertices), dtype=np.int64),
                np.ascontiguousarray(np.concatenate(owners), dtype=np.int64), owner_start)
    check_local = _local_face_guard(*geometry[:3]) if local_guard and len(geometry[0]) else None
    if native_backend.wall_sweep_available():
        source = native_backend.make_wall_sweep(*geometry)

        def clear(prior, candidate, static):
            return (check_local is None or check_local(prior, candidate, static)) and (
                native_backend.wall_sweep_clear(source, prior, candidate, static))

        return clear
    from .surgical_wall_sweep_reference import WallSweepReference

    source = WallSweepReference(*geometry)

    def clear(prior, candidate, static):
        return (check_local is None or check_local(prior, candidate, static)) and (
            source.clear(prior, candidate, static=static))

    return clear


def is_clear(engine, prior, candidate, *, static=False):
    if not prior or not candidate:
        raise ValueError('VESSEL_WALL_SWEEP_UNSUPPORTED: native score-wall geometry is required')
    key = topology(candidate)
    if not static and topology(prior) != key:
        raise ValueError('VESSEL_WALL_SWEEP_UNSUPPORTED: source topology changed inside a rod step')
    transition_check = getattr(engine, '_transition_check', None)
    if transition_check is not None:
        transition_check()
    cached = getattr(engine, '_vessel_wall_sweep', None)
    if cached is None or cached[0] != key:
        from .surgical_vessel_rods import mechanics

        cached = key, _build(candidate, local_guard=mechanics(engine) is not None)
        engine._vessel_wall_sweep = cached
    end = np.concatenate([surfaces.sweep_offsets(frame) for frame in candidate])
    start = np.concatenate([surfaces.sweep_offsets(frame) for frame in prior]) if not static else end
    clear = cached[1](start, end, static)
    if transition_check is not None:
        transition_check()
    return clear


def require_clear(engine, prior, candidate, *, static=False):
    if not is_clear(engine, prior, candidate, static=static):
        raise ValueError('VESSEL_WALL_SWEEP_UNSAFE: cannot certify native vessel wall clearance')
