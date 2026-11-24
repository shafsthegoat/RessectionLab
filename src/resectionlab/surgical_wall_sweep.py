"""Transactional clearance of the source-bound score wall between rod poses."""

import numpy as np

from . import native_backend
from . import surgical_vessel_surface as surfaces


PROPERTY = 'vesselWallMotionGuard'
POLICY = 'continuous-source-triangles-v1'


def topology(frames):
    return tuple((wall.source.label, edited.identity if edited is not None else None)
                 for wall, _, _, _, _, edited in frames)


def _build(frames):
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
    if native_backend.wall_sweep_available():
        source = native_backend.make_wall_sweep(*geometry)

        def clear(prior, candidate, static):
            return native_backend.wall_sweep_clear(source, prior, candidate, static)

        return clear
    from .surgical_wall_sweep_reference import WallSweepReference

    source = WallSweepReference(*geometry)

    def clear(prior, candidate, static):
        return source.clear(prior, candidate, static=static)

    return clear


def require_clear(engine, prior, candidate, *, static=False):
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
        cached = key, _build(candidate)
        engine._vessel_wall_sweep = cached
    end = np.concatenate([surfaces.sweep_offsets(frame) for frame in candidate])
    start = np.concatenate([surfaces.sweep_offsets(frame) for frame in prior]) if not static else end
    if not cached[1](start, end, static):
        raise ValueError('VESSEL_WALL_SWEEP_UNSAFE: cannot certify native vessel wall clearance')
    if transition_check is not None:
        transition_check()
