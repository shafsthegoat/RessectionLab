"""Opt-in bond scans for the V2 lattice (``deformation-scan``).

The reference loops evaluate one scalar ``np.linalg.norm`` per bond per
substep. Here every bond is screened with one vectorized norm, then only the
bonds that could hold the extreme value are recomputed with the reference's
own scalar expression. The vectorized norm differs from the scalar one by a
few ulps at most, so the screen keeps the true extreme and the returned value
is the reference's exact float.
"""
from __future__ import annotations

import numpy as np

_SCREEN = 1e-9  # relative slack, far above the few-ulp norm difference
_cache = [None]


def _arrays(edges):
    hit = _cache[0]
    if hit is not None and hit[0] is edges and hit[1] == len(edges):
        return hit[2:]
    first = np.fromiter((i for i, _, _ in edges), np.int64, len(edges))
    second = np.fromiter((j for _, j, _ in edges), np.int64, len(edges))
    lengths = np.fromiter((length for _, _, length in edges), float, len(edges))
    _cache[0] = (edges, len(edges), first, second, lengths)
    return first, second, lengths


def bond_scale(u, edges, max_edge_strain):
    """Exactly ``min(max_edge_strain*length/max(norm(u[j]-u[i]),1e-15), default=1.)``."""
    if not len(edges):
        return 1.
    first, second, lengths = _arrays(edges)
    stretch = np.linalg.norm(u[second] - u[first], axis=1)
    screen = max_edge_strain * lengths / np.maximum(stretch, 1e-15)
    bound = float(screen.min())
    keep = np.flatnonzero(screen <= bound + abs(bound) * _SCREEN)
    return min(max_edge_strain * edges[k][2] / max(np.linalg.norm(u[edges[k][1]] - u[edges[k][0]]), 1e-15)
               for k in keep.tolist())


def bond_ratio(u, edges):
    """Exactly ``max(float(norm(u[j]-u[i])/length), default=0.)``."""
    if not len(edges):
        return 0.
    first, second, lengths = _arrays(edges)
    screen = np.linalg.norm(u[second] - u[first], axis=1) / lengths
    bound = float(screen.max())
    keep = np.flatnonzero(screen >= bound - abs(bound) * _SCREEN)
    return max(float(np.linalg.norm(u[edges[k][1]] - u[edges[k][0]]) / edges[k][2]) for k in keep.tolist())
