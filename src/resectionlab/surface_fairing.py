"""Display-only physical fairing. Policy and Chebyshev recurrence match the
browser worker. No masks, affines, components or quantitative metrics change.
"""
import json
from importlib.resources import files
import numpy as np
from scipy import ndimage as ndi, sparse
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

POLICY = json.loads(files('resectionlab').joinpath('surface_policy.json').read_text())


def surface_kind(kind):
    return 'tumor' if kind == 'tumor' else 'vessel' if kind in {'vessel', 'vessels', 'artery', 'vein', 'sinus'} else 'context'


def coefficients(kind):
    profile = POLICY['profiles'][surface_kind(kind)]
    n, theta = profile['iterations'], np.arccos(1-profile['passBand'])
    j = np.arange(1, n+1, dtype=float)
    c = np.r_[theta/np.pi, 2*np.sin(j*theta)/(np.pi*j)*(.5+.5*np.cos(np.pi*j/(n+1)))]
    return c/c.sum()


def intersecting(a, b):
    """Vectorized triangle SAT, including coplanar in-plane separation axes."""
    origin = a[:, :1].copy()
    a, b = a-origin, b-origin
    ae, be = np.roll(a, -1, axis=1)-a, np.roll(b, -1, axis=1)-b
    an, bn = np.cross(ae[:, 0], ae[:, 1]), np.cross(be[:, 0], be[:, 1])
    hit = np.ones(len(a), bool)
    axes = [an, bn] + [np.cross(ae[:, i], be[:, j]) for i in range(3) for j in range(3)]
    axes += [np.cross(an, ae[:, i]) for i in range(3)] + [np.cross(bn, be[:, i]) for i in range(3)]
    for axis in axes:
        pa, pb = np.einsum('nki,ni->nk', a, axis), np.einsum('nki,ni->nk', b, axis)
        length = np.linalg.norm(axis, axis=1)
        separated = (pa.max(1) < pb.min(1)-1e-9*length) | (pb.max(1) < pa.min(1)-1e-9*length)
        hit &= ~(separated & (length >= 1e-15))
    return hit


def intersection_vertices(source, candidate, faces, cancelled=lambda: False):
    if not len(faces):
        return np.empty(0, int)
    triangles = candidate[faces]
    low, high = triangles.min(1), triangles.max(1)
    centers = (low+high)*.5
    radii = np.linalg.norm(high-low, axis=1)*.5
    tree = cKDTree(centers)
    bad = set()
    for start in range(0, len(faces), 512):
        if cancelled():
            raise InterruptedError('Surface intersection audit cancelled')
        end = min(start+512, len(faces))
        neighbors = tree.query_ball_point(centers[start:end], radii[start:end]+radii.max()+1e-9)
        first = np.repeat(np.arange(start, end), [len(v) for v in neighbors])
        second = np.concatenate(neighbors).astype(int)
        select = second > first
        first, second = first[select], second[select]
        if not len(first):
            continue
        select = ((low[first] <= high[second]+1e-9) & (high[first] >= low[second]-1e-9)).all(1)
        first, second = first[select], second[select]
        select = ~(faces[first, :, None] == faces[second, None, :]).any((1, 2))
        first, second = first[select], second[select]
        if not len(first):
            continue
        hits = intersecting(triangles[first], triangles[second])
        first, second = first[hits], second[hits]
        if len(first):
            introduced = ~intersecting(source[faces[first]], source[faces[second]])
            bad.update(faces[np.r_[first[introduced], second[introduced]]].ravel().tolist())
    return np.array(sorted(bad), int)


def fair_surface(points, faces, affine, kind, *, mask=None, mask_origin=None, cancelled=lambda: False):
    points, faces = np.asarray(points, np.float32), np.asarray(faces, np.uint32)
    if not np.isfinite(points).all() or (faces.size and faces.max() >= len(points)):
        raise ValueError('Invalid indexed display surface')
    if surface_kind(kind) == 'context' and len(faces) > POLICY['maxContextTriangles']:
        return points.copy(), {'policy':POLICY['version'], 'kind':'context', 'maxDisplacementMm':0.,
            'restoredComponents':0, 'backedOffVertices':0, 'intersectionCheck':'source-restored',
            'fallbackReason':'high-complexity context; original surface retained'}
    linear = np.asarray(affine, float)[:3, :3]
    h = 1/np.linalg.norm(np.linalg.inv(linear), axis=1).max()
    profile_kind = surface_kind(kind)
    profile = POLICY['profiles'][profile_kind]
    original = points.astype(float) @ linear.T
    limit = np.full(len(points), min(h*profile['voxelAltitudes'], profile.get('maximumMm', np.inf)))
    if profile_kind == 'vessel':
        limit[:] = h/8
        spacing = np.linalg.norm(linear, axis=0)
        if mask is not None and np.allclose((linear/spacing).T @ (linear/spacing), np.eye(3), atol=1e-5):
            radius = ndi.distance_transform_edt(mask, sampling=spacing)
            ix = np.floor(points-(np.zeros(3) if mask_origin is None else mask_origin)+.5).astype(int)
            valid = ((ix >= 0) & (ix < np.asarray(mask.shape))).all(1)
            r = np.zeros(len(points)); r[valid] = radius[tuple(ix[valid].T)]
            limit = np.minimum(.5*h, .25*np.maximum(r, .5*h))
    edge = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    adjacency = sparse.coo_matrix((np.ones(len(edge)*2), (np.r_[edge[:, 0], edge[:, 1]], np.r_[edge[:, 1], edge[:, 0]])), shape=(len(points), len(points))).tocsr()
    degree = np.asarray(adjacency.sum(1)).ravel()
    operator = sparse.diags(1/np.maximum(degree, 1)) @ adjacency
    unique, uses = np.unique(np.sort(edge, axis=1), axis=0, return_counts=True)
    pinned = np.zeros(len(points), bool); pinned[unique[uses != 2].ravel()] = True; pinned |= degree == 0
    _, components = connected_components(adjacency, directed=False)
    c = coefficients(kind)
    filtered = c[0]*original
    previous, current = original.copy(), np.zeros_like(original)
    for j in range(1, len(c)):
        if cancelled():
            raise InterruptedError('Display smoothing cancelled')
        following = operator @ previous if j == 1 else 2*(operator @ previous)-current
        following[pinned] = original[pinned]
        filtered += c[j]*following
        current, previous = previous, following
    delta = filtered-original
    filtered = original+delta*np.minimum(1, limit*.999/np.maximum(np.linalg.norm(delta, axis=1), 1e-30))[:, None]
    inverse = np.linalg.inv(linear)

    def quantize():
        result = (points+(filtered-original) @ inverse.T).astype(np.float32)
        physical = result.astype(float) @ linear.T
        outside = np.linalg.norm(physical-original, axis=1) > limit
        result[outside] = points[outside]; physical[outside] = original[outside]; filtered[outside] = original[outside]
        return result, physical

    result, physical = quantize()
    def normal(p):
        f = p[faces]
        return np.cross(f[:, 1]-f[:, 0], f[:, 2]-f[:, 0])
    source_normals = normal(original)
    backed_off = set()
    def repair_faces():
        nonlocal filtered, result, physical
        invalid = np.zeros(len(faces), bool)
        for attempt in range(16):
            b = normal(physical)
            invalid = (np.einsum('ij,ij->i', source_normals, b) <= 0) | (np.linalg.norm(b, axis=1) < POLICY['minimumFaceAreaRatio']*np.linalg.norm(source_normals, axis=1))
            if not invalid.any(): break
            ids = np.unique(faces[invalid]); backed_off.update(ids.tolist())
            filtered[ids] = original[ids]+(.5 if attempt < 14 else 0)*(filtered[ids]-original[ids])
            result, physical = quantize()
        return invalid
    invalid = repair_faces()
    contacts = np.empty(0, int)
    for attempt in range(4):
        contacts = intersection_vertices(original, physical, faces, cancelled)
        if not len(contacts): break
        backed_off.update(contacts.tolist()); filtered[contacts] = original[contacts]
        result, physical = quantize(); invalid = repair_faces()
    # Remaining failures roll back their component; local repairs have already
    # recovered ordinary slivers without discarding the rest of the anatomy.
    invalid = repair_faces()
    restored = set(components[faces[invalid].ravel()].tolist())
    restored.update(components[intersection_vertices(original, physical, faces, cancelled)].tolist())
    while restored:
        select = np.isin(components, list(restored)); filtered[select] = original[select]
        result, physical = quantize()
        extra = set(components[intersection_vertices(original, physical, faces, cancelled)].tolist())-restored
        if not extra:
            break
        restored.update(extra)
    return result, {'policy': POLICY['version'], 'kind': profile_kind,
                    'maxDisplacementMm': float(np.linalg.norm(physical-original, axis=1).max(initial=0)),
                    'restoredComponents': len(restored), 'backedOffVertices':len(backed_off), 'intersectionCheck': 'source-restored' if restored else 'passed'}
