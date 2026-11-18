"""Bounded fairing of display meshes that keeps the source triangle connectivity.

Native views use adaptive context bounds and closed-lesion volume checks. The
shared profile and Chebyshev recurrence are also used by browser replay.
"""
import json
from importlib.resources import files
import numpy as np
from scipy import ndimage as ndi, sparse
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

POLICY = json.loads(files('resectionlab').joinpath('surface_policy.json').read_text())
NATIVE_POLICY = POLICY['native']


def surface_kind(kind):
    return 'tumor' if kind == 'tumor' else 'vessel' if kind in {'vessel', 'vessels', 'artery', 'vein', 'sinus'} else 'context'


def coefficients(kind, *, native_context=False):
    profile = POLICY['profiles'][surface_kind(kind)]
    n, pass_band = (NATIVE_POLICY['contextIterations'], NATIVE_POLICY['contextPassBand']) if native_context else (
        profile['iterations'], profile['passBand'])
    theta = np.arccos(1-pass_band)
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
    return _intersection_vertices(source, candidate, faces, cancelled)


def _intersection_vertices(source, candidate, faces, cancelled, changed_points=None):
    moved = np.any(source != candidate, axis=1)[faces].any(axis=1)
    if not moved.any():
        return np.empty(0, int)
    dirty = np.ones(len(faces), bool) if changed_points is None else changed_points[faces].any(axis=1)
    dirty_faces = np.flatnonzero(dirty)
    if not len(dirty_faces):
        return np.empty(0, int)
    triangles = candidate[faces]
    low, high = triangles.min(1), triangles.max(1)
    centers = (low+high)*.5
    radii = np.linalg.norm(high-low, axis=1)*.5
    tree = cKDTree(centers)
    max_radius = radii.max()
    bad = set()
    for start in range(0, len(dirty_faces), 512):
        if cancelled():
            raise InterruptedError('Surface intersection audit cancelled')
        batch = dirty_faces[start:start+512]
        neighbors = tree.query_ball_point(centers[batch], radii[batch]+max_radius+1e-9)
        first = np.repeat(batch, [len(v) for v in neighbors])
        second = np.concatenate(neighbors).astype(int)
        select = ((second > first) | ~dirty[second]) & (moved[first] | moved[second])
        a, b = first[select], second[select]
        first, second = np.minimum(a, b), np.maximum(a, b)
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


def fair_surface(points, faces, affine, kind, *, mask=None, mask_origin=None, native=False, cancelled=lambda: False):
    points, faces = np.asarray(points, np.float32), np.asarray(faces, np.uint32)
    if points.ndim != 2 or points.shape[1] != 3 or faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError('Display surfaces require three coordinates and three corners per triangle')
    if not np.isfinite(points).all() or (faces.size and faces.max() >= len(points)):
        raise ValueError('Invalid indexed display surface')
    profile_kind = 'tumor' if native and kind == 'lesion-context' else surface_kind(kind)
    native_context = native and profile_kind == 'context'
    policy_version = NATIVE_POLICY['version'] if native else POLICY['version']
    triangle_budget = NATIVE_POLICY['maxContextTriangles'] if native_context else POLICY['maxContextTriangles']
    if profile_kind == 'context' and len(faces) > triangle_budget:
        return points.copy(), {'policy':policy_version, 'kind':'context', 'maxDisplacementMm':0.,
            'restoredComponents':0, 'backedOffVertices':0, 'intersectionCheck':'source-restored',
            'fallbackReason':'Context exceeds the geometry audit budget. Original shape retained.'}
    linear = np.asarray(affine, float)[:3, :3]
    inverse = np.linalg.inv(linear)
    h = 1/np.linalg.norm(inverse, axis=1).max()
    profile = POLICY['profiles'][profile_kind]
    original = points.astype(float) @ linear.T
    limit = min(h*profile['voxelAltitudes'], profile.get('maximumMm', np.inf))
    if native_context:
        limit = NATIVE_POLICY['contextMaximumMm']
    if profile_kind == 'vessel':
        limit = h/8
        spacing = np.linalg.norm(linear, axis=0)
        if mask is not None and np.allclose((linear/spacing).T @ (linear/spacing), np.eye(3), atol=1e-5):
            radius = ndi.distance_transform_edt(mask, sampling=spacing)
            ix = np.floor(points-(np.zeros(3) if mask_origin is None else mask_origin)+.5).astype(int)
            valid = ((ix >= 0) & (ix < np.asarray(mask.shape))).all(1)
            r = np.zeros(len(points)); r[valid] = radius[tuple(ix[valid].T)]
            limit = np.minimum(.5*h, .25*np.maximum(r, .5*h))
    # Packed undirected edges avoid allocating and sorting a second Nx2 array
    # for the dense meshes. Counted edges retain the original operator weights.
    n = len(points)
    edge_keys = np.empty(len(faces)*3, np.uint64)
    for edge in range(3):
        a, b = faces[:, edge], faces[:, (edge+1)%3]
        edge_keys[edge*len(faces):(edge+1)*len(faces)] = np.minimum(a, b).astype(np.uint64)*n + np.maximum(a, b)
    keys, uses = np.unique(edge_keys, return_counts=True)
    del edge_keys
    a, b = (keys//max(n, 1)).astype(np.int32), (keys%max(n, 1)).astype(np.int32)
    del keys
    adjacency = sparse.csr_matrix((np.tile(uses, 2).astype(float), (np.r_[a, b], np.r_[b, a])), shape=(n, n))
    degree = np.asarray(adjacency.sum(1)).ravel()
    pinned = degree == 0
    pinned[a[uses != 2]] = True
    pinned[b[uses != 2]] = True
    del a, b, uses
    _, components = connected_components(adjacency, directed=False)
    operator = sparse.diags(1/np.maximum(degree, 1)) @ adjacency
    del adjacency
    c = coefficients(profile_kind, native_context=native_context)
    filtered = c[0]*original
    previous, current = original.copy(), np.zeros_like(original)
    for j in range(1, len(c)):
        if cancelled():
            raise InterruptedError('Display smoothing cancelled')
        following = operator @ previous if j == 1 else 2*(operator @ previous)-current
        following[pinned] = original[pinned]
        filtered += c[j]*following
        current, previous = previous, following
    del operator, previous, current, following
    lesion_volume = native and profile_kind == 'tumor'
    if lesion_volume:
        counts = np.bincount(components)
        centers = np.column_stack([np.bincount(components, weights=axis)/counts for axis in original.T])
        face_components = components[faces[:, 0]]
        closed = np.ones(len(counts), bool)
        closed[components[pinned]] = False

        def component_volume(positions):
            triangles = positions[faces] - centers[face_components, None]
            determinants = np.einsum('ij,ij->i', triangles[:, 0], np.cross(triangles[:, 1], triangles[:, 2]))
            return np.bincount(face_components, weights=determinants, minlength=len(closed))

        source_volume = component_volume(original)
        closed &= np.abs(source_volume) > 1e-9
        candidate_volume = component_volume(filtered)
        rescale = closed & (source_volume*candidate_volume > 0)
        scale = np.ones(len(closed))
        scale[rescale] = np.cbrt(source_volume[rescale]/candidate_volume[rescale])
        filtered = centers[components] + (filtered-centers[components])*scale[components, None]

    def cap_ratio(delta):
        ratio = np.linalg.norm(delta, axis=1)/limit
        if native_context:
            # Constrain continuous movement both in source-cell units and in
            # RAS millimetres. The wide axis gets its own allowance.
            cells = np.linalg.norm(delta @ inverse.T, axis=1)
            ratio = np.maximum(ratio, cells/NATIVE_POLICY['contextVoxelAltitudes'])
        return ratio

    delta = filtered-original
    filtered = original+delta*np.minimum(1, .999/np.maximum(cap_ratio(delta), 1e-30))[:, None]
    del delta

    def quantize():
        result = (points+(filtered-original) @ inverse.T).astype(np.float32)
        physical = result.astype(float) @ linear.T
        outside = cap_ratio(physical-original) > 1
        result[outside] = points[outside]; physical[outside] = original[outside]; filtered[outside] = original[outside]
        return result, physical

    result, physical = quantize()
    audited = None
    contacts = np.empty(0, int)

    def new_contacts():
        nonlocal audited, contacts
        if physical is not audited:
            # Every face from the reported contacts is restored to source before
            # any subsequent audit. Any remaining introduced contact involves a
            # face with changed quantized coordinates from that audited result.
            changed = None if audited is None else np.any(physical != audited, axis=1)
            contacts = _intersection_vertices(original, physical, faces, cancelled, changed)
            audited = physical
        return contacts

    def normal(p):
        f = p[faces]
        return np.cross(f[:, 1]-f[:, 0], f[:, 2]-f[:, 0])
    source_normals = normal(original)
    minimum_area = POLICY['minimumFaceAreaRatio']**2*np.einsum('ij,ij->i', source_normals, source_normals)
    backed_off = set()
    def repair_faces():
        nonlocal filtered, result, physical
        invalid = np.zeros(len(faces), bool)
        for attempt in range(16):
            b = normal(physical)
            invalid = (np.einsum('ij,ij->i', source_normals, b) <= 0) | (np.einsum('ij,ij->i', b, b) < minimum_area)
            if not invalid.any(): break
            ids = np.unique(faces[invalid]); backed_off.update(ids.tolist())
            filtered[ids] = original[ids]+(.5 if attempt < 14 else 0)*(filtered[ids]-original[ids])
            result, physical = quantize()
        return invalid
    repair_faces()
    for _ in range(4):
        contacts = new_contacts()
        if not len(contacts): break
        backed_off.update(contacts.tolist()); filtered[contacts] = original[contacts]
        result, physical = quantize(); repair_faces()
    # Remaining failures roll back their component; local repairs have already
    # recovered ordinary slivers without discarding the rest of the anatomy.
    invalid = repair_faces()
    restored = set(components[faces[invalid].ravel()].tolist())
    restored.update(components[new_contacts()].tolist())
    if lesion_volume:
        def relative_volume_change(positions):
            change = np.zeros(len(closed))
            change[closed] = np.abs(component_volume(positions)[closed]/source_volume[closed]-1)
            return change

        volume_failures = np.flatnonzero(relative_volume_change(physical) > NATIVE_POLICY['lesionComponentVolumeTolerance'])
        restored.update(volume_failures.tolist())
    while restored:
        select = np.isin(components, list(restored)); filtered[select] = original[select]
        result, physical = quantize()
        extra = set(components[new_contacts()].tolist())-restored
        if not extra:
            break
        restored.update(extra)
    diagnostics = {'policy': policy_version, 'kind': profile_kind,
                   'maxDisplacementMm': float(np.linalg.norm(physical-original, axis=1).max(initial=0)),
                   'restoredComponents': len(restored), 'backedOffVertices':len(backed_off),
                   'intersectionCheck': 'source-restored' if restored else 'passed'}
    if lesion_volume:
        diagnostics['maxComponentVolumeChangeFraction'] = float(relative_volume_change(physical).max(initial=0.))
        diagnostics['volumeRestoredComponents'] = int(len(volume_failures))
    return result, diagnostics
