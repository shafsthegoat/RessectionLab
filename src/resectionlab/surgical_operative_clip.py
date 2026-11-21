"""Source-bound scalp clip on the incised edge of the generated cranial phantom."""

import numpy as np
from scipy.spatial.transform import Rotation


def rest(engine, obj):
    """The source point is the arterial pressure contact. The backing is external."""
    scene = engine.scene
    cell = obj['sourceCell']
    key = (cell, obj['radiusMm'])
    cache = getattr(scene, '_operative_clip_shape', None)
    if cache is None:
        cache = scene._operative_clip_shape = {}
    if key in cache:
        return cache[key]
    contact = scene.points([cell])[0]
    exposure = scene.properties['operativeExposureV1']
    body = exposure['bodies']['scalp']
    outward = np.asarray(body['outward'])
    margin = scene.properties['gliomaMaterialsV1']['scalpSeamCell']
    inward = scene.points([margin])[0] - contact
    inward -= outward * (inward @ outward)
    length = np.linalg.norm(inward)
    if length < 1e-6:
        raise ValueError('OPERATIVE_SCALP_CLIP_REQUIRES_ADJACENT_CUT_MARGIN')
    inward /= length
    spacing = float(scene.spacing.min())
    skin_labels = np.unique(scene.labels.ravel()[body['cells']])
    sample = contact + np.outer(np.arange(0.0, 8.0 * spacing, 0.25 * spacing), outward)
    ijk = np.rint(sample @ scene.inverse[:3, :3].T + scene.inverse[:3, 3]).astype(int)
    inside = np.all((ijk >= 0) & (ijk < scene.shape), axis=1)
    ids = np.ravel_multi_index(ijk[inside].T, scene.shape)
    skin = scene.points(ids[np.isin(scene.labels.ravel()[ids], skin_labels)])
    if not len(skin):
        raise ValueError('OPERATIVE_SCALP_CLIP_REQUIRES_EXTERNAL_SKIN')
    half = 0.5 * np.abs(scene.affine[:3, :3].T @ outward).sum()
    height = float(np.max((skin - contact) @ outward)) + half + obj['radiusMm'] + 0.08
    backing = contact - 0.9 * spacing * inward + height * outward
    jaw = contact + 0.42 * length * inward + 0.18 * spacing * outward
    neck = contact + 0.42 * length * inward + (height - 0.55 * spacing) * outward
    coords = (
        (backing, backing, obj['radiusMm'], 'backing'),
        (contact, jaw, 0.14, 'pressure-jaw'),
        (jaw, neck, 0.14, 'neck'),
        (neck, backing, 0.16, 'neck'),
    )
    cache[key] = (backing, coords, inward)
    return cache[key]


def segments(engine, obj, position=None, rotation=None):
    backing, geometry, _ = rest(engine, obj)
    if position is None and obj['status'] == 'placed':
        return geometry
    origin = np.asarray(obj['positionMm'] if position is None else position)
    transform = Rotation.from_quat(
        obj.get('carriedRotation', [0.0, 0.0, 0.0, 1.0]) if rotation is None else rotation
    )
    return [
        (origin + transform.apply(a - backing), origin + transform.apply(b - backing), r, kind)
        for a, b, r, kind in geometry
    ]


def check_retrieval_sweep(engine, old, new):
    """Check the backing and individual metal jaws against the retained grid.

    A contact may disengage from the same source cells it compressed. Each
    subsequent microstep must continue through the physically cleared incision.
    """
    start = set().union(*map(set, engine.touched_many(tuple(s[:3] for s in old))))
    swept = []
    for (a, b, r, _), (c, d, _, _) in zip(old, new):
        if np.allclose(a, b):
            swept.append((a, c, r))
            continue
        travel = max(np.linalg.norm(c - a), np.linalg.norm(d - b))
        parts = max(1, int(np.ceil(travel / 0.08)))
        for i in range(parts):
            t = (i + 0.5) / parts
            swept.append((a + t * (c - a), b + t * (d - b), r + travel / (2 * parts)))
    # The cell-cover kernel accepts at most 64 queries in a transaction.
    for offset in range(0, len(swept), 64):
        if check := getattr(engine, '_transition_check', None):
            check()
        if any(set(ids) - start for ids in engine.touched_many(swept[offset : offset + 64])):
            raise ValueError('OPERATIVE_CARRIED_OBJECT_COLLISION')
    return [(a, b, r, 'fixture') for a, b, r in swept]
