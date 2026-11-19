"""Matched sections through two independently replayed endpoints in patient space."""
import base64
from io import BytesIO

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

from .anatomy_model_registry import source_volume
from .anatomy_sections import section


def _section_normal(entries, targets, selected):
    # Both entries and the selected treatment line lie in the section. At a
    # shared entry use the two approach directions to determine the plane.
    ray = targets[selected] - entries[selected]
    ray = ray / np.linalg.norm(ray)
    for offset in (entries[1] - entries[0], targets[1-selected] - entries[selected],
                   np.eye(3)[np.argmin(abs(ray))]):
        normal = np.cross(ray, offset)
        length = np.linalg.norm(normal)
        if length > 1e-5:
            return normal / length
    raise ValueError('No section plane for the recorded approach')


def _section_cell_ids(plane, affine, shape):
    """Nearest source cell at each pixel centre; -1 outside the engine grid."""
    size = plane['size']
    y, x = (np.indices((size, size)) - (size - 1) / 2) * plane['spacingMm']
    points = plane['center'] + x[..., None] * plane['u'] + y[..., None] * plane['v']
    inv = np.linalg.inv(affine)
    ijk = np.floor(points @ inv[:3, :3].T + inv[:3, 3] + .5).astype(np.int64)
    inside = np.all((ijk >= 0) & (ijk < shape), axis=-1)
    ids = np.full(inside.shape, -1, dtype=np.int64)
    ids[inside] = np.ravel_multi_index(ijk[inside].T, shape)
    return ids


def _align_orientation(plane, first):
    """Changing approach planes should preserve the reader's left/right and up.

    The source section chooses its lattice basis independently for each
    normal. Swap/flip the pixel lattice and its RAS directions together.
    """
    u, v = np.asarray(plane['u']), np.asarray(plane['v'])
    pixels = plane['pixels']
    right, down = first['u'], first['v']
    if abs(v @ right) + abs(u @ down) > abs(u @ right) + abs(v @ down):
        pixels, u, v = pixels.swapaxes(0, 1), v, u
    if u @ right < 0:
        pixels, u = pixels[:, ::-1], -u
    if v @ down < 0:
        pixels, v = pixels[::-1], -v
    plane.update(pixels=pixels, u=u.tolist(), v=v.tolist())


def _encode_endpoint(pixels, ids, contour, removed):
    affected = np.isin(ids, removed)
    rgb = pixels[..., :3].copy()
    rgb[contour] = [90, 204, 211]
    rgb[affected] = [249, 173, 80]
    encoded = BytesIO()
    Image.fromarray(rgb).save(encoded, format='PNG')
    return {'png': base64.b64encode(encoded.getvalue()).decode('ascii'),
            'removedPixels': int(affected.sum())}


def comparison_sections(collection, arms, grids, check):
    check()
    reference = collection.series(collection.document['reference_series'])
    if not (reference.get('original') or {}).get('sha256') or not source_volume(reference):
        return {'unavailable': 'Matched sections require an acquired reference volume with a source identity.'}
    registration = collection.document.get('transforms', {}).get(reference['id'], {})
    if registration.get('status') not in ('reference', 'reviewed', 'source-provided'):
        return {'unavailable': 'The reference image lacks a reviewed patient-space placement for matched sections.'}
    # Validate the admitted geometry directly. Surface checksum failures and
    # other source errors before the general section viewer's native fallback.
    collection.patient_affine(reference)
    entries = np.array([arm['entry'] for arm in arms])
    targets = np.array([arm['target'] for arm in arms])
    affine, shape = grids[0]['affine'], grids[0]['shape']
    # The replay verifier binds a single target to an identical engine scene
    # for both arms. Keep only sparse memberships and removal IDs after replay.
    original = grids[0]['original']
    points = np.vstack((entries, targets))
    midpoint = (points.max(axis=0) + points.min(axis=0)) / 2
    planes = []
    for selected in range(2):
        check()
        normal = _section_normal(entries, targets, selected)
        center = midpoint - normal * ((midpoint - entries[selected]) @ normal)
        size = 320
        # Shared physical field fits the routes plus surrounding source anatomy.
        radius = np.linalg.norm(points - center, axis=1).max()
        pitch = max(.20, (2 * radius + 32.) / (size - 1))
        if pitch > 5.:
            return {'unavailable': 'The two approaches exceed the physical field supported by matched sections.'}
        plane = section(collection, series_id=reference['id'], center=center, normal=normal,
                        size=size, spacing_mm=pitch, check=check)
        if plane['frame'] != 'patient':
            raise ValueError('Recorded endpoint sections require a patient-space reference image')
        if planes:
            _align_orientation(plane, planes[0])
        ids = _section_cell_ids(plane, affine, shape)
        annotation = np.isin(ids, original)
        contour = annotation & ~ndi.binary_erosion(annotation, border_value=1)
        views = [_encode_endpoint(plane['pixels'], ids, contour, grid['removed']) for grid in grids]
        planes.append({**{k: plane[k] for k in ('center', 'u', 'v', 'spacingMm', 'size')}, 'arms': views})
    return {'seriesId': reference['id'], 'modality': reference['modality'],
            'sourceSha256': reference['original']['sha256'], 'planes': planes}
