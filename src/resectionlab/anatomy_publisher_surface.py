"""Load an immutable publisher surface for an exact source-label revision."""

import numpy as np

from .patient_anatomy import affine, file_hash, world

POLICY = 'publisher-vtp-ras-mm-v1'
MAX_VERTICES = 1_000_000
MAX_TRIANGLES = 2_000_000


def publisher_surface(collection, row, *, cancelled):
    attachment = row.get('publisher_surface')
    if attachment is None:
        return None
    if not isinstance(attachment, dict) or attachment.get('schema') != POLICY:
        raise ValueError('Unsupported publisher surface attachment')
    # Derived display masks and source-label corrections must show their own
    # geometry. The retained publisher asset still belongs to the original.
    if row.get('edits') or row['revision'] != attachment['mask_revision']:
        return None
    source = collection.series(row['source_series'])
    if (row.get('role') != 'context' or row['kind'] not in {'artery', 'vein', 'sinus', 'vessels'}
            or row['data']['sha256'] != attachment['mask_sha256']
            or row['label_value'] != attachment['mask_label_value']
            or row['source_series'] != attachment['source_series']
            or source['original']['sha256'] != attachment['source_sha256']
            or row['data']['shape'] != source['data']['shape']
            or not np.array_equal(row['affine'], source['affine'])
            or not np.array_equal(row['affine'], attachment['source_affine'])):
        raise ValueError('Publisher surface and source-mask binding differ')
    if cancelled():
        raise InterruptedError('Publisher surface preparation cancelled')
    if file_hash(collection.path(source['original']['path'])) != attachment['source_sha256']:
        raise ValueError('Publisher surface acquired source checksum differs')
    original = attachment['original']
    if file_hash(collection.path(original['path'])) != original['sha256']:
        raise ValueError('Publisher surface original checksum differs')
    vertices, triangles = attachment['vertices_ras_mm'], attachment['triangles']
    if not _triple_array(vertices, 'float32', MAX_VERTICES) or not _triple_array(triangles, 'uint32', MAX_TRIANGLES):
        raise ValueError('Publisher surface array layout exceeds display limits')
    ras = collection.array(vertices)
    faces = collection.array(triangles)
    if not np.isfinite(ras).all() or int(faces.max()) >= len(ras):
        raise ValueError('Publisher surface contains invalid points or indices')
    inverse = np.linalg.inv(affine(row['affine']))
    native = world(inverse, ras)
    if (np.any(native < -.5) or np.any(native > np.asarray(row['data']['shape']) - .5)):
        raise ValueError('Publisher surface lies outside the acquired source grid')
    if np.linalg.det(inverse[:3, :3]) < 0:
        faces = faces[:, ::-1].copy()
    return {'source': native, 'positions': native, 'indices': faces,
            'max_displacement_mm': 0., 'policy_version': POLICY, 'cache_hit': False,
            'publisher_source_sha256': original['sha256']}


def _triple_array(ref, dtype, limit):
    if not isinstance(ref, dict):
        return False
    shape = ref.get('shape')
    return (ref.get('dtype') == dtype and isinstance(shape, list) and len(shape) == 2
            and type(shape[0]) is int and 0 < shape[0] <= limit and shape[1] == 3)
