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


def _partition_ids(collection, supplied, whole, attachment):
    """Match a named partition to the same original points and triangles."""
    vertex_ref = attachment.get('publisher_vertex_ids')
    face_ref = attachment.get('publisher_face_ids')
    if not (vertex_ref and face_ref):
        return None
    vertices = collection.array(vertex_ref)
    faces = collection.array(face_ref)
    if (vertices.dtype != np.uint32 or faces.dtype != np.uint32
            or vertices.shape != (len(supplied['source']),) or faces.shape != (len(supplied['indices']),)
            or np.any(vertices >= len(whole['source'])) or np.any(faces >= len(whole['indices']))
            or not np.array_equal(whole['source'][vertices], supplied['source'])
            or not np.array_equal(whole['indices'][faces], vertices[supplied['indices']])):
        return None
    return vertices


def conditioned_publisher_cta(collection, row, supplied, *, cancelled):
    """Fair the complete indexed VTP once so named class seams remain coincident."""
    from .anatomy_geometry import normals
    from .surface_fairing import fair_surface

    scan = collection.series(row['source_series'])
    if scan['modality'] != 'CTA' or scan.get('units_status') != 'mm':
        return supplied
    if cancelled():
        raise InterruptedError('Publisher surface preparation cancelled')
    attachment = row['publisher_surface']
    partitioned = 'publisher_vertex_ids' in attachment or 'publisher_face_ids' in attachment
    full, whole = row, supplied
    partition = None
    if partitioned:
        parents = []
        for layer in getattr(collection, 'document', {}).get('layers', []):
            binding = layer.get('publisher_surface') or {}
            if (binding.get('original') == attachment['original']
                    and binding.get('source_sha256') == attachment['source_sha256']
                    and layer.get('source_series') == row['source_series']
                    and 'publisher_vertex_ids' not in binding and 'publisher_face_ids' not in binding):
                parents.append(layer)
        if len(parents) != 1:
            return supplied
        full, = parents
    full_attachment = full['publisher_surface']
    key = (full['id'], full['revision'], full['data']['sha256'], full['label_value'],
           full_attachment['vertices_ras_mm']['sha256'], full_attachment['triangles']['sha256'])
    cached = getattr(collection, '_publisher_cta_fairing', None)
    cache_hit = cached is not None and cached[0] == key
    if cache_hit and partitioned:
        collection.array(full_attachment['vertices_ras_mm'])
        collection.array(full_attachment['triangles'])
    if not cache_hit:
        if partitioned:
            whole = publisher_surface(collection, full, cancelled=cancelled)
            if whole is None:
                return supplied
            partition = _partition_ids(collection, supplied, whole, attachment)
            if partition is None:
                return supplied
        raw, triangles = whole['source'], whole['indices']
        if len(triangles) < 4 or len(triangles) > 200_000:
            return supplied
        mask = collection.mask(full)
        if not mask.any():
            return supplied
        # Add a two-cell halo beyond the VTP and native positive cells.
        bounds = [np.flatnonzero(mask.any(axis=tuple(i for i in range(3) if i != axis))) for axis in range(3)]
        lower = np.maximum(0, np.minimum([axis[0] for axis in bounds], np.floor(raw.min(axis=0))).astype(int) - 2)
        upper = np.minimum(mask.shape, np.maximum([axis[-1] for axis in bounds], np.ceil(raw.max(axis=0))).astype(int) + 3)
        if np.prod(upper - lower, dtype=np.int64) > 12_000_000:
            return supplied
        region = tuple(slice(a, b) for a, b in zip(lower, upper))
        positions, audit = fair_surface(raw, triangles, affine(full['affine']), full['kind'],
                                        mask=np.pad(mask[region], 1), mask_origin=lower - 1,
                                        source_shape=mask.shape, categorical=True, publisher_cta=True,
                                        cancelled=cancelled)
        cached = collection._publisher_cta_fairing = (key, whole, positions, normals(positions, triangles), audit)
    _, whole, positions, vertex_normals, audit = cached
    if audit['maxDisplacementMm'] == 0.:
        return supplied
    if partitioned:
        if partition is None:
            partition = _partition_ids(collection, supplied, whole, attachment)
        if partition is None:
            return supplied
        positions = positions[partition]
        vertex_normals = vertex_normals[partition]
    source = supplied['source'].astype(np.float32)
    displacement = (positions - source) @ affine(row['affine'])[:3, :3].T
    return {**supplied, 'positions': positions, 'normals': vertex_normals, 'shared_publisher_normals': True,
            'policy_version': audit['policy'], 'cache_hit': cache_hit,
            'max_displacement_mm': float(np.linalg.norm(displacement, axis=1).max(initial=0.)),
            'backed_off_vertices': audit['backedOffVertices'], 'restored_components': audit['restoredComponents']}
