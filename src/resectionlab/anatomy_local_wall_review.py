"""Source-bound registration review for one scored native-wall crop."""

from itertools import product

import numpy as np

from .anatomy_import import registration_landmarks
from .patient_anatomy import digest, file_hash, world


SCHEMA = 'resectionlab.native-wall-alignment-review.v1'
PROPERTY = 'native_wall_alignment_review'


def _source_binding(collection, brain, vessels):
    sid = vessels['source_series']
    ref = collection.document['reference_series']
    if sid == ref or brain['source_series'] != ref:
        raise ValueError('Select reference-scan brain tissue and a separate vascular acquisition')
    from .surgical_patient import _check_layer_export, _source_registration, _wall_layer_issue

    for role, row in [('brain', brain), ('vessels', vessels)]:
        _check_layer_export(collection, role, row)
        issue = _wall_layer_issue(role, row)
        if issue:
            raise ValueError(issue)
    series = [collection.series(id_) for id_ in (ref, sid)]
    matrix, pending = _source_registration(collection, series[1])
    if pending != [sid]:
        raise ValueError('The vascular alignment must be the only unreviewed transform in its chain')
    sources = {x['id']: {'originalSha256': x['original']['sha256'], 'timepoint': x['timepoint'],
                         'nativeAffine': x['affine'], 'nativeArray': x['data']} for x in series}
    return {'sourcePatient': collection.document['patient_id'], 'referenceSeries': ref, 'vascularSeries': sid,
            'brainLayerId': brain['id'], 'vesselLayerId': vessels['id'], 'matrix': matrix.tolist(),
            'sources': sources, 'brainLayerDigest': digest(brain), 'vesselLayerDigest': digest(vessels)}


def review_native_wall_alignment(collection, series_id, moving_points, reference_points, reviewer, scope):
    """Attach a local record, keeping the candidate unreviewed for general use."""
    if type(scope) is not dict or set(scope) != {'brainLayerId', 'vesselLayerId', 'center'}:
        raise ValueError('Select brain and vessels and the reviewed local wall center')
    brain = collection.layer(scope['brainLayerId'])
    vessels = collection.layer(scope['vesselLayerId'])
    if vessels['source_series'] != series_id:
        raise ValueError('Use paired landmarks from the selected vascular acquisition')
    binding = _source_binding(collection, brain, vessels)
    landmarks = registration_landmarks(collection, series_id, moving_points, reference_points, reviewer)
    reference_points = np.asarray(landmarks['reference'])
    moving_points = np.asarray(landmarks['moving'])
    for points in (reference_points, moving_points):
        thickness = np.linalg.svd(points - points.mean(axis=0), compute_uv=False)[-1] / np.sqrt(len(points))
        if len(points) < 4 or thickness < .25:
            raise ValueError('Local native-wall alignment review requires four non-coplanar landmark pairs')
    for sid in binding['sources']:
        original = collection.series(sid)['original']
        if file_hash(collection.path(original['path'])) != original['sha256']:
            raise ValueError('Original source checksum differs')

    from .anatomy_sampling import sample_layer
    from .surgical_patient_wall import native_wall_grid

    grid = native_wall_grid(collection, vessels, np.asarray(binding['matrix']), scope['center'])
    landmark_voxels = world(np.linalg.inv(grid.affine), reference_points)
    within = ((landmark_voxels >= -.5) & (landmark_voxels <= np.asarray(grid.mask.shape) - .5)).all(axis=1)
    residuals = np.asarray(landmarks['errors_mm'])
    near_center = np.linalg.norm(reference_points - grid.center, axis=1) <= 8.
    if np.count_nonzero(within & (residuals <= 1.5)) < 2 or not np.any(within & near_center & (residuals <= 1.)):
        raise ValueError('Place the reviewed crop near its paired source evidence')
    reference = collection.series(binding['referenceSeries'])
    a = collection.patient_affine(reference)
    corners = list(product(*zip(np.zeros(3), np.asarray(grid.mask.shape) - 1)))
    ref_voxels = world(np.linalg.inv(a) @ grid.affine, corners)
    if np.any(ref_voxels < 0.) or np.any(ref_voxels > np.asarray(reference['data']['shape']) - 1):
        raise ValueError('The reviewed crop extends outside the acquired structural grid')
    sampled, covered = sample_layer(collection, brain, grid.affine, grid.mask.shape, lambda: None, {})
    if not covered.all() or not np.any(sampled & ~grid.mask):
        raise ValueError('The reviewed crop requires covered patient tissue samples')
    record = {'schema': SCHEMA, 'binding': binding, 'landmarks': landmarks,
              'centerRasMm': grid.center.tolist(), 'cropStartIJK': grid.start.tolist(), 'cropShape': list(grid.mask.shape),
              'scope': 'One local native-wall crop for research rehearsal'}
    record['digest'] = digest(record)
    collection.document['transforms'][series_id][PROPERTY] = record
    return record


def native_wall_alignment_review(collection, brain, vessels, grid=None):
    """Resolve a current local review, optionally requiring the exact prepared crop."""
    record = collection.document['transforms'].get(vessels['source_series'], {}).get(PROPERTY)
    if not isinstance(record, dict) or record.get('schema') != SCHEMA:
        return None
    if digest({k: v for k, v in record.items() if k != 'digest'}) != record.get('digest'):
        return None
    try:
        if record.get('binding') != _source_binding(collection, brain, vessels):
            return None
    except (ValueError, OSError, StopIteration):
        return None
    if grid is not None and (record.get('cropStartIJK') != grid.start.tolist() or record.get('cropShape') != list(grid.mask.shape)):
        raise ValueError('The selected vessel cursor lies outside the reviewed local native-wall crop')
    return record
