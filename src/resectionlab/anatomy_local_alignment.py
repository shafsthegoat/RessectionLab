"""Read-only local acquisition overlap and image correspondence for scan review."""

from itertools import product

import numpy as np
from scipy import ndimage as ndi

from .patient_anatomy import file_hash, world


FIELD_HALF_MM = 12.
PROBE_SPACING_MM = .5
PROBE_HALF_MM = 3.
SEARCH_HALF_MM = 2.5
FILTER_MM = .85


def _sample_grid(center, half, spacing):
    offsets = np.arange(-round(half / spacing), round(half / spacing) + 1) * spacing
    return center + np.stack(np.meshgrid(*[offsets] * 3, indexing='ij'), axis=-1)


def _indices(points, voxel_from_patient, shape):
    indices = world(voxel_from_patient, points)
    return indices, ((indices >= 0.) & (indices <= np.asarray(shape) - 1)).all(axis=-1)


def _features(center, half, data, voxel_from_patient):
    # Filtering needs source samples beyond the comparison and shift windows.
    margin = round(3 * FILTER_MM / PROBE_SPACING_MM)
    points = _sample_grid(center, half + margin * PROBE_SPACING_MM, PROBE_SPACING_MM)
    indices, covered = _indices(points, voxel_from_patient, data.shape)
    if not covered.all():
        return None
    signal = ndi.map_coordinates(data, indices.reshape(-1, 3).T, order=1,
                                 output=np.float32, prefilter=False).reshape(covered.shape)
    if not np.isfinite(signal).all():
        return None
    gradients = np.stack([ndi.gaussian_filter(signal, FILTER_MM / PROBE_SPACING_MM,
                          order=tuple(int(i == axis) for i in range(3)), truncate=3.) for axis in range(3)], axis=-1)
    return gradients[margin:-margin, margin:-margin, margin:-margin]


def _probe(center, fixed_data, moving_data, fixed_voxel, moving_voxel):
    report = {'centerRasMm': center.tolist(), 'status': 'insufficient_source_coverage'}
    fixed = _features(center, PROBE_HALF_MM, fixed_data, fixed_voxel)
    moving = _features(center, PROBE_HALF_MM + SEARCH_HALF_MM, moving_data, moving_voxel)
    if fixed is None or moving is None:
        return report
    # Squared gradient orientation accommodates opposite local contrast (for
    # example, flowing blood appears bright in TOF and dark in structural MRI).
    fixed_strength = np.linalg.norm(fixed, axis=-1)
    moving_strength = np.linalg.norm(moving, axis=-1)
    fixed_scale, moving_scale = (np.percentile(x, 75) for x in (fixed_strength, moving_strength))
    if fixed_scale <= 1e-6 or moving_scale <= 1e-6:
        return {**report, 'status': 'weak_image_structure'}
    fixed /= np.hypot(fixed_strength, fixed_scale)[..., None]
    moving /= np.hypot(moving_strength, moving_scale)[..., None]
    limit = round(SEARCH_HALF_MM / PROBE_SPACING_MM)
    shifts = np.array(list(product(range(-limit, limit + 1), repeat=3)))
    width = len(fixed)
    # Sort equal scores by distance from the existing candidate. A flat or
    # repetitive response remains ambiguous, regardless of the selected peak.
    shifts = shifts[np.argsort(np.sum(shifts**2, axis=1), kind='stable')]
    scores = np.empty(len(shifts))
    for i, shift in enumerate(shifts):
        crop = tuple(slice(start, start + width) for start in shift + limit)
        dots = np.sum(fixed * moving[crop], axis=-1)
        scores[i] = np.mean(dots**2)
    peak = int(np.argmax(scores))
    best = shifts[peak]
    far = np.linalg.norm(shifts - best, axis=1) * PROBE_SPACING_MM >= 1.5
    separation = float(scores[peak] - scores[far].max())
    status = ('search_boundary' if np.any(abs(best) >= limit) else
              'competing_matches' if separation <= .005 else 'distinguishable_peak')
    return {**report, 'status': status, 'candidateScore': float(scores[0]), 'bestScore': float(scores[peak]),
            'separatedPeakDifference': separation, 'movingSamplingShiftMm': (best * PROBE_SPACING_MM).tolist()}


def local_alignment_evidence(collection, moving_id, center, *, check=lambda: None):
    """Screen a prospective 24 mm cube against the stored transform, without fitting it."""
    if (not isinstance(center, (list, tuple)) or len(center) != 3
            or any(type(x) not in (int, float) or not np.isfinite(x) for x in center)):
        raise ValueError('Select a finite patient RAS review point')
    center = np.asarray(center, float)
    reference = collection.series(collection.document['reference_series'])
    moving = collection.series(moving_id)
    if moving_id == reference['id']:
        raise ValueError('Choose the moving acquisition to inspect its alignment')
    transform = collection.registration_matrix(moving_id, require_geometry=True)
    fixed_voxel = np.linalg.inv(collection.patient_affine(reference))
    moving_voxel = np.linalg.inv(transform @ moving['affine'])
    for row in (reference, moving):
        check()
        original = row['original']
        if file_hash(collection.path(original['path'])) != original['sha256']:
            raise ValueError(f"Original source changed for {row['id']}")
    fixed_data = collection.array(reference['data'])
    moving_data = collection.array(moving['data'])
    field = _sample_grid(center, FIELD_HALF_MM, 1.)
    _, fixed_covered = _indices(field, fixed_voxel, fixed_data.shape)
    _, moving_covered = _indices(field, moving_voxel, moving_data.shape)
    patches = []
    for offset in product((-8., 0., 8.), repeat=3):
        check()
        patches.append(_probe(center + offset, fixed_data, moving_data, fixed_voxel, moving_voxel))
    check()
    return {'schema': 'resectionlab.local-alignment-evidence.v1', 'snapshotId': collection.identity(),
            'movingSeriesId': moving_id, 'referenceSeriesId': reference['id'],
            'movingImageSha256': moving['original']['sha256'], 'referenceImageSha256': reference['original']['sha256'],
            'movingToPatientRasMm': transform.tolist(), 'centerRasMm': center.tolist(),
            'transformStatus': collection.document['transforms'][moving_id]['status'],
            'fieldWidthMm': 2 * FIELD_HALF_MM, 'coverageSpacingMm': 1.,
            'sampleCount': fixed_covered.size, 'commonCoverageCount': int((fixed_covered & moving_covered).sum()),
            'referenceCoverageCount': int(fixed_covered.sum()), 'movingCoverageCount': int(moving_covered.sum()),
            'shiftStepMm': PROBE_SPACING_MM, 'shiftLimitMm': SEARCH_HALF_MM, 'patches': patches}
