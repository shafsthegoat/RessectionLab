"""Read-only, source-bound anatomy quality findings, never automatic edits."""
import numpy as np
from scipy import ndimage as ndi
from .patient_anatomy import digest


def mask_quality(mask, source_identity, *, probability=None, review_status='unreviewed', alignment_status=None):
    mask = np.asarray(mask, bool)
    labels, count = ndi.label(mask, np.ones((3, 3, 3)))
    sizes = np.bincount(labels.ravel())
    voxels = int(mask.sum())
    boundary = mask.copy()
    boundary[1:-1, 1:-1, 1:-1] = False
    issues = []
    if not voxels: issues.append('Empty segmentation: anatomy is missing or localization failed.')
    if count > 1: issues.append(f'{count:,} disconnected source components; inspect before correcting.')
    touching = int(boundary.sum())
    if touching: issues.append(f'{touching:,} foreground cells touch the acquisition boundary; coverage may be truncated.')
    mismatch = None
    if probability is not None:
        p = np.asarray(probability)
        if p.shape != mask.shape or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
            issues.append('Probability field is invalid or belongs to a different grid.')
        else:
            mismatch = int(np.count_nonzero((p >= .5) != mask))
            if mismatch: issues.append(f'{mismatch:,} mask cells disagree with the stored 0.5 probability threshold.')
    if alignment_status and alignment_status not in {'reference', 'reviewed', 'source-provided'}:
        issues.append('Source alignment requires review.')
    return {'sourceIdentity': source_identity, 'voxelCount': voxels, 'componentCount': int(count),
            'largestComponentFraction': float(sizes[1:].max()/voxels) if voxels else None,
            'boundaryTouchingVoxels': touching, 'probabilityMismatchVoxels': mismatch,
            'reviewStatus': review_status, 'issues': issues, 'accuracy': 'unmeasured'}


def layer_quality(collection, row):
    identity = digest({'snapshot': collection.identity(), 'data': row['data'], 'label': row['label_value'],
                       'affine': row['affine'], 'probability': row.get('probability'), 'edits': row.get('edits')})
    cache = getattr(collection, '_quality_cache', {})
    if identity not in cache:
        probability = collection.array(row['probability']) if row.get('probability') and not row.get('edits') else None
        transform = collection.document['transforms'].get(row['source_series'], {})
        cache[identity] = mask_quality(collection.mask(row), identity, probability=probability,
            review_status=row.get('review_status', 'unreviewed'), alignment_status=transform.get('status', 'unknown'))
        # Keep only the latest few generations of findings in this process.
        if len(cache) > 512: cache.pop(next(iter(cache)))
        collection._quality_cache = cache
    return cache[identity]
