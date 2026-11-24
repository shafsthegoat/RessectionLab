"""Read-only, source-bound anatomy quality findings, never automatic edits."""
import numpy as np
from scipy import ndimage as ndi
from .patient_anatomy import digest


def empty_tumor_reason(record, modality):
    """Explain a thresholded model result only when its recorded scores support it."""
    if record.get("availability") != "empty_output":
        return None
    peak = record.get("maximum_score")
    parameters = record.get("parameters")
    cutoff = parameters.get("threshold") if isinstance(parameters, dict) else None
    if not all(type(value) in (int, float) for value in (peak, cutoff)) or not 0 <= peak < cutoff <= 1:
        return None
    display_peak = f"{peak:.4g}" if float(f"{peak:.4g}") < cutoff else str(peak)
    return (f"No tumor voxels reached the model threshold of {cutoff:g} on {modality}. "
            f"The highest score was {display_peak}. Review the source scan or import a supplied tumor label for this scan.")


def mask_quality(mask, source_identity, *, probability=None, review_status='unreviewed', alignment_status=None,
                 empty_reason=None):
    mask = np.asarray(mask, bool)
    labels, count = ndi.label(mask, np.ones((3, 3, 3)))
    sizes = np.bincount(labels[mask], minlength=count + 1)
    voxels = int(mask.sum())
    issues = []
    if not voxels: issues.append(empty_reason or 'The segmentation contains no labeled voxels. The anatomy remains unassessed.')
    if count > 1: issues.append(f'{count:,} disconnected source components; inspect before correcting.')
    touching = voxels - int(mask[1:-1, 1:-1, 1:-1].sum())
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


def layer_quality(collection, row, *, verify_probability=True):
    identity = digest({'snapshot': collection.identity(), 'data': row['data'], 'label': row['label_value'],
                       'affine': row['affine'], 'probability': row.get('probability'), 'edits': row.get('edits')})
    cache = getattr(collection, '_quality_cache', {})
    cache_key = (identity, verify_probability)
    if cache_key not in cache:
        probability = collection.array(row['probability']) if verify_probability and row.get('probability') and not row.get('edits') else None
        transform = collection.document['transforms'].get(row['source_series'], {})
        reason = None
        if probability is not None and row["kind"] == "tumor":
            provenance = row.get("provenance", {})
            record = provenance.get("native_rerun", provenance)
            if record.get("availability") == "empty_output" and record.get("maximum_score") == float(probability.max()):
                reason = empty_tumor_reason(record, collection.series(row["source_series"])["modality"])
        cache[cache_key] = mask_quality(collection.mask(row), identity, probability=probability,
            review_status=row.get('review_status', 'unreviewed'), alignment_status=transform.get('status', 'unknown'),
            empty_reason=reason)
        # Keep only the latest few generations of findings in this process.
        if len(cache) > 512: cache.pop(next(iter(cache)))
        collection._quality_cache = cache
    return cache[cache_key]
