"""Derive bounded synthetic target support on the TAIR 00273 CTA lattice."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter

from resectionlab.anatomy_model_registry import ct_intensities
from resectionlab.patient_anatomy import world

FAMILY = 'TAIR:00273'
SERIES = 'tair-cta-display'
BRAIN = 'totalsegmentator-skull-e80ddcce5db3-brain'
SKULL = 'totalsegmentator-skull-d7eea0c1b2e5'
VESSEL = 'cta-enhancement-segment'
SKULL_MARGIN_MM = 4.
VESSEL_MARGIN_MM = 4.2
BRIGHT_MARGIN_MM = 1.4
DISTANCE_CAP_MM = 8.
POLICY = (
    'TAIR CTA-sample grid. Placement uses the retained brain estimate and 0.8 mm Gaussian-smoothed '
    'signal 23–74 as a tissue proxy. A neighbourhood threshold avoids treating individual noisy '
    'voxels as CSF or vessels. '
    'The lower cerebral envelope below 20% of its superior extent and the band within 16 mm of the '
    'estimated midline are excluded. Target support lies at least 4 mm inside the source FOV and '
    'at voxel-center separations of 4 mm from retained skull, 4.2 mm from the isolated enhancement '
    'segment, and 1.4 mm from strong enhancement (signal 120 or more with smoothed signal 85 or more, '
    'or signal 200 or more). Distances saturate at 8 mm. '
    'Tissue and contrast thresholds are placement proxies. Distal vessels, veins and CSF borders '
    'remain unqualified.'
)


def _bounds(array, value):
    """Find mask bounds with bounded scratch space on the memory-mapped array."""
    seen = [np.zeros(size, dtype=bool) for size in array.shape]
    for first in range(0, array.shape[0], 16):
        slab = array[first:first + 16] == value
        seen[0][first:first + len(slab)] = slab.any(axis=(1, 2))
        seen[1] |= slab.any(axis=(0, 2))
        seen[2] |= slab.any(axis=(0, 1))
    occupied = [np.flatnonzero(axis) for axis in seen]
    if any(not len(axis) for axis in occupied):
        raise ValueError('Empty retained TAIR brain estimate')
    return np.array([axis[0] for axis in occupied]), np.array([axis[-1] + 1 for axis in occupied])


def _bounded_distance(mask, spacing):
    """Exact distances up to the cap using voxel slabs with physical overlap."""
    if not mask.any():
        raise ValueError('Empty TAIR exclusion mask in the supported crop')
    result = np.empty(mask.shape, dtype=np.float32)
    halo = int(np.ceil(DISTANCE_CAP_MM / spacing[0]))
    for first in range(0, mask.shape[0], 24):
        last = min(first + 24, mask.shape[0])
        low, high = max(0, first - halo), min(last + halo, mask.shape[0])
        slab = mask[low:high]
        if slab.any():
            distance = distance_transform_edt(~slab, sampling=spacing)[first - low:last - low]
            np.minimum(distance, DISTANCE_CAP_MM, out=distance)
            result[first:last] = distance
        else:
            result[first:last] = DISTANCE_CAP_MM
    return result


def placement(study):
    """Supply the same cropped placement fields as the ISLES study adapter."""
    scan = study.series(SERIES)
    affine = np.asarray(scan['affine'], dtype=float)
    linear = affine[:3, :3]
    spacing = np.linalg.norm(linear, axis=0)
    if (study.document.get('data_case', {}).get('source_family_id') != FAMILY
            or study.document['reference_series'] != SERIES or scan['modality'] != 'CTA'
            or scan.get('units_status') != 'mm' or not ct_intensities(study, scan)
            or not np.array_equal(linear, np.diag(np.diag(linear))) or affine[2, 2] <= 0):
        raise ValueError('Expected the source-bound, scanner-aligned TAIR 00273 CTA showcase')
    rows = [study.layer(key) for key in (BRAIN, SKULL, VESSEL)]
    shape = np.asarray(scan['data']['shape'])
    for row, kind in zip(rows, ('brain', 'skull', 'vessels')):
        if (row['kind'] != kind or row['role'] != 'context' or row['source_series'] != SERIES
                or row['provenance'].get('evidence_use') != 'context_only'
                or row['data']['shape'] != scan['data']['shape']
                or not np.array_equal(row['affine'], affine)):
            raise ValueError(f'TAIR placement requires unchanged source-grid context for {row["id"]}')

    brain_native, skull_native, vessel_native = (study.array(row['data']) for row in rows)
    low, high = _bounds(brain_native, rows[0]['label_value'])
    bounds = world(affine, np.stack((low, high - 1)))
    mid = bounds.mean(0)
    superior_cut = bounds[0, 2] + .20 * np.ptp(bounds[:, 2])
    pad = np.ceil(DISTANCE_CAP_MM / spacing).astype(int) + 1
    start = np.maximum(low - pad, 0)
    start[2] = max(start[2], int(np.floor((superior_cut - affine[2, 3]) / spacing[2])) - pad[2])
    stop = np.minimum(high + pad, shape)
    crop = tuple(slice(a, b) for a, b in zip(start, stop))
    local = affine.copy()
    local[:3, 3] = world(affine, start)

    signal = study.array(scan['data'])[crop]
    smooth = gaussian_filter(signal, .8 / spacing)
    domain = brain_native[crop] == rows[0]['label_value']
    domain &= (smooth >= 23.) & (smooth <= 74.)
    bright = ((signal >= 120.) & (smooth >= 85.)) | (signal >= 200.)
    del smooth
    skull_distance = _bounded_distance(skull_native[crop] == rows[1]['label_value'], spacing)
    vessel_distance = _bounded_distance(vessel_native[crop] == rows[2]['label_value'], spacing)
    bright_distance = _bounded_distance(bright, spacing)
    domain &= (skull_distance >= SKULL_MARGIN_MM) & (vessel_distance >= VESSEL_MARGIN_MM)
    domain &= bright_distance >= BRIGHT_MARGIN_MM

    x = local[0, 0] * np.arange(domain.shape[0]) + local[0, 3]
    z = local[2, 2] * np.arange(domain.shape[2]) + local[2, 3]
    domain &= np.abs(x[:, None, None] - mid[0]) >= 16.
    domain &= z[None, None, :] >= superior_cut
    for axis, (first, last, length, size) in enumerate(zip(start, stop, shape, spacing)):
        indexes = np.arange(first, last)
        supported = np.minimum(indexes + .5, length - indexes - .5) * size >= 4.
        supported[[0, -1]] = False
        broadcast = [1, 1, 1]
        broadcast[axis] = len(supported)
        domain &= supported.reshape(broadcast)
    if not domain.any():
        raise ValueError('No TAIR tissue support remains after anatomical and source-coverage gates')
    return domain, local, start, bounds, mid, (skull_distance, vessel_distance, bright_distance)
