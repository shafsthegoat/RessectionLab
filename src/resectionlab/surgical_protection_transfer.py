"""Bounded conservative transfer of a verified inferior exclusion container.

This is geometry, not registration or anatomy review. It never grants a review
status, task permission or clinical claim. Source padding is not observed air.
The separate outside-grid mask preserves that unknown rather than clearing it.
"""
from __future__ import annotations

from dataclasses import dataclass
import itertools
import math
import re

import numpy as np

from .core import array_digest, immutable_array, semantic_digest


def _shape(value):
    shape = tuple(value)
    if (len(shape) != 3 or any(isinstance(n, (bool, np.bool_)) or
            not isinstance(n, (int, np.integer)) or n < 1 for n in shape)):
        raise ValueError('A positive three-dimensional grid is required')
    return tuple(map(int, shape))


def _affine(value):
    result = np.array(value, dtype=float, copy=True)
    if (result.shape != (4, 4) or not np.isfinite(result).all() or
            not np.allclose(result[3], [0., 0., 0., 1.], rtol=0., atol=1e-12) or
            abs(np.linalg.det(result[:3, :3])) < 1e-12):
        raise ValueError('A finite nonsingular physical affine is required')
    return result


def _bound(value, name, maximum=None):
    if (type(value) not in (int, float) or not math.isfinite(value) or value < 0 or
            maximum is not None and value > maximum):
        raise ValueError('Invalid '+name)
    return float(value)


@dataclass(frozen=True)
class InferiorSlab:
    shape: tuple[int, int, int]
    last_plane: int
    source_array_hash: str

    def declaration(self):
        return {'shape': list(self.shape), 'includedNativeK': [0, self.last_plane],
                'sourceArrayHash': self.source_array_hash}


def verify_inferior_slab(data, *, label_value=1, check=lambda: None):
    """Inspect every source cell without building a source point cloud.

    Only an exact binary full-x/y k0..K container is accepted. A hole, extra
    disconnected plane or other label refuses; a claimed recipe is insufficient.
    The returned facts carry no anatomical/registration review qualification.
    """
    data = np.asarray(data)
    shape = _shape(data.shape)
    if data.dtype.kind not in 'bui' or type(label_value) is not int or label_value < 1:
        raise ValueError('An integer/bool binary exclusion array is required')
    if not callable(check):
        raise TypeError('A callable cancellation check is required')
    last, ended = -1, False
    for k in range(shape[2]):
        check()
        plane = data[:, :, k]
        count = int(np.count_nonzero(plane == label_value))
        if count == plane.size and not ended:
            last = k
        elif count == 0 and not np.any(plane):
            ended = True
        else:
            raise ValueError('Source is not an exact contiguous inferior slab')
    if last < 0:
        raise ValueError('A nonempty source exclusion is required')
    check()
    return InferiorSlab(shape, last, array_digest(data))


def transfer_inferior_slab(slab, source_affine, target_shape, target_affine,
        source_to_target_ras_mm, *, assumed_translation_mm=0.,
        assumed_rotation_degrees=0., rotation_pivot_target_ras_mm=None,
        max_target_cells=100_000_000, chunk_cells=65_536, check=lambda: None):
    """Enclose every intersecting complete target cell for continuous stresses.

    The explicit nominal map is SOURCE-world RAS mm -> TARGET-world RAS mm.
    Assumed perturbations translate by an L2 ball and rotate about the supplied
    target-world pivot, with arbitrary axes up to the angular bound. These are
    declared stress sets, never measured registration-error distributions.

    Map each full target parallelepiped into source index coordinates. Exact
    coordinate support plus an analytic rigid-error enclosure gives an AABB;
    its overlap with the finite source slab can overexclude, never miss a true
    intersection. Source-grid escape is returned separately as unknown coverage.
    Both masks are combined in protectedMask. Interior source padding/other
    missing evidence still needs its own explicit source/coverage review.
    """
    if not isinstance(slab, InferiorSlab):
        raise TypeError('Verified slab geometry is required')
    source_shape, shape = _shape(slab.shape), _shape(target_shape)
    if (type(slab.last_plane) is not int or not 0 <= slab.last_plane < source_shape[2] or
            type(slab.source_array_hash) is not str or
            not re.fullmatch('sha256:[a-f0-9]{64}', slab.source_array_hash)):
        raise ValueError('Invalid slab geometry declaration')
    for value, maximum in ((max_target_cells, None), (chunk_cells, 262_144)):
        if type(value) is not int or value < 1 or maximum is not None and value > maximum:
            raise ValueError('Positive bounded allocation/chunk limits are required')
    total = math.prod(shape)
    if total > max_target_cells:
        raise ValueError('Target-cell allocation budget exceeded before transfer')
    if not callable(check):
        raise TypeError('A callable cancellation check is required')
    source, target, registration = map(_affine,
        (source_affine, target_affine, source_to_target_ras_mm))
    rotation = registration[:3, :3]
    if (not np.allclose(rotation.T@rotation, np.eye(3), rtol=0., atol=1e-8) or
            not np.isclose(np.linalg.det(rotation), 1., rtol=0., atol=1e-8)):
        raise ValueError('Nominal source registration must be a proper rigid RAS-mm map')
    translation = _bound(assumed_translation_mm, 'assumed translation bound')
    angle = _bound(assumed_rotation_degrees, 'assumed rotation bound', 180.)
    if rotation_pivot_target_ras_mm is None:
        if angle:
            raise ValueError('An explicit target-RAS rotation pivot is required')
        pivot = None
    else:
        pivot = np.array(rotation_pivot_target_ras_mm, float, copy=True)
        if pivot.shape != (3,) or not np.isfinite(pivot).all():
            raise ValueError('A finite target-RAS rotation pivot is required')
    check()
    inverse = np.linalg.inv(source)@np.linalg.inv(registration)
    mapping = inverse@target
    support = .5*np.abs(mapping[:3, :3]).sum(axis=1)
    error_to_index = np.linalg.norm(inverse[:3, :3], axis=1)
    corners = np.asarray(list(itertools.product((-.5, .5), repeat=3)))
    target_radius = float(np.linalg.norm(corners@target[:3, :3].T, axis=1).max())
    rotation_scale = 2.*math.sin(math.radians(angle)/2.)
    low = np.full(3, -.5)
    full_high = np.asarray(source_shape, float)-.5
    slab_high = full_high.copy(); slab_high[2] = slab.last_plane+.5
    hit, outside = np.empty(total, bool), np.empty(total, bool)
    for first in range(0, total, chunk_cells):
        check()
        last = min(total, first+chunk_cells)
        indices = np.column_stack(np.unravel_index(np.arange(first, last), shape))
        centers = indices@mapping[:3, :3].T+mapping[:3, 3]
        error = np.full(last-first, translation)
        if angle:
            world = indices@target[:3, :3].T+target[:3, 3]
            error += rotation_scale*(np.linalg.norm(world-pivot, axis=1)+target_radius)
        half = support+error[:, None]*error_to_index
        lo, hi = centers-half, centers+half
        hit[first:last] = np.all((hi >= low-1e-10) & (lo <= slab_high+1e-10), axis=1)
        # Numerical boundary uncertainty must enlarge the unknown, never trim
        # it. Closed source-face contact is conservatively unknown too.
        outside[first:last] = np.any((lo <= low+1e-10) | (hi >= full_high-1e-10), axis=1)
    check()
    hit, outside = hit.reshape(shape), outside.reshape(shape)
    combined = hit | outside
    report = {'schema': 'resectionlab.conservative-inferior-slab-transfer.v1',
        'sourceSlab': slab.declaration(), 'sourceAffineRasMm': source.tolist(),
        'targetShape': list(shape), 'targetAffineRasMm': target.tolist(),
        'sourceToTargetRasMm': registration.tolist(),
        'assumedStress': {'translationL2Mm': translation, 'rotationDegrees': angle,
            'rotationPivotTargetRasMm': None if pivot is None else pivot.tolist(),
            'meaning': 'Assumed continuous stress bounds; not measured registration error'},
        'slabOverlapCells': int(hit.sum()), 'outsideSourceGridCells': int(outside.sum()),
        'combinedProtectedCells': int(combined.sum()), 'protectedMaskHash': array_digest(combined),
        'sourceGridOutsideMaskHash': array_digest(outside), 'nativeActions': 0,
        'sourceBoundaryIndexGuard': 1e-10,
        'registrationOrAnatomyQualified': False, 'clinicalValidation': 'none',
        'method': 'Complete affine-cell coordinate support and analytic continuous rigid-stress enclosure; source-grid outside is unknown/protected'}
    report['contentHash'] = semantic_digest(report)
    return {'slabOverlapMask': immutable_array(hit),
            'outsideSourceGridMask': immutable_array(outside),
            'protectedMask': immutable_array(combined), 'report': report}
