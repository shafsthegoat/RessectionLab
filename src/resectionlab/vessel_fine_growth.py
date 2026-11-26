"""Signal-supported fine branches of a CT angiography vessel tree.

The named models and the connected single-class addition stop where a vessel
narrows to one or two voxels. The acquired CTA still shows those branches as
thin bright tubes. This module follows them outward from the tree and returns
them as a separate, display-only estimate with an uncalibrated score. It reads
the acquired image and never changes it or any label.

Rule (``FineGrowth``; thresholds are Hounsfield units and millimetres, so the
rule is refused for anything but CT):

1. Region. The intracranial envelope (supplied, or estimated here from soft
   tissue), widened by ``envelope_margin_mm``, minus everything within
   ``bone_clearance_mm`` of bone (``bone_hu`` and above) or within
   ``air_clearance_mm`` of air (``air_hu`` and below).
2. Tube response. For each scale in ``sigmas_mm`` the Hessian of the image
   clipped to ``clip_hu`` is scale-normalised; with eigenvalues ordered
   ``l1 <= l2 <= l3`` a bright tube has ``l2 < 0`` and the response is
   ``|l2| * (1 - |l3| / |l2|)`` clipped at zero. Sheets (bone edges) and blobs
   score low. The response is the largest over the scales, in HU.
3. Candidates. Region voxels outside the tree with intensity in
   ``[floor_hu, ceiling_hu]`` and response at least ``low``, minus thick parts:
   voxels within one voxel of any ball of radius ``blob_radius_mm`` of tissue
   at ``blob_floor_hu`` or brighter (enhancing glands, sinus plexus). Those
   are not fine branches, and the rim of a blob responds like a tube.
4. Hysteresis from the tree. A 26-connected candidate piece is kept when it
   touches the tree, holds a voxel with response at least ``high`` and has at
   least ``min_piece_mm3`` of volume. Nothing unconnected is ever returned.

The score of a kept voxel rises linearly from 0.5 at ``low`` to 1 at ``high``.
It orders voxels by tube response; it is not a probability.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy import ndimage as ndi

VERSION = 'ct-fine-branch-growth-v1'
FULL = np.ones((3, 3, 3), bool)


@dataclass(frozen=True)
class FineGrowth:
    """Frozen on the development CTA (TopCoW-014) only."""

    version: str = VERSION
    sigmas_mm: tuple = (0.5, 0.8, 1.2)
    low: float = 6.0
    high: float = 20.0
    floor_hu: float = 60.0
    ceiling_hu: float = 450.0
    clip_hu: tuple = (0.0, 500.0)
    bone_hu: float = 700.0
    bone_clearance_mm: float = 1.5
    # Mucosa between air cells is thin, bright against air and never a cerebral vessel.
    air_hu: float = -300.0
    air_clearance_mm: float = 3.0
    envelope_margin_mm: float = 2.0
    blob_radius_mm: float = 1.3
    # Thickness is judged on tissue clearly brighter than noisy enhanced parenchyma.
    blob_floor_hu: float = 110.0
    min_piece_mm3: float = 4.0
    # Envelope estimate: smoothed soft tissue, opened by this depth so the scalp and face separate from the brain.
    envelope_soft_hu: tuple = (5.0, 120.0)
    envelope_smooth_mm: float = 1.5
    envelope_open_mm: float = 4.0

    def __post_init__(self):
        if not 0 < self.low <= self.high:
            raise ValueError('Response thresholds must satisfy 0 < low <= high')
        if not self.floor_hu < self.ceiling_hu < self.bone_hu or not self.clip_hu[0] < self.clip_hu[1]:
            raise ValueError('Intensity bounds must satisfy floor < ceiling < bone')
        if not self.sigmas_mm or min(self.sigmas_mm) <= 0:
            raise ValueError('At least one positive scale is required')
        if min(self.bone_clearance_mm, self.air_clearance_mm, self.envelope_margin_mm, self.blob_radius_mm, self.min_piece_mm3) < 0:
            raise ValueError('Distances and volumes must not be negative')


def _spacing(spacing):
    spacing = np.asarray(spacing, float)
    if spacing.shape != (3,) or not np.isfinite(spacing).all() or (spacing <= 0).any():
        raise ValueError('Three positive voxel spacings in millimetres are required')
    return spacing


def _require_ct(modality):
    from .vascular_rooting import image_modality

    if image_modality(modality) != 'ct':
        raise ValueError(f'Fine-branch growth uses Hounsfield thresholds and is not applied to {modality!r}')


def _crop(mask, pad, shape):
    found = ndi.find_objects(mask.view(np.uint8))
    if not found or found[0] is None:
        return None
    return tuple(slice(max(0, s.start - p), min(n, s.stop + p)) for s, p, n in zip(found[0], pad, shape))


def intracranial_envelope(image, spacing, rule=FineGrowth()):
    """Estimated intracranial soft tissue of a head CT: the largest soft-tissue body left after an opening that
    separates the brain from scalp and face, grown back and filled. An estimate for limiting a search, not a
    brain segmentation."""
    spacing = _spacing(spacing)
    image = np.asarray(image, np.float32)
    # The envelope is coarse (an opening of several millimetres): work near 1 mm so the distance
    # transforms of a large grid stay small, and return it on the full grid.
    stride = np.maximum(1, np.floor(1.2 / spacing)).astype(int)
    if (stride > 1).any():
        coarse = intracranial_envelope(image[::stride[0], ::stride[1], ::stride[2]], spacing * stride, rule)
        for axis, step in enumerate(stride):
            coarse = np.repeat(coarse, step, axis=axis)
        coarse = coarse[tuple(slice(0, n) for n in image.shape)]
        return coarse & (image < rule.bone_hu)
    smooth = ndi.gaussian_filter(np.clip(image, -100, 300), rule.envelope_smooth_mm / spacing)
    soft = (smooth > rule.envelope_soft_hu[0]) & (smooth < rule.envelope_soft_hu[1])
    del smooth
    inner = ndi.distance_transform_edt(soft, sampling=spacing) > rule.envelope_open_mm
    del soft
    labels, count = ndi.label(inner)
    del inner
    if not count:
        return np.zeros(image.shape, bool)
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    core = labels == sizes.argmax()
    del labels
    grown = ndi.distance_transform_edt(~core, sampling=spacing) <= rule.envelope_open_mm + float(spacing.min())
    grown &= image < rule.bone_hu
    return ndi.binary_fill_holes(grown)


def tube_response(image, spacing, where, rule=FineGrowth(), *, check=lambda: None):
    """Bright-tube response (HU) at the voxels of ``where``; zero elsewhere."""
    spacing = _spacing(spacing)
    clipped = np.clip(np.asarray(image, np.float32), *rule.clip_hu)
    index = np.flatnonzero(where)
    response = np.zeros(image.shape, np.float32)
    if not len(index):
        return response
    best = np.zeros(len(index), np.float32)
    for sigma_mm in rule.sigmas_mm:
        hessian = np.empty((len(index), 3, 3), np.float32)
        for a in range(3):
            for b in range(a, 3):
                order = [0, 0, 0]
                order[a] += 1
                order[b] += 1
                # Derivatives per voxel become per millimetre, then scale-normalised by sigma squared.
                values = ndi.gaussian_filter(clipped, sigma_mm / spacing, order=order, mode='nearest').ravel()[index]
                hessian[:, a, b] = hessian[:, b, a] = values * (sigma_mm ** 2 / (spacing[a] * spacing[b]))
                check()
        eigenvalues = np.linalg.eigvalsh(hessian)
        del hessian
        middle, largest = np.abs(eigenvalues[:, 1]), np.abs(eigenvalues[:, 2])
        tube = np.where(eigenvalues[:, 1] < 0, middle * np.clip(1 - largest / np.maximum(middle, 1e-6), 0, 1), 0)
        np.maximum(best, tube, out=best)
        del eigenvalues, middle, largest, tube
        check()
    response.ravel()[index] = best
    return response


def grow_fine_branches(image, spacing, tree, modality, *, envelope=None, rule=FineGrowth(), check=lambda: None):
    """Follow thin bright tubes outward from ``tree`` on a CT angiogram.

    Returns ``(score, mask, receipt)``: a float32 score that is zero outside the grown voxels, the grown mask
    (disjoint from the tree, every piece 26-connected to it) and a receipt of the rule and what it found.
    """
    _require_ct(modality)
    spacing = _spacing(spacing)
    image = np.asarray(image, np.float32)
    tree = np.asarray(tree).astype(bool)
    if image.ndim != 3 or tree.shape != image.shape:
        raise ValueError('The image and the tree must share one three-dimensional grid')
    if not tree.any():
        raise ValueError('A non-empty vessel tree is required')
    voxel_mm3 = float(np.prod(spacing))
    receipt = {'version': VERSION, 'rule': asdict(rule), 'grid': {'shape': list(image.shape), 'spacingMm': spacing.round(6).tolist()},
               'scoreMeaning': 'tube response mapped linearly from 0.5 at low to 1 at high; not a probability'}
    empty = np.zeros(image.shape, np.float32), np.zeros(image.shape, bool)
    if envelope is None:
        envelope = intracranial_envelope(image, spacing, rule)
        receipt['envelope'] = {'source': 'estimated from soft tissue in this CT'}
    else:
        envelope = np.asarray(envelope).astype(bool)
        if envelope.shape != image.shape:
            raise ValueError('The envelope must be on the image grid')
        receipt['envelope'] = {'source': 'supplied'}
    receipt['envelope']['volumeMl'] = round(float(envelope.sum() * voxel_mm3 / 1000), 1)
    check()
    counts = {'candidateVoxels': 0, 'thickVoxelsExcluded': 0, 'pieces': 0, 'piecesTouchingTree': 0, 'piecesKept': 0,
              'grownVoxels': 0, 'grownVolumeMm3': 0.0}
    receipt['counts'] = counts
    pad = np.ceil((rule.envelope_margin_mm + 3 * max(rule.sigmas_mm) + 2) / spacing).astype(int)
    box = _crop(envelope, pad, image.shape)
    if box is None:
        receipt['reason'] = 'No intracranial envelope was found; nothing was grown.'
        return (*empty, receipt)
    local, local_tree = image[box], tree[box]
    region = ndi.distance_transform_edt(~envelope[box], sampling=spacing) <= rule.envelope_margin_mm
    region &= ndi.distance_transform_edt(local < rule.bone_hu, sampling=spacing) > rule.bone_clearance_mm
    region &= ndi.distance_transform_edt(local > rule.air_hu, sampling=spacing) > rule.air_clearance_mm
    check()
    eligible = region & ~local_tree & (local >= rule.floor_hu) & (local <= rule.ceiling_hu)
    del region
    response = tube_response(local, spacing, eligible, rule, check=check)
    candidates = eligible & (response >= rule.low)
    counts['candidateVoxels'] = int(candidates.sum())
    if rule.blob_radius_mm and candidates.any():
        # Thickness is judged on the bright tissue itself: the rim of a blob responds like a tube.
        core = ndi.distance_transform_edt(eligible & (local >= rule.blob_floor_hu), sampling=spacing) > rule.blob_radius_mm
        if core.any():
            reach = rule.blob_radius_mm + float(spacing.max())
            thick = candidates & (ndi.distance_transform_edt(~core, sampling=spacing) <= reach)
            counts['thickVoxelsExcluded'] = int(thick.sum())
            candidates &= ~thick
            del thick
        del core
    del eligible
    check()
    labels, count = ndi.label(candidates, FULL)
    counts['pieces'] = int(count)
    if not count:
        receipt['reason'] = 'No thin bright tube lies beside the tree.'
        return (*empty, receipt)
    sizes = np.bincount(labels.ravel(), minlength=count + 1)
    touching = np.unique(labels[ndi.binary_dilation(local_tree, FULL) & candidates])
    touching = touching[touching > 0]
    strong = np.unique(labels[candidates & (response >= rule.high)])
    kept = np.intersect1d(touching, strong[strong > 0])
    kept = kept[sizes[kept] * voxel_mm3 >= rule.min_piece_mm3]
    counts.update(piecesTouchingTree=int(len(touching)), piecesKept=int(len(kept)))
    lookup = np.zeros(count + 1, bool)
    lookup[kept] = True
    grown = lookup[labels]
    del labels, candidates
    check()
    mask = np.zeros(image.shape, bool)
    mask[box] = grown
    score = np.zeros(image.shape, np.float32)
    span = max(rule.high - rule.low, 1e-6)
    score[box][grown] = 0.5 + 0.5 * np.clip((response[grown] - rule.low) / span, 0, 1)
    counts.update(grownVoxels=int(grown.sum()), grownVolumeMm3=round(float(grown.sum() * voxel_mm3), 4))
    if grown.any():
        # Label-free evidence: what the grown voxels look like in the acquired image beside the tree they extend.
        shell = ndi.binary_dilation(grown | local_tree, FULL, iterations=4) & ~ndi.binary_dilation(grown | local_tree, FULL, iterations=2)
        lumen, background = float(np.median(local[local_tree])), float(np.median(local[shell])) if shell.any() else None
        values = local[grown]
        receipt['evidence'] = {
            'definition': 'Acquired intensity of the grown voxels, the tree (lumen) and a shell two to four voxels outside both '
                          '(background); contrast is (grown - background) / (lumen - background) on medians.',
            'grownHuP5P50P95': [round(float(v), 1) for v in np.percentile(values, [5, 50, 95])],
            'treeLumenHuMedian': round(lumen, 1), 'backgroundHuMedian': None if background is None else round(background, 1),
            'contrast': None if background is None or lumen <= background else round((float(np.median(values)) - background) / (lumen - background), 3),
            'insideEnvelopeFraction': round(float(envelope[box][grown].mean()), 3),
        }
    return score, mask, receipt
