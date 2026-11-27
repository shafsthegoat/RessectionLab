"""Root-reachable vessel display with signal-supported bridges and calibre detail.

Everything here is a display derivative of one binary vessel mask. Nothing is
deleted: every input voxel ends in exactly one of three outputs, and bridge
voxels are a fourth, separate output that never enters a source label.

``shown``
    Input voxels 26-connected to a root, directly or through an accepted bridge,
    and kept by the calibre level of detail.
``review``
    Input voxels of components that reach no root and were not bridged, and on
    CT the input voxels that lie in air.
``lod_hidden``
    Input voxels of thin terminal subtrees hidden by the level of detail.
``bridge``
    Added voxels along accepted, signal-supported gaps. They take the class of
    the tree voxel they join.

Roots are the named basilar, internal carotid and vertebral arteries, and on CT
the vein of Galen, straight sinus and superior sagittal sinus. Without named
labels the largest component is the tree, and the receipt says so.

Air on CT. With the acquired CT, an input voxel below ``ct_air_hu`` is not a
vessel whatever named or scored it: it goes to review before components are
found, so it cannot be a root, join pieces or carry a bridge. A model run on a
whole head names and scores voxels in the mastoid cells, the sinuses and the
air around the head; a brain-only field of view rarely shows this.

Bridging rule. An unattached component of at least ``min_fragment_mm3`` is
re-attached when a path from it to the shown tree exists whose centre-to-centre
length is at most ``max_gap_mm`` and whose every non-mask voxel is supported:

* acquired signal: ``(I - B) / (L - B)`` lies in ``[contrast_floor,
  contrast_ceiling]``, where ``L`` is the median signal of mask voxels and ``B``
  the median signal of a ``shell_inner_mm`` to ``shell_outer_mm`` shell outside
  the mask, both inside the fragment's window (the fragment's bounds padded by
  ``max_gap_mm + shell_outer_mm``). The criterion is used only when the window
  shows lumen contrast, ``L - B >= min_contrast_noise * sigma_B`` with sigma
  from the shell's median absolute deviation. The ceiling keeps bone and
  calcification out of a CT bridge.
* model probability: probability at least ``probability_floor``.

A voxel qualifies through either criterion that is available. The shortest
supported path is found by a geodesic search (``skimage.graph.MCP_Geometric``)
in physical millimetres. Up to ``bridge_rounds`` rounds let a chain of
fragments attach one bounded gap at a time. The path is widened to the smaller
local radius of its two ends (at most ``max_bridge_radius_mm``), only through
supported voxels.

Level of detail. Calibre is lumen diameter, estimated at each voxel as
``2 * max(EDT within calibre_reach_mm) - smallest spacing`` (the voxel-centre
distance transform overstates the radius by half a voxel on average). Calibre
``always_mm`` and above is always shown. ``local_mm`` up to ``always_mm`` is
shown within ``local_margin_mm`` of the tumour mask or of a corridor capsule
(entry to target, approach radius). Thinner voxels, and local-tier voxels away
from the tumour and corridors, are candidates. Named vessels and roots are
exempt. A candidate component is hidden only when it touches at most one
component of kept voxels: a thin terminal subtree. Thin connectors between kept
parts stay, so the shown tree never disconnects; a final check reverts the level
of detail if any shown component lost its root.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product

import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from skimage.graph import MCP_Geometric

VERSION = 'vascular-rooting-v2'
FULL = np.ones((3, 3, 3), bool)
FACE = ndi.generate_binary_structure(3, 1)
ARTERY, VEIN, UNCLASSIFIED = 1, 2, 3
CLASS_NAMES = {ARTERY: 'artery', VEIN: 'vein', UNCLASSIFIED: 'unclassified'}
LAYER_CLASSES = {'artery': ARTERY, 'vein': VEIN, 'sinus': VEIN, 'vessels': UNCLASSIFIED, 'vessel': UNCLASSIFIED}
_OFFSETS = np.array([offset for offset in product((-1, 0, 1), repeat=3) if any(offset)])
_FACE_OFFSETS = _OFFSETS[np.abs(_OFFSETS).sum(axis=1) == 1]
_JOIN_OFFSETS = _OFFSETS[np.abs(_OFFSETS).sum(axis=1) > 1]
# Depth is kept as uint8 multiples of this step (0 to 5.1 mm) so that the
# 95-million-voxel TOF grids stay within the memory budget.
DEPTH_STEP_MM = 0.02

# TopBrain v1 label maps as shipped in the UZH TopBrain 2025 image
# (/opt/app/label_jsons/labels_topbrain_CT.json, sha256 cb5a27bb71f466fd...,
# and labels_topbrain_MR.json, sha256 a2f04ada4a8d3971...). Labels 1-34 are
# shared arteries; CT adds veins and sinuses, MR adds external carotid branches.
_SHARED = {
    1: 'BA', 2: 'R-P1P2', 3: 'L-P1P2', 4: 'R-ICA', 5: 'R-M1', 6: 'L-ICA', 7: 'L-M1', 8: 'R-Pcom', 9: 'L-Pcom',
    10: 'Acom', 11: 'R-A1A2', 12: 'L-A1A2', 13: 'R-A3', 14: 'L-A3', 15: '3rd-A2', 16: '3rd-A3', 17: 'R-M2',
    18: 'R-M3', 19: 'L-M2', 20: 'L-M3', 21: 'R-P3P4', 22: 'L-P3P4', 23: 'R-VA', 24: 'L-VA', 25: 'R-SCA',
    26: 'L-SCA', 27: 'R-AICA', 28: 'L-AICA', 29: 'R-PICA', 30: 'L-PICA', 31: 'R-AChA', 32: 'L-AChA',
    33: 'R-OA', 34: 'L-OA',
}
TOPBRAIN_V1_CT = {**_SHARED, 35: 'VoG', 36: 'StS', 37: 'ICVs', 38: 'R-BVR', 39: 'L-BVR', 40: 'SSS'}
TOPBRAIN_V1_MR = {**_SHARED, 35: 'R-ECA', 36: 'L-ECA', 37: 'R-STA', 38: 'L-STA', 39: 'R-MaxA', 40: 'L-MaxA',
                  41: 'R-MMA', 42: 'L-MMA'}


@dataclass(frozen=True)
class LabelScheme:
    name: str
    modality: str
    names: dict
    veins: tuple
    roots: tuple

    def label_class(self, value):
        return VEIN if value in self.veins else ARTERY


SCHEMES = {
    # Roots: BA, R-ICA, L-ICA, R-VA, L-VA, then VoG, StS and SSS on CT.
    'topbrain-v1-ct': LabelScheme('topbrain-v1-ct', 'ct', TOPBRAIN_V1_CT, tuple(range(35, 41)),
                                  (1, 4, 6, 23, 24, 35, 36, 40)),
    'topbrain-v1-mr': LabelScheme('topbrain-v1-mr', 'mr', TOPBRAIN_V1_MR, (), (1, 4, 6, 23, 24)),
}


# Named major vessels: none of them may be left out of the default view. The
# TopBrain maps merge A1 with A2 and P1 with P2, so those segments are one label.
_MAJOR_ARTERIES = (1, 2, 3, 4, 5, 6, 7, 11, 12, 15, 17, 19, 23, 24)  # BA, P1P2, ICA, M1, A1A2, 3rd-A2, M2, VA
MAJOR_VESSELS = {
    'topbrain-v1-ct': _MAJOR_ARTERIES + (35, 36, 40),  # plus VoG, StS, SSS
    'topbrain-v1-mr': _MAJOR_ARTERIES,
}


def image_modality(value):
    """Acquisition family of a series modality: 'ct' or 'mr'."""
    text = str(value or '').strip().upper()
    if text.startswith('CT'):
        return 'ct'
    if text.startswith(('MR', 'TOF', 'PC-MR', 'SWI', 'T1', 'T2', 'FLAIR')):
        return 'mr'
    raise ValueError(f'Unknown acquisition modality {value!r}')


@dataclass(frozen=True)
class RootingPolicy:
    version: str = VERSION
    # Bridging (frozen on TopCoW-014 CTA and EDEN PTNT_4509 TOF only).
    max_gap_mm: float = 3.0
    min_fragment_mm3: float = 0.5
    contrast_floor: float = 0.5
    contrast_ceiling: float = 2.0
    min_contrast_noise: float = 3.0
    shell_inner_mm: float = 1.0
    shell_outer_mm: float = 3.0
    # Lumen and background are measured within this distance of the closest
    # fragment-to-tree voxel pair (a few vessel diameters), whatever the window.
    statistics_radius_mm: float = 6.0
    probability_floor: float = 0.25
    # Absolute CT veto: enhanced lumen stays below about 450 HU on the hero
    # CTAs; mineralised skull is above 575 HU in 95 percent of its voxels.
    ct_ceiling_hu: float = 700.0
    # Absolute CT floor: air. The fine-branch rule uses the same level
    # (vessel_fine_growth); a lumen voxel that is two fifths air is not lumen.
    ct_air_hu: float = -300.0
    bridge_rounds: int = 3
    max_bridge_radius_mm: float = 1.0
    max_fragments: int = 20_000
    # Roots without named labels: the largest component, plus components that
    # enter through the inferior face of the field of view.
    entry_min_extent_mm: float = 10.0
    # Trunks entering the head (carotid, vertebral) are wider than scalp and neck branches.
    entry_min_calibre_mm: float = 2.0
    # Level of detail.
    always_mm: float = 1.0
    local_mm: float = 0.5
    local_margin_mm: float = 10.0
    calibre_reach_mm: float = 0.6
    calibre_sphere_mm: float = 1.5
    calibre_tolerance_mm: float = 0.1

    def __post_init__(self):
        if not 0 < self.local_mm <= self.always_mm:
            raise ValueError('Calibre tiers must satisfy 0 < local <= always')
        if not 0 < self.contrast_floor < self.contrast_ceiling:
            raise ValueError('Contrast floor must be positive and below the ceiling')
        if self.max_gap_mm <= 0 or self.bridge_rounds < 0 or self.shell_inner_mm >= self.shell_outer_mm:
            raise ValueError('Invalid bridging bounds')


@dataclass
class RootedVessels:
    shown: np.ndarray
    bridge: np.ndarray
    review: np.ndarray
    lod_hidden: np.ndarray
    classes: np.ndarray
    receipt: dict
    # On CT with the acquired signal: the input voxels in air, a subset of review.
    air: np.ndarray | None = None

    @property
    def display(self):
        return self.shown | self.bridge


def _volume(value, name, shape=None, *, dtype=None):
    # asanyarray keeps a memory map recognisable, so support fields are read sequentially.
    array = np.asanyarray(value) if dtype is None else np.asarray(value, dtype=dtype)
    if array.ndim != 3 or not all(array.shape):
        raise ValueError(f'{name} must be a non-empty three-dimensional array')
    if shape is not None and array.shape != shape:
        raise ValueError(f'{name} must share the vessel mask grid')
    return array


def _frame(affine, spacing):
    if affine is not None:
        affine = np.asarray(affine, float)
        if affine.shape != (4, 4) or not np.isfinite(affine).all() or abs(np.linalg.det(affine[:3, :3])) < 1e-12:
            raise ValueError('A finite invertible 4 x 4 voxel-to-millimetre affine is required')
        measured = np.linalg.norm(affine[:3, :3], axis=0)
        if spacing is not None and not np.allclose(measured, spacing, rtol=1e-4):
            raise ValueError('Spacing disagrees with the affine')
        spacing = measured
    if spacing is None:
        raise ValueError('Voxel spacing or an affine is required')
    spacing = np.asarray(spacing, float)
    if spacing.shape != (3,) or not np.isfinite(spacing).all() or (spacing <= 0).any():
        raise ValueError('Spacing must be three positive millimetre values')
    return affine, spacing


def _crop_bounds(mask, pad):
    lo, hi = [], []
    for axis in range(3):
        present = np.flatnonzero(mask.any(axis=tuple(a for a in range(3) if a != axis)))
        lo.append(max(0, int(present[0]) - int(pad[axis])))
        hi.append(min(mask.shape[axis], int(present[-1]) + int(pad[axis]) + 1))
    return np.array(lo), np.array(hi)


def _label(mask, structure=FULL):
    try:
        return ndi.label(mask, structure, output=np.uint16)
    except RuntimeError:
        return ndi.label(mask, structure)


def _lookup(table, labels, rows=8 << 20):
    """``table[labels]`` in slabs: indexing with a uint16 grid would first widen it to int64."""
    out = np.empty(labels.shape, table.dtype)
    step = max(1, rows // max(1, int(np.prod(labels.shape[1:]))))
    for start in range(0, labels.shape[0], step):
        out[start:start + step] = table[labels[start:start + step]]
    return out


def _counts(labels, length, rows=8 << 20):
    """``np.bincount`` of a label grid in slabs, for the same reason."""
    total = np.zeros(length, np.int64)
    step = max(1, rows // max(1, int(np.prod(labels.shape[1:]))))
    for start in range(0, labels.shape[0], step):
        total += np.bincount(labels[start:start + step].ravel(), minlength=length)
    return total


def _scaled(points, spacing):
    return np.asarray(points, np.float64) * spacing


def _boundary(mask):
    """Background voxels 26-adjacent to the mask: they hold every nearest background point."""
    shell = ndi.binary_dilation(mask, FULL)
    shell[mask] = False
    return shell


def _depth(mask, spacing):
    """Distance (mm) from each mask voxel centre to the nearest background voxel centre.

    Equal to the Euclidean distance transform on the mask. The nearest
    background voxel always borders the mask, so only that shell is searched.
    Returned as uint8 multiples of DEPTH_STEP_MM, saturating at 5.1 mm.
    """
    depth = np.zeros(mask.shape, np.uint8)
    points = np.argwhere(mask)
    if not len(points):
        return depth
    cap = 255 * DEPTH_STEP_MM
    shell = np.argwhere(_boundary(mask))
    if len(shell):
        distance, _ = cKDTree(_scaled(shell, spacing)).query(_scaled(points, spacing), distance_upper_bound=cap)
    else:
        distance = np.full(len(points), np.inf)
    depth[tuple(points.T)] = np.clip(np.rint(np.minimum(distance, cap) / DEPTH_STEP_MM), 0, 255).astype(np.uint8)
    return depth


def _distance_to(points, target, spacing, upper, *, nearest=False):
    """Distance (mm) from voxel points to the nearest target voxel, inf beyond ``upper``.

    With ``nearest`` also return that target voxel for each point (-1 where none is in reach).
    """
    inner = ndi.binary_erosion(target, FULL, border_value=1)
    np.logical_xor(target, inner, out=inner)
    surface = np.argwhere(inner)
    del inner
    if not len(points) or not len(surface):
        far = np.full(len(points), np.inf)
        return (far, np.full((len(points), 3), -1)) if nearest else far
    distance, index = cKDTree(_scaled(surface, spacing)).query(_scaled(points, spacing), distance_upper_bound=upper)
    if not nearest:
        return distance
    found = np.full((len(points), 3), -1)
    hit = np.isfinite(distance)
    found[hit] = surface[index[hit]]
    return distance, found


def _load_crop(array, crop, block_bytes=64 << 20):
    """Crop of a support field.

    A memory-mapped volume is read once, front to back, in large blocks, and
    kept as float16 (ample for contrast fractions and probabilities, half the
    memory on 95-million-voxel TOF grids): fragment windows would otherwise
    fault pages all over it, which crawls on an external drive. In-memory
    arrays are used as they are.
    """
    if isinstance(array, np.memmap) and array.filename and array.flags.c_contiguous:
        import os

        if os.path.getsize(array.filename) - array.offset == array.nbytes:
            out = np.empty(tuple(s.stop - s.start for s in crop), np.float16)
            plane = int(np.prod(array.shape[1:])) * array.itemsize
            rows = max(1, block_bytes // plane)
            with open(array.filename, 'rb', buffering=0) as handle:
                for start in range(crop[0].start, crop[0].stop, rows):
                    stop = min(start + rows, crop[0].stop)
                    handle.seek(array.offset + start * plane)
                    block = np.empty((stop - start, *array.shape[1:]), array.dtype)
                    handle.readinto(memoryview(block).cast('B'))
                    out[start - crop[0].start:stop - crop[0].start] = block[:, crop[1], crop[2]]
            return out
    return np.asarray(array)[crop]


_WINDOW_EXTRA = 0  # Extra search-window padding in voxels; tests use it to show the window does not matter.


def _search_window(low, high, pad, centre, stats_pad, shape):
    """The near part of a fragment padded for the path search, joined with the
    statistics neighbourhood around the closest pair (and its background shell)."""
    first = np.minimum(low - pad, np.floor(centre).astype(int) - stats_pad) - _WINDOW_EXTRA
    last = np.maximum(high + 1 + pad, np.ceil(centre).astype(int) + 1 + stats_pad) + _WINDOW_EXTRA
    return tuple(slice(int(max(0, a)), int(min(n, b))) for a, b, n in zip(first, last, shape))


def _window(slices, pad, shape):
    return tuple(slice(max(0, s.start - p), min(n, s.stop + p)) for s, p, n in zip(slices, pad, shape))


def _robust(values):
    median = float(np.median(values))
    return median, 1.4826 * float(np.median(np.abs(values - median)))


def _support(local_mask, signal, probability, spacing, policy, *, ct=False, bone=None, ball=None, lumen_voxels=None):
    """Supported non-mask voxels in one window, with the statistics behind them.

    A voxel is supported by acquired signal or by model probability, and then
    vetoed when its signal is brighter than the contrast ceiling, above the CT
    ceiling or below the CT air floor in HU, or inside a supplied bone mask:
    probability cannot carry a bridge through bone or air. The lumen level is the median signal of ``lumen_voxels``
    and the background the median of the 1 to 3 mm shell outside the mask, both
    inside ``ball`` (a fixed physical neighbourhood the window always contains).
    """
    ball = np.ones(local_mask.shape, bool) if ball is None else ball
    lumen_voxels = local_mask if lumen_voxels is None else lumen_voxels
    supported = np.zeros(local_mask.shape, bool)
    veto = np.zeros(local_mask.shape, bool) if bone is None else np.asarray(bone, bool).copy()
    evidence = {}
    if signal is not None:
        values = np.asarray(signal, np.float32)
        if ct:
            veto |= (values > policy.ct_ceiling_hu) | (values < policy.ct_air_hu)
        outside = ndi.distance_transform_edt(~local_mask, sampling=spacing)
        shell = (outside > policy.shell_inner_mm) & (outside <= policy.shell_outer_mm) & ball
        del outside
        lumen_voxels = lumen_voxels & ball
        if shell.sum() >= 20 and lumen_voxels.any():
            lumen = float(np.median(values[lumen_voxels]))
            background, noise = _robust(values[shell])
            if lumen - background > max(policy.min_contrast_noise * noise, 1e-6):
                fraction = (values - background) / (lumen - background)
                supported |= fraction >= policy.contrast_floor
                veto |= fraction > policy.contrast_ceiling
                evidence.update(lumen=lumen, background=background, noise=noise, fraction=fraction)
            else:
                evidence['signal'] = 'no local lumen contrast'
        else:
            evidence['signal'] = 'no background shell'
    if probability is not None:
        p = np.asarray(probability, np.float32)
        supported |= p >= policy.probability_floor
        evidence['probability'] = p
    supported &= ~local_mask & ~veto
    return supported, evidence


def _radius_from_sphere_volume(volume, radius):
    """Cylinder radius whose intersection with a centred sphere has this volume.

    A sphere of radius R centred on the axis of a cylinder of radius a < R holds
    4/3 pi (R^3 - (R^2 - a^2)^(3/2)) of it. Inverting the local lumen volume
    gives a cross-section-based calibre that does not depend on how the
    centreline sits on the voxel lattice.
    """
    rest = np.clip(radius ** 3 - 3 * np.asarray(volume, float) / (4 * np.pi), 0, None)
    return np.sqrt(np.clip(radius ** 2 - np.cbrt(rest) ** 2, 0, None))


def _sphere_calibre(points, queries, spacing, voxel_mm3, policy):
    """Calibre (mm) at query voxels from the mask volume within calibre_sphere_mm."""
    if not len(queries):
        return np.empty(0)
    counts = cKDTree(_scaled(points, spacing)).query_ball_point(
        _scaled(queries, spacing), policy.calibre_sphere_mm, return_length=True)
    return 2 * _radius_from_sphere_volume(np.asarray(counts) * voxel_mm3, policy.calibre_sphere_mm)


def _inferior_face(affine, shape):
    """Voxel axis and index of the field-of-view face lowest in patient z (RAS+)."""
    if affine is None:
        return None
    axis = int(np.argmax(np.abs(affine[2, :3])))
    return axis, (0 if affine[2, axis] > 0 else shape[axis] - 1)


def _corridor_distance(points, corridors):
    """Distance in millimetres from points to the nearest capsule surface."""
    best = np.full(len(points), np.inf)
    for corridor in corridors:
        start = np.asarray(corridor['entry'], float)
        end = np.asarray(corridor['target'], float)
        radius = float(corridor['radiusMm'])
        axis = end - start
        length2 = float(axis @ axis)
        t = np.zeros(len(points)) if length2 < 1e-12 else np.clip((points - start) @ axis / length2, 0, 1)
        nearest = start + t[:, None] * axis
        best = np.minimum(best, np.linalg.norm(points - nearest, axis=1) - radius)
    return best


def _checked_corridors(corridors):
    checked = []
    for corridor in corridors or ():
        try:
            entry = np.asarray(corridor['entry'], float)
            target = np.asarray(corridor['target'], float)
            radius = float(corridor['radiusMm'])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError('Each corridor needs entry, target and radiusMm') from error
        if entry.shape != (3,) or target.shape != (3,) or not np.isfinite([*entry, *target, radius]).all() or radius < 0:
            raise ValueError('Each corridor needs finite entry, target and a non-negative radius')
        checked.append({'entry': entry.tolist(), 'target': target.tolist(), 'radiusMm': radius})
    return checked


def _named_checked(named, mask, scheme, modality):
    if named is None:
        if scheme is not None:
            raise ValueError('A label scheme was given without named labels')
        return None, None
    if scheme is None:
        raise ValueError('Named labels need their label scheme')
    if isinstance(scheme, str):
        if scheme not in SCHEMES:
            raise ValueError(f'Unknown label scheme {scheme!r}')
        scheme = SCHEMES[scheme]
    if modality is None:
        raise ValueError('Named labels need the image modality')
    if image_modality(modality) != scheme.modality:
        raise ValueError(f'{scheme.name} labels cannot be applied to a {image_modality(modality).upper()} image')
    named = _volume(named, 'Named labels', mask.shape)
    if not np.issubdtype(named.dtype, np.integer):
        raise ValueError('Named labels must be integers')
    present = np.unique(named[mask])
    unknown = set(present.tolist()) - {0} - set(scheme.names)
    if unknown:
        raise ValueError(f'Named labels outside {scheme.name}: {sorted(unknown)}')
    return named, scheme


def root_vessels(mask, *, affine=None, spacing=None, named=None, scheme=None, modality=None,
                 layer_kind='vessels', signal=None, probability=None, tumour=None, corridors=(),
                 bone=None, root_mask=None, policy=RootingPolicy(), check=lambda: None):
    """Partition a vessel mask into rooted display, review and level-of-detail outputs.

    ``named`` is an integer label map on the mask grid (for example the Phase 2
    ``named_labels`` or a supplied annotation) interpreted by ``scheme``. CT
    schemes are refused on MR images and the reverse. ``signal`` is the acquired
    scan and ``probability`` a model vessel probability on the same grid.
    ``tumour`` and ``bone`` are masks on the same grid; ``corridors`` are
    capsules ``{'entry', 'target', 'radiusMm'}`` in the millimetre frame of
    ``affine``. ``modality`` also enables the CT ceiling in HU. ``root_mask``
    supplies the roots directly (for example the shown tree a model addition
    may extend); its voxels are exempt from the level of detail.

    Memory: one component-label grid and a few boolean grids of the cropped
    mask; roots and bridge voxels are kept as index lists, and the support
    fields are read only around fragments that lie within reach of the tree.
    """
    mask = np.asarray(_volume(mask, 'The vessel mask'), dtype=bool)
    affine, spacing = _frame(affine, spacing)
    if layer_kind not in LAYER_CLASSES:
        raise ValueError(f'Unknown vessel layer kind {layer_kind!r}')
    named, scheme = _named_checked(named, mask, scheme, modality)
    signal = None if signal is None else _volume(signal, 'The acquired signal', mask.shape)
    probability = None if probability is None else _volume(probability, 'The vessel probability', mask.shape)
    tumour = None if tumour is None else _volume(tumour, 'The tumour mask', mask.shape)
    bone = None if bone is None else _volume(bone, 'The bone mask', mask.shape)
    root_mask = None if root_mask is None else _volume(root_mask, 'The root mask', mask.shape)
    ct = modality is not None and image_modality(modality) == 'ct'
    corridors = _checked_corridors(corridors)
    if corridors and affine is None:
        raise ValueError('Corridors need the voxel-to-millimetre affine')
    voxel_mm3 = float(np.prod(spacing))
    s_min = float(spacing.min())
    layer_class = LAYER_CLASSES[layer_kind]
    receipt = {'version': VERSION, 'policy': asdict(policy),
               'grid': {'shape': list(mask.shape), 'spacingMm': spacing.round(6).tolist()}}
    # Air on CT leaves the mask here; `source` keeps the input for the partition.
    source, air, air_count = mask, None, 0
    if ct and signal is not None and mask.any():
        low, high = _crop_bounds(mask, (0, 0, 0))
        box = tuple(slice(int(a), int(b)) for a, b in zip(low, high))
        # Compared as float32, like the bridge and fill vetoes: one voxel set whatever the scan's dtype.
        in_air = mask[box] & (np.asarray(_load_crop(signal, box), np.float32) < policy.ct_air_hu)
        air_count = int(in_air.sum())
        if air_count:
            air = np.zeros(mask.shape, bool)
            air[box] = in_air
            mask = mask & ~air
        del in_air
        check()

    def with_air(result):
        """Add the air voxels to the review output and say so in the receipt."""
        if ct and signal is not None:
            receipt['air'] = {
                'floorHu': policy.ct_air_hu, 'voxels': air_count, 'volumeMm3': round(air_count * voxel_mm3, 4),
                'rule': 'Input voxels below the floor in the acquired CT are in review: they are not roots and '
                        'join or bridge nothing. They are counted in the review voxels and volume, not in its '
                        'components.'}
        if air is None:
            return result
        result.review = result.review | air
        result.air = air
        if not mask.any():
            reason = 'Every voxel of this vessel mask lies in air on the acquired CT.'
            receipt['roots'] = {**receipt['roots'], 'reason': reason}
            receipt['review'] = {**receipt['review'], 'reason': reason}
        receipt['input'] = {**receipt['input'], 'voxels': receipt['input']['voxels'] + air_count,
                            'volumeMm3': round(float((receipt['input']['voxels'] + air_count) * voxel_mm3), 4)}
        receipt['review'] = {**receipt['review'], 'voxels': receipt['review']['voxels'] + air_count,
                             'volumeMm3': round(float((receipt['review']['voxels'] + air_count) * voxel_mm3), 4)}
        if named is not None:
            receipt['namedLabels'] = _named_air(receipt['namedLabels'], np.asarray(named)[air], scheme)
        receipt['partitionExact'] = bool(
            receipt['partitionExact'] and not (result.bridge & air).any()
            and np.array_equal(result.shown | result.review | result.lod_hidden, source))
        if not receipt['partitionExact']:
            raise AssertionError('Rooted display violated its partition invariant')
        return result

    if not mask.any():
        empty = np.zeros(mask.shape, bool)
        receipt.update(_empty_receipt(scheme))
        return with_air(RootedVessels(empty, empty.copy(), empty.copy(), empty.copy(), np.zeros(mask.shape, np.uint8), receipt))

    pad = np.ceil((policy.max_gap_mm + policy.shell_outer_mm + 1) / spacing).astype(int)
    lo, hi = _crop_bounds(mask, pad)
    crop = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
    whole = tuple(hi - lo) == mask.shape
    m = mask if whole else np.ascontiguousarray(mask[crop])
    nm = None if named is None else (named if whole else np.ascontiguousarray(named[crop]))
    check()

    labels, count = _label(m)
    flat_labels = labels.ravel()
    sizes = _counts(labels, count + 1)
    sizes[0] = 0
    objects = ndi.find_objects(labels)
    check()

    # Roots, kept as flat indices into the crop.
    root_info = {'source': 'largest-component', 'scheme': scheme.name if scheme else None}
    seed = np.empty(0, np.int64)
    if root_mask is not None:
        seed = np.flatnonzero((m & np.asarray(root_mask[crop], bool)).ravel())
        root_info['source'] = 'supplied root mask'
        if not len(seed):
            root_info['reason'] = 'The supplied roots do not touch this mask; the fallback roots apply.'
    elif nm is not None:
        seed = np.flatnonzero((m & np.isin(nm, scheme.roots)).ravel())
        present_roots = sorted(set(np.unique(nm.ravel()[seed]).tolist()) - {0})
        root_info['labels'] = [{'label': v, 'name': scheme.names[v]} for v in present_roots]
        if len(seed):
            root_info['source'] = 'named'
        else:
            root_info['reason'] = ('No root label lies inside this vessel mask; the largest component, and '
                                   'components entering through the inferior face, are the tree.')
    else:
        root_info['reason'] = ('No named labels; the largest component, and components entering through the '
                               'inferior face of the field of view, are the tree.')
    if not len(seed):
        seed, root_info['acceptedRoots'], refused_entries = _fallback_roots(
            labels, sizes, objects, lo, mask.shape, affine, spacing, voxel_mm3, policy)
        root_info['refusedEntries'] = refused_entries[:200]
        root_info['refusedEntryCount'] = len(refused_entries)
    root_ids = np.unique(flat_labels[seed])
    root_ids = root_ids[root_ids > 0]
    attached_lut = np.zeros(count + 1, bool)
    attached_lut[root_ids] = True
    attached_count = int(sizes[root_ids].sum())

    def classes_in(window, local_mask, local_tree):
        """Class of each voxel in a window: bridges keep theirs, then named labels, then the layer kind."""
        classes = np.full(local_mask.shape, layer_class, np.uint8)
        if nm is not None:
            local_named = nm[window]
            classes[local_named > 0] = ARTERY
            if scheme.veins:
                classes[np.isin(local_named, scheme.veins)] = VEIN
        added = np.argwhere(local_tree & ~local_mask)
        if len(added):
            flat = np.ravel_multi_index(tuple((added + [w.start for w in window]).T), m.shape)
            classes[tuple(added.T)] = [bridge_class.get(int(i), layer_class) for i in flat]
        return classes

    # Bridging.
    tree = _lookup(attached_lut, labels)
    bridged_lut = np.zeros(count + 1, bool)
    bridge_class = {}
    bridges = []
    basis = [name for name, value in (('signal', signal), ('probability', probability)) if value is not None]
    fragment_ids = np.flatnonzero(~attached_lut & (sizes > 0))
    small = sizes[fragment_ids] * voxel_mm3 < policy.min_fragment_mm3
    candidates = fragment_ids[~small]
    reasons = {int(f): 'too-small' for f in fragment_ids[small]}
    if len(candidates) > policy.max_fragments:
        order = np.argsort(-sizes[candidates], kind='stable')
        reasons.update({int(f): 'over-budget' for f in candidates[order[policy.max_fragments:]]})
        candidates = candidates[order[:policy.max_fragments]]
    reasons.update({int(f): 'too-far' if basis else 'no-support-field' for f in candidates})
    window_pad = np.ceil((policy.max_gap_mm + policy.shell_outer_mm) / spacing).astype(int)
    stats_pad = np.ceil((policy.statistics_radius_mm + policy.shell_outer_mm) / spacing).astype(int)
    reach = np.ceil(policy.calibre_reach_mm / spacing).astype(int)
    fields = _Fields(signal, probability, lo)
    rounds = 0
    pending = set(candidates.tolist()) if basis else set()
    while pending and rounds < policy.bridge_rounds:
        rounds += 1
        ids = np.array(sorted(pending))
        lut = np.zeros(count + 1, bool)
        lut[ids] = True
        members = np.flatnonzero(_lookup(lut, labels))
        distance, partner = _distance_to(np.column_stack(np.unravel_index(members, m.shape)), tree, spacing,
                                         policy.max_gap_mm + 1e-6, nearest=True)
        # Windows cover only the part of each fragment within reach of the tree.
        within = distance <= policy.max_gap_mm
        members, distance, partner = members[within], distance[within], partner[within]
        del lut, within
        owners = flat_labels[members]
        coords = np.column_stack(np.unravel_index(members, m.shape))
        low = np.full((count + 1, 3), np.iinfo(np.int64).max)
        high = np.full((count + 1, 3), -1)
        np.minimum.at(low, owners, coords)
        np.maximum.at(high, owners, coords)
        # The closest fragment-to-tree voxel pair centres the statistics neighbourhood.
        order = np.lexsort((members, distance, owners))
        first = order[np.r_[True, owners[order][1:] != owners[order][:-1]]] if len(order) else order
        centres = {int(owners[i]): (coords[i] + partner[i]) / 2 for i in first}
        near = [int(f) for f in np.unique(owners)]
        del members, owners, coords, distance, partner
        if not near:
            break
        windows = {f: _search_window(low[f], high[f], window_pad, centres[f], stats_pad, m.shape) for f in near}
        del low, high
        fields.cover(list(windows.values()))
        joined = []
        for fragment in near:
            check()
            window = windows[fragment]
            origin = np.array([w.start for w in window])
            local_mask = m[window]
            local_tree = tree[window]
            local_fragment = labels[window] == fragment
            local_signal, local_probability = fields.window(window)
            grid = np.ogrid[tuple(slice(0, w.stop - w.start) for w in window)]
            centre = centres[fragment] - origin
            ball = sum(((g - c) * s) ** 2 for g, c, s in zip(grid, centre, spacing)) <= policy.statistics_radius_mm ** 2
            supported, evidence = _support(local_mask, local_signal, local_probability, spacing, policy,
                                           ct=ct, bone=None if bone is None else bone[tuple(
                                               slice(w.start + a, w.stop + a) for w, a in zip(window, lo))],
                                           ball=ball, lumen_voxels=local_mask & (local_tree | local_fragment))
            del ball, grid
            # A fragment joins only its own class; unclassified fragments may join either.
            classes = classes_in(window, local_mask, local_tree)
            own = set(np.unique(classes[local_fragment]).tolist()) - {UNCLASSIFIED}
            allowed = local_tree & (np.isin(classes, list(own)) if own else True)
            del classes
            # One reason whatever the window holds, so the receipt does not depend on it.
            if not supported.any() or not local_tree.any():
                reasons[fragment] = 'no-supported-path'
                continue
            if not allowed.any():
                reasons[fragment] = 'no-same-class-tree'
                continue
            costs = np.where(supported | local_fragment | allowed, np.float32(1), np.float32(np.inf))
            costs[local_tree & ~allowed] = np.inf
            # Face-connected paths: the drawn surface joins voxels only across faces.
            search = MCP_Geometric(costs, sampling=tuple(float(s) for s in spacing), fully_connected=False)
            ends = np.argwhere(allowed)
            cumulative, _ = search.find_costs(np.argwhere(local_fragment), ends, find_all_ends=False)
            del costs
            reached = cumulative[allowed]
            if not np.isfinite(reached).any():
                reasons[fragment] = 'no-supported-path'
                continue
            end = tuple(ends[int(np.argmin(reached))])
            length = float(cumulative[end])
            del cumulative, reached, ends
            if length > policy.max_gap_mm + 1e-9:
                reasons[fragment] = 'supported-path-too-long'
                continue
            path = np.array(search.traceback(end))
            del search
            gap_path = np.zeros(local_mask.shape, bool)
            gap_path[tuple(path.T)] = True
            gap_path &= ~local_mask & ~local_tree & ~local_fragment
            # Each end's radius is the deepest lumen within calibre reach of it.
            depth = ndi.distance_transform_edt(local_mask, sampling=spacing)
            radii = []
            for point in (path[0], np.array(end)):
                near_box = tuple(slice(max(0, int(i) - r), int(i) + r + 1) for i, r in zip(point, reach))
                radii.append(float(depth[near_box].max()))
            del depth
            radius = float(np.clip(min(radii) - 0.5 * s_min, 0, policy.max_bridge_radius_mm))
            tube = gap_path
            if radius > 0 and gap_path.any():
                tube = ndi.distance_transform_edt(~gap_path, sampling=spacing) <= radius
                tube &= (supported | gap_path) & ~local_mask & ~local_tree
                pieces, _ = _label(tube)
                keep = np.unique(pieces[gap_path])
                tube = np.isin(pieces, keep[keep > 0])
                del pieces
            parent = int(classes_in(window, local_mask, local_tree)[end])
            added = np.ravel_multi_index(tuple((np.argwhere(tube) + origin).T), m.shape)
            bridge_class.update({int(i): parent for i in added})
            tree[window] |= local_fragment | tube
            bridged_lut[fragment] = True
            reasons.pop(fragment, None)
            joined.append(fragment)
            record = {'fragment': int(fragment), 'voxels': int(sizes[fragment]),
                      'volumeMm3': round(float(sizes[fragment] * voxel_mm3), 4),
                      'gapMm': round(length, 4), 'bridgeVoxels': int(len(added)),
                      'radiusMm': round(radius, 3), 'round': rounds, 'class': CLASS_NAMES[parent],
                      'windowShape': list(local_mask.shape),
                      'statisticsCentreIjk': [round(float(v), 1) for v in lo + centres[fragment]],
                      'pathStartIjk': [int(v) for v in lo + origin + path[0]]}
            if 'fraction' in evidence and gap_path.any():
                record['minimumContrastFraction'] = round(float(evidence['fraction'][gap_path].min()), 4)
                record['lumen'] = round(evidence['lumen'], 4)
                record['background'] = round(evidence['background'], 4)
            if 'probability' in evidence and gap_path.any():
                record['minimumProbability'] = round(float(evidence['probability'][gap_path].min()), 4)
            evidence.clear()
            if nm is not None:
                inside = np.unique(nm[window][local_fragment])
                record['namedLabels'] = [scheme.names[v] for v in inside.tolist() if v]
            bridges.append(record)
        pending -= set(joined)
        if not joined:
            break
    refused = {}
    for reason in reasons.values():
        refused[reason] = refused.get(reason, 0) + 1
    support_box = fields.box()
    # Loop locals hold views into the support fields; drop them so the fields go.
    local_signal = local_probability = supported = evidence = local_mask = local_tree = local_fragment = None
    del tree, fields, local_signal, local_probability, supported, evidence, local_mask, local_tree, local_fragment
    check()

    # Everything that needs component labels is taken before the level of detail.
    connected = _lookup(attached_lut | bridged_lut, labels)
    bridged_voxels = _lookup(bridged_lut, labels) if nm is not None else None
    bridged_source = int(sizes[bridged_lut].sum())
    named_components = _named_components(nm, scheme, m, labels, attached_lut, bridged_lut)
    del labels, flat_labels, objects
    bridge_index = np.fromiter(bridge_class, np.int64, len(bridge_class))

    # Level of detail on the rooted display (source voxels of it are display & m).
    display = connected
    del connected
    display.ravel()[bridge_index] = True
    exempt = np.zeros(m.shape, bool)
    exempt.ravel()[seed] = True
    if nm is not None:
        exempt |= display & (nm > 0)
    lod, lod_info = _level_of_detail(display, exempt, seed, spacing, s_min, voxel_mm3, policy,
                                     tumour=None if tumour is None else (tumour, lo, hi),
                                     corridors=corridors, affine=affine, origin=lo, check=check)
    del exempt
    check()

    shown_c = display & m & ~lod
    review_c = m & ~display
    lod_c = m & lod
    kept_bridges = bridge_index[~lod.ravel()[bridge_index]]
    pruned_bridges = int(len(bridge_index) - len(kept_bridges))
    del lod, display
    shown_display = shown_c.copy()
    shown_display.ravel()[kept_bridges] = True
    # The drawn surface joins voxels only across faces. Corner- or edge-only
    # contacts get a declared display-only connectivity fill, or the far side
    # goes to review; source masks are never modified.
    fill, dropped, connectivity = _face_connect(
        shown_display, seed, m, lo, spacing, policy, signal=signal, probability=probability, bone=bone, ct=ct, check=check)
    if dropped.any():
        moved = dropped & shown_c
        shown_c &= ~dropped
        review_c |= moved
        kept_bridges = kept_bridges[~dropped.ravel()[kept_bridges]]
        connectivity['droppedSourceVoxels'] = int(moved.sum())
        del moved
    del dropped
    display_labels, display_count = _label(shown_display, FACE)
    floating = int(display_count - len(set(np.unique(display_labels.ravel()[seed]).tolist()) - {0}))
    del display_labels
    classes_c = np.zeros(m.shape, np.uint8)
    classes_c[shown_c] = layer_class
    if nm is not None:
        named_shown = shown_c & (nm > 0)
        classes_c[named_shown] = ARTERY
        if scheme.veins:
            classes_c[named_shown & np.isin(nm, scheme.veins)] = VEIN
        del named_shown
    classes_c.ravel()[kept_bridges] = [bridge_class[int(i)] for i in kept_bridges]
    # A fill voxel takes the class of the face neighbour on the side it reconnects.
    for index in fill:
        point = np.unravel_index(int(index), m.shape)
        near = [classes_c[tuple(np.add(point, offset))] for offset in _FACE_OFFSETS
                if all(0 <= v < n for v, n in zip(np.add(point, offset), m.shape))]
        classes_c.ravel()[int(index)] = max((c for c in near if c), default=layer_class)
    kept_bridges = np.concatenate([kept_bridges, np.asarray(fill, np.int64)])
    del shown_display

    def full(array, dtype=bool):
        if whole:
            return array
        out = np.zeros(mask.shape, dtype)
        out[crop] = array
        return out

    bridge_c = np.zeros(m.shape, bool)
    bridge_c.ravel()[kept_bridges] = True
    named_rows = _named_report(nm, scheme, m, named_components, bridged_voxels, shown_c, review_c, lod_c)
    if scheme is not None and scheme.modality == 'mr':
        # External carotid branches are not roots on MR; list those left in review.
        root_info['externalCarotidInReview'] = [row['name'] for row in named_rows
                                                if 35 <= row['label'] <= 42 and row['review']]
    counts = {name: int(value) for name, value in (('shown', shown_c.sum()), ('review', review_c.sum()), ('lod', lod_c.sum()))}
    review_components = int(len(fragment_ids) - int(bridged_lut.sum()))
    result = RootedVessels(full(shown_c), full(bridge_c), full(review_c), full(lod_c), full(classes_c, np.uint8), receipt)
    partition = bool(counts['shown'] + counts['review'] + counts['lod'] == int(m.sum())
                     and not (result.shown & result.review).any() and not (result.shown & result.lod_hidden).any()
                     and not (result.review & result.lod_hidden).any() and not (result.bridge & mask).any()
                     and np.array_equal(result.shown | result.review | result.lod_hidden, mask))
    receipt.update({
        'roots': root_info,
        'input': {'voxels': int(m.sum()), 'volumeMm3': round(float(m.sum() * voxel_mm3), 4), 'components': int(count)},
        'attached': {'components': int(len(root_ids)), 'voxels': attached_count,
                     'volumeMm3': round(float(attached_count * voxel_mm3), 4)},
        'bridged': {'components': int(bridged_lut.sum()), 'fragmentVoxels': bridged_source,
                    'bridgeVoxels': int(len(kept_bridges)), 'bridgeVolumeMm3': round(float(len(kept_bridges) * voxel_mm3), 4),
                    'rounds': rounds, 'support': ' or '.join(basis) if basis else 'none',
                    'supportReadBox': support_box, 'bridges': bridges[:256], 'listed': min(len(bridges), 256)},
        'review': {'components': review_components, 'voxels': counts['review'],
                   'volumeMm3': round(float(counts['review'] * voxel_mm3), 4),
                   'refused': dict(sorted(refused.items())),
                   'reason': ('Components that reach no root and have no supported gap of at most '
                              f'{policy.max_gap_mm:g} mm to the tree.' if basis else
                              'No acquired signal or model probability was available to support a bridge.')},
        'lod': {**lod_info, 'voxels': counts['lod'], 'volumeMm3': round(float(counts['lod'] * voxel_mm3), 4),
                'bridgeVoxelsPruned': pruned_bridges},
        'display': {'voxels': counts['shown'] + int(len(kept_bridges)), 'sourceVoxels': counts['shown'],
                    'components': int(display_count), 'floatingComponents': floating, 'connectivity': 'face (6-connected)'},
        'connectivity': connectivity,
        'classes': _class_report(classes_c, nm is not None, layer_kind),
        'namedLabels': named_rows,
        'partitionExact': partition,
    })
    if not partition or floating:
        raise AssertionError('Rooted display violated its partition or connectivity invariant')
    return with_air(result)


def _deepest(labels, objects, component, spacing):
    """Flat crop index of the voxel deepest inside one component."""
    box = tuple(slice(max(0, s.start - 1), min(n, s.stop + 1)) for s, n in zip(objects[component - 1], labels.shape))
    inside = labels[box] == component
    local = np.unravel_index(int(np.argmax(_depth(inside, spacing))), inside.shape)
    return int(np.ravel_multi_index(tuple(int(i + s.start) for i, s in zip(local, box)), labels.shape))


def _face_connect(display, seed, mask, origin, spacing, policy, *, signal, probability, bone, ct, check):
    """Make every shown component face-connected to a root.

    ``display`` (crop grid) is extended in place by fill voxels: for each
    component that is not face-connected to a root but touches a rooted one by
    an edge or corner, the minimal face-connecting path (one voxel for an edge,
    two for a corner) with the highest signal or probability is added, vetoed
    by the bridge rules (supplied bone, the CT ceiling, and twice the local
    contrast). Components that cannot be joined are returned to be dropped from
    the display. Returns (fill flat indices, dropped mask, receipt).
    """
    shape = display.shape
    fill, dropped = [], np.zeros(shape, bool)
    receipt = {'fillVoxels': 0, 'joins': 0, 'droppedComponents': 0, 'vetoedCandidates': 0, 'rounds': 0,
               'meaning': 'display-only voxels added where shown parts touched only by an edge or corner'}
    stats = {}

    def value(source, point):
        if source is None:
            return None
        return float(np.asarray(source[tuple(int(v) for v in np.add(point, origin))], np.float32))

    def vetoed(point, anchor):
        full = tuple(int(v) for v in np.add(point, origin))
        if bone is not None and bone[full]:
            return True
        level = value(signal, point)
        if level is None:
            return False
        if ct and (level > policy.ct_ceiling_hu or level < policy.ct_air_hu):
            return True
        key = tuple(anchor)
        if key not in stats:
            pad = np.ceil((policy.statistics_radius_mm + policy.shell_outer_mm) / spacing).astype(int)
            box = tuple(slice(max(0, int(a) - p), min(n, int(a) + p + 1)) for a, p, n in zip(anchor, pad, shape))
            grid = np.ogrid[tuple(slice(0, b.stop - b.start) for b in box)]
            ball = sum(((g - (a - b.start)) * s) ** 2 for g, a, b, s in zip(grid, anchor, box, spacing)) <= policy.statistics_radius_mm ** 2
            local = np.asarray(signal[tuple(slice(b.start + o, b.stop + o) for b, o in zip(box, origin))], np.float32)
            _, evidence = _support(display[box] | mask[box], local, None, spacing, policy, ball=ball)
            stats[key] = (evidence['lumen'], evidence['background']) if 'lumen' in evidence else None
        if stats[key] is None:
            return False
        lumen, background = stats[key]
        return (level - background) / (lumen - background) > policy.contrast_ceiling

    def score(point):
        found = [v for v in (value(probability, point), value(signal, point)) if v is not None]
        return found[0] if found else 0.0

    for _ in range(64):
        check()
        receipt['rounds'] += 1
        labels, count = _label(display, FACE)
        rooted = np.zeros(count + 1, bool)
        rooted[np.unique(labels.ravel()[seed])] = True
        rooted[0] = False
        loose = ~rooted
        loose[0] = False
        if not loose.any():
            break
        coords = np.argwhere(_lookup(loose, labels))
        candidates = {}
        for offset in _JOIN_OFFSETS:
            target = coords + offset
            inside = ((target >= 0) & (target < shape)).all(axis=1)
            near = np.zeros(len(coords), bool)
            near[inside] = rooted[labels[tuple(target[inside].T)]]
            for a in coords[near][:64]:
                candidates.setdefault(int(labels[tuple(a)]), []).append((a, offset))
        changed = False
        waiting = []
        for component in np.flatnonzero(loose).tolist():
            if component not in candidates:
                waiting.append(component)  # It may touch a component joined this round.
                continue
            best = None
            for a, offset in candidates.get(component, [])[:64]:
                axes = np.flatnonzero(offset)
                steps = [np.eye(3, dtype=int)[axis] * offset[axis] for axis in axes]
                orders = [[steps[0]], [steps[1]]] if len(steps) == 2 else \
                    [[steps[i], steps[j]] for i in range(3) for j in range(3) if i != j]
                for order in orders:
                    path, point = [], np.array(a)
                    for step in order:
                        point = point + step
                        path.append(point.copy())
                    if any(display[tuple(v)] for v in path):
                        continue  # Already face-connected through another voxel.
                    if any(mask[tuple(v)] for v in path) or any(vetoed(v, a) for v in path):
                        receipt['vetoedCandidates'] += 1
                        continue
                    quality = min(score(v) for v in path)
                    if best is None or quality > best[0]:
                        best = (quality, path)
            if best is None:
                part = _lookup(np.arange(count + 1) == component, labels)
                dropped |= part
                display &= ~part
                receipt['droppedComponents'] += 1
            else:
                for v in best[1]:
                    display[tuple(v)] = True
                    fill.append(int(np.ravel_multi_index(tuple(v), shape)))
                receipt['joins'] += 1
            changed = True
        if not changed:
            # Nothing touches a rooted part any more: what is left cannot be joined.
            for component in waiting:
                part = _lookup(np.arange(count + 1) == component, labels)
                dropped |= part
                display &= ~part
                receipt['droppedComponents'] += 1
            break
    receipt['fillVoxels'] = len(fill)
    return fill, dropped, receipt


def _fallback_roots(labels, sizes, objects, lo, full_shape, affine, spacing, voxel_mm3, policy):
    """Roots without named labels: the largest component, unconditionally, plus
    components that enter the head through the inferior face of the field of view
    with calibre of at least entry_min_calibre_mm and extent of at least
    entry_min_extent_mm (the other carotid tree or a separate posterior
    circulation). Returns seeds, accepted roots and refused face components."""
    largest = int(sizes.argmax())
    seeds = [_deepest(labels, objects, largest, spacing)]
    seed_ijk = lambda flat: [int(v) for v in np.add(np.unravel_index(flat, labels.shape), lo)]
    accepted = [{'component': largest, 'voxels': int(sizes[largest]), 'reason': 'largest component',
                 'seedIjk': seed_ijk(seeds[0])}]
    refused = []
    face = _inferior_face(affine, full_shape)
    if face is None:
        accepted[0]['note'] = 'No patient frame; the inferior face is unknown.'
        return np.array(seeds, np.int64), accepted, refused
    axis, index = face
    local = index - int(lo[axis])
    if not 0 <= local < labels.shape[axis]:
        return np.array(seeds, np.int64), accepted, refused
    touching = np.unique(np.take(labels, local, axis=axis))
    for component in touching[(touching > 0) & (touching != largest)].tolist():
        box = objects[component - 1]
        extent = float(np.linalg.norm(np.array([s.stop - s.start for s in box]) * spacing))
        points = np.argwhere(labels[box] == component) + [s.start for s in box]
        # Widest sphere calibre over an even sample of the component (a deepest-voxel
        # estimate would sit on the field-of-view face, half outside the sphere).
        sample = points[::max(1, len(points) // 512)]
        calibres = _sphere_calibre(points, sample, spacing, voxel_mm3, policy)
        widest = int(np.argmax(calibres))
        calibre = float(calibres[widest]) + policy.calibre_tolerance_mm
        deepest = int(np.ravel_multi_index(tuple(sample[widest]), labels.shape))
        row = {'component': component, 'voxels': int(sizes[component]), 'calibreMm': round(calibre, 3),
               'extentMm': round(extent, 2)}
        if extent < policy.entry_min_extent_mm or calibre < policy.entry_min_calibre_mm:
            refused.append({**row, 'reason': 'too short' if extent < policy.entry_min_extent_mm else 'too narrow'})
            continue
        seeds.append(deepest)
        accepted.append({**row, 'reason': 'enters through the inferior face', 'seedIjk': seed_ijk(deepest)})
    return np.array(seeds, np.int64), accepted, refused


class _Fields:
    """Support fields read lazily over the box that covers the requested windows."""

    def __init__(self, signal, probability, origin):
        self.sources = (signal, probability)
        self.origin = np.asarray(origin)
        self.lo = self.hi = None
        self.values = (None, None)

    def cover(self, windows):
        lo = np.min([[w.start for w in window] for window in windows], axis=0)
        hi = np.max([[w.stop for w in window] for window in windows], axis=0)
        if self.lo is not None and (lo >= self.lo).all() and (hi <= self.hi).all():
            return
        if self.lo is not None:
            lo, hi = np.minimum(lo, self.lo), np.maximum(hi, self.hi)
        self.values = (None, None)
        box = tuple(slice(int(a), int(b)) for a, b in zip(lo + self.origin, hi + self.origin))
        self.values = tuple(None if source is None else _load_crop(source, box) for source in self.sources)
        self.lo, self.hi = lo, hi

    def window(self, window):
        local = tuple(slice(w.start - a, w.stop - a) for w, a in zip(window, self.lo))
        return tuple(None if values is None else values[local] for values in self.values)

    def box(self):
        return None if self.lo is None else {'lo': (self.lo + self.origin).tolist(), 'hi': (self.hi + self.origin).tolist()}


def _level_of_detail(display, exempt, seed, spacing, s_min, voxel_mm3, policy, *, tumour, corridors, affine, origin, check):
    info = {'alwaysMm': policy.always_mm, 'localMm': policy.local_mm, 'localMarginMm': policy.local_margin_mm,
            'tumour': tumour is not None and bool(tumour[0].any()), 'corridors': len(corridors), 'reverted': False}
    hidden = np.zeros(display.shape, bool)
    if not display.any():
        return hidden, {**info, 'hiddenComponents': 0}
    # Calibre from the lumen volume inside a sphere (independent of how the
    # centreline sits on the lattice), taken as the widest within the sphere
    # radius so wall voxels and branch tips read their vessel's calibre. The
    # distance transform only screens out voxels that are certainly wide.
    reach = 2 * np.ceil(policy.calibre_reach_mm / spacing).astype(int) + 1
    widest = _depth(display, spacing)
    ndi.maximum_filter(widest, size=tuple(int(r) for r in reach), output=widest)
    check()
    points = np.argwhere(display & ~exempt)
    certain = 2 * widest[tuple(points.T)].astype(np.float32) * DEPTH_STEP_MM >= policy.always_mm + 2 * float(spacing.max())
    points = points[~certain]
    del certain
    field = widest
    field[...] = 0
    field[display] = 255
    if len(points):
        radius = _sphere_calibre(np.argwhere(display), points, spacing, voxel_mm3, policy) / 2
        field[tuple(points.T)] = np.clip(np.rint(radius / DEPTH_STEP_MM), 0, 254).astype(np.uint8)
        del radius
    sphere = 2 * np.ceil(policy.calibre_sphere_mm / spacing).astype(int) + 1
    ndi.maximum_filter(field, size=tuple(int(r) for r in sphere), output=field)
    calibre = 2 * field[tuple(points.T)].astype(np.float32) * DEPTH_STEP_MM + policy.calibre_tolerance_mm
    del field, widest
    below = calibre < policy.always_mm
    points, calibre = points[below], calibre[below]
    threshold = np.full(len(points), policy.always_mm, np.float32)
    local = np.zeros(len(points), bool)
    if len(points) and info['tumour']:
        mask, lo, hi = tumour
        margin = np.ceil(policy.local_margin_mm / spacing).astype(int)
        start = np.maximum(lo - margin, 0)
        stop = np.minimum(hi + margin, mask.shape)
        region = np.ascontiguousarray(mask[tuple(slice(a, b) for a, b in zip(start, stop))])
        if region.any():
            shifted = points + origin - start
            local |= region[tuple(shifted.T)]
            local |= _distance_to(shifted, region, spacing, policy.local_margin_mm + 1e-6) <= policy.local_margin_mm
    if len(points) and corridors:
        world = (points + origin) @ affine[:3, :3].T + affine[:3, 3]
        local |= _corridor_distance(world, corridors) <= policy.local_margin_mm
    threshold[local] = policy.local_mm
    thin = np.zeros(display.shape, bool)
    thin[tuple(points[calibre < threshold].T)] = True
    info['localTierShownVoxels'] = int((local & (calibre >= policy.local_mm)).sum())
    info['belowAlwaysVoxels'] = int(len(points))
    thick = display & ~thin
    if not thick.any():
        return hidden, {**info, 'hiddenComponents': 0, 'reason': 'No voxel reaches the always-shown calibre; level of detail not applied.'}
    # Isolated thick specks on a thin branch are calibre noise, not kept anatomy.
    labels, count = _label(thick)
    sizes = _counts(labels, count + 1)
    anchored = np.bincount(labels[exempt & thick], minlength=count + 1) > 0
    minimum = np.pi * (policy.always_mm / 2) ** 2 * policy.always_mm / voxel_mm3
    noise = (sizes < minimum) & ~anchored
    noise[0] = False
    if noise.any():
        speck = _lookup(noise, labels)
        thin |= speck
        thick &= ~speck
        del speck, labels
        labels, count = _label(thick)
    del thick
    thin_labels, thin_count = _label(thin)
    check()
    if thin_count:
        coords = np.argwhere(thin)
        owner = thin_labels[tuple(coords.T)]
        pairs = []
        for offset in _OFFSETS:
            q = coords + offset
            valid = ((q >= 0) & (q < display.shape)).all(axis=1)
            touched = labels[tuple(q[valid].T)]
            keep = touched > 0
            pairs.append(np.stack([owner[valid][keep], touched[keep]], axis=1))
        pairs = np.unique(np.concatenate(pairs), axis=0) if pairs else np.empty((0, 2), int)
        degree = np.bincount(pairs[:, 0], minlength=thin_count + 1) if len(pairs) else np.zeros(thin_count + 1, int)
        terminal = degree <= 1
        terminal[0] = False
        del labels, coords, owner, pairs
        hidden = _lookup(terminal, thin_labels)
        info['hiddenComponents'] = int(terminal.sum())
    else:
        info['hiddenComponents'] = 0
    del thin_labels, thin
    kept = display & ~hidden
    kept_labels, kept_count = _label(kept)
    del kept
    if kept_count != len(set(np.unique(kept_labels.ravel()[seed]).tolist()) - {0}):
        info.update(reverted=True, hiddenComponents=0,
                    reason='Level of detail would have detached a shown component; it was not applied.')
        return np.zeros(display.shape, bool), info
    return hidden, info


def _class_report(classes, named, layer_kind):
    counts = {name: int((classes == value).sum()) for value, name in CLASS_NAMES.items()}
    report = {**counts, 'source': 'named labels, then layer kind' if named else 'layer kind'}
    if counts['unclassified']:
        report['note'] = 'Unclassified vessels are drawn in the artery colour; artery or vein is not established.'
    return report


def _named_components(named, scheme, mask, labels, attached_lut, bridged_lut):
    """Per named label: unrooted source components holding it, and how many were bridged."""
    if named is None:
        return {}
    unrooted = mask & ~_lookup(attached_lut, labels)
    pairs = np.unique(np.stack([named[unrooted].astype(np.int64), labels[unrooted].astype(np.int64)]), axis=1)
    out = {}
    for value, component in pairs.T.tolist():
        if value:
            loose, bridged = out.get(value, (0, 0))
            out[value] = (loose + 1, bridged + int(bridged_lut[component]))
    return out


def _named_report(named, scheme, mask, components, bridged_voxels, shown, review, lod):
    if named is None:
        return []
    rows = []
    for value in sorted(set(np.unique(named[mask]).tolist()) - {0}):
        here = mask & (named == value)
        loose, bridged = components.get(value, (0, 0))
        counts = {'voxels': int(here.sum()), 'shown': int((here & shown).sum()),
                  'bridgedVoxels': int((here & bridged_voxels & shown).sum()),
                  'review': int((here & review).sum()), 'lodHidden': int((here & lod).sum())}
        status = ('review' if counts['review'] == counts['voxels'] else 'partly-review' if counts['review']
                  else 'lod' if counts['lodHidden'] else 'bridged' if bridged else 'shown')
        rows.append({'label': int(value), 'name': scheme.names[value],
                     'class': CLASS_NAMES[scheme.label_class(value)], **counts,
                     'unrootedComponents': loose, 'bridgedComponents': bridged,
                     'reviewComponents': loose - bridged, 'status': status})
    return rows


def _named_air(rows, labels, scheme):
    """Named rows with the voxels of each label that lie in air counted as review."""
    rows = {row['label']: dict(row) for row in rows}
    values, counts = np.unique(labels[labels > 0], return_counts=True)
    for value, count in zip(values.tolist(), counts.tolist()):
        row = rows.setdefault(value, {
            'label': int(value), 'name': scheme.names[value], 'class': CLASS_NAMES[scheme.label_class(value)],
            'voxels': 0, 'shown': 0, 'bridgedVoxels': 0, 'review': 0, 'lodHidden': 0,
            'unrootedComponents': 0, 'bridgedComponents': 0, 'reviewComponents': 0, 'status': 'review'})
        row.update(voxels=row['voxels'] + count, review=row['review'] + count, inAir=count)
        row['status'] = 'review' if row['review'] == row['voxels'] else 'partly-review'
    return [rows[value] for value in sorted(rows)]


def _empty_receipt(scheme):
    zero = {'components': 0, 'voxels': 0, 'volumeMm3': 0.0}
    return {'roots': {'source': 'none', 'scheme': scheme.name if scheme else None, 'reason': 'Empty vessel mask'},
            'input': zero, 'attached': zero,
            'bridged': {'components': 0, 'fragmentVoxels': 0, 'bridgeVoxels': 0, 'bridgeVolumeMm3': 0.0,
                        'rounds': 0, 'support': 'none', 'bridges': [], 'listed': 0},
            'review': {**zero, 'refused': {}, 'reason': 'Empty vessel mask'},
            'lod': {'voxels': 0, 'volumeMm3': 0.0, 'hiddenComponents': 0, 'bridgeVoxelsPruned': 0, 'reverted': False},
            'display': {'voxels': 0, 'sourceVoxels': 0, 'components': 0, 'floatingComponents': 0},
            'classes': {'artery': 0, 'vein': 0, 'unclassified': 0, 'source': 'layer kind'},
            'namedLabels': [], 'partitionExact': True}
