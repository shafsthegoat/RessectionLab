"""Conservative CT cranial-interior display support, with explicit unknowns.

Only enclosed axial skull rings seeded by aligned brain are supported. No
convex hull bridges a missing skull base, craniectomy, or truncated scan.
"""
from copy import deepcopy
import numpy as np
from scipy import ndimage as ndi
from .patient_anatomy import affine, world, resample, digest
from .anatomy_display import display_affine

VERSION = 'ct-enclosed-rings-v1'


def enclosed_rings(ct, brain, spacing_mm=1.5, cancelled=lambda: False):
    from .anatomy_alignment import check_cancel
    inside = np.zeros(ct.shape, bool); outer = np.zeros(ct.shape, bool)
    supported = np.zeros(ct.shape[2], bool)
    radius = max(1, int(np.ceil(2/spacing_mm)))
    x, y = np.ogrid[-radius:radius+1, -radius:radius+1]
    disk = x*x+y*y <= radius*radius
    for z in range(ct.shape[2]):
        check_cancel(cancelled)
        seed = brain[:, :, z]
        if seed.sum() < 20: continue
        bone = ct[:, :, z] > 300
        wall = ndi.binary_closing(bone, structure=disk)
        filled = ndi.binary_fill_holes(wall)
        cavities, count = ndi.label(filled & ~wall)
        if not count: continue
        overlap = np.bincount(cavities[seed], minlength=count+1); overlap[0] = 0
        label = int(overlap.argmax())
        if label == 0: continue
        cavity = cavities == label
        # A plausible cranial cavity must cover most of this brain slice and
        # must not touch the acquisition border or grow into facial spaces.
        if overlap[label]/seed.sum() < .7 or cavity.sum()/seed.sum() > 3:
            continue
        if any(np.any(face) for face in [cavity[0], cavity[-1], cavity[:, 0], cavity[:, -1]]): continue
        ring = ndi.binary_dilation(cavity, structure=disk) & wall
        enclosure = ndi.binary_fill_holes(cavity | ring)
        inside[:, :, z] = enclosure & ~bone
        outer[:, :, z] = enclosure
        supported[z] = True
    return inside, outer, supported


def cranial_envelope(c, cancelled=lambda: False):
    from .anatomy_alignment import check_cancel
    brains = [r for r in c.document['layers'] if r['kind'] == 'brain']
    cts = [r for r in c.document['series'] if r['modality'].upper() == 'CT']
    cts.sort(key=lambda r: ('bone' not in (r['id']+' '+r.get('label', '')+' '+r['original'].get('name', '')).lower(), r['id']))
    for ct in cts:
        try: ca, _ = display_affine(c, ct, preview=True)
        except ValueError: continue
        for brain in brains:
            try: ba, _ = display_affine(c, brain, preview=True)
            except ValueError: continue
            key = digest({'version': VERSION, 'ct': ct['data'], 'ca': ca.tolist(), 'brain': brain['data'], 'ba': ba.tolist()})
            cache = getattr(c, '_cranial_display_cache', None)
            if cache and cache[0] == key: return cache[1]
            check_cancel(cancelled)
            # Axis-aligned patient RAS grid: axial ring topology must not depend
            # on source voxel ordering, left/right flips, or gantry orientation.
            corners = np.array(np.meshgrid(*[(0, n-1) for n in ct['data']['shape']], indexing='ij')).reshape(3, -1).T
            corners = world(ca, corners); spacing = 1.5
            a = np.diag([spacing, spacing, spacing, 1.]); a[:3, 3] = corners.min(0)
            shape = np.ceil((corners.max(0)-corners.min(0))/spacing).astype(int)+1
            if np.prod(shape) > 32_000_000: raise ValueError('Cranial envelope exceeds display grid budget')
            data = resample(c.array(ct['data']), ca, tuple(shape), a, order=1)
            seed = resample(c.mask(brain).astype(np.uint8), ba, tuple(shape), a, order=0) > 0
            inside, outer, supported = enclosed_rings(data, seed, spacing, cancelled)
            result = {'inside': inside, 'outer': outer, 'supported': supported, 'affine': a,
                      'ctSeries': ct['id'], 'brainLayer': brain['id'], 'key': key,
                      'supportedSlices': int(supported.sum()), 'brainSlices': int(np.count_nonzero(seed.any(axis=(0, 1))))}
            c._cranial_display_cache = (key, result)
            return result
    return None


def intracranial_vessels(c, row, *, excluded=False, cancelled=lambda: False):
    from .anatomy_alignment import check_cancel
    envelope = cranial_envelope(c, cancelled)
    full = c.mask(row)
    coords = np.argwhere(full)
    keep = np.zeros(len(coords), bool); known = np.zeros(len(coords), bool); outer = np.zeros(len(coords), bool)
    if envelope is not None:
        source, _ = display_affine(c, row, preview=True)
        xyz = world(np.linalg.inv(envelope['affine']) @ source, coords)
        in_bounds = ((xyz >= 0) & (xyz <= np.array(envelope['inside'].shape)-1)).all(axis=1)
        valid = xyz[in_bounds]
        keep[in_bounds] = ndi.map_coordinates(envelope['inside'].astype(np.uint8), valid.T, order=0) > 0
        outer[in_bounds] = ndi.map_coordinates(envelope['outer'].astype(np.uint8), valid.T, order=0) > 0
        known[in_bounds] = envelope['supported'][np.rint(valid[:, 2]).astype(int)]
    check_cancel(cancelled)
    hidden = np.zeros(full.shape, bool); hidden[tuple(coords[~keep].T)] = True
    _, fragments = ndi.label(hidden, structure=np.ones((3, 3, 3)))
    filtered = hidden if excluded else full & ~hidden
    result = deepcopy(row)
    result['data'] = c.put_array(filtered.astype(np.uint8)); result['label_value'] = 1
    result.pop('probability', None)
    result['edits'] = [{'display_only': VERSION, 'envelope': envelope['key'] if envelope else None, 'excluded': excluded}]
    if excluded: result['color'] = '#ffad54'
    volume = abs(np.linalg.det(affine(row['affine'])[:3, :3]))
    count = int((~keep).sum())
    report = {'scope': 'excluded' if excluded else 'intracranial', 'hiddenVoxels': count,
              'excludedVolumeMm3': count*volume, 'excludedFragments': int(fragments),
              'outsideVoxels': int((known & ~outer).sum()), 'boneConflictVoxels': int((known & outer & ~keep).sum()),
              'uncertainVoxels': int((~known).sum()), 'sourcePreserved': True,
              'reason': (f'{count:,} source voxels ({count*volume/1000:.2f} mL), {fragments:,} fragments '
                         + ('shown as exclusions from skull-supported view. ' if excluded else 'hidden by skull-supported filter. ')
                         + f'Unknown skull coverage: {int((~known).sum()):,}; '
                         + f'estimated outside skull: {int((known & ~outer).sum()):,}; '
                         + f'bone conflict: {int((known & outer & ~keep).sum()):,} voxels.')}
    if c.document.get('coverage_notes', {}).get('privacy_redacted'):
        report['coverageReason'] = 'privacy-redacted'
        report['reason'] += ' Missing skull regions are privacy-redacted; no removed anatomy is reconstructed.'
    if envelope is None or not envelope['supportedSlices']:
        report['reason'] += (' Skull coverage unavailable; showing the excluded source estimate.' if excluded else ' Skull coverage unavailable; this filter withholds vessels. Full source estimate remains available.')
    else:
        report.update(ctSeries=envelope['ctSeries'], supportedSlices=envelope['supportedSlices'], brainSlices=envelope['brainSlices'])
    return result, report
