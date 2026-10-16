"""Reproducible, quality-audited display alignment. Never grants planning review.

All persisted matrices map source RAS mm to reference RAS mm. Pairwise ITK
optimizers sample fixed -> moving LPS; conversion happens at this boundary only.
"""
from copy import deepcopy
from contextlib import contextmanager
import time
from dataclasses import dataclass
import numpy as np
import nibabel as nib
from scipy import ndimage as ndi
from .patient_anatomy import affine, world, digest, file_hash, reconstruction_lock

VERSION = 'rigid-quality-v2.4'
LPS = np.diag([-1., -1., 1., 1.])


def check_cancel(cancelled):
    if cancelled():
        raise InterruptedError('Alignment cancelled')


@contextmanager
def alignment_lock(cancelled, progress):
    last_notice = -float("inf")
    while True:
        check_cancel(cancelled)
        lock = reconstruction_lock()
        try:
            lock.__enter__()
            break
        except ValueError as error:
            if 'Another local reconstruction' not in str(error): raise
            if time.monotonic()-last_notice >= 1:
                progress(0, 'Waiting for the local reconstruction slot…')
                last_notice = time.monotonic()
            time.sleep(.1)
    try:
        yield
    finally:
        lock.__exit__(None, None, None)


def rigid(matrix):
    a = affine(matrix)
    if not np.allclose(a[:3, :3].T @ a[:3, :3], np.eye(3), atol=1e-5) or np.linalg.det(a[:3, :3]) < 0:
        raise ValueError('Alignment requires a proper rigid transform')
    return a


def from_itk(transform):
    origin = np.array(transform.TransformPoint((0., 0., 0.)))
    a = np.eye(4)
    a[:3, 3] = origin
    for i in range(3):
        a[:3, i] = np.array(transform.TransformPoint(tuple(np.eye(3)[i]))) - origin
    return rigid(LPS @ np.linalg.inv(a) @ LPS)


def to_itk(matrix):
    import SimpleITK as sitk
    a = LPS @ np.linalg.inv(rigid(matrix)) @ LPS
    t = sitk.Euler3DTransform()
    t.SetMatrix(a[:3, :3].ravel().tolist())
    t.SetTranslation(a[:3, 3].tolist())
    return t


@dataclass
class Grid:
    data: np.ndarray
    affine: np.ndarray
    head: np.ndarray
    brain: np.ndarray | None = None
    ct: bool = False

    def image(self, *, soft_ct=True):
        import SimpleITK as sitk
        data = np.clip(self.data, -200 if soft_ct else -500, 200 if soft_ct else 1800) if self.ct else self.data.copy()
        if not self.ct and self.head.any():
            lo, hi = np.percentile(data[self.head], [1, 99])
            data = np.clip((data-lo)/max(hi-lo, 1e-6), 0, 1)
        im = sitk.GetImageFromArray(data.transpose(2, 1, 0).astype(np.float32))
        a = LPS @ self.affine
        spacing = np.linalg.norm(a[:3, :3], axis=0)
        direction = a[:3, :3]/spacing
        if not np.allclose(direction.T @ direction, np.eye(3), atol=1e-5):
            raise ValueError('Sheared source geometry requires explicit resampling')
        im.SetSpacing(spacing.tolist()); im.SetOrigin(a[:3, 3].tolist())
        im.SetDirection(direction.ravel().tolist())
        return im


def load_grid(c, row, cancelled=lambda: False):
    check_cancel(cancelled)
    a = affine(row['affine']).copy()
    stride = np.maximum(1, np.rint(2/np.linalg.norm(a[:3, :3], axis=0)).astype(int))
    slices = tuple(slice(None, None, int(s)) for s in stride)
    data = np.asarray(c.array(row['data'])[slices], np.float32).copy()
    a[:3, :3] = a[:3, :3] @ np.diag(stride)
    ct = row['modality'].upper().startswith('CT')
    threshold = -500 if ct else max(float(np.percentile(data, 98))*.06, 1e-6)
    head = ndi.binary_fill_holes(data > threshold)
    labels, n = ndi.label(head)
    if n:
        counts = np.bincount(labels.ravel()); counts[0] = 0
        head = labels == counts.argmax()
    brain = None
    for layer in c.document['layers']:
        if layer['kind'] == 'brain' and layer['source_series'] == row['id']:
            transform = np.linalg.inv(affine(layer['affine'])) @ a
            brain = ndi.affine_transform(c.mask(layer).astype(np.uint8), transform,
                                         output_shape=data.shape, order=0) > 0
            break
    # Crop air while retaining physical coordinates and a 10 mm margin.
    if head.any():
        xyz = np.argwhere(head)
        lo = np.maximum(0, xyz.min(0)-5); hi = np.minimum(data.shape, xyz.max(0)+6)
        crop = tuple(slice(int(l), int(h)) for l, h in zip(lo, hi))
        a[:3, 3] = world(a, lo)
        data, head = data[crop].copy(), head[crop].copy()
        brain = brain[crop].copy() if brain is not None else None
    check_cancel(cancelled)
    return Grid(data, a, head, brain, ct)


def probe_points(grid, count=20000):
    ids = np.argwhere(grid.brain if grid.brain is not None and grid.brain.any() else grid.head)
    return world(grid.affine, ids[::max(1, len(ids)//count)])


def displacement(a, b, points):
    return float(np.percentile(np.linalg.norm(world(a, points)-world(b, points), axis=1), 95)) if len(points) else None


def score(fixed, moving, matrix):
    """Independent deterministic voxel-centre samples, not optimizer samples."""
    mask = ndi.binary_dilation(fixed.brain, iterations=3) & fixed.head if fixed.brain is not None else fixed.head
    ids = np.argwhere(mask)
    ids = ids[1::max(2, len(ids)//60000)]
    if len(ids) < 64:
        return {'similarity': None, 'coverage': 0., 'bone_conflict': None}
    xyz = world(np.linalg.inv(moving.affine) @ np.linalg.inv(matrix) @ fixed.affine, ids)
    valid = ((xyz >= 0) & (xyz <= np.array(moving.data.shape)-1)).all(axis=1)
    x = fixed.data[tuple(ids[valid].T)]
    y = ndi.map_coordinates(moving.data, xyz[valid].T, order=1, mode='constant', cval=-1024 if moving.ct else 0)
    support = ndi.map_coordinates(moving.head.astype(np.uint8), xyz[valid].T, order=0) > 0
    coverage = float(support.sum()/len(ids))
    conflict = None
    if fixed.brain is not None and moving.ct:
        in_brain = fixed.brain[tuple(ids[valid].T)] & support
        if in_brain.any():
            conflict = float(np.mean(y[in_brain] > 300))
    if support.sum() < 64 or np.std(x[support]) < 1e-6 or np.std(y[support]) < 1e-6:
        return {'similarity': None, 'coverage': coverage, 'bone_conflict': conflict}
    if fixed.ct and moving.ct:
        core = ndi.binary_erosion(fixed.head, iterations=2)
        moving_core = ndi.binary_erosion(moving.head, iterations=2)
        interior = core[tuple(ids[valid].T)] & (ndi.map_coordinates(moving_core.astype(np.uint8), xyz[valid].T, order=0) > 0)
        if interior.sum() >= 64: support &= interior
    x, y = x[support], y[support]
    if len(x) < 64 or np.std(x) < 1e-6 or np.std(y) < 1e-6:
        return {'similarity': None, 'coverage': coverage, 'bone_conflict': conflict}
    if fixed.ct and moving.ct:
        similarity = float(np.corrcoef(np.clip(x, -500, 1800), np.clip(y, -500, 1800))[0, 1])
    else:
        if fixed.ct: x = np.clip(x, -200, 200)
        if moving.ct: y = np.clip(y, -200, 200)
        h = np.histogram2d(x, y, bins=50)[0]; h /= h.sum()
        expected = h.sum(0)[None, :] * h.sum(1)[:, None]
        nz = h > 0
        similarity = float(np.sum(h[nz]*np.log(h[nz]/expected[nz])))
    return {'similarity': similarity, 'coverage': coverage, 'bone_conflict': conflict}


def optimize(fixed, moving, initial, cancelled):
    import SimpleITK as sitk
    check_cancel(cancelled)
    f, m = fixed.image(soft_ct=not moving.ct), moving.image(soft_ct=not fixed.ct)
    reg = sitk.ImageRegistrationMethod()
    if fixed.ct and moving.ct: reg.SetMetricAsCorrelation()
    else: reg.SetMetricAsMattesMutualInformation(50)
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(.2, 20261015)
    # Crop plus head masks avoids air/background dominating the metric.
    for g, image, setter in [(fixed, f, reg.SetMetricFixedMask), (moving, m, reg.SetMetricMovingMask)]:
        support = ndi.binary_erosion(g.head, iterations=2) if fixed.ct and moving.ct else g.head
        if support.sum() < 64: support = g.head
        mask = sitk.GetImageFromArray(support.transpose(2, 1, 0).astype(np.uint8)); mask.CopyInformation(image); setter(mask)
    reg.SetInterpolator(sitk.sitkLinear)
    reg.SetOptimizerAsRegularStepGradientDescent(1., .005, 240, .5)
    reg.SetOptimizerScalesFromPhysicalShift()
    reg.SetShrinkFactorsPerLevel([4, 2, 1]); reg.SetSmoothingSigmasPerLevel([2, 1, 0])
    reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    transform = to_itk(initial)
    # Changing centre must preserve the initial map.
    center = probe_points(fixed).mean(0) if fixed.head.any() else world(fixed.affine, (np.array(fixed.data.shape)-1)/2)
    center = (LPS[:3, :3] @ center).tolist()
    old = np.array(transform.TransformPoint((0., 0., 0.)))
    transform.SetCenter(center)
    rotation = np.array(transform.GetMatrix()).reshape(3, 3)
    transform.SetTranslation((old-np.array(center)+rotation@center).tolist())
    reg.SetInitialTransform(transform, inPlace=False)
    reg.AddCommand(sitk.sitkIterationEvent, lambda: reg.StopRegistration() if cancelled() else None)
    result = reg.Execute(f, m)
    check_cancel(cancelled)
    return from_itk(result), reg.GetOptimizerStopConditionDescription()


def estimate_pair(fixed, moving, existing=None, cancelled=lambda: False):
    initial = np.eye(4)
    centered = initial.copy()
    fp, mp = probe_points(fixed), probe_points(moving)
    if len(fp) and len(mp): centered[:3, 3] = fp.mean(0)-mp.mean(0)
    starts = [('identity', initial), ('head-centres', centered)]
    if existing is not None:
        try: starts.append(('previous', rigid(existing)))
        except ValueError: pass
    candidates = []
    for name, matrix in starts:
        check_cancel(cancelled)
        candidates.append({'name': name, 'matrix': matrix.tolist(), **score(fixed, moving, matrix)})
        try:
            fitted, stop = optimize(fixed, moving, matrix, cancelled)
            candidates.append({'name': name+'-optimized', 'matrix': fitted.tolist(), 'stop': stop, **score(fixed, moving, fitted)})
        except RuntimeError as error:
            candidates[-1]['optimizer_error'] = str(error)[:400]
    usable = [r for r in candidates if r['similarity'] is not None and r['coverage'] >= .5]
    warnings = []
    if not usable:
        best = max(candidates, key=lambda r: (r['coverage'], r['similarity'] if r['similarity'] is not None else -float('inf'), r['name'] == 'previous'))
        warnings.append('Insufficient overlap or image texture; showing best available geometry estimate')
    else:
        baseline = candidates[0]
        # An image-metric win cannot silently buy worse bone penetration.
        baseline_conflict = baseline['bone_conflict'] if baseline['coverage'] >= .9 else None
        guarded = [r for r in usable if baseline_conflict is None or r['bone_conflict'] is None or r['bone_conflict'] <= baseline_conflict+.005]
        best = max(guarded or usable, key=lambda r: (r['similarity'], r['coverage']))
        if not guarded: warnings.append('Image similarity and anatomical conflicts disagree')
        unconstrained = max(usable, key=lambda r: r['similarity'])
        if unconstrained is not best and unconstrained['similarity'] > best['similarity']+.005:
            warnings.append('Higher image similarity rejected because bone conflict increased')
    matrix = np.array(best['matrix'])
    inverse_error = None
    try:
        reverse, _ = optimize(moving, fixed, np.linalg.inv(matrix), cancelled)
        inverse_error = displacement(reverse @ matrix, np.eye(4), mp)
    except RuntimeError:
        warnings.append('Reverse registration could not be evaluated')
    fitted = [r for r in usable if r['name'].endswith('-optimized')]
    stability = max((displacement(matrix, np.array(r['matrix']), mp) or 0 for r in fitted), default=None)
    if inverse_error is not None and inverse_error > 2: warnings.append(f'Forward/reverse disagreement {inverse_error:.2f} mm exceeds 2 mm')
    if stability is not None and stability > 2: warnings.append(f'Initialization disagreement {stability:.2f} mm exceeds 2 mm')
    if best['coverage'] < .9: warnings.append(f'Limited anatomical coverage: {best["coverage"]:.1%}')
    if best['bone_conflict'] is not None and best['bone_conflict'] > .01:
        warnings.append(f'Brain mask overlaps dense bone: {best["bone_conflict"]:.1%}; inspect anatomy')
    return matrix, {'status': 'warning' if warnings else 'consistent', 'warnings': warnings,
                    'selected': best['name'], 'candidates': candidates, 'metrics': {
                        **{k: best[k] for k in ['similarity', 'coverage', 'bone_conflict']},
                        'inverse_p95_mm': inverse_error, 'initialization_p95_mm': stability},
                    'meaning': 'Automated display checks; not anatomical or clinical certification'}


def input_key(c, row, parent):
    return digest({'version': VERSION, 'patient': c.document['patient_id'], 'reference': c.document['reference_series'],
                   'source': {k: row.get(k) for k in ['id', 'original', 'data', 'affine', 'timepoint', 'modality', 'units_status']},
                   'parent': {k: parent.get(k) for k in ['id', 'original', 'data', 'affine', 'timepoint', 'modality', 'units_status']},
                   'parent_transform': c.document['transforms'].get(parent['id']),
                   'coverage_notes': c.document.get('coverage_notes'),
                   'brain_masks': [{k: l.get(k) for k in ['data', 'affine', 'source_series']} for l in c.document['layers'] if l['kind'] == 'brain']})


def ensure_alignment(c, *, force=False, cancelled=lambda: False, progress=lambda *args: None):
    """Updates an in-memory candidate document; caller commits once, atomically."""
    reference = c.series(c.document['reference_series'])
    rows = [s for s in c.document['series'] if s['id'] != reference['id']]
    if not rows: return False
    # Soft-kernel CT is preferable for CT/MRI. Other CTs connect to it by a
    # measured same-modality edge, never an assumed equal-grid identity.
    cts = [s for s in rows if s['modality'].upper() == 'CT']
    anchor = min(cts, key=lambda s: ('brain' not in (s['id']+' '+s.get('label', '')+' '+s['original'].get('name', '')).lower(), s['id'])) if cts else None
    rows.sort(key=lambda s: (s is not anchor, s['id']))
    grids = {}; changed = False
    def grid(row):
        if row['id'] not in grids: grids[row['id']] = load_grid(c, row, cancelled)
        return grids[row['id']]
    with alignment_lock(cancelled, progress):
        for row in [reference]+rows:
            check_cancel(cancelled)
            if row.get('patient_id') != c.document['patient_id'] or row['timepoint'] != reference['timepoint']:
                raise ValueError('Alignment inputs must belong to the same patient and timepoint')
            original_path = c.path(row['original']['path'])
            if file_hash(original_path) != row['original']['sha256']:
                raise ValueError('Original source changed before alignment')
            header = nib.load(original_path)
            if header.header.get_xyzt_units()[0] != 'mm':
                raise ValueError('Automatic alignment requires declared millimetre source units')
            q, qc = header.get_qform(coded=True); s, sc = header.get_sform(coded=True)
            if qc and sc and not np.allclose(q, s, atol=1e-3):
                raise ValueError('Conflicting NIfTI qform/sform; resolve source geometry before alignment')
            if not np.allclose(header.affine, row['affine'], atol=1e-5):
                raise ValueError('Source header geometry changed before alignment')
        for i, row in enumerate(rows):
            check_cancel(cancelled)
            parent = anchor if anchor is not None and row in cts and row is not anchor else reference
            key = input_key(c, row, parent)
            old = c.document['transforms'].get(row['id'])
            if not force and old and old.get('input_key') == key and old.get('quality'):
                try:
                    c.registration_matrix(row['id']); continue
                except ValueError:
                    pass
            progress(i/max(1, len(rows)), 'Aligning '+row.get('label', row['id']))
            parent_matrix = rigid(c.document['transforms'][parent['id']]['matrix'])
            previous = None
            if old and old.get('source_sha256') == row['original']['sha256'] and old.get('reference_sha256') == reference['original']['sha256']:
                try: previous = np.linalg.inv(parent_matrix) @ rigid(old['matrix'])
                except ValueError: pass
            relative, quality = estimate_pair(grid(parent), grid(row), previous, cancelled)
            if c.document.get('coverage_notes', {}).get('privacy_redacted'):
                quality['notes'] = ['Privacy-redacted regions are excluded from image similarity; removed anatomy is not reconstructed.']
                quality['warnings'] = [w for w in quality['warnings'] if not w.startswith('Limited anatomical coverage:')]
                quality['status'] = 'warning' if quality['warnings'] else 'consistent'
            composed = parent_matrix @ relative
            if old:
                c.document.setdefault('registration_history', []).append({'series_id': row['id'], 'transform': deepcopy(old)})
                try: quality['metrics']['previous_disagreement_p95_mm'] = displacement(rigid(old['matrix']), composed, probe_points(grid(row)))
                except ValueError: quality['warnings'].append('Previous transform was invalid')
            if parent is not reference:
                quality['warnings'].extend('Parent: '+w for w in c.document['transforms'][parent['id']].get('quality', {}).get('warnings', []))
                quality['metrics']['ct_relative_displacement_p95_mm'] = displacement(relative, np.eye(4), probe_points(grid(row)))
            if quality['warnings']: quality['status'] = 'warning'
            c.set_registration(row['id'], composed, status='unreviewed', method=VERSION)
            c.document['transforms'][row['id']].update(direction='source-RAS-mm-to-reference-RAS-mm',
                reference_series=reference['id'], parent_series=parent['id'], source_to_parent=relative.tolist(),
                algorithm_version=VERSION, input_key=key, quality=quality,
                transform_digest=digest({'matrix': composed.tolist(), 'relative': relative.tolist()}))
            changed = True
    return changed
