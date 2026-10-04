#!/usr/bin/env python3
"""One prospective saved-array QC; only the explicitly bound T1, mask and SDT."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import itertools
import json
import os
from pathlib import Path
import resource
import sys
import time


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    temporary.replace(path)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify(root, bindings):
    for item in bindings:
        require(sha(root / item['path']) == item['sha256'], 'Changed binding: ' + item['path'])


def corners(shape, full_cell=False):
    import numpy as np
    return np.array(list(itertools.product(*[
        (-.5, size-.5) if full_cell else (0., size-1.) for size in shape])))


def displacement(first, second, shape, full_cell=False):
    import numpy as np
    points = corners(shape, full_cell)
    return float(np.linalg.norm((np.c_[points, np.ones(8)] @ (first-second).T)[:, :3], axis=1).max())


def reconstruct(distance):
    """Match reviewed Surfa 0.6.3, including default connectivity and top-k rule."""
    import numpy as np
    from scipy import ndimage
    thresholded = distance < 1.
    labels, count = ndimage.label(thresholded)
    require(count > 0, 'Empty strict SDT<1 mm support')
    sizes = np.bincount(labels.flat)[1:]
    top = (-sizes).argsort()[:1] + 1
    selected = np.isin(labels, top)
    filled = ndimage.binary_fill_holes(selected)
    return filled, {'threshold_voxels': int(thresholded.sum()),
                    'threshold_components_6_connectivity': int(count),
                    'largest_component_label': int(top[0]),
                    'largest_component_voxels': int(selected.sum()),
                    'largest_component_size_ties': int((sizes == sizes.max()).sum()),
                    'discarded_threshold_voxels': int(thresholded.sum() - selected.sum()),
                    'filled_hole_voxels': int((filled & ~selected).sum())}


def self_test():
    import numpy as np
    # Face-disconnected voxels stay separate; strict threshold excludes exactly one.
    d = np.full((7, 7, 7), 2.)
    d[1:4, 1:4, 1:4] = 0.; d[2, 2, 2] = 2.; d[4, 4, 3] = 0.; d[6, 6, 6] = 1.
    actual, info = reconstruct(d)
    expected = np.zeros_like(d, dtype=bool); expected[1:4, 1:4, 1:4] = True
    require(np.array_equal(actual, expected), 'Component/fill/strict-threshold control failed')
    require(info['threshold_components_6_connectivity'] == 2 and info['filled_hole_voxels'] == 1,
            'Reconstruction accounting control failed')
    a = np.eye(4); b = a.copy(); b[0, 0] += .1
    require(abs(displacement(a, b, (7, 7, 7), True) - .65) < 1e-12, 'Full-cell bound control failed')
    require(abs(displacement(a, b, (7, 7, 7), False) - .6) < 1e-12, 'Centre bound control failed')
    print(json.dumps({'analytic_controls': 'passed', 'patient_arrays_opened': False}))


def geometry(image, reference, tolerance):
    import numpy as np
    import nibabel as nib
    require(image.shape == reference.shape, 'Native grid shape mismatch')
    affine = image.affine
    require(affine.shape == (4, 4) and np.isfinite(affine).all(), 'Invalid native affine')
    require(np.array_equal(affine[3], [0., 0., 0., 1.]) and abs(np.linalg.det(affine[:3, :3])) > 0.,
            'Noninvertible or nonaffine native frame')
    require(image.header.get_xyzt_units()[0] == 'mm', 'Missing native millimetre units')
    q, qc = image.get_qform(coded=True); s, sc = image.get_sform(coded=True)
    require(qc or sc, 'Missing coded native frame')
    if qc and sc:
        require(displacement(q, s, image.shape, True) <= tolerance, 'Coded spatial transforms conflict')
    center_error = displacement(affine, reference.affine, image.shape)
    cell_error = displacement(affine, reference.affine, image.shape, True)
    require(center_error <= tolerance and cell_error <= tolerance, 'Native physical corners changed')
    require(np.array_equal(affine, reference.affine), 'Exact native affine changed')
    return {'shape': list(image.shape), 'array_dtype': str(image.get_data_dtype()),
            'affine_ras_mm': affine.tolist(), 'qform_code': int(qc), 'sform_code': int(sc),
            'qform': None if q is None else q.tolist(), 'sform': None if s is None else s.tolist(),
            'units': list(image.header.get_xyzt_units()), 'orientation': list(nib.aff2axcodes(affine)),
            'spacing_mm': np.linalg.norm(affine[:3, :3], axis=0).tolist(),
            'affine_exactly_equal_to_T1': True, 'maximum_voxel_center_corner_difference_mm': center_error,
            'maximum_full_cell_corner_difference_mm': cell_error}


def render(source, mask, settings, output):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    low, high = np.percentile(source, settings['display']['T1_percentiles'])
    require(high > low, 'Degenerate T1 display window')
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    records = []
    for axis, index in enumerate(settings['planes']):
        other = [i for i in range(3) if i != axis]
        plane, envelope = np.take(source, index, axis=axis), np.take(mask, index, axis=axis)
        for row in (0, 1):
            ax = axes[row, axis]
            ax.imshow(plane.T, origin='lower', cmap='gray', vmin=low, vmax=high, interpolation='nearest')
            if row == 1 and envelope.any() and not envelope.all():
                ax.contour(envelope.T, [.5], colors=['#39dcbd'], linewidths=.85)
            ax.set_title(f'Native {"ijk"[axis]}={index} | ' + ('T1' if row == 0 else 'estimated mask outline'))
            ax.set_xlabel(f'native {"ijk"[other[0]]} voxel index')
            ax.set_ylabel(f'native {"ijk"[other[1]]} voxel index')
        records.append({'axis': axis, 'index': index, 'mask_voxels_in_plane': int(envelope.sum())})
    fig.suptitle('RESECT Case4 | fixed native planes | unreviewed envelope estimate\n'
                 'Original oblique grid; no reorientation or registration. Cyan: mask boundary; no cortical or access approval.', fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, .92))
    destination = output / 'Case4-fixed-native-planes.png'
    fig.savefig(destination, dpi=settings['display']['dpi']); plt.close(fig)
    return {'file': destination.name, 'sha256': sha(destination), 'planes': records,
            'window_percentiles': settings['display']['T1_percentiles'],
            'window_values': [float(low), float(high)], 'native_index_display_no_resampling': True,
            'anatomical_review': 'pending_root_review'}


def worker(root, settings, declaration_sha):
    output = root / settings['receipt_directory']
    path = output / 'qc-receipt.json'
    require(not path.exists(), 'Refusing another QC attempt')
    started = time.monotonic()
    report = {'status': 'running', 'started_at': datetime.now(timezone.utc).isoformat(),
              'script_sha256': sha(__file__), 'prospective_sha256': declaration_sha,
              'brain_reviewed': False, 'working_brain_mask': None, 'world_transform_accepted': False,
              'cortical_access_permitted': False, 'inference_attempts': 0, 'optimizer_updates': 0,
              'other_patient_or_motion_or_annotation_arrays_opened': False, 'stage': 'verify_bindings'}
    write(path, report)
    bindings = settings['bindings'] + list(settings['images'].values())
    try:
        verify(root, bindings)
        import numpy as np
        import scipy
        import nibabel as nib
        from scipy import ndimage
        import matplotlib
        report['runtime'] = {'python': sys.version, 'numpy': np.__version__, 'scipy': scipy.__version__,
                             'nibabel': nib.__version__, 'matplotlib': matplotlib.__version__}
        for name, module in [('numpy', np), ('scipy', scipy), ('nibabel', nib), ('matplotlib', matplotlib)]:
            pinned = settings['module_origins'][name]
            require(Path(module.__file__).resolve() == (root / pinned).resolve(), 'Unexpected module origin: ' + name)
            report['runtime'][name + '_origin'] = str(Path(module.__file__).resolve())
        require(np.__version__ == '2.2.6', 'Changed reconstruction argsort runtime')
        report['stage'] = 'read_three_native_arrays'; write(path, report)
        images = {name: nib.load(root / item['path']) for name, item in settings['images'].items()}
        reference = images['T1']
        require(list(reference.shape) == settings['shape'], 'Unexpected native T1 shape')
        report['geometry'] = {name: geometry(image, reference, settings['corner_tolerance_mm']) for name, image in images.items()}
        arrays = {name: np.asanyarray(image.dataobj) for name, image in images.items()}
        require(all(np.isfinite(array).all() for array in arrays.values()), 'Nonfinite native array')
        mask = arrays['mask']; distance = arrays['SDT']
        require(np.isin(mask, [0, 1]).all() and mask.any(), 'Nonbinary or empty saved mask')
        mask = mask.astype(bool)
        reconstructed, measures = reconstruct(distance)
        measures['reconstructed_mask_mismatch_voxels'] = int(np.count_nonzero(reconstructed != mask))
        report['reconstruction'] = measures; write(path, report)
        require(measures['reconstructed_mask_mismatch_voxels'] == 0, 'Saved mask differs from exact upstream reconstruction')
        del reconstructed
        labels, components = ndimage.label(mask)
        sizes = np.bincount(labels.flat)[1:]; del labels
        volume = abs(float(np.linalg.det(reference.affine[:3, :3])))
        contacts = {f'{"ijk"[axis]}_{side}': int(np.count_nonzero(np.take(mask, index, axis=axis)))
                    for axis in range(3) for side, index in [('low', 0), ('high', mask.shape[axis]-1)]}
        boundary = np.zeros_like(mask)
        for axis in range(3):
            for index in (0, mask.shape[axis]-1):
                selection = [slice(None)] * 3; selection[axis] = index; boundary[tuple(selection)] = True
        report['measurements'] = {'mask_voxels': int(mask.sum()), 'voxel_volume_mm3': volume,
              'mask_volume_mm3': float(mask.sum() * volume), 'mask_volume_ml': float(mask.sum() * volume / 1000),
              'mask_components_6_connectivity': int(components), 'component_sizes_descending': sorted(sizes.tolist(), reverse=True),
              'FOV_face_contact_voxels': contacts, 'FOV_boundary_union_contact_voxels': int((boundary & mask).sum()),
              'SDT_range_mm': [float(distance.min()), float(distance.max())],
              'SDT_exterior_fill_100_count': int(np.count_nonzero(distance == 100.)),
              'T1_range': [float(arrays['T1'].min()), float(arrays['T1'].max())],
              'anatomical_thresholds_or_accuracy_claims': None}
        del boundary
        report['stage'] = 'roundtrip_and_render'; write(path, report)
        raw = root / settings['roundtrip_directory']; raw.mkdir(parents=True, exist_ok=False)
        roundtrips = {}
        for name, image in images.items():
            target = raw / (name + '-native-roundtrip.nii')
            nib.save(image, target)
            reopened = nib.load(target)
            values = np.asanyarray(reopened.dataobj)
            require(values.dtype == arrays[name].dtype and np.array_equal(values, arrays[name]), 'Array roundtrip changed: ' + name)
            require(np.array_equal(reopened.affine, image.affine), 'Affine roundtrip changed: ' + name)
            require(geometry(reopened, reference, settings['corner_tolerance_mm']) == report['geometry'][name],
                    'Native header geometry roundtrip changed: ' + name)
            roundtrips[name] = {'path': str(target.relative_to(root)), 'sha256': sha(target), 'bytes': target.stat().st_size,
                               'native_array_dtype_values_affine_geometry_exact': True}
            del reopened, values
        report['roundtrip'] = roundtrips
        report['visual'] = render(arrays['T1'], mask, settings, output)
        verify(root, bindings)
        report.update(status='engineering_checks_passed_anatomy_unreviewed', stage='complete',
                      all_bound_sources_inputs_and_estimates_unchanged=True)
    except BaseException as error:
        report.update(status='failed_or_incomplete', error={'type': type(error).__name__, 'message': str(error)})
        raise
    finally:
        report.update(completed_at=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.monotonic()-started,
                      peak_self_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024)))
        write(path, report)
    print(json.dumps({'status': report['status'], 'elapsed_seconds': report['elapsed_seconds'], 'receipt_sha256': sha(path)}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path)
    parser.add_argument('--declaration', type=Path)
    parser.add_argument('--expected-declaration-sha')
    parser.add_argument('--mode', choices=['run', 'worker', 'self-test'], required=True)
    args = parser.parse_args()
    if args.mode == 'self-test':
        self_test(); return
    root = args.root.resolve(); declaration = args.declaration.resolve()
    require(sha(declaration) == args.expected_declaration_sha, 'Prospective settings changed')
    settings = json.loads(declaration.read_text())
    require(sha(__file__) == settings['script_sha256'], 'QC source changed')
    if args.mode == 'worker':
        worker(root, settings, args.expected_declaration_sha); return
    verify(root, settings['bindings'])
    spec = importlib.util.spec_from_file_location('frozen_process_supervisor', root / settings['supervisor']['path'])
    runtime = importlib.util.module_from_spec(spec); spec.loader.exec_module(runtime)
    environment = runtime.private_environment({'caps': {'thread_environment': settings['thread_environment']}})
    for key in list(environment):
        if key.startswith(('PYTHON', 'DYLD_')) or key in ('LD_PRELOAD', 'LD_LIBRARY_PATH'):
            environment.pop(key)
    environment.update(settings['thread_environment'])
    environment['MPLCONFIGDIR'] = str(root / settings['matplotlib_cache'])
    environment['MPLBACKEND'] = 'Agg'
    command = [str(root / settings['interpreter']), '-B', str(Path(__file__).resolve()), '--mode', 'worker',
               '--root', str(root), '--declaration', str(declaration), '--expected-declaration-sha', args.expected_declaration_sha]
    result = runtime.supervise(command, root / settings['receipt_directory'] / 'attempt-01', cwd=root,
                               environment=environment, seconds=settings['caps']['wall_seconds'], rss_bytes=settings['caps']['rss_bytes'])
    verify(root, settings['bindings'])
    print(json.dumps(result, indent=2))
    if result['status'] != 'completed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
