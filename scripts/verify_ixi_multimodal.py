#!/usr/bin/env python3
"""Read-only source, physical-slice, renderer-bridge and simulator QC of IXI265.

Image metrics and a source montage are software checks, not anatomical approval.
"""
import argparse
import json
from pathlib import Path
import gzip

import nibabel as nib
import numpy as np

from resectionlab.patient_anatomy import PatientImagingCollection, file_hash, resample, world
from resectionlab.anatomy_qa import layer_quality


def verify(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from resectionlab.workspace_bundle import load_workspace
    from resectionlab.surgical_environment import load_environment
    from resectionlab.surgical_engine import SurgicalEngine
    from resectionlab.surgical_rl_env import SurgicalRLEnv
    from resectionlab.desktop_bridge import BridgeSession, _Request

    root = root.resolve()
    c = PatientImagingCollection(root/'native')
    initial = c.identity()
    from resectionlab.anatomy_alignment import ensure_alignment
    if ensure_alignment(c):
        raise AssertionError('Desktop reopen would refit stored registration estimates')
    qc = root/'qc'
    qc.mkdir(exist_ok=True)
    headers = {}
    for row in c.document['series']:
        path = c.path(row['original']['path'])
        if file_hash(path) != row['original']['sha256']:
            raise ValueError('Original changed: ' + row['id'])
        # Consume the entire gzip stream to check the member CRC/trailer.
        with gzip.open(path, 'rb') as source:
            expanded = sum(len(block) for block in iter(lambda: source.read(1024**2), b''))
        image = nib.load(path)
        values = c.array(row['data'])
        headers[row['id']] = {'sha256': row['original']['sha256'], 'shape': list(image.shape),
            'spacing_mm': list(map(float, image.header.get_zooms())), 'axis_codes': list(nib.aff2axcodes(image.affine)),
            'affine_ras_mm': image.affine.tolist(), 'units': list(image.header.get_xyzt_units()),
            'qform_code': int(image.header['qform_code']), 'sform_code': int(image.header['sform_code']),
            'qform_sform_max_difference': float(np.max(abs(image.get_qform()-image.get_sform()))),
            'expanded_gzip_bytes': expanded, 'finite': bool(np.isfinite(values).all()),
            'intensity_range': [float(values.min()), float(values.max())]}
    raw = c.document['diffusion']['source_originals']
    for row in raw:
        path = c.path(row['path'])
        if file_hash(path) != row['sha256']:
            raise ValueError('Original diffusion changed')
        with gzip.open(path, 'rb') as source:
            while source.read(1024**2):
                pass
    brain = c.mask(c.layer('brain'))
    coords = np.argwhere(brain)
    points = world(np.asarray(c.layer('brain')['affine']), coords)
    lo, hi = points.min(0)-5, points.max(0)+5
    target = np.diag([2., 2., 2., 1.])
    target[:3, 3] = lo
    shape = tuple(np.ceil((hi-lo)/2).astype(int)+1)
    images, coverage = {}, {}
    for sid in ('t1', 'mra', 'dwi-b0', 'dwi-fa'):
        row = c.series(sid)
        registered = c.registration_matrix(sid) @ np.asarray(row['affine'])
        images[sid] = resample(c.array(row['data']), registered, shape, target, order=1)
        coverage[sid] = resample(np.ones(row['data']['shape'], np.uint8), registered, shape, target, order=0) > 0
    vessels = c.layer('vessels')
    mask = resample(c.mask(vessels).astype(np.uint8), c.registration_matrix('mra') @ np.asarray(vessels['affine']), shape, target, order=0)>0
    z = np.percentile(np.argwhere(mask)[:, 2], [25, 50, 75]).astype(int)
    figure, axes = plt.subplots(3, 4, figsize=(13, 10), facecolor='#131820')
    for i, slice_id in enumerate(z):
        for j, sid in enumerate(('t1', 'mra', 'dwi-b0', 'dwi-fa')):
            ax = axes[i, j]
            ax.set_facecolor('black')
            data = images[sid][:, :, slice_id]
            domain = coverage[sid][:, :, slice_id]
            data = np.ma.masked_where(~domain, data)
            vmax = 1 if sid == 'dwi-fa' else float(np.percentile(images[sid][images[sid]>0], 99))
            ax.imshow(data.T, origin='lower', cmap='gray', vmin=0, vmax=vmax)
            if mask[:, :, slice_id].any():
                ax.contour(mask[:, :, slice_id].T, levels=[.5], colors=['#ff7859'], linewidths=.4)
            ax.set_title(f'{sid} | RAS z={target[2,3]+2*slice_id:.1f} mm', color='white', fontsize=9)
            ax.axis('off')
    figure.suptitle('IXI265: same-subject acquired MRI / MRA / b0 and derived FA\nOrange: published partial vessel annotation; rigid estimates unreviewed; no tumor', color='white', fontsize=12)
    figure.tight_layout(rect=[0, 0, 1, .94])
    figure.savefig(qc/'registered-source-slices.png', dpi=140, facecolor=figure.get_facecolor())
    plt.close(figure)
    path = root/'simulator/patient.ressectionlab'
    case, artifacts, saved = load_workspace(path)
    scene = load_environment(path)
    engine = SurgicalEngine.restore(scene, artifacts['surgical_toolbox']['branches']['main'])
    if engine.guard_report()['structures'][0]['sourceCells'] == 0 or any(r.get('target') for r in scene.regions):
        raise AssertionError('Healthy source/guard contract was not preserved')
    try:
        SurgicalRLEnv.from_patient(path)
    except ValueError as error:
        if 'No unprotected explicitly designated tumor' not in str(error):
            raise
        rl_refusal = str(error)
    else:
        raise AssertionError('No-lesion case was admitted to tumor-removal RL')
    session = BridgeSession(qc/'renderer-transfers')
    call = lambda op, args: session.execute(op, args, _Request(op), lambda *_: None)
    bridge = {}
    try:
        binding = call('anatomyOpen', {'path': str(c.root/'patient-anatomy.json')})
        for sid in ('t1', 'mra', 'dwi-b0', 'dwi-fa'):
            result = call('anatomyView', {'collectionId': binding['collectionId'], 'snapshotId': c.identity(),
                'seriesId': sid, 'layers': ['brain', 'vessels'], 'previewAlignment': True, 'vesselScope': 'all'})
            if result['unavailableLayers'] or len(result['compartments']) != 2:
                raise AssertionError('Missing same-subject anatomy in renderer bridge')
            bridge[sid] = {'shape': result['shape'], 'layers': [r['name'] for r in result['compartments']],
                           'registration_status': result['registrationStatus'], 'image_transfer_bytes': result['mri']['byteLength']}
        reopened = call('loadCase', {'path': str(path)})
        bridge['simulator_case_hash'] = reopened['caseHash']
    finally:
        session.close()
    if PatientImagingCollection(c.root).identity() != initial:
        raise AssertionError('Read-only QC changed source collection')
    report = {'subject': c.document['patient_id'], 'collection_identity': initial, 'headers': headers,
        'source_diffusion_volumes': len(raw), 'source_roles_preserved': c.document['role'], 'training': False,
        'registration_estimates': c.document['transforms'],
        'anatomy_quality': {r['id']: layer_quality(c, r) for r in c.document['layers']},
        'source_slice_montage': 'registered-source-slices.png', 'renderer_bridge': bridge,
        'simulator': {'scene_id': scene.identity, 'target_cells': 0, 'guard': engine.guard_report(),
                      'originals_archived': len(saved['originals']), 'native_source_views': len(saved['sources']),
                      'rl_removal_refusal': rl_refusal},
        'limitations': c.document['missing'], 'anatomical_accuracy': 'unmeasured; no independent expert or landmark ground truth',
        'checked_by': 'Source byte/header checks, independent slice resampling, canonical engine reload, actual renderer bridge; no subagents'}
    (qc/'verification.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'status': 'pass_software_checks_anatomical_review_pending', 'report': str(qc/'verification.json')}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('testcases/environments/IXI265-multimodal'))
    verify(parser.parse_args().root)
