#!/usr/bin/env python3
"""Acquire one declared source-manual mask and compare its untouched native grid.

No registration, label generation, clinical approval or learning admission.
The scientific download reuses the source-size/version/annex verifier. A hard
deadline preserves failed/partial evidence; it never retries automatically.
"""
from __future__ import annotations

import argparse
import hashlib
from itertools import product
import json
from pathlib import Path
import pickletools
import shutil
import signal
import sys
import time

import nibabel as nib
import numpy as np

from acquire_btc_case import acquire_file
from real_intake_io import atomic_preserve, verify_source_file
import lausanne_train_intake as intake

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT/'manifests/lausanne-sub476-manual-annotation-v1.json'
MANIFEST_SHA = '39537b60333a22e8e8e81f792ac5ba9dad306ef72ff3a4a4f768f6b7ab59230f'
OUTPUT = intake.DATA/'manual-annotation-sub476-v1'


def encoded(value):
    return (json.dumps(value, indent=2, allow_nan=False)+'\n').encode()


def verify_inputs():
    payload = MANIFEST.read_bytes()
    if hashlib.sha256(payload).hexdigest() != MANIFEST_SHA:
        raise ValueError('Frozen annotation declaration changed')
    m = json.loads(payload)
    for entry in [m['cohort'], m['index'], m['original_acquisition'], *m['source_metadata'], *m['rights']]:
        verify_source_file(ROOT/entry['path'], entry)
    row = next(r for r in json.loads((ROOT/m['index']['path']).read_bytes())['sessions']
               if (r['subject'], r['session']) == (m['subject'], m['session']))
    saved = json.loads((ROOT/m['original_acquisition']['path']).read_bytes())
    intake.validate_receipt(row, saved, m['index']['sha256'])
    strong, sidecar, pointer = [(ROOT/e['path']).read_bytes() for e in m['source_metadata']]
    operations = list(pickletools.genops(strong))
    allowed = {'APPENDS', 'BINPUT', 'BINUNICODE', 'EMPTY_LIST', 'MARK', 'PROTO', 'STOP'}
    if not {op.name for op, _, _ in operations} <= allowed:
        raise ValueError('Crosswalk must contain only the inspected string-list opcodes')
    members = [value for op, value, _ in operations if op.name == 'BINUNICODE']
    if len(members) != 38 or f"{m['subject']}_{m['session']}" not in members:
        raise ValueError('Exact source strong-label membership missing')
    if json.loads(sidecar) != {'Type': 'Lesion', 'RawSources': Path(m['original_tof']['path']).name, 'Space': 'orig'}:
        raise ValueError('Source annotation linkage differs')
    if not pointer.decode().endswith('MD5E-s82420--4b8717630500d48edb362542a283a07b.nii.gz'):
        raise ValueError('Source annex declaration differs')
    if json.loads((ROOT/m['rights'][0]['path']).read_bytes())['License'] != 'CC0':
        raise ValueError('Pinned dataset rights differ')
    return m


def grid(image):
    affine = image.affine
    if len(image.shape) != 3 or not np.isfinite(affine).all():
        raise ValueError('Finite three-dimensional source grid required')
    qform, qcode = image.get_qform(coded=True)
    sform, scode = image.get_sform(coded=True)
    corners = np.array([(*p, 1) for p in product(*[(0, n-1) for n in image.shape])])
    disagreement = (float(np.linalg.norm(((qform-sform)@corners.T)[:3], axis=0).max())
                    if qcode and scode else None)
    return {'shape': list(image.shape), 'affine_ras': affine.tolist(),
            'spatial_units': image.header.get_xyzt_units()[0],
            'qform_code': int(qcode), 'sform_code': int(scode),
            'maximum_qform_sform_corner_disagreement_mm': disagreement}


def execute():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT/'attempt-started.json').open('xb') as stream:
        stream.write(encoded({'manifest_sha256': MANIFEST_SHA, 'automatic_retries': 0}))
    started = time.monotonic()
    result = {'status': 'failed_or_incomplete', 'manifest_sha256': MANIFEST_SHA,
              'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'scanner_frame_admitted': False, 'spatial_planning_admitted': False,
              'annotation_available_at': None, 'source_review_available_at': None,
              'component_optimizer_updates': 0, 'recorded_rl_transitions': 0}
    def expired(*_):
        raise TimeoutError('Declared 90-second annotation intake deadline')
    previous = signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, 90)
    try:
        m = verify_inputs()
        if shutil.disk_usage(OUTPUT).free < 2*m['file']['bytes']+1024**2:
            raise ValueError('Insufficient space for original and retained partial')
        sources = {}
        for rel in [str(Path(__file__).relative_to(ROOT)), *intake.SOURCE_NAMES]:
            payload = (ROOT/rel).read_bytes()
            sources[rel] = hashlib.sha256(payload).hexdigest()
            atomic_preserve(OUTPUT/'source-snapshot'/rel, payload)
        result['source_sha256'] = sources
        result['acquisition'] = acquire_file(m['file'], intake.DATA)
        path = intake.DATA/m['file']['path']
        result['mask_sha256'] = verify_source_file(path, m['file'])
        result['mask_bytes'] = path.stat().st_size
        image = nib.load(path)
        if len(image.shape) != 3 or np.prod(image.shape) > m['bounds']['max_mask_voxels']:
            raise ValueError('Mask dimensions exceed declared decoding limit')
        if np.prod(image.shape)*8 > m['bounds']['max_decoded_mask_bytes']:
            raise ValueError('Mask decoding exceeds declared memory size')
        values = image.get_fdata(dtype=np.float32)
        labels = np.unique(values)
        if not np.isfinite(values).all() or not np.array_equal(labels, [0, 1]):
            raise ValueError('Expected nonempty binary source annotation')
        original = nib.load(intake.DATA/m['original_tof']['path'])
        result.update(mask_grid=grid(image), original_tof_grid=grid(original),
                      labels=labels.tolist(), positive_voxels=int(np.count_nonzero(values)),
                      decoded_bytes=values.nbytes, original_tof_sha256=m['original_tof']['sha256'])
        same = (image.shape == original.shape and np.array_equal(image.affine, original.affine)
                and image.header.get_xyzt_units()[0] == original.header.get_xyzt_units()[0] == 'mm')
        result['exact_native_grid_match'] = bool(same)
        result['maximum_affine_coefficient_difference'] = float(np.max(np.abs(image.affine-original.affine)))
        result['status'] = 'binary_mask_integrity_passed_exact_grid_match' if same else 'binary_mask_integrity_passed_grid_unresolved'
        result['coverage_semantics'] = 'positive aneurysm support only; background unknown; no whole-vessel or negative-domain assertion'
        verify_inputs()
        for rel, expected in sources.items():
            if hashlib.sha256((ROOT/rel).read_bytes()).hexdigest() != expected:
                raise ValueError('Executing source changed')
    except BaseException as error:
        result.update(status='failed_or_incomplete', error={'type': type(error).__name__, 'message': str(error)})
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
        result['elapsed_seconds'] = time.monotonic()-started
        if result['elapsed_seconds'] >= 90:
            result['status'] = 'failed_or_incomplete'
        atomic_preserve(OUTPUT/'acquisition.json', encoded(result))
    print(json.dumps({key: result.get(key) for key in ('status', 'elapsed_seconds', 'mask_sha256', 'positive_voxels', 'exact_native_grid_match', 'error')}))
    return result['status'] != 'failed_or_incomplete'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if args.execute:
        raise SystemExit(0 if execute() else 1)
    verify_inputs()
    print('Pinned source metadata, rights, TRAIN role and original bytes verified; no mask downloaded.')
