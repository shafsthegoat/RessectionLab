#!/usr/bin/env python3
"""Freeze exact source arrays and two public-action views for CPU render timing."""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path
import sys
import zlib

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

import numpy as np

from resectionlab.imaging import load_case
from resectionlab.surgical_engine import SurgicalEngine
from resectionlab.surgical_environment import load_environment
from resectionlab.surgical_scene import SurgicalScene


def packed(array, dtype):
    array = np.asarray(array, dtype=dtype)
    raw = array.tobytes(order='C')
    return {'dtype': dtype, 'length': array.size, 'sha256': hashlib.sha256(raw).hexdigest(),
            'encoding': 'zlib-base64', 'data': base64.b64encode(zlib.compress(raw)).decode('ascii')}


def entry(identifier, scene, mri, masks, actions, source_file=None):
    engine = SurgicalEngine(scene)
    intact = engine.view()
    outcomes = [engine.step(action) for action in actions]
    if any(outcome['status'] == 'rejected' for outcome in outcomes):
        raise ValueError('Render fixture action was rejected')
    source = {'caseHash': scene.identity, 'shape': list(scene.shape), 'affine': scene.affine.tolist(),
              'frame': 'RAS+', 'mri': packed(mri, '<f4'),
              'compartments': [{'name': region['name'], 'color': region.get('color', '#9bacb8'),
                                'kind': region['material'], 'mask': packed(masks[region['name']], '|u1')}
                               for region in scene.regions]}
    return {'id': identifier, 'sourceFile': source_file, 'sourceProvenance': scene.source,
            'source': source, 'intactView': intact, 'updatedView': engine.view(),
            'actions': actions, 'outcomes': [{key: outcome.get(key) for key in ('status', 'reason', 'stateId')} for outcome in outcomes]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', type=Path, default=Path('outputs/surgical-patient-v2/patient.ressectionlab'))
    parser.add_argument('--actions', type=Path, default=Path('outputs/surgical-patient-v2/drill-verification.json'))
    parser.add_argument('--output', type=Path, default=Path('outputs/integrated-rehearsal-v1/render-fixture.json'))
    args = parser.parse_args()
    labels = np.zeros((24, 24, 24), np.uint16)
    labels[8:10, 8:10, 8] = 1
    small = SurgicalScene(labels, np.eye(4), [{'id': 1, 'name': 'coupon', 'material': 'tumor', 'target': True}],
                          {'provenance': 'generated engineering coupon; not patient anatomy'}, {})
    actions = [dict(slot='right', operation='place', duration=.2, parameters={'radiusMm': .6},
                    pose={'position': [8., 8., 4.], 'quaternion': [0., 0., 0., 1.]}),
               dict(slot='right', operation='apply', duration=.2,
                    pose={'position': [8., 8., 8.], 'quaternion': [0., 0., 0., 1.]})]
    entries = [entry('generated-coupon', small, np.zeros(small.shape, np.float32), {'coupon': labels == 1}, actions)]
    case = load_case(args.case)
    scene = load_environment(args.case)
    source_hash = hashlib.sha256(args.case.read_bytes()).hexdigest()
    patient_actions = [row['action'] for row in json.loads(args.actions.read_text())['actions'][:2]]
    entries.append(entry('acquired-sub06', scene, case.mri, case.compartments, patient_actions,
                         {'path': str(args.case.resolve()), 'sha256': source_hash}))
    if hashlib.sha256(args.case.read_bytes()).hexdigest() != source_hash:
        raise AssertionError('Source file changed')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({'schema': 'resectionlab.surgical-render-fixture.v1', 'cases': entries}, allow_nan=False))
    print(json.dumps({'fixture': str(args.output.resolve()), 'sha256': hashlib.sha256(args.output.read_bytes()).hexdigest(),
                      'bytes': args.output.stat().st_size, 'sourceUnchanged': True}, indent=2))


if __name__ == '__main__':
    main()
