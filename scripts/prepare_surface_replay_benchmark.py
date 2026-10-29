#!/usr/bin/env python3
"""Small source-owned renderer fixtures from an existing executed recording.

No MRI duplicates: only compressed categorical display masks plus exact source
cell deltas. The first four exposure actions are replayed and the saved final
removal set retained. No synthetic resection path or learning experience.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from resectionlab.core import array_digest
from resectionlab.surgical_engine import SurgicalEngine
from resectionlab.surgical_environment import standardize
from resectionlab.surgical_scene import from_case
from resectionlab.workspace_bundle import load_workspace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--branch', default='sustained-hold')
    args = parser.parse_args()
    if not args.output.resolve().is_relative_to(ROOT / 'testcases'):
        parser.error('Generated app-testing fixtures belong under ignored testcases/')
    args.output.mkdir(parents=True, exist_ok=False)
    case, artifacts, _ = load_workspace(args.case)
    source = artifacts['surgical_toolbox']
    saved = source['branches'][args.branch]
    scene = standardize(from_case(case, source['mapping']), physics_profile=source['physicsProfile'])
    if scene.identity != saved['sceneId']:
        raise ValueError('Source frame or declaration differs from saved episode')
    engine = SurgicalEngine(scene)
    states = [{'position': 0, 'stateId': engine.state_id, 'removed': []}]
    for action in saved['actions'][:4]:
        receipt = engine.step(action)
        if receipt['status'] == 'rejected':
            raise ValueError(receipt)
        states.append({'position': len(engine.actions), 'stateId': engine.state_id, 'removed': engine.state['removed'].copy()})
    states.append({'position': len(saved['actions']), 'stateId': saved['stateId'], 'removed': saved['checkpoint']['state']['removed']})
    layers = []
    for material, kind in (('soft_tissue', 'brain'), ('tumor', 'tumor')):
        labels = [row['id'] for row in scene.regions if row['material'] == material]
        mask = np.isin(scene.labels, labels).astype(np.uint8)
        filename = kind + '.mask.gz'
        (args.output / filename).write_bytes(gzip.compress(mask.tobytes(), mtime=0))
        previous = set(); frames = []
        for state in states:
            removed = {cell for cell in state['removed'] if mask.flat[cell]}
            if not previous <= removed:
                raise ValueError('This forward benchmark requires monotone source removal')
            frames.append({key: state[key] for key in ('position', 'stateId')} | {
                'removedCells': sorted(removed - previous), 'totalRemovedCells': len(removed),
                'removedVolumeMm3': len(removed) * scene.voxel_mm3})
            previous = removed
        layers.append({'kind': kind, 'maskFile': filename, 'maskSha256': hashlib.sha256(mask.tobytes()).hexdigest(),
            'occupiedSourceCells': int(mask.sum()), 'frames': frames})
    report = {'schema': 'resectionlab.surface-replay-fixture.v1', 'source': str(args.case.resolve()),
        'sourceSha256': hashlib.sha256(args.case.read_bytes()).hexdigest(), 'sceneId': scene.identity,
        'mriHash': array_digest(case.mri), 'labelsHash': array_digest(scene.labels), 'branch': args.branch,
        'shape': list(scene.shape), 'affine': scene.affine.tolist(), 'layers': layers,
        'scope': 'Source display surfaces at four actual exposure actions and the saved final endpoint; does not benchmark MRI or engine physics.'}
    (args.output / 'fixture.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'output': str(args.output), 'layers': [(row['kind'], len(row['frames'])) for row in layers]}))


if __name__ == '__main__':
    main()
