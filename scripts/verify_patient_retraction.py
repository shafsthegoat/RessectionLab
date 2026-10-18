#!/usr/bin/env python3
"""Verify public-action retraction on an already exposed patient brain estimate.

No source masks, removal state, holds, or tool poses are injected. A bounded
six-axis search identifies exposed estimates, then public actions exchange,
place, hold, move outward and release. This is a software check, not access QC.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
from resectionlab.core import array_digest
from resectionlab.imaging import load_case
from resectionlab.surgical_scene import from_case
from resectionlab.surgical_engine import SurgicalEngine
from verify_patient_vessels import candidates


def verify(case_path, output):
    started = time.perf_counter()
    file_hash = hashlib.sha256(Path(case_path).read_bytes()).hexdigest()
    case = load_case(case_path)
    scene = from_case(case)
    brain = next(r for r in scene.regions if r['material'] == 'soft_tissue')
    labels_hash, mri_hash = array_digest(scene.labels), array_digest(case.mri)
    attempts = []
    for axis, sign, index, _, tip, inward, quaternion in candidates(scene, brain['id'])[:80]:
        engine = SurgicalEngine(scene)
        records = []
        start = tip - 5 * inward
        def perform(operation, position=None, **extra):
            command = {'slot': 'left', 'operation': operation, 'duration': .2, **extra}
            if position is not None:
                command['pose'] = {'position': position.tolist(), 'quaternion': quaternion}
            result = engine.step(command)
            records.append({'action': command, 'status': result['status'], 'reason': result.get('reason'),
                            'stateId': result['stateId'], 'completedFraction': result['completedFraction'],
                            'events': result['events']})
            return result['status'] == 'accepted'
        print('Trying exposed brain estimate ' + str(index.tolist()), file=sys.stderr, flush=True)
        if not (perform('exchange', instrument='brain_retractor')
                and perform('place', start, parameters={'radiusMm': .6})
                and perform('hold', tip)):
            attempts.append({'index': index.tolist(), 'stage': 'hold', 'reason': records[-1]['reason']})
            continue
        hold = engine.state['holds'].get('left')
        if not hold or not hold['cells']:
            attempts.append({'index': index.tolist(), 'stage': 'hold', 'reason': 'No retained cells acquired'})
            continue
        held_cells = hold['cells'][:]
        assert all(scene.labels.ravel()[cell] == brain['id'] for cell in held_cells)
        if not perform('move', tip - 2 * inward):
            attempts.append({'index': index.tolist(), 'stage': 'outward move', 'reason': records[-1]['reason']})
            continue
        moved = engine.view()['moved']
        assert moved and engine.state['holds']['left']['offset'] != [0, 0, 0]
        occupancy = engine.occupancy()
        for source, destination in moved:
            assert occupancy[destination] == source + 1
            point = scene.points([destination])[0]
            assert source in engine.touched(point, point, .05)
        moved_state = engine.state_id
        moved_snapshot = json.loads(json.dumps(engine.snapshot()))
        moved_replay = SurgicalEngine.restore(scene, moved_snapshot)
        assert moved_replay.state_id == moved_state and moved_replay.view()['moved'] == moved
        if not perform('release'):
            attempts.append({'index': index.tolist(), 'stage': 'release', 'reason': records[-1]['reason']})
            continue
        assert not engine.state['holds'] and not engine.view()['moved'] and not engine.state['removed']
        occupancy = engine.occupancy()
        for source in held_cells:
            assert occupancy[source] == source + 1
            point = scene.points([source])[0]
            assert source in engine.touched(point, point, .05)
        destinations = {destination for _, destination in moved}
        for destination in destinations - set(held_cells):
            assert occupancy[destination] == (destination + 1 if scene.labels.ravel()[destination] else 0)
        restored = SurgicalEngine.restore(scene, json.loads(json.dumps(engine.snapshot())))
        assert restored.state_id == engine.state_id
        assert array_digest(scene.labels) == labels_hash and array_digest(case.mri) == mri_hash
        assert hashlib.sha256(Path(case_path).read_bytes()).hexdigest() == file_hash
        report = {'schema': 'resectionlab.patient-retraction-verification.v1', 'status': 'passed',
            'sourceCase': str(Path(case_path).resolve()), 'caseHash': case.semantic_hash,
            'sourceFileSha256': file_hash, 'sceneId': scene.identity, 'labelArrayHash': labels_hash,
            'intensityArrayHash': mri_hash, 'brainRegion': brain, 'sourceUnchanged': True,
            'preRemovedCells': 0, 'estimatedExposedBrainIndexIjk': index.tolist(),
            'actions': records, 'failedAttempts': attempts, 'heldCells': held_cells, 'moved': moved,
            'movedStateId': moved_state, 'movedCollisionGeometryVerified': True,
            'releaseRestoredGeometry': True, 'movedReplayVerified': True, 'releaseReplayVerified': True,
            'stateId': engine.state_id, 'elapsedSeconds': time.perf_counter() - started,
            'limitations': ['Existing unreviewed brain segmentation estimate; no anatomical accuracy claim.',
                'Already exposed surface test; no surgical exposure or operative access was fabricated.',
                'Rigid lattice patch translation and exact release; not calibrated continuum tissue mechanics.']}
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(json.dumps(report, indent=2) + '\n')
        return report
    report = {'schema': 'resectionlab.patient-retraction-verification.v1', 'status': 'not_demonstrated',
              'caseHash': case.semantic_hash, 'attempts': attempts,
              'reason': 'No bounded exposed brain-estimate candidate completed public hold/move/release.',
              'sourceUnchanged': hashlib.sha256(Path(case_path).read_bytes()).hexdigest() == file_hash,
              'elapsedSeconds': time.perf_counter() - started}
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(report, indent=2) + '\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('case')
    parser.add_argument('output')
    args = parser.parse_args()
    result = verify(args.case, args.output)
    print(json.dumps({k: result[k] for k in ('status', 'elapsedSeconds')}))
    sys.exit(0 if result['status'] == 'passed' else 1)
