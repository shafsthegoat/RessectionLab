#!/usr/bin/env python3
"""Exercise actual whole-head patient cells through public surgical transitions.

This is a deterministic software smoke test, not a route recommendation. It
searches straight voxel-axis access rays and never pre-removes anatomy. Run with
PYTHONPATH=src .venv/bin/python scripts/verify_surgical_patient.py CASE OUTPUT_DIR.
"""
from pathlib import Path
import argparse
import json
import resource
import sys
import time
import numpy as np
from scipy.spatial.transform import Rotation
from resectionlab.imaging import load_case, save_case
from resectionlab.surgical_scene import from_case
from resectionlab.surgical_engine import SurgicalEngine
from resectionlab.core import array_digest


def action(operation, position=None, quaternion=None, **kw):
    result = dict(slot='left', operation=operation, duration=.2)
    result.update(kw)
    if position is not None: result['pose'] = dict(position=list(map(float, position)), quaternion=quaternion)
    return result


def verify(case_path, output):
    started = time.perf_counter(); case = load_case(case_path); scene = from_case(case)
    source_hash = array_digest(scene.labels); intensity_hash = array_digest(case.mri)
    material = {r['material']: r['id'] for r in scene.regions}
    if not {'bone', 'soft_tissue', 'tumor', 'vessel'} <= set(material): raise ValueError('All four materials required')
    labels = scene.labels; tumor = np.argwhere(labels == material['tumor'])
    report = None
    # Search the six straight entry directions, closest surface first. No
    # clinical objective is optimized and no invisible exposure is introduced.
    for axis in (2, 0, 1):
        other = [k for k in range(3) if k != axis]
        columns = np.unique(tumor[:, other], axis=0)
        center = np.median(columns, axis=0)
        columns = columns[np.argsort(np.linalg.norm(columns-center, axis=1))]
        for sign in (-1, 1):
            inward = scene.affine[:3, axis] * sign / scene.spacing[axis]
            x = scene.affine[:3, other[0]] / scene.spacing[other[0]]
            y = np.cross(inward, x); y /= np.linalg.norm(y); x = np.cross(y, inward)
            q = Rotation.from_matrix(np.column_stack((x, y, inward))).as_quat().tolist()
            for col in columns[::max(1, len(columns)//30)]:
                index = [slice(None)] * 3
                for k, value in zip(other, col): index[k] = int(value)
                ray = labels[tuple(index)]
                targets = np.flatnonzero(ray == material['tumor'])
                if not len(targets): continue
                target_index = int(targets[-1] if sign == -1 else targets[0])
                upstream = ray[target_index:] if sign == -1 else ray[:target_index+1]
                if material['bone'] not in upstream or material['vessel'] in upstream: continue
                ijk = np.zeros(3); ijk[other] = col
                ijk[axis] = labels.shape[axis]+4 if sign == -1 else -5
                start = scene.affine[:3,:3] @ ijk + scene.affine[:3,3]
                ijk[axis] = target_index + sign * 1
                end = scene.affine[:3,:3] @ ijk + scene.affine[:3,3]
                engine = SurgicalEngine(scene); records=[]
                print(f'Trying access axis={axis} sign={sign} column={col.tolist()}', file=sys.stderr, flush=True)
                steps = [action('place',start,q,parameters={'radiusMm':3}), action('apply',end,q,duration=1),
                         action('withdraw'), action('exchange',instrument='cusa'),
                         action('place',start,q,parameters={'radiusMm':1}), action('apply',end,q,duration=1)]
                for step in steps:
                    before = engine.state_id; t = time.perf_counter()
                    preview = engine.preview(step)
                    if engine.state_id != before: raise AssertionError('Preview mutated source state')
                    result = engine.step(step)
                    if result['stateId'] != preview['stateId']: raise AssertionError('Preview and commit differ')
                    records.append(dict(action=step, result=result, previewAndStepMs=1000*(time.perf_counter()-t)))
                    print(step['operation'], result['status'], result.get('reason'), result['metrics']['targetRemovedMm3'], file=sys.stderr, flush=True)
                    if result['status'] == 'rejected': break
                if len(records) != len(steps) or records[-1]['result']['status'] == 'rejected': continue
                metrics = engine.metrics()
                if metrics['targetRemovedMm3'] <= 0 or metrics['removedByLabelMm3'].get('bone',0) <= 0: continue
                if metrics['removedByLabelMm3'].get('vessels',0): raise AssertionError('Protected vessel removed')
                assert array_digest(scene.labels) == source_hash and array_digest(case.mri) == intensity_hash
                snapshot = engine.snapshot()
                restored = SurgicalEngine.restore(scene, json.loads(json.dumps(snapshot)))
                assert restored.state_id == engine.state_id
                branch = engine.branch(2)
                assert len(branch.actions) == 2 and branch.state_id != engine.state_id
                output = Path(output); output.mkdir(parents=True,exist_ok=True)
                saved = save_case(case, output/'patient-tool-demo.ressectionlab', artifacts={'surgical_toolbox':{
                    'mapping': scene.regions, 'branches': {'main': snapshot}, 'branchId':'main',
                    'position':len(engine.actions), 'parents':{'main':None}}})
                report = dict(schema='resectionlab.patient-tool-verification.v1', sourceCase=str(Path(case_path).resolve()),
                    demoCase=str(saved), sourceIdentity=scene.identity, sourceUnchanged=True, intensityUnchanged=True,
                    explicitInitialExposure=False, status='passed', actions=records, metrics=metrics,
                    stateId=engine.state_id, replayVerified=True, branchVerified=True,
                    elapsedSeconds=time.perf_counter()-started, processPeakRssBytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                    limitations=['Software access-ray smoke test, not clinical route planning',
                                 'Unreviewed anatomy and CT registration estimates',
                                 'No vascular flow network or patient-calibrated force model'])
                (output/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
                (output/'snapshot.json').write_text(json.dumps(snapshot,indent=2)+'\n')
                return report
    raise RuntimeError('No straight checked entry reached tumor; anatomy was not modified to force a demonstration')

if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('case'); parser.add_argument('output')
    args=parser.parse_args(); result=verify(args.case,args.output)
    print(json.dumps({k:result[k] for k in ('status','demoCase','metrics','elapsedSeconds','processPeakRssBytes')},indent=2))
