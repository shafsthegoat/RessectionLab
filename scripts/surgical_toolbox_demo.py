#!/usr/bin/env python3
"""Run all fifteen mechanisms on declared engineering coupons, with replay checks.

Run from the repository: .venv/bin/python scripts/surgical_toolbox_demo.py
No patient data, measured force law, or clinical response is inferred.
"""
from __future__ import annotations
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import numpy as np
from resectionlab.surgical_scene import SurgicalScene
from resectionlab.surgical_engine import SurgicalEngine
from resectionlab.surgical_tools import TOOLS


def coupon(tool):
    definition = TOOLS[tool]
    material = definition['materials'][0] if definition['materials'] else 'soft_tissue'
    labels = np.zeros((20, 20, 20), np.uint16)
    labels[7:10, 7:10, 8] = 1
    ids = list(map(int, np.flatnonzero(labels)))
    properties = {'attachments': [], 'strands': [{'id': 'engineering-strand', 'cells': ids}],
                  'bleedingSources': [], 'functionalEvidence': [], 'boneAnchors': ids[:1]}
    if definition['effect'] in ('elevate', 'dissect'):
        properties['attachments'] = [{'id': 'engineering-bond', 'kind': 'periosteal' if tool == 'periosteal_elevator' else 'soft', 'cells': ids}]
    if definition['effect'] in ('suction', 'coagulate'):
        properties['bleedingSources'] = [{'id': 'engineering-fluid-outlet', 'cell': ids[4],
            'rateMm3PerSecond': 2., 'coagulationExposure': .5, 'patent': True}]
    if definition['effect'] == 'probe':
        properties['functionalEvidence'] = [{'id': 'engineering-evidence', 'cell': ids[4],
            'probe': tool.removesuffix('_probe'), 'result': 'synthetic test response',
            'uncertainty': 'arbitrary fixture evidence; not physiology'}]
    return SurgicalScene(labels, np.eye(4), [{'id': 1, 'name': 'Engineering coupon', 'material': material}],
                         {'provenance': 'generated mechanism demonstration; not patient anatomy'}, properties)


def run_mixed_sequence():
    """One shared scene: removal, temporary displacement, release, and fluid."""
    labels = np.zeros((40, 40, 30), np.uint16)
    labels[8, 8, 15], labels[25, 8, 15], labels[8, 25, 15] = 1, 2, 3
    scene = SurgicalScene(labels, np.eye(4), [
        {'id': 1, 'name': 'Bone coupon', 'material': 'bone'},
        {'id': 2, 'name': 'Target coupon', 'material': 'tumor', 'target': True},
        {'id': 3, 'name': 'Retained coupon', 'material': 'soft_tissue'}],
        {'provenance': 'generated mixed-tool engineering fixture'}, {'attachments': []})
    engine = SurgicalEngine(scene)
    def action(slot, operation, position=None, **kwargs):
        result = {'slot': slot, 'operation': operation, 'duration': .1, **kwargs}
        if position is not None:
            result['pose'] = {'position': position, 'quaternion': [0., 0., 0., 1.]}
        return result
    actions = [
        action('left', 'place', [8, 8, 10], parameters={'radiusMm': .6}),
        action('left', 'apply', [8, 8, 15]), action('left', 'withdraw'),
        action('left', 'exchange', instrument='brain_retractor'),
        action('left', 'place', [8, 25, 10], parameters={'radiusMm': .6}),
        action('left', 'hold', [8, 25, 15]), action('left', 'move', [10, 25, 15]),
        action('right', 'place', [25, 8, 10], parameters={'radiusMm': .6}),
        action('right', 'apply', [25, 8, 15]), action('left', 'release'),
        action('left', 'withdraw'), action('left', 'exchange', instrument='irrigation'),
        action('left', 'place', [18, 25, 10], parameters={'radiusMm': 1., 'depthMm': 2., 'flowMm3PerSecond': 20.}),
        action('left', 'apply'), action('left', 'withdraw'),
        action('left', 'exchange', instrument='frazier_suction'),
        action('left', 'place', [18, 25, 5], parameters={'radiusMm': 3., 'flowMm3PerSecond': 20.}),
        action('left', 'apply', [18, 25, 10], duration=.5),
    ]
    outcomes = []; displaced = []
    for index, step in enumerate(actions):
        result = engine.step(step)
        if result['status'] != 'accepted': raise RuntimeError(f'mixed sequence action {index}: {result}')
        if index == 6: displaced = engine.view()['moved']
        outcomes.append({'action': step, 'status': result['status'], 'stateId': result['stateId'],
                         'events': sorted({event['kind'] for event in result['events']})})
    snapshot = json.loads(json.dumps(engine.snapshot()))
    restored = SurgicalEngine.restore(scene, snapshot)
    if restored.state_id != engine.state_id: raise RuntimeError('mixed replay differed')
    metrics = engine.metrics()
    balance = metrics['fluidAddedMm3'] - metrics['fluidPresentMm3'] - metrics['fluidRemovedMm3'] - metrics['fluidEscapedMm3']
    if len(engine.state['removed']) != 2 or not displaced or engine.state['holds']:
        raise RuntimeError('mixed removal/displacement/release assertion failed')
    if metrics['fluidRemovedMm3'] <= 0 or abs(balance) > 1e-8:
        raise RuntimeError('mixed irrigation/suction conservation assertion failed')
    return {'provenance': scene.source, 'actions': outcomes, 'observedDisplacement': displaced,
            'replayVerified': True, 'checkpointVerified': 'checkpoint' in snapshot,
            'stateId': engine.state_id, 'metrics': metrics, 'fluidBalanceErrorMm3': balance}


def run_demo():
    reports = []
    for tool in TOOLS:
        scene = coupon(tool); engine = SurgicalEngine(scene)
        pose = lambda z: {'position': [8., 8., z], 'quaternion': [0., 0., 0., 1.]}
        actions = [
            {'slot': 'left', 'operation': 'exchange', 'instrument': tool},
            {'slot': 'left', 'operation': 'place', 'pose': pose(4.),
             'parameters': {'radiusMm': 1.5, 'depthMm': 2., 'jawOpeningMm': 0.}},
            {'slot': 'left', 'operation': 'apply', 'pose': pose(6.8 if tool in ('frazier_suction', 'irrigation') else 7.), 'duration': 1.},
        ]
        results = []
        for action in actions:
            result = engine.step(action)
            if result['status'] != 'accepted': raise RuntimeError(f'{tool}: {result}')
            results.append(result)
        events = [event['kind'] for event in results[-1]['events'] if event['kind'] != 'apply']
        if events == ['no_material_effect']: raise RuntimeError(f'{tool} had no effect')
        restored = SurgicalEngine.restore(scene, engine.snapshot())
        if restored.state_id != engine.state_id: raise RuntimeError(f'{tool} replay differed')
        metrics = engine.metrics()
        fluid_balance = metrics['fluidAddedMm3'] - metrics['fluidPresentMm3'] - metrics['fluidRemovedMm3'] - metrics['fluidEscapedMm3']
        if abs(fluid_balance) > 1e-8: raise RuntimeError(f'{tool} fluid balance {fluid_balance}')
        reports.append({'instrument': tool, 'events': sorted(set(events)), 'stateId': engine.state_id,
                        'replayVerified': True, 'checkpointVerified': 'checkpoint' in engine.snapshot(), 'fluidBalanceErrorMm3': fluid_balance, 'metrics': metrics})
    return {'provenance': 'Fifteen isolated generated engineering coupons; synthetic declared properties.',
            'limitations': ['Uncalibrated mechanics and energy', 'Lattice displacement and cell cuts',
                            'Supplied mapping observations only; no inferred functional anatomy',
                            'Retained mesh seam graph; no remeshed physical aperture'], 'results': reports,
            'mixedSequence': run_mixed_sequence()}


if __name__ == '__main__':
    print(json.dumps(run_demo(), indent=2))
