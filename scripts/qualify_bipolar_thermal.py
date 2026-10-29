#!/usr/bin/env python3
"""Run native bipolar programs on a generated source-owned thermal coupon."""
from dataclasses import replace
from pathlib import Path
import argparse
import copy
import json
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from qualify_vascular_graph import make_case as vascular_case
from resectionlab.core import SourceRef, array_digest, thaw_json
from resectionlab.imaging import save_case
from resectionlab.surgical_scene import from_case
from resectionlab.surgical_environment import standardize, SCHEMA as ENVIRONMENT_SCHEMA
from resectionlab.surgical_engine import SurgicalEngine
from resectionlab.simulator_qualification import removed_accounting, file_sha256
from resectionlab import surgical_bipolar_thermal as heat
from resectionlab import surgical_vascular_graph as graph

PROTOCOL = ROOT / 'manifests/experiments/bipolar-thermal-v1.json'


def make_case():
    original, _ = vascular_case()
    metadata = thaw_json(original.metadata)
    compartments = {name: values.copy() for name, values in original.compartments.items()}
    patch = np.zeros(original.mri.shape, bool); patch[3:8, 5:8, 12] = True
    for name, values in compartments.items():
        if name != 'Generated brain': patch &= ~values
    compartments['Generated brain'] |= patch
    material = {'volumetricHeatCapacityJPerMm3K': .00383, 'conductivityWPerMmK': .000553,
                'source': 'Mohammadi2021 Table3 measured calf brain33C; uniform patch scenario, including vessel labels, is an assumption'}
    cells = np.zeros(original.mri.shape, bool); cells[3:8, 5:8, 12] = True
    metadata['surgical_properties'][heat.PROPERTY] = heat.research_profile(
        list(map(int, np.flatnonzero(cells))), {'1': dict(material), '3': dict(material)},
        ambient_c=33., watts_per_activation=.02, maximum_c=60.)
    metadata['description'] = 'Generated post-exposure thermal/vascular coupon. No acquired anatomy or clinical calibration.'
    case = replace(original, case_id='generated-bipolar-thermal-v1', compartments=compartments,
        source_refs=(SourceRef(source_id='bipolar-thermal-coupon-v1', uri='synthetic://resectionlab/bipolar-thermal-v1',
            license='CC0-1.0', native_frame='RAS+', provenance='simulated'),), metadata=metadata)
    # Generated display intensity is distinct from physical temperature.
    mapping = [{'id': i, 'name': name, 'material': material_name, 'target': i == 2} for i, name, material_name in [
        (1, 'Generated brain', 'soft_tissue'), (2, 'Generated target', 'tumor'),
        (3, 'Generated artery', 'vessel'), (4, 'Generated vein', 'vessel'), (5, 'Generated sinus', 'vessel')]]
    scene = standardize(from_case(case, mapping))
    return case, scene


def prepare(output):
    output = output.resolve()
    if not output.is_relative_to(ROOT / 'testcases') or output.exists(): raise ValueError('Choose a new output under testcases/')
    output.mkdir(parents=True); started = time.perf_counter(); case, scene = make_case()
    labels_hash = array_digest(scene.labels); mri_hash = array_digest(case.mri)
    actions = []; snapshots = {'main': SurgicalEngine(scene).snapshot()}; parents = {'main': None}
    base = SurgicalEngine(scene)

    def act(e, op, point=None, *, slot='left', duration=.1, **kwargs):
        action = {'slot': slot, 'operation': op, 'duration': duration, **kwargs}
        if point is not None: action['pose'] = {'position': point, 'quaternion': [0., 0., 0., 1.]}
        result = e.step(action)
        actions.append({'action': action, 'status': result['status'], 'reason': result.get('reason'), 'stateId': e.state_id})
        (output / 'actions.json').write_text(json.dumps(actions, indent=2) + '\n')
        if result['status'] == 'rejected': raise ValueError(str(actions[-1]))
        return result

    act(base, 'configure', mechanicsRevision=2)
    snapshots['configured-start'] = base.snapshot(); parents['configured-start'] = {'branchId': 'main', 'position': 0}
    results = []
    for name in ('no-heat', 'continuous', 'pulsed', 'stop-with-residual'):
        e = base.clone(); contact_trace = []
        outside = [4.5, 6., -5.]; contact = [4.5, 6., 11.8]
        if name in ('continuous', 'pulsed'):
            act(e, 'exchange', instrument='bipolar_forceps')
            act(e, 'place', outside, parameters={'jawOpeningMm': .5, 'forceN': 0., 'power': 0., 'depthMm': 1.})
            act(e, 'apply', contact, duration=1.)
            if name == 'continuous':
                act(e, 'apply', parameters={'power': 1.}, duration=2.1)
            else:
                act(e, 'apply', parameters={'power': 1.}, duration=1.05)
                act(e, 'deactivate'); act(e, 'inspect', duration=6.)
                act(e, 'apply', parameters={'power': 1.}, duration=1.05)
            contact_trace = [event for row in e.history for event in row['events'] if event.get('kind') == 'bipolar_heat_input']
            act(e, 'deactivate'); act(e, 'withdraw', outside, duration=1.)
        if name != 'stop-with-residual':
            act(e, 'place', [11., 10., -5.], slot='right', parameters={'radiusMm': 1.5, 'power': 1., 'duty': 1.})
            for point in ([11., 10., 11.2], None, [11., 10., 11.2], None):
                act(e, 'apply', point, slot='right', duration=3.)
            act(e, 'deactivate', slot='right'); act(e, 'withdraw', [11., 10., -5.], slot='right', duration=1.)
        remaining = 40. - e.state['elapsedSeconds']
        if remaining <= 0: raise ValueError('Frozen comparison horizon exceeded')
        while remaining > 1e-8:
            act(e, 'inspect', duration=min(30., remaining))
            remaining = 40. - e.state['elapsedSeconds']
        snapshot = e.snapshot(); restored = SurgicalEngine.restore(scene, json.loads(json.dumps(snapshot)), force_verify=True)
        if restored.state != e.state: raise ValueError('Canonical thermal replay mismatch')
        if any(s['inserted'] or s['active'] for s in e.state['slots'].values()) or e.state['holds']:
            raise ValueError('STOP cleanup incomplete')
        account = removed_accounting(scene, e.state); thermal = heat.metrics(e); vascular = graph.metrics(e)
        if abs(thermal['bipolarHeatResidualJ']) > 1e-10 or abs(vascular['conservationResidualMm3']) > 1e-8:
            raise ValueError('Energy or vascular budget failed')
        if name in ('continuous', 'pulsed'):
            if abs(thermal['bipolarHeatInputJ'] - .042) > 1e-12: raise ValueError('Heat input was not total delivered power')
            if vascular['occludedEdgeCount'] != 1 or vascular['generatedMm3'] != 0: raise ValueError('Separate intact occlusion control failed')
        elif thermal['bipolarHeatInputJ'] or vascular['occludedEdgeCount']:
            raise ValueError('Unpowered control changed heat or patency')
        snapshots[name] = snapshot; parents[name] = {'branchId': 'configured-start', 'position': len(base.actions)}
        results.append({'branch': name, 'stateId': e.state_id, 'actions': len(e.actions), 'removed': account,
                        'thermal': thermal, 'vascular': vascular, 'heatContactEvents': contact_trace,
                        'canonicalReplay': True, 'stopToolsWithdrawn': True})
    active = [r for r in results if r['branch'] != 'stop-with-residual']
    if len({r['removed']['sourceIdsSha256'] for r in active}) != 1: raise ValueError('Unequal original target removals')
    if any(r['removed']['nonTargetRemovedMm3'] for r in active): raise ValueError('Unexpected non-target removal')
    if results[-1]['removed']['targetRemovedMm3'] != 0: raise ValueError('STOP removed tissue')
    record = {'environmentSchema': ENVIRONMENT_SCHEMA, 'mapping': scene.regions, 'branches': snapshots,
        'parents': parents, 'branchId': 'continuous', 'position': len(base.actions),
        'researchSetup': {'kind': 'generated-bipolar-thermal-control', 'description': 'No patient anatomy, electrical calibration or injury/patency mapping.'}}
    destination = output / 'bipolar-thermal.ressectionlab'; save_case(case, destination, artifacts={'surgical_toolbox': record})
    if array_digest(scene.labels) != labels_hash or array_digest(case.mri) != mri_hash: raise ValueError('Source arrays changed')
    report = {'schema': 'resectionlab.bipolar-thermal-result.v1', 'case': str(destination), 'sceneId': scene.identity,
        'protocolSha256': file_sha256(PROTOCOL), 'sourceLabelsHash': labels_hash, 'mriHash': mri_hash,
        'generatedNotPatient': True, 'sourceArraysPreserved': True, 'branches': results, 'commonHorizonSeconds': 40.,
        'coagulationAndTemperatureCoupled': False,
        'largeScaleRL': 'hold', 'deviceResponseValidated': False, 'patientInjuryValidated': False,
        'wallSeconds': time.perf_counter() - started}
    (output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'case': str(destination), 'branches': [{k: r[k] for k in ('branch', 'removed', 'thermal')} for r in results],
                      'wallSeconds': report['wallSeconds']}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('output', type=Path)
    prepare(parser.parse_args().output)
