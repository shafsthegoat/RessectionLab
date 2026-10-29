#!/usr/bin/env python3
"""Replayable acquired-source retraction/resection with explicit continuum assumptions.

Uses only the already admitted frozen 0007 archive; preserves its historical
exposure and all source arrays. No policy training or anatomical inference.
"""
from pathlib import Path
import argparse
import copy
import json
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from resectionlab.core import array_digest, thaw_json
from resectionlab.simulator_qualification import file_sha256, removed_accounting
from resectionlab.workspace_bundle import load_workspace, save_workspace
from resectionlab.imaging import save_case
from resectionlab.surgical_scene import from_case
from resectionlab.surgical_environment import standardize, SCHEMA
from resectionlab.surgical_engine import SurgicalEngine
from resectionlab.surgical_motion import envelopes
from resectionlab.surgical_tools import TOOLS
from resectionlab import surgical_continuum as continuum


def prepare(source, output):
    contract = json.loads((ROOT / 'manifests/experiments/simulator-qualification-v1.json').read_text())
    if file_sha256(source) != contract['baseline_archive_sha256']:
        raise ValueError('Exact frozen development archive required before decoding')
    output = output.resolve()
    if not output.is_relative_to(ROOT / 'testcases') or output.exists():
        raise ValueError('Choose a NEW output directory under testcases/')
    output.mkdir(parents=True); started = time.perf_counter()
    case, artifacts, workspace = load_workspace(source)
    previous = artifacts['surgical_toolbox']
    metadata = thaw_json(case.metadata)
    materials = {}
    for region in previous['mapping']:
        if region['material'] not in continuum.SOFT: continue
        mu = 258.77 * (2 if region['material'] == 'tumor' else 1)
        materials[str(region['id'])] = {'shearPa': mu, 'poissonRatio': .45, 'relaxationShearPa': [mu / 2],
            'source': ('Assumed tumor/brain stiffness ratio2; no patient tumor mechanics measured. ' if region['material'] == 'tumor' else '')
                + 'Hinrichsen2023 Table11 cortex258.77Pa used as a homogeneous postmortem research scenario, not a patient regional assignment; branch half-modulus/tau0.5s assumed.'}
    metadata['surgical_properties'][continuum.PROPERTY] = continuum.research_profile(materials, relaxation_seconds=[.5])
    derived = case.revised(metadata=metadata)
    scene = standardize(from_case(derived, previous['mapping']), physics_profile=previous['physicsProfile'])
    reference = standardize(from_case(case, previous['mapping']), physics_profile=previous['physicsProfile'])
    np.testing.assert_array_equal(scene.labels, reference.labels)
    source_labels = array_digest(scene.labels); mri_hash = array_digest(case.mri)
    rows = []

    def act(engine, operation, point=None, *, slot='left', duration=.1, allow_partial=False, **extra):
        action = {'slot': slot, 'operation': operation, 'duration': duration, **extra}
        if point is not None: action['pose'] = {'position': list(map(float, point)), 'quaternion': [0., 0., 0., 1.]}
        before = engine.state_id; t = time.perf_counter(); result = engine.step(action)
        row = {'action': action, 'status': result['status'], 'reason': result.get('reason'),
            'beforeStateId': before, 'stateId': engine.state_id, 'acceptedPosition': engine.state['slots'][slot]['position'],
            'wallSeconds': time.perf_counter() - t}
        rows.append(row); (output / 'actions.json').write_text(json.dumps(rows, indent=2) + '\n')
        if result['status'] == 'rejected': raise ValueError(str(row))
        if point is not None and operation != 'apply' and not allow_partial and not np.allclose(engine.state['slots'][slot]['position'], point, rtol=0., atol=1e-8):
            raise ValueError('Requested endpoint was not reached: ' + str(row))
        return result

    engine = SurgicalEngine(scene)
    for action in previous['branches']['operative-cusa']['actions'][:4]:
        result = engine.step(action)
        if result['status'] == 'rejected': raise ValueError('Original exposure no longer executes: ' + str(result))
    exposure = engine.snapshot()
    act(engine, 'configure', mechanicsRevision=2)
    act(engine, 'exchange', instrument='brain_retractor')
    outside = np.array([-84., 118., 26.]); q = [0., 0., 0., 1.]
    act(engine, 'place', outside, parameters={'radiusMm': 3., 'depthMm': 2., 'forceN': .08})
    contact = None; preceding = None
    # A fixed source-geometry first-contact scan, not a fitted action policy.
    # It changes no source masks and inspects no hidden functional/vessel data.
    for z in np.arange(30., 75. + 1e-9, .25):
        point = np.array([-84., 118., z]); working = set(); shaft = set()
        for a, b, radius, part in envelopes(TOOLS['brain_retractor'], engine.parameters(engine.state['slots']['left']), point, q):
            (shaft if part == 'shaft' else working).update(engine.touched(a, b, radius))
        if working or shaft:
            if shaft or preceding is None: raise ValueError('No unobstructed blade-first contact on frozen centerline')
            contact = point; break
        preceding = point
    if contact is None: raise ValueError('No source contact on fixed centerline')
    act(engine, 'move', preceding, duration=1., allow_partial=True)
    # The conservative swept envelope can stop a free move before its requested
    # endpoint. Acquire from that ACTUAL pose in single native integration steps
    # so hold origin, target and accepted tip coincide; never drag during hold.
    for _ in range(8):
        step = np.asarray(engine.state['slots']['left']['position']) + [0., 0., .125]
        act(engine, 'hold', step, duration=.05)
        if engine.state['holds'].get('left', {}).get('cells'): break
    if not engine.state['holds'].get('left', {}).get('cells'):
        raise ValueError('No source-bound retractor hold admitted')
    contact = np.asarray(engine.state['slots']['left']['position'])
    if not np.allclose(engine.state['holds']['left']['origin'], contact, rtol=0., atol=1e-8):
        raise ValueError('Acquisition moved after first source contact')
    common = engine.snapshot(); start = len(engine.actions)
    (output / 'contact.json').write_text(json.dumps({'position': contact.tolist(), 'holds': engine.state['holds'],
        'activeCells': len(engine.state['deformationModel']['active']),
        'continuum': continuum.descriptor(engine)['summary']}, indent=2) + '\n')
    branches = {'main': SurgicalEngine(scene).snapshot(), 'authored-exposure': exposure, 'retraction-start': common}
    parents = {'main': None, 'authored-exposure': {'branchId': 'main', 'position': 0},
        'retraction-start': {'branchId': 'authored-exposure', 'position': 4}}
    summaries = []
    for name, hold_seconds in [('short-hold', .2), ('sustained-hold', 2.)]:
        trial = engine.clone()
        act(trial, 'move', contact + [.1, 0., 0.], duration=.3)
        act(trial, 'inspect', duration=hold_seconds)
        before_resection = copy.deepcopy(trial.state[continuum.PROPERTY])
        cusa_outside = outside + [-6., 0., 0.]
        act(trial, 'exchange', slot='right', instrument='cusa')
        act(trial, 'place', cusa_outside, slot='right', parameters={'radiusMm': 1.5, 'power': 1., 'duty': 1.})
        before_removal = removed_accounting(scene, trial.state)
        act(trial, 'apply', contact + [-6., 0., 1.], slot='right', duration=3.)
        # Travel can stop at first retained contact before significant capacity
        # accumulates. Separate stationary dwells do actual unchanged2mm3/s
        # work; no whole-stroke or categorical removal shortcut is used.
        act(trial, 'apply', slot='right', duration=3.)
        act(trial, 'apply', contact + [-6., 0., 1.], slot='right', duration=3.)
        act(trial, 'apply', slot='right', duration=3.)
        after_removal = removed_accounting(scene, trial.state)
        if after_removal['sourceCellCount'] <= before_removal['sourceCellCount']:
            raise ValueError('No actual finite-capacity source tissue removal in this branch')
        act(trial, 'deactivate', slot='right')
        act(trial, 'withdraw', cusa_outside, slot='right', duration=1.)
        act(trial, 'move', contact, duration=.3)
        act(trial, 'release', duration=.05)
        act(trial, 'inspect', duration=1.)
        act(trial, 'withdraw', outside, duration=1.)
        saved = trial.snapshot(); replay_started = time.perf_counter()
        restored = SurgicalEngine.restore(scene, json.loads(json.dumps(saved)))
        replay_seconds = time.perf_counter() - replay_started
        if restored.state_id != trial.state_id or restored.state != trial.state:
            raise ValueError('Canonical continuum replay mismatch')
        branches[name] = saved; parents[name] = {'branchId': 'retraction-start', 'position': start}
        summary = {k: v for k, v in before_resection.items() if k not in ('cornerDisplacementMm', 'memoryDisplacementMm')}
        summaries.append({'branch': name, 'holdSeconds': hold_seconds, 'beforeResection': summary,
            'resectionDeltaMm3': {k: after_removal[k] - before_removal[k] for k in ('targetRemovedMm3', 'nonTargetRemovedMm3')},
            'fullEpisode': removed_accounting(scene, trial.state), 'finalStateId': trial.state_id,
            'finalContinuum': continuum.descriptor(trial)['summary'], 'exactReplay': True})
        summaries[-1].update(nativeActions=len(trial.actions), nativeSeconds=trial.state['elapsedSeconds'],
            canonicalReplayWallSeconds=replay_seconds, finalTools=copy.deepcopy(trial.state['slots']))
        (output / 'progress.json').write_text(json.dumps({'completed': name, 'stateId': trial.state_id}) + '\n')
    # A valid stop keeps the accepted prefix after an inadmissible attempted
    # continuation; the rejected action never enters the accepted replay.
    stopped = engine.clone(); before_stop = stopped.snapshot()
    candidate = {'slot': 'left', 'operation': 'move', 'duration': .3,
        'pose': {'position': (contact + [1., 0., 0.]).tolist(), 'quaternion': q}}
    refused = stopped.step(candidate)
    if refused['status'] != 'rejected' or stopped.snapshot() != before_stop:
        raise ValueError('Numerical-domain STOP did not preserve the accepted prefix')
    # Stopping the proposed continuation still releases and withdraws both
    # tools through ordinary checked actions. It does not strand a held blade.
    act(stopped, 'release', duration=.05)
    act(stopped, 'withdraw', outside, duration=1.)
    stopped_snapshot = stopped.snapshot()
    if SurgicalEngine.restore(scene, json.loads(json.dumps(stopped_snapshot))).state != stopped.state:
        raise ValueError('STOP cleanup canonical replay mismatch')
    if any(s['inserted'] or s['active'] for s in stopped.state['slots'].values()) or stopped.state['holds']:
        raise ValueError('STOP cleanup left an active/held/inserted instrument')
    branches['stop-at-model-limit'] = stopped_snapshot
    parents['stop-at-model-limit'] = {'branchId': 'retraction-start', 'position': start}
    if summaries[0]['fullEpisode']['sourceIdsSha256'] != summaries[1]['fullEpisode']['sourceIdsSha256']:
        raise ValueError('Matched resection comparison removed different source cells')
    record = {'environmentSchema': SCHEMA, 'physicsProfile': copy.deepcopy(scene.properties['tissuePhysicsV2']),
        'mapping': scene.regions, 'branches': branches, 'parents': parents, 'branchId': 'short-hold', 'position': start,
        'researchSetup': {'kind': 'explicit-continuum-retraction-research', 'exposureActions': 4,
            'description': 'Same original source labels and recorded historical exposure; small-gradient voxel continuum, assumed postmortem material transfer, tumor ratio and relaxation. No vascular/functional/clinical calibration.',
            'sourceSha256': contract['baseline_archive_sha256'], 'continuumProtocol': 'manifests/experiments/continuum-mechanics-v1.json'}}
    destination = output / 'continuum-rehearsal.ressectionlab'
    if workspace:
        save_workspace(derived, destination, sources=workspace['sources'], imaging=workspace['imagingState'],
            episode=workspace['episode'], selection=workspace['selection'], artifacts={'surgical_toolbox': record}, originals=workspace['originals'])
    else: save_case(derived, destination, artifacts={'surgical_toolbox': record})
    if array_digest(scene.labels) != source_labels or array_digest(derived.mri) != mri_hash or file_sha256(source) != contract['baseline_archive_sha256']:
        raise ValueError('Source preservation check failed')
    report = {'schema': 'resectionlab.continuum-rehearsal.v1', 'case': str(destination), 'sceneId': scene.identity,
        'sourceArchiveSha256': contract['baseline_archive_sha256'], 'labelsHash': source_labels, 'mriHash': mri_hash,
        'sourceArraysUnchanged': True, 'contactRasMm': contact.tolist(), 'commonActions': start,
        'branches': summaries, 'stop': {'reason': refused.get('reason'), 'attemptedAction': candidate,
            'acceptedPrefixUnchanged': True, 'acceptedPrefixStateId': before_stop['stateId'],
            'finalStateId': stopped.state_id, 'nativeActions': len(stopped.actions),
            'toolsWithdrawn': True, 'canonicalReplay': True,
            'fullEpisode': removed_accounting(scene, stopped.state), 'clinicalSafetyClaim': False},
        'modelAssumptions': scene.properties[continuum.PROPERTY], 'patientMechanicsValidated': False,
        'largeScaleRL': 'hold', 'wallSeconds': time.perf_counter() - started}
    report['sourceSha256'] = {name: file_sha256(ROOT / name) for name in (
        'src/resectionlab/surgical_continuum.py', 'src/resectionlab/surgical_deformation_v2.py',
        'scripts/prepare_continuum_rehearsal.py')}
    report['forwardActionWallSeconds'] = sum(r['wallSeconds'] for r in rows)
    (output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'case': str(destination), 'branches': [(r['branch'], r['resectionDeltaMm3']) for r in summaries], 'wallSeconds': report['wallSeconds']}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('source', type=Path); parser.add_argument('output', type=Path)
    args = parser.parse_args()
    try: prepare(args.source.resolve(), args.output)
    except Exception as error:
        out = args.output.resolve()
        if out.is_relative_to(ROOT / 'testcases') and out.is_dir():
            failure = out / 'failure.json'
            if not failure.exists(): failure.write_text(json.dumps({'type': type(error).__name__, 'error': str(error)}) + '\n')
        raise
