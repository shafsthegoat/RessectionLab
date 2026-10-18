#!/usr/bin/env python3
"""One bounded vascular/field continuation of the saved guarded sub-06 rehearsal."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from zipfile import ZipFile
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from resectionlab.core import array_digest, semantic_digest
from resectionlab.imaging import save_case
from resectionlab.surgical_engine import SurgicalEngine
from resectionlab.surgical_environment import load_environment
from resectionlab.workspace_bundle import load_workspace, save_workspace

ROOT = Path(__file__).resolve().parents[1]
QUATERNION = [.5, .5, -.5, -.5]
UPSTREAM = [114.53500366210938, -118.558837890625, -533.25]
FIELD = [36.535003662109375, -118.558837890625, -533.25]


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compact_manifest(path):
    """Retain exact JSON values/arrays while respecting the native text budget."""
    with ZipFile(path) as archive:
        manifest = archive.read('manifest.json')
        if len(manifest) <= 16 * 1024**2: return
        compact = json.dumps(json.loads(manifest), separators=(',', ':'), allow_nan=False).encode()
        if len(compact) > 16 * 1024**2: raise ValueError('Even compact manifest exceeds native import budget')
        fd, temporary = tempfile.mkstemp(dir=path.parent, prefix='.vascular-', suffix='.ressectionlab')
        os.close(fd)
        try:
            with ZipFile(temporary, 'w') as result:
                for info in archive.infolist():
                    result.writestr(info, compact if info.filename == 'manifest.json' else archive.read(info.filename))
            os.replace(temporary, path)
        finally: Path(temporary).unlink(missing_ok=True)


def measurements(engine):
    metrics = engine.metrics(); vessel = engine.state.get('vesselsV2', {})
    faces = list(vessel.get('faces', {}).values())
    return {'targetRemovedMm3': metrics['targetRemovedMm3'], 'removedCells': len(engine.state['removed']),
        'fluidPresentMm3': metrics['fluidPresentMm3'], 'fluidAddedMm3': metrics['fluidAddedMm3'],
        'fluidRemovedMm3': metrics['fluidRemovedMm3'], 'fluidEscapedMm3': metrics['fluidEscapedMm3'],
        'thermalExposureProxy': metrics['thermalExposure'],
        'woundCount': len(vessel.get('wounds', {})), 'damagedFaces': sum(face['damage'] > 0 for face in faces),
        'nonPatentFaces': sum(not face['patent'] for face in faces),
        'coagulationDose': sum(face.get('coagulationDose', 0.) for face in faces),
        'vesselGeneratedMm3': vessel.get('generatedMm3', 0.),
        'vesselPhysicsV2': metrics.get('vesselPhysicsV2', {})}


def check_saved(path, include_continuation=False):
    case, artifacts, workspace = load_workspace(path)
    scene = load_environment(path); record = artifacts['surgical_toolbox']
    final_snapshot = record['branches'][record['branchId']]
    rows, initial = [], None
    if include_continuation:
        engine = SurgicalEngine.restore(scene, record['branches']['main'])
        initial = measurements(engine)
        for index, command in enumerate(final_snapshot['actions'][len(engine.actions):]):
            phase = 'release-and-withdraw' if index < 4 else 'existing-vascular-recipe' if index < 15 else 'bipolar-withdrawal' if index == 15 else 'field'
            before = measurements(engine); started = time.perf_counter()
            result = engine.step(command)
            rows.append({'phase': phase, 'action': command, 'status': result['status'], 'reason': result.get('reason'),
                'reasonCode': result.get('reasonCode'), 'events': result.get('events', []), 'stateId': engine.state_id,
                'wallSeconds': time.perf_counter() - started, 'timingContext': 'deterministic-verification-replay',
                'before': before, 'after': measurements(engine)})
        if engine.state_id != final_snapshot['stateId'] or semantic_digest(engine.state) != semantic_digest(final_snapshot['checkpoint']['state']):
            raise AssertionError('Saved continuation replay differs')
    else: engine = SurgicalEngine.restore(scene, final_snapshot)
    checked = {}
    for name in ('main', 'vascular-wound', 'vascular-coagulated', 'vascular-field', 'vascular-blocked'):
        snapshot = record['branches'].get(name)
        if snapshot is None: continue
        n = len(snapshot['actions'])
        if snapshot['actions'] != engine.actions[:n]: raise AssertionError('Branch is not a verified replay prefix')
        expected = engine.history[n - 1]['stateAfter'] if n else engine.fresh().state_id
        if snapshot['stateId'] != expected or snapshot.get('protectionContract') != engine.snapshot().get('protectionContract'):
            raise AssertionError('Branch identity or protection differs from replay')
        checked[name] = expected
    return {'stateId': engine.state_id, 'sourceCaseHash': case.semantic_hash,
        'labelArrayHash': array_digest(scene.labels), 'intensityArrayHash': array_digest(case.mri),
        'protection': engine.guard_report(), 'metrics': measurements(engine), 'verifiedBranchPrefixes': checked,
        'mainSnapshotHash': semantic_digest(record['branches']['main']),
        **({'initialMeasurements': initial, 'replayedContinuation': rows} if include_continuation else {}),
        'originalFiles': len(workspace['originals']) if workspace else 0,
        'auxiliaryScans': len(workspace['sources']) if workspace else 0}


def execute(source, destination, receipt):
    source_hash = digest(source)
    case, artifacts, workspace = load_workspace(source)
    scene = load_environment(source); record = copy.deepcopy(artifacts['surgical_toolbox'])
    main = record['branches']['main']; main_hash = semantic_digest(main)
    engine = SurgicalEngine.restore(scene, main)
    if engine.guard_report()['mode'] != 'strict-research': raise AssertionError('Strict shared protection required')
    initial = measurements(engine)
    if initial['targetRemovedMm3'] != 58.: raise AssertionError('Expected the declared target-58 bimanual main prefix')
    arrays = {'labelArrayHash': array_digest(scene.labels), 'intensityArrayHash': array_digest(case.mri)}
    original_removed = engine.state['removed'][:]
    protection = engine.guard_report(); rows, snapshots, histories = [], {}, {}
    blocker = None

    def action(command, phase):
        nonlocal blocker
        before = measurements(engine); started = time.perf_counter()
        result = engine.step(command)
        rows.append({'phase': phase, 'action': command, 'status': result['status'],
            'reason': result.get('reason'), 'reasonCode': result.get('reasonCode'), 'events': result.get('events', []),
            'stateId': engine.state_id, 'wallSeconds': time.perf_counter() - started,
            'before': before, 'after': measurements(engine)})
        print(phase, len(rows), command['operation'], result['status'], result.get('reason'), flush=True)
        if result['status'] != 'accepted':
            blocker = {'phase': phase, 'action': command, 'status': result['status'],
                'reason': result.get('reason'), 'reasonCode': result.get('reasonCode')}
            return False
        return True

    def checkpoint(name):
        snapshots[name] = engine.snapshot(); histories[name] = copy.deepcopy(engine.history)

    preparations = [dict(slot='left', operation='release', duration=.1),
        dict(slot='right', operation='release', duration=.1),
        dict(slot='right', operation='withdraw', duration=.2),
        dict(slot='left', operation='withdraw', duration=.2)]
    for command in preparations:
        if not action(command, 'release-and-withdraw'): break
    recipe_path = ROOT / 'outputs/surgical-patient-v2/vessel-snapshot.json'
    recipe = json.loads(recipe_path.read_text())['actions']
    if len(recipe) != 11: raise AssertionError('Expected the pinned eleven-action vascular recipe')
    if blocker is None:
        for i, command in enumerate(recipe):
            if not action(command, 'existing-vascular-recipe'): break
            if i == 3: checkpoint('vascular-wound')
        if blocker is None: checkpoint('vascular-coagulated')

    # Park the left tool beyond the right tool's upstream shaft along the same
    # checked withdrawal axis. This is the prescribed withdrawal, not a path hunt.
    withdrawal = dict(slot='left', operation='withdraw', duration=.5,
        pose={'position': [174.53500366210938, -118.558837890625, -533.25], 'quaternion': QUATERNION})
    if blocker is None: action(withdrawal, 'bipolar-withdrawal')
    field_actions = [dict(slot='right', operation='exchange', instrument='frazier_suction', duration=.1),
        dict(slot='right', operation='place', duration=.1, parameters={'radiusMm': 3., 'flowMm3PerSecond': 20.},
            pose={'position': UPSTREAM, 'quaternion': QUATERNION}),
        dict(slot='right', operation='apply', duration=1., pose={'position': FIELD, 'quaternion': QUATERNION}),
        dict(slot='right', operation='withdraw', duration=.2),
        dict(slot='right', operation='exchange', instrument='irrigation', duration=.1),
        dict(slot='right', operation='apply', duration=1., parameters={'radiusMm': 3., 'flowMm3PerSecond': 20.},
            pose={'position': FIELD, 'quaternion': QUATERNION})]
    if blocker is None:
        for command in field_actions:
            if not action(command, 'field'): break
    checkpoint('vascular-blocked' if blocker else 'vascular-field')
    selected = 'vascular-blocked' if blocker else 'vascular-field'
    if engine.state['removed'] != original_removed: raise AssertionError('Vascular branch unexpectedly removed source tissue')
    if engine.guard_report() != protection: raise AssertionError('Protection changed')
    if array_digest(scene.labels) != arrays['labelArrayHash'] or array_digest(case.mri) != arrays['intensityArrayHash']:
        raise AssertionError('Source arrays changed')
    parent = 'main'
    for name, snapshot in snapshots.items():
        record['branches'][name] = snapshot
        record.setdefault('parents', {})[name] = {'branchId': parent, 'position': len(record['branches'][parent]['actions'])}
        record.setdefault('timeline', {})[name] = histories[name]
        parent = name
    record.update(branchId=selected, position=len(engine.actions))
    artifacts['surgical_toolbox'] = record
    if semantic_digest(record['branches']['main']) != main_hash: raise AssertionError('Main branch changed')
    destination.parent.mkdir(parents=True, exist_ok=True)
    if workspace is None: save_case(case, destination, artifacts=artifacts)
    else:
        save_workspace(case, destination, sources=workspace['sources'], imaging=workspace['imagingState'],
            episode=workspace['episode'], selection=workspace['selection'], originals=workspace['originals'], artifacts=artifacts)
    compact_manifest(destination)
    prior_attempt_path = source.with_name(source.stem + '-attempted-actions.json')
    prior_attempts = json.loads(prior_attempt_path.read_text()) if prior_attempt_path.exists() else main['actions']
    write(destination.with_name(destination.stem + '-attempted-actions.json'), prior_attempts + [row['action'] for row in rows])
    write(destination.with_name(destination.stem + '-snapshot.json'), engine.snapshot())
    child = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--check-saved', str(destination)],
        cwd=ROOT, text=True, capture_output=True, check=True)
    reopened = json.loads(child.stdout)
    if reopened['stateId'] != engine.state_id or reopened['mainSnapshotHash'] != main_hash or reopened['protection'] != protection:
        raise AssertionError('Fresh-process branch replay differs')
    if digest(source) != source_hash: raise AssertionError('Original saved rehearsal changed')
    final = measurements(engine)
    report = {'schema': 'resectionlab.integrated-vascular-verification.v1',
        'status': 'bounded-blocker' if blocker else 'passed', 'blocker': blocker,
        'source': str(source), 'sourceSha256': source_hash, 'sourceUnchanged': True,
        'sourceCaseHash': case.semantic_hash, **arrays, 'sourceArraysUnchanged': True,
        'originalMainSnapshotHash': main_hash, 'mainPreserved': True, 'anatomyRemovalPreserved': True,
        'precursorStateId': main['stateId'],
        'precursorHeldSourceCells': {slot: len(hold['cells']) for slot, hold in main['checkpoint']['state']['holds'].items()},
        'bimanualHoldEvidence': 'This earlier target-58 precursor had an empty hold; this branch does not prove retained bimanual holding.',
        'recipePath': str(recipe_path), 'recipeSha256': digest(recipe_path),
        'recipeActionLimit': 11, 'fieldCandidateLimit': 6,
        'fieldCandidatesAttempted': sum(row['phase'] == 'field' for row in rows),
        'before': initial, 'after': final, 'actions': rows, 'protection': protection,
        'savedCase': str(destination), 'savedCaseSha256': digest(destination), 'freshProcess': reopened,
        'woundDemonstrated': final['damagedFaces'] > initial['damagedFaces'],
        'bipolarExposureDemonstrated': final['coagulationDose'] > initial['coagulationDose'],
        'suctionRemovedMm3': sum(row['after']['fluidRemovedMm3'] - row['before']['fluidRemovedMm3'] for row in rows if row['phase'] == 'field'),
        'irrigationAddedMm3': sum(event.get('volumeMm3', 0.) for row in rows for event in row['events'] if event['kind'] == 'irrigation'),
        'patencyMeaning': 'Coagulation and fluid removal do not restore cut vessel patency.',
        'limitations': ['One scripted development branch, not an independent clinical evaluation.',
            'Bleeding is generated only by supported modeled wound physics; no patient bleed was injected.',
            'Guarded anatomy is limited to supplied unreviewed masks. No clinical validation.']}
    write(receipt, report)
    return report


def recover_saved(source, destination, receipt):
    """Verify existing accepted actions after fixing archive whitespace; no new trial."""
    source_hash = digest(source); case, artifacts, _ = load_workspace(source)
    scene = load_environment(source); main = artifacts['surgical_toolbox']['branches']['main']
    compact_manifest(destination)
    child = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--check-saved', str(destination), '--include-continuation'],
        cwd=ROOT, text=True, capture_output=True, check=True)
    reopened = json.loads(child.stdout); rows = reopened.pop('replayedContinuation'); initial = reopened.pop('initialMeasurements')
    if reopened['mainSnapshotHash'] != semantic_digest(main) or reopened['sourceCaseHash'] != case.semantic_hash:
        raise AssertionError('Saved precursor differs')
    if reopened['labelArrayHash'] != array_digest(scene.labels) or reopened['intensityArrayHash'] != array_digest(case.mri):
        raise AssertionError('Source arrays differ')
    if len(rows) != 22 or any(row['status'] != 'accepted' for row in rows):
        raise AssertionError('Recovery expects the single already-completed 22-action continuation')
    if initial['removedCells'] != reopened['metrics']['removedCells'] or initial['targetRemovedMm3'] != reopened['metrics']['targetRemovedMm3']:
        raise AssertionError('Anatomy removal changed')
    if digest(source) != source_hash: raise AssertionError('Preserved precursor changed')
    recipe_path = ROOT / 'outputs/surgical-patient-v2/vessel-snapshot.json'; final = reopened['metrics']
    report = {'schema': 'resectionlab.integrated-vascular-verification.v1', 'status': 'passed', 'blocker': None,
        'source': str(source), 'sourceSha256': source_hash, 'sourceUnchanged': True,
        'sourceOriginalPathAtExecution': str(source.with_name('sub06.ressectionlab')),
        'sourceCaseHash': case.semantic_hash, 'labelArrayHash': reopened['labelArrayHash'],
        'intensityArrayHash': reopened['intensityArrayHash'], 'sourceArraysUnchanged': True,
        'originalMainSnapshotHash': semantic_digest(main), 'mainPreserved': True, 'anatomyRemovalPreserved': True,
        'precursorStateId': main['stateId'],
        'precursorHeldSourceCells': {slot: len(hold['cells']) for slot, hold in main['checkpoint']['state']['holds'].items()},
        'bimanualHoldEvidence': 'This earlier target-58 precursor had an empty hold; this branch does not prove retained bimanual holding.',
        'recipePath': str(recipe_path), 'recipeSha256': digest(recipe_path), 'recipeActionLimit': 11,
        'fieldCandidateLimit': 6, 'fieldCandidatesAttempted': 6, 'before': initial, 'after': final,
        'actions': rows, 'protection': reopened['protection'], 'savedCase': str(destination),
        'savedCaseSha256': digest(destination), 'freshProcess': reopened,
        'woundDemonstrated': final['damagedFaces'] > initial['damagedFaces'],
        'bipolarExposureDemonstrated': final['coagulationDose'] > initial['coagulationDose'],
        'suctionRemovedMm3': sum(row['after']['fluidRemovedMm3'] - row['before']['fluidRemovedMm3'] for row in rows if row['phase'] == 'field'),
        'irrigationAddedMm3': sum(event.get('volumeMm3', 0.) for row in rows for event in row['events'] if event['kind'] == 'irrigation'),
        'patencyMeaning': 'Coagulation and fluid removal do not restore cut vessel patency.',
        'archiveRepair': 'Compacted manifest JSON whitespace to meet unchanged 16 MiB native import budget; all JSON values and array members preserved.',
        'instrumentationRecovery': 'Original trial completed all 22 actions and saved its snapshot. Per-action metrics/times here come from one deterministic fresh-process verification replay, not another candidate search.',
        'limitations': ['One scripted development branch on the earlier target-58 precursor, not the corrected target-57 bimanual main.',
            'Bleeding is generated only by supported modeled wound physics; no patient bleed was injected.',
            'Guarded anatomy is limited to supplied unreviewed masks. No clinical validation.']}
    write(receipt, report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT / 'outputs/integrated-rehearsal-v1/sub06-target58-precursor.ressectionlab')
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/integrated-rehearsal-v1/sub06-vascular.ressectionlab')
    parser.add_argument('--receipt', type=Path, default=ROOT / 'artifacts/integrated-rehearsal-v1/vascular-verification.json')
    parser.add_argument('--check-saved', type=Path)
    parser.add_argument('--include-continuation', action='store_true')
    parser.add_argument('--recover-saved', action='store_true')
    args = parser.parse_args()
    if args.check_saved: print(json.dumps(check_saved(args.check_saved.resolve(), args.include_continuation), allow_nan=False))
    elif args.recover_saved: recover_saved(args.source.resolve(), args.output.resolve(), args.receipt.resolve())
    else: execute(args.source.resolve(), args.output.resolve(), args.receipt.resolve())
