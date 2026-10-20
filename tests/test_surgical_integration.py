"""Cross-boundary controls: shared occupancy, motion, replay, and durable history."""
import copy
import json
import os
import subprocess
import sys

import numpy as np
import pytest
from dataclasses import replace
from scipy.spatial.transform import Rotation
from resectionlab.surgical_scene import SurgicalScene
from resectionlab.surgical_engine import SurgicalEngine
from resectionlab.desktop_bridge import BridgeSession, _Request


def test_full_case_navigation_payload_keeps_the_native_archive_private(tmp_path):
    from resectionlab.core import freeze_json, thaw_json
    session = BridgeSession(tmp_path / 'transfers')
    try:
        loaded = session.execute('createSurgicalFixture', {}, _Request('fixture'), lambda *_: None)
        key = loaded['caseHash']
        def call(args):
            return session.execute('surgicalRehearsal', {'caseHash': key, **args}, _Request('rehearsal'), lambda *_: None)
        created = call({'command': 'create'})
        entry = session.cases[key]
        artifacts = thaw_json(entry.artifacts)
        artifacts['surgical_toolbox']['fullCase'] = {'main': {
            'schema': 'resectionlab.full-case-rehearsal.v1', 'scenario': 'primary', 'status': 'running',
            'recordedBookkeeping': 'retained-private;' * 600000,
        }}
        session.cases[key] = entry = replace(entry, artifacts=freeze_json(artifacts))
        display = session._display_artifacts(entry)
        assert len(json.dumps(display).encode()) < 4096
        assert display['surgical_toolbox']['requiresReplayVerification']
        assert 'main' in display['surgical_toolbox']['branches']
        assert display['surgical_toolbox']['fullCase']['main']['schema'] == 'resectionlab.full-case-rehearsal.v1'
        assert thaw_json(entry.artifacts) == artifacts
        # A metadata summary never replaces the private replay input.
        inspected = call({'command': 'inspect'})
        assert inspected['view']['stateId'] == created['view']['stateId']
        preview = call({'command': 'preview', 'expectedStateId': inspected['view']['stateId'],
                        'action': {'slot': 'left', 'operation': 'inspect', 'duration': .1}})
        assert preview['view']['history'] == []
        assert preview['view']['removed'] == inspected['view']['removed']
    finally:
        session.close()


def test_display_timeline_compacts_waypoints_without_changing_replay_records():
    from resectionlab.surgical_bridge import display_timeline
    path = [{'position': [0., 0., i / 20.], 'quaternion': [0., 0., 0., 1.]} for i in range(512)]
    history = [{'action': {'slot': 'left', 'operation': 'apply', 'path': path},
                'status': 'accepted', 'events': [
                    {'kind': 'removal', 'cells': 2, 'volumeMm3': 2.},
                    {'kind': 'removal', 'cells': 3, 'volumeMm3': 3.}]}]
    before = copy.deepcopy(history)
    shown = display_timeline(history)
    assert history == before
    assert shown[0]['recordedPathPoseCount'] == 512
    assert shown[0]['recordedPathEndpoints'] == [path[0], path[-1]]
    assert 'path' not in shown[0]['action']
    assert shown[0]['events'][0]['cells'] == 5


def scene(affine=None, names=('A', 'B', 'C')):
    labels = np.zeros((40, 40, 30), np.uint16)
    labels[8, 8, 15], labels[25, 8, 15], labels[8, 25, 15] = 1, 2, 3
    return SurgicalScene(labels, np.eye(4) if affine is None else affine,
                         [{'id': i, 'name': name, 'material': material, 'target': i == 2}
                          for i, name, material in zip((1, 2, 3), names, ('bone', 'tumor', 'soft_tissue'))], {}, {})


def action(slot, operation, position=None, quaternion=None, **kwargs):
    result = {'slot': slot, 'operation': operation, 'duration': .1, **kwargs}
    if position is not None:
        result['pose'] = {'position': list(position), 'quaternion': quaternion or [0., 0., 0., 1.]}
    return result


def sequence():
    return [action('left', 'place', [8, 8, 10], parameters={'radiusMm': .6}),
            action('left', 'apply', [8, 8, 15]), action('left', 'withdraw'),
            action('left', 'exchange', instrument='brain_retractor'),
            action('left', 'place', [8, 25, 10], parameters={'radiusMm': .6}),
            action('left', 'hold', [8, 25, 15]), action('left', 'move', [10, 25, 15]),
            action('right', 'place', [25, 8, 10], parameters={'radiusMm': .6}),
            action('right', 'apply', [25, 8, 15]), action('left', 'release')]


def test_three_tools_share_removal_displacement_release_and_exact_replay():
    source = scene()
    engine = SurgicalEngine(source)
    tissue = np.ravel_multi_index((8, 25, 15), source.shape)
    destination = np.ravel_multi_index((10, 25, 15), source.shape)
    before = source.labels.copy()
    for index, step in enumerate(sequence()):
        outcome = engine.step(step)
        assert outcome['status'] == 'accepted', outcome
        if index == 6:
            assert engine.occupancy()[tissue] == 0
            assert engine.occupancy()[destination] == tissue + 1
            assert engine.view()['moved'] == [[tissue, destination]]
    assert len(engine.state['removed']) == 2
    assert engine.occupancy()[tissue] == tissue + 1
    assert engine.occupancy()[destination] == 0
    assert engine.state['holds'] == {}
    assert engine.metrics()['targetRemovedMm3'] == 1.
    np.testing.assert_array_equal(source.labels, before)
    restored = SurgicalEngine.restore(source, json.loads(json.dumps(engine.snapshot())))
    assert restored.state == engine.state
    assert restored.history == engine.history
    np.testing.assert_array_equal(restored.occupancy(), engine.occupancy())


def test_partial_motion_stops_before_retained_material_and_rejections_are_atomic():
    engine = SurgicalEngine(scene())
    assert engine.step(action('left', 'place', [8, 8, 10], parameters={'radiusMm': .6}))['status'] == 'accepted'
    result = engine.step(action('left', 'move', [8, 8, 20]))
    assert result['status'] == 'partial'
    assert 0 < result['completedFraction'] < 1
    assert engine.state['slots']['left']['position'][2] < 15
    assert not engine.state['removed']
    before = copy.deepcopy(engine.state)
    steps = copy.deepcopy(engine.actions)
    rejected = engine.step(action('left', 'exchange', instrument='cusa'))
    assert rejected['status'] == 'rejected'
    assert engine.state == before and engine.actions == steps
    assert SurgicalEngine.restore(engine.scene, engine.snapshot()).state == before


def test_tool_collisions_at_endpoint_and_during_rotation():
    engine = SurgicalEngine(scene())
    qleft = Rotation.from_euler('y', -90, degrees=True).as_quat().tolist()
    qright = Rotation.from_euler('y', 90, degrees=True).as_quat().tolist()
    assert engine.step(action('left', 'place', [0, 0, 0], qleft, parameters={'radiusMm': .5}))['status'] == 'accepted'
    assert engine.step(action('right', 'place', [0, 0, -20], parameters={'radiusMm': .5}))['status'] == 'accepted'
    original = copy.deepcopy(engine.state)
    result = engine.step(action('left', 'place', [0, 0, -20], qleft))
    assert result['status'] == 'rejected' and 'TOOL_COLLISION' in result['reason']
    assert engine.state == original
    result = engine.step(action('left', 'move', [0, 0, 0], qright))
    assert result['status'] == 'partial' and 'TOOL_COLLISION' in result['reason']
    # Both horizontal endpoint shafts are clear; the downward intermediate shaft is blocked.
    assert 0 < result['completedFraction'] < 1
    assert SurgicalEngine.restore(engine.scene, engine.snapshot()).state == engine.state


def test_material_behavior_is_invariant_to_names_and_oblique_world_coordinates():
    rotation = Rotation.from_euler('xyz', [15, -20, 35], degrees=True)
    affine = np.eye(4)
    affine[:3, :3] = rotation.as_matrix()
    affine[:3, 3] = [100, -70, 40]
    baseline, transformed = SurgicalEngine(scene()), SurgicalEngine(scene(affine, ('C', 'unrelated', 'B')))
    for original in sequence()[:2]:
        step = copy.deepcopy(original)
        step['pose']['position'] = (affine[:3, :3] @ step['pose']['position'] + affine[:3, 3]).tolist()
        step['pose']['quaternion'] = rotation.as_quat().tolist()
        assert baseline.step(original)['status'] == 'accepted'
        assert transformed.step(step)['status'] == 'accepted'
    assert transformed.state['removed'] == baseline.state['removed']
    assert transformed.metrics()['nonTargetRemovedMm3'] == pytest.approx(baseline.metrics()['nonTargetRemovedMm3'])


def call(session, operation, args):
    return session.execute(operation, args, _Request(operation), lambda *_: None)


def test_bridge_undo_redo_branch_and_bundle_reopen_in_fresh_process(tmp_path):
    session = BridgeSession(tmp_path / 'transfers')
    try:
        fixture = call(session, 'createSurgicalFixture', {})
        case_hash = fixture['caseHash']
        def command(name, prior=None, **extras):
            args = {'caseHash': case_hash, 'command': name, **extras}
            if prior is not None:
                args['expectedStateId'] = prior['view']['stateId']
            return call(session, 'surgicalRehearsal', args)
        created = command('create')
        first = command('step', created, action=action('right', 'place', [43, 20, 10], parameters={'radiusMm': .6}))
        second = command('step', first, action=action('right', 'apply', [43, 20, 25]))
        assert second['outcome']['status'] == 'accepted'
        assert second['view']['removed']
        undone = command('undo', second)
        assert undone['view']['stateId'] == first['view']['stateId']
        redone = command('redo', undone)
        assert redone['view']['stateId'] == second['view']['stateId']
        undone = command('undo', redone)
        branch = command('step', undone, action=action('right', 'apply', [43, 20, 24]))
        assert branch['branchId'] != second['branchId']
        selected = command('selectBranch', branch, branchId=second['branchId'])
        assert selected['view']['stateId'] == second['view']['stateId']
        path = tmp_path / 'surgical.ressectionlab'
        call(session, 'saveCase', {'caseHash': case_hash, 'path': str(path), 'workspace': {}, 'overwrite': False})
        expected = {'stateId': selected['view']['stateId'], 'removed': selected['view']['removed'],
                    'branchId': selected['branchId'], 'branches': selected['branches']}
    finally:
        session.close()
    script = '''import json, sys
from resectionlab.desktop_bridge import BridgeSession, _Request
s = BridgeSession(sys.argv[2])
def call(op, args): return s.execute(op, args, _Request(op), lambda *_: None)
try:
    c = call('loadCase', {'path': sys.argv[1]})
    r = call('surgicalRehearsal', {'caseHash': c['caseHash'], 'command': 'inspect'})
    print(json.dumps({'stateId': r['view']['stateId'], 'removed': r['view']['removed'], 'branchId': r['branchId'], 'branches': r['branches']}))
finally: s.close()
'''
    environment = dict(os.environ, PYTHONPATH=os.path.abspath('src'))
    result = subprocess.run([sys.executable, '-c', script, str(path), str(tmp_path / 'fresh')], env=environment, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    actual = json.loads(result.stdout)
    actual['branches'] = sorted(actual['branches'], key=lambda row: row['id'])
    expected['branches'] = sorted(expected['branches'], key=lambda row: row['id'])
    assert actual == expected


def test_narrow_working_tips_do_not_shrink_physical_shafts():
    engine = SurgicalEngine(scene())
    assert engine.step(action('left', 'place', [0, 0, 0], parameters={'radiusMm': .1}))['status'] == 'accepted'
    result = engine.step(action('right', 'place', [.5, 0, 0], parameters={'radiusMm': .1}))
    assert result['status'] == 'rejected', 'Two 0.5 mm radius shafts overlap despite 0.1 mm working tips'
    assert 'TOOL_COLLISION' in result['reason']


def test_bridge_preview_and_stale_actions_leave_committed_history_untouched(tmp_path):
    from resectionlab.desktop_bridge import BridgeError
    session = BridgeSession(tmp_path / 'transfers')
    try:
        fixture = call(session, 'createSurgicalFixture', {})
        key = fixture['caseHash']
        created = call(session, 'surgicalRehearsal', {'caseHash': key, 'command': 'create'})
        current = created['view']['stateId']
        args = {'caseHash': key, 'command': 'preview', 'expectedStateId': current,
                'action': action('left', 'place', [12, 15, 10])}
        preview = call(session, 'surgicalRehearsal', args)
        assert preview['status'] == 'accepted'
        assert preview['stateId'] != current
        inspected = call(session, 'surgicalRehearsal', {'caseHash': key, 'command': 'inspect'})
        assert inspected['view']['stateId'] == current and inspected['position'] == 0
        with pytest.raises(BridgeError) as error:
            call(session, 'surgicalRehearsal', {**args, 'command': 'step', 'expectedStateId': preview['stateId']})
        assert error.value.code == 'SURGICAL_STATE_STALE'
        inspected = call(session, 'surgicalRehearsal', {'caseHash': key, 'command': 'inspect'})
        assert inspected['view']['stateId'] == current and inspected['position'] == 0
    finally:
        session.close()


@pytest.mark.parametrize('kind', ['nifti', 'mesh'])
def test_import_bridge_preserves_original_and_reopens_bundle(tmp_path, kind):
    import nibabel as nib
    import zipfile
    if kind == 'nifti':
        path = tmp_path / 'labels.nii.gz'
        data = np.zeros((8, 9, 10), np.int32)
        data[3:5, 3:5, 4:6] = 70001
        im = nib.Nifti1Image(data, np.diag([2., 3., 4., 1.]))
        im.header.set_xyzt_units('mm')
        nib.save(im, path)
        options = {}
    else:
        path = tmp_path / 'sheet.obj'
        path.write_text('v 0 0 0\nv 3 0 0\nv 0 3 0\nf 1 2 3\n')
        options = {'topology': 'sheet', 'thicknessMm': .5, 'resolution': .5, 'units': 'mm'}
    original = path.read_bytes()
    session = BridgeSession(tmp_path / 'transfers')
    bundle = tmp_path / 'import.ressectionlab'
    try:
        imported = call(session, 'importSurgicalScene', {'path': str(path), 'frame': 'RAS+', 'material': 'bone', **options})
        current = call(session, 'surgicalRehearsal', {'caseHash': imported['caseHash'], 'command': 'create'})
        state_id = current['view']['stateId']
        adapter = current['view']['scene']['source']['adapter']
        assert adapter['type'] == kind
        call(session, 'saveCase', {'caseHash': imported['caseHash'], 'path': str(bundle), 'workspace': {}, 'overwrite': False})
    finally:
        session.close()
    assert path.read_bytes() == original
    with zipfile.ZipFile(bundle) as archive:
        assert any(archive.read(name) == original for name in archive.namelist() if not name.endswith('/'))
    # Reopening does not require the external source path to remain accessible.
    path.unlink()
    script = '''import json, sys
from resectionlab.desktop_bridge import BridgeSession, _Request
s = BridgeSession(sys.argv[2])
try:
    c=s.execute('loadCase', {'path':sys.argv[1]}, _Request('load'), lambda *_:None)
    r=s.execute('surgicalRehearsal', {'caseHash':c['caseHash'],'command':'inspect'}, _Request('inspect'), lambda *_:None)
    print(json.dumps({'stateId':r['view']['stateId'],'adapter':r['view']['scene']['source']['adapter']}))
finally:s.close()
'''
    result = subprocess.run([sys.executable, '-c', script, str(bundle), str(tmp_path / 'fresh')], env=dict(os.environ, PYTHONPATH=os.path.abspath('src')), capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {'stateId': state_id, 'adapter': adapter}


def native_fixture(tmp_path, units='mm', registration='reference'):
    import nibabel as nib
    from resectionlab.patient_anatomy import PatientImagingCollection
    path = tmp_path / 'native.nii'
    im = nib.Nifti1Image(np.zeros((12, 12, 12), np.int16), np.eye(4))
    im.header.set_xyzt_units(units)
    nib.save(im, path)
    collection = PatientImagingCollection.create(tmp_path / 'native', 'arbitrary')
    collection.add_series(path, series_id='scan', modality='T1', timepoint='preoperative')
    mask = np.zeros((12, 12, 12), np.uint8)
    mask[5:7, 5:7, 5:7] = 1
    collection.add_layer(mask, im.affine, layer_id='region', source_series='scan', kind='bone', role='context', color='#aaaaaa')
    collection.document['transforms']['scan']['status'] = registration
    collection.save()
    return collection


def native_binding(session, manifest):
    import hashlib
    opened = call(session, 'anatomyOpen', {'path': str(manifest)})
    source = {'collectionId': opened['collectionId'], 'snapshotId': opened['document']['snapshot_id'], 'seriesId': 'scan'}
    identity = 'sha256:' + hashlib.sha256((source['snapshotId'] + ':scan').encode()).hexdigest()
    return {'caseHash': identity, 'source': source}


def test_native_hot_admission_hashes_disk_content_and_rejects_replacement_corruption(tmp_path, monkeypatch):
    import resectionlab.surgical_bridge as bridge
    collection = native_fixture(tmp_path)
    session = BridgeSession(tmp_path / 'transfers')
    try:
        binding = native_binding(session, collection.root / 'patient-anatomy.json')
        initial = call(session, 'surgicalRehearsal', {**binding, 'command': 'create'})
        destination = collection.root / 'surgical-toolbox.json'
        original = destination.read_text()
        hashes, restored = [], []
        digest = bridge.semantic_digest
        restore = SurgicalEngine.restore.__func__
        monkeypatch.setattr(bridge, 'semantic_digest', lambda value: hashes.append(value) or digest(value))
        monkeypatch.setattr(SurgicalEngine, 'restore', classmethod(lambda cls, *a, **kw: restored.append(a[1]) or restore(cls, *a, **kw)))
        same = call(session, 'surgicalRehearsal', {**binding, 'command': 'inspect'})
        assert same['view'] == initial['view'] and restored == []
        assert len(hashes) == 2, 'native input and committed output remain fully hashed'
        authority = session.surgical_sessions[binding['caseHash']][1]
        corrupt = json.loads(original)
        corrupt['surgical_toolbox']['branches']['main']['checkpoint']['state']['fluidEscapedMm3'] = 999
        destination.write_text(json.dumps(corrupt))
        with pytest.raises(ValueError, match='Checkpoint differs'):
            call(session, 'surgicalRehearsal', {**binding, 'command': 'inspect'})
        assert len(restored) == 1
        assert session.surgical_sessions[binding['caseHash']][1] is authority
        # Even an identical prior archive must be admitted afresh on exchange.
        equivalent = json.loads(original)
        equivalent['surgical_prior_scenes'] = [copy.deepcopy(equivalent['surgical_toolbox'])]
        destination.write_text(json.dumps(equivalent))
        selected = call(session, 'surgicalRehearsal', {**binding, 'command': 'selectScene', 'sceneIndex': 0})
        assert len(restored) == 2 and selected['view'] == initial['view']
    finally:
        session.close()


def test_native_bridge_package_fresh_reopen_preserves_source(tmp_path):
    collection = native_fixture(tmp_path)
    manifest = collection.root / 'patient-anatomy.json'
    original_manifest = manifest.read_bytes()
    original_arrays = {p.name: p.read_bytes() for p in (collection.root / 'arrays').glob('*.npy')}
    session = BridgeSession(tmp_path / 'transfers')
    package = tmp_path / 'native.rsworkspace'
    try:
        binding = native_binding(session, manifest)
        current = call(session, 'surgicalRehearsal', {**binding, 'command': 'create'})
        current = call(session, 'surgicalRehearsal', {**binding, 'command': 'step', 'expectedStateId': current['view']['stateId'],
                        'action': action('left', 'place', [5, 5, -2], parameters={'radiusMm': .6})})
        # Standard V2 drilling accumulates work before removing bone.
        for _ in range(8):
            current = call(session, 'surgicalRehearsal', {**binding, 'command': 'step', 'expectedStateId': current['view']['stateId'],
                            'action': action('left', 'apply', [5, 5, 5], duration=1.)})
            assert current['outcome']['status'] in ('accepted', 'partial')
            if current['view']['removed']:
                break
        assert current['view']['removed']
        call(session, 'anatomySavePackage', {**binding['source'], 'path': str(package)})
    finally:
        session.close()
    assert manifest.read_bytes() == original_manifest
    assert {p.name: p.read_bytes() for p in (collection.root / 'arrays').glob('*.npy')} == original_arrays
    script = '''import hashlib,json,sys
from resectionlab.desktop_bridge import BridgeSession,_Request
s=BridgeSession(sys.argv[2])
def call(op,args):return s.execute(op,args,_Request(op),lambda *_:None)
try:
    c=call('anatomyOpen',{'path':sys.argv[1],'workingRoot':sys.argv[3]})
    source={'collectionId':c['collectionId'],'snapshotId':c['document']['snapshot_id'],'seriesId':'scan'}
    key='sha256:'+hashlib.sha256((source['snapshotId']+':scan').encode()).hexdigest()
    r=call('surgicalRehearsal',{'caseHash':key,'source':source,'command':'inspect'})
    print(json.dumps({'stateId':r['view']['stateId'],'removed':r['view']['removed']}))
finally:s.close()
'''
    result = subprocess.run([sys.executable, '-c', script, str(package), str(tmp_path / 'fresh'), str(tmp_path / 'opened')], env=dict(os.environ, PYTHONPATH=os.path.abspath('src')), capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {'stateId': current['view']['stateId'], 'removed': current['view']['removed']}


@pytest.mark.parametrize('units,registration', [('unknown', 'reference'), ('mm', 'unreviewed')])
def test_native_bridge_rejects_unknown_units_or_unreviewed_registration(tmp_path, units, registration):
    collection = native_fixture(tmp_path, units, registration)
    session = BridgeSession(tmp_path / 'transfers')
    try:
        binding = native_binding(session, collection.root / 'patient-anatomy.json')
        with pytest.raises(ValueError, match='units|Registration'):
            call(session, 'surgicalRehearsal', {**binding, 'command': 'create'})
    finally:
        session.close()


def test_available_patient_bundle_uses_same_engine_without_source_mutation():
    from pathlib import Path
    import hashlib
    from resectionlab.imaging import load_case
    from resectionlab.surgical_scene import from_case
    path = Path(__file__).resolve().parents[1] / 'outputs/poc-cases/UCSF-PDGM-0004-rehearsal-ready.ressectionlab'
    if not path.exists():
        pytest.skip('Local patient fixture is not distributed with source checkout')
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    patient = load_case(path)
    original_identity = patient.semantic_hash
    adapted = from_case(patient)
    engine = SurgicalEngine(adapted)
    state_id = engine.state_id
    result = engine.preview(action('left', 'inspect'))
    assert result['status'] in ('accepted', 'rejected')
    assert engine.state_id == state_id and engine.actions == []
    assert engine.scene.shape == patient.mri.shape
    assert {r['name'] for r in adapted.regions} >= set(patient.compartments)
    assert SurgicalEngine.restore(adapted, engine.snapshot()).state_id == state_id
    compatible = [r['id'] for r in adapted.regions if r['material'] in ('soft_tissue', 'tumor')]
    coordinates = np.argwhere(np.isin(adapted.labels, compatible))
    assert len(coordinates), 'Patient fixture must declare at least one CUSA-compatible material'
    exposed = coordinates[np.argmin(coordinates[:, 2])]
    point = adapted.points([np.ravel_multi_index(exposed, adapted.shape)])[0]
    for step in (action('left', 'exchange', instrument='cusa'),
                 action('left', 'place', point - [0, 0, 5], parameters={'radiusMm': .6}),
                 action('left', 'apply', point)):
        assert engine.step(step)['status'] == 'accepted'
    assert engine.state['removed']
    assert SurgicalEngine.restore(adapted, engine.snapshot()).state == engine.state
    assert patient.semantic_hash == original_identity
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before


@pytest.mark.parametrize('properties', [
    {'bleedingSources': [{'id': 'bad', 'cell': 0, 'rateMm3PerSecond': -5, 'coagulationExposure': 1}]},
    {'bleedingSources': [{'id': 'bad', 'cell': 0, 'rateMm3PerSecond': float('nan'), 'coagulationExposure': 1}]},
    {'bleedingSources': [{'id': 'bad', 'cell': 0, 'rateMm3PerSecond': 1, 'coagulationExposure': -1}]},
    {'bleedingSources': [{'id': 'bad', 'cell': 0, 'rateMm3PerSecond': 1, 'coagulationExposure': 1, 'reservoirMm3': -1}]},
    {'bleedingSources': [{'id': 'bad', 'cell': 0, 'rateMm3PerSecond': 1, 'coagulationExposure': 1, 'exposure': float('inf')}]},
    {'bleedingSources': [{'id': 'bad', 'cell': 0, 'rateMm3PerSecond': 1, 'coagulationExposure': 1, 'exposure': -1}]},
    {'attachments': [{'id': 'bad', 'kind': 'soft', 'cells': [-1]}]},
    {'strands': [{'id': 'bad', 'cells': [99999]}]},
    {'functionalEvidence': [{'id': 'duplicate', 'cell': 0, 'probe': 'cortical'}, {'id': 'duplicate', 'cell': 1, 'probe': 'cortical'}]},
])
def test_malformed_physics_cannot_enter_simulation(properties):
    with pytest.raises(ValueError):
        SurgicalScene(np.zeros((8, 8, 8), np.uint16), np.eye(4), [], {}, properties)
