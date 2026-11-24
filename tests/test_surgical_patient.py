import hashlib
import json
from pathlib import Path
import nibabel as nib
import numpy as np
import pytest
from resectionlab.patient_anatomy import PatientImagingCollection
from resectionlab.surgical_patient import prepare_surgical_environment, prepare_surgical_patient, surgical_environment_options, ORDER
from resectionlab.imaging import load_case
from resectionlab.surgical_scene import from_case


def fixture(tmp_path, *, status='reviewed'):
    affine = np.diag([-.5, .5, .5, 1.])
    affine[:3, 3] = [4, 2, -3]
    intensity = np.indices((12, 12, 12)).sum(axis=0).astype(np.float32)
    image = nib.Nifti1Image(intensity, affine)
    image.header.set_xyzt_units('mm')
    path = tmp_path / 'scan.nii.gz'
    nib.save(image, path)
    c = PatientImagingCollection.create(tmp_path / 'native', 'test')
    c.add_series(path, series_id='mri', modality='T1', timepoint='preoperative')
    c.add_series(path, series_id='ct', modality='CT', timepoint='preoperative')
    c.set_registration('ct', np.eye(4), status=status, method='explicit test registration')
    arrays = {name: np.zeros((12, 12, 12), np.uint8) for name in ORDER}
    arrays['brain'][2:10, 2:10, 2:10] = 1
    arrays['tumor'][4:8, 4:8, 4:8] = 1
    arrays['bone'][5, 4:8, 4:8] = 1
    arrays['bone'][0, :, :] = 1  # Thin sheet not sampled by target cell centers.
    arrays['vessels'][2, 2, :] = 1
    for name, array in arrays.items():
        c.add_layer(array, affine, layer_id=name, source_series='ct' if name == 'bone' else 'mri', kind=name, role='context', color='#aaaaaa')
    c.save()
    return c


def test_export_preserves_sources_overlaps_coordinates_and_thin_structures(tmp_path):
    c = fixture(tmp_path)
    before = {str(p.relative_to(c.root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in c.root.rglob('*') if p.is_file()}
    options = surgical_environment_options(c)
    assert options['referenceIssue'] is None
    assert {row['layerId']: row['status'] for row in options['layers']} == {key: 'ready' for key in ORDER}
    report = prepare_surgical_patient(c.root, tmp_path / 'output', layer_ids={k: k for k in ORDER})
    case = load_case(report['casePath'])
    assert case.mri.shape == (6, 6, 6)
    expected_kinds = {'brain': 'brain', 'bone': 'skull', 'tumor': 'tumor', 'vessels': 'vessels'}
    for name, layer in case.metadata['anatomy_layers'].items():
        assert layer['kind'] == expected_kinds[name]
        assert layer['provenance'] == 'estimated'
        assert layer['review_status'] == 'unreviewed'
        assert layer['model']['name']
        assert layer['coverage']
    np.testing.assert_allclose(case.affine[:3, 3], [3.75, 2.25, -2.75])
    assert case.mri[2, 2, 2] == pytest.approx(13.5)
    assert all(case.compartments[k].any() for k in ORDER)
    assert np.any(case.compartments['brain'] & case.compartments['tumor'])
    assert report['overlapCells']['brain:tumor'] > 0
    assert report['sourceLayers']['bone']['sampling'].startswith('any-hit')
    assert all(report['ownershipCells'].values())
    adapted = from_case(case)
    for r in adapted.regions:
        assert np.count_nonzero(adapted.labels == r['id']) == report['ownershipCells'][r['name']]
    after = {str(p.relative_to(c.root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in c.root.rglob('*') if p.is_file()}
    assert before == after
    assert json.loads((tmp_path / 'output/report.json').read_text())['caseHash'] == case.semantic_hash


def test_unreviewed_transform_opt_in_preserves_status(tmp_path):
    c = fixture(tmp_path, status='unreviewed')
    assert next(row for row in surgical_environment_options(c)['layers'] if row['layerId'] == 'bone')['status'] == 'estimated_alignment'
    with pytest.raises(ValueError, match='opt-in'):
        prepare_surgical_patient(c.root, tmp_path / 'rejected', layer_ids={k: k for k in ORDER})
    report = prepare_surgical_patient(c.root, tmp_path / 'output', layer_ids={k: k for k in ORDER}, allow_unreviewed_registration=True)
    assert report['sourceManifest']['transforms']['ct']['status'] == 'unreviewed'
    assert PatientImagingCollection(c.root).document['transforms']['ct']['status'] == 'unreviewed'
    assert any('unreviewed registration' in warning for warning in report['warnings'])


def test_reject_missing_or_empty_selected_anatomy(tmp_path):
    c = fixture(tmp_path)
    with pytest.raises(ValueError, match='four distinct'):
        prepare_surgical_patient(c.root, tmp_path / 'output', layer_ids={'brain': 'brain'})
    row = c.layer('vessels')
    row['label_value'] = 2
    c.save()
    assert next(row for row in surgical_environment_options(c)['layers'] if row['layerId'] == 'vessels')['status'] == 'unavailable'
    with pytest.raises(ValueError, match='no occupied'):
        prepare_surgical_patient(c.root, tmp_path / 'output', layer_ids={k: k for k in ORDER})


@pytest.mark.parametrize('name,provenance,expected', [
    ('brain', {'inference': {'model': {'name': 'SynthStrip'}}}, 'SynthStrip'),
    ('vessels', {'inference': {'model': {'name': 'vesselFM'}}}, 'vesselFM'),
    ('tumor', {'model': 'Raidionics meningioma proposal'}, 'Raidionics meningioma proposal'),
    ('bone', {'method': 'native CT 300 HU boundary proposal'}, 'native CT 300 HU boundary proposal'),
])
def test_anatomy_model_attribution_comes_from_native_provenance(name, provenance, expected):
    from resectionlab.surgical_patient import _anatomy_metadata
    layer = _anatomy_metadata(name, {'provenance': provenance})
    assert layer['model']['name'] == expected
    assert layer['provenance'] == 'estimated'


def native_wall_fixture(tmp_path, *, registration='reviewed', isolated_pair=False):
    shape = (65, 65, 65)
    affine = np.diag([-.5, .5, .5, 1.])
    affine[:3, 3] = [16., -16., -16.]
    x, y, z = np.ogrid[:shape[0], :shape[1], :shape[2]]
    scores = np.broadcast_to(np.clip(.5 + (2.45 - np.hypot(y - 32, z - 34)) / 4, 0, 1), shape).astype(np.float32)
    if isolated_pair:
        scores[33:37, 49:52, 32:35] = .42
        scores[34, 50, 33], scores[35, 50, 33] = .67, .59
    brain = np.broadcast_to((x > 4) & (z > 8) & (z <= 32), shape).astype(np.uint8)
    study = PatientImagingCollection.create(tmp_path / 'native-wall', 'native-wall-fixture')
    for series, modality, values in (('t1', 'T1', brain + np.broadcast_to(z / 64, shape)), ('tof', 'TOF', scores)):
        image = nib.Nifti1Image(values.astype(np.float32), affine)
        image.header.set_xyzt_units('mm')
        path = tmp_path / f'{series}.nii.gz'
        nib.save(image, path)
        study.add_series(path, series_id=series, modality=modality, timepoint='preoperative')
    study.set_registration('tof', np.eye(4), status=registration, method='explicit test registration')
    study.add_layer(brain, affine, layer_id='brain', source_series='t1', kind='brain', role='context', color='#aaaaaa')
    study.add_layer((scores >= .5).astype(np.uint8), affine, layer_id='vessels', source_series='tof', kind='vessels',
                    role='context', color='#aa2222', probability=scores)
    study.save()
    return study, scores, brain


def test_native_wall_catalog_keeps_cursor_pairs_when_standard_grid_is_too_large(tmp_path, monkeypatch):
    from resectionlab import surgical_patient

    study, _, brain = native_wall_fixture(tmp_path)
    study.add_layer(brain, study.series('tof')['affine'], layer_id='tof-brain', source_series='tof',
                    kind='brain', role='context', color='#aaaaaa')
    study.save()
    standard_reference_issue = 'Requested derivative exceeds 80 million cells'

    def oversized_grid(*_):
        raise ValueError(standard_reference_issue)

    monkeypatch.setattr(surgical_patient, '_research_grid', oversized_grid)
    options = surgical_environment_options(study)
    assert options['referenceIssue'] == standard_reference_issue
    assert options['nativeWall'] == {'referenceIssue': None, 'pairs': [
        {'brainLayerId': 'brain', 'vesselLayerId': 'vessels', 'scope': 'cursor'},
        {'brainLayerId': 'tof-brain', 'vesselLayerId': 'vessels', 'scope': 'cursor'},
    ]}


def test_native_wall_catalog_and_preparation_refuse_incompatible_score_lattice(tmp_path):
    study, scores, _ = native_wall_fixture(tmp_path)
    study.layer('vessels')['probability'] = study.put_array(scores[:-1])
    study.save()
    options = surgical_environment_options(study)
    vessel = next(row for row in options['layers'] if row['role'] == 'vessels')
    assert options['nativeWall'] == {'referenceIssue': None, 'pairs': []}
    assert vessel['status'] == 'ready'
    assert vessel['nativeWallIssue'].endswith('an intact native angiography score layer')
    with pytest.raises(ValueError, match=vessel['nativeWallIssue']):
        prepare_surgical_environment(study.root, tmp_path / 'result', layer_ids={'brain': 'brain', 'vessels': 'vessels'},
                                     native_wall_center=[0., 0., 1.])


def test_native_wall_source_inspection_keeps_live_state_and_checks_relocated_sources(tmp_path):
    from resectionlab.anatomy_local_wall_review import review_native_wall_alignment
    from resectionlab.anatomy_package import save_package
    from resectionlab.desktop_bridge import BridgeSession, _Request

    study, _, _ = native_wall_fixture(tmp_path, registration='unreviewed')
    transform = np.eye(4)
    transform[0, 3] = .5
    study.set_registration('tof', transform, status='unreviewed', method='local review fixture')
    center = [0., 0., 1.]
    fixed = np.array([[-4., -4., 0.], [4., -3., 0.], [-1., 3., 0.], [1., 1., 4.]])
    moving = fixed - [.5, 0., 0.]
    review_native_wall_alignment(study, 'tof', moving.tolist(), fixed.tolist(), 'local reviewer',
                                 {'brainLayerId': 'brain', 'vesselLayerId': 'vessels', 'center': center})
    study.save()
    prepared = prepare_surgical_environment(study.root, tmp_path / 'rehearsal', layer_ids={'brain': 'brain', 'vessels': 'vessels'},
                                            native_wall_center=center)
    archive = tmp_path / 'original.rsworkspace'
    save_package(study, archive)
    manifest = study.root.rename(tmp_path / 'relocated') / 'patient-anatomy.json'

    def files():
        return {str(p.relative_to(manifest.parent)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in manifest.parent.rglob('*') if p.is_file()}

    source_before = files()
    session = BridgeSession(tmp_path / 'bridge')
    try:
        def call(op, args):
            return session.execute(op, args, _Request(op), lambda *_: None)

        case = call('loadCase', {'path': prepared['casePath']})
        key = {'caseHash': case['caseHash']}
        entry = session._get_case(case['caseHash'])
        initial = call('surgicalRehearsal', {**key, 'command': 'inspect'})
        before = call('surgicalRehearsal', {**key, 'command': 'step', 'expectedStateId': initial['view']['stateId'],
                      'action': {'slot': 'left', 'operation': 'exchange', 'instrument': 'dissector', 'duration': .08}})
        assert before['outcome']['status'] == 'accepted' and before['position'] == 1
        assert call('inspectNativeWallSource', key) is None
        original = manifest.read_bytes()
        manifest.write_bytes(original + b'\n')
        with pytest.raises(ValueError, match='exact study revision'):
            call('inspectNativeWallSource', {**key, 'path': str(manifest)})
        manifest.write_bytes(original)
        acquisition = manifest.parent / study.series('tof')['original']['path']
        original_scan = acquisition.read_bytes()
        acquisition.write_bytes(original_scan + b'changed')
        with pytest.raises(ValueError, match='acquired source file differs'):
            call('inspectNativeWallSource', {**key, 'path': str(manifest)})
        acquisition.write_bytes(original_scan)
        for path in (manifest, archive):
            opened = call('inspectNativeWallSource', {**key, 'path': str(path)})
            assert opened['locallyReviewed'] and opened['centerRasMm'] == center
            assert [(scan['seriesId'], scan['centerRasMm'], scan['independent']) for scan in opened['scans']] == [
                ('t1', center, False), ('tof', [-.5, 0., 1.], True)]
            binding = {name: opened[name] for name in ('collectionId', 'snapshotId')}
            with pytest.raises(ValueError, match='read-only'):
                call('anatomySaveView', binding)
            with pytest.raises(ValueError, match='outside the saved rehearsal source'):
                call('anatomyView', {**binding, 'seriesId': 'other'})
            for scan, layer, status in [('t1', 'brain', 'patient frame'), ('tof', 'vessels', 'independent source frame')]:
                # Study-wide preview and arbitrary overlays must remain outside the inspection.
                args = {**binding, 'seriesId': scan, 'layers': ['other'], 'previewAlignment': True}
                view = call('anatomyView', {**args, 'sourceExtent': 'local'})
                assert view['registrationStatus'] == status and not view['alignmentPreviews']
                assert view['shape'] == [49, 49, 49] and view['displayPreview'] is None
                field = next(row['localField'] for row in opened['scans'] if row['seriesId'] == scan)
                assert view['sourceField'] == field and field['nativeShape'] == [65, 65, 65]
                start = np.asarray(field['startIJK'])
                crop = tuple(slice(i, i + n) for i, n in zip(start, field['shape']))
                native = np.array(study.series(scan)['affine'])
                native[:3, 3] += native[:3, :3] @ start
                np.testing.assert_array_equal(view['affine'], native)
                volume = study.series(scan)
                collection = session.native_wall_source.collection
                source_samples = collection.array(volume['data'])[crop].astype('<f4')
                assert Path(view['mri']['path']).read_bytes() == source_samples.tobytes()
                comp, = view['compartments']
                assert (comp['name'], comp['sourceSeries']) == (layer, scan)
                source_mask = (collection.array(study.layer(layer)['data'])[crop] == 1).astype(np.uint8)
                assert Path(comp['mask']['path']).read_bytes() == source_mask.tobytes()
                if scan == 'tof':
                    assert field['startIJK'] == opened['cropStartIJK']
                    faces = np.fromfile(comp['surface']['indices']['path'], '<u4').reshape(-1, 3)
                    raw = np.fromfile(comp['surface']['source']['path'], '<f4').reshape(-1, 3)
                    points = np.fromfile(comp['surface']['positions']['path'], '<f4').reshape(-1, 3)
                    assert comp['surfaceDiagnostics']['policy'] == 'native-bounded-open-vessel-v2'
                    assert not comp['surfaceDiagnostics']['sourceSurfaceOnly']
                    assert not np.array_equal(raw, points)
                    edges = np.sort(faces[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2), axis=1)
                    unique_edges, counts = np.unique(edges, axis=0, return_counts=True)
                    boundary = unique_edges[counts == 1]
                    # The long vessel reaches both x faces without synthesized end caps.
                    assert set(np.unique(points[boundary, 0])) == {0., 48.}
                    np.testing.assert_array_equal(raw[boundary], points[boundary])
                full = call('anatomyView', {**args, 'sourceExtent': 'full'})
                assert 'sourceField' not in full and full['shape'] == [65, 65, 65]
                assert full['registrationStatus'] == status
            scratch = session.native_wall_source.workspace
            call('closeNativeWallSource', {'collectionId': binding['collectionId']})
            assert not scratch.exists()
        opened = call('inspectNativeWallSource', {**key, 'path': str(manifest)})
        binding = {name: opened[name] for name in ('collectionId', 'snapshotId')}
        acquisition.write_bytes(original_scan + b'changed after verification')
        with pytest.raises(ValueError, match='study changed'):
            call('anatomyView', {**binding, 'seriesId': 'tof', 'sourceExtent': 'local'})
        acquisition.write_bytes(original_scan)
        call('closeNativeWallSource', {'collectionId': binding['collectionId']})
        assert files() == source_before
        assert session._get_case(case['caseHash']) is entry
        after = call('surgicalRehearsal', {**key, 'command': 'inspect'})
        assert (after['branchId'], after['position'], after['view']['stateId'], after['view']['slots']) == (
            before['branchId'], before['position'], before['view']['stateId'], before['view']['slots'])
    finally:
        session.close()


def test_anatomy_vessel_field_keeps_the_full_view_cache_and_native_preview_coordinates(tmp_path):
    from resectionlab.desktop_bridge import BridgeSession, _Request

    study, _, _ = native_wall_fixture(tmp_path, registration='unreviewed')
    estimated = np.eye(4)
    estimated[:3, 3] = [.5, -.25, .75]
    study.set_registration('tof', estimated, status='unreviewed', method='display test')
    study.save()
    manifest = study.root / 'patient-anatomy.json'
    original = manifest.read_bytes()
    session = BridgeSession(tmp_path / 'bridge')
    try:
        def call(op, args):
            return session.execute(op, args, _Request(op), lambda *_: None)

        binding = call('anatomyOpen', {'path': str(manifest)})
        selection = {'collectionId': binding['collectionId'], 'snapshotId': binding['document']['snapshot_id'], 'seriesId': 'tof'}
        full = call('anatomyView', {**selection, 'layers': ['vessels'], 'previewAlignment': True})
        cache = session.anatomy_view_cache
        transfers = session.anatomy_transfer_paths.copy()
        assert full['registrationStatus'] == 'alignment preview' and full['shape'] == [65, 65, 65]
        request = {'layerId': 'vessels', 'center': [.5, -.25, 1.75], 'frame': 'alignment preview'}
        field = call('anatomyView', {**selection, 'localVesselField': request})
        assert field['sourceFieldCursorRasMm'] == [0., 0., 1.]
        assert field['registrationStatus'] == 'independent source frame' and not field['patientFrameReady']
        assert field['shape'] == [49, 49, 49]
        native = np.array(study.series('tof')['affine'])
        start = np.asarray(field['sourceField']['startIJK'])
        crop = tuple(slice(i, i + 49) for i in start)
        native[:3, 3] += native[:3, :3] @ start
        np.testing.assert_array_equal(field['affine'], native)
        assert Path(field['mri']['path']).read_bytes() == study.array(study.series('tof')['data'])[crop].astype('<f4').tobytes()
        assert session.anatomy_view_cache is cache and session.anatomy_transfer_paths == transfers
        layer, = field['compartments']
        assert layer['surfaceDiagnostics']['policy'] == 'native-bounded-open-vessel-v2'
        assert layer['surfaceDiagnostics']['sectionStations'] > 0
        with pytest.raises(ValueError, match='on its active acquisition'):
            call('anatomyView', {**selection, 'seriesId': 't1', 'localVesselField': request})
        with pytest.raises(ValueError, match='Choose a scored vessel layer'):
            call('anatomyView', {**selection, 'localVesselField': {**request, 'radius': 99}})
        with pytest.raises(ValueError, match='Anatomy changed'):
            call('anatomyView', {**selection, 'snapshotId': 'wrong', 'localVesselField': request})
        assert original == manifest.read_bytes()
    finally:
        session.close()


def test_native_wall_catalog_offers_only_the_source_bound_reviewed_pair(tmp_path):
    from resectionlab.anatomy_local_wall_review import review_native_wall_alignment

    study, _, brain = native_wall_fixture(tmp_path, registration='unreviewed')
    study.add_layer(brain, study.series('t1')['affine'], layer_id='other-brain', source_series='t1',
                    kind='brain', role='context', color='#aaaaaa')
    center = [0., 0., 1.]
    landmarks = [[-4., -4., 0.], [4., -3., 0.], [-1., 3., 0.], [1., 1., 4.]]
    review_native_wall_alignment(study, 'tof', landmarks, landmarks, 'engineering scope control',
                                 {'brainLayerId': 'brain', 'vesselLayerId': 'vessels', 'center': center})
    study.save()
    options = surgical_environment_options(study)
    assert options['nativeWall'] == {'referenceIssue': None, 'pairs': [
        {'brainLayerId': 'brain', 'vesselLayerId': 'vessels', 'scope': 'reviewed-crop', 'centerRasMm': center},
    ]}
    assert study.document['transforms']['tof']['status'] == 'unreviewed'
    study.layer('brain')['revision'] += 1
    study.save()
    assert surgical_environment_options(study)['nativeWall'] == {'referenceIssue': None, 'pairs': []}


def test_standard_environment_local_native_wall_preserves_source_and_reopens(tmp_path):
    from resectionlab import surgical_wall_sweep as wall_sweep
    from resectionlab.surgical_environment import load_environment
    from resectionlab.surgical_engine import SurgicalEngine
    from resectionlab.surgical_motion import AXIAL_SWEEP_POLICY, AXIAL_SWEEP_PROPERTY
    from resectionlab.surgical_rl_protection import GuardConfig
    from resectionlab.surgical_vessel_rods import (ATTACHMENT_COUPLING_POLICY, ATTACHMENT_COUPLING_PROPERTY,
                                                 TOOL_CONTACT_CLOCK_POLICY, TOOL_CONTACT_CLOCK_PROPERTY,
                                                 WORKING_AXIS_POLICY, WORKING_AXIS_PROPERTY, view)

    study, scores, brain = native_wall_fixture(tmp_path)
    report = prepare_surgical_environment(study.root, tmp_path / 'result', layer_ids={'brain': 'brain', 'vessels': 'vessels'},
                                          native_wall_center=[0., 0., 1.])
    case = load_case(report['casePath'])
    wall = report['nativeWall']
    start = wall['sourceStartIJK']
    crop = tuple(slice(i, i + size) for i, size in zip(start, case.mri.shape))
    np.testing.assert_array_equal(case.vascular_probabilities['vessels'], scores[crop])
    np.testing.assert_array_equal(case.compartments['brain'], brain[crop])
    assert report['sourceLayers']['brain']['sampledCoverageCells'] == case.mri.size
    assert wall['sourceImageSha256'] == study.series('tof')['original']['sha256']
    assert wall['sourceScoresSha256'] == study.layer('vessels')['probability']['sha256']
    assert report['resolutionMm'] == [.5, .5, .5]
    scene = load_environment(report['casePath'])
    engine = SurgicalEngine(scene, guard_config=GuardConfig(require_critical_coverage=False))
    assert scene.properties[AXIAL_SWEEP_PROPERTY] == AXIAL_SWEEP_POLICY
    assert scene.properties[TOOL_CONTACT_CLOCK_PROPERTY] == wall[TOOL_CONTACT_CLOCK_PROPERTY] == TOOL_CONTACT_CLOCK_POLICY
    assert scene.properties[ATTACHMENT_COUPLING_PROPERTY] == wall[ATTACHMENT_COUPLING_PROPERTY] == ATTACHMENT_COUPLING_POLICY
    assert view(engine)['attachmentCouplingPolicy'] == ATTACHMENT_COUPLING_POLICY
    assert scene.properties[WORKING_AXIS_PROPERTY] == wall[WORKING_AXIS_PROPERTY] == WORKING_AXIS_POLICY
    assert view(engine)['workingAxisLimitedTicks'] == {}
    assert scene.properties['gliomaMaterialsV1'][wall_sweep.PROPERTY] == wall[wall_sweep.PROPERTY] == wall_sweep.CURRENT_POLICY
    assert report['environment']['sceneId'] == scene.identity
    assert scene._vessel_source_walls[0].geometry_binding == wall['geometryBinding']
    assert {edge['edgeId'] for edge in view(engine)['segments']} == set(wall['branches'])
    assert any(site['activeSupportCells'] for site in view(engine)['sites'])
    assert study.identity() == PatientImagingCollection(study.root).identity()


def test_target_free_native_wall_saves_keep_their_declared_sweep_policy(tmp_path, monkeypatch):
    from dataclasses import replace

    from resectionlab import surgical_motion as motion
    from resectionlab.core import thaw_json
    from resectionlab.imaging import save_case
    from resectionlab.surgical_engine import SurgicalEngine
    from resectionlab.surgical_rl_protection import GuardConfig
    from resectionlab.surgical_vessel_rods import ATTACHMENT_COUPLING_PROPERTY, TOOL_CONTACT_CLOCK_PROPERTY, WORKING_AXIS_PROPERTY

    study, _, _ = native_wall_fixture(tmp_path)
    prepared = prepare_surgical_environment(study.root, tmp_path / 'prepared', layer_ids={'brain': 'brain', 'vessels': 'vessels'},
                                            native_wall_center=[0., 0., 1.])
    fresh = load_case(prepared['casePath'])
    metadata = thaw_json(fresh.metadata)
    del metadata['surgical_adapter']['nativeWall'][WORKING_AXIS_PROPERTY]
    with pytest.raises(ValueError, match='Saved patient native wall differs'):
        from_case(replace(fresh, metadata=metadata))
    del metadata['surgical_properties'][WORKING_AXIS_PROPERTY]
    del metadata['surgical_adapter']['nativeWall'][ATTACHMENT_COUPLING_PROPERTY]
    with pytest.raises(ValueError, match='Saved patient native wall differs'):
        from_case(replace(fresh, metadata=metadata))
    del metadata['surgical_properties'][ATTACHMENT_COUPLING_PROPERTY]
    del metadata['surgical_properties'][motion.AXIAL_SWEEP_PROPERTY]
    del metadata['surgical_properties'][TOOL_CONTACT_CLOCK_PROPERTY]
    del metadata['surgical_adapter']['nativeWall'][TOOL_CONTACT_CLOCK_PROPERTY]
    historical_path = tmp_path / 'legacy-native-wall.ressectionlab'
    save_case(replace(fresh, metadata=metadata), historical_path)
    historical = from_case(load_case(historical_path))
    current = from_case(fresh)
    assert historical.identity != current.identity
    assert all(not row.get('target') for row in historical.regions)
    assert historical._vessel_source_walls[0].geometry_binding == current._vessel_source_walls[0].geometry_binding
    changed_extent = {'slot': 'left', 'operation': 'move', 'pose': {
        'position': [0., 0., 5.8], 'quaternion': [1., 0., 0., 0.]},
        'parameters': {'depthMm': .5}, 'duration': .08}
    fixed_extent = {**changed_extent, 'pose': {**changed_extent['pose'], 'position': [0., 0., 5.7]}}
    del fixed_extent['parameters']
    calls = []
    axial = motion.axial_dissector_sweep

    def observe(*args):
        calls.append(args)
        return axial(*args)

    monkeypatch.setattr(motion, 'axial_dissector_sweep', observe)
    snapshots = []
    for scene in (historical, current):
        engine = SurgicalEngine(scene, guard_config=GuardConfig(require_critical_coverage=False))
        for action in ({'slot': 'left', 'operation': 'exchange', 'instrument': 'dissector', 'duration': .08},
                       {'slot': 'left', 'operation': 'place', 'pose': {**changed_extent['pose'], 'position': [0., 0., 6.]}, 'duration': .08},
                       changed_extent):
            assert engine.step(action)['status'] == 'accepted'
        assert not calls
        assert engine.step(fixed_extent)['status'] == 'accepted'
        assert bool(calls) == (scene is current)
        snapshots.append(engine.snapshot())
    with pytest.raises(ValueError, match='Saved scene or engine version differs'):
        SurgicalEngine.restore(current, snapshots[0], force_verify=True)
    calls.clear()
    restored = SurgicalEngine.restore(historical, snapshots[0], force_verify=True)
    assert restored.snapshot() == snapshots[0] and not calls


def test_native_wall_requires_reviewed_alignment_even_when_estimated_is_allowed(tmp_path):
    study, _, _ = native_wall_fixture(tmp_path, registration='unreviewed')
    assert surgical_environment_options(study)['nativeWall'] == {'referenceIssue': None, 'pairs': []}
    with pytest.raises(ValueError, match='reviewed source alignment: tof'):
        prepare_surgical_environment(study.root, tmp_path / 'result', layer_ids={'brain': 'brain', 'vessels': 'vessels'},
                                     native_wall_center=[0., 0., 1.], allow_unreviewed_registration=True)
    assert not (tmp_path / 'result').exists()


@pytest.mark.parametrize('evidence_use', ['context_only', 'native-lattice inspection only'])
def test_local_native_wall_refuses_inspection_only_brain(tmp_path, evidence_use):
    study, _, _ = native_wall_fixture(tmp_path)
    study.layer('brain')['provenance']['evidence_use'] = evidence_use
    study.save()
    assert surgical_environment_options(study)['nativeWall'] == {'referenceIssue': None, 'pairs': []}
    with pytest.raises(ValueError, match='cannot use brain as mechanical anatomy'):
        prepare_surgical_environment(study.root, tmp_path / 'result', layer_ids={'brain': 'brain', 'vessels': 'vessels'},
                                     native_wall_center=[0., 0., 1.])
    assert not (tmp_path / 'result').exists()


def test_local_native_wall_checks_anatomical_roles(tmp_path):
    study, _, _ = native_wall_fixture(tmp_path)
    with pytest.raises(ValueError, match='cannot use vessels for the brain material'):
        prepare_surgical_environment(study.root, tmp_path / 'result', layer_ids={'brain': 'vessels', 'vessels': 'brain'},
                                     native_wall_center=[0., 0., 1.])
    assert not (tmp_path / 'result').exists()


def test_local_native_wall_samples_only_covered_brain_and_vascular_labels(tmp_path):
    study, _, brain = native_wall_fixture(tmp_path)
    study.layer('brain')['coverage_fields'] = {'supported_axial_indices': [30, 31, 32]}
    study.save()
    layers = {'brain': 'brain', 'vessels': 'vessels'}
    center = [0., 0., 1.]
    report = prepare_surgical_environment(study.root, tmp_path / 'result', layer_ids=layers, native_wall_center=center)
    case = load_case(report['casePath'])
    start = report['nativeWall']['sourceStartIJK']
    shape = case.mri.shape
    crop = tuple(slice(i, i + n) for i, n in zip(start, shape))
    expected = brain[crop].astype(bool) & np.isin(np.arange(start[2], start[2] + shape[2]), [30, 31, 32])
    np.testing.assert_array_equal(case.compartments['brain'], expected)
    assert report['sourceLayers']['brain']['sampledCoverageCells'] == 3 * shape[0] * shape[1]
    assert any('Brain source coverage is partial' in warning for warning in report['warnings'])

    study.layer('vessels')['coverage_fields'] = {'inference_bounds_voxel_inclusive': [[10, 0, 0], [64, 64, 64]]}
    study.save()
    with pytest.raises(ValueError, match='outside scored vascular coverage'):
        prepare_surgical_environment(study.root, tmp_path / 'uncovered', layer_ids=layers, native_wall_center=center)
    assert not (tmp_path / 'uncovered').exists()


def test_local_native_wall_desktop_contact_and_replay_keep_the_installed_geometry(tmp_path):
    from resectionlab.desktop_bridge import BridgeSession, _Request
    from resectionlab.surgical_vessel_surface import ISOLATED_SOURCE_GEOMETRY

    study, _, _ = native_wall_fixture(tmp_path, isolated_pair=True)

    def call(session, operation, args):
        return session.execute(operation, args, _Request(operation), lambda *_: None)

    def check_surface(payload):
        layer = next(layer for layer in payload['compartments'] if layer['name'] == 'vessels')
        surface = layer['vesselSurface']
        receipt = payload['metadata']['surgical_adapter']['nativeWall']
        assert receipt['geometryPolicy'] == ISOLATED_SOURCE_GEOMETRY
        assert receipt['sourceRetainedIsolatedComponents'] == 1
        assert ':'.join(surface[key]['sha256'] for key in ('positions', 'indices', 'sourceCells')) == receipt['geometryBinding']
        assert receipt['geometryBinding'] != receipt['sourceCrossingBinding']
        assert 'scoreLighting' in surface
        return receipt['geometryBinding']

    session = BridgeSession(tmp_path / 'transfers')
    try:
        opened = call(session, 'anatomyOpen', {'path': str(study.root / 'patient-anatomy.json')})
        binding = {'collectionId': opened['collectionId'], 'snapshotId': opened['document']['snapshot_id']}
        source_graph = call(session, 'anatomyGraph', {**binding, 'layerId': 'vessels', 'displaySeriesId': 'tof', 'point': [0., 0., 1.]})
        payload = call(session, 'prepareSurgicalEnvironment', {**binding, 'layers': {'brain': 'brain', 'vessels': 'vessels'},
            'nativeWallCenter': [0., 0., 1.]})
        geometry = check_surface(payload)
        receipt = payload['metadata']['surgical_adapter']['nativeWall']
        assert receipt['branchIdentityBasis'] == 'source study branch graph'
        assert set(receipt['branches']) == {branch['branch_id'] for branch in source_graph['branches']}
        case_args = {'caseHash': payload['caseHash']}
        state = call(session, 'surgicalRehearsal', {**case_args, 'command': 'inspect'})
        assert not state['planningAdmission']['available']
        actions = [{'operation': 'exchange', 'instrument': 'dissector', 'duration': .08},
                   {'operation': 'place', 'pose': {'position': [0., 0., 6.], 'quaternion': [1., 0., 0., 0.]}, 'duration': .08},
                   {'operation': 'move', 'pose': {'position': [0., 0., 3.], 'quaternion': [1., 0., 0., 0.]}, 'duration': .4}]
        for action in actions:
            state = call(session, 'surgicalRehearsal', {**case_args, 'command': 'step', 'expectedStateId': state['view']['stateId'],
                'action': {**action, 'slot': 'left'}})
        motion = state['view']['vesselMechanics']
        assert motion['sectionModel'] == 'linear-annular-research-contact-v5'
        assert motion['contactedCells']['left']
        assert motion['vertexDisplacementsByLabel']['2']['geometryBinding'] == geometry
        assert motion['vertexDisplacementsByLabel']['2']['indices']
        assert state['position'] == len(actions)
        destination = tmp_path / 'saved.ressectionlab'
        call(session, 'saveCase', {**case_args, 'path': str(destination), 'workspace': {}, 'overwrite': False})
    finally:
        session.close()
    cold = BridgeSession(tmp_path / 'cold-transfers')
    try:
        reopened = call(cold, 'loadCase', {'path': str(destination)})
        assert check_surface(reopened) == geometry
        replay = call(cold, 'surgicalRehearsal', {**case_args, 'command': 'inspect', 'branchId': state['branchId']})
        assert replay['view']['stateId'] == state['view']['stateId']
        assert replay['view']['vesselMechanics'] == motion
    finally:
        cold.close()
