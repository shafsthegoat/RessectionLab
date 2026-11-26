import json
from copy import deepcopy
from zipfile import ZipFile
import numpy as np
import pytest
from test_patient_anatomy import patient
from resectionlab.patient_anatomy import PatientImagingCollection as Collection, world
from resectionlab.anatomy_display import display_affine, brain_vessel_display
from resectionlab.anatomy_package import save_package, open_package
from resectionlab.desktop_bridge import BridgeSession, _Request


def test_native_planning_target_uses_label_when_prepared_focus_is_outside(patient, tmp_path):
    target = patient.layer('tumor')
    labels = np.zeros(target['data']['shape'], dtype=np.uint8)
    labels[3, 5, 5] = labels[7, 5, 5] = target['label_value']
    target['data'] = patient.put_array(labels)
    transform = patient.patient_affine(target)
    prepared = world(transform, [5, 5, 5]).tolist()
    entry = world(transform, [5, 5, 12]).tolist()
    patient.document['display'] = {'planning_targets': {'tumor': {'cursor_ras_mm': prepared, 'entry_ras_mm': entry}}}
    patient.save('target picker sources')
    bridge = BridgeSession(tmp_path / 'target-options-transfer')
    opened = bridge.execute('anatomyOpen', {'path': str(patient.root / 'patient-anatomy.json')}, _Request('open'), lambda *args: None)
    plan = bridge.execute('anatomyPlan', {'collectionId': opened['collectionId'], 'snapshotId': patient.identity(), 'action': 'list'}, _Request('plan'), lambda *args: None)
    option = next(row for row in plan['targetOptions'] if row['layerId'] == 'tumor')
    assert option['target'] in [world(transform, [3, 5, 5]).tolist(), world(transform, [7, 5, 5]).tolist()]
    assert option['entry'] == entry
    assert 'nearest' in option['focusMeaning']
    assert not (patient.root / 'anatomy-approaches.json').exists()


def test_estimated_display_alignment_does_not_authorize_patient_operations(patient, tmp_path):
    c = patient
    c.add_series(tmp_path/'mri.nii.gz', series_id='ct', modality='CT')
    transform = np.eye(4)
    transform[:3, 3] = [2, -1, 3]
    c.set_registration('ct', transform, status='unreviewed', method='test estimate')
    c.save()
    with pytest.raises(ValueError, match='needs review'):
        c.patient_affine(c.series('ct'))
    with pytest.raises(ValueError, match='No usable'):
        display_affine(c, c.series('ct'))
    displayed, estimated = display_affine(c, c.series('ct'), preview=True)
    assert estimated
    assert np.allclose(displayed, transform @ np.array(c.series('ct')['affine']))
    bridge = BridgeSession(tmp_path/'transfer')
    def call(op, args):
        return bridge.execute(op, args, _Request('test'), lambda *x: None)
    opened = call('anatomyOpen', {'path': str(c.root/'patient-anatomy.json')})
    args = {'collectionId': opened['collectionId'], 'snapshotId': c.identity(), 'seriesId': 'ct', 'layers': ['tumor', 'vessels']}
    hidden = call('anatomyView', args)
    assert hidden['unavailableLayers'] == ['tumor', 'vessels']
    preview = call('anatomyView', {**args, 'previewAlignment': True})
    assert preview['unavailableLayers'] == []
    assert [r['name'] for r in preview['compartments']] == ['tumor', 'vessels']
    assert preview['registrationStatus'] == 'alignment preview'
    assert preview['alignmentPreviews'] == ['ct']
    assert np.allclose(preview['affine'], displayed)
    # A source-space surface is transformed once into the displayed CT grid.
    from resectionlab.anatomy_geometry import mesh
    surface = preview['compartments'][0]['surface']['source']
    positions = np.fromfile(surface['path'], dtype='<f4').reshape(-1,3)
    expected = world(c.patient_affine(c.layer('tumor')), mesh(c,c.layer('tumor'))['source'])
    assert np.allclose(world(displayed, positions), expected, atol=1e-5)
    with pytest.raises(ValueError, match='needs review'):
        c.patient_affine(c.series('ct'))
    # A stale estimate cannot be admitted even in preview mode.
    c.document['transforms']['ct']['source_sha256'] = '0'*64
    with pytest.raises(ValueError, match='does not match'):
        display_affine(c,c.series('ct'),preview=True)


def test_brain_filter_removes_extracranial_display_only(patient):
    c = patient
    brain = c.layer('brain')
    mask = np.zeros(tuple(brain['data']['shape']), np.uint8)
    mask[9:16, 8:16, 8:14] = 1
    brain['data'] = c.put_array(mask)
    before = deepcopy(c.document)
    vessel = c.layer('vessels')
    full = c.mask(vessel).copy()
    result, report = brain_vessel_display(c, vessel)
    assert report['hiddenVoxels'] > 0
    assert np.array_equal(c.mask(result), full & mask.astype(bool))
    assert np.array_equal(c.mask(vessel), full)
    assert c.document == before
    # Filtering cannot create a connection between the two source branches.
    from scipy.ndimage import label
    assert label(c.mask(result), structure=np.ones((3,3,3)))[1] == 2


def test_vessel_display_filter_preserves_the_untouched_native_surface(patient, tmp_path):
    from resectionlab.anatomy_geometry import mesh, vascular_cleanup_source

    c = patient
    brain, vessel = c.layer('brain'), c.layer('vessels')
    shape = tuple(vessel['data']['shape'])
    probability = np.full(shape, .28, dtype=np.float32)
    probability[5:8, 5:8, 4:11] = .83
    probability[14:17, 13:16, 12:17] = .61
    vessel.update(data=c.put_array((probability >= .5).astype(np.uint8)), label_value=1,
                  probability=c.put_array(probability), edits=[])
    support = np.zeros(shape, np.uint8)
    support[2:11, 2:11, 2:14] = 1
    brain.update(data=c.put_array(support), label_value=1)
    filtered, report = brain_vessel_display(c, vessel)
    assert report['hiddenVoxels'] == 45
    full = mesh(c, vessel)
    result = mesh(c, filtered, display_source=vessel)
    assert str(full['surface_field']) == 'native-probability'
    assert str(result['surface_field']) == 'filtered-native-probability'
    retained_vertices = full['source'][:, 0] < 11
    assert np.array_equal(result['source'], full['source'][retained_vertices])
    assert np.allclose(result['positions'], full['positions'][retained_vertices], atol=1e-5)
    assert not np.array_equal(result['source'], mesh(c, filtered)['source'])

    companion = deepcopy(filtered)
    companion.update(id='cleanup', edits=[], provenance={'vascular_cleanup': {
        'source_layer_id': vessel['id'], 'source_mask_sha256': vessel['data']['sha256'],
        'source_probability_sha256': vessel['probability']['sha256'],
        'result_mask_sha256': filtered['data']['sha256'],
    }})
    assert vascular_cleanup_source(c, companion) is vessel
    c.document['layers'].append(companion)
    c.save('test native vessel display')
    bridge = BridgeSession(tmp_path/'vascular-display-transfer')
    opened = bridge.execute('anatomyOpen', {'path': str(c.root/'patient-anatomy.json')}, _Request('open'), lambda *a: None)
    record = bridge.execute('anatomyView', {'collectionId': opened['collectionId'], 'snapshotId': c.identity(),
                            'seriesId': vessel['source_series'], 'layers': ['cleanup']}, _Request('view'), lambda *a: None)['compartments'][0]
    assert record['surfaceDiagnostics']['surfaceField'] == 'filtered-native-probability'
    assert record['displayFilter'] is None
    raw_record = bridge.execute('anatomyView', {'collectionId': opened['collectionId'], 'snapshotId': c.identity(),
                                'seriesId': vessel['source_series'], 'layers': ['cleanup'], 'vesselSurfaceMode': 'source'}, _Request('raw-view'), lambda *a: None)['compartments'][0]
    assert raw_record['surfaceDiagnostics']['sourceSurfaceOnly']
    assert raw_record['surfaceDiagnostics']['maxDisplacementMm'] == 0
    assert raw_record['surface']['positions'] == raw_record['surface']['source']
    assert raw_record['surface']['normals'] == raw_record['surface']['sourceNormals']
    assert raw_record['surface']['source']['sha256'] == record['surface']['source']['sha256']
    assert np.array_equal(c.array(vessel['probability']), probability)
    companion['provenance']['vascular_cleanup']['source_probability_sha256'] = '0'*64
    assert vascular_cleanup_source(c, companion) is None


def test_raw_native_mesh_reopens_with_one_copy_of_the_source_positions(patient, tmp_path):
    from resectionlab.anatomy_geometry import mesh

    layer = patient.layer('vessels')
    cache_dir = tmp_path / 'raw-mesh-cache'
    first = mesh(patient, layer, conditioned=False, cache_directory=cache_dir)
    assert first['source'].size and not first['cache_hit']
    [cache] = list(cache_dir.glob('*.npz'))
    with np.load(cache, allow_pickle=False) as stored:
        assert 'source' in stored and 'positions' not in stored
    reopened = mesh(patient, layer, conditioned=False, cache_directory=cache_dir)
    assert reopened['cache_hit'] and reopened['positions'] is reopened['source']
    for key in ('positions', 'normals', 'indices'):
        np.testing.assert_array_equal(reopened[key], first[key])


def test_prepared_branch_query_preserves_source_ids_and_excludes_hidden_paths(patient):
    from resectionlab.anatomy_geometry import inspect_vessel_geometry, source_branch_cache_path
    from resectionlab.vascular_source_branches import build_source_branches

    c = patient
    source = c.layer('vessels')
    mask = np.zeros(source['data']['shape'], bool)
    mask[3:9, 4, 5] = True
    mask[14:20, 16, 17] = True
    source.update(data=c.put_array(mask.astype(np.uint8)), label_value=1,
                  probability=c.put_array(mask.astype(np.float32)), edits=[])
    candidate_mask = mask.copy()
    candidate_mask[12:] = False
    candidate = deepcopy(source)
    candidate.pop('probability')
    candidate.update(id='cleanup', data=c.put_array(candidate_mask.astype(np.uint8)))
    candidate['provenance']['vascular_cleanup'] = {
        'source_layer_id': source['id'], 'source_mask_sha256': source['data']['sha256'],
        'source_probability_sha256': source['probability']['sha256'],
        'result_mask_sha256': candidate['data']['sha256'],
    }
    c.document['layers'].append(candidate)
    scan = c.series(source['source_series'])
    graph = build_source_branches(mask, source['affine'], source_scan_sha256=scan['original']['sha256'])
    path = source_branch_cache_path(c, source)
    path.parent.mkdir()
    path.write_text(json.dumps(graph))
    first, last = [world(np.asarray(source['affine']), voxel) for voxel in ([6, 4, 5], [17, 16, 17])]
    raw_first = inspect_vessel_geometry(c, source['id'], first, scan['id'])
    raw_last = inspect_vessel_geometry(c, source['id'], last, scan['id'])
    cleaned = inspect_vessel_geometry(c, candidate['id'], last, scan['id'])
    assert cleaned['source_graph_sha256'] == raw_first['source_graph_sha256']
    assert cleaned['branches'][0]['branch_id'] == raw_first['branches'][0]['branch_id']
    assert cleaned['branches'][0]['branch_id'] != raw_last['branches'][0]['branch_id']
    assert cleaned['nearest']['distance_mm'] > raw_last['nearest']['distance_mm']
    assert candidate_mask.ravel()[cleaned['branches'][0]['source_cells']].all()


def test_pre_and_postoperative_studies_are_separate(patient, tmp_path):
    before = patient.identity()
    with pytest.raises(ValueError, match='separate patient study files'):
        patient.add_series(tmp_path/'mri.nii.gz', series_id='post', modality='T1', timepoint='postoperative')
    assert patient.identity() == before
    post = Collection.create(tmp_path/'postop', patient.document['patient_id'], timepoint='postoperative')
    post.add_series(tmp_path/'mri.nii.gz',series_id='post',modality='T1',timepoint='postoperative')
    assert post.document['patient_id'] == patient.document['patient_id']
    assert post.document['study_timepoint'] == 'postoperative'


def test_portable_study_roundtrip_preserves_sources_and_reopens_edits(patient, tmp_path):
    dest = tmp_path/'patient.rsworkspace'
    result = save_package(patient, dest)
    assert result['files'] >= 5
    manifest = open_package(dest, tmp_path/'working')
    assert manifest.read_bytes() == (patient.root/'patient-anatomy.json').read_bytes()
    restored = Collection(manifest.parent)
    assert restored.document == patient.document
    for row in restored.document['series'] + restored.document['layers']:
        assert np.array_equal(restored.array(row['data']), patient.array(row['data']))
    restored.document['missing'].append('round-trip edit')
    restored.save('test')
    assert Collection(open_package(dest,tmp_path/'working').parent).identity() == restored.identity()
    with ZipFile(dest) as archive:
        assert not any(name.startswith(('meshes/','reconstructions/')) for name in archive.namelist())


@pytest.mark.parametrize('name', ['../escape','/escape','folder/../../escape','folder\\escape'])
def test_study_archive_rejects_path_escape(patient,tmp_path,name):
    dest = tmp_path/'bad.rsworkspace'
    with ZipFile(dest,'w') as archive:
        archive.writestr('patient-anatomy.json', json.dumps(patient.document))
        archive.writestr(name, b'bad')
    with pytest.raises(ValueError,match='Unsafe'):
        open_package(dest,tmp_path/'working')


def test_study_archive_rejects_changed_sources(patient,tmp_path):
    dest = tmp_path/'bad.rsworkspace'
    save_package(patient,dest)
    other = tmp_path/'changed.rsworkspace'
    with ZipFile(dest) as source, ZipFile(other,'w') as archive:
        for name in source.namelist():
            archive.writestr(name,b'changed' if name==patient.series('mri')['data']['path'] else source.read(name))
    with pytest.raises(ValueError,match='checksum'):
        open_package(other,tmp_path/'working')


def test_header_save_updates_the_open_portable_file(patient,tmp_path):
    dest = tmp_path/'patient.rsworkspace'
    save_package(patient,dest)
    bridge = BridgeSession(tmp_path/'transfer')
    def call(op,args):
        return bridge.execute(op,args,_Request('test'),lambda *x: None)
    opened = call('anatomyOpen',{'path':str(dest),'workingRoot':str(tmp_path/'working')})
    state = {'schema':'patient-anatomy-view-v1','patientId':patient.document['patient_id'],
             'snapshotId':patient.identity(),'seriesId':'mri','vesselScope':'all',
             'states':{'mri':{'cursor':None,'visible':{'tumor':True},'settings':{}}}}
    result = call('anatomySaveView',{'collectionId':opened['collectionId'],
                  'snapshotId':patient.identity(),'viewState':state})
    assert result['portable'] is True
    with ZipFile(dest) as archive:
        assert json.loads(archive.read('workspace-view.json')) == state


@pytest.mark.parametrize('scope,version', [('all', 2), ('brain', 2), ('intracranial', 2), ('excluded', 2),
                                           ('rooted', 3), ('rooted-review', 3), ('all', 3), ('exterior-air', 3)])
def test_versioned_vessel_scope_roundtrip_preserves_anatomy(patient, scope, version):
    from resectionlab.anatomy_workspace import save_view, read_view
    before = deepcopy(patient.document)
    state = {'schema': 'patient-anatomy-view-v1', 'patientId': patient.document['patient_id'],
             'snapshotId': patient.identity(), 'seriesId': 'mri', 'vesselScope': scope,
             'vesselDisplayVersion': version, 'states': {'mri': {'cursor': None, 'visible': {}, 'settings': {}}}}
    save_view(patient, state)
    assert read_view(patient) == state
    assert patient.document == before
    for invalid in [True, 1, 4, '2', None]:
        with pytest.raises(ValueError, match='vessel display version'):
            save_view(patient, {**state, 'vesselDisplayVersion': invalid})


@pytest.mark.parametrize('scope', ['rooted', 'rooted-review'])
def test_rooted_scopes_need_the_rooted_display_version(patient, scope):
    from resectionlab.anatomy_workspace import save_view
    state = {'schema': 'patient-anatomy-view-v1', 'patientId': patient.document['patient_id'],
             'snapshotId': patient.identity(), 'seriesId': 'mri', 'vesselScope': scope,
             'vesselDisplayVersion': 2, 'states': {'mri': {'cursor': None, 'visible': {}, 'settings': {}}}}
    with pytest.raises(ValueError, match='display version 3'):
        save_view(patient, state)
    with pytest.raises(ValueError, match='vessel display scope'):
        save_view(patient, {**state, 'vesselScope': 'rooted-everything', 'vesselDisplayVersion': 3})


def test_rooted_vessel_view_hides_detached_parts_behind_a_review_overlay(tmp_path):
    import nibabel as nib
    from scipy import ndimage as ndi
    from resectionlab.vascular_cleanup import ROOTED_REVIEW_PREFIX, VESSEL_CLASS_COLORS
    shape = (48, 32, 32)
    signal = np.full(shape, 40., np.float32)
    tree = np.zeros(shape, bool); tree[4:30, 14:18, 14:18] = True
    loose = np.zeros(shape, bool); loose[40:46, 14:18, 14:18] = True
    signal[tree | loose] = 400.
    image = nib.Nifti1Image(signal, np.diag([.5, .5, .5, 1])); image.header.set_xyzt_units('mm')
    nib.save(image, tmp_path/'tof.nii.gz')
    c = Collection.create(tmp_path/'rooted', 'rooted-vessels')
    c.add_series(tmp_path/'tof.nii.gz', series_id='tof', modality='TOF')
    vessel = (tree | loose).astype(np.uint8)
    c.add_layer(vessel, np.diag([.5, .5, .5, 1]), layer_id='vessels', source_series='tof', kind='vessels',
                role='context', color='#ef5679')
    c.save(); before = deepcopy(c.document); arrays = set((c.root/'arrays').iterdir())
    bridge = BridgeSession(tmp_path/'transfer')
    def call(op, args): return bridge.execute(op, args, _Request('rooted-test'), lambda *x: None)
    opened = call('anatomyOpen', {'path': str(c.root/'patient-anatomy.json')})
    args = {'collectionId': opened['collectionId'], 'snapshotId': c.identity(), 'seriesId': 'tof', 'layers': ['vessels'],
            'previewAlignment': True}
    def read(record):
        return np.fromfile(record['mask']['path'], dtype=np.uint8).reshape(shape)
    for scope in ['rooted', 'rooted-review', 'all', 'rooted-review']:
        compartments = call('anatomyView', {**args, 'vesselScope': scope})['compartments']
        shown = read(compartments[0])
        if scope == 'all':
            assert len(compartments) == 1 and np.array_equal(shown, vessel)
            continue
        assert np.array_equal(shown > 0, tree)
        assert ndi.label(shown)[1] == 1
        report = compartments[0]['displayFilter']
        assert report['scope'] == scope and report['sourcePreserved'] is True
        assert report['excludedFragments'] == 1 and report['hiddenVoxels'] == int(loose.sum())
        assert report['rooting']['roots']['source'] == 'largest-component'
        assert report['rooting']['display']['floatingComponents'] == 0
        assert 'artery colour' in report['reason']
        assert compartments[0]['color'] == VESSEL_CLASS_COLORS['artery']
        if scope == 'rooted':
            assert len(compartments) == 1
        else:
            overlay = compartments[1]
            assert overlay['name'] == ROOTED_REVIEW_PREFIX + 'vessels' and overlay['color'] == '#ffad54'
            assert np.array_equal(read(overlay) > 0, loose)
            assert 'kind' not in overlay
    assert c.document == before and Collection(c.root).document == before
    assert np.array_equal(c.mask(c.layer('vessels')), tree | loose)
    assert arrays <= set((c.root/'arrays').iterdir())


@pytest.mark.parametrize('style,mode', [('contour', 'scan'), ('contour', 'segmentation'), ('fill', 'vascular')])
def test_anatomy_presentation_survives_native_and_portable_reopen(patient, tmp_path, style, mode):
    from resectionlab.anatomy_workspace import save_view, read_view
    before = deepcopy(patient.document)
    state = {'schema': 'patient-anatomy-view-v1', 'patientId': patient.document['patient_id'],
             'snapshotId': patient.identity(), 'seriesId': 'mri', 'vesselScope': 'all',
             'states': {'mri': {'cursor': [0., 0., 0.], 'visible': {'vessels': True, 'tumor': True},
                                'settings': {'sliceOverlay': style, 'presentation': mode,
                                             'showOrientationGuide': mode == 'segmentation',
                                             'contextStyle': 'shell', 'sliceFraming': {'fit': 'anatomy', 'zoom': 1.25},
                                             'volumeRendering': {'preset': 'auto', 'density': .085}}}}}
    if mode == 'vascular':
        state['states']['mri']['settings']['sourcedVesselContext'] = True
    save_view(patient, state)
    assert read_view(Collection(patient.root)) == state
    dest = tmp_path/'presentation.rsworkspace'
    save_package(patient, dest)
    restored = Collection(open_package(dest, tmp_path/'presentation-open').parent)
    assert read_view(restored) == state
    assert restored.document == before == patient.document
    for row in patient.document['series'] + patient.document['layers']:
        assert np.array_equal(restored.array(row['data']), patient.array(row['data']))
    for key in ('sliceOverlay', 'presentation'):
        for invalid in ('unknown', 1, None, [], {}):
            broken = deepcopy(state)
            broken['states']['mri']['settings'][key] = invalid
            with pytest.raises(ValueError, match='Invalid (slice annotation style|anatomy presentation)'):
                save_view(patient, broken)
    for invalid in (None, 0, 1, 'false', [], {}):
        broken = deepcopy(state)
        broken['states']['mri']['settings']['showOrientationGuide'] = invalid
        with pytest.raises(ValueError, match='orientation guide'):
            save_view(patient, broken)
    broken = deepcopy(state)
    broken['states']['mri']['settings']['sourcedVesselContext'] = 'true'
    with pytest.raises(ValueError, match='sourced vessel context flag'):
        save_view(patient, broken)
    assert read_view(patient) == state


@pytest.mark.parametrize('settings', [
    {'contextStyle': 'unknown'}, {'contextStyle': []},
    {'sliceFraming': None}, {'sliceFraming': []},
    {'sliceFraming': {'fit': 'brain', 'zoom': 1}},
    {'sliceFraming': {'fit': 'image', 'zoom': True}},
    {'sliceFraming': {'fit': 'image', 'zoom': .49}},
    {'sliceFraming': {'fit': 'anatomy', 'zoom': 4.01}},
    {'sliceFraming': {'fit': 'image', 'zoom': 1, 'alterSource': True}},
])
def test_view_rejects_invalid_context_and_slice_framing(patient, settings):
    from resectionlab.anatomy_workspace import save_view
    state = {'schema': 'patient-anatomy-view-v1', 'patientId': patient.document['patient_id'],
             'snapshotId': patient.identity(), 'seriesId': 'mri',
             'states': {'mri': {'cursor': None, 'visible': {}, 'settings': settings}}}
    with pytest.raises(ValueError, match='Invalid (source context style|slice framing)'):
        save_view(patient, state)


def test_full_source_preserves_continuous_vessel_across_unknown_skull_slice(tmp_path):
    import nibabel as nib
    from scipy import ndimage as ndi
    from resectionlab.anatomy_intracranial import intracranial_vessels
    x, y, z = np.indices((64, 64, 12))
    radius = np.hypot(x-32, y-32)
    ct = np.where(radius > 26, -1000., np.where(radius > 23, 1000., 30.)).astype(np.float32)
    ct[31:39, 50:, 6] = 0  # One open skull ring; source vessel stays continuous.
    for name, data in [('mri', np.ones(ct.shape, np.float32)), ('ct', ct)]:
        image = nib.Nifti1Image(data, np.eye(4)); image.header.set_xyzt_units('mm')
        nib.save(image, tmp_path/f'{name}.nii.gz')
    c = Collection.create(tmp_path/'continuous', 'continuous-vessel')
    c.add_series(tmp_path/'mri.nii.gz', series_id='mri', modality='T1CE')
    c.add_series(tmp_path/'ct.nii.gz', series_id='ct', modality='CT')
    c.set_registration('ct', np.eye(4), status='source-provided', method='fixture')
    vessel = np.zeros(ct.shape, np.uint8); vessel[31:34,31:34,1:11] = 1
    c.add_layer(vessel,np.eye(4),layer_id='vessels',source_series='mri',kind='vessels',role='critical',color='#ef5679',probability=vessel*.8+.1)
    c.add_layer((radius<19).astype(np.uint8),np.eye(4),layer_id='brain',source_series='mri',kind='brain',role='context',color='#91c4b6')
    c.save(); before = deepcopy(c.document)
    strict, report = intracranial_vessels(c,c.layer('vessels'))
    assert report['uncertainVoxels'] > 0
    assert report['outsideVoxels'] == report['boneConflictVoxels'] == 0
    assert ndi.label(c.mask(strict))[1] == 2
    bridge = BridgeSession(tmp_path/'transfer')
    def call(op, args): return bridge.execute(op,args,_Request('gap-test'),lambda *x:None)
    opened = call('anatomyOpen',{'path':str(c.root/'patient-anatomy.json')})
    args = {'collectionId':opened['collectionId'],'snapshotId':c.identity(),'seriesId':'mri','layers':['vessels'],'previewAlignment':True}
    for scope in ['intracranial','all','intracranial','all']:
        result = call('anatomyView',{**args,'vesselScope':scope})['compartments'][0]
        shown = np.fromfile(result['mask']['path'],dtype=np.uint8).reshape(vessel.shape)
        if scope == 'all':
            assert np.array_equal(shown,vessel)
            assert ndi.label(shown)[1] == 1
            assert result['displayFilter'] is None
        else:
            assert np.array_equal(shown,c.mask(strict))
            assert 'Unknown skull coverage' in result['displayFilter']['reason']
    assert c.document == before
    assert Collection(c.root).document == before


def test_named_tract_display_and_selected_approach_survive_native_view_roundtrip(patient, tmp_path):
    import nibabel as nib
    from resectionlab.anatomy_import import import_anatomy_derivative
    from resectionlab.anatomy_workspace import save_view, read_view
    points = world(patient.patient_affine(patient.series('mri')), np.array([[8, 8, 8], [12, 12, 12]]))
    filename = tmp_path / 'named.trk'
    nib.streamlines.save(nib.streamlines.Tractogram([points], affine_to_rasmm=np.eye(4)), filename)
    tract = import_anatomy_derivative(patient, filename, source_series='mri', kind='tractography', label='Supplied CST')
    patient.save()
    identity = patient.identity()
    state = {'schema': 'patient-anatomy-view-v1', 'patientId': patient.document['patient_id'],
             'snapshotId': identity, 'seriesId': 'mri', 'selectedStructure': tract['id'],
             'selectedApproachId': 'manual-source-path', 'states': {'mri': {
                 'cursor': points[0].tolist(), 'visible': {tract['id']: False, 'brain': True},
                 'settings': {'layerOpacity': {tract['id']: .35}, 'layerColor': {tract['id']: '#65bfd3'}}}}}
    save_view(patient, state)
    assert read_view(Collection(patient.root)) == state
    assert patient.identity() == identity
    with pytest.raises(ValueError, match='visibility'):
        save_view(patient, {**state, 'states': {'mri': {**state['states']['mri'], 'visible': {'invented-tract': True}}}})


def test_rooted_display_binds_supplied_topbrain_ct_labels_and_never_on_mr(tmp_path):
    import nibabel as nib
    from resectionlab.vascular_cleanup import rooted_vessel_layer
    shape, frame = (40, 24, 24), np.diag([.5, .5, .5, 1])
    labels = np.zeros(shape, np.int16)
    labels[2:20, 10:14, 10:14] = 4    # R-ICA, a root
    labels[20:26, 10:14, 10:14] = 5   # R-M1, attached to it
    labels[33:38, 10:14, 10:14] = 17  # R-M2, more than 3 mm from the tree
    labels[2:10, 2:5, 2:5] = 36       # Straight sinus, outside the arterial grouping
    for name, data in (('scan', np.where(labels > 0, 400., 40.).astype(np.float32)), ('topcow_ct_001', labels)):
        image = nib.Nifti1Image(data, frame); image.header.set_xyzt_units('mm')
        nib.save(image, tmp_path/f'{name}.nii.gz')
    arteries = (labels > 0) & (labels <= 34)
    for modality, expected in (('CTA', 'named'), ('TOF', 'largest-component')):
        c = Collection.create(tmp_path/modality, f'topbrain-{modality.lower()}')
        c.add_series(tmp_path/'scan.nii.gz', series_id='scan', modality=modality)
        c.series('scan')['publisher_dataset'] = 'https://zenodo.org/records/21972006'
        supplied = c.original(tmp_path/'topcow_ct_001.nii.gz')
        c.add_layer(arteries.astype(np.uint8), frame, layer_id='arteries', source_series='scan', kind='artery',
                    role='context', color='#f06455',
                    provenance={'kind': 'supplied_annotation', 'source': supplied, 'source_label_values': list(range(1, 35))})
        before = deepcopy(c.document)
        display, receipt, review, split = rooted_vessel_layer(c, c.layer('arteries'), review=True)
        assert split is None
        rooting = receipt['rooting']
        assert rooting['roots']['source'] == expected
        assert np.array_equal(c.mask(display) | c.mask(review), arteries)
        assert not (c.mask(display) & c.mask(review)).any()
        if modality == 'CTA':
            assert [row['name'] for row in rooting['roots']['labels']] == ['R-ICA']
            rows = {row['name']: row for row in rooting['namedLabels']}
            assert rows['R-ICA']['status'] == rows['R-M1']['status'] == 'shown'
            assert rows['R-M2']['status'] == 'review' and 'StS' not in rows
            assert np.array_equal(c.mask(review), labels == 17)
        else:
            assert 'never applied' in receipt['namedLabelsUnavailable']
            assert rooting['namedLabels'] == []
        assert c.document == before


def _topbrain_ct_study(tmp_path, name, labels, *, modality='CTA', dictionary=None, original_name='topcow_ct_001'):
    import nibabel as nib
    frame = np.diag([.5, .5, .5, 1])
    folder = tmp_path / f'{name}-sources'
    folder.mkdir()
    for stem, data in (('scan', np.where(labels > 0, 400., 40.).astype(np.float32)), (original_name, labels)):
        image = nib.Nifti1Image(data, frame); image.header.set_xyzt_units('mm')
        nib.save(image, folder / f'{stem}.nii.gz')
    c = Collection.create(tmp_path / name, name)
    c.add_series(folder / 'scan.nii.gz', series_id='scan', modality=modality)
    c.series('scan')['publisher_dataset'] = 'https://zenodo.org/records/21972006'
    supplied = c.original(folder / f'{original_name}.nii.gz')
    provenance = {'kind': 'supplied_annotation', 'source': supplied, 'source_label_values': list(range(1, 35))}
    if dictionary:
        provenance['publisher_label_dictionary'] = dictionary
    c.add_layer(((labels > 0) & (labels <= 34)).astype(np.uint8), frame, layer_id='arteries', source_series='scan',
                kind='artery', role='context', color='#f06455', provenance=provenance)
    return c


def _two_class_labels():
    labels = np.zeros((40, 24, 24), np.uint8)
    labels[2:20, 10:14, 10:14] = 4     # R-ICA
    labels[20:26, 10:14, 10:14] = 5    # R-M1
    labels[33:38, 10:14, 10:14] = 17   # R-M2, detached
    labels[2:30, 2:5, 2:5] = 36        # Straight sinus
    return labels


def test_rooted_display_splits_vein_voxels_of_a_mixed_layer_into_their_own_compartment(tmp_path):
    from resectionlab.vascular_cleanup import VESSEL_CLASS_COLORS, rooted_vessel_layer
    labels = _two_class_labels()
    c = _topbrain_ct_study(tmp_path, 'mixed', labels)
    row = c.add_layer((labels > 0).astype(np.uint8), np.diag([.5, .5, .5, 1]), layer_id='ensemble',
                      source_series='scan', kind='vessels', role='context', color='#ef5679')
    row['named_labels'] = {'data': c.put_array(labels), 'scheme': 'topbrain-v1-ct'}
    display, receipt, review, split = rooted_vessel_layer(c, row)
    assert np.array_equal(c.mask(display), np.isin(labels, (4, 5)))
    assert np.array_equal(c.mask(split), labels == 36)
    assert split['kind'] == 'vein' and split['color'] == VESSEL_CLASS_COLORS['vein']
    assert display['color'] == VESSEL_CLASS_COLORS['artery'] and review is None
    assert receipt['rooting']['classes']['vein'] == int((labels == 36).sum())
    # A venous layer keeps its veins and splits off any arteries.
    row['kind'] = 'vein'
    display, _, _, split = rooted_vessel_layer(c, row)
    assert np.array_equal(c.mask(display), labels == 36) and split['kind'] == 'artery'


def test_rooted_view_falls_back_to_the_source_when_the_derivative_fails(tmp_path):
    labels = _two_class_labels()
    c = _topbrain_ct_study(tmp_path, 'failing', labels)
    bad = labels.copy(); bad[bad == 17] = 99  # Outside the TopBrain CT label map.
    row = c.add_layer((labels > 0).astype(np.uint8), np.diag([.5, .5, .5, 1]), layer_id='ensemble',
                      source_series='scan', kind='vessels', role='context', color='#ef5679')
    row['named_labels'] = {'data': c.put_array(bad), 'scheme': 'topbrain-v1-ct'}
    c.save()
    bridge = BridgeSession(tmp_path / 'transfer')
    opened = bridge.execute('anatomyOpen', {'path': str(c.root / 'patient-anatomy.json')}, _Request('open'), lambda *x: None)
    view = bridge.execute('anatomyView', {'collectionId': opened['collectionId'], 'snapshotId': c.identity(),
                                          'seriesId': 'scan', 'layers': ['ensemble', 'arteries'],
                                          'previewAlignment': True, 'vesselScope': 'rooted'},
                          _Request('view'), lambda *x: None)
    records = {record['name']: record for record in view['compartments']}
    failed = records['ensemble']['displayFilter']
    assert failed['failed'] is True and 'Named labels outside' in failed['reason'] and 'full source' in failed['reason']
    shown = np.fromfile(records['ensemble']['mask']['path'], dtype=np.uint8).reshape(labels.shape)
    assert np.array_equal(shown > 0, labels > 0)
    assert records['arteries']['displayFilter']['rooting']['roots']['source'] == 'named'


def test_rooted_derived_compartments_never_exceed_the_viewer_layer_budget(tmp_path, monkeypatch):
    import resectionlab.anatomy_bridge as anatomy_bridge
    labels = _two_class_labels()
    c = _topbrain_ct_study(tmp_path, 'budget', labels)
    c.save()
    bridge = BridgeSession(tmp_path / 'transfer')
    opened = bridge.execute('anatomyOpen', {'path': str(c.root / 'patient-anatomy.json')}, _Request('open'), lambda *x: None)
    args = {'collectionId': opened['collectionId'], 'snapshotId': c.identity(), 'seriesId': 'scan',
            'layers': ['arteries'], 'previewAlignment': True, 'vesselScope': 'rooted-review'}
    full = bridge.execute('anatomyView', args, _Request('full'), lambda *x: None)
    assert [r['name'] for r in full['compartments']] == ['arteries', '__vessel_review:arteries']
    monkeypatch.setattr(anatomy_bridge, 'MAX_DISPLAY_COMPARTMENTS', 1)
    capped = bridge.execute('anatomyView', args, _Request('capped'), lambda *x: None)
    assert [r['name'] for r in capped['compartments']] == ['arteries']
    assert capped['omittedDerivedCompartments'] == 1


def test_a_tumour_on_another_registered_series_enables_the_local_tier(tmp_path):
    import nibabel as nib
    from resectionlab.vascular_cleanup import rooted_vessel_layer
    labels = _two_class_labels()
    c = _topbrain_ct_study(tmp_path, 'tumour', labels)
    t1 = np.zeros((20, 12, 12), np.float32)
    image = nib.Nifti1Image(t1, np.diag([1., 1., 1., 1])); image.header.set_xyzt_units('mm')
    nib.save(image, tmp_path / 't1.nii.gz')
    c.add_series(tmp_path / 't1.nii.gz', series_id='t1', modality='T1CE')
    tumour = np.zeros(t1.shape, np.uint8); tumour[16:18, 5:7, 5:7] = 1
    c.add_layer(tumour, np.diag([1., 1., 1., 1]), layer_id='lesion', source_series='t1', kind='tumor',
                role='target', color='#ff00ff')
    _, receipt, _, _ = rooted_vessel_layer(c, c.layer('arteries'))
    lod = receipt['rooting']['lod']
    assert lod['tumourLayers'] == [] and 'no registration' in lod['tumourReason']
    c.set_registration('t1', np.eye(4), status='source-provided', method='fixture')
    _, receipt, _, _ = rooted_vessel_layer(c, c.layer('arteries'))
    lod = receipt['rooting']['lod']
    assert lod['tumourLayers'] == ['lesion'] and lod['tumourReason'] is None and lod['tumour'] is True


def test_supplied_labels_whose_dictionary_and_file_name_disagree_are_refused(tmp_path):
    from resectionlab.vascular_cleanup import rooted_vessel_layer
    labels = _two_class_labels()
    c = _topbrain_ct_study(tmp_path, 'disagree', labels, original_name='topcow_mr_001',
                           dictionary='TopBrain/labelmap_jsons/labels_topbrain_v1_ct.json')
    _, receipt, _, _ = rooted_vessel_layer(c, c.layer('arteries'))
    assert 'disagree' in receipt['namedLabelsUnavailable']
    assert receipt['rooting']['roots']['source'] == 'largest-component'


def test_a_missing_label_original_is_not_cached_for_the_session(tmp_path):
    from resectionlab.vascular_cleanup import rooted_vessel_layer
    labels = _two_class_labels()
    c = _topbrain_ct_study(tmp_path, 'missing', labels)
    original = c.path(c.layer('arteries')['provenance']['source']['path'])
    moved = original.with_suffix('.away')
    original.rename(moved)
    _, receipt, _, _ = rooted_vessel_layer(c, c.layer('arteries'))
    assert 'missing' in receipt['namedLabelsUnavailable']
    assert not getattr(c, '_rooted_display_cache', {})
    moved.rename(original)
    _, receipt, _, _ = rooted_vessel_layer(c, c.layer('arteries'))
    assert receipt['rooting']['roots']['source'] == 'named' and len(c._rooted_display_cache) == 1
