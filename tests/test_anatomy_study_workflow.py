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


@pytest.mark.parametrize('scope', ['all', 'brain', 'intracranial', 'excluded'])
def test_versioned_vessel_scope_roundtrip_preserves_anatomy(patient, scope):
    from resectionlab.anatomy_workspace import save_view, read_view
    before = deepcopy(patient.document)
    state = {'schema': 'patient-anatomy-view-v1', 'patientId': patient.document['patient_id'],
             'snapshotId': patient.identity(), 'seriesId': 'mri', 'vesselScope': scope,
             'vesselDisplayVersion': 2, 'states': {'mri': {'cursor': None, 'visible': {}, 'settings': {}}}}
    save_view(patient, state)
    assert read_view(patient) == state
    assert patient.document == before
    for invalid in [True, 1, 3, '2', None]:
        with pytest.raises(ValueError, match='vessel display version'):
            save_view(patient, {**state, 'vesselDisplayVersion': invalid})


@pytest.mark.parametrize('style,mode', [('contour', 'segmentation'), ('fill', 'vascular')])
def test_anatomy_presentation_survives_native_and_portable_reopen(patient, tmp_path, style, mode):
    from resectionlab.anatomy_workspace import save_view, read_view
    before = deepcopy(patient.document)
    state = {'schema': 'patient-anatomy-view-v1', 'patientId': patient.document['patient_id'],
             'snapshotId': patient.identity(), 'seriesId': 'mri', 'vesselScope': 'all',
             'states': {'mri': {'cursor': [0., 0., 0.], 'visible': {'vessels': True, 'tumor': True},
                                'settings': {'sliceOverlay': style, 'presentation': mode,
                                             'volumeRendering': {'preset': 'auto', 'density': .085}}}}}
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
    assert read_view(patient) == state


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
