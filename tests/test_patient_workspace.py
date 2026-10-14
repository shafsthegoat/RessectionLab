"""Source identity, portable originals and display settings are persistence contracts."""
from dataclasses import replace
from hashlib import sha256
import copy
import json
from zipfile import ZipFile

import nibabel as nib
import numpy as np
import pytest

from resectionlab.imaging import load_nifti_case
from resectionlab.workspace_bundle import OriginalFile, load_workspace, save_workspace, view_state
from resectionlab.workspace_imaging import DisplaySource, source_registration


def image(path,offset=0):
    affine=np.diag([1.,2.,3.,1.]);affine[0,3]=offset
    data=np.arange(120,dtype=np.float32).reshape((4,5,6))
    value=nib.Nifti1Image(data,affine);value.header.set_xyzt_units('mm')
    value.set_qform(affine,1);value.set_sform(affine,1);nib.save(value,path)
    return load_nifti_case(path,case_id='generated-persistence-fixture')


def registered(primary,auxiliary):
    declaration={'kind':'source-provided-coregistration','patient_id':primary.case_id,'group_id':'generated-test-group',
        'evidence_sha256':'a'*64,'evidence_url':'https://example.invalid/fixture-registration',
        'source_sha256_by_modality':{'T1CE':primary.source_refs[0].sha256,'T2':auxiliary.source_refs[0].sha256}}
    return (replace(primary,metadata={**primary.metadata,'selected_modality':'T1CE','workspace_registration':declaration}),
            replace(auxiliary,metadata={**auxiliary.metadata,'selected_modality':'T2','workspace_registration':declaration}))


def test_original_bytes_and_viewer_preferences_survive_removed_paths(tmp_path):
    path=tmp_path/'image.nii.gz';case=image(path)
    original=OriginalFile.from_path(path,role='imaging');state=view_state(case,[])
    state['states']['primary']['viewer']={'camera':{'position':[100,100,100],'target':[1,3,5],'up':[0,0,1]},
        'clipping':{'axis':'x','value':1.5,'enabled':True,'flipped':False},
        'fusion':{'seriesId':None,'opacity':.3},'layout':'3d-focus','activePlane':'coronal','showPlane':False,'contrast':1.2}
    bundle=tmp_path/'case.ressectionlab'
    identity=save_workspace(case,bundle,sources=[],imaging=state,episode=None,selection=None,artifacts={},originals=[original])
    path.unlink()
    reopened,_,saved=load_workspace(bundle)
    assert reopened.semantic_hash==case.semantic_hash and saved['sessionHash']==identity
    assert saved['imagingState']==state and saved['originals']==[original]
    second=tmp_path/'second.ressectionlab'
    assert save_workspace(reopened,second,sources=saved['sources'],imaging=saved['imagingState'],episode=None,selection=None,artifacts={},originals=saved['originals'])==identity
    with ZipFile(second) as archive:
        assert archive.read('workspace/original-0.bin')==original.data


def test_original_byte_corruption_refuses_even_if_zip_crc_is_recomputed(tmp_path):
    path=tmp_path/'image.nii.gz';case=image(path);bundle=tmp_path/'case.ressectionlab'
    save_workspace(case,bundle,sources=[],imaging=None,episode=None,selection=None,artifacts={},originals=[OriginalFile.from_path(path,role='imaging')])
    with ZipFile(bundle) as archive:members={name:archive.read(name) for name in archive.namelist()}
    payload=members['workspace/original-0.bin'];members['workspace/original-0.bin']=b'X'+payload[1:]
    with ZipFile(bundle,'w') as archive:
        for name,payload in members.items():archive.writestr(name,payload)
    with pytest.raises(ValueError,match='Original source bytes'):load_workspace(bundle)


def test_equal_grid_alone_never_enables_fusion(tmp_path):
    first=image(tmp_path/'one.nii.gz');second=image(tmp_path/'two.nii.gz')
    assert not source_registration(first,second)[1]['overlayPermitted']
    source=DisplaySource(second,'T2','none');state=view_state(first,[source])
    state['states']['primary']['viewer']={'fusion':{'seriesId':source.identity(first),'opacity':.5}}
    with pytest.raises(ValueError,match='explicitly source-registered'):view_state(first,[source],state)


def test_declaration_and_actual_hashes_and_frames_are_all_required(tmp_path):
    first,second=registered(image(tmp_path/'one.nii.gz'),image(tmp_path/'two.nii.gz'))
    association,registration=source_registration(first,second)
    assert association['samePersonVerified'] and not association['sameTimeVerified']
    assert registration['status']=='source-provided' and registration['overlayPermitted']
    for changed in [replace(second,case_id='different-patient'),replace(second,affine=second.affine+np.diag([0,0,.1,0])),
                    replace(second,source_refs=(replace(second.source_refs[0],sha256='0'*64),))]:
        assert not source_registration(first,changed)[1]['overlayPermitted']
    source=DisplaySource(second,'T2','none');state=view_state(first,[source])
    state['states']['primary']['viewer']={'fusion':{'seriesId':source.identity(first),'opacity':.5}}
    assert view_state(first,[source],state)==state


@pytest.mark.parametrize('viewer',[
    {'clipping':{'axis':'x','value':1000,'enabled':True}},
    {'layerOpacity':{'not-a-layer':.5}}, {'layerOpacity':{} ,'contrast':float('nan')},
    {'camera':{'position':[0,0,0],'target':[0,0,0],'up':[0,0,1]}},
    {'camera':{'position':[0,0,1],'target':[0,0,0],'up':[0,0,0]}},
    {'layout':'unsupported'}, {'showPlane':1}, {'activePlane':'oblique'},
])
def test_invalid_display_settings_refuse_without_materializing_a_bundle(tmp_path,viewer):
    case=image(tmp_path/'one.nii.gz');state=view_state(case,[]);state['states']['primary']['viewer']=viewer
    with pytest.raises(ValueError):view_state(case,[],state)


def test_brain_visibility_requires_explicit_anatomy_metadata(tmp_path):
    case=image(tmp_path/'one.nii.gz');mask=np.ones(case.mri.shape,dtype=bool)
    legacy=replace(case,brain_mask=mask)
    assert 'brain' not in view_state(legacy,[])['states']['primary']['visibleLayers']
    workbench=replace(legacy,metadata={**legacy.metadata,'anatomy_layers':{'brain':{'provenance':'estimated'}}})
    state=view_state(workbench,[]);assert state['states']['primary']['visibleLayers']['brain']
    state['states']['primary']['visibleLayers']['brain']=False
    assert view_state(workbench,[],state)==state


def model_fixture(tmp_path):
    image_path=tmp_path/'ct.nii.gz';case=image(image_path)
    mask_path=tmp_path/'skull_mask.nii.gz';values=(case.mri>50).astype(np.uint8)
    mask=nib.Nifti1Image(values,case.affine);mask.header.set_xyzt_units('mm');mask.set_qform(case.affine,1);mask.set_sform(case.affine,1);nib.save(mask,mask_path)
    weights=tmp_path/'generated-test-weights.bin';weights.write_bytes(b'generated unit test weight bytes; not an actual model')
    digest=lambda p:sha256(p.read_bytes()).hexdigest()
    report={'schema':'resectionlab-poc-inference-v1','status':'succeeded','case':case.case_id,'cache_key':'generated-fixture-cache',
        'model':{'name':'generated-unit-test-only','version':'1','license':'test'},
        'weights':[{'path':str(weights),'sha256':digest(weights),'bytes':weights.stat().st_size}],
        'input':{'path':str(image_path),'sha256':digest(image_path),'modality':'CT','shape':list(case.mri.shape),'affine_ras_mm':case.affine.tolist()},
        'outputs':[{'path':str(mask_path),'sha256':digest(mask_path),'name':mask_path.name}], 'elapsed_seconds':.01}
    report_path=tmp_path/'inference.json';report_path.write_text(json.dumps(report))
    metadata={'case_id':case.case_id,'source_url':'https://example.invalid/generated-fixture','source_license':'test',
        'source_name':'generated unit test only','source_image_sha256':digest(image_path)}
    return report_path,metadata,[image_path,mask_path,weights]


def test_model_case_keeps_estimated_layer_and_reopens_without_weights_or_sources(tmp_path):
    from resectionlab.patient_workspace import prepare_model_workspace
    report,metadata,paths=model_fixture(tmp_path);bundle=tmp_path/'model.ressectionlab'
    result=prepare_model_workspace(report,bundle,source_metadata=metadata)
    for path in [report,*paths]:path.unlink()
    case,_,saved=load_workspace(bundle)
    assert result['reopened_identical'] and result['reinference_performed'] is False
    assert case.metadata['anatomy_workbench_only'] is True
    assert case.metadata['anatomy_layers']['skull']['provenance']=='estimated'
    assert next(ref for ref in case.source_refs if ref.source_id=='model_skull').provenance=='estimated'
    assert case.compartments['skull'].any() and len(saved['originals'])==3


def test_model_case_rejects_other_patient_or_other_acquisition(tmp_path):
    from resectionlab.patient_workspace import prepare_model_workspace
    report,metadata,_=model_fixture(tmp_path)
    for changed in [{**metadata,'case_id':'different-case'},{**metadata,'source_image_sha256':'0'*64}]:
        with pytest.raises(ValueError,match='patient|hash'):
            prepare_model_workspace(report,tmp_path/'refused.ressectionlab',source_metadata=changed)
    assert not (tmp_path/'refused.ressectionlab').exists()
