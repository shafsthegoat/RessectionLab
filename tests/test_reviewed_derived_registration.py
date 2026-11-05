"""Reviewed physical derivations differ from supplied upstream registration."""
from dataclasses import replace
import copy
import numpy as np
import nibabel as nib
import pytest

from resectionlab.imaging import load_nifti_case
from resectionlab.workspace_imaging import DisplaySource, source_registration
from resectionlab.workspace_bundle import load_workspace, save_workspace, view_state
from resectionlab.core import thaw_json


def fixture(tmp_path):
    cases=[]
    for index in range(2):
        path=tmp_path/f'image-{index}.nii.gz'
        image=nib.Nifti1Image((np.arange(120,dtype=np.float32).reshape(4,5,6)+index),np.diag([-1.,2.,3.,1.]))
        image.header.set_xyzt_units('mm');nib.save(image,path)
        cases.append(load_nifti_case(path,case_id='IXI-test-integrated'))
    forward=np.eye(4);forward[:2,:2]=[[0.,-1.],[1.,0.]];forward[:3,3]=[12.,-4.,7.]
    declaration={'kind':'independently-reviewed-derived-registration-v1','patient_id':cases[0].case_id,
        'source_person_id':'IXI:test','group_id':'generated-alignment-control',
        'source_sha256_by_modality':dict(zip(['T1','MRA registered'],[c.source_refs[0].sha256 for c in cases])),
        'evidence_sha256':'a'*64,'evidence_path':'qc/independent/generated-review.json','reviewer':'generated regression fixture',
        'review_decision':'accepted_for_registered_overlay_inspection_with_alignment_uncertainty',
        'method_report_sha256':'b'*64,'original_moving_image_sha256':'c'*64,'original_annotation_sha256':'d'*64,
        'original_moving_to_reference_ras_mm':forward.tolist(),'reference_to_original_moving_ras_mm':np.linalg.inv(forward).tolist(),
        'registered_grid':'reference-native-grid'}
    return tuple(replace(c,metadata={'selected_modality':modality,'workspace_registration':copy.deepcopy(declaration),
        'data_case':{'person_id':'IXI:test'}}) for c,modality in zip(cases,['T1','MRA registered']))


def test_reviewed_derivative_uses_identity_after_resampling_and_survives_native(tmp_path):
    primary,derived=fixture(tmp_path)
    association,result=source_registration(primary,derived)
    assert result['status']=='verified' and result['overlayPermitted'] and not association['sameTimeVerified']
    assert np.array_equal(result['sourceToReferenceRasMm'],np.eye(4)) # Do not apply the original rigid transform twice.
    assert 'uncertainty' in result['reason'] and 'unsupported' in result['reason']
    source=DisplaySource(derived,'MRA','none');state=view_state(primary,[source])
    state['states']['primary']['viewer']={'fusion':{'seriesId':source.identity(primary),'opacity':.3}}
    path=tmp_path/'native.ressectionlab'
    save_workspace(primary,path,sources=[source],imaging=state,episode=None,selection=None,artifacts={})
    reopened,_,saved=load_workspace(path)
    assert source_registration(reopened,saved['sources'][0].case)[1]==result


@pytest.mark.parametrize('fault',['inverse','shear','person','hash','review','evidence','deformation'])
def test_unreviewed_stale_or_wrong_physical_derivations_never_enable_overlay(tmp_path,fault):
    primary,derived=fixture(tmp_path)
    metadata=thaw_json(derived.metadata);declaration=copy.deepcopy(metadata['workspace_registration'])
    if fault=='inverse':declaration['reference_to_original_moving_ras_mm'][0][3]+=1
    elif fault=='shear':declaration['original_moving_to_reference_ras_mm'][0][1]+=.2
    elif fault=='person':metadata['data_case']={'person_id':'other'}
    elif fault=='hash':declaration['source_sha256_by_modality']['MRA registered']='0'*64
    elif fault=='review':declaration['review_decision']='pending'
    elif fault=='evidence':declaration['evidence_path']='../outside.json'
    elif fault=='deformation':metadata['data_case']={'person_id':'IXI:test','deformation':{'unreviewed':'field'}}
    metadata['workspace_registration']=declaration
    # For matrix faults also mutate the shared declaration: equality alone is insufficient.
    if fault in {'inverse','shear'}:
        primary=replace(primary,metadata={**dict(primary.metadata),'workspace_registration':declaration})
    assert not source_registration(primary,replace(derived,metadata=metadata))[1]['overlayPermitted']
