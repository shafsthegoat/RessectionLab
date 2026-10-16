"""Small generated fixtures exercise software gates; no patient imaging is read."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from resectionlab import patient_corpus as pc
from resectionlab.core import semantic_digest
from resectionlab.imaging import file_sha256, load_nifti_case, save_case
from resectionlab.workspace_bundle import OriginalFile, save_workspace
from resectionlab.workspace_imaging import DisplaySource


def nifti(path, values):
    affine=np.diag([2.,3.,4.,1.]);affine[:3,3]=[-10.,2.,-5.]
    image=nib.Nifti1Image(values,affine);image.header.set_xyzt_units('mm')
    image.set_qform(affine,1);image.set_sform(affine,1);nib.save(image,path)
    return path


def fixture(root, number='05'):
    """Canonical IDs below exercise only generated test input, not actual people."""
    folder=root/number;folder.mkdir()
    image=nifti(folder/'image.nii.gz',np.arange(120,dtype=np.float32).reshape(4,5,6))
    labels=np.zeros((4,5,6),np.uint8);labels[2:4,2,3]=1
    annotation=nifti(folder/'annotation.nii.gz',labels)
    patient='BTC:sub-PAT'+number;case_id='BTC-ds001226-sub-PAT'+number+'-preop'
    case=load_nifti_case(image,annotation,case_id=case_id,label_map={1:'target'},license='CC0-1.0',metadata={
        'selected_modality':'T1','split':{'role':'TRAIN','patient_group':patient},
        'anatomy_layers':{'target':{'kind':'tumor','provenance':'source_provided',
            'source_role':'imported_source_annotation','source_sha256':file_sha256(annotation),'review_status':'unreviewed'},
            'brain':{'kind':'brain','provenance':'model_inference_unreviewed','review_status':'unreviewed',
                'model':{'name':'generated test model','license':'CC0-1.0'}}}})
    brain=np.zeros(labels.shape,bool);brain[2,2,3]=True
    case=replace(case,brain_mask=brain)
    path=folder/('BTC-sub-PAT'+number+'.ressectionlab');save_case(case,path)
    table=folder/'participants.tsv';table.write_text('participant_id\nsub-PAT'+number+'\n')
    candidate={'patient_group':patient,'case_id':case_id,'role':'TRAIN','existing_materialized':True,
        'bundle':str(path.relative_to(root)),'rights':{'license':'CC0-1.0','source':'https://example.org/generated-software-fixture'},
        'participant':{'path':str(table.relative_to(root)),'sha256':file_sha256(table),'delimiter':'\t',
            'key':'participant_id','value':'sub-PAT'+number},
        'source_files':[{'path':str(p.relative_to(root)),'sha256':file_sha256(p),'role':role}
            for p,role in [(image,'imaging'),(annotation,'annotation')]]}
    return candidate,case,path


def plan(*candidates):
    document={'schema':pc.VERSION+'.plan','candidates':list(candidates),'metadata_evidence':[],
        'existing_materialized_people':len(candidates),'new_patient_acquisition_allowed':False}
    document['plan_hash']=semantic_digest(document)
    return document


def catalog(output):return json.loads((output/'catalog.json').read_text())


def forbidden(*args,**kwargs):raise AssertionError('Protected or cached numeric payload was opened')


@pytest.mark.parametrize('identity,role',[
    ('BTC:sub-PAT26','TRAIN'),('BTC:sub-PAT27','SELECT'),('BTC:sub-PAT29','TRAIN'),
    ('BTC:sub-PAT31','TRAIN'),('UPENN:001','TRAIN'),('UTSW:001','TRAIN'),
    ('ReMIND:013','TRAIN'),('Lausanne:sub-001','TRAIN'),('BTC:sub-PAT05','DEVELOPMENT')])
def test_protected_or_nonallowlisted_candidate_refuses_before_payload(tmp_path,monkeypatch,identity,role):
    candidate={'patient_group':identity,'role':role,'bundle':'must-not-be-opened',
        'existing_materialized':True,'rights':{'license':'CC0-1.0','source':'https://example.org'}}
    monkeypatch.setattr(pc,'_manifest_preflight',forbidden)
    with pytest.raises(pc.QualificationError):pc.run_qualification(tmp_path,tmp_path/'out',plan(candidate))
    assert not (tmp_path/'out').exists()


def test_missing_rights_and_duplicate_people_refuse_metadata_only(tmp_path):
    candidate,_,_=fixture(tmp_path)
    absent=deepcopy(candidate);absent['rights']={}
    with pytest.raises(pc.QualificationError,match='rights'):pc.validate_plan(plan(absent))
    with pytest.raises(pc.QualificationError,match='different people'):pc.validate_plan(plan(candidate,candidate))
    assert pc.canonical_patient('BTC-sub-PAT005')=='BTC:sub-PAT05'
    assert pc.canonical_patient('UCSF-PDGM-0004')=='UCSF-PDGM:004'


@pytest.mark.parametrize('nested',[{'split':{'role':'SELECT'}},{'old_record':{'splitRole':'MEASUREMENT_EVAL'}},
    {'parent':{'partition':{'name':'held-out'}}},{'parent':{'patient_group':'BTC:sub-PAT26'}},
    {'ancestry':[{'development_role':'unopened_transfer'}]}, {'roles':{'SELECT':True}},
    {'split':{'SELECT':True}}, {'roles':{'held_out':True}}, {'source_collection':{'name':'UPENN-GBM'}},
    {'dataset_name':'UTSW'}, {'source_dataset':{'id':'UPENN-GBM'}}, {'source_role':'SELECT'}])
def test_nested_protected_metadata_refuses_before_any_arrays(tmp_path,monkeypatch,nested):
    candidate,case,path=fixture(tmp_path)
    save_case(replace(case,metadata={**dict(case.metadata),'parent':nested}),path)
    monkeypatch.setattr(pc,'_case_layout',forbidden);monkeypatch.setattr(pc,'load_workspace',forbidden)
    result=pc.run_qualification(tmp_path,tmp_path/'out',plan(candidate))
    assert result['counts']['rejected']==1
    assert result['rejections'][0]['failure_code'] in {'PROTECTED_ROLE','PATIENT_MISMATCH'}


def test_explicit_false_protected_role_flag_does_not_activate_that_role(tmp_path):
    candidate,case,path=fixture(tmp_path)
    save_case(replace(case,metadata={**dict(case.metadata),'roles':{'SELECT':False,'TRAIN':True},
        'split':{'SELECT':False,'TRAIN':True}}),path)
    result=pc.run_qualification(tmp_path,tmp_path/'out',plan(candidate))
    assert result['counts']['qualified_existing_people']==1,result


def test_symlink_to_protected_bundle_refuses_before_zip(tmp_path,monkeypatch):
    candidate,_,path=fixture(tmp_path);path.unlink()
    path.symlink_to(tmp_path/'BTC-sub-PAT26.ressectionlab')
    monkeypatch.setattr(pc,'_manifest_preflight',forbidden)
    result=pc.run_qualification(tmp_path,tmp_path/'out',plan(candidate))
    assert result['rejections'][0]['failure_code']=='PROTECTED_PATH'


def test_qualification_records_presence_unknowns_and_source_union_without_approval(tmp_path):
    candidate,_,_=fixture(tmp_path);out=tmp_path/'out'
    result=pc.run_qualification(tmp_path,out,plan(candidate))
    assert result['counts']['qualified_existing_people']==1,result
    row=catalog(out)['entries'][0]
    assert row['anatomy_present']==['brain','tumor']
    assert row['missing_anatomy']==['skull','ventricles','vessels']
    assert all(not layer['accepted_anatomical_truth'] for layer in row['layers'])
    assert next(layer for layer in row['layers'] if layer['name']=='brain')['model']['license']=='CC0-1.0'
    assert row['tumor_location']['lobe']==row['tumor_location']['hemisphere']=='UNKNOWN'
    assert row['tumor_location']['centroid_ras_mm']==[-5.,8.,7.]
    assert row['tumor_location']['source_label_union_volume_ml']==pytest.approx(.048)
    assert row['support_conflicts'][0]['source_target_outside_support_voxels']==1
    assert not row['support_conflicts'][0]['target_clipped']
    assert not row['training_admission_established'] and not row['new_patient']
    assert result['counts']['newly_acquired_people']==result['counts']['anatomical_truth_accepted_people']==0
    assert not row['originals']['portable_originals_complete']


def test_resume_reuses_bound_receipt_and_queries_never_decode_imaging(tmp_path,monkeypatch):
    candidate,_,_=fixture(tmp_path);out=tmp_path/'out';frozen=plan(candidate)
    first=pc.run_qualification(tmp_path,out,frozen)
    monkeypatch.setattr(pc,'load_workspace',forbidden)
    second=pc.run_qualification(tmp_path,out,frozen)
    assert second['new_qualification_receipts']==0 and second['reused_qualification_receipts']==1
    assert first['catalog_hash']==second['catalog_hash']
    assert len(pc.query_catalog(out,anatomy='tumor',modality='T1',role='TRAIN'))==1
    assert not pc.query_catalog(out,anatomy='skull')
    assert len(pc.query_catalog(out,text='sub-pat05'))==1


def test_resumable_limit_advances_past_previously_qualified_people(tmp_path):
    a,_,_=fixture(tmp_path,'05');b,_,_=fixture(tmp_path,'16');out=tmp_path/'out';frozen=plan(a,b)
    first=pc.run_qualification(tmp_path,out,frozen,limit=1)
    assert first['counts']['pending']==1 and first['new_qualification_receipts']==1
    second=pc.run_qualification(tmp_path,out,frozen,limit=1)
    assert second['counts']['pending']==0 and second['new_qualification_receipts']==1
    assert second['reused_qualification_receipts']==1 and second['counts']['existing_materialized_people']==2
    assert len(list((out/'catalogs').glob('*.json')))==2


def test_reused_validator_change_invalidates_cached_qualification(tmp_path,monkeypatch):
    candidate,_,_=fixture(tmp_path);out=tmp_path/'out';frozen=plan(candidate)
    pc.run_qualification(tmp_path,out,frozen)
    bindings=pc._code_bindings();bindings['workspace_bundle.py']='f'*64
    monkeypatch.setattr(pc,'_code_bindings',lambda:bindings)
    result=pc.run_qualification(tmp_path,out,frozen)
    assert result['new_qualification_receipts']==1 and result['reused_qualification_receipts']==0


def test_original_changed_during_qualification_cannot_receive_success(tmp_path,monkeypatch):
    candidate,_,_=fixture(tmp_path);original=pc._mask_record
    def changed(*args,**kwargs):
        (tmp_path/candidate['source_files'][1]['path']).write_bytes(b'changed during validation')
        return original(*args,**kwargs)
    monkeypatch.setattr(pc,'_mask_record',changed)
    result=pc.run_qualification(tmp_path,tmp_path/'out',plan(candidate))
    assert result['rejections'][0]['failure_code']=='INPUT_CHANGED'


def test_original_change_refuses_stale_cache_and_keeps_failure_in_denominator(tmp_path,monkeypatch):
    candidate,_,_=fixture(tmp_path);out=tmp_path/'out';frozen=plan(candidate)
    pc.run_qualification(tmp_path,out,frozen)
    (tmp_path/candidate['source_files'][0]['path']).write_bytes(b'changed source')
    monkeypatch.setattr(pc,'load_workspace',forbidden)
    result=pc.run_qualification(tmp_path,out,frozen)
    assert result['counts']['existing_materialized_people']==1 and result['counts']['rejected']==1
    assert result['rejections'][0]['failure_code']=='SOURCE_HASH'
    assert result['reused_qualification_receipts']==0


def test_missing_participant_row_refuses_before_array_decode(tmp_path,monkeypatch):
    candidate,_,_=fixture(tmp_path)
    table=tmp_path/candidate['participant']['path'];table.write_text('participant_id\nwrong-person\n')
    candidate['participant']['sha256']=file_sha256(table)
    monkeypatch.setattr(pc,'load_workspace',forbidden)
    result=pc.run_qualification(tmp_path,tmp_path/'out',plan(candidate))
    assert result['rejections'][0]['failure_code']=='PARTICIPANT_ID'


@pytest.mark.parametrize('change',['grid','voxels','rights','lineage'])
def test_valid_bundle_cannot_lie_about_acquired_image(tmp_path,change):
    candidate,case,path=fixture(tmp_path)
    if change=='grid':case=replace(case,affine=case.affine@np.diag([2.,1.,1.,1.]))
    if change=='voxels':case=replace(case,mri=case.mri+1)
    if change=='rights':case=replace(case,source_refs=(replace(case.source_refs[0],license='CC-BY-4.0'),*case.source_refs[1:]))
    if change=='lineage':case=replace(case,source_refs=(case.source_refs[0],replace(case.source_refs[1],sha256='0'*64)))
    save_case(case,path)
    result=pc.run_qualification(tmp_path,tmp_path/'out',plan(candidate))
    expected={'grid':'IMAGE_GRID','voxels':'IMAGE_ARRAY','rights':'RIGHTS_MISMATCH','lineage':'SOURCE_LINEAGE'}
    assert result['rejections'][0]['failure_code']==expected[change]


def auxiliary(root,case,candidate,*,other_patient=False):
    path=nifti(root/'aux.nii.gz',np.arange(120,dtype=np.float32).reshape(4,5,6)+2)
    candidate['source_files'].append({'path':'aux.nii.gz','sha256':file_sha256(path),'role':'imaging'})
    aux=load_nifti_case(path,case_id='BTC-ds001226-sub-PAT16-preop' if other_patient else case.case_id,
        license='CC0-1.0',metadata={'selected_modality':'T2'})
    return DisplaySource(aux,'T2','none')


def test_cross_patient_auxiliary_refuses_before_numeric_decoding(tmp_path,monkeypatch):
    candidate,case,path=fixture(tmp_path);source=auxiliary(tmp_path,case,candidate,other_patient=True)
    save_workspace(case,path,sources=[source],imaging=None,episode=None,selection=None,artifacts={})
    monkeypatch.setattr(pc,'_case_layout',forbidden);monkeypatch.setattr(pc,'load_workspace',forbidden)
    result=pc.run_qualification(tmp_path,tmp_path/'out',plan(candidate))
    assert result['rejections'][0]['failure_code']=='PATIENT_MISMATCH'


def test_equal_grids_alone_cannot_approve_registration_and_originals_are_verified(tmp_path):
    candidate,case,path=fixture(tmp_path);source=auxiliary(tmp_path,case,candidate)
    originals=[OriginalFile.from_path(tmp_path/row['path'],role=row['role']) for row in candidate['source_files']]
    save_workspace(case,path,sources=[source],imaging=None,episode=None,selection=None,artifacts={},originals=originals)
    result=pc.run_qualification(tmp_path,tmp_path/'out',plan(candidate))
    assert result['counts']['qualified_existing_people']==1,result
    row=catalog(tmp_path/'out')['entries'][0];registration=row['registrations'][0]
    assert registration['status']=='unreviewed' and not registration['overlayPermitted']
    assert not registration['independently_validated_registration']
    assert row['originals']['embedded_count']==3 and row['originals']['portable_originals_complete']
    assert len(row['imaging'])==2


def test_cached_receipt_tampering_and_changed_plan_are_refused(tmp_path,monkeypatch):
    candidate,_,_=fixture(tmp_path);out=tmp_path/'out';frozen=plan(candidate)
    pc.run_qualification(tmp_path,out,frozen)
    receipt=next((out/'receipts').glob('*.json'));document=json.loads(receipt.read_text());document['role']='SELECT'
    receipt.write_text(json.dumps(document));monkeypatch.setattr(pc,'load_workspace',forbidden)
    result=pc.run_qualification(tmp_path,out,frozen)
    assert result['rejections'][0]['failure_code']=='RECEIPT_HASH'
    changed=deepcopy(candidate);changed['rights']['use']='changed'
    with pytest.raises(pc.QualificationError,match='new corpus output version'):
        pc.run_qualification(tmp_path,out,plan(changed))
