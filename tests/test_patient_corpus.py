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


def frozen_ixi():
    document,_=pc._frozen_release()
    return document['additional_candidate']


def test_explicit_ten_person_release_is_metadata_only_and_default_stays_nine(monkeypatch):
    root=Path(__file__).resolve().parents[1]
    monkeypatch.setattr(pc,'load_workspace',forbidden);monkeypatch.setattr(pc,'_manifest_preflight',forbidden)
    monkeypatch.setattr(pc,'file_sha256',forbidden)
    monkeypatch.setattr(nib,'load',forbidden)
    original=pc.default_plan(root);expanded=pc.release_plan(root,pc.IXI_RELEASE)
    assert original==pc.release_plan(root)
    assert original['plan_hash']=='sha256:2945f141cd2e91dd661e4502e06544ca636adeac98f57c354bf56ea3486bad7f'
    assert len(original['candidates'])==9 and len(expanded['candidates'])==10
    assert expanded['plan_hash']==pc.IXI_PLAN_HASH
    assert [r['patient_group'] for r in expanded['candidates']].count('IXI:265')==1
    assert all('bundle_sha256' in row for row in expanded['candidates'])
    assert frozen_ixi()['training_admitted'] is False


def test_ixi_remains_forbidden_without_explicit_release():
    candidate=frozen_ixi()
    with pytest.raises(pc.QualificationError,match='nine-person'):pc.require_candidate(candidate)
    pc.require_candidate(candidate,pc.IXI_RELEASE)
    assert pc.canonical_patient('IXI265-Guys-0845')=='IXI:265'


@pytest.mark.parametrize('field,value',[
    ('patient_group','IXI:266'),('role','SELECT'),('training_admitted',True),
    ('bundle','outputs/poc-ixi265/virtual-FK-pilot-001/IXI265-virtual.ressectionlab'),
    ('bundle_sha256','0'*64),('case_id','IXI265-virtual-FK-pilot-001')])
def test_new_release_never_accepts_arbitrary_or_descendant_candidate(field,value):
    candidate=frozen_ixi();candidate[field]=value
    with pytest.raises(pc.QualificationError):pc.require_candidate(candidate,pc.IXI_RELEASE)


@pytest.mark.parametrize('change',['drop','duplicate','metadata','scope','unknown_release'])
def test_frozen_release_plan_cannot_be_self_resigned_to_change_scope(change,tmp_path,monkeypatch):
    frozen=pc.release_plan(Path(__file__).resolve().parents[1],pc.IXI_RELEASE)
    if change=='drop':frozen['candidates'].pop()
    if change=='duplicate':frozen['candidates'][0]=deepcopy(frozen['candidates'][6])
    if change=='metadata':frozen['metadata_evidence']=[]
    if change=='scope':frozen['virtual_descendants_admitted']=True
    if change=='unknown_release':frozen['release_id']='caller-supplied-release'
    frozen['plan_hash']=semantic_digest({k:v for k,v in frozen.items() if k!='plan_hash'})
    monkeypatch.setattr(pc,'_input_binding',forbidden)
    with pytest.raises(pc.QualificationError):pc.run_qualification(tmp_path,tmp_path/'out',frozen)
    assert not (tmp_path/'out').exists()


def test_frozen_bundle_hash_is_checked_before_zip_or_source_access(tmp_path,monkeypatch):
    candidate=frozen_ixi();path=tmp_path/candidate['bundle'];path.parent.mkdir(parents=True)
    path.write_bytes(b'counterfeit named IXI bundle')
    monkeypatch.setattr(pc,'_manifest_preflight',forbidden)
    with pytest.raises(pc.QualificationError,match='before decoding'):
        pc._input_binding(tmp_path,candidate,{'release_id':pc.IXI_RELEASE,'metadata_evidence':[]})


@pytest.mark.parametrize('metadata',[{'is_synthetic':True},{'is_synthetic':False,'virtual_glioma':{'host_patient_group':'IXI:265'}},
    {'is_synthetic':False,'split':{'role':'SELECT'}},{'is_synthetic':False,'patient_group':'IXI:266'}])
def test_ixi_manifest_rejects_generated_roles_or_cross_patient_ancestry(metadata):
    with pytest.raises(pc.QualificationError):
        pc.require_manifest({'case_id':'IXI265-Guys-0845','metadata':metadata},frozen_ixi(),pc.IXI_RELEASE)


def test_search_does_not_mistake_shared_release_provenance_for_patient_identity(tmp_path):
    rows=[{'patient_group':'BTC:sub-PAT05','case_id':'BTC-sub-PAT05','role':'TRAIN','status':'qualified_with_limitations',
        'input_binding':{'metadata_evidence':['IXI265 shared release role evidence']}},
        {'patient_group':'IXI:265','case_id':'IXI265-Guys-0845','role':'TRAIN','status':'qualified_with_limitations'}]
    pc._write_search_index(tmp_path/'catalog.sqlite',{'entries':rows})
    assert [row['patient_group'] for row in pc.query_catalog(tmp_path,text='IXI')]==['IXI:265']


@pytest.mark.parametrize('lineage',[{'virtual_geometry':{}},{'virtual_geometry':{'host_patient_group':'IXI:265'}},
    {'anatomy_layers':{'lesion':{'kind':'tumor','provenance':'generated'}}},
    {'anatomy_layers':{'lesion':{'kind':'tumor','provenance':'simulated_geometry'}}}])
@pytest.mark.parametrize('release',[pc.DEFAULT_RELEASE,pc.IXI_RELEASE])
def test_generated_anatomy_metadata_rejected_without_opening_payloads(tmp_path,lineage,release,monkeypatch):
    candidate=frozen_ixi() if release==pc.IXI_RELEASE else fixture(tmp_path)[0]
    monkeypatch.setattr(pc,'load_workspace',forbidden);monkeypatch.setattr(pc,'_manifest_preflight',forbidden)
    with pytest.raises(pc.QualificationError) as error:
        pc.require_manifest({'case_id':candidate['case_id'],'metadata':{'is_synthetic':False,**lineage}},candidate,release)
    assert error.value.code=='GENERATED_DESCENDANT'


@pytest.mark.parametrize('provenance',['simulated','generated'])
def test_simulated_source_reference_rejected_in_manifest_before_decoding(tmp_path,provenance):
    candidate,_,_=fixture(tmp_path)
    with pytest.raises(pc.QualificationError) as error:
        pc.require_manifest({'case_id':candidate['case_id'],'metadata':{'is_synthetic':False},
            'source_refs':[{'provenance':provenance}]},candidate)
    assert error.value.code=='GENERATED_DESCENDANT'


def test_estimated_layer_survives_synthetic_model_training_lineage():
    candidate=frozen_ixi()
    pc.require_manifest({'case_id':candidate['case_id'],'metadata':{'is_synthetic':False,
        'anatomy_layers':{'estimated_vessels':{'kind':'vessels','provenance':'estimated',
            'model':{'training':{'provenance':'simulated','sources':['D_real','D_drand','D_flow']}}}}},
        'source_refs':[{'provenance':'observed'},{'provenance':'estimated'}]},candidate,pc.IXI_RELEASE)


def test_prospective_twelve_preparation_reads_metadata_only_and_retains_exact_ten(monkeypatch):
    root=Path(__file__).resolve().parents[1]
    real_open=Path.open
    def metadata_only(path,*args,**kwargs):
        assert path.suffix=='.json','Unexpected payload/code access: '+str(path)
        return real_open(path,*args,**kwargs)
    monkeypatch.setattr(Path,'open',metadata_only)
    monkeypatch.setattr(pc,'file_sha256',forbidden);monkeypatch.setattr(pc,'load_workspace',forbidden)
    monkeypatch.setattr(pc,'_manifest_preflight',forbidden);monkeypatch.setattr(nib,'load',forbidden)
    current=pc.release_plan(root,pc.IXI_RELEASE);prepared=pc.prospective_plan(root)
    new_ids={'SynthRAD2023:1BB033','SynthRAD2023:1BC077'}
    assert len(prepared['candidates'])==12
    assert [row for row in prepared['candidates'] if row['patient_group'] not in new_ids]==current['candidates']
    assert prepared['qualification_released'] is prepared['source_image_decoding_released'] is False
    assert pc.PROSPECTIVE_RELEASE not in pc.RELEASES
    assert prepared['plan_hash']==pc.PROSPECTIVE_PLAN_HASH


@pytest.mark.parametrize('field,value',[('role','SELECT'),('training_admitted',True),('bundle','another.ressectionlab'),
    ('bundle_sha256','0'*64),('patient_group','SynthRAD2023:1BC078'),('source_files',[])])
def test_prospective_candidate_identity_cannot_be_changed_before_future_access(field,value):
    frozen,_=pc._prospective_release();candidate=deepcopy(frozen['additional_candidates'][0]);candidate[field]=value
    with pytest.raises(pc.QualificationError):pc.require_prospective_candidate(candidate)


def test_prospective_new_people_stay_outside_both_live_releases():
    frozen,_=pc._prospective_release()
    for candidate in frozen['additional_candidates']:
        pc.require_prospective_candidate(candidate)
        for release in pc.RELEASES:
            with pytest.raises(pc.QualificationError):pc.require_candidate(candidate,release)


@pytest.mark.parametrize('change',['none','self_release','drop','duplicate','metadata','schema'])
def test_prospective_plan_cannot_execute_or_self_release_before_payload_or_output(tmp_path,monkeypatch,change):
    prepared=pc.prospective_plan(Path(__file__).resolve().parents[1])
    if change=='self_release':prepared['qualification_released']=True
    if change=='drop':prepared['candidates'].pop()
    if change=='duplicate':prepared['candidates'][0]=deepcopy(prepared['candidates'][-1])
    if change=='metadata':prepared['metadata_evidence']=[]
    if change=='schema':prepared['schema']=pc.VERSION+'.plan'
    prepared['plan_hash']=semantic_digest({k:v for k,v in prepared.items() if k!='plan_hash'})
    if change!='none':
        with pytest.raises(pc.QualificationError):pc.validate_prospective_plan(prepared)
    monkeypatch.setattr(pc,'_input_binding',forbidden);monkeypatch.setattr(pc,'_manifest_preflight',forbidden)
    monkeypatch.setattr(pc,'load_workspace',forbidden);monkeypatch.setattr(nib,'load',forbidden)
    with pytest.raises(pc.QualificationError) as error:pc.run_qualification(tmp_path,tmp_path/'out',prepared)
    assert error.value.code=='QUALIFICATION_NOT_RELEASED'
    assert not (tmp_path/'out').exists()


def copy_prospective_metadata(root):
    """Copy only pinned JSON metadata to exercise preparation persistence."""
    repository=Path(__file__).resolve().parents[1];plan=pc.prospective_plan(repository)
    for row in plan['metadata_evidence']:
        path=root/row['path'];path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes((repository/row['path']).read_bytes())


def test_prospective_preparation_resumes_without_creating_qualified_catalog(tmp_path,monkeypatch):
    copy_prospective_metadata(tmp_path)
    monkeypatch.setattr(pc,'load_workspace',forbidden);monkeypatch.setattr(pc,'_input_binding',forbidden)
    monkeypatch.setattr(nib,'load',forbidden)
    first=pc.prepare_prospective_release(tmp_path);path=tmp_path/first['prepared_plan'];before=path.read_bytes()
    second=pc.prepare_prospective_release(tmp_path)
    assert first['reused_preparation'] is False and second['reused_preparation'] is True
    assert path.read_bytes()==before and first['plan_hash']==second['plan_hash']
    assert second['qualified_people_this_step']==0 and second['payloads_opened'] is False
    assert not (tmp_path/pc.PROSPECTIVE_OUTPUT/'catalog.json').exists()
    assert not (tmp_path/'outputs/patient-corpus-v1').exists()
    assert not (tmp_path/'outputs/patient-corpus-ixi265-v1').exists()


def test_prospective_preparation_refuses_redirect_into_previous_release(tmp_path):
    copy_prospective_metadata(tmp_path)
    old=tmp_path/'old-release';old.mkdir();(tmp_path/pc.PROSPECTIVE_OUTPUT).symlink_to(old,target_is_directory=True)
    with pytest.raises(pc.QualificationError) as error:pc.prepare_prospective_release(tmp_path)
    assert error.value.code=='OUTPUT_SCOPE' and not list(old.iterdir())


def copy_execution_metadata(root,*,launch=False):
    repository=Path(__file__).resolve().parents[1];document=pc.execution_plan(repository)
    for row in document['metadata_evidence']:
        path=root/row['path'];path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes((repository/row['path']).read_bytes())
    if launch:
        # Generated temporary authority exercises the supervisor, never real inputs.
        guide=json.loads((repository/'docs/patient-corpus-twelve-execution-guide.json').read_text())
        path=root/pc.EXECUTION_OUTPUT/'launch-admission.json';path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(guide['launch_gate']['document'],indent=2,sort_keys=True)+'\n')
    return document


def test_execution_preparation_reads_only_json_preserves_exact_scope_and_old_plan_hashes(monkeypatch):
    root=Path(__file__).resolve().parents[1];real_open=Path.open
    def metadata_only(path,*args,**kwargs):
        assert path.suffix=='.json','Unexpected binary/code access: '+str(path)
        return real_open(path,*args,**kwargs)
    monkeypatch.setattr(Path,'open',metadata_only)
    monkeypatch.setattr(pc,'file_sha256',forbidden);monkeypatch.setattr(pc,'load_workspace',forbidden)
    monkeypatch.setattr(pc,'_manifest_preflight',forbidden);monkeypatch.setattr(nib,'load',forbidden)
    document=pc.execution_plan(root);parent=pc.prospective_plan(root)
    assert document['candidates']==parent['candidates'] and len(document['candidates'])==12
    assert pc.EXECUTION_RELEASE not in pc.RELEASES and pc.RELEASES==(pc.DEFAULT_RELEASE,pc.IXI_RELEASE)
    assert pc.release_plan(root,pc.IXI_RELEASE)['plan_hash']==pc.IXI_PLAN_HASH
    assert pc.default_plan(root)['plan_hash']=='sha256:2945f141cd2e91dd661e4502e06544ca636adeac98f57c354bf56ea3486bad7f'
    assert document['source_image_decoding_released'] is False
    assert document['plan_hash']==pc.EXECUTION_PLAN_HASH


@pytest.mark.parametrize('change',['drop','duplicate','role','bundle','source','metadata','budget','release','launch'])
def test_execution_plan_cannot_self_resign_scope_pins_or_budgets(tmp_path,monkeypatch,change):
    document=pc.execution_plan(Path(__file__).resolve().parents[1])
    if change=='drop':document['candidates'].pop()
    if change=='duplicate':document['candidates'][0]=deepcopy(document['candidates'][-1])
    if change=='role':document['candidates'][0]['role']='SELECT'
    if change=='bundle':document['candidates'][-1]['bundle']='other.ressectionlab'
    if change=='source':document['candidates'][-1]['source_files'][0]['sha256']='0'*64
    if change=='metadata':document['metadata_evidence']=[]
    if change=='budget':document['resource_bounds']['process_family_plus_supervisor_rss_bytes']*=2
    if change=='release':document['release_id']='caller-release'
    if change=='launch':document['source_image_decoding_released']=True
    document['plan_hash']=semantic_digest({k:v for k,v in document.items() if k!='plan_hash'})
    monkeypatch.setattr(pc,'_input_binding',forbidden);monkeypatch.setattr(pc.subprocess,'Popen',forbidden)
    with pytest.raises(pc.QualificationError):pc.run_qualification(tmp_path,tmp_path/pc.EXECUTION_OUTPUT,document)
    assert not (tmp_path/pc.EXECUTION_OUTPUT).exists()


@pytest.mark.parametrize('source',[False,True])
@pytest.mark.parametrize('nested',[{'roles':{'SELECT':True}},{'split':{'name':'validation'}},
    {'source_collection':{'name':'UPENN-GBM'}},{'patient_group':'SynthRAD2023:1BC077'}])
def test_execution_recursively_rejects_protected_or_wrong_identity_before_decode(source,nested,monkeypatch):
    candidate=next(row for row in pc.execution_plan(Path(__file__).resolve().parents[1])['candidates']
        if row['patient_group']=='SynthRAD2023:1BB033')
    manifest={'case_id':candidate['case_id'],'metadata':{'is_synthetic':False},'source_refs':[]}
    if source:manifest['source_refs']=[{'provenance':'observed','metadata':nested}]
    else:manifest['metadata']['parent']=nested
    monkeypatch.setattr(pc,'load_workspace',forbidden);monkeypatch.setattr(pc,'_case_layout',forbidden)
    with pytest.raises(pc.QualificationError) as error:pc.require_manifest(manifest,candidate,pc.EXECUTION_RELEASE)
    assert error.value.code in {'PROTECTED_ROLE','PATIENT_MISMATCH'}


def test_execution_prepare_resumes_but_cannot_materialize_launch_or_decode(tmp_path,monkeypatch):
    document=copy_execution_metadata(tmp_path)
    monkeypatch.setattr(pc,'_input_binding',forbidden);monkeypatch.setattr(pc.subprocess,'Popen',forbidden)
    first=pc.prepare_execution_release(tmp_path);path=tmp_path/first['prepared_plan'];before=path.read_bytes()
    second=pc.prepare_execution_release(tmp_path)
    assert not first['reused_preparation'] and second['reused_preparation'] and path.read_bytes()==before
    assert not (tmp_path/pc.EXECUTION_OUTPUT/'launch-admission.json').exists()
    with pytest.raises(pc.QualificationError) as error:pc.run_qualification(tmp_path,tmp_path/pc.EXECUTION_OUTPUT,document)
    assert error.value.code=='QUALIFICATION_NOT_RELEASED'
    assert not (tmp_path/pc.EXECUTION_OUTPUT/'catalog.json').exists()


@pytest.mark.parametrize('kind',['wrong_output','symlink_output','wrong_launch','launch_symlink','limit'])
def test_execution_launch_cannot_redirect_or_substitute_authority(tmp_path,monkeypatch,kind):
    document=copy_execution_metadata(tmp_path,launch=True);output=tmp_path/pc.EXECUTION_OUTPUT
    if kind=='wrong_output':output=tmp_path/'outputs/patient-corpus-ixi265-v1'
    if kind=='symlink_output':
        moved=tmp_path/'elsewhere';output.rename(moved);output.symlink_to(moved,target_is_directory=True)
    if kind=='wrong_launch':(output/'launch-admission.json').write_text('{}\n')
    if kind=='launch_symlink':
        moved=output/'elsewhere.json';(output/'launch-admission.json').rename(moved);(output/'launch-admission.json').symlink_to(moved)
    monkeypatch.setattr(pc,'_input_binding',forbidden);monkeypatch.setattr(pc.subprocess,'Popen',forbidden)
    with pytest.raises(pc.QualificationError):pc.run_qualification(tmp_path,output,document,limit=1 if kind=='limit' else None)


def safe_execution_sample(bounds):
    return {'rss':bounds['process_family_plus_supervisor_rss_bytes'],'available':bounds['minimum_available_bytes'],
        'elapsed':bounds['wall_seconds'],'output_bytes':bounds['maximum_metadata_output_bytes'],
        'free_disk':bounds['minimum_free_disk_bytes']}


@pytest.mark.parametrize('field,delta,code',[('rss',1,'RSS_LIMIT'),('available',-1,'AVAILABLE_MEMORY_FLOOR'),
    ('elapsed',0.001,'WALL_TIME_LIMIT'),('output_bytes',1,'OUTPUT_LIMIT'),('free_disk',-1,'DISK_RESERVE')])
def test_execution_resource_limits_accept_exact_boundary_and_refuse_excess(field,delta,code):
    admission,_=pc._execution_admission();bounds=admission['bounds'];sample=safe_execution_sample(bounds)
    assert bounds=={'process_family_plus_supervisor_rss_bytes':1610612736,'minimum_available_bytes':1073741824,
        'wall_seconds':600,'numerical_threads':2,'worker_count':1,'sequential_people':True,'automatic_retries':0,
        'minimum_free_disk_bytes':53687091200,'maximum_metadata_output_bytes':536870912,'monitor_interval_seconds':0.1}
    assert pc._execution_resource_violation(bounds,**sample) is None
    sample[field]+=delta
    assert pc._execution_resource_violation(bounds,**sample)==code


@pytest.mark.parametrize('phase',['preflight','running'])
def test_execution_resource_refusal_stops_before_payload_or_kills_single_worker(tmp_path,monkeypatch,phase):
    document=copy_execution_metadata(tmp_path,launch=True);safe=safe_execution_sample(document['resource_bounds'])
    samples=[{**safe,'rss':safe['rss']+1}] if phase=='preflight' else [safe,{**safe,'rss':safe['rss']+1}]
    monkeypatch.setattr(pc,'_execution_sample',lambda *args: samples.pop(0))
    launched=[];killed=[]
    class Worker:
        pid=987654321;returncode=None
        def poll(self):return self.returncode
        def wait(self,timeout):self.returncode=-9;return -9
    worker=Worker()
    def popen(command,**kwargs):
        launched.append(command)
        assert kwargs['start_new_session'] is True
        assert kwargs['env']['OMP_NUM_THREADS']==kwargs['env']['OPENBLAS_NUM_THREADS']=='2'
        return worker
    monkeypatch.setattr(pc.subprocess,'Popen',popen);monkeypatch.setattr(pc.os,'killpg',lambda *args:killed.append(args))
    monkeypatch.setattr(pc,'_input_binding',forbidden)
    with pytest.raises(pc.QualificationError) as error:pc.run_qualification(tmp_path,tmp_path/pc.EXECUTION_OUTPUT,document)
    assert error.value.code=='RSS_LIMIT'
    assert len(launched)==len(killed)==(0 if phase=='preflight' else 1)
    receipts=list((tmp_path/pc.EXECUTION_OUTPUT/'supervision').glob('*/supervision.json'))
    assert len(receipts)==(0 if phase=='preflight' else 1)
    if receipts:
        receipt=json.loads(receipts[0].read_text())
        assert receipt['status']=='failed' and receipt['automatic_retries']==0 and receipt['worker_exit_code']==-9


def test_execution_supervisor_completes_one_worker_with_metadata_receipt(tmp_path,monkeypatch):
    document=copy_execution_metadata(tmp_path,launch=True);sample=safe_execution_sample(document['resource_bounds'])
    monkeypatch.setattr(pc,'_execution_sample',lambda *args:sample)
    launched=[]
    class Worker:
        pid=987654321;returncode=0
        def poll(self):return 0
    expected={'generated_software_control':True,'reused_qualification_receipts':12}
    def popen(command,**kwargs):
        launched.append(command);Path(command[-1]).write_text(json.dumps(expected));return Worker()
    monkeypatch.setattr(pc.subprocess,'Popen',popen);monkeypatch.setattr(pc,'_input_binding',forbidden)
    assert pc.run_qualification(tmp_path,tmp_path/pc.EXECUTION_OUTPUT,document)==expected
    assert len(launched)==1
    receipts=list((tmp_path/pc.EXECUTION_OUTPUT/'supervision').glob('*/supervision.json'))
    assert len(receipts)==1 and json.loads(receipts[0].read_text())['status']=='completed'


def test_execution_output_budget_rejects_nested_symlinks(tmp_path):
    output=tmp_path/'out';output.mkdir();(output/'redirect').symlink_to(tmp_path/'elsewhere')
    with pytest.raises(pc.QualificationError) as error:pc._execution_directory_bytes(output)
    assert error.value.code=='OUTPUT_SCOPE'


def test_execution_revalidates_admission_bytes_before_candidate_paths(tmp_path,monkeypatch):
    document=copy_execution_metadata(tmp_path,launch=True)
    (tmp_path/pc.EXECUTION_ADMISSION_PATH).write_text('{}\n')
    monkeypatch.setattr(pc,'_input_binding',forbidden);monkeypatch.setattr(pc.subprocess,'Popen',forbidden)
    with pytest.raises(pc.QualificationError) as error:pc.run_qualification(tmp_path,tmp_path/pc.EXECUTION_OUTPUT,document)
    assert error.value.code=='METADATA_HASH'


def test_execution_worker_cannot_be_started_without_supervisor(tmp_path,monkeypatch):
    copy_execution_metadata(tmp_path,launch=True)
    monkeypatch.delenv('RESSECTIONLAB_CORPUS_SUPERVISOR_PID',raising=False)
    monkeypatch.setattr(pc,'_run_qualification',forbidden)
    with pytest.raises(pc.QualificationError) as error:
        pc._execution_worker(tmp_path,tmp_path/pc.EXECUTION_OUTPUT/'supervision/test/worker-result.json')
    assert error.value.code=='SUPERVISOR_REQUIRED'


def test_execution_explicit_interrupt_records_failure_and_kills_without_retry(tmp_path,monkeypatch):
    document=copy_execution_metadata(tmp_path,launch=True);sample=safe_execution_sample(document['resource_bounds'])
    def sampling(*args):
        if len(args)==4:raise KeyboardInterrupt()
        return sample
    monkeypatch.setattr(pc,'_execution_sample',sampling)
    launched=[];killed=[]
    class Worker:
        pid=987654321;returncode=None
        def poll(self):return self.returncode
        def wait(self,timeout):self.returncode=-9;return -9
    worker=Worker()
    def popen(*args,**kwargs):launched.append(args);return worker
    monkeypatch.setattr(pc.subprocess,'Popen',popen);monkeypatch.setattr(pc.os,'killpg',lambda *args:killed.append(args))
    with pytest.raises(KeyboardInterrupt):pc.run_qualification(tmp_path,tmp_path/pc.EXECUTION_OUTPUT,document)
    receipts=list((tmp_path/pc.EXECUTION_OUTPUT/'supervision').glob('*/supervision.json'))
    assert len(launched)==len(killed)==len(receipts)==1
    receipt=json.loads(receipts[0].read_text())
    assert receipt['status']=='failed' and receipt['failure_code']=='INTERRUPTED' and receipt['worker_exit_code']==-9


def test_execution_cleanup_tolerates_worker_exit_between_poll_and_kill(monkeypatch):
    waited=[]
    class Worker:
        pid=987654321
        def poll(self):return None
        def wait(self,timeout):waited.append(timeout);return 0
    def vanished(*args):raise ProcessLookupError()
    monkeypatch.setattr(pc.os,'killpg',vanished)
    pc._stop_execution_worker(Worker())
    assert waited==[5]
