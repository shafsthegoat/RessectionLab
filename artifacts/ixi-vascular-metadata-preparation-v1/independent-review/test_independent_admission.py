"""Generated metadata only: no payload arrays, model or runtime bridge."""
from pathlib import Path
import sys
import pytest
ROOT=Path(__file__).resolve().parents[2];C=ROOT/'build/ixi-vascular-admission-v1'
sys.path.insert(0,str(C))
import test_admission as t


def test_missing_annotation_or_pair_refuses_without_reassignment():
    value=t.fixture()
    member=next(m for m in t.PEOPLE if m['role']==value['role'] and not m['annotation_available'])
    value['person_id']=member['person_group']
    with pytest.raises(ValueError,match='paired_annotated_person_required_no_reassignment'):
        t.prepare(value)
    assert value['role']==member['role']


def test_other_person_member_cannot_stand_in_for_frozen_pair():
    value=t.fixture();member=next(m for m in t.PEOPLE if m['person_group']!=value['person_id'] and 'T1' in m['raw_files'])
    value['sources']['T1']['member']=member['raw_files']['T1']['name']
    with pytest.raises(ValueError,match='source_kind_or_member_mismatch'):
        t.prepare(value)


def test_private_frame_registration_and_coverage_changes_do_not_change_actor_projection():
    value=t.fixture();before=t.prepare(value)
    for name in ('MRA','vessel_annotation'):
        row=value['sources'][name]
        row['grid']=t.grid(2.)
        row['source_file_sha256']=t.digest('CHANGED-PRIVATE-'+name)
        row['qc_record_sha256']=t.digest('CHANGED-PRIVATE-QC-'+name)
    mra=value['sources']['MRA'];label=value['sources']['vessel_annotation']
    registration=value['registration'];registration['to_source_sha256']=mra['source_file_sha256'];registration['to_frame_sha256']=mra['grid']['frame_sha256']
    registration['matrix_ras_mm'][0][3]=2.
    registration['valid_domain_sha256']=t.digest('CHANGED-PRIVATE-REGISTRATION-DOMAIN')
    coverage=value['coverage'];coverage['annotation_source_sha256']=label['source_file_sha256'];coverage['frame_sha256']=label['grid']['frame_sha256']
    coverage['registration_valid_domain_sha256']=registration['valid_domain_sha256'];coverage['coverage_sha256']=t.digest('CHANGED-PRIVATE-COVERAGE')
    after=t.prepare(value)
    assert before.actor_contract==after.actor_contract
    assert before.evaluator_contract!=after.evaluator_contract
    assert after.evaluator_contract['actor_contract_sha256']==t.a.semantic_digest(after.actor_contract)
    assert after.execution_status()['execution_admitted'] is False


@pytest.mark.parametrize('part',['T1','MRA','vessel_annotation','support','task','registration','coverage'])
def test_scope_cannot_be_omitted_while_retaining_generic_pass(part):
    value=t.fixture();row=value['sources'][part] if part in value['sources'] else value[part]
    del row['qc_scope']
    assert row['qc_status']=='pass'
    with pytest.raises(ValueError,match='fields'):
        t.prepare(value)


@pytest.mark.parametrize('shape',[[True,9,9],[9.,9,9],[4096,4096,4096]])
def test_source_grid_dimensions_are_exact_bounded_integers(shape):
    value=t.fixture();value['sources']['T1']['grid']['shape']=shape
    with pytest.raises(ValueError,match='bounded_3d_grid_required'):
        t.prepare(value)


def test_unknown_private_times_are_not_converted_to_actor_availability():
    prepared=t.prepare()
    for part in ('MRA','vessel_annotation'):
        assert prepared.evaluator_contract[part]['acquired_at'] is None
        assert prepared.evaluator_contract[part]['available_at'] is None
        assert prepared.evaluator_contract[part]['availability_record_sha256'] is None
    assert prepared.actor_contract['T1']['acquired_at'] is None
    assert prepared.actor_contract['preoperative_claim'] is False


def test_nested_projection_is_detached_and_generated_non_executable():
    value=t.fixture();prepared=t.prepare(value)
    value['support']['project_fit_roles'].clear();value['registration']['matrix_ras_mm'][0][3]=100.
    assert prepared.actor_contract['support']['project_fit_roles']==('TRAIN',)
    assert prepared.evaluator_contract['registration']['matrix_ras_mm'][0][3]==0.
    for contract in (prepared.actor_contract,prepared.evaluator_contract):
        assert contract['evidence_domain']=='generated_metadata_control'
        with pytest.raises(TypeError):contract['role']='TRAIN'
    assert prepared.execution_status()=={'execution_admitted':False,'patient_payloads_validated':False,'required_bridges':list(t.a.EXECUTION_GAPS)}
