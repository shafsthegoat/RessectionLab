"""Frozen transfer guards and denominator arithmetic; no patient data loading."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest
import torch

ROOT=Path(__file__).parents[1]
spec=importlib.util.spec_from_file_location('training_transfer',ROOT/'scripts/run_real_training_transfer.py')
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)


@pytest.fixture
def metadata(tmp_path,monkeypatch):
    import prepare_real_training_cases as prep
    cohort=tmp_path/runner.COHORT_PATH;cohort.parent.mkdir(parents=True)
    shutil.copy2(ROOT/runner.COHORT_PATH,cohort)
    checkpoint=tmp_path/'anchor/augmented_latest.pt';checkpoint.parent.mkdir();checkpoint.write_bytes(b'metadata-only checkpoint double')
    (checkpoint.parent/'output-sha256.json').write_text(json.dumps({'augmented_latest.pt':runner.sha256(checkpoint)}))
    monkeypatch.setattr(runner,'ROOT',tmp_path);monkeypatch.setattr(runner,'ANCHOR',checkpoint.parent);monkeypatch.setattr(runner,'CHECKPOINT',checkpoint)
    monkeypatch.setattr(runner,'source_inventory',lambda:{'fixed-source':'abc'})
    monkeypatch.setattr(prep,'preparation_inputs',lambda:{'subjects':list(runner.SUBJECTS),'common_task':'fixed'})
    return runner.declaration()


def test_exact_five_train_patients_no_select_or_patient_omission(metadata):
    assert runner.validate(metadata)
    for subjects in [list(runner.SUBJECTS[:-1]),['sub-PAT26',*runner.SUBJECTS[1:]],['sub-PAT29',*runner.SUBJECTS[1:]]]:
        altered=deepcopy(metadata);altered['subjects']=subjects
        with pytest.raises(ValueError,match='remaining-TRAIN'):runner.validate(altered)
    altered=deepcopy(metadata);altered['settings']['optimizer_updates']=1
    with pytest.raises(ValueError,match='zero-update'):runner.validate(altered)


def completed_methods():
    return {name:{'status':'complete','outcomes':{'total_reward':0.},'independent_evaluation_accepted':True,
        'initial_parameter_hash':runner.POLICY_HASH,'final_parameter_hash':runner.POLICY_HASH}for name in runner.PRIMARY_METHODS}


def closure():
    return {'sources_unchanged':True,'preparation_inputs_unchanged':True,'checkpoint_bytes_unchanged':True}


def test_optional_random_timeout_does_not_erase_audited_primary_pair(tmp_path):
    subject=runner.SUBJECTS[0];d=tmp_path/subject;d.mkdir()
    receipt={**runner.blank_receipt(subject),'status':'failed','methods':{**completed_methods(),
        'random_legal':{'status':'running','outcomes':None}}}
    (d/'receipt.json').write_text(json.dumps(receipt));(d/'supervisor.json').write_text(json.dumps({'status':'failed','timed_out':True}))
    (d/'closure-check.json').write_text(json.dumps(closure()))
    summary=runner.summarize_patients(tmp_path)
    assert summary['patients_prescribed']==5 and len(summary['patients'])==5
    assert summary['complete_pairs']==1 and summary['paired_return_difference'][subject]==0.
    assert summary['patients'][0]['methods']['random_legal']['outcomes'] is None
    assert summary['patients'][1]['status']=='not_executed'


@pytest.mark.parametrize('failure',['source','audit','model','incomplete'])
def test_primary_failure_is_never_a_zero_or_successful_pair(failure):
    patient={'methods':completed_methods(),'closure_check':closure()}
    if failure=='source':patient['closure_check']['sources_unchanged']=False
    if failure=='audit':patient['methods']['greedy_search']['independent_evaluation_accepted']=False
    if failure=='model':patient['methods']['frozen_policy']['final_parameter_hash']='changed'
    if failure=='incomplete':patient['methods']['greedy_search'].update(status='running',outcomes=None)
    assert not runner.primary_pair_eligible(patient)


def test_blocked_support_remains_in_denominator(tmp_path):
    d=tmp_path/runner.SUBJECTS[0];d.mkdir()
    row={**runner.blank_receipt(runner.SUBJECTS[0]),'status':'preparation_blocked','preparation_status':'blocked_support_conflict'}
    (d/'receipt.json').write_text(json.dumps(row))
    summary=runner.summarize_patients(tmp_path)
    assert summary['patients_prescribed']==5 and summary['complete_pairs']==0
    assert summary['patients'][0]['status']=='preparation_blocked'
    assert all(m['outcomes']is None for m in summary['patients'][0]['methods'].values())


def test_frozen_loader_disables_gradients_and_never_initializes_adam(tmp_path,monkeypatch):
    from resectionlab.spatial_policy import SpatialPolicy,SpatialPolicyConfig,parameter_hash
    torch.set_num_threads(1)
    policy=SpatialPolicy(SpatialPolicyConfig(critic_candidate_context=True));expected=parameter_hash(policy)
    monkeypatch.setattr(runner,'POLICY_HASH',expected)
    monkeypatch.setattr(torch.optim,'Adam',lambda *a,**k:pytest.fail('zero-update comparison initialized Adam'))
    checkpoint=tmp_path/'checkpoint.pt'
    torch.save({'policy':policy.state_dict(),'architecture':policy.architecture_record(),
        'parameter_hash':expected,'additional_updates':8},checkpoint)
    loaded=runner.load_frozen_policy(checkpoint,runner.sha256(checkpoint))
    assert parameter_hash(loaded)==expected and not loaded.training
    assert all(not p.requires_grad and p.grad is None for p in loaded.parameters())
    with checkpoint.open('ab')as f:f.write(b'changed')
    with pytest.raises(ValueError,match='bytes changed'):runner.load_frozen_policy(checkpoint,'wrong')
