"""Metadata-only refit publication controls; no model/task/checkpoint decode."""
import copy
from pathlib import Path
import pytest
from resectionlab import contact_train_refit_release as release
from resectionlab.core import semantic_digest


@pytest.fixture
def publication(monkeypatch,tmp_path):
    family={'family_hash':'sha256:'+'f'*64,'source_bindings':[{'layout_id':f'pcf-{i:02d}','role':'TRAIN'} for i in range(12)]}
    keys=[(r['layout_id'],g) for r in family['source_bindings'] for g in ('surface','deep')]
    experiment={'version':'generated-public-contact-full-teacher-refit-v1','family_manifest':family,'teacher_states':[]}
    exp_hash=semantic_digest(experiment);monkeypatch.setattr(release,'EXPERIMENT_HASH',exp_hash)
    checkpoint={'sha256':release.CHECKPOINT_SHA256,'parameter_hash':release.PARAMETER_HASH}
    fit={'status':'complete_fixed_full40_TRAIN_refit','experiment_hash':exp_hash,'optimizer_updates':32,
         'loss_forward_calls':1280,'readout_forward_calls':80,'SELECT_or_MEASUREMENT_reads':0,'patient_reads':0,'final_checkpoint':checkpoint}
    rows=[{'layout_id':l,'goal_id':g,'role':'TRAIN','status':'complete','actions':['STOP'],
           'metrics':{'goal_contacted_and_retained':False,'total_reward':0.},
           'saved_search_metrics':{'goal_contacted_and_retained':False}} for l,g in keys]
    rollout={'status':'complete_fixed24_TRAIN_greedy_pass','experiment_hash':exp_hash,'checkpoint':checkpoint,
             'parameter_hash_after':release.PARAMETER_HASH,'rows':rows,
             **{k:0 for k in ('optimizer_updates','new_search_calls','new_teacher_calls','patient_reads','SELECT_or_MEASUREMENT_reads')}}
    file_records={key:{'relativePath':relative,'sha256':release.CHECKPOINT_SHA256 if key=='checkpoint' else str(i)*64,'bytes':32}
                  for i,(key,(relative,_)) in enumerate(release.SLOTS.items(),1)}
    receipt=lambda target:{'status':'complete','exit_code':0,'cleanup_errors':[],'final_owned_pids':[],
                          'worker_termination_confirmed':True,'result_sha256':file_records[target]['sha256']}
    records={'experiment':experiment,'fitResult':fit,'rolloutResult':rollout,
        'fitSupervision':receipt('fitResult'),'rolloutSupervision':receipt('rolloutResult'),
        'independentAudit':{'status':'PASS_SAVED_ONLY_ROLLOUT_PACKAGE'}}
    manifest={'version':release.RELEASE_VERSION,'variant':release.VARIANT,'scope':'TRAIN_only_generated_extra_compute_refit',
        'experimentHash':exp_hash,'familyHash':family['family_hash'],'parameterHash':release.PARAMETER_HASH,'files':file_records,
        'trainingBudget':{'updates':32,'statesPerUpdate':40,'lossForwards':1280,'fixedReadoutForwards':80},
        'knownTRAINOutcome':{'tasks':24,'goalContacts':0,'savedSEARCHContacts':0,'STOPOnly':24,'meanReturn':0.,
                             'scope':'generated_TRAIN_native_results_no_heldout_claim'}}
    monkeypatch.setattr(release,'RELEASE_MANIFEST_SHA256','a'*64)
    calls=[]
    def fixed_file(root,relative,expected,limit,digest,size):
        assert relative==expected;calls.append(relative);return Path(relative)
    monkeypatch.setattr(release,'_file',fixed_file)
    mapping={value[0]:records[key] for key,value in release.SLOTS.items() if key!='checkpoint'}
    mapping[release.RELEASE_RELATIVE_PATH]=manifest
    monkeypatch.setattr(release,'_json_file',lambda path,*args:mapping[str(path)])
    return tmp_path,family,manifest,records,calls


def test_distinct_publication_admits_all_train_keys_and_exposes_negative_results(publication):
    root,family,manifest,records,calls=publication
    checked=release.read_published_train_refit(root,family_manifest=family)
    assert len(checked['allowedTRAINKeys'])==24 and checked['knownTRAINOutcome']['STOPOnly']==24
    assert checked['manifest']['variant']=='IL_TRAIN_REFIT' and checked['manifest']['scope'].startswith('TRAIN_only')
    release.require_train_refit_request(checked,layout_id='pcf-00',goal_id='surface')
    with pytest.raises(release.ContactReleaseUnavailable):release.require_train_refit_request(checked,layout_id='other',goal_id='surface')


def test_partial_or_changed_saved_rollout_cannot_publish(publication):
    root,family,manifest,records,_=publication
    records['rolloutResult']['rows'][0]['status']='failed_or_capped'
    with pytest.raises(release.ContactReleaseUnavailable):release.read_published_train_refit(root,family_manifest=family)


def test_published_outcomes_and_extra_training_budget_cannot_be_relabelled(publication):
    root,family,manifest,records,_=publication
    manifest['knownTRAINOutcome']['goalContacts']=24
    with pytest.raises(release.ContactReleaseUnavailable):release.read_published_train_refit(root,family_manifest=family)
    manifest['knownTRAINOutcome']['goalContacts']=0;manifest['trainingBudget']['statesPerUpdate']=4
    with pytest.raises(release.ContactReleaseUnavailable):release.read_published_train_refit(root,family_manifest=family)


def test_nontrain_owned_load_refused_before_publication_or_checkpoint_read(monkeypatch):
    from resectionlab import public_contact_family
    monkeypatch.setattr(public_contact_family,'layout_metadata',lambda layout:{'role':'SELECT'})
    def forbidden(*args,**kwargs):raise AssertionError('Non-TRAIN request read publication/model')
    monkeypatch.setattr(release,'read_published_train_refit',forbidden)
    with pytest.raises(release.ContactReleaseUnavailable,match='TRAIN_REFIT_ONLY'):
        release.load_train_refit_for_owned_inference(Path('.'),layout_id='pcf-00',goal_id='surface')
