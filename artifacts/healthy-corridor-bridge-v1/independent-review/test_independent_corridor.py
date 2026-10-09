"""Bounded generated regressions for corridor identity, choices and accounting."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[2];C=ROOT/'build/healthy-corridor-bridge-v1'
sys.path.insert(0,str(C))
import test_corridor as t
tmp_path=t.tmp_path


def test_two_interior_cells_produce_two_distinct_waypoints():
    task,_,_=t.fixture();support=np.zeros(task.support.shape,bool);support[6:9,6:9,10:14]=True
    actor=t.thaw_json(task.actor_contract);actor['support']['output_sha256']=t.array_digest(support);actor['task']['support_output_sha256']=t.array_digest(support)
    tiny=t.c.CorridorTask(actor,task.t1,support,task.coverage,task.definition)
    assert len(tiny.waypoints)==2 and len(set(tiny.waypoints))==2


@pytest.mark.parametrize('method',['SEARCH','HYBRID'])
def test_resealed_exhaustive_method_cannot_claim_suboptimal_STOP(method):
    task,_,_=t.fixture()
    accounting={'policy_callback_calls':int(method=='HYBRID'),'nominal_candidate_comparisons':sum(r['legal'] for r in task.candidates)+1,'actor_forward_calls':0,'optimizer_updates':0}
    record=t.c._strategy(task,method,'STOP',accounting)
    with pytest.raises(ValueError):t.c._preflight(task,record,method)


def test_STOP_remains_valid_exhaustive_optimum_when_effort_exceeds_reach(tmp_path):
    task,prepared,arrays=t.fixture(definition_change=lambda d:d.update(reach_credit=.01,effort_per_mm=1.))
    assert all(r['nominal_score']<0 for r in task.candidates)
    sealed=t.seal(task,tmp_path/'seals');report=t.score(task,prepared,arrays,sealed,tmp_path/'score')
    assert report['status']=='evaluated_generated_corridor_batch'
    for method in ('SEARCH','HYBRID'):
        record=t.read_strategy(sealed,method)
        assert record['action_ids']==['STOP'] and record['public_metrics']['waypoint_reached'] is False
        assert report['methods'][method]['evaluation']['contacts']['whole_tool']['touched_reference_cells']==0


def rewrite_manifest(sealed,mutate):
    path=Path(sealed['manifest_path']);manifest=json.loads(path.read_text());mutate(manifest);path.write_text(json.dumps(manifest))
    return {**sealed,'manifest_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}


def test_noncomplete_slot_cannot_launder_actor_or_optimizer_accounting(tmp_path):
    task,prepared,arrays=t.fixture();selectors=t.controls();selectors['IL']=lambda _:None
    sealed=t.seal(task,tmp_path/'seals',selectors)
    sealed=rewrite_manifest(sealed,lambda m:m['methods']['IL']['accounting'].update(actor_forward_calls=1,optimizer_updates=1))
    calls=[]
    with pytest.raises(ValueError):t.score(task,prepared,arrays,sealed,tmp_path/'score',lambda:calls.append(True))
    assert calls==[]


def test_boolean_manifest_budget_does_not_equal_exact_integer_budget(tmp_path):
    task,prepared,arrays=t.fixture();sealed=t.seal(task,tmp_path/'seals')
    sealed=rewrite_manifest(sealed,lambda m:m['budgets'].update(horizon=True))
    calls=[]
    with pytest.raises(ValueError):t.score(task,prepared,arrays,sealed,tmp_path/'score',lambda:calls.append(True))
    assert calls==[]


@pytest.mark.parametrize('mutation', ['kind','fit_role','exposure','support_time','task_time','task_availability_hash'])
def test_actor_only_admission_invariants_cannot_be_weakened_after_preparation(mutation):
    task,_,_=t.fixture();actor=t.thaw_json(task.actor_contract)
    if mutation=='kind':actor['support']['kind']='private_MRA_derived_support'
    elif mutation=='fit_role':actor['support']['project_fit_roles']=['SELECT']
    elif mutation=='exposure':actor['support']['pretrained_exposure']='unseen_claim_without_enum'
    elif mutation=='support_time':actor['support']['available_at']='2026-10-08T23:59:00Z'
    elif mutation=='task_time':actor['task']['available_at']='2026-10-09T00:00:00Z'
    else:actor['task']['availability_record_sha256']='not-a-hash'
    with pytest.raises(ValueError):t.c.CorridorTask(actor,task.t1,task.support,task.coverage,task.definition)


def test_private_arrays_are_snapshotted_before_cross_method_scoring(tmp_path,monkeypatch):
    task,prepared,arrays=t.fixture();sealed=t.seal(task,tmp_path/'seals');original=t.c.contact.evaluate_contacts;calls=[]
    def counted(*args,**kwargs):
        calls.append(True)
        arrays['mask'][:]=False
        return original(*args,**kwargs)
    monkeypatch.setattr(t.c.contact,'evaluate_contacts',counted)
    report=t.score(task,prepared,arrays,sealed,tmp_path/'score')
    assert report['status']=='evaluated_generated_corridor_batch' and len(calls)==4
    for method in ('SEARCH','IL','HYBRID'):
        assert report['methods'][method]['evaluation']['contacts']['whole_tool']['positive_reference_cells']==2


def test_tissue_exposure_is_reported_without_surgical_feasibility_or_removal_claim(tmp_path):
    task,prepared,arrays=t.fixture();sealed=t.seal(task,tmp_path/'seals');record=t.read_strategy(sealed,'SEARCH')
    history=record['physical_history'][0]
    assert history['geometry']['starting_pose']['exposure_volume_mm3']==0.
    assert history['geometry']['insertion']['exposure_volume_mm3']>0.
    assert history['interpretation']=='hypothetical_geometric_corridor_no_removal_or_penetration_model'
    assert record['tissue_removal_assessed'] is False
