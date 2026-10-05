"""Essential label/budget/optimizer control checks, without real case loading."""
from copy import deepcopy
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT=Path(__file__).parents[1]
spec=importlib.util.spec_from_file_location('visited_bc',ROOT/'scripts/run_real_patient_visited_imitation.py')
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)


def label_fixture():
    observation=SimpleNamespace(action_ids=('STOP','A','B'),action_mask=np.array([True,True,True]))
    metrics={'steps':1};inventory={'cavity_state_hash':'current'}
    accounting={'complete':True,'decisions':[{'step':1,'source_state_hash':'current',
        'scores':[{'action_id':'STOP','reward':0.},{'action_id':'A','reward':1.},{'action_id':'B','reward':-.1}],
        'all_current_legal_actions_scored':True,'legal_nonstop_actions':2,'scored_nonstop_actions':2,'selected_action_id':'A'}]}
    return observation,metrics,inventory,('A','STOP'),accounting


def test_live_teacher_returns_only_first_label_not_appended_stop():
    assert runner.validate_label(*label_fixture())=='A'
    args=label_fixture();args[-1]['decisions'][0]['scores'][1]['reward']=0.
    args[-1]['decisions'][0]['selected_action_id']='STOP'
    assert runner.validate_label(*args[:3],('STOP',),args[-1])=='STOP'


@pytest.mark.parametrize('kind',['state','mask','missing','terminal','maximum'])
def test_misaligned_or_incomplete_teacher_label_rejected(kind):
    obs,metrics,inventory,sequence,accounting=label_fixture()
    row=accounting['decisions'][0]
    if kind=='state':row['source_state_hash']='stale'
    if kind=='mask':obs.action_mask[1]=False
    if kind=='missing':row['scores'].pop()
    if kind=='terminal':accounting['decisions']=[];sequence=()
    if kind=='maximum':row['scores'][2]['reward']=2.
    with pytest.raises(ValueError):runner.validate_label(obs,metrics,inventory,sequence,accounting)


def test_six_examples_equal_workload_and_duplicate_initial_is_retained():
    original=(('initial','a'),('teacher2','b'),('teacher3','c'))
    visited=(('initial','a'),('visited2','d'),('visited3','e'))
    batches=runner.make_batches(original,visited)
    assert len(batches['control'])==len(batches['augmented'])==6
    assert len(set(batches['control']))==3 and len(set(batches['augmented']))==5
    assert batches['control'].count(('initial','a'))==batches['augmented'].count(('initial','a'))==2
    with pytest.raises(ValueError):runner.make_batches(original,visited[:2])


def test_separate_restores_isolate_model_and_actual_adam_tensor_state(tmp_path,monkeypatch):
    from resectionlab.spatial_policy import SpatialPolicy,parameter_hash
    torch.set_num_threads(1)
    policy=SpatialPolicy();optimizer=torch.optim.Adam(policy.parameters(),lr=.001)
    parameter=next(policy.parameters())
    optimizer.state[parameter]={'step':torch.tensor(8.),'exp_avg':torch.full_like(parameter,.1),'exp_avg_sq':torch.full_like(parameter,.2)}
    initial_hash=parameter_hash(policy)
    torch.save({'policy':policy.state_dict(),'optimizer':optimizer.state_dict(),'update':8},tmp_path/'latest.pt')
    monkeypatch.setattr(runner,'ANCHOR',tmp_path)
    definition={'policy_config':{},'expected_policy_architecture_hash':policy.architecture_hash}
    p1,o1,h1=runner.restore_branch(definition,initial_hash)
    with torch.no_grad():next(p1.parameters()).add_(3)
    o1.state[next(p1.parameters())]['exp_avg'].add_(9)
    p2,o2,h2=runner.restore_branch(definition,initial_hash)
    assert h1==h2==runner.optimizer_digest(optimizer.state_dict())
    assert parameter_hash(p2)==initial_hash and parameter_hash(p1)!=initial_hash
    assert torch.equal(o2.state[next(p2.parameters())]['exp_avg'],torch.full_like(parameter,.1))
    assert o1.state[next(p1.parameters())]['exp_avg'].data_ptr()!=o2.state[next(p2.parameters())]['exp_avg'].data_ptr()
