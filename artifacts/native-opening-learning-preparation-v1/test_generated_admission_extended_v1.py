from dataclasses import replace
from copy import deepcopy
import pytest
import torch
from resectionlab.data_policy import GeneratedDevelopmentContext
from resectionlab.native_spatial_task import make_native_opening_task, NativeSpatialTask
from resectionlab.spatial_policy import SpatialPolicy, imitation_loss, gradient_step, parameter_hash

@pytest.fixture(scope='module')
def task():
    torch.set_num_threads(1)
    return make_native_opening_task()

def ctx(task):return GeneratedDevelopmentContext('1'*64,(task.case.source_hash,),task.decision_model_hash)
def model():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(4)
        return SpatialPolicy()

def test_task_reward_model_mismatch_refused(task):
    changed=NativeSpatialTask(task.case,max_steps=2,reward=replace(task.reward_spec,normal_per_mm3=.5))
    assert changed.case.source_hash==task.case.source_hash
    assert changed.decision_model_hash!=task.decision_model_hash
    with pytest.raises(ValueError,match='source/reward/horizon'):
        ctx(task).require_task(changed)

def test_context_admits_exact_frozen_task(task):
    ctx(task).require_task(task)

def test_loss_one_use_even_when_zero_learning_rate_keeps_same_weights(task):
    m=model();context=ctx(task)
    loss,_=imitation_loss(m,[(task.observation(),'STOP')],learning_context=context)
    before=parameter_hash(m)
    optimizer=torch.optim.Adam(m.parameters(),lr=0.)
    gradient_step(m,optimizer,loss,learning_context=context)
    assert before==parameter_hash(m)
    with pytest.raises(ValueError,match='matching admitted loss'):
        gradient_step(m,optimizer,loss,learning_context=context)

def test_same_model_replaced_equal_weight_parameter_objects_refused_before_backward(task):
    m=model();context=ctx(task)
    loss,_=imitation_loss(m,[(task.observation(),'STOP')],learning_context=context)
    before=parameter_hash(m); m.actor=deepcopy(m.actor)
    assert before==parameter_hash(m)
    called=[];loss.register_hook(lambda g:called.append(True))
    error=None; result=None
    try:result=gradient_step(m,torch.optim.Adam(m.parameters()),loss,learning_context=context)
    except Exception as exc:error=exc
    assert isinstance(error,ValueError), 'Replaced graph parameter objects admitted: '+repr(result)
    assert not called

def test_changed_forward_configuration_refused_before_backward(task):
    m=model();context=ctx(task)
    loss,_=imitation_loss(m,[(task.observation(),'STOP')],learning_context=context)
    before=parameter_hash(m); old_arch=m.architecture_hash
    m.config=replace(m.config,physical_reference_mm=20.)
    assert before==parameter_hash(m) and old_arch!=m.architecture_hash
    called=[];loss.register_hook(lambda g:called.append(True))
    error=None; result=None
    try:result=gradient_step(m,torch.optim.Adam(m.parameters()),loss,learning_context=context)
    except Exception as exc:error=exc
    assert isinstance(error,ValueError), 'Changed policy normalization admitted: '+repr(result)
    assert not called
