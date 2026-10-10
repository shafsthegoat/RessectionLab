"""Tiny generated relation/lineage controls; no native task or optimizer step."""
from dataclasses import replace
import hashlib
import numpy as np
import pytest
import torch
from resectionlab.goal_relation_spatial_policy import (GoalRelationSpatialPolicy,
    physical_goal_tip_relations,public_goal_relations,legal_relation_summary,CHECKPOINT_VERSION)
from resectionlab.goal_mode_spatial_policy import GoalModeSpatialPolicy
from resectionlab.contact_learning_contract import (freeze_contact_experiment,freeze_full_teacher_refit,
    freeze_goal_relation_fit,common_initial_policies,RELATION_FIT_VERSION)
from resectionlab.contact_checkpoint import encode_contact_checkpoint,decode_contact_checkpoint,initial_lineage
from resectionlab.spatial_policy import SpatialPolicyConfig,parameter_hash
from test_goal_mode_spatial_policy import observation,context
from test_contact_full_teacher_refit import states


def test_physical_units_anisotropic_spacing_and_joint_rigid_transform():
    affine=np.diag([2.,3.,4.,1.]);goal=np.array([2.,6.,12.]);tips=np.array([[0.,0.,0.],[2.,3.,8.]])
    expected=np.array([[.2,.6,1.2,np.sqrt(184)/10],[0.,.3,.4,.5]])
    actual=physical_goal_tip_relations(goal,tips,affine,reference_mm=10.)
    np.testing.assert_allclose(actual,expected,rtol=0,atol=1e-12)
    rotation=np.array([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]]);translation=np.array([100.,-30.,7.])
    moved=affine.copy();moved[:3,:3]=rotation@affine[:3,:3];moved[:3,3]=translation
    np.testing.assert_allclose(physical_goal_tip_relations(rotation@goal+translation,
        tips@rotation.T+translation,moved,reference_mm=10.),actual,rtol=0,atol=1e-12)
    np.testing.assert_allclose(physical_goal_tip_relations(goal*2,tips*2,affine*np.array([2,2,2,1])[:,None],
        reference_mm=10.),actual*2,rtol=0,atol=1e-12)
    shear=affine.copy();shear[0,1]=.2
    with pytest.raises(ValueError,match='orthogonal'):physical_goal_tip_relations(goal,tips,shear,reference_mm=10.)


def test_goal_swap_changes_public_relation_but_private_property_does_not():
    obs=observation();other=observation(goal=(3,2,3))
    before=public_goal_relations(obs,context=context(obs),reference_mm=10.)
    after=public_goal_relations(other,context=context(other),reference_mm=10.)
    assert not np.array_equal(before,after)
    assert np.array_equal(before[0],np.zeros(4))
    class Poison:
        def __array__(self,*args,**kwargs):raise AssertionError('Private property read')
        def __getattr__(self,name):raise AssertionError('Private property read')
    object.__setattr__(obs,'private_reference_target',Poison())
    object.__setattr__(obs,'future_reward',Poison())
    np.testing.assert_array_equal(public_goal_relations(obs,context=context(obs),reference_mm=10.),before)
    with pytest.raises(ValueError):public_goal_relations(other,context=context(obs),reference_mm=10.)


def test_legal_mode_summary_excludes_stop_masked_rows_and_is_permutation_invariant():
    values=torch.tensor([[9.,9.,9.,9.],[1.,2.,3.,4.],[5.,6.,7.,8.],[99.,99.,99.,99.]])
    mask=torch.tensor([True,True,True,False]);modes=('stop','aspirate','probe','aspirate')
    result=legal_relation_summary(values,mask,modes)
    torch.testing.assert_close(result[:4],values[1]);torch.testing.assert_close(result[9:13],values[2])
    order=[0,2,3,1]
    torch.testing.assert_close(legal_relation_summary(values[order],mask[order],tuple(modes[i] for i in order)),result)
    values[0]=1000;values[3]=-1000
    torch.testing.assert_close(legal_relation_summary(values,mask,modes),result)
    torch.testing.assert_close(legal_relation_summary(values,torch.tensor([True,False,False,False]),modes),torch.zeros(18))


def pair():
    config=SpatialPolicyConfig(encoder_channels=(2,3),hidden_features=8,ray_samples=3,critic_candidate_context=True)
    with torch.random.fork_rng(devices=[]):torch.manual_seed(11);old=GoalModeSpatialPolicy(config)
    with torch.random.fork_rng(devices=[]):torch.manual_seed(11);new=GoalRelationSpatialPolicy(config)
    return old,new


def test_shared_tensors_exact_added_columns_zero_and_initial_functions_match():
    old,new=pair()
    for name,tensor in old.state_dict().items():
        expanded=new.state_dict()[name]
        if name in ('actor.0.weight','stop.0.weight'):
            assert torch.equal(expanded[:,:tensor.shape[1]],tensor)
            assert torch.count_nonzero(expanded[:,tensor.shape[1]:])==0
        else:assert torch.equal(expanded,tensor)
    obs=observation()
    with torch.no_grad():
        a,av=old(obs,context=context(obs));b,bv=new(obs,context=context(obs))
    torch.testing.assert_close(a,b,rtol=1e-6,atol=1e-7);torch.testing.assert_close(av,bv,rtol=0,atol=0)
    assert old.architecture_hash!=new.architecture_hash and parameter_hash(old)!=parameter_hash(new)


def test_controlled_relation_weights_affect_candidate_and_stop_scores():
    _,model=pair()
    with torch.no_grad():
        for parameter in model.parameters():parameter.zero_()
        model.actor[0].weight[0,-4]=1.;model.actor[-1].weight[0,0]=1.
        model.stop[0].weight[0,-18]=1.;model.stop[-1].weight[0,0]=1.
        first=observation();second=observation(goal=(3,2,3))
        a,_=model(first,context=context(first));b,_=model(second,context=context(second))
    assert float(b[0])>float(a[0]) and float(b[1])>float(a[1])
    calls=[];hook=model.encoder.register_forward_pre_hook(lambda *unused:calls.append(True))
    try:
        with pytest.raises(ValueError):model(second,context=context(first))
        assert calls==[]
    finally:hook.remove()


def test_explicit_fit_keeps_defaults_and_refuses_cross_architecture_checkpoint():
    baseline=freeze_contact_experiment();assert baseline.fingerprint=='sha256:a01e6ba30fbf1e6efabf70f9637d42c0813931624cb50d2d393858029d4fd894'
    corpus=states();old=freeze_full_teacher_refit(corpus);new=freeze_goal_relation_fit(corpus)
    assert new.record()['version']==RELATION_FIT_VERSION and old.policy_type is GoalModeSpatialPolicy
    assert new.policy_type is GoalRelationSpatialPolicy and new.protocol['batch_size']==40 and new.protocol['updates']==32
    model=common_initial_policies(new)['IL'];payload=encode_contact_checkpoint(model,new,initial_lineage(new))
    digest=hashlib.sha256(payload).hexdigest()
    restored,metadata=decode_contact_checkpoint(payload,expected_sha256=digest,experiment=new,kind='initial')
    assert metadata['version']==CHECKPOINT_VERSION and parameter_hash(restored)==parameter_hash(model)
    with pytest.raises(ValueError):decode_contact_checkpoint(payload,expected_sha256=digest,experiment=old,kind='initial')
