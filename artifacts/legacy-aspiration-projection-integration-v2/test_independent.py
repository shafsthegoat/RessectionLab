"""Projection/search and binding controls only: no forward, checkpoint bytes or fit."""
from pathlib import Path
import resectionlab
import pytest
import numpy as np
resectionlab.__path__.insert(0,str(Path(__file__).resolve().parents[1]/'shared-legacy-aspiration-desktop-v2/resectionlab'))
from resectionlab.development_episode import make_development_task
from resectionlab.legacy_aspiration_projection import AspirationOnlyNativeTask,project_aspiration_observation
from resectionlab.sequential_spatial_observation import SequentialSpatialObservation
from resectionlab.spatial_observations import SpatialObservation
from resectionlab.observed_search import observed_beam_search
from resectionlab.native_spatial_task import NativeSpatialTask

def select(native,mode):
    obs=native.observation()
    return next(action for action,current in zip(obs.action_ids,obs.action_modes) if current==mode)

def opened():
    native=make_development_task();native.step(select(native,'aspirate'));return native

def test_direct_projection_refuses_real_nonzero_contact():
    native=opened();native.step(select(native,'probe'))
    assert native.observation().observed_probe_contact_grid.any()
    with pytest.raises(ValueError):project_aspiration_observation(native.observation())

def test_constructor_checks_history_independently_of_zero_contact_mask(monkeypatch):
    # Isolate the history guard; actual native probes require contact today.
    native=make_development_task();assert not native._engine.probe_contact_mask.any()
    metrics=native.metrics();metrics['history']=[{'interaction_mode':'probe'}]
    monkeypatch.setattr(native,'metrics',lambda:metrics)
    with pytest.raises(ValueError):AspirationOnlyNativeTask(native)

def test_existing_wrapper_refuses_new_probe_through_native_alias():
    native=make_development_task();view=AspirationOnlyNativeTask(native)
    view.step(select(native,'aspirate'));native.step(select(native,'probe'))
    with pytest.raises(ValueError):view.observation()

@pytest.mark.parametrize('branch',['clone','planning_clone','fresh'])
def test_all_branch_paths_refuse_probe_and_keep_native_parent_unchanged(branch):
    native=opened();view=AspirationOnlyNativeTask(native);parent=native._engine.state_hash
    child=getattr(view,branch)()
    if branch=='fresh':child.step(select(child._native,'aspirate'))
    probe=select(child._native,'probe');before=child._engine.state_hash
    for method in ('step','advance_planning'):
        with pytest.raises(ValueError):getattr(child,method)(probe)
        assert child._engine.state_hash==before
    assert type(child.observation()) is SpatialObservation
    assert native._engine.state_hash==parent

@pytest.mark.parametrize('mode',['eager','lazy_planning'])
def test_real_search_never_commits_probe(mode,monkeypatch):
    view=AspirationOnlyNativeTask(opened());committed=[]
    for name in ('step','advance_planning'):
        original=getattr(NativeSpatialTask,name)
        def record(self,action,_original=original):
            result=_original(self,action);committed.append(result.info['interaction_mode']);return result
        monkeypatch.setattr(NativeSpatialTask,name,record)
    sequence,accounting=observed_beam_search(view,max_calls=3,beam_width=1,seconds=5.,transition_mode=mode)
    assert committed and set(committed)<={'aspirate','stop'}
    assert accounting['actor_forward_calls']==0
    assert accounting['model_transition_calls']<=3
    assert sequence and all(type(action) is str for action in sequence)

@pytest.fixture(autouse=True)
def forbid_weights_and_inference(monkeypatch):
    import torch
    from resectionlab.spatial_policy import SpatialPolicy
    def forbidden(*args,**kwargs):raise AssertionError('Review forbids weights/inference/training')
    monkeypatch.setattr(torch,'load',forbidden)
    monkeypatch.setattr(SpatialPolicy,'forward',forbidden)
    monkeypatch.setattr(SpatialPolicy,'act',forbidden)


def bound_control():
    from resectionlab.spatial_policy import SpatialPolicy, SpatialPolicyConfig
    from resectionlab.shared_episode import identity_for_untrained_spatial_policy
    from resectionlab.legacy_aspiration_projection import BoundAspirationTransferPolicy
    task=AspirationOnlyNativeTask(make_development_task())
    actor=SpatialPolicy(SpatialPolicyConfig(encoder_channels=(2,2),hidden_features=4,ray_samples=2)).eval()
    identity=identity_for_untrained_spatial_policy(actor,policy_id='independent-binding-software-control')
    return task,actor,identity,BoundAspirationTransferPolicy(actor,identity,task)


def test_unchanged_binding_passes_without_forward():
    task,actor,identity,bound=bound_control()
    bound.assert_binding_intact()
    assert bound.projection_receipt['base_parameter_hash']==identity.parameter_hash
    assert bound.projection_receipt['scope']=='generated_transfer_not_checkpoint_training_domain'


@pytest.mark.parametrize('field,value',[
    ('expected_source_hash','sha256:'+'0'*64),
    ('expected_decision_model_hash','sha256:'+'1'*64),
    ('aspirator_tool_ids',frozenset({'development-probe'})),
    ('_integration_verified_checkpoint_sha256','sha256:'+'2'*64),
])
def test_live_identity_drift_refuses_before_forward(field,value):
    task,actor,identity,bound=bound_control()
    setattr(bound,field,value)
    with pytest.raises(RuntimeError,match='changed after binding'):
        bound.act(task.observation())


def test_base_architecture_drift_refuses_before_forward():
    task,actor,identity,bound=bound_control()
    from dataclasses import replace
    actor.config=replace(actor.config,ray_samples=3)
    with pytest.raises(RuntimeError,match='changed after binding'):
        bound.act(task.observation())


def test_base_parameter_drift_refuses_before_forward():
    import torch
    task,actor,identity,bound=bound_control()
    with torch.no_grad():next(actor.parameters()).add_(1.)
    with pytest.raises(RuntimeError,match='changed after binding'):
        bound.act(task.observation())


def test_wrapper_identity_drift_refuses_before_forward():
    from dataclasses import replace
    task,actor,identity,bound=bound_control()
    bound.identity=replace(bound.identity,policy_id='changed-policy-claim')
    with pytest.raises(RuntimeError,match='changed after binding'):
        bound.act(task.observation())


def test_synthetic_checkpoint_receipt_stays_distinct_and_tag_drift_refuses():
    # Artificial metadata tests the trained-branch contract, not authentic checkpoint admission.
    from dataclasses import replace
    from resectionlab.legacy_aspiration_projection import BoundAspirationTransferPolicy
    from resectionlab.core import semantic_digest
    task,actor,identity,unused=bound_control()
    original=replace(identity,training_status='trained_checkpoint',
        checkpoint_sha256='sha256:'+'4'*64,
        checkpoint_validation_receipt_sha256='sha256:'+'5'*64)
    actor._integration_verified_checkpoint_sha256=original.checkpoint_sha256
    bound=BoundAspirationTransferPolicy(actor,original,task)
    bound.assert_binding_intact()
    assert bound.projection_receipt['checkpoint_validation_receipt_sha256']==original.checkpoint_validation_receipt_sha256
    assert bound.identity.checkpoint_validation_receipt_sha256==semantic_digest(bound.projection_receipt)
    assert bound.identity.checkpoint_validation_receipt_sha256!=original.checkpoint_validation_receipt_sha256
    assert bound.identity.checkpoint_sha256==original.checkpoint_sha256
    actor._integration_verified_checkpoint_sha256='sha256:'+'6'*64
    with pytest.raises(RuntimeError,match='changed after binding'):
        bound.act(task.observation())
