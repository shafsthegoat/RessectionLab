"""Generated small-array/forward controls only; no patient, native step or update."""
from dataclasses import replace
import hashlib
from pathlib import Path
from types import SimpleNamespace
import sys
import numpy as np
import pytest
import torch
import resectionlab

STAGE=Path(__file__).resolve().parents[1]/'src/resectionlab'
if str(STAGE) not in resectionlab.__path__:resectionlab.__path__.insert(0,str(STAGE))
from resectionlab.core import array_digest,freeze_json,semantic_digest,thaw_json
from resectionlab.geometry import AccessWindow,ToolGeometry
from resectionlab.spatial_observations import ObservedChannel,SpatialInputs,ObservedProcedureState,SpatialAction,build_spatial_observation
from resectionlab.spatial_policy import SpatialPolicy,SpatialPolicyConfig,parameter_hash
from resectionlab.public_target_context import VERSION,prepare_public_target,public_target_features
from resectionlab.patient_planning_admission import PatientPlanningContext,COHORT_SHA256,_public_observation_binding
from resectionlab.patient_planning_learning import PREFLIGHT_PROTOCOL,common_patient_policies


@pytest.fixture(autouse=True)
def one_thread():
    torch.set_num_threads(1)


def observation(*,goal=(9,3,3),affine=None,unknown=False,zero=False,domain=None,cavity=None,stop_only=False,support=None,crop_origin=(0,0,0)):
    shape=(12,8,8);crop=(4,4,4);region=tuple(slice(a,a+n) for a,n in zip(crop_origin,crop))
    affine=np.diag([2.,3.,4.,1.]) if affine is None else np.asarray(affine,np.float64)
    target=np.zeros(shape,np.float32)
    if not zero and not unknown:target[goal]=1.
    support=np.ones(shape,bool) if support is None else support
    domain=np.ones(shape,bool) if domain is None else domain
    cavity=np.zeros(shape,bool) if cavity is None else cavity
    identity=semantic_digest({'fixture':'permitted_target','target':array_digest(np.where(domain,target,0)),
        'domain':array_digest(domain),'affine':array_digest(affine),'unknown':unknown,'support':array_digest(support)})
    prepared=prepare_public_target(nominal_target=None if unknown else target,target_domain=None if unknown else domain,
        observed_support=support,affine_ras_mm=affine,crop_origin=crop_origin,crop_shape=crop,source_hash=identity,
        source_kind='unavailable' if unknown else 'supplied_annotation')
    channels={'structural_intensity':ObservedChannel(np.arange(64,dtype=np.float32).reshape(crop)/64,source_kind='observed_scan'),
        'nominal_tissue':ObservedChannel(support[region],source_kind='supplied_annotation'),
        'observed_cavity':ObservedChannel(cavity[region],source_kind='observed_procedure_state')}
    if not unknown:channels['nominal_target']=ObservedChannel(target[region],domain[region],'supplied_annotation')
    world=lambda point:affine[:3,:3]@np.asarray(point)+affine[:3,3]
    access=AccessWindow(tuple(world((0.,1.,1.))),tuple(affine[:3,0]/np.linalg.norm(affine[:3,0])),3.)
    tool=ToolGeometry('tool',.4,.2,50.,tip_length_mm=.5)
    actions=[SpatialAction('STOP')]
    if not stop_only:actions.append(SpatialAction('candidate',tuple(world((0.,1.,1.))),tuple(world((2.,1.,1.))),tool))
    crop_affine=affine.copy();crop_affine[:3,3]+=affine[:3,:3]@crop_origin
    base=build_spatial_observation(SpatialInputs(channels,crop_affine,'annotation_assisted',identity),actions,
        ObservedProcedureState(access,0,6,None))
    return replace(base,public_target_context=prepared.observe(cavity)),base,prepared


def policy(*,variant=True):
    config=SpatialPolicyConfig(encoder_channels=(2,3),hidden_features=8,ray_samples=3,critic_candidate_context=True)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        return SpatialPolicy(config,**({'public_target_context_variant':VERSION} if variant else {}))


def test_distant_target_visible_with_identical_empty_local_target():
    first,a,_=observation();second,b,_=observation(goal=(10,3,3))
    np.testing.assert_array_equal(a.image_channels,b.image_channels)
    assert not a.image_channels[2].any() and first.public_target_context.static['crop_mass_fraction']==0
    x,rx=public_target_features(first,reference_mm=10.);y,ry=public_target_features(second,reference_mm=10.)
    assert x.shape==(16,) and rx.shape==(2,4) and not np.array_equal(x,y) and not np.array_equal(rx,ry)
    assert first.public_target_context.static['mass_mm3']==pytest.approx(24.)
    assert rx[1,0]==pytest.approx(1.4) and np.array_equal(rx[0],np.zeros(4))


def test_unknown_known_zero_and_partial_domain_are_distinct():
    unknown,_,_=observation(unknown=True);zero,_,_=observation(zero=True)
    u,_=public_target_features(unknown,reference_mm=10.);z,_=public_target_features(zero,reference_mm=10.)
    assert tuple(u[:3])==(0,0,0) and tuple(z[:3])==(1,1,0)
    assert unknown.public_target_context.static['centroid_ras_mm'] is None
    domain=np.ones((12,8,8),bool);domain[:4]=False
    partial,_,_=observation(domain=domain)
    p,_=public_target_features(partial,reference_mm=10.)
    assert p[0]==1 and p[1]==pytest.approx(2/3) and p[2]==1
    assert not partial.coverage[2].any() and partial.public_target_context.static['crop_mass_fraction']==0


def test_unknown_payload_does_not_change_summary_or_identity():
    support=np.ones((5,5,5),bool);domain=np.ones(support.shape,bool);domain[0]=False
    a=np.zeros(support.shape,np.float32);a[4,4,4]=1;b=a.copy();b[0]=np.nan
    kwargs=dict(target_domain=domain,observed_support=support,affine_ras_mm=np.eye(4),crop_origin=(0,0,0),
        crop_shape=(3,3,3),source_hash='sha256:'+'a'*64)
    first=prepare_public_target(nominal_target=a,**kwargs);second=prepare_public_target(nominal_target=b,**kwargs)
    assert first.fingerprint==second.fingerprint


def test_unsupported_target_and_committed_overlap_keep_full_denominator():
    support=np.ones((12,8,8),bool);support[9,3,3]=False
    observed,_,_=observation(support=support)
    assert observed.public_target_context.static['mass_mm3']==pytest.approx(24.)
    assert observed.public_target_context.static['unsupported_mass_fraction']==1
    cavity=np.zeros((12,8,8),bool);cavity[9,3,3]=True
    progress,_,_=observation(cavity=cavity)
    values,_=public_target_features(progress,reference_mm=10.)
    assert values[-1]==1 and progress.public_target_context.static['mass_mm3']==pytest.approx(24.)
    assert not progress.image_channels[3].any()  # Progress is outside the local crop.


def test_immutable_frame_source_context_and_patient_binding():
    obs,base,prepared=observation();context=obs.public_target_context
    with pytest.raises((ValueError,TypeError)):context.static['mass_mm3']=0
    with pytest.raises(ValueError):prepared._target.setflags(write=True)
    with pytest.raises(ValueError):replace(base,source_id='sha256:'+'0'*64,public_target_context=context)
    shifted=base.affine_ras_mm.copy();shifted[0,3]+=1
    with pytest.raises(ValueError):replace(base,affine_ras_mm=shifted,public_target_context=context)
    changed_support=base.image_channels.copy();changed_support[1,0,0,0]=0
    with pytest.raises(ValueError):replace(base,image_channels=changed_support,public_target_context=context)
    record=freeze_json({'source_hash':obs.source_id,'observation_track':obs.track,'max_steps':6,
        'public_observation_binding':_public_observation_binding(obs)})
    admitted=PatientPlanningContext(record,semantic_digest(record));admitted.require_observations([obs])
    other,_,_=observation(goal=(10,3,3))
    with pytest.raises(ValueError):admitted.require_observations([other])
    object.__setattr__(context,'committed_overlap_mm3',1.)
    with pytest.raises(ValueError):obs.assert_intact()


def test_joint_rigid_transform_and_anisotropic_physical_scale():
    affine=np.diag([2.,3.,4.,1.]);a,_,_=observation(affine=affine)
    rotation=np.array([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])
    moved=affine.copy();moved[:3,:3]=rotation@affine[:3,:3];moved[:3,3]=[100,-30,7]
    b,_,_=observation(affine=moved)
    x,rx=public_target_features(a,reference_mm=10.);y,ry=public_target_features(b,reference_mm=10.)
    np.testing.assert_allclose(x,y,rtol=0,atol=1e-12);np.testing.assert_allclose(rx,ry,rtol=0,atol=1e-12)
    scaled=affine.copy();scaled[:3,:3]*=2;c,_,_=observation(affine=scaled)
    z,rz=public_target_features(c,reference_mm=10.)
    np.testing.assert_allclose(z[3:12],x[3:12]*2,rtol=0,atol=1e-12)
    np.testing.assert_allclose(rz,rx*2,rtol=0,atol=1e-12)


def test_nonzero_crop_origin_and_reflected_source_frame():
    first,_,_=observation(goal=(8,3,3))
    origin=(5,2,1);reflected=np.diag([-2.,3.,4.,1.]);reflected[:3,3]=[40,-10,7]
    second,_,_=observation(goal=(8,3,3),affine=reflected,crop_origin=origin)
    assert second.image_channels[2,3,1,2]==1
    expected=reflected.copy();expected[:3,3]+=reflected[:3,:3]@origin
    np.testing.assert_array_equal(second.affine_ras_mm,expected)
    a,ra=public_target_features(first,reference_mm=10.);b,rb=public_target_features(second,reference_mm=10.)
    np.testing.assert_allclose(a[:13],b[:13],rtol=0,atol=1e-12)
    np.testing.assert_allclose(ra,rb,rtol=0,atol=1e-12)
    assert a[13]==0 and b[13]==1  # Only local target retention changed.
    cavity=np.zeros((12,8,8),bool);cavity[8,3,3]=True
    changed,_,_=observation(goal=(8,3,3),affine=reflected,crop_origin=origin,cavity=cavity)
    assert changed.image_channels[3,3,1,2]==1
    assert public_target_features(changed,reference_mm=10.)[0][-1]==1


def test_private_future_and_task_objects_have_no_read_path():
    obs,_,_=observation()
    class Poison:
        def __array__(self,*a,**k):raise AssertionError('Forbidden payload read')
        def __getattr__(self,name):raise AssertionError('Forbidden payload read')
    before=public_target_features(obs,reference_mm=10.)
    for name in ('reference_target','future_reward','task','private_labels','future_cavity'):
        object.__setattr__(obs,name,Poison())
    after=public_target_features(obs,reference_mm=10.)
    for a,b in zip(before,after):np.testing.assert_array_equal(a,b)
    with pytest.raises(TypeError):prepare_public_target(reference_target=Poison())
    with torch.no_grad():policy()(obs)


def test_zero_added_columns_preserve_baseline_tensors_logits_and_value():
    baseline,expanded=policy(variant=False),policy()
    for name,old in baseline.state_dict().items():
        new=expanded.state_dict()[name]
        if name in ('actor.0.weight','stop.0.weight','critic.0.weight'):
            assert torch.equal(old,new[:,:old.shape[1]])
            assert torch.count_nonzero(new[:,old.shape[1]:])==0
        else:assert torch.equal(old,new)
    obs,base,_=observation()
    with torch.no_grad():a,av=baseline(base);b,bv=expanded(obs)
    torch.testing.assert_close(a,b,rtol=1e-6,atol=1e-7);torch.testing.assert_close(av,bv,rtol=1e-6,atol=1e-7)
    assert 'public_target_context_variant' not in baseline.architecture_record()
    assert baseline.architecture_hash!=expanded.architecture_hash
    with pytest.raises(ValueError,match='variant'):baseline(obs)
    with pytest.raises(ValueError,match='variant'):expanded(base)


@pytest.mark.parametrize('stop_only',[False,True])
def test_controlled_target_response_in_actor_STOP_and_critic(stop_only):
    model=policy()
    with torch.no_grad():
        for parameter in model.parameters():parameter.zero_()
        model.actor[0].weight[0,-4]=1.;model.actor[-1].weight[0,0]=1.
        for head in (model.stop,model.critic):head[0].weight[0,-16+3]=1.;head[-1].weight[0,0]=1.
        first,_,_=observation(stop_only=stop_only);second,_,_=observation(goal=(10,3,3),stop_only=stop_only)
        a,av=model(first);b,bv=model(second)
    assert float(b[0])>float(a[0]) and float(bv)>float(av)
    if not stop_only:assert float(b[1])>float(a[1])


def test_native_case_hook_is_observation_only_with_no_preview_or_step(monkeypatch):
    from resectionlab import native_spatial_task as native
    target=np.zeros((12,8,8),np.float32);target[9,3,3]=1
    support=np.ones(target.shape,bool);image=np.zeros(target.shape,np.float32)
    access=AccessWindow((-.5,3.,3.),(1.,0.,0.),2.)
    tool=ToolGeometry('tool',.4,.2,30.,tip_length_mm=.5)
    kwargs=dict(structural_intensity=image,observed_support=support,reference_target=target,affine_ras_mm=np.eye(4),
        access=access,tools=(tool,),nominal_target=target,target_derivation='generated supplied target',crop_shape=(4,4,4))
    plain=native.NativeSpatialCase(**kwargs)
    enhanced=native.NativeSpatialCase(**kwargs,public_target_context_variant=VERSION,public_target_domain=np.ones(target.shape,bool))
    assert plain._crop_origin==enhanced._crop_origin and plain._candidate_voxels==enhanced._candidate_voxels
    for name in ('structural_intensity','nominal_target','observed_support','reference_target'):
        np.testing.assert_array_equal(getattr(plain,name),getattr(enhanced,name))
    assert native.MAX_NATIVE_SPATIAL_STEPS==24
    owner=SimpleNamespace(case=enhanced,_engine=SimpleNamespace(removed_mask=np.zeros(target.shape,bool)),
        _steps=0,max_steps=6,_current_tool=None,tool_modes=None,_prepare_inventory=lambda:{})
    result=native.NativeSpatialTask.observation(owner)
    assert result.action_ids==('STOP',) and result.public_target_context.static['mass_mm3']==1
    assert result.public_target_context.static['crop_mass_fraction']==0


def test_opt_in_common_learning_is_declared():
    protocol=thaw_json(PREFLIGHT_PROTOCOL)
    protocol.update(version='generated-public-target-context-control',updates_per_method=8,
        public_target_context_variant=VERSION)
    protocol=freeze_json(protocol)
    TRAIN=('ReMIND-008','ReMIND-010','ReMIND-020','ReMIND-025')
    records=[]
    for subject in TRAIN:
        record=freeze_json({'role':'TRAIN','cohort_sha256':COHORT_SHA256,'scope':'patient_native_planning_experiment',
            'max_optimizer_updates':16,'patient_group':'ReMIND:'+subject[-3:],
            'learning_protocol_hash':semantic_digest(protocol),'public_target_context_variant':VERSION})
        records.append(PatientPlanningContext(record,semantic_digest(record)))
    il,rl=common_patient_policies(records,protocol)
    assert parameter_hash(il)==parameter_hash(rl) and il.public_target_context_variant==VERSION
