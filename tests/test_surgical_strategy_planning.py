"""Real-engine delayed-consequence checks on a small generated rescue scene."""
import copy
import numpy as np
import pytest
from resectionlab.surgical_engine import SurgicalEngine
from resectionlab.surgical_scene import SurgicalScene
from resectionlab.surgical_physics_v2 import default_profile
from resectionlab.surgical_rl_protection import ProtectionSet, ProtectionAnnotation, GuardConfig
from resectionlab.surgical_strategy_planning import compare_rescue, rescue_recipes
from resectionlab.surgical_planning import validate_saved_comparison
from resectionlab.core import semantic_digest


@pytest.fixture
def rescue():
    labels=np.zeros((24,24,24),np.uint16)
    labels[3:6,3:6,8:11]=1;labels[11,9:14,10]=2
    scene=SurgicalScene(labels,np.eye(4),[
        dict(id=1,name='Generated target',material='tumor',target=True),
        dict(id=2,name='Generated vessel',material='vessel',target=False)],
        dict(provenance='generated rescue contract fixture'),{'tissuePhysicsV2':default_profile()})
    mask=np.zeros(scene.shape,bool);mask[0,0,0]=True
    protection=ProtectionSet(scene,[ProtectionAnnotation.from_mask(scene,'brainstem',mask,provenance='generated exclusion')])
    engine=SurgicalEngine(scene,protection=protection,guard_config=GuardConfig())
    pose=lambda z:dict(position=[11.,11.,float(z)],quaternion=[0.,0.,0.,1.])
    def action(operation,duration=.2,**values):return dict(slot='left',operation=operation,duration=duration,**values)
    prefix=[action('exchange',instrument='micro_scissors'),
        action('place',pose=pose(6),parameters=dict(radiusMm=1.5,depthMm=2.,jawOpeningMm=0.,forceN=0.)),
        action('apply',duration=1.,pose=pose(10))]
    for a in prefix:assert engine.step(a)['status']!='rejected'
    assert engine.metrics()['activeVesselWounds']>0
    tail=[action('deactivate'),action('inspect',duration=2.),action('inspect',duration=5.),action('withdraw'),
        action('exchange',instrument='bipolar_forceps'),
        action('place',pose=pose(-4),parameters=dict(radiusMm=2.,power=1.,jawOpeningMm=0.,forceN=0.)),
        action('apply',duration=1.,pose=pose(10)),action('apply',duration=3.),action('deactivate'),action('inspect',duration=5.),action('withdraw'),
        action('exchange',instrument='irrigation'),
        action('place',pose=pose(-4),parameters=dict(radiusMm=3.,depthMm=.1,flowMm3PerSecond=2.,forceN=0.)),
        action('apply',duration=1.,pose=pose(8)),action('apply',duration=2.),action('deactivate'),action('withdraw'),
        action('exchange',instrument='frazier_suction'),
        action('place',pose=pose(-4),parameters=dict(radiusMm=3.,depthMm=.1,flowMm3PerSecond=2.,forceN=0.)),
        action('apply',duration=1.,pose=pose(8)),action('apply',duration=2.),action('deactivate'),action('withdraw'),
        action('inspect',duration=5.),dict(slot='right',operation='inspect',duration=.2)]
    original=engine.clone()
    for a in tail:assert original.step(a)['status']!='rejected'
    record=dict(branches={'reference':original.snapshot()},fullCase={'reference':dict(scenario='vascular')})
    return engine,record


def test_three_real_strategies_have_delayed_consequences_and_preserve_source(rescue):
    engine,record=rescue;before=engine.snapshot();source=engine.scene.labels.copy()
    recipes=rescue_recipes(engine,record)
    assert [len(actions) for _,_,actions in recipes]==[23,23,25]
    # Same intervention dose; suction-first changes order, not action count.
    assert sorted(map(semantic_digest,recipes[0][2]))==sorted(map(semantic_digest,recipes[1][2]))
    plans,report=compare_rescue(engine,record,acknowledge_unknowns=True)
    assert engine.snapshot()==before;np.testing.assert_array_equal(engine.scene.labels,source)
    assert len(plans)==3 and report['task']=='vascular-rescue'
    for candidate,p in plans:
        assert p['invalidActions']==0 and p['stopReason']=='sequence_complete'
        assert p['verification']['exactStateMatch'] and p['verification']['endpointVectorMatch']
        assert p['incrementalEndpoints']['tumor_removed_mm3']==0
        assert candidate.actions[:len(engine.actions)]==engine.actions
    assert plans[2][1]['summary']['additionalWoundFluidMm3']>plans[0][1]['summary']['additionalWoundFluidMm3']
    assert plans[2][1]['incrementalEndpoints']['elapsed_seconds']-plans[0][1]['incrementalEndpoints']['elapsed_seconds']==pytest.approx(7.)


def test_opt_in_cancellation_and_three_branch_persistence(rescue):
    engine,record=rescue;before=engine.snapshot()
    with pytest.raises(ValueError,match='acknowledge'):compare_rescue(engine,record)
    def cancel():raise InterruptedError('cancelled')
    with pytest.raises(InterruptedError):compare_rescue(engine,record,acknowledge_unknowns=True,check=cancel)
    assert engine.snapshot()==before
    plans,report=compare_rescue(engine,record,acknowledge_unknowns=True)
    for i,(candidate,plan) in enumerate(plans):
        plan['branchId']='strategy-'+str(i);record['branches'][plan['branchId']]=candidate.snapshot()
    report['comparisonSeal']=semantic_digest(report)
    validate_saved_comparison(engine.scene,record,report)
    tampered=copy.deepcopy(report);tampered['alternatives'].pop();tampered.pop('comparisonSeal');tampered['comparisonSeal']=semantic_digest(tampered)
    with pytest.raises(ValueError,match='distinct'):validate_saved_comparison(engine.scene,record,tampered)
