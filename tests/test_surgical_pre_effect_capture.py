"""Pre-effect observations cannot leak rejected/rolled-back native attempts."""
import copy
import numpy as np
import pytest

from resectionlab import surgical_cusa_v3 as cusa,surgical_deformation_v2 as brain
from resectionlab.surgical_engine import SurgicalEngine
from resectionlab.surgical_scene import SurgicalScene
from resectionlab.surgical_continuum_capture import MechanicsCaptureEngine
from resectionlab.surgical_pre_effect_capture import PreEffectCaptureEngine,capture_pre_effects
from test_surgical_continuum_capture import engines


def prepared(position=(9.,9.,7.25)):
    template,_=engines();labels=np.zeros((20,20,20),np.uint16);labels[1:6,1:6,1:6]=1;labels[9,9,9]=1
    props=copy.deepcopy(template.scene.properties);props['cusaV3']=cusa.default_profile();props['tissuePhysicsV2']['brain']['radiusMm']=3.
    scene=SurgicalScene(labels,np.eye(4),template.scene.regions,{'kind':'generated pre-effect control'},props)
    result=[MechanicsCaptureEngine(scene),PreEffectCaptureEngine(scene,selected_ticks=range(1,200))]
    with capture_pre_effects():
        for engine in result:
            engine.state['toolMechanicsRevision']=2
            cell=int(np.ravel_multi_index((3,3,3),scene.shape))
            brain.begin_hold(engine,'left',[cell],[3.,3.,3.]);brain.set_target(engine,'left',[3.01,3.,3.])
            engine.state['slots']['right']['position']=list(position)
            engine.step({'slot':'right','operation':'inspect','duration':.1})
    assert result[0].snapshot()==result[1].snapshot()
    return result


@pytest.mark.parametrize('fail_at',[1,2,None])
def test_only_fully_committed_microticks_publish_precise_old_state_and_cusa_calls(monkeypatch,fail_at):
    base,observed=prepared();original=SurgicalEngine._physical_substep
    def physical(self,*args):
        events=original(self,*args);self._unit_calls=getattr(self,'_unit_calls',0)+1
        if self._unit_calls==fail_at:raise ValueError('UNIT_LATE_FAILURE')
        return events
    monkeypatch.setattr(SurgicalEngine,'_physical_substep',physical)
    functions=(cusa.contact_limited_pose,cusa.apply,cusa._ordered_contacts)
    action={'slot':'right','operation':'apply','duration':.3}
    before=copy.deepcopy(observed.state)
    with capture_pre_effects():expected=base.step(action);actual=observed.step(action)
    assert actual==expected and observed.snapshot()==base.snapshot()
    assert (cusa.contact_limited_pose,cusa.apply,cusa._ordered_contacts)==functions
    frames=observed.pre_effect_frames();assert len(frames)==(0 if fail_at==1 else 1 if fail_at==2 else 3)
    if frames:
        frame=frames[0];assert frame['geometryPhase']=='before-physical-substep'
        assert frame['cusaBefore']==before['cusaV3'] and frame['removedSourceIds']==before['removed']
        assert len(frame['limiterCalls'])==len(frame['applicationCalls'])==1
        assert frame['limiterCalls'][0]['resolvedMm']==[9.,9.,7.25]
        app=frame['applicationCalls'][0]
        assert app['cusaBefore']==frame['cusaBefore'] and app['cusaAfter']!=app['cusaBefore']
        assert len(app['orderedQueries'])==1
        assert np.max(np.abs(frame['cornerDisplacementMm']))>0
        frame['cornerDisplacementMm'][:]=100.;frame['cusaBefore']['inputCapacityMm3']=999.
        assert np.max(observed.pre_effect_frames()[0]['cornerDisplacementMm'])<1.
        assert observed.pre_effect_frames()[0]['cusaBefore']['inputCapacityMm3']!=999.


def test_whole_cell_removal_is_bound_to_actual_pre_effect_budget():
    base,observed=prepared();cell=int(np.ravel_multi_index((9,9,9),base.scene.shape))
    with capture_pre_effects():
        for _ in range(5):
            action={'slot':'right','operation':'apply','duration':.1}
            assert base.step(action)==observed.step(action)
    assert observed.snapshot()==base.snapshot();frame=observed.pre_effect_frames()[-1]
    assert cell not in frame['removedSourceIds'] and cell in frame['removedSourceIdsAfter']
    app=frame['applicationCalls'][0]
    assert app['events'][0]['removedCells']==[cell] and str(cell) in app['cusaBefore']['progressExact']
    assert str(cell) not in app['cusaAfter']['progressExact']


def test_actual_limiter_query_trace_and_clipped_pose_are_not_requested_pose():
    base,observed=prepared(position=(9.,9.,3.));action={'slot':'right','operation':'apply','duration':.1,
        'pose':{'position':[9.,9.,11.],'quaternion':[0.,0.,0.,1.]}}
    with capture_pre_effects():assert base.step(action)==observed.step(action)
    assert base.snapshot()==observed.snapshot();frames=observed.pre_effect_frames()
    assert any(f['limiterCalls'][0]['requestedMm']!=f['limiterCalls'][0]['resolvedMm'] for f in frames)
    assert any(len(f['limiterCalls'][0]['queries'])>2 for f in frames)
    assert all(f['resolvedSlotsAfter']['right']['position']==f['limiterCalls'][0]['resolvedMm'] for f in frames)


def test_root_cancellation_discards_candidate_ledgers_and_restores_hooks():
    _,observed=prepared();before=observed.snapshot();functions=(cusa.apply,cusa._ordered_contacts,cusa.contact_limited_pose);calls=0
    def check():
        nonlocal calls
        calls+=1
        if calls==4:raise InterruptedError('Unit pre-effect cancellation')
    with pytest.raises(InterruptedError),capture_pre_effects():
        observed.step({'slot':'right','operation':'apply','duration':.3},check=check)
    assert observed.snapshot()==before and observed.pre_effect_frames()==[]
    assert (cusa.apply,cusa._ordered_contacts,cusa.contact_limited_pose)==functions
