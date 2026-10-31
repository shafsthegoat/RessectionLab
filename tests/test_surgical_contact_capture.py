"""Matching native BE phases, clipped poses and rollback-safe geometry capture."""
import copy
import numpy as np
import pytest

from resectionlab.surgical_contact_capture import ContactCaptureEngine,capture_contact_geometry
from resectionlab.surgical_engine import SurgicalEngine
from resectionlab.surgical_scene import SurgicalScene
from resectionlab import surgical_continuum as c,surgical_deformation_v2 as brain
from test_surgical_continuum_capture import engines
from resectionlab.surgical_continuum_capture import MechanicsCaptureEngine,capture_mechanics_inputs
from resectionlab.surgical_cusa_v3 import default_profile as cusa_profile


@pytest.mark.parametrize('fail_at',[1,2,None])
@pytest.mark.parametrize('radius',[2.,3.])
def test_geometry_capture_is_exact_and_only_committed_microticks_publish(monkeypatch,fail_at,radius):
    template,_=engines();props=copy.deepcopy(template.scene.properties);props['tissuePhysicsV2']['brain']['radiusMm']=radius
    scene=SurgicalScene(template.scene.labels,template.scene.affine,template.scene.regions,template.scene.source,props)
    base=SurgicalEngine(scene);observed=ContactCaptureEngine(scene,selected_phases=[(0,0),(0,11),(1,0)])
    cell=int(np.ravel_multi_index((4,4,4),base.scene.shape));brain.begin_hold(observed,'left',[cell],[4.,4.,4.]);brain.set_target(observed,'left',[4.01,4.,4.])
    brain.begin_hold(base,'left',[cell],[4.,4.,4.]);brain.set_target(base,'left',[4.01,4.,4.])
    original=SurgicalEngine._physical_substep
    def physical(self,*args):
        events=original(self,*args);self._unit_calls=getattr(self,'_unit_calls',0)+1
        if self._unit_calls==fail_at:raise ValueError('UNIT_AFTER_CONTINUUM')
        return events
    monkeypatch.setattr(SurgicalEngine,'_physical_substep',physical)
    functions=(c.tick,c._system,c.update_memory)
    with capture_contact_geometry():
        action={'slot':'right','operation':'inspect','duration':.3}
        expected=base.step(action);actual=observed.step(action)
    assert (c.tick,c._system,c.update_memory)==functions
    assert actual==expected and observed.snapshot()==base.snapshot()
    frames=observed.contact_frames();assert len(frames)==(0 if fail_at==1 else 2 if fail_at==2 else 3)
    if frames:
        assert frames[0]['withinTick']==0 and frames[1]['atCommittedTickEnd']
        assert frames[0]['sourceCells']==frames[1]['sourceCells']
        # Radius2's generated diamond has fully constrained centre corners;
        # radius3 also exercises a genuinely moving interior source element.
        assert (np.max(np.abs(frames[0]['cornerDisplacementMm']))>0)==(radius==3.)
        assert frames[0]['slots']['left']['position']==observed.state['slots']['left']['position']
        frames[0]['cornerDisplacementMm'][:]=123.
        assert np.max(observed.contact_frames()[0]['cornerDisplacementMm'])<1.


def test_geometry_capture_cancellation_restores_hooks_and_root_state():
    base,_=engines();observed=ContactCaptureEngine(base.scene,selected_phases=[(0,0)])
    observed.state=copy.deepcopy(base.state)
    before=observed.snapshot();functions=(c.tick,c._system,c.update_memory);calls=0
    def check():
        nonlocal calls
        calls+=1
        if calls==4:raise InterruptedError('Unit cancellation')
    with pytest.raises(InterruptedError),capture_contact_geometry():observed.step({'slot':'right','operation':'inspect','duration':.3},check=check)
    assert observed.snapshot()==before and observed.contact_frames()==[]
    assert (c.tick,c._system,c.update_memory)==functions


def test_unavailable_fixed_step_is_an_observer_failure_not_a_rejected_physics_action():
    base,_=engines();observed=ContactCaptureEngine(base.scene,selected_phases=[(0,12)])
    cell=int(np.ravel_multi_index((4,4,4),base.scene.shape));brain.begin_hold(observed,'left',[cell],[4.,4.,4.])
    before=observed.snapshot()
    with pytest.raises(RuntimeError,match='incomplete'),capture_contact_geometry():
        observed.step({'slot':'right','operation':'inspect','duration':.1})
    assert observed.snapshot()==before and observed.contact_frames()==[]


def test_actual_cusa_limiter_captures_resolved_pose_instead_of_requested_endpoint():
    template,_=engines();labels=np.zeros((20,20,20),np.uint16);labels[1:6,1:6,1:6]=1;labels[9,9,9]=1
    props=copy.deepcopy(template.scene.properties);props['cusaV3']=cusa_profile()
    scene=SurgicalScene(labels,np.eye(4),template.scene.regions,{'kind':'generated clipped-pose control'},props)
    base=MechanicsCaptureEngine(scene);cell=int(np.ravel_multi_index((3,3,3),scene.shape))
    brain.begin_hold(base,'left',[cell],[3.,3.,3.]);base.state['slots']['right']['position']=[9.,9.,3.]
    initial=copy.deepcopy(base.state)
    action={'slot':'right','operation':'apply','duration':.1,
        'pose':{'position':[9.,9.,11.],'quaternion':[0.,0.,0.,1.]}}
    with capture_mechanics_inputs():expected=base.step(action)
    ticks=base.mechanics_inputs()['ticks'];index=max(i for i,tick in enumerate(ticks) if tick['nativeFixedSteps']>0)
    observed=ContactCaptureEngine(scene,selected_phases=[(index,ticks[index]['nativeFixedSteps']-1)]);observed.state=initial
    with capture_contact_geometry():actual=observed.step(action)
    assert actual==expected and observed.snapshot()==base.snapshot()
    assert actual['status']=='partial' and actual['reasonCode']=='TISSUE_ASPIRATION_IN_PROGRESS'
    frame=observed.contact_frames()[0]
    assert frame['slots']['right']['position']==base.state['slots']['right']['position']
    assert frame['requestedPose']['position']==[9.,9.,11.]
    assert frame['slots']['right']['position'][2]<8.
