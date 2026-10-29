"""Native observation is transactional through candidates and failed microsteps."""
import numpy as np
import pytest

from resectionlab.surgical_engine import SurgicalEngine
from resectionlab.surgical_continuum_capture import MechanicsCaptureEngine,capture_mechanics_inputs
from resectionlab import surgical_continuum as c,surgical_deformation_v2 as brain
from resectionlab.surgical_physics_v2 import default_profile
from resectionlab.surgical_scene import SurgicalScene


def engines():
    props={'tissuePhysicsV2':default_profile(),c.PROPERTY:c.research_profile({'1':{
        'shearPa':1000.,'poissonRatio':.3,'relaxationShearPa':[500.],'source':'Generated capture control'}},relaxation_seconds=[.5])}
    props['tissuePhysicsV2']['brain']['radiusMm']=2.
    scene=SurgicalScene(np.ones((9,9,9),np.uint16),np.eye(4),
        [{'id':1,'name':'Generated soft block','material':'soft_tissue'}],{'kind':'generated capture control'},props)
    result=[SurgicalEngine(scene),MechanicsCaptureEngine(scene)]
    for e in result:
        cell=int(np.ravel_multi_index((4,4,4),scene.shape));brain.begin_hold(e,'left',[cell],[4.,4.,4.]);brain.set_target(e,'left',[4.01,4.,4.])
    return result


@pytest.mark.parametrize('fail_at',[1,2,None])
def test_only_committed_successful_continuum_inputs_publish(monkeypatch,fail_at):
    base,observed=engines();original=SurgicalEngine._physical_substep
    def physical(self,*args):
        result=original(self,*args);self._unit_calls=getattr(self,'_unit_calls',0)+1
        if self._unit_calls==fail_at:raise ValueError('UNIT_POST_PHYSICS_STOP')
        return result
    monkeypatch.setattr(SurgicalEngine,'_physical_substep',physical)
    tick=c.tick;action={'slot':'right','operation':'inspect','duration':.3}
    with capture_mechanics_inputs():
        expected=base.step(action);actual=observed.step(action)
    assert c.tick is tick and actual==expected and observed.snapshot()==base.snapshot()
    trace=observed.mechanics_inputs();expected_ticks=0 if fail_at==1 else 1 if fail_at==2 else 3
    assert len(trace['ticks'])==expected_ticks
    assert expected['status']==('rejected' if fail_at==1 else 'partial' if fail_at==2 else 'accepted')
    assert all(row['nativeFixedSteps']==12 and row['bodyForce']=='zero' for row in trace['ticks'])
    trace['ticks'].clear();assert len(observed.mechanics_inputs()['ticks'])==expected_ticks


def test_capture_cancellation_does_not_publish_candidate_or_leave_global_hook():
    _,observed=engines();before=observed.snapshot();tick=c.tick;calls=0
    def check():
        nonlocal calls
        calls+=1
        if calls==4:raise InterruptedError('Unit cancellation')
    with pytest.raises(InterruptedError),capture_mechanics_inputs():
        observed.step({'slot':'right','operation':'inspect','duration':.3},check=check)
    assert c.tick is tick and observed.snapshot()==before and observed.mechanics_inputs()['ticks']==[]


def test_removed_contact_owner_is_absent_from_post_tick_inputs():
    base,observed=engines()
    for e in [base,observed]:
        cell=e.state['holds']['left']['cells'][0]
        e.state['removed']=[cell]
    with capture_mechanics_inputs():
        a=base.step({'slot':'right','operation':'inspect','duration':.1})
        b=observed.step({'slot':'right','operation':'inspect','duration':.1})
    assert a==b and base.snapshot()==observed.snapshot()
    trace=observed.mechanics_inputs()
    assert trace['ticks'] and trace['ticks'][0]['contacts']==[]
    assert cell not in trace['topologies'][trace['ticks'][0]['topologyId']]['nativeActiveSourceIds']
