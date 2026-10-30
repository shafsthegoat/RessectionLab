"""Exact native oracle, one-step independence, old-node and capacity controls."""
import copy
from fractions import Fraction
from types import SimpleNamespace
import numpy as np
import pytest

from resectionlab import surgical_cusa_v3 as cusa
from resectionlab.geometry import GeometryScene
from resectionlab.surgical_continuum_program import field_hash
from resectionlab.surgical_pre_effect_capture import capture_pre_effects
from resectionlab.surgical_pre_effect_witness import (pre_event_fields,capacity_prediction,
    pre_effect_prediction,require_native_prediction_exact,require_pre_effect_context)
from test_surgical_pre_effect_capture import prepared


def test_native_prediction_matches_actual_clipped_limiter_capacity_and_sweep_context():
    _,engine=prepared(position=(9.,9.,3.))
    with capture_pre_effects():engine.step({'slot':'right','operation':'apply','duration':.1,
        'pose':{'position':[9.,9.,11.],'quaternion':[0.,0.,0.,1.]}})
    frames=engine.pre_effect_frames();grid=GeometryScene(np.zeros(engine.scene.shape,bool),engine.scene.affine)
    chosen=[f for f in frames if f['mechanics']['nativeFixedSteps']>0]
    assert chosen
    for frame in chosen:
        prediction=pre_effect_prediction(engine.scene,grid,frame,np.asarray(frame['sourceCells']),frame['cornerDisplacementMm'])
        require_native_prediction_exact(frame,prediction)
        assert not prediction['predictedStatePropagated'] and not prediction['largerDomainPublicActionsAdmitted']
        assert all(not piece['actualSweptCollisionEstablished'] for piece in prediction['covers'])
        wrong=copy.deepcopy(prediction);wrong['events'][0]['inputCapacityMm3']+=1e-9
        with pytest.raises(ValueError,match='NATIVE_PREDICTION_DIFFERS'):require_native_prediction_exact(frame,wrong)
        for key,value in [('positionMm',[100.,100.,100.]),('candidateSourceIds',[-1])]:
            damaged=copy.deepcopy(frame);damaged['applicationCalls'][0]['orderedQueries'][0][key]=value
            with pytest.raises(ValueError,match='NATIVE_PREDICTION_DIFFERS'):require_native_prediction_exact(damaged,prediction)


@pytest.mark.parametrize('dt,power,duty',[(.1,.9,.7),(.5,1.,1.),(0.,1.,1.),(.15,.25,.125)])
@pytest.mark.parametrize('active',[True,False])
def test_capacity_allocation_matches_original_exact_float_budget_and_rational_remainders(dt,power,duty,active):
    params={'radiusMm':1.5,'power':power,'duty':duty};profile=cusa.default_profile()
    scene=SimpleNamespace(properties={'cusaV3':profile},labels=np.ones(3,np.uint16),voxel_mm3=1.,
        affine=np.eye(4),points=lambda ids:np.array([[float(i),0.,0.] for i in ids]))
    engine=SimpleNamespace(scene=scene,state={'removed':[],'holds':{},'toolMechanicsRevision':2},
        by_label={1:{'material':'soft_tissue'}},touched=lambda a,b,r:[0,1,2])
    cusa.initialize(engine);cusa.apply(engine,[0,1,2],.15,{'radiusMm':1.5},position=[0.,0.,0.])
    before=copy.deepcopy(engine.state['cusaV3']);old=copy.deepcopy(before)
    prediction=capacity_prediction(before,[],[0,1,2],1.,profile,params,dt,active)
    expected=cusa.apply(engine,[0,1,2],dt,params,position=[0.,0.,0.],active=active)
    assert prediction['cusaAfter']==engine.state['cusaV3'] and prediction['events']==expected
    assert prediction['removedSourceIdsAfter']==engine.state['removed'] and before==old
    if active and dt==.1:
        budget=Fraction.from_float(2.*power*duty*dt)
        assert budget!=Fraction.from_float(2.)*Fraction.from_float(power)*Fraction.from_float(duty)*Fraction.from_float(dt)
        assert Fraction(*prediction['cusaAfter']['progressExact']['0'])==Fraction(*before['progressExact']['0'])+budget


def test_reconstruction_keeps_removed_only_nonzero_history_and_refuses_one_ulp_damage():
    u=np.arange(48,dtype=float).reshape((16,3))*.001;z=u[None]*.7
    old={'cells':np.array([1,2]),'nodes':np.zeros((16,3)),'elementNodes':np.arange(16).reshape((2,8)),'freeDofs':np.ones(48,bool)}
    current={'cells':np.array([2]),'nodes':np.zeros((8,3)),'elementNodes':np.arange(8)[None,:],'freeDofs':np.ones(24,bool)}
    data={'freeIndices':np.arange(24),'beforeDisplacement':u[8:].ravel(),'beforeMemories':z[:,8:].reshape((1,24)),
        'orphanOldNodeIndices':np.arange(8),'orphanOldDisplacement':u[:8].copy(),'orphanOldMemories':z[:,:8].copy()}
    row={'event':{'removedSourceIds':[1]},'withinTick':0,'previousArrayHash':field_hash(u.ravel(),z.reshape((1,-1)))}
    result,memory=pre_event_fields(current,old,data,row)
    assert np.array_equal(result,u[old['elementNodes']]) and np.array_equal(memory,z[:,old['elementNodes']])
    damaged=copy.deepcopy(data);damaged['orphanOldMemories'][0,0,0]=np.nextafter(damaged['orphanOldMemories'][0,0,0],1.)
    with pytest.raises(ValueError,match='PREVIOUS_FIELD_HASH'):pre_event_fields(current,old,damaged,row)
    damaged=copy.deepcopy(data);damaged['orphanOldNodeIndices']=np.arange(7)
    with pytest.raises(ValueError,match='ORPHAN_SUPPORT'):pre_event_fields(current,old,damaged,row)


def test_each_conditional_event_resets_to_its_own_native_budget_and_context_refuses_wrong_phase():
    _,engine=prepared()
    with capture_pre_effects():engine.step({'slot':'right','operation':'apply','duration':.1})
    frame=engine.pre_effect_frames()[0];grid=GeometryScene(np.zeros(engine.scene.shape,bool),engine.scene.affine)
    before=copy.deepcopy(frame);cells=np.array(frame['sourceCells']);u=frame['cornerDisplacementMm']
    a=pre_effect_prediction(engine.scene,grid,frame,cells,u);a['cusaAfter']['progressExact'].clear()
    b=pre_effect_prediction(engine.scene,grid,frame,cells,u)
    require_native_prediction_exact(frame,b);assert frame.keys()==before.keys() and frame['cusaBefore']==before['cusaBefore']
    frame['geometryPhase']='after-physics'
    with pytest.raises(ValueError,match='CONTEXT_UNADMITTED'):require_pre_effect_context(frame)


@pytest.mark.parametrize('protected,configured',[([999],False),([],True)])
def test_protected_context_refuses_until_pre_effect_protection_is_modeled(protected,configured):
    _,engine=prepared()
    with capture_pre_effects():engine.step({'slot':'right','operation':'apply','duration':.1})
    frame=engine.pre_effect_frames()[0];frame['protectedSourceIds']=protected;frame['guardReport']['configured']=configured
    with pytest.raises(ValueError,match='CONTEXT_UNADMITTED'):require_pre_effect_context(frame)
