"""Full-history compliance algebra and source-owned topology changes."""
import copy

import numpy as np
import pytest
from scipy import sparse

from resectionlab import surgical_continuum as c, surgical_deformation_v2 as brain
from resectionlab.surgical_continuum_solve import ProportionalHistorySolve,PROPERTY,profile
from resectionlab.surgical_scene import SurgicalScene
from resectionlab.surgical_engine import SurgicalEngine
from resectionlab.surgical_physics_v2 import default_profile


def dense_problem(branches=(.5,.2),contacts=True,free_all=False):
    rng=np.random.default_rng(121);n=30
    a=rng.normal(size=(n,n));k=a.T@a+np.eye(n)
    matrices=tuple(sparse.csr_matrix(v*k) for v in [1.,*branches])
    free=np.ones(n,bool)
    if not free_all:free[:3]=False
    holds=[('left',0,np.arange(8),.02),('right',1,np.arange(2,10),.03)] if contacts else []
    materials={'1':{'shearPa':1000.,'relaxationShearPa':[1000*v for v in branches]}}
    ratios=np.array([.9,.7][:len(branches)])
    return rng,k,matrices,free,holds,materials,ratios


@pytest.mark.parametrize('branches',[(),(.5,),(.5,.2),(0.,.2)])
@pytest.mark.parametrize('contacts',[False,True])
def test_arbitrary_distributed_history_and_changed_gravity_match_dense_oracle(branches,contacts):
    rng,k,matrices,free,holds,materials,ratios=dense_problem(branches,contacts)
    solver=ProportionalHistorySolve(matrices,free,holds,ratios,materials)
    z=rng.normal(size=(len(branches),len(free)))*.001;z[:,~free]=0.
    # Assemble the spring operator independently from dense averaging rows.
    avg=np.zeros((3*len(holds),len(free)));spring=[]
    for i,(_,_,nodes,stiffness) in enumerate(holds):
        for axis in range(3):avg[3*i+axis,3*nodes+axis]=1/8.;spring.append(stiffness)
    spring=np.array(spring);beta=1+sum(ratios*np.asarray(branches))
    a=beta*k+avg.T@np.diag(spring)@avg
    for step in range(12):
        body=rng.normal(size=len(free))*.001 if step%3 else np.zeros(len(free))
        target=rng.normal(size=3*len(holds))*.01
        rows=[(key,nodes,stiffness,target[3*i:3*i+3]) for i,(key,_,nodes,stiffness) in enumerate(holds)]
        actual=solver.advance(z,body,rows)
        rhs=body+avg.T@(spring*target)
        for r,alpha,memory in zip(ratios,branches,z):rhs+=r*alpha*(k@memory)
        expected=np.zeros(len(free));expected[free]=np.linalg.solve(a[np.ix_(free,free)],rhs[free])
        np.testing.assert_allclose(actual,expected,rtol=1e-11,atol=1e-13)
        # Arbitrary spatial memory remains full-dimensional, not fitted to B.
        z=.9*z+.1*actual


def test_fixed_memory_nonproportional_matrix_and_no_free_dofs_are_explicit():
    rng,k,matrices,free,holds,materials,ratios=dense_problem((.5,))
    solver=ProportionalHistorySolve(matrices,free,holds,ratios,materials)
    z=np.zeros((1,len(free)));z[0,0]=.001
    with pytest.raises(ValueError,match='FIXED_HISTORY'):solver.advance(z,np.zeros(len(free)),[])
    bad=[matrices[0],matrices[1]+sparse.eye(len(free))*.001]
    with pytest.raises(ValueError,match='MATRIX_PROPORTIONALITY'):
        ProportionalHistorySolve(bad,free,holds,ratios,materials)
    clamped=ProportionalHistorySolve(matrices,np.zeros(len(free),bool),holds,ratios,materials)
    target=[(key,nodes,stiffness,np.ones(3)*.01) for key,_,nodes,stiffness in holds]
    np.testing.assert_array_equal(clamped.advance(np.zeros_like(z),np.ones(len(free)),target),np.zeros(len(free)))


def coupon(enabled=False,*,branches=(500.,),times=(.5,)):
    properties={'tissuePhysicsV2':default_profile(),c.PROPERTY:c.research_profile({'1':{
        'shearPa':1000.,'poissonRatio':.3,'relaxationShearPa':list(branches),'source':'Generated numerical coupon'}},relaxation_seconds=list(times))}
    if enabled:properties[PROPERTY]=profile('detailed' if enabled=='detailed' else 'proportional-history-compliance')
    scene=SurgicalScene(np.ones((13,13,13),np.uint16),np.eye(4),
        [{'id':1,'name':'Generated block','material':'soft_tissue'}],{'kind':'generated numerical coupon'},properties)
    return SurgicalEngine(scene)


def assert_fields_equal(a,b):
    for field in ['cornerDisplacementMm','memoryDisplacementMm','exposureBySourceLabel']:
        def check(x,y):
            if isinstance(x,dict):
                assert set(x)==set(y)
                for key in x:check(x[key],y[key])
            elif isinstance(x,list) and x and isinstance(x[0],dict):
                assert len(x)==len(y)
                for first,second in zip(x,y):check(first,second)
            else:np.testing.assert_allclose(x,y,atol=1e-10,rtol=1e-9)
        check(a.state[c.PROPERTY][field],b.state[c.PROPERTY][field])
    for field in ['dissipatedEnergyNmm','removedStoredEnergyNmm','numericalDissipationNmm','removedElementCount']:
        assert b.state[c.PROPERTY][field]==pytest.approx(a.state[c.PROPERTY][field],rel=1e-9,abs=1e-12)


def test_removal_cut_release_and_cache_rebuild_keep_distributed_source_history():
    detailed=coupon();fast=coupon(True);center=int(np.ravel_multi_index((6,6,6),detailed.scene.shape))
    for e in [detailed,fast]:
        brain.begin_hold(e,'left',[center],[6.,6.,6.]);brain.set_target(e,'left',[6.1,6.,6.]);brain.tick(e,.2)
    assert_fields_equal(detailed,fast);first_system=fast._continuum_system_v1[1]
    removed=int(np.ravel_multi_index((5,6,6),detailed.scene.shape))
    for e in [detailed,fast]:e.state['removed']=[removed];brain.tick(e,.1)
    assert_fields_equal(detailed,fast)
    assert fast._continuum_system_v1[1] is not first_system
    assert fast.state[c.PROPERTY]['removedStoredEnergyNmm']>0
    assert fast.state[c.PROPERTY]['removedElementCount']==1
    after_removal=fast.state[c.PROPERTY]['removedStoredEnergyNmm'];second_system=fast._continuum_system_v1[1]
    for e in [detailed,fast]:
        active=set(e.state['deformationModel']['active']);cuts=[]
        for cell in active:
            ijk=list(np.unravel_index(cell,e.scene.shape))
            if ijk[0]!=6:continue
            other=int(np.ravel_multi_index((7,ijk[1],ijk[2]),e.scene.shape))
            if other in active:cuts.append([cell,other])
        e.state['cutEdges']=cuts;brain.tick(e,.1)
    assert_fields_equal(detailed,fast);assert fast._continuum_system_v1[1] is not second_system
    assert fast.state[c.PROPERTY]['removedStoredEnergyNmm']==after_removal
    for e in [detailed,fast]:brain.release(e,'left');brain.tick(e,.5)
    assert_fields_equal(detailed,fast)
    assert fast._continuum_system_v1[1]['history_solve'].contacts==()


def test_opt_in_profile_is_explicit_and_unqualified_reward_stays_blocked():
    from resectionlab.surgical_rl_rewards import RewardTracker
    old=coupon();new=coupon(True)
    assert old.scene.identity!=new.scene.identity
    with pytest.raises(ValueError,match='CONTINUUM_RL_HOLD'):RewardTracker(new)
    properties=copy.deepcopy(new.scene.properties);properties.pop(c.PROPERTY)
    with pytest.raises(ValueError,match='REQUIRES_CONTINUUM'):
        SurgicalScene(new.scene.labels,new.scene.affine,new.scene.regions,new.scene.source,properties)


def test_loaded_solver_replacement_disables_verified_replay_cache(monkeypatch):
    from resectionlab.surgical_verified_cache import implementation_fingerprint
    assert implementation_fingerprint() is not None
    monkeypatch.setattr(ProportionalHistorySolve,'advance',lambda *args,**kwargs:None)
    assert implementation_fingerprint() is None


def test_inactive_source_removal_releases_support_and_warm_cache_matches_fresh():
    engines=[coupon('detailed'),coupon(True)]
    center=int(np.ravel_multi_index((6,6,6),engines[0].scene.shape))
    active=[int(np.ravel_multi_index((x,y,z),engines[0].scene.shape)) for x in range(4,9) for y in range(4,9) for z in range(4,9)]
    anchors=[cell for cell in active if any(v in (4,8) for v in np.unravel_index(cell,engines[0].scene.shape))]
    outside=[int(np.ravel_multi_index((9,y,z),engines[0].scene.shape)) for y in range(5,8) for z in range(5,8)]
    old=[]
    for e in engines:
        brain.begin_hold(e,'left',[center],[6.,6.,6.]);brain.set_target(e,'left',[6.1,6.,6.])
        e.state['deformationModel'].update(active=active[:],anchors=anchors[:]);brain.tick(e,.1)
        old.append(e._continuum_system_v1[1]);e.state['removed']=outside[:]
    forced=engines[0].clone();del forced._continuum_system_v1
    for e in [*engines,forced]:brain.tick(e,.1)
    for e,previous in zip(engines,old):
        assert e._continuum_system_v1[1] is not previous
        assert e._continuum_system_v1[1]['free'].sum()>previous['free'].sum()
    assert_fields_equal(engines[0],forced);assert_fields_equal(engines[0],engines[1])
