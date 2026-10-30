"""Stateful tool/continuum controls with an independently solved full system."""
import copy
from fractions import Fraction
from types import SimpleNamespace
import numpy as np
import pytest
from scipy import sparse
from scipy.spatial.transform import Rotation

from resectionlab import surgical_continuum as c,surgical_cusa_v3 as cusa
from resectionlab.surgical_scene import SurgicalScene
from resectionlab.surgical_physics_v2 import default_profile
from resectionlab.geometry import GeometryScene
from resectionlab.surgical_continuum_topology import SourceDomain
from resectionlab.surgical_continuum_exterior import source_stiffness
from resectionlab.surgical_continuum_events import source_operator_identity
from resectionlab.surgical_continuum_program import contact_matrix,spring,targets
from resectionlab.surgical_source_contacts import SourceContactGeometry
from resectionlab.surgical_closed_loop import CusaContinuumEpisode,CoupledPreparation,ClosedLoopRefusal,hold_contacts

LIMITS={'forceN':1e-10,'energyNmm':1e-10,'balanceNmm':1e-8,
    'maxGradientNorm':.2,'maxCenterDisplacementMm':3.,'maxReactionN':.08}


def metrics(g,free,u,z):
    full=np.zeros(g['nodes'].size);full[free]=u;local=full.reshape((-1,3))[g['elementNodes']]
    gradient=np.einsum('eai,gaj->egij',local,c.element_operators(g['affine'])[2],optimize=True)
    return np.zeros(2),float(np.linalg.norm(gradient,axis=(2,3)).max()),float(np.linalg.norm(local.mean(axis=1),axis=1).max())


class DirectBackend:
    def __init__(self,domain,alpha,times):self.domain=domain;self.alpha=alpha;self.times=times;self.current=None;self.builds=[]
    def checkpoint(self):return self.current
    def rollback(self,value):self.current=value
    def prepare(self,stage,contacts,removed):
        g=stage['geometry'];free=np.flatnonzero(g['freeDofs']);k=source_stiffness(g,source_order=self.domain.cells)[free][:,free].toarray()
        contact=contact_matrix(g,contacts)[:,free].toarray();d=spring(contacts);alpha=self.alpha;times=self.times
        class Solve:
            operator=SimpleNamespace(k=sparse.csr_matrix(k))
            def contact_energy(self,u,target):
                e=contact@u-target;return float(.5*e@(d*e))
            def step(self,u,z,body,target,h,*,previous_target):
                r=times/(times+h);matrix=(1+alpha@r)*k+contact.T@np.diag(d)@contact
                rhs=body+contact.T@(d*target)+sum((a*rj*(k@old) for a,rj,old in zip(alpha,r,z)),np.zeros(len(u)))
                new=np.linalg.solve(matrix,rhs);memory=(times[:,None]*z+h*new)/(times[:,None]+h)
                def energy(x,mem,t):
                    e=contact@x-t;return .5*x@k@x+.5*e@(d*e)+sum(.5*a*(x-v)@k@(x-v) for a,v in zip(alpha,mem))
                delta=new-u;before_contact=contact@u-previous_target;after_contact=contact@new-target
                num=.5*delta@k@delta+.5*(after_contact-before_contact)@(d*(after_contact-before_contact));loss=0.
                for a,tau,v,old in zip(alpha,times,memory,z):
                    dz=v-old;dv=(new-v)-(u-old);num+=.5*a*dv@k@dv;loss+=a*tau/h*dz@k@dz
                work=float(body@delta+(-d*after_contact)@(target-previous_target));old=energy(u,z,previous_target);total=energy(new,memory,target)
                residual=(1+sum(alpha))*(k@new)-sum((a*(k@v) for a,v in zip(alpha,memory)),np.zeros(len(new)))+contact.T@(d*after_contact)-body
                return {'displacement':new,'memories':memory,'maxFreeResidualN':float(np.abs(residual).max()),
                    'energyNmm':float(total),'oldEnergyNmm':float(old),'contactEnergyNmm':self.contact_energy(new,target),
                    'bodyWorkNmm':float(body@delta),'targetWorkNmm':work-float(body@delta),'workNmm':work,
                    'numericalDissipationNmm':float(num),'dissipatedEnergyNmm':float(loss),
                    'energyBalanceResidualNmm':float(work-total+old-loss-num),'contactReactionN':d*(target-contact@new)}
        self.current=Solve();return self.current,source_operator_identity(g,alpha,times,core_free_dofs=stage['coreFreeDofs'],removed_source_ids=removed)


def fixture(*,two_tip_candidates=False):
    labels=np.zeros((10,10,10),np.uint16);labels[3:6,3:6,3:6]=1
    if two_tip_candidates:labels[4,3,3]=labels[4,4,3]=0
    props={c.PROPERTY:c.research_profile({'1':{'shearPa':1000.,'poissonRatio':.3,'relaxationShearPa':[500.],
        'source':'Generated closed-loop control'}},relaxation_seconds=[.5]),'cusaV3':cusa.default_profile(),
        'tissuePhysicsV2':default_profile()}
    scene=SurgicalScene(labels,np.eye(4),[{'id':1,'material':'soft_tissue'}],{'kind':'generated closed-loop block'},props)
    ids=np.flatnonzero(labels);mesh=c.voxel_mesh(scene,ids);free=np.repeat(~np.isclose(mesh.nodes[:,0],2.5),3)
    g={'cells':np.asarray(mesh.cells),'shape':np.array(scene.shape),'nodes':mesh.nodes,'elementNodes':mesh.element_nodes,
        'affine':scene.affine,'freeDofs':free,'muPa':np.full(len(ids),1000.),'nu':np.full(len(ids),.3)}
    e=np.flatnonzero(np.all(np.isclose(mesh.nodes,[3.5,2.5,5.5]),axis=1))[0]
    indexes=np.flatnonzero(free);core=np.flatnonzero(~np.isin(indexes,np.arange(3*e,3*e+3)))
    domain=SourceDomain(g,core);g=domain.after()['geometry'];free=np.flatnonzero(g['freeDofs'])
    held=[int(np.ravel_multi_index((5,5,z),scene.shape)) for z in (3,4,5)]
    holds={'left':{'cells':held,'origin':[5.75,5.,5.],'target':[5.77,5.,5.],'baseDisplacements':{},'offset':[0,0,0]}}
    contacts=hold_contacts(scene,holds);a=np.array([.5]);times=np.array([.5]);h=1/120
    k=source_stiffness(g)[free][:,free].toarray();cc=contact_matrix(g,contacts)[:,free].toarray();d=spring(contacts);target=targets(contacts)
    u=np.zeros(len(free));z=np.zeros((1,len(free)));r=times/(times+h);matrix=(1+a@r)*k+cc.T@np.diag(d)@cc
    for _ in range(12):
        u=np.linalg.solve(matrix,cc.T@(d*target)+a[0]*r[0]*(k@z[0]));z=(times[:,None]*z+h*u)/(times[:,None]+h)
    full=np.zeros(g['nodes'].size);full[free]=u;memory=np.zeros((1,len(full)));memory[:,free]=z
    assert np.max(np.abs(memory[:,np.setdiff1d(free,free[core])]))>0
    slots={'left':{'instrument':'brain_retractor','position':[5.77,5.,5.],
        'quaternion':Rotation.from_euler('y',-90,degrees=True).as_quat().tolist(),'active':False},
        'right':{'instrument':'cusa','position':[3.,3.,1.25],'quaternion':[0.,0.,0.,1.],'active':False}}
    params={'left':{'radiusMm':3.,'depthMm':2.,'contactGeometryRevision':2},
        'right':{'radiusMm':1.5,'depthMm':2.,'power':1.,'duty':1.,'contactGeometryRevision':2}}
    grid=GeometryScene(np.zeros(scene.shape,bool),scene.affine);geo=SourceContactGeometry(scene,grid,[],g['cells'],full.reshape((-1,3))[g['elementNodes']],holds=holds,detached_fragments=[])
    view=SimpleNamespace(scene=scene,by_label=geo.by_label,state={'toolMechanicsRevision':2,'holds':holds,'removed':[],
        'deformation':{str(int(cell)):v.tolist() for cell,v in zip(geo.field.cells,geo.u.mean(axis=1))}},
        touched=lambda p,q,radius:geo.occupancy.material_ids(geo.grid_cover(p,q,radius)))
    cusa.initialize(view)
    for _ in range(4):cusa.apply(view,view.touched(slots['right']['position'],slots['right']['position'],1.5),.1,params['right'],position=slots['right']['position'])
    assert not view.state['removed']
    initial={'displacement':full,'memories':memory,'removedSourceIds':[],'holds':holds,'slots':slots,
        'parametersBySlot':params,'cusaBefore':view.state['cusaV3'],'elapsedSeconds':.1,'remainderSeconds':0.,'toolMechanicsRevision':2}
    return scene,grid,domain,initial,a,times,h


def commands(initial):
    apply={'slot':'right','operation':'apply','effectiveActive':True,'durationSeconds':.1,
        'pose':{'position':[3.,3.,2.4],'quaternion':[0.,0.,0.,1.]}}
    return [copy.deepcopy(apply) for _ in range(6)]+[
        {'slot':'left','operation':'release','effectiveActive':False,'durationSeconds':.1,'pose':'current'},
        *[{'slot':'right','operation':'inspect','effectiveActive':False,'durationSeconds':.1,'pose':'current'} for _ in range(2)]]


def test_dynamic_source_decisions_fields_energy_release_and_replay_match_direct_full_solve():
    scene,grid,domain,initial,a,t,h=fixture();left=CusaContinuumEpisode(scene,grid,domain,initial,a,t,h,metrics,LIMITS)
    direct=CusaContinuumEpisode(scene,grid,domain,initial,a,t,h,metrics,LIMITS,backend=DirectBackend(domain,a,t))
    replay=CusaContinuumEpisode(scene,grid,domain,initial,a,t,h,metrics,LIMITS);removed=[];hashes=[];events=[]
    for command in commands(initial):
        expected=direct.advance(command);actual=left.advance(command);again=replay.advance(command);events.append(actual)
        assert actual==again and left.state_hash()==replay.state_hash()
        assert actual['orderedSourceIds']==expected['orderedSourceIds'] and actual['cusaEvents']==expected['cusaEvents']
        assert left.state['removed']==direct.state['removed'] and left.state['cusa']==direct.state['cusa']
        np.testing.assert_allclose(left.state['u'],direct.state['u'],atol=1e-12,rtol=1e-10)
        np.testing.assert_allclose(left.state['z'],direct.state['z'],atol=1e-12,rtol=1e-10)
        if actual['sourceEvent']:removed.extend(actual['sourceEvent']['removedSourceIds'])
        hashes.append(left.state_hash())
    assert len(removed)>=2 and len(set(removed))==len(removed)
    assert left.state['fixedSteps']==108 and left.state['remainderSeconds']==0.
    assert left.state['initialEnergyNmm']>0 and left.state['removedEnergyNmm']>0 and left.state['contactJumpNmm']<0
    assert np.max(np.abs(left.state['z']))>0 and not left.state['holds']
    assert events[-1]['cusaEvents']==[] and events[6]['cusaEvents']==[]
    assert any(row['retainedExterior'] for row in left.backend.builds)


@pytest.mark.parametrize('kind',['late-shaft','second-BE','cancel'])
def test_rejected_tick_rolls_back_every_physical_and_numeric_binding_then_replays_exactly(monkeypatch,kind):
    scene,grid,domain,initial,a,t,h=fixture();episode=CusaContinuumEpisode(scene,grid,domain,initial,a,t,h,metrics,LIMITS)
    clean=CusaContinuumEpisode(scene,grid,domain,initial,a,t,h,metrics,LIMITS);before=episode.state_hash();old_factor=episode.backend.prepared
    original=episode._covers
    original_guard=episode._field_guard;calls=0
    def covers(*args,**kwargs):
        if not kwargs['swept']:raise ClosedLoopRefusal('UNIT_LATE_SHAFT')
        return original(*args,**kwargs)
    def fail(*args):raise InterruptedError('Unit failure')
    def guard(*args):
        nonlocal calls
        calls+=1
        if calls==2:raise ClosedLoopRefusal('UNIT_SECOND_BE_FAILURE')
        return original_guard(*args)
    command=commands(initial)[0]
    if kind=='late-shaft':monkeypatch.setattr(episode,'_covers',covers)
    if kind=='second-BE':monkeypatch.setattr(episode,'_field_guard',guard)
    if kind=='cancel':episode.check=fail
    with pytest.raises((InterruptedError,ClosedLoopRefusal)):
        episode.advance(command)
    assert episode.state_hash()==before and episode.backend.prepared is old_factor
    assert episode.last_committed_batch() is None
    monkeypatch.setattr(episode,'_covers',original);monkeypatch.setattr(episode,'_field_guard',original_guard);episode.check=lambda:None
    assert episode.advance(command)==clean.advance(command) and episode.state_hash()==clean.state_hash()


def test_fractional_clock_and_invalid_initial_physics_are_not_silently_reset():
    scene,grid,domain,initial,a,t,h=fixture();initial['remainderSeconds']=.003
    episode=CusaContinuumEpisode(scene,grid,domain,initial,a,t,h,metrics,LIMITS)
    step=episode.advance(commands(initial)[0]);assert step['fixedSteps']==12 and step['remainderSeconds']==.003
    broken=copy.deepcopy(initial);broken['memories'][0,0]=.1
    with pytest.raises(ValueError,match='CONSTRAINED_HISTORY'):
        CusaContinuumEpisode(scene,grid,domain,broken,a,t,h,metrics,LIMITS)


def test_only_committed_batches_publish_and_returned_payloads_cannot_mutate_state():
    scene,grid,domain,initial,a,t,h=fixture();episode=CusaContinuumEpisode(scene,grid,domain,initial,a,t,h,metrics,LIMITS)
    result=episode.advance(commands(initial)[0]);before=episode.state_hash();batch=episode.last_committed_batch()
    assert len(batch['records'])==12 and batch['afterStateHash']==before
    batch['stage']['geometry']['cells'][0]=-1;batch['records'][0][1]['displacement'][:]=100.
    result['sourceEvent']['removedSourceIds'].clear()
    assert episode.state_hash()==before and episode.last_committed_batch()['stage']['geometry']['cells'][0]>=0
    assert episode.state['sourceLedger'].events[0]['removedSourceIds']


def test_every_be_uses_original_four_fraction_obstacle_guard_and_rolls_back(monkeypatch):
    from resectionlab import surgical_deformation_v2 as brain
    scene,grid,domain,initial,a,t,h=fixture();episode=CusaContinuumEpisode(scene,grid,domain,initial,a,t,h,metrics,LIMITS)
    before=episode.state_hash();original=brain._obstacle_collision;calls=0
    def obstacle(*args):
        nonlocal calls
        calls+=1
        return True if calls==3 else original(*args)
    monkeypatch.setattr(brain,'_obstacle_collision',obstacle)
    with pytest.raises(ClosedLoopRefusal,match='OBSTACLE_COLLISION'):episode.advance(commands(initial)[0])
    assert calls==3 and episode.state_hash()==before and episode.last_committed_batch() is None


@pytest.mark.parametrize('rates',[np.array(-1.),np.array([-1.,0.]),np.array([np.nan,0.]),np.zeros(3)])
def test_invalid_dose_observations_cannot_change_accepted_state(rates):
    scene,grid,domain,initial,a,t,h=fixture();episode=CusaContinuumEpisode(scene,grid,domain,initial,a,t,h,metrics,LIMITS)
    before=episode.state_hash();a[:]=0.;t[:]=99.
    assert episode.alpha[0]==.5 and episode.times[0]==.5
    episode.measure=lambda *args:(rates,.01,.01)
    with pytest.raises(ClosedLoopRefusal,match='APPROXIMATION_DOMAIN'):episode.advance(commands(initial)[0])
    assert episode.state_hash()==before


def test_fresh_exterior_update_is_an_explicit_bounded_refusal_not_stale_reuse():
    scene,grid,domain,initial,a,t,h=fixture();episode=CusaContinuumEpisode(scene,grid,domain,initial,a,t,h,metrics,LIMITS)
    factor=episode.backend.prepared;cell=int(np.ravel_multi_index((3,3,5),scene.shape));stage=domain.after([cell])
    with pytest.raises(ClosedLoopRefusal,match='FRESH_EXTERIOR_UPDATE_REQUIRED'):
        episode.backend.prepare(stage,episode.state['contacts'],[cell])
    assert episode.backend.prepared is factor and episode.backend.builds[-1]['status']=='refused-fresh-update'


def test_multiple_whole_owner_removals_are_transferred_atomically_from_actual_budget():
    scene,grid,domain,initial,a,t,h=fixture();g=domain.after()['geometry']
    geometry=SourceContactGeometry(scene,grid,[],g['cells'],initial['displacement'].reshape((-1,3))[g['elementNodes']],
        holds=initial['holds'],detached_fragments=[])
    owners=geometry.cusa_order(initial['slots']['right']['position'])[:2];value=Fraction.from_float(.9)
    ledger=initial['cusaBefore'];ledger['progressMm3']={str(i):.9 for i in owners}
    ledger['progressExact']={str(i):[value.numerator,value.denominator] for i in owners}
    ledger['contactSeconds']={str(i):.45 for i in owners};ledger['inputCapacityMm3']=1.8
    episode=CusaContinuumEpisode(scene,grid,domain,initial,a,t,h,metrics,LIMITS);result=episode.advance(commands(initial)[0])
    assert result['sourceEvent']['removedElementCount']==2 and set(result['sourceEvent']['removedSourceIds'])==set(owners)
    assert set(episode.state['removed'])==set(owners) and episode.state['sourceLedger'].removed_source_ids==set(owners)
    assert result['sourceEvent']['removedStoredEnergyNmm']>0 and np.max(np.abs(episode.state['z']))>0


def test_removal_outside_declared_mechanics_refuses_before_losing_energy_or_history():
    scene,_,domain,initial,a,t,h=fixture();labels=np.array(scene.labels);labels[3,3,2]=1
    scene=SurgicalScene(labels,scene.affine,scene.regions,{'kind':'generated outside-domain control'},scene.properties)
    grid=GeometryScene(np.zeros(scene.shape,bool),scene.affine);initial['slots']['right']['position']=[3.,3.,.75]
    g=domain.after()['geometry'];geometry=SourceContactGeometry(scene,grid,[],g['cells'],
        initial['displacement'].reshape((-1,3))[g['elementNodes']],holds=initial['holds'],detached_fragments=[])
    view=SimpleNamespace(scene=scene,by_label=geometry.by_label,state={'removed':[],'holds':initial['holds'],'toolMechanicsRevision':2},
        touched=lambda p,q,r:geometry.occupancy.material_ids(geometry.grid_cover(p,q,r)))
    cusa.initialize(view)
    for _ in range(4):cusa.apply(view,view.touched([3.,3.,.75],[3.,3.,.75],1.5),.1,initial['parametersBySlot']['right'],position=[3.,3.,.75])
    initial['cusaBefore']=view.state['cusaV3'];episode=CusaContinuumEpisode(scene,grid,domain,initial,a,t,h,metrics,LIMITS)
    before=episode.state_hash()
    with pytest.raises(ClosedLoopRefusal,match='REMOVAL_OUTSIDE_DOMAIN'):episode.advance(commands(initial)[0])
    assert episode.state_hash()==before and not episode.state['sourceLedger'].events


def test_held_owner_removal_refuses_without_silently_redistributing_contacts():
    scene,grid,domain,initial,a,t,h=fixture();first=int(np.ravel_multi_index((3,3,3),scene.shape))
    initial['displacement'][:]=0.;initial['memories'][:]=0.;initial['holds']['left']['target']=initial['holds']['left']['origin'][:]
    initial['holds']['left']['cells'].append(first)
    episode=CusaContinuumEpisode(scene,grid,domain,initial,a,t,h,metrics,LIMITS);before=episode.state_hash()
    with pytest.raises(ClosedLoopRefusal,match='HELD_OWNER_REMOVAL'):episode.advance(commands(initial)[0])
    assert episode.state_hash()==before


def test_smaller_algebraic_core_with_counted_fresh_factors_preserves_full_fields_and_rollback(monkeypatch):
    scene,grid,original,initial,a,t,h=fixture();g=original.geometry;free=np.flatnonzero(g['freeDofs'])
    core=np.unique(contact_matrix(g,hold_contacts(scene,initial['holds']))[:,free].indices)
    domain=SourceDomain(g,core);backend=CoupledPreparation(domain,a,t,allow_fresh_exterior_updates=True)
    episode=CusaContinuumEpisode(scene,grid,domain,initial,a,t,h,metrics,LIMITS,backend=backend)
    reference=CusaContinuumEpisode(scene,grid,original,initial,a,t,h,metrics,LIMITS)
    before=episode.state_hash();prepared=backend.prepared;covers=episode._covers
    def fail(*args,**kwargs):
        if not kwargs['swept']:raise ClosedLoopRefusal('UNIT_FRESH_FACTOR_ROLLBACK')
        return covers(*args,**kwargs)
    monkeypatch.setattr(episode,'_covers',fail)
    with pytest.raises(ClosedLoopRefusal,match='FRESH_FACTOR_ROLLBACK'):episode.advance(commands(initial)[0])
    assert episode.state_hash()==before and backend.prepared is prepared
    monkeypatch.setattr(episode,'_covers',covers)
    for command in commands(initial):
        result=episode.advance(command);expected=reference.advance(command)
        assert result['orderedSourceIds']==expected['orderedSourceIds'] and result['cusaEvents']==expected['cusaEvents']
        np.testing.assert_allclose(episode.state['u'],reference.state['u'],atol=1e-12,rtol=1e-10)
        np.testing.assert_allclose(episode.state['z'],reference.state['z'],atol=1e-12,rtol=1e-10)
    assert len(core)<len(original.original_core)
    assert all(row['totalDofs']>=episode.state['stage']['geometry']['nodes'].size for row in backend.builds)
    assert sum(not row['retainedExterior'] for row in backend.builds)>=3
