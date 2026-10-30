"""Offline state-driven CUSA/continuum microticks with full source history.

This restricted research episode is not a SurgicalEngine or live-cap override.
Initial fields/ledgers are explicit conditions. Future removals are computed
from current geometry and accumulated work, never prescribed native masks.
"""
import copy
import gc
import hashlib
import json
import time
from types import SimpleNamespace

import numpy as np

from . import surgical_cusa_v3 as cusa
from . import surgical_deformation_v2 as brain
from .core import immutable_array
from .surgical_continuum import PROPERTY
from .surgical_continuum_events import SourceHistoryLedger,stored_energy_by_source,source_operator_identity
from .surgical_continuum_exterior import ExteriorOperator,ExteriorCoreFactor,ExteriorHistorySolve,source_stiffness
from .surgical_continuum_program import contact_matrix,contact_key,targets,spring,field_hash
from .surgical_motion import envelopes,swept_envelopes,pair_conflict
from .surgical_pre_effect_witness import limiter_prediction,capacity_prediction
from .surgical_source_contacts import SourceContactGeometry
from .surgical_tools import TOOLS


class ClosedLoopRefusal(ValueError):pass


class CoupledPreparation:
    """Checkpointed preparation with a bounded refusal for fresh-E updates."""
    def __init__(self,domain,alpha,times,*,allow_fresh_exterior_updates=False,check=lambda:None):
        if type(allow_fresh_exterior_updates) is not bool:raise ValueError('CLOSED_LOOP_FRESH_UPDATE_DECLARATION')
        self.domain=domain;self.alpha=immutable_array(np.asarray(alpha,float));self.times=immutable_array(np.asarray(times,float));self.check=check
        self.allow_fresh_exterior_updates=allow_fresh_exterior_updates
        self.prepared=None;self.solver=None;self.identity=None;self.lineage=None;self.binding=None;self.builds=[]

    def invalidate(self):
        self.solver=None;self.prepared=None;self.identity=None;self.lineage=None;self.binding=None

    def checkpoint(self):return self.prepared,self.solver,self.identity,self.lineage,self.binding

    def rollback(self,value):self.prepared,self.solver,self.identity,self.lineage,self.binding=value

    def prepare(self,stage,contacts,removed):
        g=stage['geometry'];free=np.flatnonzero(g['freeDofs']);core=stage['coreFreeDofs']
        identity=source_operator_identity(g,self.alpha,self.times,core_free_dofs=core,removed_source_ids=removed)
        if identity!=self.identity:
            begin=time.perf_counter();self.check();full=source_stiffness(g,source_order=self.domain.cells,check=self.check)
            k=full[free][:,free].tocsr();del full
            next_prepared=None;refusal=None;lineage=stage['originalFreeDofForNew']
            if self.prepared is not None:
                if len(np.unique(self.lineage))==len(self.lineage) and len(np.unique(lineage))==len(lineage) and set(lineage)<=set(self.lineage):
                    inverse={int(v):i for i,v in enumerate(self.lineage)};parents=np.asarray([inverse[int(v)] for v in lineage])
                    try:next_prepared=self.prepared.rebuild(k,core,parents,check=self.check)
                    except ValueError as error:
                        refusal=str(error)
                        if refusal not in ('EXTERIOR_RETAINED_EXTERIOR_COEFFICIENTS','EXTERIOR_RETAINED_COUPLING_COEFFICIENTS',
                                'EXTERIOR_RETAINED_PARTITION_LINEAGE','EXTERIOR_RETAINED_FREE_LINEAGE'):raise
                else:refusal='NONBIJECTIVE_OR_NEW_SOURCE_LINEAGE'
            retained=next_prepared is not None
            if not retained:
                if self.prepared is not None and not self.allow_fresh_exterior_updates:
                    # Retaining an old factor for atomic rollback while also
                    # allocating a fresh E factor needs separate peak admission.
                    self.builds.append({'status':'refused-fresh-update','reason':refusal,
                        'totalDofs':g['nodes'].size,'freeDofs':len(free),'wallSeconds':time.perf_counter()-begin})
                    raise ClosedLoopRefusal('CLOSED_LOOP_FRESH_EXTERIOR_UPDATE_REQUIRED:'+str(refusal))
                # An explicit profile/pilot declaration charges coexistence of
                # the old rollback checkpoint and the new fresh E factor.
                gc.collect();self.check()
                next_prepared=ExteriorCoreFactor(ExteriorOperator(k,core,check=self.check),check=self.check)
            self.solver=None;self.prepared=next_prepared;self.identity=identity;self.lineage=lineage.copy();self.binding=None
            probe=np.sin(np.arange(len(core))*.013+.3)*1e-3
            error=float(np.max(np.abs(next_prepared.schur@probe-next_prepared.operator.apply(probe)),initial=0.))
            if error>1e-10:raise ClosedLoopRefusal('CLOSED_LOOP_CURRENT_SCHUR_PROBE')
            self.builds.append({'sourceOperatorIdentity':identity,'sourceCells':len(g['cells']),'totalDofs':g['nodes'].size,
                'freeDofs':len(free),'retainedExterior':retained,'freshExteriorReason':refusal,'wallSeconds':time.perf_counter()-begin,
                'schurProbeMaximumN':error,'operator':dict(next_prepared.operator.stats)})
        binding=contact_key(contacts)
        if binding!=self.binding:
            self.solver=ExteriorHistorySolve(self.prepared.operator,contact_matrix(g,contacts)[:,free],spring(contacts),
                self.alpha,self.times,prepared=self.prepared,check=self.check);self.binding=binding
        return self.solver,self.identity


def hold_contacts(scene,holds):
    stiffness=scene.properties[PROPERTY]['handleStiffnessNPerMm'];result=[]
    for slot,hold in sorted(holds.items()):
        if 'target' not in hold or not hold['cells']:raise ClosedLoopRefusal('CLOSED_LOOP_UNSUPPORTED_HOLD')
        cells=sorted(set(hold['cells']))
        if len(cells)!=len(hold['cells']) or any(not isinstance(cell,(int,np.integer)) or isinstance(cell,(bool,np.bool_)) for cell in cells):
            raise ClosedLoopRefusal('CLOSED_LOOP_HOLD_OWNER_DECLARATION')
        for cell in cells:
            target=np.subtract(hold['target'],hold['origin'])+np.asarray(hold.get('baseDisplacements',{}).get(str(cell),[0.,0.,0.]))
            if target.shape!=(3,) or not np.isfinite(target).all():raise ClosedLoopRefusal('CLOSED_LOOP_HOLD_TARGET')
            result.append({'slot':slot,'sourceCell':int(cell),'stiffnessNPerMm':stiffness/len(cells),'targetDisplacementMm':target.tolist()})
    return result


class CusaContinuumEpisode:
    """Atomic microticks; no prescribed future source-event interface exists."""
    def __init__(self,scene,grid,domain,initial,alpha,times,h,measure,limits,*,backend=None,check=lambda:None):
        scene.require_cache_source();self.scene=scene;self.grid=grid;self.domain=domain;self.alpha=immutable_array(np.asarray(alpha,float))
        self.times=immutable_array(np.asarray(times,float));self.h=float(h);self.measure=measure;self.limits=copy.deepcopy(limits);self.check=check
        self._committed_batch=None
        if (not np.isfinite(h) or h<=0 or self.alpha.ndim!=1 or len(self.alpha)>2 or self.times.shape!=self.alpha.shape
                or not np.isfinite(self.alpha).all() or np.any(self.alpha<0) or not np.isfinite(self.times).all() or np.any(self.times<=0)):
            raise ValueError('CLOSED_LOOP_PHYSICAL_STEP_OR_SPECTRUM')
        if (initial.get('protectedSourceIds') or initial.get('guardReport',{}).get('configured') or initial.get('cutFaces')
                or initial.get('detachedFragments') or initial.get('toolMechanicsRevision')!=2
                or 'patientPosition' in initial or initial.get('bodyForce') not in (None,'zero')
                or any(v['active'] for v in initial['slots'].values())):raise ClosedLoopRefusal('CLOSED_LOOP_INITIAL_CONTEXT')
        if not 0<=initial['remainderSeconds']<h or not np.isfinite(initial['elapsedSeconds']) or initial['elapsedSeconds']<0:
            raise ValueError('CLOSED_LOOP_INITIAL_CLOCK')
        removed=sorted(set(initial['removedSourceIds']));within=sorted(set(removed)&set(map(int,domain.cells)))
        stage=domain.after(within);g=stage['geometry'];u=np.array(initial['displacement'],float,copy=True);z=np.array(initial['memories'],float,copy=True)
        if u.shape!=(g['nodes'].size,) or z.shape!=(len(self.alpha),len(u)):raise ValueError('CLOSED_LOOP_INITIAL_FIELDS')
        tissue=float(stored_energy_by_source(g,u,z,self.alpha).sum());holds=copy.deepcopy(initial['holds']);contacts=hold_contacts(scene,holds)
        if any(c['sourceCell'] not in set(map(int,g['cells'])) for c in contacts):raise ClosedLoopRefusal('CLOSED_LOOP_CONTACT_OUTSIDE_DOMAIN')
        if any(np.linalg.norm(c['targetDisplacementMm'])>limits['maxCenterDisplacementMm'] for c in contacts):raise ClosedLoopRefusal('CLOSED_LOOP_TARGET_DOMAIN')
        target=targets(contacts);c=contact_matrix(g,contacts);stretch=c@u-target;energy=tissue+float(.5*stretch@(spring(contacts)*stretch))
        self.backend=CoupledPreparation(domain,self.alpha,self.times,check=check) if backend is None else backend
        self.state={'stage':stage,'u':u,'z':z,'holds':holds,'contacts':contacts,'target':target,'removed':removed,
            'slots':copy.deepcopy(initial['slots']),'parametersBySlot':copy.deepcopy(initial['parametersBySlot']),
            'cusa':copy.deepcopy(initial['cusaBefore']),'elapsedSeconds':float(initial['elapsedSeconds']),
            'remainderSeconds':float(initial['remainderSeconds']),'energyNmm':energy,'initialEnergyNmm':energy,
            'removedEnergyNmm':0.,'contactJumpNmm':0.,'workNmm':0.,'lossNmm':0.,'exposure':np.zeros(2),
            'fixedSteps':0,'ticks':0,'sourceLedger':SourceHistoryLedger(),'sourceOperatorIdentity':None}
        self._field_guard(self.state,None,{'contactReactionN':spring(contacts)*(target-c@u)})
        self._increment_guard(self.state,u.reshape((-1,3))[g['elementNodes']].mean(axis=1))
        key=next((key for key,value in initial['slots'].items() if value['instrument']=='cusa'),None)
        if key is None:raise ClosedLoopRefusal('CLOSED_LOOP_CUSA_SLOT_REQUIRED')
        self._covers(self.state,key,self.state['slots'][key]['position'],self.state['slots'][key]['quaternion'],swept=False)
        checked=capacity_prediction(self.state['cusa'],removed,[],float(scene.voxel_mm3),
            cusa.profile(SimpleNamespace(scene=scene,state={'toolMechanicsRevision':2})),initial['parametersBySlot'][key],0.,False)
        if checked['cusaAfter']!=self.state['cusa'] or checked['events']:raise ClosedLoopRefusal('CLOSED_LOOP_STALE_INITIAL_CAPACITY')
        solver,identity=self.backend.prepare(stage,contacts,removed)
        free=np.flatnonzero(g['freeDofs']);internal=(1+sum(self.alpha))*u[free]-np.sum(self.alpha[:,None]*z[:,free],axis=0)
        residual=solver.operator.k@internal+c[:,free].T@(spring(contacts)*stretch)
        if np.max(np.abs(residual),initial=0.)>limits['forceN']:raise ClosedLoopRefusal('CLOSED_LOOP_INITIAL_EQUILIBRIUM')
        self.state['sourceOperatorIdentity']=identity

    def snapshot(self):return copy.deepcopy(self.state)

    def last_committed_batch(self):return copy.deepcopy(self._committed_batch)

    def state_hash(self):return self._state_hash(self.state)

    def _state_hash(self,s):
        digest=hashlib.sha256(field_hash(s['u'],s['z']).encode())
        keys=('holds','contacts','removed','slots','parametersBySlot','cusa','elapsedSeconds','remainderSeconds',
            'energyNmm','initialEnergyNmm','removedEnergyNmm','contactJumpNmm','workNmm','lossNmm','fixedSteps','ticks','sourceOperatorIdentity')
        digest.update(json.dumps({key:s[key] for key in keys},sort_keys=True,allow_nan=False).encode())
        ledger=s['sourceLedger']
        digest.update(json.dumps({'removedSourceIds':sorted(ledger.removed_source_ids),'removedEnergyNmm':ledger.removed_stored_energy_nmm,
            'events':ledger.events},sort_keys=True,allow_nan=False).encode())
        digest.update(s['target'].tobytes());digest.update(s['exposure'].tobytes());return digest.hexdigest()

    def _geometry(self,s):
        g=s['stage']['geometry'];return SourceContactGeometry(self.scene,self.grid,s['removed'],g['cells'],
            s['u'].reshape((-1,3))[g['elementNodes']],holds=s['holds'],detached_fragments=[],check=self.check)

    def _covers(self,s,key,p,q,*,swept):
        geometry=self._geometry(s);moving=swept_envelopes(TOOLS[s['slots'][key]['instrument']],s['parametersBySlot'][key],
            s['slots'][key],{'position':p,'quaternion':q}) if swept else None;rows=[];pieces={}
        for slot,value in s['slots'].items():
            tool=TOOLS[value['instrument']];own=set(s['holds'].get(slot,{}).get('cells',[]))
            pieces[slot]=moving if slot==key and swept else envelopes(tool,s['parametersBySlot'][slot],value['position'],value['quaternion'])
            for a,b,radius,part in pieces[slot]:
                hit=geometry.capsule(a,b,radius);possible=set(hit['witnessedSourceIds'])|set(hit['indeterminateSourceIds'])
                if part=='shaft' and (set(hit['rasterSourceIds'])|possible)-own:raise ClosedLoopRefusal('CLOSED_LOOP_SHAFT_COVER_OVERLAP:'+slot)
                if part!='shaft' and any(geometry.by_label[int(geometry.labels[c])]['material'] not in tool['materials'] for c in (possible|set(hit['rasterSourceIds']))-own):
                    raise ClosedLoopRefusal('CLOSED_LOOP_UNSUPPORTED_WORKING_MATERIAL')
                rows.append({'slot':slot,'part':part,'rasterSourceIds':hit['rasterSourceIds'],
                    'coverPointSourceIds':hit['witnessedSourceIds'],'indeterminateCoverSourceIds':hit['indeterminateSourceIds'],
                    'actualSweptCollisionEstablished':False})
        for other in pieces:
            if other!=key and pair_conflict(pieces[key],pieces[other]):raise ClosedLoopRefusal('CLOSED_LOOP_TOOL_COVER_CONFLICT:'+other)
        return rows

    def _increment_guard(self,s,old_centers):
        """Preserve the native four-fraction source-center obstacle/FOV guard."""
        g=s['stage']['geometry'];centers=s['u'].reshape((-1,3))[g['elementNodes']].mean(axis=1)
        rest=self.scene.points(g['cells']);removed=np.asarray(s['removed'],dtype=np.int64)
        labels=np.array([r['id'] for r in self.scene.regions if r['material'] in ('bone','vessel','unknown')],dtype=self.scene.labels.dtype)
        for fraction in (.25,.5,.75,1.):
            world=rest+old_centers+fraction*(centers-old_centers)
            voxel=np.rint(world@self.scene.inverse[:3,:3].T+self.scene.inverse[:3,3]).astype(int)
            inside=np.all((voxel>=0)&(voxel<np.asarray(self.scene.shape)),axis=1)
            if brain._obstacle_collision(SimpleNamespace(scene=self.scene),voxel,inside,removed,labels):
                raise ClosedLoopRefusal('CLOSED_LOOP_CONTINUUM_OBSTACLE_COLLISION')

    def _field_guard(self,s,solver,result):
        g=s['stage']['geometry'];free=np.flatnonzero(g['freeDofs']);rates,gradient,center=self.measure(g,free,s['u'][free],s['z'][:,free])
        rates=np.asarray(rates)
        if (rates.shape!=(2,) or not np.isfinite(rates).all() or np.any(rates<0) or not np.isfinite([gradient,center]).all()
                or gradient<0 or center<0 or gradient>self.limits['maxGradientNorm'] or center>self.limits['maxCenterDisplacementMm']):
            raise ClosedLoopRefusal('CLOSED_LOOP_APPROXIMATION_DOMAIN')
        if result is not None:
            force={}
            for contact,f in zip(s['contacts'],result['contactReactionN'].reshape((-1,3))):force[contact['slot']]=force.get(contact['slot'],np.zeros(3))+f
            reaction=max((float(np.linalg.norm(f)) for f in force.values()),default=0.)
            if not np.isfinite(reaction) or reaction>self.limits['maxReactionN']:raise ClosedLoopRefusal('CLOSED_LOOP_REACTION_DOMAIN')
        return np.asarray(rates),float(gradient),float(center)

    def advance(self,command):
        """Commit all physical state once; failed/cancelled ticks leave it intact."""
        preparation=self.backend.checkpoint()
        try:return self._advance(command)
        except BaseException:
            self.backend.rollback(preparation);raise

    def _advance(self,command):
        self.check();s=copy.deepcopy(self.state);op=command['operation'];key=command['slot'];dt=command['durationSeconds'];active=command['effectiveActive']
        if type(dt) not in (float,int) or not np.isfinite(dt) or dt<=0 or type(active) is not bool or key not in s['slots']:
            raise ValueError('CLOSED_LOOP_COMMAND')
        steps=int(np.floor((s['remainderSeconds']+dt+1e-12)/self.h))
        if steps>120 or s['ticks']>=128:raise ClosedLoopRefusal('CLOSED_LOOP_STEP_BUDGET')
        if op not in ('apply','release','inspect') or active!=(op=='apply'):raise ClosedLoopRefusal('CLOSED_LOOP_OPERATION')
        pose=s['slots'][key] if command['pose']=='current' else command['pose'];p=np.asarray(pose['position'],float);q=np.asarray(pose['quaternion'],float)
        if p.shape!=(3,) or q.shape!=(4,) or not np.isfinite(p).all() or not np.isfinite(q).all():raise ValueError('CLOSED_LOOP_POSE')
        before_hash=self.state_hash();old_contact=float(.5*(contact_matrix(s['stage']['geometry'],s['contacts'])@s['u']-s['target'])@
            (spring(s['contacts'])*(contact_matrix(s['stage']['geometry'],s['contacts'])@s['u']-s['target'])))
        event=None;cusa_event=[];order=[];limiter=None
        if op=='apply':
            if s['slots'][key]['instrument']!='cusa' or key in s['holds']:raise ClosedLoopRefusal('CLOSED_LOOP_CUSA_ONLY')
            geometry=self._geometry(s);context={'slot':key,'slots':s['slots'],'holds':s['holds'],'toolMechanicsRevision':2,
                'parametersBySlot':s['parametersBySlot'],'requestedPose':{'position':p.tolist(),'quaternion':q.tolist()}}
            limiter=limiter_prediction(geometry,context);p=np.asarray(limiter['resolvedMm']);order=geometry.cusa_order(p)
            self._covers(s,key,p,q,swept=True)
            owned={'removed':s['removed'][:],'holds':s['holds'],'toolMechanicsRevision':2,'cusaV3':copy.deepcopy(s['cusa']),
                'deformation':{str(int(c)):v.tolist() for c,v in zip(geometry.field.cells,geometry.u.mean(axis=1))}}
            view=SimpleNamespace(scene=self.scene,state=owned,by_label=geometry.by_label,_protected_ids=frozenset(),
                touched=lambda a,b,r:geometry.occupancy.material_ids(geometry.grid_cover(a,b,r)))
            cells=view.touched(p,p,s['parametersBySlot'][key]['radiusMm'])
            predicted=capacity_prediction(s['cusa'],s['removed'],order,float(self.scene.voxel_mm3),cusa.profile(view),s['parametersBySlot'][key],dt,True)
            cusa_event=cusa.apply(view,cells,dt,s['parametersBySlot'][key],position=p,active=True)
            if predicted['events']!=cusa_event or predicted['cusaAfter']!=owned['cusaV3'] or predicted['removedSourceIdsAfter']!=owned['removed']:
                raise RuntimeError('CLOSED_LOOP_CAPACITY_ORACLE')
            new=set(owned['removed'])-set(s['removed']);held={c for h in s['holds'].values() for c in h['cells']}
            if new&held:raise ClosedLoopRefusal('CLOSED_LOOP_HELD_OWNER_REMOVAL')
            if not new<=set(map(int,self.domain.cells)):raise ClosedLoopRefusal('CLOSED_LOOP_REMOVAL_OUTSIDE_DOMAIN')
            s['cusa']=owned['cusaV3'];s['removed']=owned['removed']
            if new:
                stage=self.domain.after(sorted(set(s['removed'])&set(map(int,self.domain.cells))))
                transfer=s['sourceLedger'].transfer(s['stage']['geometry'],stage['geometry'],s['u'],s['z'],self.alpha)
                s['u'],s['z'],event=transfer['displacement'],transfer['memories'],transfer['event'];s['stage']=stage
                s['removedEnergyNmm']+=event['removedStoredEnergyNmm'];s['energyNmm']-=event['removedStoredEnergyNmm']
        else:
            if not np.array_equal(p,s['slots'][key]['position']) or not np.array_equal(q,s['slots'][key]['quaternion']):raise ClosedLoopRefusal('CLOSED_LOOP_STATIONARY_RELEASE_OR_INSPECT')
            if op=='release':
                if key not in s['holds']:raise ClosedLoopRefusal('CLOSED_LOOP_RELEASE_WITHOUT_HOLD')
                del s['holds'][key]
            self._covers(s,key,p,q,swept=True)
        s['slots'][key].update(position=p.tolist(),quaternion=q.tolist())
        contacts=hold_contacts(self.scene,s['holds']);target=targets(contacts);g=s['stage']['geometry'];free=np.flatnonzero(g['freeDofs'])
        solver,identity=self.backend.prepare(s['stage'],contacts,s['removed']);new_contact=solver.contact_energy(s['u'][free],target)
        jump=new_contact-old_contact;s['contactJumpNmm']+=jump;s['energyNmm']+=jump;s['contacts']=contacts;s['target']=target;s['sourceOperatorIdentity']=identity
        records=[]
        for within in range(steps):
            self.check();before_u=s['u'][free].copy();before_z=s['z'][:,free].copy();prior=field_hash(s['u'],s['z'])
            old_centers=s['u'].reshape((-1,3))[g['elementNodes']].mean(axis=1)
            result=solver.step(before_u,before_z,np.zeros(len(free)),target,self.h,previous_target=target)
            if (result['maxFreeResidualN']>self.limits['forceN'] or abs(result['energyBalanceResidualNmm'])>self.limits['energyNmm']
                    or abs(result['oldEnergyNmm']-s['energyNmm'])>self.limits['energyNmm']):raise ClosedLoopRefusal('CLOSED_LOOP_FORCE_OR_ENERGY')
            s['u']=np.zeros(g['nodes'].size);s['u'][free]=result['displacement'];s['z']=np.zeros((len(self.alpha),len(s['u'])));s['z'][:,free]=result['memories']
            self._increment_guard(s,old_centers)
            s['energyNmm']=result['energyNmm'];s['workNmm']+=result['workNmm'];s['lossNmm']+=result['dissipatedEnergyNmm']+result['numericalDissipationNmm']
            rates,gradient,center=self._field_guard(s,solver,result);s['exposure']+=rates*self.h;s['fixedSteps']+=1
            balance=s['energyNmm']-s['initialEnergyNmm']+s['removedEnergyNmm']-s['contactJumpNmm']+s['lossNmm']-s['workNmm']
            if abs(balance)>self.limits['balanceNmm']:raise ClosedLoopRefusal('CLOSED_LOOP_CUMULATIVE_ENERGY')
            record={'tick':s['ticks'],'withinTick':within,'fixedStep':s['fixedSteps'],'sourceOperatorIdentity':identity,
                'previousArrayHash':prior,'arrayHash':field_hash(s['u'],s['z']),'cumulativeEventBalanceResidualNmm':balance,
                'maxGradientNorm':gradient,'maxCenterDisplacementMm':center,'rates':rates.tolist(),
                **{k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in result.items() if k not in ('displacement','memories')}}
            records.append((record,{'freeIndices':free,'beforeDisplacement':before_u,'beforeMemories':before_z,
                'displacement':s['u'][free].copy(),'memories':s['z'][:,free].copy(),'target':target.copy(),'h':np.asarray(self.h)}))
        s['elapsedSeconds']+=dt;s['remainderSeconds']=max(0.,round(s['remainderSeconds']+dt-steps*self.h,14));s['ticks']+=1
        self._field_guard(s,solver,{'contactReactionN':spring(contacts)*(target-contact_matrix(g,contacts)@s['u'])})
        balance=s['energyNmm']-s['initialEnergyNmm']+s['removedEnergyNmm']-s['contactJumpNmm']+s['lossNmm']-s['workNmm']
        if abs(balance)>self.limits['balanceNmm']:raise ClosedLoopRefusal('CLOSED_LOOP_CUMULATIVE_ENERGY')
        post=self._covers(s,key,p,q,swept=False);self.check();after_hash=self._state_hash(s)
        result=copy.deepcopy({'command':command,'beforeStateHash':before_hash,'afterStateHash':after_hash,
            'resolvedPose':{'position':p.tolist(),'quaternion':q.tolist()},'limiter':limiter,'orderedSourceIds':order,
            'cusaEvents':cusa_event,'sourceEvent':event,'contactModelEnergyJumpNmm':jump,'fixedSteps':steps,
            'remainderSeconds':s['remainderSeconds'],'elapsedSeconds':s['elapsedSeconds'],'postEndpointCovers':post,
            'predictedStatePropagated':True,'largerDomainPublicActionsAdmitted':False,'physicalValidationPassed':False})
        batch={'beforeStateHash':before_hash,'afterStateHash':after_hash,'stage':s['stage'],'records':records}
        self.state=s;self._committed_batch=batch
        return result
