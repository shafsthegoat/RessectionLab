"""Offline full-field replay of frozen, committed native mechanics inputs.

Prescribing a native world's inputs does not admit a different world's public
actions/collisions. Episode fields, clocks and event ledgers remain independent
when immutable tissue operators are shared by a stage scheduler.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np
from scipy import sparse

from .surgical_continuum_events import SourceHistoryLedger


def contact_key(contacts):
    return tuple((row['slot'],int(row['sourceCell']),float(row['stiffnessNPerMm']).hex()) for row in contacts)


def targets(contacts):
    return np.asarray([row['targetDisplacementMm'] for row in contacts],float).reshape((-1,))


def contact_matrix(geometry,contacts):
    lookup={int(cell):row for row,cell in enumerate(geometry['cells'])};rows=[];columns=[];values=[]
    for i,contact in enumerate(contacts):
        if int(contact['sourceCell']) not in lookup:raise ValueError('PROGRAM_CONTACT_OWNER_ABSENT')
        nodes=geometry['elementNodes'][lookup[int(contact['sourceCell'])]]
        for axis in range(3):
            rows.extend([3*i+axis]*8);columns.extend((3*nodes+axis).tolist());values.extend([1/8]*8)
    return sparse.coo_matrix((values,(rows,columns)),shape=(3*len(contacts),geometry['nodes'].size)).tocsr()


def spring(contacts):
    return np.repeat(np.asarray([row['stiffnessNPerMm'] for row in contacts],float),3)


def field_hash(displacement,memories):
    digest=hashlib.sha256(b'resectionlab.full-source-fields.v1')
    for a in (displacement,memories):
        array=np.asarray(a,dtype=np.float64)
        digest.update(str(array.shape).encode());digest.update(array.tobytes())
    return digest.hexdigest()


def record_hash(record):
    return hashlib.sha256(json.dumps(record,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def prepare_program(name,data,masks,*,prefix_actions=15):
    if data['schema']!='resectionlab.committed-native-mechanics-inputs.v1':raise ValueError('PROGRAM_CAPTURE_SCHEMA')
    prefix=data['prefixActivation'];summary=data['summary'];base=set(prefix['removedSourceIds'])
    if (prefix['actions']!=prefix_actions or not prefix['continuumFieldsZero']
            or prefix['nativeRemainderSeconds']!=0. or summary['branch']!=name
            or not summary['canonicalEveryActionExact'] or not summary['canonicalFullSnapshotExact']
            or len(data['actions'])!=summary['actions'] or len(data['actionReceipts'])!=summary['actions']):
        raise ValueError('PROGRAM_NATIVE_PREFIX_OR_RECEIPTS')
    prior=[tick for tick in data['ticks'] if tick['actionIndex']<prefix_actions]
    suffix=[tick for tick in data['ticks'] if tick['actionIndex']>=prefix_actions]
    if not prior or not suffix:raise ValueError('PROGRAM_EMPTY_CLOCK')
    initial=prior[-1]['contacts'];previous_contacts=initial;clock=prefix['nativeRemainderSeconds'];time=prefix['timeSeconds']
    if np.any(targets(initial)) or prior[-1]['nativeRemainderAfterSeconds']!=clock:
        raise ValueError('PROGRAM_INITIAL_SPRING_OR_CLOCK')
    stages={tuple(mask):index for index,mask in enumerate(masks)};segments={};last_stage=0;last_action=prefix_actions;fixed=0
    for index,tick in enumerate(suffix):
        h=tick['fixedStepSeconds'];dt=tick['durationSeconds'];steps=tick['nativeFixedSteps'];contacts=tick['contacts']
        if (tick['bodyForce']!='zero' or not tick['nativeContinuumActive'] or not np.isfinite([h,dt]).all()
                or h<=0 or dt<=0 or type(steps) is not int or steps<=0
                or not last_action<=tick['actionIndex']<summary['actions']):raise ValueError('PROGRAM_INPUT_OR_CLOCK')
        if tick['nativeRemainderBeforeSeconds']!=clock:raise ValueError('PROGRAM_CLOCK_CHAIN')
        if int(np.floor((clock+dt+1e-12)/h))!=steps:raise ValueError('PROGRAM_FIXED_STEP_COUNT')
        clock=max(0.,round(clock+dt-steps*h,14))
        if clock!=tick['nativeRemainderAfterSeconds']:raise ValueError('PROGRAM_FIXED_CLOCK_ROUNDING')
        time+=dt
        if abs(time-tick['timeSeconds'])>1e-9:raise ValueError('PROGRAM_PHYSICAL_ELAPSED_TIME')
        top=data['topologies'][tick['topologyId']];whole=set(top['removedSourceIds'])
        if top['cutFaces'] or top['rigidSourceIds'] or not base<=whole:raise ValueError('PROGRAM_UNADMITTED_TOPOLOGY')
        mask=tuple(sorted(whole-base))
        if mask not in stages:raise ValueError('PROGRAM_SOURCE_MASK_OUTSIDE_FROZEN_GEOMETRY')
        stage=stages[mask]
        if stage<last_stage or stage>last_stage+1:raise ValueError('PROGRAM_NONMONOTONE_SOURCE_STAGES')
        if len(contacts)>21 or len({(c['slot'],c['sourceCell']) for c in contacts})!=len(contacts):
            raise ValueError('PROGRAM_CONTACT_DECLARATION')
        for contact in contacts:
            if (not isinstance(contact['slot'],str) or type(contact['sourceCell']) is not int
                    or not np.isfinite(contact['stiffnessNPerMm']) or contact['stiffnessNPerMm']<=0
                    or np.asarray(contact['targetDisplacementMm']).shape!=(3,)
                    or not np.isfinite(contact['targetDisplacementMm']).all()):raise ValueError('PROGRAM_CONTACT_DECLARATION')
        # These captured protocols contain no simultaneous topology/binding or
        # topology/target changes; do not invent an instantaneous work policy.
        if stage!=last_stage and (contact_key(contacts)!=contact_key(previous_contacts)
                or not np.array_equal(targets(contacts),targets(previous_contacts))):
            raise ValueError('PROGRAM_SIMULTANEOUS_MODEL_AND_TARGET_CHANGE')
        segments.setdefault(stage,[]).append((index,tick));last_stage=stage;last_action=tick['actionIndex']
        previous_contacts=contacts;fixed+=steps
    if (len(suffix)!=summary['suffixTicks'] or fixed!=summary['suffixFixedSteps']
            or abs(sum(t['durationSeconds'] for t in suffix)-summary['suffixDurationSeconds'])>1e-9
            or list(masks[last_stage])!=summary['additionalRemovedSourceIds']):raise ValueError('PROGRAM_CAPTURE_COVERAGE')
    return {'name':name,'capture':data,'prefix':prefix,'initialContacts':initial,'ticks':suffix,'segments':segments,
        'finalStage':last_stage,'fixedSteps':fixed}


class Episode:
    """Independent full source fields and actual spring/event/BE accounting."""
    def __init__(self,program,geometry,alpha,times):
        self.program=program;self.geometry=geometry;self.alpha=np.asarray(alpha);self.times=np.asarray(times)
        self.u=np.zeros(geometry['nodes'].size);self.z=np.zeros((len(alpha),len(self.u)))
        self.contacts=program['initialContacts'];self.modeled_target=targets(self.contacts)
        self.ledger=SourceHistoryLedger();self.steps=0;self.next_tick=0;self.stage=0;self.energy=0.
        self.removed=0.;self.jump=0.;self.work=0.;self.loss=0.;self.exposure=np.zeros(2)
        self.chain=hashlib.sha256(b'resectionlab.full-source-program.v1');self.pending_event=None
        self.pending_orphans=None;self.pending_jump=0.;self.previous_array_hash=field_hash(self.u,self.z)

    def _contact_energy(self,geometry,u,contacts,target):
        stretch=contact_matrix(geometry,contacts)@u-target
        return float(.5*stretch@(spring(contacts)*stretch))

    def enter_geometry(self,geometry,stage,first_contacts):
        if stage!=self.stage+1 or contact_key(first_contacts)!=contact_key(self.contacts):
            raise ValueError('PROGRAM_STAGE_OR_BINDING_TRANSFER')
        old=self.geometry;old_u=self.u;old_z=self.z
        old_contact=self._contact_energy(old,old_u,self.contacts,self.modeled_target)
        transferred=self.ledger.transfer(old,geometry,old_u,old_z,self.alpha)
        # Export only vanished old-node values. All other pre-event fields can
        # be reconstructed losslessly from the new before-state and owner map.
        old_lookup={int(cell):i for i,cell in enumerate(old['cells'])}
        retained_rows=np.asarray([old_lookup[int(cell)] for cell in geometry['cells']])
        orphan=np.setdiff1d(np.arange(len(old['nodes'])),np.unique(old['elementNodes'][retained_rows]))
        self.pending_orphans={'orphanOldNodeIndices':orphan,
            'orphanOldDisplacement':old_u.reshape((-1,3))[orphan].copy(),
            'orphanOldMemories':old_z.reshape((len(self.alpha),-1,3))[:,orphan].copy()}
        self.u,self.z=transferred['displacement'],transferred['memories'];self.pending_event=transferred['event']
        jump=self._contact_energy(geometry,self.u,self.contacts,self.modeled_target)-old_contact
        removed=self.pending_event['removedStoredEnergyNmm'];self.removed+=removed;self.jump+=jump
        self.pending_jump+=jump;self.energy+=jump-removed;self.geometry=geometry;self.stage=stage

    def bind(self,contacts):
        if contact_key(contacts)==contact_key(self.contacts):return False
        old=self._contact_energy(self.geometry,self.u,self.contacts,self.modeled_target)
        target=targets(contacts);new=self._contact_energy(self.geometry,self.u,contacts,target)
        jump=new-old;self.jump+=jump;self.pending_jump+=jump;self.energy+=jump
        self.contacts=contacts;self.modeled_target=target
        return True

    def advance(self,solver,tick,tick_index,within_tick,measure,limits,domain):
        if tick_index!=self.next_tick:raise ValueError('PROGRAM_BRANCH_TICK_ORDER')
        free=np.flatnonzero(self.geometry['freeDofs']);before=self.u[free].copy();old_z=self.z[:,free].copy()
        target=targets(tick['contacts']);previous=self.modeled_target.copy();h=tick['fixedStepSeconds']
        result=solver.step(before,old_z,np.zeros(len(free)),target,h,previous_target=previous)
        if result['maxFreeResidualN']>limits['forceN']:raise ValueError('PROGRAM_FULL_FREE_FORCE_BUDGET')
        if abs(result['oldEnergyNmm']-self.energy)>limits['energyNmm']:raise ValueError('PROGRAM_EVENT_TO_STEP_ENERGY')
        old_hash=self.previous_array_hash
        self.u=np.zeros(self.geometry['nodes'].size);self.u[free]=result['displacement']
        self.z=np.zeros((len(self.alpha),len(self.u)));self.z[:,free]=result['memories']
        self.steps+=1;self.modeled_target=target.copy();self.energy=result['energyNmm']
        self.work+=result['workNmm'];self.loss+=result['dissipatedEnergyNmm']+result['numericalDissipationNmm']
        rates,gradient,center=measure(self.geometry,free,result['displacement'],result['memories'])
        self.exposure+=rates*h
        forces={}
        for contact,force in zip(tick['contacts'],result['contactReactionN'].reshape((-1,3))):
            forces[contact['slot']]=forces.get(contact['slot'],np.zeros(3))+force
        max_reaction=max((float(np.linalg.norm(force)) for force in forces.values()),default=0.)
        if (gradient>domain['maxGradientNorm'] or center>domain['maxCenterDisplacementMm']
                or max_reaction>domain['maxReactionN']):raise ValueError('PROGRAM_APPROXIMATION_DOMAIN')
        balance=self.energy+self.removed-self.jump+self.loss-self.work
        if abs(balance)>limits['balanceNmm']:raise ValueError('PROGRAM_CUMULATIVE_EVENT_ENERGY')
        self.previous_array_hash=field_hash(self.u,self.z)
        record={'branch':self.program['name'],'step':self.steps,'tissueState':self.stage,'suffixTick':tick_index,
            'withinTick':within_tick,'actionIndex':tick['actionIndex'],'operation':tick['operation'],'slot':tick['slot'],
            'h':h,'target':target.tolist(),'previousTarget':previous.tolist(),'rates':rates.tolist(),
            'cumulativeExposure':self.exposure.tolist(),'maxGradientNorm':gradient,'maxCenterDisplacementMm':center,
            'maxHandleReactionN':max_reaction,'removedStoredEnergyNmm':self.removed,'contactModelEnergyJumpNmm':self.pending_jump,
            'cumulativeContactModelEnergyJumpNmm':self.jump,'cumulativeLossNmm':self.loss,'cumulativeWorkNmm':self.work,
            'cumulativeEventBalanceResidualNmm':balance,'event':self.pending_event,'previousArrayHash':old_hash,
            'arrayHash':self.previous_array_hash,**{key:value.tolist() if isinstance(value,np.ndarray) else value
                for key,value in result.items() if key not in ('displacement','memories')}}
        digest=record_hash(record);self.chain.update(bytes.fromhex(digest))
        fields={'freeIndices':free,'displacement':result['displacement'],'memories':result['memories'],
            'beforeDisplacement':before,'beforeMemories':old_z,'target':target,'previousTarget':previous,'h':h,
            **(self.pending_orphans or {'orphanOldNodeIndices':np.empty(0,dtype=np.int64),
                'orphanOldDisplacement':np.empty((0,3)),'orphanOldMemories':np.empty((len(self.alpha),0,3))})}
        self.pending_event=None;self.pending_orphans=None;self.pending_jump=0.
        if within_tick+1==tick['nativeFixedSteps']:self.next_tick+=1
        return record,digest,fields

    def summary(self):
        p=self.program;capture=p['capture'];source=capture['summary']['fullSourceAccounting'];prefix=p['prefix']['fullSourceAccounting']
        owners=sorted(self.ledger.removed_source_ids)
        if (self.steps!=p['fixedSteps'] or self.next_tick!=len(p['ticks'])
                or owners!=capture['summary']['additionalRemovedSourceIds']
                or prefix['sourceCellCount']+len(owners)!=source['sourceCellCount']):
            raise ValueError('PROGRAM_FINAL_CLOCK_OR_SOURCE_CONSERVATION')
        return {'branch':p['name'],'steps':self.steps,'ticks':self.next_tick,'finalTissueState':self.stage,
            'arrayHash':self.previous_array_hash,'replayChainSha256':self.chain.hexdigest(),'energyNmm':self.energy,
            'removedSourceIds':owners,'removedStoredEnergyNmm':self.removed,'contactModelEnergyJumpNmm':self.jump,
            'workNmm':self.work,'lossNmm':self.loss,'cumulativeExposure':self.exposure.tolist(),
            'cumulativeEventBalanceResidualNmm':self.energy+self.removed-self.jump+self.loss-self.work,
            'fullSourceAccounting':source,'unmodeledPrefixAccounting':prefix,'publicActions':len(capture['actions']),
            'nativeActionReceiptsPreserved':True,'largerDomainPublicActionsAdmitted':False}
